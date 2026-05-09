"""
pt-svc - FastAPI application entry point.

Hosts the PyTorch deployment endpoints:
    - D2: /predict/dnn          (UCI HAR activity classification)
    - D3: /predict/gan/sample   (DCGAN image generation - later phase)
    - D4: /predict/qlearning/taxi (Q-table action lookup - later phase)

USAGE (from deployment/services/pt-svc/):
    .venv\\Scripts\\uvicorn.exe app.main:app --reload --port 8002

Port convention across services:
    sklearn-svc -> 8001
    pt-svc      -> 8002    (this service)
    tf-svc      -> 8003

WHAT THIS FILE CONTAINS NOW (after Step 3.5):
    - FastAPI() instance with metadata for OpenAPI/Swagger docs
    - Middleware stack (request_id + structured logging + metrics)
    - Lifespan event that loads the DNN + scaler AND the DCGAN
      generator at startup (sequential, ~100ms each)
    - /health (liveness) - cheap, never does work
    - /ready (readiness) - returns 503 until BOTH models are loaded;
      200 with a per-model `models` dict once loaded
    - /predict/dnn         (D2 router)
    - /predict/gan/sample  (D3 router)
    - /metrics (Prometheus scrape endpoint)

WHAT WILL BE ADDED LATER:
    - Phase 4: include POST /predict/qlearning/taxi router (D4)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.middleware.logging import LoggingMiddleware, configure_logging
from app.middleware.metrics import MetricsMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.routers import dnn as dnn_router
from app.routers import gan as gan_router
from app.services import dnn_loader, gan_loader

# Configure structured (JSON) logging at module import. Runs ONCE per
# process. Must happen before any module-level logger is created so
# everything emits through the structlog pipeline (including the
# dnn_loader logger that fires during the lifespan event below).
configure_logging()

"""
Lifespan event - runs once per process, before any request is served.
Code BEFORE `yield` is startup; code AFTER is shutdown. We block on
the model load here so:
  - failures crash the container immediately (k8s/Docker see the
    non-zero exit and trigger restart/alerts)
  - /ready stays 503 throughout the load window
  - the first real request arrives to a fully-warm cache
No shutdown work yet - process exit reclaims the in-memory model.
"""

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan: load every model this service hosts at startup.

    Loads run SEQUENTIALLY, not in parallel. Each load is ~100 ms
    (registry query + .pth read + state_dict apply + .eval()), so
    total startup is ~200 ms - asyncio.gather would save ~100 ms at
    the cost of harder-to-read error paths. Worth it only if loads
    were network-bound (e.g., S3 downloads), not local file reads.

    If ANY load fails, the exception propagates out of lifespan,
    uvicorn logs the traceback, and the process exits non-zero.
    Docker/k8s see the failure and trigger restart/alerts. /ready
    will never return 200 in that scenario - the process is dead.
    """
    dnn_loader.load_dnn_model()
    gan_loader.load_gan_model()
    yield
    # Shutdown - intentionally empty.


app = FastAPI(
    title="pt-svc",
    version="0.1.0",
    description=(
        "Deployment service for PyTorch models: D2 DNN (UCI HAR activity "
        "recognition, 96.03% test accuracy), D3 DCGAN (CIFAR-10 image "
        "generation, FID 30.57), D4 Q-Learning V1 Tabular (Taxi-v4). "
        "Loads from the consolidated MLflow registry at "
        "`deployment/mlflow.db`."
    ),
    lifespan=lifespan,
)


# Middleware - run on every request, in REVERSE-add order.
# Starlette executes the LAST add_middleware as the OUTERMOST layer
# (sees the request first, the response last). For incoming requests:
#     request_id (outermost - generates the ID)
#         -> logging   (binds request_id into structlog contextvars,
#                       emits request_started + request_finished)
#             -> metrics (counter + histogram + gauge per request)
#                 -> handler
# So we add in REVERSE: innermost first, outermost last. Metrics is
# innermost so its measured latency is just the handler work, not the
# bookkeeping of the outer layers.
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)


# Mount domain routers. Each router groups related endpoints under a
# shared prefix. include_router() registers all the router's routes
# onto the main app at startup. Order doesn't affect routing (FastAPI
# matches by exact path + method); we list in the same order as the
# health-check `models` dict for readability.
app.include_router(dnn_router.router)
app.include_router(gan_router.router)


# Health check endpoints
# /health = "is the process alive?" - cheap, no work.
# /ready  = "should the load balancer route traffic to me?" - gates
# on model load completion so traffic doesn't hit a not-yet-warm pod.


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """
    Liveness check - the process is alive and responding.

    Returns:
        {"status": "alive"} with HTTP 200.
    """
    return {"status": "alive"}


@app.get("/ready", tags=["health"])
async def ready() -> dict[str, Any]:
    """
    Readiness check - the service can accept traffic.

    pt-svc hosts MULTIPLE models (DNN now, GAN now, Q-learning later),
    so the response shape reports per-model status keyed by model
    name. This is a deliberate departure from sklearn-svc's single-
    model shape - pt-svc was always going to outgrow that contract.

    Returns:
        - HTTP 200 with payload:
            {
              "status": "ready",
              "models": {
                "pt-dnn":         {"loaded": true, "version": "1"},
                "pt-gan-dcgan":   {"loaded": true, "version": "1"}
              }
            }
          once the lifespan event finishes loading EVERY model.
        - HTTP 503 with detail="model_not_loaded" while ANY model is
          still loading. The whole-service-or-nothing semantic is the
          right call: if /predict/dnn would 503 because DNN isn't
          ready, the load balancer shouldn't route ANY traffic to the
          pod. The detail string is grep-friendly for log scrapers.

    Used by:
        - Kubernetes readinessProbe
        - Rolling-deploy systems waiting before draining old pods
    """
    if not (dnn_loader.is_loaded() and gan_loader.is_loaded()):
        raise HTTPException(status_code=503, detail="model_not_loaded")

    return {
        "status": "ready",
        "models": {
            dnn_loader.MODEL_NAME: {
                "loaded": True,
                "version": dnn_loader.get_model_version(),
            },
            gan_loader.MODEL_NAME: {
                "loaded": True,
                "version": gan_loader.get_model_version(),
            },
        },
    }


"""
Prometheus scrape endpoint
Plain GET that returns the current state of all registered metrics
in Prometheus text exposition format. We return a raw Response (not
Pydantic) because the body is plain text in a specific format -
Pydantic JSON serialization would break it. CONTENT_TYPE_LATEST is
"text/plain; version=0.0.4; charset=utf-8" which is what Prometheus
servers expect.
"""
# This path is in EXCLUDED_PATHS in middleware/metrics.py so scrapes
# don't inflate the very counters they're reading.


@app.get("/metrics", tags=["observability"])
async def metrics() -> Response:
    """
    Expose registered Prometheus metrics in text exposition format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
