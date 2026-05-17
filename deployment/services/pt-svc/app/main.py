"""
pt-svc - FastAPI application entry point.

Hosts the PyTorch deployment endpoints:
    - /predict/dnn              (UCI HAR activity classification)
    - /predict/gan/sample       (DCGAN image generation)
    - /predict/qlearning/taxi   (Q-learning Taxi-v4 policy lookup)

USAGE (from deployment/services/pt-svc/):
    .venv\\Scripts\\uvicorn.exe app.main:app --reload --port 8002

WHAT THIS FILE CONTAINS:
    - FastAPI() instance with metadata for OpenAPI/Swagger docs
    - Middleware stack (request_id + structured logging + metrics)
    - Lifespan event that loads every registered model at startup
      (DNN + scaler, then DCGAN, then Q-table; sequential)
    - /health (liveness) - cheap, never does work
    - /ready (readiness) - 503 until every model is loaded; 200 with
      a per-model `models` dict once loaded
    - Mounted routers: /predict/dnn, /predict/gan/sample,
      /predict/qlearning/taxi
    - /metrics (Prometheus scrape endpoint)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.middleware import inference_tracking
from app.middleware.logging import LoggingMiddleware, configure_logging
from app.middleware.metrics import MetricsMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.routers import dnn as dnn_router
from app.routers import gan as gan_router
from app.routers import qlearning as qlearning_router
from app.services import dnn_loader, gan_loader, qlearning_loader

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
    qlearning_loader.load_qlearning_model()
    yield
    # Shutdown - intentionally empty.


app = FastAPI(
    title="pt-svc",
    version="0.1.0",
    description=(
        "Deployment service for PyTorch models: DNN classifier (UCI HAR "
        "activity recognition, 96.03% test accuracy), DCGAN image "
        "generator (CIFAR-10, FID 30.57), and tabular Q-learning "
        "policy (Gymnasium Taxi-v4). Loads from the consolidated "
        "MLflow registry at `deployment/mlflow.db`."
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
app.include_router(qlearning_router.router)


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

    This service hosts multiple models, so the response shape reports
    per-model status keyed by model name. A multi-model dict is the
    right shape regardless of how many models are loaded - one entry
    today scales to N entries with no contract change for clients.

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
    if not (
        dnn_loader.is_loaded()
        and gan_loader.is_loaded()
        and qlearning_loader.is_loaded()
    ):
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
            qlearning_loader.MODEL_NAME: {
                "loaded": True,
                "version": qlearning_loader.get_model_version(),
            },
        },
    }


@app.get("/health/dnn", tags=["health"])
async def health_dnn() -> dict[str, Any]:
    """
    Per-model freshness check for the DNN endpoint.

    Returns 200 with diagnostic body when the model is loaded AND the
    last activity (load or inference) is within the staleness threshold
    (MODEL_INFERENCE_STALENESS_SECONDS env var, default 3600s). Returns
    503 detail="model_not_loaded" when the cache is empty, or 503
    detail="inference_stale" when loaded but past the threshold.

    Differs from /ready: /ready is service-level (all-or-nothing across
    every model in pt-svc); /health/dnn is per-model + adds a freshness
    dimension that /ready does not track.
    """
    if not dnn_loader.is_loaded():
        raise HTTPException(status_code=503, detail="model_not_loaded")
    if not inference_tracking.is_fresh(dnn_loader):
        raise HTTPException(status_code=503, detail="inference_stale")
    return {
        "status": "healthy",
        "model_name": dnn_loader.MODEL_NAME,
        "version": dnn_loader.get_model_version(),
        "last_inference_age_seconds": round(
            inference_tracking.get_age(dnn_loader), 2
        ),
        "staleness_threshold_seconds": (
            inference_tracking._resolve_staleness_threshold()
        ),
    }


@app.get("/health/gan", tags=["health"])
async def health_gan() -> dict[str, Any]:
    """
    Per-model freshness check for the GAN endpoint. See /health/dnn for
    the full contract; the gating logic is identical, only the loader
    module passed to inference_tracking changes.
    """
    if not gan_loader.is_loaded():
        raise HTTPException(status_code=503, detail="model_not_loaded")
    if not inference_tracking.is_fresh(gan_loader):
        raise HTTPException(status_code=503, detail="inference_stale")
    return {
        "status": "healthy",
        "model_name": gan_loader.MODEL_NAME,
        "version": gan_loader.get_model_version(),
        "last_inference_age_seconds": round(
            inference_tracking.get_age(gan_loader), 2
        ),
        "staleness_threshold_seconds": (
            inference_tracking._resolve_staleness_threshold()
        ),
    }


@app.get("/health/qlearning", tags=["health"])
async def health_qlearning() -> dict[str, Any]:
    """
    Per-model freshness check for the Q-learning endpoint. See /health/dnn
    for the full contract; the gating logic is identical, only the loader
    module passed to inference_tracking changes.
    """
    if not qlearning_loader.is_loaded():
        raise HTTPException(status_code=503, detail="model_not_loaded")
    if not inference_tracking.is_fresh(qlearning_loader):
        raise HTTPException(status_code=503, detail="inference_stale")
    return {
        "status": "healthy",
        "model_name": qlearning_loader.MODEL_NAME,
        "version": qlearning_loader.get_model_version(),
        "last_inference_age_seconds": round(
            inference_tracking.get_age(qlearning_loader), 2
        ),
        "staleness_threshold_seconds": (
            inference_tracking._resolve_staleness_threshold()
        ),
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
