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

WHAT THIS FILE CONTAINS NOW (after Step 2.4c):
    - FastAPI() instance with metadata for OpenAPI/Swagger docs
    - Lifespan event that loads the DNN + scaler at startup
    - /health (liveness) - cheap, never does work
    - /ready (readiness) - returns 503 until the DNN is loaded;
      200 with model_name + model_version once loaded

WHAT WILL BE ADDED LATER:
    - Step 2.5: include POST /predict/dnn router
    - Step 2.6: middleware stack (request_id + logging + metrics)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.services import dnn_loader

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
    FastAPI lifespan: load the DNN + scaler at startup.
    """
    dnn_loader.load_dnn_model()
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
async def ready() -> dict[str, str | bool]:
    """
    Readiness check - the service can accept traffic.

    Returns:
        - HTTP 200 with {status, model_loaded, model_name, model_version}
          once the lifespan event finishes loading.
        - HTTP 503 with detail="model_not_loaded" until then. The
          detail string is grep-friendly for log scrapers.

    Used by:
        - Kubernetes readinessProbe
        - Rolling-deploy systems waiting before draining old pods
    """
    if not dnn_loader.is_loaded():
        raise HTTPException(status_code=503, detail="model_not_loaded")

    return {
        "status": "ready",
        "model_loaded": True,
        "model_name": dnn_loader.MODEL_NAME,
        "model_version": dnn_loader.get_model_version(),
    }
