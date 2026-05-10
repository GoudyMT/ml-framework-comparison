"""
tf-svc - FastAPI application entry point.

Hosts the TensorFlow translation endpoint (encoder-decoder Transformer
trained on Tatoeba EN-ES with shared 8K BPE vocab).

USAGE (from deployment/services/tf-svc/):
    .venv\\Scripts\\uvicorn.exe app.main:app --reload --port 8003

WHAT THIS FILE CONTAINS:
    - FastAPI() instance with metadata for OpenAPI/Swagger docs
    - Lifespan event that loads the Transformer + tokenizer at
      startup, including a warm-up forward pass to compile TF's
      oneDNN ops before the first user request
    - /health (liveness) - cheap, never does work
    - /ready (readiness) - 503 until both model and tokenizer are
      loaded; 200 with a per-model `models` dict once loaded
    - /metrics (Prometheus scrape endpoint)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.services import translation_loader


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan: load the Transformer + tokenizer at startup.

    Includes a warm-up forward pass inside the loader so TF's lazy
    oneDNN op compilation happens here (visible in startup time)
    rather than on the first user request (where it would look like
    a 5-10s latency spike).

    If the load fails, the exception propagates out of lifespan,
    uvicorn logs the traceback, and the process exits non-zero.
    Docker/k8s see the failure and trigger restart/alerts. /ready
    will never return 200 in that scenario - the process is dead.
    """
    translation_loader.load_translation_model()
    yield
    # Shutdown - intentionally empty.


app = FastAPI(
    title="tf-svc",
    version="0.1.0",
    description=(
        "Deployment service for TensorFlow models: encoder-decoder "
        "Transformer translating English to Spanish (Tatoeba EN-ES, "
        "BLEU 0.4456 with beam search, shared 8K BPE vocab). Loads "
        "from the consolidated MLflow registry at "
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
async def ready() -> dict[str, Any]:
    """
    Readiness check - the service can accept traffic.

    Returns:
        - HTTP 200 with payload:
            {
              "status": "ready",
              "models": {
                "tf-transformer-translation": {"loaded": true, "version": "1"}
              }
            }
          once the lifespan event finishes loading the Transformer
          AND the tokenizer (both required for the inference path).
        - HTTP 503 with detail="model_not_loaded" while either is
          still loading. The detail string is grep-friendly for log
          scrapers.

    The multi-model `models` dict shape matches pt-svc's contract -
    even though this service hosts a single model today, the dict
    structure scales naturally if a second TF model is ever added
    (no contract change for clients).

    Used by:
        - Kubernetes readinessProbe
        - Rolling-deploy systems waiting before draining old pods
    """
    if not translation_loader.is_loaded():
        raise HTTPException(status_code=503, detail="model_not_loaded")

    return {
        "status": "ready",
        "models": {
            translation_loader.MODEL_NAME: {
                "loaded": True,
                "version": translation_loader.get_model_version(),
            },
        },
    }


# Prometheus scrape endpoint
"""
Plain GET that returns the current state of all registered metrics
in Prometheus text exposition format. We return a raw Response (not
Pydantic) because the body is plain text in a specific format -
Pydantic JSON serialization would break it. CONTENT_TYPE_LATEST is
"text/plain; version=0.0.4; charset=utf-8" which is what Prometheus
servers expect.
"""
# Once the metrics middleware lands, that path is added to its
# EXCLUDED_PATHS so scrapes don't inflate the very counters they're
# reading.


@app.get("/metrics", tags=["observability"])
async def metrics() -> Response:
    """
    Expose registered Prometheus metrics in text exposition format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
