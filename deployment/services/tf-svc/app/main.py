"""
tf-svc - FastAPI application entry point.

Hosts the TensorFlow translation endpoint (encoder-decoder Transformer
trained on Tatoeba EN-ES with shared 8K BPE vocab).

USAGE (from deployment/services/tf-svc/):
    .venv\\Scripts\\uvicorn.exe app.main:app --reload --port 8003

WHAT THIS FILE CONTAINS:
    - FastAPI() instance with metadata for OpenAPI/Swagger docs
    - Lifespan event placeholder (the loader integration lands once
      app/services/translation_loader.py is in place)
    - /health (liveness) - cheap, never does work
    - /ready (readiness) - currently always returns 503 because no
      model has been loaded yet; this gets gated on the loader's
      is_loaded() check once the loader is in place
    - /metrics (Prometheus scrape endpoint)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan: load every model this service hosts at startup.

    Currently a no-op pending the loader. Once translation_loader is
    in place, the body becomes `translation_loader.load_translation_model()`
    so that:
      - failures crash the container immediately (k8s/Docker see the
        non-zero exit and trigger restart/alerts)
      - /ready stays 503 throughout the load window
      - the first real request arrives to a fully-warm cache
    """
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
        - HTTP 503 with detail="model_not_loaded" because no model
          has been loaded yet. The detail string is grep-friendly
          for log scrapers.

    Once the loader is in place, this becomes 200 with a per-model
    `models` dict once the lifespan event finishes loading the
    Transformer + tokenizer; 503 stays for the load window.

    Used by:
        - Kubernetes readinessProbe
        - Rolling-deploy systems waiting before draining old pods
    """
    raise HTTPException(status_code=503, detail="model_not_loaded")


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
