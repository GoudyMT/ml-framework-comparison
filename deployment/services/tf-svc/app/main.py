"""
tf-svc - FastAPI application entry point.

Hosts the TensorFlow translation endpoint (encoder-decoder Transformer
trained on Tatoeba EN-ES with shared 8K BPE vocab).

USAGE (from deployment/services/tf-svc/):
    .venv\\Scripts\\uvicorn.exe app.main:app --reload --port 8003

WHAT THIS FILE CONTAINS:
    - FastAPI() instance with metadata for OpenAPI/Swagger docs
    - Middleware stack (request_id + structured logging + metrics)
    - Lifespan event that loads the Transformer + tokenizer at
      startup, including a warm-up forward pass to compile TF's
      oneDNN ops before the first user request
    - /health (liveness) - cheap, never does work
    - /ready (readiness) - 503 until both model and tokenizer are
      loaded; 200 with a per-model `models` dict once loaded
    - Mounted router: /translate
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
from app.routers import translation as translation_router
from app.services import translation_loader

# Configure structured (JSON) logging at module import. Runs ONCE per
# process. Must happen before any module-level logger is created so
# everything emits through the structlog pipeline (including the
# translation_loader logger that fires during the lifespan event below).
configure_logging()


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
    # OpenAPI contact + license surface in the auto-generated /docs UI
    # header. Contact points operators at the public repo for issues /
    # discussions (no email - GitHub handle is the established public
    # surface). License matches the root LICENSE file + the README badge.
    contact={
        "name": "Goudy",
        "url": "https://github.com/GoudyMT/ml-framework-comparison",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    # Tag descriptions surface as headings in /docs. Order here controls
    # display order in Swagger UI. Single per-model tag for translation
    # matches the per-router convention used in the other services.
    openapi_tags=[
        {
            "name": "translation",
            "description": (
                "English-to-Spanish translation endpoint. Encoder-decoder "
                "Transformer with SentencePiece BPE tokenization (shared "
                "EN+ES 8K vocab); greedy autoregressive decode."
            ),
        },
        {
            "name": "health",
            "description": (
                "Liveness (`/health`), readiness (`/ready`), and per-model "
                "freshness (`/health/translation`) checks for orchestrator "
                "probes."
            ),
        },
        {
            "name": "observability",
            "description": (
                "Prometheus metrics scrape endpoint (`/metrics`). Returns "
                "HTTP + per-model inference metrics in text exposition "
                "format. See `docs/monitoring.md` for the metric catalog."
            ),
        },
    ],
    lifespan=lifespan,
)


"""
Middleware - run on every request, in REVERSE-add order.
Starlette executes the LAST add_middleware as the OUTERMOST layer
(sees the request first, the response last). For incoming requests:
    request_id (outermost - generates the ID)
        -> logging   (binds request_id into structlog contextvars,
                      emits request_started + request_finished)
            -> metrics (counter + histogram + gauge per request)
                -> handler
So we add in REVERSE: innermost first, outermost last. Metrics is
innermost so its measured latency is just the handler work, not the
bookkeeping of the outer layers.
"""
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)


"""
Mount domain routers. Each router groups related endpoints; this
service hosts a single one (/translate). include_router registers
its routes on the main app at startup.
"""
app.include_router(translation_router.router)


# Health check endpoints
# /health = "is the process alive?" - cheap, no work.
# /ready  = "should the load balancer route traffic to me?" - gates
# on model load completion so traffic doesn't hit a not-yet-warm pod.


@app.get(
    "/health",
    tags=["health"],
    summary="Liveness probe - is the process alive?",
    description=(
        "Trivial liveness check that does no work: no model load, no DB "
        "query, no network call. Returns `{\"status\": \"alive\"}` with "
        "HTTP 200 whenever the process is responding. Used by Kubernetes "
        "livenessProbe + Docker HEALTHCHECK + load-balancer 'is the "
        "backend up?' pings. For 'can the service handle traffic?' use "
        "`/ready` instead."
    ),
    responses={
        200: {"description": "Process is alive and responding."},
    },
)
async def health() -> dict[str, str]:
    """
    Liveness check - the process is alive and responding.

    Returns:
        {"status": "alive"} with HTTP 200.
    """
    return {"status": "alive"}


@app.get(
    "/ready",
    tags=["health"],
    summary="Readiness probe - can the service handle traffic?",
    description=(
        "Readiness check covering both the Transformer model AND the "
        "SentencePiece tokenizer (both must be loaded for `/translate` "
        "to function). Returns HTTP 200 with `{status, models: {<name>: "
        "{loaded, version}}}` once both artifacts are loaded from the "
        "registry. Returns HTTP 503 detail=`model_not_loaded` while "
        "either is still loading. For per-model freshness check use "
        "`/health/translation` instead."
    ),
    responses={
        200: {
            "description": (
                "Both model + tokenizer loaded. Per-model dict in body "
                "shows loaded=true + version for tf-transformer-translation."
            ),
        },
        503: {
            "description": (
                "Model or tokenizer not loaded yet. Body shape: "
                "`{\"detail\": \"model_not_loaded\"}`."
            ),
        },
    },
)
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


@app.get(
    "/health/translation",
    tags=["health"],
    summary="Per-model freshness check for the translation endpoint",
    description=(
        "Per-model health + freshness check that `/ready` cannot express. "
        "Returns HTTP 200 with diagnostic body when the Transformer is "
        "loaded AND the last activity (load or inference) is within the "
        "staleness threshold. Returns HTTP 503 detail=`model_not_loaded` "
        "if either the model or tokenizer cache is empty, or HTTP 503 "
        "detail=`inference_stale` if loaded but past the threshold. "
        "Threshold is operator-tunable via "
        "`MODEL_INFERENCE_STALENESS_SECONDS` env var (default 3600s; "
        "`0` disables the freshness check, leaving loaded-only health). "
        "See `docs/monitoring.md` for the full design."
    ),
    responses={
        200: {
            "description": (
                "Transformer loaded + recent activity within threshold. "
                "Body includes age + threshold for at-a-glance comparison."
            ),
        },
        503: {
            "description": (
                "Either `model_not_loaded` (cache empty; lifespan not "
                "complete) or `inference_stale` (loaded but no activity "
                "within threshold). Detail string in the JSON body "
                "distinguishes the two causes for log scrapers."
            ),
        },
    },
)
async def health_translation() -> dict[str, Any]:
    """
    Per-model freshness check for the translation endpoint.

    Returns 200 with diagnostic body when the model is loaded AND the
    last activity (load or inference) is within the staleness threshold
    (MODEL_INFERENCE_STALENESS_SECONDS env var, default 3600s). Returns
    503 detail="model_not_loaded" when the cache is empty, or 503
    detail="inference_stale" when loaded but past the threshold.

    Differs from /ready: /ready is service-level; /health/translation
    is per-model + adds a freshness dimension that /ready does not track.
    """
    if not translation_loader.is_loaded():
        raise HTTPException(status_code=503, detail="model_not_loaded")
    if not inference_tracking.is_fresh(translation_loader):
        raise HTTPException(status_code=503, detail="inference_stale")
    return {
        "status": "healthy",
        "model_name": translation_loader.MODEL_NAME,
        "version": translation_loader.get_model_version(),
        "last_inference_age_seconds": round(
            inference_tracking.get_age(translation_loader), 2
        ),
        "staleness_threshold_seconds": (
            inference_tracking._resolve_staleness_threshold()
        ),
    }


# Prometheus scrape endpoint
"""
Plain GET that returns the current state of all registered metrics
in Prometheus text exposition format. We return a raw Response (not
Pydantic) because the body is plain text in a specific format -
Pydantic JSON serialization would break it. CONTENT_TYPE_LATEST is
"text/plain; version=0.0.4; charset=utf-8" which is what Prometheus
servers expect.

This path is in EXCLUDED_PATHS in middleware/metrics.py so scrapes
don't inflate the very counters they're reading.
"""


@app.get(
    "/metrics",
    tags=["observability"],
    summary="Prometheus metrics scrape endpoint",
    description=(
        "Returns the current state of every registered metric in "
        "Prometheus text exposition format. Scraped on a schedule by a "
        "Prometheus server (default 15s interval). Body is plain text "
        "with content-type `text/plain; version=0.0.4; charset=utf-8`. "
        "Exposes the HTTP four-golden-signals metrics "
        "(`http_requests_total`, `http_request_duration_seconds`, "
        "`http_requests_in_flight`) plus the per-model "
        "`model_inference_duration_seconds` Histogram "
        "(model_name=`tf-transformer-translation`). See "
        "`docs/monitoring.md` for the full metric catalog + sample PromQL "
        "queries. This endpoint is excluded from request instrumentation "
        "so scrapes don't inflate the counters they're reading."
    ),
    responses={
        200: {
            "description": (
                "Prometheus text exposition format. One series per line "
                "plus `# HELP` and `# TYPE` comment lines per metric."
            ),
        },
    },
)
async def metrics() -> Response:
    """
    Expose registered Prometheus metrics in text exposition format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
