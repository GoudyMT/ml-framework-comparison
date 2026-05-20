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

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol

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
    # display order in Swagger UI. Three per-model tags match the
    # established per-router convention; health + observability group
    # the cross-cutting endpoints from main.py.
    openapi_tags=[
        {
            "name": "dnn",
            "description": (
                "Activity classification endpoint. UCI HAR 561-feature "
                "vector -> 6-class softmax (WALKING / WALKING_UPSTAIRS / "
                "WALKING_DOWNSTAIRS / SITTING / STANDING / LAYING)."
            ),
        },
        {
            "name": "gan",
            "description": (
                "Image generation endpoint. DCGAN samples N latent "
                "vectors and returns N base64-encoded 32x32 RGB PNGs."
            ),
        },
        {
            "name": "qlearning",
            "description": (
                "Tabular Q-learning policy lookup. Maps a Gymnasium "
                "Taxi-v4 state ID to the argmax action + full Q-vector."
            ),
        },
        {
            "name": "health",
            "description": (
                "Liveness (`/health`), readiness (`/ready`), and per-model "
                "freshness (`/health/dnn`, `/health/gan`, "
                "`/health/qlearning`) checks for orchestrator probes."
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


@app.get(
    "/health",
    tags=["health"],
    operation_id="health_check",
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
    operation_id="readiness_check",
    summary="Readiness probe - can the service handle traffic?",
    description=(
        "All-or-nothing readiness check across all 3 models hosted by "
        "this service. Returns HTTP 200 with `{status, models: {<name>: "
        "{loaded, version}}}` once every model (DNN + scaler, DCGAN, "
        "Q-table) is loaded from the registry. Returns HTTP 503 "
        "detail=`model_not_loaded` while ANY model is still loading - "
        "the load balancer shouldn't route traffic to a partially-loaded "
        "pod, since any individual `/predict/*` call would 503 on the "
        "missing model. For per-model freshness checks use "
        "`/health/<model>` instead."
    ),
    responses={
        200: {
            "description": (
                "All 3 models loaded; per-model dict in body shows "
                "loaded=true + version for each (pt-dnn, pt-gan-dcgan, "
                "pt-qlearning-taxi)."
            ),
        },
        503: {
            "description": (
                "At least one model not loaded yet. Body shape: "
                "`{\"detail\": \"model_not_loaded\"}`."
            ),
        },
    },
)
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
                "pt-dnn":              {"loaded": true, "version": "1"},
                "pt-gan-dcgan":        {"loaded": true, "version": "1"},
                "pt-qlearning-taxi":   {"loaded": true, "version": "1"}
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


# Per-model freshness endpoints
"""
Three /health/<model> endpoints, one per deployed model. The handler
logic is identical across all three - only the loader module passed
to inference_tracking varies. A factory generates the handler bound
to one loader; each endpoint registers via app.add_api_route() with
its own OpenAPI metadata (summary, description, operation_id).

The triplication that lived here through Phase 9 was a real cost: the
three blocks were one self-edit away from drifting (Phase 10 caught a
description-string drift between /health/dnn and the other two and had
to manually re-sync). The factory makes drift impossible; a fourth
/health/<model> addition is one new app.add_api_route call.
"""

_HEALTH_RESPONSES: dict[int | str, dict[str, str]] = {
    200: {
        "description": (
            "Model loaded + recent activity within threshold. Body "
            "includes age + threshold for at-a-glance comparison."
        ),
    },
    503: {
        "description": (
            "Either `model_not_loaded` (cache empty) or "
            "`inference_stale` (loaded but stale). Detail string "
            "distinguishes the two causes."
        ),
    },
}


class _HealthLoader(Protocol):
    """
    Structural contract for any loader module the `/health/<model>`
    factory accepts. Wider than `inference_tracking._InferenceLoader`
    because the factory also calls `is_loaded()` + `get_model_version()`
    directly on the loader (inference_tracking only needs the
    timestamp + name). Defining it here keeps the cross-service
    middleware Protocol minimal while still giving mypy strict a real
    contract to enforce at the factory boundary - a future fourth
    loader missing one of these attributes will fail type-check
    instead of breaking at request time.
    """

    MODEL_NAME: str
    _LAST_INFERENCE_TS: float

    def is_loaded(self) -> bool: ...

    def get_model_version(self) -> str: ...


def _make_health_route(
    loader: _HealthLoader,
) -> Callable[[], Awaitable[dict[str, Any]]]:
    """
    Build a /health/<model> handler bound to one loader module.

    The returned handler returns:
        - HTTP 200 with diagnostic body when `loader.is_loaded()` AND
          `inference_tracking.is_fresh(loader)` are both true.
        - HTTP 503 detail="model_not_loaded" when the cache is empty.
        - HTTP 503 detail="inference_stale" when loaded but past the
          staleness threshold (`MODEL_INFERENCE_STALENESS_SECONDS` env
          var; default 3600s; `0` disables the freshness check).

    The loader must satisfy the `_HealthLoader` Protocol above. Every
    loader module in this service (dnn_loader, gan_loader,
    qlearning_loader) does, by exposing the required module-level
    attributes + accessor functions.

    Differs from /ready: /ready is service-level (all-or-nothing across
    every model in this service); the handler this factory builds is
    per-model + adds a freshness dimension /ready does not track.
    """
    async def health_endpoint() -> dict[str, Any]:
        if not loader.is_loaded():
            raise HTTPException(status_code=503, detail="model_not_loaded")
        if not inference_tracking.is_fresh(loader):
            raise HTTPException(status_code=503, detail="inference_stale")
        return {
            "status": "healthy",
            "model_name": loader.MODEL_NAME,
            "version": loader.get_model_version(),
            "last_inference_age_seconds": round(
                inference_tracking.get_age(loader), 2
            ),
            "staleness_threshold_seconds": (
                inference_tracking.get_staleness_threshold()
            ),
        }
    return health_endpoint


app.add_api_route(
    "/health/dnn",
    _make_health_route(dnn_loader),
    methods=["GET"],
    tags=["health"],
    operation_id="health_dnn",
    summary="Per-model freshness check for the DNN endpoint",
    description=(
        "Per-model health + freshness check that `/ready` cannot express. "
        "Returns HTTP 200 with diagnostic body when the DNN is loaded "
        "AND the last activity (load or inference) is within the "
        "staleness threshold. Returns HTTP 503 detail=`model_not_loaded` "
        "if the cache is empty, or HTTP 503 detail=`inference_stale` "
        "if loaded but past the threshold. Threshold is operator-tunable "
        "via `MODEL_INFERENCE_STALENESS_SECONDS` env var (default 3600s). "
        "See `docs/monitoring.md` for the full design."
    ),
    responses=_HEALTH_RESPONSES,
)


app.add_api_route(
    "/health/gan",
    _make_health_route(gan_loader),
    methods=["GET"],
    tags=["health"],
    operation_id="health_gan",
    summary="Per-model freshness check for the GAN endpoint",
    description=(
        "Per-model health + freshness check that `/ready` cannot express. "
        "Returns HTTP 200 with diagnostic body when the DCGAN generator "
        "is loaded AND the last activity (load or inference) is within "
        "the staleness threshold. Returns HTTP 503 "
        "detail=`model_not_loaded` if the cache is empty, or HTTP 503 "
        "detail=`inference_stale` if loaded but past the threshold. "
        "Threshold is operator-tunable via "
        "`MODEL_INFERENCE_STALENESS_SECONDS` env var (default 3600s). "
        "See `docs/monitoring.md` for the full design."
    ),
    responses=_HEALTH_RESPONSES,
)


app.add_api_route(
    "/health/qlearning",
    _make_health_route(qlearning_loader),
    methods=["GET"],
    tags=["health"],
    operation_id="health_qlearning",
    summary="Per-model freshness check for the Q-learning endpoint",
    description=(
        "Per-model health + freshness check that `/ready` cannot express. "
        "Returns HTTP 200 with diagnostic body when the Taxi-v4 Q-table "
        "is loaded AND the last activity (load or inference) is within "
        "the staleness threshold. Returns HTTP 503 "
        "detail=`model_not_loaded` if the cache is empty, or HTTP 503 "
        "detail=`inference_stale` if loaded but past the threshold. "
        "Threshold is operator-tunable via "
        "`MODEL_INFERENCE_STALENESS_SECONDS` env var (default 3600s). "
        "See `docs/monitoring.md` for the full design."
    ),
    responses=_HEALTH_RESPONSES,
)


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


@app.get(
    "/metrics",
    tags=["observability"],
    operation_id="metrics",
    summary="Prometheus metrics scrape endpoint",
    description=(
        "Returns the current state of every registered metric in "
        "Prometheus text exposition format. Scraped on a schedule by a "
        "Prometheus server (default 15s interval). Body is plain text "
        "with content-type `text/plain; version=0.0.4; charset=utf-8`. "
        "Exposes the HTTP four-golden-signals metrics "
        "(`http_requests_total`, `http_request_duration_seconds`, "
        "`http_requests_in_flight`) plus the per-model "
        "`model_inference_duration_seconds` Histogram with one series "
        "per deployed model (pt-dnn, pt-gan-dcgan, pt-qlearning-taxi). "
        "See `docs/monitoring.md` for the full metric catalog + sample "
        "PromQL queries. This endpoint is excluded from request "
        "instrumentation so scrapes don't inflate the counters they're "
        "reading."
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
