"""
sklearn-svc - FastAPI application entry point.

This is the file `uvicorn` looks at to start the HTTP server. The variable
`app` (an instance of `fastapi.FastAPI`) IS the application; uvicorn imports
this module and serves `app` over HTTP.

USAGE (from deployment/services/sklearn-svc/):
    .venv\\Scripts\\uvicorn.exe app.main:app --reload --port 8001

The `app.main:app` notation means: in the module `app.main`, find the
attribute named `app`. The `--reload` flag auto-restarts the server on file
changes (DEV ONLY - never use --reload in production; it adds overhead).

WHAT THIS FILE CONTAINS:
    - FastAPI() instance with metadata for OpenAPI/Swagger docs
    - Middleware stack (request_id + structured logging + metrics)
    - Lifespan event that loads the PCA at startup
    - /health (liveness) - cheap, never does work
    - /ready (readiness) - 503 until the model is loaded; 200 with
      model_name + model_version once loaded
    - Mounted router: /predict/pca
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
from app.routers import pca as pca_router
from app.services import pca_loader

# Configure structured (JSON) logging at module import. Runs ONCE per
# process. Must happen before any module-level logger is created so
# everything in the app emits through the structlog pipeline.
configure_logging()

# Lifespan event
"""
FastAPI's modern startup/shutdown hook. Runs ONCE per process:
  - Code BEFORE `yield` -> startup
  - The `yield` itself  -> the entire serving window (could be days)
  - Code AFTER `yield`  -> shutdown (SIGTERM, ctrl-C, etc.)

Wrapped with @asynccontextmanager because FastAPI expects an async context
manager. The `app` parameter is the FastAPI instance - unused here, but
part of the required signature so FastAPI can pass references through.

WHY EAGER LOAD AT STARTUP (not on first request):
  - Failures are surfaced immediately, before any traffic arrives. The
    container exits, k8s sees the crash, alerts fire, deploy rolls back.
    Lazy load would let the container come up "alive" while every
    subsequent request 500s.
  - The 1-2s load cost is paid once at boot. Lazy would slow the first
    request unpredictably (cold-start tax).
  - readinessProbe handles the load window: /ready returns 503 until the
    loader finishes, so load balancers don't route to a not-yet-ready pod.

IF THE LOAD FAILS:
  pca_loader.load_pca_model() raises RuntimeError. We do NOT catch it
  here - we want it to propagate. Uvicorn will log the traceback and the
  process will exit non-zero. Docker / k8s will detect the failure and
  handle restart / alerting / rollback per their configured policy.

Shutdown work (after yield) is intentionally empty - the model is
in-memory only and process exit reclaims it.
"""

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan: load the PCA at startup; nothing to do at shutdown.
    """
    # Startup
    pca_loader.load_pca_model()

    yield

    # Shutdown - intentionally empty.


# FastAPI application
"""
The FastAPI() constructor accepts metadata that powers the auto-generated
OpenAPI/Swagger UI at /docs. Things to know:

  title         = human-readable service name; shown at top of /docs
  version       = API version. Separate from the package version in
                  pyproject.toml because APIs can evolve independently
                  of internal code.
  description   = markdown-supported overview text. Shows under the title.
  lifespan      = the async context manager defined above
  docs_url      = path for Swagger UI (default /docs)
  redoc_url     = path for ReDoc UI (default /redoc) - alternative renderer
  openapi_url   = path for raw openapi.json schema (default /openapi.json)
"""
app = FastAPI(
    title="sklearn-svc",
    version="0.1.0",
    description=(
        "Deployment service for D1: SK PCA dimensionality reduction "
        "(Fashion-MNIST, 150 components, 90.85% explained variance). "
        "Built for the ML-framework-comparisons portfolio's deployment phase. "
        "Loads the PCA from the consolidated MLflow registry to test and practice real world deployment "
        "(`models:/sk-pca@production`)."
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
    # display order in Swagger UI. Per-model tags (one per router) match
    # the established convention from the other services in this repo.
    openapi_tags=[
        {
            "name": "pca",
            "description": (
                "Dimensionality-reduction endpoint. Projects a flat "
                "Fashion-MNIST image (784 pixels) through the registered "
                "PCA to a 150-component vector."
            ),
        },
        {
            "name": "health",
            "description": (
                "Liveness (`/health`), readiness (`/ready`), and per-model "
                "freshness (`/health/pca`) checks for orchestrator probes."
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
Middleware - run on every request, in reverse-add order.
FastAPI/Starlette executes middleware as a stack: the LAST add_middleware
call becomes the OUTERMOST layer (sees the request first, the response
last). Registration order for incoming flow:
    request_id (outermost - generates the ID)
        -> logging   (binds request_id into structlog contextvars,
                      logs request_started + request_finished)
            -> metrics (counter+histogram+gauge per request)
                -> handler

So we add_middleware in REVERSE: innermost first, outermost last.
Metrics is innermost so the latency it measures is purely handler time,
not the bookkeeping done by the outer layers.
"""
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)


"""
Mount domain routers
Each router (in app/routers/) groups related endpoints under a shared
prefix. include_router() registers all the router's routes onto the
main app at startup. Order doesn't matter for routing, but grouping
imports near the top + mounts here keeps the wiring obvious.
"""
app.include_router(pca_router.router)


# Health check endpoints
"""
Two distinct concepts that orchestrators (Docker, k8s, load balancers)
query independently. /health = "is the process alive?" /ready = "should
I send real traffic?" A starting-up service is alive but not ready.

Both are tagged `health` so they group together in the Swagger UI.
"""

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

    This endpoint must NEVER do real work (no model load, no DB query, no
    network call). If it does and that work fails, the orchestrator
    (Docker / k8s) will restart the process unnecessarily. Keep it dumb.

    Used by:
        - Kubernetes `livenessProbe`
        - Docker `HEALTHCHECK` directive
        - Load balancer "is the backend up?" pings
    """
    return {"status": "alive"}


@app.get(
    "/ready",
    tags=["health"],
    operation_id="readiness_check",
    summary="Readiness probe - can the service handle traffic?",
    description=(
        "All-or-nothing readiness check. Returns HTTP 200 with "
        "`{status, model_loaded, model_name, model_version}` once the PCA "
        "is loaded from the registry and the service can accept "
        "`/predict/pca` calls. Returns HTTP 503 detail=`model_not_loaded` "
        "during the startup window before the lifespan event finishes "
        "loading the model. Used by Kubernetes readinessProbe + rolling-"
        "deploy systems waiting before draining old pods. For per-model "
        "freshness checks use `/health/pca` instead."
    ),
    responses={
        200: {"description": "Service is ready; PCA loaded with version echo."},
        503: {
            "description": (
                "PCA not loaded yet. Body shape: "
                "`{\"detail\": \"model_not_loaded\"}`."
            ),
        },
    },
)
async def ready() -> dict[str, str | bool]:
    """
    Readiness check - the service can accept traffic.

    Returns:
        - HTTP 200 with {status, model_loaded, model_name, model_version}
          when the PCA is loaded and the service is ready for /predict calls.
        - HTTP 503 (Service Unavailable) when the PCA hasn't loaded yet.
          The body has detail="model_not_loaded" so log scrapers can grep
          for it without parsing free-form text.

    Differs from /health: a service starting up is ALIVE but not READY.
    During the ~1-2 seconds it takes the lifespan event to load the PCA,
    /health returns 200 (process is alive) but /ready returns 503
    (don't send real traffic yet).

    Used by:
        - Kubernetes `readinessProbe` - controls whether the pod receives
          traffic from the Service load balancer
        - Rolling-deploy systems - wait for /ready before draining old pods
    """
    if not pca_loader.is_loaded():
        # 503 Service Unavailable = "I exist but can't serve requests now."
        # Standard HTTP code for "still warming up." Distinct from 500
        # (something broke) and 404 (route doesn't exist).
        raise HTTPException(status_code=503, detail="model_not_loaded")

    # Echo the resolved version + name so operators (and clients) can
    # confirm WHICH model is running without a separate registry call.
    return {
        "status": "ready",
        "model_loaded": True,
        "model_name": pca_loader.MODEL_NAME,
        "model_version": pca_loader.get_model_version(),
    }


@app.get(
    "/health/pca",
    tags=["health"],
    operation_id="health_pca",
    summary="Per-model freshness check for the PCA endpoint",
    description=(
        "Per-model health + freshness check that `/ready` cannot express. "
        "Returns HTTP 200 with diagnostic body "
        "(`status`, `model_name`, `version`, `last_inference_age_seconds`, "
        "`staleness_threshold_seconds`) when the PCA is loaded AND the "
        "last activity (load or inference) is within the staleness "
        "threshold. Returns HTTP 503 detail=`model_not_loaded` if the "
        "cache is empty, or HTTP 503 detail=`inference_stale` if loaded "
        "but past the threshold. Staleness threshold is operator-tunable "
        "via `MODEL_INFERENCE_STALENESS_SECONDS` env var (default 3600s; "
        "`0` disables the freshness check, leaving loaded-only health). "
        "See `docs/monitoring.md` for the full design + use cases."
    ),
    responses={
        200: {
            "description": (
                "Model loaded + recent activity within threshold. Body "
                "includes age + threshold for at-a-glance comparison."
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
async def health_pca() -> dict[str, Any]:
    """
    Per-model freshness check for the PCA endpoint.

    Returns:
        - HTTP 200 with diagnostic body when the model is loaded AND the
          last activity (load or inference) is within the staleness
          threshold (MODEL_INFERENCE_STALENESS_SECONDS env var, default
          3600s). Body includes the resolved age + threshold so operators
          can diff at a glance without a second call.
        - HTTP 503 detail="model_not_loaded" when the cache is empty -
          matches /ready's contract for the same condition.
        - HTTP 503 detail="inference_stale" when the model is loaded but
          age >= threshold. Distinct cause string lets log scrapers
          separate "still warming up" from "serving but dead".

    Differs from /ready: /ready is service-level (all-or-nothing across
    every model in the service); /health/pca is per-model + adds a
    freshness dimension that /ready does not track.
    """
    if not pca_loader.is_loaded():
        raise HTTPException(status_code=503, detail="model_not_loaded")
    if not inference_tracking.is_fresh(pca_loader):
        raise HTTPException(status_code=503, detail="inference_stale")
    return {
        "status": "healthy",
        "model_name": pca_loader.MODEL_NAME,
        "version": pca_loader.get_model_version(),
        "last_inference_age_seconds": round(
            inference_tracking.get_age(pca_loader), 2
        ),
        "staleness_threshold_seconds": (
            inference_tracking.get_staleness_threshold()
        ),
    }


# Prometheus metrics endpoint
"""
Plain GET that returns the current state of all registered metrics in
Prometheus exposition format. Prometheus servers scrape this endpoint
on a schedule (default 15s) and store the time series.

We return a raw Response (not a Pydantic model) because the body is
plain text in a specific format - Pydantic JSON serialization would
break it. CONTENT_TYPE_LATEST is "text/plain; version=0.0.4; charset=utf-8"
which is what Prometheus servers expect.

This endpoint is excluded from MetricsMiddleware instrumentation
(EXCLUDED_PATHS in middleware/metrics.py) so scrapes don't inflate
the very counters they're reading.
"""

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
        "`model_inference_duration_seconds` Histogram. See "
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
