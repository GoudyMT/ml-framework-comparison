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

WHAT THIS FILE CONTAINS NOW (after Step 1.2):
    - The FastAPI() instance with metadata for OpenAPI/Swagger docs
    - Two health-check endpoints: /health (liveness) + /ready (readiness)
    - NO ML routes yet - those come in Step 1.5

WHAT WILL BE ADDED LATER:
    - Step 1.5: mount the /predict/pca router from app.routers.pca
    - Step 1.6: wire in middleware (request ID, structured logging, metrics)
    - Step 1.4: ready endpoint will report whether the PCA model is loaded
"""

from fastapi import FastAPI

# The FastAPI() constructor accepts metadata that powers the auto-generated
# OpenAPI/Swagger UI at /docs. Things to know:
#
#   title         = human-readable service name; shown at top of /docs
#   version       = API version. Separate from the package version in
#                   pyproject.toml because APIs can evolve independently
#                   of internal code.
#   description   = markdown-supported overview text. Shows under the title.
#   docs_url      = path for Swagger UI (default /docs)
#   redoc_url     = path for ReDoc UI (default /redoc) - alternative renderer
#   openapi_url   = path for raw openapi.json schema (default /openapi.json)
#                   This JSON is what Swagger UI and ReDoc render from.

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
)


# Health check endpoints
# Two distinct concepts that orchestrators (Docker, k8s, load balancers)
# query independently. See module-level comments for the full explanation.
#
# We tag both with `tags=["health"]` so they appear grouped together in the
# Swagger UI at /docs - cleaner than having them inline with the ML routes.


@app.get("/health", tags=["health"])
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


@app.get("/ready", tags=["health"])
async def ready() -> dict[str, str]:
    """
    Readiness check - the service can accept traffic.

    Returns:
        {"status": "ready", ...} with HTTP 200 when ready;
        will return 503 when the model isn't loaded yet (added in Step 1.4).

    Differs from /health: a service starting up is ALIVE but not READY.
    During the ~2 seconds it takes to load the PCA model on startup,
    /health returns 200 (process is alive) but /ready should return 503
    (don't send real traffic yet).

    Used by:
        - Kubernetes `readinessProbe` - controls whether the pod receives
          traffic from the Service load balancer
        - Rolling-deploy systems - wait for /ready before draining old pods
    """
    # In Step 1.4, this will check whether the PCA model has been loaded
    # into memory and return 503 if not. For now we just say "ready"
    # since there's no model to wait for.
    return {"status": "ready", "model_loaded": "not yet (Step 1.4 adds this)"}