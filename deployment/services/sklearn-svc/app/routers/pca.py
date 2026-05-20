"""
Router for the /predict/pca endpoint.

WHAT THIS FILE IS:
    The HTTP-facing layer for D1 PCA inference. One POST endpoint that:
        1. Receives a PCARequest (Pydantic-validated by FastAPI)
        2. Reshapes the features into a numpy array
        3. Calls the cached PCA's .transform()
        4. Returns a PCAResponse with the 150 components

    Routers are intentionally THIN. No business logic, no model loading,
    no I/O - those live in app/services/. This file is purely "HTTP <->
    Python objects" plumbing.

WHY APIRouter (and not @app.post directly in main.py):
    APIRouter lets us group related endpoints into their own module.
    main.py mounts the router via app.include_router(). Two benefits:
        1. main.py stays small - it's the wiring file, not the endpoint file
        2. Tests can import THIS router and test it standalone, without
           booting the whole app
    The pattern scales cleanly: pt-svc will have routers/dnn.py,
    routers/gan.py, routers/qlearning.py - same shape, different domain.

URL SHAPE:
    `prefix="/predict"` on the router + `@router.post("/pca")` on the
    handler means the final URL is `POST /predict/pca`. The split lets
    every endpoint in this service share the `/predict` namespace
    without each handler repeating the prefix.

DEFENSE IN DEPTH:
    The lifespan event in main.py guarantees the model is loaded before
    any request hits this handler. We still defensively catch RuntimeError
    from get_pca_model() and return 503 - same shape /ready uses for the
    same condition. "Should never fail" eventually does, at 3am.
"""

from fastapi import APIRouter, HTTPException

from app.middleware.inference_tracking import track_inference
from app.middleware.input_distribution import maybe_log_input_distribution
from app.schemas.pca import OUTPUT_DIM, PCARequest, PCAResponse
from app.services import pca_loader
from app.services.preprocessing import apply_preprocessing

"""
APIRouter() instances are like mini FastAPI apps. They collect routes
under a shared prefix + tag set, then get mounted on the main app via
app.include_router(router) in main.py.

  prefix="/predict" - all routes in this file are under /predict/...
  tags=["pca"]      - Swagger UI groups them under a "pca" heading,
                      separate from the "health" tag we put on /health
                      and /ready.
"""
router = APIRouter(prefix="/predict", tags=["pca"])


@router.post(
    "/pca",
    response_model=PCAResponse,
    operation_id="predict_pca",
    summary="Reduce a 784-dim Fashion-MNIST image to 150 PCA components",
    description=(
        "Accepts a flat 784-element list of raw Fashion-MNIST pixel values "
        "(uint8 range 0-255), applies the standardization scaler from the "
        "training pipeline (divide-by-255 + StandardScaler), then projects "
        "through the registered PCA. Returns the 150-component projection "
        "plus the model's cumulative explained-variance ratio (0.9085 on "
        "the training set). Inference latency is sub-millisecond for a "
        "single sample; network round-trip dominates total request time. "
        "The service owns all preprocessing - clients send raw pixels."
    ),
    responses={
        200: {
            "description": (
                "Successful projection. Body conforms to PCAResponse: 150 "
                "real-valued components + n_components echo + the loaded "
                "model's cumulative explained-variance ratio."
            ),
        },
        422: {
            "description": (
                "Request validation failed. Common causes: wrong feature "
                "count (must be exactly 784), pixel value outside [0, 255], "
                "non-numeric values in the features list, or missing body. "
                "Error body identifies the offending field + value."
            ),
        },
        503: {
            "description": (
                "Model not loaded. The lifespan startup either has not "
                "completed yet or failed to load the PCA from the registry. "
                "Same body shape as /ready returns for the same condition: "
                "`{\"detail\": \"model_not_loaded\"}`."
            ),
        },
    },
)
async def predict_pca(req: PCARequest) -> PCAResponse:
    """
    Apply the registered PCA to a single Fashion-MNIST flat image.

    Args:
        req: A PCARequest carrying `features` - 784 normalized pixels.
            FastAPI parses the JSON body into this model BEFORE this
            function runs; bad input never reaches us (HTTP 422 returned
            by Pydantic instead).

    Returns:
        PCAResponse with the 150-component projection plus the model's
        cumulative explained variance.

    Raises:
        HTTPException 503: If the PCA isn't loaded (extremely unlikely
            given the lifespan event, but kept as defense-in-depth).
    """
    # Defensive: pull the cached model + scaler. The except clause should
    # never trigger in production - lifespan loads both before we accept
    # requests. If it does, return 503 "model_not_loaded" matching /ready's
    # contract. We treat scaler-not-loaded the same way: from the client's
    # perspective the service is just not ready.
    try:
        model = pca_loader.get_pca_model()
        scaler = pca_loader.get_scaler()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="model_not_loaded") from exc

    # Sample the raw input distribution every Nth request (per-model
    # counter, env-var-configurable interval). Drift-detection signal -
    # logs summary stats of what clients are sending, before any
    # preprocessing transforms the values. See input_distribution.py
    # for the sampling design rationale.
    maybe_log_input_distribution(req.features, model_name=pca_loader.MODEL_NAME)

    """
    Apply the modeling-phase preprocessing pipeline:
      raw uint8 features -> /255 -> StandardScaler -> ready for PCA.
    `apply_preprocessing` lives in app/services/preprocessing.py - pure
    function, no I/O, easy to unit-test. Returns shape (1, INPUT_DIM)
    float32, which is exactly what PCA.transform expects.
    """
    X = apply_preprocessing(req.features, scaler)

    """
    The actual inference. transform() applies the learned linear
    projection: out = (X - pca.mean_) @ pca.components_.T
    For a single sample this is a few microseconds of matmul - the
    network round-trip dominates total request time, not the math.

    track_inference wraps two coupled side effects: observe the
    model_inference_duration_seconds histogram (measures the forward
    only, separate from preprocessing + serialization) AND stamp the
    loader's _LAST_INFERENCE_TS on successful exit (powers the
    /health/pca freshness check). A raised exception still observes
    the metric but skips the stamp - a failed inference must not keep
    the model looking "fresh".
    """
    with track_inference(pca_loader):
        components_array = model.transform(X)  # shape (1, OUTPUT_DIM)

    """
    Convert back to a plain Python list for JSON serialization.
    `[0]` selects the single batch row; `.tolist()` recursively
    converts numpy.float32 entries to Python floats (Pydantic + JSON
    don't natively understand numpy scalar types).
    """
    components: list[float] = components_array[0].tolist()

    """
    Construct the response Pydantic model. response_model=PCAResponse
    on the decorator means FastAPI re-validates this object before
    serializing - if we returned the wrong shape (e.g., mismatched
    length, missing field), the client gets a 500 instead of garbage.
    """
    return PCAResponse(
        components=components,
        n_components=OUTPUT_DIM,
        variance_explained=pca_loader.get_variance_explained(),
    )
