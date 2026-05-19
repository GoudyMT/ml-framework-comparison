"""
Router for the /predict/dnn endpoint.

WHAT THIS FILE IS:
    The HTTP-facing layer for DNN inference. One POST handler:
        1. Receives a validated DNNRequest (Pydantic enforced 561 floats
           in [-1, 1] before this code runs)
        2. Standardizes via the cached StandardScaler
        3. Runs the cached DNN under torch.no_grad()
        4. Builds a DNNResponse with predicted_class, predicted_label,
           and per-class probabilities

    Routers stay THIN. The model lifecycle, registry resolution, and
    architecture all live in app/services/. This file is purely the
    HTTP <-> Python objects boundary plus the data-shape conversions
    needed at that boundary (list -> ndarray -> tensor -> ndarray -> list).

URL SHAPE:
    `prefix="/predict"` on the router + `@router.post("/dnn")` on the
    handler -> final URL is `POST /predict/dnn`. The other endpoints
    in this service (/predict/gan/sample, /predict/qlearning/taxi)
    share the same /predict prefix - the prefix lives on the router
    instead of being repeated on every handler decorator.

INFERENCE PATTERN - torch.no_grad():
    The model.eval() that the loader applied switches BatchNorm +
    Dropout to inference mode, but autograd is still tracking
    operations on every forward pass by default. That's wasted work
    at inference - we'll never call .backward(). Wrapping the forward
    in `with torch.no_grad():` skips the autograd bookkeeping,
    saving roughly 10-20% on compute and a noticeable amount of
    memory (autograd builds a computation graph; no_grad tells it
    not to). Standard pattern around any inference call.

DEFENSE IN DEPTH:
    The lifespan event guarantees the model + scaler are loaded
    before any request. We still defensively catch RuntimeError
    from the accessors and return 503 - same shape /ready uses for
    the same condition. "Should never fail" eventually does, at 3am.
"""

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from app.middleware.inference_tracking import track_inference
from app.middleware.input_distribution import maybe_log_input_distribution
from app.schemas.dnn import CLASS_NAMES, ClassLabel, DNNRequest, DNNResponse
from app.services import dnn_loader

router = APIRouter(prefix="/predict", tags=["dnn"])


@router.post(
    "/dnn",
    response_model=DNNResponse,
    summary="Classify a 561-feature UCI HAR sample into one of 6 activities",
    description=(
        "Accepts a 561-element UCI HAR feature vector (each value in "
        "[-1, 1], the dataset's pre-normalized range), applies the "
        "StandardScaler from training, then runs the registered DNN "
        "(561 -> 256 -> 128 -> 6 logits) under `torch.no_grad()`. "
        "Returns the predicted class index + human-readable label + "
        "the full 6-element softmax probability distribution so clients "
        "can apply their own confidence thresholds or display top-k "
        "predictions. The 6 classes are WALKING / WALKING_UPSTAIRS / "
        "WALKING_DOWNSTAIRS / SITTING / STANDING / LAYING. Inference "
        "is sub-millisecond per request."
    ),
    responses={
        200: {
            "description": (
                "Successful classification. Body conforms to DNNResponse: "
                "predicted_class index + predicted_label string + "
                "probabilities list (length 6, sums to ~1.0)."
            ),
        },
        422: {
            "description": (
                "Request validation failed. Common causes: wrong feature "
                "count (must be exactly 561), feature value outside "
                "[-1, 1] (UCI's pre-normalized range), or non-numeric "
                "values in the features list."
            ),
        },
        503: {
            "description": (
                "DNN or scaler not loaded; lifespan startup hasn't "
                "completed. Body: `{\"detail\": \"model_not_loaded\"}`."
            ),
        },
    },
)
async def predict_dnn(req: DNNRequest) -> DNNResponse:
    """
    Run the registered DNN on a single UCI HAR feature vector.

    Args:
        req: A DNNRequest carrying `features` - 561 pre-normalized
            UCI floats in [-1, 1]. FastAPI parses + validates the JSON
            body via Pydantic BEFORE this function runs; bad input
            never reaches us (HTTP 422 returned by FastAPI instead).

    Returns:
        DNNResponse with predicted_class (int 0-5), predicted_label
        (string from the 6-activity enum), and probabilities (length-6
        softmax output in CLASS_NAMES order).

    Raises:
        HTTPException 503: If the DNN or scaler isn't loaded
            (extremely unlikely given the lifespan event; defense-in-
            depth so partial-load states surface cleanly).
    """
    # Defensive accessor pulls. Either accessor can raise RuntimeError
    # if its cache slot is None; we translate to 503 matching /ready's
    # contract. Catching both with one try/except keeps the response
    # consistent regardless of which artifact is missing.
    try:
        model = dnn_loader.get_dnn_model()
        scaler = dnn_loader.get_scaler()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="model_not_loaded") from exc

    # Sample the raw input distribution every Nth request (per-model
    # counter, env-var-configurable interval). Drift-detection signal -
    # logs summary stats of what clients are sending, before any
    # preprocessing transforms the values. See input_distribution.py
    # for the sampling design rationale.
    maybe_log_input_distribution(req.features, model_name=dnn_loader.MODEL_NAME)

    # Step 1: list -> 2D ndarray.
    # Sklearn's transform expects (batch, n_features). reshape(1, -1)
    # makes a single-row batch; the -1 means "infer from data" (= 561).
    # dtype=float32 matches both the trained network's parameter dtype
    # and the scaler's internal precision - no implicit upcasting.
    X = np.asarray(req.features, dtype=np.float32).reshape(1, -1)

    # Step 2: apply the StandardScaler. The fitted scaler subtracts
    # per-feature means and divides by per-feature stds (both vectors
    # of length 561 from training). After this, X has the same
    # distribution the network was trained on.
    X_scaled = scaler.transform(X)

    # Step 3: numpy -> torch.Tensor + run inference.
    # from_numpy shares memory with the source array (zero copy);
    # .float() ensures dtype is float32 (some older sklearn versions
    # return float64). no_grad disables autograd tracking - faster
    # and leaner without changing the math.
    #
    # track_inference wraps two coupled side effects: observe the
    # model_inference_duration_seconds histogram (covers the forward +
    # softmax + argmax; excludes scaler.transform above and response
    # prep below) AND stamp dnn_loader._LAST_INFERENCE_TS on successful
    # exit (powers the /health/dnn freshness check). A raised exception
    # still observes the metric but skips the stamp.
    X_tensor = torch.from_numpy(X_scaled).float()
    with track_inference(dnn_loader), torch.no_grad():
        logits = model(X_tensor)               # shape (1, 6)
        probabilities = torch.softmax(logits, dim=1)[0]  # shape (6,)
        predicted_class = int(logits.argmax(dim=1).item())

    # Step 4: tensor -> Python primitives for JSON serialization.
    # .tolist() recursively converts torch numeric types to Python
    # int/float (Pydantic + JSON don't natively understand torch
    # scalars). Index lookup gives the human-readable label - the
    # Literal type on DNNResponse.predicted_label re-validates that
    # the value is one of the 6 known classes.
    predicted_label: ClassLabel = CLASS_NAMES[predicted_class]  # type: ignore[assignment]
    probabilities_list: list[float] = probabilities.tolist()

    # response_model=DNNResponse on the decorator means FastAPI
    # re-validates this object before serializing - if the handler
    # ever returned the wrong shape, the client would get a 500
    # instead of garbage. Defense in depth complementing Pydantic's
    # input validation.
    return DNNResponse(
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        probabilities=probabilities_list,
    )
