"""
Router for the /predict/qlearning/taxi endpoint.

WHAT THIS FILE IS:
    The HTTP-facing layer for tabular Q-learning Taxi-v4 inference.
    One POST handler:
        1. Receives a validated TaxiRequest (Pydantic enforced
           state in [0, 500) before this code runs)
        2. Indexes the cached Q-table by state ID
        3. Runs argmax over the 6-element Q-row to pick an action
        4. Builds a TaxiResponse with state echoed, action ID,
           action name, and the full Q-vector

    Routers stay THIN. The model lifecycle and registry resolution
    live in app/services/qlearning_loader.py. This file is purely
    the HTTP <-> Python objects boundary plus the index-and-argmax
    needed at that boundary.

URL SHAPE:
    `prefix="/predict"` on the router + `@router.post("/qlearning/taxi")`
    on the handler -> final URL is `POST /predict/qlearning/taxi`.
    The "/taxi" sub-path is explicit (rather than just "/qlearning")
    so future Q-learning endpoints (FrozenLake, CliffWalking, etc.)
    fit cleanly under the same `/qlearning/` namespace without
    renaming this one.

WHY THIS HANDLER IS THE SIMPLEST IN THE SERVICE:
    Every other router does substantial work between input and
    output: DNN runs StandardScaler -> matmul -> softmax; GAN runs
    randn -> 4 transposed convs -> denormalize -> PIL -> base64.
    This handler does ONE numpy operation: argmax of a 6-element
    vector. There's no preprocessing, no autograd, no encoding,
    no batching. The Q-table IS the policy.

DEFENSE IN DEPTH:
    The lifespan event guarantees the Q-table is loaded before any
    request. We still defensively catch RuntimeError from
    get_qtable() and return 503 - same shape /ready uses for the
    same condition.
"""

from typing import cast

import numpy as np
from fastapi import APIRouter, HTTPException

from app.middleware.metrics import MODEL_INFERENCE_DURATION_SECONDS
from app.schemas.qlearning import (
    ACTION_NAMES,
    ActionLabel,
    TaxiRequest,
    TaxiResponse,
)
from app.services import qlearning_loader

router = APIRouter(prefix="/predict", tags=["qlearning"])


@router.post(
    "/qlearning/taxi",
    response_model=TaxiResponse,
    summary="Pick the best action for a Gymnasium Taxi-v4 state",
)
async def predict_qlearning_taxi(req: TaxiRequest) -> TaxiResponse:
    """
    Look up the trained policy's action for a given state.

    Args:
        req: A TaxiRequest carrying `state` - an int in [0, 500).
            FastAPI parses + validates the JSON body via Pydantic
            BEFORE this function runs; out-of-range states never
            reach us (HTTP 422 returned by FastAPI instead).

    Returns:
        TaxiResponse with `state` (echoed), `action` (int 0-5,
        argmax of Q[state]), `action_name` (the human-readable
        Gymnasium action), and `q_values` (the full 6-element
        Q-vector for explainability).

    Raises:
        HTTPException 503: If the Q-table isn't loaded (extremely
            unlikely given the lifespan event; defense-in-depth so
            partial-load states surface cleanly).
    """
    # Defensive accessor pull. RuntimeError -> 503 mirrors the
    # contract /ready uses for the same "service not yet ready" state.
    try:
        qtable = qlearning_loader.get_qtable()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="model_not_loaded") from exc

    # Step 1+2: index the Q-table by state, then argmax over actions.
    # The .time() context manager observes elapsed seconds into the
    # model_inference_duration_seconds histogram - this is the cheapest
    # inference in the portfolio (one numpy index + one argmax over 6
    # floats), measured for consistency with the other endpoints. The
    # int() cast on argmax converts numpy int64 to plain Python int
    # for clean JSON serialization.
    with MODEL_INFERENCE_DURATION_SECONDS.labels(
        model_name=qlearning_loader.MODEL_NAME
    ).time():
        q_row = qtable[req.state]
        action = int(np.argmax(q_row))

    # Step 3: convert numpy floats to Python floats for the response.
    # .tolist() on a 1-d numpy array returns a list of plain floats,
    # which Pydantic + JSON serialize cleanly. Without this, numpy
    # float64 values would either raise serialization errors or
    # round-trip through their repr() form.
    q_values: list[float] = q_row.tolist()

    # Step 4: look up the human-readable action name. The cast to
    # ActionLabel narrows mypy's view: ACTION_NAMES is a tuple[str,
    # ...] so indexing returns plain str, but the schema's
    # action_name field requires the 6-string Literal. We know by
    # construction that action is in [0, 5] (argmax of a length-6
    # vector), so the lookup is safe.
    action_name = cast(ActionLabel, ACTION_NAMES[action])

    return TaxiResponse(
        state=req.state,
        action=action,
        action_name=action_name,
        q_values=q_values,
    )
