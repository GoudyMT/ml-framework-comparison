"""
Pydantic schemas for the /predict/dnn endpoint.

WHAT THIS FILE IS:
    The API contract for the PyTorch DNN endpoint (UCI HAR activity
    classification).
    Two classes:
        - DNNRequest:  what the client MUST send (validated on the way in)
        - DNNResponse: what the service WILL return (validated on the way out)

    FastAPI introspects these to validate incoming JSON, generate the
    OpenAPI schema for /docs, and serialize handler return values.

THE TRAINING PIPELINE THIS CONTRACT REFLECTS:
    raw UCI features [-1.0, 1.0]  (already normalized by the dataset publishers)
        -> StandardScaler.transform()   (fit on the training split)
        -> DNN(561 -> 256 -> 128 -> 6)  (logits)
        -> softmax                      (probabilities; argmax for class)

    The service applies StandardScaler internally - clients send the
    raw [-1, 1] features, we handle the rest. Keeps the API contract
    framework-agnostic: any client with UCI HAR-format input can call
    us correctly without knowing how the model was trained.

DESIGN DECISIONS:
    - STRICT [-1.0, 1.0] feature range. Out-of-range values almost
      certainly indicate the client sent the wrong data shape (e.g.,
      already-scaled values that overshoot). Reject loudly at the
      boundary.
    - Predicted label is a Literal type (not free-form str) - the API
      contract documents the exact 6-class enum, OpenAPI shows it to
      clients, and mypy catches typos at handler-construction time.
    - Probabilities are exposed (not just the predicted class) so
      callers can implement their own confidence thresholds, abstain
      on low-confidence inputs, or display top-k predictions.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Module constants
# ---------------------------------------------------------------------------
# Pulled from the modeling phase: data/processed/dnn/preprocessing_info.json
# and the trained network architecture (561 input -> 6 output).
#
# At module level (not inside a class) so the loader, the router, the
# tests, and the smoke test can all import them without re-deriving.

INPUT_DIM: int = 561                  # UCI HAR feature vector length
N_CLASSES: int = 6                    # 6 activities in the label space
FEATURE_MIN: float = -1.0             # UCI's pre-normalized lower bound
FEATURE_MAX: float = 1.0              # UCI's pre-normalized upper bound

# Activity labels in 0-indexed order (the modeling phase shifted UCI's
# original 1-6 labels down to 0-5). The order MATTERS - it's the order
# the network's output logits correspond to. Index i of the 6-vector
# probability output is the probability of CLASS_NAMES[i].
CLASS_NAMES: tuple[str, ...] = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)

# Literal type for the predicted_label field.
#
# Literal["A", "B", ...] declares "this value must be EXACTLY one of
# these strings". Pydantic validates it at runtime; mypy validates it
# statically; FastAPI exposes it in OpenAPI as an enum, so Swagger UI
# shows clients a dropdown of valid values.

# We could just type the field as `str` and rely on the handler to
# always produce a known label. Using Literal makes the contract
# self-enforcing - the schema rejects any return value that's not in
# the enum, even if the handler accidentally produced one.
ClassLabel = Literal[
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


# Request schema


class DNNRequest(BaseModel):
    """
    Client payload for POST /predict/dnn.

    Attributes:
        features: 561-element UCI HAR feature vector. Each value is a
            pre-normalized signal statistic in [-1.0, 1.0] (UCI's
            standard format - the dataset publishers normalize from
            raw accelerometer/gyroscope readings).

    Example payload (truncated):
        {"features": [0.288, -0.020, -0.132, ..., 0.179]}
    """

    features: list[float] = Field(
        ...,
        min_length=INPUT_DIM,
        max_length=INPUT_DIM,
        description=(
            f"UCI HAR feature vector: exactly {INPUT_DIM} pre-normalized "
            f"floats in [{FEATURE_MIN}, {FEATURE_MAX}]. The service applies "
            "StandardScaler internally - send the values as UCI provides them."
        ),
        examples=[[0.0] * INPUT_DIM],
    )

    @field_validator("features")
    @classmethod
    def _features_in_uci_range(cls, value: list[float]) -> list[float]:
        """
        Reject feature values outside UCI HAR's documented [-1.0, 1.0] range.

        Out-of-range values almost always mean the client sent the wrong
        data shape - typically already-StandardScaler'd output (range
        roughly [-3, +3]) or unnormalized raw sensor readings. Rejecting
        loudly at the boundary prevents silent classification on garbage
        input.
        """
        # next() with a generator finds the first offender lazily -
        # we don't scan all 561 elements once a bad one is found.
        bad = next(
            (
                (i, v) for i, v in enumerate(value)
                if not (FEATURE_MIN <= v <= FEATURE_MAX)
            ),
            None,
        )
        if bad is not None:
            idx, val = bad
            raise ValueError(
                f"features[{idx}] = {val} is out of range; each feature "
                f"must be a UCI HAR pre-normalized value in "
                f"[{FEATURE_MIN}, {FEATURE_MAX}]. The service applies "
                f"StandardScaler internally - do not pre-scale the input."
            )
        return value


# Response schema


class DNNResponse(BaseModel):
    """
    Service payload returned from POST /predict/dnn.

    Attributes:
        predicted_class: Integer class index in [0, 5]. The argmax of
            the network's softmax output. Matches CLASS_NAMES[i].
        predicted_label: Human-readable class string. Constrained to
            the exact 6-activity enum - clients can switch on this
            without worrying about case or typos.
        probabilities: Per-class softmax probabilities, length 6, in
            CLASS_NAMES order. Each in [0, 1]; the whole vector sums
            to ~1.0 (float precision). Exposing the full distribution
            lets callers apply their own confidence thresholds or
            display top-k predictions instead of just the argmax.

    Example payload:
        {
          "predicted_class": 4,
          "predicted_label": "STANDING",
          "probabilities": [0.001, 0.002, 0.001, 0.012, 0.972, 0.012]
        }
    """

    predicted_class: int = Field(
        ...,
        ge=0,
        le=N_CLASSES - 1,
        description=(
            f"Predicted activity class index in [0, {N_CLASSES - 1}]. "
            "Argmax of the softmax output. Use predicted_label for the "
            "human-readable name."
        ),
    )

    predicted_label: ClassLabel = Field(
        ...,
        description=(
            "Predicted activity name. Always one of the 6 UCI HAR classes."
        ),
    )

    probabilities: list[float] = Field(
        ...,
        min_length=N_CLASSES,
        max_length=N_CLASSES,
        description=(
            f"Per-class softmax probabilities in CLASS_NAMES order "
            f"(length {N_CLASSES}). Each value in [0, 1]; full vector "
            "sums to ~1.0."
        ),
    )

    @field_validator("probabilities")
    @classmethod
    def _probabilities_in_unit_range(cls, value: list[float]) -> list[float]:
        """
        Each probability must be in [0, 1].

        Pydantic's Field doesn't constrain individual list elements
        (only the list itself), so this catches misuse - e.g., a future
        bug where the handler returns logits instead of post-softmax
        probabilities. Defensive: we'd rather reject our own malformed
        response than silently return values >1 to clients.
        """
        bad = next(
            ((i, v) for i, v in enumerate(value) if not (0.0 <= v <= 1.0)),
            None,
        )
        if bad is not None:
            idx, val = bad
            raise ValueError(
                f"probabilities[{idx}] = {val} is out of [0, 1] range. "
                "This indicates a handler bug - softmax output should "
                "always be in [0, 1]."
            )
        return value
