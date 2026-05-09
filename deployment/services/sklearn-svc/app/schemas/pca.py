"""
Pydantic schemas for the /predict/pca endpoint.

WHAT THIS FILE IS:
    The API contract for D1 (SK PCA, Fashion-MNIST). Two classes:
        - PCARequest:  what the client MUST send (validated on the way in)
        - PCAResponse: what the service WILL return (validated on the way out)

    These schemas are the single source of truth for the endpoint's shape.
    FastAPI introspects them to:
        1. Validate incoming JSON       -> auto HTTP 422 on bad input
        2. Generate the OpenAPI schema  -> /docs Swagger UI
        3. Type-check the handler       -> mypy catches mismatches
        4. Serialize outgoing responses -> dict -> JSON with correct types

WHY SCHEMAS LIVE IN THEIR OWN FILE (and their own package):
    Separation of concerns. Routers handle HTTP plumbing, services handle
    business logic, schemas define the contract. Keeping them apart means:
        - Schemas can be imported by tests, docs, or client SDKs without
          dragging in FastAPI / sklearn / mlflow as dependencies
        - Contract changes are reviewed in isolation - one file, one purpose
        - The package scales: each endpoint added in the future gets its
          own schema file alongside this one

PYDANTIC v2 NOTES (we pinned >=2.5):
    - BaseModel is the parent class for all schemas
    - Field(...) attaches metadata: descriptions, constraints, examples
    - @field_validator runs custom checks beyond Field's built-in constraints
    - model_config replaces v1's `class Config:` inner class
    - The Rust core makes v2 ~10-50x faster than v1 (matters at high RPS)

DESIGN DECISIONS LOCKED FOR THIS ENDPOINT:
    - Pixel range: STRICT 0.0 <= x <= 255.0 (raw uint8 pixel range).
      The SERVICE owns preprocessing - clients send raw pixels, we
      apply the modeling pipeline (divide-by-255 -> StandardScaler -> PCA)
      in app/services/preprocessing.py. Out-of-range values still get
      rejected loudly (likely indicate a client sent the wrong data).
    - No request echo in response. Clients correlate via the
      X-Request-ID header (set by the request_id middleware on every
      response) and the matching field in structured logs.
"""

from pydantic import BaseModel, Field, field_validator

# Constants tied to the trained PCA model (Fashion-MNIST).
# Pulled from the modeling phase artifacts:
#   - Input  = 28 x 28 = 784 pixels (flat row-major)
#   - Output = 150 components (n_components=150 at fit time)
#   - Variance explained = 0.9085 cumulative

# These are at module level (not inside a class) so they can be imported by
# the loader, the router, and the tests without re-deriving them.

INPUT_DIM: int = 784           # Fashion-MNIST flattened image length
OUTPUT_DIM: int = 150          # PCA n_components fit at training time
# Raw pixel range. Clients send unnormalized 0-255 values - the SERVICE
# does the divide-by-255 + StandardScaler in app/services/preprocessing.py.
# This API contract means clients don't have to know preprocessing details:
# any client with raw pixel data can call us correctly.
PIXEL_MIN: float = 0.0
PIXEL_MAX: float = 255.0


# Step 1: Request schema
# ----------------------
# This is what the CLIENT sends in the POST body. FastAPI sees a parameter
# typed `PCARequest`, parses the request body as JSON, and constructs a
# PCARequest instance from it. If construction fails (missing field, wrong
# type, validator raised), FastAPI returns HTTP 422 with a JSON error body
# explaining which field broke and why - all without the handler running.

class PCARequest(BaseModel):
    """
    Client payload for POST /predict/pca.

    Attributes:
        features: Flattened 28x28 Fashion-MNIST image as a list of 784
            raw pixel values in [0.0, 255.0]. Row-major order (the same
            order numpy's `arr.flatten()` produces). The service applies
            the modeling-phase preprocessing (divide-by-255 + StandardScaler)
            internally - send pixels as-is.

    Example payload (truncated):
        {"features": [0.0, 0.0, ..., 200.0, 232.0, ..., 0.0]}
    """

    features: list[float] = Field(
        ...,
        min_length=INPUT_DIM,
        max_length=INPUT_DIM,
        description=(
            f"Flattened {INPUT_DIM}-element Fashion-MNIST image. "
            f"Each value must be a raw pixel in [{PIXEL_MIN}, {PIXEL_MAX}]. "
            "Row-major order (same as numpy's flatten()). The service "
            "handles all preprocessing - no client-side normalization needed."
        ),
        examples=[[0.0] * INPUT_DIM],  # Swagger shows a 784-zero example
    )

    # @field_validator runs AFTER Field's built-in checks pass.
    # We use it for the per-element range check, which Field can't express
    # (Field constrains the LIST, not its elements).
    @field_validator("features")
    @classmethod
    def _features_in_pixel_range(cls, value: list[float]) -> list[float]:
        """
        Reject values outside the raw pixel range [0.0, 255.0].

        Out-of-range values almost certainly indicate the client sent
        wrong-shape data (e.g., already-normalized 0-1 floats, or
        already-standardized features). Better to fail loudly here
        than silently produce nonsense components.
        """
        # next() with a generator finds the first offender lazily - we
        # don't need to scan all 784 elements once we've found one bad value.
        bad = next(
            ((i, v) for i, v in enumerate(value) if not (PIXEL_MIN <= v <= PIXEL_MAX)),
            None,
        )
        if bad is not None:
            idx, val = bad
            raise ValueError(
                f"features[{idx}] = {val} is out of range; each pixel must "
                f"be a raw value in [{PIXEL_MIN}, {PIXEL_MAX}]. "
                f"The service applies normalization internally - send raw "
                f"uint8 pixel values, not pre-normalized floats."
            )
        return value


# Step 2: Response schema
# -----------------------
# This is what the SERVICE returns. The handler builds a PCAResponse and
# returns it; FastAPI serializes it to JSON automatically. Defining it as
# a Pydantic model (instead of just `dict`) gives us:
#   - Type-checked construction (mypy catches missing/wrong fields)
#   - Auto-documented response schema in /docs
#   - Consistent serialization rules (e.g., floats stay floats, no surprise
#     numpy.float64 -> "0.78321..." string conversions)

class PCAResponse(BaseModel):
    """
    Service payload returned from POST /predict/pca.

    Attributes:
        components: PCA-transformed vector of length 150. Each value is a
            principal component coordinate (NOT bounded - components are
            real-valued, can be negative, no fixed range).
        n_components: Echo of the component count (= 150). Lets clients
            self-verify dimensionality without re-counting the list.
        variance_explained: Cumulative explained variance ratio of the
            loaded PCA model (= 0.9085 for D1). Constant per model load,
            but echoed in every response so monitoring / logs can track
            which model version produced the prediction without a separate
            registry call.

    Example payload (truncated):
        {"components": [-1.23, 0.87, ...],
         "n_components": 150,
         "variance_explained": 0.9085}
    """

    components: list[float] = Field(
        ...,
        min_length=OUTPUT_DIM,
        max_length=OUTPUT_DIM,
        description=(
            f"PCA-transformed coordinates: {OUTPUT_DIM} principal components. "
            "Real-valued, unbounded, can be negative."
        ),
    )

    n_components: int = Field(
        ...,
        ge=OUTPUT_DIM,
        le=OUTPUT_DIM,
        description=f"Number of components in the response (always {OUTPUT_DIM}).",
    )

    # variance_explained is a ratio in [0, 1]. We don't pin it to exactly
    # 0.9085 because the model could be retrained later with a different
    # cumulative variance, and we don't want the schema to reject a
    # legitimately-loaded new model.
    variance_explained: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Cumulative explained variance ratio of the loaded PCA model. "
            "Property of the model, not the request - same value for every "
            "response from a given model load."
        ),
    )
