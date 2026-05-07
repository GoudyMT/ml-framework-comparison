"""
Tests for POST /predict/pca.

WHAT THESE TESTS COVER:
    Happy path:
        - 784 valid floats -> 200 with PCAResponse-shaped JSON
          (150 components, n_components=150, variance_explained in [0,1]).

    Validation rejections (HTTP 422 from Pydantic):
        - Wrong length (too short): Field min_length kicks in.
        - Wrong length (too long): Field max_length kicks in.
        - Out-of-range pixel: our @field_validator kicks in.
        - Wrong type: a non-number in the list.

    Loader-state rejection:
        - 503 when the model isn't loaded (uses client_unloaded fixture).

WHY 422 (not 400) FOR VALIDATION FAILURES:
    HTTP 422 "Unprocessable Entity" is the FastAPI/Pydantic convention
    for "the request was syntactically valid JSON but failed schema
    validation." Distinct from 400 ("malformed request" - bad JSON,
    missing Content-Type) and 404 ("route doesn't exist"). Clients can
    tell from the status code whether to fix their JSON or fix their
    data.

NOTE ON FAKEPCA OUTPUT:
    FakePCA.transform() returns shape (1, 150) where every entry is
    the row mean. For all-zeros input that's [0.0]*150; for all-0.5
    input that's [0.5]*150. Tests don't check exact values - they
    check shape + types + status codes. The actual math is sklearn's
    job; we mocked it out specifically to avoid that being part of
    these tests.
"""

from fastapi.testclient import TestClient

from app.schemas.pca import INPUT_DIM, OUTPUT_DIM

# Helper - building a 784-float request payload is something every
# test below does. Centralizing keeps tests focused on the assertion,
# not the setup.

def _features_payload(value: float = 0.0, length: int = INPUT_DIM) -> dict[str, list[float]]:
    """Return a {features: [...]} dict with `length` copies of `value`."""
    return {"features": [value] * length}


# Happy path


def test_predict_pca_happy_path(client: TestClient) -> None:
    """
    784 valid floats -> 200 with the right response shape.

    We don't check exact component values - those depend on FakePCA's
    transform implementation (just row_means tiled). What matters here
    is that the request flowed through validation, the handler called
    the cached model, and the response satisfied PCAResponse's schema.
    """
    # 100.0 is a typical raw pixel value (mid-tone). The schema accepts
    # [0, 255]; the service applies divide-by-255 + StandardScaler before PCA.
    response = client.post("/predict/pca", json=_features_payload(100.0))

    assert response.status_code == 200

    body = response.json()
    assert "components" in body
    assert "n_components" in body
    assert "variance_explained" in body

    # Length and types match the schema contract.
    assert len(body["components"]) == OUTPUT_DIM
    assert all(isinstance(c, float) for c in body["components"])
    assert body["n_components"] == OUTPUT_DIM
    # FakePCA in conftest sets _VARIANCE_EXPLAINED to 1.0.
    assert body["variance_explained"] == 1.0


# Validation rejections (Field constraints)


def test_predict_pca_too_few_features(client: TestClient) -> None:
    """List shorter than INPUT_DIM -> 422 from Field(min_length=...)."""
    response = client.post("/predict/pca", json=_features_payload(length=100))

    assert response.status_code == 422
    # Pydantic's error body has detail = list of error dicts. For our
    # length violation, the error type is "too_short". We don't pin the
    # exact message - that's Pydantic's wording and could change.
    detail = response.json()["detail"]
    assert any(err["type"] == "too_short" for err in detail)


def test_predict_pca_too_many_features(client: TestClient) -> None:
    """List longer than INPUT_DIM -> 422 from Field(max_length=...)."""
    response = client.post(
        "/predict/pca", json=_features_payload(length=INPUT_DIM + 1)
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "too_long" for err in detail)


# Validation rejections (custom @field_validator)


def test_predict_pca_pixel_out_of_range(client: TestClient) -> None:
    """
    Pixel > 255.0 -> 422 from our _features_in_pixel_range validator.

    Out-of-range values almost always mean the client sent already-
    normalized or already-standardized data. The validator's error
    message tells them to send raw uint8 pixels instead. We pin the
    actionable hint so future refactors don't accidentally drop it.
    """
    payload = _features_payload(0.0)
    payload["features"][42] = 300.0  # above the 255 raw-pixel ceiling

    response = client.post("/predict/pca", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]

    # Find the validation error for the features field.
    feature_errors = [err for err in detail if "features" in err["loc"]]
    assert feature_errors, f"Expected features error, got: {detail}"

    # Pin the actionable hint so the client-facing fix instruction
    # stays in place across refactors.
    msg = feature_errors[0]["msg"]
    assert "raw uint8 pixel values" in msg


def test_predict_pca_wrong_type(client: TestClient) -> None:
    """
    Non-number in features -> 422 from Pydantic's type coercion.

    Pydantic v2 will try to coerce a numeric string ('0.5' -> 0.5) but
    rejects truly non-numeric values like None or arbitrary strings.
    """
    payload = _features_payload(0.0)
    payload["features"][0] = "not-a-number" # type: ignore

    response = client.post("/predict/pca", json=payload)

    assert response.status_code == 422


# Loader-state rejection


def test_predict_pca_model_unloaded(client_unloaded: TestClient) -> None:
    """
    /predict/pca returns 503 when the model cache is empty.

    Defense-in-depth: the lifespan event in production guarantees a
    model is loaded before requests arrive. But if get_pca_model()
    raises (e.g., during a hot-reload window or pathological state),
    the router catches RuntimeError and returns 503 - same shape
    /ready returns for the same condition.
    """
    response = client_unloaded.post("/predict/pca", json=_features_payload(0.0))

    assert response.status_code == 503
    assert response.json() == {"detail": "model_not_loaded"}
