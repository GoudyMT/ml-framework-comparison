"""
Tests for POST /predict/dnn.

WHAT THESE TESTS COVER:
    Happy path:
        - 561 valid floats -> 200 with DNNResponse-shaped JSON
          (predicted_class, predicted_label, probabilities length 6).

    Validation rejections (HTTP 422 from Pydantic):
        - Wrong length (too short): Field min_length=561 kicks in.
        - Wrong length (too long): Field max_length=561 kicks in.
        - Out-of-range feature: our @field_validator kicks in.
        - Wrong type: a non-number in the features list.

    Loader-state rejection:
        - 503 when the model + scaler aren't loaded (uses
          client_unloaded fixture).

NOTE ON FAKEDNN OUTPUT:
    FakeDNN.forward() returns logits with index 3 set to 5.0 and the
    rest 0.0. After softmax, that becomes a high probability on
    class 3 (SITTING per CLASS_NAMES order). Tests assert on this
    deterministic mapping, NOT on the actual classification quality
    of a real DNN - that's verified end-to-end by scripts/smoke_test.py
    against the real registry artifact.
"""

from fastapi.testclient import TestClient

from app.schemas.dnn import INPUT_DIM, N_CLASSES

# Helper - building a 561-float payload is repeated in every test.

def _features_payload(value: float = 0.0, length: int = INPUT_DIM) -> dict[str, list[float]]:
    """Return a {features: [...]} dict with `length` copies of `value`."""
    return {"features": [value] * length}


# Happy path


def test_predict_dnn_happy_path(client: TestClient) -> None:
    """
    561 valid floats -> 200 with the right response shape.

    FakeDNN's deterministic logits (index 3 = 5.0, rest = 0.0)
    produce predicted_class=3 and predicted_label='SITTING' regardless
    of input, so the assertions are stable. Real classification
    quality is verified by the smoke test against the registered model.
    """
    response = client.post("/predict/dnn", json=_features_payload(0.0))

    assert response.status_code == 200

    body = response.json()
    assert "predicted_class" in body
    assert "predicted_label" in body
    assert "probabilities" in body

    # FakeDNN biases toward class 3 (SITTING).
    assert body["predicted_class"] == 3
    assert body["predicted_label"] == "SITTING"

    # Probabilities shape contract.
    assert len(body["probabilities"]) == N_CLASSES
    assert all(isinstance(p, float) for p in body["probabilities"])
    # Softmax outputs sum to ~1.0 (float precision).
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-5
    # Each in [0, 1].
    assert all(0.0 <= p <= 1.0 for p in body["probabilities"])


# Validation rejections (Field constraints)


def test_predict_dnn_too_few_features(client: TestClient) -> None:
    """List shorter than INPUT_DIM -> 422 from Field(min_length=...)."""
    response = client.post("/predict/dnn", json=_features_payload(length=100))

    assert response.status_code == 422
    detail = response.json()["detail"]
    # Pydantic's error type for "list too short" is 'too_short'. We
    # don't pin the exact message - that's Pydantic's wording and
    # could change across versions.
    assert any(err["type"] == "too_short" for err in detail)


def test_predict_dnn_too_many_features(client: TestClient) -> None:
    """List longer than INPUT_DIM -> 422 from Field(max_length=...)."""
    response = client.post(
        "/predict/dnn", json=_features_payload(length=INPUT_DIM + 1)
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "too_long" for err in detail)


# Validation rejections (custom @field_validator)


def test_predict_dnn_feature_out_of_range(client: TestClient) -> None:
    """
    Feature > 1.0 -> 422 from our _features_in_uci_range validator.

    The custom validator's ValueError text is part of our API contract
    (it tells clients NOT to pre-scale - the service applies
    StandardScaler internally). We pin the actionable hint so future
    refactors don't accidentally drop it.
    """
    payload = _features_payload(0.0)
    payload["features"][42] = 5.0  # Looks like already-scaled output

    response = client.post("/predict/dnn", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]

    # Find the validation error for the features field.
    feature_errors = [err for err in detail if "features" in err["loc"]]
    assert feature_errors, f"Expected features error, got: {detail}"

    # Pin the actionable hint so the client-facing fix instruction
    # stays in place across refactors.
    msg = feature_errors[0]["msg"]
    assert "do not pre-scale" in msg


def test_predict_dnn_wrong_type(client: TestClient) -> None:
    """
    Non-number in features -> 422 from Pydantic's type coercion.

    Pydantic v2 will try to coerce a numeric string ('0.5' -> 0.5) but
    rejects truly non-numeric values like None or arbitrary strings.
    """
    payload = _features_payload(0.0)
    payload["features"][0] = "not-a-number"  # type: ignore[call-overload]

    response = client.post("/predict/dnn", json=payload)

    assert response.status_code == 422


# Loader-state rejection


def test_predict_dnn_model_unloaded(client_unloaded: TestClient) -> None:
    """
    /predict/dnn returns 503 when the model + scaler cache is empty.

    Defense-in-depth: the lifespan event in production guarantees a
    loaded model before requests arrive. But if get_dnn_model() or
    get_scaler() raises (e.g., during a hot-reload window or
    pathological state), the router catches RuntimeError and returns
    503 - same shape /ready returns for the same condition.
    """
    response = client_unloaded.post("/predict/dnn", json=_features_payload(0.0))

    assert response.status_code == 503
    assert response.json() == {"detail": "model_not_loaded"}


# Inference-latency histogram


def test_predict_dnn_records_inference_duration(client: TestClient) -> None:
    """
    A successful /predict/dnn records the model_inference_duration_seconds
    histogram with model_name="pt-dnn".

    Verifies the .time() context manager wrapping the no_grad forward in
    the router emits a labeled sample. The substring asserted below is
    the `_count` line prometheus_client auto-generates for any Histogram
    with at least one observation; without the wrap, this substring
    never appears in /metrics.
    """
    response = client.post("/predict/dnn", json=_features_payload(0.0))
    assert response.status_code == 200, "setup precondition failed"

    body = client.get("/metrics").text

    assert (
        'model_inference_duration_seconds_count{model_name="pt-dnn"}' in body
    )
