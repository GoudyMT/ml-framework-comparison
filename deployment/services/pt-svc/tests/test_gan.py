"""
Tests for POST /predict/gan/sample.

WHAT THESE TESTS COVER:
    Happy path:
        - n_samples=1, no seed -> 200 with GANResponse-shaped JSON
          (images list of len 1, generation_time_ms >= 0, seed echoed).
        - n_samples=16 (upper boundary of the schema constraint).
        - n_samples omitted -> defaults to 1 via Pydantic Field default.

    Validation rejections (HTTP 422 from Pydantic Field bounds):
        - n_samples=0 -> Field ge=1 violation ('greater_than_equal').
        - n_samples=17 -> Field le=16 violation ('less_than_equal').
        - n_samples=-5 -> sanity, also < ge=1.

    Seed plumbing:
        - Same seed produces identical image bytes across calls
          (proves torch.manual_seed -> torch.randn determinism).
        - No seed produces different bytes across calls (proves the
          RNG advances by default).
        - Seed is echoed back in the response (or null if omitted).

    Image content:
        - Each base64 string decodes to a valid 32x32 RGB PNG (proves
          the full encode pipeline: tensor -> uint8 -> PIL -> PNG ->
          base64 -> string).

    Loader-state rejection:
        - 503 when the model cache is empty (uses client_unloaded
          fixture).

NOTE ON FAKEGENERATOR DETERMINISM:
    conftest.py's FakeGenerator.forward(z) is z-dependent:
        first3 = z[:, :3, :, :]
        return torch.tanh(first3.expand(-1, -1, 32, 32))
    Different z -> different output; same z -> identical output to
    the bit. This is what the seed-reproducibility test relies on -
    if FakeGenerator ignored z, the test would pass even if the
    router's seed plumbing was broken. With z-dependence, identical
    request bytes only result when torch.manual_seed() is being
    called correctly in the router.
"""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.schemas.gan import IMAGE_CHANNELS, IMAGE_SIZE, N_SAMPLES_MAX
from app.services import gan_loader

# Helper - decode a single base64 string to PIL.Image. Used by tests
# that need to validate the actual image bytes.

def _decode_base64_to_image(b64_string: str) -> Image.Image:
    """Decode a base64 PNG string into a PIL.Image for inspection."""
    return Image.open(io.BytesIO(base64.b64decode(b64_string)))


# Happy path


def test_predict_gan_happy_path_n_samples_1(client: TestClient) -> None:
    """
    n_samples=1 with a seed -> 200 with the right response shape.

    Asserts the structural contract: images is a list of one base64
    string, generation_time_ms is a non-negative float, seed is the
    echoed integer.
    """
    response = client.post(
        "/predict/gan/sample", json={"n_samples": 1, "seed": 42}
    )

    assert response.status_code == 200

    body = response.json()
    assert "images" in body
    assert "generation_time_ms" in body
    assert "seed" in body

    assert isinstance(body["images"], list)
    assert len(body["images"]) == 1
    assert isinstance(body["images"][0], str)
    assert len(body["images"][0]) > 0          # non-empty base64

    assert isinstance(body["generation_time_ms"], int | float)
    assert body["generation_time_ms"] >= 0.0

    assert body["seed"] == 42


def test_predict_gan_happy_path_n_samples_max(client: TestClient) -> None:
    """
    n_samples=16 (upper boundary) -> 200 with 16 images.

    Boundary check: the schema's le=16 should ALLOW exactly 16 (not
    reject it). The list must have exactly that many entries; each
    must be non-empty.
    """
    response = client.post(
        "/predict/gan/sample", json={"n_samples": N_SAMPLES_MAX, "seed": 0}
    )

    assert response.status_code == 200

    body = response.json()
    assert len(body["images"]) == N_SAMPLES_MAX
    assert all(isinstance(s, str) and len(s) > 0 for s in body["images"])
    assert body["seed"] == 0


def test_predict_gan_default_n_samples(client: TestClient) -> None:
    """
    n_samples omitted -> Pydantic default=1 applies.

    Verifies the schema's `default=1` actually works in the request
    pipeline (some Pydantic v2 misconfigurations can drop defaults).
    """
    response = client.post("/predict/gan/sample", json={"seed": 7})

    assert response.status_code == 200

    body = response.json()
    assert len(body["images"]) == 1
    assert body["seed"] == 7


# Validation rejections (Field bounds)


def test_predict_gan_n_samples_zero_rejected(client: TestClient) -> None:
    """n_samples=0 -> 422 from Field(ge=1)."""
    response = client.post("/predict/gan/sample", json={"n_samples": 0})

    assert response.status_code == 422
    detail = response.json()["detail"]
    # Pydantic's error type for 'value < ge' is 'greater_than_equal'.
    assert any(err["type"] == "greater_than_equal" for err in detail)


def test_predict_gan_n_samples_over_max_rejected(client: TestClient) -> None:
    """n_samples=17 -> 422 from Field(le=16)."""
    response = client.post(
        "/predict/gan/sample", json={"n_samples": N_SAMPLES_MAX + 1}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "less_than_equal" for err in detail)


def test_predict_gan_n_samples_negative_rejected(client: TestClient) -> None:
    """n_samples=-5 -> 422 from Field(ge=1) (sanity)."""
    response = client.post("/predict/gan/sample", json={"n_samples": -5})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "greater_than_equal" for err in detail)


# Seed plumbing


def test_predict_gan_seed_reproducibility(client: TestClient) -> None:
    """
    Same seed across two calls -> identical image bytes.

    Proves the seed plumbing: torch.manual_seed(seed) is called BEFORE
    torch.randn(), resetting the RNG state. Same z -> same FakeGenerator
    output -> same denorm -> same PNG bytes -> same base64 string.

    If the router forgot to seed (or seeded after randn), this test
    would fail because the second call would inherit advanced RNG state
    and produce a different z.
    """
    payload = {"n_samples": 1, "seed": 42}

    r1 = client.post("/predict/gan/sample", json=payload)
    r2 = client.post("/predict/gan/sample", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["images"][0] == r2.json()["images"][0]


def test_predict_gan_seed_different_seeds_differ(client: TestClient) -> None:
    """
    Different seeds -> different image bytes.

    Sanity complement to the reproducibility test: we shouldn't get
    identical output for non-identical input. If this passes when the
    reproducibility test also passes, we have BOTH directions covered:
    same-seed -> same; different-seed -> different.
    """
    r1 = client.post("/predict/gan/sample", json={"n_samples": 1, "seed": 42})
    r2 = client.post("/predict/gan/sample", json={"n_samples": 1, "seed": 43})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["images"][0] != r2.json()["images"][0]


def test_predict_gan_no_seed_produces_different_outputs(
    client: TestClient,
) -> None:
    """
    Two unseeded calls produce DIFFERENT bytes.

    Relies on torch.randn advancing the global RNG state by default.
    If the router accidentally always seeded (e.g., bug like
    torch.manual_seed(0) when seed is None), every unseeded call
    would produce the same image and this test would fail.
    """
    payload: dict[str, int] = {"n_samples": 1}

    r1 = client.post("/predict/gan/sample", json=payload)
    r2 = client.post("/predict/gan/sample", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["images"][0] != r2.json()["images"][0]


def test_predict_gan_seed_echoed_when_provided(client: TestClient) -> None:
    """seed=42 in request -> seed=42 in response."""
    response = client.post(
        "/predict/gan/sample", json={"n_samples": 1, "seed": 42}
    )

    assert response.status_code == 200
    assert response.json()["seed"] == 42


def test_predict_gan_seed_null_when_omitted(client: TestClient) -> None:
    """
    seed omitted in request -> seed=null in response.

    Important: we do NOT auto-generate a seed server-side when None
    is passed. The schema's GANResponse.seed = None is the honest
    answer ("no seed was used"); fabricating one would mislead clients
    into thinking the call was reproducible.
    """
    response = client.post("/predict/gan/sample", json={"n_samples": 1})

    assert response.status_code == 200
    assert response.json()["seed"] is None


# Image content


def test_predict_gan_decoded_image_is_valid_rgb_32x32(
    client: TestClient,
) -> None:
    """
    Each base64 string decodes to a valid 32x32 RGB PNG.

    End-to-end content contract check: the entire encode pipeline
    (tensor -> uint8 -> permute -> numpy -> PIL.Image.fromarray ->
    BytesIO PNG -> base64) must produce something that can round-trip
    back through PIL.Image.open and report the right mode + size.
    """
    response = client.post(
        "/predict/gan/sample", json={"n_samples": 4, "seed": 1}
    )

    assert response.status_code == 200
    images = response.json()["images"]
    assert len(images) == 4

    for i, b64 in enumerate(images):
        img = _decode_base64_to_image(b64)
        assert img.mode == "RGB", (
            f"images[{i}].mode = {img.mode!r}, expected 'RGB'"
        )
        assert img.size == (IMAGE_SIZE, IMAGE_SIZE), (
            f"images[{i}].size = {img.size}, expected "
            f"({IMAGE_SIZE}, {IMAGE_SIZE})"
        )
        # tobytes() length check: H * W * channels * 1 byte/uint8.
        # Confirms PIL agrees on the channel count, not just metadata.
        assert len(img.tobytes()) == IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS


# Loader-state rejection


def test_predict_gan_model_unloaded(client_unloaded: TestClient) -> None:
    """
    /predict/gan/sample returns 503 when the GAN cache is empty.

    Defense-in-depth: the lifespan event guarantees the model is
    loaded before requests in production. But if get_gan_model()
    raises (e.g., during a hot-reload window or pathological state),
    the router catches RuntimeError and returns 503 - same shape
    /ready returns for the same condition.
    """
    response = client_unloaded.post(
        "/predict/gan/sample", json={"n_samples": 1, "seed": 42}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "model_not_loaded"}


# Inference-latency histogram


def test_predict_gan_records_inference_duration(client: TestClient) -> None:
    """
    A successful /predict/gan/sample records the
    model_inference_duration_seconds histogram with
    model_name="pt-gan-dcgan".

    Verifies the .time() context manager wrapping the no_grad generator
    forward in the router emits a labeled sample. The substring asserted
    below is the `_count` line prometheus_client auto-generates for any
    Histogram with at least one observation; without the wrap, this
    substring never appears in /metrics.
    """
    response = client.post(
        "/predict/gan/sample", json={"n_samples": 1, "seed": 42}
    )
    assert response.status_code == 200, "setup precondition failed"

    body = client.get("/metrics").text

    assert (
        'model_inference_duration_seconds_count{model_name="pt-gan-dcgan"}'
        in body
    )


# Per-model freshness endpoint /health/gan


def test_health_gan_loaded_fresh_returns_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    /health/gan returns 200 with diagnostic body when the model is loaded
    AND _LAST_INFERENCE_TS is within the staleness threshold.
    """
    monkeypatch.delenv("MODEL_INFERENCE_STALENESS_SECONDS", raising=False)
    response = client.get("/health/gan")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_name"] == gan_loader.MODEL_NAME
    assert body["version"] == "1"
    assert body["staleness_threshold_seconds"] == 3600
    assert 0.0 <= body["last_inference_age_seconds"] < 5.0


def test_health_gan_loaded_stale_returns_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    /health/gan returns 503 inference_stale when the model is loaded but
    the last inference timestamp is beyond the staleness threshold.
    """
    monkeypatch.setenv("MODEL_INFERENCE_STALENESS_SECONDS", "1")
    monkeypatch.setattr(gan_loader, "_LAST_INFERENCE_TS", 0.0)

    response = client.get("/health/gan")

    assert response.status_code == 503
    assert response.json() == {"detail": "inference_stale"}


def test_health_gan_unloaded_returns_503(client_unloaded: TestClient) -> None:
    """
    /health/gan returns 503 model_not_loaded when the cache is empty.
    """
    response = client_unloaded.get("/health/gan")

    assert response.status_code == 503
    assert response.json() == {"detail": "model_not_loaded"}
