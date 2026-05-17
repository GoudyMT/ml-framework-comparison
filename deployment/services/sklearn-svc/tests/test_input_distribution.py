"""
Tests for the input-distribution sampling helper.

WHAT THESE TESTS COVER:
    Helper unit behavior:
        - On the Nth call per model, emits a structured
          `input_distribution_sample` event with the expected fields.
        - Below the threshold (calls 1..N-1) the helper emits nothing.
        - Summary stats computed from the input vector match the
          straightforward numpy reference values.
        - The INPUT_DISTRIBUTION_SAMPLE_EVERY env var overrides the
          default sample interval.

    Integration through the router:
        - Hitting /predict/pca N times via TestClient causes exactly
          one captured `input_distribution_sample` event - proves the
          router actually calls the helper.

ISOLATION:
    Tests reset the per-model counter at the start of each test (the
    counter is module-level state shared across tests). The env-var
    test uses monkeypatch.setenv so the override is scoped to that
    test only.

STRUCTLOG capture_logs:
    structlog.testing.capture_logs() context manager intercepts all
    structlog events for the duration of the block, returning them as
    a list of dicts. Works alongside the existing JSON-renderer pipeline
    configured in app/middleware/logging.py.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.middleware import input_distribution
from app.schemas.pca import INPUT_DIM


@pytest.fixture(autouse=True)
def reset_counters() -> Iterator[None]:
    """
    Clear the helper's per-model counters before every test.

    The counter dict is module-level state. Without a reset between
    tests, an earlier test that called the helper 50 times would leave
    counter=50 in place; a later test asserting "the 100th call should
    log" would actually be testing call number 150. The fixture is
    autouse so every test in this file gets a clean slate.
    """
    input_distribution._COUNTERS.clear()
    yield
    input_distribution._COUNTERS.clear()


# Helper unit tests


def test_helper_logs_on_nth_call() -> None:
    """
    Calling the helper exactly DEFAULT_SAMPLE_EVERY times emits one
    input_distribution_sample event with the expected shape.
    """
    features = [0.5] * INPUT_DIM
    with capture_logs() as captured:
        for _ in range(input_distribution.DEFAULT_SAMPLE_EVERY):
            input_distribution.maybe_log_input_distribution(
                features, model_name="sk-pca"
            )

    samples = [
        c for c in captured if c.get("event") == "input_distribution_sample"
    ]
    assert len(samples) == 1, f"expected 1 sample event, got {len(samples)}"

    event = samples[0]
    assert event["model_name"] == "sk-pca"
    assert event["sample_index"] == input_distribution.DEFAULT_SAMPLE_EVERY
    assert event["n_features"] == INPUT_DIM
    # Each summary stat must be present and a finite float.
    for field in ("mean", "std", "min", "max", "l2_norm", "zero_fraction"):
        assert field in event, f"missing summary field: {field}"
        assert isinstance(event[field], float)


def test_helper_does_not_log_below_threshold() -> None:
    """
    Calling the helper DEFAULT_SAMPLE_EVERY - 1 times emits no
    input_distribution_sample event.
    """
    features = [0.5] * INPUT_DIM
    with capture_logs() as captured:
        for _ in range(input_distribution.DEFAULT_SAMPLE_EVERY - 1):
            input_distribution.maybe_log_input_distribution(
                features, model_name="sk-pca"
            )

    samples = [
        c for c in captured if c.get("event") == "input_distribution_sample"
    ]
    assert samples == [], (
        f"expected no sample events below threshold, got {len(samples)}"
    )


def test_helper_stats_correctness() -> None:
    """
    Summary stats computed by the helper match the reference values for
    a hand-constructed feature vector.

    Vector: [1.0, 0.0, 2.0, 0.0]
        mean          = 0.75
        std (popn)    = 0.829156 (numpy default ddof=0)
        min           = 0.0
        max           = 2.0
        l2_norm       = sqrt(1^2 + 2^2) = 2.236068
        zero_fraction = 2/4 = 0.5
    """
    features = [1.0, 0.0, 2.0, 0.0]
    with capture_logs() as captured:
        # First call advances counter to 1; complete the cycle to N
        # so the Nth call emits.
        for _ in range(input_distribution.DEFAULT_SAMPLE_EVERY):
            input_distribution.maybe_log_input_distribution(
                features, model_name="sk-pca-test"
            )

    samples = [
        c for c in captured if c.get("event") == "input_distribution_sample"
    ]
    assert len(samples) == 1
    event = samples[0]
    assert event["mean"] == pytest.approx(0.75, abs=1e-6)
    assert event["std"] == pytest.approx(0.829156, abs=1e-5)
    assert event["min"] == 0.0
    assert event["max"] == 2.0
    assert event["l2_norm"] == pytest.approx(2.236068, abs=1e-5)
    assert event["zero_fraction"] == pytest.approx(0.5, abs=1e-6)


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Setting INPUT_DISTRIBUTION_SAMPLE_EVERY changes the sample interval.

    With N=5 the helper logs on calls 5, 10, 15, ... With N=0 the helper
    is disabled entirely (no log even at the canonical 100th call).
    """
    # Override to N=5; expect a log on the 5th call.
    monkeypatch.setenv(input_distribution.SAMPLE_EVERY_OVERRIDE_ENV, "5")
    features = [0.1] * 10
    with capture_logs() as captured:
        for _ in range(5):
            input_distribution.maybe_log_input_distribution(
                features, model_name="sk-pca"
            )
    samples = [
        c for c in captured if c.get("event") == "input_distribution_sample"
    ]
    assert len(samples) == 1, f"expected 1 sample event with N=5, got {len(samples)}"
    assert samples[0]["sample_index"] == 5

    # Reset counters + override to N=0 (disabled). No events expected.
    input_distribution._COUNTERS.clear()
    monkeypatch.setenv(input_distribution.SAMPLE_EVERY_OVERRIDE_ENV, "0")
    with capture_logs() as captured_disabled:
        for _ in range(200):
            input_distribution.maybe_log_input_distribution(
                features, model_name="sk-pca"
            )
    samples_disabled = [
        c for c in captured_disabled
        if c.get("event") == "input_distribution_sample"
    ]
    assert samples_disabled == [], (
        f"expected no events with N=0, got {len(samples_disabled)}"
    )


# Integration through the router


def test_router_emits_through_predict_endpoint(client: TestClient) -> None:
    """
    Hitting /predict/pca DEFAULT_SAMPLE_EVERY times causes the helper to
    emit exactly one captured input_distribution_sample event.

    Proves the router actually calls the helper - without the wiring,
    the counter never advances and no event is emitted regardless of how
    many requests arrive.
    """
    payload = {"features": [100.0] * INPUT_DIM}
    with capture_logs() as captured:
        for _ in range(input_distribution.DEFAULT_SAMPLE_EVERY):
            response = client.post("/predict/pca", json=payload)
            assert response.status_code == 200

    samples = [
        c for c in captured if c.get("event") == "input_distribution_sample"
    ]
    assert len(samples) == 1, (
        f"expected 1 sample event after {input_distribution.DEFAULT_SAMPLE_EVERY} "
        f"requests, got {len(samples)}"
    )
    assert samples[0]["model_name"] == "sk-pca"
    assert samples[0]["n_features"] == INPUT_DIM
