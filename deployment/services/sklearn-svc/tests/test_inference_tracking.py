"""
Tests for the inference-tracking helper.

WHAT THESE TESTS COVER:
    Threshold resolution:
        - Env var unset returns DEFAULT_STALENESS_SECONDS.
        - Env var set to a valid int returns that int.
        - Env var set to "0" returns 0 (disabled mode).
        - Env var set to garbage or negative falls back to default.

    Per-loader timestamp ops:
        - `stamp(loader)` updates loader._LAST_INFERENCE_TS to a near-now value.
        - `get_age(loader)` returns seconds since the stamped timestamp.
        - `is_fresh(loader)` returns True when age < threshold, False otherwise.
        - `is_fresh(loader)` returns True when threshold == 0 (disabled mode).

    track_inference context manager:
        - Successful exit stamps loader._LAST_INFERENCE_TS to now.
        - An exception inside the with-block propagates AND skips the stamp
          (failed inference must not keep the model "fresh").

ISOLATION:
    Each test uses a fresh fake-loader (types.SimpleNamespace) rather than
    poking at the real pca_loader module. Keeps tests independent of which
    loaders the service hosts and avoids state leak between tests.
"""

from types import SimpleNamespace

import pytest

from app.middleware import inference_tracking


def _fake_loader(ts: float = 0.0, model_name: str = "test-model") -> SimpleNamespace:
    """
    Build a stand-in loader module with the two attributes track_inference
    + stamp + get_age + is_fresh read: `_LAST_INFERENCE_TS` and `MODEL_NAME`.

    SimpleNamespace is a tiny shim - no behavior, just attribute storage.
    Each test gets its own instance, so per-test state never leaks.
    """
    return SimpleNamespace(_LAST_INFERENCE_TS=ts, MODEL_NAME=model_name)


# Threshold resolution


def test_resolve_threshold_default_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env var unset -> returns DEFAULT_STALENESS_SECONDS (3600)."""
    monkeypatch.delenv(inference_tracking.STALENESS_OVERRIDE_ENV, raising=False)
    assert (
        inference_tracking._resolve_staleness_threshold()
        == inference_tracking.DEFAULT_STALENESS_SECONDS
    )


def test_resolve_threshold_env_override_returns_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env var set to a positive int returns that int."""
    monkeypatch.setenv(inference_tracking.STALENESS_OVERRIDE_ENV, "60")
    assert inference_tracking._resolve_staleness_threshold() == 60


def test_resolve_threshold_zero_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env var set to '0' returns 0 (disabled-mode sentinel)."""
    monkeypatch.setenv(inference_tracking.STALENESS_OVERRIDE_ENV, "0")
    assert inference_tracking._resolve_staleness_threshold() == 0


def test_resolve_threshold_invalid_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env var set to non-integer or negative falls back to default."""
    monkeypatch.setenv(inference_tracking.STALENESS_OVERRIDE_ENV, "not-a-number")
    assert (
        inference_tracking._resolve_staleness_threshold()
        == inference_tracking.DEFAULT_STALENESS_SECONDS
    )
    monkeypatch.setenv(inference_tracking.STALENESS_OVERRIDE_ENV, "-5")
    assert (
        inference_tracking._resolve_staleness_threshold()
        == inference_tracking.DEFAULT_STALENESS_SECONDS
    )


# Per-loader timestamp ops


def test_stamp_updates_timestamp_to_near_now() -> None:
    """stamp(loader) sets loader._LAST_INFERENCE_TS to a near-now value."""
    import time

    loader = _fake_loader(ts=0.0)
    before = time.time()
    inference_tracking.stamp(loader)
    after = time.time()
    assert before <= loader._LAST_INFERENCE_TS <= after


def test_get_age_returns_delta_from_stamp() -> None:
    """get_age(loader) returns approximately (now - _LAST_INFERENCE_TS)."""
    import time

    # Stamp explicitly 10 seconds in the past; get_age should return ~10.
    loader = _fake_loader(ts=time.time() - 10.0)
    age = inference_tracking.get_age(loader)
    # Tolerate a wide window (anything from 9.5 to 10.5) - the assertion
    # is "approximately 10", not "exactly 10". Wall-clock jitter on
    # different test runners varies.
    assert 9.5 <= age <= 10.5


def test_is_fresh_within_threshold_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recent timestamp is within the staleness threshold."""
    import time

    monkeypatch.delenv(inference_tracking.STALENESS_OVERRIDE_ENV, raising=False)
    loader = _fake_loader(ts=time.time() - 5.0)  # 5 seconds ago
    assert inference_tracking.is_fresh(loader) is True


def test_is_fresh_beyond_threshold_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timestamp older than the threshold is stale."""
    import time

    # Tight 1-second threshold; stamp 10 seconds in the past.
    monkeypatch.setenv(inference_tracking.STALENESS_OVERRIDE_ENV, "1")
    loader = _fake_loader(ts=time.time() - 10.0)
    assert inference_tracking.is_fresh(loader) is False


def test_is_fresh_disabled_threshold_always_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env-disabled threshold (0) reports fresh regardless of age."""
    monkeypatch.setenv(inference_tracking.STALENESS_OVERRIDE_ENV, "0")
    # Ancient timestamp - would normally be very stale.
    loader = _fake_loader(ts=0.0)
    assert inference_tracking.is_fresh(loader) is True


# track_inference context manager


def test_track_inference_stamps_on_successful_exit() -> None:
    """Exiting the with-block normally stamps the loader timestamp."""
    import time

    loader = _fake_loader(ts=0.0)
    before = time.time()
    with inference_tracking.track_inference(loader):
        pass
    after = time.time()
    assert before <= loader._LAST_INFERENCE_TS <= after


def test_track_inference_skips_stamp_on_exception() -> None:
    """
    An exception inside the with-block propagates AND leaves the
    timestamp unchanged - a failed inference must not keep the model
    looking 'fresh'.
    """
    loader = _fake_loader(ts=0.0)
    with (
        pytest.raises(RuntimeError, match="boom"),
        inference_tracking.track_inference(loader),
    ):
        raise RuntimeError("boom")
    # Timestamp is still the original 0.0 - the stamp line after yield
    # never executed because the generator raised through it.
    assert loader._LAST_INFERENCE_TS == 0.0
