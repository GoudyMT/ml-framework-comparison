"""
Per-model inference-activity tracking for the /health/<model> endpoints.

WHAT THIS FILE PROVIDES:
    A small helper that wraps the existing per-model inference latency
    metric with a freshness timestamp. Routers replace
    `with MODEL_INFERENCE_DURATION_SECONDS.labels(...).time():` with
    `with track_inference(loader_module):` - same metric observation,
    plus a stamp of `loader_module._LAST_INFERENCE_TS` on successful exit.

    The /health/<model> endpoints in main.py then call:
        - `is_fresh(loader_module)` to decide 200 vs 503
        - `get_age(loader_module)` to populate the diagnostic response body

WHY A SINGLE HELPER MODULE:
    Five wrap sites across three services need the same coupled behavior
    (observe metric + stamp timestamp). A shared helper centralizes the
    coupling, the staleness threshold, and the env-var-override logic in
    one place. Loaders only need to declare `_LAST_INFERENCE_TS` as
    module-level state; everything else lives here.

WHY MODULETYPE PASSING:
    Each loader owns its own `_LAST_INFERENCE_TS` (per-model state) and
    its own `MODEL_NAME` constant (per-model label). The helper takes a
    loader module as argument and reads + mutates those attributes
    duck-typed. No per-model branches inside the helper; the loader
    module IS the per-model identity.

WHY EVERY-CALL ENV VAR LOOKUP:
    `_resolve_staleness_threshold()` reads the env var on every call
    rather than caching at import. Matches the established pattern in
    input_distribution.py - tests can monkeypatch the env var and see
    the change immediately, no import-time caching to fight.

SINGLE-WORKER ASSUMPTION:
    Timestamps are in-memory and per-process. Multi-worker uvicorn would
    give each worker its own timestamp - one worker's recent inference
    does not refresh another worker's staleness. Acceptable for a
    per-model freshness signal; aggregated across workers via the
    Prometheus inference-count metric tells the same story at the
    cluster level.
"""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

from app.middleware.metrics import MODEL_INFERENCE_DURATION_SECONDS


class _InferenceLoader(Protocol):
    """
    Structural type for a loader module passed to this helper.

    Loader modules are not classes - they are Python modules with a few
    well-known module-level attributes. Protocol lets mypy enforce the
    structural contract (must have MODEL_NAME + _LAST_INFERENCE_TS)
    without requiring loaders to inherit from anything. The underscore
    prefix on _LAST_INFERENCE_TS is unusual for a Protocol attribute but
    matches the established loader-internal-state convention used for
    _MODEL / _SCALER / _MODEL_VERSION across the codebase.
    """

    MODEL_NAME: str
    _LAST_INFERENCE_TS: float

# Env-var override pattern mirrors MODEL_ALIAS, MLFLOW_ARTIFACT_ROOT_OVERRIDE,
# and INPUT_DISTRIBUTION_SAMPLE_EVERY elsewhere in the codebase: a small
# _resolve_* helper reads the env var on every call and falls back to a
# module constant. 0 disables the freshness check entirely (loaded-only
# health); negative or non-integer values quiet-fallback to the default.
STALENESS_OVERRIDE_ENV: str = "MODEL_INFERENCE_STALENESS_SECONDS"
DEFAULT_STALENESS_SECONDS: int = 3600


def _resolve_staleness_threshold() -> int:
    """
    Resolve the staleness threshold in seconds from env var or default.

    Returns 0 to disable the freshness check entirely. Negative or
    non-integer env values fall back silently to DEFAULT_STALENESS_SECONDS
    - quiet degradation rather than a startup failure on a typo (health
    freshness is observability, not safety-critical).
    """
    raw = os.environ.get(STALENESS_OVERRIDE_ENV)
    if raw is None:
        return DEFAULT_STALENESS_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_STALENESS_SECONDS
    if value < 0:
        return DEFAULT_STALENESS_SECONDS
    return value


def stamp(loader_module: _InferenceLoader) -> None:
    """
    Set loader_module._LAST_INFERENCE_TS to the current wall-clock time.

    Called from track_inference() on successful exit, and inline at the
    end of each loader's load_*_model() function so a freshly loaded
    model is considered "active" from the moment it is ready - no
    chicken-and-egg with /health/<model> at startup.
    """
    loader_module._LAST_INFERENCE_TS = time.time()


def get_age(loader_module: _InferenceLoader) -> float:
    """
    Return seconds since loader_module._LAST_INFERENCE_TS.

    On a freshly-started process where load has not yet run, the loader's
    `_LAST_INFERENCE_TS = 0.0` module-level default yields a very large
    age (~current Unix time). The /health/<model> endpoint surfaces that
    via the existing model_not_loaded path, not via stale-age handling.
    """
    return time.time() - loader_module._LAST_INFERENCE_TS


def is_fresh(loader_module: _InferenceLoader) -> bool:
    """
    Return True if the loader's last activity is within the staleness threshold.

    Threshold of 0 disables the freshness check entirely - always returns
    True (the /health/<model> endpoint still gates on `is_loaded()` first,
    so a disabled threshold means "loaded-only health").
    """
    threshold = _resolve_staleness_threshold()
    if threshold == 0:
        return True
    return get_age(loader_module) < threshold


@contextmanager
def track_inference(loader_module: _InferenceLoader) -> Iterator[None]:
    """
    Wrap a model-forward call with the inference-duration metric and an
    activity-timestamp stamp.

    Behavior:
        - Inner `with MODEL_INFERENCE_DURATION_SECONDS.labels(...).time():`
          observes elapsed seconds into the histogram on exit. Fires on
          both successful and failed exits (latency telemetry never lies).
        - `stamp(loader_module)` runs only on SUCCESSFUL exit. The line
          after `yield` in a @contextmanager generator is skipped when
          the wrapped block raises - standard contextlib semantics. A
          failed inference must not leave the model looking "fresh".

    Routers use this in place of the bare metric .time() wrap - same
    metric observation, plus the freshness stamp.
    """
    with MODEL_INFERENCE_DURATION_SECONDS.labels(
        model_name=loader_module.MODEL_NAME
    ).time():
        yield
    stamp(loader_module)
