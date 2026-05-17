"""
Input-distribution sampling for downstream drift detection.

WHAT THIS FILE PROVIDES:
    `maybe_log_input_distribution(features, model_name)` - a helper called
    from prediction routers that, every Nth request per model, emits a
    structured log line with summary statistics of the incoming feature
    vector.

WHY A HELPER, NOT A MIDDLEWARE:
    Only 2 of the 5 deployed endpoints (PCA, DNN) carry numeric feature
    vectors. The other 3 take config (GAN: n_samples + seed), a single
    integer state (Q-learning), or text (translation) - none have a
    distribution shape compatible with mean/std/L2 metrics. A middleware
    would need path-matching + JSON-body replay machinery to opt-in two
    endpoints; an explicit helper call is one line per relevant router
    and keeps the logic discoverable.

WHY EVERY NTH, NOT EVERY REQUEST:
    Logging summary stats on every request inflates log volume by ~250
    bytes per request - acceptable at hundreds of RPS but wasteful at
    sustained traffic. Sampling at 1% (N=100 default) keeps the drift
    signal trackable while the log stream stays bounded. The operator
    can override via the INPUT_DISTRIBUTION_SAMPLE_EVERY env var:
        N=0   disables sampling entirely
        N=1   logs every request (debug only)
        N=100 default (~1% sampling)

WHY WHOLE-VECTOR STATS, NOT PER-FEATURE:
    Per-feature stats are needed for direct PSI (Population Stability
    Index) input - 561 means + 561 stds for DNN, 784+784 for PCA - which
    is ~20 KB per log line. We emit aggregate stats (mean, std, min, max,
    L2 norm, zero-fraction) instead: a "something shifted" smoke alarm,
    not a localized PSI input. Per-feature can be added later if drift
    detection ever lands for real production.

WHY PER-MODEL COUNTER:
    Each model's traffic is independent; using one global counter would
    skew the sampling cadence toward the busier endpoint. A dict keyed
    by model_name gives each model a steady "every Nth of its own
    requests" cadence so the resulting time series stays even per model.

SINGLE-WORKER ASSUMPTION:
    Counters are in-memory and per-process. Multi-worker uvicorn would
    give each worker its own counter, slightly desynchronizing sampling
    across workers - acceptable for drift detection since the aggregate
    signal still holds.
"""

import os

import numpy as np
import structlog

# Env-var override pattern mirrors MODEL_ALIAS and
# MLFLOW_ARTIFACT_ROOT_OVERRIDE elsewhere in the codebase: a small
# _resolve_* helper reads the env var on each call and falls back to a
# module constant. Reading on each call (rather than caching at import)
# keeps tests isolated - monkeypatching the env var takes effect
# immediately.
SAMPLE_EVERY_OVERRIDE_ENV: str = "INPUT_DISTRIBUTION_SAMPLE_EVERY"
DEFAULT_SAMPLE_EVERY: int = 100

# Per-model request counters keyed by model_name. The dict.get +
# assignment update on each call is safe under asyncio because the event
# loop dispatches one coroutine at a time per worker - no preemption
# between the read and the write. Counters live only for the process
# lifetime (no persistence); a restart resets them all to zero. New
# model_name entries start at zero via dict.get's default.
_COUNTERS: dict[str, int] = {}


def _resolve_sample_every() -> int:
    """
    Resolve the sampling interval N from env var or default.

    Returns 0 to disable sampling entirely. Negative or non-integer env
    values fall back to DEFAULT_SAMPLE_EVERY - quiet degradation rather
    than a startup failure on a typo (drift logging is observability,
    not safety-critical).
    """
    raw = os.environ.get(SAMPLE_EVERY_OVERRIDE_ENV)
    if raw is None:
        return DEFAULT_SAMPLE_EVERY
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SAMPLE_EVERY
    if value < 0:
        return DEFAULT_SAMPLE_EVERY
    return value


def maybe_log_input_distribution(
    features: list[float], model_name: str
) -> None:
    """
    Sample summary stats of the incoming feature vector every Nth call.

    Increments this model's request counter on every call. When the
    counter is a positive multiple of the resolved sample interval N,
    computes whole-vector summary statistics and emits a structured log
    line via structlog.

    Args:
        features: The raw input vector as the client sent it (after
            Pydantic validation but before any preprocessing). Logging
            the raw input matches what drift detection cares about -
            client-side distribution shift, not transformations we
            apply server-side.
        model_name: Identifier matching the loader's MODEL_NAME constant
            (e.g., "sk-pca", "pt-dnn"). Used both for per-model counter
            keying and as a log-line label.

    Side effects:
        - Mutates _COUNTERS[model_name].
        - Emits one structlog `input_distribution_sample` event when
          the counter hits a sampling boundary; emits nothing otherwise.

    Returns None - purely side-effecting.
    """
    sample_every = _resolve_sample_every()

    # Increment first, then check. With N=100 this means the 100th, 200th,
    # 300th request log - never the 1st. Keeps smoke tests / canaries
    # from triggering a log line on every cold start.
    _COUNTERS[model_name] = _COUNTERS.get(model_name, 0) + 1
    current = _COUNTERS[model_name]

    # Disabled mode: N=0 skips all emission. Counter still increments
    # (cheap, harmless) so re-enabling mid-process picks up at the right
    # cadence.
    if sample_every == 0:
        return

    if current % sample_every != 0:
        return

    # Compute summary stats over the whole vector. np.asarray copies the
    # list into a contiguous float32 array; one pass each for mean / std /
    # min / max / L2 / zero-fraction. Round to 6 decimals so the JSON log
    # line stays compact (full float64 precision would emit 17 digits per
    # number).
    arr = np.asarray(features, dtype=np.float32)
    structlog.get_logger(__name__).info(
        "input_distribution_sample",
        model_name=model_name,
        sample_index=current,
        n_features=int(arr.shape[0]),
        mean=round(float(arr.mean()), 6),
        std=round(float(arr.std()), 6),
        min=round(float(arr.min()), 6),
        max=round(float(arr.max()), 6),
        l2_norm=round(float(np.linalg.norm(arr)), 6),
        zero_fraction=round(float((arr == 0.0).mean()), 6),
    )
