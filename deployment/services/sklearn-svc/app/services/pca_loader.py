"""
PCA model loader for sklearn-svc.

WHAT THIS FILE IS:
    The bridge between the MLflow Model Registry (single source of truth
    for trained models) and the FastAPI handler (hot path that needs the
    model in memory). Loads the SK PCA estimator ONCE at service startup,
    caches it, and exposes accessors for the rest of the app.

WHY A SEPARATE "loader" MODULE:
    Routers should be thin - they translate HTTP <-> Python and delegate.
    "How do I load a model from MLflow" is business logic, not HTTP logic.
    Putting it here means:
        - The router stays focused on request/response shape
        - Tests can mock `get_pca_model()` without touching MLflow
        - If we ever switch from MLflow to a flat .joblib file, we change
          ONE file (this one) and the router is untouched

KEY DESIGN CHOICES:
    - Eager load at startup. Fail fast: if the model is missing/broken,
      the service refuses to come up.
    - Module-level cache. Modules are Python singletons - first import
      initializes, every subsequent import reuses. Free process-wide cache.
    - Production alias (`models:/sk-pca@production`), not a pinned version.
      Lets us roll out new model versions by moving the alias in the
      registry, no code change or redeploy needed.
    - Env-var-with-fallback for the tracking URI. Production sets
      MLFLOW_TRACKING_URI explicitly in the Docker env; dev auto-derives
      the path to `deployment/mlflow.db` from this file's location.

LIFECYCLE:
    1. main.py's lifespan event calls `load_pca_model()` at startup
    2. That populates the module-level cache (_MODEL, _VARIANCE_EXPLAINED)
    3. /ready checks `is_loaded()` - returns 200 once True
    4. The /predict/pca handler calls `get_pca_model()` for every request
       (cheap dict lookup, no I/O)
    5. On shutdown the process exits and the cache vanishes with it
"""

import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import joblib
import mlflow
import structlog
from mlflow.tracking import MlflowClient

# Module-level structlog logger. Bound name "pca_loader" appears as
# `logger=pca_loader` in JSON output, letting log queries filter by
# component without parsing the stack.
log = structlog.get_logger("pca_loader")

# Configuration
"""
We use MLflow's registry for ALIAS RESOLUTION only - given the name + alias,
the registry tells us which version is "production" and where its artifact
lives on disk. We then load the joblib file directly with `joblib.load()`,
bypassing `mlflow.sklearn.load_model()`.

Why bypass: Phase 0's promote_to_registry.py logged the model with
`mlflow.log_artifact(pca_model.joblib)` (raw artifact upload), NOT
`mlflow.sklearn.log_model(...)` (sklearn-flavor structured directory). So
the registry points at a bare `pca_model.joblib` file, not a flavored
model directory with `MLmodel` metadata. `mlflow.sklearn.load_model()`
expects the flavor structure - it would fail to find `MLmodel` regardless
of platform.

This is a defensible choice for our case: we trust our own training
pipeline's joblib output, we're not consuming third-party model packages,
and skipping the flavor dance keeps the loader trivial. We retain the
benefits we want from MLflow (registry, aliases, version tracking) and
drop the parts we don't need (flavor wrappers, conda env spec, etc.).

Production-wise, this means: when the artifact format changes, we update
this loader. The registry contract (name + alias) stays stable.
"""
MODEL_NAME: str = "sk-pca"
MODEL_ALIAS: str = "production"

# Filenames of the artifacts inside the registered version's artifact
# directory. promote_to_registry.py logs both files at the top level
# of artifacts/, so we can load them by name from the same dir.
PCA_FILENAME: str = "pca_model.joblib"
SCALER_FILENAME: str = "scaler.pkl"

# Backward-compat alias (some pre-Phase-B references): same value as
# PCA_FILENAME. Remove once any external readers migrate.
ARTIFACT_FILENAME: str = PCA_FILENAME

"""
Env var name MLflow's client reads automatically when no URI is passed
explicitly to the API. Standard convention - same name in every MLflow
deployment, so production sysadmins know what to set.
"""
TRACKING_URI_ENV: str = "MLFLOW_TRACKING_URI"


def _default_tracking_uri() -> str:
    """
    Compute a sensible default tracking URI for local development.

    Resolves to `deployment/mlflow.db` based on this file's path. This
    works because the project layout is fixed:

        deployment/
        +-- mlflow.db                                 <- target
        +-- services/
            +-- sklearn-svc/
                +-- app/
                    +-- services/
                        +-- pca_loader.py             <- this file

    `Path(__file__).parents[4]` walks up 4 directories to `deployment/`.

    Returns:
        A `sqlite:///` URI (NOT `file:///`). MLflow's tracking URI accepts
        SQLite paths via the `sqlite:///` scheme; the registry metadata
        lives inside the .db, while artifact files (the actual joblib
        bytes) live in `deployment/mlruns/` alongside.
    """
    db_path = Path(__file__).parents[4] / "mlflow.db"
    # `as_posix()` normalizes Windows backslashes to forward slashes -
    # MLflow URIs are POSIX-style on every platform.
    return f"sqlite:///{db_path.as_posix()}"


def _file_uri_to_path(uri: str) -> Path:
    """
    Convert a `file://` URI to a platform-correct local Path.

    MLflow stores artifact locations as URIs (e.g., `file:///C:/foo/bar` on
    Windows, `file:///home/user/foo` on POSIX). To open the file with stdlib
    we need a Path the OS understands - filesystem APIs don't accept URIs.

    Args:
        uri: A `file://` URI string.

    Returns:
        A pathlib.Path pointing at the same filesystem location.

    Raises:
        ValueError: If the URI scheme isn't `file`.

    Why we hand-roll this:
        Python 3.13 added `Path.from_uri()` which would do the same thing,
        but we're on 3.12. The conversion is small enough to inline.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(
            f"Expected file:// URI, got scheme {parsed.scheme!r}: {uri}"
        )

    # urlparse on `file:///C:/foo` gives parsed.path = '/C:/foo'.
    # On POSIX `file:///home/foo` -> parsed.path = '/home/foo' (correct).
    # On Windows we need to drop the leading '/' before the drive letter
    # so '/C:/foo' becomes 'C:/foo'.
    path_str = unquote(parsed.path)
    if (
        os.name == "nt"
        and len(path_str) >= 3
        and path_str[0] == "/"
        and path_str[2] == ":"
    ):
        path_str = path_str[1:]
    return Path(path_str)



# Module-level cache
"""
These are SINGLE underscores, not double - they're "module-private by
convention" but accessible if you really need them (e.g., from tests
resetting state between cases). Double underscore would name-mangle.

Initialized to None; populated by `load_pca_model()`. The accessors
below check for None and raise a clear error if called too early -
better than letting the caller hit a confusing AttributeError later.
"""
_MODEL: Any | None = None
_VARIANCE_EXPLAINED: float | None = None
_MODEL_VERSION: str | None = None  # Set to e.g. "1" once the alias is resolved
_SCALER: Any | None = None         # Dict {'mean': ndarray(784), 'std': ndarray(784)}


# Public API

def load_pca_model() -> None:
    """
    Load the PCA from the MLflow registry into the module cache.

    Called ONCE at service startup (from main.py's lifespan event). After
    this returns, `get_pca_model()` and `get_variance_explained()` are
    safe to call. Calling this twice is a no-op (idempotent guard) -
    convenient for tests that re-run startup.

    Raises:
        RuntimeError: If the model cannot be loaded. The original MLflow
            exception is chained so `__cause__` shows the real reason.

    Side effects:
        - Sets MLFLOW_TRACKING_URI in the process env if it wasn't set
          already (so any other MLflow call later in the process picks
          up the same registry).
        - Mutates module-level _MODEL and _VARIANCE_EXPLAINED.
    """
    global _MODEL, _VARIANCE_EXPLAINED, _MODEL_VERSION, _SCALER

    # Idempotency guard: if the cache is already populated, skip.
    # Useful for test suites that call lifespan startup multiple times.
    if _MODEL is not None:
        return

    # Resolve the tracking URI: env var wins, else compute the dev default.
    tracking_uri = os.environ.get(TRACKING_URI_ENV) or _default_tracking_uri()
    # Setting it on the env (not just calling mlflow.set_tracking_uri)
    # means any other MLflow client in the same process sees the same
    # registry without us having to wire it through.
    os.environ[TRACKING_URI_ENV] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    log.info(
        "pca_load_start",
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        tracking_uri=tracking_uri,
    )

    try:
        # Step 1: resolve the alias to a concrete ModelVersion.
        # MlflowClient queries the registry tables directly - no artifact
        # download yet. Returns a ModelVersion with .version and .source
        # (where .source is a properly-formed file:/// URI).
        client = MlflowClient(tracking_uri=tracking_uri)
        version = client.get_model_version_by_alias(
            name=MODEL_NAME, alias=MODEL_ALIAS
        )
        log.info(
            "pca_alias_resolved",
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
            version=version.version,
            run_id=version.run_id[:8] if version.run_id else None,
        )

        # Step 2: convert the file:// URI to a local Path, then joblib.load
        # the artifact directly. The artifact lives at:
        #     <artifact_dir>/<ARTIFACT_FILENAME>
        # which on disk is e.g.
        #     deployment/mlruns/<run-id>/artifacts/pca_model.joblib
        #
        # MLflow types `version.source` as `str | None` because the registry
        # schema technically allows null. In practice every version we register
        # has a source, but mypy strict mode demands we prove it - hence the
        # explicit guard. If this fires, the registry entry is malformed and
        # we want a clear error, not an opaque AttributeError later.
        if version.source is None:
            raise RuntimeError(
                f"Registered version {MODEL_NAME} v{version.version} has no "
                f"source URI. The registry entry is malformed."
            )
        artifact_dir = _file_uri_to_path(version.source)

        # Verify both expected artifacts exist BEFORE loading either.
        # Failing fast here gives a clear error; loading partial state
        # would let pca load succeed and only blow up later when the
        # router tries to use the missing scaler.
        model_file = artifact_dir / PCA_FILENAME
        scaler_file = artifact_dir / SCALER_FILENAME
        for required, label in (
            (model_file, "PCA model"),
            (scaler_file, "scaler"),
        ):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Expected {label} artifact at {required}, but it "
                    f"doesn't exist. Check promote_to_registry.py logged "
                    f"both {PCA_FILENAME} and {SCALER_FILENAME} to the "
                    f"registered version."
                )

        # joblib.load deserializes the pickle. Same call sklearn docs use
        # everywhere - no MLflow magic in the hot path.
        # The scaler is a plain dict {'mean': ndarray, 'std': ndarray} - we
        # do the standardization arithmetic in app/services/preprocessing.py
        # without instantiating a sklearn StandardScaler.
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
    except Exception as exc:
        # Re-raise as RuntimeError with chained cause. The lifespan
        # handler in main.py will catch this and Docker/k8s will see
        # the container exit with a clear error in stderr.
        raise RuntimeError(
            f"Failed to load {MODEL_NAME}@{MODEL_ALIAS} from {tracking_uri}. "
            f"Verify the registry exists and the alias is set."
        ) from exc

    # Sklearn PCA exposes `explained_variance_ratio_` - a numpy array of
    # length n_components, each entry the fraction of variance explained
    # by that component. Sum = cumulative ratio (= 0.9085 for our model).
    # `.sum()` returns numpy.float64; cast to plain float for clean JSON.
    variance_explained = float(model.explained_variance_ratio_.sum())

    _MODEL = model
    _VARIANCE_EXPLAINED = variance_explained
    _MODEL_VERSION = str(version.version)
    _SCALER = scaler

    log.info(
        "pca_load_complete",
        model_name=MODEL_NAME,
        version=str(version.version),
        n_components=int(model.n_components_),
        variance_explained=round(variance_explained, 4),
        scaler_loaded=True,
        scaler_features=int(scaler["mean"].shape[0]),
    )


def get_pca_model() -> Any:
    """
    Return the cached PCA estimator.

    Returns:
        The sklearn PCA instance. Has `.transform(X)` for inference.

    Raises:
        RuntimeError: If `load_pca_model()` hasn't been called yet. This
            should never happen in the running service (lifespan loads
            before requests are routed) but guards against misuse from
            tests or scripts that import this module directly.
    """
    if _MODEL is None:
        raise RuntimeError(
            "PCA model not loaded. Call load_pca_model() first "
            "(usually done by the FastAPI lifespan event)."
        )
    return _MODEL


def get_scaler() -> Any:
    """
    Return the cached scaler dict.

    Returns:
        Dict with two keys, 'mean' and 'std', each a numpy.ndarray of
        shape (784,) and dtype float32. The router uses these for the
        per-pixel standardization step before PCA.transform.

    Raises:
        RuntimeError: If `load_pca_model()` hasn't been called yet.
    """
    if _SCALER is None:
        raise RuntimeError(
            "Scaler not loaded. Call load_pca_model() first."
        )
    return _SCALER


def get_variance_explained() -> float:
    """
    Return the cumulative explained variance ratio of the loaded model.

    Returns:
        Float in [0, 1]. For D1 SK PCA = 0.9085.

    Raises:
        RuntimeError: If the model hasn't been loaded yet.
    """
    if _VARIANCE_EXPLAINED is None:
        raise RuntimeError(
            "PCA model not loaded. Call load_pca_model() first."
        )
    return _VARIANCE_EXPLAINED


def get_model_version() -> str:
    """
    Return the resolved registry version of the loaded model.

    Useful for /ready, log lines, and the response payload - lets clients
    and operators see exactly which version is serving traffic without a
    separate registry call.

    Returns:
        The version as a string (e.g., "1"). Stringified because MLflow
        stores it that way; keeping the same type avoids surprises.

    Raises:
        RuntimeError: If the model hasn't been loaded yet.
    """
    if _MODEL_VERSION is None:
        raise RuntimeError(
            "PCA model not loaded. Call load_pca_model() first."
        )
    return _MODEL_VERSION


def is_loaded() -> bool:
    """
    Cheap status check for the /ready endpoint.

    Returns:
        True iff `load_pca_model()` has populated BOTH cache slots
        (PCA + scaler). Both are required for inference, so partial
        load is "not ready" from the service's perspective.
    """
    return _MODEL is not None and _SCALER is not None
