"""
Q-learning Q-table loader for pt-svc.

WHAT THIS FILE IS:
    The bridge between the MLflow Model Registry and the FastAPI
    handler. At service startup (lifespan event), resolves the
    `pt-qlearning-taxi@production` alias, loads the trained Q-table
    via numpy, and caches the array in a module-level slot. Hot-path
    requests read from the cache - no I/O per request.

LIFECYCLE:
    1. main.py's lifespan event calls `load_qlearning_model()` at startup.
    2. _QTABLE, _MODEL_VERSION populate.
    3. /ready checks `is_loaded()` -> 200 once True.
    4. /predict/qlearning/taxi handler calls `get_qtable()` per request
       (cheap dict lookup; no I/O).
    5. Process exit reclaims everything.

WHY THIS LOADER IS SIMPLER THAN THE OTHERS IN THIS SERVICE:
    The Q-table IS the model - a numpy array of shape (500, 6) where
    each row is a state and each column is an action's expected
    cumulative reward. There is no:
        - class to instantiate (no `qlearning_model.py` file)
        - state_dict to assemble (the .npy IS the parameters)
        - eval()/train() distinction (numpy has no such modes)
        - preprocessing scaler (input is a plain int state ID)
        - autograd to disable (no gradients in tabular RL)

    The whole loading process is `np.load(path)`. The cache is one
    array. The accessor returns it. That's the entire model lifecycle.

KEY NUMPY CONCEPTS USED HERE:
    `np.load(path)`:
        Deserializes a `.npy` file - numpy's native binary format for
        single arrays. The format spec embeds shape + dtype + raw
        bytes; reading is just a header parse + memory read. Unlike
        pickle, the .npy reader can ONLY produce numpy arrays of
        primitive dtypes - there's no arbitrary-code-execution path
        the way `torch.load(weights_only=False)` has, so no
        `weights_only=True` equivalent flag is needed (or exists).

    Defensive shape assertion:
        After load, we verify `qtable.shape == (N_STATES, N_ACTIONS)`.
        Catches artifact rot at startup - if a future retraining
        shipped a different-shape table, this raises before any
        request can hit a malformed Q-row. Failing fast at boot
        beats silent wrong-shape indexing at request time.
"""

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlflow
import numpy as np
import structlog
from mlflow.tracking import MlflowClient

from app.schemas.qlearning import N_ACTIONS, N_STATES

log = structlog.get_logger("qlearning_loader")


# Configuration
# ---------------------------------------------------------------------------

"""
Registered model name + alias. We resolve them at runtime via
MlflowClient.get_model_version_by_alias() rather than the combined
"models:/pt-qlearning-taxi@production" URI - the alias-resolved
`version.source` is a properly-formed file:/// URI we can hand
directly to np.load, sidestepping known issues with MLflow's
two-stage download path on Windows.
"""
MODEL_NAME: str = "pt-qlearning-taxi"
# DEFAULT alias. Resolved at startup unless the MODEL_ALIAS env var is
# set, in which case the env var wins. Lets a container be pointed at a
# non-default alias (canary / staging) without rebuilding the image.
MODEL_ALIAS: str = "production"

# Env var name read inside load_qlearning_model() to override the default
# above at runtime. The actual alias used is logged in qlearning_load_start
# + qlearning_alias_resolved events so operators can verify which alias
# was queried against the registry.
ALIAS_OVERRIDE_ENV: str = "MODEL_ALIAS"

# Filename inside the registered version's artifact directory.
# promote_to_registry.py logs this single file at the artifact root.
QTABLE_FILENAME: str = "v1_qtable_taxi.npy"

# Standard MLflow env var. The client reads it automatically when no
# tracking URI is passed explicitly. Production Docker sets this in
# the container env; dev computes a sensible default below.
TRACKING_URI_ENV: str = "MLFLOW_TRACKING_URI"

# Optional override env var read by _resolve_artifact_dir() below. Set this
# when the service runs on a different host than the one that populated the
# registry, so the absolute paths baked into version.source can be re-anchored
# under a runtime-known root (e.g., a container's bind-mounted /srv/mlflow).
# Unset = legacy behavior: use the registered URI verbatim, no relocation.
ARTIFACT_ROOT_OVERRIDE_ENV: str = "MLFLOW_ARTIFACT_ROOT_OVERRIDE"


def _default_tracking_uri() -> str:
    """
    Compute a dev-default tracking URI pointing at deployment/mlflow.db.

    Resolves from this file's location:
        deployment/
            services/
                pt-svc/
                    app/
                        services/
                            qlearning_loader.py     <- __file__
        => parents[4] = deployment/

    Returns:
        `sqlite:///` URI (NOT `file:///`). MLflow accepts SQLite
        registries via the sqlite scheme; metadata lives in the .db,
        artifact bytes in the sibling mlruns/ directory.
    """
    db_path = Path(__file__).parents[4] / "mlflow.db"
    return f"sqlite:///{db_path.as_posix()}"


def _file_uri_to_path(uri: str) -> Path:
    """
    Convert a `file://` URI to a platform-correct local Path.

    MLflow stores artifact_uri as `file:///C:/...` on Windows or
    `file:///home/user/...` on POSIX. Filesystem APIs (Path, open()
    etc.) want a native path string, not a URI.

    Args:
        uri: A `file://` URI string.

    Returns:
        A pathlib.Path pointing at the same location.

    Raises:
        ValueError: If the URI scheme isn't `file`.

    Note: Python 3.13 added Path.from_uri() which would replace this
    helper. We're on 3.12 so we hand-roll the conversion.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(
            f"Expected file:// URI, got scheme {parsed.scheme!r}: {uri}"
        )

    # urlparse on Windows: `file:///C:/foo` -> parsed.path = '/C:/foo'.
    # On POSIX: `file:///home/foo` -> parsed.path = '/home/foo' (already correct).
    # On Windows we strip the leading slash before the drive letter.
    path_str = unquote(parsed.path)
    if (
        os.name == "nt"
        and len(path_str) >= 3
        and path_str[0] == "/"
        and path_str[2] == ":"
    ):
        path_str = path_str[1:]
    return Path(path_str)


def _resolve_artifact_dir(source_uri: str) -> Path:
    """
    Resolve a registered model's source URI to a local artifact directory,
    optionally re-anchoring under a runtime-supplied root.

    The MLflow registry stores `version.source` as an absolute file:// URI
    captured at promotion time. When the service runs on a different host
    than the one that populated the registry (a Linux container reading a
    Windows-populated mlflow.db, a cloud node reading a CI-populated one),
    those absolute paths don't exist locally even though the artifact bytes
    are accessible under a different prefix.

    Setting MLFLOW_ARTIFACT_ROOT_OVERRIDE declares "the host-side prefix in
    the registry doesn't apply here; my mlruns/ tree lives at this path."
    The helper extracts everything from `/mlruns/` onwards in the
    registered URI's path and re-anchors under the override. With the env
    var unset, behavior is identical to a bare _file_uri_to_path call.

    Args:
        source_uri: The `file://` URI from version.source.

    Returns:
        A pathlib.Path pointing at the artifact directory, either as
        recorded in the registry or as relocated under the override.

    Raises:
        ValueError: If source_uri isn't a file:// URI (propagated from
            _file_uri_to_path when no override is in effect).
        RuntimeError: If the override is set but source_uri's path has
            no `/mlruns/` segment to anchor against. Indicates a registry
            layout we don't know how to relocate; the operator should
            either unset the override or run against a registry whose
            URIs include /mlruns/.
    """
    override = os.environ.get(ARTIFACT_ROOT_OVERRIDE_ENV)
    if override:
        # Re-anchor under the override root: extract everything from
        # `/mlruns/` onwards in the registered URI, prepend override.
        # Example:
        #   source_uri = "file:///C:/Users/Max/.../deployment/mlruns/<run-id>/artifacts"
        #   override   = "/srv/mlflow"
        #   result     = "/srv/mlflow/mlruns/<run-id>/artifacts"
        src_path = urlparse(source_uri).path
        idx = src_path.find("/mlruns/")
        if idx < 0:
            raise RuntimeError(
                f"MLFLOW_ARTIFACT_ROOT_OVERRIDE is set to {override!r} "
                f"but source URI {source_uri!r} has no '/mlruns/' "
                f"segment to anchor against. Either unset the override "
                f"or point it at a registry whose URIs include /mlruns/."
            )
        # idx + 1 drops the leading '/' so the join produces a clean
        # "<override>/mlruns/<run-id>/artifacts" path.
        return Path(override) / src_path[idx + 1 :]
    return _file_uri_to_path(source_uri)


def _resolve_alias() -> str:
    """
    Resolve the alias used to query the registry.

    Reads the MODEL_ALIAS env var (via ALIAS_OVERRIDE_ENV); falls back
    to the MODEL_ALIAS module constant when the env var is unset or
    empty. The resolved value is logged in qlearning_load_start so
    operators can verify which alias was queried against the registry.

    Returns:
        The alias string (e.g. "production", "staging").
    """
    return os.environ.get(ALIAS_OVERRIDE_ENV) or MODEL_ALIAS


# Module-level cache
# ---------------------------------------------------------------------------
"""
Single-underscore "private by convention" - tests can monkeypatch
these directly via pytest's monkeypatch fixture without name-mangling
getting in the way. Initialized to None; populated by
load_qlearning_model() exactly once per process.
"""

_QTABLE: np.ndarray | None = None
_MODEL_VERSION: str | None = None  # e.g. "1" once the alias is resolved


# Public API
# ---------------------------------------------------------------------------


def load_qlearning_model() -> None:
    """
    Load the Q-table from the MLflow registry into the cache.

    Called ONCE at service startup (from main.py's lifespan event).
    After this returns, get_qtable() is safe to call. Calling twice
    is a no-op (idempotency guard).

    Raises:
        RuntimeError: If the artifact can't be loaded or has the wrong
            shape. Original exception chained via `from`. The lifespan
            handler does NOT catch - we want the process to exit so
            Docker/k8s sees the failure and triggers a restart/alert.

    Side effects:
        - Sets MLFLOW_TRACKING_URI in os.environ if it wasn't set.
        - Mutates module-level _QTABLE, _MODEL_VERSION.
    """
    global _QTABLE, _MODEL_VERSION

    # Idempotency: don't re-load if cache is already populated.
    # Useful for tests that re-run startup.
    if _QTABLE is not None:
        return

    # Resolve tracking URI: env var wins, else compute dev default.
    tracking_uri = os.environ.get(TRACKING_URI_ENV) or _default_tracking_uri()
    os.environ[TRACKING_URI_ENV] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    # Resolve the model alias: MODEL_ALIAS env var wins, else use the default.
    alias = _resolve_alias()

    log.info(
        "qlearning_load_start",
        model_name=MODEL_NAME,
        model_alias=alias,
        tracking_uri=tracking_uri,
    )

    try:
        # Step 1: alias -> ModelVersion. Pure registry query, no
        # artifact download yet.
        client = MlflowClient(tracking_uri=tracking_uri)
        version = client.get_model_version_by_alias(
            name=MODEL_NAME, alias=alias
        )
        log.info(
            "qlearning_alias_resolved",
            model_name=MODEL_NAME,
            model_alias=alias,
            version=version.version,
            run_id=version.run_id[:8] if version.run_id else None,
        )

        # MLflow types version.source as `str | None` because the
        # registry schema technically allows null. In practice we
        # always have a source; mypy strict needs the explicit check.
        if version.source is None:
            raise RuntimeError(
                f"Registered version {MODEL_NAME} v{version.version} has "
                f"no source URI. The registry entry is malformed."
            )

        # Step 2: convert file:// URI to a local Path; verify the
        # Q-table file exists before touching it. Failing fast here
        # gives a single clear error instead of a half-loaded state.
        artifact_dir = _resolve_artifact_dir(version.source)
        qtable_file = artifact_dir / QTABLE_FILENAME
        if not qtable_file.is_file():
            raise FileNotFoundError(
                f"Expected Q-table at {qtable_file}, but it doesn't "
                f"exist. Check promote_to_registry.py logged "
                f"{QTABLE_FILENAME} to the registered version."
            )

        # Step 3: load the Q-table. np.load reads the .npy header +
        # raw bytes - no pickle, no arbitrary-code-execution path,
        # so no security flag equivalent to torch's weights_only is
        # needed.
        qtable = np.load(qtable_file)

        # Step 4: defensive shape assertion. Catches artifact rot
        # (e.g., a future retraining ships a (1000, 6) table for an
        # extended observation space). Failing at boot beats silent
        # wrong-shape indexing at request time.
        expected_shape = (N_STATES, N_ACTIONS)
        if qtable.shape != expected_shape:
            raise RuntimeError(
                f"Q-table shape mismatch: expected {expected_shape}, "
                f"got {qtable.shape}. The registered artifact does not "
                f"match the schema's N_STATES/N_ACTIONS constants. "
                f"Update both or re-promote a matching artifact."
            )

    except Exception as exc:
        # Re-raise as RuntimeError with chained cause. uvicorn will
        # log the full traceback; Docker/k8s will see the container
        # exit non-zero.
        raise RuntimeError(
            f"Failed to load {MODEL_NAME}@{alias} from "
            f"{tracking_uri}. Verify the registry exists and the "
            f"alias is set."
        ) from exc

    _QTABLE = qtable
    _MODEL_VERSION = str(version.version)

    log.info(
        "qlearning_load_complete",
        model_name=MODEL_NAME,
        version=str(version.version),
        shape=list(qtable.shape),
        dtype=str(qtable.dtype),
        size_bytes=int(qtable.nbytes),
    )


def get_qtable() -> np.ndarray:
    """
    Return the cached Q-table.

    Returns:
        A (N_STATES, N_ACTIONS) numpy array. Each row is a state,
        each column is an action's expected cumulative reward. The
        handler indexes by state ID and runs argmax to pick the
        action.

    Raises:
        RuntimeError: If load_qlearning_model() hasn't run yet.
    """
    if _QTABLE is None:
        raise RuntimeError(
            "Q-table not loaded. Call load_qlearning_model() first "
            "(usually done by the FastAPI lifespan event)."
        )
    return _QTABLE


def get_model_version() -> str:
    """
    Return the resolved registry version string (e.g. "1").

    Useful for logging + the /ready response payload so operators can
    confirm exactly which version is serving traffic.

    Raises:
        RuntimeError: If load_qlearning_model() hasn't run yet.
    """
    if _MODEL_VERSION is None:
        raise RuntimeError(
            "Q-table not loaded. Call load_qlearning_model() first."
        )
    return _MODEL_VERSION


def is_loaded() -> bool:
    """
    Cheap status check for the /ready endpoint.

    Returns:
        True iff load_qlearning_model() has populated the cache. The
        Q-table is the only artifact this loader manages, so one
        cache slot is the complete check.
    """
    return _QTABLE is not None
