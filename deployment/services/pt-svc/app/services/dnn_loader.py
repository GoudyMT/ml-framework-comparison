"""
DNN model loader for pt-svc.

WHAT THIS FILE IS:
    The bridge between the MLflow Model Registry and the FastAPI
    handler. At service startup (lifespan event), resolves the
    `pt-dnn@production` alias, loads the trained state_dict + the
    bundled StandardScaler, and caches both in module-level slots.
    Hot-path requests read from the cache - no I/O per request.

LIFECYCLE:
    1. main.py's lifespan event calls `load_dnn_model()` at startup.
    2. _MODEL, _SCALER, _MODEL_VERSION populate.
    3. /ready checks `is_loaded()` -> 200 once True.
    4. /predict/dnn handler calls `get_dnn_model()` + `get_scaler()`
       per request (cheap dict lookup; no I/O).
    5. Process exit reclaims everything.

KEY PYTORCH CONCEPTS USED HERE:

    `torch.load(path, weights_only=True)`:
        Deserializes a .pth file. The `weights_only=True` flag
        restricts unpickling to a safe allow-list (tensors and
        primitives only) instead of running arbitrary pickle.
        SECURITY: pickle without weights_only can execute arbitrary
        code during deserialization - a malicious .pth file would
        own the process. weights_only=True is the new default in
        PyTorch 2.6+; setting it explicitly here makes us safe on
        any version. Our artifact is a state_dict (a dict of tensors),
        so weights_only=True is the correct choice.

    `model.load_state_dict(sd)`:
        Overwrites the freshly-instantiated model's random weights
        with the trained values. Returns a NamedTuple of
        `(missing_keys, unexpected_keys)` - both empty lists when the
        architecture matches the saved state_dict exactly. We don't
        check the return value because mismatch raises by default
        (since strict=True is the default).

    `model.eval()`:
        Switches the module from training mode to inference mode.
        Two layers behave differently:
            - BatchNorm1d uses its stored running_mean/running_var
              instead of computing batch statistics from the current
              input. Critical for single-sample inference (a batch
              of 1 has zero variance, which would explode).
            - Dropout becomes a no-op pass-through (vs. training
              mode where it zeros out random elements).
        Forgetting eval() is a classic deployment bug - predictions
        become nondeterministic and incorrect.

    `torch.no_grad()` context:
        Used at INFERENCE time in the router (not here). Disables
        autograd's gradient tracking, saving memory and ~10-20%
        compute. Doesn't affect correctness, only speed/memory.
"""

import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import joblib
import mlflow
import structlog
import torch
from mlflow.tracking import MlflowClient

from app.services.dnn_model import DNN

log = structlog.get_logger("dnn_loader")


# Configuration
# ---------------------------------------------------------------------------

"""
Registered model name + alias. We resolve them at runtime via
MlflowClient.get_model_version_by_alias() rather than the
combined "models:/pt-dnn@production" URI - the alias-resolved
`version.source` is a properly-formed file:/// URI we can hand
directly to torch.load / joblib.load, sidestepping known issues
with MLflow's two-stage download path on Windows.
"""
MODEL_NAME: str = "pt-dnn"
# DEFAULT alias. Resolved at startup unless the MODEL_ALIAS env var is
# set, in which case the env var wins. Lets a container be pointed at a
# non-default alias (canary / staging) without rebuilding the image.
MODEL_ALIAS: str = "production"

# Env var name read inside load_dnn_model() to override the default above
# at runtime. The actual alias used is logged in dnn_load_start +
# dnn_alias_resolved events so operators can verify which alias was
# queried against the registry.
ALIAS_OVERRIDE_ENV: str = "MODEL_ALIAS"

"""
Filenames inside the registered version's artifact directory.
promote_to_registry.py logs both files at the artifact root, so
they sit side-by-side at <artifact_dir>/{filename}.
"""
DNN_WEIGHTS_FILENAME: str = "dnn_model.pth"
SCALER_FILENAME: str = "scaler.pkl"

"""
Standard MLflow env var. The client reads it automatically when no
tracking URI is passed explicitly. Production Docker sets this in
the container env; dev computes a sensible default below.
"""
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
                            dnn_loader.py     <- __file__
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
        #   source_uri = "file:///<workspace-root>/deployment/mlruns/<run-id>/artifacts"
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
    empty. The resolved value is logged in dnn_load_start so operators
    can verify which alias was queried against the registry.

    Returns:
        The alias string (e.g. "production", "staging").
    """
    return os.environ.get(ALIAS_OVERRIDE_ENV) or MODEL_ALIAS



# Module-level cache
# ---------------------------------------------------------------------------
"""
Single-underscore "private by convention" - tests can monkeypatch
these directly via pytest's monkeypatch fixture without name-
mangling getting in the way. Initialized to None; populated by
load_dnn_model() exactly once per process.
"""

_MODEL: DNN | None = None
_SCALER: Any | None = None         # sklearn StandardScaler instance
_MODEL_VERSION: str | None = None  # e.g. "1" once the alias is resolved

# Wall-clock timestamp of the most recent activity (load OR inference).
# Powers the /health/dnn endpoint's freshness check. Updated in two
# places: (1) inline at the end of load_dnn_model() so a freshly-loaded
# model is considered active from the moment it is ready, and (2) on
# every successful /predict/dnn call via inference_tracking.stamp().
# Module default 0.0 means "no activity yet" - the /health/dnn endpoint
# gates on is_loaded() first, so the 0.0 default only matters in the
# narrow window between import and the lifespan load.
_LAST_INFERENCE_TS: float = 0.0


# Public API
# ---------------------------------------------------------------------------


def load_dnn_model() -> None:
    """
    Load the DNN + scaler from the MLflow registry into the cache.

    Called ONCE at service startup (from main.py's lifespan event).
    After this returns, get_dnn_model() and get_scaler() are safe to
    call. Calling twice is a no-op (idempotency guard).

    Raises:
        RuntimeError: If either artifact can't be loaded. Original
            exception chained via `from`. The lifespan handler does
            NOT catch - we want the process to exit so Docker/k8s
            sees the failure and triggers a restart/alert.

    Side effects:
        - Sets MLFLOW_TRACKING_URI in os.environ if it wasn't set.
        - Mutates module-level _MODEL, _SCALER, _MODEL_VERSION.
    """
    global _MODEL, _SCALER, _MODEL_VERSION, _LAST_INFERENCE_TS

    # Idempotency: don't re-load if cache is already populated.
    # Useful for tests that re-run startup.
    if _MODEL is not None:
        return

    # Resolve tracking URI: env var wins, else compute dev default.
    tracking_uri = os.environ.get(TRACKING_URI_ENV) or _default_tracking_uri()
    os.environ[TRACKING_URI_ENV] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    # Resolve the model alias: MODEL_ALIAS env var wins, else use the default.
    alias = _resolve_alias()

    log.info(
        "dnn_load_start",
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
            "dnn_alias_resolved",
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

        # Step 2: convert file:// URI to a local Path; verify both
        # required files exist before touching either. Failing fast
        # here gives a single clear error instead of a half-loaded
        # state.
        artifact_dir = _resolve_artifact_dir(version.source)
        weights_file = artifact_dir / DNN_WEIGHTS_FILENAME
        scaler_file = artifact_dir / SCALER_FILENAME
        for required, label in (
            (weights_file, "DNN weights"),
            (scaler_file, "scaler"),
        ):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Expected {label} artifact at {required}, but it "
                    f"doesn't exist. Check promote_to_registry.py logged "
                    f"both {DNN_WEIGHTS_FILENAME} and {SCALER_FILENAME} "
                    f"to the registered version."
                )

        # Step 3: load + assemble the model.
        # weights_only=True restricts unpickling to a safe type
        # allow-list - prevents arbitrary code execution if someone
        # ever ships us a malicious .pth file. Required setting for
        # production deployment of any pickle-based artifact.
        # map_location='cpu' forces CPU-side load even if a GPU is
        # available on the host (deployment is CPU-only).
        state_dict = torch.load(
            weights_file, map_location="cpu", weights_only=True
        )

        model = DNN()
        model.load_state_dict(state_dict)
        model.eval()  # CRITICAL: BatchNorm uses running stats, Dropout off.

        # Step 4: load the scaler. The promoted artifact is a fitted
        # sklearn.preprocessing.StandardScaler instance; joblib.load
        # returns it ready to call .transform(). No further setup.
        scaler = joblib.load(scaler_file)

    except Exception as exc:
        # Re-raise as RuntimeError with chained cause. uvicorn will
        # log the full traceback; Docker/k8s will see the container
        # exit non-zero.
        raise RuntimeError(
            f"Failed to load {MODEL_NAME}@{alias} from "
            f"{tracking_uri}. Verify the registry exists and the "
            f"alias is set."
        ) from exc

    _MODEL = model
    _SCALER = scaler
    _MODEL_VERSION = str(version.version)

    # Stamp _LAST_INFERENCE_TS at load completion so a freshly-loaded
    # model is considered "active" from the moment it is ready - the
    # /health/dnn endpoint returns 200 without needing a prior request.
    # Subsequent /predict/dnn calls re-stamp via track_inference.
    _LAST_INFERENCE_TS = time.time()

    log.info(
        "dnn_load_complete",
        model_name=MODEL_NAME,
        version=str(version.version),
        n_params=sum(p.numel() for p in model.parameters()),
        scaler_loaded=True,
        scaler_features=int(scaler.mean_.shape[0]),
    )


def get_dnn_model() -> DNN:
    """
    Return the cached DNN (in eval mode).

    Returns:
        The DNN nn.Module. Already in eval() mode - safe to call
        directly inside `with torch.no_grad():`.

    Raises:
        RuntimeError: If load_dnn_model() hasn't run yet.
    """
    if _MODEL is None:
        raise RuntimeError(
            "DNN not loaded. Call load_dnn_model() first (usually "
            "done by the FastAPI lifespan event)."
        )
    return _MODEL


def get_scaler() -> Any:
    """
    Return the cached StandardScaler.

    Returns:
        A fitted sklearn.preprocessing.StandardScaler. The handler
        calls `scaler.transform(X)` to apply the training-time
        per-feature standardization to incoming raw [-1, 1] features.

    Raises:
        RuntimeError: If load_dnn_model() hasn't run yet.
    """
    if _SCALER is None:
        raise RuntimeError(
            "Scaler not loaded. Call load_dnn_model() first."
        )
    return _SCALER


def get_model_version() -> str:
    """
    Return the resolved registry version string (e.g. "1").

    Useful for logging + the /ready response payload so operators can
    confirm exactly which version is serving traffic.

    Raises:
        RuntimeError: If load_dnn_model() hasn't run yet.
    """
    if _MODEL_VERSION is None:
        raise RuntimeError(
            "DNN not loaded. Call load_dnn_model() first."
        )
    return _MODEL_VERSION


def is_loaded() -> bool:
    """
    Cheap status check for the /ready endpoint.

    Returns:
        True iff load_dnn_model() has populated BOTH model + scaler.
        Partial load is "not ready" - the inference path needs both.
    """
    return _MODEL is not None and _SCALER is not None
