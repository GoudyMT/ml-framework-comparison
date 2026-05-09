"""
DCGAN generator loader for pt-svc.

WHAT THIS FILE IS:
    The bridge between the MLflow Model Registry and the FastAPI
    handler. At service startup (lifespan event), resolves the
    `pt-gan-dcgan@production` alias, loads the trained state_dict,
    and caches the assembled DCGenerator in a module-level slot.
    Hot-path requests read from the cache - no I/O per request.

LIFECYCLE:
    1. main.py's lifespan event calls `load_gan_model()` at startup.
    2. _MODEL, _MODEL_VERSION populate.
    3. /ready checks `is_loaded()` -> 200 once True.
    4. /predict/gan/sample handler calls `get_gan_model()` per request
       (cheap dict lookup; no I/O).
    5. Process exit reclaims everything.

WHY ONE ARTIFACT (NOT TWO LIKE DNN):
    PCA bundled `pca_model.joblib` + `scaler.pkl`. DNN bundled
    `dnn_model.pth` + `scaler.pkl`. DCGAN needs only `dcgan_generator.pth`
    because there's no fitted preprocessing state to carry alongside.
    The training-time normalization (`pixel/127.5 - 1.0` -> [-1, 1])
    is two named constants, not a fitted artifact - they live in the
    router as plain numbers, with a comment pointing at
    `data/processed/gans/preprocessing_info.json` as source-of-truth.

    Lesson: "bundle preprocessing alongside the model" is a guideline,
    not a rule. Bundle when there's fitted state to preserve (a scaler
    fit on the training split, a tokenizer trained on the corpus).
    Don't bundle when the entire transform is `(x + 1) * 127.5`.

KEY PYTORCH CONCEPTS USED HERE:
    Same as dnn_loader (torch.load + weights_only=True; load_state_dict;
    model.eval()). One detail worth restating for DCGAN specifically:

    BatchNorm2d in eval mode:
        DCGAN's generator has three BatchNorm2d layers (after the
        first three transposed convolutions). At training time they
        compute statistics from the current batch. At eval time they
        use the stored running_mean / running_var that were updated
        during training. For a generator at inference - sampling 1-16
        random latent vectors - using batch statistics would normalize
        each tiny "batch" to itself and the output would look nothing
        like the training distribution. eval() is mandatory; not
        something to skip.

    Single-precision float32:
        Our trained weights are float32 (1,069,827 params * 4 bytes
        = ~4 MB, plus BN buffers). We don't need float16 or bfloat16
        for CPU inference - 1 MB difference, negligible. The router's
        `torch.randn(..., dtype=torch.float32)` matches the model dtype
        so no casting overhead.
"""

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlflow
import structlog
import torch
from mlflow.tracking import MlflowClient

from app.services.gan_model import DCGenerator

log = structlog.get_logger("gan_loader")


# Configuration
# ---------------------------------------------------------------------------

"""
Registered model name + alias. We resolve them at runtime via
MlflowClient.get_model_version_by_alias() rather than the combined
"models:/pt-gan-dcgan@production" URI - the alias-resolved
`version.source` is a properly-formed file:/// URI we can hand
directly to torch.load, sidestepping known issues with MLflow's
two-stage download path on Windows.
"""
MODEL_NAME: str = "pt-gan-dcgan"
MODEL_ALIAS: str = "production"

# Filename inside the registered version's artifact directory.
# promote_to_registry.py logs this single file at the artifact root.
GAN_WEIGHTS_FILENAME: str = "dcgan_generator.pth"

# Standard MLflow env var. The client reads it automatically when no
# tracking URI is passed explicitly. Production Docker sets this in
# the container env; dev computes a sensible default below.
TRACKING_URI_ENV: str = "MLFLOW_TRACKING_URI"


def _default_tracking_uri() -> str:
    """
    Compute a dev-default tracking URI pointing at deployment/mlflow.db.

    Resolves from this file's location:
        deployment/
            services/
                pt-svc/
                    app/
                        services/
                            gan_loader.py     <- __file__
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


# Module-level cache
# ---------------------------------------------------------------------------
"""
Single-underscore "private by convention" - tests can monkeypatch
these directly via pytest's monkeypatch fixture without name-mangling
getting in the way. Initialized to None; populated by load_gan_model()
exactly once per process.
"""

_MODEL: DCGenerator | None = None
_MODEL_VERSION: str | None = None  # e.g. "1" once the alias is resolved


# Public API


def load_gan_model() -> None:
    """
    Load the DCGAN generator from the MLflow registry into the cache.

    Called ONCE at service startup (from main.py's lifespan event).
    After this returns, get_gan_model() is safe to call. Calling
    twice is a no-op (idempotency guard).

    Raises:
        RuntimeError: If the artifact can't be loaded. Original
            exception chained via `from`. The lifespan handler does
            NOT catch - we want the process to exit so Docker/k8s
            sees the failure and triggers a restart/alert.

    Side effects:
        - Sets MLFLOW_TRACKING_URI in os.environ if it wasn't set.
        - Mutates module-level _MODEL, _MODEL_VERSION.
    """
    global _MODEL, _MODEL_VERSION

    # Idempotency: don't re-load if cache is already populated.
    # Useful for tests that re-run startup.
    if _MODEL is not None:
        return

    # Resolve tracking URI: env var wins, else compute dev default.
    tracking_uri = os.environ.get(TRACKING_URI_ENV) or _default_tracking_uri()
    os.environ[TRACKING_URI_ENV] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    log.info(
        "gan_load_start",
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        tracking_uri=tracking_uri,
    )

    try:
        # Step 1: alias -> ModelVersion. Pure registry query, no
        # artifact download yet.
        client = MlflowClient(tracking_uri=tracking_uri)
        version = client.get_model_version_by_alias(
            name=MODEL_NAME, alias=MODEL_ALIAS
        )
        log.info(
            "gan_alias_resolved",
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
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
        # weights file exists before touching it. Failing fast here
        # gives a single clear error instead of a half-loaded state.
        artifact_dir = _file_uri_to_path(version.source)
        weights_file = artifact_dir / GAN_WEIGHTS_FILENAME
        if not weights_file.is_file():
            raise FileNotFoundError(
                f"Expected DCGAN weights at {weights_file}, but it doesn't "
                f"exist. Check promote_to_registry.py logged "
                f"{GAN_WEIGHTS_FILENAME} to the registered version."
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

        model = DCGenerator()
        model.load_state_dict(state_dict)
        model.eval()  # CRITICAL: BatchNorm uses running stats, not batch.

    except Exception as exc:
        # Re-raise as RuntimeError with chained cause. uvicorn will
        # log the full traceback; Docker/k8s will see the container
        # exit non-zero.
        raise RuntimeError(
            f"Failed to load {MODEL_NAME}@{MODEL_ALIAS} from "
            f"{tracking_uri}. Verify the registry exists and the "
            f"alias is set."
        ) from exc

    _MODEL = model
    _MODEL_VERSION = str(version.version)

    log.info(
        "gan_load_complete",
        model_name=MODEL_NAME,
        version=str(version.version),
        n_params=sum(p.numel() for p in model.parameters()),
    )


def get_gan_model() -> DCGenerator:
    """
    Return the cached DCGenerator (in eval mode).

    Returns:
        The DCGenerator nn.Module. Already in eval() mode - safe to
        call directly inside `with torch.no_grad():`.

    Raises:
        RuntimeError: If load_gan_model() hasn't run yet.
    """
    if _MODEL is None:
        raise RuntimeError(
            "DCGAN not loaded. Call load_gan_model() first (usually "
            "done by the FastAPI lifespan event)."
        )
    return _MODEL


def get_model_version() -> str:
    """
    Return the resolved registry version string (e.g. "1").

    Useful for logging + the /ready response payload so operators can
    confirm exactly which version is serving traffic.

    Raises:
        RuntimeError: If load_gan_model() hasn't run yet.
    """
    if _MODEL_VERSION is None:
        raise RuntimeError(
            "DCGAN not loaded. Call load_gan_model() first."
        )
    return _MODEL_VERSION


def is_loaded() -> bool:
    """
    Cheap status check for the /ready endpoint.

    Returns:
        True iff load_gan_model() has populated the model cache.
        Unlike DNN's loader (which checks model + scaler), DCGAN has
        a single artifact - one cache slot to check.
    """
    return _MODEL is not None
