# Pylance / Pyright suppression for known-false-positive Keras stubs.
# Pylance's bundled TF stubs declare Layer.__call__(self, inputs) -
# a tight signature that rejects the multi-positional Keras pattern
# (e.g., `model(src, tgt, training=False)`). Keras's __call__
# dispatches via *args, **kwargs at runtime so any signature works,
# and mypy strict (the CLI gate) doesn't flag it because it treats
# TF imports as Any. Suppressing only this one rule keeps every
# other type check active.
# pyright: reportCallIssue=false

"""
Translation model + tokenizer loader for tf-svc.

WHAT THIS FILE IS:
    The bridge between the MLflow Model Registry and the FastAPI
    handler. At service startup (lifespan event), resolves the
    `tf-transformer-translation@production` alias, loads BOTH the
    Transformer weights (.h5) AND the SentencePiece BPE tokenizer
    (.model), runs a warm-up forward pass to compile TF's oneDNN
    ops, and caches everything in module-level slots. Hot-path
    requests read from the cache - no I/O per request.

LIFECYCLE:
    1. main.py's lifespan event calls `load_translation_model()`.
    2. _MODEL, _TOKENIZER, _MODEL_VERSION populate.
    3. /ready checks `is_loaded()` -> 200 once True.
    4. /translate handler calls `get_model()` + `get_tokenizer()`
       per request (cheap dict lookups; no I/O).
    5. Process exit reclaims everything.

WHY MULTI-ARTIFACT:
    Translation needs BOTH a model and a tokenizer to be useful.
    Loading either alone leaves the service half-functional - the
    inference path encodes English text via SentencePiece into BPE
    IDs, runs them through the Transformer, then decodes the output
    IDs back to Spanish text via the same SentencePiece processor.
    is_loaded() requires both before reporting True.

WHY WARM-UP:
    TensorFlow's first inference call after load_weights triggers
    lazy oneDNN op compilation + CPU cache fills, which can take
    5-10 seconds. Without warm-up, the first user request hits
    that cost. With warm-up, the lifespan eats it at startup
    (where the orchestrator's readinessProbe expects load time)
    and per-request latency is normal from the first hit.

KEY TENSORFLOW + KERAS CONCEPTS USED HERE:
    Lazy weight materialization:
        Keras layers don't materialize their weight tensors until
        the first forward pass tells them what input shape they're
        operating on. load_weights() needs the layers built already
        - if you call it on a fresh Transformer() instance with no
        forward pass first, Keras silently leaves layers at random
        init. Solution: run a dummy forward of the right shape
        BEFORE load_weights, then a warm-up forward AFTER for the
        oneDNN compile.

    .h5 vs .keras vs SavedModel:
        We use the .h5 weights-only format (file ends in
        `.weights.h5`). Stores per-layer weight tensors keyed by
        layer name; doesn't include the architectural graph (that's
        in app/services/translation_model.py). Forward-compatible
        across TF/Keras minor versions for our recipe.
"""

import os
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

import mlflow
import sentencepiece as spm
import structlog
import tensorflow as tf
from mlflow.tracking import MlflowClient

from app.services.translation_model import (
    D_FF,
    D_MODEL,
    DROPOUT,
    MAX_LEN,
    N_DEC_LAYERS,
    N_ENC_LAYERS,
    N_HEADS,
    PAD_IDX,
    VOCAB_SIZE,
    Transformer,
)

log = structlog.get_logger("translation_loader")


# Configuration
# ---------------------------------------------------------------------------

"""
Registered model name + alias. We resolve via
MlflowClient.get_model_version_by_alias() rather than the combined
"models:/tf-transformer-translation@production" URI - the alias-
resolved `version.source` is a properly-formed file:/// URI we
can hand directly to filesystem APIs, sidestepping known issues
with MLflow's two-stage download path on Windows.
"""
MODEL_NAME: str = "tf-transformer-translation"
# DEFAULT alias. Resolved at startup unless the MODEL_ALIAS env var is
# set, in which case the env var wins. Lets a container be pointed at a
# non-default alias (canary / staging) without rebuilding the image.
MODEL_ALIAS: str = "production"

# Env var name read inside load_translation_model() to override the
# default above at runtime. The actual alias used is logged in
# translation_load_start + translation_alias_resolved events so operators
# can verify which alias was queried against the registry.
ALIAS_OVERRIDE_ENV: str = "MODEL_ALIAS"

"""
Filenames inside the registered version's artifact directory.
promote_to_registry.py logs both files at the artifact root, so
they sit side-by-side at <artifact_dir>/{filename}.
"""
WEIGHTS_FILENAME: str = "translation_transformer.weights.h5"
TOKENIZER_FILENAME: str = "bpe.model"

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
                tf-svc/
                    app/
                        services/
                            translation_loader.py     <- __file__
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
    empty. The resolved value is logged in translation_load_start so
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
load_translation_model() exactly once per process.
"""

_MODEL: Transformer | None = None
_TOKENIZER: Any | None = None      # sentencepiece.SentencePieceProcessor instance
_MODEL_VERSION: str | None = None  # e.g. "1" once the alias is resolved


# Public API
# ---------------------------------------------------------------------------


def load_translation_model() -> None:
    """
    Load the Transformer + tokenizer from the MLflow registry.

    Called ONCE at service startup (from main.py's lifespan event).
    After this returns, get_model() and get_tokenizer() are safe to
    call. Calling twice is a no-op (idempotency guard).

    Raises:
        RuntimeError: If either artifact can't be loaded. Original
            exception chained via `from`. The lifespan handler does
            NOT catch - we want the process to exit so Docker/k8s
            sees the failure and triggers a restart/alert.

    Side effects:
        - Sets MLFLOW_TRACKING_URI in os.environ if it wasn't set.
        - Mutates module-level _MODEL, _TOKENIZER, _MODEL_VERSION.
    """
    global _MODEL, _TOKENIZER, _MODEL_VERSION

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
        "translation_load_start",
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
            "translation_alias_resolved",
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
        weights_file = artifact_dir / WEIGHTS_FILENAME
        tokenizer_file = artifact_dir / TOKENIZER_FILENAME
        for required, label in (
            (weights_file, "Transformer weights"),
            (tokenizer_file, "BPE tokenizer"),
        ):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Expected {label} at {required}, but it doesn't "
                    f"exist. Check promote_to_registry.py logged both "
                    f"{WEIGHTS_FILENAME} and {TOKENIZER_FILENAME} to "
                    f"the registered version."
                )

        # Step 3: load the SentencePiece tokenizer. The .model file
        # is a small protobuf (~360 KB); load takes <100ms.
        # SentencePieceProcessor.Load() returns False on failure
        # rather than raising, so we check the return.
        tokenizer = spm.SentencePieceProcessor()
        if not tokenizer.Load(str(tokenizer_file)):
            raise RuntimeError(
                f"sentencepiece failed to load tokenizer from {tokenizer_file}"
            )
        log.info(
            "translation_tokenizer_loaded",
            vocab_size=tokenizer.GetPieceSize(),
            file_size_kb=round(tokenizer_file.stat().st_size / 1024, 1),
        )

        # Step 4: build the Transformer architecture with the
        # constants from translation_model.py. These match what was
        # used to train the registered weights; changing them
        # produces a state_dict layout load_weights cannot map onto
        # the .h5 contents.
        model = Transformer(
            src_vocab=VOCAB_SIZE,
            tgt_vocab=VOCAB_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_enc_layers=N_ENC_LAYERS,
            n_dec_layers=N_DEC_LAYERS,
            d_ff=D_FF,
            max_len=MAX_LEN,
            dropout_rate=DROPOUT,
            pad_idx=PAD_IDX,
        )

        # Step 5: dummy forward pass to materialize Keras weights.
        # Keras lazy-builds layers on first call - load_weights
        # silently leaves layers at random init if they aren't
        # built yet. The dummy input shape doesn't matter as long
        # as it matches the expected input rank/dtype.
        dummy_src = tf.zeros((1, MAX_LEN), dtype=tf.int32)
        dummy_tgt = tf.zeros((1, MAX_LEN), dtype=tf.int32)
        _ = model(dummy_src, dummy_tgt, training=False)

        # Step 6: load the trained weights from disk.
        model.load_weights(str(weights_file))

        # count_params() is Keras's idiomatic helper - returns the
        # total int of trainable + non-trainable scalar weights.
        # Cleaner than summing tf.reduce_prod(v.shape) over each
        # trainable_weights entry, and avoids a TensorShape-vs-Tensor
        # type quirk in some stubs.
        n_params = model.count_params()
        log.info(
            "translation_weights_loaded",
            n_params=n_params,
            file_size_mb=round(weights_file.stat().st_size / 1024 / 1024, 1),
        )

        # Step 7: warm-up forward pass with the loaded weights.
        # Compiles oneDNN custom ops + fills CPU caches with the
        # actual weight tensors so the first user request doesn't
        # eat that one-time cost. Eager mode doesn't do per-shape
        # graph caching so a single shape covers it; the router's
        # autoregressive loop reuses the same encoder output across
        # decode steps anyway.
        warmup_start = time.perf_counter()
        _ = model(dummy_src, dummy_tgt, training=False)
        warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

    except Exception as exc:
        # Re-raise as RuntimeError with chained cause. uvicorn will
        # log the full traceback; Docker/k8s will see the container
        # exit non-zero.
        raise RuntimeError(
            f"Failed to load {MODEL_NAME}@{alias} from "
            f"{tracking_uri}. Verify the registry exists, the alias "
            f"is set, and both {WEIGHTS_FILENAME} and "
            f"{TOKENIZER_FILENAME} are bundled with the version."
        ) from exc

    """
     cast() narrows Pylance's view: subclassing keras.Model (which
    resolves to a generic Model[Unknown, Unknown] under bundled
    stubs) loses the Transformer subclass identity at assignment
    time. The runtime type IS Transformer; cast just tells the
    type-checker what we already know.
    """
    _MODEL = cast(Transformer, model)  # type: ignore[redundant-cast]
    _TOKENIZER = tokenizer
    _MODEL_VERSION = str(version.version)

    log.info(
        "translation_load_complete",
        model_name=MODEL_NAME,
        version=str(version.version),
        n_params=n_params,
        vocab_size=tokenizer.GetPieceSize(),
        warmup_ms=round(warmup_ms, 2),
    )


def get_model() -> Transformer:
    """
    Return the cached Transformer (with weights loaded + warmed).

    Returns:
        The Transformer keras.Model. Already in eval mode for our
        purposes (we always pass training=False at inference).

    Raises:
        RuntimeError: If load_translation_model() hasn't run yet.
    """
    if _MODEL is None:
        raise RuntimeError(
            "Transformer not loaded. Call load_translation_model() "
            "first (usually done by the FastAPI lifespan event)."
        )
    return _MODEL


def get_tokenizer() -> Any:
    """
    Return the cached SentencePieceProcessor.

    Returns:
        A loaded sentencepiece.SentencePieceProcessor. Use
        .EncodeAsIds(text) and .DecodeIds(ids) to round-trip
        between Python strings and BPE token IDs.

    Raises:
        RuntimeError: If load_translation_model() hasn't run yet.
    """
    if _TOKENIZER is None:
        raise RuntimeError(
            "Tokenizer not loaded. Call load_translation_model() first."
        )
    return _TOKENIZER


def get_model_version() -> str:
    """
    Return the resolved registry version string (e.g. "1").

    Useful for logging + the /ready response payload so operators can
    confirm exactly which version is serving traffic.

    Raises:
        RuntimeError: If load_translation_model() hasn't run yet.
    """
    if _MODEL_VERSION is None:
        raise RuntimeError(
            "Translation model not loaded. Call load_translation_model() first."
        )
    return _MODEL_VERSION


def is_loaded() -> bool:
    """
    Cheap status check for the /ready endpoint.

    Returns:
        True iff load_translation_model() has populated BOTH the
        model and the tokenizer. Partial load is "not ready" - the
        inference path needs both to round-trip text through the
        Transformer.
    """
    return _MODEL is not None and _TOKENIZER is not None
