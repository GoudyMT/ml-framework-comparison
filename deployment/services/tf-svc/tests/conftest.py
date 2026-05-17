"""
Shared pytest fixtures for tf-svc tests.

WHAT THIS FILE PROVIDES:
    - `FakeTokenizer` - duck-typed stand-in for SentencePieceProcessor.
      EncodeAsIds returns a configurable list of ints; DecodeIds
      returns a configurable string. Lets tests assert on exact
      response shape without loading the real 360 KB BPE model.
    - `FakeTransformer` - duck-typed stand-in for the encoder-decoder
      Transformer. encode() returns a fixed-shape stub tensor;
      decode() returns logits where the argmax of the last position
      walks through a configurable decode_sequence list. Lets tests
      drive the autoregressive loop to specific outcomes (early EOS,
      max_length cap) without loading the real 11.68M-param model.
    - `client` - TestClient with both loader caches populated by the
      fakes (default decode_sequence produces 3 tokens then EOS).
    - `client_unloaded` - TestClient with both caches cleared.

WHY MOCK THE MODEL + TOKENIZER:
    The real Transformer + tokenizer load takes ~2.5 seconds and
    requires the MLflow registry + .h5 + .model files on disk - none
    of which are available in CI environments that pull only source
    code. Mocking the cache slots directly bypasses MLflow entirely
    while keeping every other code path (routes, middleware,
    validation, schema serialization, decode loop logic) running
    for real.

LIFESPAN BEHAVIOR:
    `TestClient(app)` (without `with`) does NOT trigger the lifespan
    event. So translation_loader.load_translation_model() does NOT
    run during tests - the fixtures manually populate the cache slots
    instead.
"""

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import tensorflow as tf
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.translation import TRANSLATION_MAX_LENGTH
from app.services import translation_loader
from app.services.translation_model import EOS_IDX, VOCAB_SIZE


class FakeTokenizer:
    """
    Duck-typed stand-in for sentencepiece.SentencePieceProcessor.

    Exposes the surface the deployment code touches:
        - EncodeAsIds(text)  -> configurable list of ints
        - DecodeIds(ids)     -> configurable string
        - GetPieceSize()     -> 8000 (matches real vocab)
    """

    def __init__(
        self,
        encode_ids: list[int] | None = None,
        decode_string: str = "fake translation",
    ) -> None:
        # Default encode produces 5 tokens; tests that need specific
        # input-token counts can pass their own list.
        self._encode_ids = encode_ids if encode_ids is not None else [10, 20, 30, 40, 50]
        self._decode_string = decode_string

    def EncodeAsIds(self, text: str) -> list[int]:  # noqa: N802 (sentencepiece API name)
        return list(self._encode_ids)

    def DecodeIds(self, ids: list[int]) -> str:  # noqa: N802
        return self._decode_string

    def GetPieceSize(self) -> int:  # noqa: N802
        return VOCAB_SIZE


class FakeTransformer:
    """
    Duck-typed stand-in for the encoder-decoder Transformer.

    encode(src, src_mask, training) -> stub tensor of the expected
        rank (the router only passes it through to decode; the actual
        values aren't read).
    decode(tgt, encoder_out, src_mask, tgt_mask, training) -> logits
        where argmax of the LAST position equals decode_sequence[step].
        Stepping is internal: each call advances by one. Construct
        with decode_sequence=[42, 43, EOS_IDX] to generate 2 tokens
        then stop, or with a long list of non-EOS values to test the
        max_length cap path.

    Not a real keras.Model subclass - duck typing is enough for the
    router (which only calls .encode and .decode by name).
    """

    def __init__(self, decode_sequence: list[int] | None = None) -> None:
        # Default: 3 tokens then EOS - representative of a normal
        # short translation.
        self.decode_sequence = (
            decode_sequence if decode_sequence is not None else [100, 200, 300, EOS_IDX]
        )
        self._step = 0

    def encode(
        self, src: tf.Tensor, src_mask: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        # Shape (batch, src_len, d_model). The router caches this and
        # passes it to decode() unchanged; values don't matter.
        return tf.zeros((1, TRANSLATION_MAX_LENGTH, 256), dtype=tf.float32)

    def decode(
        self,
        tgt: tf.Tensor,
        encoder_out: tf.Tensor,
        src_mask: tf.Tensor,
        tgt_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        # Determine which token to emit next. If we've exhausted the
        # sequence, emit EOS to terminate the loop (defensive).
        if self._step < len(self.decode_sequence):
            next_id = self.decode_sequence[self._step]
        else:
            next_id = EOS_IDX
        self._step += 1

        # Build logits where argmax of the LAST position is next_id.
        # The router's `tf.argmax(logits[0, -1, :])` finds it.
        seq_len = int(tgt.shape[1])
        logits = np.zeros((1, seq_len, VOCAB_SIZE), dtype=np.float32)
        logits[0, -1, next_id] = 100.0
        return tf.constant(logits)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """
    TestClient with both loader caches pre-populated by fakes.

    Default decode_sequence produces 3 tokens then EOS - representative
    of a normal short translation. Tests needing different behavior
    can re-monkeypatch translation_loader._MODEL with a custom
    FakeTransformer.

    monkeypatch.setattr swaps the module attribute for the duration
    of THIS test only - automatically restored at teardown.
    """
    fake_model: Any = FakeTransformer()
    fake_tokenizer: Any = FakeTokenizer()
    monkeypatch.setattr(translation_loader, "_MODEL", fake_model)
    monkeypatch.setattr(translation_loader, "_TOKENIZER", fake_tokenizer)
    monkeypatch.setattr(translation_loader, "_MODEL_VERSION", "1")
    # Mirror the real load's _LAST_INFERENCE_TS stamp - a freshly-loaded
    # model is considered active from the moment it is ready, so the
    # /health/translation endpoint returns 200 without needing a prior
    # request.
    monkeypatch.setattr(translation_loader, "_LAST_INFERENCE_TS", time.time())

    yield TestClient(app)


@pytest.fixture
def client_unloaded(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """
    TestClient with both loader caches CLEARED.

    Use for tests exercising the "model not loaded" path - /ready
    returning 503, /translate returning 503. Mirrors the real-world
    startup window before the lifespan event finishes loading.
    """
    monkeypatch.setattr(translation_loader, "_MODEL", None)
    monkeypatch.setattr(translation_loader, "_TOKENIZER", None)
    monkeypatch.setattr(translation_loader, "_MODEL_VERSION", None)
    # Reset to module default - the /health/translation endpoint's
    # is_loaded() check fires first, so this 0.0 only matters if a prior
    # test left the TS in a non-default state. Explicit reset is defensive.
    monkeypatch.setattr(translation_loader, "_LAST_INFERENCE_TS", 0.0)

    yield TestClient(app)
