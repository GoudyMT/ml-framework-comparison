"""
Tests for POST /translate.

WHAT THESE TESTS COVER:
    Happy path:
        - Valid English text -> 200 with TranslationResponse-shaped JSON
          (source echoed, translation string, n_input_tokens,
          n_output_tokens, generation_time_ms).
        - Whitespace stripping: leading/trailing whitespace gets removed
          before encoding; the response's `source` field is the
          stripped form.
        - max_length omitted -> default of 25 applies.

    Field validation rejections (HTTP 422):
        - Empty text -> Field min_length=1 violation.
        - Whitespace-only text -> custom validator (post-strip empty).
        - Text > 1000 chars -> Field max_length violation.
        - max_length=0 -> Field ge=1 violation.
        - max_length=26 -> Field le=25 violation.
        - max_length=-5 -> Field ge=1 violation.

    Decode-loop control (drives FakeTransformer.decode_sequence):
        - Early EOS: tokenizer emits sequence ending in EOS at step 2,
          n_output_tokens should equal 2 (tokens emitted before EOS).
        - max_length cap: tokenizer never emits EOS, n_output_tokens
          should equal req.max_length.

    Loader-state rejection:
        - 503 when either the model or tokenizer cache is empty.

NOTE ON FAKETRANSFORMER + FAKETOKENIZER:
    conftest.py's defaults produce 3 tokens then EOS, with the
    tokenizer always returning the string "fake translation". Tests
    that need different decode behavior or different decoded output
    construct their own FakeTransformer / FakeTokenizer and re-
    monkeypatch the loader cache slots.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.schemas.translation import TEXT_MAX_LENGTH, TRANSLATION_MAX_LENGTH
from app.services import translation_loader
from app.services.translation_model import EOS_IDX
from tests.conftest import FakeTokenizer, FakeTransformer

# Happy path


def test_translate_happy_path(client: TestClient) -> None:
    """
    Simple text -> 200 with the right response shape.

    FakeTokenizer's default DecodeIds returns 'fake translation';
    FakeTransformer's default emits 3 tokens then EOS.
    """
    response = client.post("/translate", json={"text": "Hello, how are you?"})

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "Hello, how are you?"
    assert body["translation"] == "fake translation"
    assert body["n_input_tokens"] == 5  # FakeTokenizer default
    assert body["n_output_tokens"] == 3  # 3 tokens before EOS
    assert isinstance(body["generation_time_ms"], int | float)
    assert body["generation_time_ms"] >= 0.0


def test_translate_whitespace_stripped(client: TestClient) -> None:
    """
    Leading/trailing whitespace is stripped before encoding;
    the response's source field is the stripped form.
    """
    response = client.post("/translate", json={"text": "  Hello world  \n"})

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "Hello world"  # post-strip


def test_translate_max_length_default(client: TestClient) -> None:
    """max_length omitted -> Pydantic default 25 applies."""
    response = client.post("/translate", json={"text": "Hello"})

    assert response.status_code == 200
    # Default decode_sequence produces 3 tokens then EOS, so we don't
    # actually exercise the cap here; the existence test is enough.
    assert response.json()["source"] == "Hello"


# Validation rejections


def test_translate_empty_text_rejected(client: TestClient) -> None:
    """Empty text -> 422 from Field(min_length=1)."""
    response = client.post("/translate", json={"text": ""})

    assert response.status_code == 422


def test_translate_whitespace_only_text_rejected(client: TestClient) -> None:
    """
    Whitespace-only text -> 422 from custom validator.

    The Field min_length=1 alone passes (the raw '   ' is 3 chars),
    so the post-strip empty check in the validator does the rejection.
    """
    response = client.post("/translate", json={"text": "   \n\t  "})

    assert response.status_code == 422
    detail = response.json()["detail"]
    msg = "; ".join(err["msg"] for err in detail)
    assert "non-whitespace" in msg


def test_translate_text_too_long_rejected(client: TestClient) -> None:
    """Text > TEXT_MAX_LENGTH chars -> 422 from Field(max_length)."""
    too_long = "a" * (TEXT_MAX_LENGTH + 1)
    response = client.post("/translate", json={"text": too_long})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "string_too_long" for err in detail)


def test_translate_max_length_zero_rejected(client: TestClient) -> None:
    """max_length=0 -> 422 from Field(ge=1)."""
    response = client.post("/translate", json={"text": "Hello", "max_length": 0})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "greater_than_equal" for err in detail)


def test_translate_max_length_over_25_rejected(client: TestClient) -> None:
    """max_length=26 -> 422 from Field(le=25)."""
    response = client.post(
        "/translate",
        json={"text": "Hello", "max_length": TRANSLATION_MAX_LENGTH + 1},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "less_than_equal" for err in detail)


def test_translate_max_length_negative_rejected(client: TestClient) -> None:
    """max_length=-5 -> 422 (sanity, well below ge=1)."""
    response = client.post("/translate", json={"text": "Hello", "max_length": -5})

    assert response.status_code == 422


# Decode-loop control


def test_translate_early_eos_stops_loop(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """
    Decoder emits EOS after 2 tokens -> n_output_tokens == 2.

    Overrides the default FakeTransformer with one whose sequence is
    [99, 100, EOS]. The router's loop should append 99 then 100 then
    see EOS and break - generated == [BOS, 99, 100], output_ids = [99, 100].
    """
    fake_model: Any = FakeTransformer(decode_sequence=[99, 100, EOS_IDX])
    monkeypatch.setattr(translation_loader, "_MODEL", fake_model)

    response = client.post("/translate", json={"text": "Hello", "max_length": 25})

    assert response.status_code == 200
    assert response.json()["n_output_tokens"] == 2


def test_translate_max_length_cap(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """
    Decoder never emits EOS -> n_output_tokens == max_length.

    Configures FakeTransformer with a long non-EOS sequence so the
    loop runs until req.max_length is hit. Validates the cap path.
    """
    # Use max_length=5 to keep the test fast; never-EOS sequence is
    # all the same dummy id 42.
    fake_model: Any = FakeTransformer(decode_sequence=[42] * 25)
    monkeypatch.setattr(translation_loader, "_MODEL", fake_model)

    response = client.post("/translate", json={"text": "Hello", "max_length": 5})

    assert response.status_code == 200
    assert response.json()["n_output_tokens"] == 5  # capped at max_length


# Input-token reporting


def test_translate_n_input_tokens_capped_at_max(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """
    Tokenizer produces > TRANSLATION_MAX_LENGTH IDs -> n_input_tokens
    is reported as TRANSLATION_MAX_LENGTH (the cap), signaling to the
    client that their input was truncated.
    """
    fake_tokenizer: Any = FakeTokenizer(encode_ids=list(range(50)))  # 50 > 25
    monkeypatch.setattr(translation_loader, "_TOKENIZER", fake_tokenizer)

    response = client.post("/translate", json={"text": "Long text"})

    assert response.status_code == 200
    assert response.json()["n_input_tokens"] == TRANSLATION_MAX_LENGTH


# Loader-state rejection


def test_translate_model_unloaded(client_unloaded: TestClient) -> None:
    """
    /translate returns 503 when either cache is empty.

    Defense-in-depth: the lifespan event guarantees both are loaded
    in production. But if get_model() / get_tokenizer() raises, the
    router returns 503 - same shape /ready uses for the same condition.
    """
    response = client_unloaded.post("/translate", json={"text": "Hello"})

    assert response.status_code == 503
    assert response.json() == {"detail": "model_not_loaded"}
