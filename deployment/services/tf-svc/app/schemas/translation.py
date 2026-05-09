"""
Pydantic schemas for the /translate endpoint.

WHAT THIS FILE IS:
    The API contract for the English-to-Spanish translation endpoint.
    Two classes:
        - TranslationRequest:  what the client MUST send
        - TranslationResponse: what the service WILL return

    FastAPI introspects these to validate incoming JSON, generate the
    OpenAPI schema for /docs, and serialize handler return values.

THE MODEL THIS CONTRACT REFLECTS:
    An encoder-decoder Transformer trained on Tatoeba EN-ES pairs
    (manythings.org/anki spa-eng.zip; 144,873 train pairs after
    filtering). Tokenization is BPE via SentencePiece with a shared
    8,000-token vocab covering both languages. The model has
    d_model=256, 8 heads, 3 encoder layers, 3 decoder layers, and
    was trained with sequence length 25 BPE tokens.

THE INFERENCE PIPELINE:
    Client sends raw English text. The service:
        1. Strips whitespace
        2. Encodes to BPE token IDs (truncated to TRANSLATION_MAX_LENGTH
           if longer)
        3. Runs the encoder once
        4. Greedy autoregressive decode: starts with <s>=1, generates
           one token at a time via argmax over the vocabulary, stops
           on </s>=2 or after max_length tokens
        5. Decodes the generated BPE IDs back to a Spanish string
           (skipping <s>, </s>, and <pad>)

    The client controls only the input text and how many output
    tokens to allow. Tokenization, model state, and decoding all
    happen server-side.

DESIGN DECISIONS:
    - max_length is strictly bounded to [1, 25] to match the
      training-time positional encoding range. The sinusoidal PE
      only saw positions 0-24 during training; generating beyond
      that produces garbage even though the math is technically
      defined.
    - Input text is bounded to 1000 chars defensively. Anything
      longer would be truncated to ~25 BPE tokens by the encoder
      anyway, so longer input just wastes bandwidth.
    - The response separates n_input_tokens from n_output_tokens
      instead of a single ambiguous n_tokens. If n_input_tokens
      equals TRANSLATION_MAX_LENGTH, the client knows their input
      was truncated; n_output_tokens shows whether decoding hit
      </s> naturally or maxed out.
    - source is echoed in the response (the post-strip version, so
      clients can confirm what was actually translated).
"""

from pydantic import BaseModel, Field, field_validator

# Module constants
# ---------------------------------------------------------------------------
"""
Pulled from the modeling phase: data/processed/transformers_translation/
preprocessing_info.json + the trained model's positional-encoding limit.

At module level (not inside a class) so the loader, the router, the
tests, and the smoke test can all import them without re-deriving.

Defensive cap on the input string. The encoder truncates to
TRANSLATION_MAX_LENGTH BPE tokens internally - anything longer than
~150-200 English words just wastes bandwidth.
"""
TEXT_MAX_LENGTH: int = 1000

# Maximum BPE tokens for both encoding (input truncation) and decoding
# (output generation cap). Matches the training-time max sequence
# length; the sinusoidal positional encoding only saw positions 0-24.
TRANSLATION_MAX_LENGTH: int = 25


# Request schema


class TranslationRequest(BaseModel):
    """
    Client payload for POST /translate.

    Attributes:
        text: English sentence to translate. Whitespace is stripped
            before encoding; empty strings are rejected. Max length
            1000 characters (defensive bound; the encoder truncates
            to 25 BPE tokens internally regardless).
        max_length: Maximum BPE tokens to generate during decoding.
            Capped at 25 to match the model's training-time
            positional encoding range. The decoder may stop earlier
            if it produces </s>.

    Example payloads:
        {"text": "Hello, how are you?"}
        {"text": "I love machine learning.", "max_length": 15}
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=TEXT_MAX_LENGTH,
        description=(
            f"English text to translate. 1 to {TEXT_MAX_LENGTH} chars "
            "after whitespace stripping. The service handles BPE "
            "tokenization internally."
        ),
        examples=[
            "Hello, how are you?",
            "I love machine learning.",
            "The cat is on the table.",
        ],
    )

    max_length: int = Field(
        default=TRANSLATION_MAX_LENGTH,
        ge=1,
        le=TRANSLATION_MAX_LENGTH,
        description=(
            f"Maximum BPE tokens to generate, in [1, {TRANSLATION_MAX_LENGTH}]. "
            "Defaults to the training-time max sequence length. "
            "Decoding may stop earlier if the model produces </s>."
        ),
        examples=[10, 25],
    )

    @field_validator("text")
    @classmethod
    def _strip_whitespace(cls, value: str) -> str:
        """
        Strip leading/trailing whitespace, then re-check non-empty.

        Clients sometimes send "Hello\\n" or " Hello " expecting it
        to encode the same as "Hello". Stripping here gives that
        intuition. The post-strip empty-check guards against inputs
        that are pure whitespace (which would otherwise pass the
        Field min_length=1 check on the raw string).
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "text must contain non-whitespace characters; "
                "received only whitespace or empty string"
            )
        return stripped


# Response schema


class TranslationResponse(BaseModel):
    """
    Service payload returned from POST /translate.

    Attributes:
        source: Echo of the post-strip input text. Lets clients
            confirm what was actually translated (vs what was sent
            with extra whitespace).
        translation: Spanish output as a plain string. Decoded from
            the model's BPE token IDs via SentencePiece; <s>, </s>,
            and <pad> are skipped during detokenization.
        n_input_tokens: How many BPE tokens the source was encoded
            into. If this equals TRANSLATION_MAX_LENGTH (25), the
            input was truncated to fit; clients should treat the
            translation as a partial response.
        n_output_tokens: How many BPE tokens the decoder generated
            (excluding <s> but including </s> if it was produced).
            If less than max_length, decoding stopped naturally on
            </s>; if equal to max_length, decoding hit the cap.
        generation_time_ms: Wall-clock time the server spent on
            tokenization + encoder + decoder + detokenization, in
            milliseconds. Excludes JSON serialization and network
            transit.

    Example payload:
        {
          "source": "Hello, how are you?",
          "translation": "Hola, ¿cómo estás?",
          "n_input_tokens": 7,
          "n_output_tokens": 6,
          "generation_time_ms": 142.3
        }
    """

    source: str = Field(
        ...,
        description="Echo of the post-strip input text.",
    )

    translation: str = Field(
        ...,
        description=(
            "Spanish translation. Empty if the decoder produced no "
            "tokens before </s> (rare; indicates the model gave up)."
        ),
    )

    n_input_tokens: int = Field(
        ...,
        ge=0,
        le=TRANSLATION_MAX_LENGTH,
        description=(
            f"BPE token count of the encoded source, in "
            f"[0, {TRANSLATION_MAX_LENGTH}]. If equal to "
            f"{TRANSLATION_MAX_LENGTH}, the input was truncated."
        ),
    )

    n_output_tokens: int = Field(
        ...,
        ge=0,
        le=TRANSLATION_MAX_LENGTH,
        description=(
            f"BPE token count of the decoder output, in "
            f"[0, {TRANSLATION_MAX_LENGTH}]. If less than max_length, "
            "decoding stopped on </s>; if equal, decoding hit the cap."
        ),
    )

    generation_time_ms: float = Field(
        ...,
        ge=0.0,
        description=(
            "Wall-clock time spent on tokenize + encode + decode + "
            "detokenize, in milliseconds. Excludes JSON serialization "
            "and network transit."
        ),
    )
