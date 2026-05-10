# Pylance / Pyright suppression for known-false-positive Keras stubs.
# Same Layer.__call__(self, inputs) issue as in translation_model.py
# and translation_loader.py - mypy strict (the CLI gate) doesn't flag
# it because TF imports resolve to Any. Suppressing only this one
# rule keeps every other type check active.
# pyright: reportCallIssue=false

"""
Router for the /translate endpoint.

WHAT THIS FILE IS:
    The HTTP-facing layer for English-to-Spanish translation. One
    POST handler that orchestrates the multi-step inference pipeline:
        1. Receive a validated TranslationRequest (Pydantic checked
           the input string is non-empty/<=1000 chars and stripped
           whitespace before this code runs)
        2. Tokenize the text via SentencePiece into BPE IDs
        3. Pad the source to fixed length and run the encoder ONCE
        4. Greedy autoregressive decode: starting from <s>, generate
           one token at a time via argmax over the vocabulary, stop
           on </s> or after max_length tokens
        5. Detokenize the generated IDs back to a Spanish string

    Routers stay THIN. The model lifecycle, registry resolution, and
    architecture all live in app/services/. This file is purely the
    HTTP <-> Python objects boundary plus the orchestration of
    tokenize -> tensor -> model.encode -> model.decode (in a loop)
    -> tensor -> detokenize.

URL SHAPE:
    `POST /translate` (no /predict prefix unlike the other services).
    Translation is a distinct task category from prediction
    (regression/classification/generation), so the URL groups it
    separately. The deployment plan locked this naming.

KEY CONCEPTS:

    1. AUTOREGRESSIVE GREEDY DECODING:
        Translation is sequence-to-sequence: the decoder produces
        one token at a time, each conditioned on the encoder output
        AND all previously-generated tokens. The greedy strategy
        picks the argmax (most probable next token) at every step.

        Loop structure:
            tgt = [<s>]
            for step in range(max_length):
                logits = model.decode(tgt, encoder_out, ...)
                next_token = argmax(logits[0, -1])  # last position
                if next_token == </s>: break
                tgt.append(next_token)

        Greedy is simpler and faster than beam search but can produce
        suboptimal translations (no backtracking). The deployment
        plan picks greedy for first ship; beam search is a future
        enhancement if portfolio demo benefits.

    2. ENCODE-ONCE, DECODE-MANY:
        The encoder runs ONCE with the source - its output is cached
        as a local variable and reused across every decode step.
        Avoids re-encoding the source for every generated token (the
        Transformer encoder is the expensive part; rerunning it
        would multiply inference time by max_length).

    3. FIXED-LENGTH SOURCE, GROWING TARGET:
        The encoder expects a fixed (1, MAX_LEN) shape - padded with
        PAD_IDX to match the training distribution. The decoder
        accepts any target length and uses a causal mask, so we let
        the target tensor grow by one token each step (no padding
        needed; eager mode makes shape changes free).

    4. BPE ROUND-TRIP:
        SentencePiece's EncodeAsIds(text) -> list[int] handles
        word -> subword splitting, OOV handling, and special-char
        normalization in one call. DecodeIds(ids) -> str reverses
        it, including stripping any <pad>, <s>, </s>, <unk> special
        tokens (those have empty surface forms in the BPE vocab).

DEFENSE IN DEPTH:
    The lifespan event guarantees both the model and tokenizer are
    loaded before any request. We still defensively catch
    RuntimeError from get_model() / get_tokenizer() and return 503
    - same shape /ready uses for the same condition.
"""

import time

import tensorflow as tf
from fastapi import APIRouter, HTTPException

from app.schemas.translation import (
    TRANSLATION_MAX_LENGTH,
    TranslationRequest,
    TranslationResponse,
)
from app.services import translation_loader
from app.services.translation_model import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    create_causal_mask,
    create_pad_mask,
)

router = APIRouter(tags=["translation"])


@router.post(
    "/translate",
    response_model=TranslationResponse,
    summary="Translate an English sentence to Spanish",
)
async def translate(req: TranslationRequest) -> TranslationResponse:
    """
    Translate an English sentence to Spanish via the Transformer.

    Args:
        req: A TranslationRequest carrying `text` (1-1000 char EN
            string, whitespace stripped by Pydantic) and `max_length`
            (1-25 BPE tokens to generate). FastAPI validates the body
            BEFORE this function runs.

    Returns:
        TranslationResponse with `source` (post-strip echo),
        `translation` (Spanish output), `n_input_tokens` and
        `n_output_tokens` (BPE token counts), and `generation_time_ms`
        (wall-clock time for tokenize + encode + decode + detokenize).

    Raises:
        HTTPException 503: If the Transformer or tokenizer isn't
            loaded (extremely unlikely given the lifespan event;
            defense-in-depth so partial-load states surface cleanly).
    """
    # Defensive accessor pulls. RuntimeError -> 503 mirrors the
    # contract /ready uses for the same "service not yet ready" state.
    try:
        model = translation_loader.get_model()
        tokenizer = translation_loader.get_tokenizer()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="model_not_loaded") from exc

    # Start the clock. Covers tokenize + encode + decode + detokenize -
    # the same scope the schema's generation_time_ms field documents.
    t0 = time.perf_counter()

    # Step 1: tokenize the source string into BPE IDs.
    # EncodeAsIds returns a list[int]; SentencePiece handles word
    # splitting, OOV recovery, and special-char normalization
    # internally. Truncate to TRANSLATION_MAX_LENGTH if longer; the
    # encoder's positional encoding only saw positions 0-24 during
    # training, so longer inputs would produce garbage in those
    # positions.
    src_ids: list[int] = tokenizer.EncodeAsIds(req.text)
    n_input_tokens = min(len(src_ids), TRANSLATION_MAX_LENGTH)
    src_ids = src_ids[:TRANSLATION_MAX_LENGTH]

    # Pad with PAD_IDX to the fixed encoder length. Training-time
    # samples were post-padded (BPE_tokens + <pad>...), so we match.
    src_ids_padded = src_ids + [PAD_IDX] * (TRANSLATION_MAX_LENGTH - len(src_ids))

    # Step 2: encode the source ONCE.
    # Convert to a (1, MAX_LEN) int32 tensor; build the pad mask;
    # call model.encode(). The encoder output is shape
    # (1, MAX_LEN, d_model) and gets cached for the decode loop.
    src = tf.convert_to_tensor([src_ids_padded], dtype=tf.int32)
    src_mask = create_pad_mask(src, PAD_IDX)
    encoder_out = model.encode(src, src_mask, training=False)

    # Step 3: greedy autoregressive decode.
    # Start with [BOS_IDX]. At each step, build the target as a
    # (1, len) tensor, run model.decode(), pick the argmax of the
    # last position's logits, stop on EOS_IDX or after max_length
    # tokens.
    generated: list[int] = [BOS_IDX]
    for _ in range(req.max_length):
        tgt = tf.convert_to_tensor([generated], dtype=tf.int32)
        # tgt_mask combines the pad mask (no pads in our growing
        # target, but the model expects this shape) and the causal
        # mask (don't attend to future positions). tf.maximum is
        # the elementwise OR for {0.0, 1.0}-valued masks.
        tgt_pad_mask = create_pad_mask(tgt, PAD_IDX)
        tgt_causal = create_causal_mask(tf.shape(tgt)[1])
        tgt_mask = tf.maximum(tgt_pad_mask, tgt_causal)
        # logits shape: (1, current_target_len, VOCAB_SIZE).
        logits = model.decode(tgt, encoder_out, src_mask, tgt_mask, training=False)
        # The prediction for the NEXT token comes from the LAST
        # position of the current logits. argmax along the vocab
        # dimension (-1) gives an int64 scalar; cast to plain int.
        next_id = int(tf.argmax(logits[0, -1, :]).numpy())
        if next_id == EOS_IDX:
            break
        generated.append(next_id)

    # Step 4: strip the leading BOS, then detokenize.
    # n_output_tokens counts the generated tokens AFTER BOS - the
    # response field documents this scope. SentencePiece's
    # DecodeIds skips any remaining special tokens (PAD/UNK) with
    # empty surface forms.
    output_ids = generated[1:]
    n_output_tokens = len(output_ids)
    translation = tokenizer.DecodeIds(output_ids)

    generation_time_ms = (time.perf_counter() - t0) * 1000.0

    # response_model=TranslationResponse on the decorator means
    # FastAPI re-validates this object before serializing; defensive
    # against handler-side bugs that might construct the response
    # with the wrong shape.
    return TranslationResponse(
        source=req.text,
        translation=translation,
        n_input_tokens=n_input_tokens,
        n_output_tokens=n_output_tokens,
        generation_time_ms=generation_time_ms,
    )
