# Pylance / Pyright suppression for known-false-positive Keras stubs.
# Same Layer.__call__(self, inputs) issue as in translation_model.py
# and translation_loader.py - Pylance's bundled TF stubs reject the
# multi-positional Keras pattern (e.g., `model(src, tgt, training=False)`),
# but Keras dispatches via *args/**kwargs at runtime so any signature
# works. Mypy strict (the CLI gate) doesn't flag it because TF imports
# resolve to Any.
# pyright: reportCallIssue=false

"""
End-to-end smoke test for the tf-svc /translate endpoint.

WHAT THIS SCRIPT DOES:
    1. Loads the real Transformer + SentencePiece tokenizer from the
       modeling phase (TensorFlow/16-transformers/results/...).
    2. For a fixed set of EN sentences, manually runs the same
       pipeline the router runs: tokenize -> encode -> greedy
       autoregressive decode -> detokenize. This is the EXPECTED
       output.
    3. POSTs each sentence to a running tf-svc.
    4. String-exact compares the response against the manual
       pipeline output (translation, source echo, n_input_tokens,
       n_output_tokens).

WHY STRING-EXACT:
    Greedy decoding is deterministic given identical model weights
    and identical input. The router's tokenize + encode + argmax
    loop + detokenize chain has no randomness, no per-call state,
    and no float-precision amplification points capable of flipping
    an argmax decision (Q-values in this network are well-separated
    at typical inference time). So the manual pipeline and the
    server pipeline should produce byte-identical translations.

USAGE (from deployment/services/tf-svc/):
    1. Boot the server:
        .venv\\Scripts\\uvicorn.exe app.main:app --port 8003
    2. Run this script:
        .venv\\Scripts\\python.exe scripts/smoke_test.py

ASSUMPTIONS:
    - The server is running on localhost:8003 (tf-svc port).
    - The service's loaded Transformer is the same artifact as
      TensorFlow/16-transformers/results/translation_transformer.weights.h5.
    - The service's loaded tokenizer is the same artifact as
      data/processed/transformers_translation/bpe.model (the
      preprocessing output that promote_to_registry.py copies into
      the registry).
"""

import os
import sys
from pathlib import Path
from typing import cast

# Quiet TF info/warning spam BEFORE importing tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Make `app` importable. When this script runs directly (not via
# uvicorn or pytest), Python only adds the script's own directory
# (`scripts/`) to sys.path - not the parent service dir. Inserting
# the parent makes `from app.x import y` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import sentencepiece as spm  # noqa: E402
import tensorflow as tf  # noqa: E402

from app.schemas.translation import TRANSLATION_MAX_LENGTH  # noqa: E402
from app.services.translation_model import (  # noqa: E402
    BOS_IDX,
    D_FF,
    D_MODEL,
    DROPOUT,
    EOS_IDX,
    MAX_LEN,
    N_DEC_LAYERS,
    N_ENC_LAYERS,
    N_HEADS,
    PAD_IDX,
    VOCAB_SIZE,
    Transformer,
    create_causal_mask,
    create_pad_mask,
)

# Path resolution

SCRIPT_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT  = SCRIPT_DIR.parents[3]
WEIGHTS_PATH  = (
    PROJECT_ROOT / "TensorFlow" / "16-transformers" / "results"
    / "translation_transformer.weights.h5"
)
# bpe.model is sourced from the data preprocessing output (the same
# place promote_to_registry.py copies it from); the modeling-phase
# notebook saves it there alongside the train/val/test BPE-encoded
# arrays.
TOKENIZER_PATH = (
    PROJECT_ROOT / "data" / "processed" / "transformers_translation" / "bpe.model"
)

SERVICE_URL  = "http://localhost:8003/translate"
TIMEOUT_SEC  = 30.0   # generous - greedy decode runs the decoder up to 25 times

# Targeted EN sentences. Cover greeting, technical phrase, simple
# SVO, question, and a slightly longer sentence to exercise the
# tokenization + truncation path.
TEST_SENTENCES: tuple[tuple[str, int], ...] = (
    ("Hello, how are you?", 25),
    ("I love machine learning.", 25),
    ("The cat is on the table.", 25),
    ("Where is the bank?", 25),
    ("She walks to the store every morning.", 25),
)


def _translate_locally(
    model: Transformer, tokenizer: spm.SentencePieceProcessor,
    text: str, max_length: int,
) -> tuple[str, int, int]:
    """
    Re-run the router's pipeline locally for forward verification.

    Returns:
        (translation_str, n_input_tokens, n_output_tokens) tuple.
    """
    src_ids: list[int] = tokenizer.EncodeAsIds(text)
    n_input_tokens = min(len(src_ids), TRANSLATION_MAX_LENGTH)
    src_ids = src_ids[:TRANSLATION_MAX_LENGTH]
    src_ids_padded = src_ids + [PAD_IDX] * (TRANSLATION_MAX_LENGTH - len(src_ids))

    src = tf.convert_to_tensor([src_ids_padded], dtype=tf.int32)
    src_mask = create_pad_mask(src, PAD_IDX)
    encoder_out = model.encode(src, src_mask, training=False)

    generated: list[int] = [BOS_IDX]
    for _ in range(max_length):
        tgt = tf.convert_to_tensor([generated], dtype=tf.int32)
        tgt_pad_mask = create_pad_mask(tgt, PAD_IDX)
        tgt_causal = create_causal_mask(tf.shape(tgt)[1])
        tgt_mask = tf.maximum(tgt_pad_mask, tgt_causal)
        logits = model.decode(tgt, encoder_out, src_mask, tgt_mask, training=False)
        next_id = int(tf.argmax(logits[0, -1, :]).numpy())
        if next_id == EOS_IDX:
            break
        generated.append(next_id)

    output_ids = generated[1:]
    n_output_tokens = len(output_ids)
    translation = tokenizer.DecodeIds(output_ids)
    return translation, n_input_tokens, n_output_tokens


def main() -> int:
    print("=" * 70)
    print("tf-svc /translate smoke test")
    print("=" * 70)

    # Step 1: load modeling-phase artifacts.
    print("\n[1/4] Loading Transformer + tokenizer from modeling phase...")
    tokenizer = spm.SentencePieceProcessor()
    if not tokenizer.Load(str(TOKENIZER_PATH)):
        print(f"      ERROR: failed to load tokenizer from {TOKENIZER_PATH}")
        return 1
    print(f"      tokenizer: vocab_size={tokenizer.GetPieceSize()}")

    model = Transformer(
        src_vocab=VOCAB_SIZE, tgt_vocab=VOCAB_SIZE,
        d_model=D_MODEL, n_heads=N_HEADS,
        n_enc_layers=N_ENC_LAYERS, n_dec_layers=N_DEC_LAYERS,
        d_ff=D_FF, max_len=MAX_LEN,
        dropout_rate=DROPOUT, pad_idx=PAD_IDX,
    )
    # Dummy forward to materialize Keras's lazy weights, then load.
    dummy_src = tf.zeros((1, MAX_LEN), dtype=tf.int32)
    dummy_tgt = tf.zeros((1, MAX_LEN), dtype=tf.int32)
    _ = model(dummy_src, dummy_tgt, training=False)
    model.load_weights(str(WEIGHTS_PATH))
    n_params = model.count_params()
    print(f"      model:     n_params={n_params:,}")

    # Step 2: open one httpx client and reuse the connection.
    try:
        client = httpx.Client(timeout=TIMEOUT_SEC)
    except Exception as exc:
        print(f"      ERROR creating httpx client: {exc}")
        return 1

    # Step 3: per-sentence forward verification.
    print(f"\n[2/4] Per-sentence forward verification ({len(TEST_SENTENCES)} sentences)...")
    failures: list[str] = []
    try:
        # cast() narrows Pylance's view: subclassing keras.Model (which
        # resolves to Model[Unknown, Unknown] under bundled stubs)
        # loses the Transformer subclass identity at the constructor's
        # return type. The runtime type IS Transformer; cast just tells
        # the type-checker what we already know. Mypy treats TF imports
        # as Any so it considers the cast redundant - the per-line
        # ignore silences mypy without affecting Pylance.
        model_typed = cast(Transformer, model)  # type: ignore[redundant-cast]

        for text, max_length in TEST_SENTENCES:
            # Manual pipeline.
            exp_translation, exp_n_in, exp_n_out = _translate_locally(
                model_typed, tokenizer, text, max_length,
            )

            # Server pipeline.
            try:
                response = client.post(
                    SERVICE_URL,
                    json={"text": text, "max_length": max_length},
                )
            except httpx.ConnectError:
                print(f"      ERROR: cannot connect to {SERVICE_URL}.")
                print("      Is the server running? Try in another terminal:")
                print("        .venv\\Scripts\\uvicorn.exe app.main:app --port 8003")
                return 1

            if response.status_code != 200:
                print(f"      FAIL: server returned {response.status_code} for "
                      f"{text!r}: {response.text[:200]}")
                return 1

            body = response.json()
            svc_source = body["source"]
            svc_translation = body["translation"]
            svc_n_in = body["n_input_tokens"]
            svc_n_out = body["n_output_tokens"]

            # Bit-exact comparison.
            ok = (
                svc_source == text
                and svc_translation == exp_translation
                and svc_n_in == exp_n_in
                and svc_n_out == exp_n_out
            )
            mark = "OK  " if ok else "FAIL"
            print(f"      [{mark}] EN: {text!r}")
            print(f"             ES: {svc_translation!r}")
            print(f"             n_input={svc_n_in}  n_output={svc_n_out}  "
                  f"gen_ms={body['generation_time_ms']:.1f}")
            if not ok:
                if svc_translation != exp_translation:
                    print(f"             expected ES: {exp_translation!r}")
                if svc_n_in != exp_n_in:
                    print(f"             expected n_input: {exp_n_in}")
                if svc_n_out != exp_n_out:
                    print(f"             expected n_output: {exp_n_out}")
                if svc_source != text:
                    print(f"             expected source: {text!r}")
                failures.append(text)

        # Step 4: summary.
        print("\n[3/4] Summary")
        print(f"      sentences tested: {len(TEST_SENTENCES)}")
        print(f"      failures:         {len(failures)}")

        # Step 5 (additional): /metrics traffic confirmation.
        print("\n[4/4] /metrics traffic confirmation...")
        m = client.get("http://localhost:8003/metrics").text
        translate_count_lines = [
            line for line in m.split("\n")
            if line.startswith("http_requests_total{")
            and 'path="/translate"' in line
        ]
        for line in translate_count_lines:
            print(f"      {line}")

    finally:
        client.close()

    if failures:
        print("\n" + "=" * 70)
        print(f"  FAIL - {len(failures)} of {len(TEST_SENTENCES)} sentences disagreed")
        print("=" * 70)
        return 1

    print("\n" + "=" * 70)
    print("  PASS - service-decoded translations bit-match manual forward")
    print(f"  Coverage: {len(TEST_SENTENCES)} sentences, all string-exact")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
