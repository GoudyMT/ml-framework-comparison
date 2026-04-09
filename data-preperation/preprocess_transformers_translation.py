"""
Transformer Translation Preprocessing — Tatoeba English-Spanish
Prepares sentence pairs for Transformer models.

- BPE subword tokenization via SentencePiece (shared EN+ES vocab)
- Single shared vocabulary (8K tokens, zero UNK rate)
- Special tokens: <pad>=0, <s>=1, </s>=2, <unk>=3
- Post-padded to MAX_LENGTH=25
- 80/10/10 train/val/test split

Key differences from Attention #15 (preprocess_attention.py):
  - BPE replaces word-level tokenization (eliminates UNK entirely)
  - Shared EN+ES vocab (8K) replaces separate vocabs (10K each)
  - SentencePiece .model binary replaces JSON vocab files
  - MAX_LENGTH=25 (up from 20, BPE sequences ~1.37x longer)

Usage: python preprocess_transformers_translation.py
       (run from project root)
"""

import numpy as np
import json
import unicodedata
import re
import tempfile
import sentencepiece as spm
from pathlib import Path

RANDOM_STATE = 113
BPE_VOCAB_SIZE = 8000    # Shared EN+ES (EDA: 8K is sweet spot)
MAX_LENGTH = 25          # BPE P99=20, +BOS/EOS=22, 25 covers 99%+
MAX_RAW_TOKENS = 23      # MAX_LENGTH - 2 (room for BOS + EOS on target)
OUTPUT_DIR = Path('./data/processed/transformers_translation')

# SentencePiece special token indices
PAD_IDX = 0   # <pad>
BOS_IDX = 1   # <s>  (start of sequence)
EOS_IDX = 2   # </s> (end of sequence)
UNK_IDX = 3   # <unk>


def normalize_text(sentence):
    """
    Lowercase and normalize unicode for consistent tokenization.

    Unlike #15's tokenize(), we don't split punctuation manually —
    SentencePiece handles subword boundaries including punctuation.
    We only need to normalize casing and unicode.

    Args:
        sentence: Raw sentence string.

    Returns:
        Cleaned lowercase string.
    """
    sentence = unicodedata.normalize('NFC', sentence)
    sentence = sentence.lower().strip()
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def main():
    print("=" * 60)
    print("PREPROCESSING: TRANSFORMERS TRANSLATION (Tatoeba EN→ES)")
    print("=" * 60)

    # [1/8] Load raw data
    print("\n[1/8] Loading raw Tatoeba EN→ES pairs...")
    raw_path = Path('./data/raw/attention/spa.txt')
    pairs = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[0].strip(), parts[1].strip()))

    print(f"    Total pairs loaded: {len(pairs):,}")

    # [2/8] Clean and normalize
    print("\n[2/8] Cleaning and normalizing text...")
    en_clean = [normalize_text(en) for en, es in pairs]
    es_clean = [normalize_text(es) for en, es in pairs]

    print(f"    Sample: '{en_clean[100]}' → '{es_clean[100]}'")

    # [3/8] Train SentencePiece BPE model
    print(f"\n[3/8] Training SentencePiece BPE ({BPE_VOCAB_SIZE:,} shared vocab)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpe_prefix = str(OUTPUT_DIR / 'bpe')

    # Write combined EN+ES corpus to temp file for training
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                     encoding='utf-8') as tmp:
        for en, es in zip(en_clean, es_clean):
            tmp.write(en + '\n')
            tmp.write(es + '\n')
        tmp_path = tmp.name

    spm.SentencePieceTrainer.train( # type: ignore
        input=tmp_path,
        model_prefix=bpe_prefix,
        vocab_size=BPE_VOCAB_SIZE,
        model_type='bpe',
        pad_id=PAD_IDX,
        bos_id=BOS_IDX,
        eos_id=EOS_IDX,
        unk_id=UNK_IDX,
        character_coverage=1.0,
        shuffle_input_sentence=True,
    )

    # Clean up temp file
    Path(tmp_path).unlink()

    # Load trained BPE model
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_prefix + '.model') # type: ignore
    print(f"    Vocab size: {sp.get_piece_size():,}") # type: ignore
    print(f"    Model saved: {bpe_prefix}.model")

    # Show tokenization examples
    print(f"\n    Tokenization examples:")
    for i in [0, 100, 5000]:
        en_tokens = sp.encode(en_clean[i], out_type=str) # type: ignore
        es_tokens = sp.encode(es_clean[i], out_type=str) # type: ignore
        print(f"      EN: '{en_clean[i][:60]}' → {en_tokens[:8]}{'...' if len(en_tokens) > 8 else ''} ({len(en_tokens)} tokens)")
        print(f"      ES: '{es_clean[i][:60]}' → {es_tokens[:8]}{'...' if len(es_tokens) > 8 else ''} ({len(es_tokens)} tokens)")

    # [4/8] Encode all sentences with BPE
    print("\n[4/8] Encoding all sentences with BPE...")
    en_encoded = [sp.encode(s) for s in en_clean] # type: ignore
    es_encoded = [sp.encode(s) for s in es_clean] # type: ignore

    en_lengths = [len(s) for s in en_encoded]
    es_lengths = [len(s) for s in es_encoded]
    print(f"    EN mean length: {np.mean(en_lengths):.1f} BPE tokens")
    print(f"    ES mean length: {np.mean(es_lengths):.1f} BPE tokens")

    # [5/8] Filter by BPE token length
    print(f"\n[5/8] Filtering by BPE length (max {MAX_RAW_TOKENS} tokens per side)...")
    en_filtered = []
    es_filtered = []
    filtered_count = 0

    for en_ids, es_ids in zip(en_encoded, es_encoded):
        if len(en_ids) <= MAX_RAW_TOKENS and len(es_ids) <= MAX_RAW_TOKENS:
            en_filtered.append(en_ids)
            es_filtered.append(es_ids)
        else:
            filtered_count += 1

    print(f"    Kept: {len(en_filtered):,} pairs")
    print(f"    Filtered (too long): {filtered_count:,} pairs")
    print(f"    Retention: {len(en_filtered)/len(pairs)*100:.1f}%")

    # [6/8] Add BOS/EOS to target + pad to MAX_LENGTH
    print(f"\n[6/8] Adding BOS/EOS to target, padding to MAX_LENGTH={MAX_LENGTH}...")

    # Source (EN): raw BPE tokens + padding (no BOS/EOS — encoder input)
    src_array = np.full((len(en_filtered), MAX_LENGTH), PAD_IDX, dtype=np.int32)
    for i, ids in enumerate(en_filtered):
        src_array[i, :len(ids)] = ids

    # Target (ES): BOS + BPE tokens + EOS + padding (decoder input/output)
    tgt_array = np.full((len(es_filtered), MAX_LENGTH), PAD_IDX, dtype=np.int32)
    for i, ids in enumerate(es_filtered):
        seq = [BOS_IDX] + ids + [EOS_IDX]
        tgt_array[i, :len(seq)] = seq

    print(f"    Source shape: {src_array.shape}, dtype: {src_array.dtype}")
    print(f"    Target shape: {tgt_array.shape}, dtype: {tgt_array.dtype}")
    print(f"    Source sample [0]: {src_array[0][:12]}...")
    print(f"    Target sample [0]: {tgt_array[0][:12]}...")

    # Validate
    assert np.all(tgt_array[:, 0] == BOS_IDX), "All targets must start with BOS"
    assert src_array.min() >= 0, "Negative index in source"
    assert tgt_array.min() >= 0, "Negative index in target"
    assert src_array.max() < BPE_VOCAB_SIZE, "Source index exceeds vocab"
    assert tgt_array.max() < BPE_VOCAB_SIZE, "Target index exceeds vocab"
    print("    Validation: all checks passed")

    # [7/8] Split train/val/test (80/10/10)
    print("\n[7/8] Splitting into train/val/test (80/10/10)...")
    np.random.seed(RANDOM_STATE)
    n = len(en_filtered)
    indices = np.random.permutation(n)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    src_train, tgt_train = src_array[train_idx], tgt_array[train_idx]
    src_val, tgt_val = src_array[val_idx], tgt_array[val_idx]
    src_test, tgt_test = src_array[test_idx], tgt_array[test_idx]

    print(f"    Train: {src_train.shape[0]:,} pairs")
    print(f"    Val:   {src_val.shape[0]:,} pairs")
    print(f"    Test:  {src_test.shape[0]:,} pairs")

    # [8/8] Save to folder
    print(f"\n[8/8] Saving to {OUTPUT_DIR}...")

    # Save arrays
    np.save(OUTPUT_DIR / 'src_train.npy', src_train)
    np.save(OUTPUT_DIR / 'src_val.npy', src_val)
    np.save(OUTPUT_DIR / 'src_test.npy', src_test)
    np.save(OUTPUT_DIR / 'tgt_train.npy', tgt_train)
    np.save(OUTPUT_DIR / 'tgt_val.npy', tgt_val)
    np.save(OUTPUT_DIR / 'tgt_test.npy', tgt_test)

    # BPE model already saved during training (bpe.model + bpe.vocab)

    # Save metadata
    info = {
        "dataset": "Tatoeba EN→ES",
        "source": "manythings.org/anki/ (spa-eng.zip)",
        "task": "machine_translation",
        "tokenization": "BPE (SentencePiece), shared EN+ES",
        "total_raw_pairs": len(pairs),
        "filtered_pairs": len(en_filtered),
        "removed_pairs": filtered_count,
        "n_train": int(src_train.shape[0]),
        "n_val": int(src_val.shape[0]),
        "n_test": int(src_test.shape[0]),
        "max_length": MAX_LENGTH,
        "max_raw_tokens": MAX_RAW_TOKENS,
        "bpe_vocab_size": BPE_VOCAB_SIZE,
        "special_tokens": {
            "<pad>": PAD_IDX,
            "<s>": BOS_IDX,
            "</s>": EOS_IDX,
            "<unk>": UNK_IDX
        },
        "src_language": "English",
        "tgt_language": "Spanish",
        "padding": "post-pad with <pad>=0",
        "tgt_format": "<s> + BPE_tokens + </s> + <pad>",
        "src_format": "BPE_tokens + <pad>",
        "bpe_expansion_ratio": "~1.37x vs word-level",
        "random_state": RANDOM_STATE,
        "comparison_with_attention_15": {
            "tokenization": "BPE shared 8K vs word-level separate 10K each",
            "unk_rate": "0% vs EN 2.6% / ES 7.3%",
            "max_length": "25 vs 20",
            "vocab_files": "bpe.model binary vs JSON word2idx/idx2word"
        },
        "notes": "Same raw data as Attention #15. BPE eliminates UNK tokens. "
                 "Shared vocab captures EN/ES cognates naturally. "
                 "BLEU comparison with #15 requires decoding BPE back to words."
    }
    with open(OUTPUT_DIR / 'preprocessing_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # Print file summary
    print("\n    Files saved:")
    for fpath in sorted(OUTPUT_DIR.iterdir()):
        size_kb = fpath.stat().st_size / 1024
        print(f"      {fpath.name:35s} {size_kb:>10.1f} KB")

    print("\n    Preprocessing complete!")

if __name__ == '__main__':
    main()