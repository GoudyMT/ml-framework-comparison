"""
Attention Preprocessing — Tatoeba English-Spanish Translation
Prepares sentence pairs for seq2seq attention models (Model #15).

- Word-level tokenization with punctuation separation
- Separate EN/ES vocabularies (10K each, frequency-based)
- Special tokens: <PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3
- Post-padded to MAX_LENGTH=20
- 80/10/10 train/val/test split

Usage: python preprocess_attention.py
"""

import numpy as np
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter

RANDOM_STATE = 113
VOCAB_SIZE = 10000       # Per language (top 10K most frequent words)
MAX_LENGTH = 20          # Including <SOS> and <EOS> tokens
MAX_RAW_TOKENS = 18      # MAX_LENGTH - 2 (room for <SOS> + <EOS>)
OUTPUT_DIR = Path('./data/processed/attention')

# Special token indices — reserved at the start of every vocabulary
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def tokenize(sentence):
    """
    Lowercase, normalize unicode, and split into tokens.

    Separates punctuation from words so the model sees them as
    distinct tokens (e.g., "don't" → ["don", "'", "t"]).
    Preserves Spanish accented characters (á, é, í, ó, ú, ñ, ü)
    and inverted punctuation (¿, ¡).

    Args:
        sentence: Raw sentence string.

    Returns:
        List of lowercase token strings.
    """
    # Normalize unicode (e.g., composed vs decomposed accents)
    sentence = unicodedata.normalize('NFC', sentence)
    sentence = sentence.lower().strip()

    # Separate punctuation from words: "don't" -> "don ' t"
    # Also separates ¿ ¡ . , ! ? ; : from adjacent words
    sentence = re.sub(r"([.!?¿¡,;:\"\'])", r" \1 ", sentence)

    # Collapse multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    return sentence.split()


def build_vocab(tokenized_sentences, vocab_size):
    """
    Build word->index vocabulary from tokenized sentences.

    Takes the top `vocab_size` most frequent words and assigns
    indices starting at 4 (0-3 reserved for special tokens).
    All other words map to <UNK>=3 during numericalization.

    Args:
        tokenized_sentences: List of token lists.
        vocab_size: Max vocabulary size (excluding special tokens).

    Returns:
        word2idx: Dict mapping word -> integer index.
        idx2word: Dict mapping integer index → word.
    """
    word_counts = Counter()
    for tokens in tokenized_sentences:
        word_counts.update(tokens)

    # Reserve indices 0-3 for special tokens
    word2idx = {'<PAD>': PAD_IDX, '<SOS>': SOS_IDX,
                '<EOS>': EOS_IDX, '<UNK>': UNK_IDX}

    # Add top vocab_size words starting at index 4
    for word, _ in word_counts.most_common(vocab_size):
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def numericalize(tokenized_sentences, word2idx, max_length, add_sos_eos=False):
    """
    Convert token lists to padded integer arrays.

    Maps each word to its vocabulary index (or <UNK> if not found).
    Optionally prepends <SOS> and appends <EOS> for target sequences.
    Post-pads with <PAD>=0 to max_length.

    Args:
        tokenized_sentences: List of token lists.
        word2idx: Vocabulary mapping.
        max_length: Fixed output length (with padding).
        add_sos_eos: If True, wrap sequence with <SOS> and <EOS>.

    Returns:
        NumPy array of shape (n_sentences, max_length), dtype int32.
    """
    result = np.zeros((len(tokenized_sentences), max_length), dtype=np.int32)

    for i, tokens in enumerate(tokenized_sentences):
        if add_sos_eos:
            indices = [SOS_IDX]
            indices += [word2idx.get(t, UNK_IDX) for t in tokens]
            indices.append(EOS_IDX)
        else:
            indices = [word2idx.get(t, UNK_IDX) for t in tokens]

        # Truncate if needed (shouldn't happen after filtering, but safe)
        indices = indices[:max_length]
        result[i, :len(indices)] = indices

    return result


def main():
    print("=" * 60)
    print("PREPROCESSING: ATTENTION (Tatoeba EN→ES)")
    print("=" * 60)

    # [1/7] Load raw data
    print("\n[1/7] Loading raw Tatoeba EN→ES pairs...")
    raw_path = Path('./data/raw/attention/spa.txt')
    pairs = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))

    print(f"    Total pairs loaded: {len(pairs):,}")

    # [2/7] Tokenize and filter by length
    print("\n[2/7] Tokenizing and filtering by length...")
    en_tokenized = []
    es_tokenized = []
    filtered_count = 0

    for en_raw, es_raw in pairs:
        en_tokens = tokenize(en_raw)
        es_tokens = tokenize(es_raw)

        # Keep pairs where BOTH sides fit within MAX_RAW_TOKENS
        if len(en_tokens) <= MAX_RAW_TOKENS and len(es_tokens) <= MAX_RAW_TOKENS:
            en_tokenized.append(en_tokens)
            es_tokenized.append(es_tokens)
        else:
            filtered_count += 1

    print(f"    Kept: {len(en_tokenized):,} pairs")
    print(f"    Filtered (too long): {filtered_count:,} pairs")
    print(f"    Sample: {en_tokenized[100]} → {es_tokenized[100]}")

    # [3/7] Build vocabularies
    print(f"\n[3/7] Building vocabularies (top {VOCAB_SIZE:,} per language)...")
    en_word2idx, en_idx2word = build_vocab(en_tokenized, VOCAB_SIZE)
    es_word2idx, es_idx2word = build_vocab(es_tokenized, VOCAB_SIZE)

    print(f"    EN vocab: {len(en_word2idx):,} words (4 special + {len(en_word2idx)-4:,} learned)")
    print(f"    ES vocab: {len(es_word2idx):,} words (4 special + {len(es_word2idx)-4:,} learned)")

    # Compute <UNK> rate
    en_total = sum(len(t) for t in en_tokenized)
    es_total = sum(len(t) for t in es_tokenized)
    en_unk = sum(1 for tokens in en_tokenized for t in tokens if t not in en_word2idx)
    es_unk = sum(1 for tokens in es_tokenized for t in tokens if t not in es_word2idx)
    print(f"    EN <UNK> rate: {en_unk/en_total*100:.2f}%")
    print(f"    ES <UNK> rate: {es_unk/es_total*100:.2f}%")

    # [4/7] Numericalize sequences
    print("\n[4/7] Numericalizing sequences...")
    # Source (EN): no <SOS>/<EOS> — encoder just reads the input
    en_sequences = numericalize(en_tokenized, en_word2idx, MAX_LENGTH, add_sos_eos=False)
    # Target (ES): <SOS> + tokens + <EOS> — decoder learns start/stop signals
    es_sequences = numericalize(es_tokenized, es_word2idx, MAX_LENGTH, add_sos_eos=True)

    print(f"    EN shape: {en_sequences.shape}, dtype: {en_sequences.dtype}")
    print(f"    ES shape: {es_sequences.shape}, dtype: {es_sequences.dtype}")
    print(f"    EN sample [0]: {en_sequences[0][:10]}...")
    print(f"    ES sample [0]: {es_sequences[0][:10]}...")

    # [5/7] Validate data quality
    print("\n[5/7] Validating data quality...")
    assert not np.any(np.isnan(en_sequences)), "NaN in EN sequences"
    assert not np.any(np.isnan(es_sequences)), "NaN in ES sequences"
    assert en_sequences.min() >= 0, "Negative index in EN"
    assert es_sequences.min() >= 0, "Negative index in ES"
    assert en_sequences.max() < len(en_word2idx), "EN index exceeds vocab"
    assert es_sequences.max() < len(es_word2idx), "ES index exceeds vocab"

    # Every target sequence should start with <SOS>
    assert np.all(es_sequences[:, 0] == SOS_IDX), "Target sequences must start with <SOS>"
    print("    Status: all checks passed ✓")

    # [6/7] Split into train/val/test (80/10/10)
    print("\n[6/7] Splitting into train/val/test (80/10/10)...")
    np.random.seed(RANDOM_STATE)
    n = len(en_sequences)
    indices = np.random.permutation(n)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    src_train, tgt_train = en_sequences[train_idx], es_sequences[train_idx]
    src_val, tgt_val = en_sequences[val_idx], es_sequences[val_idx]
    src_test, tgt_test = en_sequences[test_idx], es_sequences[test_idx]

    print(f"    Train: {src_train.shape[0]:,} pairs")
    print(f"    Val:   {src_val.shape[0]:,} pairs")
    print(f"    Test:  {src_test.shape[0]:,} pairs")

    # [7/7] Save to disk
    print(f"\n[7/7] Saving to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(OUTPUT_DIR / 'src_train.npy', src_train)
    np.save(OUTPUT_DIR / 'src_val.npy', src_val)
    np.save(OUTPUT_DIR / 'src_test.npy', src_test)
    np.save(OUTPUT_DIR / 'tgt_train.npy', tgt_train)
    np.save(OUTPUT_DIR / 'tgt_val.npy', tgt_val)
    np.save(OUTPUT_DIR / 'tgt_test.npy', tgt_test)

    # Save vocabularies
    with open(OUTPUT_DIR / 'src_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(en_word2idx, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / 'tgt_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(es_word2idx, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / 'src_vocab_inv.json', 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in en_idx2word.items()}, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / 'tgt_vocab_inv.json', 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in es_idx2word.items()}, f, ensure_ascii=False, indent=2)

    # Save metadata
    info = {
        "dataset": "Tatoeba EN→ES",
        "source": "manythings.org/anki/ (spa-eng.zip)",
        "task": "machine_translation",
        "total_raw_pairs": len(pairs),
        "filtered_pairs": len(en_tokenized),
        "removed_pairs": filtered_count,
        "n_train": int(src_train.shape[0]),
        "n_val": int(src_val.shape[0]),
        "n_test": int(src_test.shape[0]),
        "max_length": MAX_LENGTH,
        "max_raw_tokens": MAX_RAW_TOKENS,
        "src_vocab_size": len(en_word2idx),
        "tgt_vocab_size": len(es_word2idx),
        "vocab_frequency_cutoff": VOCAB_SIZE,
        "special_tokens": {
            "<PAD>": PAD_IDX,
            "<SOS>": SOS_IDX,
            "<EOS>": EOS_IDX,
            "<UNK>": UNK_IDX
        },
        "tokenization": "word-level, lowercase, punctuation separated",
        "src_language": "English",
        "tgt_language": "Spanish",
        "padding": "post-pad with <PAD>=0",
        "tgt_format": "<SOS> + tokens + <EOS> + <PAD>",
        "src_format": "tokens + <PAD>",
        "random_state": RANDOM_STATE,
        "notes": "Duplicates kept (27.1% of EN have multiple ES translations). "
                 "Accented characters preserved. ¿/¡ as separate tokens."
    }
    with open(OUTPUT_DIR / 'preprocessing_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # Print file summary
    print("\nFiles saved:")
    for fpath in sorted(OUTPUT_DIR.iterdir()):
        size_kb = fpath.stat().st_size / 1024
        print(f"    {fpath.name:30s} {size_kb:>10.1f} KB")

    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()