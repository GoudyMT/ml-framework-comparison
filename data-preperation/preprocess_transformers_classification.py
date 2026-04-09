"""
Transformer Classification Preprocessing — AG News (4-class)
Prepares news articles for encoder-only Transformer + DistilBERT fine-tuning (Model #16).

- AG News: World (0), Sports (1), Business (2), Sci/Tech (3)
- Text cleaning: HTML entities, malformed quotes, escaped newlines
- BPE subword tokenization via SentencePiece (16K English vocab)
- Special tokens: <pad>=0, <s>=1, </s>=2, <unk>=3
- Truncated/padded to MAX_LENGTH=128
- Train 108K / Val 12K / Test 7.6K (stratified)

Note: The BPE preprocessing is for the from-scratch encoder-only
Transformer. DistilBERT fine-tuning uses its own WordPiece tokenizer
and does NOT use these preprocessed files.

Usage: python preprocess_transformers_classification.py
       (run from project root)
"""

import numpy as np
import json
import re
import html
import tempfile
import pandas as pd
import sentencepiece as spm
from pathlib import Path

RANDOM_STATE = 113
BPE_VOCAB_SIZE = 16000   # English only (EDA: 16K covers 91.2% word-level)
MAX_LENGTH = 128          # BPE P99 ≈ 98 estimated, 128 covers all
OUTPUT_DIR = Path('./data/processed/transformers_classification')

# SentencePiece special token indices
PAD_IDX = 0   # <pad>
BOS_IDX = 1   # <s>
EOS_IDX = 2   # </s>
UNK_IDX = 3   # <unk>

CLASS_NAMES = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

def clean_text(text):
    """
    Clean AG News text artifacts found during EDA.

    Addresses specific issues in order:
      1. html.unescape() — proper entities (&lt; &gt; &amp;)
      2. Strip HTML tags — leftovers from unescape (<strong> etc.)
      3. Fix malformed quotes — ' quot;' and '#39;' (26.7% of data)
      4. Fix escaped newlines — backslashes (11.0% of data)
      5. Collapse whitespace + lowercase

    Args:
        text: Raw AG News text string.

    Returns:
        Cleaned lowercase string.
    """
    # 1. Decode proper HTML entities
    text = html.unescape(text)

    # 2. Strip any remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. Fix malformed quote artifacts (not proper HTML entities)
    text = text.replace(' quot;', '"').replace('quot;', '"')
    text = text.replace("#39;", "'")

    # 4. Fix escaped newlines/characters
    text = text.replace('\\', ' ')

    # 5. Normalize whitespace + lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()

    return text

def main():
    print("=" * 60)
    print("PREPROCESSING: TRANSFORMERS CLASSIFICATION (AG News)")
    print("=" * 60)

    # [1/8] Load AG News CSVs
    print("\n[1/8] Loading AG News data...")
    train_path = Path('./data/raw/transformers/ag_news/train.csv')
    test_path = Path('./data/raw/transformers/ag_news/test.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"    Train: {len(train_df):,} samples")
    print(f"    Test:  {len(test_df):,} samples")
    print(f"    Classes: {CLASS_NAMES}")

    # [2/8] Clean text
    print("\n[2/8] Cleaning text (HTML entities, quot;, #39;, backslashes)...")
    train_df['text_clean'] = train_df['text'].apply(clean_text)
    test_df['text_clean'] = test_df['text'].apply(clean_text)

    # Show cleaning examples
    dirty_idx = train_df[train_df['text'].str.contains('quot;', na=False)].index[0]
    print(f"    Before: {train_df.loc[dirty_idx, 'text'][:100]}...") # type: ignore
    print(f"    After:  {train_df.loc[dirty_idx, 'text_clean'][:100]}...") # type: ignore

    # Null check after cleaning
    null_count = train_df['text_clean'].isna().sum() + test_df['text_clean'].isna().sum()
    empty_count = (train_df['text_clean'].str.len() == 0).sum() + (test_df['text_clean'].str.len() == 0).sum()
    print(f"    Null after cleaning: {null_count}")
    print(f"    Empty after cleaning: {empty_count}")

    # [3/8] Train SentencePiece BPE model (on training text ONLY)
    print(f"\n[3/8] Training SentencePiece BPE ({BPE_VOCAB_SIZE:,} vocab, train set only)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bpe_prefix = str(OUTPUT_DIR / 'bpe')

    # Write training text to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                     encoding='utf-8') as tmp:
        for text in train_df['text_clean']:
            tmp.write(text + '\n')
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

    Path(tmp_path).unlink()

    sp = spm.SentencePieceProcessor()
    sp.load(bpe_prefix + '.model') # type: ignore
    print(f"    Vocab size: {sp.get_piece_size():,}") # type: ignore
    print(f"    Model saved: {bpe_prefix}.model")

    # [4/8] Encode all texts with BPE
    print("\n[4/8] Encoding all texts with BPE...")
    train_encoded = [sp.encode(t) for t in train_df['text_clean']] # type: ignore
    test_encoded = [sp.encode(t) for t in test_df['text_clean']] # type: ignore

    train_bpe_lens = [len(s) for s in train_encoded]
    print(f"    Train BPE lengths: mean={np.mean(train_bpe_lens):.1f}, "
          f"P95={np.percentile(train_bpe_lens, 95):.0f}, "
          f"P99={np.percentile(train_bpe_lens, 99):.0f}, "
          f"max={max(train_bpe_lens)}")

    # [5/8] Truncate/pad to MAX_LENGTH
    print(f"\n[5/8] Truncating/padding to MAX_LENGTH={MAX_LENGTH}...")

    def encode_to_array(encoded_list, max_length):
        array = np.full((len(encoded_list), max_length), PAD_IDX, dtype=np.int32)
        truncated = 0
        for i, ids in enumerate(encoded_list):
            if len(ids) > max_length:
                ids = ids[:max_length]
                truncated += 1
            array[i, :len(ids)] = ids
        return array, truncated

    X_train_full, tr_trunc = encode_to_array(train_encoded, MAX_LENGTH)
    X_test, te_trunc = encode_to_array(test_encoded, MAX_LENGTH)

    print(f"    Train truncated: {tr_trunc:,} / {len(train_encoded):,} ({tr_trunc/len(train_encoded)*100:.2f}%)")
    print(f"    Test truncated:  {te_trunc:,} / {len(test_encoded):,} ({te_trunc/len(test_encoded)*100:.2f}%)")
    print(f"    X_train shape: {X_train_full.shape}, dtype: {X_train_full.dtype}")
    print(f"    X_test shape:  {X_test.shape}, dtype: {X_test.dtype}")

    # [6/8] Extract labels
    print("\n[6/8] Extracting labels...")
    y_train_full = train_df['label'].values.astype(np.int32)
    y_test = test_df['label'].values.astype(np.int32)

    assert y_train_full.min() == 0 and y_train_full.max() == 3, "Labels must be 0-3" # type: ignore
    assert y_test.min() == 0 and y_test.max() == 3, "Labels must be 0-3" # type: ignore
    print(f"    Labels: {np.unique(y_train_full)} (already 0-indexed)") # type: ignore

    # [7/8] Split train → train + val (90/10, stratified)
    print("\n[7/8] Splitting train into train/val (90/10, stratified)...")
    np.random.seed(RANDOM_STATE)

    # Stratified split: maintain class balance
    train_indices = []
    val_indices = []
    for label in sorted(CLASS_NAMES.keys()):
        label_idx = np.where(y_train_full == label)[0]
        np.random.shuffle(label_idx)
        n_val = int(0.1 * len(label_idx))
        val_indices.extend(label_idx[:n_val])
        train_indices.extend(label_idx[n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    X_train = X_train_full[train_indices]
    y_train = y_train_full[train_indices]
    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]

    print(f"    Train: {X_train.shape[0]:,} samples")
    print(f"    Val:   {X_val.shape[0]:,} samples")
    print(f"    Test:  {X_test.shape[0]:,} samples")

    # Verify stratification
    print(f"\n    Class balance verification:")
    for label in sorted(CLASS_NAMES.keys()):
        tr_pct = (y_train == label).sum() / len(y_train) * 100 # type: ignore
        va_pct = (y_val == label).sum() / len(y_val) * 100 # type: ignore
        te_pct = (y_test == label).sum() / len(y_test) * 100 # type: ignore
        print(f"      {CLASS_NAMES[label]:<10} train={tr_pct:.1f}%  val={va_pct:.1f}%  test={te_pct:.1f}%")

    # [8/8] Save to disk
    print(f"\n[8/8] Saving to {OUTPUT_DIR}...")

    np.save(OUTPUT_DIR / 'X_train.npy', X_train)
    np.save(OUTPUT_DIR / 'X_val.npy', X_val)
    np.save(OUTPUT_DIR / 'X_test.npy', X_test)
    np.save(OUTPUT_DIR / 'y_train.npy', y_train) # type: ignore
    np.save(OUTPUT_DIR / 'y_val.npy', y_val) # type: ignore
    np.save(OUTPUT_DIR / 'y_test.npy', y_test) # type: ignore

    # BPE model already saved during training

    # Save metadata
    info = {
        "dataset": "AG News",
        "source": "HuggingFace datasets (ag_news)",
        "task": "text_classification",
        "n_classes": 4,
        "class_names": CLASS_NAMES,
        "tokenization": "BPE (SentencePiece), English only",
        "bpe_vocab_size": BPE_VOCAB_SIZE,
        "max_length": MAX_LENGTH,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "total_train_before_split": int(len(train_df)),
        "special_tokens": {
            "<pad>": PAD_IDX,
            "<s>": BOS_IDX,
            "</s>": EOS_IDX,
            "<unk>": UNK_IDX
        },
        "cleaning_steps": [
            "html.unescape()",
            "strip HTML tags",
            "replace ' quot;' → '\"'",
            "replace '#39;' → \"'\"",
            "replace '\\\\' → ' '",
            "collapse whitespace",
            "strip + lowercase"
        ],
        "padding": "post-pad with <pad>=0, truncate if > MAX_LENGTH",
        "split_strategy": "original test set preserved, train split 90/10 stratified",
        "random_state": RANDOM_STATE,
        "notes": "BPE preprocessing for from-scratch encoder-only Transformer. "
                 "DistilBERT fine-tuning uses its own WordPiece tokenizer, not this BPE model."
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