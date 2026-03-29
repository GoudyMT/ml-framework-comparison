"""
LSTM Preprocessing — ECG5000 Augmentation + IMDB Padding
Two datasets in one script for LSTM Model #13.

Part A: Load RNN's preprocessed ECG5000, augment minority classes,
        save to data/processed/lstm/ecg/
Part B: Load IMDB via keras, pad/truncate to 300 tokens,
        save to data/processed/lstm/imdb/
"""

import numpy as np
import json
from pathlib import Path

RANDOM_STATE = 113

# Output directories
ECG_DIR = Path('./data/processed/lstm/ecg')
IMDB_DIR = Path('./data/processed/lstm/imdb')


def main():
    print("=" * 60)
    print("LSTM — Preprocessing (ECG5000 Augmentation + IMDB Padding)")
    print("=" * 60)

    # =========================================================
    # PART A: ECG5000 Augmentation
    # =========================================================
    print("\n[1/7] Loading ECG5000 from RNN preprocessing...")
    import sys
    sys.path.insert(0, '.')

    rnn_dir = Path('./data/processed/rnn')
    X_train = np.load(rnn_dir / 'X_train.npy')
    X_test = np.load(rnn_dir / 'X_test.npy')
    y_train = np.load(rnn_dir / 'y_train.npy')
    y_test = np.load(rnn_dir / 'y_test.npy')

    with open(rnn_dir / 'preprocessing_info.json', 'r') as f:
        rnn_meta = json.load(f)

    print(f"  Original train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")

    # [2/7] Show original class distribution
    print("\n[2/7] Original class distribution...")
    classes, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(classes, counts):
        print(f"  Class {cls} ({rnn_meta['class_names'][cls]}): {count}")

    # [3/7] Augment minority classes
    print("\n[3/7] Augmenting minority classes...")
    from utils.rnn_utils import augment_minority_classes

    X_train_aug, y_train_aug = augment_minority_classes(
        X_train, y_train, target_ratio=0.5, random_state=RANDOM_STATE
    )

    print(f"  Augmented train: {X_train_aug.shape} (was {X_train.shape[0]})")
    classes_aug, counts_aug = np.unique(y_train_aug, return_counts=True)
    for cls, count in zip(classes_aug, counts_aug):
        orig = counts[classes == cls][0]
        added = count - orig
        print(f"  Class {cls} ({rnn_meta['class_names'][cls]}): {count} (+{added} synthetic)")

    # [4/7] Save augmented ECG data
    print(f"\n[4/7] Saving augmented ECG to {ECG_DIR}")
    ECG_DIR.mkdir(parents=True, exist_ok=True)

    np.save(ECG_DIR / 'X_train.npy', X_train_aug)
    np.save(ECG_DIR / 'X_test.npy', X_test)
    np.save(ECG_DIR / 'y_train.npy', y_train_aug)
    np.save(ECG_DIR / 'y_test.npy', y_test)

    # Recompute class weights for augmented distribution
    aug_classes, aug_counts = np.unique(y_train_aug, return_counts=True)
    n_samples = len(y_train_aug)
    n_classes = len(aug_classes)
    aug_weights = {int(cls): float(n_samples / (n_classes * count))
                   for cls, count in zip(aug_classes, aug_counts)}

    ecg_info = {
        'dataset': 'ECG5000 (augmented)',
        'source': 'Augmented from data/processed/rnn/',
        'n_train_original': int(len(y_train)),
        'n_train_augmented': int(len(y_train_aug)),
        'n_test': int(len(y_test)),
        'sequence_length': int(X_train.shape[1]),
        'n_features': int(X_train.shape[2]) if X_train.ndim == 3 else 1,
        'n_classes': int(n_classes),
        'class_names': rnn_meta['class_names'],
        'class_weights_augmented': aug_weights,
        'class_weights_original': rnn_meta['class_weights'],
        'augmentation': {
            'target_ratio': 0.5,
            'methods': ['jitter', 'scaling', 'time_warp'],
            'random_state': RANDOM_STATE
        },
        'normalization': rnn_meta['normalization'],
        'random_state': RANDOM_STATE
    }

    with open(ECG_DIR / 'preprocessing_info.json', 'w') as f:
        json.dump(ecg_info, f, indent=2)

    for fname in sorted(ECG_DIR.glob('*.npy')):
        arr = np.load(fname)
        size_kb = fname.stat().st_size / 1024
        print(f"  {fname.name}: {arr.shape} {arr.dtype} ({size_kb:.1f} KB)")
    print(f"  preprocessing_info.json saved")

    # =========================================================
    # PART B: IMDB Padding
    # =========================================================
    print(f"\n{'='*60}")
    print("PART B: IMDB Sentiment Analysis")
    print(f"{'='*60}")

    # [5/7] Load IMDB
    print("\n[5/7] Loading IMDB...")
    from tensorflow.keras.datasets import imdb # type: ignore
    from tensorflow.keras.utils import pad_sequences # type: ignore

    VOCAB_SIZE = 10000
    MAX_LENGTH = 300

    (X_train_imdb, y_train_imdb), (X_test_imdb, y_test_imdb) = imdb.load_data(
        num_words=VOCAB_SIZE
    )
    print(f"  Train: {len(X_train_imdb):,} | Test: {len(X_test_imdb):,}")
    print(f"  Vocab size: {VOCAB_SIZE:,}")

    # [6/7] Pad/truncate sequences
    print(f"\n[6/7] Padding to max_length={MAX_LENGTH} (pre-padding)...")
    X_train_padded = pad_sequences(X_train_imdb, maxlen=MAX_LENGTH,
                                    padding='pre', truncating='pre')
    X_test_padded = pad_sequences(X_test_imdb, maxlen=MAX_LENGTH,
                                   padding='pre', truncating='pre')
    print(f"  Train: {X_train_padded.shape} | Test: {X_test_padded.shape}")
    print(f"  Dtype: {X_train_padded.dtype}")

    # [7/7] Save IMDB data
    print(f"\n[7/7] Saving IMDB to {IMDB_DIR}")
    IMDB_DIR.mkdir(parents=True, exist_ok=True)

    np.save(IMDB_DIR / 'X_train.npy', X_train_padded)
    np.save(IMDB_DIR / 'X_test.npy', X_test_padded)
    np.save(IMDB_DIR / 'y_train.npy', y_train_imdb)
    np.save(IMDB_DIR / 'y_test.npy', y_test_imdb)

    # Save word index for decoding in pipelines
    word_index = imdb.get_word_index()
    with open(IMDB_DIR / 'word_index.json', 'w') as f:
        json.dump(word_index, f)

    imdb_info = {
        'dataset': 'IMDB Sentiment Analysis',
        'source': 'keras.datasets.imdb',
        'n_train': int(len(y_train_imdb)),
        'n_test': int(len(y_test_imdb)),
        'vocab_size': VOCAB_SIZE,
        'max_length': MAX_LENGTH,
        'padding': 'pre',
        'truncating': 'pre',
        'n_classes': 2,
        'class_names': ['Negative', 'Positive'],
        'class_balance': 'Perfectly balanced (50/50)',
        'random_state': RANDOM_STATE
    }

    with open(IMDB_DIR / 'preprocessing_info.json', 'w') as f:
        json.dump(imdb_info, f, indent=2)

    for fname in sorted(IMDB_DIR.glob('*.npy')):
        arr = np.load(fname)
        size_mb = fname.stat().st_size / (1024 * 1024)
        print(f"  {fname.name}: {arr.shape} {arr.dtype} ({size_mb:.1f} MB)")
    print(f"  word_index.json + preprocessing_info.json saved")

    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"  ECG: {ECG_DIR} ({len(y_train_aug):,} augmented train samples)")
    print(f"  IMDB: {IMDB_DIR} ({len(y_train_imdb):,} padded train samples)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()