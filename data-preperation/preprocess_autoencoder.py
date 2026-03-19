"""
Preprocessing script for Autoencoders — CIFAR-10 dataset.

Downloads CIFAR-10 via Keras, normalizes to [0,1], saves both
flattened (for dense AE) and image-shaped (for conv AE) arrays.
Labels kept for downstream evaluation only — autoencoders are self-supervised.

Usage:
    python preprocess_autoencoder.py
"""

import numpy as np
import json
from pathlib import Path

# Configuration
RANDOM_STATE = 113
OUTPUT_DIR = Path("./data/processed/autoencoder")


def main():
    print("=" * 60)
    print("PREPROCESSING: Autoencoders (CIFAR-10)")
    print("=" * 60)

    # Step 1: Load CIFAR-10
    print("\n[1/5] Loading CIFAR-10 via Keras...")
    from tensorflow.keras.datasets import cifar10
    (X_train_raw, y_train), (X_test_raw, y_test) = cifar10.load_data()

    # Flatten labels from (N, 1) to (N,)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print(f"    Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")
    print(f"    Labels: {y_train.shape} / {y_test.shape}")
    print(f"    Dtype: {X_train_raw.dtype}, range [{X_train_raw.min()}, {X_train_raw.max()}]")

    # Step 2: Validate data quality
    print("\n[2/5] Validating data quality...")
    X_train_f = X_train_raw.astype(np.float32)
    X_test_f = X_test_raw.astype(np.float32)

    assert not np.isnan(X_train_f).any(), "NaN found in train set"
    assert not np.isnan(X_test_f).any(), "NaN found in test set"
    assert not np.isinf(X_train_f).any(), "Inf found in train set"
    assert not np.isinf(X_test_f).any(), "Inf found in test set"
    assert X_train_raw.min() >= 0 and X_train_raw.max() <= 255, "Pixel range outside [0, 255]"
    print("    No NaN, no Inf, pixel range [0, 255] confirmed")

    # Step 3: Normalize to [0, 1] float32
    print("\n[3/5] Normalizing to [0, 1] float32...")
    X_train_norm = X_train_raw.astype(np.float32) / 255.0
    X_test_norm = X_test_raw.astype(np.float32) / 255.0
    print(f"    Train range: [{X_train_norm.min():.1f}, {X_train_norm.max():.1f}]")
    print(f"    Test range:  [{X_test_norm.min():.1f}, {X_test_norm.max():.1f}]")
    print(f"    Dtype: {X_train_norm.dtype}")

    # Step 4: Create flattened and image-shaped versions
    print("\n[4/5] Creating flattened + image-shaped arrays...")

    # Image-shaped: (N, 32, 32, 3) — for conv autoencoders (PT/TF)
    X_train_img = X_train_norm  # already (N, 32, 32, 3)
    X_test_img = X_test_norm

    # Flattened: (N, 3072) — for dense autoencoders (SK)
    X_train_flat = X_train_norm.reshape(len(X_train_norm), -1)
    X_test_flat = X_test_norm.reshape(len(X_test_norm), -1)

    print(f"    Flattened — Train: {X_train_flat.shape} | Test: {X_test_flat.shape}")
    print(f"    Image     — Train: {X_train_img.shape} | Test: {X_test_img.shape}")

    # Step 5: Save everything
    print("\n[5/5] Saving to disk...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Flattened (default for load_processed_data)
    np.save(OUTPUT_DIR / "X_train.npy", X_train_flat)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_flat)

    # Image-shaped (loaded directly in conv AE pipelines)
    np.save(OUTPUT_DIR / "X_train_img.npy", X_train_img)
    np.save(OUTPUT_DIR / "X_test_img.npy", X_test_img)

    # Labels (for evaluation only)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    # Metadata
    info = {
        "dataset": "CIFAR-10",
        "source": "tensorflow.keras.datasets.cifar10",
        "n_train": int(len(X_train_flat)),
        "n_test": int(len(X_test_flat)),
        "n_features_flat": int(X_train_flat.shape[1]),
        "image_shape": [32, 32, 3],
        "n_classes": 10,
        "class_names": ["airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"],
        "normalization": "[0, 1] divide by 255",
        "scaler": None,
        "label_encoding": "0-indexed (0-9), used for evaluation only",
        "random_state": RANDOM_STATE,
        "notes": "Labels not used in training. Noise added at training time."
    }
    with open(OUTPUT_DIR / "preprocessing_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"    Saved to: {OUTPUT_DIR}")

    # Summary
    print("\nFiles saved:")
    for fpath in sorted(OUTPUT_DIR.iterdir()):
        size_mb = fpath.stat().st_size / (1024 ** 2)
        print(f"    {fpath.name:30s} {size_mb:>8.2f} MB")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()