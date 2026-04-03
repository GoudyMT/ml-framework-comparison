"""
Preprocessing script for GANs — CIFAR-10 dataset.

Downloads CIFAR-10 via Keras, normalizes to [-1, 1] for tanh generator output.
Same dataset as Autoencoders (#10) but different normalization range.
Labels kept for conditional GAN training.

Usage:
    python preprocess_gans.py
"""

import numpy as np
import json
from pathlib import Path

# Configuration
RANDOM_STATE = 113
OUTPUT_DIR = Path("./data/processed/gans")


def main():
    print("=" * 60)
    print("PREPROCESSING: GANs (CIFAR-10)")
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

    # Step 3: Normalize to [-1, 1] for tanh generator output
    print("\n[3/5] Normalizing to [-1, 1] float32...")
    X_train_norm = X_train_raw.astype(np.float32) / 127.5 - 1.0
    X_test_norm = X_test_raw.astype(np.float32) / 127.5 - 1.0
    print(f"    Train range: [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
    print(f"    Test range:  [{X_test_norm.min():.3f}, {X_test_norm.max():.3f}]")
    print(f"    Train mean:  {X_train_norm.mean():.4f}")
    print(f"    Dtype: {X_train_norm.dtype}")

    # Step 4: Save arrays (image-shaped only — GANs are always convolutional)
    print("\n[4/5] Saving to disk...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Image-shaped: (N, 32, 32, 3) — channel-last format
    # PyTorch pipelines will transpose to (N, 3, 32, 32) at load time
    np.save(OUTPUT_DIR / "X_train.npy", X_train_norm)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_norm)

    # Labels for conditional GAN
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    # Step 5: Save metadata
    print("\n[5/5] Saving metadata...")
    info = {
        "dataset": "CIFAR-10",
        "source": "tensorflow.keras.datasets.cifar10",
        "n_train": int(len(X_train_norm)),
        "n_test": int(len(X_test_norm)),
        "image_shape": [32, 32, 3],
        "n_classes": 10,
        "class_names": ["airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"],
        "normalization": "[-1, 1] pixel / 127.5 - 1.0 (tanh generator output)",
        "pixel_stats": {
            "mean_per_channel": [125.31, 122.95, 113.87],
            "std_per_channel": [62.99, 62.09, 66.70],
            "overall_mean_normalized": -0.0533
        },
        "label_encoding": "0-indexed (0-9), used for conditional GAN",
        "random_state": RANDOM_STATE,
        "notes": "Same dataset as Autoencoders (#10) but [-1,1] normalization for GAN tanh output. Test set reserved for FID computation only."
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