"""
Preprocessing script for PCA (Model #08) — Fashion-MNIST dataset.

Loads raw Fashion-MNIST via Keras, flattens 28x28 images to 784 features,
applies StandardScaler (fit on train only), and saves processed .npy files
plus the scaler object for reconstruction in pipeline notebooks.

Google-style docstring conventions throughout.
"""

import numpy as np
import pickle
import json
from pathlib import Path
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warning messages

from tensorflow.keras.datasets import fashion_mnist #type: ignore

# Configuration
RANDOM_STATE = 113
OUTPUT_DIR = Path("./data/processed/pca")


def main():
    # Load, preprocess, and save Fashion-MNIST data for PCA pipelines.
    print("=" * 60)
    print("PCA Preprocessing — Fashion-MNIST")
    print("=" * 60)

    # Step 1: Load raw Fashion-MNIST
    print("\n[1/4] Loading Fashion-MNIST...")
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    print(f"  Raw shapes: train={X_train.shape}, test={X_test.shape}")

    # Step 2: Flatten 28x28 images to 784-dim vectors
    # Convert to float32 for StandardScaler math
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
    print(f"\n[2/4] Flattened: train={X_train.shape}, test={X_test.shape}")
    print(f"  Dtype: {X_train.dtype}, Range: [{X_train.min():.0f}, {X_train.max():.0f}]")

    # Step 3: StandardScaler — fit on train only, transform both
    # Critical for PCA: pixel variances range 0–10,745 (from EDA)
    # Without scaling, PCA would just find "brightest pixels" not structure
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    # Count zero-std pixels before replacing (constant corners)
    zero_std_count = (train_std == 0).sum()

    # Replace zero-std with 1.0 to avoid division by zero
    # These constant pixels will scale to 0 regardless
    train_std[train_std == 0] = 1.0

    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std

    print(f"\n[3/4] StandardScaler applied (fit on train only)")
    print(f"  Zero-std pixels (constant): {zero_std_count}")
    print(f"  Train — mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.4f}")
    print(f"  Test  — mean: {X_test_scaled.mean():.6f}, std: {X_test_scaled.std():.4f}")

    # Step 4: Save processed data + scaler
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X_train.npy", X_train_scaled)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_scaled)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    # Save scaler params for reconstruction in pipeline notebooks
    scaler = {"mean": train_mean, "std": train_std}
    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save preprocessing metadata (consistent with other models)
    metadata = {
        "dataset": "Fashion-MNIST",
        "source": "tensorflow.keras.datasets.fashion_mnist",
        "n_train": int(X_train_scaled.shape[0]),
        "n_test": int(X_test_scaled.shape[0]),
        "n_features": int(X_train_scaled.shape[1]),
        "n_classes": 10,
        "class_names": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Bag", "Sneaker", "Ankle boot"
        ],
        "scaling": "StandardScaler (manual, fit on train)",
        "original_shape": [28, 28],
        "pixel_range_raw": [0, 255],
        "scaler_file": "scaler.pkl"
    }
    with open(OUTPUT_DIR / "preprocessing_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[4/4] Saved to {OUTPUT_DIR}/")
    print(f"  X_train: {X_train_scaled.shape} ({X_train_scaled.nbytes / 1e6:.1f} MB)")
    print(f"  X_test:  {X_test_scaled.shape} ({X_test_scaled.nbytes / 1e6:.1f} MB)")
    print(f"  y_train: {y_train.shape}, y_test: {y_test.shape}")
    print(f"  scaler.pkl: mean + std vectors (for reconstruction)")
    print(f"  preprocessing_info.json: dataset metadata")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()