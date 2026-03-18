"""
Preprocessing script for DNN — UCI HAR dataset.

Loads pre-split UCI HAR data (subject-wise train/test), applies
StandardScaler (fit on train only), converts 1-indexed labels to
0-indexed for softmax, and saves processed .npy files.

Key decisions from EDA:
    - StandardScaler needed: means skewed (-0.66 median), stds vary 18x
    - Keep all 561 features: DNN handles redundancy via learned weights
    - Labels 1-6 → 0-5 for softmax output layer
    - Preserve subject-wise split (21 train / 9 test subjects)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

# Configuration
RANDOM_STATE = 113
RAW_DIR = Path("./data/raw/UCI HAR Dataset")
OUTPUT_DIR = Path("./data/processed/dnn")

ACTIVITY_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
]

def main():
    """Load, preprocess, and save UCI HAR data for DNN pipelines."""
    print("=" * 60)
    print("DNN Preprocessing — UCI HAR Dataset")
    print("=" * 60)

    # Step 1: Load raw data (pre-split by subject)
    print("\n[1/6] Loading UCI HAR raw data...")

    # Deduplicate feature names (UCI HAR has repeated bandsEnergy names)
    feature_names_raw = pd.read_csv(
        RAW_DIR / 'features.txt', sep=r'\s+', header=None, names=['idx', 'name']
    )['name'].tolist()

    counts = {}
    feature_names = []
    for name in feature_names_raw:
        if name in counts:
            counts[name] += 1
            feature_names.append(f"{name}_{counts[name]}")
        else:
            counts[name] = 0
            feature_names.append(name)

    X_train = pd.read_csv(
        RAW_DIR / 'train' / 'X_train.txt', sep=r'\s+', header=None, names=feature_names
    ).values.astype(np.float32)
    X_test = pd.read_csv(
        RAW_DIR / 'test' / 'X_test.txt', sep=r'\s+', header=None, names=feature_names
    ).values.astype(np.float32)

    y_train = pd.read_csv(
        RAW_DIR / 'train' / 'y_train.txt', header=None
    ).values.ravel()
    y_test = pd.read_csv(
        RAW_DIR / 'test' / 'y_test.txt', header=None
    ).values.ravel()

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Labels — train: {y_train.shape}, test: {y_test.shape}")
    print(f"  Label range: [{y_train.min()}, {y_train.max()}] (1-indexed)")

    # Step 2: Validate data quality (confirmed clean in EDA)
    print(f"\n[2/6] Data quality check...")
    assert np.isnan(X_train).sum() == 0, "NaN in train!"
    assert np.isnan(X_test).sum() == 0, "NaN in test!"
    assert np.isinf(X_train).sum() == 0, "Inf in train!"
    assert np.isinf(X_test).sum() == 0, "Inf in test!"
    print(f"  No NaN, no Inf — confirmed clean")
    print(f"  Feature range: [{X_train.min():.4f}, {X_train.max():.4f}]")

    # Step 3: Convert labels from 1-indexed to 0-indexed
    # UCI HAR labels are 1-6, softmax needs 0-5
    print(f"\n[3/6] Converting labels: 1-indexed → 0-indexed...")
    y_train = y_train - 1
    y_test = y_test - 1
    print(f"  New label range: [{y_train.min()}, {y_train.max()}]")

    for i, name in enumerate(ACTIVITY_NAMES):
        n_train = (y_train == i).sum()
        n_test = (y_test == i).sum()
        print(f"  Class {i} ({name:25s}): train={n_train:,}, test={n_test:,}")

    # Step 4: StandardScaler — fit on train only, transform both
    # EDA showed: means skewed (median -0.66), stds vary 18x (0.04 to 0.75)
    # Without scaling, features with high variance dominate gradient updates
    print(f"\n[4/6] StandardScaler (fit on train only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to float32 (StandardScaler outputs float64)
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)

    print(f"  Train — mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.4f}")
    print(f"  Test  — mean: {X_test_scaled.mean():.6f}, std: {X_test_scaled.std():.4f}")

    # Step 5: Save processed data
    print(f"\n[5/6] Saving to {OUTPUT_DIR}/...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X_train.npy", X_train_scaled)
    np.save(OUTPUT_DIR / "X_test.npy", X_test_scaled)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    # Save scaler for potential inference pipeline use
    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"  X_train: {X_train_scaled.shape} ({X_train_scaled.nbytes / 1e6:.1f} MB)")
    print(f"  X_test:  {X_test_scaled.shape} ({X_test_scaled.nbytes / 1e6:.1f} MB)")
    print(f"  y_train: {y_train.shape}, y_test: {y_test.shape}")
    print(f"  scaler.pkl: fitted StandardScaler")

    # Step 6: Save preprocessing metadata
    print(f"\n[6/6] Saving preprocessing_info.json...")
    metadata = {
        "dataset": "UCI HAR (Human Activity Recognition)",
        "source": "https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones",
        "n_train": int(X_train_scaled.shape[0]),
        "n_test": int(X_test_scaled.shape[0]),
        "n_features": int(X_train_scaled.shape[1]),
        "n_classes": 6,
        "class_names": ACTIVITY_NAMES,
        "scaling": "StandardScaler (sklearn, fit on train)",
        "label_encoding": "0-indexed (original 1-6 shifted to 0-5)",
        "split_method": "Subject-wise (21 train / 9 test, pre-split by UCI)",
        "feature_range_raw": [-1.0, 1.0],
        "scaler_file": "scaler.pkl"
    }
    with open(OUTPUT_DIR / "preprocessing_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  preprocessing_info.json: dataset metadata")
    print("\n" + "=" * 60)
    print("DNN PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()