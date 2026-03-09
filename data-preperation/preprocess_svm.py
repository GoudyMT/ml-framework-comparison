"""
Preprocessing script for SVM — MAGIC Gamma Telescope dataset.

Dataset: UCI MAGIC Gamma Telescope (Bock et al., 2004)
    - 19,020 samples, 10 continuous features, 1 binary target.
    - Source: https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data
    - Target: 'g' (gamma / signal) → 1, 'h' (hadron / background) → 0.
    - All features continuous — no categorical encoding needed.
    - 115 duplicate rows removed (0.6%) — EDA confirmed.
    - StandardScaler applied — SVM is scale-sensitive (kernel distances
      depend on feature magnitudes, unlike trees which split on thresholds).
    - Stratified 80/20 split, random_state=113.

Run once before training any SVM models in this project.
"""

import numpy as np
import pandas as pd
import json
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
RANDOM_STATE = 113
TEST_SIZE = 0.2
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data'
RAW_DIR = './data/raw'
RAW_PATH = f'{RAW_DIR}/magic04.data'
OUTPUT_DIR = './data/processed/svm'

# MAGIC dataset has no header row — define column names from UCI documentation
FEATURE_NAMES = [
    'fLength',   # Major axis of ellipse [mm]
    'fWidth',    # Minor axis of ellipse [mm]
    'fSize',     # 10-log of sum of content of all pixels [in #phot]
    'fConc',     # Ratio of sum of two highest pixels over fSize [ratio]
    'fConc1',    # Ratio of highest pixel over fSize [ratio]
    'fAsym',     # Distance from highest pixel to center, projected onto major axis [mm]
    'fM3Long',   # 3rd root of third moment along major axis [mm]
    'fM3Trans',  # 3rd root of third moment along minor axis [mm]
    'fAlpha',    # Angle of major axis with vector to origin [deg]
    'fDist'      # Distance from origin to center of ellipse [mm]
]
TARGET_NAME = 'class'


def main():
    print("=" * 60)
    print("SVM PREPROCESSING — MAGIC GAMMA TELESCOPE")
    print("=" * 60)

    # Step 1: Download dataset (if not already present)
    print("\n[1/6] Downloading MAGIC Gamma Telescope dataset...")
    if os.path.exists(RAW_PATH):
        print(f"    Already exists: {RAW_PATH}")
    else:
        os.makedirs(RAW_DIR, exist_ok=True)
        print(f"    Downloading from UCI...")
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
        print(f"    Saved to: {RAW_PATH}")

    # Step 2: Load and inspect
    print("\n[2/6] Loading and inspecting dataset...")
    column_names = FEATURE_NAMES + [TARGET_NAME]
    df = pd.read_csv(RAW_PATH, header=None, names=column_names)
    print(f"    Shape: {df.shape}")
    print(f"    Features: {FEATURE_NAMES}")
    target_counts = df[TARGET_NAME].value_counts()
    print(f"    Class distribution: g(gamma)={target_counts['g']:,} "
          f"({target_counts['g']/len(df)*100:.1f}%), "
          f"h(hadron)={target_counts['h']:,} "
          f"({target_counts['h']/len(df)*100:.1f}%)")

    # Step 3: Data quality — missing values + duplicates
    # EDA confirmed: 0 NaN, 115 duplicates (0.6%)
    print("\n[3/6] Data quality checks...")

    # Missing values
    total_missing = df.isnull().sum().sum()
    print(f"    Missing values: {total_missing}")

    # Duplicates — remove to prevent inflated support vector counts
    n_duplicates = df.duplicated().sum()
    print(f"    Duplicate rows: {n_duplicates}")
    if n_duplicates > 0:
        print(f"    Dropping {n_duplicates} duplicates...")
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"    Shape after dedup: {df.shape}")

    # Step 4: Encode target
    # 'g' (gamma ray / signal) → 1, 'h' (hadron / background noise) → 0
    # Convention: positive class = the signal we're trying to detect
    print("\n[4/6] Encoding target variable...")
    df[TARGET_NAME] = (df[TARGET_NAME] == 'g').astype(int)
    print(f"    Encoded: g (gamma/signal) → 1, h (hadron/background) → 0")
    print(f"    Class 0 (hadron): {(df[TARGET_NAME] == 0).sum():,}")
    print(f"    Class 1 (gamma):  {(df[TARGET_NAME] == 1).sum():,}")

    # Step 5: Stratified split + feature scaling
    # Split FIRST, then scale — fit scaler on train only (no data leakage)
    # SVM is scale-sensitive: kernel distances (especially RBF: exp(-gamma * ||x-z||^2))
    # depend on feature magnitudes. EDA showed 1531x range disparity across features.
    print("\n[5/6] Stratified split + StandardScaler...")

    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = df[TARGET_NAME].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y # type:ignore
    )

    # Fit scaler on training data only — transform both
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"    Training: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"    Test:     {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"    Train class dist: 0={np.sum(y_train==0):,}, 1={np.sum(y_train==1):,}")
    print(f"    Test class dist:  0={np.sum(y_test==0):,}, 1={np.sum(y_test==1):,}")
    print(f"    Scaler means (first 3): {scaler.mean_[:3]}") # type:ignore
    print(f"    Scaler stds  (first 3): {scaler.scale_[:3]}") # type:ignore

    # Step 6: Save processed data
    print("\n[6/6] Saving processed data...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f'{OUTPUT_DIR}/X_train.npy', X_train)
    np.save(f'{OUTPUT_DIR}/X_test.npy', X_test)
    np.save(f'{OUTPUT_DIR}/y_train.npy', y_train)
    np.save(f'{OUTPUT_DIR}/y_test.npy', y_test)

    metadata = {
        'dataset': 'MAGIC Gamma Telescope (UCI)',
        'source': 'https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data',
        'description': 'Major Atmospheric Gamma Imaging Cherenkov Telescope — '
                       'classify gamma ray signals vs hadron background noise',
        'total_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'n_classes': 2,
        'class_names': ['hadron (background)', 'gamma (signal)'],
        'feature_names': FEATURE_NAMES,
        'scaling': 'StandardScaler (fit on train, transform both)',
        'scaler_means': scaler.mean_.tolist(), # type:ignore
        'scaler_stds': scaler.scale_.tolist(), # type:ignore
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'class_distribution': {
            'train': {'0_hadron': int(np.sum(y_train == 0)), '1_gamma': int(np.sum(y_train == 1))},
            'test': {'0_hadron': int(np.sum(y_test == 0)), '1_gamma': int(np.sum(y_test == 1))}
        },
        'duplicates_removed': int(n_duplicates),
        'missing_values': int(total_missing),
        'preprocessing_notes': [
            'All 10 features are continuous — no categorical encoding needed',
            'StandardScaler applied — SVM kernels are distance-based, need normalized features',
            'Class imbalance ~65/35 — use class_weight=balanced in SVM',
            'No feature removal — all 10 physics-based features are meaningful',
            'Duplicates removed (115 rows) — prevents inflated support vector counts',
            'Labels stored as {0, 1} — convert to {-1, +1} at training time via svm_utils'
        ]
    }

    with open(f'{OUTPUT_DIR}/preprocessing_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved to: {OUTPUT_DIR}/")
    print(f"      - X_train.npy: {X_train.shape}")
    print(f"      - X_test.npy:  {X_test.shape}")
    print(f"      - y_train.npy: {y_train.shape}")
    print(f"      - y_test.npy:  {y_test.shape}")
    print(f"      - preprocessing_info.json")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()