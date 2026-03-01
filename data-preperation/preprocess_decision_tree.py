"""
Preprocessing script for Decision Tree / Random Forest — Bank Marketing dataset.

Dataset: UCI Bank Marketing (Moro et al., 2014)
    - 41,188 samples, 20 input features, 1 binary target (term deposit: yes/no).
    - Source: bank-additional-full.csv (downloaded from UCI in EDA notebook).
    - Drop `duration` (data leakage — call duration only known after outcome).
    - Ordinal encoding for categoricals (trees split on thresholds, not distances).
    - No feature scaling (trees are scale-invariant).
    - Stratified 80/20 split, random_state=113.

Run once before training any Decision Tree / Random Forest models in this project.
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Configuration
RANDOM_STATE = 113
TEST_SIZE = 0.2
RAW_PATH = './data/raw/bank-additional-full.csv'
OUTPUT_DIR = './data/processed/decision_tree'


def main():
    print("=" * 60)
    print("DECISION TREE / RANDOM FOREST PREPROCESSING")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[1/6] Loading Bank Marketing dataset...")
    df = pd.read_csv(RAW_PATH, sep=';')
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    target_counts = df['y'].value_counts()
    print(f"    Class distribution: no={target_counts['no']:,} ({target_counts['no']/len(df)*100:.1f}%), "
          f"yes={target_counts['yes']:,} ({target_counts['yes']/len(df)*100:.1f}%)")

    # Step 2: Drop duration (data leakage)
    # EDA confirmed: +150.5% mean difference between classes,
    # but duration is only known AFTER the call — not available at prediction time
    print("\n[2/6] Dropping 'duration' column (data leakage)...")
    print(f"    Before: {df.shape[1]} columns")
    df = df.drop(columns=['duration'])
    print(f"    After:  {df.shape[1]} columns")

    # Step 3: Audit "unknown" values
    # EDA finding: 6 columns have "unknown" — keep as its own category
    # Trees route unknowns naturally via learned splits
    print("\n[3/6] Auditing 'unknown' values (keeping as category)...")
    object_cols = df.select_dtypes(include='object').columns.drop('y')
    for col in object_cols:
        n_unknown = (df[col] == 'unknown').sum()
        if n_unknown > 0:
            pct = n_unknown / len(df) * 100
            print(f"    {col:15s}: {n_unknown:6,} unknown ({pct:.1f}%)")

    # Step 4: Encode categorical features
    # Ordinal encoding — NOT one-hot. Trees split on thresholds,
    # so ordinal integers work naturally. One-hot would dilute
    # high-cardinality features (job has 12 categories) into sparse columns.
    print("\n[4/6] Ordinal encoding categorical features...")
    categorical_cols = list(object_cols)
    print(f"    Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    # Build encoding maps for interpretability
    encoding_maps = {}
    for i, col in enumerate(categorical_cols):
        categories = encoder.categories_[i]
        encoding_maps[col] = {cat: int(idx) for idx, cat in enumerate(categories)} # type: ignore
        print(f"    {col:15s}: {len(categories)} categories → {list(encoding_maps[col].items())[:3]}...") # type: ignore

    # Encode target: yes → 1, no → 0
    df['y'] = (df['y'] == 'yes').astype(int)
    print(f"    Target encoded: no → 0, yes → 1")

    # Step 5: Stratified train/test split
    print("\n[5/6] Stratified train/test split...")
    feature_names = [c for c in df.columns if c != 'y']
    X = df[feature_names].values.astype(np.float64)
    y = df['y'].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y # type: ignore
    )

    print(f"    Training: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"    Test:     {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"    Train class dist: 0={np.sum(y_train==0):,}, 1={np.sum(y_train==1):,}")
    print(f"    Test class dist:  0={np.sum(y_test==0):,}, 1={np.sum(y_test==1):,}")

    # Identify which columns are categorical vs continuous (by index)
    categorical_indices = [feature_names.index(c) for c in categorical_cols]
    continuous_cols = [c for c in feature_names if c not in categorical_cols]
    continuous_indices = [feature_names.index(c) for c in continuous_cols]

    # Step 6: Save processed data
    print("\n[6/6] Saving processed data...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f'{OUTPUT_DIR}/X_train.npy', X_train)
    np.save(f'{OUTPUT_DIR}/X_test.npy', X_test)
    np.save(f'{OUTPUT_DIR}/y_train.npy', y_train)
    np.save(f'{OUTPUT_DIR}/y_test.npy', y_test)

    metadata = {
        'dataset': 'Bank Marketing (UCI)',
        'source': 'https://archive.ics.uci.edu/dataset/222/bank+marketing',
        'total_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'n_classes': 2,
        'class_names': ['no', 'yes'],
        'feature_names': feature_names,
        'categorical_features': categorical_cols,
        'categorical_indices': categorical_indices,
        'continuous_features': continuous_cols,
        'continuous_indices': continuous_indices,
        'encoding_maps': encoding_maps,
        'dropped_features': {'duration': 'data leakage — only known after call ends'},
        'unknown_handling': 'kept as own category (ordinal encoded alongside other values)',
        'scaling': 'None (trees are scale-invariant)',
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'class_distribution': {
            'train': {'0_no': int(np.sum(y_train == 0)), '1_yes': int(np.sum(y_train == 1))},
            'test': {'0_no': int(np.sum(y_test == 0)), '1_yes': int(np.sum(y_test == 1))}
        },
        'eda_notes': [
            'Class imbalance 88.7%/11.3% — use class_weight=balanced',
            'duration dropped — +150.5% mean diff, data leakage confirmed',
            'pdays=999 sentinel kept as-is — trees split naturally at ~998',
            'Outliers kept — trees are robust (split-based, not distance-based)',
            'Economic indicators highly correlated (VIF 31-64) — no action for trees',
            'poutcome=success is strongest categorical predictor (65.1% subscription rate)'
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