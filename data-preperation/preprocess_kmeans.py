"""
Preprocessing script for k-means - Dry Beans Dataset.

Loads the dry beans csv, inspects data quality, encodes starting labels
to integers, applies stratified train/test split, scales features,
and saves to data/processed/kmeans/.

Dataset: 13,611 samples, 16 geometric features, 7 bean types.
Source: UCI ML Repository (via Kaggle download).

Run once before training any K-means models inside this project."
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configuration
RANDOM_STATE = 113
TEST_SIZE = 0.2
RAW_PATH = './data/raw/Dry_Bean.csv'
OUTPUT_DIR = './data/processed/kmeans'

def main():
    print("=" * 60)
    print("K-MEANS PREPROCESSING — DRY BEANS DATASET")
    print("=" * 60)

    # Step 1: Load and clean raw data
    print("\n[1/5] Loading and cleaning raw data...")
    df = pd.read_csv(RAW_PATH)
    print(f"    Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"    Nulls: {df.isnull().sum().sum()}")
    print(f"    Duplicates: {df.duplicated().sum()}")

    # Drop exact duplicate rows (data entry artifacts)
    df = df.drop_duplicates(keep='first')
    print(f"    After removing duplicate rows: {df.shape[0]:,} rows")

    # Step 2: Seperate features and encode labels
    # Dry beans has string class names: ('SEKER', 'BOMBAY', etc.)
    # LabelEncoder maps them to integers (0-6) for model compatability
    print("\n[2/5] Seperating features and encoding labels...")
    X = df.drop('Class', axis=1).values
    feature_names = [col for col in df.columns if col != 'Class']

    le = LabelEncoder()
    y = le.fit_transform(df['Class'])

    CLASS_NAMES = le.classes_.tolist()
    print(f"    Features: {X.shape[1]}")
    print(f"    Classes: {CLASS_NAMES}")
    print(f"    Label mapping: {dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))}")

    # Step 3: Stratified train/test split (same pattern as previous models)
    # K-means is unsupervised - it never sees labels during training.
    # We still split so we can evaluate if clusters generalize to unseen data,
    # not just memorize the training set. Labels are only used for evaluation (ARI).
    print("\n[3/5]")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"    Training:   {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"    Test:       {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Step 4: Feature scaling (fit on train only - prevents data leakage)
    print("\n[4/5] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"    Before scaling — Train mean: {X_train[:, 0].mean():.2f}, std: {X_train[:, 0].std():.2f}")
    print(f"    After scaling  — Train mean: {X_train_scaled[:, 0].mean():.2f}, std: {X_train_scaled[:, 0].std():.2f}")

    # Step 5: Save processed data
    print("\n[5/5] Saving processed data...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f'{OUTPUT_DIR}/X_train.npy', X_train_scaled)
    np.save(f'{OUTPUT_DIR}/X_test.npy', X_test_scaled)
    np.save(f'{OUTPUT_DIR}/y_train.npy', y_train)
    np.save(f'{OUTPUT_DIR}/y_test.npy', y_test)

    metadata = {
        'dataset': 'Dry Beans (UCI ML Repository)',
        'source': 'Kaggle download — data/raw/Dry_Bean.csv',
        'total_samples': int(len(df)),
        'n_features': int(X.shape[1]),
        'n_classes': len(CLASS_NAMES),
        'class_names': CLASS_NAMES,
        'feature_names': feature_names,
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'scaling': 'StandardScaler',
        'duplicates_removed': 68,
        'label_encoding': dict(zip(CLASS_NAMES, range(len(CLASS_NAMES)))),
        'scaler_mean': scaler.mean_.tolist(), # type: ignore
        'scaler_std': scaler.scale_.tolist()  # type: ignore  
    }

    with open(f'{OUTPUT_DIR}/preprocessing_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved to: {OUTPUT_DIR}/")
    print(f"      - X_train.npy: {X_train_scaled.shape}")
    print(f"      - X_test.npy:  {X_test_scaled.shape}")
    print(f"      - y_train.npy: {y_train.shape}")
    print(f"      - y_test.npy:  {y_test.shape}")
    print(f"      - preprocessing_info.json")

    print(f"\n{'=' * 60}")
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()