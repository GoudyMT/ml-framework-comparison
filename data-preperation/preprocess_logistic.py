"""
Preprocessing script for Logistic Regression.
Handles: Loading, Splitting, Scaling, SMOTE, and Filtering.

All frameworks will load the same preprocessed .npy files,
ensuring consistency and eliminating redundant code.
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Configuration
RANDOM_STATE = 113      # Seed for reproducibility
TEST_SIZE = 0.2         # 80/20 train/test split
SMOTE_RATIO = 3.5       # Oversample fraud to 3.5x legit count (overshoot, then filter to 50/50)
FILTER_STD = 1.5        # Keep synthetic samples within 1.5 std of real fraud (Lower is closer to real)

RAW_PATH = './data/raw/creditcard.csv'
OUTPUT_DIR = './data/processed/logistic_regression'

# Main Preprocessing Pipeline
def main():
    print("=" * 60)
    print("LOGISTIC REGRESSION PREPROCESSING")
    print("=" * 60)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load raw data
    print("\n[1/7] Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Step 2: Seperate features and target
    print("\n[2/7] Seperating features and target...")
    X = df.drop('Class', axis=1).values     # All columns except 'Class'
    y = df['Class'].values                 # Target: 0 = Legit, 1 = Fraud
    feature_names = [col for col in df.columns if col != 'Class']

    # Show class distribution
    fraud_count = np.sum(y == 1)
    legit_count = np.sum(y == 0)
    print(f"    Legitmate: {legit_count:,} ({legit_count/len(y)*100:.2f}%)")
    print(f"    Fraud: {fraud_count:,} ({fraud_count/len(y)*100:.4f}%)")

    # Step 3: Train/test split BEFORE any resampling
    # We stratify to maintain class proportions in both sets
    print("\n[3/7] Train/test split (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Keeps same fraud % in train and test # type: ignore
    )
    print(f"    Training set: {len(X_train):,} samples")
    print(f"    Test set:   {len(X_test):,} samples")

    # Step 4: Feature scaling (fit on train, transform on both)
    # StandardScaler: z = (x - mean) / std
    print("\n[4/7] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Learn mean/std from train
    X_test_scaled = scaler.transform(X_test)        # Apply same transform to test
    print(f"    Scaler fitted on training data only (no data leakage)")

    # Step 5: SMOTE oversampling on training data only
    # Creates synthetic fraud samples by interpolating between real ones
    print("\n[5/7] Applying SMOTE...")
    train_fraud_before = np.sum(y_train == 1)   # Actual fraud count
    train_legit_count = np.sum(y_train == 0)    # Legit count for ratio
    target_fraud_count = int(train_legit_count * SMOTE_RATIO)

    smote = SMOTE(sampling_strategy={1: target_fraud_count}, random_state=RANDOM_STATE) # type: ignore
    X_train_smote , y_train_smote = smote.fit_resample(X_train_scaled, y_train) # type: ignore

    train_fraud_after = np.sum(y_train_smote == 1)
    synthetic_count = train_fraud_after - train_fraud_before
    print(f"    Original fraud samples: {train_fraud_before:,}")
    print(f"    Synthetic fraud generated: {synthetic_count:,}")
    print(f"    Total fraud after SMOTE: {train_fraud_after:,}")

    # Step 6: Filter unrealistic synthetic samples
    # Keep only synthetic samples within N std of real fraud distribution
    print(f"\n[6/7] Filtering unrealistic synthetic samples...")

    # Calculate statistics from REAL fraud samples only (before SMOTE)
    real_fraud_mask = y_train == 1
    real_fraud = X_train_scaled[real_fraud_mask]
    fraud_mean = real_fraud.mean(axis=0)    # Mean of each feature
    fraud_std = real_fraud.std(axis=0)      # Std of each feature

    # Define acceptable bounds for each feature
    lower_bound = fraud_mean - FILTER_STD * fraud_std
    upper_bound = fraud_mean + FILTER_STD * fraud_std

    # SMOTE appends synthetic samples after original data
    n_original = len(X_train_scaled)

    # Split into original and synthetic portions
    X_original = X_train_smote[:n_original]
    y_original = y_train_smote[:n_original]
    X_synthetic = X_train_smote[n_original:]
    y_synthetic = y_train_smote[n_original:]

    # Check which synthetic samples fall within bounds (all features must pass)
    within_bounds = np.all((X_synthetic >= lower_bound) & (X_synthetic <= upper_bound), axis=1)

    X_synthetic_filtered = X_synthetic[within_bounds]
    y_synthetic_filtered = y_synthetic[within_bounds]

    filtered_out = len(X_synthetic) - len(X_synthetic_filtered)
    print(f"    Synthetic before filter: {len(X_synthetic):,}")
    print(f"    Removed as unrealistic: {filtered_out}")
    print(f"    Synthetic after filter: {len(X_synthetic_filtered):,}")

    # Step 6b. Trim to 50/50 balance
    # If we still have more synthetic fraud than legit, random sample down
    print("\n   Trimming to 50/50 balance...")

    # We want fraud count to equal legit count
    n_legit = np.sum(y_original == 0)
    n_real_fraud = np.sum(y_original == 1)
    n_synthetic_needed = n_legit - n_real_fraud # How many synthetic to keep

    if len(X_synthetic_filtered) > n_synthetic_needed:
        # Randomly select the synthetic samples we need
        np.random.seed(RANDOM_STATE)
        keep_indices = np.random.choice(
            len(X_synthetic_filtered),
            size=n_synthetic_needed,
            replace=False
        )
        X_synthetic_final = X_synthetic_filtered[keep_indices]
        y_synthetic_final = y_synthetic_filtered[keep_indices]
        print(f"    > Trimmed synthetic from {len(X_synthetic_filtered):,} to {n_synthetic_needed:,}")
    else:
        # Keep all filtered synethcif (couldn't reach 50/50)
        X_synthetic_final = X_synthetic_filtered
        y_synthetic_final = y_synthetic_filtered
        print(f"    Keeping all {len(X_synthetic_filtered):,} filtered synthetic samples")

    # Combine original + trimmed synthetic for final training set
    X_train_final = np.vstack([X_original, X_synthetic_final])
    y_train_final = np.concatenate([y_original, y_synthetic_final])

    # Step 7 Save preprocessed arrays
    print("\n[7/7] Saving processed arrays...")
    np.save(f'{OUTPUT_DIR}/X_train.npy', X_train_final)
    np.save(f'{OUTPUT_DIR}/X_test.npy', X_test_scaled)
    np.save(f'{OUTPUT_DIR}/y_train.npy', y_train_final)
    np.save(f'{OUTPUT_DIR}/y_test.npy', y_test)

    # Save metadata for reference and reproducibility
    metadata = {
        'random_state': RANDOM_STATE,
        'test_size': TEST_SIZE,
        'smote_ratio': SMOTE_RATIO,
        'filter_std': FILTER_STD,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'original_total': len(df),
        'original_fraud': int(fraud_count),
        'original_legit': int(legit_count),
        'train_samples_final': len(X_train_final),
        'train_fraud_final': int(np.sum(y_train_final == 1)),
        'train_legit_final': int(np.sum(y_train_final == 0)),
        'test_samples': len(X_test),
        'test_fraud': int(np.sum(y_test == 1)),
        'test_legit': int(np.sum(y_test == 0)),
        'synthetic_generated': int(synthetic_count),
        'synthetic_after_filter': int(len(X_synthetic_filtered)),
        'synthetic_final': int(len(X_synthetic_final)),
        'scaler_mean': scaler.mean_.tolist(), # type: ignore
        'scaler_std': scaler.scale_.tolist() # type: ignore
    }

    with open(f'{OUTPUT_DIR}/preprocessing_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved to {OUTPUT_DIR}/")
    print(f"    - X_train.npy: {X_train_final.shape}")
    print(f"    - X_test.npy: {X_test_scaled.shape}")
    print(f"    - y_train.npy: {y_train_final.shape}")
    print(f"    - y_test.npy: {y_test.shape}")
    print(f"    - preprocessing_info.json")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    final_fraud = np.sum(y_train_final == 1)
    final_legit = np.sum(y_train_final == 0)
    print(f"Training Set: {len(X_train_final):,} samples")
    print(f"  - Legitimate: {final_legit:,} ({final_legit/len(y_train_final)*100:.1f}%)")
    print(f"  - Fraud: {final_fraud:,} ({final_fraud/len(y_train_final)*100:.1f}%)")
    print(f"\nTest Set: {len(X_test):,} samples (original distribution preserved)")
    print(f"  - Legitimate: {np.sum(y_test == 0):,}")
    print(f"  - Fraud: {np.sum(y_test == 1):,}")
    print("\nPreprocessing complete!")

if __name__ == '__main__':
    main()