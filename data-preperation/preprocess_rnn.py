"""
ECG5000 Preprocessing for RNN Pipelines
Combines UCR splits, stratified re-split, StandardScaler, label remapping
Source: UCR Time Series Archive via aeon toolkit
"""

import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from aeon.datasets import load_classification

RANDOM_STATE = 113
OUTPUT_DIR = Path('./data/processed/rnn')

CLASS_NAMES = ['Normal', 'R-on-T PVC', 'PVC', 'SP', 'UB']
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def main():
    print("=" * 60)
    print("ECG5000 — Preprocessing for RNN")
    print("=" * 60)

    # [1/7] Load ECG5000 via aeon (official UCR archive toolkit)
    print("\n[1/7] Loading ECG5000...")
    X_train_raw, y_train_raw = load_classification("ECG5000", split="train") # type: ignore
    X_test_raw, y_test_raw = load_classification("ECG5000", split="test") # type: ignore

    # aeon returns (n_samples, n_channels, n_timesteps) — squeeze channel dim
    X_train_raw = X_train_raw.squeeze(1).astype(np.float32) # type: ignore
    X_test_raw = X_test_raw.squeeze(1).astype(np.float32) # type: ignore
    y_train_raw = y_train_raw.astype(int)
    y_test_raw = y_test_raw.astype(int)

    print(f"  UCR split: {X_train_raw.shape[0]} train / {X_test_raw.shape[0]} test")
    print(f"  Sequence length: {X_train_raw.shape[1]} timesteps")

    # [2/7] Validate
    print("\n[2/7] Validating data quality...")
    X_all = np.concatenate([X_train_raw, X_test_raw])
    y_all = np.concatenate([y_train_raw, y_test_raw])
    assert not np.isnan(X_all).any(), "NaN found"
    assert not np.isinf(X_all).any(), "Inf found"
    assert X_all.shape == (5000, 140), f"Unexpected shape: {X_all.shape}"
    assert set(np.unique(y_all)) == {1, 2, 3, 4, 5}, f"Unexpected labels: {np.unique(y_all)}"
    print(f"  No NaN/Inf, shape verified, all 5 classes present")

    # [3/7] Combine + stratified re-split
    print("\n[3/7] Stratified 80/20 re-split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # [4/7] Remap labels 1-5 -> 0-4
    print("\n[4/7] Remapping labels (1-indexed -> 0-indexed)...")
    y_train = np.array([LABEL_MAP[y] for y in y_train])
    y_test = np.array([LABEL_MAP[y] for y in y_test])
    print(f"  Labels: {np.unique(y_train)} (was 1-5, now 0-4)")

    # [5/7] StandardScaler (per-timestep, fit on train)
    print("\n[5/7] StandardScaler (per-timestep, fit on train)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"  Train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  Test range:  [{X_test.min():.4f}, {X_test.max():.4f}]")

    # [6/7] Reshape to (N, 140, 1) for RNN input
    print("\n[6/7] Reshaping for RNN input...")
    X_train = X_train.reshape(-1, 140, 1).astype(np.float32)
    X_test = X_test.reshape(-1, 140, 1).astype(np.float32)
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # [7/7] Save
    print("\n[7/7] Saving to", OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / 'X_train.npy', X_train)
    np.save(OUTPUT_DIR / 'X_test.npy', X_test)
    np.save(OUTPUT_DIR / 'y_train.npy', y_train)
    np.save(OUTPUT_DIR / 'y_test.npy', y_test)

    with open(OUTPUT_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Compute class weights (inverse frequency) from training set
    class_counts = np.bincount(y_train)
    n_classes = len(class_counts)
    class_weights = len(y_train) / (n_classes * class_counts)
    class_weight_dict = {int(i): float(w) for i, w in enumerate(class_weights)}

    info = {
        'dataset': 'ECG5000',
        'source': 'UCR Time Series Archive via aeon toolkit',
        'original_source': 'PhysioNet MIT-BIH',
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'sequence_length': 140,
        'n_features': 1,
        'n_classes': n_classes,
        'class_names': CLASS_NAMES,
        'label_mapping': {str(k): int(v) for k, v in LABEL_MAP.items()},
        'class_weights': class_weight_dict,
        'normalization': 'StandardScaler (per-timestep, fit on train)',
        'random_state': RANDOM_STATE,
        'split': 'stratified 80/20 (combined UCR train+test)',
        'class_distribution_train': {
            CLASS_NAMES[i]: int(class_counts[i]) for i in range(n_classes)
        }
    }

    with open(OUTPUT_DIR / 'preprocessing_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    # Print file sizes
    for fname in sorted(OUTPUT_DIR.glob('*')):
        size_kb = fname.stat().st_size / 1024
        print(f"  {fname.name}: {size_kb:.1f} KB")

    print(f"\nClass weights: {class_weight_dict}")
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()