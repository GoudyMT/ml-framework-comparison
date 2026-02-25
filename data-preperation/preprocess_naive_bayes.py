"""
Preprocessing script for Naive Bayes - two datasets.

Part 1: Breast Cancer Wisconsin (GaussianNB baseline)
    - 569 samples, 30 continuous features, 2 classes (malignant/benign).
    - Source: sklearn.datasets.load_breast_cancer()
    - Stratified 80/20 split, StandardScaler.

Part 2: 20 Newsgroups (MultinomialNB main event)
    - 18,846 documents, 20 newgroup categories.
    - Source: sklearn.dataset.fetch_20newsgroups()
    - Built-in train/test split (~60/40), TF-IDF vectorization (10k features).
    - Headers/footers/quotes removed to prevent data leakage.

Run once before training any Naive Bayes models in this project.
"""

import numpy as np
import json
import os
from sklearn.datasets import load_breast_cancer, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration
RANDOM_STATE = 113
TEST_SIZE = 0.2
MAX_FEATURES = 10000
GAUSSIAN_OUTPUT_DIR = './data/processed/naive_bayes_gaussian'
TEXT_OUTPUT_DIR = './data/processed/naive_bayes_text'

def preprocess_gaussian():
    """
    Part 1: Breast Cancer for GaussianNB baseline.
    """
    print("\n" + "=" * 60)
    print("PART 1: BREAST CANCER (GaussianNB Baseline)")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[1/4] Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target # type: ignore
    feature_names = data.feature_names.tolist() # type: ignore
    class_names = data.target_names.tolist() # type: ignore
    print(f"    Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"    Classes: {class_names}")
    print(f"    Class distribution: {dict(zip(class_names, np.bincount(y)))}")

    # Step 2: Stratified train/test split
    print("\n[2/4] Stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"    Training: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"    Test:     {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Step 3: Feature scaling (fit on train only)
    print("\n[3/4] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"    Before — Train mean: {X_train[:, 0].mean():.2f}, std: {X_train[:, 0].std():.2f}")
    print(f"    After  — Train mean: {X_train_scaled[:, 0].mean():.2f}, std: {X_train_scaled[:, 0].std():.2f}")

    # Step 4: Save processed data
    print("\n[4/4] Saving processed data...")
    os.makedirs(GAUSSIAN_OUTPUT_DIR, exist_ok=True)

    np.save(f'{GAUSSIAN_OUTPUT_DIR}/X_train.npy', X_train_scaled)
    np.save(f'{GAUSSIAN_OUTPUT_DIR}/X_test.npy', X_test_scaled)
    np.save(f'{GAUSSIAN_OUTPUT_DIR}/y_train.npy', y_train)
    np.save(f'{GAUSSIAN_OUTPUT_DIR}/y_test.npy', y_test)

    metadata = {
        'dataset': 'Breast Cancer Wisconsin (sklearn built-in)',
        'total_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'n_classes': len(class_names),
        'class_names': class_names,
        'feature_names': feature_names,
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'scaling': 'StandardScaler',
        'scaler_mean': scaler.mean_.tolist(), # type: ignore
        'scaler_std': scaler.scale_.tolist() # type: ignore
    }

    with open(f'{GAUSSIAN_OUTPUT_DIR}/preprocessing_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved to: {GAUSSIAN_OUTPUT_DIR}/")
    print(f"      - X_train.npy: {X_train_scaled.shape}")
    print(f"      - X_test.npy:  {X_test_scaled.shape}")
    print(f"      - y_train.npy: {y_train.shape}")
    print(f"      - y_test.npy:  {y_test.shape}")
    print(f"      - preprocessing_info.json")

def preprocess_text():
    """
    Part 2: 20 Newsgroups for MultinomialNB main event.
    """
    print("\n" + "=" * 60)
    print("PART 2: 20 NEWSGROUPS (MultinomialNB Main Event)")
    print("=" * 60)

    # Step 1: Load dataset (built-in train/test split)
    # Remove headers, footers, quotes to prevent data leakage —
    # forces model to learn from content, not email formatting
    print("\n[1/5] Loading 20 Newsgroups dataset...")
    train_data = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        random_state=RANDOM_STATE
    )
    test_data = fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        random_state=RANDOM_STATE
    )

    class_names = train_data.target_names # type: ignore
    print(f"    Training documents: {len(train_data.data):,}") # type: ignore
    print(f"    Test documents:     {len(test_data.data):,}") # type: ignore
    print(f"    Categories: {len(class_names)}")
    print(f"    Split ratio: {len(train_data.data)/(len(train_data.data)+len(test_data.data))*100:.1f}% / " # type: ignore
          f"{len(test_data.data)/(len(train_data.data)+len(test_data.data))*100:.1f}% (built-in benchmark split)") # type: ignore

    # Step 2: Inspect class distribution
    print("\n[2/5] Class distribution...")
    train_counts = np.bincount(train_data.target) # type: ignore
    for i, name in enumerate(class_names):
        print(f"    {name}: {train_counts[i]} train docs")

    # Step 3: TF-IDF vectorization (fit on train only)
    # sublinear_tf: use 1 + log(tf) instead of raw term frequency —
    # prevents very long documents from dominating
    print(f"\n[3/5] TF-IDF vectorization (max_features={MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words='english',
        sublinear_tf=True
    )

    X_train_sparse = vectorizer.fit_transform(train_data.data) # type: ignore
    X_test_sparse = vectorizer.transform(test_data.data) # type: ignore
    print(f"    Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"    Sparse shape — Train: {X_train_sparse.shape}, Test: {X_test_sparse.shape}")
    print(f"    Sparsity: {(1 - X_train_sparse.nnz / (X_train_sparse.shape[0] * X_train_sparse.shape[1]))*100:.1f}%") # type: ignore

    # Step 4: Convert to dense float32 arrays
    # Dense saves as .npy (fits existing data_loader pattern).
    # float32 halves storage vs float64 — TF-IDF values don't need double precision.
    print("\n[4/5] Converting to dense float32 arrays...")
    X_train = X_train_sparse.toarray().astype(np.float32) # type: ignore
    X_test = X_test_sparse.toarray().astype(np.float32) # type: ignore
    y_train = train_data.target # type: ignore
    y_test = test_data.target # type: ignore
    print(f"    X_train: {X_train.shape} ({X_train.nbytes / 1e6:.1f} MB)")
    print(f"    X_test:  {X_test.shape} ({X_test.nbytes / 1e6:.1f} MB)")

    # Step 5: Save processed data
    print("\n[5/5] Saving processed data...")
    os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

    np.save(f'{TEXT_OUTPUT_DIR}/X_train.npy', X_train)
    np.save(f'{TEXT_OUTPUT_DIR}/X_test.npy', X_test)
    np.save(f'{TEXT_OUTPUT_DIR}/y_train.npy', y_train)
    np.save(f'{TEXT_OUTPUT_DIR}/y_test.npy', y_test)

    # Save vocabulary separately (10K terms)
    vocabulary = vectorizer.vocabulary_
    # Convert numpy int keys to regular ints for JSON serialization
    vocab_serializable = {term: int(idx) for term, idx in vocabulary.items()}
    with open(f'{TEXT_OUTPUT_DIR}/vocabulary.json', 'w') as f:
        json.dump(vocab_serializable, f, indent=2)

    # Save IDF weights (needed for analysis, not for training)
    np.save(f'{TEXT_OUTPUT_DIR}/idf_weights.npy', vectorizer.idf_)

    metadata = {
        'dataset': '20 Newsgroups (sklearn built-in)',
        'total_samples': int(len(train_data.data) + len(test_data.data)), # type: ignore
        'n_features': MAX_FEATURES,
        'n_classes': len(class_names),
        'class_names': list(class_names),
        'train_samples': int(len(train_data.data)), # type: ignore
        'test_samples': int(len(test_data.data)), # type: ignore
        'split': 'Built-in benchmark split (~60/40)',
        'random_state': RANDOM_STATE,
        'vectorizer': 'TfidfVectorizer',
        'max_features': MAX_FEATURES,
        'stop_words': 'english',
        'sublinear_tf': True,
        'removed': ['headers', 'footers', 'quotes'],
        'dtype': 'float32'
    }

    with open(f'{TEXT_OUTPUT_DIR}/preprocessing_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved to: {TEXT_OUTPUT_DIR}/")
    print(f"      - X_train.npy: {X_train.shape}")
    print(f"      - X_test.npy:  {X_test.shape}")
    print(f"      - y_train.npy: {y_train.shape}")
    print(f"      - y_test.npy:  {y_test.shape}")
    print(f"      - vocabulary.json ({len(vocab_serializable):,} terms)")
    print(f"      - idf_weights.npy")
    print(f"      - preprocessing_info.json")

def main():
    print("=" * 60)
    print("NAIVE BAYES PREPROCESSING")
    print("=" * 60)

    preprocess_gaussian()
    preprocess_text()

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()