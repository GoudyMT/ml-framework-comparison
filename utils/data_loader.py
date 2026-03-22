"""
Generic data loader for preprocessed datasets.
Reduced redundant loading code across all framework notebooks.
"""

import numpy as np
import json
import os

def load_processed_data(model_name):
    """
    Load preprocessed data for any model type.

    Each model's preprocessing script saves data to data/processed/{model_name}.
    This function loads the standarized .npy files and metadata JSON.

    Args:
        model_name: Folder name in data/processed/ ('knn, 'logistic_regression')

    Returns:
        tuple: (X_train, X_test, y_train, y_test, metadata_dict)
        - X_train: Training features (numpy array)
        - X_test: Test features (numpy array)
        - y_train: Training labels (numpy array)
        - y_test: Test labels (numpy array)
        - metadata: Dict with preprocessing info (feature names, scaler params, etc.)

    Example:
        from utils.data_loader import load_processed_data
        X_train, X_test, y_train, y_test, meta = load_processed_data('knn')
    """
    # Build path relative to framework notebooks (which are 2 levels deep)
    base_path = f'../../data/processed/{model_name}'

    # Load feature arrays (always present)
    X_train = np.load(f'{base_path}/X_train.npy')
    X_test = np.load(f'{base_path}/X_test.npy')

    # Load labels — supports both standard (y_train.npy) and
    # hierarchical datasets (y_train_fine.npy + y_train_coarse.npy)
    y_train_path = f'{base_path}/y_train.npy'
    y_fine_path = f'{base_path}/y_train_fine.npy'

    if os.path.exists(y_train_path):
        # Standard: single label set
        y_train = np.load(y_train_path)
        y_test = np.load(f'{base_path}/y_test.npy')
    elif os.path.exists(y_fine_path):
        # Hierarchical: fine labels as primary, coarse in metadata
        y_train = np.load(y_fine_path)
        y_test = np.load(f'{base_path}/y_test_fine.npy')
    else:
        raise FileNotFoundError(f"No label files found in {base_path}")

    # Load preprocessing metadata (feature names, scaler params, etc.)
    with open(f'{base_path}/preprocessing_info.json', 'r') as f:
        metadata = json.load(f)

    # Load coarse labels into metadata if they exist
    y_coarse_path = f'{base_path}/y_train_coarse.npy'
    if os.path.exists(y_coarse_path):
        metadata['y_train_coarse'] = np.load(y_coarse_path)
        metadata['y_test_coarse'] = np.load(f'{base_path}/y_test_coarse.npy')

    return X_train, X_test, y_train, y_test, metadata