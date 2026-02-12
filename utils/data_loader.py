"""
Generic data loader for preprocessed datasets.
Reduced redundant loading code across all framework notebooks.
"""

import numpy as np
import json

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

    # Load the four standard numpy arrays
    X_train = np.load(f'{base_path}/X_train.npy')
    X_test = np.load(f'{base_path}/X_test.npy')
    y_train = np.load(f'{base_path}/y_train.npy')
    y_test = np.load(f'{base_path}/y_test.npy')

    # Load preprocessing metadata (feature names, scaler params, etc.)
    with open(f'{base_path}/preprocessing_info.json', 'r') as f:
        metadata = json.load(f)

    return X_train, X_test, y_train, y_test, metadata