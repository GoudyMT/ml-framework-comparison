"""
Preprocessing script for KNN - Covertype Dataset.

Downloads the Forest Cover Type dataset from sklearn, applies stratified
train/test split, scales features, and saves to data/processed/knn/.

Dataset: 581,012 samples, 54 features, 7 forest cover types.
Source: UCI ML Repository via sklearn.datasets.fetch_covtype

Run once before training any KNN models.
"""

import numpy as np
import json
import os
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Project-wide random seed for reproducibility
RANDOM_SEED = 113

# Step 1: Load Covertype Dataset

print("\nLoading Covertype dataset... Takes about 90 seconds on first load...")
covtype = fetch_covtype()
X, y = covtype.data, covtype.target # type: ignore

print(f"Full dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"Target classes: {np.unique(y)}")

# Forest cover type names (1-7 in original, we'll keep as-is)
# These are actual forest types from Roosevelt National Forest, Colorado
CLASS_NAMES = [
    'Spruce/Fir',           # Class 1
    'Lodgepole Pine',       # Class 2
    'Ponderosa Pine',       # Class 3
    'Cottonwood/Willow',    # Class 4
    'Aspen',                # Class 5
    'Douglas-fir',          # Class 6
    'Krummholz'             # Class 7
]

# Feature names for the 54 features
# 10 continuous + 4 binary wilderness areas + 40 binary soil types
FEATURE_NAMES = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
    'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
] + [f'Wilderness_Area_{i}' for i in range(1, 5)] + [f'Soil_Type_{i}' for i in range(1, 41)]

# Step 2: Train/Test Split (Stratified)

print(f"\n{"=" * 50}")
print("TRAIN/TEST SPLIT")
print(f"{"=" * 50}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,              # 80/20 Split
    random_state=RANDOM_SEED,   # Reproducibility
    stratify=y                  # Maintain class proportions in both sets    
)

print(f"Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Step 3: Feature Scaling (StandardScaler)

"""
KNN uses distance calculations, so features must be on the same scale.
Without scaling, high-magnitude features (like elevation 3000)
would dominate over low-magnitude features (like Slope 14).
"""

print(f"\n{"=" * 50}")
print("FEATURE SCALING")
print("=" * 50)

scaler = StandardScaler()

# Fit on training data only (prevent data leakage)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Scaler fit on training data: {X_train.shape[0]:,} samples")
print(f"Before scaling - Train mean: {X_train[:, 0].mean():.2f}, std: {X_train[:, 0].std():.2f}")
print(f"After scaling  - Train mean: {X_train_scaled[:, 0].mean():.2f}, std: {X_train_scaled[:, 0].std():.2f}")

# Step 4: Save Processed Data

print(f"\n{"=" * 50}")
print("SAVING PROCESSED DATA")
print("=" * 50)

# Create output directory
output_dir = './data/processed/knn'
os.makedirs(output_dir, exist_ok=True)

# Save numpy arrays
np.save(f'{output_dir}/X_train.npy', X_train_scaled)
np.save(f'{output_dir}/X_test.npy', X_test_scaled)
np.save(f'{output_dir}/y_train.npy', y_train)
np.save(f'{output_dir}/y_test.npy', y_test)

# Save Metadata for reference
metadata = {
    'dataset': 'Covertype (Forest Cover Type)',
    'source': 'sklearn.datasets.fetch_covtype / UCI ML Repository',
    'total_samples': int(X.shape[0]),
    'n_features': int(X.shape[1]),
    'n_classes': len(CLASS_NAMES),
    'class_names': CLASS_NAMES,
    'feature_names': FEATURE_NAMES,
    'train_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'test_size': 0.2,
    'random_seed': RANDOM_SEED,
    'scaling': 'StandardScaler',
    'scaler_mean': scaler.mean_.tolist(),   # type: ignore
    'scaler_std': scaler.scale_.tolist()    # type: ignore
}

with open(f'{output_dir}/preprocessing_info.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved to: {output_dir}/")
print(f"  - X_train.npy: {X_train_scaled.shape}")
print(f"  - X_test.npy: {X_test_scaled.shape}")
print(f"  - y_train.npy: {y_train.shape}")
print(f"  - y_test.npy: {y_test.shape}")
print(f"  - preprocessing_info.json")

print(f"\n{'='*50}")
print("PREPROCESSING COMPLETE!")
print(f"{'='*50}")