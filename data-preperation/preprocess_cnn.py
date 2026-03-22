"""
CIFAR-100 Preprocessing for CNN Pipelines
Normalizes to [0,1], saves image-shaped arrays + both label granularities
"""

import numpy as np
import json
from pathlib import Path
from tensorflow.keras.datasets import cifar100 # type: ignore

RANDOM_STATE = 113
OUTPUT_DIR = Path("./data/processed/cnn")

# CIFAR-100 class names from official documentation:
# https://www.cs.toronto.edu/~kriz/cifar.html
# Fine names are alphabetically ordered (index 0-99)
FINE_NAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Coarse (superclass) names are alphabetically ordered (index 0-19)
COARSE_NAMES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers',
    'fruit_and_vegetables', 'household_electrical_devices',
    'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores', 'medium_mammals',
    'non-insect_invertebrates', 'people', 'reptiles',
    'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
]

def main():
    print("=" * 60)
    print("CIFAR-100 — Preprocessing for CNN")
    print("=" * 60)

    # [1/5] Load CIFAR-100 with both fine and coarse labels
    print("\n[1/5] Loading CIFAR-100...")
    (X_train, y_train_fine), (X_test, y_test_fine) = cifar100.load_data(label_mode='fine')
    (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')

    # Flatten labels from (N, 1) to (N,)
    y_train_fine = y_train_fine.ravel()
    y_test_fine = y_test_fine.ravel()
    y_train_coarse = y_train_coarse.ravel()
    y_test_coarse = y_test_coarse.ravel()

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Fine labels: {len(np.unique(y_train_fine))} classes")
    print(f"  Coarse labels: {len(np.unique(y_train_coarse))} superclasses")

    # [2/5] Validate data quality
    print("\n[2/5] Validating data quality...")
    assert not np.isnan(X_train.astype(float)).any(), "NaN found in train"
    assert not np.isnan(X_test.astype(float)).any(), "NaN found in test"
    assert not np.isinf(X_train.astype(float)).any(), "Inf found in train"
    assert not np.isinf(X_test.astype(float)).any(), "Inf found in test"
    assert X_train.shape == (50000, 32, 32, 3), f"Unexpected train shape: {X_train.shape}"
    assert X_test.shape == (10000, 32, 32, 3), f"Unexpected test shape: {X_test.shape}"
    assert X_train.dtype == np.uint8, f"Unexpected dtype: {X_train.dtype}"
    print(f"  No NaN/Inf detected")
    print(f"  Shapes verified: train {X_train.shape}, test {X_test.shape}")
    print(f"  Dtype: {X_train.dtype}, range: [{X_train.min()}, {X_train.max()}]")

    # [3/5] Normalize to [0,1] float32
    print("\n[3/5] Normalizing to [0,1] float32...")
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    print(f"  Dtype: {X_train.dtype}")
    print(f"  Pixel range: [{X_train.min():.1f}, {X_train.max():.1f}]")

    # [4/5] Save arrays
    print("\n[4/5] Saving to", OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / 'X_train.npy', X_train)
    np.save(OUTPUT_DIR / 'X_test.npy', X_test)
    np.save(OUTPUT_DIR / 'y_train_fine.npy', y_train_fine)
    np.save(OUTPUT_DIR / 'y_test_fine.npy', y_test_fine)
    np.save(OUTPUT_DIR / 'y_train_coarse.npy', y_train_coarse)
    np.save(OUTPUT_DIR / 'y_test_coarse.npy', y_test_coarse)

    # Print file sizes
    for fname in sorted(OUTPUT_DIR.glob('*.npy')):
        size_mb = fname.stat().st_size / (1024 * 1024)
        arr = np.load(fname)
        print(f"  {fname.name}: {arr.shape} {arr.dtype} ({size_mb:.1f} MB)")

    # [5/5] Save preprocessing metadata
    print("\n[5/5] Saving preprocessing_info.json...")

    # Build superclass -> fine class mapping
    superclass_map = {}
    for sc_idx in range(20):
        mask = y_train_coarse == sc_idx
        fine_in_sc = np.unique(y_train_fine[mask])
        superclass_map[COARSE_NAMES[sc_idx]] = [FINE_NAMES[i] for i in fine_in_sc]

    info = {
        'dataset': 'CIFAR-100',
        'source': 'tensorflow.keras.datasets.cifar100',
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'image_shape': [32, 32, 3],
        'n_fine_classes': 100,
        'n_coarse_classes': 20,
        'normalization': '[0,1] (X / 255.0)',
        'random_state': RANDOM_STATE,
        'fine_class_names': FINE_NAMES,
        'coarse_class_names': COARSE_NAMES,
        'superclass_mapping': superclass_map
    }

    with open(OUTPUT_DIR / 'preprocessing_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  Saved preprocessing_info.json")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files: 6 .npy + 1 .json")
    print("=" * 60)


if __name__ == '__main__':
    main()