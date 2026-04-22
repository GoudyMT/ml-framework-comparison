"""
VAE Preprocessing — MNIST + CIFAR-10

Downloads and prepares two datasets for Model #19:
  - MNIST (28x28 grayscale, 10 classes): canonical VAE teaching dataset -
    Kingma & Welling 2013's demo, gives 2D latent-space visualizations
    and disentanglement traversals. Small, clean, educational.
  - CIFAR-10 (32x32 RGB, 10 classes): portfolio hero dataset. Enables
    direct FID comparison against GANs #14 (DCGAN FID 30.57) - the
    honest "VAE is worse on natural images" data point.

Normalization: [0, 1] float32 for both. Diverges from GANs #14 which used
[-1, 1] for tanh generator output; VAE uses BCE-compatible [0, 1] range.
Keeps the normalization choice paired with the model choice.

Usage: python preprocess_vae.py
"""

import json
import numpy as np
from pathlib import Path
from tensorflow.keras.datasets import mnist, cifar10  # type: ignore

RANDOM_STATE = 113
OUTPUT_DIR = Path('./data/processed/vae')
MNIST_DIR = OUTPUT_DIR / 'mnist'
CIFAR10_DIR = OUTPUT_DIR / 'cifar10'

# Class names from dataset documentation
MNIST_CLASS_NAMES = [str(i) for i in range(10)]  # '0' through '9'
CIFAR10_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def preprocess_mnist():
    # Load MNIST, normalize to [0, 1], validate, save
    print("\n[1/3] Loading MNIST (28x28 grayscale)...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Validate raw
    assert X_train.shape == (60000, 28, 28), f"MNIST train shape: {X_train.shape}"
    assert X_test.shape == (10000, 28, 28), f"MNIST test shape: {X_test.shape}"
    assert X_train.dtype == np.uint8, f"MNIST dtype: {X_train.dtype}"
    assert not np.isnan(X_train.astype(float)).any()

    # Normalize to float32 [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Dtype: {X_train.dtype} | Range: [{X_train.min():.1f}, {X_train.max():.1f}]")
    print(f"    Classes: {len(np.unique(y_train))} (balanced: {np.bincount(y_train).min()}-{np.bincount(y_train).max()} per class)")

    MNIST_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MNIST_DIR / 'X_train.npy', X_train)
    np.save(MNIST_DIR / 'X_test.npy', X_test)
    np.save(MNIST_DIR / 'y_train.npy', y_train)
    np.save(MNIST_DIR / 'y_test.npy', y_test)

    info = {
        'dataset': 'MNIST',
        'source': 'tensorflow.keras.datasets.mnist',
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'image_shape': list(X_train.shape[1:]),
        'n_classes': 10,
        'class_names': MNIST_CLASS_NAMES,
        'normalization': '[0, 1] (X / 255.0, float32)',
        'notes_for_vae': 'Grayscale, add channel dim at load time if needed: X[..., None]',
        'random_state': RANDOM_STATE,
    }
    with open(MNIST_DIR / 'preprocessing_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    for fname in sorted(MNIST_DIR.glob('*.npy')):
        arr = np.load(fname)
        size_mb = fname.stat().st_size / (1024 * 1024)
        print(f"    {fname.name}: {arr.shape} {arr.dtype} ({size_mb:.1f} MB)")


def preprocess_cifar10():
    # Load CIFAR-10, normalize to [0, 1], validate, save
    print("\n[2/3] Loading CIFAR-10 (32x32 RGB)...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Validate raw
    assert X_train.shape == (50000, 32, 32, 3), f"CIFAR-10 train shape: {X_train.shape}"
    assert X_test.shape == (10000, 32, 32, 3), f"CIFAR-10 test shape: {X_test.shape}"
    assert X_train.dtype == np.uint8, f"CIFAR-10 dtype: {X_train.dtype}"
    assert not np.isnan(X_train.astype(float)).any()

    # Normalize to float32 [0, 1] (diverges from GAN #14's [-1, 1])
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Dtype: {X_train.dtype} | Range: [{X_train.min():.1f}, {X_train.max():.1f}]")
    print(f"    Classes: {len(np.unique(y_train))} (balanced: {np.bincount(y_train).min()}-{np.bincount(y_train).max()} per class)")

    CIFAR10_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CIFAR10_DIR / 'X_train.npy', X_train)
    np.save(CIFAR10_DIR / 'X_test.npy', X_test)
    np.save(CIFAR10_DIR / 'y_train.npy', y_train)
    np.save(CIFAR10_DIR / 'y_test.npy', y_test)

    info = {
        'dataset': 'CIFAR-10',
        'source': 'tensorflow.keras.datasets.cifar10',
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'image_shape': list(X_train.shape[1:]),
        'n_classes': 10,
        'class_names': CIFAR10_CLASS_NAMES,
        'normalization': '[0, 1] (X / 255.0, float32)',
        'notes_for_vae': 'Channel-last (H, W, C). PT pipelines transpose to (C, H, W) at load.',
        'notes_vs_gans14': 'GANs #14 used [-1, 1] for tanh output; VAE uses [0, 1] for BCE/Gaussian decoder likelihood',
        'random_state': RANDOM_STATE,
    }
    with open(CIFAR10_DIR / 'preprocessing_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    for fname in sorted(CIFAR10_DIR.glob('*.npy')):
        arr = np.load(fname)
        size_mb = fname.stat().st_size / (1024 * 1024)
        print(f"    {fname.name}: {arr.shape} {arr.dtype} ({size_mb:.1f} MB)")


def main():
    print("=" * 60)
    print("VAE - Preprocessing MNIST + CIFAR-10")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    preprocess_mnist()
    preprocess_cifar10()

    print("\n[3/3] Summary")
    print(f"    MNIST root:    {MNIST_DIR}")
    print(f"    CIFAR-10 root: {CIFAR10_DIR}")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()