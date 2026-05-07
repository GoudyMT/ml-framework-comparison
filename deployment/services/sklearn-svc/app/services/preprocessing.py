"""
Preprocessing for D1 SK PCA inference.

WHAT THIS FILE PROVIDES:
    `apply_preprocessing(features, scaler)` - a pure function that
    transforms a raw 784-pixel Fashion-MNIST sample into the input
    space the PCA was fit on.

WHY A SEPARATE MODULE:
    Routers should be thin (HTTP plumbing). Loaders should load (I/O).
    Preprocessing is BUSINESS LOGIC: "what is the exact transformation
    the model expects?" Putting it here means:
        - Tests can verify the math in isolation, without booting FastAPI
        - The router stays declarative ("validate -> preprocess -> predict
          -> respond") with no inline math
        - When future endpoints (D5 TF Translation has BPE tokenization)
          need their own preprocessing, the pattern is established

WHY THE SERVICE OWNS PREPROCESSING (not the client):
    The deployed service is the only thing that knows EXACTLY how the
    model was trained. Pushing preprocessing to clients means:
        - Every client must re-implement the exact pipeline
        - Drift between training and clients silently corrupts predictions
        - Clients in different languages/frameworks get it slightly different
    Owning preprocessing in the service means clients send the most
    natural representation of their data (raw pixel values 0-255) and
    we guarantee correct inference regardless of who's calling.

THE TRAINING PIPELINE (per data/processed/pca/preprocessing_info.json):
    raw uint8 [0, 255]
        -> divide by 255    -> float32 [0, 1]
        -> StandardScaler   -> standardized (mean=0, std=1 per pixel,
                               using mean+std fit on TRAIN set)
        -> PCA.transform    -> 150 components

    The scaler is stored as a dict {'mean': ndarray(784), 'std':
    ndarray(784)} - manual standardization stored as numpy arrays
    rather than a sklearn StandardScaler instance. We do the math
    inline; no need to import sklearn here.
"""

from typing import TypedDict, cast

import numpy as np


class ScalerDict(TypedDict):
    """
    Type contract for the scaler artifact.

    The training pipeline saved the scaler as a plain dict with two
    numpy arrays. TypedDict gives us the typing surface without
    introducing a class - it's still a dict at runtime, just better
    documented and mypy-checkable.
    """

    mean: np.ndarray
    std: np.ndarray


def apply_preprocessing(
    features: list[float],
    scaler: ScalerDict,
) -> np.ndarray:
    """
    Reproduce the modeling-phase preprocessing on a single sample.

    Args:
        features: Raw pixel values for a flattened 28x28 image.
            Length 784 (validated upstream by the PCARequest schema),
            values in [0, 255].
        scaler: Dict with 'mean' and 'std' numpy arrays of shape (784,)
            from the training pipeline.

    Returns:
        ndarray of shape (1, 784), float32, ready for PCA.transform().
        The leading 1 is the batch dimension - sklearn always wants 2D
        input even for a single sample.

    Why each step:
        1. np.asarray + reshape: convert Python list -> 2D numpy array.
           dtype=float32 matches the PCA's internal precision (it was
           fit on float32 data) and is half the memory of float64.
        2. /255.0: normalize raw pixel range into [0, 1]. Without this,
           the StandardScaler would be applied to values 255x larger
           than what it was fit on, producing nonsense.
        3. (X - mean) / std: equivalent to StandardScaler.transform(X)
           but done with raw arithmetic since we serialized the scaler
           as a dict. Per-pixel standardization: each of the 784 pixels
           gets its own mean and std from the training set.
    """
    # Step 1: list -> (1, 784) float32 ndarray
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)

    # Step 2: normalize raw 0-255 to 0-1
    X = X / 255.0

    # Step 3: per-pixel standardization. Broadcasting handles the
    # batch dim - mean has shape (784,), X has (1, 784); numpy
    # broadcasts mean across the batch dimension automatically.
    # cast() narrows the inferred type back to ndarray - mypy strict
    # otherwise widens to Any when arithmetic mixes TypedDict-typed
    # values with the ndarray X.
    return cast(np.ndarray, (X - scaler["mean"]) / scaler["std"])
