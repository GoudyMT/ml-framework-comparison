"""
Unit tests for app.services.preprocessing.

WHAT THESE TESTS COVER:
    - Output shape and dtype contract (handlers depend on these).
    - Identity scaler case: with mean=0, std=1, only the /255 step
      happens. Lets us verify the normalization step in isolation.
    - Realistic scaler case: non-zero mean and non-unit std,
      verifying the full (X - mean) / std formula end to end.
    - Edge cases: all-zero pixels, all-max pixels.

WHY UNIT-TEST THIS SEPARATELY FROM THE ROUTER:
    apply_preprocessing is a pure function with no I/O. Testing it
    here means we can verify the math precisely without booting
    FastAPI, mocking the loader, or thinking about HTTP. If a
    /predict/pca test ever fails for unclear reasons, this file
    is the first place to look - is the math right? - before
    chasing routing or schema issues.
"""

import numpy as np

from app.schemas.pca import INPUT_DIM
from app.services.preprocessing import ScalerDict, apply_preprocessing
from tests.conftest import make_identity_scaler


def test_output_shape_and_dtype() -> None:
    """The handler depends on (1, 784) float32 output."""
    scaler = make_identity_scaler()
    X = apply_preprocessing([100.0] * INPUT_DIM, scaler)

    assert X.shape == (1, INPUT_DIM)
    assert X.dtype == np.float32


def test_identity_scaler_only_normalizes() -> None:
    """
    With mean=zeros, std=ones, the (X - mean) / std step is a no-op.
    The full preprocessing chain reduces to X / 255.
    """
    scaler = make_identity_scaler()

    # Pixel = 0 -> /255 = 0
    X = apply_preprocessing([0.0] * INPUT_DIM, scaler)
    np.testing.assert_allclose(X, np.zeros((1, INPUT_DIM), dtype=np.float32))

    # Pixel = 255 -> /255 = 1
    X = apply_preprocessing([255.0] * INPUT_DIM, scaler)
    np.testing.assert_allclose(X, np.ones((1, INPUT_DIM), dtype=np.float32))

    # Pixel = 127.5 -> /255 = 0.5
    X = apply_preprocessing([127.5] * INPUT_DIM, scaler)
    np.testing.assert_allclose(
        X, np.full((1, INPUT_DIM), 0.5, dtype=np.float32)
    )


def test_realistic_scaler_full_pipeline() -> None:
    """
    Verify (X - mean) / std with a non-trivial scaler.

    Constructs a scaler equivalent to "subtract 0.5 (post-normalize),
    then divide by 0.25 (multiply by 4)". For each input pixel, the
    expected output is computable by hand:

        raw 127.5 -> /255 = 0.5 -> -0.5 = 0.0   -> /0.25 =  0.0
        raw 255.0 -> /255 = 1.0 -> -0.5 = 0.5   -> /0.25 =  2.0
        raw   0.0 -> /255 = 0.0 -> -0.5 = -0.5  -> /0.25 = -2.0
    """
    scaler = ScalerDict(
        mean=np.full(INPUT_DIM, 0.5, dtype=np.float32),
        std=np.full(INPUT_DIM, 0.25, dtype=np.float32),
    )

    X = apply_preprocessing([127.5] * INPUT_DIM, scaler)
    np.testing.assert_allclose(
        X, np.zeros((1, INPUT_DIM), dtype=np.float32), atol=1e-5
    )

    X = apply_preprocessing([255.0] * INPUT_DIM, scaler)
    np.testing.assert_allclose(
        X, np.full((1, INPUT_DIM), 2.0, dtype=np.float32), atol=1e-5
    )

    X = apply_preprocessing([0.0] * INPUT_DIM, scaler)
    np.testing.assert_allclose(
        X, np.full((1, INPUT_DIM), -2.0, dtype=np.float32), atol=1e-5
    )


def test_per_pixel_scaling_with_varying_mean_std() -> None:
    """
    Different pixels can have different means and stds (real scaler does).
    Verify the broadcasting works correctly: pixel i is standardized using
    mean[i] and std[i].
    """
    # Mean ramps from 0 to ~1 across pixels; std fixed at 0.5.
    mean = np.linspace(0.0, 1.0, INPUT_DIM, dtype=np.float32)
    std = np.full(INPUT_DIM, 0.5, dtype=np.float32)
    scaler = ScalerDict(mean=mean, std=std)

    # Send all-127.5 (normalize -> 0.5 for every pixel).
    # Expected per-pixel: (0.5 - mean[i]) / 0.5
    X = apply_preprocessing([127.5] * INPUT_DIM, scaler)

    expected = ((0.5 - mean) / 0.5).reshape(1, -1)
    np.testing.assert_allclose(X, expected, atol=1e-5)
