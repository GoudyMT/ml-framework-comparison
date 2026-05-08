"""
Shared pytest fixtures for pt-svc tests.

WHAT THIS FILE PROVIDES:
    - `FakeDNN` - a real nn.Module subclass with the same I/O contract
      as the deployed DNN (input (batch, 561) -> output (batch, 6)
      logits). Real Module so the router's `with torch.no_grad():`
      and `model.eval()` checks all behave correctly.
    - `FakeStandardScaler` - duck-typed stand-in for sklearn's
      StandardScaler. Exposes `.transform(X)` and `.mean_` /
      `.scale_` attributes so the loader's logging path doesn't break.
    - `client` - TestClient with the loader cache populated by the
      fakes. Use for tests expecting "loaded model" state.
    - `client_unloaded` - TestClient with the cache cleared. Use for
      503 / unloaded behavior tests.

WHY MOCK THE MODEL:
    The real DNN + scaler load takes ~2 seconds and requires the
    MLflow registry + .pth artifact + scaler.pkl on disk - none of
    which are available in CI environments that pull only source
    code. Mocking the cache slots directly bypasses MLflow entirely
    while keeping every other code path (routes, middleware,
    validation, schema serialization) running for real.

LIFESPAN BEHAVIOR:
    `TestClient(app)` (without `with`) does NOT trigger the lifespan
    event. So our dnn_loader.load_dnn_model() does NOT run during
    tests - we manually populate _MODEL / _SCALER / _MODEL_VERSION
    in the fixtures instead.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from app.main import app
from app.services import dnn_loader


class FakeDNN(nn.Module):
    """
    Minimal nn.Module stand-in for the real DNN.

    Returns deterministic logits favoring class index 3 (SITTING) so
    the predicted_class assertion in the happy-path test is stable
    regardless of input. The router's softmax + argmax will produce
    predicted_class=3, predicted_label='SITTING'.

    Inheriting from nn.Module (not just defining a forward()) means:
        - model.eval() and model.training work as expected
        - `with torch.no_grad():` correctly disables grad on this fake
        - tests exercise the same code path as the real model
    """

    def __init__(self, n_classes: int = 6) -> None:
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return shape (batch, n_classes) with class 3 favored."""
        batch = x.shape[0]
        # All logits = 0 except index 3 = 5.0 (high softmax probability).
        out = torch.zeros((batch, self.n_classes), dtype=torch.float32)
        out[:, 3] = 5.0
        return out


class FakeStandardScaler:
    """
    Duck-typed stand-in for sklearn.preprocessing.StandardScaler.

    Exposes the surface the deployment code touches:
        - `.transform(X)`: identity transform - returns input unchanged.
          The real scaler would standardize per-feature; for tests we
          want the math to be a no-op so input -> output is predictable.
        - `.mean_`: numpy array of shape (n_features,). The loader's
          pca_load_complete log line reads `.mean_.shape[0]`, so the
          attribute must exist with the right shape.
        - `.scale_`: numpy array of shape (n_features,). Symmetric
          with mean_; included for completeness even though tests
          don't currently read it.
    """

    def __init__(self, n_features: int = 561) -> None:
        self.mean_ = np.zeros(n_features, dtype=np.float32)
        self.scale_ = np.ones(n_features, dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Identity transform - input passes through unchanged."""
        return np.asarray(X, dtype=np.float32)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """
    TestClient with the loader cache pre-populated by FakeDNN +
    FakeStandardScaler.

    Use this fixture in any test expecting a "loaded model" state -
    the production state after lifespan runs. The router's
    get_dnn_model() returns the FakeDNN; get_scaler() returns the
    FakeStandardScaler; get_model_version() returns "1".

    monkeypatch.setattr swaps the module attribute for the duration
    of THIS test only - automatically restored at teardown. No manual
    cleanup; no risk of state leaking between tests.
    """
    fake_model: Any = FakeDNN()
    fake_model.eval()  # Mirror the loader's eval-mode invariant.
    fake_scaler: Any = FakeStandardScaler()
    monkeypatch.setattr(dnn_loader, "_MODEL", fake_model)
    monkeypatch.setattr(dnn_loader, "_SCALER", fake_scaler)
    monkeypatch.setattr(dnn_loader, "_MODEL_VERSION", "1")

    # TestClient(app) without `with` does NOT trigger lifespan, so
    # our manually-populated cache is what get_dnn_model() sees.
    yield TestClient(app)


@pytest.fixture
def client_unloaded(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """
    TestClient with the loader cache CLEARED.

    Use this fixture for tests exercising the "model not loaded"
    path - /ready returning 503, /predict/dnn returning 503. Mirrors
    the real-world startup window before the lifespan event finishes.
    """
    monkeypatch.setattr(dnn_loader, "_MODEL", None)
    monkeypatch.setattr(dnn_loader, "_SCALER", None)
    monkeypatch.setattr(dnn_loader, "_MODEL_VERSION", None)

    yield TestClient(app)
