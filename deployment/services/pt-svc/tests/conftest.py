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
    - `FakeGenerator` - real nn.Module stand-in for the DCGAN generator.
      Forward IS z-dependent (different noise -> different output) so
      future seed-reproducibility tests can verify the seed plumbing
      without loading the 1M-param real model.
    - `client` - TestClient with BOTH loader caches populated by the
      fakes. Use for tests expecting "all loaded" state.
    - `client_unloaded` - TestClient with BOTH caches cleared. Use for
      503 / unloaded behavior tests.

WHY MOCK THE MODELS:
    The real DNN + scaler + DCGAN load takes ~2 seconds and requires
    the MLflow registry + .pth artifacts on disk - none of which are
    available in CI environments that pull only source code. Mocking
    the cache slots directly bypasses MLflow entirely while keeping
    every other code path (routes, middleware, validation, schema
    serialization) running for real.

LIFESPAN BEHAVIOR:
    `TestClient(app)` (without `with`) does NOT trigger the lifespan
    event. So our dnn_loader.load_dnn_model() and
    gan_loader.load_gan_model() do NOT run during tests - we manually
    populate the cache slots in the fixtures instead.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn

from app.main import app
from app.services import dnn_loader, gan_loader


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
          dnn_load_complete log line reads `.mean_.shape[0]`, so the
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


class FakeGenerator(nn.Module):
    """
    Real nn.Module stand-in for the DCGAN generator.

    I/O contract matches the real DCGenerator:
        Input  z:  (batch, 100, 1, 1)  float32
        Output:    (batch, 3, 32, 32)  float32 in [-1, 1]

    Forward IS z-dependent: different noise produces different output.
    Critical for future seed-reproducibility tests - if the fake
    ignored z and returned constants, "POST with seed=42 twice"
    would always look identical regardless of whether the seed
    plumbing actually worked. With z-dependence, identical seeds
    produce identical bytes only when torch.manual_seed() is being
    called correctly in the router.

    The math is intentionally simple (broadcast + tanh) so test
    output is deterministic given z, with no Conv2d / BatchNorm /
    ReLU layers. State_dict has zero entries; FakeGenerator() is
    stateless apart from the nn.Module scaffold.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Deterministic z-dependent output in [-1, 1].

        Takes the first 3 channels of z, broadcasts (1, 1) -> (32, 32),
        and tanh-bounds. Different z -> different output; same z ->
        same output to the bit.
        """
        first3 = z[:, :3, :, :]              # (B, 3, 1, 1)
        broadcast = first3.expand(-1, -1, 32, 32)  # (B, 3, 32, 32)
        return torch.tanh(broadcast)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """
    TestClient with BOTH loader caches pre-populated by fakes.

    Use this fixture in any test expecting an "all loaded" state -
    the production state after lifespan finishes. The DNN loader's
    get_dnn_model()/get_scaler()/get_model_version() return the
    FakeDNN/FakeStandardScaler/"1"; the GAN loader's
    get_gan_model()/get_model_version() return the FakeGenerator/"1".

    monkeypatch.setattr swaps the module attribute for the duration
    of THIS test only - automatically restored at teardown. No manual
    cleanup; no risk of state leaking between tests.
    """
    # DNN loader cache slots
    fake_dnn: Any = FakeDNN()
    fake_dnn.eval()  # Mirror the loader's eval-mode invariant.
    fake_scaler: Any = FakeStandardScaler()
    monkeypatch.setattr(dnn_loader, "_MODEL", fake_dnn)
    monkeypatch.setattr(dnn_loader, "_SCALER", fake_scaler)
    monkeypatch.setattr(dnn_loader, "_MODEL_VERSION", "1")

    # GAN loader cache slots
    fake_gen: Any = FakeGenerator()
    fake_gen.eval()
    monkeypatch.setattr(gan_loader, "_MODEL", fake_gen)
    monkeypatch.setattr(gan_loader, "_MODEL_VERSION", "1")

    # TestClient(app) without `with` does NOT trigger lifespan, so
    # our manually-populated caches are what the loaders' getters see.
    yield TestClient(app)


@pytest.fixture
def client_unloaded(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """
    TestClient with BOTH loader caches CLEARED.

    Use this fixture for tests exercising the "model not loaded"
    path - /ready returning 503, /predict/dnn returning 503,
    /predict/gan/sample returning 503. Mirrors the real-world
    startup window before the lifespan event finishes loading.
    """
    # DNN loader cache slots
    monkeypatch.setattr(dnn_loader, "_MODEL", None)
    monkeypatch.setattr(dnn_loader, "_SCALER", None)
    monkeypatch.setattr(dnn_loader, "_MODEL_VERSION", None)

    # GAN loader cache slots
    monkeypatch.setattr(gan_loader, "_MODEL", None)
    monkeypatch.setattr(gan_loader, "_MODEL_VERSION", None)

    yield TestClient(app)
