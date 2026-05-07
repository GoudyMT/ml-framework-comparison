"""
Shared pytest fixtures for sklearn-svc tests.

WHAT THIS FILE PROVIDES:
    - `FakePCA` - a stand-in for sklearn's PCA estimator, sufficient
      to make the router happy. Just enough surface to look like a
      real PCA from the handler's perspective.
    - `client` - a TestClient with the loader's cache populated by
      a FakePCA instance. Use this for tests that expect a loaded model.
    - `client_unloaded` - a TestClient with the loader's cache cleared.
      Use this for tests that expect 503 / unloaded behavior.

WHY conftest.py:
    pytest auto-discovers `conftest.py` and shares its fixtures across
    every test file in the same directory tree. We define the fixtures
    once here; test_health.py / test_pca.py / test_middleware.py all
    use them by name.

WHY MOCK THE MODEL (not load the real one):
    Tests should be deterministic and fast. Loading from MLflow:
        - Takes 1-2 seconds per test
        - Requires the registry DB + artifact files to exist
        - Requires the modeling-phase artifacts to be available in CI,
          which they're not (CI pulls only the code, not mlruns/)
    Mocking the cache directly bypasses MLflow entirely while keeping
    every other code path (routes, middleware, validation, schema
    serialization) running for real. That's exactly the right test
    boundary - we trust sklearn's PCA, we want to verify OUR plumbing.

LIFESPAN BEHAVIOR:
    `TestClient(app)` (without `with`) does NOT trigger the lifespan
    event. So our pca_loader.load_pca_model() does NOT run during tests
    - we manually populate _MODEL et al. in the fixture instead. If a
    future test needs lifespan to fire, it would use:
        with TestClient(app) as client: ...
    For everything we're testing now, the manual-populate approach is
    cleaner.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import pca_loader


class FakePCA:
    """
    Minimal sklearn PCA look-alike for tests.

    Exposes just enough attributes/methods for the router to work:
        - n_components_       (int)
        - explained_variance_ratio_  (numpy array, length n_components)
        - transform(X)        (callable: ndarray (batch, 784) -> ndarray (batch, 150))

    The transform implementation returns deterministic dummy components
    (a simple linear projection of the row mean). Real PCA would use the
    learned components_ matrix. Tests don't check the exact values - they
    check that the response shape + Pydantic validation work end-to-end.
    """

    def __init__(self, n_components: int = 150) -> None:
        self.n_components_ = n_components
        # explained_variance_ratio_ in real PCA sums to <=1; we use a
        # uniform distribution so total = 1.0 (a "perfect" PCA, useful
        # only as a placeholder).
        self.explained_variance_ratio_ = np.full(
            n_components, 1.0 / n_components, dtype=np.float32
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Stub transform: return shape (batch, n_components).

        Returns the row mean repeated across the output dimensions.
        Deterministic, fast, no real linear algebra. Tests that check
        shape pass; tests that check actual values would need a real PCA.
        """
        # X shape: (batch, 784). Return (batch, 150) by broadcasting
        # row means out to 150 columns.
        row_means = X.mean(axis=1, keepdims=True).astype(np.float32)
        # tile to (batch, n_components)
        return np.tile(row_means, (1, self.n_components_))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """
    TestClient with the loader cache pre-populated by a FakePCA.

    Use this fixture in any test that expects a "loaded model" - which
    is the production state after lifespan runs. The router's
    get_pca_model() reads from the cache and returns the FakePCA;
    get_variance_explained() returns 1.0; get_model_version() returns "1".

    monkeypatch.setattr swaps the module attribute for the duration of
    the test ONLY - automatically restored at teardown. No manual cleanup
    needed; no risk of state leaking between tests.
    """
    fake_model: Any = FakePCA()
    monkeypatch.setattr(pca_loader, "_MODEL", fake_model)
    monkeypatch.setattr(pca_loader, "_VARIANCE_EXPLAINED", 1.0)
    monkeypatch.setattr(pca_loader, "_MODEL_VERSION", "1")

    # Yield a TestClient. Without `with`, lifespan does NOT run - so
    # our manually-set cache is what get_pca_model() sees, not the real
    # MLflow load.
    yield TestClient(app)


@pytest.fixture
def client_unloaded(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """
    TestClient with the loader cache CLEARED.

    Use this fixture in any test that exercises the "model not loaded"
    path - /ready returning 503, /predict/pca returning 503. Mirrors
    the real-world startup window before the lifespan event finishes.
    """
    monkeypatch.setattr(pca_loader, "_MODEL", None)
    monkeypatch.setattr(pca_loader, "_VARIANCE_EXPLAINED", None)
    monkeypatch.setattr(pca_loader, "_MODEL_VERSION", None)

    yield TestClient(app)
