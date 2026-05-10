"""
Tests for /health and /ready.

WHAT THESE TESTS COVER:
    /health:
        - Returns 200 with {"status": "alive"} regardless of model state.
          Liveness must NEVER do real work (k8s would needlessly restart
          the container if /health flaps).

    /ready:
        - Returns 200 with the multi-model payload (status + per-model
          loaded/version dict keyed by model name) when the Transformer
          AND the tokenizer are cached.
        - Returns 503 with detail="model_not_loaded" when either cache
          is empty. Mirrors the readinessProbe contract.

WHAT THESE TESTS DO NOT COVER:
    - Real MLflow load path (covered by scripts/smoke_test.py against
      a live server with the real registered artifacts). Tests stay
      deterministic.
    - Performance (covered by metrics endpoint tests + production
      monitoring, not unit tests).
"""

from fastapi.testclient import TestClient

# /health


def test_health_returns_alive(client: TestClient) -> None:
    """/health returns 200 with status=alive when the model is loaded."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_alive_even_when_model_unloaded(
    client_unloaded: TestClient,
) -> None:
    """
    /health does NOT depend on model state.

    A starting-up service is alive but not ready. /health must keep
    returning 200 during the load window so the orchestrator doesn't
    restart the pod prematurely.
    """
    response = client_unloaded.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


# /ready


def test_ready_when_loaded(client: TestClient) -> None:
    """
    /ready returns 200 with the multi-model payload when loaded.

    Shape is {"status": "ready", "models": {<name>: {loaded, version}}}.
    Single-model service today, dict structure is forward-compatible
    if a second TF model ever ships.
    """
    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"

    models = body["models"]
    assert models["tf-transformer-translation"] == {"loaded": True, "version": "1"}


def test_ready_when_unloaded(client_unloaded: TestClient) -> None:
    """/ready returns 503 with detail=model_not_loaded when not loaded."""
    response = client_unloaded.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "model_not_loaded"}
