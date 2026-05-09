"""
Tests for /health and /ready.

WHAT THESE TESTS COVER:
    /health:
        - Returns 200 with {"status": "alive"} regardless of model state.
          Liveness should never gate on real work - if it did, k8s would
          needlessly restart the container during the load window.

    /ready:
        - Returns 200 with the multi-model payload (status + per-model
          loaded/version dict keyed by model name) when EVERY model
          is cached.
        - Returns 503 with detail="model_not_loaded" when ANY cache is
          empty. Mirrors the readinessProbe contract Kubernetes expects.

WHAT THESE TESTS DO NOT COVER:
    - Real MLflow load path (covered by the smoke test scripts that
      run against a live server with the real registered artifacts).
      Tests stay deterministic.
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

    The shape is {"status": "ready", "models": {<name>: {loaded,
    version}, ...}}. Every model the service hosts must appear in
    the dict. Versions come from the conftest fixtures (both set to
    "1" via monkeypatch).
    """
    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"

    models = body["models"]
    assert models["pt-dnn"] == {"loaded": True, "version": "1"}
    assert models["pt-gan-dcgan"] == {"loaded": True, "version": "1"}
    assert models["pt-qlearning-taxi"] == {"loaded": True, "version": "1"}


def test_ready_when_unloaded(client_unloaded: TestClient) -> None:
    """/ready returns 503 with detail=model_not_loaded when not loaded."""
    response = client_unloaded.get("/ready")

    assert response.status_code == 503
    # FastAPI's HTTPException renders as {"detail": <whatever>}
    assert response.json() == {"detail": "model_not_loaded"}
