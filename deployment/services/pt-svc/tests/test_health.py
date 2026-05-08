"""
Tests for /health and /ready.

WHAT THESE TESTS COVER:
    /health:
        - Returns 200 with {"status": "alive"} regardless of model state.
          Liveness should never gate on real work - if it did, k8s would
          needlessly restart the container during the load window.

    /ready:
        - Returns 200 with full status payload (status, model_loaded,
          model_name, model_version) when both DNN and scaler are
          cached.
        - Returns 503 with detail="model_not_loaded" when the cache is
          empty. Mirrors the readinessProbe contract Kubernetes expects.

WHAT THESE TESTS DO NOT COVER:
    - Real MLflow load path (covered by manual smoke testing in
      Step 2.8 with a real UCI HAR sample). Tests stay deterministic.
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
    """/ready returns 200 with full payload when loaded."""
    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["model_name"] == "pt-dnn"
    # model_version comes from the conftest fixture (set to "1").
    assert body["model_version"] == "1"


def test_ready_when_unloaded(client_unloaded: TestClient) -> None:
    """/ready returns 503 with detail=model_not_loaded when not loaded."""
    response = client_unloaded.get("/ready")

    assert response.status_code == 503
    # FastAPI's HTTPException renders as {"detail": <whatever>}
    assert response.json() == {"detail": "model_not_loaded"}
