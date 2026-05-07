"""
Tests for /health and /ready.

WHAT THESE TESTS COVER:
    /health:
        - Returns 200 with {"status": "alive"} regardless of model state.
          Liveness should NEVER do real work (k8s will needlessly restart
          the container if /health flaps), so it should pass in any state.

    /ready:
        - Returns 200 with full status payload when the model is loaded.
        - Returns 503 with detail="model_not_loaded" when the model isn't.
          This is the kubernetes readinessProbe contract - 503 keeps
          traffic away from a not-yet-warm pod.

WHAT THESE TESTS DO NOT COVER:
    - The actual MLflow load path (covered by manual smoke testing in
      Step 1.8 with a real Fashion-MNIST sample). Tests stay deterministic.
    - Performance (covered by metrics endpoint tests + production
      monitoring, not unit tests).
"""

from fastapi.testclient import TestClient

# /health


def test_health_returns_alive(client: TestClient) -> None:
    """/health must always return 200 with status=alive."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_alive_even_when_model_unloaded(
    client_unloaded: TestClient,
) -> None:
    """
    /health must NOT depend on model state.

    A starting-up service is alive but not ready. /health must keep
    returning 200 during the load window so k8s doesn't restart the
    pod prematurely.
    """
    response = client_unloaded.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


# /ready


def test_ready_when_loaded(client: TestClient) -> None:
    """/ready returns 200 with full status payload when loaded."""
    response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["model_name"] == "sk-pca"
    # Version comes from the fixture - we set it to "1" in conftest.
    assert body["model_version"] == "1"


def test_ready_when_unloaded(client_unloaded: TestClient) -> None:
    """/ready returns 503 with detail=model_not_loaded when not loaded."""
    response = client_unloaded.get("/ready")

    assert response.status_code == 503
    # FastAPI's HTTPException renders as {"detail": <whatever>}
    assert response.json() == {"detail": "model_not_loaded"}
