"""
Tests for POST /predict/qlearning/taxi.

WHAT THESE TESTS COVER:
    Happy path:
        - Mid-range state -> 200 with TaxiResponse-shaped JSON
          (state echoed, action int 0-5, action_name from the
          Gymnasium 6-action enum, q_values list of 6 floats).
        - state=0 (lower bound) and state=499 (upper bound) both
          accepted.

    Action name mapping (parametrized over all 6 actions):
        - state s -> action (s % 6) -> ACTION_NAMES[s % 6].
          Cycles through south/north/east/west/pickup/dropoff and
          confirms the schema's Literal enum stays in lockstep with
          the trained Q-table's column order.

    Validation rejections (HTTP 422 from Pydantic Field bounds):
        - state=-1 -> Field ge=0 violation (greater_than_equal).
        - state=500 -> Field lt=500 violation (less_than).
        - state=99999 -> sanity, far above range.
        - state="foo" -> wrong type, Pydantic int parse fails.

    Loader-state rejection:
        - 503 when the Q-table cache is empty (uses client_unloaded
          fixture).

NOTE ON FAKEQTABLE BEHAVIOR:
    conftest.py's make_fake_qtable() builds a deterministic
    (500, 6) array where row s has exactly one non-zero entry at
    column (s % 6). Therefore argmax(qtable[s]) == s % 6 by
    construction. Tests rely on this mapping to assert specific
    argmax outcomes. The real Q-table has different values, but
    the router code path tested here is the same: index by state,
    argmax over 6, look up action_name.
"""

import pytest
from fastapi.testclient import TestClient

from app.schemas.qlearning import ACTION_NAMES, N_ACTIONS, N_STATES
from app.services import qlearning_loader

# Helper - building a {state: N} payload is repeated.

def _state_payload(state: int) -> dict[str, int]:
    """Return a {state: int} dict for posting."""
    return {"state": state}


# Happy path


def test_predict_qlearning_happy_path(client: TestClient) -> None:
    """
    Mid-range state -> 200 with the right response shape.

    state=42 maps to action 42 % 6 = 0 ('south') under FakeQtable's
    cycle. Asserts every field in the TaxiResponse contract is
    populated with the right type.
    """
    response = client.post("/predict/qlearning/taxi", json=_state_payload(42))

    assert response.status_code == 200

    body = response.json()
    assert "state" in body
    assert "action" in body
    assert "action_name" in body
    assert "q_values" in body

    assert body["state"] == 42
    assert body["action"] == 42 % N_ACTIONS  # FakeQtable cycle
    assert body["action_name"] == ACTION_NAMES[42 % N_ACTIONS]
    assert isinstance(body["q_values"], list)
    assert len(body["q_values"]) == N_ACTIONS
    assert all(isinstance(q, float) for q in body["q_values"])


def test_predict_qlearning_state_lower_bound(client: TestClient) -> None:
    """state=0 (lower boundary of [0, 500)) is accepted."""
    response = client.post("/predict/qlearning/taxi", json=_state_payload(0))

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == 0
    # FakeQtable: state 0 -> action 0 -> 'south'.
    assert body["action"] == 0
    assert body["action_name"] == "south"


def test_predict_qlearning_state_upper_bound(client: TestClient) -> None:
    """state=499 (one below the strict upper bound 500) is accepted."""
    response = client.post(
        "/predict/qlearning/taxi", json=_state_payload(N_STATES - 1)
    )

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == N_STATES - 1
    # FakeQtable: state 499 -> action 499 % 6 = 5 -> 'dropoff'.
    assert body["action"] == (N_STATES - 1) % N_ACTIONS
    assert body["action_name"] == ACTION_NAMES[(N_STATES - 1) % N_ACTIONS]


def test_predict_qlearning_state_echoed(client: TestClient) -> None:
    """
    The state field in the response equals the request's state.

    Echoing the state lets clients confirm what was queried without
    having to remember.
    """
    response = client.post(
        "/predict/qlearning/taxi", json=_state_payload(123)
    )

    assert response.status_code == 200
    assert response.json()["state"] == 123


# Action name mapping (parametrized over all 6 states/actions)


@pytest.mark.parametrize(
    "state,expected_action,expected_name",
    [
        (0, 0, "south"),
        (1, 1, "north"),
        (2, 2, "east"),
        (3, 3, "west"),
        (4, 4, "pickup"),
        (5, 5, "dropoff"),
    ],
)
def test_predict_qlearning_action_name_mapping(
    client: TestClient,
    state: int,
    expected_action: int,
    expected_name: str,
) -> None:
    """
    State s -> action (s % 6) -> ACTION_NAMES[s % 6] for all 6 actions.

    Walks the first 6 states (which directly map to actions 0-5
    under FakeQtable's cycle) and confirms each one returns both the
    right action ID and the matching human-readable name. If the
    schema's Literal enum ever drifts from the trained Q-table's
    column order, this is the canary.
    """
    response = client.post("/predict/qlearning/taxi", json=_state_payload(state))

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == expected_action
    assert body["action_name"] == expected_name


# Validation rejections (Field bounds)


def test_predict_qlearning_state_negative_rejected(client: TestClient) -> None:
    """state=-1 -> 422 from Field(ge=0)."""
    response = client.post(
        "/predict/qlearning/taxi", json=_state_payload(-1)
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "greater_than_equal" for err in detail)


def test_predict_qlearning_state_at_500_rejected(client: TestClient) -> None:
    """
    state=500 -> 422 from Field(lt=500).

    The bound is strictly less-than, so 500 itself is rejected
    even though it's the size of the observation space. Valid
    states are [0, 499] inclusive.
    """
    response = client.post(
        "/predict/qlearning/taxi", json=_state_payload(N_STATES)
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "less_than" for err in detail)


def test_predict_qlearning_state_far_too_large_rejected(
    client: TestClient,
) -> None:
    """state=99999 -> 422 (sanity, well above range)."""
    response = client.post(
        "/predict/qlearning/taxi", json=_state_payload(99999)
    )

    assert response.status_code == 422


def test_predict_qlearning_wrong_type(client: TestClient) -> None:
    """
    Non-int state -> 422 from Pydantic's int parse.

    Pydantic v2 will coerce a numeric string ('42' -> 42) but rejects
    truly non-numeric values like 'foo'.
    """
    response = client.post(
        "/predict/qlearning/taxi", json={"state": "foo"}
    )

    assert response.status_code == 422


# Loader-state rejection


def test_predict_qlearning_model_unloaded(client_unloaded: TestClient) -> None:
    """
    /predict/qlearning/taxi returns 503 when the Q-table cache is empty.

    Defense-in-depth: the lifespan event guarantees the Q-table is
    loaded before any request in production. But if get_qtable()
    raises (e.g., during a hot-reload window or pathological state),
    the router catches RuntimeError and returns 503 - same shape
    /ready returns for the same condition.
    """
    response = client_unloaded.post(
        "/predict/qlearning/taxi", json=_state_payload(42)
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "model_not_loaded"}


# Inference-latency histogram


def test_predict_qlearning_records_inference_duration(
    client: TestClient,
) -> None:
    """
    A successful /predict/qlearning/taxi records the
    model_inference_duration_seconds histogram with
    model_name="pt-qlearning-taxi".

    Verifies the .time() context manager wrapping the Q-table lookup +
    argmax in the router emits a labeled sample. Q-learning's inference
    is the cheapest in the portfolio (one index + one argmax over 6
    floats); we measure it for consistency with the other endpoints.
    The substring asserted below is the `_count` line prometheus_client
    auto-generates for any Histogram with at least one observation.
    """
    response = client.post(
        "/predict/qlearning/taxi", json=_state_payload(42)
    )
    assert response.status_code == 200, "setup precondition failed"

    body = client.get("/metrics").text

    assert (
        'model_inference_duration_seconds_count{model_name="pt-qlearning-taxi"}'
        in body
    )


# Per-model freshness endpoint /health/qlearning


def test_health_qlearning_loaded_fresh_returns_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    /health/qlearning returns 200 with diagnostic body when the Q-table is
    loaded AND _LAST_INFERENCE_TS is within the staleness threshold.
    """
    monkeypatch.delenv("MODEL_INFERENCE_STALENESS_SECONDS", raising=False)
    response = client.get("/health/qlearning")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_name"] == qlearning_loader.MODEL_NAME
    assert body["version"] == "1"
    assert body["staleness_threshold_seconds"] == 3600
    assert 0.0 <= body["last_inference_age_seconds"] < 5.0


def test_health_qlearning_loaded_stale_returns_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    /health/qlearning returns 503 inference_stale when the Q-table is
    loaded but the last inference timestamp is beyond the staleness
    threshold.
    """
    monkeypatch.setenv("MODEL_INFERENCE_STALENESS_SECONDS", "1")
    monkeypatch.setattr(qlearning_loader, "_LAST_INFERENCE_TS", 0.0)

    response = client.get("/health/qlearning")

    assert response.status_code == 503
    assert response.json() == {"detail": "inference_stale"}


def test_health_qlearning_unloaded_returns_503(
    client_unloaded: TestClient,
) -> None:
    """
    /health/qlearning returns 503 model_not_loaded when the Q-table cache
    is empty.
    """
    response = client_unloaded.get("/health/qlearning")

    assert response.status_code == 503
    assert response.json() == {"detail": "model_not_loaded"}
