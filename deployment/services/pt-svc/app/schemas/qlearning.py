"""
Pydantic schemas for the /predict/qlearning/taxi endpoint.

WHAT THIS FILE IS:
    The API contract for the Q-learning Taxi-v4 endpoint. Two classes:
        - TaxiRequest:  what the client MUST send (validated on the way in)
        - TaxiResponse: what the service WILL return (validated on the way out)

    FastAPI introspects these to validate incoming JSON, generate the
    OpenAPI schema for /docs, and serialize handler return values.

THE MODEL THIS CONTRACT REFLECTS:
    A tabular Q-learning policy trained on Gymnasium's Taxi-v4
    environment. The "model" is a numpy array of shape (500, 6):
        - 500 discrete states (5x5 grid * 5 passenger locations *
          4 destinations; 100 of those are terminal/unreachable)
        - 6 discrete actions: south, north, east, west, pickup, dropoff

    Inference is just `Q[state].argmax()` - no preprocessing, no
    learnable transform, no batch dimension. The Q-table IS the model.

GYMNASIUM TAXI-v4 ACTION MAPPING (locked by the environment):
    0: south   1: north   2: east   3: west   4: pickup   5: dropoff
    See https://gymnasium.farama.org/environments/toy_text/taxi/ for
    the canonical state encoding (taxi position, passenger location,
    destination). Clients constructing a state from the structured
    form can use the standard formula:
        state = ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + dest

DESIGN DECISIONS:
    - state is a plain int in [0, 500), matching Gymnasium's
      Discrete(500) observation_space. We do NOT accept the structured
      (row, col, passenger_loc, dest) form - clients encode it
      themselves before posting. Keeps the API minimal and avoids
      validating internal consistency of the structured form.
    - Response includes the FULL 6-element Q-vector, not just the
      argmax. Lets clients see WHY the agent chose this action (top
      Q-value vs runner-up gap) and detect terminal states themselves
      (sum(q_values) == 0 on the 100 unreachable absorbing states)
      without us inventing a special is_terminal flag.
    - action_name is a Literal type so the Swagger UI shows the
      6-action enum and mypy catches typos at handler-construction
      time.
    - state is echoed in the response (request/response symmetry).
      Lets clients confirm what was queried without having to remember.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Module constants
# ---------------------------------------------------------------------------
# Pulled from the modeling phase: Gymnasium's Taxi-v4 environment +
# the trained Q-table at PyTorch/20-q-learning/results/v1_qtable_taxi.npy.

# At module level (not inside a class) so the loader, the router, the
# tests, and the smoke test can all import them without re-deriving.

N_STATES: int = 500              # Gymnasium Taxi-v4 Discrete(500) observation space
N_ACTIONS: int = 6               # 6 discrete actions

# Action names in the canonical Gymnasium order. Index i of the
# 6-vector Q-row corresponds to ACTION_NAMES[i] - critical that the
# order matches the trained Q-table's action layout.
ACTION_NAMES: tuple[str, ...] = (
    "south",
    "north",
    "east",
    "west",
    "pickup",
    "dropoff",
)

# Literal type for the action_name field. Same self-enforcing-enum
# pattern as DNNResponse.predicted_label: Pydantic validates at
# runtime, mypy validates statically, FastAPI exposes the enum to
# the Swagger UI's "Try it out" panel as a dropdown.
ActionLabel = Literal[
    "south",
    "north",
    "east",
    "west",
    "pickup",
    "dropoff",
]


# Request schema


class TaxiRequest(BaseModel):
    """
    Client payload for POST /predict/qlearning/taxi.

    Attributes:
        state: Integer state ID in [0, 500) per Gymnasium's Taxi-v4
            observation space. The 100 states where the passenger has
            already been delivered are technically valid inputs but
            return all-zero Q-values (clients should treat them as
            episode-terminal and call env.reset()).

    Example payloads:
        {"state": 0}     # taxi at (0,0), passenger at R, dest R (terminal)
        {"state": 328}   # mid-episode state; reached during training
        {"state": 499}   # taxi at (4,4), passenger at B, dest B (terminal)
    """

    state: int = Field(
        ...,
        ge=0,
        lt=N_STATES,
        description=(
            f"Integer state ID in [0, {N_STATES}) per Gymnasium's "
            "Taxi-v4 Discrete(500) observation space. Clients can "
            "encode from the structured form via "
            "((row*5 + col)*5 + passenger_loc)*4 + dest."
        ),
        examples=[0, 328, 499],
    )


# Response schema


class TaxiResponse(BaseModel):
    """
    Service payload returned from POST /predict/qlearning/taxi.

    Attributes:
        state: Echo of the requested state. Lets clients confirm
            which state's policy was queried.
        action: Integer action ID in [0, 5] - the argmax of
            Q[state]. Use action_name for the human-readable name.
        action_name: Human-readable action. Always one of the 6
            Gymnasium Taxi-v4 actions. Constrained to the exact
            6-action enum.
        q_values: The full 6-element Q-vector for this state, in
            ACTION_NAMES order. Real-valued; can be negative
            (penalty paths) or large positive (successful-dropoff
            paths near +20 reward). Sum is 0 for the 100 terminal
            states - clients can detect those via sum(q_values) == 0.

    Example payload:
        {
          "state": 328,
          "action": 1,
          "action_name": "north",
          "q_values": [4.7, 9.6, 2.4, 5.0, -3.1, -2.8]
        }
    """

    state: int = Field(
        ...,
        ge=0,
        lt=N_STATES,
        description="Echo of the requested state.",
    )

    action: int = Field(
        ...,
        ge=0,
        le=N_ACTIONS - 1,
        description=(
            f"Predicted action ID in [0, {N_ACTIONS - 1}]. Argmax of "
            "Q[state]. Use action_name for the human-readable name."
        ),
    )

    action_name: ActionLabel = Field(
        ...,
        description=(
            "Predicted action name. Always one of the 6 Gymnasium "
            "Taxi-v4 actions."
        ),
    )

    q_values: list[float] = Field(
        ...,
        min_length=N_ACTIONS,
        max_length=N_ACTIONS,
        description=(
            f"Full {N_ACTIONS}-element Q-vector for this state in "
            "ACTION_NAMES order. Real-valued and unbounded (can be "
            "negative). Sum is 0 for terminal states; clients can "
            "detect those via sum(q_values) == 0."
        ),
    )
