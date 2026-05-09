"""
End-to-end smoke test for the pt-svc /predict/qlearning/taxi endpoint.

WHAT THIS SCRIPT DOES:
    1. Loads the real Q-table from the modeling phase
       (PyTorch/20-q-learning/results/v1_qtable_taxi.npy).
    2. Computes the EXPECTED (action, q_values) for a fixed set of
       targeted spot-check states and a stratified sweep of every
       50th state across [0, 500).
    3. POSTs each state to a running pt-svc instance.
    4. Bit-compares server output against the manual forward:
        - action must equal int(np.argmax(Q[state]))
        - q_values must equal Q[state].tolist() within float64
          JSON round-trip (no precision amplification in Q-learning)

WHY THIS SMOKE TEST IS SIMPLER THAN THE DNN/GAN ONES:
    Tabular Q-learning has no preprocessing, no autograd, no
    randomness. The "inference" is one numpy operation: argmax of
    a 6-element vector. There's no way for float-precision artifacts
    to amplify across operations (because there's only ONE operation),
    so the server's response should match the manual forward exactly,
    not within tolerance.

USAGE (from deployment/services/pt-svc/):
    1. Boot the server:
        .venv\\Scripts\\uvicorn.exe app.main:app --port 8002
    2. Run this script:
        .venv\\Scripts\\python.exe scripts/smoke_test_qlearning.py

ASSUMPTIONS:
    - The server is running on localhost:8002 (pt-svc port).
    - The service's loaded Q-table is the same artifact as
      PyTorch/20-q-learning/results/v1_qtable_taxi.npy (verified by
      promote_to_registry.py when the model was registered).
"""

import sys
from pathlib import Path
from typing import Any

# Make `app` importable. When this script runs directly (not via
# uvicorn or pytest), Python only adds the script's own directory
# (`scripts/`) to sys.path - not the parent service dir. Inserting
# the parent makes `from app.x import y` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from app.schemas.qlearning import ACTION_NAMES, N_STATES  # noqa: E402

# Path resolution

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
QTABLE_PATH  = (
    PROJECT_ROOT / "PyTorch" / "20-q-learning" / "results" / "v1_qtable_taxi.npy"
)

SERVICE_URL  = "http://localhost:8002/predict/qlearning/taxi"
TIMEOUT_SEC  = 5.0    # Q-learning is sub-millisecond; 5s is generous

# Targeted spot-check states. Cover terminal (all-zero row), two
# known-action cases from pre-flight inspection, mid-range, and the
# upper boundary.
SPOT_CHECK_STATES: tuple[int, ...] = (0, 1, 4, 42, 499)

# Stratified sweep - every 50th state across the space. Light
# coverage that catches systemic drift without 500 HTTP calls.
SWEEP_STATES: tuple[int, ...] = tuple(range(0, N_STATES, 50))


def _post_and_decode(
    client: httpx.Client, state: int
) -> dict[str, Any]:
    """POST a state to the service and return the parsed JSON body."""
    response = client.post(SERVICE_URL, json={"state": state})
    if response.status_code != 200:
        raise RuntimeError(
            f"Service returned {response.status_code} for state={state}: "
            f"{response.text[:200]}"
        )
    return dict(response.json())


def main() -> int:
    print("=" * 70)
    print("pt-svc /predict/qlearning/taxi smoke test")
    print("=" * 70)

    # Step 1: load the modeling-phase Q-table.
    print("\n[1/4] Loading Q-table from modeling phase...")
    qtable = np.load(QTABLE_PATH)
    print(f"      file:    {QTABLE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"      shape:   {qtable.shape}")
    print(f"      dtype:   {qtable.dtype}")
    print(f"      range:   [{qtable.min():.4f}, {qtable.max():.4f}]")

    # Open one httpx client; reuse the connection across requests.
    try:
        client = httpx.Client(timeout=TIMEOUT_SEC)
    except Exception as exc:
        print(f"      ERROR creating httpx client: {exc}")
        return 1

    # Step 2: targeted spot-checks. Print full output so a human can
    # eyeball each one.
    print(f"\n[2/4] Targeted spot-checks ({len(SPOT_CHECK_STATES)} states)...")
    try:
        for state in SPOT_CHECK_STATES:
            expected_action = int(np.argmax(qtable[state]))
            expected_q = qtable[state].tolist()
            expected_name = ACTION_NAMES[expected_action]

            try:
                body = _post_and_decode(client, state)
            except httpx.ConnectError:
                print(f"      ERROR: cannot connect to {SERVICE_URL}.")
                print("      Is the server running? Try in another terminal:")
                print("        .venv\\Scripts\\uvicorn.exe app.main:app --port 8002")
                return 1

            service_action = int(body["action"])
            service_name = body["action_name"]
            service_q = list(body["q_values"])
            service_state = int(body["state"])

            action_match = service_action == expected_action
            name_match = service_name == expected_name
            state_match = service_state == state
            # Q-values: convert both to numpy for elementwise compare.
            q_diff = np.abs(np.asarray(service_q) - np.asarray(expected_q))
            q_match = bool(q_diff.max() == 0)

            ok = action_match and name_match and state_match and q_match
            mark = "OK  " if ok else "FAIL"
            print(
                f"      [{mark}] state={state:3d}  "
                f"action={service_action} ({service_name})  "
                f"max_q_diff={q_diff.max():.2e}"
            )
            if not ok:
                print(f"             expected: action={expected_action} "
                      f"({expected_name})  q={[round(q, 4) for q in expected_q]}")
                print(f"             got:      action={service_action} "
                      f"({service_name})  q={[round(q, 4) for q in service_q]}")
                return 1

        # Step 3: stratified sweep. argmax-only check across the space.
        print(f"\n[3/4] Stratified sweep ({len(SWEEP_STATES)} states, every 50th)...")
        mismatches: list[tuple[int, int, int]] = []
        for state in SWEEP_STATES:
            expected_action = int(np.argmax(qtable[state]))
            body = _post_and_decode(client, state)
            service_action = int(body["action"])
            if service_action != expected_action:
                mismatches.append((state, expected_action, service_action))

        if mismatches:
            print(f"      FAIL: {len(mismatches)} state(s) disagreed:")
            for s, exp, got in mismatches:
                print(f"        state={s}: expected action={exp}, got {got}")
            return 1
        print(f"      OK: all {len(SWEEP_STATES)} swept states agree with manual argmax")

        # Step 4: summary.
        print("\n[4/4] Summary")
        # Detail lines: count terminal vs non-terminal states queried.
        all_queried = set(SPOT_CHECK_STATES) | set(SWEEP_STATES)
        terminal_count = sum(
            1 for s in all_queried if (qtable[s] == 0).all()
        )
        print(f"      states queried:  {len(all_queried)}")
        print(f"        terminal:      {terminal_count}")
        print(f"        non-terminal:  {len(all_queried) - terminal_count}")
        print("      bit-exact match: 100%")

    finally:
        client.close()

    print("\n" + "=" * 70)
    print("  PASS - service argmax + q_values bit-match modeling-phase Q-table")
    print(f"  Q-table:  {qtable.shape} {qtable.dtype}")
    print(f"  Coverage: {len(SPOT_CHECK_STATES)} spot-checks + "
          f"{len(SWEEP_STATES)} swept states")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
