"""
End-to-end smoke test for the pt-svc /predict/dnn endpoint.

WHAT THIS SCRIPT DOES:
    1. Loads a real UCI HAR test sample from sample_raw.npy (561
       pre-normalized floats in [-1, 1], extracted once from
       data/raw/UCI HAR Dataset/test/X_test.txt). Sample 0 is true
       label 4 = STANDING.
    2. Forward-pipelines the raw sample through StandardScaler +
       DNN + softmax + argmax using the same artifacts the service
       loaded. This is the EXPECTED output.
    3. POSTs the raw features to a running pt-svc instance.
    4. Compares service output vs expected. They should match
       bit-exact - same float32 math on the same input through the
       same code paths.

WHY FORWARD-ONLY VERIFICATION:
    Earlier draft considered reverse-engineering raw input from the
    already-standardized data/processed/dnn/X_test.npy. That path
    is precision-lossy (StandardScaler division can amplify tiny
    float errors at low-std features), and the network's linear
    layers then spread those errors across all logits. Forward-
    pipelining from raw has no such amplification - bit-exact
    comparison is achievable.

USAGE (from deployment/services/pt-svc/):
    1. Boot the server:
        .venv\\Scripts\\uvicorn.exe app.main:app --port 8002
    2. Run this script:
        .venv\\Scripts\\python.exe scripts/smoke_test.py

ASSUMPTIONS:
    - The server is running on localhost:8002
    - sample_raw.npy lives next to this script (committed; ~2KB)
    - The service's loaded DNN + scaler match the modeling-phase
      artifacts at PyTorch/09-dnn/results/dnn_model.pth and
      data/processed/dnn/scaler.pkl

REGENERATING sample_raw.npy:
    The fixture was extracted once from the raw UCI HAR test set:
        from pathlib import Path
        import numpy as np
        with open('data/raw/UCI HAR Dataset/test/X_test.txt') as f:
            line = f.readline()
        np.save('sample_raw.npy', np.array(line.split(), dtype=np.float32))
    deployment/.gitignore has a negation rule preserving this file
    despite the global *.npy ignore.
"""

import sys
from pathlib import Path

# Make `app` importable. When this script runs directly (not via
# uvicorn or pytest), Python only adds the script's own directory
# (`scripts/`) to sys.path - not the parent service dir. Inserting
# the parent makes `from app.x import y` resolve. Standard pattern
# for service-local tooling scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.schemas.dnn import CLASS_NAMES  # noqa: E402
from app.services.dnn_model import DNN  # noqa: E402

# Path resolution

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]

SAMPLE_PATH  = SCRIPT_DIR / "sample_raw.npy"
SCALER_PATH  = PROJECT_ROOT / "data" / "processed" / "dnn" / "scaler.pkl"
WEIGHTS_PATH = PROJECT_ROOT / "PyTorch" / "09-dnn" / "results" / "dnn_model.pth"

SERVICE_URL  = "http://localhost:8002/predict/dnn"
TIMEOUT_SEC  = 5.0

# Tolerances. With forward-only verification on float32 + JSON
# round-trip, observed delta is typically 1e-7. atol=1e-3 is a
# generous ceiling that still detects real divergence.
PROBABILITY_ATOL = 1e-3


def main() -> int:
    print("=" * 70)
    print("pt-svc /predict/dnn smoke test")
    print("=" * 70)

    # Step 1: load fixture + modeling-phase artifacts.
    print("\n[1/4] Loading sample + modeling-phase artifacts...")
    sample_raw = np.load(SAMPLE_PATH).astype(np.float32)
    scaler = joblib.load(SCALER_PATH)
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model = DNN()
    model.load_state_dict(state_dict)
    model.eval()
    print(f"      sample_raw: shape={sample_raw.shape}, "
          f"range=[{sample_raw.min():.4f}, {sample_raw.max():.4f}]")
    print(f"      scaler: {type(scaler).__name__}, "
          f"mean_.shape={scaler.mean_.shape}")
    print(f"      DNN: n_params={sum(p.numel() for p in model.parameters()):,}, "
          f"eval_mode={not model.training}")

    # Step 2: forward-pipeline manually to produce the EXPECTED
    # output. Identical chain to what the service runs in the router.
    print("\n[2/4] Forward-pipelining sample to expected output...")
    X_scaled = scaler.transform(sample_raw.reshape(1, -1))
    X_tensor = torch.from_numpy(X_scaled).float()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        expected_class = int(logits.argmax(dim=1).item())
    expected_probs = probs.tolist()
    expected_label = CLASS_NAMES[expected_class]
    print(f"      X_scaled range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    print(f"      expected_class:        {expected_class}")
    print(f"      expected_label:        {expected_label}")
    print(f"      expected_probs:        "
          f"{[round(p, 4) for p in expected_probs]}")

    # Step 3: POST raw to the running service.
    print(f"\n[3/4] POST {SERVICE_URL}")
    try:
        response = httpx.post(
            SERVICE_URL,
            json={"features": sample_raw.tolist()},
            timeout=TIMEOUT_SEC,
        )
    except httpx.ConnectError:
        print(f"      ERROR: cannot connect to {SERVICE_URL}.")
        print("      Is the server running? Try in another terminal:")
        print("        .venv\\Scripts\\uvicorn.exe app.main:app --port 8002")
        return 1

    print(f"      Status: {response.status_code}")
    print(f"      X-Request-ID: {response.headers.get('X-Request-ID', '?')}")
    if response.status_code != 200:
        print(f"      Body: {response.text[:200]}")
        return 1

    body = response.json()
    service_class = int(body["predicted_class"])
    service_label = body["predicted_label"]
    service_probs = body["probabilities"]
    print(f"      service_class:         {service_class}")
    print(f"      service_label:         {service_label}")
    print(f"      service_probs:         "
          f"{[round(p, 4) for p in service_probs]}")

    # Step 4: compare. With forward-only on float32 + JSON-stable
    # round-trip, these should match bit-exact.
    print("\n[4/4] Comparing service vs expected...")
    if service_class != expected_class:
        print(f"      FAIL: predicted_class differs "
              f"({service_class} vs {expected_class})")
        return 1
    if service_label != expected_label:
        print(f"      FAIL: predicted_label differs "
              f"({service_label!r} vs {expected_label!r})")
        return 1

    delta = np.abs(np.asarray(service_probs) - np.asarray(expected_probs))
    print(f"      max abs prob diff:  {delta.max():.6e}")
    print(f"      mean abs prob diff: {delta.mean():.6e}")

    if delta.max() <= PROBABILITY_ATOL:
        print("\n" + "=" * 70)
        print(f"  PASS - service matches forward pipeline within "
              f"{PROBABILITY_ATOL} tolerance")
        print(f"  Predicted: class={service_class} ({service_label})")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"  FAIL - probability mismatch (max diff "
              f"{delta.max():.4e} > {PROBABILITY_ATOL})")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
