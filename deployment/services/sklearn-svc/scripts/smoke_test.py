"""
End-to-end smoke test for the sklearn-svc /predict/pca endpoint.

WHAT THIS SCRIPT DOES:
    1. Loads a real Fashion-MNIST test sample from sample_raw.npy
       (raw uint8 pixels in [0, 255], 784-element flat vector,
       label 9 = "Ankle boot"). The .npy is committed alongside this
       script so the test runs without TensorFlow.
    2. Manually forward-pipelines the raw sample through /255 ->
       StandardScaler -> PCA, using the same artifacts the service
       loaded. This is the EXPECTED output.
    3. POSTs the raw pixels to a running sklearn-svc instance.
    4. Compares the service's components against the expected. They
       should match within float32 precision tolerance (~1e-3).

WHY FORWARD-ONLY VERIFICATION:
    An earlier draft tried to reverse-engineer raw pixels from the
    already-standardized X_test.npy. That path is precision-lossy:
    pixels with very small std (e.g., corner pixels in Fashion-MNIST
    that are usually black) amplify float-rounding errors by ~11x
    when reverse-standardizing, then PCA's linear combinations
    propagate those errors across all components. The forward path
    (raw -> standardize -> PCA) has no such amplification.

USAGE (from deployment/services/sklearn-svc/):
    1. Boot the server:
        .venv\\Scripts\\uvicorn.exe app.main:app --port 8001
    2. Run this script:
        .venv\\Scripts\\python.exe scripts/smoke_test.py

ASSUMPTIONS:
    - The server is running on localhost:8001
    - sample_raw.npy lives next to this script
    - The service's loaded PCA + scaler match the modeling-phase
      artifacts at data/processed/pca/ and Scikit-Learn/08-pca/results/
"""

import sys
from pathlib import Path

import httpx
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]

SAMPLE_PATH  = SCRIPT_DIR / "sample_raw.npy"
SCALER_PATH  = PROJECT_ROOT / "data" / "processed" / "pca" / "scaler.pkl"
PCA_PATH     = PROJECT_ROOT / "Scikit-Learn" / "08-pca" / "results" / "pca_model.joblib"

SERVICE_URL  = "http://localhost:8001/predict/pca"
TIMEOUT_SEC  = 5.0

# Float32 round-trip through the network/JSON loses some precision.
# 1e-3 is comfortably above typical observed delta (~1e-5).
COMPONENT_ATOL = 1e-3


def main() -> int:
    print("=" * 70)
    print("sklearn-svc /predict/pca smoke test")
    print("=" * 70)

    # Step 1: load the real Fashion-MNIST sample + the artifacts the
    # modeling phase produced.
    print("\n[1/4] Loading sample + modeling-phase artifacts...")
    sample_raw = np.load(SAMPLE_PATH).astype(np.float32)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    print(f"      sample_raw: shape={sample_raw.shape}, "
          f"range=[{int(sample_raw.min())}, {int(sample_raw.max())}], "
          f"mean={sample_raw.mean():.2f}")
    print(f"      scaler keys: {list(scaler.keys())}")
    print(f"      pca: n_components_={pca.n_components_}")

    # Step 2: forward-pipeline manually to produce the EXPECTED
    # components. This is the exact same chain the service runs;
    # any discrepancy points at a service-side bug.
    print("\n[2/4] Forward-pipelining sample to expected components...")
    X_norm = sample_raw / 255.0
    X_std = (X_norm - scaler["mean"]) / scaler["std"]
    expected = pca.transform(X_std.reshape(1, -1))[0].astype(np.float32)
    print(f"      X_norm range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")
    print(f"      X_std range:  [{X_std.min():.2f}, {X_std.max():.2f}]")
    print(f"      expected[:5]: {expected[:5].tolist()}")

    # Step 3: POST raw pixels to the running service.
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
        print("        .venv\\Scripts\\uvicorn.exe app.main:app --port 8001")
        return 1

    print(f"      Status: {response.status_code}")
    print(f"      X-Request-ID: {response.headers.get('X-Request-ID', '?')}")
    if response.status_code != 200:
        print(f"      Body: {response.text[:200]}")
        return 1

    body = response.json()
    server_components = np.asarray(body["components"], dtype=np.float32)
    print(f"      n_components:       {body['n_components']}")
    print(f"      variance_explained: {body['variance_explained']:.4f}")
    print(f"      service[:5]:        {server_components[:5].tolist()}")

    # Step 4: compare. With forward-only verification, these should
    # match within float32 + JSON-serialization precision.
    print("\n[4/4] Comparing service vs expected (forward-only)...")
    delta = np.abs(server_components - expected)
    print(f"      max abs diff:  {delta.max():.6e}")
    print(f"      mean abs diff: {delta.mean():.6e}")

    if delta.max() <= COMPONENT_ATOL:
        print("\n" + "=" * 70)
        print(f"  PASS - service matches forward pipeline within "
              f"{COMPONENT_ATOL} tolerance")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"  FAIL - service differs from expected (max diff "
              f"{delta.max():.4e} > {COMPONENT_ATOL})")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
