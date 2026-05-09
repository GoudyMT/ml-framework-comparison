"""
End-to-end smoke test for the pt-svc /predict/gan/sample endpoint.

WHAT THIS SCRIPT DOES:
    1. Loads the real DCGenerator class + weights from the modeling
       phase (PyTorch/14-gans/results/dcgan_generator.pth).
    2. With a fixed seed, manually runs the forward pipeline:
           torch.manual_seed -> torch.randn -> model(z) -> denormalize
           -> permute NCHW -> NHWC -> uint8 pixels.
       This is the EXPECTED pixel array.
    3. POSTs {"n_samples": 1, "seed": <same>} to a running pt-svc.
    4. Decodes the base64 PNG from the response back to a numpy
       uint8 array via PIL.Image.open.
    5. Bit-compares expected vs decoded pixels - PNG is lossless,
       so the round-trip MUST preserve every pixel exactly.
    6. Additionally exercises:
        - n_samples=16 batch path (real DCGenerator handles the
          upper-bound batch through real conv + BN layers).
        - Repeat-call reproducibility (same seed across two calls
          -> identical bytes, proves server-side seed plumbing
          works under live HTTP, not just FakeGenerator).

WHY DECODE-AND-COMPARE-PIXELS, NOT BYTE-COMPARE-BASE64:
    PIL's PNG encoder is generally deterministic given identical
    input pixels, but the bit-pattern of the compressed output can
    drift slightly across PIL/zlib versions or compression-level
    defaults. The decoded PIXELS are guaranteed identical because
    PNG is lossless. Comparing pixels gives bit-exactness without
    coupling the test to PIL's serialization details.

WHY FORWARD-ONLY VERIFICATION:
    Same logic as the DNN smoke test: reverse-engineering "what z
    must have been" from the output PNG requires inverting a non-
    invertible transform (the discriminator's-eye view of the
    image). Forward-pipelining from a known seed produces an
    EXPECTED pixel array we can compare against without amplifying
    any float precision artifacts.

USAGE (from deployment/services/pt-svc/):
    1. Boot the server:
        .venv\\Scripts\\uvicorn.exe app.main:app --port 8002
    2. Run this script:
        .venv\\Scripts\\python.exe scripts/smoke_test_gan.py

ASSUMPTIONS:
    - The server is running on localhost:8002 (pt-svc port).
    - The service's loaded DCGenerator is the same artifact as
      PyTorch/14-gans/results/dcgan_generator.pth (verified at
      Phase 0 promote_to_registry.py time).
"""

import sys
from pathlib import Path

"""
Make `app` importable. When this script runs directly (not via
uvicorn or pytest), Python only adds the script's own directory
(`scripts/`) to sys.path - not the parent service dir. Inserting
the parent makes `from app.x import y` resolve.
"""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import base64  # noqa: E402
import io  # noqa: E402

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from app.schemas.gan import IMAGE_CHANNELS, IMAGE_SIZE, LATENT_DIM, N_SAMPLES_MAX  # noqa: E402
from app.services.gan_model import DCGenerator  # noqa: E402

# Path resolution

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
WEIGHTS_PATH = PROJECT_ROOT / "PyTorch" / "14-gans" / "results" / "dcgan_generator.pth"

SERVICE_URL  = "http://localhost:8002/predict/gan/sample"
TIMEOUT_SEC  = 10.0   # GAN forward + encode is slower than DNN; give headroom

# The seed is the only "knob" the smoke test pins. Any int works as
# long as both the manual forward AND the service request use the
# same value.
SEED = 42


def _decode_to_pixels(b64_string: str) -> np.ndarray:
    """
    Decode a base64-encoded PNG to a (H, W, 3) uint8 numpy array.

    The intermediate PIL.Image is discarded; only the pixel array
    matters for the comparison. Asserting mode == 'RGB' here protects
    against grayscale / RGBA corruption from a malformed handler.
    """
    img = Image.open(io.BytesIO(base64.b64decode(b64_string)))
    if img.mode != "RGB":
        raise ValueError(f"Expected RGB image, got mode={img.mode!r}")
    return np.asarray(img, dtype=np.uint8)


def main() -> int:
    print("=" * 70)
    print("pt-svc /predict/gan/sample smoke test")
    print("=" * 70)

    # Step 1: load the modeling-phase DCGenerator weights.
    print("\n[1/5] Loading DCGenerator + weights...")
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model = DCGenerator()
    model.load_state_dict(state_dict)
    model.eval()
    print(f"      weights file: {WEIGHTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"      n_params:     {sum(p.numel() for p in model.parameters()):,}")
    print(f"      eval_mode:    {not model.training}")

    # Step 2: forward-pipeline with seed=SEED to get EXPECTED pixels.
    # Mirror the router's exact pipeline so the manual output matches
    # what the service computes server-side.
    print(f"\n[2/5] Forward-pipelining seed={SEED} -> expected pixels...")
    torch.manual_seed(SEED)
    z = torch.randn(1, LATENT_DIM, 1, 1, dtype=torch.float32)
    with torch.no_grad():
        out = model(z)  # (1, 3, 32, 32) in [-1, 1]
    out_uint8 = ((out + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    expected_pixels: np.ndarray = (
        out_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()[0]
    )  # (32, 32, 3)
    print(f"      z range:       [{z.min().item():.4f}, {z.max().item():.4f}]")
    print(f"      out range:     [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"      expected_pixels: shape={expected_pixels.shape} dtype={expected_pixels.dtype}")
    print(f"      pixel range:   [{expected_pixels.min()}, {expected_pixels.max()}]")

    # Step 3: POST {n_samples: 1, seed: SEED} to the running service.
    print(f"\n[3/5] POST {SERVICE_URL}  payload=(n_samples=1, seed={SEED})")
    try:
        response = httpx.post(
            SERVICE_URL,
            json={"n_samples": 1, "seed": SEED},
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
    print(f"      generation_time_ms: {body['generation_time_ms']:.2f}")
    print(f"      seed echoed:        {body['seed']}")
    print(f"      images:             list of {len(body['images'])}")
    print(f"      image[0] base64:    {len(body['images'][0])} chars")

    if body["seed"] != SEED:
        print(f"      FAIL: seed not echoed correctly ({body['seed']} vs {SEED})")
        return 1

    # Step 4: decode + bit-compare pixels. PNG is lossless so the
    # round-trip via base64 must preserve every pixel exactly.
    print("\n[4/5] Decoding service response + bit-comparing pixels...")
    service_pixels = _decode_to_pixels(body["images"][0])
    print(f"      service_pixels: shape={service_pixels.shape} dtype={service_pixels.dtype}")

    if service_pixels.shape != expected_pixels.shape:
        print(f"      FAIL: shape mismatch "
              f"({service_pixels.shape} vs {expected_pixels.shape})")
        return 1

    diff = np.abs(
        service_pixels.astype(np.int32) - expected_pixels.astype(np.int32)
    )
    print(f"      max abs pixel diff:  {diff.max()}")
    print(f"      mean abs pixel diff: {diff.mean():.6f}")
    print(f"      n_pixels differing:  {int((diff > 0).sum())}")

    if diff.max() != 0:
        print(f"\n      FAIL: pixel mismatch (expected bit-exact, got max diff {diff.max()})")
        return 1

    # Step 5: bonus checks - n_samples=16 batch + repeat-call reproducibility.
    print(f"\n[5/5] Bonus checks: n_samples={N_SAMPLES_MAX} batch + reproducibility...")

    # 5a: n_samples=16 batch returns 16 valid 32x32 RGB images.
    response_batch = httpx.post(
        SERVICE_URL,
        json={"n_samples": N_SAMPLES_MAX, "seed": 0},
        timeout=TIMEOUT_SEC,
    )
    if response_batch.status_code != 200:
        print(f"      FAIL: n_samples=16 returned {response_batch.status_code}")
        return 1
    batch_body = response_batch.json()
    if len(batch_body["images"]) != N_SAMPLES_MAX:
        print(f"      FAIL: expected {N_SAMPLES_MAX} images, got {len(batch_body['images'])}")
        return 1
    for i, b64 in enumerate(batch_body["images"]):
        pixels = _decode_to_pixels(b64)
        if pixels.shape != (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS):
            print(f"      FAIL: batch image {i} has shape {pixels.shape}")
            return 1
    print(f"      OK: n_samples={N_SAMPLES_MAX} returned {N_SAMPLES_MAX} valid 32x32 RGB images "
          f"in {batch_body['generation_time_ms']:.1f} ms")

    # 5b: same seed twice -> identical base64 bytes (exercises seeding
    # under live HTTP across two separate request handlers).
    r1 = httpx.post(
        SERVICE_URL,
        json={"n_samples": 1, "seed": SEED},
        timeout=TIMEOUT_SEC,
    )
    r2 = httpx.post(
        SERVICE_URL,
        json={"n_samples": 1, "seed": SEED},
        timeout=TIMEOUT_SEC,
    )
    img1 = r1.json()["images"][0]
    img2 = r2.json()["images"][0]
    if img1 != img2:
        print("      FAIL: same-seed bytes differ across two calls "
              "(server-side seed plumbing broken under HTTP)")
        return 1
    print(f"      OK: same seed={SEED} produces byte-identical bytes across two calls")

    print("\n" + "=" * 70)
    print("  PASS - service-decoded pixels bit-match manual forward")
    print(f"  Seed:               {SEED}")
    print(f"  Image:              {service_pixels.shape} {service_pixels.dtype}, "
          f"range [{service_pixels.min()}, {service_pixels.max()}]")
    print(f"  Single-sample time: {body['generation_time_ms']:.2f} ms")
    print(f"  Batch-16 time:      {batch_body['generation_time_ms']:.2f} ms "
          f"({batch_body['generation_time_ms'] / N_SAMPLES_MAX:.2f} ms/img)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
