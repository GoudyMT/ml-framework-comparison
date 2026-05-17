"""
Router for the /predict/gan/sample endpoint.

WHAT THIS FILE IS:
    The HTTP-facing layer for DCGAN image generation. One POST
    handler:
        1. Receives a validated GANRequest (Pydantic enforced
           n_samples in [1, 16] + optional seed before this code runs)
        2. Optionally seeds torch's RNG for reproducibility
        3. Samples latent noise z ~ N(0, 1) of shape (n, 100, 1, 1)
        4. Runs the cached DCGenerator under torch.no_grad()
        5. Denormalizes the (n, 3, 32, 32) output from [-1, 1] to
           uint8 [0, 255]
        6. Encodes each image as PNG bytes -> base64 string
        7. Builds a GANResponse with the list of base64 images,
           generation time, and the echoed seed

    Routers stay THIN: the model lifecycle, registry resolution, and
    architecture all live in app/services/. This file is purely the
    HTTP <-> Python objects boundary plus the data-shape conversions
    needed at that boundary (noise -> tensor -> ndarray -> PIL ->
    PNG bytes -> base64 string).

URL SHAPE:
    `prefix="/predict"` on the router + `@router.post("/gan/sample")`
    on the handler -> final URL is `POST /predict/gan/sample`. The
    "/sample" sub-path is explicit (rather than just "/gan") so future
    GAN endpoints (latent interpolation, conditional generation, etc.)
    fit cleanly under the same `/gan/` namespace without renaming this
    one.

KEY CONCEPTS:

    1. SERVER-SIDE NOISE SAMPLING:
        The handler GENERATES its own input rather than receiving it
        from the client: a noise vector z drawn from N(0, 1). The
        client controls only how many to draw and (optionally) a seed
        for reproducibility. Why server-side: "POST {} -> get an
        image" stays trivial for callers. Pushing latent-vector
        construction onto the caller would add complexity for zero
        educational gain.

    2. torch.manual_seed FOR REPRODUCIBILITY:
        torch.randn() draws from torch's global RNG. Calling
        torch.manual_seed(N) BEFORE torch.randn() makes the noise
        deterministic - same seed -> same z -> same image bytes (down
        to the byte). This is a TESTABLE property (the test suite
        verifies it via decode-and-compare). Note: this seeds ONLY
        torch's RNG, not numpy or Python's random module - we don't
        use those here, so torch alone is sufficient.

    3. TENSOR DENORMALIZATION:
        The generator's tanh output is in [-1, 1]. To get displayable
        pixels we need [0, 255] uint8. The inverse of the training-
        time `pixel/127.5 - 1.0` is:
            (out + 1.0) * 127.5  -> [0, 255]
        We additionally `.clamp(0, 255)` defensively: tanh is
        mathematically bounded, but float-precision artifacts could
        produce 255.0000001 or -0.0000001 at the edges, and `.to(uint8)`
        on out-of-range floats yields wraparound garbage. Then
        `.to(torch.uint8)` truncates to integer.

    4. NCHW -> HWC LAYOUT FLIP:
        PyTorch conv layers store images as (N, C, H, W) - channels
        first (the "PT" convention). PIL's Image.fromarray() and most
        image libraries expect (H, W, C) - channels last (the "TF"/
        OpenCV convention). `.permute(0, 2, 3, 1)` swaps the axes
        without copying data; the resulting tensor is contiguous after
        a `.contiguous()` call (PIL needs contiguous numpy memory).

    5. JSON-INCOMPATIBLE BINARY -> base64 STRING:
        JSON has no native binary type. PNG bytes contain arbitrary
        non-ASCII octets that would break a JSON parser. Base64
        encoding maps every 3 bytes to 4 ASCII characters from the
        safe alphabet [A-Za-z0-9+/], so the entire PNG fits cleanly
        inside a JSON string field. The ~33% size overhead is
        acceptable because individual 32x32 RGB PNGs are ~3 KB.

DEFENSE IN DEPTH:
    The lifespan event guarantees the model is loaded before any
    request. We still defensively catch RuntimeError from
    get_gan_model() and return 503 - same shape /ready uses for the
    same condition. "Should never fail" eventually does, at 3am.
"""

import base64
import io
import time

import torch
from fastapi import APIRouter, HTTPException
from PIL import Image

from app.middleware.inference_tracking import track_inference
from app.schemas.gan import LATENT_DIM, GANRequest, GANResponse
from app.services import gan_loader

router = APIRouter(prefix="/predict", tags=["gan"])


@router.post(
    "/gan/sample",
    response_model=GANResponse,
    summary="Generate N CIFAR-10-style 32x32 RGB images via DCGAN",
)
async def predict_gan_sample(req: GANRequest) -> GANResponse:
    """
    Sample N images from the registered DCGAN generator.

    Args:
        req: A GANRequest carrying `n_samples` (1-16) and optional
            `seed`. FastAPI parses + validates the JSON body via
            Pydantic BEFORE this function runs; out-of-range
            n_samples never reaches us (HTTP 422 returned by FastAPI
            instead).

    Returns:
        GANResponse with `images` (list of base64-encoded 32x32 RGB
        PNGs, length matches n_samples), `generation_time_ms` (wall-
        clock time for forward + denorm + PNG encode), and `seed`
        (the value the client sent, echoed back).

    Raises:
        HTTPException 503: If the DCGAN isn't loaded (extremely
            unlikely given the lifespan event; defense-in-depth so
            partial-load states surface cleanly).
    """
    # Defensive accessor pull. RuntimeError -> 503 mirrors the contract
    # /ready uses for the same "service not yet ready" state.
    try:
        model = gan_loader.get_gan_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="model_not_loaded") from exc

    # Step 1: optionally seed torch's RNG for reproducible sampling.
    # Must happen BEFORE the torch.randn call. If req.seed is None we
    # leave the RNG state alone, so output is fresh random each call.
    if req.seed is not None:
        torch.manual_seed(req.seed)

    # Step 2: sample latent noise z ~ N(0, 1).
    # Shape (n_samples, LATENT_DIM=100, 1, 1) matches DCGenerator's
    # input contract. dtype=float32 matches the generator's weight
    # dtype - any other dtype would force an upcast inside the forward
    # pass and silently double memory.
    z = torch.randn(req.n_samples, LATENT_DIM, 1, 1, dtype=torch.float32)

    # Time the work: forward + denormalization + PNG encoding. We start
    # the clock here (after seed setup but before the forward) so the
    # measurement matches what the GANResponse.generation_time_ms field
    # documents.
    t0 = time.perf_counter()

    # Step 3: forward pass under no_grad().
    """
    eval() mode was already applied by the loader (BatchNorm uses
    running stats, not batch stats - critical for single-sample-ish
    inference). no_grad disables autograd's computation-graph
    tracking - no .backward() will ever run on this output, so the
    bookkeeping is wasted compute and memory.

    track_inference wraps two coupled side effects: observe the
    model_inference_duration_seconds histogram (measures only the
    generator forward; narrower than the existing t0 scope, which also
    covers denormalization + PIL/PNG/base64 encoding) AND stamp
    gan_loader._LAST_INFERENCE_TS on successful exit (powers the
    /health/gan freshness check).
    """
    with track_inference(gan_loader), torch.no_grad():
        out = model(z)  # shape (n, 3, 32, 32) in [-1, 1]

    # Step 4: denormalize [-1, 1] float -> [0, 255] uint8.
    """
    Inverse of the training-time `pixel/127.5 - 1.0` normalization.
    The constants live here (not in schemas/gan.py) because they
    describe the math, not the API contract. See
    `data/processed/gans/preprocessing_info.json` for source-of-
    truth on the [-1, 1] training range.
      .clamp(0, 255): defensive against float-precision artifacts at
        the boundaries (mathematically bounded, but 255.0000001
        truncates to 255 unsafely otherwise)
      .to(torch.uint8): converts to the dtype PIL expects for RGB.
      """
    images_uint8 = ((out + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)

    # Step 5: NCHW -> NHWC for PIL, then to numpy.
    """
    PyTorch stores images channels-first (PT convention); PIL wants
    channels-last (HxWxC). permute reorders axes without copying;
    .contiguous() makes the result memory-contiguous (numpy view of
    a non-contiguous tensor would be wrong shape). .cpu() is a no-op
    on CPU-only deployments but keeps the pattern correct if we ever
    move to GPU.
    """
    images_np = images_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    # Step 6: per-image PNG encode + base64.
    """
    The pattern for one image:
      numpy(H, W, 3) uint8 -> PIL.Image -> BytesIO PNG -> base64 str
    No data:image/png;base64, URI prefix - we ship raw base64, the
    client adds the prefix if it needs to embed in HTML.
    """
    images_b64: list[str] = []
    for arr in images_np:
        # Image.fromarray reads numpy directly without an extra copy
        # when arr is contiguous + uint8 + (H, W, 3). mode="RGB" is
        # explicit even though it's inferred from the (H, W, 3) shape.
        img = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        # PNG is lossless - critical for reproducibility tests where
        # the same seed must produce identical bytes. JPEG would re-
        # introduce non-determinism via lossy compression.
        img.save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    # Stop the clock. Multiply by 1000 to convert seconds -> ms.
    generation_time_ms = (time.perf_counter() - t0) * 1000.0

    # response_model=GANResponse on the decorator means FastAPI
    # re-validates this object before serializing. The defensive
    # base64 validator on GANResponse.images runs here too - catches
    # any handler bug that produced a malformed string.
    return GANResponse(
        images=images_b64,
        generation_time_ms=generation_time_ms,
        seed=req.seed,
    )
