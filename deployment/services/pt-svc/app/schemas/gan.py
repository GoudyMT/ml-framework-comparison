"""
Pydantic schemas for the /predict/gan/sample endpoint.

WHAT THIS FILE IS:
    The API contract for the PyTorch DCGAN endpoint (CIFAR-10 image
    generation).
    Two classes:
        - GANRequest:  what the client MUST send (validated on the way in)
        - GANResponse: what the service WILL return (validated on the way out)

    FastAPI introspects these to validate incoming JSON, generate the
    OpenAPI schema for /docs, and serialize handler return values.

THE TRAINING PIPELINE THIS CONTRACT REFLECTS:
    z ~ N(0, 1)  shape (n_samples, 100, 1, 1)
        -> DCGenerator (4 transposed convs + BN + ReLU; tanh output)
        -> tensor in [-1, 1] shape (n_samples, 3, 32, 32)
        -> denormalize:  (x + 1) * 127.5  -> uint8 [0, 255]
        -> PIL Image.fromarray  -> PNG bytes  -> base64 string

    The *input* to the model is server-side noise, not user-provided
    data. The client controls only:
        - how many samples to draw (n_samples)
        - whether to seed the RNG for reproducibility (seed)

WHY BASE64 PNG INSTEAD OF RAW TENSORS:
    JSON has no native binary type. Three real options exist:
      1. Raw float arrays (precise but ~12 KB/image of float text;
         client has to denormalize themselves; defeats the "service
         owns preprocessing" rule).
      2. Streaming binary `Response(media_type="image/png")` (smallest
         payload but only works for n_samples=1; can't ship metadata
         like generation_time_ms or seed alongside the image).
      3. Base64-encoded PNG inside JSON (one response carries N images
         + metadata; client trivially renders via
         `<img src="data:image/png;base64,...">`; lossless for 32x32).

    Option 3 is the standard ML-serving pattern for small images.
    The ~33% size overhead from base64 is acceptable because the PNGs
    themselves are tiny (~3 KB each at 32x32 RGB).

DESIGN DECISIONS:
    - n_samples STRICTLY bounded to [1, 16]. Caps response payload at
      ~48 KB (16 PNGs * ~3 KB) so callers can't accidentally request
      thousands of samples and OOM the service.
    - seed is optional. Passing seed=42 produces deterministic output
      across calls (testable). Omitting it produces fresh random each
      call.
    - The seed is ECHOED in the response - if the client passed 42 we
      return 42; if they passed None we return None. We do NOT
      auto-generate a seed server-side when None is sent (that would
      misleadingly suggest the response could be reproduced when in
      fact the RNG state was inherited from prior calls).
    - The image format (PNG, 32x32 RGB) is fixed by the service
      contract; not a request field. Documented in docstrings, not
      duplicated as a configurable parameter.
"""

import base64

from pydantic import BaseModel, Field, field_validator

# Module constants
# ---------------------------------------------------------------------------
# Pulled from the modeling phase: PyTorch/14-gans/pipeline.ipynb
# and data/processed/gans/preprocessing_info.json.

# At module level (not inside a class) so the loader, the router, the
# tests, and the smoke test can all import them without re-deriving.

LATENT_DIM: int = 100                   # z-vector size; first ConvTranspose2d
                                        # input channels in DCGenerator.main.0
IMAGE_SIZE: int = 32                    # CIFAR-10 spatial dimension
IMAGE_CHANNELS: int = 3                 # CIFAR-10 RGB

N_SAMPLES_MIN: int = 1                  # caller must request at least 1
N_SAMPLES_MAX: int = 16                 # response payload cap (~48 KB total)

# DCGAN training used tanh output in [-1, 1]; the inverse is
# (x + 1) * 127.5 to recover [0, 255] uint8. These constants live in
# the router (where the denormalization happens), not here, because
# the schema describes the API contract not the math.


# Request schema


class GANRequest(BaseModel):
    """
    Client payload for POST /predict/gan/sample.

    Attributes:
        n_samples: How many CIFAR-10-style images to generate. The
            server samples that many independent z vectors from N(0, 1)
            and runs them through the generator in a single batched
            forward pass.
        seed: Optional RNG seed. If provided, `torch.manual_seed(seed)`
            is called before sampling, making the output deterministic
            across calls (same seed -> same images). If omitted, the
            server uses whatever RNG state is current.

    Example payloads:
        {"n_samples": 4}                    # 4 fresh random images
        {"n_samples": 1, "seed": 42}        # 1 deterministic image
        {"n_samples": 16, "seed": 0}        # 16 reproducible images
    """

    n_samples: int = Field(
        default=1,
        ge=N_SAMPLES_MIN,
        le=N_SAMPLES_MAX,
        description=(
            f"Number of images to generate, in "
            f"[{N_SAMPLES_MIN}, {N_SAMPLES_MAX}]. The cap bounds the "
            "response payload (each image is ~3 KB as base64 PNG)."
        ),
        examples=[1, 4, 16],
    )

    seed: int | None = Field(
        default=None,
        description=(
            "Optional RNG seed for reproducible sampling. If provided, "
            "the server calls torch.manual_seed(seed) before drawing "
            "the noise vectors, so the same seed produces the same "
            "images. If omitted (None), output is fresh random each call."
        ),
        examples=[None, 42, 113],
    )


# Response schema


class GANResponse(BaseModel):
    """
    Service payload returned from POST /predict/gan/sample.

    Attributes:
        images: A list of base64-encoded PNG strings, length matches
            the request's n_samples. Each PNG is a 32x32 RGB image.
            To render in HTML: `<img src="data:image/png;base64,{s}">`.
            To decode in Python: `PIL.Image.open(io.BytesIO(
                base64.b64decode(s)))`.
        generation_time_ms: Wall-clock time the server spent on the
            forward pass + denormalization + PNG encoding, in
            milliseconds. Useful for benchmarking without instrumenting
            the client.
        seed: The seed value the server used. Echoes back the request's
            seed - if the client passed 42, this is 42; if they passed
            None, this is None. Lets clients confirm reproducibility
            without having to remember what they sent.

    Example payload (truncated):
        {
          "images": ["iVBORw0KGgoAAAANSUhEUgAAACAAA...", "..."],
          "generation_time_ms": 18.4,
          "seed": 42
        }
    """

    images: list[str] = Field(
        ...,
        max_length=N_SAMPLES_MAX,
        description=(
            f"Base64-encoded PNG strings, length matches request "
            f"n_samples (max {N_SAMPLES_MAX}). Each image is 32x32 RGB. "
            "No 'data:image/png;base64,' URI prefix is included - "
            "raw base64 only; clients add the prefix if they need it."
        ),
    )

    generation_time_ms: float = Field(
        ...,
        ge=0.0,
        description=(
            "Wall-clock time spent on forward pass + denormalization + "
            "PNG encoding, in milliseconds. Excludes JSON serialization "
            "and network transit."
        ),
    )

    seed: int | None = Field(
        ...,
        description=(
            "Echo of the request's seed field. Same value the client "
            "sent (or null if no seed was provided). Lets the client "
            "confirm what was used for reproducibility."
        ),
    )

    @field_validator("images")
    @classmethod
    def _images_are_valid_base64(cls, value: list[str]) -> list[str]:
        """
        Each image string must be non-empty and decode as valid base64.

        Pydantic's Field doesn't validate string contents (only length
        of the list). This catches handler bugs - e.g., accidentally
        returning raw bytes, an empty list element, or a malformed
        base64 string. Defensive: we'd rather reject our own malformed
        response than ship corrupt data to clients.

        Cost: O(N) base64 parse over at most 16 strings of ~4 KB each;
        ~negligible compared to the ~20 ms generator forward pass.
        """
        for i, s in enumerate(value):
            if not s:
                raise ValueError(
                    f"images[{i}] is an empty string. The handler must "
                    "produce a base64-encoded PNG for every requested "
                    "sample."
                )
            try:
                # validate=True rejects non-base64 characters; without
                # it, b64decode silently strips garbage and returns a
                # truncated payload. b64decode raises binascii.Error on
                # invalid input, which is a subclass of ValueError in
                # Python 3 - so a single except clause covers both.
                base64.b64decode(s, validate=True)
            except ValueError as exc:
                raise ValueError(
                    f"images[{i}] is not valid base64: {exc}. This "
                    "indicates a handler bug - the encoding step "
                    "should always produce a base64-decodable string."
                ) from exc
        return value
