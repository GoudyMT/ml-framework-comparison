# Adding a New Model

How to extend the deployment with a 6th model, using the patterns established by the existing 5. Read [`architecture.md`](architecture.md) first for the WHY; this doc covers the HOW.

**Two paths.** Adding a model to an **existing service** (shares a framework runtime with one of `sklearn-svc`, `pt-svc`, `tf-svc`) is the common case and gets the full walkthrough below. Adding a **new service** (a different framework — JAX, ONNX, etc.) is a bigger lift covered briefly at the end.

**Worked example.** Throughout this doc, the example model is **PT VAE** — a PyTorch Variational Autoencoder from modeling phase #19. It fits in `pt-svc` (shares the torch runtime) and adds a generative endpoint that complements the existing GAN. Replace `vae` / `pt-vae` with your model's identifiers throughout.

## Prerequisites

Before starting the deployment work, confirm:

1. **A trained artifact exists** — `.joblib`, `.pt`/`.pth`, `.npy`, `.h5`+`.model`, or whatever your framework expects. The modeling phase produced it.
2. **A registered model name + version** in the MLflow registry. If you haven't promoted yet, do that first (see [Step 8](#step-8--promote-to-the-registry)) and come back.
3. **A framework runtime decision** — does this model share a runtime with an existing service? If yes, follow the steps below. If no (e.g., JAX), see [Adding a New Service](#adding-a-new-service).
4. **A clear request + response shape** — what does the client send, what does the service return? Sketch the Pydantic schema before writing code.

## Step 1 — Pick the service + name

For PT VAE: target service `pt-svc` (torch runtime), model name `pt-vae` (matches the dash-lowercase pattern of `pt-dnn`, `pt-gan-dcgan`, `pt-qlearning-taxi`), endpoint path `/predict/vae/sample`.

The naming patterns in use:

| Aspect | Convention |
|---|---|
| `MODEL_NAME` constant | `<framework>-<model>` (lowercase, dash-separated): `pt-dnn`, `sk-pca`, `tf-transformer-translation` |
| Loader filename | `app/services/<model>_loader.py`: `vae_loader.py` |
| Router filename | `app/routers/<model>.py`: `vae.py` |
| Schema filename | `app/schemas/<model>.py`: `vae.py` |
| Endpoint path | `/predict/<model>` or `/predict/<model>/<subaction>` if multiple ops |
| Health endpoint | `/health/<model>` (short label, matches the router's purpose): `/health/vae` |

## Step 2 — Add the loader

`app/services/vae_loader.py`. Copy `gan_loader.py` (closest analog — generative, server-side noise input) and adapt the names + artifact-loading line.

Required exports:

```python
MODEL_NAME: str = "pt-vae"
ALIAS_OVERRIDE_ENV: str = "MODEL_ALIAS"
_DEFAULT_ALIAS: str = "production"
ARTIFACT_ROOT_OVERRIDE_ENV: str = "MLFLOW_ARTIFACT_ROOT_OVERRIDE"

_MODEL: VAEModel | None = None
_MODEL_VERSION: str | None = None
_LAST_INFERENCE_TS: float = 0.0   # Phase 9.4 - per-model freshness tracking

def load_vae_model() -> None: ...      # called from main.py lifespan
def get_vae_model() -> VAEModel: ...   # called from router; raises RuntimeError if not loaded
def get_model_version() -> str: ...
def is_loaded() -> bool: ...
def _resolve_alias() -> str: ...       # private helper, MODEL_ALIAS env or default
def _resolve_artifact_dir(source: str) -> Path: ...  # ARTIFACT_ROOT_OVERRIDE remap
```

`_LAST_INFERENCE_TS` is stamped at two points: at the end of `load_vae_model()` (so a just-loaded model has a recent timestamp; the freshness probe doesn't fire on a fresh boot) and inside the router's `track_inference()` context manager (so every successful inference resets it).

## Step 3 — Add the schemas

`app/schemas/vae.py`. Two Pydantic models — `VAERequest` and `VAEResponse`. Module-level constants for any input bounds:

```python
from pydantic import BaseModel, ConfigDict, Field

LATENT_DIM: int = 128                  # VAE latent vector size
N_SAMPLES_MIN: int = 1
N_SAMPLES_MAX: int = 16                # bounds response payload, same cap as GAN

class VAERequest(BaseModel):
    n_samples: int = Field(default=1, ge=N_SAMPLES_MIN, le=N_SAMPLES_MAX, ...)
    seed: int | None = Field(default=None, ...)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"n_samples": 1, "seed": 42},
                {"n_samples": 16, "seed": 0},
            ],
        },
    )

class VAEResponse(BaseModel):
    images: list[str] = Field(..., max_length=N_SAMPLES_MAX, ...)  # base64 PNGs
    generation_time_ms: float = Field(..., ge=0.0, ...)
    seed: int | None = Field(..., ...)

    model_config = ConfigDict(json_schema_extra={"examples": [...]})
```

Phase 10 lessons baked in:

- **Use `model_config.json_schema_extra.examples` for top-level Request/Response examples**, NOT per-field `examples=`. Per-field examples render redundantly alongside the model-level ones in Swagger UI.
- **Don't include `None` in `examples=` lists** — older Swagger UI versions render it as the literal string "None".
- **Constants at module level**, not inside the class — loaders + router + tests + smoke test can all import without re-deriving.

## Step 4 — Add the router

`app/routers/vae.py`. Copy `gan.py` and adapt. Required shape:

```python
from fastapi import APIRouter, HTTPException

from app.middleware.inference_tracking import track_inference
from app.schemas.vae import N_SAMPLES_MAX, VAERequest, VAEResponse
from app.services import vae_loader

router = APIRouter(prefix="/predict", tags=["vae"])

@router.post(
    "/vae/sample",
    response_model=VAEResponse,
    summary="Generate N=[1,16] images by sampling the VAE prior",
    description=(
        "Server-side draws N latent vectors from the VAE prior, decodes "
        "them through the trained decoder, and returns the rendered "
        "images as base64-encoded PNG strings."
    ),
    responses={
        200: {"description": "Generation succeeded; `images` length matches the request `n_samples`."},
        422: {"description": "Request validation failed (e.g., `n_samples` out of range)."},
        503: {"description": "Model not loaded yet; retry after `/ready` returns 200."},
    },
)
async def predict_vae_sample(request: VAERequest) -> VAEResponse:
    try:
        model = vae_loader.get_vae_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="model_not_loaded") from exc

    with track_inference(vae_loader):
        images, generation_time_ms = _sample_from_vae(model, request)

    return VAEResponse(images=images, generation_time_ms=generation_time_ms, seed=request.seed)
```

The `track_inference(vae_loader)` context manager from `app/middleware/inference_tracking.py` does three things at once: stamps `_LAST_INFERENCE_TS` on the loader, records to the `model_inference_duration_seconds` histogram, and re-raises on exception so the freshness stamp doesn't update on failed inferences.

## Step 5 — Wire up `main.py`

Three additions to `app/main.py`:

**5.1 Import the new router + loader.**

```python
from app.routers import vae as vae_router
from app.services import vae_loader
```

**5.2 Load it in the lifespan event.**

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    dnn_loader.load_dnn_model()
    gan_loader.load_gan_model()
    qtable_loader.load_qtable()
    vae_loader.load_vae_model()       # add this
    yield
```

**5.3 Add the openapi_tag + register the router.**

```python
openapi_tags=[
    {"name": "dnn", "description": "..."},
    {"name": "gan", "description": "..."},
    {"name": "qlearning", "description": "..."},
    {"name": "vae", "description": "Generative endpoint - VAE prior sampling, 32x32 RGB output."},  # add
    {"name": "health", "description": "..."},
    {"name": "observability", "description": "..."},
],

# After app construction:
app.include_router(vae_router.router)
```

**5.4 Include in `/ready`.**

```python
async def ready() -> dict[str, str | bool | dict[str, dict[str, str | bool]]]:
    if not (dnn_loader.is_loaded() and gan_loader.is_loaded()
            and qtable_loader.is_loaded() and vae_loader.is_loaded()):
        raise HTTPException(status_code=503, detail="not_all_models_loaded")

    return {
        "status": "ready",
        "models": {
            "pt-dnn": {"loaded": True, "version": dnn_loader.get_model_version()},
            "pt-gan-dcgan": {"loaded": True, "version": gan_loader.get_model_version()},
            "pt-qlearning-taxi": {"loaded": True, "version": qtable_loader.get_model_version()},
            "pt-vae": {"loaded": True, "version": vae_loader.get_model_version()},  # add
        },
    }
```

## Step 6 — Add `/health/<model>`

In `app/main.py`, add a new `/health/vae` endpoint matching the shape of the existing `/health/dnn`. The endpoint should call `inference_tracking._resolve_staleness_threshold()` for the timeout (operator-tunable via `MODEL_INFERENCE_STALENESS_SECONDS`) and return either:

- 200 with `{"status": "healthy", "model_name": "pt-vae", "version": ..., "last_inference_age_seconds": ..., "staleness_threshold_seconds": ...}`
- 503 with `{"detail": "model_not_loaded"}` or `{"detail": "inference_stale"}`

The exact endpoint body is ~30 lines; the three existing `/health/dnn` / `/health/gan` / `/health/qlearning` blocks in `main.py` are the templates. (If a `_make_health_route(loader, tag_label)` factory lands in a future hardening pass, the addition collapses to a single call — see the open question in the deployment-phase plan.)

## Step 7 — Add tests

Five test files touched. None of them load real artifacts.

**7.1 `tests/test_vae.py` (new file).** Endpoint tests using a `FakeVAE` fixture. ~10-15 tests covering valid input, validation errors (422), `/ready` integration, response shape, base64 validity if generating images.

**7.2 `tests/test_alias_resolution.py` (modify).** Parametrize over loaders to include `vae_loader`:

```python
@pytest.mark.parametrize("loader_module", [
    dnn_loader, gan_loader, qtable_loader, vae_loader,
])
def test_default_alias_when_env_unset(monkeypatch, loader_module): ...
```

**7.3 `tests/test_inference_tracking.py` (modify).** Add `vae_loader` to the loader parametrize list — `test_track_inference_updates_last_inference_ts` etc. cover the new loader for free.

**7.4 `tests/test_input_distribution.py` (modify; only if VAE takes numeric features).** PT VAE samples from the prior server-side (no user features), so skip this one — same as GAN.

**7.5 `tests/conftest.py` (modify).** Add a `FakeVAE` fixture that returns predetermined latent vectors → predetermined image bytes. Pattern matches the existing `FakeDCGenerator` fixture.

Run the suite:

```powershell
.\.venv\Scripts\pytest -q
```

All previously-passing tests should still pass; the new ones should pass too. Pre-existing test count was 99 for pt-svc; after PT VAE addition it lands at ~115-120 depending on how many endpoint cases you cover.

## Step 8 — Promote to the registry

```powershell
cd deployment
# Upload the artifact + register as a new model
python scripts/promote_to_registry.py --name pt-vae --run-id <mlflow-run-id>

# Set the production alias
python scripts/promote_to_registry.py --name pt-vae --alias production --version 1

# Verify
python scripts/promote_to_registry.py --list
```

The artifact path the registry records is your host-machine absolute path; inside the container, the loader's `_resolve_artifact_dir` helper substitutes `MLFLOW_ARTIFACT_ROOT_OVERRIDE` → `/srv/mlflow` to find it. No code change needed for portability — the helper is already there.

## Step 9 — Smoke test

Rebuild the container (or restart uvicorn on the host venv):

```powershell
# Docker path
cd deployment
docker compose up -d --build pt-svc

# Host path
cd deployment/services/pt-svc
.\.venv\Scripts\uvicorn.exe app.main:app --port 8002 --reload
```

Quick probe:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8002/ready | % Content
# Expect: "models" includes "pt-vae": {"loaded": true, "version": "1"}

Invoke-WebRequest -UseBasicParsing http://localhost:8002/predict/vae/sample `
  -Method POST -ContentType 'application/json' -Body '{"n_samples":1,"seed":42}' | % Content
# Expect: 200 with images: ["iVBORw0KGgo..."], generation_time_ms, seed: 42

Invoke-WebRequest -UseBasicParsing http://localhost:8002/health/vae | % Content
# Expect: 200 with last_inference_age_seconds + staleness_threshold_seconds
```

Optional: add `scripts/smoke_test_vae.py` for forward-vs-service parity (load the artifacts standalone, run a known seed through the decoder manually, compare against the service response).

## Step 10 — Polish

Before committing, run the polish standard:

```powershell
.\.venv\Scripts\ruff check .
.\.venv\Scripts\mypy --strict app
.\.venv\Scripts\pytest -q
```

All three must pass. Then do a fresh-eyes polish review on the new files (loader + schema + router + main.py edits + tests) — read each file as a stranger would, checking that descriptions match what the code actually does, that schemas don't duplicate metadata across model_config + Field-level surfaces, and that the endpoint summaries in `/docs` read cleanly to someone who has never seen the model before.

## Adding a New Service

If the model's framework doesn't fit any of the existing services (e.g., JAX, ONNX runtime, a custom inference engine), you're forking the layout. High-level steps:

1. **Fork `sklearn-svc` as the template** — smallest existing service, easiest to read and adapt.
2. **Pick a port** (currently 8001/8002/8003 are taken; 8004 is the natural next).
3. **Update `pyproject.toml`** with the new framework's pin + regenerate both lockfiles (Step 5.4 of the runbook).
4. **Rewrite the loader** for the new framework's artifact format.
5. **Update the Dockerfile** if the new framework needs system packages (`apt-get install` in the builder stage).
6. **Add the service to `docker-compose.yml`** with its own port mapping + the same MLflow registry bind-mount.
7. **Add a new matrix entry** to `.github/workflows/ci.yml` and `.github/workflows/build.yml`.
8. **Write a per-service README** matching the existing 3 (template at any of the per-service READMEs).
9. **Update [`../README.md`](../README.md)** + this doc + [`architecture.md`](architecture.md) to reflect the new service in the topology.

The 3-service pattern was a portfolio scope decision, not a hard limit — adding a 4th service is straightforward.

## Common Pitfalls

| Pitfall | How to avoid |
|---|---|
| Forgetting to wire the loader into `lifespan` | `/ready` will still return 200 even though the model isn't loaded. **Fix**: every loader must be called from `lifespan` AND its `is_loaded()` must be in the `/ready` check |
| Forgetting to add the openapi_tag | The router still works but Swagger UI shows the endpoint under "default" instead of the per-model heading. **Fix**: add the tag to the `openapi_tags=` list in main.py before the router uses it |
| Skipping the `track_inference()` wrap | `model_inference_duration_seconds_count` won't increment for the new model, AND `/health/<model>` will return `inference_stale` after the threshold. **Fix**: every successful inference must run inside `with track_inference(loader): ...` |
| Per-field `examples=` on the schema | Renders redundantly in Swagger UI alongside the model_config examples. **Fix**: use `model_config.json_schema_extra.examples` as the single source of truth |
| Hand-editing the requirements lockfiles | Hashes go stale; `--require-hashes` install fails | Regenerate via [Step 5.4 of the runbook](deployment-runbook.md#54-regenerate-the-requirements-lockfiles) |
| Forgetting `MODEL_ALIAS` plumbing | The loader hardcodes `"production"`; operators can't roll a candidate without a code change. **Fix**: every loader must read `_resolve_alias()` and pass the resolved value into `MlflowClient.get_model_version_by_alias` |

## Further Reading

- [`architecture.md`](architecture.md) — design decisions across all services
- [`deployment-runbook.md`](deployment-runbook.md) — operations + verification + failure modes
- [`registry-strategy.md`](registry-strategy.md) — MLflow registry contract + alias workflow
- [`monitoring.md`](monitoring.md) — what `/metrics` exposes + per-model histogram cardinality
- Per-service READMEs at `deployment/services/{sklearn-svc,pt-svc,tf-svc}/README.md` — concrete examples of the patterns in use
