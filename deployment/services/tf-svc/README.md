# tf-svc

FastAPI service for **D5 TF Transformer Translation** — English → Spanish via an encoder-decoder Transformer (3+3 layers, d_model=256, 11.68M params) trained on the Tatoeba EN-ES pairs with a shared 8,000-token SentencePiece BPE vocabulary. BLEU 0.4456 on the test set (modeling phase #16).

One service, one model, one endpoint — but two artifacts. The Transformer weights live in a `.weights.h5` file; SentencePiece's BPE tokenizer is a separate `.model` file. The service is only useful with both, so the loader resolves both at startup and gates `/ready` on both being present.

| | |
|---|---|
| **Port** | 8003 |
| **Base image** | `python:3.12.2-slim` (multi-stage) |
| **Registry alias** | `models:/tf-transformer-translation@production` (override with `MODEL_ALIAS`) |
| **Tests** | 47 (pytest); ruff + mypy strict clean |
| **Image size** | ~4.05 GB (dominated by TF 2.21 CPU wheel + dependencies) |

## Quick start

### Via docker compose (recommended)

From `deployment/`:

```powershell
docker compose up -d --build tf-svc
docker compose ps   # wait for "(healthy)" - first boot includes the oneDNN warm-up
```

Smoke-test from another terminal:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8003/translate `
  -Method POST -ContentType 'application/json' `
  -Body '{"text":"Hello, how are you?"}'
```

The response body is UTF-8 JSON; PowerShell's `$response.Content` decodes it as Latin-1, which misrenders Spanish accents on display. For a faithful view:

```powershell
$r = Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8003/translate `
  -Method POST -ContentType 'application/json' `
  -Body '{"text":"Hello, how are you?"}'
[System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray())
```

Tear down:

```powershell
docker compose down
```

### Standalone (host venv, no Docker)

From `deployment/services/tf-svc/`:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-windows.txt
.\.venv\Scripts\uvicorn.exe app.main:app --port 8003 --reload
```

Two-lockfile pattern per [`../../docs/dependency-strategy.md`](../../docs/dependency-strategy.md) — `requirements.txt` (Linux) drives Docker, `requirements-windows.txt` is the host-dev counterpart.

### Real-data smoke test

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

Verifies 5/5 sentences match string-exact between a manual forward run (Transformer + SentencePiece tokenizer loaded standalone) and the service's `/translate` response.

## Endpoints

| Method | Path | Tag | Purpose |
|---|---|---|---|
| GET | `/health` | `health` | Liveness — `{"status": "alive"}`, never does work |
| GET | `/ready` | `health` | Readiness — 503 until BOTH weights and tokenizer loaded; 200 with `{loaded, model_name, model_version}` once ready |
| GET | `/health/translation` | `health` | Per-model freshness — 200 with last-inference age, 503 `inference_stale` after threshold |
| POST | `/translate` | `translation` | Translate one English sentence (1-1000 chars) to Spanish |
| GET | `/metrics` | `observability` | Prometheus text exposition (HTTP + per-model inference metrics) |
| GET | `/docs` | — | Swagger UI |
| GET | `/redoc` | — | ReDoc UI |
| GET | `/openapi.json` | — | Raw OpenAPI 3 schema |

The endpoint is `/translate`, not `/predict/translate` — translation is the verb, "predict" misframes it. Other than the path naming, the contract matches the sklearn/pt services. Request schema accepts `text` (1-1000 chars) + optional `max_length` capped at 25 BPE tokens (matches the training-time positional encoding range; longer would generate noise). Out-of-range inputs return 422 with precise validation errors.

## Environment variables

| Name | Default | What it controls |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:////srv/mlflow/mlflow.db` (Docker) / auto-derived (host) | Path/URI to the MLflow SQLite registry |
| `MLFLOW_ARTIFACT_ROOT_OVERRIDE` | `/srv/mlflow` (Docker) / unset (host) | Prefix substitution for artifact URIs whose host-side absolute path doesn't exist inside the container. See [`../../docs/volume-mount-strategy.md`](../../docs/volume-mount-strategy.md) |
| `MODEL_ALIAS` | `production` | Which registry alias to resolve. Applies to the parent model directory; tokenizer + weights resolve from the same `version.source` |
| `INPUT_DISTRIBUTION_SAMPLE_EVERY` | unused | tf-svc has no numeric feature vector to summarize — `/translate` takes text, which is a different signal. The env var is read but no drift events are emitted from this service |
| `MODEL_INFERENCE_STALENESS_SECONDS` | `3600` | Freshness threshold for `/health/translation` |

## Health + monitoring

`/ready` requires BOTH artifacts loaded — half-loaded (e.g., weights present, tokenizer missing) is unusable, so the service stays 503 until both succeed. Per-model `/health/translation` adds the freshness layer on top.

`/metrics` exposes `model_inference_duration_seconds{model_name="tf-transformer-translation"}`. The histogram wraps the FULL greedy-decode loop (one encoder pass + N decoder passes for N output tokens), so a slow histogram often means a slow decode loop rather than a slow handler. Full catalog + PromQL in [`../../docs/monitoring.md`](../../docs/monitoring.md).

The first inference after a fresh load takes 5-10 s for TensorFlow's lazy oneDNN op compilation. The lifespan loader runs a warm-up forward pass to eat that cost at startup — `/ready` doesn't return 200 until warm-up completes, so the first user request hits normal latency, not the cold-start tax.

## Local development

From `deployment/services/tf-svc/`:

```powershell
.\.venv\Scripts\pip install -r requirements-dev.txt
.\.venv\Scripts\ruff check .
.\.venv\Scripts\mypy --strict app
.\.venv\Scripts\pytest -q
```

Test suite is hermetic — `FakeTransformer` + `FakeTokenizer` fixtures drive deterministic paths through the decode loop (predetermined token sequence + BOS/EOS handling) without loading real `.h5` or `.model` files. Two pytest `filterwarnings` entries in `pyproject.toml` suppress upstream sentencepiece SWIG `DeprecationWarning`s that are emitted unconditionally on import.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Container takes 30-60 s to reach `(healthy)` on first boot | Warm-up forward pass + first-time TF oneDNN op compile | Expected. Subsequent boots reuse the cached image and are faster (~10-15 s) since op compile is already done. `docker compose logs tf-svc` and look for `translation_warmup_start` → `translation_warmup_complete` |
| Container exits with `RuntimeError: ... alias 'production' not found` | Registry missing the alias or wrong `MLFLOW_TRACKING_URI` | List aliases via `python ../../scripts/promote_to_registry.py --list` (from `deployment/`); promote/move with `--name tf-transformer-translation --alias production --version <n>` |
| Container exits with `RuntimeError: tokenizer artifact not found alongside weights` | Promotion uploaded `.h5` but not `.model` (or vice versa) | Both files must live in the same `version.source` directory. Re-promote with `python scripts/promote_to_registry.py --name tf-transformer-translation` ensuring both files are passed |
| Spanish accents render as `Ã³`, `Â¿` etc. in PowerShell `$r.Content` | UTF-8 → Latin-1 decoding by PowerShell on `Invoke-WebRequest` | Use the explicit decode shown in Quick start: `[System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray())`. The wire bytes are correct; this is purely a display issue |
| Translation output looks low-quality | Model is BLEU 0.4456 on the test set; real outputs are often imperfect | Expected behavior for this model. Idealized examples in the schema docs reflect best-case output, not every actual decode; see `app/schemas/translation.py` docstrings for the model's expected character |
| `/translate` returns 422 `max_length ... not in [1, 25]` | Client requested a longer generation | The 25-token cap matches the training-time positional encoding range (sinusoidal PE only saw positions 0-24); generating beyond that produces noise. Split longer translations client-side |

## Further reading

- [`../../README.md`](../../README.md) — deployment-phase overview
- [`../../docs/architecture.md`](../../docs/architecture.md) — design decisions across all 3 services
- [`../../docs/deployment-runbook.md`](../../docs/deployment-runbook.md) — clone-to-running step-by-step
- [`../../docs/adding-a-new-model.md`](../../docs/adding-a-new-model.md) — extension pattern for a 6th model
