# sklearn-svc

FastAPI service for **D1 SK PCA** — projects a flat Fashion-MNIST image (784 pixels) through the registered scikit-learn PCA estimator to a 150-component vector that captures 90.85% of the input variance.

One service, one model, one prediction endpoint. The minimal example of the per-service deployment pattern used across the portfolio.

| | |
|---|---|
| **Port** | 8001 |
| **Base image** | `python:3.12.2-slim` (multi-stage) |
| **Registry alias** | `models:/sk-pca@production` (override with `MODEL_ALIAS`) |
| **Tests** | 49 (pytest); ruff + mypy strict clean |
| **Image size** | ~1.27 GB |

## Quick start

### Via docker compose (recommended)

From `deployment/`:

```powershell
docker compose up -d --build sklearn-svc
docker compose ps   # wait for "(healthy)"
```

Smoke-test from another terminal:

```powershell
$payload = @{ features = @(1..784 | ForEach-Object { 0.0 }) } | ConvertTo-Json -Compress
Invoke-WebRequest -UseBasicParsing `
  -Uri http://localhost:8001/predict/pca `
  -Method POST -ContentType 'application/json' -Body $payload
```

Tear down:

```powershell
docker compose down
```

### Standalone (host venv, no Docker)

From `deployment/services/sklearn-svc/`:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-windows.txt
.\.venv\Scripts\uvicorn.exe app.main:app --port 8001 --reload
```

The `requirements-windows.txt` lockfile resolves wheels for the host platform; the Docker build uses `requirements.txt` (Linux-resolved). See [`../../docs/dependency-strategy.md`](../../docs/dependency-strategy.md) for the two-lockfile rationale.

### Real-data smoke test

A committed Fashion-MNIST sample drives an end-to-end forward-vs-service parity check:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

The script standardizes a raw 784-pixel sample through the same `/255 -> StandardScaler -> PCA` pipeline the service runs, then compares against the service response. Match tolerance ~1e-3 (float32 precision).

## Endpoints

| Method | Path | Tag | Purpose |
|---|---|---|---|
| GET | `/health` | `health` | Liveness — `{"status": "alive"}`, never does work |
| GET | `/ready` | `health` | Readiness — 503 until PCA loaded; 200 with `model_name` + `model_version` once loaded |
| GET | `/health/pca` | `health` | Per-model freshness — 200 with last-inference age + threshold, 503 with `inference_stale` after the threshold |
| POST | `/predict/pca` | `pca` | Project a 784-pixel vector to 150 PCA components |
| GET | `/metrics` | `observability` | Prometheus text exposition (HTTP + per-model inference metrics) |
| GET | `/docs` | — | Swagger UI |
| GET | `/redoc` | — | ReDoc UI |
| GET | `/openapi.json` | — | Raw OpenAPI 3 schema |

Full request/response schemas, validation rules, and example payloads are in `/docs`. The `/predict/pca` request schema enforces `len(features) == 784` and individual values in float range; out-of-range pixels return 422 with a precise validation error.

## Environment variables

| Name | Default | What it controls |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:////srv/mlflow/mlflow.db` (Docker) / auto-derived (host) | Path/URI to the MLflow SQLite registry the loader reads |
| `MLFLOW_ARTIFACT_ROOT_OVERRIDE` | `/srv/mlflow` (Docker) / unset (host) | Prefix substitution for artifact URIs whose host-side absolute path doesn't exist inside the container. See [`../../docs/volume-mount-strategy.md`](../../docs/volume-mount-strategy.md) |
| `MODEL_ALIAS` | `production` | Which registry alias to resolve. `MlflowClient.get_model_version_by_alias("sk-pca", <alias>)` picks the model version |
| `INPUT_DISTRIBUTION_SAMPLE_EVERY` | `100` | Emit an `input_distribution_sample` structlog event every Nth `/predict/pca` request. `0` disables drift-signal logging entirely |
| `MODEL_INFERENCE_STALENESS_SECONDS` | `3600` | Freshness threshold for `/health/pca`. Returns 503 `inference_stale` once `now - _LAST_INFERENCE_TS` exceeds this many seconds |

`docker-compose.yml` sets the first two; the rest fall back to the defaults baked into the loader + middleware.

## Health + monitoring

Three health concerns, three distinct endpoints:

- **`/health`** — process is alive. Dumb on purpose; orchestrators restart on failure here.
- **`/ready`** — service can accept traffic. 503 during the ~1-2 s lifespan-load window; 200 once `pca_loader.is_loaded()` returns True.
- **`/health/pca`** — per-model freshness. Returns 200 with `last_inference_age_seconds` + `staleness_threshold_seconds` while inference is current; 503 with `{"detail": "inference_stale"}` after the threshold elapses with no requests.

`/metrics` exposes the four golden signals (`http_requests_total`, `http_request_duration_seconds`, `http_requests_in_flight`) plus a per-model latency histogram (`model_inference_duration_seconds{model_name="sk-pca"}`). The full metric catalog + sample PromQL queries are in [`../../docs/monitoring.md`](../../docs/monitoring.md).

Structured JSON logs go to stdout via structlog. Pipe through `jq` for ad-hoc inspection; an `input_distribution_sample` event surfaces every Nth request with summary stats (mean/std/min/max/L2/zero-fraction) of the input vector — the upstream signal a real drift pipeline would consume.

## Local development

From `deployment/services/sklearn-svc/`:

```powershell
.\.venv\Scripts\pip install -r requirements-dev.txt
.\.venv\Scripts\ruff check .
.\.venv\Scripts\mypy --strict app
.\.venv\Scripts\pytest -q
```

Test suite is hermetic — fakes the PCA loader and uses `structlog.testing.capture_logs()`; no MLflow registry or model artifacts required. Real-data parity is verified separately via `scripts/smoke_test.py` (needs the service running).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Container exits immediately with `RuntimeError: ... alias 'production' not found` | Registry SQLite or artifacts unreachable inside the container | Confirm `deployment/mlflow.db` + `deployment/mlruns/` exist on the host and that `docker-compose.yml`'s `.:/srv/mlflow:ro` mount is in place |
| `/ready` returns 503 with `model_not_loaded` past 5 s | Loader still resolving the alias — usually a stale registry, missing `mlflow.db`, or wrong `MLFLOW_TRACKING_URI` | `docker compose logs sklearn-svc` and look for the `pca_load_start` / `pca_alias_resolved` events; the chained error in the next event shows what failed |
| `/predict/pca` returns 422 `len(features) ... not equal to 784` | Wrong-shape input | The PCA was trained on flat 28×28 Fashion-MNIST; client must flatten the image to a 784-element vector before posting |
| `/health/pca` returns 503 `inference_stale` | No traffic in the last hour | Either expected (idle service) or set `MODEL_INFERENCE_STALENESS_SECONDS` higher / `0` to disable the check |
| `requirements-windows.txt` install fails on a wheel | Lockfile drift after upstream wheel re-publish | Regenerate via `pip-compile` per [`../../docs/dependency-strategy.md`](../../docs/dependency-strategy.md); never edit the lockfile by hand |

## Further reading

- [`../../README.md`](../../README.md) — deployment-phase overview
- [`../../docs/architecture.md`](../../docs/architecture.md) — design decisions across all 3 services
- [`../../docs/deployment-runbook.md`](../../docs/deployment-runbook.md) — clone-to-running step-by-step
- [`../../docs/adding-a-new-model.md`](../../docs/adding-a-new-model.md) — extension pattern for a 6th model
