# pt-svc

FastAPI service for three PyTorch models, all served by one process:

- **D2 PT DNN** — 96.03% test accuracy on UCI HAR; 561-feature accelerometer/gyroscope sample → one of 6 activity classes
- **D3 PT DCGAN** — FID 30.57 on CIFAR-10; generates 32×32 RGB images from server-side noise (no user pixels in)
- **D4 PT Q-Learning Taxi-v4** — tabular Q-table parity to the modeling-phase Gymnasium run; integer state → action + full 6-element Q-vector

Three models in one service because they share a runtime (`torch==2.5.1` CPU build), making a single image cheaper than three.

| | |
|---|---|
| **Port** | 8002 |
| **Base image** | `python:3.12.2-slim` (multi-stage) |
| **Registry aliases** | `models:/pt-dnn@production`, `models:/pt-gan-dcgan@production`, `models:/pt-qlearning-taxi@production` (override with `MODEL_ALIAS`; same alias applies to all three) |
| **Tests** | 99 (pytest); ruff + mypy strict clean |
| **Image size** | ~2.33 GB (dominated by torch CPU wheel) |

## Quick start

### Via docker compose (recommended)

From `deployment/`:

```powershell
docker compose up -d --build pt-svc
docker compose ps   # wait for "(healthy)"
```

Smoke-test from another terminal:

```powershell
# DNN: 561-feature vector (zeros for a quick uptime probe)
$dnn = @{ features = @(1..561 | ForEach-Object { 0.0 }) } | ConvertTo-Json -Compress
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8002/predict/dnn `
  -Method POST -ContentType 'application/json' -Body $dnn

# GAN: 1 deterministic image
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8002/predict/gan/sample `
  -Method POST -ContentType 'application/json' -Body '{"n_samples":1,"seed":42}'

# Q-Learning: mid-episode Taxi-v4 state
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8002/predict/qlearning/taxi `
  -Method POST -ContentType 'application/json' -Body '{"state":328}'
```

Tear down:

```powershell
docker compose down
```

### Standalone (host venv, no Docker)

From `deployment/services/pt-svc/`:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-windows.txt
.\.venv\Scripts\uvicorn.exe app.main:app --port 8002 --reload
```

Two-lockfile pattern per [`../../docs/dependency-strategy.md`](../../docs/dependency-strategy.md) — `requirements.txt` (Linux) drives the Docker build; `requirements-windows.txt` is the host-dev counterpart.

### Real-data smoke tests

Three scripts, one per model, all start from raw inputs and run forward-vs-service parity checks:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py            # DNN: real UCI HAR sample
.\.venv\Scripts\python.exe scripts/smoke_test_gan.py        # DCGAN: bit-exact pixel parity (0 differing pixels)
.\.venv\Scripts\python.exe scripts/smoke_test_qlearning.py  # Q-Learning: 0 action disagreements across 15 states
```

Each script loads the same artifacts the service loaded, runs them forward independently, and confirms the service response matches.

## Endpoints

| Method | Path | Tag | Purpose |
|---|---|---|---|
| GET | `/health` | `health` | Liveness — `{"status": "alive"}`, never does work |
| GET | `/ready` | `health` | Readiness — 503 until ALL THREE models loaded; 200 with per-model `loaded` + `version` dict once ready |
| GET | `/health/dnn` | `health` | Per-model freshness for the DNN |
| GET | `/health/gan` | `health` | Per-model freshness for the DCGAN |
| GET | `/health/qlearning` | `health` | Per-model freshness for the Q-table |
| POST | `/predict/dnn` | `dnn` | Classify a 561-feature UCI HAR sample into one of 6 activities |
| POST | `/predict/gan/sample` | `gan` | Generate N=[1,16] CIFAR-10-style images as base64 PNGs |
| POST | `/predict/qlearning/taxi` | `qlearning` | Look up the policy action + Q-vector for a Taxi-v4 state in [0, 500) |
| GET | `/metrics` | `observability` | Prometheus text exposition (HTTP + per-model inference metrics) |
| GET | `/docs` | — | Swagger UI (grouped by per-model tag) |
| GET | `/redoc` | — | ReDoc UI |
| GET | `/openapi.json` | — | Raw OpenAPI 3 schema |

The `/ready` response carries a dict of all 3 models so a single probe answers "is the whole service ready" without per-model round-trips. Full request/response schemas + validation rules are in `/docs`; common 422 cases include `features.length != 561` (DNN), `n_samples not in [1,16]` (GAN), and `state not in [0,500)` (Q-Learning).

## Environment variables

Same five as sklearn-svc; `MODEL_ALIAS` applies to all 3 models (the registry has one alias per model, all pointing at the production version).

| Name | Default | What it controls |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:////srv/mlflow/mlflow.db` (Docker) / auto-derived (host) | Path/URI to the MLflow SQLite registry |
| `MLFLOW_ARTIFACT_ROOT_OVERRIDE` | `/srv/mlflow` (Docker) / unset (host) | Prefix substitution for artifact URIs whose host-side absolute path doesn't exist inside the container. See [`../../docs/volume-mount-strategy.md`](../../docs/volume-mount-strategy.md) |
| `MODEL_ALIAS` | `production` | Which registry alias to resolve for all 3 models |
| `INPUT_DISTRIBUTION_SAMPLE_EVERY` | `100` | Emit `input_distribution_sample` events every Nth request on `/predict/dnn` only (GAN has no user features; Q-learning takes one int). `0` disables |
| `MODEL_INFERENCE_STALENESS_SECONDS` | `3600` | Freshness threshold for all three `/health/<model>` endpoints. Each model tracks its own `_LAST_INFERENCE_TS` independently |

## Health + monitoring

`/ready` is all-or-nothing (200 only when all 3 models loaded); per-model `/health/<model>` lets operators pinpoint which model went stale. The freshness check is per-model independently, so an idle GAN can return 503 `inference_stale` while DNN + Q-learning return 200 — useful when one model gets traffic and the others don't.

`/metrics` exposes `model_inference_duration_seconds` with `model_name` partitioning all three models — query the histogram per model for tail-latency comparisons. Full catalog + PromQL queries in [`../../docs/monitoring.md`](../../docs/monitoring.md).

Drift sampling (`input_distribution_sample` events) runs on `/predict/dnn` only — GAN's input is server-side noise (no user features to summarize), and Q-Learning's input is a single integer (no useful distribution shape).

## Local development

From `deployment/services/pt-svc/`:

```powershell
.\.venv\Scripts\pip install -r requirements-dev.txt
.\.venv\Scripts\ruff check .
.\.venv\Scripts\mypy --strict app
.\.venv\Scripts\pytest -q
```

Test suite is hermetic — `FakeDNN`, `FakeDCGAN`, and `FakeQtable` fixtures drive deterministic paths through each router without loading real artifacts. Real-data parity is verified separately via the three smoke scripts (need the service running).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Container exits with `RuntimeError: ... alias 'production' not found` for one of the 3 models | One model's alias not set in the registry | List aliases via `python scripts/promote_to_registry.py --list` (from `deployment/scripts/`); promote/move with `--name <model> --alias production --version <n>` |
| `/ready` returns 503 with `not_all_models_loaded` past 10 s | Slowest loader still resolving — usually DCGAN's state_dict load on first cold-boot | `docker compose logs pt-svc` and look for `dnn_load_start` / `gan_load_start` / `qtable_load_start` events; whichever hasn't emitted its `*_alias_resolved` is the bottleneck |
| `/predict/gan/sample` returns 422 `n_samples ... not in [1, 16]` | Client over-requesting | The 16-sample cap bounds response payload at ~48 KB (16 × ~3 KB base64 PNGs); split larger requests client-side |
| `/predict/dnn` returns 422 `features ... not in [-1.0, 1.0]` | Client sent already-scaled features (typically in [-3, 3] after standardization) | The schema enforces `[-1, 1]` strictly at the API boundary, before any internal processing. Send raw publisher-format UCI HAR features (already pre-normalized in `[-1, 1]`); the service applies its own StandardScaler internally |
| `/predict/qlearning/taxi` returns all-zero `q_values` | Terminal Taxi state queried (100 of 500 are absorbing) | Expected — clients can detect via `sum(q_values) == 0` and call `env.reset()` for a new episode |
| GAN responses unusually large or memory pressure rising | `n_samples=16` on every request | Payload + decoder memory scale linearly with `n_samples`; request fewer if the cap isn't needed |

## Further reading

- [`../../README.md`](../../README.md) — deployment-phase overview
- [`../../docs/architecture.md`](../../docs/architecture.md) — design decisions across all 3 services
- [`../../docs/deployment-runbook.md`](../../docs/deployment-runbook.md) — clone-to-running step-by-step
- [`../../docs/adding-a-new-model.md`](../../docs/adding-a-new-model.md) — extension pattern for a 6th model
