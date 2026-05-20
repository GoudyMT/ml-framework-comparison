# Deployment Runbook

Step-by-step procedure for going from a fresh clone to running endpoints. If you want to understand WHY the deployment looks the way it does, read [`architecture.md`](architecture.md) first; this doc tells you HOW to operate it.

**Audience.** Someone setting up the deployment on their own machine — a hiring manager kicking the tires, a future Max returning to the repo, anyone forking the layout. Assumes Windows 11 + PowerShell; Linux/macOS readers can substitute path separators and shell idioms.

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Git | 2.40+ | Clone the repo |
| Python | 3.12.2 (exact for hash-locked installs to match) | Host-side venvs |
| Docker Desktop | 24.0+ (Compose v2 bundled) | Containerized path |
| PowerShell | 5.1 or 7.x | All commands in this doc use PS-native syntax |

You do not need a GPU. Every service is CPU-only — the PyTorch image installs `torch==2.5.1+cpu` wheels; TF runs on the CPU build with oneDNN ops; sklearn is CPU by definition.

The repo expects ~10 GB free disk for the three Docker images plus the MLflow registry and modeling-phase artifacts.

## Step 1 — Clone and orient

```powershell
git clone https://github.com/GoudyMT/ml-framework-comparison.git
cd ml-framework-comparison
```

Skim the top-level [`README.md`](../../README.md) for portfolio context, then read [`../README.md`](../README.md) for the deployment phase overview. The phase docs you'll likely consult during this runbook:

- [`architecture.md`](architecture.md) — design decisions
- [`registry-strategy.md`](registry-strategy.md) — how the MLflow registry resolves model versions
- [`monitoring.md`](monitoring.md) — what `/metrics` exposes + sample PromQL
- Per-service READMEs at `deployment/services/{sklearn-svc,pt-svc,tf-svc}/README.md`

## Step 2 — Choose your path

Two ways to run the services:

| Path | When to choose |
|---|---|
| **Docker Compose** (Step 3) | You want all three services up, the full end-to-end surface, the most production-like setup. The default. |
| **Host venv per service** (Step 4) | You're hacking on one service, want hot-reload + debugger attachment, don't want the container layer in the loop. |

Either path uses the same MLflow registry at `deployment/mlflow.db` + `deployment/mlruns/`.

## Step 3 — Docker Compose path

From `deployment/`:

### 3.1 Build + boot

```powershell
cd deployment
docker compose up -d --build
```

First-time cold build: ~6 minutes (downloads + pip installs). Subsequent rebuilds with code-only changes: ~15 seconds (layer cache reuses everything except the `COPY app/` step).

### 3.2 Verify health

```powershell
docker compose ps
```

Expected output: three containers, all `Up X seconds (healthy)`. The HEALTHCHECK probe in each Dockerfile takes 5-10 seconds to flip from `(starting)` to `(healthy)` after the lifespan loader finishes.

Per-service health from a second terminal:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8001/ready | % Content
Invoke-WebRequest -UseBasicParsing http://localhost:8002/ready | % Content
Invoke-WebRequest -UseBasicParsing http://localhost:8003/ready | % Content
```

Each returns 200 with a JSON body showing `model_loaded: true` (sklearn-svc) or `models: {<name>: {loaded: true, version: "1"}}` (pt-svc, tf-svc).

### 3.3 Smoke-test each prediction endpoint

```powershell
# sklearn-svc: PCA on a flat 784-pixel vector (all zeros is fine for the smoke)
$pca = @{ features = @(1..784 | ForEach-Object { 0.0 }) } | ConvertTo-Json -Compress
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8001/predict/pca `
  -Method POST -ContentType 'application/json' -Body $pca

# pt-svc: DNN (UCI HAR), GAN (1 deterministic image), Q-Learning (mid-episode state)
$dnn = @{ features = @(1..561 | ForEach-Object { 0.0 }) } | ConvertTo-Json -Compress
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8002/predict/dnn `
  -Method POST -ContentType 'application/json' -Body $dnn
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8002/predict/gan/sample `
  -Method POST -ContentType 'application/json' -Body '{"n_samples":1,"seed":42}'
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8002/predict/qlearning/taxi `
  -Method POST -ContentType 'application/json' -Body '{"state":328}'

# tf-svc: translation (English -> Spanish)
$r = Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8003/translate `
  -Method POST -ContentType 'application/json' -Body '{"text":"Hello, how are you?"}'
[System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray())
```

The translation response needs the explicit UTF-8 decode because PowerShell's `$r.Content` defaults to Latin-1, which mojibakes Spanish accents (`ó` becomes `Ã³`). The wire bytes are UTF-8; this is purely a display issue.

### 3.4 Browse the Swagger UI

Three URLs in a browser:

- http://localhost:8001/docs — sklearn-svc
- http://localhost:8002/docs — pt-svc
- http://localhost:8003/docs — tf-svc

Each renders the full OpenAPI 3 surface with grouped tags, per-endpoint summaries + descriptions, request schemas with examples, and the "Try it out" panel for in-browser smoke testing.

### 3.5 Tear down

```powershell
docker compose down
```

Removes the containers + the named network. Images stay cached for the next `docker compose up`.

## Step 4 — Host venv path (per service)

Skip Step 3 if you took this path. Pick the service you want to run.

### 4.1 Create the venv

```powershell
cd deployment/services/sklearn-svc      # or pt-svc / tf-svc
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-windows.txt
```

Each service has its own `.venv` because the dependency sets conflict (TF + PT in the same venv is fragile). The `requirements-windows.txt` lockfile resolves wheels for the host platform; the Docker build uses `requirements.txt` (Linux-resolved). See [`dependency-strategy.md`](dependency-strategy.md) for the two-lockfile rationale.

### 4.2 Boot uvicorn

```powershell
# Port choice per service:
#   sklearn-svc -> 8001
#   pt-svc      -> 8002
#   tf-svc      -> 8003
.\.venv\Scripts\uvicorn.exe app.main:app --port 8001 --reload
```

`--reload` watches the source tree and restarts on changes (dev-only; never use it in production).

### 4.3 Smoke-test

The smoke commands from [Step 3.3](#33-smoke-test-each-prediction-endpoint) apply unchanged — the only difference is which port is up.

### 4.4 Real-data parity smoke

Each service ships `scripts/smoke_test.py` that compares an independent forward pass against the service response. With the service running:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

pt-svc has three (`smoke_test.py` for DNN, `smoke_test_gan.py`, `smoke_test_qlearning.py`); the others have one. Each prints PASS/FAIL with the diff if any.

## Step 5 — Common operations

### 5.1 Tail structured logs

```powershell
# Docker path: tail one service
docker compose logs sklearn-svc -f

# Host path: logs go to stdout where you booted uvicorn
```

The log lines are JSON. Pipe through `jq` (or `ConvertFrom-Json` in PS) to filter:

```powershell
# Show only request lifecycle events
docker compose logs sklearn-svc -f | Select-String 'request_started|request_finished'

# Show only the drift-signal samples (one every 100 requests by default)
docker compose logs pt-svc -f | Select-String 'input_distribution_sample'
```

### 5.2 Read Prometheus metrics

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8001/metrics |
  Select-Object -ExpandProperty Content |
  Out-String -Stream |
  Select-String '^(http_requests_total|model_inference)'
```

The full metric catalog + sample PromQL is in [`monitoring.md`](monitoring.md).

### 5.3 Run the test suite

```powershell
cd deployment/services/sklearn-svc
.\.venv\Scripts\pip install -r requirements-dev.txt    # one-time
.\.venv\Scripts\ruff check .
.\.venv\Scripts\mypy --strict app
.\.venv\Scripts\pytest -q
```

195/195 tests pristine across the three services; ruff + mypy strict clean. CI runs the same three commands in matrix on every push.

### 5.4 Regenerate the requirements lockfiles

```powershell
cd deployment/services/sklearn-svc
.\.venv\Scripts\pip install pip-tools
.\.venv\Scripts\pip-compile --generate-hashes --output-file=requirements-windows.txt pyproject.toml

# Linux lockfile must be regenerated inside a Linux image (host wheels differ):
docker run --rm -v ${PWD}:/work -w /work python:3.12.2-slim sh -c `
  "pip install pip-tools && pip-compile --generate-hashes --output-file=requirements.txt pyproject.toml"
```

Never hand-edit the lockfiles. Full workflow in [`dependency-strategy.md`](dependency-strategy.md).

### 5.5 Roll a new model version

The registry is the source of truth — no code change needed for a version bump:

```powershell
# Promote a new run's artifact as version N
cd deployment
python scripts/promote_to_registry.py --name <model-name> --run-id <mlflow-run-id>

# Move the production alias to the new version
python scripts/promote_to_registry.py --name <model-name> --alias production --version N
```

Restart the service (Docker compose restart or uvicorn restart) and the loader picks up the new version on next boot. To roll back, move the alias to the prior version. To test a candidate without affecting production, point a service at it via `MODEL_ALIAS=candidate`.

## Step 6 — Verification checklist

After Step 3 or Step 4, confirm:

- [ ] All target services show `Up X seconds (healthy)` in `docker compose ps` (Docker path) or accept HTTP on the right port (host path)
- [ ] `/health` returns 200 `{"status": "alive"}` on each running service
- [ ] `/ready` returns 200 with `model_loaded: true` (sklearn-svc) or `models: {...}` populated (pt-svc, tf-svc)
- [ ] `/health/<model>` returns 200 with `last_inference_age_seconds` + `staleness_threshold_seconds`
- [ ] Each `/predict/*` (and `/translate`) returns 200 with a correctly-shaped response body
- [ ] `/metrics` shows both `http_requests_total` and `model_inference_duration_seconds_count{model_name=...}` populated for each model after the smoke
- [ ] `/docs` renders the Swagger UI with grouped tags + per-endpoint summaries
- [ ] `/openapi.json` parses as valid JSON and contains `info.contact`, `info.license`, and per-endpoint `summary` + `description` + `responses`

If any item fails, jump to [Step 7](#step-7--common-failure-modes).

## Step 7 — Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker compose up` fails with "no space left on device" | Docker Desktop's WSL2 image is full | Prune via `docker system prune --volumes` (deletes stopped containers + dangling images + unused volumes; ~5-10 GB freed in typical situations) |
| Container exits at startup with `RuntimeError: ... alias 'production' not found` | The MLflow registry doesn't know the alias | List aliases via `python deployment/scripts/promote_to_registry.py --list`; promote/move with `--name <model> --alias production --version <n>` |
| `/ready` returns 503 with `model_not_loaded` past 30 seconds | The lifespan loader is stuck — usually a stale registry or wrong `MLFLOW_TRACKING_URI` | `docker compose logs <service>` and look for the `*_load_start` event without a matching `*_alias_resolved` |
| `/predict/*` returns 422 with a validation error | Wrong-shape request body | The error body shows the exact field + constraint that failed. The per-service README troubleshooting tables list the common shape mistakes |
| Host venv install fails with hash mismatch | Lockfile drift after a wheel re-publish on PyPI | Regenerate via [Step 5.4](#54-regenerate-the-requirements-lockfiles); never edit by hand |
| Spanish accents render as `Ã³`, `Â¿` in PowerShell | UTF-8 → Latin-1 decoding by PowerShell on `$response.Content` | Use the explicit UTF-8 decode shown in [Step 3.3](#33-smoke-test-each-prediction-endpoint); the wire bytes are correct |
| Browser to `/docs` shows the Swagger UI but no endpoints | Service hasn't finished booting; OpenAPI surface is rendered before the lifespan completes | Refresh after the `(healthy)` flip in `docker compose ps` |
| `pytest` fails with `ModuleNotFoundError: No module named 'app'` | Tests run from the wrong directory | `cd deployment/services/<service>` first; pytest's `pyproject.toml` configures the import root from there |

## Where to go next

- Extend the deployment with a 6th model → [`adding-a-new-model.md`](adding-a-new-model.md)
- Understand a specific design decision → [`architecture.md`](architecture.md) + the linked deep-dive docs
- Operate a single service in detail → the per-service README at `deployment/services/<service>/README.md`
- Inspect the CI workflows → `.github/workflows/{ci,build,deploy}.yml`
