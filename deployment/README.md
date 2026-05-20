# Deployment Phase

[![CI](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/ci.yml/badge.svg)](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/ci.yml)
[![Build and Push](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/build.yml/badge.svg)](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/build.yml)

Production-grade FastAPI services for the staged deployment winners from the modeling phase (#01-#20). Demonstrates every distinct deployment pattern the portfolio produced via 5 representative models across 3 framework runtimes.

> **Status: Phase 11 complete; Phase 12 (Git close-out) pending.** Five models live across three FastAPI services with MLflow-registry-backed loading, multi-stage Docker images, GitHub Actions CI/CD, complete observability surface (per-model inference-latency histogram + drift-signal log event + per-model freshness endpoints with operator-tunable threshold), exhaustive OpenAPI/Swagger metadata + `operation_id` on every endpoint, and the full operator + architecture documentation set. 195/195 tests pristine; ruff + mypy strict clean. Containerized end-to-end smoke check on 2026-05-19 confirmed every health, prediction, `/metrics`, and `/openapi.json` surface live in the built images. See [`docs/architecture.md`](docs/architecture.md) for design rationale, [`docs/deployment-runbook.md`](docs/deployment-runbook.md) for the clone-to-running guide, [`docs/adding-a-new-model.md`](docs/adding-a-new-model.md) for the extension pattern, and the per-service READMEs at `services/<svc>/README.md` for operator how-tos.

## Models Deployed

| Service | Model | Pattern | Endpoint |
|---------|-------|---------|----------|
| `sklearn-svc` | SK PCA | Tabular sklearn | `POST /predict/pca` |
| `pt-svc` | PT DNN (UCI HAR 96.03%) | PT classifier | `POST /predict/dnn` |
| `pt-svc` | PT GANs (DCGAN FID 30.57) | PT generative | `POST /predict/gan/sample` |
| `pt-svc` | PT Q-Learning (Taxi-v4) | PT reinforcement learning | `POST /predict/qlearning/taxi` |
| `tf-svc` | TF Transformer Translation | TF + sentencepiece | `POST /translate` |

## Architecture

- **Unified FastAPI multi-endpoint design**: 3 services (one per framework runtime), each hosting multiple model routers
- **Multi-stage Docker builds** with non-root users; framework-specific base images keep size proportional
- **MLflow Model Registry** as source of truth; local-file fallback for dev iteration
- **GitHub Actions CI/CD**: lint -> mypy -> pytest -> build -> push to ghcr.io -> manual-trigger deploy
- **Structured JSON logging** (structlog) + Prometheus metrics endpoint per service
- **Per-model observability**: `model_inference_duration_seconds` Histogram (isolates model time from request time) + `input_distribution_sample` log event (every Nth-request drift signal) + per-model `/health/<model>` freshness endpoints; full rationale in `docs/monitoring.md`
- **Pydantic v2** schemas + auto-generated OpenAPI/Swagger docs at `/docs`

## Quick Start

```powershell
# Bring up all three services (sklearn-svc, pt-svc, tf-svc)
cd deployment
docker compose up -d --build

# Verify all three are healthy
docker compose ps

# Hit a prediction endpoint (Fashion-MNIST PCA in this example)
$payload = @{ features = @(0..783 | ForEach-Object { 100.0 }) } | ConvertTo-Json -Compress
Invoke-WebRequest -UseBasicParsing `
  -Uri http://localhost:8001/predict/pca `
  -Method POST -ContentType 'application/json' -Body $payload

# Tear down (containers + network removed; images stay cached)
docker compose down
```

See [`docs/dependency-strategy.md`](docs/dependency-strategy.md) for the two-lockfile pattern (Linux for Docker, Windows for host dev) and [`docs/volume-mount-strategy.md`](docs/volume-mount-strategy.md) for the registry bind-mount + `MLFLOW_ARTIFACT_ROOT_OVERRIDE` mechanism that keeps artifact paths portable across hosts.

## Documentation

### Per-service operator guides

- [`services/sklearn-svc/README.md`](services/sklearn-svc/README.md) - D1 PCA service: boot, env vars, endpoints, troubleshooting
- [`services/pt-svc/README.md`](services/pt-svc/README.md) - PyTorch service (DNN + DCGAN + Q-Learning) covering 3 models in one process
- [`services/tf-svc/README.md`](services/tf-svc/README.md) - TF Transformer translation: 2-artifact loader + oneDNN warm-up notes

### Architecture + operations

- [`docs/architecture.md`](docs/architecture.md) - design decisions, service topology, framework-runtime boundaries, cross-cutting patterns, explicit out-of-scope table
- [`docs/deployment-runbook.md`](docs/deployment-runbook.md) - step-by-step clone-to-running procedure with two paths (docker compose + host venv)
- [`docs/adding-a-new-model.md`](docs/adding-a-new-model.md) - extension pattern walking through a 6th model across all file touches

### Decision deep dives

- [`docs/dependency-strategy.md`](docs/dependency-strategy.md) - two-lockfile pattern, pip-tools workflow, hash-locked installs
- [`docs/registry-strategy.md`](docs/registry-strategy.md) - aliases vs Stages, direct file-open vs flavor wrappers, `MODEL_ALIAS` override
- [`docs/volume-mount-strategy.md`](docs/volume-mount-strategy.md) - bind-mount registry, artifact-root override, cross-host portability
- [`docs/monitoring.md`](docs/monitoring.md) - `/metrics` catalog, sample PromQL, drift-signal log shape, HEALTHCHECK noise note

## Phase Progress

- [x] **Phase 0 - Pre-flight** (scaffolding + MLflow consolidated registry + dependency strategy + artifact verification)
- [x] **Phase 1 - FastAPI scaffolding + D1 SK PCA endpoint** (sklearn-svc: schemas, lifespan loader with scaler bundling, /predict/pca router, request_id + structlog + prometheus middleware, 20 pytest tests, real Fashion-MNIST smoke test bit-exact PASS)
- [x] **Phase 2 - D2 PT DNN endpoint** (pt-svc: torch CPU-only build, DNN architecture class + state_dict loader, scaler bundling, /predict/dnn router with softmax/argmax + Literal-typed predicted_label, 16 pytest tests, real UCI HAR smoke test bit-exact PASS - real sample classified as STANDING with ~100% confidence)
- [x] **Phase 3 - D3 PT DCGAN endpoint** (pt-svc: DCGenerator architecture class + state_dict loader, /predict/gan/sample router with server-side noise sampling + optional seed for reproducibility + base64-PNG response, /ready refactored to multi-model dict shape, Pillow promoted from transitive to direct dep, 13 new pytest tests including seed plumbing + decode-and-shape, real DCGAN smoke test bit-exact PASS - 0 pixels differ between manual forward and service round-trip)
- [x] **Phase 4 - D4 PT Q-Learning Taxi-v4 endpoint** (pt-svc: Q-table loader with shape assertion, /predict/qlearning/taxi router with state-as-int input + action/action_name/q_values response, /ready expanded to report 3 models, FakeQtable test fixture with deterministic state to action cycle, 15 new pytest tests including parametrized action_name mapping across all 6 Gymnasium actions, real Q-table smoke test bit-exact PASS - 0 disagreements across 15 states)
- [x] **Phase 5 - D5 TF Transformer translation endpoint** (tf-svc: new service from scratch, TF 2.21 CPU runtime + sentencepiece BPE tokenizer, 6-class Transformer architecture (encoder-decoder, 3+3 layers, d_model=256, 11.68M params) hardcoded in translation_model.py to match the registered .h5 weights, multi-artifact loader with warm-up forward pass for oneDNN op compilation, /translate router with greedy autoregressive decode loop (encode-once + decode-many optimization), /ready refactored to multi-model dict shape, full middleware stack copied from sklearn-svc, 23 pytest tests (FakeTokenizer + FakeTransformer fixtures driving decode-loop control), real Transformer smoke test bit-exact PASS - 5/5 sentences string-exact match between manual forward and service forward)
- [x] **Phase 6 - Containerization** (per-service Dockerfiles for sklearn-svc + pt-svc + tf-svc on python:3.12.2-slim with multi-stage builder/runtime layout, non-root UID 1000, JSON-array CMD for SIGTERM forwarding, stdlib-urllib HEALTHCHECK; portable artifact paths via MLFLOW_ARTIFACT_ROOT_OVERRIDE env var + per-loader _resolve_artifact_dir helpers across 5 loaders, 20 new unit tests across 3 services bringing total to 107/107; two-lockfile pattern per service - requirements.txt Linux-resolved for Docker COPY, requirements-windows.txt for host dev; sklearn-svc mlflow alignment to >=3.12 to match the registry schema written by pt-svc/tf-svc; docker-compose.yml stitching all 3 services on ml-fc-network bridge with bind-mounted registry at /srv/mlflow:ro; volume-mount-strategy.md + dependency-strategy.md architectural docs; fresh-shell cold-start smoke PASS - all 3 images rebuild from layer cache in 13s, all reach healthy in 14s, all 5 prediction endpoints respond 200 through compose orchestration)
- [x] **Phase 7 - MLflow Model Registry stage/alias promotion** (audited: all 5 models at version 1 with `@production` alias; `current_stage = 'None'` across the board, legacy Stages never used. Decision: aliases not Stages - documented in `docs/registry-strategy.md`. Loader pattern: `MlflowClient.get_model_version_by_alias` resolves name+alias to a `ModelVersion`, then direct file-open from `version.source` via `joblib.load`/`torch.load`/`np.load`/`tf.load_weights`/`spm.Load` - bypasses `mlflow.<flavor>.load_model` because `promote_to_registry.py` used `log_artifact` (raw upload) not `log_model` (flavor-structured); re-promoting in flavors would require Q-learning's `.npy` to use generic Python flavor with no functional upside. Phase 7.4 implementation: `MODEL_ALIAS` env var override resolved via `_resolve_alias` helper in each of 5 loaders + 25 new unit tests (5 in sklearn-svc + 15 parametrized x 3 loaders in pt-svc + 5 in tf-svc) bringing total to 132/132. Verified with functional smoke: env unset uses default `production`, explicit `production` same outcome, `nonexistent` fails cleanly with the resolved alias visible in the chained error)
- [x] **Phase 8 - GitHub Actions CI/CD** (three workflows in `.github/workflows/`: `ci.yml` runs ruff + mypy strict + pytest in matrix across 3 services on every PR and push to main with `actions/cache@v4` pip-wheel cache keyed on requirements.txt hash, ~2 min wall-clock; `build.yml` builds + pushes 3 Docker images to `ghcr.io/<owner>/<repo>/<service>:{latest,sha-<short>}` on every push to main via `docker/build-push-action@v6` with `type=gha` layer cache scoped per-service for ~3x speedup on cache-hit runs; `deploy.yml` is a `workflow_dispatch` placeholder demonstrating the cloud-deploy pipeline shape with `image_tag` + `environment` inputs and no actual deploy. Authentication via the workflow-scoped GITHUB_TOKEN; no PAT secrets needed for same-repo ghcr.io pushes. Permissions: minimum scope per workflow (`contents: read` for ci/deploy; +`packages: write` for build). Verified - `0f87dc3` push fired CI cleanly across all 3 matrix jobs (2m 24s); `8af854f` push fired CI + Build successfully (build 6m 24s cold-build, 0% cache; all 3 images live at ghcr.io); `deploy.yml` `workflow_dispatch` tested manually - displays plan in runner log + writes Markdown step summary. CI + Build status badges added to root README.md + this README at the top.)
- [x] **Phase 9 - Monitoring + drift-detection observability** (`model_inference_duration_seconds` Histogram with `model_name` label across all 5 deployed models via a `track_inference` context manager wrapping each router's forward call; `input_distribution_sample` structlog event sampling whole-vector summary stats every Nth request per PCA + DNN, configurable via `INPUT_DISTRIBUTION_SAMPLE_EVERY`; per-model `/health/<model>` endpoints in main.py per service returning 200 diagnostic body / 503 bare-detail (`model_not_loaded` or `inference_stale`); `MODEL_INFERENCE_STALENESS_SECONDS` env var gating freshness; `_LAST_INFERENCE_TS` stamped at load completion + each inference; full architectural rationale + sample PromQL queries + jq log-tailing patterns in `docs/monitoring.md`; 132/132 -> 195/195 tests pristine across 3 services; CI green on every push)
- [x] **Phase 10 - OpenAPI/Swagger polish** (per-service FastAPI `contact` + `license_info` + `openapi_tags` metadata; `summary` + `description` + `responses` on every `/health` + `/predict` + `/metrics` endpoint; `model_config(json_schema_extra={"examples": [...]})` Request examples on all 5 schemas plus Response examples on DNN + Q-learning + translation; cross-service polish review caught + fixed 1 real contract bug (`n_output_tokens` description vs router behavior) + 3 cross-service polish items (duplicate `examples=`, GAN seed `None`, pt-svc `/health/<model>` description drift); containerized E2E smoke check 2026-05-19 verified Phase 9 + Phase 10 surface live in built images.)
- [x] **Phase 11 - Documentation** (full operator + architecture doc set landed: 3 per-service READMEs at `services/<svc>/README.md` covering boot + env vars + endpoint reference + troubleshooting per service; `docs/architecture.md` synthesizing design decisions across all services with explicit out-of-scope table; `docs/deployment-runbook.md` covering both the docker compose path and the host venv path with verification checklist + common failure modes; `docs/adding-a-new-model.md` walking through a 6th model across all 10 file touches; smoke-check observation carryover (monitoring.md HEALTHCHECK noise note, tf-svc translation schema example clarification); cross-doc polish-review sweep tightened naming consistency + scope discipline across all active markdown surface. OpenAPI hardening sub-phase added explicit `operation_id` on all 19 endpoints (clean SDK method names), promoted the private staleness-threshold accessor to public `get_staleness_threshold()`, and refactored pt-svc 3x `/health/<model>` blocks into a `_make_health_route` factory + `_HealthLoader` Protocol (~165 lines of triplication collapse to ~80 lines; the Protocol catches missing loader attributes at type-check time instead of at request time). 195/195 tests pristine across 3 services; ruff + mypy strict clean throughout.)
- [ ] Phase 12 - Git close-out
