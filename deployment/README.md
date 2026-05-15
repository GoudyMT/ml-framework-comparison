# Deployment Phase

[![CI](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/ci.yml/badge.svg)](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/ci.yml)
[![Build and Push](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/build.yml/badge.svg)](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/build.yml)

Production-grade FastAPI services for the staged deployment winners from the modeling phase (#01-#20). Demonstrates every distinct deployment pattern the portfolio produced via 5 representative models across 3 framework runtimes.

> **Status: Phase 8 complete - GitHub Actions CI/CD live with three workflows. `ci.yml` runs ruff + mypy strict + pytest in matrix across 3 services on every PR and push to main (~2 min wall-clock). `build.yml` builds + pushes 3 Docker images to `ghcr.io/<owner>/<repo>/<service>:{latest,sha-<short>}` on every push to main via `docker/build-push-action@v6` with `type=gha` layer cache scoped per-service (cold-build ~6 min; cached runs ~2-3 min). `deploy.yml` is a `workflow_dispatch` placeholder for the cloud-deploy pipeline shape. Authentication via the workflow-scoped GITHUB_TOKEN; no PAT secrets needed. Moving to Phase 9 (monitoring + drift-detection middleware).** This README will be filled in at Phase 11 once all phases land.

## Models Deployed

| Service | Model | Pattern | Endpoint |
|---------|-------|---------|----------|
| `sklearn-svc` | SK PCA | Tabular sklearn | `POST /predict/pca` |
| `pt-svc` | PT DNN (UCI HAR 96.03%) | PT classifier | `POST /predict/dnn` |
| `pt-svc` | PT GANs (DCGAN FID 30.57) | PT generative | `POST /predict/gan/sample` |
| `pt-svc` | PT Q-Learning V1 (Taxi-v4) | PT reinforcement learning | `POST /predict/qlearning/taxi` |
| `tf-svc` | TF Transformer Translation | TF + tokenization preprocessing | `POST /translate` |

## Architecture

- **Unified FastAPI multi-endpoint design**: 3 services (one per framework runtime), each hosting multiple model routers
- **Multi-stage Docker builds** with non-root users; framework-specific base images keep size proportional
- **MLflow Model Registry** as source of truth; local-file fallback for dev iteration
- **GitHub Actions CI/CD**: lint -> mypy -> pytest -> build -> push to ghcr.io -> manual-trigger deploy
- **Structured JSON logging** (structlog) + Prometheus metrics endpoint per service
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

- `docs/architecture.md` - design decisions + rationale
- `docs/deployment-runbook.md` - step-by-step "from clone to running endpoints"
- `docs/adding-a-new-model.md` - pattern for extending to a 6th model
- `docs/monitoring.md` - `/metrics` endpoint + log structure

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
- [ ] Phase 9 - Monitoring + drift-detection middleware
- [ ] Phase 10 - OpenAPI/Swagger polish
- [ ] Phase 11 - Documentation
- [ ] Phase 12 - Git close-out
