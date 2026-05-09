# Deployment Phase

Production-grade FastAPI services for the staged deployment winners from the modeling phase (#01-#20). Demonstrates every distinct deployment pattern the portfolio produced via 5 representative models across 3 framework runtimes.

> **Status: Phase 4 complete - pt-svc D4 Q-Learning Taxi-v4 endpoint live + verified end-to-end (real Q-table forward bit-matches service output across 15 states with max_q_diff=0.0; sub-millisecond inference). Moving to Phase 5 (D5 TF Transformer Translation - new tf-svc service).** This README will be filled in at Phase 11 once all services are running.

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

(Filled in at later date once `docker compose up` works end-to-end.)

```bash
# Coming soon
docker compose up
curl -X POST http://localhost:8000/predict/dnn -d '...'
```

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
- [ ] Phase 5 - D5 TF Transformer endpoint (tokenization preprocessing)
- [ ] Phase 6 - Containerization (Dockerfiles + docker-compose)
- [ ] Phase 7 - MLflow Model Registry stage promotion
- [ ] Phase 8 - GitHub Actions CI/CD
- [ ] Phase 9 - Monitoring + drift-detection middleware
- [ ] Phase 10 - OpenAPI/Swagger polish
- [ ] Phase 11 - Documentation
- [ ] Phase 12 - Git close-out
