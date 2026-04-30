# Deployment Phase

Production-grade FastAPI services for the staged deployment winners from the modeling phase (#01-#20). Demonstrates every distinct deployment pattern the portfolio produced via 5 representative models across 3 framework runtimes.

> **Status: Phase 0 - scaffolding in progress.** This README will be filled in at Phase 11 once all services are running.

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

- [ ] Phase 0 - Pre-flight (scaffolding + MLflow audit + artifact verification)
- [ ] Phase 1 - FastAPI scaffolding + D1 SK PCA endpoint
- [ ] Phase 2 - D2 PT DNN endpoint
- [ ] Phase 3 - D3 PT GAN endpoint (image-bytes response)
- [ ] Phase 4 - D4 PT Q-Learning endpoint (state -> action)
- [ ] Phase 5 - D5 TF Transformer endpoint (tokenization preprocessing)
- [ ] Phase 6 - Containerization (Dockerfiles + docker-compose)
- [ ] Phase 7 - MLflow Model Registry stage promotion
- [ ] Phase 8 - GitHub Actions CI/CD
- [ ] Phase 9 - Monitoring + drift-detection middleware
- [ ] Phase 10 - OpenAPI/Swagger polish
- [ ] Phase 11 - Documentation
- [ ] Phase 12 - Git close-out
