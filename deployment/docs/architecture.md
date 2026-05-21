# Deployment Phase Architecture

The deployment phase ships **5 representative models from the 20-model modeling phase** as production-shaped FastAPI services. This document records the cross-cutting design decisions: WHY the services look the way they do, what was deliberately excluded, and where to find the per-decision deep dives.

If you want to operate the deployment, read [`deployment-runbook.md`](deployment-runbook.md). If you want to extend it with a 6th model, read [`adding-a-new-model.md`](adding-a-new-model.md). This doc is for evaluators: hiring managers reading the portfolio, future Max wondering why a past choice was made, or anyone forking the layout for their own work.

## Goals + Non-Goals

**Goals.** Demonstrate the full deployment surface a real ML engineer touches: registry-backed model loading, framework-runtime isolation, OCI containers, hash-locked supply chain, structured logging + Prometheus metrics + per-model health, CI matrix + image registry push, exhaustive OpenAPI/Swagger documentation. Five models across three frameworks prove the patterns generalize beyond one happy-path example.

**Non-Goals.** No cloud deployment (the `deploy.yml` workflow is a documented placeholder, not a working pipeline). No drift-detection compute (only the upstream log signal — PSI/KS computation is downstream tooling out of scope). No model-quality A/B testing, request-shadowing, canary routing, or feature-flagged rollouts. No web UI for the registry (MLflow's bundled UI plus the CLI is enough). No multi-tenant isolation or per-client rate limiting — the services are single-tenant by design.

The boundary is deliberate. Building all of the above would dilute the depth of what's here. The deferred concerns are listed in [What's Out of Scope](#whats-out-of-scope) with notes on where they'd plug in.

## Service Topology

Three services, five models, three framework runtimes:

| Service | Models | Framework | Port | Image size |
|---|---|---|---|---|
| `sklearn-svc` | SK PCA | scikit-learn | 8001 | ~1.27 GB |
| `pt-svc` | PT DNN, PT DCGAN, PT Q-Learning Taxi-v4 | PyTorch (CPU) | 8002 | ~2.33 GB |
| `tf-svc` | TF Transformer Translation | TensorFlow (CPU) | 8003 | ~4.05 GB |

**Why split by framework, not by model?** Each framework brings a multi-hundred-MB runtime (torch CPU is ~800 MB; TF + dependencies is ~2.5 GB). One image per model would duplicate that runtime per model, multiplying disk + RAM footprint without buying anything. Co-located models that share a runtime (the three PyTorch models in `pt-svc`) sit in the same image and reuse the loaded framework.

**Why three services, not one?** Three reasons. (1) Conflicting transitive dependencies: TF's numpy floor and PyTorch's protobuf pin clash; resolving a single environment with both would mean version-pinning to the intersection — fragile and undocumented. (2) Independent rollouts: pinning a new TF version shouldn't force a sklearn redeploy. (3) Independent failure modes: a TF oneDNN warm-up failure shouldn't take down PCA.

**Why unified FastAPI multi-endpoint within a service?** Within a framework, model loaders + routers + schemas are cheap to colocate (one process, one OpenAPI surface, one /metrics scrape). Splitting `pt-svc` into three microservices would triple the per-model overhead (3× process, 3× scrape target, 3× health check) without separating any failure modes — they share the runtime regardless.

The shape mirrors how real ML teams ship: framework as the deployment boundary, models as routes within.

## Registry as Source of Truth

Every loader resolves models via the MLflow Model Registry rather than a path or a URL. The registry is the single answer to "which version of which model is production?".

Two locked decisions, both with full rationale in [`registry-strategy.md`](registry-strategy.md):

1. **Aliases, not Stages.** MLflow 3.x deprecated the legacy Stage workflow (`Staging` / `Production` / `Archived` enum); aliases (`@production`, `@candidate`, free-form text) replaced it. The portfolio uses aliases from day one — no migration burden.
2. **Direct file-open from `version.source`, not flavor wrappers.** Each loader resolves the alias to a `ModelVersion`, then calls `joblib.load` / `torch.load` / `np.load` / `tf.load_weights` / `spm.Load` directly on the artifact path. Bypasses `mlflow.<flavor>.load_model` because `promote_to_registry.py` used `log_artifact` (raw upload), not `log_model` (flavor-structured). The result: zero flavor wrapper code, full registry contract (name + alias + version tracking), trivial loaders.

The `MODEL_ALIAS` environment variable lets operators override the alias at deploy time without changing code — point to `@staging` for a canary, `@production` for the default.

## Dependency Strategy

Every service has **two lockfiles** maintained by pip-tools:

| File | Resolved for | Used by |
|---|---|---|
| `requirements.txt` | Linux x86_64 | Docker build (`COPY` + `pip install --require-hashes`) |
| `requirements-windows.txt` | Windows x86_64 | Host-side dev venv (`pip install` on the developer's machine) |

**Why two lockfiles?** Wheel hashes differ between platforms (the Linux + Windows wheels are different binaries). A single lockfile would resolve correctly for one platform and either fail or fall back to non-hash-locked installs on the other. Maintaining both keeps `--require-hashes` enforceable everywhere.

**Why `--require-hashes`?** Supply-chain safety. If a malicious wheel replaces a legitimate one on PyPI, the hash mismatch fails the build instead of silently shipping the malicious wheel. Worth the bookkeeping; trivial to regenerate via pip-compile.

Full rationale + regeneration workflow in [`dependency-strategy.md`](dependency-strategy.md).

## Container Strategy

Every service ships as a multi-stage OCI image built from `python:3.12.2-slim`:

```
[builder stage]            [runtime stage]
python:3.12.2-slim         python:3.12.2-slim
+ pip install all deps     + COPY /opt/venv from builder
  into /opt/venv           + COPY app/ source
                           + create non-root user app:1000
                           + USER app  (drops privileges)
                           + HEALTHCHECK via stdlib urllib
                           + CMD ["uvicorn", "app.main:app", ...]
```

**Why multi-stage?** Build-time tooling (compilers, pip metadata, wheel caches) stays out of the runtime image. Only the resolved `/opt/venv` directory crosses the stage boundary. Final images are 30-40% smaller than a single-stage equivalent.

**Why non-root UID 1000?** Predictable file ownership when the host bind-mounts a volume (the MLflow registry). The container user matches the typical Linux first-user UID, so files written by the host process are readable by the container process without `chown` gymnastics.

**Why stdlib-urllib HEALTHCHECK, not curl?** `curl` isn't in `python:slim`. Adding it would mean an `apt-get install` layer for a single HTTP probe. The stdlib `urllib.request` is already available — no extra dep.

**Why JSON-array CMD?** `CMD ["uvicorn", ...]` runs `uvicorn` as PID 1 directly; signals (`SIGTERM` from `docker stop`) reach it cleanly for graceful shutdown. The shell form (`CMD uvicorn ...`) wraps in `/bin/sh -c`, which doesn't forward signals — containers would have to be killed forcefully on shutdown.

## Volume + Registry Mount Strategy

The MLflow registry (SQLite database at `deployment/mlflow.db` + artifact tree at `deployment/mlruns/`) is bind-mounted read-only into each container at `/srv/mlflow`. Loaders read from the mount; no service can mutate the registry by accident.

**Why bind-mount instead of baking the artifacts into the image?**

| Option | Pro | Con |
|---|---|---|
| Bake artifacts into image | Self-contained image | New model version → image rebuild + push; image bloat per model |
| Bind-mount registry | Model rollouts via registry alias move (no rebuild); image stays small | Image isn't self-contained; needs the mount to function |

Bind-mount wins for the local-dev + portfolio-demo use case (fast iteration on model versions). A real cloud deploy would swap the bind-mount for an artifact-store URL (S3, Azure Blob), keeping the same loader code.

The `MLFLOW_ARTIFACT_ROOT_OVERRIDE` env var handles the cross-host portability problem: when `promote_to_registry.py` records the artifact's source path, it records the host-machine absolute path (e.g., `C:/Users/.../deployment/mlruns/...`). That path doesn't exist inside the container. The override env var tells each loader's `_resolve_artifact_dir` helper to substitute the registered prefix with the container-side mount prefix at request time. Full rationale + worked example in [`volume-mount-strategy.md`](volume-mount-strategy.md).

## API Contract Design

Every service exposes the same surface shape:

| Path | Purpose |
|---|---|
| `/health` | Liveness — process is responding |
| `/ready` | Readiness — model(s) loaded; ready for traffic |
| `/health/<model>` | Per-model freshness — last inference within the staleness threshold |
| `/predict/*` or `/translate` | Inference |
| `/metrics` | Prometheus text exposition |
| `/docs`, `/redoc`, `/openapi.json` | OpenAPI surface |

**Why this exact set?** The three health concepts (liveness vs readiness vs freshness) answer three different orchestrator questions. Liveness = "restart the process if this fails." Readiness = "stop routing traffic if this fails." Freshness = "the model loaded but hasn't seen traffic in N hours — is something upstream broken?" Conflating any two of these leaves a real failure mode untracked.

**Why Pydantic v2 schemas?** Three benefits in one type definition: runtime validation (incoming JSON), static type-checking (mypy strict catches handler bugs), and OpenAPI generation (Swagger UI gets schemas + examples + validation rules for free). The single source of truth means schema + handler + docs can't drift apart.

**Why exhaustive OpenAPI metadata (summary + description + responses per endpoint)?** A `/docs` page that just lists path + method is useless for ops. Phase 10 added explicit `summary` (one-line), `description` (paragraph with rationale), and `responses` (status code → meaning) on every endpoint. Operators reading the Swagger UI now see what they need to know without cross-referencing the source.

## Observability

Three pillars, all per-service:

1. **Structured JSON logs (structlog)** to stdout. Every log line includes `request_id` (per-request UUID propagated through structlog contextvars), `model_name` where relevant, and a stable `event` key. Pipe through `jq` for ad-hoc filtering; ship to any log aggregator without a parser.
2. **Prometheus metrics at `/metrics`.** Three HTTP-layer metrics cover the SRE four golden signals (`http_requests_total`, `http_request_duration_seconds`, `http_requests_in_flight`); one per-model metric (`model_inference_duration_seconds{model_name}`) isolates model time from request handling. Sample PromQL queries + bucket choice + cardinality analysis in [`monitoring.md`](monitoring.md).
3. **Per-model health endpoints.** Each `/health/<model>` returns 200 with diagnostic body when fresh (`last_inference_age_seconds` + `staleness_threshold_seconds`), 503 with grep-friendly detail (`model_not_loaded` or `inference_stale`) otherwise. Operator-tunable threshold via `MODEL_INFERENCE_STALENESS_SECONDS`.

**Drift signal.** The `input_distribution_sample` structlog event fires on PCA + DNN routers every Nth request (operator-tunable via `INPUT_DISTRIBUTION_SAMPLE_EVERY`, default 100). The event carries whole-vector summary stats (mean / std / min / max / L2 / zero-fraction); a downstream drift pipeline would consume this log stream and compute PSI / KS / etc. The portfolio scope ends at emitting the signal — actual drift compute is a separate tool.

## CI/CD

Three GitHub Actions workflows under `.github/workflows/`:

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | PR + push to main | Matrix-parallel ruff + mypy strict + pytest across all 3 services; ~2 min wall-clock |
| `build.yml` | Push to main + manual dispatch | Builds + pushes 3 images to `ghcr.io/<owner>/<repo>/<service>:{latest,sha-<short>}`; per-service Docker layer cache |
| `deploy.yml` | Manual dispatch only | Documented placeholder — displays the would-be deploy plan + writes a step summary; no actual deploy |

**Why the placeholder `deploy.yml`?** Cloud deploy is intentionally out of scope (see [Non-Goals](#goals--non-goals)), but the workflow shape (inputs + permissions + step structure) demonstrates what a real deploy job would look like. A team forking the layout for a real cloud target replaces the placeholder body — the trigger + auth + image-tag selection are already wired.

**Why GHCR over Docker Hub?** Same-repo auth via the workflow-scoped `GITHUB_TOKEN` — no PAT secrets to manage. The image path lives next to the source, version-tagged + commit-tagged for rollback.

## Cross-Cutting Patterns

Three patterns appear in every service. They're the deployment-phase equivalent of utility modules:

**Lifespan loaders.** Each `app/services/<model>_loader.py` exposes a `load_*_model()` called once from FastAPI's lifespan event. Eager-load-at-startup means model-load failures crash the process at boot, not on the first user request. The loader caches the model in module-level slots; routers read via `get_*_model()` (cheap dict lookup, no I/O).

**Per-loader `_resolve_*` helpers.** Each loader resolves three things from env vars with sensible defaults: tracking URI, artifact root override, alias. The pattern (`<NAME>_OVERRIDE_ENV` constant + `_resolve_*` private function) is identical across all 5 loaders — predictable to read, predictable to test.

**Byte-identical middleware copies.** `app/middleware/{request_id,logging,metrics,inference_tracking,input_distribution}.py` are byte-for-byte identical across the three services. The duplication is deliberate: shared-library hell (lockstep version bumps, hidden inter-service coupling) is a real cost; copy-now-and-diff-on-change is cheap. A cross-service diff after any change that touches more than one service catches drift before it ships — Phase 10 surfaced a duplicate `examples=` pattern across the three Request schemas that uniform application had hidden.

**Hermetic test fixtures.** Each service has `FakeX` fixtures driving deterministic paths through the routers without loading real artifacts. Tests don't need MLflow, don't need the real registry, don't need GPU. Real-data parity is checked separately via per-service `scripts/smoke_test.py` that needs the service running.

## What's Out of Scope

| Concern | Why deferred | Where it would plug in |
|---|---|---|
| Cloud deployment | Out-of-scope per [Non-Goals](#goals--non-goals); requires real cloud account + costs | Replace `deploy.yml` placeholder body with real `kubectl apply` / Cloud Run `gcloud run deploy` / AWS ECS task-definition update |
| Drift computation (PSI, KS, etc.) | Downstream tooling concern; portfolio scope ends at emitting the signal | Consume the `input_distribution_sample` log stream from any log aggregator; PSI/KS over rolling windows; alert on threshold breach |
| Model-quality A/B testing | Requires traffic splitting + offline metric infra | The `MODEL_ALIAS` env var + the per-model `/health/<model>` freshness are the building blocks; need a router layer above the services for split traffic |
| Request shadowing | Requires a sidecar or proxy layer | An envoy filter or service mesh would shadow `/predict/*` from prod to a canary instance running `MODEL_ALIAS=candidate` |
| Real-time monitoring dashboards | Requires a Grafana/Datadog instance and dashboard JSON | The `/metrics` Prometheus exposition + the JSON log shape are stable; a real dashboard imports both |
| Multi-tenancy / rate limiting | Requires a per-client identity model + a rate-limit store | Add a middleware that reads `X-Client-Id` or an auth token; gate on Redis or in-memory token-bucket |
| Model-version UI | MLflow's bundled UI + the CLI is sufficient for portfolio scope | Start the MLflow tracking server (`mlflow ui --backend-store-uri ...`) against the same SQLite |

Every deferred concern is one we know how to add, not one we don't understand. The choice is depth over breadth.

## Further Reading

Deep dives that this document points at:

- [`dependency-strategy.md`](dependency-strategy.md) — two-lockfile pattern, pip-tools workflow, hash-locked installs
- [`volume-mount-strategy.md`](volume-mount-strategy.md) — bind-mount registry, artifact-root override, cross-host portability
- [`registry-strategy.md`](registry-strategy.md) — aliases vs Stages, direct file-open vs flavor wrappers, MODEL_ALIAS override
- [`monitoring.md`](monitoring.md) — `/metrics` catalog, sample PromQL, drift-signal log shape, four golden signals mapping

Companion operator + extension docs:

- [`deployment-runbook.md`](deployment-runbook.md) — clone-to-running step-by-step
- [`adding-a-new-model.md`](adding-a-new-model.md) — extension pattern for a 6th model
- [`../README.md`](../README.md) — deployment-phase overview + Phase Progress
- [`../services/sklearn-svc/README.md`](../services/sklearn-svc/README.md), [`../services/pt-svc/README.md`](../services/pt-svc/README.md), [`../services/tf-svc/README.md`](../services/tf-svc/README.md) — per-service operator how-tos
