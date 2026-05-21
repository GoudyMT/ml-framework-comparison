# Reflections + Future Work

The retrospective companion to this portfolio. Four months of building 20 models across 4 frameworks and shipping 5 of them as production-shaped FastAPI services produces a lot of evidence about what worked, what would evolve next, and which engineering instincts are worth carrying into future projects. This doc records that retrospective for hiring managers asking "does the author know what they don't know?", for engineers forking the layout wondering "what would I want to know before building on this?", and for the original author returning later with the same question.

The "Areas to Evolve" section is grounded in a comprehensive cross-cutting examination of the codebase — modeling code, deployment services, documentation, infrastructure + tooling, and cross-cutting concerns. Each item is a real observation paired with what it would unlock next.

Companion: [`DECISIONS.md`](DECISIONS.md) — the "why" behind every model, framework, and architectural choice.

## Table of Contents

- [What Worked Well](#what-worked-well)
- [Areas to Evolve](#areas-to-evolve)
- [Engineering Practices Worth Carrying Forward](#engineering-practices-worth-carrying-forward)
- [Lessons Learned](#lessons-learned)
- [Reading Order for Newcomers](#reading-order-for-newcomers)

---

## What Worked Well

The patterns that consistently held up under independent review. These are the things I'd want to do again in any future ML/SWE project.

- **Honest negative results documented as portfolio assets.** V5 PER did not replicate Schaul 2016 (0/3 LunarLander seeds vs V4's 2/3). V4 GIN underperformed V1 GCN on arxiv (-2.7pp). ViT V3 distillation landed -12.75pp below CNN #11. V4 VQ-VAE failed three times before working. The project rule was to surface these, not bury them — each negative result teaches more than the wins it sits next to.

- **Decision-tabled deep-dive docs.** The four deployment-phase strategy docs ([`dependency-strategy.md`](deployment/docs/dependency-strategy.md), [`registry-strategy.md`](deployment/docs/registry-strategy.md), [`volume-mount-strategy.md`](deployment/docs/volume-mount-strategy.md), [`monitoring.md`](deployment/docs/monitoring.md)) present every non-obvious choice as a side-by-side options matrix with pro/con reasoning. This is the strongest documentation pattern in the set — internal-design-doc quality, not learner's-project quality.

- **Three-tier health taxonomy in observability.** `/health` (process alive), `/ready` (service can accept traffic), `/health/<model>` (per-model freshness within an operator-tunable threshold). Each answers a distinct orchestrator question with grep-friendly + machine-parseable failure detail strings (`model_not_loaded`, `inference_stale`). Conflating any two would leave a real failure mode untracked.

- **Hermetic test boundary + real-data smoke separation.** `tests/` exercise routes, middleware, validation, and serialization end-to-end without MLflow or model artifacts on disk via `FakeX` fixtures that subclass real `nn.Module` (so `eval()` and `torch.no_grad()` behave correctly). Real-data parity is a separate `scripts/smoke_test.py` per service that runs against a live process. Two layers, two purposes, neither contaminating the other.

- **Supply-chain hygiene end-to-end.** Hash-locked installs (`--require-hashes`) enforced in both Dockerfile and CI. `torch.load(weights_only=True)` consistent across all PyTorch loaders with explicit security comments. Non-root UID 1000 containers, JSON-array CMD for clean SIGTERM delivery, stdlib-urllib HEALTHCHECK avoiding adding `curl` as a dependency. Pinning is half of supply-chain hygiene; the explicit `weights_only=True` is the other half that often gets skipped.

- **Cross-framework comparison rigor (structurally).** Identical preprocessed `.npy` artifacts under `data/processed/<model>/` consumed by every framework via one `load_processed_data()` call. Same train/test split, same scaler params, same metric implementations. The comparison tables ("PT 80.0% vs TF 79.5%") actually mean something because the comparison machinery is real.

- **Pedagogical comment quality.** Every loader, router, schema, and middleware file opens with a WHAT/WHY block, and inline comments explain *intent* (e.g., the `weights_only=True` security note; the `eval()`/`no_grad()` distinction; lazy weight materialization in TF). Reads as on-ramp documentation for future maintainers, not noise — the project rule was "comments as instructions to someone who has never read code before" and it held across 47 notebooks and 50+ deployment source files.

---

## Areas to Evolve

The next directions. None of these are blockers for the v1.0.0 release; all are honest pointers to what would graduate this from "production-shaped" to "production-grade" if the work continued.

### 1. Deterministic-ops regime for cross-framework reproducibility claims

The "bit-identical Q-table across PT/TF" claim holds because the V1 algorithm is pure NumPy. Most other GPU-touching models lack `torch.use_deterministic_algorithms(True)` / `tf.config.experimental.enable_op_determinism()`. Reproducibility is documented as "3-seed reporting" (Henderson 2018-style honesty) but not pinned by determinism flags. Next direction: a `utils/seeding.py` `seed_everything(seed=113)` helper that touches NumPy + torch + TF + Python's `random` + `PYTHONHASHSEED`, plus a "reproducibility regime" section in the root README clarifying which models are seed-deterministic vs only seed-stable-in-distribution.

### 2. Container smoke tests in CI

`ci.yml` runs hermetic pytest only; container smoke tests require a running server + real registry artifacts (`smoke_test.py`). A registry corruption or loader regression that only surfaces at container-runtime is currently caught by manual smoke runs, not by CI. Next direction: a `compose-up + smoke-script-loop + compose-down` job triggered nightly or on PRs touching `services/*/app/services/*_loader.py`. Closes the seam between hermetic unit coverage and full-stack-behavior verification.

### 3. Container security + multi-arch + SBOM + signing

`build.yml` ships the image and stops. A senior ops reviewer expects Trivy or Grype in the pipeline, `cosign` image signing, an SPDX/CycloneDX SBOM uploaded as a release artifact, and multi-arch builds (`platforms: linux/amd64,linux/arm64`) so ARM Mac developers + AWS Graviton / GCP Tau-T2A nodes can run images natively. Next direction: a 20-line `security-scan` job in `build.yml` running `aquasecurity/trivy-action` post-build gated on severity HIGH/CRITICAL, plus `anchore/sbom-action`, plus the `platforms:` argument on the build step (accepts qemu-emulated arm64 cold-build time roughly doubling for tf-svc).

### 4. Dependabot + dependency-review-action

`.github/workflows/` has no automated dependency-update plumbing. Hash-locked deps drift toward stale without it; manual pin bumps don't scale. Next direction: a `.github/dependabot.yml` covering pip + GitHub Actions ecosystems, plus `actions/dependency-review-action` on the PR workflow. Pairs with the existing `--require-hashes` discipline.

### 5. Authentication + rate limiting on `/predict/*` endpoints

Explicitly out-of-scope per [`docs/architecture.md`](deployment/docs/architecture.md), but worth a one-paragraph documentation note in [`deployment-runbook.md`](deployment/docs/deployment-runbook.md) under "Operational caveats": "endpoints are unauthenticated by design; for any non-portfolio deployment, terminate behind an auth proxy or add OAuth2 dependency injection." Next direction: when this fronts real traffic, add a middleware reading `X-Client-Id` or an auth token + a Redis-backed per-client token-bucket rate limiter.

### 6. CORS + TrustedHost + GZip + request-size middleware

A grep across all three services returns zero matches for `CORS|trusted_host|max_request_size|GZip`. PCA accepts a 784-float list; DNN accepts 561; GAN returns up to 16 base64 PNGs. Bounded by Pydantic, but an unbounded JSON parser still allocates before validation runs. Next direction: `starlette.middleware.gzip.GZipMiddleware` for the response side and a custom request-body-size limit middleware ordered before `RequestIDMiddleware`.

### 7. Structured error responses (RFC 7807 problem+json)

503 returns `{"detail": "model_not_loaded"}`; 422 returns Pydantic's default error array; client-facing error shape varies by status code. Next direction: a `RFC 7807 problem+json` error envelope (`type`, `title`, `status`, `detail`, `instance`) for consistency across status codes — a senior ops engineer hooking this into a multi-service mesh will appreciate uniform error shape.

### 8. Loader boilerplate consolidation

The 5 loaders contain ~150 lines of duplicated boilerplate (`_default_tracking_uri`, `_file_uri_to_path`, `_resolve_artifact_dir`, `_resolve_alias`, idempotency guard, `MlflowClient.get_model_version_by_alias` + source-URI check). A `_BaseLoader` helper or a `RegistryArtifactResolver` class would reduce surface area + lock in consistency. Acknowledged trade-off: the duplication keeps each loader readable in isolation; centralizing would tighten the cross-loader contract.

### 9. Typed loader returns (Protocol-driven)

`get_pca_model() -> Any`, `get_scaler() -> Any`, `get_tokenizer() -> Any` leak `Any` into router handlers. The `pt-svc` `_HealthLoader` Protocol pattern shows the direction: declare a `PCAEstimator` Protocol (just `.transform` + `.explained_variance_ratio_`) and a `Scaler` Protocol; cascades type safety into routers without changing runtime behavior. Next direction: a small `app/services/protocols.py` per service declaring the consumer-side contracts.

### 10. Per-process freshness/sampling state vs multi-worker reality

`inference_tracking._LAST_INFERENCE_TS` and `input_distribution._COUNTERS` are documented as "single-worker assumption" (correctly), but production Uvicorn typically runs `--workers N`. Two reasonable next directions: a Redis-backed shared store for cross-worker freshness consensus, or accept the desync and lean on Prometheus counters as the cluster-level signal (the comments already hint at the latter being acceptable for a freshness check).

### 11. Distributed tracing (OpenTelemetry)

`request_id` middleware generates a UUID per request, but propagation across services never happens today because the services are independent. The moment a 6th model adds an upstream-call dependency (feature store, embedding service), tracing becomes mandatory. Next direction: `opentelemetry-instrumentation-fastapi` wiring + OTLP exporter is one focused PR; mark as a prerequisite for any multi-service composition.

### 12. SDK generator pipeline

Every endpoint carries explicit `operation_id` (clean Python/TypeScript SDK method names: `client.predictPca()`, `client.translate()`, etc.), but no generator pipeline is wired. Next direction: an `openapi-generator-cli` invocation in `build.yml` producing typed client SDKs as workflow artifacts; downstream consumers `pip install` (or `npm install`) the generated client instead of hand-writing `requests` / `fetch` calls against `/openapi.json`.

### 13. Cloud deployment (replace `deploy.yml` placeholder)

`deploy.yml` is a documented placeholder demonstrating the workflow shape (inputs, permissions, deploy plan summary) but performs no actual deploy. Next direction: replace the placeholder body with a real target (kubectl apply / Cloud Run / ECS task-definition update / Fargate); the trigger + auth + image-tag-selection scaffolding is already in place.

### 14. Drift computation pipeline

The `input_distribution_sample` structlog event emits whole-vector summary stats every Nth request per PCA + DNN. A downstream pipeline that consumes the log stream and computes PSI / KS over rolling windows + alerts on threshold breach is the natural next step — the upstream signal is the harder part, and that already exists. Next direction: a small Python service or notebook periodically pulling logs (Loki query / CloudWatch Insights query), computing the divergence metric, posting to Prometheus pushgateway.

---

## Engineering Practices Worth Carrying Forward

Disciplines from this project that should travel to future work.

- **Decision-tabled deep-dive docs alongside operator how-tos.** For every non-obvious architectural choice, write an internal-design-doc-style markdown file with the alternatives considered + the trade-off accepted. Operators get a separate how-to file. Two audiences, two doc shapes — don't conflate.

- **Verification before completion.** No claim of "tests passing" without running the test command in the same message. No claim of "linter clean" without running it. Evidence before assertions, always. CI exists; running it locally before commit is cheaper than the round-trip.

- **Per-decision rationale captured in source.** Every Dockerfile line earns its place (the `--chown` during COPY vs separate `chown -R`; the JSON-array CMD; the stdlib-urllib HEALTHCHECK avoiding `curl`); every `pyproject.toml` pin earns its comment (the `+cpu` torch wheel divergence; the `disallow_subclassing_any = false` carve-out in tf-svc). When a future engineer reads "why is this here?", the answer is in the file.

- **Two-lockfile cross-platform pattern (or its equivalent).** Hash-locked installs only work if the platform-specific wheel set actually resolves. Maintaining `requirements.txt` for Linux + `requirements-windows.txt` for host dev (or whatever the platform split is) keeps `--require-hashes` enforceable everywhere. The bookkeeping cost is trivial via `pip-compile`.

- **Hermetic tests + real-data smoke separation.** Unit tests should never depend on registry/network/disk. Real-data parity is a separate smoke step. Mixing the two produces brittle CI and slow developer feedback loops.

- **Honest negative results as portfolio assets.** Publish what didn't work, not just the wins. "V5 PER does not replicate Schaul 2016 (0/3 seeds)" is a stronger portfolio signal than another marginal improvement on a happy path. This is the discipline that separates "good portfolio" from "hire-this-engineer".

---

## Lessons Learned

The surprises from the 4-month journey. Things I didn't know going in.

- **Framework choice is cosmetic when the algorithm is deterministic.** For Linear Regression / KNN / K-Means / Naive Bayes / PCA / SVM / Q-Learning V1, all 4 frameworks produce numerically identical (or bit-identical) outputs. Differences emerge only when stochastic initialization + non-deterministic optimizer paths + floating-point order-of-operations compound across thousands of training steps. The portfolio's strongest claim is that "framework choice doesn't affect quality" *for the right class of algorithm* — backed by parity tables, not assertion.

- **Documentation drift accumulates faster than code drift.** The deployment phase landed ~12 markdown docs across 3 services. The hardest discipline wasn't getting them right initially — it was keeping them in sync as the underlying code evolved. Independent cross-references caught drift more reliably than self-review, because writers anchor on what they intended, not what they shipped.

- **Pre-training dominates on small data, even for "more expressive" architectures.** ViT V4 (pre-trained ImageNet-21k -> CIFAR-100) hit 91.16%, beating V1-V2-V3 combined (+17.15pp via vanilla -> DeiT -> distillation). ViT's lack of convolutional inductive bias is a real cost without pretraining. The lesson generalizes: architectural ambition without scale-matched pre-training underperforms simpler architectures that fit the dataset's complexity.

- **The cheapest improvement is usually the one you didn't try yet.** Adding beam search to the Transformer (#16) bought +0.0164 BLEU with zero retraining. CutMix dropped CNN #11 overfitting from 22% to 5% gap. SGD with momentum gave +12.8% over Adam on ResNet-20 because Adam interferes with BatchNorm dynamics. The high-leverage moves are often boring training-recipe choices, not architectural reinvention.

- **`Any`-typed boundaries leak everywhere they touch.** A loader returning `Any` makes every router handler that calls it `Any`-tainted. Protocol types are the right tool — declare the structural contract at the boundary, mypy strict enforces it across consumers. The `_HealthLoader` Protocol pattern in pt-svc was a quiet but high-impact refactor.

- **"Production-shaped" earns its qualifier through the boring details, not the headline ones.** Multi-stage Docker, hash-locked installs, non-root containers, JSON-array CMD for signal forwarding, three-tier health endpoints, bounded Prometheus label cardinality, byte-identical middleware copies with explicit drift-prevention discipline — none of these are individually impressive, but their *together* is what distinguishes a portfolio service from a tutorial demo.

---

## Reading Order for Newcomers

Where to start if you're reading this from cold.

**Hiring managers** with 5-10 minutes:
1. Root [`README.md`](README.md) — headline + 20-model table + 5-deployment summary
2. [`DECISIONS.md`](DECISIONS.md) — the "why" behind each model, framework, and architectural choice (skim Models, read Frameworks + Deployment Decisions in full)
3. This `REFLECTIONS.md` — self-awareness check; "Areas to Evolve" shows what the author knows would improve next

**Engineers** forking the layout for their own work:
1. [`deployment/docs/architecture.md`](deployment/docs/architecture.md) — service topology + cross-cutting patterns
2. [`deployment/docs/adding-a-new-model.md`](deployment/docs/adding-a-new-model.md) — extension pattern walking through a 6th model
3. The four deployment strategy docs ([`dependency-strategy.md`](deployment/docs/dependency-strategy.md), [`registry-strategy.md`](deployment/docs/registry-strategy.md), [`volume-mount-strategy.md`](deployment/docs/volume-mount-strategy.md), [`monitoring.md`](deployment/docs/monitoring.md))
4. The "Areas to Evolve" section above — known limitations + what would address them

**Operators** running the services:
1. Per-service READMEs at [`deployment/services/sklearn-svc/README.md`](deployment/services/sklearn-svc/README.md), [`pt-svc/README.md`](deployment/services/pt-svc/README.md), [`tf-svc/README.md`](deployment/services/tf-svc/README.md)
2. [`deployment/docs/deployment-runbook.md`](deployment/docs/deployment-runbook.md) for the clone-to-running walkthrough

**ML engineers** wanting cross-framework comparisons:
1. [`docs/modeling/cross-framework-findings.md`](docs/modeling/cross-framework-findings.md) — parity results + speed/memory hierarchy
2. The per-framework deep dives ([`pytorch.md`](docs/modeling/pytorch.md), [`tensorflow.md`](docs/modeling/tensorflow.md), [`scikit-learn.md`](docs/modeling/scikit-learn.md), [`no-framework.md`](docs/modeling/no-framework.md))
3. The "Lessons Learned" section above — surprises that didn't make it into the per-model writeups
