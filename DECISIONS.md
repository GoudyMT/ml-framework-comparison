# Decisions + Reasoning

The "why" companion to this portfolio. Every model picked, every framework chosen, every deployment-phase decision recorded with the alternatives considered and the trade-offs accepted. For hiring managers evaluating the portfolio's engineering judgment, for engineers forking the layout for their own work, and for the original author returning later wanting to know "why did I do it that way?"

Three top-level sections: **Models** (the 20 implementations + what each taught), **Frameworks** (the 4 framework runtimes + where each won or lost), **Deployment Decisions** (~10 architecture decisions made during the production-shaping phase). A short **Tools** + **Standards** section closes the doc. Reading order pointers at the bottom for different audiences.

Companion: [`REFLECTIONS.md`](REFLECTIONS.md) — retrospective on what worked and what would evolve next.

## Table of Contents

- [Models (20)](#models-20)
- [Frameworks (4)](#frameworks-4)
- [Deployment Decisions (10)](#deployment-decisions-10)
- [Tools](#tools)
- [Standards + Reproducibility](#standards--reproducibility)
- [Reading Order for Newcomers](#reading-order-for-newcomers)

---

## Models (20)

20 models across 4 paradigms (supervised, unsupervised, generative, reinforcement). Each ships with metrics + per-framework comparisons; the headline below is the strongest result + the most-portable lesson.

### #01. Linear Regression

- **Problem**: predict vehicle prices from numeric + categorical features
- **Frameworks**: all 4 implemented identically (R² ≈ 0.50, RMSE ≈ $10,100)
- **Headline**: framework choice doesn't matter when the algorithm is closed-form; sklearn wins on dev speed (0.03s vs PT 3.4s) because there's no neural network overhead to amortize

### #02. Logistic Regression

- **Problem**: credit-card fraud detection (binary, severe class imbalance)
- **Frameworks**: all 4 land at ~83% recall on fraud class
- **Headline**: sklearn L-BFGS converges in 0.32s (57× faster than NF SGD); the autograd payoff starts showing on PyTorch (7.8× over NF)

### #03. KNN

- **Problem**: Forest Cover Type (581K samples, 7 classes)
- **Frameworks**: all 4 hit 93.77% identically (algorithm is deterministic)
- **Headline**: sklearn's KD-tree (O(log n) lookups) beats PyTorch GPU brute-force (2,000 pred/sec vs 1,164) — algorithmic intelligence > hardware muscle at this scale

### #04. K-Means

- **Problem**: Dry Bean clustering (7 cultivars, 13 morphological features)
- **Frameworks**: all 4 identical (inertia ~9,976, ARI ~0.6684)
- **Headline**: sklearn wins on wall-clock (0.06s) by 5.7× over PT GPU — kernel launch overhead doesn't pay off at 10K samples

### #05. Naive Bayes

- **Problem**: 20 Newsgroups text classification (20 classes, TF-IDF features)
- **Frameworks**: all 4 identical (accuracy 0.6683, log-loss 1.5576)
- **Headline**: PT GPU fastest at 3.5 µs/sample inference; sklearn's `CalibratedClassifierCV` cuts ECE from 0.32 -> 0.14 — same accuracy, much honester probabilities

### #06. Decision Trees / Random Forest

- **Problem**: heart-disease prediction (Cleveland UCI, 14 features)
- **Frameworks**: F1 ≈ 0.48 across all 4
- **Headline**: sklearn won via `GridSearchCV` + MLflow tracking + joblib persistence — the production-ready combo NF/PT/TF can't match without rebuilding the tooling layer

### #07. SVM

- **Problem**: MAGIC Gamma telescope (binary signal/background, 19 features)
- **Frameworks**: all 4 converge (F1 ~0.90, AUC ~0.91)
- **Headline**: PT GPU dual gradient descent 17.7× faster than NF (9.03s vs 160s); sklearn has the best calibration AUC (0.9164 via Platt scaling)

### #08. PCA — D1 deployment winner

- **Problem**: Fashion-MNIST dimensionality reduction (784 -> 150 components)
- **Frameworks**: all 4 identical (0.9085 explained variance, 0.0951 reconstruction MSE)
- **Headline**: PT GPU eigendecomposition fastest (0.11s fit, 9.1× CUDA LAPACK speedup); sklearn wins for deployment because the registered .joblib artifact is the simplest production wrapper
- **Deployed as D1 in `sklearn-svc`** — flat 784-pixel input -> 150-component vector, sub-millisecond inference

### #09. DNN — D2 deployment winner

- **Problem**: UCI HAR human-activity recognition (561 features, 6 activities)
- **Frameworks**: PT GPU 96.03% best; SK MLPClassifier 95.04%; TF 92.70% (lost on same architecture due to CPU eager dispatch overhead)
- **Headline**: regularization is the differentiator — BatchNorm + Dropout + ReduceLROnPlateau LR scheduling gives the same 256-128 architecture +1.0% over plain. SK's MLPClassifier cannot express these structurally
- **Deployed as D2 in `pt-svc`** — 561-feature input -> softmax over 6 activities + Literal-typed `predicted_label`

### #10. Autoencoders

- **Problem**: MNIST reconstruction (clean + denoising)
- **Frameworks**: PT GPU MSE 0.0037 vs SK 0.0133 (3.6× better) at comparable param count
- **Headline**: conv-based architecture dominates dense; conv denoising AE removes 86.9% of σ=0.2 noise; reconstruction quality ≠ classification quality (denser bottlenecks force semantic compression)

### #11. CNN

- **Problem**: CIFAR-100 (100 fine classes, 32×32 RGB)
- **Frameworks**: PT GPU only (TF dropped due to oneDNN compile latency)
- **Headline**: ResNet-20 + CutMix + Label Smoothing + Nesterov SGD -> 80.1% via 10 systematic experiments (56.9% -> 80.1%); SGD with momentum the single biggest gain (+12.8% vs Adam) because Adam interferes with BatchNorm dynamics
- **Lesson**: cosine annealing must complete its full cycle — early-stopping cosine LR is counterproductive

### #12. RNN

- **Problem**: ECG5000 arrhythmia classification (5 classes, 121.6× class imbalance)
- **Frameworks**: PT GPU GRU-128, 91.8% accuracy / 0.55 macro F1
- **Headline**: **macro F1 is the only honest metric** — 91.8% accuracy looks great but macro F1 0.55 reveals failure on 3 of 5 classes (PVC 19 samples, SP 39, UB 5 training); no architecture fixes a data problem

### #13. LSTM

- **Problem**: dual-dataset (ECG5000 arrhythmia + IMDB sentiment)
- **Frameworks**: PT GPU LSTM-128
- **Headline**: ECG augmentation (jitter, scaling, time warp) drove 85% of total improvement — augmentation > architecture for imbalanced sequences; LSTM vs GRU adds only +0.008 macro F1 at 140 timesteps (gating matters more for longer dependencies); IMDB hits 87.8% accuracy / 0.946 AUC

### #14. GANs — D3 deployment winner

- **Problem**: CIFAR-10 generative modeling
- **Frameworks**: PT GPU, 4 variants on the same data (Vanilla MLP FID 261, DCGAN 30.57, WGAN-GP 55, cGAN 148)
- **Headline**: convolutional architecture dominates (DCGAN 8.5× better FID than MLP); BCE loss beats Wasserstein at fixed budget; **for deployment, simple DCGAN + BCE is the pragmatic choice**
- **Deployed as D3 in `pt-svc`** — server-side noise sampling, 1-16 images per call, base64-PNG response

### #15. Attention

- **Problem**: Tatoeba EN->ES translation (143K pairs, word-level tokenization)
- **Frameworks**: PT GPU, 4 attention variants
- **Headline**: Bahdanau additive attention wins (BLEU 0.380) over Luong (0.297) and Multi-Head (0.368) — **pre-GRU context injection beats post-GRU** because the decoder gets full 1024-dim context BEFORE the state update. Reproduces Bahdanau et al. (2014)
- **Lesson**: BLEU 0.38 is "gist quality" — production MT needs BPE + deeper models + beam search + 100× more data, all addressed in #16

### #16. Transformers — D5 deployment winner

- **Problem**: two tasks (Tatoeba EN->ES translation + AG News 4-class classification)
- **Frameworks**: TF Recipe BLEU 0.4456 wins translation (vs PT 0.3625); PT only for classification (DistilBERT fine-tuned 94.45%)
- **Headline**: **TF beats PT on translation by +0.0831 BLEU** on identical architecture/data/hyperparams — random init + optimizer epsilon defaults compound over thousands of training steps; built from `nn.Linear` only (no `nn.Transformer` / `nn.MultiheadAttention` shortcuts); SentencePiece BPE eliminates the 2.6%-7.3% UNK rates from #15's word-level
- **Deployed as D5 in `tf-svc`** — text input, greedy decode (encode-once + decode-many), oneDNN warm-up at startup

### #17. Vision Transformers

- **Problem**: CIFAR-100 (same dataset as #11 for direct CNN-vs-ViT comparison)
- **Frameworks**: PT GPU, 4 variants
- **Headline**: V1 Vanilla 50.33% -> V4 Pre-trained ViT-B/16 91.16%. **Pre-training dominates on small data** — V4's +23.68pp jump from V3 is larger than V1->V2->V3 combined. V3 distillation (67.48%) lands -12.75pp below CNN #11 — **ViT's lack of convolutional inductive bias is a real cost without pretraining; honest portfolio finding**
- **Note**: V4 weights deleted during git cleanup (327 MB exceeded GitHub's 100 MB limit); deployment uses V3

### #18. GNN

- **Problem**: dual-dataset (Cora citation network + ogbn-arxiv 169K nodes)
- **Frameworks**: PT GPU, 4 variants
- **Headline**: V3 GAT wins both (Cora 0.8310 matches Velickovic 2018 exactly; arxiv OGB 0.7025); V4 GIN underperforms V1 GCN by 2.7pp on arxiv — **more theoretical expressiveness doesn't auto-transfer to node classification**. Per-class F1 tracks edge homophily monotonically (high-homophily classes hit F1 ~0.90; low-homophily class with 29 papers gets F1=0.00)

### #19. VAE

- **Problem**: dual-dataset (MNIST + CIFAR-10), 5 variants culminating in VQ-VAE + PixelCNN prior
- **Frameworks**: PT GPU
- **Headline**: **the FID ladder is the headline result** — GAN #14 (30.57) < V4 VQ-VAE reconstruction (97.81) < V5 PixelCNN-sampled (117.27) < V2 prior (165) < V4 random codes (202). V5 - V4 random = -85 FID is the learned-prior contribution. **This is the two-stage recipe that powers DALL-E 1, Stable Diffusion, Muse, EnCodec**
- **Lesson**: V4 VQ-VAE failed 3 times before working (gradient-killing detach, codebook collapse, scale mismatch) — each diagnosis required tracing actual forward/backward, not docstring pattern-matching

### #20. Q-Learning — D4 deployment winner

- **Problem**: 3 environments (Taxi-v4 tabular + CartPole DQN family + LunarLander dueling/PER)
- **Frameworks**: PT only for V2-V5 (TF for V1 parity); 5 variants
- **Headline**: V1 PT vs TF is **bit-identical** (0.00% max relative diff across all 3,000 Q-table entries); V2 DQN perfect 3/3 seeds on CartPole; V3 Double DQN's overestimation fix replicates cleanly (-32%); **V5 PER does NOT replicate Schaul 2016** (0/3 seeds; honest negative result, Henderson 2018 in miniature)
- **Deployed as D4 in `pt-svc`** — V1 Q-table is the smallest deployment artifact (`(500, 6)` numpy array, ~12 KB)

---

## Frameworks (4)

Four runtimes, each ran most or all of the 20 models. The comparison is the point — same dataset, same train/test splits, same metrics across implementations forces honest framework differentiation.

### No-Framework (pure NumPy)

- **Range**: #01-#08 only. Retired after PCA (the boundary where from-scratch builds stop teaching and start slowing down)
- **Wins**: foundational understanding. Every classifier and optimizer hand-implemented; eigen vs SVD numerical equivalence shown in #08; manual OOB error in #06; from-scratch dual gradient ascent in #07
- **Loses**: scaling. No autograd means every backward pass is hand-derived; no GPU acceleration; model sizes balloon (DT/RF 55 MB vs PT/TF tensor representations at 29 MB)
- **Insight**: "from-scratch builds solidify fundamentals but scale poorly for complex models" — the #08 retirement boundary is real, not arbitrary

### Scikit-Learn

- **Range**: #01-#10 only. Retired after Autoencoders (sklearn's MLPClassifier hits its expressiveness ceiling without BatchNorm/Dropout/LR scheduling)
- **Wins**: classical ML production-readiness. `GridSearchCV` + `Pipeline` + `joblib` + `CalibratedClassifierCV` are the tools other frameworks rebuild. Best calibration AUC on SVM (0.9164); ECE 0.32 -> 0.14 on Naive Bayes; first deployment win (D1 PCA) because `.joblib` is the simplest production wrapper
- **Loses**: modern deep learning. No first-class autograd; MLPClassifier can't express BatchNorm + Dropout; no GPU acceleration in sklearn proper
- **Insight**: "high-level frameworks accelerate development for standard tasks but hide mechanics" — speed-vs-understanding trade-off

### PyTorch

- **Range**: all 20 models. Won deployment for 14 of 17 deployable artifacts
- **Wins**: dynamic graphs cut debug time on custom architectures (GNN sparse adjacency, RL replay loops, VAE masked-conv causality); eager mode is 14-24× faster than TF eager for tight per-step loops (12 min vs 4 hr for VAE/RL training); `nn.Module` subclassing makes from-scratch reimplementation idiomatic — Transformer/ViT/GAT/VQ-VAE/DQN-family all built from primitives, no high-level shortcuts
- **Loses**: TF beats PT on #16 translation by +0.0831 BLEU on identical setup (random init + optimizer epsilon defaults compound); torch.compile limited on Windows in 2.5.1 (TorchInductor backend incomplete)
- **Insight**: dominant deep-learning framework for the portfolio because the speed + debuggability stack lets the writer focus on architecture, not framework friction

### TensorFlow

- **Range**: #01-#20 with WSL2 scope reductions on 5 models. Won D5 (Translation)
- **Wins**: `@tf.function` graph mode reaches PT GPU parity once compile cost is amortized; Keras callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`) are first-class; cuDNN integration in WSL2 gives competitive deep-learning speeds; **+0.0831 BLEU win on #16 translation** is the portfolio's clearest TF-beats-PT result
- **Loses**: eager mode is 14-24× slower than PT eager for tight loops (#19 VAE 156s vs 11s; #20 Q-Learning 288 min vs 12 min); Windows-native install painful (WSL2 required for cuDNN); per-step dispatch overhead unavoidable in eager
- **Insight**: stays portfolio-relevant for parity verification + the specific corners where Keras tooling or `@tf.function` graph mode genuinely win

---

## Deployment Decisions (10)

The deployment phase shapes the 5 representative models into production-grade FastAPI services. Each decision below records the alternatives considered + the trade-off accepted.

### 1. FastAPI vs Flask vs Django vs Starlette

- **Chose**: FastAPI. Async-native, Pydantic v2 integration, OpenAPI auto-generation from schema, built-in dependency injection
- **Considered**: Flask (mature but no async-first), Django (heavy for an API-only service), Starlette (FastAPI's own foundation — too low-level for the schema/docs combo we wanted)
- **Trade-off accepted**: FastAPI's async-first design adds complexity over Flask's synchronous simplicity; we don't currently need async I/O in the model-forward path, but the framework keeps the door open

### 2. Pydantic v2 vs Marshmallow vs attrs

- **Chose**: Pydantic v2. Single schema definition becomes runtime validator + mypy type + OpenAPI surface
- **Considered**: Marshmallow (more flexibility but no static-typing integration), attrs (excellent dataclass alternative but no validation/serialization built-in)
- **Trade-off accepted**: Pydantic v2's strict types can require workarounds for legitimate Any cases (e.g., loader modules); the `Protocol` pattern works around that cleanly

### 3. MLflow Model Registry vs flat files vs DVC

- **Chose**: MLflow Model Registry as the source of truth for "which version of which model is production"
- **Considered**: flat-file registry per service (simplest, but no version history or rollout primitives), DVC (excellent for data versioning, less natural for model promotion workflows)
- **Trade-off accepted**: MLflow's flavor/log_model contract is bypassed in favor of `MlflowClient.get_model_version_by_alias` + direct file-open (joblib/torch.load/np.load/tf.load_weights/spm.Load) — captured in [`deployment/docs/registry-strategy.md`](deployment/docs/registry-strategy.md)

### 4. MLflow aliases vs Stages

- **Chose**: aliases (`@production`, `@candidate`). MLflow 3.x deprecated the legacy `current_stage` Staging/Production/Archived enum
- **Considered**: legacy Stages (still supported but deprecated; would require migration if used)
- **Trade-off accepted**: nothing real — the portfolio used aliases from day one, so the migration burden is zero. `MODEL_ALIAS` env var lets operators override at deploy time (e.g., `MODEL_ALIAS=candidate` for a canary)

### 5. Multi-stage Docker vs single-stage

- **Chose**: multi-stage builds (builder + runtime). `--require-hashes` pip install in builder; only `/opt/venv` crosses into runtime
- **Considered**: single-stage (simpler Dockerfile, larger image)
- **Trade-off accepted**: multi-stage adds Dockerfile complexity but final images are 30-40% smaller. Builder-stage tooling (compilers, pip cache, build artifacts) never ships. Non-root UID 1000 in runtime drops privileges. JSON-array CMD ensures SIGTERM forwarding for graceful shutdown

### 6. Two-lockfile pattern (pip-tools) vs Poetry vs single lockfile

- **Chose**: pip-tools with two `requirements*.txt` per service — Linux for Docker, Windows for host dev
- **Considered**: Poetry (cross-platform lock issues in poetry.lock metadata; bundled venv manager conflicts with existing .venv convention), single Linux lockfile (host-side dev installs would fall back to non-hash-locked resolution on Windows)
- **Trade-off accepted**: maintaining two lockfiles per service; documented + scripted regeneration via `pip-compile`. Full rationale in [`deployment/docs/dependency-strategy.md`](deployment/docs/dependency-strategy.md)

### 7. structlog vs std logging

- **Chose**: structlog with JSON renderer. Per-request `request_id` bound via contextvars
- **Considered**: stdlib `logging` (works, no structured output by default), loguru (third-party, less integration with structured-event patterns)
- **Trade-off accepted**: structlog adds one runtime dep; payoff is that every log line is JSON-parseable for downstream aggregators (Loki, CloudWatch, etc.) without a parser, and request lifecycle events carry the `request_id` for free

### 8. Prometheus metrics design (golden signals + per-model histogram)

- **Chose**: three HTTP-layer metrics (`http_requests_total`, `http_request_duration_seconds`, `http_requests_in_flight`) cover SRE's four golden signals; one per-model metric (`model_inference_duration_seconds{model_name}`) isolates model time from request handling
- **Considered**: StatsD (text protocol vs Prometheus pull model), custom (reinvent the wheel)
- **Trade-off accepted**: bounded label cardinality discipline (`method`, `status`, `path`, `model_name` — never user IDs or request bodies); `model_name` carries 5 known values across the portfolio (safe for prod). Full design in [`deployment/docs/monitoring.md`](deployment/docs/monitoring.md)

### 9. /health vs /ready vs /health/<model> separation

- **Chose**: three separate endpoints. `/health` = process alive (HEALTHCHECK target), `/ready` = service can accept traffic (load-balancer gating), `/health/<model>` = per-model freshness (model-loaded AND last activity within threshold)
- **Considered**: a single `/health` that conflates all three concerns
- **Trade-off accepted**: three endpoints to maintain instead of one. Worth it because each answers a distinct orchestrator question (restart? route traffic? alert on staleness?); conflating any two leaves a real failure mode untracked. Staleness threshold tunable via `MODEL_INFERENCE_STALENESS_SECONDS`

### 10. GitHub Actions CI/CD vs alternatives

- **Chose**: GitHub Actions. Matrix-parallel lint + mypy + pytest across 3 services; build/push to ghcr.io on push to main; workflow_dispatch deploy placeholder
- **Considered**: GitLab CI (repo is on GitHub), CircleCI / Buildkite (external service to configure + bill), local-only (no continuous verification)
- **Trade-off accepted**: tied to GitHub. Workflow-scoped `GITHUB_TOKEN` handles both git operations and ghcr.io pushes — no PAT secrets to manage. Per-service pip-wheel cache via `actions/cache@v4` keyed on lockfile hash. Layer cache via `type=gha` scoped per-service for ~3× cold-build speedup

---

## Tools

- **Python 3.12.2** — pinned exactly across pyproject.toml `requires-python`, Dockerfile `FROM`, CI `actions/setup-python` to match hash-locked wheels (any 3.12.x patch would invalidate hashes)
- **ruff** — lints + formats faster than flake8/black combined; configured per-service to enforce E/F/W + I + B + UP + C4 + SIM rule families
- **mypy --strict** — stricter than the default; `ignore_missing_imports = true` + `disable_error_code = ["import-untyped"]` accept stubless third-party imports (joblib, mlflow, sentencepiece, tensorflow) without per-module overrides
- **pytest + pytest-asyncio + httpx** — async-aware test runner + FastAPI TestClient. Hermetic tests via `FakeX` fixtures; no MLflow registry or model artifacts needed at test time
- **pip-tools** — `pip-compile` regenerates per-platform hash-locked lockfiles from a `pyproject.toml` source manifest; `pip install --require-hashes` enforces supply-chain integrity
- **Docker (multi-stage)** — `python:3.12.2-slim` base; non-root UID 1000; stdlib-urllib HEALTHCHECK (no curl dep added)
- **MLflow** — registry only (aliases + version metadata); not used for experiment tracking in this portfolio (modeling-phase metrics live in per-model notebook output cells)
- **GitHub Actions** — `actions/checkout@v4`, `actions/setup-python@v5`, `actions/cache@v4`, `docker/build-push-action@v6`, `docker/setup-buildx-action@v3`, `docker/login-action@v3`, `docker/metadata-action@v5`

---

## Standards + Reproducibility

- **`RANDOM_STATE = 113`** across all 4 frameworks (matches sklearn convention; not `RANDOM_SEED`)
- **Same dataset per model type** across implementations — proves framework choice doesn't affect output quality for equivalent algorithms (verified bit-identical for #01-#05, #08, #20 V1)
- **Identical train/test splits** for direct metric comparison; same metric definitions (e.g., one canonical F1 macro, not framework-default variants)
- **Hand-typed code** — no auto-completion, no copy-paste. Forces real understanding; surfaces bugs at write-time, not at debug-time
- **Frequent commits** with meaningful messages — repo treated as professional work; commit history is part of the portfolio
- **Best-comment practices** — first occurrence of a concept gets full explanation, repeats get brief reminders, obvious operations get nothing; comments written as instructions to someone who has never read code before
- **Verification before completion** — gate runs (ruff + mypy + pytest) live in CI and run locally before any commit claims "passing"; no completion claims without fresh evidence

---

## Reading Order for Newcomers

**Hiring managers** with 5-10 minutes:
1. Root [`README.md`](README.md) for headline + 20-model table + 5-deployment summary
2. This `DECISIONS.md` for the "why" behind each choice (skim Models, read Frameworks + Deployment Decisions in full)
3. [`REFLECTIONS.md`](REFLECTIONS.md) for "what would evolve next" — signals self-awareness + planning rigor

**Engineers** forking the layout for their own work:
1. [`deployment/docs/architecture.md`](deployment/docs/architecture.md) — service topology + cross-cutting patterns
2. [`deployment/docs/adding-a-new-model.md`](deployment/docs/adding-a-new-model.md) — extension pattern walking through a 6th model
3. The four deployment strategy docs ([`dependency-strategy.md`](deployment/docs/dependency-strategy.md), [`registry-strategy.md`](deployment/docs/registry-strategy.md), [`volume-mount-strategy.md`](deployment/docs/volume-mount-strategy.md), [`monitoring.md`](deployment/docs/monitoring.md)) for per-decision deep dives

**Operators** running the services:
1. Per-service READMEs at [`deployment/services/sklearn-svc/README.md`](deployment/services/sklearn-svc/README.md), [`pt-svc/README.md`](deployment/services/pt-svc/README.md), [`tf-svc/README.md`](deployment/services/tf-svc/README.md)
2. [`deployment/docs/deployment-runbook.md`](deployment/docs/deployment-runbook.md) for the clone-to-running walkthrough

**ML engineers** wanting cross-framework comparisons:
1. [`docs/modeling/cross-framework-findings.md`](docs/modeling/cross-framework-findings.md) — parity results + speed/memory hierarchy + framework-specific showcase pattern
2. [`docs/modeling/pytorch.md`](docs/modeling/pytorch.md) / [`tensorflow.md`](docs/modeling/tensorflow.md) / [`scikit-learn.md`](docs/modeling/scikit-learn.md) / [`no-framework.md`](docs/modeling/no-framework.md) — per-framework deep dives with per-model findings
