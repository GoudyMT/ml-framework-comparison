# Modeling Findings: TensorFlow

> Implemented #01-#20 with scope reductions from #15 onward. Won deployment for #16 Transformers Translation. Best for production seq2seq; weaker on research-pace workflows requiring fast iteration.

## TensorFlow's role in the portfolio

- **Won deployment for #16 Transformers Translation**: BLEU 0.4456 vs PT's 0.3625 (+17% over the #15 Bahdanau baseline). Keras 3's recipe-style training was actively helpful for production seq2seq
- **Eager mode is 14-24x slower than PT for tight loops**: measured on VAE V1 (1048s vs 73s) and Q-Learning V2 (288 min vs 12 min per seed). Production TF RL (`tf-agents`, `tensorflow/agents`) uses `@tf.function` decorators throughout for this reason — we don't, to maintain "PT-eager parity" for honest comparison
- **WSL2 GPU constraints forced scope drops on 5 models**: ViT V3, GNN V2/V4, VAE V2, Q-Learning V3-V5

## Scope reductions (documented as portfolio findings, not gaps)

| # | Model | Variants Skipped | Reason |
|---|---|---|---|
| 17 | Vision Transformers | V3 Distillation | TF 2.21 on WSL2 fails on Conv2D ops (`No DNN in stream executor`). Forced CPU teacher fallback caused two system crashes even with core affinity capping. cuDNN 9.1.0 runtime vs 9.3.0 compiled mismatch |
| 18 | GNN | V2 GraphSAGE, V4 GIN | Spektral 1.3.1 broken on Keras 3 (list-of-Nones mask bug). Plan pivoted to from-scratch TF primitives for V1/V3 only |
| 19 | VAE | V2 Conv VAE | Same Conv2D cuDNN constraint as ViT V3 |
| 20 | Q-Learning | V3 Double DQN, V4 Dueling DQN, V5 PER | Marginal cross-framework value vs implementation cost. V3 is one-line target-computation change from V2; V4 architectural decomposition; V5 sum-tree priority buffer — each substantial implementation work for limited additional findings beyond PT coverage |

## Models Implemented (#01-#20)

| # | Model | Folder | Notes |
|---|---|---|---|
| 01 | Linear Regression | [TensorFlow/01-linear-regression](../../TensorFlow/01-linear-regression/) | — |
| 02 | Logistic Regression | [TensorFlow/02-logistic-regression](../../TensorFlow/02-logistic-regression/) | — |
| 03 | KNN | [TensorFlow/03-knn](../../TensorFlow/03-knn/) | TF 2.11+ no Windows GPU |
| 04 | K-Means | [TensorFlow/04-k-means](../../TensorFlow/04-k-means/) | — |
| 05 | Naive Bayes | [TensorFlow/05-naive-bayes](../../TensorFlow/05-naive-bayes/) | — |
| 06 | Decision Trees / RF | [TensorFlow/06-decision-trees-random-forests](../../TensorFlow/06-decision-trees-random-forests/) | Slowest framework (199 min) |
| 07 | SVM | [TensorFlow/07-svm](../../TensorFlow/07-svm/) | — |
| 08 | PCA | [TensorFlow/08-pca](../../TensorFlow/08-pca/) | — |
| 09 | DNN | [TensorFlow/09-dnn](../../TensorFlow/09-dnn/) | CPU |
| 10 | Autoencoders | [TensorFlow/10-autoencoders](../../TensorFlow/10-autoencoders/) | Dense AE only — Conv crashes on CPU |
| 11 | CNN | [TensorFlow/11-cnn](../../TensorFlow/11-cnn/) | First WSL2 GPU model |
| 12 | RNN | [TensorFlow/12-rnn](../../TensorFlow/12-rnn/) | CPU |
| 13 | LSTM | [TensorFlow/13-lstm](../../TensorFlow/13-lstm/) | CPU |
| 14 | GANs | [TensorFlow/14-gans](../../TensorFlow/14-gans/) | DCGAN only |
| 15 | Attention | [TensorFlow/15-attention](../../TensorFlow/15-attention/) | Bahdanau only |
| 16 | Transformers | [TensorFlow/16-transformers](../../TensorFlow/16-transformers/) | ✓ Deploy winner (D5) |
| 17 | ViT | [TensorFlow/17-vit](../../TensorFlow/17-vit/) | V2 Recipe only — V3 cuDNN-blocked |
| 18 | GNN | [TensorFlow/18-gnn](../../TensorFlow/18-gnn/) | V1 GCN + V3 GAT — V2/V4 Spektral-blocked |
| 19 | VAE | [TensorFlow/19-vae](../../TensorFlow/19-vae/) | V1 only — V2 cuDNN-blocked |
| 20 | Q-Learning | [TensorFlow/20-q-learning](../../TensorFlow/20-q-learning/) | V1 + V2 only — V3-V5 PT-only by plan |

## Per-Model Findings

### #01 Linear Regression

- Keras `model.fit()` abstraction is the simplest code of all 4 frameworks
- Slowest (23.58s vs sklearn's 0.03s, NF's 0.38s) — Keras overhead for simple tasks
- All 4 frameworks identical accuracy (R² ≈ 0.50, RMSE ≈ $10,100)

### #02 Logistic Regression

- 52.95s training (slowest) due to full-batch overhead
- `model.fit()` provides simplest code — but the abstraction tax is real for trivially-fitted models
- ~83% recall on fraud detection — same as all other frameworks

### #03 KNN

- **Chunked broadcasting on CPU** (TF 2.11+ no Windows GPU). 110 predictions/sec
- No `torch.cdist` equivalent — TF requires chunked broadcasting for pairwise distances, creating memory management challenges PyTorch avoids
- 93.77% accuracy — same as all other frameworks

### #04 K-Means

- **Slowest of all 4 frameworks**: 2.01s training. 33x slower than sklearn (0.06s), 2x slower than NF's pure NumPy (1.02s)
- TF eager-mode dispatch overhead is the bottleneck — every tensor op crosses Python→C++ even at 10K samples
- `tf.TensorArray` for immutable tensors — the framework-specific implementation note
- 0.3064 silhouette, 0.6684 ARI — within rounding of all other frameworks

### #05 Naive Bayes

- **CPU tensor ops** (TF 2.20.0, no Windows GPU). 0.10s training, 7.62 µs/sample inference
- **The framework-specific showcase**: `tf.function` eager vs graph mode (1.12x speedup)
- 2x slower than PyTorch GPU but competitive with No-Framework/sklearn at 0.6683 accuracy
- List + `tf.stack()` pattern avoided the float32 NaN issue PyTorch hit with `torch.zeros()`

### #06 Decision Trees / RF

- **Slowest framework: 199 min training** (6.8x slower than No-Framework's 29 min). TF eager dispatch kills CPU tree performance — every tensor op crosses the Python→C++ bridge, and tree recursion triggers millions of these crossings
- **Vectorization doesn't guarantee CPU speedup**: TF's `tf.cumsum` evaluates all 32,949 thresholds simultaneously, yet NumPy's sequential for-loop with O(1) incremental updates is 1.06x faster — memory allocation overhead for intermediate tensors outweighs computation savings
- Model size (29.50 MB) nearly identical to PyTorch (29.47 MB) — both store the same Python dict trees. The 47% reduction vs NF (55.23 MB) comes from tree structure differences, not tensor representation

### #07 SVM

- **CPU tensor-based dual gradient descent**: 85.77s training. 1.9x faster than NumPy (NF's 160s) — TF's C++ matmul kernels are more optimized than NumPy's BLAS bindings
- Dual gradient descent converges to obj=231.83 — algorithm is implementation-agnostic
- 15.55 µs/sample inference — 26x slower than PyTorch GPU's 0.59 µs but faster than sklearn's 36.63 µs

### #08 PCA

- **CPU eager-mode tensor ops**: 0.17s fit, 0.93 µs/sample
- **Eager vs `tf.function` showcase**: 1.10x graph speedup. PCA is dominated by a single LAPACK eigh call, so graph compilation has little to fuse. Real benefit is lower timing variance (3.31 ms vs 12.01 ms std)
- All 4 frameworks produce identical results (0.9085 explained variance, 85.99% downstream KNN accuracy)

### #09 DNN

- **CPU training only** (Windows TF 2.11+, pre-WSL2): 9.14s training, 31.68 µs/sample inference
- 94.23% accuracy with Keras Sequential 128-64 — PT's 256-128 with same architecture underperformed on TF CPU (92.70%). GPU training dynamics differ from CPU for BatchNorm + wider layers
- **Keras callbacks are the code simplicity winner**: 5 lines of callback config replaces ~40 lines of PyTorch manual training loop for identical functionality (EarlyStopping + ReduceLROnPlateau + restore_best_weights)
- Inference 31.68 µs/sample — much slower than PT GPU's 0.35 µs and sklearn's 0.65 µs. Keras `model.predict()` has per-call Python overhead; production would use TF Serving

### #10 Autoencoders

- **Dense AE only**: 128-dim bottleneck, MSE 0.0096, 15K subset. Conv AE skipped — TF CPU crashes on conv training with color images
- **TF CPU cannot handle conv autoencoders on color images**: repeated system crashes (exit code 0xC0000005) with 50K, 15K, and even subset training. CPU memory allocator cannot manage conv layer activation maps + gradients on Windows. Dense AE worked fine — issue is specific to convolutional architectures
- The framework-specific finding: **CPU limitation documentation — when GPU matters**

### #11 CNN — first WSL2 GPU model

- **First TF model on GPU via WSL2**: TF 2.11+ dropped native Windows GPU. WSL2 Ubuntu + CUDA libraries enables GPU access. One-time setup that benefits all 9 remaining models
- **79.5% accuracy on CIFAR-100** with ResNet-20 + CutMix + Nesterov SGD — within 0.6pp of PT's 80.1%
- **11x slower training than PT**: manual `tf.GradientTape` loop with per-batch augmentation has significant Python-level overhead. Not a fundamental TF limitation — `model.fit()` + `@tf.function` would be faster, but CutMix required manual loop
- The framework-specific showcase: Keras Functional API for ResNet, WSL2 GPU setup, documented eager mode performance characteristics

### #12 RNN

- **CPU training**: BiGRU-64 (2 layers), 89.8% accuracy, 0.54 macro F1 on ECG5000
- 218s CPU training (vs PT GPU 4s) — WSL2 GPU setup friction isn't worth it when training takes minutes
- The framework-specific showcase: Keras `model.fit` with `class_weight` dict, custom `MacroF1Callback`, `Bidirectional` layer wrapper

### #13 LSTM

- **CPU training**: ECG: LSTM-128 0.607 macro F1 (augmented). IMDB: LSTM-128 86.5% accuracy
- 8 min ECG, 24 min IMDB on CPU — viable for baseline confirmation but experimentation (architecture sweeps, ablations) requires GPU
- Architecture sweep skipped (PT covered) — CPU makes 8-model sweep impractical (5+ hours)
- The framework-specific showcase: `model.fit` with `class_weight` + `mask_zero` embedding + concise callback API

### #14 GANs

- **DCGAN only on WSL2 GPU**: 79 min training (15x slower than PT's 5.3 min on the same RTX 4090)
- **WSL2 filesystem is the TF bottleneck, not compute**: data lives on Windows `/mnt/c/` — every batch read goes through 9P protocol translation. Moving data to Linux filesystem would close much of this gap
- **FID computation requires framework alignment**: `compute_fid` uses PyTorch InceptionV3, which runs CPU-only in the WSL2 TF venv. Cross-framework metric tools need careful environment planning
- WGAN-GP skipped (6h+ estimate, worse FID in PT)

### #15 Attention

- **Bahdanau only on WSL2 GPU**: BLEU 0.3368, 278 min training
- **17x slower than PT (16 min)** — `reset_after=False` forces a non-CuDNN GRU kernel on WSL2, changing gate computation order. Combined with `/mnt/c/` filesystem overhead
- Identical architecture to PT's 0.380 — confirms Bahdanau's pre-GRU context injection cross-framework. Quality gap entirely explained by infrastructure penalty

### #16 Transformers — DEPLOYMENT WIN (D5)

- **TF Translation BLEU (0.4456) dramatically exceeds PT (0.3625)**: same architecture, same data, same hyperparameters. **TF greedy decode alone beats PT beam search**
- **TF beats #15 Bahdanau by +17%**: 0.4456 vs 0.3803. The Transformer DOES surpass RNN+attention — just not in PyTorch. Framework implementation details matter more than architecture when models haven't converged
- Random initialization + optimizer implementation details produce meaningfully different training trajectories at this scale
- Classification: TF 92.20% vs PT 91.22% (+0.98%). Both below DistilBERT fine-tuned 94.45% (PT-only)
- **WSL2 adds 8.5-17x speed overhead**: Translation training 241 min (TF) vs 28 min (PT). Inference 377 ms vs 22.6 ms
- The framework-specific showcase: `tf.keras.Model` subclassing, `@tf.function` graph compilation, `tf.GradientTape`, custom `LearningRateSchedule`
- **Deployed as D5** — see [deployment/services/tf-svc/](../../deployment/services/tf-svc/) (planned Phase 5)

### #17 Vision Transformers — V2 only (V3 cuDNN-blocked)

- **V2 Recipe only**: 63.60% test accuracy on CIFAR-100. -0.85pp from PT V2 (64.45%) — DeiT recipe is robust to framework choice
- **TF trained ~30% faster** than PT on this model — simplified augmentation stack (no RandAugment / RandomErasing). TF inference ~2x slower due to framework overhead + Dense-based patch embed workaround
- **V3 Distillation skipped due to cuDNN Conv2D env constraint**: TF 2.21 on WSL2 build fails on Conv2D ops. Forced CPU teacher fallback caused two system crashes even with core affinity capping. PT V3 findings preserved
- The framework-specific showcase: `tf.keras.Model` subclassing, `@tf.function` + `tf.GradientTape` custom loop, cross-framework recipe confirmation

### #18 GNN — V1 GCN + V3 GAT only

- **V2 GraphSAGE + V4 GIN dropped**: Spektral 1.3.1 broken on Keras 3 (list-of-Nones mask bug). Plan pivoted to from-scratch TF primitives
- **V1 GCN: Cora 0.8090, arxiv OGB 0.7045** — slightly beats TF V3 GAT on both datasets. Intra-framework reversal from PT (where GAT wins). Framework matters more than architecture at this margin: Keras 3's AdamW-style decoupled weight decay + Glorot init combination doesn't favor attention as clearly as PT's L2 Adam + Kaiming init
- **TF V1 GCN at arxiv OGB 0.7045 even edges PT V3 GAT (0.7028) cross-framework** — attention's lift on these graphs is <1pp and framework-sensitive
- V3 GAT: Cora 0.8060, arxiv OGB 0.7004
- The framework-specific showcase: from-scratch with `tf.sparse` + `tf.math.unsorted_segment_*`, manual `tf.GradientTape` with masked transductive loss, OGB Evaluator for leaderboard parity

### #19 VAE — V1 only (V2 cuDNN-blocked)

- **V1 Vanilla VAE only**: MNIST test NLL **102.09 nats**, parity delta **+0.11 nats** vs PT (101.98) — **the tightest reproducibility result in the portfolio**
- **V2 cuDNN-blocked**: 9.1.0 runtime vs 9.3.0 compiled — same blocker that surfaced on ViT #17. V3-V5 PT-only by plan: single-coefficient β change (V3) doesn't exercise new TF primitives; STE codebook (V4) and masked-conv autoregressive sampling (V5) are substantial scope for marginal cross-framework value
- **WSL2 GPU eager mode 14x slower** (1048s vs 73s) — converges to the same ELBO. **The math is the math; the framework is just plumbing**

### #20 Q-Learning — V1 + V2 only

- **V1 Tabular (Taxi-v4)**: parity 0.00% (bit-identical Q-tables). Same numpy training loop, same seed, same Taxi-v4 transitions. The TF involvement is `tf.constant` + `tf.argmax` at the inference boundary — cosmetic but auditable. **The math is the math; tabular Q-learning is numpy-native**
- **V2 DQN (CartPole-v1)**: eval +403.20 vs PT V2's +500.00 (3-seed all-perfect). 19.36% gap — within Henderson 2018 RL noise floor
- **TF eager mode is 24x slower than PT for tight RL loops**: 288 min/seed vs 12 min/seed on V2. TF eager dispatches small graphs ~250K times per training run; PT eager amortizes much better
- Same algorithm, same hyperparameters, same seed, **different framework** → mean max-Q values remarkably close (133.49 vs 135.01, both overestimating ~34 the same way) but final-policy quality diverges. Default Dense init schemes (Glorot vs Kaiming), Adam epsilon defaults (1e-7 vs 1e-8), and divergent RNG paths during replay sampling produce ~100-point eval gaps even with identical hyperparameters

## Progress Log (TensorFlow entries, chronological)

| Date | Model | Notes |
|---|---|---|
| 2026-04-27 | Q-Learning | V1 Tabular (Taxi-v4) + V2 DQN (CartPole-v1) only. **V1 parity 0.00% (bit-identical Q-tables)**. V2 eval +403.20 vs PT +500.00 (19.36% gap). Wall-clock 24x slower than PT in eager mode (288 min vs 12 min/seed) |
| 2026-04-24 | VAE | V1 Vanilla VAE only (V2 cuDNN-blocked). MNIST test NLL **102.09** nats, parity delta **+0.11** vs PT |
| 2026-04-20 | GNN | V1 GCN + V3 GAT from TF primitives. Spektral dropped. WSL2 GPU. V1 GCN: Cora **0.8090**, arxiv OGB **0.7045**. V3 GAT: Cora 0.8060, arxiv OGB 0.7004 |
| 2026-04-17 | Vision Transformers | V2 Recipe only (WSL2 GPU). **63.60% test**. V3 Distillation skipped |
| 2026-04-12 | Transformers | Recipe variant (WSL2 GPU). Translation **BLEU 0.4456**. Classification **92.20% acc**. 241 min translation training (8.5x WSL2 overhead) |
| 2026-04-06 | Attention | Bahdanau only (WSL2 GPU). **BLEU 0.3368**, 278 min training (17x slower) |
| 2026-04-03 | GANs | DCGAN only (WSL2 GPU). 79 min training (15x slower — filesystem overhead). WGAN-GP skipped |
| 2026-03-30 | LSTM | ECG: LSTM-128 **0.607 macro F1** (augmented). IMDB: LSTM-128 **86.5% acc**. CPU training |
| 2026-03-28 | RNN | BiGRU-64 (2 layers), **89.8% accuracy, 0.54 macro F1** on ECG5000. Keras Sequential + `model.fit` + custom MacroF1Callback. CPU training (218s) |
| 2026-03-26 | CNN | ResNet-20 via Keras Functional API + CutMix + Nesterov SGD, **79.5% accuracy** on CIFAR-100. **WSL2 GPU (RTX 4090)** |
| 2026-03-21 | Autoencoders | Dense AE only (128-dim, MSE 0.0096, 15K subset). Conv AE skipped — TF CPU crashes on conv training with color images |
| 2026-03-18 | DNN | Keras Sequential + callbacks, 128-64 architecture, 94.23% accuracy. 9.14s training, 31.68 µs/sample |
| 2026-03-16 | PCA | CPU eager-mode tensor ops (0.17s fit, 0.93 µs/sample). Eager vs `tf.function` showcase: 1.10x graph speedup |
| 2026-03-11 | SVM | CPU tensor-based dual gradient descent (85.77s training, 1.9x faster than NF). 15.55 µs/sample |
| 2026-03-07 | Decision Trees & RF | CPU tensor ops DT+RF. NumPy 1.06x faster than TF for split search. **Slowest framework (199 min)** |
| 2026-02-28 | Naive Bayes | CPU tensor ops (TF 2.20.0, no Windows GPU). 0.10s training, 7.62 µs/sample |
| 2026-02-24 | K-Means | CPU tensor ops, `tf.TensorArray` for immutable tensors. **Slowest at 2.01s** |
| 2026-02-15 | KNN | Chunked broadcasting on CPU (TF 2.11+ no Windows GPU). 93.77% accuracy, 110/sec |
| 2026-02-10 | Logistic Regression | Keras `model.fit()` abstraction. Slowest (52.95s) |
| 2026-02-08 | Linear Regression | Keras `model.fit()` abstraction. Slowest (23.58s) but simplest code |

## Cross-cutting takeaways

1. **TF's strength is production seq2seq**: the #16 Translation win (BLEU 0.4456 vs PT 0.3625) isn't an anomaly — Keras 3's recipe-style training (`@tf.function` + `tf.GradientTape` + `LearningRateSchedule`) maps cleanly to converged training. The same advantage doesn't show up on research-pace work because TF's iteration overhead grows with the variant count.

2. **The eager-mode penalty is concrete and quantified**: VAE V1 1048s vs 73s (14x), Q-Learning V2 288 min vs 12 min/seed (24x). For tight per-step loops in RL/replay/ELBO contexts, TF eager dispatches small graphs hundreds of thousands of times per run; PT eager amortizes better. Production TF avoids this with `@tf.function`; we don't, deliberately, to keep the framework comparison honest.

3. **WSL2 GPU is real but expensive**: enabling TF GPU from #11 onward unblocked 9 models. Cost: `/mnt/c/` filesystem overhead (15x penalty on GANs), cuDNN version mismatches (blocked ViT V3 + VAE V2), Spektral library incompatibility (blocked GNN V2/V4). The scope reductions are documented as portfolio findings, not gaps.

4. **TF wins where convergence + recipes matter, loses where iteration speed matters**: this maps cleanly to "production deployment" vs "research exploration." Keras callbacks (DNN), `model.fit` simplicity (LSTM), `@tf.function` + custom training loops (CNN) all show TF's production-readiness — but the same simplicity that helps deployment hurts when you need 4-5 variants per model in fast iteration.

5. **Cross-framework parity is achievable when math dominates**: V1 Tabular Q-Learning (0.00% parity), VAE V1 (+0.11 nats), PCA (identical eigenvalues to 1e-5). When the algorithm is implementation-agnostic, framework choice is cosmetic. The parity gaps that DO appear (Q-Learning V2's 19% on CartPole, Transformers' 8pp on translation) trace to optimizer/initialization defaults, not framework capability.
