# Cross-Framework Findings

> Observations that span multiple frameworks — parity results, "all 4 identical" patterns, framework-vs-framework comparisons, and general insights from the 20-model modeling phase.

For framework-specific deep dives, see:
- [no-framework.md](no-framework.md) — pure NumPy/SciPy, #01-#08
- [scikit-learn.md](scikit-learn.md) — #01-#10
- [pytorch.md](pytorch.md) — all 20 models
- [tensorflow.md](tensorflow.md) — all 20 with WSL2 scope reductions

## Parity results: when frameworks agree

The portfolio's strongest cross-framework reproducibility findings, ordered by tightness:

| Model | Result | Frameworks |
|---|---|---|
| **#20 Q-Learning V1** (Taxi-v4) | **0.00% relative diff** across all 3,000 (state, action) Q-table entries | PT vs TF |
| **#19 VAE V1** (MNIST) | **0.11 nats** test NLL difference (TF 102.09 vs PT 101.98) | PT vs TF |
| **#08 PCA** (Fashion-MNIST) | Identical 0.9085 explained variance, 0.0951 reconstruction MSE, 85.99% downstream KNN accuracy at 150 components | All 4 frameworks |
| **#07 SVM** (MAGIC Gamma) | accuracy 0.8611, F1 0.8990, AUC 0.9105 across NF/PT/TF; SK 0.8606/0.8942/0.9164 (different by 0.0005-0.006) | All 4 frameworks |
| **#05 Naive Bayes** (20 Newsgroups) | accuracy 0.6683, macro F1 0.6394, log-loss 1.5576, Brier 0.6008, ECE 0.3229 | All 4 frameworks |
| **#04 K-Means** (Dry Bean) | inertia ~9,976, silhouette ~0.3064, ARI ~0.6684 | All 4 frameworks |
| **#03 KNN** (Forest Cover) | 93.77% accuracy | All 4 frameworks |
| **#01 Linear Regression** (Vehicle prices) | R² ≈ 0.50, RMSE ≈ $10,100 | All 4 frameworks |

**The pattern**: when the algorithm is mathematically deterministic (eigendecomposition, Q-table updates, kernel SVM dual ascent, KMeans Lloyd's algorithm, Naive Bayes likelihood), framework choice is cosmetic. The math is the math.

The cases where frameworks DON'T agree (#06 DT/RF, #09 DNN, #11+ deep learning) all involve stochastic initialization, non-deterministic optimizer paths, or floating-point order-of-operations differences that compound across thousands of training steps.

## Where frameworks diverge

### Speed/memory hierarchy across the modeling phase

| Workload class | Framework hierarchy (fastest first) |
|---|---|
| Classical ML training | sklearn (Cython) > PT GPU > NF (NumPy) > TF (eager dispatch overhead) |
| Classical ML inference | PT GPU >> sklearn > TF CPU > NF |
| Deep learning training (#09+) | PT GPU > TF GPU (with `@tf.function`) > TF GPU (eager) > sklearn (limited to MLP) |
| Deep learning inference | PT GPU > TF (with TF Serving) > Keras `model.predict()` per-call |
| RL tight loops | PT eager (12 min/seed) >> TF eager (288 min/seed) — 24x gap |
| Eigendecomposition | PT GPU CUDA LAPACK > sklearn LAPACK > NF NumPy > TF eager |

### Model size patterns

- **Sklearn keeps diagnostic arrays** alongside parameters (Naive Bayes: 3.05 MB vs NF's 1.53 MB)
- **PT/TF tensor representations are more compact** than dict/list structures: DT/RF 29.47 MB (PT) and 29.50 MB (TF) vs 55.23 MB (NF)
- **NLP embedding layers dominate model size**: LSTM IMDB 5.1 MB embedding / 5.9 MB total (87%)

### Surprising framework wins

- **TF beats PT on #16 Translation** by +0.0831 BLEU (0.4456 vs 0.3625) on identical architecture/data/hyperparams — random init + optimizer epsilon defaults compound
- **Sklearn beats PT GPU on #03 KNN**: KD-tree's O(log n) lookups beat brute-force GPU at 464K samples (57s vs 100s)
- **NF beats PT GPU on #06 DT inference**: 169.46 µs/sample (NF) vs 279.79 µs (PT) — flattening 100 tree dicts to GPU tensors per call adds conversion overhead that outweighs GPU prediction speed
- **TF eager CPU beats raw NumPy on #07 SVM**: 85.77s vs 160s (1.9x faster) — TF's C++ matmul kernels are more optimized than NumPy's BLAS bindings

## Cross-framework lessons

### "Each framework showcased a unique strength" (the per-model showcase pattern)

Every model from #04 onward includes a "framework-specific showcase" — a feature only that framework demonstrates well, even when all frameworks produce equivalent metrics. The pattern locked in early:

| Model | NF showcase | SK showcase | PT showcase | TF showcase |
|---|---|---|---|---|
| K-Means | from-scratch Lloyd | KMeans + MiniBatchKMeans | `torch.cdist` + `torch.vmap`/`torch.compile` | `tf.TensorArray` |
| Naive Bayes | log-sum-exp trick | CalibratedClassifierCV (ECE 0.32→0.14) | CPU vs GPU matmul (2.8x) | `tf.function` eager vs graph (1.12x) |
| DT/RF | manual OOB computation | GridSearchCV + MLflow + joblib | hybrid CPU/GPU split search | (slow) eager-dispatch documentation |
| SVM | from-scratch dual gradient ascent | SMO with optimal calibration | GPU dual gradient descent | `@tf.function` graph |
| PCA | eigen vs SVD numerical equivalence | IncrementalPCA out-of-core | CUDA LAPACK eigh | eager vs `tf.function` |
| DNN | (retired) | activation function comparison | BatchNorm + Dropout + LR scheduling | Keras callbacks |

The discipline forces honest framework differentiation — "this is what THIS framework is uniquely good at" — even when the underlying metric is identical.

### General insights

- **High-level frameworks (Scikit-Learn) accelerate development for standard tasks but hide mechanics** — the speed/understanding trade-off
- **Deep learning libraries (PyTorch/TensorFlow) offer control over modern architectures while providing tools** like autograd, optimizers, schedulers
- **From-scratch builds solidify fundamentals but scale poorly** for complex models — the #08 PCA retirement boundary is real
- **Framework choice depends on data type, team needs, and deployment goals** — there is no universal winner

### Honest negative results documented as portfolio assets

- **#20 Q-Learning V5 PER** does NOT replicate Schaul 2016: 0/3 seeds solved on LunarLander vs V4's 2/3. Paper claims do not auto-replicate without retuning (Henderson 2018 in miniature)
- **#18 GNN V4 GIN underperforms V1 GCN by 2.7pp on arxiv**: more theoretical expressiveness does not auto-transfer to node classification with pre-trained features
- **#17 ViT V3 Distillation -12.75pp below CNN #11**: ViT's lack of convolutional inductive bias is a real cost without pretraining on small data
- **#19 VAE V4 VQ-VAE failed three times before working**: gradient-killing detach, codebook collapse, scale mismatch — each diagnosis required tracing the actual forward/backward pass
- **#11 CNN's 10-cell progression** documents 56.9% → 80.1% as a sequence of clearly-attributable wins, not a single "magic" recipe

## The four-paradigm coverage

The 20-model portfolio covers all four classical ML paradigms:

- **Supervised**: #01 Linear Regression, #02 Logistic Regression, #03 KNN, #05 Naive Bayes, #06 DT/RF, #07 SVM, #09 DNN, #11 CNN, #12 RNN, #13 LSTM, #15 Attention, #16 Transformers, #17 ViT, #18 GNN
- **Unsupervised**: #04 K-Means, #08 PCA, #10 Autoencoders
- **Generative**: #14 GANs, #19 VAE
- **Reinforcement**: #20 Q-Learning

`#20 Q-Learning` completes the set. The deployment phase (Phase 1 onward) selects 5 representative models across these paradigms for production wrapping (D1 PCA, D2 DNN, D3 GANs, D4 Q-Learning, D5 Transformer Translation).
