# Modeling Findings: PyTorch

> Implemented all 20 models. Won deployment for **14 of 17 deployable artifacts**. The dominant framework for #09-#20.

## Why PyTorch dominates the deep-learning portfolio

- **Eager mode is 14-24x faster than TF eager for tight per-step loops** (measured on VAE V1 and Q-Learning V2). For RL replay loops and ELBO training, this gap becomes the difference between a 12-minute training run and a 4-hour one.
- **Dynamic graphs cut debug time on custom architectures** — GNN sparse adjacency operations, RL replay integration, VAE masked-conv causality. Eager-by-default lets you `print(tensor.shape)` mid-forward without recompiling a graph.
- **`nn.Module` subclassing makes from-scratch reimplementation idiomatic** — every PyTorch model in this repo (Transformer, ViT, GAT, VQ-VAE, DQN family) was built from `nn.Linear` / `nn.Conv2d` / `nn.Embedding` primitives, no high-level `nn.Transformer` or `nn.MultiheadAttention` shortcuts.

## Models Implemented (#01-#20)

| # | Model | Folder | Deployment Win? |
|---|---|---|---|
| 01 | Linear Regression | [PyTorch/01-linear-regression](../../PyTorch/01-linear-regression/) | — |
| 02 | Logistic Regression | [PyTorch/02-logistic-regression](../../PyTorch/02-logistic-regression/) | — |
| 03 | KNN | [PyTorch/03-knn](../../PyTorch/03-knn/) | — |
| 04 | K-Means | [PyTorch/04-k-means](../../PyTorch/04-k-means/) | — |
| 05 | Naive Bayes | [PyTorch/05-naive-bayes](../../PyTorch/05-naive-bayes/) | — |
| 06 | Decision Trees / RF | [PyTorch/06-decision-trees-random-forests](../../PyTorch/06-decision-trees-random-forests/) | — |
| 07 | SVM | [PyTorch/07-svm](../../PyTorch/07-svm/) | — |
| 08 | PCA | [PyTorch/08-pca](../../PyTorch/08-pca/) | — |
| 09 | DNN | [PyTorch/09-dnn](../../PyTorch/09-dnn/) | ✓ Deploy winner (D2) |
| 10 | Autoencoders | [PyTorch/10-autoencoders](../../PyTorch/10-autoencoders/) | ✓ Deploy winner |
| 11 | CNN | [PyTorch/11-cnn](../../PyTorch/11-cnn/) | ✓ Deploy winner |
| 12 | RNN | [PyTorch/12-rnn](../../PyTorch/12-rnn/) | ✓ Deploy winner |
| 13 | LSTM | [PyTorch/13-lstm](../../PyTorch/13-lstm/) | ✓ Deploy winner (×2 datasets) |
| 14 | GANs | [PyTorch/14-gans](../../PyTorch/14-gans/) | ✓ Deploy winner (D3) |
| 15 | Attention | [PyTorch/15-attention](../../PyTorch/15-attention/) | ✓ Deploy winner |
| 16 | Transformers | [PyTorch/16-transformers](../../PyTorch/16-transformers/) | ✓ Deploy winner (×2 tasks) |
| 17 | ViT | [PyTorch/17-vit](../../PyTorch/17-vit/) | ✓ Deploy winner |
| 18 | GNN | [PyTorch/18-gnn](../../PyTorch/18-gnn/) | ✓ Deploy winner (×2 datasets) |
| 19 | VAE | [PyTorch/19-vae](../../PyTorch/19-vae/) | ✓ Deploy winner (×2 datasets) |
| 20 | Q-Learning | [PyTorch/20-q-learning](../../PyTorch/20-q-learning/) | ✓ Deploy winner (D4) |

## Per-Model Findings

### #01 Linear Regression

- Autograd vs manual gradients comparison. Slower (3.44s) and more memory (54 MB) than sklearn — overhead is the foundation cost for neural networks
- All 4 frameworks identical accuracy (R² ≈ 0.50, RMSE ≈ $10,100) — proves framework choice doesn't affect quality for equivalent algorithms

### #02 Logistic Regression

- Autograd + SGD, 7.8x faster than No-Framework (2.36s) — the autograd payoff starts showing
- All 4 frameworks: ~83% recall on fraud detection — consistent across implementations

### #03 KNN

- **GPU-accelerated `torch.cdist`** uses 7.2 GB VRAM for 464K training samples
- 1,164 predictions/sec — 776x faster than No-Framework's 1.5/sec, but still slower than sklearn's KD-tree at 2,000/sec
- Lesson: GPU helps, but tree-based algorithms beat brute-force GPU at index lookups

### #04 K-Means

- **GPU `torch.cdist` + `torch.vmap`/`torch.compile` showcases** — 1.15x speedup for parallel n_init runs
- 0.34s training (5.7x slower than sklearn's 0.06s) — GPU kernel launch overhead doesn't pay off at 10K samples
- `torch.compile` limited on Windows: TorchInductor backend doesn't fully support Windows in 2.5.1. Manual broadcasting was 1.9x faster than `torch.cdist`

### #05 Naive Bayes

- **Fastest of all 4 frameworks**: 0.028s training, 3.5 µs/sample inference on RTX 4090
- Float64 required for GaussianNB: float32 caused NaN log-loss and ECE mismatch due to `torch.zeros()` defaulting to float32 regardless of input dtype
- The framework-specific showcase: CPU vs GPU matmul (2.8x speedup)

### #06 Decision Trees / RF

- **Hybrid CPU/GPU approach**: recursive dicts on CPU, `torch.sort` + `torch.cumsum` split search on GPU. 16% faster training than No-Framework — modest because GPU kernel launch overhead partially offsets parallelism at 32K samples
- Inference overhead from architecture mismatch: 279.79 µs/sample (slower than NF's 169.46 µs/sample) — flattening 100 tree dicts to GPU tensors per call adds conversion overhead that outweighs GPU prediction speed
- Model size halves with torch tensors: 29.47 MB vs NF's 55.23 MB — fewer intermediate Python objects when heavy computation happens in-place

### #07 SVM

- **GPU-accelerated dual gradient descent**: 9.03s training, 17.7x faster than NF (160s) and 9.7x faster than TF CPU (85.77s). The O(n²) kernel matrix-vector product each iteration is embarrassingly parallel
- Best inference speed of all 4 frameworks: 0.59 µs/sample — 260x faster than NF, 26x faster than TF CPU, 62x faster than sklearn

### #08 PCA

- **GPU eigendecomposition fastest across the board**: 0.11s fit (1.7x vs SK, 2.1x vs NF), 0.39 µs/sample inference. 9.1x GPU speedup over CPU eigh on 784×784 matrix
- 599 MB GPU memory used. CUDA LAPACK eigh pays off even at this scale
- The framework-specific showcase: GPU vs CPU eigendecomposition (50-run benchmark)

### #09 DNN — DEPLOYMENT WIN (D2)

- **Best accuracy: 96.03%** with 256-128 RegularizedDNN (BatchNorm + Dropout + ReduceLROnPlateau), 178K params, 96.02% F1
- Regularization is the differentiator: same 128-64 architecture jumps +1.0% with BatchNorm + Dropout + LR scheduling. SK's MLPClassifier cannot express these structurally
- Wider architectures benefit from GPU + regularization: 256-128 (178K params) hits 96.03%, but the same architecture underperformed on TF CPU (92.70%)
- Inference: 0.35 µs/sample — fastest of all frameworks
- The framework-specific showcase: BatchNorm + Dropout + LR scheduling
- **Deployed as D2 in Phase 2** — see [deployment/services/pt-svc/](../../deployment/services/pt-svc/) for the production wrapper

### #10 Autoencoders — DEPLOYMENT WIN

- **Best reconstruction: MSE 0.0037** with 64-128-256 filters + 256-dim latent (2.8M params). 3.6x better than SK's dense AE (0.0133) with comparable parameters
- **Conv denoising AE** removes 86.9% of noise (σ=0.2) while maintaining clean reconstruction. Even at σ=0.5 (heavy noise), denoised output (0.0099) beats SK's clean reconstruction (0.0133)
- Reconstruction quality ≠ classification quality: dense AE (worse MSE) learns more class-separable latent features. KNN accuracy: dense 40.3% > conv 36.2%. Tighter bottlenecks force semantic compression
- Architecture sweep identifies optimal: Large (64-128-256, lat=256) beats Medium by 31%, Wide by 28%, Small by 56%

### #11 CNN — DEPLOYMENT WIN

- **Best accuracy: 80.1%** on CIFAR-100 (100 fine classes, 32×32×3, 60K images). ResNet-20 + CutMix + Label Smoothing + Nesterov SGD
- **Progressive improvement from 56.9% to 80.1%** through 10 systematic experiments — plain CNN baseline → architecture sweep (Wide wins) → cosine LR refinement → ResNet-20 → SGD discovery → cosine cycles → CutMix + label smoothing
- **SGD with momentum is the single biggest gain (+12.8%)**: Adam's adaptive per-parameter learning rates interfere with ResNet's BatchNorm dynamics. SGD lr=0.05 with weight decay 5e-4 trained the full 200-epoch cosine cycle while Adam early-stopped at epoch 37-51
- **CutMix collapsed overfitting from 22% to 5% gap**: replaced patches with real image content from other classes, forcing partial-feature learning. More effective than Cutout or label smoothing alone
- ResNet-20 is the sweet spot for CIFAR-100 at 32×32: ResNet-32 (7.45M params) matched ResNet-20 (4.35M) at 77.4%
- Cosine annealing must complete its full cycle: early stopping with cosine LR is counterproductive — best results always in final 10-15% of training
- Superclass analysis validates EDA: People (girl/boy/man/woman) hardest fine classes (F1 0.54-0.63), trees/flowers easiest

### #12 RNN — DEPLOYMENT WIN

- **GRU-128 (2 layers), 91.8% accuracy, 0.55 macro F1** on ECG5000 (5-class heartbeat, 121.6× class imbalance)
- **Macro F1 is the only honest metric**: 91.8% accuracy sounds great, but macro F1 of 0.55 reveals failure on 3/5 classes. Accuracy is misleading when 58% of data is one class
- Vanishing gradients didn't vanish at 140 timesteps: both vanilla RNN and GRU show healthy gradients (ratio 1.2-4.2x). Theoretical vanishing requires hundreds/thousands of steps
- GRU barely beats vanilla RNN on short sequences: +0.002 macro F1. Gating matters more for longer dependencies
- The performance ceiling is data, not architecture: 19 PVC, 39 SP, 5 UB training samples. No RNN variant can fix this — augmentation is the next step (#13 LSTM)
- Inference: 4.32 µs/sample

### #13 LSTM — DEPLOYMENT WIN (×2 datasets)

- **Two datasets staged**: ECG5000 (LSTM-128, 0.603 macro F1 augmented, broke RNN's 0.55 ceiling) + IMDB sentiment (LSTM-128, 87.8% accuracy, 0.946 AUC)
- **Data augmentation is the #1 tool for imbalanced sequence data**: time-series augmentation (jitter, scaling, time warp) drove 85% of the total improvement. GRU on augmented data: 0.5950 F1 (+0.047 over original); LSTM added only +0.008 on top
- LSTM adds minimal value over GRU for short sequences: at 140 ECG timesteps, +0.008 macro F1. The cell state pathway doesn't accumulate meaningfully different info
- Sequence length has diminishing returns for IMDB: +3.2% from 100 to 200 tokens, +0.1% from 200 to 300. Cut sequence length 33% with <0.1% accuracy loss
- BiLSTM underperformed: pre-padding means backward LSTM processes uninformative padding tokens first
- Embedding layers dominate NLP model size: 10K vocab × 128 dims = 5.1 MB (87% of total 5.9 MB)

### #14 GANs — DEPLOYMENT WIN (D3)

- **Best FID: 30.57** with DCGAN. 4 variants on CIFAR-10: Vanilla MLP (FID 261) → DCGAN (30.57) → WGAN-GP (55) → cGAN (148)
- **Convolutional architecture dominates GAN image quality**: DCGAN's FID dwarfs Vanilla MLP by 8.5x. Loss function choice (BCE vs Wasserstein) and conditional generation both made quality worse. **For deployment, simple DCGAN + BCE is the pragmatic choice**
- WGAN-GP trades quality for stability at fixed budgets: Wasserstein distance smoothly decreasing (11.5 → 1.4) is the only meaningful GAN training metric, but n_critic=5 means 5x fewer generator updates
- Generator architectures are parameter-efficient: DCGAN generator (1.07M params) produces recognizable 32×32 images vs CNN's ResNet-20 (4.35M params) for classification
- The framework-specific showcase: 4-variant progressive build, GPU FID computation, all loss types

### #15 Attention — DEPLOYMENT WIN

- **Best BLEU: 0.3803** with Bahdanau additive attention. 4 variants on Tatoeba EN→ES (143K pairs, word-level)
- **Pre-GRU context injection is the dominant factor**: Bahdanau (0.380) feeds full 1024-dim encoder context INTO the GRU before the state update. Luong (0.297) and Multi-Head (0.368) compute attention AFTER the GRU, getting only compressed 512-dim context
- Multi-Head attention nearly closes the gap: 8 parallel heads with Q·K^T/√d_k scoring (0.368) recover most of Bahdanau's advantage despite post-GRU disadvantage. **This IS the Transformer mechanism wrapped in an RNN**
- Attention eliminates the length penalty: no-attention BLEU drops 27% on long sentences (0.340→0.248); Bahdanau drops only 8% (0.387→0.357). Reproduces Bahdanau et al. (2014)
- Correct paper implementation matters: Luong without Eq. 5 had 25.1M params (fc_out doing both fusion AND projection). Adding tanh W_c dropped to 16.4M with cleaner architecture
- BLEU 0.38 is "gist quality": understandable translations with visible errors. Production MT needs subword tokenization (BPE), deeper models, beam search, 100x more training data — all addressed in #16

### #16 Transformers — DEPLOYMENT WIN (×2 tasks)

- **Two tasks**: Tatoeba EN→ES translation + AG News 4-class classification. 5 variants
- **Translation: PT Beam Search BLEU 0.3625** (TF Recipe 0.4456 wins on quality but PT staged for portable .pt format)
- **Classification: AG News PT 91.22%, DistilBERT fine-tuned 94.45%** (PT only, same pre-trained weights)
- **Built from scratch with `nn.Linear`** — `MultiHeadAttention`, `TransformerEncoderLayer`, `TransformerDecoderLayer` all defined manually. No `nn.Transformer` or `nn.MultiheadAttention` shortcuts
- **BPE subword tokenization (0% UNK rate)**: SentencePiece shared 8K EN+ES vocab. Eliminates the 2.6%/7.3% UNK rates from #15's word-level. BPE expansion ratio ~1.37x
- Training recipe matters as much as architecture: PT Vanilla 0.3289 → PT Recipe 0.3462 (+0.0172 BLEU) for zero architecture change. Warmup scheduler + label smoothing 0.1 + dropout 0.15
- Beam search is the cheapest improvement: k=5 with length_penalty=0.6 added +0.0164 BLEU with zero retraining
- DistilBERT quantifies pre-training advantage: +3.22% accuracy for 9.2x more parameters on AG News

### #17 Vision Transformers — DEPLOYMENT WIN

- **4 variants isolate distinct levers**: V1 Vanilla 50.33% → V2 DeiT Recipe 64.45% (+14.12pp) → V3 Distillation 67.48% (+3.03pp) → V4 Pre-trained ViT-B/16 91.16% (+23.68pp)
- ViT-Small (6L, d=384, 6 heads, patch=4, 10.73M params) **built from `nn.Linear`** — no `nn.TransformerEncoder` or `nn.MultiheadAttention`
- **Pre-training dominates on small data**: V4's +23.68pp jump from V3 is larger than V1→V2→V3 combined (+17.15pp). ImageNet-21k features transfer cleanly to CIFAR-100 despite the 7x resolution upscale
- V4 beats CNN #11 by +10.93pp (91.16% vs 80.23%), but at 29x inference cost (1117 µs vs 38.9 µs per sample). **CNN stays Pareto-optimal for low-latency deployment**
- **From-scratch ViT underperforms CNN on small data**: V3 Distillation (67.48%) lands -12.75pp below CNN #11. ViT's lack of convolutional inductive bias is a real cost without pretraining. **Honest portfolio finding**
- DeiT distillation with own CNN teacher: CNN #11 (80.23%) supervises ViT student, absorbs ~57% of teacher's advantage over vanilla. Hard distillation ([DIST] token + teacher argmax)
- Attention visualization confirms learned spatial structure: per-head CLS attention maps show emergent specialization. On baby samples, one head clearly learned "face features" (two bright spots where eyes are)
- **V4 pre-trained weights deleted during git cleanup** — 327 MB HuggingFace state_dict exceeded GitHub's 100 MB file limit. `git filter-repo` stripped from history. Re-generating requires re-running Cell 7 (~1h training). Deployment falls back to V3 as the servable artifact

### #18 GNN — DEPLOYMENT WIN (×2 datasets)

- **Two datasets**: Cora (2,708 nodes, 7 classes, transductive) + ogbn-arxiv (169K nodes, 40 classes, temporal split). Mirrors LSTM's dual-dataset pattern
- **First non-Euclidean model in the portfolio**: every prior model assumed regular structure. GNNs handle arbitrary neighbor counts, no canonical ordering. Shifted core primitive from dense matmul to `torch.sparse.mm` over message-passing operators
- **V3 GAT wins both datasets**: Cora 0.8310 (matches Velickovic 2018 exactly), arxiv OGB 0.7025. V1 GCN 0.6985, V2 SAGE 0.6948, V4 GIN 0.6714
- 4 variants: V1 GCN, V2 GraphSAGE, V3 GAT, V4 GIN. V2 and V4 are PT-only — their lever is library-dependent machinery (`NeighborLoader`, `GINConv`)
- **V4 GIN underperforms V1 GCN by 2.7pp on arxiv**: more theoretical expressiveness (WL-equivalent) does not auto-transfer to node classification with pre-trained word2vec features. Baseline's learned epsilons (+2.14, +2.50, +0.19) showed the model partially rejecting GIN's paradigm — upweighting each node's own features 3x over summed neighbors. **GIN's native domain is graph-level classification (molecules, proteins); arxiv is the wrong task family**
- **Optuna 20-trial sweep surfaced `num_layers=2` (not paper's 3) as the #1 arxiv fix**: 400-epoch budget per trial with MedianPruner. Top-5 configs agree: 2 layers, hidden=256, train_eps=True, lr 5-10x lower than paper's 0.01
- V4b tuned recovered +1.80pp val acc and +5.41pp macro F1 — but lost -1.01pp test accuracy to **concept-drift tax** (val set is 2018, test is 2019-2020; cs.lg +14pp, cs.it -12pp class-proportion shifts)
- **V3 GAT on arxiv costs 13.4 GB GPU** (from-scratch, materializes `(E+N, H, D)` attention intermediates per layer). PyG's `GATConv` with fused scatter kernels uses ~4 GB for the same accuracy. Honest cost: 3x memory of production-ready implementations, but per-edge computation explicit
- Per-class F1 tracks edge homophily monotonically: Cora Neural_Networks (0.91 homophily) F1 0.905, Case_Based (0.70) 0.724. arxiv cs.cv/cs.it/cs.cl reach F1 ~0.85; cs.gl (29 papers, 0.04 homophily) gets F1=0.00 across every variant. **Message passing only works as well as the graph lets it**

### #19 VAE — DEPLOYMENT WIN (×2 datasets)

- **5 variants**: V1 Vanilla, V2 Conv, V3 β-VAE sweep, V4 VQ-VAE, V5 VQ-VAE+PixelCNN prior. Two datasets: MNIST + CIFAR-10
- **First likelihood-based deep generative model in the portfolio**: AE #10 (deterministic, no generation) and GANs #14 (implicit, adversarial) are prior generative entries. VAE fills the third philosophical slot — explicit probabilistic latent, ELBO-trained, samplable
- **The FID ladder is the portfolio's headline result**: GAN #14 (30.57) < V4 reconstruction (97.81) < **V5 prior-sampled (117.27)** < V2 prior (165) < V4 random-codes (202). Each gap isolates a lever
- **V5 - V4 random = -85 FID is the learned-prior contribution** (PixelCNN doing its job). V4 recon - V5 = -19 is the prior's information loss vs the encoder's ceiling. GAN - V4 recon = -67 is the **honest VAE-vs-adversarial gap** that explains why modern generation moved to GAN, then diffusion
- **The prior does the generation, not the decoder**: V4 and V5 share the exact same decoder. V4 from uniform-random codes = FID 202 (noise). V5 from PixelCNN-sampled codes = FID 117 (coherent skies, animals, horizons). **This is the whole thesis of the two-stage recipe (DALL-E 1, Stable Diffusion, Muse, EnCodec)**
- **Discrete latents reconstruct better than continuous VAEs at similar scale**: V4 (138K params, MSE 0.0071) beats V2 (582K params, MSE 0.0166) by 2.3x with 4.2x fewer parameters. Codebook acts as a learned 512-point quantization grid that prevents latent drift into under-trained regions. **Quantization is a feature, not a bug**
- β-VAE's disentanglement is a Pareto trade-off: V3 β=4 cleanly separates digit identity (dim 3) from style (other dims) but pays 44 nats in NLL vs V1. β=10 collapses most dims. Sweet spot is task-dependent — no universal β
- **V4 VQ-VAE failed three times before working**: gradient-killing detach in `vq_straight_through` util (fix: inline VQ math), winner-take-all codebook collapse with random init (fix: data-dependent codebook init), 6-orders-of-magnitude scale mismatch between sum-reduced reconstruction and mean-reduced VQ losses (fix: mean-reduce all three loss terms). **Each diagnosis required tracing actual forward/backward, not pattern-matching the docstring**
- Information gain (uniform_baseline_CE - test_CE) is a cleaner prior-quality metric than FID: V5 captured 2.893 of 6.238 nats/position possible = 46% of maximum. Reports prior tightness independent of decoder image-quality ceiling
- Two-stage cost accounting matters for deployment: V5 is "1.4M params" in isolation, but its inference pipeline requires V4's 138K params + codebook → 1.52M total. Same for train time: V5 70.5s + V4 156.6s = 227s end-to-end

### #20 Q-Learning — DEPLOYMENT WIN (D4)

- **5 variants on PyTorch**: V1 Tabular (Taxi-v4) + V2 DQN + V3 Double DQN (CartPole-v1) + V4 Dueling DQN + V5 PER (LunarLander-v3). V3-V5 are PT-only by plan
- **Only reinforcement-learning model in the 20-model portfolio** — completes the four-paradigm coverage. Watkins 1989 → Mnih 2015 → van Hasselt 2016 → Wang 2016 → Schaul 2016 lineage with one isolated lever per variant
- **V1 cross-framework parity is bit-identical**: max relative diff 0.00% across all 3,000 (state, action) Q-table entries. Same numpy training loop, same seed, same Taxi-v4 transitions
- **V2 perfect 3/3 seeds** on CartPole-v1 (+500.00). V3 Double DQN's overestimation fix replicates cleanly at -32% (V2 mean max-Q 133.49 vs V3 122.58). **Algorithmic claim verified**
- **V5 PER does NOT replicate Schaul 2016**: 0/3 seeds solved on LunarLander vs V4's 2/3. Two contributors: (1) PER amplified Q-divergence on seed 115 — high-TD-error transitions ARE the divergent ones; (2) Python sum-tree was the wall-clock bottleneck (134 min/seed = 5x V4's 26 min). **Honest negative result documented as portfolio asset, not embarrassment**. Paper claims do not auto-replicate without retuning (Henderson 2018 in miniature)
- **Best-checkpoint tracking matters for DQN-family**: V3 seed 115 demonstrated catastrophic forgetting late in training (rolling-100 +436 at ep 700, +125 at ep 900, eval-on-final-weights +149). Added best-rolling-100 weight tracking in V4/V5; deployed weights are the peak observed during training, not final-epoch
- **Multi-seed reporting catches failures invisible to single-seed**: V3 seed 115 collapsed; V4 seed 114 had a 400-episode catastrophic-forgetting window; V5 all 3 seeds underperformed. Single-seed reporting could have shown +500 (lucky) or +149 (unlucky) for V3 — neither is the truth. Following Henderson 2018's standard for portfolio honesty
- CartPole-scale epsilon decay is not Atari-scale: first V2 run used `eps_decay_steps=50_000`. Result: 1/3 seeds barely cleared random because epsilon was still 0.4-0.5 at end of training. Fixed to 5,000; all 3 seeds then hit 500.00. **Lesson: decay budget must scale to total env-step budget, not be ported rotely from Atari conventions**
- **Deployed as D4 in Phase 4** (planned) — the V1 Q-table is the simplest deployment artifact in the portfolio (`(500, 6)` numpy array, ~12 KB)

## Progress Log (PyTorch entries, chronological)

| Date | Model | Notes |
|---|---|---|
| 2026-04-26 | Q-Learning | 5 variants on 3 envs. V1 Tabular **+8.38** (Taxi-v4). V2 DQN **+500.00 perfect 3/3** (CartPole-v1). V3 Double DQN -32% overestimation. V4 Dueling DQN **+206.90 mean, 2/3 solved** (LunarLander-v3). V5 PER honest negative result (0/3 seeds, 5x V4 wall-clock) |
| 2026-04-23 | VAE | 5 variants on MNIST + CIFAR-10. V1 Vanilla NLL **101.98**, **V5 VQ-VAE+PixelCNN FID 117.27** (DALL-E 1 two-stage recipe). Built from `nn.Linear`/`nn.Conv2d`/`nn.Embedding` + masked-conv causality |
| 2026-04-19 | GNN | 4 variants on Cora + ogbn-arxiv: V3 GAT (Cora **0.8310** matches Velickovic / arxiv **0.7025**) |
| 2026-04-15 | Vision Transformers | 4 variants on CIFAR-100. V1 Vanilla 50.33% → V4 Pre-trained ViT-B/16 **91.16%**. Built from `nn.Linear` with no `nn.TransformerEncoder` |
| 2026-04-11 | Transformers | 5 variants across 2 tasks. Translation: Beam Search **0.3625**. Classification (AG News): DistilBERT fine-tuned **94.45%**. Built from `nn.Linear` with no `nn.Transformer` |
| 2026-04-05 | Attention | 4 variants: **Bahdanau (BLEU 0.380)** Pre-GRU context injection >> post-GRU attention on short sentences |
| 2026-04-02 | GANs | 4 variants: Vanilla (FID 261), **DCGAN (FID 30.57)**, WGAN-GP (FID 55), cGAN (FID 148). Progressive build |
| 2026-03-29 | LSTM | ECG: LSTM-128 **0.603 macro F1** (augmented, broke 0.55 ceiling). IMDB: LSTM-128 **87.8% acc**, 0.946 AUC. Sequence length ablation (300 optimal) |
| 2026-03-27 | RNN | GRU-128 (2 layers), **91.8% accuracy, 0.55 macro F1** on ECG5000. Vanilla RNN vs GRU + gradient flow analysis |
| 2026-03-25 | CNN | ResNet-20 + CutMix + Label Smoothing + Nesterov SGD, **80.1% accuracy** on CIFAR-100. Progression 56.9% → 80.1%. Superclass accuracy 87.9% |
| 2026-03-20 | Autoencoders | GPU conv denoising AE (64-128-256, lat=256), MSE **0.0037** (3.6x better than SK). Architecture sweep + noise level sweep. 0.07 µs/sample |
| 2026-03-18 | DNN | GPU-accelerated RegularizedDNN 256-128, **96.03% accuracy**, 96.02% F1 |
| 2026-03-15 | PCA | GPU eigendecomposition fastest (0.11s fit, 0.39 µs/sample). 9.1x GPU vs CPU speedup |
| 2026-03-10 | SVM | GPU dual gradient descent (9.03s, 17.7x faster than NF). 0.59 µs/sample |
| 2026-03-04 | Decision Trees & RF | Hybrid CPU/GPU DT+RF. 16% faster training than No-Framework |
| 2026-02-27 | Naive Bayes | GPU-accelerated NB on RTX 4090. Fastest: 0.028s training, 3.5 µs/sample |
| 2026-02-22 | K-Means | GPU `torch.cdist` + `torch.vmap`/`torch.compile` showcases |
| 2026-02-14 | KNN | GPU-accelerated `torch.cdist`, 7.2 GB VRAM. 93.77% accuracy, 1,164/sec |
| 2026-02-07 | Logistic Regression | Autograd + SGD, 7.8x faster than No-Framework (2.36s) |
| 2026-02-07 | Linear Regression | Autograd vs manual gradients. Slower (3.44s) and more memory (54 MB) |

## Cross-cutting takeaways

1. **PyTorch's eager-mode + dynamic-graph design is decisive for research-pace work**: every variant exploration in #11-#20 (CNN's 10-step progression, Transformers' 5-variant beam-search progression, GAT/GIN comparison, VAE's failed-three-times debug, Q-Learning's multi-seed catastrophic-forgetting analysis) leveraged the ability to print, inspect, and re-run mid-forward without graph recompilation.

2. **From-scratch is idiomatic in PyTorch**: every deep learning model in #11-#20 was built from `nn.Linear` / `nn.Conv2d` / `nn.Embedding` primitives — no `nn.Transformer`, no `nn.MultiheadAttention`, no `torch_geometric.GCNConv` for the V1 implementations. This produces educational depth that high-level wrappers can't.

3. **GPU acceleration boundary is sharper than expected**: GPU dominates compute-heavy operations (matmul-bound: SVM, PCA, DNN, SVMs, anything with attention) but loses on tree structures (DT/RF), small batches (K-Means at 10K samples), and Python-recursion-heavy patterns. The RTX 4090 is a 17.7x SVM speedup AND a 1.6x slowdown for tree inference vs NumPy — same hardware, different problem shapes.

4. **Honest negative results are deliberate portfolio assets**: V5 PER 0/3 seeds, V4 GIN underperforming V1 GCN, V3 ViT distillation -12.75pp below CNN, V4 ViT pre-trained weights purged from git history. None of these are bugs — all are documented design boundaries that match real practitioner experience.

5. **PyTorch wins 14 of 17 deployment slots not because the framework is better, but because deep-learning workloads dominate #09-#20, and PyTorch's iteration speed advantage compounds across the variant-exploration patterns we used**. TF won #16 Translation specifically because Keras's recipe-style training was actively helpful for production seq2seq — a problem class where the PyTorch advantage doesn't apply.
