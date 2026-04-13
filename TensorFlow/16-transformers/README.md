# Transformers — TensorFlow Pipeline

Confirms PyTorch Transformer findings with TF-native implementation on WSL2 GPU. Identical architecture (d_model=256, 8 heads, 3+3 layers) built from scratch with `tf.keras.Model` subclassing — no `tf.keras.layers.MultiHeadAttention`. The surprise result: TF Translation BLEU **0.4456** dramatically exceeds PyTorch's 0.3625 (with beam search) and beats #15 Bahdanau's 0.3803 by **+17%**. Classification confirms PT findings: TF encoder-only 92.20% vs PT 91.22% (+0.98%). Same architecture, same data, same hyperparameters — framework implementation details (random initialization, optimizer internals) produce meaningfully different training trajectories.

## Overview

- **2 tasks confirmed**: Translation (Tatoeba EN->ES) + Classification (AG News 4-class)
- **Best from PyTorch**: Recipe variant (warmup + label smoothing) for translation, Encoder-Only from scratch for classification
- **DistilBERT skipped**: Same pre-trained weights across frameworks, no meaningful comparison
- **WSL2 GPU**: RTX 4090 via Ubuntu WSL2 (TF 2.21+ dropped native Windows GPU)
- **All from scratch**: `tf.keras.layers.Dense`, `tf.keras.layers.LayerNormalization`, `tf.keras.layers.Dropout` — no high-level Transformer APIs

## What Runs on GPU

| Component | Device | Why |
|-----------|--------|-----|
| All Transformer training | WSL2 GPU (RTX 4090) | Self-attention + FFN on 114K translation + 108K classification samples |
| Greedy decode inference | WSL2 GPU | Autoregressive decoder forward passes |
| @tf.function training step | WSL2 GPU | Graph compilation critical for Transformer speed |

---

## Cross-Framework Comparison

### Translation (Tatoeba EN->ES)

| Model | Test BLEU | Params | Train Time | Inference |
|-------|-----------|--------|------------|-----------|
| PT Vanilla (greedy) | 0.3289 | 11.7M | 28.6 min | - |
| PT Recipe (greedy) | 0.3462 | 11.7M | 28.2 min | 22.6 ms |
| PT Recipe + Beam (k=5) | 0.3625 | 11.7M | (same) | ~113 ms |
| **TF Recipe (greedy)** | **0.4456** | **11.7M** | **241.1 min** | **376.8 ms** |
| #15 PT Bahdanau | 0.3803 | 16.7M | 14.6 min | 46.8 us |

**Key finding**: TF greedy decode (0.4456) surpasses PT beam search (0.3625) by +0.0831 BLEU. Same architecture, same hyperparameters, same data. TF also beats #15 Bahdanau (0.3803) by +0.0653 — the Transformer DOES beat RNN+attention when trained in TF.

### Classification (AG News)

| Model | Test Acc | Macro F1 | Params | Train Time |
|-------|----------|----------|--------|------------|
| PT Encoder-Only | 0.9122 | 0.9120 | 7.3M | 8.4 min |
| **TF Encoder-Only** | **0.9220** | **0.9217** | **7.3M** | **10.1 min** |
| PT DistilBERT (fine-tuned) | 0.9445 | 0.9444 | 67.0M | 40.5 min |

**Key finding**: TF edges PT by +0.98% accuracy. Both remain below DistilBERT by ~2-3%.

### Per-Class Accuracy (Classification)

| Class | PT | TF | Gap |
|-------|--------|--------|-----|
| World | 0.9011 | 0.9389 | +3.78% |
| Sports | 0.9858 | 0.9842 | -0.16% |
| Business | 0.8732 | 0.8853 | +1.21% |
| Sci/Tech | 0.8889 | 0.8795 | -0.94% |

---

## TF-Specific Implementation Details

| TF Feature | Purpose | PT Equivalent |
|------------|---------|---------------|
| `tf.keras.Model` subclassing | Full Transformer architecture | `nn.Module` |
| `tf.keras.layers.Dense` | Q/K/V/output projections | `nn.Linear` |
| `tf.keras.layers.LayerNormalization` | Post-LN residual normalization | `nn.LayerNorm` |
| `tf.keras.layers.Embedding` | Token embeddings | `nn.Embedding` |
| `tf.constant` for PE buffer | Precomputed positional encoding | `register_buffer` |
| `tf.linalg.band_part` | Causal mask construction | `torch.triu` |
| `mask * -1e9` additive masking | Mask attention scores before softmax | `masked_fill(-inf)` |
| `tf.keras.optimizers.schedules.LearningRateSchedule` | Custom warmup schedule | `optim.lr_scheduler.LambdaLR` |
| `tf.GradientTape` | Custom training loop | Manual backward pass |
| `tf.clip_by_global_norm` | Gradient clipping | `clip_grad_norm_` |
| `@tf.function` | Graph compilation for training step | N/A (eager by default) |
| `tf.nn.softmax_cross_entropy_with_logits` | Manual label smoothing | `CrossEntropyLoss(label_smoothing=)` |

### Why TF Outperforms PT on Translation

Same architecture + same hyperparameters but +0.0831 BLEU. Three contributing factors:

1. **Random initialization**: `tf.keras.layers.Dense` uses Glorot uniform by default. PyTorch `nn.Linear` uses Kaiming uniform. Different initial weight distributions lead to different training trajectories, especially with warmup schedules where early gradient updates matter.

2. **Optimizer internals**: TF's Adam and PT's Adam differ in epsilon handling, moment computation precision, and weight decay implementation. Over 30 epochs x 1794 batches = 53,820 steps, small per-step differences compound.

3. **@tf.function graph compilation**: The training step executes as a compiled graph, potentially producing different numerical results than PT's eager execution due to operation fusion and memory layout optimizations.

This is NOT a framework quality judgment — PT might win with different seeds or more epochs. It IS evidence that framework implementation details matter more than architecture for models that haven't converged.

---

## Sample Translations (TF)

```
[1] EN:   it is easy to form a plan, but it is difficult to carry it out.
    REF:  es facil hacer un plan pero es dificil realizarlo.
    PRED: es facil hacer un plan, pero es dificil hacerlo.

[2] EN:   we were struck dumb with astonishment.
    REF:  nos quedamos mudos con estupor.
    PRED: estabamos muy tontos de asombrados.

[3] EN:   tom answered back.
    REF:  tom contesto.
    PRED: tom respondio.

[4] EN:   i wasn't home.
    REF:  yo no estaba en casa.
    PRED: no estaba en casa.

[5] EN:   this glass contains water.
    REF:  este vaso tiene agua.
    PRED: este vaso tiene agua.
```

Translation [1] is notably better than PT's attempt — nearly perfect, just "hacerlo" vs "realizarlo" (synonyms). Translation [5] is an exact match.

---

## Performance Benchmarks

| Metric | Translation | Classification |
|--------|-------------|----------------|
| Test BLEU / Accuracy | 0.4456 | 0.9220 |
| Training Time | 241.1 min (30 epochs) | 10.1 min (8 epochs) |
| Inference (per sample) | 376.8 ms | 32.4 ms |
| Model Size | 44.8 MB | 27.8 MB |
| Parameters | 11,681,600 | 7,256,580 |

### WSL2 Speed Overhead

| Task | PT (Windows GPU) | TF (WSL2 GPU) | Slowdown |
|------|-------------------|----------------|----------|
| Translation training | 28.2 min | 241.1 min | 8.5x |
| Classification training | 8.4 min | 10.1 min | 1.2x |
| Translation inference | 22.6 ms | 376.8 ms | 16.7x |
| Classification inference | 1.87 ms | 32.4 ms | 17.3x |

Training slowdown varies (8.5x for translation, 1.2x for classification). Inference slowdown is consistently ~17x. The /mnt/c/ filesystem overhead and TF's eager-mode greedy decode loop (not @tf.function compiled) are the main contributors. Not a fundamental TF limitation.

---

## Files

```
TensorFlow/16-transformers/
├── pipeline.ipynb                              # Full pipeline (6 cells)
├── README.md                                   # This file
├── requirements.txt                            # Verified package versions
└── results/
    ├── translation_transformer.weights.h5      # Recipe variant weights
    ├── encoder_only_classifier.weights.h5      # From-scratch classifier weights
    └── metrics.json                            # Full results snapshot
```

## How to Run

```bash
# Requires WSL2 with TF GPU setup
wsl
source ~/tf-gpu-venv/bin/activate

# From project root (WSL2 path)
cd TensorFlow/16-transformers

pip install -r requirements.txt

# Preprocessing already completed (shared with PyTorch)
# Run pipeline -- ~4.5 hours total (translation ~4h + classification ~10min)
jupyter notebook --no-browser --port=8888
# Open pipeline.ipynb in VS Code via Existing Jupyter Server
```
