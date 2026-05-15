# ML Framework Comparisons: From Scratch to Production-Ready

U.S. Navy veteran rapidly promoted to E-6 in under 6 years — leading technical teams of up to 44 personnel in high-tempo deployed environments while restoring $7.8M+ in RADAR, navigation, and satellite-communication systems at 94-98% sustained uptime. This portfolio applies that same systems-thinking to ML/AI engineering: **20 models across Scikit-learn, PyTorch, TensorFlow, and from-scratch NumPy** — achieving up to **96.03% accuracy and 0.97 F1**, with production-ready FastAPI deployment pipelines covering GANs, Vision Transformers, GNNs, and reinforcement learning. Currently pursuing a B.A. in Computer Science at SNHU (expected Apr 2027, **3.92 GPA**), targeting high-impact roles in AI/ML engineering and defense technology.

[![CI](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/ci.yml/badge.svg)](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/ci.yml)
[![Build and Push](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/build.yml/badge.svg)](https://github.com/GoudyMT/ml-framework-comparison/actions/workflows/build.yml)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)

## Status

1. **Modeling phase complete (#01-#20).** All four learning paradigms covered: supervised + unsupervised + generative + reinforcement.
2. **Deployment phase IN PROGRESS** — see [`deployment/README.md`](deployment/README.md) for live status.

## Headline Results

20 models trained across 4 frameworks. Selected results — full per-framework breakdown in [`docs/modeling/`](docs/modeling/).

| # | Model | Headline result | Best framework |
|---|---|---|---|
| 01 | Linear Regression | R² 0.50, RMSE $10,100 (all 4 identical) | sklearn (0.03s, 90% less code) |
| 02 | Logistic Regression | 83% recall on fraud detection | sklearn L-BFGS (0.32s, 57x vs NF) |
| 03 | KNN | 93.77% accuracy (all 4 identical) | sklearn KD-tree (2,000/sec) |
| 04 | K-Means | ARI 0.6684 (all 4 identical) | sklearn (0.06s) |
| 05 | Naive Bayes | accuracy 0.6683 (all 4 identical) | PT GPU (3.5 µs/sample) |
| 06 | Decision Trees / RF | F1 0.48 (sklearn-tuned) | sklearn (GridSearchCV + MLflow) |
| 07 | SVM | F1 0.90, AUC 0.92 | sklearn (best calibration AUC 0.9164) |
| 08 | PCA | 0.9085 explained variance, 85.99% downstream KNN | sklearn (deployed as D1) |
| 09 | DNN | **96.03% accuracy** (UCI HAR) | PyTorch GPU (deployed as D2) |
| 10 | Autoencoders | **MSE 0.0037** (3.6x better than sklearn) | PyTorch GPU |
| 11 | CNN | **80.1% on CIFAR-100** (ResNet-20 + CutMix) | PyTorch GPU |
| 12 | RNN | macro F1 0.55 on ECG5000 | PyTorch GPU |
| 13 | LSTM | ECG F1 0.60 + IMDB 87.8% acc | PyTorch GPU |
| 14 | GANs | **FID 30.57** (DCGAN on CIFAR-10) | PyTorch GPU |
| 15 | Attention | **BLEU 0.3803** (Bahdanau pre-GRU context) | PyTorch GPU |
| 16 | Transformers | **BLEU 0.4456** (Translation) + 92.20% (Classification) | TensorFlow Translation (deployed as D5) |
| 17 | Vision Transformers | 91.16% (V4 pre-trained), 67.48% (V3 distillation) | PyTorch GPU |
| 18 | GNN | Cora 0.8310 + arxiv OGB 0.7025 (V3 GAT) | PyTorch GPU |
| 19 | VAE | NLL 101.98 nats + FID 117.27 (DALL-E 1 recipe) | PyTorch GPU |
| 20 | Q-Learning | CartPole +500.00 (3/3 seeds), Taxi-v4 bit-identical parity | PyTorch (deployed as D4) |

5 representative winners are being deployed as production FastAPI services in 3 framework runtimes: **D1 SK PCA**, **D2 PT DNN**, **D3 PT DCGAN**, **D4 PT Q-Learning**, **D5 TF Transformer Translation**.

## Project Rules

1. **Hand-typed models** — no auto-fill, no AI copy-paste
2. **Same dataset across frameworks** for any given model type, enabling direct metric comparison
3. **Reproducibility**: fixed random seed (113) across all frameworks
4. **Identical train/test splits** and metrics for models of the same type
5. **Best-comment practices** — comments written as instructions to someone who has never read code before
6. **Frequent commits** with meaningful messages — repo treated as professional work
7. **Standard installs only** — pinned versions in `requirements.txt`; no niche dependencies unless absolutely needed

## Quick Start

```bash
# Clone + install root dependencies
git clone <repo-url>
cd ml-framework-comparisons
python -m venv .venv && .\.venv\Scripts\activate    # Windows
pip install -r requirements.txt

# Run a modeling notebook (example)
jupyter notebook PyTorch/09-dnn/pipeline.ipynb

# Or boot a deployed service (see deployment/README.md for the full guide)
cd deployment/services/sklearn-svc
.\.venv\Scripts\uvicorn.exe app.main:app --port 8001
# -> http://localhost:8001/docs for Swagger UI
```

## Project Structure (top-level)

```text
.
├── README.md                       # this file
├── docs/                           # design + findings documentation (NEW)
│   └── modeling/                   # per-framework deep dives + cross-framework findings
├── data/                           # raw datasets + preprocessed arrays (.gitignore'd)
├── data-preperation/               # preprocess scripts + EDA notebooks per model
├── utils/                          # shared utilities (#02 onward)
├── No-Framework/                   # #01-#08 only (retired after PCA)
├── Scikit-Learn/                   # #01-#10 only (retired after Autoencoders)
├── PyTorch/                        # all 20 models; deployment winner for 14 of 17 artifacts
├── TensorFlow/                     # #01-#20 with WSL2 cuDNN scope reductions on 5 models
├── deployment/                     # ACTIVE: production FastAPI services for D1-D5
└── results/                        # cross-framework comparison artifacts
```

Full tree with per-folder annotations: [`docs/modeling/folder-structure.md`](docs/modeling/folder-structure.md).

## Documentation

### Modeling phase

- [`docs/modeling/no-framework.md`](docs/modeling/no-framework.md) — pure NumPy/SciPy, #01-#08 (retired after PCA)
- [`docs/modeling/scikit-learn.md`](docs/modeling/scikit-learn.md) — #01-#10 (retired after Autoencoders); won 3 deployment slots
- [`docs/modeling/pytorch.md`](docs/modeling/pytorch.md) — all 20 models; won 14 of 17 deployment artifacts
- [`docs/modeling/tensorflow.md`](docs/modeling/tensorflow.md) — #01-#20 with documented scope reductions; won D5 Translation
- [`docs/modeling/cross-framework-findings.md`](docs/modeling/cross-framework-findings.md) — parity results, framework-vs-framework patterns, general insights
- [`docs/modeling/utilities.md`](docs/modeling/utilities.md) — the shared `utils/` package (architecture + every function added)
- [`docs/modeling/folder-structure.md`](docs/modeling/folder-structure.md) — full project tree with annotations

### Deployment phase

- [`deployment/README.md`](deployment/README.md) — phase progress + architecture + per-service progress
- [`deployment/docs/dependency-strategy.md`](deployment/docs/dependency-strategy.md) — Python/pip-tools/version-pinning rationale
- Per-service READMEs land at [`deployment/services/<svc>/README.md`](deployment/services/) as each service ships

## License

MIT License. See [LICENSE](LICENSE) for details.
