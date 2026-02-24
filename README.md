# ML Framework Comparisons: From Scratch to Production-Ready

U.S. Navy veteran (promoted to senior rank in under 6 years) transitioning to ML/AI Engineering. Over the past 1.5 years, I have built and tuned models achieving 91-98% accuracy and F1 scores up to 0.97 using Scikit-learn, XGBoost, and modern NLP techniques. Previously led 40-person technical teams and restored $7.8M in critical navigation/comms systems under deployment conditions. Currently pursuing a B.A. in Computer Science (expected Apr 2027). This repository combines my ML engineering hands-on work with proven leadership and systems-thinking, aimed at high-impact roles in Big Tech or defense.

This project is my hands-on portfolio to deepen understanding of machine learning and deep learning pipelines. I implement the same models across four approaches: Scikit-Learn (high-level classical ML), PyTorch (flexible deep learning), TensorFlow (production-oriented deep learning), and No-Framework (pure NumPy/SciPy from scratch). The goal is to compare how each makes building easier (time saved, built-in tools) or better suits certain data situations (tabular vs. images/sequences, scalability, custom needs). Every model of the same type uses the identical dataset for fair metric comparisons (e.g., same MSE or accuracy).

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

## Project Rules & Philosophy

1. Each model is hand-typed, with no auto-fill, AI copy-paste, or external file management tools.
2. Models covering the same type of ML (e.g., Logistic Regression) utilize the same dataset across all applicable frameworks.
3. AI guidance is acceptable for ideas, debugging, structure suggestions, and explanations, but it cannot break rule 1.
4. Utilize best comment practices, writing code comments as if providing instructions to someone who has never read code before.
5. Reproducibility first: Use a fixed random seed (113) in all random operations across frameworks.
6. Consistent evaluation: Apply identical train/test splits and metrics for models of the same type to enable direct comparisons.
7. Commit often with meaningful messages: Treat this repo like professional work—commit after each major pipeline step. Use descriptive commit messages to show iterative development process.
8. No external heavy dependencies beyond core libraries: Stick to standard installs without niche packages unless absolutely needed for a specific advanced model. Document exact versions in a root requirements.txt.

## Key Highlights & Framework Insights

(Updated as models complete)

- Scikit-Learn: Fastest prototyping for tabular/classical ML; built-in pipelines and cross-validation save significant time.
- PyTorch: Dynamic computation graphs excel for custom architectures, sequences, and research-like flexibility.
- TensorFlow: Strong for production workflows, with Keras for rapid builds and tools for deployment/scaling.
- No-Framework: Reveals core math (manual gradients, matrix operations); slower but builds deepest conceptual understanding.

## Table of Contents

- [Models Covered](#models-covered)
- [Folder Structure](#folder-structure)
- [Shared Utilities Architecture](#shared-utilities-architecture)
- [Progress Log](#progress-log)
- [How to Run / Setup](#how-to-run--setup)
- [Overall Learnings & Conclusions](#overall-learnings--conclusions)
- [Future Plans](#future-plans)
- [License](#license)

## Models Covered

Models progress from beginner (basic concepts) to advanced (latest deep learning tech). Not all are implemented in every framework due to practicality.

| Model Name                        | Type
|-----------------------------------|----------------------------
| Linear Regression                 | Supervised Regression
| Logistic Regression               | Supervised Classification
| K-Nearest Neighbors (KNN)         | Supervised (non-parametric)  
| K-Means Clustering                | Unsupervised Clustering
| Naive Bayes                       | Supervised Probabilistic
| Decision Trees / Random Forests   | Supervised Ensemble
| Support Vector Machines (SVM)     | Supervised with Kernels
| Principal Component Analysis      | Unsupervised Dim Reduction
| Deep Neural Networks (DNN)        | Supervised Feedforward
| Convolutional Neural Networks     | Image Supervised
| Recurrent Neural Networks (RNN)   | Sequence Supervised
| Long Short-Term Memory (LSTM)     | Advanced Sequence
| Autoencoders                      | Unsupervised Reconstruction
| Generative Adversarial Networks   | Unsupervised Generative
| Attention Mechanisms              | Sequence Focus
| Transformers                      | Self-Attention Models
| Vision Transformers (ViT)         | Image Attention
| Graph Neural Networks (GNN)       | Graph Data
| Variational Autoencoders (VAE)    | Probabilistic Generative
| Q-Learning (RL Basics)            | Reinforcement Learning

## Folder Structure

```text
├── README.md
├── LICENSE
├── data/       # .gitignore for entire folder (large files + processed data from data-preperation)
│   ├── raw/
│   │   ├── vehicles.csv
│   │   └── creditcard.csv
│   ├── processed/
│   │   ├── linear_regression/
│   │   ├── logistic_regression/
│   │   ├── knn/
│   │   └── kmeans/
│   └── results/            # Cross-framework comparison JSONs (one per model)
│       └── kmeans.json
├── data-preperation/
│   ├── clean_vehicles.py
│   ├── preprocess_logistic.py
│   ├── preprocess_knn.py
│   └── preprocess_kmeans.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── performance.py
│   ├── visualization.py
│   └── results.py
├── No-Framework/
│   ├── 01-linear-regression/
│   ├── 02-logistic-regression/
│   ├── 03-knn/
│   └── 04-k-means/
├── Scikit-Learn/
│   ├── 01-linear-regression/
│   ├── 02-logistic-regression/
│   ├── 03-knn/
│   └── 04-k-means/
├── PyTorch/
│   ├── 01-linear-regression/
│   ├── 02-logistic-regression/
│   ├── 03-knn/
│   └── 04-k-means/
└── TensorFlow/
    ├── 01-linear-regression/
    ├── 02-logistic-regression/
    ├── 03-knn/
    └── 04-k-means/
```

Each model subfolder contains: pipeline notebook/script, README with framework notes/time estimates, results (plots/metrics), and data loading consistent with root guidelines.

## Shared Utilities Architecture

Began implementation during Logistic Regression and introduced a shared `utils/` package to avoid duplicating code across frameworks.
The package evolves organically: during the planning phase when new model types are started, recurring patterns are identified and moved here.

### Current Utilities

(Newest additions at top; table grows as new model types introduce shared patterns)

| Module | Functions | Added In | Purpose |
|--------|-----------|----------|---------|
| `results.py` | `save_results`, `add_result`, `print_comparison` | K-Means | Cross-framework result saving and comparison |
| `metrics.py` | `inertia`, `silhouette_score`, `silhouette_samples`, `adjusted_rand_index` | K-Means | Unsupervised clustering evaluation |
| `visualization.py` | `plot_elbow_curve`, `plot_silhouette_comparison`, `plot_silhouette_analysis`, `plot_convergence_curve` | K-Means | Clustering visualizations |
| `performance.py` | `track_performance(gpu=True)` | KNN (PyTorch) | GPU memory tracking for PyTorch/TensorFlow |
| `data_loader.py` | `load_processed_data` | KNN | Generic data loader for any model |
| `metrics.py` | `confusion_matrix_multiclass`, `macro_f1_score` | KNN | Multi-class evaluation |
| `visualization.py` | `plot_confusion_matrix_multiclass`, `plot_validation_curve`, `plot_per_class_f1` | KNN | Multi-class visualizations |
| `metrics.py` | `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix_values`, `roc_curve`, `auc_score` | Logistic Regression | Classification evaluation |
| `performance.py` | `track_performance()` | Logistic Regression | Context manager for timing and CPU memory tracking |
| `visualization.py` | `plot_cost_curve`, `plot_confusion_matrix`, `plot_roc_curve`, `plot_feature_importance` | Logistic Regression | Consistent plots across frameworks |

### Benefits

- **Zero inconsistency**: All frameworks use identical metric calculations
- **Faster development**: 3 imports vs 50 lines of boilerplate per notebook
- **Easier maintenance**: Fix a bug once, applies everywhere
- **Framework-agnostic**: Works with NumPy arrays from any framework

### Usage Pattern
```python
from utils.metrics import accuracy, precision, recall, f1_score, auc_score
from utils.performance import track_performance
from utils.visualization import plot_confusion_matrix, plot_roc_curve

# CPU-only tracking (Scikit-Learn, No-Framework)
with track_performance() as perf:
    # Training code here
    pass
print(f"Time: {perf['time']:.2f}s, Memory: {perf['memory']:.2f} MB")

# GPU tracking (PyTorch, TensorFlow)
with track_performance(gpu=True) as perf:
    # GPU training/inference code here
    torch.cuda.synchronize()  # Ensure GPU ops complete
print(f"Time: {perf['time']:.2f}s, GPU Memory: {perf['gpu_memory']:.2f} MB")
```

## Progress Log

(Newest entries at top; grows downward as we complete models)

- 2026-02-22 | K-Means / PyTorch | GPU-accelerated torch.cdist + torch.vmap/torch.compile showcases. 0.3064 silhouette, 0.6684 ARI. | [PyTorch/04-k-means](PyTorch/04-k-means/)
- 2026-02-21 | K-Means / No-Framework | From-scratch Lloyd's algorithm, K-Means++ init. Matches sklearn metrics, 17x slower. | [No-Framework/04-k-means](No-Framework/04-k-means/)
- 2026-02-18 | K-Means / Scikit-Learn | KMeans + MiniBatchKMeans comparison. K=7, 0.3061 silhouette, 0.6686 ARI. | [Scikit-Learn/04-k-means](Scikit-Learn/04-k-means/)
- 2026-02-17 | K-Means utilities | Preprocessing script, results.py, clustering metrics + visualizations in utils/ | [utils/](utils/)
- **2026-02-15 | KNN Summary: *All 4 frameworks achieve 93.77% accuracy | Scikit-Learn KD-tree fastest (57s)***
- 2026-02-15 | KNN / TensorFlow | Chunked broadcasting on CPU (TF 2.11+ no Windows GPU). 93.77% accuracy, 110/sec. | [TensorFlow/03-knn](TensorFlow/03-knn/)
- 2026-02-14 | KNN / PyTorch | GPU-accelerated torch.cdist, 7.2GB VRAM. 93.77% accuracy, 1,164/sec. | [PyTorch/03-knn](PyTorch/03-knn/)
- 2026-02-14 | KNN / No-Framework | Manual Manhattan distance + weighted voting. 93.79% accuracy, ~1,300x slower. | [No-Framework/03-knn](No-Framework/03-knn/)
- 2026-02-12 | KNN / Scikit-Learn | GridSearchCV tuning, K=3 manhattan distance. 93.77% accuracy. | [Scikit-Learn/03-knn](Scikit-Learn/03-knn/)
- **2026-02-10 | Logistic Regression Summary: *All 4 frameworks achieve 83% recall on fraud detection | 70% Time saved with `utils/`***
- 2026-02-10 | Logistic Regression / TensorFlow | Keras model.fit() abstraction. Slowest (52.95s). | [TensorFlow/02-logistic-regression](TensorFlow/02-logistic-regression/)
- 2026-02-10 | Logistic Regression / PyTorch | Autograd + SGD, 7.8x faster than No-Framework (2.36s). | [PyTorch/02-logistic-regression](PyTorch/02-logistic-regression/)
- 2026-02-09 | Logistic Regression / Scikit-Learn | L-BFGS solver, 57x faster than No-Framework (0.32s). | [Scikit-Learn/02-logistic-regression](Scikit-Learn/02-logistic-regression/)
- 2026-02-09 | Logistic Regression / No-Framework | Manual sigmoid, BCE loss, gradient descent. 18.3s training. | [No-Framework/02-logistic-regression](No-Framework/02-logistic-regression/)
- **2026-02-08 | Linear Regression Summary: *All 4 frameworks achieve identical accuracy: R²=0.50, RMSE=$10,105***
- 2026-02-08 | Linear Regression / TensorFlow | Keras model.fit() abstraction. Slowest (23.58s) but simplest code. | [TensorFlow/01-linear-regression](TensorFlow/01-linear-regression/)
- 2026-02-07 | Linear Regression / PyTorch | Autograd vs manual gradients. Slower (3.44s) and more memory (54MB) | [PyTorch/01-linear-regression](PyTorch/01-linear-regression/)
- 2026-02-05 | Linear Regression / Scikit-Learn | Normal Equation vs Gradient Descent. 13x faster, 7.5x more memory. 90% less code. | [Scikit-Learn/01-linear-regression](Scikit-Learn/01-linear-regression/)
- 2026-02-04 | Linear Regression / No-Framework | Built from scratch with NumPy: gradient descent, MSE cost, z-score scaling. | [No-Framework/01-linear-regression](No-Framework/01-linear-regression/)

## How to Run / Setup

1. Python 3.10+ recommended.
2. Install dependencies per framework (see each subfolder's README for specifics; common: numpy, pandas, matplotlib, scikit-learn, torch, tensorflow).
3. Navigate to a model subfolder and run the notebook/script.
4. Use consistent random seeds 113 for reproducibility.

## Overall Learnings & Conclusions

(Updated over time)

### K-Means Clustering (In Progress — 3/4 frameworks)

- **First unsupervised model** — no labels during training. Evaluation shifts from accuracy/F1 to inertia, silhouette score, and ARI
- **Metrics match across all 3 frameworks** (inertia ~9,976, silhouette ~0.3064, ARI ~0.6684) — algorithm is implementation-agnostic
- **K=3 vs K=7 tradeoff**: Silhouette peaks at K=3 (3 natural geometric groupings), but K=7 matches ground truth bean types for ARI evaluation
- **Scikit-Learn fastest (0.06s)**, No-Framework 17x slower (1.02s), PyTorch GPU in between (0.34s) — GPU minimal benefit at 10K samples
- **torch.compile limited on Windows**: TorchInductor backend doesn't fully support Windows in PyTorch 2.5.1. Manual broadcasting was 1.9x faster than torch.cdist
- **torch.vmap works**: 1.15x speedup for parallel n_init runs. Modest at this scale but demonstrates vectorized map pattern
- **New `results.py` utility**: Automated cross-framework comparison — `add_result()` collects results as each framework finishes, `print_comparison()` displays aligned table

### K-Nearest Neighbors (Completed)

- **All 4 frameworks achieve identical accuracy** (93.77%) — proving KNN results are implementation-agnostic
- **Scikit-Learn's KD-tree wins**: O(log n) lookups beat brute-force GPU computation for 464K training samples (57s vs 100s)
- **GPU helps but isn't magic**: PyTorch GPU (1,164/sec) is 776x faster than No-Framework (1.5/sec) but still slower than Scikit-Learn's optimized trees (2,000/sec)
- **TensorFlow limited by Windows GPU support**: TF 2.11+ dropped native Windows GPU, forcing CPU-only execution (110/sec). WSL2 planned for neural networks
- **No tf.cdist equivalent**: TensorFlow requires chunked broadcasting for pairwise distances, creating memory management challenges PyTorch avoids with `torch.cdist`
- **Rare classes struggle across all frameworks**: Cottonwood/Willow (0.47% of data) consistently has lowest F1 (~0.81)

### Logistic Regression (Completed)

- **All 4 frameworks achieve similar recall** (82-83%) on fraud detection — consistent results across implementations
- **Scikit-Learn dominates speed**: L-BFGS optimizer converges in 0.32s (57x faster than No-Framework)
- **Class imbalance is the real challenge**: 98.9% accuracy is misleading; precision (12%) matters more than accuracy for fraud detection
- **SMOTE + filtering works well**: Oversampling then filtering unrealistic samples creates balanced training without losing model quality
- **TensorFlow slowest for simple models**: 52.95s due to full-batch overhead, but `model.fit()` provides simplest code

### Linear Regression (Completed)

- **All 4 frameworks achieve identical accuracy** (R²≈0.50, RMSE≈$10,100) — proving framework choice doesn't affect model quality for equivalent algorithms
- **Scikit-Learn is best for simple ML**: Normal Equation solves instantly (0.03s), 90% less code than manual implementation
- **No-Framework builds understanding**: Manual gradient descent reveals the math behind the magic (but 0.38s vs 0.03s)
- **PyTorch/TensorFlow have overhead for simple tasks**: Autograd and Keras abstraction add time (3.4s and 23.6s) but provide foundation for neural networks
- **Memory vs Speed tradeoff**: No-Framework uses least memory (2MB), Scikit-Learn trades memory for speed (15MB), PyTorch uses most (54MB)

### General Insights

- High-level frameworks (Scikit-Learn) accelerate development for standard tasks but hide mechanics
- Deep learning libraries (PyTorch/TensorFlow) offer control over modern architectures while providing tools like autograd/optimizers
- From-scratch builds solidify fundamentals but scale poorly for complex models
- Framework choice depends on data type, team needs, and deployment goals

## Future Plans

- ~~Complete Linear Regression across all 4 frameworks~~
- ~~Complete Logistic Regression across all 4 frameworks~~
- ~~Complete KNN across all 4 frameworks~~
- Complete K-Means across all 4 frameworks (3/4 Completed)
- Complete remaining beginner models (Naive Bayes)
- Add deployment examples (Flask/Streamlit wrappers)
- Explore real-world datasets beyond toys
- Compare inference speed and memory on larger inputs

## License

MIT License. See the [LICENSE](LICENSE) file for details.
