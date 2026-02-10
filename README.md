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
├── data/
│   ├── raw/
│   │   ├── vehicles.csv
│   │   └── creditcard.csv
│   └── processed/
│       ├── vehicles_clean.csv
│       ├── encoding_mappings.json
│       └── logistic_regression/
│           ├── X_train.npy
│           ├── X_test.npy
│           ├── y_train.npy
│           ├── y_test.npy
│           └── preprocessing_info.json
├── data-preperation/
│   └── preprocess_logistic.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── performance.py
│   └── visualization.py
├── No-Framework/
│   ├── 01-linear-regression/
│   └── 02-logistic-regression/
├── Scikit-Learn/
│   ├── 01-linear-regression/
│   └── 02-logistic-regression/
├── PyTorch/
│   ├── 01-linear-regression/
│   └── 02-logistic-regression/
└── TensorFlow/
    ├── 01-linear-regression/
    └── 02-logistic-regression/
```

Each model subfolder contains: pipeline notebook/script, README with framework notes/time estimates, results (plots/metrics), and data loading consistent with root guidelines.

## Progress Log

(Newest entries at top; grows downward as we complete models)

- 2025-02-09 | Logistic Regression / No-Framework | Manual sigmoid, BCE loss, gradient descent. 18.3s training, 83% recall on fraud detection. | [No-Framework/02-logistic-regression](No-Framework/02-logistic-regression/)
- **2025-02-08 | Linear Regression Summary: *All 4 frameworks achieve identical accuracy: R²=0.50, RMSE=$10,105***
- 2025-02-08 | Linear Regression / TensorFlow | Keras model.fit() abstraction. Slowest (23.58s) but simplest code. | [TensorFlow/01-linear-regression](TensorFlow/01-linear-regression/)
- 2025-02-07 | Linear Regression / PyTorch | Autograd vs manual gradients. Slower (3.44s) and more memory (54MB) due to computational graph overhead. | [PyTorch/01-linear-regression](PyTorch/01-linear-regression/)
- 2025-02-05 | Linear Regression / Scikit-Learn | Normal Equation vs Gradient Descent. 13x faster, 7.5x more memory. Same results with 90% less code. | [Scikit-Learn/01-linear-regression](Scikit-Learn/01-linear-regression/)
- 2025-02-04 | Linear Regression / No-Framework | Built from scratch with NumPy: gradient descent, MSE cost, z-score scaling. | [No-Framework/01-linear-regression](No-Framework/01-linear-regression/)

## How to Run / Setup

1. Python 3.10+ recommended.
2. Install dependencies per framework (see each subfolder's README for specifics; common: numpy, pandas, matplotlib, scikit-learn, torch, tensorflow).
3. Navigate to a model subfolder and run the notebook/script.
4. Use consistent random seeds 113 for reproducibility.

## Overall Learnings & Conclusions

(Updated over time)

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
- Complete Logistic Regression — *In Progress (No-Framework ✓, 3 remaining)*
- Complete remaining beginner models (KNN, K-Means, Naive Bayes)
- Add deployment examples (Flask/Streamlit wrappers)
- Explore real-world datasets beyond toys
- Compare inference speed and memory on larger inputs

## License

MIT License. See the [LICENSE](LICENSE) file for details.
