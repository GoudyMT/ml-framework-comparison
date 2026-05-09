# Modeling Findings: Scikit-Learn

> Implemented #01-#10, retired after Autoencoders. Won deployment for #06 DT/RF, #07 SVM, and #08 PCA. Normal Equation solving in 0.03s vs No-Framework's 0.38s established the speed-vs-understanding trade-off early.

## Why Scikit-Learn was retired after #10

Hit its ceiling at #10 Autoencoders — sklearn's `MLPRegressor` could not compete with PyTorch's conv denoising AE (MSE 0.0037 vs sklearn's 0.0103). The line between "classical ML" and "deep learning" is approximately the line between "sklearn dominates" and "PT/TF dominates":

- Full-batch only training (no mini-batch SGD)
- No convolutional layers
- No batch normalization, no dropout per-layer customization
- MLP-only neural net architecture

Future models (#11 CNN onward) continue with PyTorch and TensorFlow only.

## Models Implemented (#01-#10)

| # | Model | Folder | Deployment Win? |
|---|---|---|---|
| 01 | Linear Regression | [Scikit-Learn/01-linear-regression](../../Scikit-Learn/01-linear-regression/) | — |
| 02 | Logistic Regression | [Scikit-Learn/02-logistic-regression](../../Scikit-Learn/02-logistic-regression/) | — |
| 03 | KNN | [Scikit-Learn/03-knn](../../Scikit-Learn/03-knn/) | — |
| 04 | K-Means | [Scikit-Learn/04-k-means](../../Scikit-Learn/04-k-means/) | — |
| 05 | Naive Bayes | [Scikit-Learn/05-naive-bayes](../../Scikit-Learn/05-naive-bayes/) | — |
| 06 | Decision Trees / RF | [Scikit-Learn/06-decision-trees-random-forests](../../Scikit-Learn/06-decision-trees-random-forests/) | ✓ Deploy winner |
| 07 | SVM | [Scikit-Learn/07-svm](../../Scikit-Learn/07-svm/) | ✓ Deploy winner |
| 08 | PCA | [Scikit-Learn/08-pca](../../Scikit-Learn/08-pca/) | ✓ Deploy winner (D1) |
| 09 | DNN | [Scikit-Learn/09-dnn](../../Scikit-Learn/09-dnn/) | — |
| 10 | Autoencoders | [Scikit-Learn/10-autoencoders](../../Scikit-Learn/10-autoencoders/) | — (last SK model) |

## Per-Model Findings

### #01 Linear Regression

- **Best for simple ML**: Normal Equation solves instantly (0.03s), 90% less code than manual implementation. R² ≈ 0.50, RMSE ≈ $10,100
- 13x faster than No-Framework's manual gradient descent (0.38s) — established the early speed-vs-understanding trade-off
- Memory vs speed trade-off: SK uses 15 MB (vs NF's 2 MB) but trades it for 13x speed

### #02 Logistic Regression

- **L-BFGS solver dominates speed**: 0.32s training, 57x faster than No-Framework
- All 4 frameworks achieved similar recall (~83%) on fraud detection
- Class imbalance is the real challenge: 98.9% accuracy is misleading; precision (12%) matters more than accuracy for fraud detection
- SMOTE + filtering works well: oversampling then filtering unrealistic samples creates balanced training without losing model quality

### #03 KNN

- **KD-tree wins**: O(log n) lookups beat brute-force GPU computation for 464K training samples (57s vs PyTorch GPU's 100s)
- 2,000 predictions/sec — 1.7x faster than PyTorch GPU's 1,164/sec, ~1,300x faster than No-Framework's 1.5/sec
- All 4 frameworks achieved 93.77% accuracy with K=3 manhattan distance via GridSearchCV tuning

### #04 K-Means

- **Fastest of all 4 frameworks**: 0.06s training (vs PT GPU 0.34s, NF 1.02s, TF CPU 2.01s) — KMeans + MiniBatchKMeans comparison
- K=7 chosen for ARI evaluation (matches ground truth bean types); silhouette peaks at K=3 (3 natural geometric groupings)
- 0.3061 silhouette, 0.6686 ARI — within rounding of all other frameworks

### #05 Naive Bayes

- **CalibratedClassifierCV is the framework-specific showcase**: ECE drops from 0.32 to 0.14 with calibration
- GaussianNB on Breast Cancer: 89.5% accuracy (30 continuous features). MultinomialNB on 20 Newsgroups: 66.8% accuracy (10K TF-IDF features)
- All 4 frameworks produce identical metrics: accuracy 0.6683, macro F1 0.6394, log-loss 1.5576
- Largest model size (3.05 MB) — sklearn keeps diagnostic arrays alongside parameters; NF's pure-NumPy implementation is 2x smaller

### #06 Decision Trees / RF — DEPLOYMENT WIN

- **GridSearchCV tuning produces best F1 (0.48)** and AUC (0.80). 21s training (83x faster than from-scratch's 29 minutes)
- Cython under the hood does the heavy lifting — sklearn's `DecisionTreeClassifier` is C-compiled; pure-Python alternatives can't compete
- **First MLflow integration + model export (joblib)** — established the deployment pattern reused by all subsequent winners
- Bank Marketing dataset (41,188 samples, 19 features, 88.7/11.3 imbalanced). `duration` feature dropped for data leakage
- Economic indicators dominate: euribor3m, nr.employed, emp.var.rate top features across DT and RF — client demographics matter less than macroeconomic conditions
- F1 > accuracy for imbalanced data: 85.5% accuracy sounds good but tuned RF recall of 60.1% means 40% of subscribers still missed

### #07 SVM — DEPLOYMENT WIN

- **Best calibration**: AUC 0.9164, log-loss 0.3486 — sklearn's optimized SMO produces the smoothest probability outputs
- **Fewest support vectors (5,343 at 35.3%)** vs from-scratch's 11,426 (75.5%) — better convergence, more compact model
- Polynomial kernel chosen via GridSearchCV (degree=3, C=10). Kernel comparison showcase locked the choice for cross-framework comparison
- Slightly different from from-scratch frameworks (0.8606 acc vs NF/PT/TF's 0.8611) due to optimized SMO vs our dual gradient descent — same dual objective, different convergence path
- 36.63 µs/sample inference — slower than PyTorch GPU's 0.59 µs but with cleaner probability outputs

### #08 PCA — DEPLOYMENT WIN (D1)

- **IncrementalPCA + sklearn Pipeline** — out-of-core support, lowest memory (11.74 MB) of all 4 frameworks
- 150 components retain 90.85% variance, KNN downstream accuracy 85.99% — same as every other framework (eigendecomposition is implementation-agnostic)
- 0.19s fit, 0.52 µs/sample — second only to PyTorch GPU's 0.11s/0.39 µs
- Sample covariance (1/(n-1)) vs population (1/n in NF/PT/TF) — shifts variance thresholds but component ordering and downstream accuracy are identical
- **Deployed as D1 in Phase 1** — see [deployment/services/sklearn-svc/](../../deployment/services/sklearn-svc/) for the production wrapper

### #09 DNN

- **MLPClassifier matches DL frameworks within 1% accuracy** — 94.91% with 128-64 architecture (vs PT's 96.03% with 256-128 + BatchNorm)
- 2.42s training (fastest of the 3 frameworks) — but cannot express BatchNorm, per-layer Dropout, or LR scheduling
- Activation function is nearly irrelevant on pre-engineered features: ReLU 94.40%, Tanh 94.37%, Logistic 94.77% — all within 0.4%. Real difference is convergence speed (ReLU 30 epochs vs Logistic 73)
- 0.65 µs/sample inference — competitive with PyTorch GPU (0.35 µs), much faster than TF CPU (31.68 µs)
- **The framework-specific showcase**: activation function comparison across logistic/tanh/ReLU
- UCI HAR dataset: 10,299 samples, 561 pre-engineered sensor features, 6 activity classes (subject-wise split, no leakage)
- SITTING vs STANDING is the performance ceiling — 72 misclassifications between these classes (sensor profiles nearly identical when phone is in pocket)

### #10 Autoencoders — last sklearn model

- **Dense AE only**: MLPRegressor with 128-dim bottleneck, MSE 0.0133, 24x compression ratio
- 3.6x worse MSE than PyTorch's conv denoising AE (0.0037 with 64-128-256 filters) — the compute boundary that ended sklearn for the project
- Bottleneck dimension sweep (the framework-specific showcase): tested 64/128/256/512 latent sizes
- KNN accuracy on latent: 34.3% (worst of 3 frameworks) — sklearn's MLPRegressor learns less class-separable features than PT/TF dense AE
- **Last sklearn model — retired after this**

## Progress Log (Scikit-Learn entries, chronological)

| Date | Model | Notes |
|---|---|---|
| 2026-03-20 | Autoencoders | Dense AE (MLPRegressor), 128-dim bottleneck, MSE 0.0133, 24x compression. **SK's LAST model — retired** |
| 2026-03-17 | DNN | MLPClassifier 128-64 architecture, 94.91% accuracy, 94.93% F1 |
| 2026-03-13 | PCA | 150 components retain 90.85% variance, KNN accuracy 85.99%. IncrementalPCA showcase. 0.19s fit, 0.52 µs/sample |
| 2026-03-09 | SVM | Poly kernel SVC (C=10, F1 0.89, AUC 0.92). Kernel comparison showcase + MLflow + model export |
| 2026-03-01 | Decision Trees & RF | GridSearchCV tuned RF (F1 0.48, AUC 0.80). **First MLflow + model export — pattern reused for all future winners** |
| 2026-02-25 | Naive Bayes | GaussianNB (89.5%) + MultinomialNB (66.8%). CalibratedClassifierCV: ECE 0.32 → 0.14 |
| 2026-02-18 | K-Means | KMeans + MiniBatchKMeans comparison. K=7, 0.3061 silhouette, 0.6686 ARI. Fastest framework |
| 2026-02-12 | KNN | GridSearchCV tuning, K=3 manhattan distance. 93.77% accuracy. KD-tree dominates |
| 2026-02-09 | Logistic Regression | L-BFGS solver, 57x faster than No-Framework (0.32s) |
| 2026-02-05 | Linear Regression | Normal Equation vs Gradient Descent. 13x faster, 7.5x more memory. 90% less code |

## Cross-cutting takeaways

1. **Sklearn is the speed champion for classical ML**: in #01-#08, sklearn was either fastest or within 2x of fastest on every model. The L-BFGS / SMO / Cython kernels are decades of optimization that pure-Python (NF) and eager-tensor (PT/TF on CPU) implementations can't match without GPU acceleration.

2. **Sklearn is also the deployment standard for classical ML**: 3 of 5 D1-D5 deployments are sklearn artifacts (DT/RF, SVM, PCA — though only PCA reached production). The MLflow + joblib pattern established at #06 became the project-wide deployment template.

3. **Sklearn's ceiling is exactly where its design says it is**: classical ML + MLPRegressor. The retirement at #10 wasn't a loss — it was sklearn doing exactly what it was designed for, then handing off to PyTorch/TensorFlow when the work moved to deep learning architectures sklearn can't express.

4. **Calibration matters more than raw accuracy**: SVM's +0.0033 AUC over from-scratch (0.9164 vs 0.9105) and Naive Bayes's ECE 0.32→0.14 with `CalibratedClassifierCV` show sklearn's probabilistic outputs are production-ready in ways from-scratch implementations aren't.
