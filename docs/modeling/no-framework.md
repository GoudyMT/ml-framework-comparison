# Modeling Findings: No-Framework (Pure NumPy/SciPy)

> Implemented #01-#08, retired after PCA. From-scratch builds for solidifying math fundamentals — every parameter's role becomes explicit when you write `theta -= lr * gradient` directly.

## Why No-Framework was retired after PCA

Scaled poorly past #08:
- SVM's dual gradient ascent took 5K+ iterations to match sklearn's converged solution
- Matrix-heavy operations couldn't compete with optimized BLAS calls
- Beyond classical ML, framework-level optimizations dominate, and from-scratch becomes teaching theater rather than honest comparison

The line between "classical ML" and "deep learning" is approximately the line between "from-scratch teaches you something" and "from-scratch teaches you nothing the framework wouldn't have taught more efficiently."

## Models Implemented (#01-#08)

| # | Model | Folder |
|---|---|---|
| 01 | Linear Regression | [No-Framework/01-linear-regression](../../No-Framework/01-linear-regression/) |
| 02 | Logistic Regression | [No-Framework/02-logistic-regression](../../No-Framework/02-logistic-regression/) |
| 03 | KNN | [No-Framework/03-knn](../../No-Framework/03-knn/) |
| 04 | K-Means | [No-Framework/04-k-means](../../No-Framework/04-k-means/) |
| 05 | Naive Bayes | [No-Framework/05-naive-bayes](../../No-Framework/05-naive-bayes/) |
| 06 | Decision Trees / RF | [No-Framework/06-decision-trees-random-forests](../../No-Framework/06-decision-trees-random-forests/) |
| 07 | SVM | [No-Framework/07-svm](../../No-Framework/07-svm/) |
| 08 | PCA | [No-Framework/08-pca](../../No-Framework/08-pca/) |

## Per-Model Findings

### #01 Linear Regression

- Manual gradient descent reveals the math behind the magic — but at a cost (0.38s vs sklearn's 0.03s, 13x slower)
- Lowest memory of all 4 frameworks (2 MB vs sklearn's 15 MB, PyTorch's 54 MB)
- All frameworks produced identical accuracy (R² ≈ 0.50, RMSE ≈ $10,100) — proving algorithm choice matters more than implementation language

### #02 Logistic Regression

- Manual sigmoid + BCE loss + gradient descent in raw NumPy. 18.3s training time
- 57x slower than sklearn's L-BFGS solver (0.32s) and 7.8x slower than PyTorch's autograd + SGD (2.36s) — the cost of writing the math by hand
- All 4 frameworks achieved similar recall (~83%) on fraud detection, confirming algorithm-equivalence across implementations

### #03 KNN

- Manual Manhattan distance + weighted voting. 93.79% accuracy
- ~1,300x slower than sklearn (1.5/sec vs sklearn's KD-tree at 2,000/sec) — sklearn's tree-based lookup dominates brute-force at scale
- All 4 frameworks achieved 93.77% accuracy — proving KNN results are implementation-agnostic
- Rare classes (Cottonwood/Willow at 0.47% of data) consistently had lowest F1 (~0.81) regardless of framework

### #04 K-Means

- From-scratch Lloyd's algorithm with K-Means++ initialization. Matches sklearn metrics within rounding
- 17x slower than sklearn (1.02s vs 0.06s) — but still 2x faster than TensorFlow's eager-CPU dispatch overhead
- All 4 frameworks identical clustering quality: inertia ~9,976, silhouette ~0.3064, ARI ~0.6684

### #05 Naive Bayes

- Pure NumPy GaussianNB + MultinomialNB with the **log-sum-exp trick** for numerical stability — the framework-specific showcase
- 0.13s training (faster than sklearn's 0.21s on this dataset), 18x less memory than sklearn
- Model size halves per abstraction layer: sklearn 3.05 MB (diagnostic arrays) → No-Framework 1.53 MB (float64) → PyTorch/TF 0.76 MB (float32)
- All 4 frameworks produced identical metrics: accuracy 0.6683, macro F1 0.6394

### #06 Decision Trees / Random Forests

- From-scratch DT+RF (F1 0.41, AUC 0.78). Gini vs Entropy comparison + manual OOB computation showcases
- **Pure Python is 83x slower than Cython**: 100-tree RF took 29 minutes from scratch vs sklearn's 21 seconds. The sorted-scan split search is the bottleneck — each node sorts feature values and evaluates all thresholds in pure Python loops
- OOB matches test accuracy within 0.5% — manual OOB computation confirms 1-1/e theory (36.8 average OOB trees per sample) — free validation without holdout sets
- Gini and Entropy are interchangeable: 99.5% prediction agreement, same root split, same feature rankings — theoretical differences negligible in practice
- **Inference faster than PyTorch's hybrid CPU/GPU**: 169.46 µs/sample (NF) vs 279.79 µs/sample (PT) — flattening 100 tree dicts to GPU tensors per call adds conversion overhead that outweighs GPU prediction speed for tree structures
- **Model size 47% larger than tensor-based**: 55.23 MB (NF) vs 29.47 MB (PyTorch) — fewer intermediate Python objects when heavy computation happens in-place
- **6.8x faster than TF eager CPU** for the same algorithm (NF 29 min vs TF 199 min) — every TF tensor op crosses the Python→C++ bridge, and tree recursion triggers millions of these crossings

### #07 SVM

- Poly kernel SVM via dual gradient descent (C=10, F1 0.90, AUC 0.91). From-scratch projected gradient ascent with adaptive LR via quadratic line search
- **From-scratch frameworks match exactly** — accuracy 0.8611, F1 0.8990, AUC 0.9105 across NF, PyTorch, and TensorFlow. Scikit-Learn's optimized SMO produces slightly different (0.8606 acc, 0.8942 F1, 0.9164 AUC) but converges to the same dual objective
- Dual gradient descent converges to obj=231.83 consistently across NF, PT, and TF — algorithm is implementation-agnostic
- **17.7x slower than PyTorch GPU** (160s vs 9.03s) — the O(n²) kernel matrix-vector product each iteration is embarrassingly parallel, and NumPy can't compete with cuBLAS
- **TF eager CPU beats raw NumPy** (85.77s vs 160s, 1.9x faster) for the same algorithm — TF's C++ matmul kernels are more optimized than NumPy's BLAS bindings
- Inference speed hierarchy: PyTorch GPU (0.59 µs) >> TF CPU (15.55 µs) >> SK (36.63 µs) >> NF (153.57 µs) — GPU prediction with batched matmul is 260x faster than NumPy loops
- 75.5% support vectors (11,426 SVs vs SK's 5,343 at 35.3%) — dual gradient descent hadn't fully converged at 3,000 iterations, but accuracy matched. Educational implementation, not production optimization

### #08 PCA

- From-scratch eigendecomposition matches SK exactly: 0.9085 explained variance, 0.8599 KNN downstream accuracy at 150 components. 0.23s fit, 0.89 µs/sample
- **Eigendecomposition and SVD produce identical PCA** — verified numerically: max eigenvalue diff 1.53e-05, max projection diff 2.43e-03 across 60K samples. Different LAPACK routines, same math. **The framework-specific NF showcase**
- Population vs sample covariance is a non-issue: NF/PT/TF use 1/n, SK uses 1/(n-1) — shifts 90%/95% thresholds but component ordering and downstream accuracy are identical
- All 4 frameworks produced identical results — eigendecomposition algorithm is truly implementation-agnostic. The math is the math; the framework is just plumbing

## Progress Log (No-Framework entries, chronological)

| Date | Model | Notes |
|---|---|---|
| 2026-03-14 | PCA | From-scratch eigendecomposition matches SK exactly (0.9085 variance, 0.8599 KNN). 0.23s fit, 0.89 µs/sample. **Last NF model — retired after this** |
| 2026-03-10 | SVM | Poly kernel SVM via dual gradient descent (C=10, F1 0.90, AUC 0.91). From-scratch projected gradient ascent |
| 2026-03-03 | Decision Trees & RF | From-scratch DT+RF (F1 0.41, AUC 0.78). Gini vs Entropy + manual OOB showcases |
| 2026-02-26 | Naive Bayes | Pure NumPy GaussianNB + MultinomialNB. Faster (0.13s vs 0.21s vs sklearn), 18x less memory |
| 2026-02-21 | K-Means | From-scratch Lloyd's algorithm, K-Means++ init. Matches sklearn metrics, 17x slower |
| 2026-02-14 | KNN | Manual Manhattan distance + weighted voting. 93.79% accuracy, ~1,300x slower |
| 2026-02-09 | Logistic Regression | Manual sigmoid, BCE loss, gradient descent. 18.3s training |
| 2026-02-04 | Linear Regression | Built from scratch with NumPy: gradient descent, MSE cost, z-score scaling |

## Cross-cutting takeaways

1. **Framework-equivalence is the headline result**: across #01-#08, NF achieved metrics within rounding of every other implementation. The eigendecomposition (PCA), gradient descent (LR/LogReg/SVM), and Lloyd's algorithm (K-Means) are all implementation-agnostic. **The math is the math; the framework is just plumbing.**

2. **The retirement boundary is exactly where it should be**: #01-#08 are all "matrix algebra you can write by hand in 100 lines." Anything past that (DNNs onward) requires autograd, batching primitives, GPU kernels, layer abstractions — at which point "no framework" stops meaning "I understand the math" and starts meaning "I'm reimplementing PyTorch badly."

3. **NF was the speed/memory floor for measurement**: every framework's speedup or overhead in #01-#08 was reported relative to the NF baseline (e.g., "PyTorch GPU SVM is 17.7x faster than NF"). That comparison only works because NF gives an honest, optimization-free reference point.
