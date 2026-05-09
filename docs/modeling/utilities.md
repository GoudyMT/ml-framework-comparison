# Shared Utilities Architecture

> The `utils/` package began during Logistic Regression (#02) to avoid duplicating code across frameworks. The package evolves organically: during the planning phase when new model types are started, recurring patterns are identified and moved here.

## Benefits

- **Zero inconsistency**: All frameworks use identical metric calculations
- **Faster development**: 3 imports vs 50 lines of boilerplate per notebook
- **Easier maintenance**: Fix a bug once, applies everywhere
- **Framework-agnostic**: Works with NumPy arrays from any framework

## Usage Pattern

```python
from utils.metrics import evaluate_classifier, print_metrics
from utils.performance import track_performance, track_inference, get_model_size
from utils.visualization import plot_calibration_curve

# Training with performance tracking
with track_performance() as perf:
    model.fit(X_train, y_train)

# Streamlined evaluation (auto-detects binary/multiclass, includes probabilistic metrics)
train_metrics = evaluate_classifier(y_train, train_pred, train_proba)
test_metrics = evaluate_classifier(y_test, test_pred, test_proba)
print_metrics(train_metrics, test_metrics, title='MultinomialNB — 20 Newsgroups')

# Inference speed + model size (new from Naive Bayes onward)
inference = track_inference(model.predict_proba, X_test, n_runs=100)
model_size = get_model_size(model, framework='sklearn')
```

## Current Utilities

(Newest additions at top; table grows as new model types introduce shared patterns)

| Module | Functions | Added In | Purpose |
|--------|-----------|----------|---------|
| `rl_utils.py` | `seed_everything`, `epsilon_greedy`, `LinearEpsilonSchedule`, `ReplayBuffer`, `PrioritizedReplayBuffer` (with `_SumTree`), `soft_target_update`, `evaluate_policy`, `plot_learning_curve` | Q-Learning | Eight RL primitives shared between PT + TF pipelines: cyclic replay buffer (V2-V4), prioritized replay via O(log N) sum-tree (V5; proportional-sampling unit-tested at <5% relative error), epsilon-greedy action selection, linear epsilon decay schedule, Polyak target-net update (PT lazy torch import), deterministic policy rollout, learning curve plotter with rolling mean overlay. `seed_everything` plumbs Python random + numpy + torch (lazy) + gymnasium env action_space sampler in one call |
| `vae_utils.py` | `reparameterize`, `kl_divergence_gaussian`, `vq_straight_through`, `vq_commit_loss`, `latent_traversal`, `interpolate_latent` | VAE | Six probabilistic-latent primitives: reparameterization trick (z = mu + exp(0.5*logvar)*eps), analytical KL divergence between N(mu, sigma^2) and N(0, I), VQ-VAE straight-through estimator + commitment loss, β-VAE single-dim latent traversals, latent-space interpolation. Note: V4 + V5 inline VQ math; the util's `vq_straight_through` has a known gradient-flow bug deferred to future cleanup |
| `gnn_utils.py` | `normalize_adj_symmetric`, `edge_softmax`, `evaluate_ogb` | GNN | Symmetric-normalized adjacency (`D^(-1/2)(A+I)D^(-1/2)` as sparse COO) for GCN, scatter-softmax wrapper for from-scratch GAT attention, OGB Evaluator shape-shim from logits/labels to leaderboard metric |
| `vit_utils.py` | `apply_mixup_cutmix`, `distillation_loss`, `attention_rollout`, `cls_attention_map` | Vision Transformers | Batch-level MixUp/CutMix with soft labels, DeiT dual-head distillation loss (hard/soft modes), Abnar & Zuidema attention rollout, CLS-to-patches heatmap upsampling |
| `visualization.py` | `plot_attention_overlay`, `plot_attention_grid_samples` | Vision Transformers | Single-image CLS attention overlay + 2xN grid of originals vs attention for portfolio viz. `plot_bleu_progression` extended with `ylabel` param to serve as generic variant progression chart |
| `transformer_utils.py` | `create_pad_mask`, `create_causal_mask`, `greedy_decode`, `compute_bleu_greedy`, `beam_search_decode` | Transformers | Mask creation (padding + causal), autoregressive greedy decoding, BPE-aware BLEU computation, beam search with length normalization |
| `visualization.py` | `plot_bleu_progression`, `plot_multihead_grid` | Transformers | BLEU comparison across variants with baseline line + multi-head attention grid showing head specialization |
| `attention_utils.py` | `compute_bleu`, `bleu_by_length` | Attention | BLEU-4 corpus scoring with smoothing + per-length-bucket analysis. Reusable for Transformers. |
| `visualization.py` | `plot_attention_heatmap`, `plot_attention_comparison`, `plot_bleu_by_length` | Attention | Attention weight heatmaps (single + side-by-side comparison), BLEU by sentence length bar charts |
| `gan_utils.py` | `compute_fid` | GANs | Frechet Inception Distance via InceptionV3 pool3 features. Gold-standard generative model metric. Reusable for VAE. |
| `visualization.py` | `plot_generated_grid` | GANs | Grid display of generated images with auto [-1,1]→[0,1] rescaling and channel format detection. Reusable for VAE. |
| `rnn_utils.py` | `jitter`, `scaling`, `time_warp`, `augment_minority_classes` | LSTM | Time-series augmentation for imbalanced datasets. Offline augmentation applied once during preprocessing. |
| `rnn_utils.py` | `MacroF1Callback`, `extract_hidden_states` | LSTM | Keras early stopping on macro F1 (reusable for all TF sequence models). Hidden/cell state extraction for LSTM analysis (PT + TF). |
| `visualization.py` | `plot_ecg_augmentation_samples`, `plot_class_distribution_comparison`, `plot_sequence_length_distribution` | LSTM | ECG augmentation viz, before/after class balance chart, variable-length sequence histogram |
| `rnn_utils.py` | `compute_gradient_norms` | RNN | Per-layer gradient norms for vanishing gradient analysis (PyTorch + TensorFlow). Reusable for LSTM, Attention, Transformers |
| `visualization.py` | `plot_gradient_flow`, `plot_ecg_predictions`, `plot_hidden_state_evolution` | RNN | Gradient norm bar charts (side-by-side models), ECG waveforms colored by prediction correctness, hidden state dimensions over timesteps |
| `visualization.py` | `plot_superclass_confusion`, `plot_augmentation_samples` | CNN | Superclass-level confusion matrix with hierarchical evaluation, generic augmentation sample grid |
| `data_loader.py` | `load_processed_data` (updated) | CNN | Auto-detects hierarchical labels (fine/coarse) — backward compatible with all existing models |
| `performance.py` | `get_model_size` (fixed) | CNN | Fixed TF dtype bug — `v.dtype.name` with `np.dtype().itemsize` handles both PT and TF |
| `visualization.py` | `plot_latent_space`, `plot_reconstruction_grid`, `plot_training_history` | Autoencoders | Latent space t-SNE/PCA projection, RGB reconstruction grids, model-agnostic training history title |
| `results.py` | `add_result`, `print_comparison` (fixed) | Autoencoders | Removed hardcoded `/4` framework count — now dynamic for 3-framework and 2-framework models |
| `visualization.py` | `plot_training_history` | DNN | Dual-panel training loss + accuracy curves (handles optional val_loss/val_acc) |
| `visualization.py` | `plot_scree`, `plot_reconstruction_grid`, `plot_pca_components`, `plot_component_accuracy` | PCA | Scree/cumulative variance plots, reconstruction comparison grids, PC visualization, accuracy vs components |
| `svm_utils.py` | `to_svm_labels`, `to_std_labels`, `platt_calibrate`, `platt_predict_proba` | SVM | Label conversion {0,1}↔{-1,+1} + Platt probability calibration for from-scratch SVMs |
| `visualization.py` | `plot_kernel_comparison`, `plot_svm_convergence` | SVM | Kernel comparison showcase (3-panel grouped bars) + dual objective convergence |
| `tree_utils.py` | `compute_feature_importance`, `flatten_tree`, `predict_batch` | Decision Trees | Shared DT/RF operations — Gini importance, flat array conversion, batch prediction |
| `results.py` | `build_results_dict`, `_format_value` | Decision Trees | Standardized results construction + human-readable unit formatting (seconds, MB, µs) |
| `visualization.py` | `plot_tree_depth_analysis`, `plot_forest_convergence` | Decision Trees | DT overfitting analysis (train vs test across max_depth) + RF convergence (accuracy vs n_estimators) |
| `metrics.py` | `log_loss`, `brier_score`, `expected_calibration_error` | Naive Bayes | Probabilistic evaluation (calibration quality) |
| `metrics.py` | `evaluate_classifier`, `print_metrics` | Naive Bayes | Streamlined evaluation helpers — auto-detect binary/multiclass, formatted tables |
| `performance.py` | `track_inference`, `get_model_size` | Naive Bayes | Inference speed (per-sample μs, throughput) and model size tracking |
| `visualization.py` | `plot_calibration_curve`, `plot_calibration_comparison` | Naive Bayes | Reliability diagrams — single model + multi-model overlay for before/after comparison |
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
