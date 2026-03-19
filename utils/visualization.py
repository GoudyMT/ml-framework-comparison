"""
Shared visualization function for all frameworks.
Each function takes a 'framework' parameter to cusomize titles.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_cost_curve(costs, framework, save_path=None):
    """
    Plot training cost/loss over iterations.

    Args:
        costs: List of array of cost values per iteration
        framework: Name for the title (e.g. "PyTorch", "No-Framework")
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title(f'{framework} - Training Cost Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, framework, save_path=None):
    """
    Plot confusion matrix as a heatmap.

    Args:
        y_true: Actual labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        framework: Name for the title
        save_path: Optinal path to save the figure
    """
    # Import here to avoid circular dependency
    from .metrics import confusion_matrix_values

    tp, fp, tn, fn = confusion_matrix_values(y_true, y_pred)

    # Arrange as 2x2 matrix: rows = actual, cols = predicted
    # [[TN, FP], [FN, TP]]
    cm = np.array([[tn, fp],
                   [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()

    # Label the axes
    classes = ['Legitimate (0)', 'Fraud (1)']
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    # Add count annotations to each cell
    thresh = cm.max() / 2   # Threshold for text color
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], ',d'),
                     ha='center', va='center', fontsize=14,
                     color='white' if cm[i, j] > thresh else 'black')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'{framework} - Confusion Matrix', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_proba, framework, save_path=None):
    """
    Plot ROC curve with AUC score.

    Args:
        y_true: Actual labels (0 or 1)
        y_proba: Predicted probabilities (0.0 to 1.0)
        framework: Name for the title
        save_path: Optinal path to save the figure
    """
    from .metrics import roc_curve, auc_score

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0,1], [0,1], 'r--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{framework} - ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_feature_importance(weights, feature_names, framework, save_path=None, top_n=15, mode='coefficient'):
    """
    Plot feature importance as horizontal bar chart.

    Two modes for different model types:
    - 'coefficient': Green/red for positive/negative coefficients (linear models)
    - 'importance': Single color for non-negative importances (tree-based models)

    Args:
        weights: Model coefficients/weights array (or importance scores)
        feature_names: List of feature names
        framework: Name for the title
        save_path: Optional path to save the figure
        top_n: Number of top features to display
        mode: 'coefficient' (default) for linear models, 'importance' for tree models
    """
    if mode == 'importance':
        # Tree-based models: importances are always non-negative
        # Sort by value descending (no absolute value needed)
        indices = np.argsort(weights)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), weights[indices], color='steelblue', alpha=0.7)
        plt.xlabel('Importance Score', fontsize=12)
    else:
        # Linear models: coefficients can be positive or negative
        # Sort by absolute value to find most impactful features
        indices = np.argsort(np.abs(weights))[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        colors = ['green' if weights[i] > 0 else 'red' for i in indices]
        plt.barh(range(len(indices)), weights[indices], color=colors, alpha=0.7)
        plt.xlabel('Coefficient Value', fontsize=12)

    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'{framework} - Feature Importance (Top {top_n})', fontsize=14)
    plt.gca().invert_yaxis()    # Highest importance at top
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# MULTI-CLASS & KNN VISUALIZATIONS (Added during KNN Prep)

def plot_confusion_matrix_multiclass(y_true, y_pred, class_names, framework, save_path=None):
    """
    Plot NxN confusion matrix heatmap for multi-class classification.

    Unlike binary confusion matrix (2x2), this handles any number of classes.
    Rows represent actual classes, columns represent predicted classes.
    Diagonal shows correct predictions; off-diagonal shows misclassifcation.

    Args:
        y_true: Actual class labels (integers 0 to n_classes-1)
        y_pred: Predicted class labels (integers 0 to n_classes-1)
        class_names: List of class name strings for axis labels
        framework: Name for the title (e.g. "Scikit_learn", "PyTorch")
        save_path: Optinal save path to save the figure
    """
    from .metrics import confusion_matrix_multiclass

    n_classes = len(class_names)
    cm = confusion_matrix_multiclass(y_true, y_pred, n_classes)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()

    # Label axes with class names
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)

    # Add count annotations to each cell
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, format(cm[i, j], ',d'),
                     ha='center', va='center', fontsize=8,
                     color='white' if cm[i, j] > thresh else 'black')
            
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'{framework} - Confusion Matrix ({n_classes} Classes)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_validation_curve(k_values, train_scores, test_scores, framework, save_path=None):
    """
    Plot train vs test scores across K values for KNN.

    Shows bias-variance tradeoff:
    - Low K (K=1): High variance (overfitting) - train score high, test score low
    - High K: High bias (underfitting) - both scores lower but closer together
    - Optimal K: Best test score, good generalization

    Args:
        k_values: List of K values tested
        train_scores: List of training scores for each K value
        test_scores: List of test scores for each K value
        framework: Name for the title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))

    plt.plot(k_values, train_scores, 'b-o', linewidth=2, markersize=8, label='Training Score')
    plt.plot(k_values, test_scores, 'r-o', linewidth=2, markersize=8, label='Test Score')

    # Highlight the best K (highest test score)
    best_idx = np.argmax(test_scores)
    best_k = k_values[best_idx]
    best_score = test_scores[best_idx]
    plt.axvline(x=best_k, color='green', linestyle='--', alpha=0.7,
                label=f'Best K={best_k} (Test={best_score:.4f})')
    
    plt.xlabel('K (Number of Neighbors)', fontsize=12)
    plt.ylabel('Score (Macro F1)', fontsize=12)
    plt.title(f'{framework} - Validation Curve (Bias-Variance Tradeoff)', fontsize=14)
    plt.xticks(k_values)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_per_class_f1(y_true, y_pred, class_names, framework, save_path=None):
    """
    Bar chart showing F1 score for each class.

    Helps identify which classes the model struggle with.
    Classes with low F1 scores are candidates for further investigation
    (e.g., more data, feature engineering, or class-specific strategies).

    Args:
        y_true: Actual class labels
        y_pred: Predicted class labels
        class_names: List of class name strings
        framework: Name for the title
        save_path: Optinal path to save the figure
    """
    from .metrics import macro_f1_score

    n_classes = len(class_names)

    # Get per-class F1 scores from macro_f1_score (reduces redundancy)
    _, f1_scores = macro_f1_score(y_true, y_pred, return_per_class=True) # type: ignore

    # Create bar chart
    plt.figure(figsize=(12,6))
    bars = plt.bar(range(n_classes), f1_scores, color='steelblue', alpha=0.8)

    # Add value labels on top of each bar
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(range(n_classes), class_names, rotation=45, ha='right', fontsize=10)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title(f'{framework} - Per-Class F1 Scores', fontsize=14)
    plt.ylim(0, 1.1)    # F1 ranges from 0 to 1, extra room for labels
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# CLUSTERING VISUALIZATIONS (Added during K-Means)

def plot_elbow_curve(k_values, inertias, framework, best_k=None, save_path=None):
    """
    Inertia vs K plot for finding optimal cluster count.

    The "elbow" point where inertia stops dropping sharply suggests
    the best K — adding more clusters beyond that gives diminishing returns.

    Args:
        k_values: List of K values tested.
        inertias: Inertia (WCSS) for each K.
        framework: Name for the title.
        best_k: Optional K to highlight with a vertical line.
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'b-o', linewidth=2, markersize=8)

    if best_k is not None:
        plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7,
                    label=f'Best K={best_k}')
        plt.legend(fontsize=11)

    plt.xlabel('K (Number of Clusters)', fontsize=12)
    plt.ylabel('Inertia (WCSS)', fontsize=12)
    plt.title(f'{framework} — Elbow Curve', fontsize=14)
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_silhouette_comparison(k_values, silhouette_scores, framework, best_k=None, save_path=None):
    """
    Silhouette Score vs K to confirm optimal cluster count.

    Complements the elbow curve — the K with the highest silhouette
    score has the best-defined cluster boundaries.

    Args:
        k_values: List of K values tested.
        silhouette_scores: Silhouette score for each K.
        framework: Name for the title.
        best_k: Optional K to highlight with a vertical line.
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'g-o', linewidth=2, markersize=8)

    if best_k is not None:
        plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7,
                    label=f'Best K={best_k}')
        plt.legend(fontsize=11)

    plt.xlabel('K (Number of Clusters)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title(f'{framework} — Silhouette Comparison', fontsize=14)
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_silhouette_analysis(X, labels, framework, save_path=None):
    """
    Per-sample silhouette plot grouped by cluster.

    Each cluster is shown as a horizontal blade of sorted silhouette values.
    Wide, uniform blades = well-defined clusters. Thin or negative blades =
    fuzzy/overlapping clusters. More insightful than 2D PCA scatter
    for high-dimensional data (16 features).

    Args:
        X: Feature matrix (n_samples, n_features).
        labels: Cluster assignments (n_samples,).
        framework: Name for the title.
        save_path: Optional path to save the figure.
    """
    from .metrics import silhouette_samples, silhouette_score

    sample_scores = silhouette_samples(X, labels)
    mean_score = np.mean(sample_scores)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    plt.figure(figsize=(10, 8))
    y_lower = 10  # Starting y position for first cluster blade

    for i, label in enumerate(unique_labels):
        # Get silhouette values for this cluster, sorted ascending
        cluster_scores = sample_scores[labels == label]
        cluster_scores.sort()

        cluster_size = len(cluster_scores)
        y_upper = y_lower + cluster_size

        # Fill between creates the blade shape
        color = plt.cm.tab10(i / n_clusters)    # type: ignore
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_scores,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label each cluster at its midpoint
        plt.text(-0.05, y_lower + 0.5 * cluster_size, str(label), fontsize=10)

        y_lower = y_upper + 10  # Gap between clusters

    # Vertical line for mean silhouette score
    plt.axvline(x=mean_score, color='red', linestyle='--',  # type: ignore
                label=f'Mean: {mean_score:.3f}')

    plt.xlabel('Silhouette Score', fontsize=12)
    plt.ylabel('Cluster', fontsize=12)
    plt.title(f'{framework} — Silhouette Analysis ({n_clusters} Clusters)', fontsize=14)
    plt.legend(fontsize=11)
    plt.yticks([])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_convergence_curve(inertia_history, framework, save_path=None):
    """
    Inertia per iteration showing algorithm convergence.

    A well-behaved K-Means run shows inertia dropping steeply at first,
    then flattening as centroids stabilize.

    Args:
        inertia_history: List of inertia values, one per iteration.
        framework: Name for the title.
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(inertia_history) + 1), inertia_history,
             'b-o', linewidth=2, markersize=6)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Inertia (WCSS)', fontsize=12)
    plt.title(f'{framework} — Convergence Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# PROBABILISTIC VISUALIZATIONS (Added during Naive Bayes)

def plot_calibration_curve(y_true, y_proba, framework, n_bins=10, save_path=None):
    """
    Reliability diagram showing predicted vs actual probability.

    A perfectly calibrated model follows the diagonal (y=x).
    Points above the diagonal: model is under-confident.
    Points below the diagonal: model is over-confident.

    Two-panel layout:
    - Top: calibration curve with perfect diagonal reference
    - Bottom: histogram of prediction counts per bin

    For multiclass: uses max predicted probability and checks if
    the predicted class matches the true class.

    Args:
        y_true: True labels (n_samples,).
        y_proba: Predicted probabilities. Binary: (n_samples,).
            Multiclass: (n_samples, n_classes).
        framework: Name for the title.
        n_bins: Number of bins for grouping predictions.
        save_path: Optional path to save the figure.
    """
    # Extract confidences and correctness
    if y_proba.ndim == 1:
        confidences = y_proba
        correct = (y_true == (y_proba >= 0.5).astype(int)).astype(float)
    else:
        confidences = np.max(y_proba, axis=1)
        correct = (np.argmax(y_proba, axis=1) == y_true).astype(float)

    # Bin predictions by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        count = np.sum(mask)
        bin_counts.append(count)

        if count == 0:
            continue

        bin_centers.append(np.mean(confidences[mask]))
        bin_accuracies.append(np.mean(correct[mask]))

    # Two-panel layout: calibration curve on top, histogram on bottom
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Top panel: calibration curve
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect calibration')
    ax1.plot(bin_centers, bin_accuracies, 'b-o', linewidth=2, markersize=8,
             label=f'{framework}')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title(f'{framework} — Calibration Curve (Reliability Diagram)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Bottom panel: histogram of prediction counts
    ax2.bar(range(n_bins), bin_counts, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Confidence Bin', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels([f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}'
                         for i in range(n_bins)], rotation=45, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_calibration_comparison(y_true, probas_dict, n_bins=10, save_path=None):
    """
    Overlay multiple models' calibration curves for before/after comparison.

    Same two-panel layout as plot_calibration_curve but with multiple
    models on the same axes. Useful for comparing uncalibrated vs
    calibrated probabilities (e.g., NB vs Platt-scaled NB).

    Args:
        y_true: True labels (n_samples,).
        probas_dict: Dict of {model_name: y_proba}. Each y_proba is
            either (n_samples,) for binary or (n_samples, n_classes)
            for multiclass.
        n_bins: Number of bins for grouping predictions.
        save_path: Optional path to save the figure.
    """
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # blue, green, orange, red
    bin_edges = np.linspace(0, 1, n_bins + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Perfect calibration reference line
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect calibration')

    bar_width = 0.8 / len(probas_dict)

    for idx, (name, y_proba) in enumerate(probas_dict.items()):
        # Extract confidences and correctness (same logic as plot_calibration_curve)
        if y_proba.ndim == 1:
            confidences = y_proba
            correct = (y_true == (y_proba >= 0.5).astype(int)).astype(float)
        else:
            confidences = np.max(y_proba, axis=1)
            correct = (np.argmax(y_proba, axis=1) == y_true).astype(float)

        # Bin predictions by confidence
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
            count = np.sum(mask)
            bin_counts.append(count)

            if count == 0:
                continue

            bin_centers.append(np.mean(confidences[mask]))
            bin_accuracies.append(np.mean(correct[mask]))

        color = colors[idx % len(colors)]

        # Top panel: calibration curve
        ax1.plot(bin_centers, bin_accuracies, '-o', color=color,
                 linewidth=2, markersize=8, label=name)

        # Bottom panel: grouped histogram bars
        offsets = np.arange(n_bins) + idx * bar_width
        ax2.bar(offsets, bin_counts, width=bar_width, color=color,
                alpha=0.7, label=name)

    # Format top panel
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Comparison (Reliability Diagram)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Format bottom panel
    ax2.set_xlabel('Confidence Bin', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    center_offsets = np.arange(n_bins) + bar_width * (len(probas_dict) - 1) / 2
    ax2.set_xticks(center_offsets)
    ax2.set_xticklabels([f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}'
                         for i in range(n_bins)], rotation=45, fontsize=8)
    ax2.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# TREE-BASED MODEL VISUALIZATIONS (Added during DT/RF prep)

def plot_tree_depth_analysis(depth_values, train_scores, test_scores, framework, save_path=None):
    """
    Plot train vs test accuracy across max_depth values.

    Shows the overfitting point where train and test scores diverge.
    Used for Decision Tree depth tuning — helps find optimal max_depth
    that balances bias (underfitting) and variance (overfitting).

    Args:
        depth_values: List of max_depth values tested (can include None for full tree)
        train_scores: List of training accuracies (or F1) for each depth
        test_scores: List of test accuracies (or F1) for each depth
        framework: Name for the title (e.g., 'Scikit-Learn', 'No-Framework')
        save_path: Optional path to save the figure
    """
    # Convert None to string label for x-axis (None = no depth limit)
    x_labels = [str(d) if d is not None else 'None' for d in depth_values]
    x_positions = range(len(depth_values))

    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, train_scores, 'b-o', linewidth=2, markersize=8, label='Train')
    plt.plot(x_positions, test_scores, 'r-o', linewidth=2, markersize=8, label='Test')

    # Highlight best test score
    best_idx = np.argmax(test_scores)
    plt.axvline(x=best_idx, color='green', linestyle='--', alpha=0.5, # type: ignore
                label=f'Best depth: {x_labels[best_idx]}')
    plt.scatter([best_idx], [test_scores[best_idx]], color='green',
                s=200, zorder=5, edgecolors='black', linewidths=2)

    plt.xticks(x_positions, x_labels, fontsize=11)
    plt.xlabel('max_depth', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'{framework} - Decision Tree Depth Analysis', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_forest_convergence(n_estimators_values, train_scores, test_scores, framework,
                            oob_scores=None, save_path=None):
    """
    Plot train/test accuracy vs number of trees in Random Forest.

    Shows how ensemble performance improves with more trees and where
    it plateaus. Optionally overlays OOB (out-of-bag) score to demonstrate
    the "free" validation that bagging provides.

    Args:
        n_estimators_values: List of n_estimators values tested
        train_scores: List of training accuracies for each n_estimators
        test_scores: List of test accuracies for each n_estimators
        framework: Name for the title
        oob_scores: Optional list of OOB scores (only for sklearn/from-scratch)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_values, train_scores, 'b-o', linewidth=2,
             markersize=6, label='Train')
    plt.plot(n_estimators_values, test_scores, 'r-o', linewidth=2,
             markersize=6, label='Test')

    if oob_scores is not None:
        plt.plot(n_estimators_values, oob_scores, 'g--s', linewidth=2,
                 markersize=6, label='OOB Score', alpha=0.8)

    plt.xlabel('Number of Trees (n_estimators)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'{framework} - Random Forest Convergence', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # SUPPORT VECTOR MACHINES MODEL VISUALIZATIONS (Added during SVM prep)

def plot_kernel_comparison(kernel_results, framework, save_path=None):
    """
    Plot a 3-panel comparison of SVM kernels side by side.

    Shows how different kernel functions (linear, RBF, polynomial) perform
    on the same dataset. Three panels compare: classification metrics
    (accuracy, F1, AUC), training time, and support vector count.
    This is the SVM showcase visualization used by all 4 frameworks.

    Args:
        kernel_results: Dict of {kernel_name: {accuracy, f1, auc,
            training_time, n_support_vectors}}. Example:
            {'Linear': {'accuracy': 0.85, 'f1': 0.82, 'auc': 0.90,
                        'training_time': 1.2, 'n_support_vectors': 500},
             'RBF': {...}, 'Polynomial': {...}}
        framework: Name for the title (e.g., 'Scikit-Learn')
        save_path: Optional path to save the figure
    """
    kernels = list(kernel_results.keys())
    n_kernels = len(kernels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Classification Metrics (grouped bars)
    metrics = ['accuracy', 'f1', 'auc']
    metric_labels = ['Accuracy', 'F1 Score', 'AUC']
    x = np.arange(len(metrics))
    width = 0.8 / n_kernels
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    for i, kernel in enumerate(kernels):
        values = [kernel_results[kernel][m] for m in metrics]
        offset = (i - (n_kernels - 1) / 2) * width
        bars = axes[0].bar(x + offset, values, width, label=kernel,
                          color=colors[i % len(colors)], alpha=0.85)
        # Value labels on each bar
        for bar, val in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels, fontsize=11)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Classification Metrics', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Panel 2: Training Time
    times = [kernel_results[k]['training_time'] for k in kernels]
    bars = axes[1].bar(kernels, times, color=colors[:n_kernels], alpha=0.85)
    for bar, val in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                    f'{val:.2f}s', ha='center', va='bottom', fontsize=10)
    axes[1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1].set_title('Training Time', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Panel 3: Support Vector Count
    sv_counts = [kernel_results[k]['n_support_vectors'] for k in kernels]
    bars = axes[2].bar(kernels, sv_counts, color=colors[:n_kernels], alpha=0.85)
    for bar, val in zip(bars, sv_counts):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sv_counts) * 0.02,
                    f'{val:,}', ha='center', va='bottom', fontsize=10)
    axes[2].set_ylabel('Count', fontsize=12)
    axes[2].set_title('Support Vectors', fontsize=13)
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'{framework} - SVM Kernel Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_svm_convergence(objective_values, framework, save_path=None):
    """
    Plot dual objective value over training iterations for SVM.

    Shows convergence of the projected gradient ascent optimizer
    on the SVM dual problem. A rising, plateauing curve indicates
    the optimizer found the maximum of the dual objective (= optimal
    separating hyperplane). Used by PyTorch and TensorFlow implementations
    which solve the dual via gradient-based optimization.

    Args:
        objective_values: List or array of dual objective values per iteration
        framework: Name for the title (e.g., 'PyTorch')
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(objective_values) + 1)
    plt.plot(iterations, objective_values, 'b-', linewidth=2, alpha=0.8)

    # Mark final value
    final_val = objective_values[-1]
    plt.axhline(y=final_val, color='red', linestyle='--', alpha=0.5,
                label=f'Final: {final_val:.4f}')
    plt.scatter([len(objective_values)], [final_val], color='red',
                s=100, zorder=5, edgecolors='black', linewidths=1.5)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Dual Objective Value', fontsize=12)
    plt.title(f'{framework} - SVM Dual Objective Convergence', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
# PCA VISUALIZATIONS (Added during PCA prep)

def plot_scree(explained_var_ratio, framework, save_path=None):
    """
    Dual-panel scree plot: individual + cumulative explained variance.

    Left panel shows per-component variance (eigenvalue decay).
    Right panel shows cumulative variance with 90%/95% threshold lines
    and markers for how many components reach each threshold.

    Args:
        explained_var_ratio: Array of explained variance ratios per component.
        framework: Name for the title (e.g., 'Scikit-Learn').
        save_path: Optional path to save the figure.
    """
    cumulative = np.cumsum(explained_var_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: individual explained variance (scree)
    n_show = min(100, len(explained_var_ratio))
    axes[0].plot(range(1, n_show + 1), explained_var_ratio[:n_show],
                 'b-', linewidth=1.5)
    axes[0].set_xlabel('Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title(f'{framework} — Scree Plot', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # Right: cumulative explained variance
    n_show_cum = min(200, len(cumulative))
    axes[1].plot(range(1, n_show_cum + 1), cumulative[:n_show_cum],
                 'r-', linewidth=2)

    # Threshold lines at 90% and 95%
    for thresh, color, style in [(0.90, 'gray', '--'), (0.95, 'black', '--')]:
        axes[1].axhline(thresh, color=color, linestyle=style, alpha=0.7,
                        label=f'{thresh:.0%} variance')
        n_comp = np.searchsorted(cumulative, thresh) + 1
        axes[1].axvline(n_comp, color=color, linestyle=':', alpha=0.5)
        axes[1].annotate(f'n={n_comp}', xy=(n_comp, thresh),
                         xytext=(n_comp + 5, thresh - 0.05),
                         fontsize=10, color=color)

    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title(f'{framework} — Cumulative Variance', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_reconstruction_grid(originals, reconstructions_dict, image_shape=(28, 28),
                              n_samples=5, save_path=None, title=None,
                              row_label_prefix='n', framework=None):
    """
    Grid comparing original images to reconstructions at different compression levels.

    First row shows original images. Subsequent rows show reconstructions
    at each level, with MSE displayed below each image.
    Supports both grayscale (H, W) and RGB (H, W, C) images.

    Args:
        originals: Array of original images, shape (n_samples, n_features).
        reconstructions_dict: Dict of {level: reconstructed_array}.
            Each array shape (n_samples, n_features).
        image_shape: Tuple for reshaping — (H, W) for grayscale or (H, W, C) for RGB.
        n_samples: Number of sample images to show (columns).
        save_path: Optional path to save the figure.
        title: Custom title. Defaults to 'PCA Reconstruction Quality' (backward compatible).
        row_label_prefix: Label prefix for rows. Defaults to 'n' (PCA). Use 'dim' for AE.
        framework: Optional framework name prepended to title.
    """
    is_rgb = len(image_shape) == 3
    n_rows = 1 + len(reconstructions_dict)
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(n_samples * 2.5, n_rows * 2.5))

    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    # Row 0: originals
    for j in range(n_samples):
        img = originals[j].reshape(image_shape)
        if is_rgb:
            axes[0, j].imshow(np.clip(img, 0, 1))
        else:
            axes[0, j].imshow(img, cmap='gray')
        axes[0, j].axis('off')
        if j == 0:
            axes[0, j].set_ylabel('Original', fontsize=11, rotation=0,
                                   labelpad=60, va='center')

    # Remaining rows: reconstructions at each level
    for i, (level, recon) in enumerate(reconstructions_dict.items(), start=1):
        for j in range(n_samples):
            img = recon[j].reshape(image_shape)
            mse = np.mean((originals[j] - recon[j]) ** 2)
            if is_rgb:
                axes[i, j].imshow(np.clip(img, 0, 1))
            else:
                axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'MSE={mse:.4f}', fontsize=8)
            if j == 0:
                axes[i, j].set_ylabel(f'{row_label_prefix}={level}', fontsize=11,
                                       rotation=0, labelpad=60, va='center')

    # Title
    if title is None:
        title = 'PCA Reconstruction Quality'
    if framework:
        title = f'{framework} — {title}'
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_pca_components(components, image_shape=(28, 28), n_components=10,
                         framework='', save_path=None):
    """
    Visualize top principal components as images.

    Each component is a direction in pixel space — reshaping to the
    original image dimensions reveals what spatial patterns each
    component captures (edges, textures, shapes).

    Args:
        components: Array of shape (n_total_components, n_features).
            Rows are principal component vectors.
        image_shape: Tuple (height, width) for reshaping.
        n_components: Number of top components to display.
        framework: Name for the title.
        save_path: Optional path to save the figure.
    """
    n_show = min(n_components, len(components))
    n_cols = 5
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = np.atleast_2d(axes)

    for i in range(n_rows * n_cols):
        row, col = divmod(i, n_cols)
        if i < n_show:
            img = components[i].reshape(image_shape)
            axes[row, col].imshow(img, cmap='RdBu_r')
            axes[row, col].set_title(f'PC {i + 1}', fontsize=10)
        axes[row, col].axis('off')

    fig.suptitle(f'{framework} — Top {n_show} Principal Components', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_component_accuracy(n_components_list, accuracies, framework, save_path=None):
    """
    Downstream classifier accuracy vs number of PCA components.

    Shows the extrinsic evaluation of PCA — how well a simple classifier
    performs on PCA-reduced data at different compression levels.
    Helps identify the sweet spot between dimensionality reduction
    and classification performance.

    Args:
        n_components_list: List of component counts tested.
        accuracies: List of classifier accuracies at each component count.
        framework: Name for the title.
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_list, accuracies, 'b-o', linewidth=2, markersize=8)

    # Highlight best accuracy
    best_idx = np.argmax(accuracies)
    best_n = n_components_list[best_idx]
    best_acc = accuracies[best_idx]
    plt.scatter([best_n], [best_acc], color='red', s=200, zorder=5,
                edgecolors='black', linewidths=2,
                label=f'Best: n={best_n} ({best_acc:.4f})')

    # Add value labels
    for n, acc in zip(n_components_list, accuracies):
        plt.annotate(f'{acc:.3f}', (n, acc), textcoords='offset points',
                     xytext=(0, 12), ha='center', fontsize=9)

    plt.xlabel('Number of PCA Components', fontsize=12)
    plt.ylabel('Classifier Accuracy', fontsize=12)
    plt.title(f'{framework} — Accuracy vs PCA Components', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# DNN VISUALIZATIONS (Added during DNN prep)

def plot_training_history(history, framework, save_path=None):
    """
    Plot training/validation loss and accuracy over epochs.
    Dual-panel figure: loss on the left, accuracy on the right.

    Designed for DNN training — works with any framework that provides
    a history dict. Handles optional val_loss/val_acc gracefully.

    Args:
        history: Dict with keys from {'train_loss', 'val_loss',
                 'train_acc', 'val_acc'}. Only 'train_loss' is required.
        framework: Name for the title (e.g. "Scikit-Learn", "PyTorch")
        save_path: Optional path to save the figure
    """
    has_val_loss = 'val_loss' in history and len(history['val_loss']) > 0
    has_acc = 'train_acc' in history and len(history['train_acc']) > 0
    has_val_acc = 'val_acc' in history and len(history['val_acc']) > 0

    # Determine layout: 2 panels if accuracy exists, 1 if loss only
    n_panels = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    epochs = range(1, len(history['train_loss']) + 1)

    # Left panel: Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    if has_val_loss:
        ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{framework} — Training Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right panel: Accuracy (if available)
    if has_acc:
        ax = axes[1]
        ax.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
        if has_val_acc:
            ax.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{framework} — Training Accuracy', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# AUTOENCODER VISUALIZATIONS (Added during autoencoder prep)

def plot_latent_space(latent_vectors, labels, class_names, framework,
                       method='tsne', save_path=None):
    """
    2D projection of latent space colored by class label.

    Reduces high-dimensional latent vectors to 2D using t-SNE or PCA,
    then plots a scatter with points colored by true class. Reveals
    whether the encoder learns class-separable representations.

    Args:
        latent_vectors: Array (n_samples, latent_dim).
        labels: Array (n_samples,) of integer class labels.
        class_names: List of class name strings.
        framework: Name for the title (e.g. "PyTorch", "Scikit-Learn").
        method: 'tsne' or 'pca' for dimensionality reduction.
        save_path: Optional path to save the figure.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=113, perplexity=30, max_iter=1000)
        coords = reducer.fit_transform(latent_vectors)
        method_label = 't-SNE'
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=113)
        coords = reducer.fit_transform(latent_vectors)
        method_label = 'PCA'
    else:
        raise ValueError(f"method must be 'tsne' or 'pca', got '{method}'")

    plt.figure(figsize=(10, 8))
    n_classes = len(class_names)
    cmap = plt.cm.get_cmap('tab10', n_classes)

    for i, name in enumerate(class_names):
        mask = labels == i
        plt.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)],
                    label=name, s=5, alpha=0.5)

    plt.xlabel(f'{method_label} 1', fontsize=12)
    plt.ylabel(f'{method_label} 2', fontsize=12)
    plt.title(f'{framework} — Latent Space ({method_label})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=8, markerscale=3, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()