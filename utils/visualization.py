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

def plot_feature_importance(weights, feature_names, framework, save_path=None, top_n=15):
    """
    Plot feature importance as horizontal bar chart.
    Green = positive coefficient
    Red = negative coefficient

    Args:
        weights: Model coefficients/weights array
        feature_names: List of feature names
        framework: Name for the title
        save_path: Optinal path to save the figure
        top_n: Number of top features to display
    """
    # Sort by absolute value to find most important features
    indices = np.argsort(np.abs(weights))[::-1][:top_n]

    plt.figure(figsize=(10, 8))

    # Color based on coefficient sign
    colors = ['green' if weights[i] > 0 else 'red' for i in indices]

    plt.barh(range(len(indices)), weights[indices], color=colors, alpha=0.7)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel('Coefficient Value', fontsize=12)
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