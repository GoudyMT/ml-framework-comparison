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
