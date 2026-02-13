"""
Shared evaluation metrics for all frameworks.
Included both regression metrics (from Linear Regression) and
classification metrics (For Logistic Regression and beyond).
"""

import numpy as np

# REGRESSION METRICS

def mse(y_true, y_pred):
    """
    Mean Squared Error.
    Average of squared difference between predictions and actual values.
    """
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    Square root of MSE - gives error in same units as target variable.
    """
    return np.sqrt(mse(y_true, y_pred))

def r_squared(y_true, y_pred):
    """
    Coefficent of Determination (R^2).
    Proportion of variance in y explained by the model.
    1.0 = Perfect, 0.0 = no better than predicting the mean
    """
    ss_res = np.sum((y_true - y_pred) ** 2)             # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)    # Total sum of squares
    return 1 - (ss_res / ss_tot)

# CLASSIFICATION METRICS

def accuracy(y_true, y_pred):
    """
    Classification accuracy.
    Percentage of correct predictions. Can be misleading with imbalanced data.
    """
    return np.mean(y_true == y_pred)

def confusion_matrix_values(y_true, y_pred):
    """
    Calculate confusion matrix components.

    Returns:
        tuple: (TP, FP, TN, FN)
        - TP: True Positive (predicted 1, actual 1)
        - FP: False Positive (predicted 1, actual 0) - "False alarm"
        - TN: True Negative (prediced 0, actual 0)
        - FN: False Negative (predicted 0, actual 1) - "Missed detection
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, fp, tn, fn

def precision(y_true, y_pred):
    """
    Precision: TP / (TP / FP)
    "Of all predicted positives, how many were actually positive?
    High precision = few false alarms.
    """                
    tp, fp, _, _ = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    """
    Recall (Sensitivity): TP / (TP + FN)
    "Of all actual positives, how many did we catch?"
    High recall = few missed detections. Critical for fraud detection.
    """
    tp, _, _, fn = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    """
    F1 Score: Harmonic mean of precision and recall.
    Balances the tradeoff between precision and recall.
    Range: 0 to 1, higher is better.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

def roc_curve(y_true, y_proba, n_thresholds=100):
    """
    Compute ROC (Receiver Operating Characteristic) curve.

    The ROC curve shows the tradeoff between:
    - True Positive Rate (TPR/Recall): catching fraud
    - False Positive Rate (FPR): falsely flagging legitimate transactions

    Args:
        y_true: Actual labels (0 to 1)
        y_proba: Predicted probabilities (0.0 to 1.0)
        n_thresholds: Number of threshold points to evaluate

    Returns:
        fpr: False Positive Rates at each threshold
        tpr: True Positive Rates at each threshold
        thresholds: The thresholds values used
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        # Convert probabilities to predictions at this threshold
        y_pred = (y_proba >= thresh).astype(int)
        tp, fp, tn, fn = confusion_matrix_values(y_true, y_pred)

        # TPR = TP / (TP + FN) - same as recall
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        # FPR = FP / (FP + TN) - false alarm rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def auc_score(y_true, y_proba):
    """
    Area Under the ROC Curve (AUC).

    Measures the model's ability to distinguish between classes.
    - 1.0 = perfect discrimination
    - 0.5 = random guessing (no discrimination)
    - < 0.5 = worse than random (predictions inverted)

    Uses trapezoidal rule for numerical integration.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    # Sort by FPR for proper integration (low to high)
    sorted_idx = np.argsort(fpr)
    fpr_sorted = fpr[sorted_idx]
    tpr_sorted = tpr[sorted_idx]

    # Trapezoidal integration: area under the curve
    return np.trapezoid(tpr_sorted, fpr_sorted)

# MULTI-CLASS METRICS (Added during KNN Prep)

def confusion_matrix_multiclass(y_true, y_pred, n_classes):
    """
    Build NxN confusion matrix for multi-class classification.

    Unlike binary confusion matrix (2x2), this handles any number of classes.
    Rows represent actual classes, columns represent predicted classes.
    Diagonal values are correct predictions; off-diagonal are misclassifications.

    Args:
        y_true: Actual class labels (0 to n_classes-1)
        y_pred: Predicted class labels (0 to n_classes-1)
        n_classes: Total number of classes

    Returns: 
        numpy array: NxN confusion matrix where cm[i,j] = count of samples with true label i predicted as label j
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Get unique classes and create mapping to 0-index positions
    # This handles labels like 1-7 (Covertype) or 0-6 or any range
    classes = np.unique(y_true)
    classes_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for true_label, pred_label in zip(y_true, y_pred):
        cm[classes_to_idx[true_label], classes_to_idx[pred_label]] += 1
    return cm

def macro_f1_score(y_true, y_pred, return_per_class=False):
    """
    Macro-averaged F1 score for multi-class classification.

    Calculates F1 score for each class independently (one-vs-all),
    then takes the unweighted average. This treats all classes equally,
    regardless of their frequency in the dataset.

    Args:
        y_true: Actual class labels
        y_pred: Predicted class labels
        return_per_class: If true, also return the per-class F1 score list

    Returns:
        float: Average F1 score across all classes
        OR
        tuple: (macro_f1, per_class_score) if return_per_class=True

    Note:
        Macro F1 is preferred over accuracy for imbalanced multi-class problems
        because it doesn't let majority classes dominate the metric.
    """
    classes = np.unique(y_true)
    f1_scores = []

    for cls in classes:
        # Create binary masks: "this class" vs "all other classes"
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        # Use existing f1_score function (already defined above)
        f1_scores.append(f1_score(y_true_binary, y_pred_binary))

    macro = np.mean(f1_scores)

    if return_per_class:
        return macro, f1_scores
    return macro