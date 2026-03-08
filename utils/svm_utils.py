"""
Shared utilities for SVM models.

Framework-agnostic: operates on numpy arrays.
Used by No-Framework, PyTorch, and TensorFlow implementations.
Scikit-Learn uses built-in probability=True (Platt scaling internally),
so these utilities will serve the 3 from-scratch frameworks only.

Functions:
    to_svm_labels: Convert {0,1} labels to {-1,+1} for SVM math
    to_std_labels: Convert {-1,+1} labels back to {0,1} for evaluation
    platt_calibrate: Fit sigmoid parameters (A, B) on decision values
    platt_predict_proba: Apply fitted sigmoid to get P(y=1)
"""

import numpy as np


def to_svm_labels(y):
    """
    Convert standard {0, 1} labels to SVM {-1, +1} labels.

    SVM math requires labels in {-1, +1} for the decision function
    f(x) = sign(sum(alpha_i * y_i * K(x_i, x)) + b). Our preprocessed
    data stores labels as {0, 1} to stay compatible with evaluate_classifier
    and other shared utilities.

    Args:
        y: numpy array of labels with values in {0, 1}

    Returns:
        numpy array of labels with values in {-1, +1}
    """
    return 2 * y - 1


def to_std_labels(y):
    """
    Convert SVM {-1, +1} labels back to standard {0, 1} labels.

    After SVM prediction, convert back to {0, 1} before calling
    evaluate_classifier, plot_confusion_matrix, or any other shared utility.

    Args:
        y: numpy array of labels with values in {-1, +1}

    Returns:
        numpy array of labels with values in {0, 1}
    """
    return ((y + 1) / 2).astype(int)


def platt_calibrate(decision_values, y_true, max_iter=200, lr=0.01):
    """
    Fit Platt scaling to convert SVM decision values into probabilities.

    Learns parameters A and B for the sigmoid:
        P(y=1 | f) = 1 / (1 + exp(A * f + B))

    where f is the raw SVM decision value (distance from hyperplane).
    This is the same calibration method sklearn uses internally when
    SVC(probability=True) is set. We implement it here for the 3
    from-scratch frameworks.

    Uses gradient descent on the negative log-likelihood of the sigmoid
    model. Target values are smoothed: t_i = (N+ + 1)/(N+ + 2) for
    positive samples and t_i = 1/(N- + 2) for negative samples, following
    Platt's original formulation to avoid overfitting.

    Args:
        decision_values: numpy array of shape (n_samples,) — raw SVM
            decision function output (positive = class 1 side)
        y_true: numpy array of shape (n_samples,) — true labels in {0, 1}
        max_iter: Maximum gradient descent iterations (default 200)
        lr: Learning rate for gradient descent (default 0.01)

    Returns:
        tuple (A, B): Fitted sigmoid parameters
    """
    # Smoothed targets (Platt's formulation to avoid 0/1 boundary issues)
    # t_i = (N+ + 1)/(N+ + 2) for y=1, t_i = 1/(N- + 2) for y=0
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    t = np.where(y_true == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    # Initialize parameters
    A = 0.0
    B = 0.0

    for _ in range(max_iter):
        # Sigmoid: p = 1 / (1 + exp(A*f + B))
        exponent = A * decision_values + B
        # Clip to avoid overflow in exp
        exponent = np.clip(exponent, -500, 500)
        p = 1.0 / (1.0 + np.exp(exponent))

        # Gradients of negative log-likelihood
        # NLL = -sum(t * log(p) + (1-t) * log(1-p))
        # dNLL/dA = sum((p - t) * f)
        # dNLL/dB = sum(p - t)
        diff = p - t
        grad_A = np.dot(diff, decision_values)
        grad_B = np.sum(diff)

        # Update
        A -= lr * grad_A / len(y_true)
        B -= lr * grad_B / len(y_true)

    return A, B


def platt_predict_proba(decision_values, A, B):
    """
    Apply fitted Platt scaling to convert decision values to probabilities.

    Uses the sigmoid learned by platt_calibrate:
        P(y=1 | f) = 1 / (1 + exp(A * f + B))

    Args:
        decision_values: numpy array of shape (n_samples,) — raw SVM
            decision function output
        A: Fitted sigmoid slope parameter (from platt_calibrate)
        B: Fitted sigmoid intercept parameter (from platt_calibrate)

    Returns:
        numpy array of shape (n_samples,) — calibrated probabilities P(y=1)
    """
    exponent = A * decision_values + B
    exponent = np.clip(exponent, -500, 500)
    return 1.0 / (1.0 + np.exp(exponent))