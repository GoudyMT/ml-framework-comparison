"""
RNN-specific computation util.
Reusable for RNN (#12), LSTM (#13), Attention (#15), Transformers (#16).

Functions:
    compute_gradient_norms — Per-layer gradient magnitudes (vanishing gradient demo)
"""

import numpy as np


def compute_gradient_norms(model, loss_fn, X_batch, y_batch, framework='pytorch'):
    """
    Compute per-layer gradient norms after a backward pass.

    Used to demonstrate vanishing gradients in vanilla RNN vs healthy
    gradients in GRU/LSTM. Returns a dict of {layer_name: gradient_norm}.

    Args:
        model: Trained model (PyTorch nn.Module or TF keras.Model).
        loss_fn: Loss function (CrossEntropyLoss for PT, SparseCCE for TF).
        X_batch: Small input batch for gradient computation.
        y_batch: Corresponding labels.
        framework: 'pytorch' or 'tensorflow'.

    Returns:
        dict: {layer_name: float} mapping layer names to L2 gradient norms.
    """
    gradient_norms = {}

    if framework == 'pytorch':
        import torch
        model.train()
        model.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_norms[name] = float(param.grad.norm().item())

    elif framework == 'tensorflow':
        import tensorflow as tf
        with tf.GradientTape() as tape:
            logits = model(X_batch, training=True)
            loss = loss_fn(y_batch, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        for var, grad in zip(model.trainable_variables, grads):
            if grad is not None:
                gradient_norms[var.name] = float(tf.norm(grad).numpy())

    return gradient_norms