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

# TIME-SERIES AUGMENTATION (Added during LSTM #13)
# Offline augmentation for imbalanced sequence datasets.
# Applied once during preprocessing so both frameworks train on identical data.

def jitter(x, sigma=0.03):
    """
    Add random Gaussian noise to a sequence.

    Simulates realistic sensor noise in ECG recordings.

    Args:
        x: Sequence array, shape (seq_len,) or (seq_len, n_features).
        sigma: Standard deviation of Gaussian noise.

    Returns:
        Augmented sequence, same shape as input.
    """
    return x + np.random.normal(0, sigma, x.shape).astype(x.dtype)


def scaling(x, sigma=0.1):
    """
    Multiply entire sequence by a random scalar.

    Simulates amplitude variation from different patient lead
    placements or signal gain differences.

    Args:
        x: Sequence array, shape (seq_len,) or (seq_len, n_features).
        sigma: Standard deviation of scaling factor around 1.0.

    Returns:
        Augmented sequence, same shape as input.
    """
    factor = np.random.normal(1.0, sigma)
    return (x * factor).astype(x.dtype)


def time_warp(x, sigma=0.2, n_knots=4):
    """
    Smooth temporal distortion via cubic spline.

    Most impactful augmentation for ECG — creates realistic
    heart rate variation by warping the time axis.

    Args:
        x: Sequence array, shape (seq_len,) or (seq_len, n_features).
        sigma: Magnitude of warp at each knot point.
        n_knots: Number of interior warp control points.

    Returns:
        Augmented sequence, same shape as input.
    """
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(len(x))
    knot_positions = np.linspace(0, len(x) - 1, n_knots + 2)
    warp_magnitudes = np.random.normal(1.0, sigma, n_knots + 2)
    warp_magnitudes[0] = 1.0
    warp_magnitudes[-1] = 1.0
    warped_positions = knot_positions * warp_magnitudes
    warped_positions = np.clip(warped_positions, 0, len(x) - 1)
    warped_positions = np.sort(warped_positions)

    cs = CubicSpline(warped_positions, knot_positions)
    new_indices = cs(orig_steps)
    new_indices = np.clip(new_indices, 0, len(x) - 1)

    if x.ndim == 1:
        return np.interp(new_indices, orig_steps, x).astype(x.dtype)
    else:
        result = np.zeros_like(x)
        for feat in range(x.shape[1]):
            result[:, feat] = np.interp(new_indices, orig_steps, x[:, feat])
        return result.astype(x.dtype)


def augment_minority_classes(X, y, target_ratio=0.5, random_state=113):
    """
    Augment minority classes to reduce imbalance.

    For each class with fewer samples than target_ratio * majority_count,
    generates synthetic samples by applying random augmentation combinations
    (jitter + scaling + time_warp) to randomly selected originals.

    Args:
        X: Training data, shape (n_samples, seq_len) or (n_samples, seq_len, n_features).
        y: Training labels, shape (n_samples,).
        target_ratio: Target count as fraction of majority class. 0.5 = 50%.
        random_state: Random seed for reproducibility.

    Returns:
        tuple: (X_augmented, y_augmented) with original + synthetic samples.
    """
    rng = np.random.RandomState(random_state)
    np.random.seed(random_state)

    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()
    target_count = int(majority_count * target_ratio)

    augmentations = [jitter, scaling, time_warp]
    X_new, y_new = [X.copy()], [y.copy()]

    for cls, count in zip(classes, counts):
        if count >= target_count:
            continue

        n_needed = target_count - count
        cls_mask = y == cls
        cls_samples = X[cls_mask]

        synthetic = []
        for _ in range(n_needed):
            idx = rng.randint(0, len(cls_samples))
            sample = cls_samples[idx].copy()

            n_augs = rng.randint(1, 4)
            chosen = rng.choice(len(augmentations), n_augs, replace=False)
            for aug_idx in chosen:
                sample = augmentations[aug_idx](sample)

            synthetic.append(sample)

        X_new.append(np.array(synthetic))
        y_new.append(np.full(n_needed, cls, dtype=y.dtype))

    return np.concatenate(X_new), np.concatenate(y_new)

# KERAS CALLBACK (Added during LSTM #13)
# Reusable for any TensorFlow sequence model with imbalanced classes.

class MacroF1Callback:
    """
    Keras callback for early stopping on validation macro F1.

    Keras built-in EarlyStopping only monitors standard metrics.
    This tracks macro F1 each epoch and stores best weights.
    Import macro_f1_score from utils.metrics before using.

    Args:
        X_val: Validation input array.
        y_val: Validation labels.
        patience: Epochs to wait after best F1.

    Usage:
        from utils.rnn_utils import MacroF1Callback
        cb = MacroF1Callback(X_val, y_val, patience=10)
        model.fit(..., callbacks=[cb])
        model.set_weights(cb.best_weights)
    """
    def __init__(self, X_val, y_val, patience=10):
        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience
        self.best_f1 = 0.0
        self.wait = 0
        self.best_weights = None
        self.val_f1s = []
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        from utils.metrics import macro_f1_score
        preds = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        f1 = float(macro_f1_score(self.y_val, preds))
        self.val_f1s.append(f1)

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

# HIDDEN/CELL STATE EXTRACTION (Added during LSTM #13)
# Reusable for RNN, GRU, LSTM, Attention models.

def extract_hidden_states(model, X_sample, framework='pytorch'):
    """
    Extract per-timestep hidden states (and cell states for LSTM).

    For LSTM: returns both hidden state (h) and cell state (c).
    For GRU/RNN: returns hidden state only (cell_states=None).

    Args:
        model: Trained sequence model.
            PyTorch: nn.Module with a .rnn/.gru/.lstm attribute.
            TensorFlow: keras.Model or Sequential with RNN layers.
        X_sample: Single sample or batch. Shape (seq_len, features)
            or (batch, seq_len, features).
        framework: 'pytorch' or 'tensorflow'.

    Returns:
        tuple: (hidden_states, cell_states)
            hidden_states: shape (seq_len, hidden_dim) for single sample
            cell_states: shape (seq_len, hidden_dim) for LSTM, None for GRU/RNN
    """
    if framework == 'pytorch':
        import torch
        model.eval()
        if X_sample.dim() == 2:
            X_sample = X_sample.unsqueeze(0)

        with torch.no_grad():
            # Find the recurrent layer
            rnn_layer = None
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM)):
                    rnn_layer = module
                    break

            if rnn_layer is None:
                raise ValueError("No RNN/GRU/LSTM layer found in model")

            output, hidden = rnn_layer(X_sample)
            # output: (batch, seq_len, hidden_dim)
            hidden_states = output.squeeze(0).cpu().numpy()

            cell_states = None
            if isinstance(rnn_layer, torch.nn.LSTM):
                # For LSTM, need to run step-by-step to get cell states
                # hidden is (h_n, c_n) but only for final timestep
                # Re-run manually to capture per-timestep cell states
                h = torch.zeros(rnn_layer.num_layers, 1, rnn_layer.hidden_size,
                                device=X_sample.device)
                c = torch.zeros_like(h)
                cell_list = []
                for t in range(X_sample.size(1)):
                    _, (h, c) = rnn_layer(X_sample[:, t:t+1, :], (h, c))
                    cell_list.append(c[-1].squeeze(0).cpu().numpy())
                cell_states = np.array(cell_list)

        return hidden_states, cell_states

    elif framework == 'tensorflow':
        import tensorflow as tf
        from tensorflow import keras

        if X_sample.ndim == 2:
            X_sample = X_sample[np.newaxis, ...]

        # Rebuild model with return_sequences=True on all layers
        # to get per-timestep outputs
        layers = model.layers
        rnn_layers = [l for l in layers
                      if isinstance(l, (keras.layers.SimpleRNN,
                                        keras.layers.GRU,
                                        keras.layers.LSTM,
                                        keras.layers.Bidirectional))]

        if not rnn_layers:
            raise ValueError("No RNN/GRU/LSTM layer found in model")

        # Build extractor with return_sequences=True
        extractor = keras.Sequential()
        extractor.add(keras.layers.Input(shape=X_sample.shape[1:]))

        for layer in layers:
            if layer == layers[0] and isinstance(layer, keras.layers.InputLayer):
                continue
            if isinstance(layer, keras.layers.Dense):
                break
            if isinstance(layer, keras.layers.Bidirectional):
                inner = layer.forward_layer
                new_inner = type(inner)(
                    inner.units, return_sequences=True
                )
                extractor.add(keras.layers.Bidirectional(new_inner))
            elif isinstance(layer, (keras.layers.SimpleRNN,
                                     keras.layers.GRU,
                                     keras.layers.LSTM)):
                new_layer = type(layer)(layer.units, return_sequences=True)
                extractor.add(new_layer)
            else:
                extractor.add(layer)

        # Copy weights
        for src, dst in zip(model.layers, extractor.layers):
            try:
                dst.set_weights(src.get_weights())
            except (ValueError, AttributeError):
                pass

        all_hidden = extractor.predict(X_sample, verbose=0)
        hidden_states = all_hidden[0]  # First sample

        # TF LSTM cell states require return_state=True rebuild
        # which is complex for Sequential models. Return None for now.
        cell_states = None

        return hidden_states, cell_states

    raise ValueError(f"Unknown framework: {framework}")