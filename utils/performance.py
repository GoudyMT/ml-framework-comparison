"""
Performance tracking utilities.
Provides a simple way to measure training time and peak memory usage.
Supports both CPU (Python) memory and GPU memory (PyTorch/TensorFlow).
"""

import time
import tracemalloc
from contextlib import contextmanager

def _detect_gpu_backend():
    """
    Detect which GPU framework is available (if any).

    Returns:
    str: 'pytorch', 'tensorflow', or None
    """
    # Try PyTorch first (more common for ML)
    try:
        import torch
        if torch.cuda.is_available():
            return 'pytorch'
    except ImportError:
        pass

    # Try TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return 'tensorflow'
    except ImportError:
        pass

    return None

def _reset_gpu_memory_stats(backend):
    # Reset GPU peak memory statistics before measurements.
    if backend == 'pytorch':
        import torch
        torch.cuda.reset_peak_memory_stats()
    elif backend == 'tensorflow':
        import tensorflow as tf
        tf.config.experimental.reset_memory_stats('GPU:0')

def _get_gpu_peak_memory(backend):
    """
    Get peak GPU memory usage in bytes.

    Returns:
        int: Peak memory in bytes, or 0 if unavailable
    """
    if backend == 'pytorch':
        import torch
        return torch.cuda.max_memory_allocated()
    elif backend == 'tensorflow':
        import tensorflow as tf
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        return mem_info['peak']
    return 0

@contextmanager
def track_performance(gpu=False):
    """
    Context manager for timing and memory tracking.
    
    A context manager lets us wrap code with setup/cleanup logic.
    The 'with' statement handles starting and stopping automatically.

    Args:
        gpu: If True, also track GPU memory usage (requires PyTorch or TensorFlow
             with CUDA available). Automatically detects whcih framework to use.

    Usage:
        # CPU only (default)
        with track_performance() as perf:
            # ... your training code here ...
        print(f"Time: {perf['time']:.4f}s")
        print(f"Memory: {perf['memory']:.2f} MB")

        # With GPU tracking (PyTorch/TensorFlow)
        with track_performance(gpu=True) as perf:
            y_pred = model.predict(X_test)
            torch.cuda.synchronize()  # Ensure GPU ops complete
        print(f"GPU Memory: {perf['gpu_memory']:.2f} MB")

    Returns:
        dict: Contains 'time' (seconds), 'memory' (CPU MB),
              and 'gpu_memory' (GPU MB, only if gpu=True)
    """
    # Detect and setup GPU tracking if requested
    gpu_backend = None
    if gpu:
        gpu_backend = _detect_gpu_backend()
        if gpu_backend:
            _reset_gpu_memory_stats(gpu_backend)

    # Start CPU memory tracking
    tracemalloc.start()
    start_time = time.time()

    # This dict will be filled in after the 'with' block completes
    result = {'time': 0.0, 'memory': 0.0, 'gpu_memory': 0.0}

    # 'yield' pauses here - the code inside the 'with' block runs
    yield result

    # After the 'with' block completes, we resume here
    result['time'] = time.time() - start_time

    # Get CPU peak memory (tracemalloc tracks Python heap allocations)
    _, peak_bytes = tracemalloc.get_traced_memory()
    result['memory'] = peak_bytes / (1024 * 1024)  # Convert bytes to MB

    # Get GPU peak memory if tracking was enabled
    if gpu_backend:
        gpu_peak_bytes = _get_gpu_peak_memory(gpu_backend)
        result['gpu_memory'] = gpu_peak_bytes / (1024 * 1024)  # Convert to MB

    # Clean up CPU memory tracking
    tracemalloc.stop()

# INFERENCE & MODEL SIZE TRACKING (Added during Naive Bayes)

def track_inference(predict_fn, X, n_runs=100):
    """
    Measure inference speed by timing repeated predictions.

    Runs predict_fn(X) n_runs times and reports average time.
    First run is a warmup (excluded from timing) to avoid
    cold-start effects (JIT compilation, cache loading, etc.).

    Args:
        predict_fn: Callable that takes X and returns predictions.
            Example: model.predict, lambda x: model(x).argmax(1)
        X: Input data to predict on (full test set).
        n_runs: Number of timed runs (excluding warmup).

    Returns:
        dict: {
            'total_time': float (avg seconds for full batch),
            'per_sample_us': float (microseconds per sample),
            'samples_per_sec': float (throughput)
        }
    """
    # warmup run - excludes JIT, cache, lazy initalization overhead
    _ = predict_fn(X)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = predict_fn(X)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    n_samples = len(X)

    return {
        'total_time': avg_time,
        'per_sample_us': (avg_time / n_samples) * 1e6,
        'samples_per_sec': n_samples / avg_time
    }

def get_model_size(model, framework='numpy'):
    """
    Estimate model size in bytes from its learned parameters.

    Different frameworks store parameters differently:
    - numpy: accepts a dict of numpy arrays, sum .nbytes
    - sklearn: sums .nbytes of known learn attributes (ending in _)
    - pytorch: sum param.nelement() * param.element_size()
    - tensorflow: sums np.prod(v.shape) * v.dtype.size

    Args:
        model: Trained model object, or dict of numpy arrays for 'numpy'.
        framework: One of 'numpy', 'sklearn', 'pytorch', 'tensorflow'.

    Returns:
        int: Total model size in bytes.
    """
    import numpy as np

    if framework == 'numpy':
        # Expect a dict of {name: np.ndarray}
        if isinstance(model, dict):
            return sum(arr.nbytes for arr in model.values()
                       if isinstance(arr, np.ndarray))
        # Single array fallback
        if isinstance(model, np.ndarray):
            return model.nbytes
        return 0

    elif framework == 'sklearn':
        # Sum .nbytes of all learned attributes (sklearn convention: end with _)
        total = 0
        for attr_name in dir(model):
            if attr_name.endswith('_') and not attr_name.startswith('__'):
                attr = getattr(model, attr_name, None)
                if isinstance(attr, np.ndarray):
                    total += attr.nbytes
        return total

    elif framework == 'pytorch':
        import torch
        return sum(p.nelement() * p.element_size()
                   for p in model.parameters())

    elif framework == 'tensorflow':
        return sum(np.prod(v.shape) * v.dtype.size
                   for v in model.trainable_variables)

    return 0 