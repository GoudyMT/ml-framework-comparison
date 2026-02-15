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

