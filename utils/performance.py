"""
Performance tracking utilities.
Provides a simple way to measure training time and peak memory usage.
"""

import time
import tracemalloc
from contextlib import contextmanager

@contextmanager
def track_performance():
    """
    Context manager for timing and memory tracking.
    
    A context manager lets us wrap code with setup/cleanup logic.
    The 'with statment handles starting and stopping automatically.

    Usage:
        with track_performance() as perf:
        # ... your training code here ...

        print(f"Time: {perf['time']:.4f}s)
        print(f"Memory: {perf['memory]:.2f} MB)

    Returns:
    dict: Contains 'time (seconds) and 'memory (MB) after completion
    """

    # Start memory tracking before anything else
    tracemalloc.start()
    start_time = time.time()

    # This dict will be filled in after the 'with' block completes
    result = {'time': 0.0, 'memory': 0.0}

    # 'yield' pauses here - the code inside the 'with' block runs
    yield result

    # After the 'with' block completes, we resume here
    result['time'] = time.time() - start_time

    # get_tracked_memory() returns (current, peak) - we want peak usage
    _, peak_bytes = tracemalloc.get_traced_memory()
    result['memory'] = peak_bytes / (1024 * 1024)   # Convert bytes to MB

    # Clean up memory tracking
    tracemalloc.stop()

