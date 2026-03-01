"""
Cross-framework results tracking and comparison.

Provides utilities to save per-framework results and compare
across all 4 frameworks once each has been evaluated.
Designed to work with any model type (supervised or unsupervised)
by auto-detecting whatever metrics are passed in.

Usage:
    from utils.results import save_results, add_result, print_comparison, build_results_dict

    # Build results dict (replaces manual dict construction):
    results = build_results_dict('Scikit-Learn', 'RandomForest', test_metrics, perf, inference_stats, model_size)

    # After evaluating a framework:
    save_results(results, save_dir='results')
    add_result('decision_tree', results)

    # After all frameworks are done:
    print_comparison('decision_tree')
"""

import json
import os
from pathlib import Path

# Resolve project root from this file's location (utils/ is one level down)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'

def build_results_dict(framework, model_name, test_metrics, perf, inference_stats, model_size, **extra):
    """
    Build standardized results dict for cross-framework comparison.

    Combines test metrics, performance stats, and inference benchmarks
    into the format expected by save_results() and add_result().

    Args:
        framework: Framework name ('Scikit-Learn', 'PyTorch')
        model_name: Model name ('RandomForest', 'MultinomialNB')
        test_metrics: Dict from evaluate_classifier() (accuracy, f1, auc, etc.)
        perf: Dict from track_performance() context manager (time, memory)
        inference_stats: Dict from track_inference() (per_sample_us, samples_per_sec)
        model_size: Int from get_model_size() (bytes)
        **extra: Any additional model-specific fields (n_estimators=100)

    Returns:
        dict: Ready for save_results() and add_result()
    """
    results = {
        'framework': framework,
        'model': model_name,
        'training_time': float(perf['time']),
        'inference_time_per_sample_us': float(inference_stats['per_sample_us']),
        'model_size_bytes': int(model_size),
        'peak_memory_mb': float(perf['memory'])
    }
    # Add all classification/regression metrics (auto-cast to float)
    for key, val in test_metrics.items():
        results[key] = float(val)
    # Add model-specific extras (n_estimators, oob_score, dt_baseline, etc.)
    for key, val in extra.items():
        results[key] = val
    return results

def save_results(results, save_dir='results'):
    """
    Save a results dictionary to a JSON file.

    Replaces the repeated JSON-saving boilerplate in each notebook.
    Creates the directory if it doesn't exist.

    Args:
        results: Dictionary of metrics/results to save.
        save_dir: Directory to save metrics.json into.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"    Results saved to: {path}")

def add_result(model_name, result_dict):
    """
    Append a framework's results to the shared comparison file.

    Write to {PROJECT_ROOT}/data/results/{model_name}.json.
    If the framework already exists in the file, overwrites that entry.
    This lets us re-run a notebook without creating duplicates.

    Args:
        model_name: Model identifier ('kmeans', 'knn')
        result_dict: Dict of metrics. Must include 'framework' key.
    """
    if 'framework' not in result_dict:
        raise ValueError("result_dict must include 'framework' key")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = RESULTS_DIR / f'{model_name}.json' # type: ignore

    # Load existing results or start fresh
    if path.exists():
        with open(path, 'r') as f:
            results_list = json.load(f)
    else:
        results_list = []

    # Overwrite if this framework already has an entry, otherwise append
    framework = result_dict['framework']
    results_list = [r for r in results_list if r.get('framework') != framework]
    results_list.append(result_dict)

    with open(path, 'w') as f:
        json.dump(results_list, f, indent=2)

    print(f"    Added '{framework}' to {path}")
    print(f"    Frameworks recorded: {len(results_list)}/4")

def _format_value(key, val):
    """
    Format a metric value with appropriate units for readable display.

    Handles unit conversion and labeling for known metric keys.
    Unknown keys fall back to 4-decimal float or string representation.

    Args:
        key: Metric key name (e.g., 'training_time', 'model_size_bytes')
        val: Raw metric value

    Returns:
        str: Human-readable formatted value with units
    """
    if val == 'N/A':
        return 'N/A'

    # Time metrics
    if key == 'training_time':
        if val >= 60:
            return f"{val / 60:.1f} min"
        return f"{val:.2f} s"

    if key == 'inference_time_per_sample_us':
        if val >= 1000:
            return f"{val / 1000:.2f} ms"
        return f"{val:.2f} µs"

    # Size metrics
    if key == 'model_size_bytes':
        if val >= 1024 ** 3:
            return f"{val / (1024 ** 3):.2f} GB"
        if val >= 1024 ** 2:
            return f"{val / (1024 ** 2):.2f} MB"
        if val >= 1024:
            return f"{val / 1024:.1f} KB"
        return f"{val} B"

    if key == 'peak_memory_mb':
        if val >= 1024:
            return f"{val / 1024:.2f} GB"
        return f"{val:.2f} MB"

    # Default formatting
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

def print_comparison(model_name):
    """
    Pretty-print cross-framework comparison table.

    Reads {PROJECT_ROOT}/data/results/{model_name}.json and formats
    an aligned table. Auto-detects metrics from the stored results,
    so it works for both supervised and unsupervised models.
    Values are displayed with human-readable units (seconds, MB, µs).

    Args:
        model_name: Model identifier (e.g., 'kmeans', 'knn')
    """
    path = RESULTS_DIR / f'{model_name}.json'

    if not path.exists():
        print(f"    No results found for '{model_name}'")
        return

    with open(path, 'r') as f:
        results_list = json.load(f)

    if not results_list:
        print(f"    No entries in {path}")
        return

    # Auto-detect all metric keys (exclude 'framework')
    all_keys = []
    for r in results_list:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    all_keys.remove('framework')

    # Print header
    print(f"\n{'=' * 60}")
    print(f"CROSS-FRAMEWORK COMPARISON: {model_name.upper()}")
    print(f"{'=' * 60}")

    # Format all values first to determine column widths
    formatted = {}
    for key in all_keys:
        formatted[key] = []
        for r in results_list:
            val = r.get(key, 'N/A')
            formatted[key].append(_format_value(key, val))

    # Column widths based on formatted values and framework names
    metric_width = max(len(k) for k in all_keys) + 2
    fw_width = max(
        max(len(r['framework']) for r in results_list),
        max(len(v) for vals in formatted.values() for v in vals)
    ) + 2

    # Header row
    header = f"{'Metric':<{metric_width}}"
    for r in results_list:
        header += f"{r['framework']:>{fw_width}}"
    print(header)
    print("-" * len(header))

    # Data rows
    for key in all_keys:
        row = f"{key:<{metric_width}}"
        for val_str in formatted[key]:
            row += f"{val_str:>{fw_width}}"
        print(row)

    print(f"\n    Frameworks: {len(results_list)}/4 recorded")