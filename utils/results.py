"""
Cross-framework results tracking and comparison.

Provides utilities to save per-framework results and compare
across all 4 frameworks once each has been evaluated.
Designed to work with any model type (supervised or unsupervised)
by auto-detecting whatever metrics are passed in.

Usage:
    from utils.results import save_results, add_result, print_comparison

    # After evaluating a framework:
    save_results(results_dict, save_dir='results')
    add_result('kmeans', {'framework': 'Scikit-Learn', 'silhouette': 0.45, ...})

    # After all frameworks are done:
    print_comparison('kmeans')
"""

import json
import os
from pathlib import Path

# Resolve project root from this file's location (utils/ is one level down)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'

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

def print_comparison(model_name):
    """
    Pretty-print cross-framework comparison table.

    Reads {PROJECT_ROOT}/data/results/{model_name}.json and formats
    an aligned table. Auto-detects metrics from the stored results, 
    so it works for both supervised and unsupervised models.

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

    # Column widths: metric name + one column per framework
    metric_width = max(len(k) for k in all_keys) + 2
    fw_width = max(len(r['framework']) for r in results_list) + 2

    # Header row
    header = f"{'Metric':<{metric_width}}"
    for r in results_list:
        header += f"{r['framework']:>{fw_width}}"
    print(header)
    print("-" * len(header))

    # Data rows
    for key in all_keys:
        row = f"{key:<{metric_width}}"
        for r in results_list:
            val = r.get(key, 'N/A')
            if isinstance(val, float):
                row += f"{val:>{fw_width}.4f}"
            else:
                row += f"{str(val):>{fw_width}}"
        print(row)

    print(f"\n    Frameworks: {len(results_list)}/4 recorded")