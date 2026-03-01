"""
Shared utilities for tree-based models (Decision tree, random forests),

Framework-agnostic: operates on dict-based tree structures and nujmpy arrays.
Used by No-framework, pytorch and tensorflow implementations.
Sklearn has its own built-in feature_importances+ and predict, so these
utilities serve the 3 from-scratch frameworks only.

Functions:
    compute_feature_importance: Gini importance from recursive dict tree
    flatten_tree: Convert recursive dict to flat arrays (for vectorized predict)
    predict_batch: batch prediction using flat arrays (no python recursion)
"""

import numpy as np
from collections import deque

def compute_feature_importance(tree_dict, n_features):
    """
    Calculate Gini importance (mean decrease in impurity) for each feature.

    Traverses the tree recursively, accumulating the weighted impurity
    decrease at each split node. Normalizes so importances sum to 1.0.
    This is the same algorithm sklearn uses for feature_importances_.

    Args:
        tree_dict: Recursive dict tree node with keys:
            Split nodes: 'feature', 'threshold', 'left', 'right',
                         'n_samples', 'impurity'
            Leaf nodes: 'value', 'n_samples', 'impurity'
        n_features: Total number of features in the dataset

    Returns:
        numpy array of shape (n_features,) with normalized importance scores
        (sums to 1.0, or all zeros if tree is a single leaf)
    """
    importances = np.zeros(n_features)

    def _recurse(node):
        # Leaf node — no split, no importance contribution
        if 'value' in node:
            return

        # Split node — compute weighted impurity decrease
        # Importance = (n_samples * impurity) - (n_left * imp_left) - (n_right * imp_right)
        n = node['n_samples']
        left = node['left']
        right = node['right']

        weighted_decrease = (
            n * node['impurity']
            - left['n_samples'] * left['impurity']
            - right['n_samples'] * right['impurity']
        )

        importances[node['feature']] += weighted_decrease

        # Recurse into children
        _recurse(left)
        _recurse(right)

    _recurse(tree_dict)

    # Normalize so importances sum to 1.0
    total = importances.sum()
    if total > 0:
        importances /= total

    return importances


def flatten_tree(tree_dict):
    """
    Convert recursive dict tree to flat arrays for vectorized prediction.

    Maps each node to an integer ID via BFS (breadth-first) ordering.
    Returns parallel arrays where array[node_id] gives that node's properties.
    Used by PyTorch and TensorFlow for batch prediction without Python recursion.

    Args:
        tree_dict: Recursive dict tree node (root of the tree)

    Returns:
        dict with keys:
            'feature_indices': int array — feature index per node (-1 for leaves)
            'thresholds': float array — split threshold per node (0.0 for leaves)
            'left_children': int array — left child node_id (-1 for leaves)
            'right_children': int array — right child node_id (-1 for leaves)
            'values': 2D float array — class distribution per node (n_nodes, n_classes)
    """
    # BFS to assign node IDs and collect properties
    feature_indices = []
    thresholds = []
    left_children = []
    right_children = []
    values = []

    # Queue holds (node_dict, node_id) pairs
    queue = deque()
    queue.append(tree_dict)
    node_id_map = {id(tree_dict): 0}
    next_id = 1

    # First pass: assign IDs to all nodes via BFS
    all_nodes = [tree_dict]
    bfs_queue = deque([tree_dict])
    while bfs_queue:
        node = bfs_queue.popleft()
        if 'value' not in node:  # Split node has children
            for child in [node['left'], node['right']]:
                node_id_map[id(child)] = next_id
                next_id += 1
                all_nodes.append(child)
                bfs_queue.append(child)

    # Second pass: build flat arrays using assigned IDs
    for node in all_nodes:
        if 'value' in node:
            # Leaf node
            feature_indices.append(-1)
            thresholds.append(0.0)
            left_children.append(-1)
            right_children.append(-1)
            values.append(node['value'])
        else:
            # Split node
            feature_indices.append(node['feature'])
            thresholds.append(node['threshold'])
            left_children.append(node_id_map[id(node['left'])])
            right_children.append(node_id_map[id(node['right'])])
            # Split nodes also store class distribution for probability estimates
            values.append(node.get('value', np.zeros_like(all_nodes[0].get('value', [0]))))

    return {
        'feature_indices': np.array(feature_indices, dtype=np.int32),
        'thresholds': np.array(thresholds, dtype=np.float64),
        'left_children': np.array(left_children, dtype=np.int32),
        'right_children': np.array(right_children, dtype=np.int32),
        'values': np.array(values, dtype=np.float64)
    }


def predict_batch(flat_tree, X):
    """
    Batch prediction using flat tree arrays (pure numpy, no Python recursion).

    All samples start at the root (node 0). At each iteration, samples still
    at internal nodes are routed left or right based on their feature value
    vs the node's threshold. Continues until all samples reach leaf nodes.

    Args:
        flat_tree: Dict from flatten_tree() with keys:
            'feature_indices', 'thresholds', 'left_children',
            'right_children', 'values'
        X: numpy array of shape (n_samples, n_features)

    Returns:
        predictions: int array of shape (n_samples,) — predicted class labels
        probabilities: float array of shape (n_samples, n_classes) — class probabilities
    """
    n_samples = X.shape[0]
    feature_indices = flat_tree['feature_indices']
    thresholds = flat_tree['thresholds']
    left_children = flat_tree['left_children']
    right_children = flat_tree['right_children']
    values = flat_tree['values']

    # All samples start at root (node 0)
    node_ids = np.zeros(n_samples, dtype=np.int32)

    # Route samples until all reach leaves (feature_index == -1)
    while True:
        # Which samples are still at internal (non-leaf) nodes?
        at_internal = feature_indices[node_ids] != -1
        if not at_internal.any():
            break

        # Get the feature and threshold for each sample's current node
        current_features = feature_indices[node_ids[at_internal]]
        current_thresholds = thresholds[node_ids[at_internal]]

        # Get each sample's value for its current node's split feature
        sample_values = X[at_internal, current_features]

        # Route: left if value <= threshold, right otherwise
        go_left = sample_values <= current_thresholds

        # Update node IDs for routed samples
        internal_indices = np.where(at_internal)[0]
        node_ids[internal_indices[go_left]] = left_children[node_ids[internal_indices[go_left]]]
        node_ids[internal_indices[~go_left]] = right_children[node_ids[internal_indices[~go_left]]]

    # All samples are now at leaf nodes — extract predictions
    leaf_values = values[node_ids]  # (n_samples, n_classes)

    # Normalize to probabilities
    row_sums = leaf_values.sum(axis=1, keepdims=True)
    probabilities = np.where(row_sums > 0, leaf_values / row_sums, 0.0)

    # Predicted class = argmax of class distribution
    predictions = np.argmax(leaf_values, axis=1)

    return predictions, probabilities