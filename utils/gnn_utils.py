"""
Graph Neural Network utilities for Model #18.
PyTorch primitives for from-scratch GCN (#V1) and GAT (#V3) layers,
plus an OGB evaluator wrapper that bridges the leaderboard metric
to our standard metrics dict.

Shapes throughout:
    N = num_nodes
    E = num_edges  (edge_index is (2, E), row 0 = source, row 1 = target)
    F = feature dim
    H = num_heads
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops as pyg_add_self_loops
from torch_scatter import scatter_add, scatter_softmax

def normalize_adj_symmetric(edge_index, num_nodes, add_self_loops=True):
    """
    Compute symmetric-normalized adjacency as a torch sparse tensor:

        A_hat = D^(-1/2) (A + I) D^(-1/2)

    This is the GCN (Kipf & Welling 2017) message-passing operator.
    The (A + I) self-loop augmentation ensures each node sees its own
    previous-layer features; without it, a node's hidden state would
    be entirely a function of its neighbors.

    Args:
        edge_index: LongTensor (2, E). Should NOT already contain
            self-loops — they are added here when add_self_loops=True.
        num_nodes: Total node count N.
        add_self_loops: If True, augment A with I before normalizing.
            Default True (standard GCN).

    Returns:
        torch.sparse_coo_tensor of shape (N, N) holding A_hat.
        Used inside a GCN forward pass as:
            output = torch.sparse.mm(A_hat, X @ W)
    """
    if add_self_loops:
        edge_index, _ = pyg_add_self_loops(edge_index, num_nodes=num_nodes)

    row, col = edge_index
    deg = scatter_add(
        torch.ones(row.size(0), device=row.device), row, dim=0, dim_size=num_nodes,
    )
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0  # defensive; shouldn't fire with self-loops
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return torch.sparse_coo_tensor(
        edge_index, edge_weight, (num_nodes, num_nodes),
    ).coalesce()

def edge_softmax(scores, target_idx, num_nodes):
    """
    Softmax over edges grouped by target node — the attention
    normalization step in GAT and related layers.

        alpha_ij = softmax over j in N(i) of scores_ij

    Implemented via scatter_softmax so each target node's incoming
    edges sum to 1.0 in the returned alpha.

    Args:
        scores: Tensor (E,) or (E, H) of raw attention logits per edge.
        target_idx: LongTensor (E,), the target node of each edge
            (i.e., edge_index[1] in PyG convention).
        num_nodes: Total node count (used to size the scatter).

    Returns:
        alpha: Same shape as scores, normalized so that for every
            target node i the incoming edges' alpha values sum to 1.
    """
    return scatter_softmax(scores, target_idx, dim=0, dim_size=num_nodes)

def evaluate_ogb(logits, y_true, evaluator):
    """
    Run OGB's official evaluator and return its scalar metric.

    OGB expects y_pred and y_true shaped (N, 1); this wrapper handles
    the argmax and shape gymnastics so pipeline code stays clean.

    Args:
        logits: Tensor (N, C) of class logits for the evaluation split.
        y_true: Tensor (N,) or (N, 1) of ground-truth class indices.
        evaluator: `ogb.nodeproppred.Evaluator` instance.

    Returns:
        Float scalar — the metric OGB's Evaluator emits for this task.
        For ogbn-arxiv that is classification accuracy.
    """
    y_pred = logits.argmax(dim=-1, keepdim=True)
    y_true_2d = y_true.view(-1, 1)
    result = evaluator.eval({'y_true': y_true_2d, 'y_pred': y_pred})
    return float(result[evaluator.eval_metric])