"""
GNN Preprocessing — Cora + ogbn-arxiv

Downloads and applies industry-standard cleaning to two graph datasets:
  - Cora (PyG Planetoid): 2708 nodes, 7 classes, transductive baseline
  - ogbn-arxiv (OGB): 169,343 nodes, 40 classes, public leaderboard benchmark

Cleaning applied:
  1. ogbn-arxiv: symmetrize directed citation edges (OGB leaderboard standard)
  2. ogbn-arxiv: remove duplicate edges after symmetrization
  3. Cora:       row-normalize binary BoW features (Kipf & Welling 2017 standard)

Not applied here (correctly deferred elsewhere):
  - Self-loops: added per-layer inside GCN/GAT (preprocessing would double-add)
  - Feature standardization for ogbn-arxiv: word2vec embeddings already normalized
  - Custom splits: OGB temporal split is the benchmark (train <2018, val 2018, test 2019)

Usage: python preprocess_gnn.py
"""

import json
from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import coalesce, is_undirected
from ogb.nodeproppred import PygNodePropPredDataset

RANDOM_STATE = 113
OUTPUT_DIR = Path('./data/processed/gnn')
CORA_ROOT = OUTPUT_DIR         # Planetoid creates data/processed/gnn/Cora/
ARXIV_ROOT = OUTPUT_DIR        # OGB creates   data/processed/gnn/ogbn_arxiv/

# Industry-standard transforms applied at dataset load time.
# Pipelines import these to guarantee identical cleaning across variants.
CORA_TRANSFORM = T.Compose([
    T.NormalizeFeatures(),           # row-normalize 0/1 BoW features (Kipf 2017)
])

ARXIV_TRANSFORM = T.Compose([
    T.ToUndirected(),                # symmetrize citation edges (OGB baseline standard)
])

def validate_graph(data, name, expected_n=None, expected_e=None, expected_classes=None):
    """
    Run health checks on a PyG Data object.

    Args:
        data: torch_geometric.data.Data instance.
        name: Dataset name for assertion messages.
        expected_n: Expected node count (None to skip).
        expected_e: Expected edge count (None to skip).
        expected_classes: Expected class count (None to skip).

    Returns:
        Dict of graph statistics (node/edge/feature/class counts, degree stats,
        symmetry flag, duplicate edge count).
    """
    n = data.num_nodes
    e = data.num_edges
    f = data.num_features
    y = data.y.view(-1)
    n_classes = int(y.max().item()) + 1

    assert not torch.isnan(data.x).any(), f"{name}: NaN in node features"
    assert not torch.isinf(data.x).any(), f"{name}: Inf in node features"
    assert data.edge_index.min().item() >= 0, f"{name}: negative edge index"
    assert data.edge_index.max().item() < n, f"{name}: edge index out of node range"

    if expected_n is not None:
        assert n == expected_n, f"{name}: expected {expected_n} nodes, got {n}"
    if expected_e is not None:
        assert e == expected_e, f"{name}: expected {expected_e} edges, got {e}"
    if expected_classes is not None:
        assert n_classes == expected_classes, (
            f"{name}: expected {expected_classes} classes, got {n_classes}"
        )

    degrees = torch.bincount(data.edge_index[1], minlength=n).float()

    # Count duplicate edges (coalesce dedupes; compare before/after count)
    _, dedup_count = coalesce(
        data.edge_index, num_nodes=n, is_sorted=False, sort_by_row=True,
    ), None
    deduped = coalesce(data.edge_index, num_nodes=n)
    n_duplicates = int(data.edge_index.size(1) - deduped.size(1))

    return {
        'n_nodes': int(n),
        'n_edges': int(e),
        'n_features': int(f),
        'n_classes': int(n_classes),
        'feature_dtype': str(data.x.dtype),
        'label_dtype': str(y.dtype),
        'degree_mean': float(degrees.mean()),
        'degree_median': float(degrees.median()),
        'degree_max': int(degrees.max()),
        'isolated_nodes': int((degrees == 0).sum()),
        'is_undirected': bool(is_undirected(data.edge_index, num_nodes=n)),
        'n_duplicate_edges': n_duplicates,
    }

def feature_stats(x):
    """
    Compute per-row norm statistics to confirm normalization worked.

    Args:
        x: Node feature tensor, shape (N, F).

    Returns:
        Dict with min/mean/max of L1 row norms.
    """
    row_sums = x.sum(dim=1)
    return {
        'row_l1_min': float(row_sums.min()),
        'row_l1_mean': float(row_sums.mean()),
        'row_l1_max': float(row_sums.max()),
    }

def main():
    print("=" * 60)
    print("GNN — Preprocessing Cora + ogbn-arxiv")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # [1/6] Download Cora, validate raw, then apply NormalizeFeatures
    print("\n[1/6] Loading Cora via PyG Planetoid (raw)...")
    cora_raw = Planetoid(root=str(CORA_ROOT), name='Cora')[0]
    cora_raw_stats = validate_graph(
        cora_raw, 'Cora', expected_n=2708, expected_e=10556, expected_classes=7,
    )
    cora_raw_feat_stats = feature_stats(cora_raw.x) # type: ignore
    print(f"    Raw: {cora_raw}")
    print(f"    Row L1 sums (raw): "
          f"min={cora_raw_feat_stats['row_l1_min']:.1f}, "
          f"mean={cora_raw_feat_stats['row_l1_mean']:.1f}, "
          f"max={cora_raw_feat_stats['row_l1_max']:.1f}")

    print("\n[2/6] Applying Cora transform (row-normalize features, Kipf 2017)...")
    cora = Planetoid(root=str(CORA_ROOT), name='Cora', transform=CORA_TRANSFORM)[0]
    cora_feat_stats = feature_stats(cora.x) # type: ignore
    print(f"    Row L1 sums (normalized): "
          f"min={cora_feat_stats['row_l1_min']:.4f}, "
          f"mean={cora_feat_stats['row_l1_mean']:.4f}, "
          f"max={cora_feat_stats['row_l1_max']:.4f}")
    assert abs(cora_feat_stats['row_l1_mean'] - 1.0) < 1e-5, (
        "Cora row-normalization failed: row sums should average to 1.0"
    )

    # [3/6] Download ogbn-arxiv, validate raw directed graph
    print("\n[3/6] Loading ogbn-arxiv via OGB (raw, directed)...")
    arxiv_raw_dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=str(ARXIV_ROOT))
    arxiv_raw = arxiv_raw_dataset[0]
    split_idx = arxiv_raw_dataset.get_idx_split()
    arxiv_raw_stats = validate_graph(
        arxiv_raw, 'ogbn-arxiv',
        expected_n=169343, expected_e=1166243, expected_classes=40,
    )
    print(f"    Raw: {arxiv_raw}")
    print(f"    Undirected: {arxiv_raw_stats['is_undirected']} "
          f"(expected False — raw is directed citations)")
    print(f"    Duplicate edges (raw): {arxiv_raw_stats['n_duplicate_edges']}")

    # [4/6] Apply arxiv transform (symmetrize)
    print("\n[4/6] Applying ogbn-arxiv transform (symmetrize, OGB baseline)...")
    arxiv_dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', root=str(ARXIV_ROOT), transform=ARXIV_TRANSFORM,
    )
    arxiv = arxiv_dataset[0]
    arxiv_stats = validate_graph(arxiv, 'ogbn-arxiv (symmetrized)')
    print(f"    Symmetrized: {arxiv}")
    print(f"    Undirected: {arxiv_stats['is_undirected']}")
    print(f"    Edge count: {arxiv_raw_stats['n_edges']:,} -> {arxiv_stats['n_edges']:,} "
          f"(+{arxiv_stats['n_edges'] - arxiv_raw_stats['n_edges']:,} reverse edges)")
    print(f"    Duplicate edges after ToUndirected: {arxiv_stats['n_duplicate_edges']} "
          f"(PyG ToUndirected dedupes internally)")
    assert arxiv_stats['is_undirected'], "arxiv symmetrization failed"

    # [5/6] Verify OGB splits partition all nodes
    print("\n[5/6] Verifying OGB split integrity...")
    split_total = (
        len(split_idx['train']) + len(split_idx['valid']) + len(split_idx['test']) # type: ignore
    )
    assert split_total == arxiv.num_nodes, ( # type: ignore
        f"ogbn-arxiv split mismatch: {split_total} != {arxiv.num_nodes}" # type: ignore
    )
    print(f"    train={len(split_idx['train']):,} + " # type: ignore
          f"valid={len(split_idx['valid']):,} + " # type: ignore
          f"test={len(split_idx['test']):,} = " # type: ignore
          f"{split_total:,} (matches {arxiv.num_nodes:,} nodes)") # type: ignore

    # [6/6] Emit preprocessing metadata
    print("\n[6/6] Writing preprocessing_info.json...")
    info = {
        'datasets': {
            'cora': {
                'source': 'PyG torch_geometric.datasets.Planetoid',
                'root': str(CORA_ROOT),
                'raw_stats': cora_raw_stats,
                'raw_feature_row_l1': cora_raw_feat_stats,
                'transform': 'NormalizeFeatures (row-normalize BoW features)',
                'transform_rationale': 'Kipf & Welling 2017 — papers with more words should not produce louder messages',
                'post_transform_feature_row_l1': cora_feat_stats,
                'split': {
                    'train_nodes': int(cora.train_mask.sum()), # type: ignore
                    'val_nodes': int(cora.val_mask.sum()), # type: ignore
                    'test_nodes': int(cora.test_mask.sum()), # type: ignore
                    'type': 'transductive_mask',
                },
            },
            'ogbn_arxiv': {
                'source': 'ogb.nodeproppred.PygNodePropPredDataset',
                'root': str(ARXIV_ROOT / 'ogbn_arxiv'),
                'raw_stats': arxiv_raw_stats,
                'transform': 'ToUndirected (symmetrize citation edges)',
                'transform_rationale': 'OGB leaderboard baseline — directed citations halve message-passing reach; symmetrization is standard',
                'post_transform_stats': arxiv_stats,
                'split': {
                    'train_nodes': len(split_idx['train']), # type: ignore
                    'val_nodes': len(split_idx['valid']), # type: ignore
                    'test_nodes': len(split_idx['test']), # type: ignore
                    'type': 'ogb_idx_split',
                    'split_basis': 'publication_year (train <2018, val 2018, test 2019)',
                },
            },
        },
        'not_applied': {
            'self_loops': 'added per-layer inside GCN/GAT (preprocessing would double-add)',
            'feature_standardization_arxiv': 'word2vec embeddings already normalized',
            'custom_splits': 'OGB temporal split is the benchmark standard',
        },
        'random_state': RANDOM_STATE,
    }

    info_path = OUTPUT_DIR / 'preprocessing_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"    Saved {info_path}")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"    Cora root: {CORA_ROOT / 'Cora'}")
    print(f"    ogbn-arxiv root: {ARXIV_ROOT / 'ogbn_arxiv'}")
    print(f"    Pipelines: import CORA_TRANSFORM and ARXIV_TRANSFORM from this module")
    print("=" * 60)

if __name__ == '__main__':
    main()