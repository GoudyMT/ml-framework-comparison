"""
Artifact Load Verification.

Loads each D1-D5 artifact from the consolidated MLflow registry
(deployment/mlflow.db) using the production alias. Tests use the same
library calls each service will use at runtime - catches artifact rot
or dependency-missing surprises before any service code runs against
the registry.

Run this script after every promote_to_registry.py refresh and any
time the modeling-phase artifacts change on disk.

Usage (from project root):
    .venv\\Scripts\\python.exe deployment\\scripts\\verify_artifacts.py
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import mlflow
from mlflow.exceptions import MlflowException

# Path resolution
SCRIPT_DIR     = Path(__file__).resolve().parent
DEPLOYMENT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT   = DEPLOYMENT_DIR.parent
DEPLOYMENT_DB  = DEPLOYMENT_DIR / 'mlflow.db'


def file_url_to_path(file_url: str) -> Path:
    """Convert MLflow's file:/// URI to a local Path."""
    parsed = urlparse(file_url)
    # On Windows: file:///C:/path -> /C:/path -> strip leading /
    raw = parsed.path
    if raw.startswith('/') and len(raw) > 2 and raw[2] == ':':
        raw = raw[1:]
    return Path(raw)


def get_artifact_dir(name: str) -> Path:
    """Resolve the artifact directory for a registered model's production version."""
    mlflow.set_tracking_uri(f'sqlite:///{DEPLOYMENT_DB.resolve()}')
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(name, 'production')
    artifact_dir = file_url_to_path(version.source) # type: ignore
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact dir does not exist: {artifact_dir}")
    return artifact_dir



# Per-model verification functions
def test_sk_pca(artifact_dir: Path) -> str:
    """D1: full functional check - load + transform synthetic input."""
    import joblib
    import numpy as np
    pca = joblib.load(artifact_dir / 'pca_model.joblib')
    assert hasattr(pca, 'n_components_'), 'Not a fitted PCA'
    dummy = np.random.randn(1, 784).astype(np.float32)
    transformed = pca.transform(dummy)
    return (f"PCA loaded - n_components={pca.n_components_}, "
            f"transform({dummy.shape}) -> {transformed.shape}, "
            f"explained_var_sum={pca.explained_variance_ratio_.sum():.4f}")


def test_pt_dnn(artifact_dir: Path) -> str:
    """D2: full functional check - load state_dict, inspect keys + shapes."""
    import torch
    state_dict = torch.load(
        artifact_dir / 'dnn_model.pth',
        weights_only=True, map_location='cpu',
    )
    n_keys = len(state_dict)
    n_params = sum(t.numel() for t in state_dict.values())
    first_key = next(iter(state_dict.keys()))
    first_shape = list(state_dict[first_key].shape)
    return (f"state_dict loaded - {n_keys} tensors, {n_params:,} total params, "
            f"first key '{first_key}' shape={first_shape}")


def test_pt_gan(artifact_dir: Path) -> str:
    """D3: full functional check - load generator state_dict."""
    import torch
    state_dict = torch.load(
        artifact_dir / 'dcgan_generator.pth',
        weights_only=True, map_location='cpu',
    )
    n_keys = len(state_dict)
    n_params = sum(t.numel() for t in state_dict.values())
    return f"generator state_dict loaded - {n_keys} tensors, {n_params:,} total params"


def test_pt_qlearning(artifact_dir: Path) -> str:
    """D4: full functional check - load Q-table, verify shape + argmax."""
    import numpy as np
    Q = np.load(artifact_dir / 'v1_qtable_taxi.npy')
    assert Q.shape == (500, 6), f"Expected (500, 6), got {Q.shape}"
    action = int(np.argmax(Q[0]))
    assert 0 <= action < 6, f"argmax invalid: {action}"
    nonzero = (Q != 0).sum()
    return (f"Q-table loaded - shape={Q.shape}, dtype={Q.dtype}, "
            f"nonzero entries={nonzero}, sample argmax(Q[0])={action}")


def test_tf_translation(artifact_dir: Path) -> str:
    """D5: graceful fallback - sentencepiece + file checks; TF skipped (not in Windows venv)."""
    info = {}

    # 1. preprocessing_info.json - always loadable (stdlib)
    info_path = artifact_dir / 'preprocessing_info.json'
    info_dict = json.loads(info_path.read_text(encoding='utf-8'))
    info['preprocessing_keys'] = list(info_dict.keys())[:5]

    # 2. sentencepiece BPE model - try to load if installed
    bpe_path = artifact_dir / 'bpe.model'
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(bpe_path)) # type: ignore
        info['bpe_vocab_size'] = sp.get_piece_size() # type: ignore
        encoded = sp.encode_as_ids("Hello world") # type: ignore
        info['bpe_sample_encode'] = encoded[:5]
        info['bpe_full_load'] = 'OK'
    except ImportError:
        info['bpe_full_load'] = 'SKIPPED (sentencepiece not in .venv - tested by tf-svc)'
        info['bpe_file_size_kb'] = round(bpe_path.stat().st_size / 1024, 1)

    # 3. TF .h5 weights - file existence + size only (TF not in Windows venv)
    weights_path = artifact_dir / 'translation_transformer.weights.h5'
    info['tf_weights_size_mb'] = round(weights_path.stat().st_size / 1024 / 1024, 1)
    info['tf_full_load'] = 'SKIPPED (tensorflow not in Windows .venv - tested by tf-svc on WSL2)'

    return f"artifacts loaded - {info}"



# Driver

VERIFICATIONS = [
    ('sk-pca', test_sk_pca, 'D1 SK PCA'),
    ('pt-dnn', test_pt_dnn, 'D2 PT DNN'),
    ('pt-gan-dcgan', test_pt_gan, 'D3 PT DCGAN'),
    ('pt-qlearning-taxi', test_pt_qlearning, 'D4 PT Q-Learning'),
    ('tf-transformer-translation', test_tf_translation, 'D5 TF Translation'),
]


if __name__ == '__main__':
    print('=' * 78)
    print('  PHASE 0 STEP 0.4 - ARTIFACT LOAD VERIFICATION')
    print('=' * 78)
    print(f'  Registry: {DEPLOYMENT_DB}')
    print('=' * 78)

    pass_count = 0
    fail_count = 0
    for model_name, test_fn, label in VERIFICATIONS:
        print(f"\n  {label} ({model_name})")
        print('  ' + '-' * 70)
        try:
            artifact_dir = get_artifact_dir(model_name)
            result = test_fn(artifact_dir)
            print(f"  PASS - {result}")
            pass_count += 1
        except Exception as e:
            print(f"  FAIL - {type(e).__name__}: {e}")
            fail_count += 1

    print('\n' + '=' * 78)
    print(f'  SUMMARY: {pass_count} passed, {fail_count} failed')
    print('=' * 78)
    if fail_count > 0:
        raise SystemExit(1)