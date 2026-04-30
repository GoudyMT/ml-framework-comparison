"""
Creates `deployment/mlflow.db` and registers all 5 deployment-phase models
with `production` aliases. Each model gets:
  - A new run in the consolidated DB tagged with source-run lineage
  - Artifacts re-logged under `deployment/mlruns/<run_id>/artifacts/`
  - A registered model entry with v1
  - `production` alias on v1

D1-D4 + D5 PT all have existing source MLflow runs (from modeling phase) and
their metrics are pulled forward as lineage tags. D5 TF has no MLflow tracking,
so it's registered from saved artifacts directly.

Re-runnable: if the registered models already exist, a new version (v2, v3, ...)
is created and the `production` alias moves to the new version. The old
versions remain in the registry for audit.

Usage (from project root):
    .venv\\Scripts\\python.exe deployment\\scripts\\promote_to_registry.py
"""

from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

# Path resolution: this script lives at deployment/scripts/promote_to_registry.py
SCRIPT_DIR     = Path(__file__).resolve().parent          # deployment/scripts/
DEPLOYMENT_DIR = SCRIPT_DIR.parent                        # deployment/
PROJECT_ROOT   = DEPLOYMENT_DIR.parent                    # project root
DEPLOYMENT_DB  = DEPLOYMENT_DIR / 'mlflow.db'
DEPLOYMENT_DB.parent.mkdir(parents=True, exist_ok=True)

PROMOTIONS = [
    {
        'name':          'sk-pca',
        'source_db':     'Scikit-Learn/08-pca/mlflow.db',
        'source_run_id': '378a610e2eed470db16065cf9584634a',
        'artifacts':     ['Scikit-Learn/08-pca/results/pca_model.joblib'],
        'framework':     'Scikit-Learn',
        'paradigm':      'unsupervised_dim_reduction',
        'description':   'D1: SK PCA dimensionality reduction (Fashion-MNIST, 150 components)',
    },
    {
        'name':          'pt-dnn',
        'source_db':     'PyTorch/09-dnn/mlflow.db',
        'source_run_id': 'aa31927a343e4c4a848a8938508f689f',
        'artifacts':     ['PyTorch/09-dnn/results/dnn_model.pth'],
        'framework':     'PyTorch',
        'paradigm':      'supervised_classifier',
        'description':   'D2: PT DNN classifier (UCI HAR, 96.03% test accuracy)',
    },
    {
        'name':          'pt-gan-dcgan',
        'source_db':     'PyTorch/14-gans/mlflow.db',
        'source_run_id': '5c5e31008fcd4640b60152757d520b50',
        'artifacts':     ['PyTorch/14-gans/results/dcgan_generator.pth'],
        'framework':     'PyTorch',
        'paradigm':      'generative',
        'description':   'D3: PT DCGAN generator (CIFAR-10, FID 30.57)',
    },
    {
        'name':          'pt-qlearning-taxi',
        'source_db':     'PyTorch/20-q-learning/mlflow.db',
        'source_run_id': '91377d65d8554afaba89ad7e8e81be1b',
        'artifacts':     ['PyTorch/20-q-learning/results/v1_qtable_taxi.npy'],
        'framework':     'PyTorch',
        'paradigm':      'reinforcement_learning',
        'description':   'D4: PT Q-Learning V1 Tabular (Taxi-v4, eval +8.38)',
    },
    {
        'name':          'tf-transformer-translation',
        'source_db':     None,             # No source MLflow tracking - register from disk
        'source_run_id': None,
        'artifacts': [
            'TensorFlow/16-transformers/results/translation_transformer.weights.h5',
            'data/processed/transformers_translation/bpe.model',
            'data/processed/transformers_translation/bpe.vocab',
            'data/processed/transformers_translation/preprocessing_info.json',
        ],
        'framework':     'TensorFlow',
        'paradigm':      'supervised_seq2seq',
        'description':   'D5: TF Transformer Translation (EN-ES, BLEU 0.4456) bundled with sentencepiece BPE tokenizer + preprocessing config',
    },
]


def get_source_metrics(source_db_rel: str, source_run_id: str) -> dict:
    """Pull metrics from the source run for lineage. Returns empty dict if unavailable."""
    if not source_db_rel or not source_run_id:
        return {}
    src_db = PROJECT_ROOT / source_db_rel
    if not src_db.exists():
        return {}
    mlflow.set_tracking_uri(f'sqlite:///{src_db.resolve()}')
    client = mlflow.MlflowClient()
    try:
        run = client.get_run(source_run_id)
        return {k: float(v) for k, v in run.data.metrics.items()}
    except MlflowException:
        return {}


def promote_model(entry: dict):
    print('=' * 78)
    print(f"  Promoting: {entry['name']}")
    print(f"  {entry['description']}")
    print('=' * 78)

    # 1. Pull source metrics for lineage (if a source MLflow run exists)
    source_metrics = get_source_metrics(entry['source_db'], entry['source_run_id'])
    if source_metrics:
        print(f"  Source metrics ({len(source_metrics)} keys): "
              f"{list(source_metrics.keys())[:5]}{'...' if len(source_metrics) > 5 else ''}")
    else:
        print(f"  No source metrics (D5 TF or unavailable)")

    # 2. Switch tracking to consolidated deployment DB
    mlflow.set_tracking_uri(f'sqlite:///{DEPLOYMENT_DB.resolve()}')

    exp_name = 'deployment-models'
    if mlflow.get_experiment_by_name(exp_name) is None:
        # Explicit artifact_location keeps artifacts under deployment/mlruns/
        # rather than MLflow's default ./mlruns/ relative to CWD
        artifact_location = (DEPLOYMENT_DB.parent / 'mlruns').as_uri()
        mlflow.create_experiment(exp_name, artifact_location=artifact_location)
    mlflow.set_experiment(exp_name)

    # 3. New run in consolidated DB - log lineage tags + metrics + artifacts
    with mlflow.start_run(run_name=f"promote-{entry['name']}") as run:
        mlflow.set_tags({
            'framework':       entry['framework'],
            'paradigm':        entry['paradigm'],
            'description':     entry['description'],
            'source_db':       entry['source_db'] or '(registered from disk; no source MLflow run)',
            'source_run_id':   entry['source_run_id'] or 'none',
            'promotion_phase': 'deployment-phase-step-0.2',
        })

        for k, v in source_metrics.items():
            mlflow.log_metric(k, v)

        for art_rel in entry['artifacts']:
            art_path = PROJECT_ROOT / art_rel
            if not art_path.exists():
                raise FileNotFoundError(f"Artifact not found: {art_path}")
            mlflow.log_artifact(str(art_path))
            print(f"    Logged: {art_rel}")

        artifact_uri = mlflow.get_artifact_uri()
        promotion_run_id = run.info.run_id

    # 4. Register model + create version + set production alias
    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(entry['name'])
        print(f"  Created registered model: {entry['name']}")
    except MlflowException:
        print(f"  Registered model exists; will create new version")

    version = client.create_model_version(
        name=entry['name'],
        source=artifact_uri,
        run_id=promotion_run_id,
        description=entry['description'],
    )
    print(f"  Created version: v{version.version}")

    client.set_registered_model_alias(
        name=entry['name'],
        alias='production',
        version=version.version,
    )
    print(f"  Set alias: production -> v{version.version}")
    print()


if __name__ == '__main__':
    print(f"Consolidated registry: {DEPLOYMENT_DB}")
    print(f"Promoting {len(PROMOTIONS)} models...\n")

    for entry in PROMOTIONS:
        promote_model(entry)

    # Final verification - aliases live on RegisteredModel as a dict
    mlflow.set_tracking_uri(f'sqlite:///{DEPLOYMENT_DB.resolve()}')
    client = mlflow.MlflowClient()
    print('=' * 78)
    print('  PROMOTION COMPLETE - Final Registry State')
    print('=' * 78)
    for rm in client.search_registered_models():
        # rm.aliases is a dict: {alias_name: version_number}
        aliases_dict = dict(rm.aliases) if rm.aliases else {}
        print(f"\n  {rm.name}")
        for v in client.search_model_versions(f"name='{rm.name}'"):
            # Reverse-lookup: which aliases point at this version?
            v_aliases = [a for a, ver in aliases_dict.items() if str(ver) == str(v.version)]
            print(f"    v{v.version}  aliases={v_aliases}")
            print(f"        run_id: {v.run_id}")
            print(f"        source: {v.source}")