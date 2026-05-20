# MLflow Registry Strategy

Locked-in decisions for how the deployment services use the MLflow Model Registry: alias-based versioning (not Stages), direct file-open from `version.source` (not `mlflow.<flavor>.load_model`), and runtime alias override via the `MODEL_ALIAS` env var.

## Decision Summary

| Concern | Choice |
|---------|--------|
| Version selection mechanism | **Aliases** (`@production`, `@staging`, ...) — NOT legacy Stages |
| Registry query | **`MlflowClient.get_model_version_by_alias(name, alias)`** per loader |
| Artifact load | **Direct file open** from `version.source` (joblib / torch.load / np.load / spm.Load / tf.load_weights), bypassing `mlflow.<flavor>.load_model` |
| Runtime alias selection | **`MODEL_ALIAS` env var** with fallback to the loader's module-level default (`"production"`) |
| Override env var name | **`MODEL_ALIAS`** (project-defined, not an MLflow-standard env var) |

## Aliases, Not Stages

MLflow 3.x deprecated the legacy `current_stage` field (`None` / `Staging` / `Production` / `Archived`) in favor of **registered model aliases** — arbitrary strings that point at specific versions. Our `promote_to_registry.py` has used aliases from day one; the `model_versions` table shows `current_stage = 'None'` across all 5 registered models because Stages were never set.

| Aspect | Stages (legacy) | Aliases (modern) |
|--------|-----------------|------------------|
| Available labels | Fixed enum: None, Staging, Production, Archived | Arbitrary strings (production, staging, canary, blue, green, v1-frozen, ...) |
| Multi-target | One Stage per version max | A version can carry multiple aliases simultaneously |
| API | `MlflowClient.transition_model_version_stage(name, version, stage)` | `MlflowClient.set_registered_model_alias(name, alias, version)` |
| Status as of MLflow 3.x | Deprecated; emits DeprecationWarning | Recommended |
| Our adoption | Never used | All 5 models tagged `@production` v1 |

Why this matters: anywhere docs / plans / older tutorials say "promote Staging to Production", read it as "move the `@production` alias to a new version." The lever is the same; the API call is `set_registered_model_alias`.

## Direct File-Open, Not `mlflow.<flavor>.load_model`

Each loader resolves a `ModelVersion` from the registry, then opens the artifact file directly:

```python
client = MlflowClient(tracking_uri=...)
version = client.get_model_version_by_alias(name="sk-pca", alias="production")

# version.source is a file:// URI managed by the registry
artifact_dir = _resolve_artifact_dir(version.source)
model_file = artifact_dir / "pca_model.joblib"
model = joblib.load(model_file)   # direct file open
```

The MLflow-standard alternative would be:

```python
model_uri = "models:/sk-pca@production"
model = mlflow.sklearn.load_model(model_uri)
```

We chose direct file-open. Reasoning:

| Aspect | Direct file-open (chosen) | `mlflow.<flavor>.load_model` |
|--------|---------------------------|------------------------------|
| Artifact format expected | Raw file (`.joblib`, `.pth`, `.npy`, `.h5`, `.model`) | Flavor-structured directory (`MLmodel` metadata + framework-specific files) |
| Promotion script wrote them how | `mlflow.log_artifact(file)` — raw upload | Would need `mlflow.sklearn.log_model(model, name)` etc. — flavor-structured logging |
| Re-promotion cost to switch | (status quo) | Re-promote all 5 models with their respective flavor APIs; Q-learning has no MLflow flavor at all (would need generic Python flavor) |
| Registry contract still satisfied | Yes — alias resolution still happens via `get_model_version_by_alias` | Yes |
| Q-learning fit | Native (numpy `.npy` opens with `np.load`) | Awkward (would need `python_function` flavor wrapper) |

The registry contract we care about (name + alias -> ModelVersion) is satisfied either way. The flavor-based loading adds a layer that buys nothing for our 5 models — we already trust our own training pipeline's artifact format, we don't consume third-party model packages, and the framework-specific load APIs (`joblib.load`, `torch.load(weights_only=True)`, `np.load`, `tf.load_weights`, `spm.Load`) are documented + obvious.

If a future model needed cross-org distribution (e.g., shipping a model to consumers who would call `mlflow.X.load_model` themselves), we would re-promote it in the appropriate flavor at that time.

## MODEL_ALIAS Env Var (Runtime Override)

Each loader resolves the alias via:

```python
def _resolve_alias() -> str:
    return os.environ.get(ALIAS_OVERRIDE_ENV) or MODEL_ALIAS
```

Where:
- `ALIAS_OVERRIDE_ENV = "MODEL_ALIAS"` — the env var name read at runtime
- `MODEL_ALIAS = "production"` — the module-level default when the env var is unset or empty

### Why an override env var

Two A/B-style deployment scenarios this enables, both without rebuilding the image:

1. **Canary**: promote a new model version with `@canary` alias; run one container with `-e MODEL_ALIAS=canary` alongside production traffic; monitor; promote to `@production` once verified.
2. **Staging environment**: a non-prod environment runs the same image but with `-e MODEL_ALIAS=staging` to load models tagged `@staging` instead.

The override is identical across the three services, so a single `MODEL_ALIAS=staging` env var on the compose stack switches all 5 model loads at once (or a subset, if only some services receive the env var).

### Set via docker run

```powershell
docker run --rm -d -p 8001:8001 `
  -e MLFLOW_TRACKING_URI=sqlite:////srv/mlflow/mlflow.db `
  -e MLFLOW_ARTIFACT_ROOT_OVERRIDE=/srv/mlflow `
  -e MODEL_ALIAS=staging `
  -v "${PWD}/deployment:/srv/mlflow:ro" `
  ml-fc-sklearn-svc:dev
```

### Set via docker compose

Either inline in `docker-compose.yml`:

```yaml
environment:
  MLFLOW_TRACKING_URI: sqlite:////srv/mlflow/mlflow.db
  MLFLOW_ARTIFACT_ROOT_OVERRIDE: /srv/mlflow
  MODEL_ALIAS: staging
```

Or via shell env before `docker compose up`:

```powershell
$env:MODEL_ALIAS = "staging"
docker compose up -d
```

The compose file's `${VAR}` syntax would pick it up if added to the `environment:` block as `MODEL_ALIAS: ${MODEL_ALIAS:-production}` (host env wins, falls back to "production").

### Observability

The resolved alias appears in two structured-log events per service:

- `<svc>_load_start` — logged before the registry query. Confirms what the loader is about to ask for.
- `<svc>_alias_resolved` — logged after `get_model_version_by_alias` returns. Confirms the version that backs the alias right now.

Sample log line:

```json
{
  "model_name": "sk-pca",
  "model_alias": "staging",
  "version": 3,
  "run_id": "78fd7c9d",
  "event": "pca_alias_resolved",
  "level": "info",
  "timestamp": "2026-05-14T19:42:11.503827Z"
}
```

Operators inspecting `docker compose logs sklearn-svc` can see which alias was used and which version it backed without shelling in.

### Failure mode

Setting `MODEL_ALIAS=nonexistent` (no such alias in the registry) raises at lifespan startup:

```
RuntimeError: Failed to load sk-pca@nonexistent from sqlite:////srv/mlflow/mlflow.db.
Verify the registry exists and the alias is set.
```

The original `mlflow.exceptions.MlflowException` is chained via `from exc` so the full traceback is preserved in `docker logs`.

## What This Document Does Not Cover

- **Promoting new versions**: that is `promote_to_registry.py`'s job (host-side script, not part of the running services). It uploads new artifacts, registers a new model version, and sets the `@production` alias to point at it.
- **Multi-alias per version**: a version can carry multiple aliases (e.g., both `@production` and `@v1-frozen`). The loaders read whichever alias they are pointed at via `MODEL_ALIAS`; they do not introspect the full alias set.
- **Registry write paths**: containers mount the registry read-only. Anything that mutates the registry (new promotions, alias moves) runs on the host.
- **Cross-service alias selection**: the current implementation uses one shared `MODEL_ALIAS` env var for all loaders within a service AND across services. Per-model alias granularity (e.g., pt-dnn on `@production` while pt-gan on `@canary`) would require either per-model env vars or a per-model lookup table — a future change, not in scope today.
