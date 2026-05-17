# Volume Mount Strategy for Model Artifacts

Locked-in decision for how the running services access trained model artifacts. Covers the bind-mount approach, the `MLFLOW_ARTIFACT_ROOT_OVERRIDE` mechanism that makes the registry portable across hosts, and the tradeoff matrix vs alternatives.

## Decision Summary

| Concern | Choice |
|---------|--------|
| Artifact storage at runtime | **Bind-mount the host's `deployment/` at `/srv/mlflow:ro`** inside each container |
| Mount permission | **Read-only (`:ro`)** — containers never mutate the registry |
| Path portability | **`MLFLOW_ARTIFACT_ROOT_OVERRIDE` env var** + `_resolve_artifact_dir()` helper in each loader |
| Registry source of truth | **`deployment/mlflow.db`** (SQLite) + `deployment/mlruns/<run-id>/artifacts/...` on host |

## Why a Bind-Mount

Three alternatives were considered and rejected:

### Option A — Bake artifacts into each image

`COPY deployment/mlruns/<run>/artifacts/ /app/artifacts/` in the Dockerfile. Container reads from `/app/artifacts/` at runtime.

| Pro | Con |
|-----|-----|
| Self-contained image (no external mount) | Image size jumps: tf-svc would go from 917 MB to ~960 MB; pt-svc DCGAN weights add ~4 MB |
| Reproducible builds (image + artifacts pinned together) | Every model retrain requires a full image rebuild + repush. The 600 MB TF wheel never changes between retrains; only the small artifact bytes differ. |
| | Couples image versioning to model versioning. Image v1.2.3 conceptually owns model v1, but the registry tracks model versions independently — easy to get out of step. |
| | No hot-swap: changing the production alias on the registry has no effect on a running container. |

### Option B — Named Docker volume (Docker-managed, not host-bound)

`volumes: ml-artifacts:/srv/mlflow` instead of `./:/srv/mlflow`. Docker stores the volume contents in its internal area.

| Pro | Con |
|-----|-----|
| Portable across hosts (no host-path dependency in compose) | The volume needs to be populated manually (`docker cp` or a seed container) — extra setup step |
| | Tied to Docker's lifecycle: `docker volume prune` wipes it. The registry is project data, not Docker scratch. |
| | The repo already contains `deployment/mlflow.db` + `mlruns/` (with binary artifacts gitignored where size demands). Bind-mount reuses what's already on disk. |

### Option C — Cloud blob storage (S3 / GCS / Azure Blob)

MLflow tracking URI points at a remote DB; `artifact_uri` becomes `s3://...` or similar. Container needs cloud credentials.

| Pro | Con |
|-----|-----|
| Production-grade; matches how real ML services run at scale | Adds cloud-account dependency + IAM setup for local dev |
| MLflow client supports this natively; no code changes | Out of scope for the local-with-cloud-deployable target. The deployment plan is "ready to move to cloud", not "deployed to cloud". |
| The switch from local to cloud is a tracking-URI change, not a code refactor | |

**Chosen: bind-mount.** Keeps images small, decouples artifact versioning from image versioning, works without cloud credentials, and the registry data lives in the repo where it is already version-controlled.

## The Portable-Paths Mechanism

### The problem

MLflow's registry stores `version.source` as an absolute `file://` URI captured at promotion time on the host that ran `promote_to_registry.py`:

```
file:///<workspace-root>/ml-framework-comparisons/deployment/mlruns/78fd7c.../artifacts
```

This URI is correct on the promotion host but invalid inside a Linux container — the host-Windows `<workspace-root>/...` path does not exist there.

### The solution

Each loader implements `_resolve_artifact_dir(source_uri: str) -> Path`:

```python
def _resolve_artifact_dir(source_uri: str) -> Path:
    override = os.environ.get(ARTIFACT_ROOT_OVERRIDE_ENV)
    if override:
        # Extract everything from /mlruns/ onwards in the registered URI,
        # then re-anchor it under the override root.
        src_path = urlparse(source_uri).path
        idx = src_path.find("/mlruns/")
        if idx < 0:
            raise RuntimeError(
                "MLFLOW_ARTIFACT_ROOT_OVERRIDE is set but source URI has "
                "no '/mlruns/' segment to anchor against."
            )
        return Path(override) / src_path[idx + 1 :]
    return _file_uri_to_path(source_uri)
```

When `MLFLOW_ARTIFACT_ROOT_OVERRIDE=/srv/mlflow` is set inside the container, the relocation runs:

| Step | Value |
|------|-------|
| Registered URI | `file:///<workspace-root>/deployment/mlruns/78fd7c.../artifacts` |
| `urlparse(...).path` | `/<workspace-root>/deployment/mlruns/78fd7c.../artifacts` |
| Tail extracted (from `/mlruns/` onwards) | `mlruns/78fd7c.../artifacts` |
| Re-anchored under override root | `/srv/mlflow/mlruns/78fd7c.../artifacts` |

The override prefix matches the container-side bind-mount target, so the relocated path resolves to a real file.

### Backwards compatibility

With `MLFLOW_ARTIFACT_ROOT_OVERRIDE` **unset** (the default on host-side dev), `_resolve_artifact_dir()` falls through to a bare `_file_uri_to_path()` call. When the same machine both writes and reads the registry, no relocation is needed.

The override is opt-in: containers and cloud nodes set it; host devs do not.

### Where it lives

All five loaders ship an identical copy of the helper:

- `services/sklearn-svc/app/services/pca_loader.py`
- `services/pt-svc/app/services/dnn_loader.py`
- `services/pt-svc/app/services/gan_loader.py`
- `services/pt-svc/app/services/qlearning_loader.py`
- `services/tf-svc/app/services/translation_loader.py`

Each service is its own Python project with its own venv, so the duplication is conscious — matches the existing pattern of `_file_uri_to_path()` and `_default_tracking_uri()` being copied across loaders. A shared helper module would couple the services in a way they otherwise avoid.

Per-service unit tests in `tests/test_artifact_resolution.py` cover four cases each: env-var unset; Windows-style source URI with override; POSIX-style source URI with override; URI missing `/mlruns/` segment raises `RuntimeError`.

## Tradeoffs Matrix

| Approach | Image size | Reproducible | Hot-swap models | Cloud-portable | Local dev setup |
|----------|-----------|--------------|-----------------|----------------|-----------------|
| **Bind-mount (chosen)** | Smallest | Per-version | Yes (re-promote, restart container) | Needs swap to blob storage | None — host registry already in repo |
| Bake into image | +0-50 MB per service | Image+artifact pinned | Requires rebuild + repush | Image runs anywhere as-is | None |
| Named docker volume | Smallest | Per-version | Yes | Needs seed step | Manual populate |
| Cloud blob storage | Smallest | Per-version | Yes | Native | IAM + credentials |

## CI / CD Implications

GitHub Actions runners do not have `deployment/mlruns/` populated by default — the repo `.gitignore` excludes `*.npy`, `*.pth`, `*.joblib`, `*.h5`, `*.keras` for size reasons. Three options for the CI/CD work:

1. **Commit a thin synthetic registry to git for CI use only** — tiny dummy model files plus a small `mlflow.db`. Container tests run against this fixture; production loads from the real registry.
2. **Pull artifacts from cloud blob storage in CI**, pointed at by a `MLFLOW_TRACKING_URI` secret. Adds a cloud dependency to CI but matches how production would work.
3. **Mock the loader cache entirely in CI tests** — the current 107/107 host test suite already takes this approach (every fixture monkeypatches the loader's module-level slots so no real `MlflowClient` is ever instantiated).

Option 3 is the current default. Option 2 fits when end-to-end container tests run in CI and need to exercise the real load path.

## Production Deployment Path

The `MLFLOW_TRACKING_URI` env var is the single switch between local and cloud:

```bash
# Local (bind-mount):
export MLFLOW_TRACKING_URI=sqlite:////srv/mlflow/mlflow.db
export MLFLOW_ARTIFACT_ROOT_OVERRIDE=/srv/mlflow

# Cloud (Postgres tracking DB + S3 artifacts):
export MLFLOW_TRACKING_URI=postgresql://user:pass@db.example.com/mlflow
unset MLFLOW_ARTIFACT_ROOT_OVERRIDE   # cloud URIs are remote-readable as-is
```

When cloud blob storage is used, MLflow's client reads `s3://bucket/path/artifacts/...` URIs natively — no `_resolve_artifact_dir` substitution needed (the URI is reachable from any host with valid credentials). The override mechanism is local dev's bridge over the host-path-vs-container-path mismatch; in cloud it is irrelevant.

## Failure Modes + Diagnostics

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError` opening `/C:/Users/<...>/mlruns/<run-id>/artifacts/<file>.joblib` inside container | `MLFLOW_ARTIFACT_ROOT_OVERRIDE` not set in the container env | Set the env var via `docker run -e` or the compose `environment:` block |
| `mlflow.exceptions.MlflowException: Detected out-of-date database schema` | Container's mlflow version older than the one that wrote the DB | Bump the service's mlflow pin in `pyproject.toml`, regen the Linux lockfile, rebuild the image |
| `RuntimeError: MLFLOW_ARTIFACT_ROOT_OVERRIDE is set ... but source URI has no '/mlruns/' segment` | Registry URI does not follow the standard `mlruns/<run-id>/artifacts` layout | Either unset the override (host-side) or re-promote artifacts into a standard-layout registry |
| Container exits with `Application startup failed. Exiting.` and no other context | Lifespan load raised something the loader did not catch | Check `docker logs <container>` — the loader chains exceptions via `raise ... from exc`, so the original cause is in the traceback |
| Permission denied opening files under the mount on Linux hosts | Host file UID does not match container UID 1000 | `chown -R 1000:1000` the mount source on the host, or bind-mount with `:ro,Z` on SELinux-enabled distros, or override the container user via the compose `user:` directive |

## What This Document Does Not Cover

- **Multi-replica orchestration**: running multiple instances of a service (Kubernetes, horizontal scaling) means every replica needs access to the same artifacts. Bind-mount with a shared filesystem works; named volumes need to be cluster-aware. Cloud blob storage is the simplest answer at scale.
- **Registry write permission**: containers mount `:ro`. The only thing that writes the registry is `promote_to_registry.py` running on the host. Services that need to log new artifacts at runtime (A/B test experiment tracking, online metric capture) would need write access — a separate decision from this document.
- **Encryption at rest**: the bind-mount inherits the host filesystem's encryption settings (BitLocker, LUKS, etc.). Cloud blob storage has its own encryption story (KMS, customer-managed keys) that belongs in a cloud-deployment doc, not here.
