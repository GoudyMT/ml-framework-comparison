# Dependency Management Strategy

Locked-in decision for the deployment phase. Covers Python version, lock-file tooling, per-service vs project-wide split, and the daily workflow.

## Decision Summary

| Concern | Choice |
|---------|--------|
| Python version | **3.12.2** (matches modeling-phase environment) |
| Lock-file tool | **pip-tools** (`pip-compile` + `pip-sync`) |
| Manifest format | **`pyproject.toml`** (PEP 621 standard) |
| Scope | **Per-service** (each service has its own pyproject.toml + lock file) |
| Hash pinning | **Yes** (`pip-compile --generate-hashes`) for production builds |

## Why These Choices

### Python 3.12.2 (pinned)

The modeling phase trained models on Python 3.12.2 in both Windows `.venv` and WSL2 `tf-gpu-venv`. Sklearn / PyTorch / TensorFlow can occasionally produce slightly different model behavior across Python versions due to stdlib changes (e.g., dict ordering, hash randomization, RNG paths). Pinning the inference environment to the same version eliminates this variable.

Pinned in three places:
1. `pyproject.toml` `[project] requires-python = "==3.12.2"` per service
2. Dockerfile base image `FROM python:3.12.2-slim`
3. CI matrix `python-version: '3.12.2'` in GitHub Actions

### pip-tools over alternatives

| Option | Why rejected (or chosen) |
|--------|--------------------------|
| **pip-tools** | Chosen. Industry-standard, well-documented, deterministic. Splits abstract deps (`requirements.in`) from locked deps (`requirements.txt` with all transitives + hashes). Works with stock pip - no new tool to install in production containers. |
| **Plain `requirements.txt`** | What the modeling phase used. Insufficient for production: no transitive-dep locking (`pip install -r requirements.txt` resolves transitives at install time, can produce different versions on different machines), no hash verification (supply-chain risk), no dev/prod split. |
| **Poetry** | Rejected. Cross-platform lock-file edge cases (Windows venv paths sometimes mismatch Linux container paths in poetry.lock metadata). Slower installs. Bundles its own venv manager which conflicts with the project's existing `.venv` convention. |
| **uv** | Modern faster alternative (Rust-based, built by Astral). Mentioned for awareness; pip-tools is the safer pedagogical choice in 2026 due to broader documentation. **Migration path open**: `uv pip compile` reads pip-tools' `requirements.in` syntax directly, so we can switch later without rewriting source manifests. |
| **conda / mamba** | Rejected. Conda's strength is binary packages with non-Python deps (CUDA, MKL); for our CPU-inference Docker images, pip wheels are sufficient. Mixing conda + pip in containers adds layers without value. |

### Per-service scope (not project-wide)

The deployment phase has 3 services with **disjoint runtime requirements**:
- `sklearn-svc` needs `scikit-learn`, `joblib` — ~50 MB total
- `pt-svc` needs `torch` — ~200 MB
- `tf-svc` needs `tensorflow` + `sentencepiece` — ~500 MB

A project-wide lock file would force every container to install all three frameworks, ballooning image sizes 5-10x. Per-service lock files keep each container minimal.

**Tradeoff**: shared dependencies (FastAPI, Pydantic, structlog, prometheus-client) duplicate across all 3 lock files. We accept this for image-size benefit. If a security CVE hits a shared dep, we update 3 files instead of 1 — small ongoing maintenance cost.

## Per-Service File Layout

Each service follows this structure:

```text
services/<svc-name>/
├── pyproject.toml              # PEP 621 manifest - source of truth for abstract deps
├── requirements.in             # Optional: pip-tools input format (alternative to pyproject.toml)
├── requirements.txt            # Auto-generated locked deps (DO NOT EDIT BY HAND)
├── requirements-dev.txt        # Auto-generated dev tooling locks
├── app/
│   └── ... (service code)
└── tests/
    └── ... (service tests)
```

## pyproject.toml Template

```toml
[project]
name = "sklearn-svc"
version = "0.1.0"
description = "FastAPI service for SK PCA (D1)"
requires-python = "==3.12.2"

# Abstract runtime dependencies - what the service needs
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.5",
    "scikit-learn>=1.5",
    "joblib>=1.4",
    "mlflow>=3.10",
    "structlog>=24.0",
    "prometheus-client>=0.20",
]

# Dev tooling (separate group; not installed in production container)
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",        # for FastAPI TestClient
    "ruff>=0.6",
    "mypy>=1.11",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

## Daily Workflow

### When adding a new dependency

```powershell
# 1. Edit pyproject.toml - add the abstract requirement
# (e.g., add "redis>=5.0" to dependencies = [...])

# 2. Re-compile the lock file
cd deployment/services/sklearn-svc
pip-compile pyproject.toml --output-file requirements.txt --generate-hashes

# 3. Sync your local venv to match the lock file
pip-sync requirements.txt
```

### When upgrading dependencies

```powershell
# Upgrade everything to latest compatible versions
pip-compile pyproject.toml --output-file requirements.txt --generate-hashes --upgrade

# Or upgrade a specific package
pip-compile pyproject.toml --output-file requirements.txt --generate-hashes --upgrade-package fastapi
```

### In Dockerfile (production)

```dockerfile
# Multi-stage: builder installs from locked file
FROM python:3.12.2-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes -r requirements.txt --target=/install

# Runtime: copy installed deps, no pip needed
FROM python:3.12.2-slim AS runtime
COPY --from=builder /install /usr/local/lib/python3.12/site-packages
COPY app/ /app/
WORKDIR /app
USER 1000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`--require-hashes` blocks installation of any package whose sha256 doesn't match the lock file. **Supply-chain attack mitigation** — even if PyPI gets compromised, an attacker would need to also compromise our git repo to swap hashes.

## Onboarding (New Contributor)

```powershell
# 1. Install pip-tools in the project venv (one-time)
.\.venv\Scripts\pip.exe install pip-tools

# 2. For each service you'll work on, sync to the lock file
cd deployment/services/sklearn-svc
pip-sync requirements.txt requirements-dev.txt

# 3. Run the service locally
uvicorn app.main:app --reload
```

## Critical Version Pins from Modeling Phase

When pinning service dependencies, the **inference framework versions must match the training versions** to avoid pickle/state-dict compatibility warnings or silent numerical drift. Pinned values discovered during Phase 0 Step 0.4 artifact verification:

| Service | Library | Training version | Notes |
|---------|---------|------------------|-------|
| `sklearn-svc` | scikit-learn | **1.8.0** | Phase 0 verification flagged `InconsistentVersionWarning` when loading with 1.7.2; pin to 1.8.0 in `pyproject.toml` |
| `pt-svc` | torch | 2.5.1+cu121 (CPU wheel for inference) | Production containers use `torch==2.5.1+cpu`; no CUDA in deployment images |
| `tf-svc` | tensorflow | 2.21.0 | Same as WSL2 modeling-phase env; full GPU not required for inference |
| `tf-svc` | sentencepiece | (any 0.2+) | Tokenizer model is version-agnostic; minor versions OK |

When the `pyproject.toml` for each service is written in Phase 1+, these exact pins go in `dependencies = [...]`.

## Migration Path to uv (Future)

If we decide to switch to `uv` later for speed:
- `uv pip compile pyproject.toml` reads the same source format as `pip-compile`
- `uv pip sync requirements.txt` works identically to `pip-sync`
- The `pyproject.toml` doesn't change at all
- Lock-file format compatible (both produce standard pip-format requirements.txt)

This is intentional - the source-of-truth format (`pyproject.toml`) is tool-agnostic. Only the compile/sync commands would change.
