# syntax=docker/dockerfile:1.7

# ============================================================================
# STAGE 1: BUILDER - install pip-locked deps into an isolated venv
# ============================================================================
# Multi-stage pattern: build-time tooling and pip metadata stay in this stage
# and are discarded; only /opt/venv carries forward to runtime.
FROM python:3.12.2-slim AS builder

# Standard production-Python env vars: no .pyc on disk, unbuffered stdout for
# real-time `docker logs`, no pip wheel cache (the stage is thrown away), and
# skip the noisy "you should upgrade pip" self-check.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy lockfile only; pip install layer caches on the file's hash so app code
# edits don't trigger reinstall. The TensorFlow + sentencepiece install is
# the slowest of the three services (~150-200s; TF wheel is ~600 MB by
# itself) - getting the cache right matters most here.
COPY requirements.txt .

# Build the venv from the hash-locked lockfile. Linux PyPI ships tensorflow
# CPU-only by default from TF 2.15 onward (GPU lives behind the
# `tensorflow[and-cuda]` extra we don't reference), so no nvidia-* CUDA
# packages enter the venv - one less reason for image bloat.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --require-hashes -r requirements.txt


# ============================================================================
# STAGE 2: RUNTIME - the image we ship
# ============================================================================
FROM python:3.12.2-slim AS runtime

# PATH munging puts /opt/venv/bin first so `uvicorn` and `python` resolve to
# the venv's binaries, not the system Python.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Non-root `app` user at fixed UID/GID 1000 (predictable for volume-mount
# permission interop with the host). nologin shell + create-home: the
# service runs as `app` but interactive logins go nowhere; uvicorn / TF
# still get a $HOME for transient state.
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Copy the prebuilt venv with ownership set during the copy (avoids a
# duplicate-layer `chown -R` that would double the venv's disk footprint).
COPY --from=builder --chown=app:app /opt/venv /opt/venv

# App code last so the cache invalidates only when the application changes.
COPY --chown=app:app app/ ./app/

USER app

# tf-svc port allocation: sklearn-svc=8001, pt-svc=8002, tf-svc=8003.
# EXPOSE is metadata only; the actual publish is `docker run -p` or compose.
EXPOSE 8003

# HEALTHCHECK against /health every 30s. start-period of 15s (sklearn-svc
# and pt-svc use 10s) adds headroom for the Transformer's lifespan: dummy-
# forward + load_weights + warm-up forward all run through TF's eager-mode
# op compiler on a cold-cache container. Host-side wall-clock is ~2.5s,
# but a freshly-pulled container with no warm TF caches can drift several
# seconds longer. stdlib urllib keeps the slim image free of a curl dep.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8003/health').status == 200 else 1)"

# JSON-array (exec) form so SIGTERM from `docker stop` reaches uvicorn
# directly for a graceful shutdown - the shell form would interpose /bin/sh
# and drop the signal. --host 0.0.0.0 binds all interfaces inside the
# container so the host's `docker run -p 8003:8003` mapping can reach the
# service; the 127.0.0.1 default would be unreachable from outside.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
