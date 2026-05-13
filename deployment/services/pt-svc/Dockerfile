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

# COPY only the lockfile here, NOT the app code. Layer-caching discipline: pip
# install is the slow step (~120s for pt-svc - torch CPU wheel is ~200 MB),
# so we want its layer to invalidate ONLY when requirements.txt changes.
COPY requirements.txt .

# Build the venv from the hash-locked lockfile. pt-svc pins torch==2.5.1+cpu,
# which lives at https://download.pytorch.org/whl/cpu rather than PyPI;
# pip-compile embedded the matching `--extra-index-url` directive at the top
# of requirements.txt, so pip reads it from there - no flag needed here.
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

# Non-root `app` user at fixed UID/GID 1000. Predictable for volume-mount
# permission interop with the host; nologin shell + create-home gives the
# service its own $HOME (uvicorn / mlflow write transient state there) while
# blocking interactive logins as a defense-in-depth measure.
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Copy the prebuilt venv with ownership set during the copy (avoids a
# duplicate-layer `chown -R` that would double the venv's disk footprint).
COPY --from=builder --chown=app:app /opt/venv /opt/venv

# App code last so the cache invalidates only when the application changes.
COPY --chown=app:app app/ ./app/

USER app

# pt-svc port allocation: sklearn-svc=8001, pt-svc=8002, tf-svc=8003.
# EXPOSE is metadata only; `docker run -p` or compose `ports:` does the actual publish.
EXPOSE 8002

# HEALTHCHECK against /health every 30s. 10s start-period is conservative
# even for pt-svc's three sequential loaders (DNN ~200ms + GAN ~50ms + Q-table
# ~50ms; ~300ms wall-clock total). stdlib urllib avoids dragging curl into
# the slim image just for a probe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8002/health').status == 200 else 1)"

# JSON-array (exec) form so SIGTERM from `docker stop` reaches uvicorn
# directly for a graceful shutdown - the shell form would interpose /bin/sh
# and drop the signal. --host 0.0.0.0 binds all interfaces inside the
# container so the host's `docker run -p 8002:8002` mapping can reach the
# service; the 127.0.0.1 default would be unreachable from outside.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
