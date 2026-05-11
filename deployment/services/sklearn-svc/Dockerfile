# syntax=docker/dockerfile:1.7
# Tells the BuildKit frontend which Dockerfile syntax version to use.
# Pinning gives reproducible parsing across daemon updates - if a future
# Docker version changes default semantics, our build still parses the
# same way it did the day we wrote it.

# ============================================================================
# STAGE 1: BUILDER - install pip-locked deps into an isolated venv
# ============================================================================
# Why a separate builder stage: keeps build-time tooling (compilers, pip
# metadata caches, intermediate downloads) OUT of the final runtime image.
# We only ship the resolved /opt/venv directory across to runtime - nothing
# else from this stage survives.

# `AS builder` names this stage so the runtime stage can reference it via
# `COPY --from=builder ...`. Stage names are arbitrary; "builder" is conventional.
FROM python:3.12.2-slim AS builder

# Environment variables that control Python + pip behavior inside the container.
#   PYTHONDONTWRITEBYTECODE=1: don't write .pyc files. They're useless in an
#     immutable image and waste a few MB.
#   PYTHONUNBUFFERED=1: force stdout/stderr to flush immediately. Without it,
#     Python buffers logs and `docker logs` shows nothing until shutdown.
#   PIP_NO_CACHE_DIR=1: don't cache wheels in ~/.cache/pip. They're discarded
#     when this stage is thrown away anyway; saves ~100 MB transient.
#   PIP_DISABLE_PIP_VERSION_CHECK=1: skip pip's "you should upgrade pip" call.
#     Saves a network round-trip and a noisy log line.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# /build is throwaway - this stage's filesystem doesn't ship to runtime.
WORKDIR /build

# COPY only the lockfile here, NOT the app code. This is the layer-caching
# discipline: pip install is the slow step (~20s for sklearn-svc), so we want
# its layer to invalidate ONLY when requirements.txt changes - not every time
# a Python file in app/ changes. Without this discipline, every code edit
# would trigger a full reinstall of all 80+ deps.
COPY requirements.txt .

# Create an isolated venv at /opt/venv and install all hash-locked deps into it.
#   - python -m venv /opt/venv: stdlib way to create a venv. Resulting venv
#     is a self-contained directory (bin/, lib/) we can COPY across stages.
#   - --require-hashes: enforce that every install matches a hash from the
#     lockfile. Supply-chain security: if PyPI gets compromised and a
#     malicious wheel replaces a legitimate one, the hash mismatch fails
#     the build instead of silently shipping the malicious wheel.
#   - All deps are wheel-only on python:3.12.2-slim x86_64 (sklearn ships
#     manylinux wheels), so no compiler toolchain is needed in this stage.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --require-hashes -r requirements.txt


# ============================================================================
# STAGE 2: RUNTIME - the image we actually ship
# ============================================================================
# Fresh FROM line means a fresh filesystem - the builder stage's /build dir,
# pip metadata, etc. are discarded. We only carry forward what we explicitly COPY.
FROM python:3.12.2-slim AS runtime

# Same env vars as builder PLUS PATH munging so /opt/venv/bin is found first
# (so `uvicorn` and `python` resolve to the venv's binaries, not the system Python).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Create a non-root user + group named `app` with fixed UID/GID 1000.
# Why fixed UID 1000:
#   - Predictable file ownership when the host mounts a volume into the
#     container (e.g., the MLflow registry). If a host file is owned by
#     UID 1000 and the container user is UID 1000, no permission gymnastics.
#   - 1000 is the conventional first-non-system-user UID on Debian/Ubuntu.
# Why --shell /usr/sbin/nologin: the user account is for running the service,
#   not for interactive login. If someone shells into the container as `app`,
#   they get nothing. Defense-in-depth.
# Why --create-home: some Python tools (uvicorn, mlflow) write transient
#   state under $HOME. Without a home dir they error out at startup.
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Copy the prebuilt venv from the builder stage. --chown= sets ownership
# during the copy (cheaper than a separate `chown -R` afterward, which would
# create a duplicate layer doubling the venv's disk footprint in the image).
COPY --from=builder --chown=app:app /opt/venv /opt/venv

# Copy app source code LAST. This is the cache-discipline payoff: when you
# edit app/main.py, only this layer + everything below rebuilds. Everything
# above (the slow pip install) is reused from cache, so rebuilds are seconds
# instead of minutes.
COPY --chown=app:app app/ ./app/

# Drop privileges. From here on, the image runs as UID 1000.
USER app

# Document which port the service listens on. EXPOSE is metadata only - it
# does NOT actually publish the port to the host (that's `docker run -p` or
# the compose `ports:` directive). It's used by `docker inspect` and by
# compose to autowire networking between services.
EXPOSE 8001

# Container-level healthcheck. Docker pings this every 30s; status is visible
# in `docker ps` and used by compose to gate dependent services from starting
# until this one is healthy.
#   --interval=30s: time between checks
#   --timeout=3s: max time per check before counting as failed
#   --start-period=10s: grace window after container start where failures
#     don't count toward the unhealthy threshold (PCA load takes ~1s; this
#     is conservative)
#   --retries=3: 3 consecutive failures before marking unhealthy
# We avoid `curl` (not in slim image) by using stdlib urllib - one less
# system dep to install.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health').status == 200 else 1)"

# Use the JSON-array (exec) form of CMD - this runs uvicorn as PID 1 directly,
# without an intermediary `/bin/sh -c`. Critical because:
#   - SIGTERM from `docker stop` reaches uvicorn directly (graceful shutdown)
#   - One less process layer; faster startup
# The shell form (`CMD uvicorn ...`) wraps in sh, which doesn't forward signals
# by default - container would have to be killed forcefully on shutdown.
#
# --host 0.0.0.0: bind to all interfaces inside the container so the host's
# `docker run -p 8001:8001` port mapping can reach the service. Default is
# 127.0.0.1 which would only be reachable from inside the container itself.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
