"""
Unit tests for pca_loader._resolve_artifact_dir.

The helper translates a registered model's source URI (an absolute path
captured at promotion time on the host that ran promote_to_registry.py)
into a local pathlib.Path. When MLFLOW_ARTIFACT_ROOT_OVERRIDE is set,
the helper re-anchors the URI's artifact path under that override root -
the relocation hook that lets a container read a registry whose URIs
were captured on a different host.

These tests exercise the helper directly: pure URI manipulation, no
MLflow setup or model artifacts required.
"""

from pathlib import Path

import pytest

from app.services.pca_loader import (
    ARTIFACT_ROOT_OVERRIDE_ENV,
    _file_uri_to_path,
    _resolve_artifact_dir,
)

# Sample URIs covering the two host-side shapes promote_to_registry.py
# can produce. Windows hosts produce file:///C:/... (drive letter retained
# inside the URI). POSIX hosts produce file:///home/... (no drive letter).
WINDOWS_SOURCE_URI = (
    "file:///C:/workspaces/ml-framework-comparisons/"
    "deployment/mlruns/abc123/artifacts"
)
POSIX_SOURCE_URI = "file:///home/user/deployment/mlruns/abc123/artifacts"

# Overlay path the container would advertise to the loader. Arbitrary;
# the helper only cares about the prefix substitution, not the value.
OVERRIDE_ROOT = "/srv/mlflow"


def test_no_override_matches_file_uri_to_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Env var unset -> _resolve_artifact_dir returns the same Path as the
    bare _file_uri_to_path call. Host-side dev (where the same machine
    writes and reads the registry) needs no relocation.
    """
    monkeypatch.delenv(ARTIFACT_ROOT_OVERRIDE_ENV, raising=False)
    assert _resolve_artifact_dir(WINDOWS_SOURCE_URI) == _file_uri_to_path(
        WINDOWS_SOURCE_URI
    )
    assert _resolve_artifact_dir(POSIX_SOURCE_URI) == _file_uri_to_path(
        POSIX_SOURCE_URI
    )


def test_override_reanchors_windows_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Env var set + Windows-shaped source URI -> path is re-anchored under
    the override root. The mlruns/<run-id>/artifacts tail is preserved
    so the loader still finds the correct version's artifact directory
    on the runtime host.
    """
    monkeypatch.setenv(ARTIFACT_ROOT_OVERRIDE_ENV, OVERRIDE_ROOT)
    result = _resolve_artifact_dir(WINDOWS_SOURCE_URI)
    assert result == Path(OVERRIDE_ROOT) / "mlruns/abc123/artifacts"


def test_override_reanchors_posix_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Env var set + POSIX-shaped source URI -> same re-anchoring behavior.
    The override logic is platform-agnostic; only the /mlruns/ segment
    matters for the relocation.
    """
    monkeypatch.setenv(ARTIFACT_ROOT_OVERRIDE_ENV, OVERRIDE_ROOT)
    result = _resolve_artifact_dir(POSIX_SOURCE_URI)
    assert result == Path(OVERRIDE_ROOT) / "mlruns/abc123/artifacts"


def test_override_without_mlruns_segment_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Env var set + URI missing /mlruns/ segment -> RuntimeError. Surfaces
    unknown registry layouts at boot rather than silently producing a
    wrong path the loader would try to open and then fail with a less
    informative FileNotFoundError later.
    """
    monkeypatch.setenv(ARTIFACT_ROOT_OVERRIDE_ENV, OVERRIDE_ROOT)
    bad_uri = "file:///some/other/path/that/has/no/standard/segment"
    with pytest.raises(RuntimeError, match="MLFLOW_ARTIFACT_ROOT_OVERRIDE"):
        _resolve_artifact_dir(bad_uri)
