"""
Unit tests for each loader's _resolve_artifact_dir helper.

The pt-svc service hosts three loaders (DNN, GAN, Q-learning), each with
its own private copy of _resolve_artifact_dir. We parametrize the same
test body across all three so any divergence between them gets caught
in one test sweep.

The helper itself is pure URI manipulation: no MLflow setup or model
artifacts required to exercise it.
"""

from pathlib import Path
from types import ModuleType

import pytest

from app.services import dnn_loader, gan_loader, qlearning_loader

# Same test body across all 3 loaders; ids= gives readable pytest output
# (e.g., test_no_override_matches_file_uri_to_path[gan]).
LOADERS = pytest.mark.parametrize(
    "loader",
    [dnn_loader, gan_loader, qlearning_loader],
    ids=["dnn", "gan", "qlearning"],
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


@LOADERS
def test_no_override_matches_file_uri_to_path(
    loader: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Env var unset -> _resolve_artifact_dir returns the same Path as the
    bare _file_uri_to_path call. Host-side dev (where the same machine
    writes and reads the registry) needs no relocation.
    """
    monkeypatch.delenv(loader.ARTIFACT_ROOT_OVERRIDE_ENV, raising=False)
    assert loader._resolve_artifact_dir(WINDOWS_SOURCE_URI) == loader._file_uri_to_path(
        WINDOWS_SOURCE_URI
    )
    assert loader._resolve_artifact_dir(POSIX_SOURCE_URI) == loader._file_uri_to_path(
        POSIX_SOURCE_URI
    )


@LOADERS
def test_override_reanchors_windows_uri(
    loader: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Env var set + Windows-shaped source URI -> path is re-anchored under
    the override root. The mlruns/<run-id>/artifacts tail is preserved
    so the loader still finds the correct version's artifact directory.
    """
    monkeypatch.setenv(loader.ARTIFACT_ROOT_OVERRIDE_ENV, OVERRIDE_ROOT)
    result = loader._resolve_artifact_dir(WINDOWS_SOURCE_URI)
    assert result == Path(OVERRIDE_ROOT) / "mlruns/abc123/artifacts"


@LOADERS
def test_override_reanchors_posix_uri(
    loader: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Env var set + POSIX-shaped source URI -> same re-anchoring behavior.
    The override logic is platform-agnostic; only the /mlruns/ segment
    matters for the relocation.
    """
    monkeypatch.setenv(loader.ARTIFACT_ROOT_OVERRIDE_ENV, OVERRIDE_ROOT)
    result = loader._resolve_artifact_dir(POSIX_SOURCE_URI)
    assert result == Path(OVERRIDE_ROOT) / "mlruns/abc123/artifacts"


@LOADERS
def test_override_without_mlruns_segment_raises(
    loader: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Env var set + URI missing /mlruns/ segment -> RuntimeError. Surfaces
    unknown registry layouts at boot rather than silently producing a
    wrong path the loader would try to open and then fail with a less
    informative FileNotFoundError later.
    """
    monkeypatch.setenv(loader.ARTIFACT_ROOT_OVERRIDE_ENV, OVERRIDE_ROOT)
    bad_uri = "file:///some/other/path/that/has/no/standard/segment"
    with pytest.raises(RuntimeError, match="MLFLOW_ARTIFACT_ROOT_OVERRIDE"):
        loader._resolve_artifact_dir(bad_uri)
