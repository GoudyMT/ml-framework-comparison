"""
Unit tests for each loader's _resolve_alias helper.

The pt-svc service hosts three loaders (DNN, GAN, Q-learning), each
with its own private copy of _resolve_alias. We parametrize the same
test body across all three so any divergence between them gets caught
in one test sweep.

The helper reads the MODEL_ALIAS env var (via ALIAS_OVERRIDE_ENV) and
falls back to the MODEL_ALIAS module constant when the env var is
unset or empty. This is what makes a container point at a non-default
alias (canary / staging) without rebuilding the image.

These tests exercise the helper directly - no MlflowClient mocking,
no model artifacts required.
"""

from types import ModuleType

import pytest

from app.services import dnn_loader, gan_loader, qlearning_loader

# Same test body across all 3 loaders; ids= gives readable pytest output
# (e.g., test_resolve_alias_returns_env_value_when_set[gan]).
LOADERS = pytest.mark.parametrize(
    "loader",
    [dnn_loader, gan_loader, qlearning_loader],
    ids=["dnn", "gan", "qlearning"],
)


@LOADERS
def test_default_alias_constant_is_production(loader: ModuleType) -> None:
    """MODEL_ALIAS default value is 'production' to match the registry alias set by promote_to_registry.py."""
    assert loader.MODEL_ALIAS == "production"


@LOADERS
def test_override_env_constant_is_model_alias(loader: ModuleType) -> None:
    """ALIAS_OVERRIDE_ENV is the literal env-var name 'MODEL_ALIAS'."""
    assert loader.ALIAS_OVERRIDE_ENV == "MODEL_ALIAS"


@LOADERS
def test_resolve_alias_uses_default_when_env_unset(
    loader: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env var unset -> _resolve_alias returns MODEL_ALIAS default."""
    monkeypatch.delenv(loader.ALIAS_OVERRIDE_ENV, raising=False)
    assert loader._resolve_alias() == "production"


@LOADERS
def test_resolve_alias_returns_env_value_when_set(
    loader: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env var set -> _resolve_alias returns the env-supplied value."""
    monkeypatch.setenv(loader.ALIAS_OVERRIDE_ENV, "staging")
    assert loader._resolve_alias() == "staging"


@LOADERS
def test_resolve_alias_falls_back_when_env_empty(
    loader: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Env var set to empty string -> _resolve_alias falls back to default.

    Empty string is falsy in Python, so the `or` short-circuits to the
    MODEL_ALIAS default. This matches the documented behavior: empty
    means "use default", not "use empty-named alias" (which the
    registry would reject anyway).
    """
    monkeypatch.setenv(loader.ALIAS_OVERRIDE_ENV, "")
    assert loader._resolve_alias() == "production"
