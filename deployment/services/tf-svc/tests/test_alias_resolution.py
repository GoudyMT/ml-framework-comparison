"""
Unit tests for translation_loader._resolve_alias.

The helper reads the MODEL_ALIAS env var (via ALIAS_OVERRIDE_ENV) and
falls back to the MODEL_ALIAS module constant when the env var is
unset or empty. This is what makes a container point at a non-default
alias (canary / staging) without rebuilding the image.

These tests exercise the helper directly - no MlflowClient mocking,
no model artifacts required.
"""

import pytest

from app.services.translation_loader import (
    ALIAS_OVERRIDE_ENV,
    MODEL_ALIAS,
    _resolve_alias,
)


def test_default_alias_constant_is_production() -> None:
    """MODEL_ALIAS default value is 'production' to match the registry alias set by promote_to_registry.py."""
    assert MODEL_ALIAS == "production"


def test_override_env_constant_is_model_alias() -> None:
    """ALIAS_OVERRIDE_ENV is the literal env-var name 'MODEL_ALIAS'."""
    assert ALIAS_OVERRIDE_ENV == "MODEL_ALIAS"


def test_resolve_alias_uses_default_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env var unset -> _resolve_alias returns MODEL_ALIAS default."""
    monkeypatch.delenv(ALIAS_OVERRIDE_ENV, raising=False)
    assert _resolve_alias() == "production"


def test_resolve_alias_returns_env_value_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env var set -> _resolve_alias returns the env-supplied value."""
    monkeypatch.setenv(ALIAS_OVERRIDE_ENV, "staging")
    assert _resolve_alias() == "staging"


def test_resolve_alias_falls_back_when_env_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Env var set to empty string -> _resolve_alias falls back to default.

    Empty string is falsy in Python, so the `or` short-circuits to the
    MODEL_ALIAS default. This matches the documented behavior: empty
    means "use default", not "use empty-named alias" (which the
    registry would reject anyway).
    """
    monkeypatch.setenv(ALIAS_OVERRIDE_ENV, "")
    assert _resolve_alias() == "production"
