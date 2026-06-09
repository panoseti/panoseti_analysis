"""Tests for the model registry (algorithms/registry.py)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from panoseti_analysis.algorithms.registry import (  # noqa: E402
    build_model,
    get_model,
    register_model,
    registered_ids,
)

# ── fixture: a test-only model that won't pollute the real registry ────────────


@pytest.fixture()
def temp_model_id() -> str:
    """Returns a model id used only in this test session."""
    return "_test_registry_model_42"


def test_register_and_get(temp_model_id: str) -> None:
    """Registering a model class and looking it up by id works."""

    @register_model(temp_model_id)
    class _TinyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    cls = get_model(temp_model_id)
    assert cls is _TinyModel


def test_build_model(temp_model_id: str) -> None:
    """build_model returns an instantiated nn.Module."""
    model = build_model(temp_model_id)
    assert isinstance(model, nn.Module)


def test_registered_ids_contains_known_models() -> None:
    """The two production models are registered at import time."""
    import panoseti_analysis.algorithms.cloud_detector  # noqa: F401

    ids = registered_ids()
    assert "cloud_detector" in ids
    assert "cloud_detector_v2" in ids


def test_duplicate_registration_raises(temp_model_id: str) -> None:
    """Registering the same id twice raises ValueError."""
    with pytest.raises(ValueError, match="already registered"):
        register_model(temp_model_id)(build_model(temp_model_id).__class__)  # type: ignore


def test_unknown_id_raises() -> None:
    """get_model with an unknown id raises KeyError with the available ids listed."""
    with pytest.raises(KeyError, match="Registered ids"):
        get_model("_nonexistent_model_id_xyz")


def test_non_module_raises() -> None:
    """@register_model on a non-nn.Module class raises TypeError."""
    with pytest.raises(TypeError, match=r"nn\.Module subclass"):

        @register_model("_test_bad_class_42")
        class _NotAModule:  # type: ignore[misc]
            pass
