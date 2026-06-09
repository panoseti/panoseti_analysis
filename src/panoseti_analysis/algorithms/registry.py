"""Model registry — ``@register_model("id")`` + ``get_model`` / ``build_model``.

Layer A pure (torch only; no ray/zarr/io).  Any module that imports this file can
register a model class.  The registry is populated at import time by decorating the
class definition:

    @register_model("cloud_detector_v2")
    class CloudDetectionV2(nn.Module): ...

    # Instantiate without hard-coding the class:
    model = build_model("cloud_detector_v2")
    # or with constructor kwargs:
    model = build_model("beta_vae", latent_dim=16)
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(model_id: str) -> Any:
    """Class decorator that registers an ``nn.Module`` subclass under ``model_id``."""

    def _decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if not issubclass(cls, nn.Module):
            raise TypeError(f"@register_model requires an nn.Module subclass, got {cls}")
        if model_id in _REGISTRY:
            raise ValueError(
                f"Model id {model_id!r} is already registered to {_REGISTRY[model_id].__name__}"
            )
        _REGISTRY[model_id] = cls
        return cls

    return _decorator


def get_model(model_id: str) -> type[nn.Module]:
    """Return the registered ``nn.Module`` class for ``model_id``.

    Raises ``KeyError`` with a helpful message listing available ids.
    """
    if model_id not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise KeyError(f"Unknown model id {model_id!r}. Registered ids: {available}")
    return _REGISTRY[model_id]


def build_model(model_id: str, **kwargs: Any) -> nn.Module:
    """Instantiate the registered model class with optional constructor kwargs."""
    return get_model(model_id)(**kwargs)


def registered_ids() -> list[str]:
    """Return a sorted list of all registered model ids."""
    return sorted(_REGISTRY)
