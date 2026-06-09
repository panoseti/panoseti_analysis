"""Tests for the config-driven optimizer + scheduler (algorithms/optim.py)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from panoseti_analysis.algorithms.optim import build_optimizer, build_scheduler  # noqa: E402


@pytest.fixture()
def tiny_model() -> nn.Module:
    return nn.Linear(4, 2)


def test_build_adamw(tiny_model: nn.Module) -> None:
    opt = build_optimizer(tiny_model.parameters(), {"optimizer": "adamw", "lr": 1e-3})
    assert isinstance(opt, torch.optim.AdamW)


def test_build_adam(tiny_model: nn.Module) -> None:
    opt = build_optimizer(tiny_model.parameters(), {"optimizer": "adam"})
    assert isinstance(opt, torch.optim.Adam)


def test_build_sgd(tiny_model: nn.Module) -> None:
    opt = build_optimizer(tiny_model.parameters(), {"optimizer": "sgd", "momentum": 0.9})
    assert isinstance(opt, torch.optim.SGD)


def test_unknown_optimizer_raises(tiny_model: nn.Module) -> None:
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(tiny_model.parameters(), {"optimizer": "rmsprop_xyz"})


def test_build_scheduler_callable(tiny_model: nn.Module) -> None:
    opt = build_optimizer(tiny_model.parameters(), {})
    step = build_scheduler(opt, {"gamma": 0.9})
    assert callable(step)
    step(0.5)  # must not raise


def test_no_schedulers(tiny_model: nn.Module) -> None:
    """scheduler_exp=False + scheduler_plateau=False → no-op callable."""
    opt = build_optimizer(tiny_model.parameters(), {})
    step = build_scheduler(opt, {"scheduler_exp": False, "scheduler_plateau": False})
    step(0.5)  # must not raise


def test_default_matches_legacy_cloud(tiny_model: nn.Module) -> None:
    """Default config (no keys) must produce an AdamW + schedulers identical to the legacy
    build_cloud_optimizer behaviour."""
    from panoseti_analysis.algorithms.cloud_train import build_cloud_optimizer

    hp = {"lr": 1e-3, "weight_decay": 1e-5, "gamma": 0.9}
    opt_new = build_optimizer(tiny_model.parameters(), hp)
    opt_legacy, _ = build_cloud_optimizer(tiny_model, hp)
    # Same optimizer class and same hyper-params
    assert type(opt_new) is type(opt_legacy)
    assert opt_new.param_groups[0]["lr"] == opt_legacy.param_groups[0]["lr"]
    assert opt_new.param_groups[0]["weight_decay"] == opt_legacy.param_groups[0]["weight_decay"]
