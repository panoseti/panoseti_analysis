"""Tests for TrainingScalingConfig and the multi-node DDP guard in runner.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from panoseti_analysis.config.models import TrainingScalingConfig

# ── TrainingScalingConfig validation ─────────────────────────────────────────


def test_single_worker_with_accelerator_type() -> None:
    """num_workers=1 with accelerator_type parses cleanly."""
    cfg = TrainingScalingConfig.model_validate({"num_workers": 1, "accelerator_type": "A6000"})
    assert cfg.num_workers == 1
    assert cfg.accelerator_type == "A6000"
    assert cfg.allow_multinode is False


def test_defaults() -> None:
    """Empty dict uses safe defaults."""
    cfg = TrainingScalingConfig.model_validate({})
    assert cfg.num_workers == 1
    assert cfg.accelerator_type is None
    assert cfg.allow_multinode is False


def test_extra_keys_are_ignored() -> None:
    """_IgnoreExtra base: recipe keys we don't own are silently dropped."""
    cfg = TrainingScalingConfig.model_validate(
        {"num_workers": 1, "accelerator_type": "A6000", "wandb_entity": "should-be-ignored"}
    )
    assert cfg.num_workers == 1


def test_allow_multinode_flag() -> None:
    """allow_multinode: true is accepted."""
    cfg = TrainingScalingConfig.model_validate({"num_workers": 2, "allow_multinode": True})
    assert cfg.num_workers == 2
    assert cfg.allow_multinode is True


# ── _build_scaling_and_run_cfg guard ─────────────────────────────────────────


def _call_build(scaling_cfg: dict, tmp_path: Path) -> tuple:
    """Call _build_scaling_and_run_cfg with Ray and torch.cuda mocked out.

    We patch at the source level inside the function body: ``ray.train`` is only
    imported *inside* ``_build_scaling_and_run_cfg`` (local import), so we keep a
    persistent entry in sys.modules for the duration of each call and remove it
    afterwards to leave no pollution between tests.
    """
    import sys

    import panoseti_analysis.adapters.ml.runner as runner_mod

    mock_scaling_cls = MagicMock(name="ScalingConfig")
    mock_run_cfg_cls = MagicMock(name="RunConfig")
    mock_ray_train = MagicMock()
    mock_ray_train.ScalingConfig = mock_scaling_cls
    mock_ray_train.RunConfig = mock_run_cfg_cls

    # Inject fake ray.train *before* calling so the local import inside the function
    # picks it up. Remove afterwards to keep sys.modules clean.
    prev_ray = sys.modules.get("ray")
    prev_ray_train = sys.modules.get("ray.train")
    sys.modules["ray"] = MagicMock()
    sys.modules["ray.train"] = mock_ray_train
    try:
        with patch.object(runner_mod.torch.cuda, "is_available", return_value=False):
            return runner_mod._build_scaling_and_run_cfg(scaling_cfg, tmp_path)
    finally:
        if prev_ray is None:
            sys.modules.pop("ray", None)
        else:
            sys.modules["ray"] = prev_ray
        if prev_ray_train is None:
            sys.modules.pop("ray.train", None)
        else:
            sys.modules["ray.train"] = prev_ray_train


def test_num_workers_gt1_without_allow_multinode_raises(tmp_path: Path) -> None:
    """num_workers=2 without allow_multinode must raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="allow_multinode"):
        _call_build({"num_workers": 2}, tmp_path)


def test_num_workers_gt1_with_allow_multinode_does_not_raise(tmp_path: Path) -> None:
    """num_workers=2 + allow_multinode: true must NOT raise."""
    _call_build({"num_workers": 2, "allow_multinode": True}, tmp_path)


def test_num_workers_1_does_not_raise(tmp_path: Path) -> None:
    """num_workers=1 (the default for cloud recipes) must NOT raise."""
    _call_build({"num_workers": 1, "accelerator_type": "A6000"}, tmp_path)
