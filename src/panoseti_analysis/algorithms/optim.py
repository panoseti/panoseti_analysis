"""Config-driven optimizer and LR-scheduler construction (Layer A, pure torch).

Usage::

    optimizer = build_optimizer(model.parameters(), cfg)
    step_schedulers = build_scheduler(optimizer, cfg)
    # step_schedulers(val_loss) — call once per epoch

This generalises ``cloud_train.build_cloud_optimizer`` so a new model only needs a
recipe edit to change the optimizer, not a code change.  The legacy cloud schedule
(AdamW + ExponentialLR + ReduceLROnPlateau) is reproduced exactly when
``optimizer="adamw"``, ``scheduler_exp=True``, and ``scheduler_plateau=True``.

Supported optimizer names (case-insensitive):
    ``"adamw"`` → ``torch.optim.AdamW``
    ``"adam"``  → ``torch.optim.Adam``
    ``"sgd"``   → ``torch.optim.SGD`` (requires ``momentum`` in cfg, defaults 0.9)

Schedulers are opt-in via bool recipe keys:
    ``scheduler_exp``     (default True)  → ExponentialLR(gamma=cfg.gamma)
    ``scheduler_plateau`` (default True)  → ReduceLROnPlateau(patience=5, factor=0.5)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

_OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}


def build_optimizer(
    params: Any,  # model.parameters() or a list of param groups
    cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    """Build an optimizer from a config dict.

    Expected keys (all optional — defaults match the legacy cloud schedule):
        ``optimizer`` (str, default ``"adamw"``), ``lr`` (float, default 1e-3),
        ``weight_decay`` (float, default 1e-5), ``momentum`` (float, default 0.9, SGD only).
    """
    name = str(cfg.get("optimizer", "adamw")).lower()
    if name not in _OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer {name!r}. Supported: {sorted(_OPTIMIZER_REGISTRY)}")
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 1e-5))
    kw: dict[str, Any] = {"lr": lr, "weight_decay": wd}
    if name == "sgd":
        kw["momentum"] = float(cfg.get("momentum", 0.9))
    return _OPTIMIZER_REGISTRY[name](params, **kw)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> Callable[[float], None]:
    """Build a combined scheduler step-function from a config dict.

    Returns a callable ``step_schedulers(val_loss: float)`` that advances all
    enabled schedulers.  Matches the signature expected by ``algorithms.training.fit``.

    Expected keys (all optional):
        ``gamma`` (float, default 0.9),
        ``scheduler_exp`` (bool, default True),
        ``scheduler_plateau`` (bool, default True),
        ``plateau_patience`` (int, default 5),
        ``plateau_factor`` (float, default 0.5).
    """
    gamma = float(cfg.get("gamma", 0.9))
    use_exp = bool(cfg.get("scheduler_exp", True))
    use_plateau = bool(cfg.get("scheduler_plateau", True))

    schedulers: list[Any] = []
    if use_exp:
        schedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma))
    if use_plateau:
        patience = int(cfg.get("plateau_patience", 5))
        factor = float(cfg.get("plateau_factor", 0.5))
        schedulers.append(
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=patience, factor=factor
            )
        )

    def step_schedulers(val_loss: float) -> None:
        for sched in schedulers:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(val_loss)
            else:
                sched.step()

    return step_schedulers
