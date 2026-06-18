"""Thin wrappers over pypff's read-side API (the only module that imports pypff)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pypff import PanosetiRun

if TYPE_CHECKING:
    from pypff.io2 import PFFSequence


def read_pff_run(obs_dir: str | Path) -> PanosetiRun:
    """Open a ``.pffd`` observation directory as a pypff ``PanosetiRun``."""
    return PanosetiRun(str(obs_dir))


def open_pff_product(obs_dir: str | Path, dp: str, module: int | str) -> PFFSequence:
    """Open a single (data_product, module) PFFSequence from a .pffd observation dir.

    Args:
        obs_dir: Path to the ``.pffd`` observation directory.
        dp:      Full data_product name, e.g. ``"dp_img16.bpp_2.module_1"``.
                 Passed directly to ``PanosetiRun.get_product(dp)``.
        module:  Module ID (int) or full dp string (str) — present for API symmetry
                 but the lookup is driven entirely by ``dp``.

    Returns:
        The ``PFFSequence`` for the requested (data_product, module) pair.

    Raises:
        KeyError: if ``dp`` is not found in the run.
    """
    run = read_pff_run(obs_dir)
    return run.get_product(dp)


def read_hk(obs_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Parse the run's ``hk.pff`` head-node telemetry into ``{hashset: {field: ndarray}}``.

    Returns ``{}`` when the run has no housekeeping file.
    """
    return read_pff_run(obs_dir).get_hk()
