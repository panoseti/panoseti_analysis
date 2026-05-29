"""Thin wrappers over pypff's read-side API (the only module that imports pypff)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pypff import PanosetiRun


def read_pff_run(obs_dir: str | Path) -> PanosetiRun:
    """Open a ``.pffd`` observation directory as a pypff ``PanosetiRun``."""
    return PanosetiRun(str(obs_dir))


def read_hk(obs_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Parse the run's ``hk.pff`` head-node telemetry into ``{hashset: {field: ndarray}}``.

    Returns ``{}`` when the run has no housekeeping file.
    """
    return read_pff_run(obs_dir).get_hk()
