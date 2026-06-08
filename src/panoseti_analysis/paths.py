"""Canonical repo-relative paths.

One place that resolves the repository layout, so code and notebooks never hand-roll
``Path(__file__).parent.parent...`` chains, ``os.path`` joins, or ``../../`` string
literals. Resolved once from this module's own location.

Usage::

    from panoseti_analysis.paths import ASSETS, REPO_ROOT
    model = ASSETS / "models" / "cloud_detector_v1.pt"

In notebooks, prefer importing these over ``sys.path`` hacks: the package is installed
editable (``uv sync``), so ``import panoseti_analysis`` works from any working directory.
"""

from __future__ import annotations

from pathlib import Path

# This file lives at <repo>/src/panoseti_analysis/paths.py, so parents[2] is the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

SRC: Path = REPO_ROOT / "src"
ASSETS: Path = REPO_ROOT / "assets"
MODELS: Path = ASSETS / "models"
RECIPES: Path = REPO_ROOT / "recipes"
ML: Path = REPO_ROOT / "ml"

__all__ = ["ASSETS", "ML", "MODELS", "RECIPES", "REPO_ROOT", "SRC"]
