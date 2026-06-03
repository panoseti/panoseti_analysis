"""<Your model name> research bench — the *volatile* layer.

Copy this directory to ``ml/<your-model>/`` and edit freely with ``%autoreload 2``.

HOW TO USE:
1. Register your model class:
       from panoseti_analysis.algorithms.registry import register_model
       @register_model("my_model_v1")
       class MyModel(nn.Module): ...
   (Put it in ``src/panoseti_analysis/algorithms/<yourfile>.py`` for a
   reliable tested module, or keep it here for early R&D.)

2. Set MODEL_ARCH below to your registry id.

3. Call quick_train(FEATURE_CACHE, {"arch": MODEL_ARCH, "epochs": 10}).

4. When the model is mature, promote it:
   - Register the class in algorithms/ (if not done yet).
   - Add a recipe in recipes/<name>.yml.
   - Run pa-train-cloud (or pa-tune-cloud for a sweep).

Promotion gradient:
  notebook/lab.py (volatile) → algorithms/ + recipe (tested) → shared util (general)

Package layer (do NOT reimplement here):
- epoch loop ......... ``panoseti_analysis.algorithms.training.fit``
- model registry ..... ``panoseti_analysis.algorithms.registry``
- data loading ....... ``panoseti_analysis.adapters.ml.data``
- figure builders .... ``panoseti_analysis.adapters.ml.viz``
- shared bench ....... ``panoseti_analysis.adapters.ml.bench``
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

# ── shared batteries-included helpers ─────────────────────────────────────────
from panoseti_analysis.adapters.ml.bench import quick_train

# ── data loading ──────────────────────────────────────────────────────────────
# ── model registry ─────────────────────────────────────────────────────────────
# ── training engine ───────────────────────────────────────────────────────────
from panoseti_analysis.algorithms.training import TrainResult

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Define and register your model.
#   Move this class to src/panoseti_analysis/algorithms/ once it's stable.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ARCH = "my_model_v1"  # ← change this to your registry id

# Uncomment and fill in your model:
#
# @register_model(MODEL_ARCH)
# class MyModel(nn.Module):
#     input_shape = (2, 32, 32)  # match your feature shape
#
#     def __init__(self) -> None:
#         super().__init__()
#         # your layers here
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # your forward pass here
#         ...


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Point at your feature cache.
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_CACHE = "/mnt/beegfs/features/features.<hash>.zarr"  # ← your path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Quick train (run this cell in the notebook).
# ─────────────────────────────────────────────────────────────────────────────


def run_quick_train(epochs: int = 10, **hp_overrides: Any) -> tuple[nn.Module, TrainResult]:
    """Train MODEL_ARCH for a quick smoke-test.  Returns (best model, result)."""
    hp = {"arch": MODEL_ARCH, "epochs": epochs, **hp_overrides}
    return quick_train(FEATURE_CACHE, hp)
