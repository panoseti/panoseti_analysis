"""Shared ML adapter utilities (Layer B).

The common, framework-aware glue every model reuses — feature-cache loading
(:mod:`.data`), the generic Ray ``TorchTrainer`` runner (:mod:`.runner`), and experiment
tracking (:mod:`.tracking`). Per-model adapters (``adapters/ray/train_cloud.py``,
``adapters/ray/train_vae.py``) wire their model + hooks on top of these, so adding a new
model does not mean copy-pasting training boilerplate.
"""
