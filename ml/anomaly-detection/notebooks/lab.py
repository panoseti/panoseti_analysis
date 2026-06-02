"""PH anomaly-detection (BetaVAE) research bench — the *volatile* layer.

Edit freely from the notebooks (``%autoreload 2``). Composes the *trusted, tested* package
layer and never reimplements it:

- epoch loop ......... ``panoseti_analysis.algorithms.training.fit``
- model + hooks ...... ``panoseti_analysis.algorithms.vae_train`` / ``.ph_vae``
- data loading ....... ``panoseti_analysis.adapters.ml.data.load_unlabeled_feature_cache``
- repo paths ......... ``panoseti_analysis.paths``

The VAE is unsupervised: per-sample reconstruction MSE is the anomaly score. Keep durable,
reusable logic in the package (guarded by tests); keep experiments here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from panoseti_analysis.adapters.ml.data import load_unlabeled_feature_cache
from panoseti_analysis.algorithms.ph_vae import BetaVAE
from panoseti_analysis.algorithms.training import TrainResult
from panoseti_analysis.algorithms.vae_train import fit_vae, vae_reconstruction_errors


def get_device() -> torch.device:
    """Pick the best available device (never hard-coded cuda)."""
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def load_features(
    feature_cache: str | Path, *, val_prop: float = 0.1, seed: int = 1984
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(X_train, X_val)`` from a PH feature cache (seeded carve-out if it has no split)."""
    return load_unlabeled_feature_cache(Path(feature_cache), val_prop=val_prop, seed=seed)


def quick_train_vae(
    feature_cache: str | Path,
    hp: dict[str, Any] | None = None,
    *,
    device: torch.device | None = None,
    on_epoch: Any = None,
) -> tuple[BetaVAE, TrainResult]:
    """Load a PH feature cache and train a BetaVAE — returns (best model, result)."""
    device = device or get_device()
    defaults = {
        "latent_dim": 32,
        "hidden_dim": 64,
        "beta": 4e-9,
        "sparsity_weight": 0.0,
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 30,
    }
    defaults.update(hp or {})

    x_train, x_val = load_features(feature_cache)
    result = fit_vae(x_train, x_val, defaults, device=device, on_epoch=on_epoch)
    model = BetaVAE(latent_dim=int(defaults["latent_dim"]), hidden_dim=int(defaults["hidden_dim"]))
    model.load_state_dict(result.best_state)
    model.to(device).eval()
    return model, result


def plot_history(result: TrainResult, keys: tuple[str, ...] = ("train_loss", "val_loss")) -> None:
    """Line-plot selected metrics across epochs from a TrainResult.history."""
    epochs = [int(h["epoch"]) for h in result.history]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for key in keys:
        ys = [h.get(key) for h in result.history]
        if any(y is not None for y in ys):
            ax.plot(epochs, ys, marker=".", label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def reconstruction_errors(
    model: BetaVAE, X: torch.Tensor, device: torch.device | None = None
) -> np.ndarray:
    """Per-sample reconstruction MSE (the anomaly score) for X → numpy (N,)."""
    device = device or get_device()
    return vae_reconstruction_errors(
        model.state_dict(), X, device, latent_dim=model.latent_dim, hidden_dim=model._hidden_dim
    )


def plot_error_hist(model: BetaVAE, X: torch.Tensor, device: torch.device | None = None) -> None:
    """Histogram of per-sample reconstruction error — the anomaly-score distribution."""
    errors = reconstruction_errors(model, X, device)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(errors, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("per-sample reconstruction MSE")
    ax.set_ylabel("count")
    ax.set_title("Anomaly-score distribution")
    fig.tight_layout()
    plt.show()


def plot_reconstructions(
    model: BetaVAE, X: torch.Tensor, n: int = 8, device: torch.device | None = None
) -> None:
    """Show n input frames (top) vs their reconstructions (bottom)."""
    device = device or get_device()
    xb = X[:n].to(device)
    model.eval()
    with torch.no_grad():
        recon, _mu, _logvar = model(xb)
    xb_np = xb.cpu().numpy()
    recon_np = recon.cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.2))
    for j in range(n):
        axes[0, j].imshow(xb_np[j, 0], cmap="viridis")
        axes[1, j].imshow(recon_np[j, 0], cmap="viridis")
        for row in (0, 1):
            axes[row, j].axis("off")
    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("recon")
    fig.suptitle("Input (top) vs reconstruction (bottom)")
    fig.tight_layout()
    plt.show()
