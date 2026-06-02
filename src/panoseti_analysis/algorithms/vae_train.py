"""BetaVAE training: model-specific hooks + a notebook-friendly wrapper.

Mirrors :mod:`panoseti_analysis.algorithms.cloud_train`, but for the unsupervised PH
anomaly detector. The epoch loop lives in :mod:`panoseti_analysis.algorithms.training`;
here we supply the VAE-specific pieces — loss, validation metrics, optimizer — plus
:func:`fit_vae` for plain (non-Ray) training in notebooks. The Ray training adapter reuses
the *same* hooks so there is one source of truth for how this model trains.

Reconstruction error (per-sample MSE) is the anomaly score: high-error frames are the
candidate anomalies. :func:`vae_reconstruction_errors` produces those scores for reporting.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from panoseti_analysis.algorithms.ph_vae import BetaVAE, beta_vae_loss_function
from panoseti_analysis.algorithms.training import TrainResult, fit


def make_vae_loss_fn(
    beta: float, sparsity_weight: float
) -> Callable[[torch.nn.Module, Any], tuple[torch.Tensor, dict[str, float]]]:
    """Build a BetaVAE loss closure over ``(features,)`` batches (TensorDataset of X only)."""

    def vae_loss_fn(model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        (xb,) = batch
        recon, mu, logvar = model(xb)
        loss = beta_vae_loss_function(recon, xb, mu, logvar, beta, sparsity_weight)
        mse = float(F.mse_loss(recon, xb, reduction="mean").item())
        return loss, {"loss": loss.item(), "recon_mse": mse}

    return vae_loss_fn


def make_vae_val_eval(
    X_val: torch.Tensor, beta: float, sparsity_weight: float, device: torch.device
) -> Callable[[torch.nn.Module], dict[str, float]]:
    """Build a validation closure returning ``val_loss`` + reconstruction-error summaries."""

    def val_eval(model: torch.nn.Module) -> dict[str, float]:
        xb = X_val.to(device)
        recon, mu, logvar = model(xb)
        loss = float(beta_vae_loss_function(recon, xb, mu, logvar, beta, sparsity_weight).item())
        per_sample = F.mse_loss(recon, xb, reduction="none").flatten(1).mean(1)
        return {
            "val_loss": loss,
            "val_recon_mse": float(per_sample.mean().item()),
            "val_recon_mse_p99": float(torch.quantile(per_sample, 0.99).item()),
        }

    return val_eval


def vae_reconstruction_errors(
    state_dict: dict[str, torch.Tensor],
    X: torch.Tensor,
    device: torch.device,
    *,
    latent_dim: int = 32,
    hidden_dim: int = 64,
) -> Any:
    """Per-sample reconstruction MSE (the anomaly score) for a saved state_dict → numpy (N,)."""
    model = BetaVAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        recon, _mu, _logvar = model(X.to(device))
        per_sample = F.mse_loss(recon, X.to(device), reduction="none").flatten(1).mean(1)
    return per_sample.cpu().numpy()


def build_vae_optimizer(model: torch.nn.Module, hp: dict[str, Any]) -> torch.optim.Optimizer:
    """Adam optimizer matching the production BetaVAE schedule (no LR scheduler)."""
    lr = float(hp.get("lr", 1e-3))
    weight_decay = float(hp.get("weight_decay", 0.0))
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def fit_vae(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    hp: dict[str, Any],
    *,
    device: torch.device,
    model: torch.nn.Module | None = None,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainResult:
    """Train a :class:`BetaVAE` with plain PyTorch (no Ray) — the notebook-native entry point.

    Composes the VAE hooks with the shared :func:`~panoseti_analysis.algorithms.training.fit`
    engine. The Ray worker uses the same hooks but wires ``fit`` itself (so it can DDP-wrap
    the model first).
    """
    latent_dim = int(hp.get("latent_dim", 32))
    hidden_dim = int(hp.get("hidden_dim", 64))
    beta = float(hp.get("beta", 4e-9))
    sparsity_weight = float(hp.get("sparsity_weight", 0.0))
    batch_size = int(hp.get("batch_size", 256))
    epochs = int(hp.get("epochs", 100))

    if model is None:
        model = BetaVAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    train_loader: DataLoader[Any] = DataLoader(
        TensorDataset(X_train), batch_size=batch_size, shuffle=True, drop_last=True
    )
    return fit(
        model,
        train_loader,
        loss_fn=make_vae_loss_fn(beta, sparsity_weight),
        optimizer=build_vae_optimizer(model, hp),
        epochs=epochs,
        device=device,
        val_eval=make_vae_val_eval(X_val, beta, sparsity_weight, device),
        on_epoch=on_epoch,
        monitor="val_loss",
    )
