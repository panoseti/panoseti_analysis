"""Tests for the BetaVAE training hooks (algorithms/vae_train.py)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from panoseti_analysis.algorithms.ph_vae import BetaVAE  # noqa: E402
from panoseti_analysis.algorithms.training import TrainResult  # noqa: E402
from panoseti_analysis.algorithms.vae_train import (  # noqa: E402
    fit_vae,
    make_vae_loss_fn,
    make_vae_val_eval,
    vae_reconstruction_errors,
)

CPU = torch.device("cpu")


def test_vae_loss_fn_returns_loss_and_logs() -> None:
    model = BetaVAE(latent_dim=8, hidden_dim=16)
    xb = torch.randn(4, 1, 16, 16)
    loss_fn = make_vae_loss_fn(beta=4e-9, sparsity_weight=0.0)
    loss, logs = loss_fn(model, (xb,))
    assert loss.requires_grad
    assert "loss" in logs and "recon_mse" in logs


def test_vae_val_eval_keys() -> None:
    model = BetaVAE(latent_dim=8, hidden_dim=16)
    x_val = torch.randn(6, 1, 16, 16)
    val_eval = make_vae_val_eval(x_val, beta=4e-9, sparsity_weight=0.0, device=CPU)
    model.eval()
    with torch.no_grad():
        m = val_eval(model)
    for key in ("val_loss", "val_recon_mse", "val_recon_mse_p99"):
        assert key in m


def test_fit_vae_smoke() -> None:
    torch.manual_seed(0)
    x_train = torch.randn(32, 1, 16, 16)
    x_val = torch.randn(8, 1, 16, 16)
    hp = {"latent_dim": 8, "hidden_dim": 16, "batch_size": 8, "epochs": 2, "lr": 1e-3}

    result = fit_vae(x_train, x_val, hp, device=CPU)

    assert isinstance(result, TrainResult)
    assert result.history, "expected per-epoch history"
    assert "val_loss" in result.history[-1]
    # best_state must load cleanly into a fresh model.
    BetaVAE(latent_dim=8, hidden_dim=16).load_state_dict(result.best_state)


def test_vae_reconstruction_errors_shape() -> None:
    x = torch.randn(10, 1, 16, 16)
    state = BetaVAE(latent_dim=8, hidden_dim=16).state_dict()
    errors = vae_reconstruction_errors(state, x, CPU, latent_dim=8, hidden_dim=16)
    assert errors.shape == (10,)
    assert (errors >= 0).all()
