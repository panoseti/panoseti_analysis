"""Tests for algorithms/ph_vae.py — BetaVAE for pulse-height anomaly detection."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from panoseti_analysis.algorithms.ph_vae import BetaVAE, beta_vae_loss_function

_PH_VAE_FILE = Path(__file__).parents[2] / "src/panoseti_analysis/algorithms/ph_vae.py"


@pytest.fixture
def small_model() -> BetaVAE:
    """Small BetaVAE suitable for fast CPU tests."""
    return BetaVAE(latent_dim=8, hidden_dim=16)


@pytest.fixture
def batch_input() -> torch.Tensor:
    return torch.randn(2, 1, 16, 16)


def test_forward_pass_shape(small_model: BetaVAE, batch_input: torch.Tensor) -> None:
    """Forward pass returns (recon, mu, logvar) with correct shapes."""
    recon, mu, logvar = small_model(batch_input)
    assert recon.shape == (2, 1, 16, 16), f"recon shape {recon.shape}"
    assert mu.shape == (2, 8), f"mu shape {mu.shape}"
    assert logvar.shape == (2, 8), f"logvar shape {logvar.shape}"


def test_encode_shape(small_model: BetaVAE, batch_input: torch.Tensor) -> None:
    """encode() returns mu and logvar both of shape (batch, latent_dim)."""
    mu, logvar = small_model.encode(batch_input)
    assert mu.shape == (2, 8), f"mu shape {mu.shape}"
    assert logvar.shape == (2, 8), f"logvar shape {logvar.shape}"


def test_reparameterize_eval_is_mu(small_model: BetaVAE, batch_input: torch.Tensor) -> None:
    """In eval mode reparameterize returns mu exactly."""
    small_model.eval()
    mu, logvar = small_model.encode(batch_input)
    z = small_model.reparameterize(mu, logvar)
    assert torch.equal(z, mu), "reparameterize should return mu in eval mode"


def test_loss_returns_scalar(small_model: BetaVAE, batch_input: torch.Tensor) -> None:
    """beta_vae_loss_function returns a scalar (0-d) tensor."""
    recon, mu, logvar = small_model(batch_input)
    loss = beta_vae_loss_function(recon, batch_input, mu, logvar, beta=4e-9)
    assert loss.ndim == 0, f"loss should be a scalar, got shape {loss.shape}"


def test_loss_is_non_negative(small_model: BetaVAE, batch_input: torch.Tensor) -> None:
    """Loss must be non-negative for typical inputs."""
    recon, mu, logvar = small_model(batch_input)
    loss = beta_vae_loss_function(recon, batch_input, mu, logvar, beta=4e-9)
    assert loss.item() >= 0.0, f"Loss was negative: {loss.item()}"


def test_no_device_placement() -> None:
    """ph_vae.py must not contain hard-coded CUDA device placement (AST check)."""
    src = _PH_VAE_FILE.read_text()
    forbidden = (
        ".cuda(",
        "torch.cuda",
        '.to("cuda"',
        ".to('cuda'",
        'device="cuda"',
        "device='cuda'",
    )
    offenders = [p for p in forbidden if p in src]
    assert not offenders, f"ph_vae.py contains hard-coded CUDA references: {offenders}"
