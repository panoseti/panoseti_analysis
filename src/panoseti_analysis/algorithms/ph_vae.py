"""BetaVAE for PANOSETI pulse-height anomaly detection.

Pure Layer A module: nn.Module definitions + loss function only.
No device placement, no I/O, no Ray.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from panoseti_analysis.algorithms.registry import register_model


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_ch, out_ch),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            UpConv(in_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register_model("beta_vae")
class BetaVAE(nn.Module):
    """BetaVAE for 16x16 single-channel PANOSETI pulse-height frames.

    Input shape: (N, 1, 16, 16).
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self._hidden_dim = hidden_dim

        # Encoder: 1->hidden_dim->hidden_dim*2 with downsampling
        self.encoder = nn.Sequential(
            DownBlock(1, hidden_dim),  # (N, hidden_dim, 8, 8)
            DownBlock(hidden_dim, hidden_dim * 2),  # (N, hidden_dim*2, 4, 4)
        )
        flat_dim = hidden_dim * 2 * 4 * 4  # 4x4 spatial after two MaxPool2d(2)
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            UpBlock(hidden_dim * 2, hidden_dim),  # (N, hidden_dim, 8, 8)
            UpBlock(hidden_dim, hidden_dim // 2),  # (N, hidden_dim//2, 16, 16)
        )
        self.out_conv = nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, hidden_dim: int | None = None) -> torch.Tensor:
        hd = hidden_dim if hidden_dim is not None else self._hidden_dim
        h = self.fc_decode(z)
        h = h.view(h.size(0), hd * 2, 4, 4)
        h = self.decoder(h)
        return self.out_conv(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def beta_vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 4e-9,
    sparsity_weight: float = 0.0,
) -> torch.Tensor:
    """BetaVAE ELBO loss.

    MSE(reduction='mean') + beta * KL(sum) + sparsity_weight * L1(mu).

    Note: beta=4e-9 is asymmetrically small because KL uses sum (not mean)
    over the latent dim, so beta scales the total KL contribution.
    """
    mse = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    sparsity = mu.abs().mean() if sparsity_weight > 0 else torch.tensor(0.0)
    return mse + beta * kl + sparsity_weight * sparsity
