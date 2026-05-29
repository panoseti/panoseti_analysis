"""Quick-look sidecar writers (PNG + JSON). I/O side of quicklook.

The pure statistics (``Stats``/``summarize``) live in ``algorithms/quicklook.py``;
these functions only render/serialize and are called from adapters, never kernels.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def make_preview_png(da: xr.DataArray, out_path: str | Path, *, title: str) -> None:
    """Write a two-panel quick-look PNG: mean image + mean-pixel time series."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(da.compute())  # (T, H, W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_img = np.nanmean(arr, axis=0)
        mean_ts = np.nanmean(arr, axis=(1, 2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=10)

    im = axes[0].imshow(mean_img, origin="lower", aspect="equal", cmap="viridis")
    axes[0].set_title("Mean image (all frames)")
    axes[0].set_xlabel("x [pix]")
    axes[0].set_ylabel("y [pix]")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].plot(np.where(np.isfinite(mean_ts), mean_ts, np.nan), lw=0.8)
    axes[1].set_title("Mean pixel value vs frame")
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_summary_json(payload: dict[str, Any], out_path: str | Path) -> None:
    """Write a quick-look summary JSON (e.g. ``Stats.to_json()`` merged with extras)."""
    Path(out_path).write_text(json.dumps(payload, indent=2))

def generate_cloud_quicklook(ds_l2: xr.Dataset, out_path: str | Path) -> None:
    """Generate a diagnostic quicklook for cloud detection results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "T_l2" not in ds_l2.sizes or ds_l2.sizes["T_l2"] == 0:
        return

    scores = ds_l2["cloud_score"].values
    t_ns = ds_l2["unix_t_ns"].values

    # Find frame with max cloud score
    max_idx = np.argmax(scores)
    max_score = scores[max_idx]

    fft = ds_l2["feature_raw_fft"].values[max_idx]
    deriv_fft = ds_l2["feature_deriv_fft"].values[max_idx]

    fig = plt.figure(figsize=(10, 8))

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])

    ax_ts = fig.add_subplot(gs[0, :])
    t_s = (t_ns - t_ns[0]) / 1e9

    ax_ts.scatter(t_s, scores, c=scores, cmap="viridis", vmin=0, vmax=1)
    ax_ts.plot(t_s, scores, alpha=0.3, color="gray")
    ax_ts.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold=0.5")
    ax_ts.set_title("Cloud Detection Score vs Time")
    ax_ts.set_xlabel("Time (s relative to observation start)")
    ax_ts.set_ylabel("Cloud Score (0=Clear, 1=Cloud)")
    ax_ts.set_ylim(-0.05, 1.05)
    ax_ts.legend()

    ax_ts.scatter(t_s[max_idx], max_score, color="red", s=100, marker="*")

    ax_fft = fig.add_subplot(gs[1, 0])
    im1 = ax_fft.imshow(fft, cmap="viridis")
    ax_fft.set_title(f"Raw FFT @ t={t_s[max_idx]:.1f}s\nScore: {max_score:.2f}")
    ax_fft.axis('off')
    fig.colorbar(im1, ax=ax_fft, fraction=0.046, pad=0.04)

    ax_deriv = fig.add_subplot(gs[1, 1])
    im2 = ax_deriv.imshow(deriv_fft, cmap="viridis")
    ax_deriv.set_title(f"Deriv FFT @ t={t_s[max_idx]:.1f}s")
    ax_deriv.axis('off')
    fig.colorbar(im2, ax=ax_deriv, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
