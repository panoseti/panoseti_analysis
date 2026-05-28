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
