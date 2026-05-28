"""Shared utilities for the PanoSETI Zarr calibration pipeline."""
from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
import zarr
from zarr.codecs import ZstdCodec


def open_l0(store: Path) -> xr.Dataset:
    """Open an L0 Zarr store produced by pypff.zarr.convert_run."""
    ds = xr.open_zarr(str(store), consolidated=False, chunks={})
    assert ds["unix_t_ns"].dtype == np.int64, "unix_t_ns must be int64"
    return ds


def infer_kind(ds: xr.Dataset) -> Literal["ph", "img"]:
    """Return 'ph' for pulse-height products, 'img' for image-mode products."""
    dp: str = ds.attrs.get("data_product", "")
    if dp.startswith("ph"):
        return "ph"
    if dp.startswith("img"):
        return "img"
    raise ValueError(f"Cannot infer kind from data_product={dp!r}")


def write_l1(
    arrays: dict[str, xr.DataArray],
    out_path: Path,
    *,
    attrs: dict,
    codec: str = "zstd",
    level: int = 5,
) -> tuple[int, float]:
    """
    Write L1 arrays to a Zarr v3 store. Returns (bytes_written, compression_ratio_vs_images).

    arrays: mapping of variable name → DataArray (must share the 'time' dimension)
    attrs: root-level attributes to embed in the store
    """
    if out_path.exists():
        shutil.rmtree(out_path)

    compressor = ZstdCodec(level=level)
    encoding = {name: {"compressors": [compressor]} for name in arrays}

    ds = xr.Dataset(arrays)
    ds.attrs.update(attrs)
    ds.to_zarr(
        str(out_path),
        mode="w",
        zarr_format=3,
        consolidated=False,
        encoding=encoding,
    )

    total = sum(
        f.stat().st_size for f in out_path.rglob("*") if f.is_file()
    )
    return total, 0.0  # ratio computed by caller if desired


def frame_rate_hz(ds: xr.Dataset) -> float:
    """Estimate frame rate from the median inter-frame interval in unix_t_ns."""
    ts = ds["unix_t_ns"].values
    if len(ts) < 2:
        return float("nan")
    diffs = np.diff(ts)
    median_ns = float(np.median(diffs[diffs > 0]))
    return 1e9 / median_ns if median_ns > 0 else float("nan")


@dataclass
class Stats:
    n_frames: int
    n_pixels: int
    mean: float
    std: float
    p01: float
    p99: float
    nonzero_frac: float
    n_hot: int    # pixels with mean > 10× global mean
    n_dead: int   # pixels with mean == 0

    def to_json(self) -> dict:
        return asdict(self)


def summarize(da: xr.DataArray) -> Stats:
    """Compute summary statistics over (T, H, W) DataArray. Loads into memory."""
    arr: np.ndarray = np.asarray(da.compute())
    finite = arr[np.isfinite(arr)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pixel_means = np.nanmean(arr, axis=0).ravel()
        global_mean = float(np.nanmean(pixel_means))
    hot_thresh = 10.0 * global_mean if global_mean > 0 else np.inf
    return Stats(
        n_frames=arr.shape[0],
        n_pixels=arr.shape[1] * arr.shape[2],
        mean=global_mean,
        std=float(np.nanstd(finite)),
        p01=float(np.nanpercentile(finite, 1)) if len(finite) else float("nan"),
        p99=float(np.nanpercentile(finite, 99)) if len(finite) else float("nan"),
        nonzero_frac=float(np.count_nonzero(finite) / finite.size) if finite.size else 0.0,
        n_hot=int(np.sum(pixel_means > hot_thresh)),
        n_dead=int(np.sum(pixel_means == 0)),
    )


def make_preview_png(da: xr.DataArray, out_path: Path, *, title: str) -> None:
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
    axes[1].set_ylabel("ADC counts")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_summary_json(stats: Stats, out_path: Path, extra: dict | None = None) -> None:
    payload = stats.to_json()
    if extra:
        payload.update(extra)
    out_path.write_text(json.dumps(payload, indent=2))
