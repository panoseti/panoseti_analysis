"""Movie-mode calibration kernel (img8, img16) — Layer A, pure.

Pixel semantics: counts of above-threshold SiPM spikes per integration window (>= 0).
Steps: cast to int32 -> spatial block-median (coarsen over a strided subset) -> upsample
-> subtract -> temporal supermedian -> subtract -> ADC->PE scaling. No I/O.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from panoseti_analysis.config.models import ImgCalibParams


def calibrate_img(ds: xr.Dataset, params: ImgCalibParams) -> xr.Dataset:
    """Calibrate one movie-mode store; return a new L1 Dataset (no I/O)."""
    images = ds["images"].astype("int32")
    height, width = images.shape[1], images.shape[2]
    print("hello")

    subset = images[:: params.frame_stride]
    # xarray's Coarsen reduction methods are generated dynamically (absent from stubs).
    block_coarsen = subset.coarsen(y=params.block_size, x=params.block_size, boundary="trim")
    block_medians = block_coarsen.median().median("time")  # type: ignore[attr-defined]
    upsampled_np = np.repeat(
        np.repeat(block_medians.values, params.block_size, axis=0), params.block_size, axis=1
    )[:height, :width]
    upsampled = xr.DataArray(upsampled_np, dims=("y", "x"))

    spatial_sub = images - upsampled
    supermedian = spatial_sub[:: params.frame_stride].median("time")
    calibrated_int = spatial_sub - supermedian
    calibrated = (calibrated_int / params.adc_to_pe).astype("float32")
    calibrated.name = "median_subtracted"

    global_med_mean = float(upsampled.mean())
    hot = (upsampled > 3.0 * global_med_mean).astype("uint8")
    hot.name = "hot_pixel_mask"
    dead = (upsampled == 0).astype("uint8")
    dead.name = "dead_pixel_mask"

    carried = {k: ds[k] for k in ds.data_vars if k != "images"}
    out = xr.Dataset(
        {"median_subtracted": calibrated, "hot_pixel_mask": hot, "dead_pixel_mask": dead, **carried}
    )
    return out.pano.stamp(
        data_level="L1",
        calibration={"kind": "img", **params.model_dump()},
        carry_from=ds,
    )
