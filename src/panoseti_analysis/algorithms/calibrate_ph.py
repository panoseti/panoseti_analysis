"""Pulse-height calibration kernel (ph256, ph1024) — Layer A, pure.

Pixel semantics: int16 ADC intensity counts (signed; pedestal sits above zero but
fluctuations go below). Steps: optional ph1024 quabo-quadrant fix -> add baseline
offset -> per-pixel pedestal/sigma over a strided subset -> subtract -> n-sigma mask
-> hot/dead masks. Performs no I/O; assumes input already monotonic (the adapter runs
``repair_timestamps`` first).
"""

from __future__ import annotations

import xarray as xr

from panoseti_analysis.config import levels
from panoseti_analysis.config.models import PhCalibParams
from panoseti_analysis.config.versions import (
    CALIBRATION_KEY,
    DATA_LEVEL_KEY,
    PANOSETI_ANALYSIS_STORAGE_VERSION,
    STORAGE_VERSION_KEY,
)


def _fix_ph1024_quabo_order(images: xr.DataArray) -> xr.DataArray:
    """Swap top-right and bottom-left 16x16 quadrants of a 32x32 ph1024 grid.

    ph1024 packs 4 quabos as [q0|q2] / [q1|q3]; the DAQ writes a different order, so
    the top-right and bottom-left sub-arrays must be exchanged for correct spatial layout.
    """
    arr = images.values.copy()  # (T, 32, 32)
    tmp = arr[:, :16, 16:].copy()
    arr[:, :16, 16:] = arr[:, 16:, :16]
    arr[:, 16:, :16] = tmp
    return xr.DataArray(arr, dims=images.dims, coords=images.coords, attrs=images.attrs)


def calibrate_ph(ds: xr.Dataset, params: PhCalibParams) -> xr.Dataset:
    """Calibrate one pulse-height store; return a new L1 Dataset (no I/O)."""
    images = ds["images"].astype("float32")
    if images.shape[1] == 32:  # ph1024
        images = _fix_ph1024_quabo_order(images)

    subset = images[:: params.frame_stride]
    shifted = subset + params.baseline_offset
    pedestal = shifted.median(dim="time")
    sigma = shifted.std(dim="time")

    calibrated = (images + params.baseline_offset) - pedestal
    above = calibrated.where(calibrated > (params.sigma_threshold * sigma))
    above.name = "pedestal_subtracted"

    global_ped_mean = float(pedestal.mean())
    hot = (pedestal > 3.0 * global_ped_mean).astype("uint8")
    hot.name = "hot_pixel_mask"
    dead = (pedestal == 0).astype("uint8")
    dead.name = "dead_pixel_mask"

    carried = {k: ds[k] for k in ds.data_vars if k != "images"}
    out = xr.Dataset(
        {"pedestal_subtracted": above, "hot_pixel_mask": hot, "dead_pixel_mask": dead, **carried}
    )
    out.attrs = {
        **ds.attrs,
        DATA_LEVEL_KEY: levels.validate_level("L1"),
        STORAGE_VERSION_KEY: PANOSETI_ANALYSIS_STORAGE_VERSION,
        CALIBRATION_KEY: {"kind": "ph", **params.model_dump()},
    }
    return out
