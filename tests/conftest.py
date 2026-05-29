"""Shared synthetic ``xarray.Dataset`` fixtures for Layer A + io tests.

These mirror the L0 store layout pypff emits (``images`` + ``unix_t_ns`` + header
arrays + root attrs) so kernels and io can be exercised entirely in memory — no PFF
files, no Nextflow.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

_EPOCH_NS = 1_700_000_000_000_000_000  # arbitrary fixed base for reproducible timestamps


def make_l0_ph(
    n: int = 60,
    size: int = 16,
    *,
    data_product: str = "ph256",
    cadence_ns: int = 1_000_000,
    seed: int = 0,
) -> xr.Dataset:
    """Synthetic L0 pulse-height store (int16 ADC), monotonic 1 ms-ish cadence."""
    rng = np.random.default_rng(seed)
    images = rng.normal(800.0, 5.0, size=(n, size, size)).astype("int16")
    t = _EPOCH_NS + np.arange(n, dtype="int64") * cadence_ns
    ds = xr.Dataset(
        data_vars={
            "images": (("time", "y", "x"), images),
            "unix_t_ns": ("time", t),
            "pkt_num": ("time", np.arange(n, dtype="uint32")),
            "quabo_num": ("time", np.zeros(n, dtype="uint8")),
        },
        attrs={
            "data_product": data_product,
            "module": "1",
            "bytes_per_pixel": 2,
            "total_frames": n,
            "panoseti_pff_zarr_version": "1.0",
        },
    )
    return ds


def make_l0_img(
    n: int = 60,
    size: int = 32,
    *,
    data_product: str = "img16",
    dtype: str = "uint16",
    cadence_ns: int = 20_000,
    seed: int = 0,
) -> xr.Dataset:
    """Synthetic L0 movie-mode store (uint8/uint16 counts), monotonic 20 us cadence."""
    rng = np.random.default_rng(seed)
    images = rng.poisson(3.0, size=(n, size, size)).astype(dtype)
    t = _EPOCH_NS + np.arange(n, dtype="int64") * cadence_ns
    ds = xr.Dataset(
        data_vars={
            "images": (("time", "y", "x"), images),
            "unix_t_ns": ("time", t),
            "quabo_0_pkt_num": ("time", np.arange(n, dtype="uint32")),
        },
        attrs={
            "data_product": data_product,
            "module": "1",
            "bytes_per_pixel": 1 if dtype == "uint8" else 2,
            "total_frames": n,
            "panoseti_pff_zarr_version": "1.0",
        },
    )
    return ds


@pytest.fixture
def l0_ph_ds() -> xr.Dataset:
    return make_l0_ph()


@pytest.fixture
def l0_img_ds() -> xr.Dataset:
    return make_l0_img()


@pytest.fixture
def make_ph():  # type: ignore[no-untyped-def]
    """Factory fixture: call with kwargs to build a synthetic L0 ph Dataset."""
    return make_l0_ph


@pytest.fixture
def make_img():  # type: ignore[no-untyped-def]
    """Factory fixture: call with kwargs to build a synthetic L0 img Dataset."""
    return make_l0_img
