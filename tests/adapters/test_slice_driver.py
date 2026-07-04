"""Tests for slice_driver — in-memory PFF→L1/L2 chain.

All tests mock ``sequence_to_dataset`` to return a synthetic L0 Dataset
matching the expected L0 schema, so no real PFF files are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from panoseti_analysis.adapters._common import SuspectTimestamps
from panoseti_analysis.adapters.slice_driver import slice_to_l1, slice_to_l2_cloud

# ── helpers ───────────────────────────────────────────────────────────────────

_EPOCH_NS = 1_700_000_000_000_000_000


def _make_synthetic_l0_img(n: int = 60) -> xr.Dataset:
    """Synthetic L0 img16 dataset, mirrors the conftest make_l0_img shape."""
    rng = np.random.default_rng(0)
    images = rng.poisson(3.0, size=(n, 32, 32)).astype("uint16")
    t = _EPOCH_NS + np.arange(n, dtype="int64") * 20_000  # 20 µs cadence
    return xr.Dataset(
        data_vars={
            "images": (("time", "y", "x"), images),
            "unix_t_ns": ("time", t),
            "quabo_0_pkt_num": ("time", np.arange(n, dtype="uint32")),
        },
        attrs={
            "data_product": "img16",
            "module": "1",
            "bytes_per_pixel": 2,
            "total_frames": n,
            "panoseti_pff_zarr_version": "1.0",
        },
    )


def _make_synthetic_l0_ph(n: int = 60) -> xr.Dataset:
    """Synthetic L0 ph256 dataset, mirrors the conftest make_l0_ph shape."""
    rng = np.random.default_rng(1)
    images = rng.normal(800.0, 5.0, size=(n, 16, 16)).astype("int16")
    t = _EPOCH_NS + np.arange(n, dtype="int64") * 1_000_000  # 1 ms cadence
    return xr.Dataset(
        data_vars={
            "images": (("time", "y", "x"), images),
            "unix_t_ns": ("time", t),
            "pkt_num": ("time", np.arange(n, dtype="uint32")),
            "quabo_num": ("time", np.zeros(n, dtype="uint8")),
        },
        attrs={
            "data_product": "ph256",
            "module": "1",
            "bytes_per_pixel": 2,
            "total_frames": n,
            "panoseti_pff_zarr_version": "1.0",
        },
    )


def _mock_seq(ds: xr.Dataset) -> MagicMock:
    """Build a minimal PFFSequence mock — only needs to be passed through."""
    return MagicMock(name="PFFSequence")


# ── tests: kind inference ─────────────────────────────────────────────────────


def test_slice_to_l1_img_produces_median_subtracted() -> None:
    """An img16 sequence produces a dataset with median_subtracted (not pedestal_subtracted)."""
    ds_l0 = _make_synthetic_l0_img(n=60)
    seq = _mock_seq(ds_l0)

    with patch("panoseti_analysis.io.zarr_compat.sequence_to_dataset", return_value=ds_l0):
        out = slice_to_l1(seq, img_stride=5, block=8)

    assert "median_subtracted" in out.data_vars
    assert "pedestal_subtracted" not in out.data_vars


def test_slice_to_l1_ph_produces_pedestal_subtracted() -> None:
    """A ph256 sequence produces a dataset with pedestal_subtracted (not median_subtracted)."""
    ds_l0 = _make_synthetic_l0_ph(n=60)
    seq = _mock_seq(ds_l0)

    with patch("panoseti_analysis.io.zarr_compat.sequence_to_dataset", return_value=ds_l0):
        out = slice_to_l1(seq, ph_stride=5)

    assert "pedestal_subtracted" in out.data_vars
    assert "median_subtracted" not in out.data_vars


# ── tests: decimation shape ───────────────────────────────────────────────────


def test_slice_to_l1_decimation_shape() -> None:
    """With decimate=5 on a 50-frame seq, L1 has ~10 frames (±1 for boundary rounding)."""
    n_total = 50
    decimate = 5
    ds_l0 = _make_synthetic_l0_img(n=n_total // decimate)  # seq_to_dataset already decimates

    # Simulate the decimated L0 returned by sequence_to_dataset (10 frames)
    seq = _mock_seq(ds_l0)

    with patch(
        "panoseti_analysis.io.zarr_compat.sequence_to_dataset", return_value=ds_l0
    ) as mock_s2d:
        out = slice_to_l1(seq, decimate=decimate, img_stride=2, block=8)
        # Verify sequence_to_dataset was called with step=decimate
        mock_s2d.assert_called_once()
        _, kwargs = mock_s2d.call_args
        assert kwargs.get("step") == decimate

    expected_frames = n_total // decimate
    assert abs(out.sizes["time"] - expected_frames) <= 1


# ── tests: fail_on_suspect behaviour ─────────────────────────────────────────


def test_slice_to_l1_fail_on_suspect_false_default_does_not_raise() -> None:
    """fail_on_suspect=False (default) must not raise even with SUSPECT timestamps."""
    ds_l0 = _make_synthetic_l0_img(n=30)

    # Inject a corruption-grade backward jump (>1000x cadence = 20 µs × 1000 = 20 ms)
    t = ds_l0["unix_t_ns"].values.copy()
    t[15] -= 10**14  # large backward jump → SUSPECT
    ds_l0 = ds_l0.assign(unix_t_ns=("time", t))

    seq = _mock_seq(ds_l0)

    with patch("panoseti_analysis.io.zarr_compat.sequence_to_dataset", return_value=ds_l0):
        # Must not raise — default fail_on_suspect=False
        out = slice_to_l1(seq, img_stride=5, block=8, fail_on_suspect=False)

    assert isinstance(out, xr.Dataset)


def test_slice_to_l1_fail_on_suspect_true_raises() -> None:
    """fail_on_suspect=True must raise SuspectTimestamps on corruption-grade timestamps."""
    ds_l0 = _make_synthetic_l0_img(n=30)

    t = ds_l0["unix_t_ns"].values.copy()
    t[15] -= 10**14  # corruption-grade backward jump
    ds_l0 = ds_l0.assign(unix_t_ns=("time", t))

    seq = _mock_seq(ds_l0)

    with (
        patch("panoseti_analysis.io.zarr_compat.sequence_to_dataset", return_value=ds_l0),
        pytest.raises(SuspectTimestamps),
    ):
        slice_to_l1(seq, img_stride=5, block=8, fail_on_suspect=True)


# ── tests: slice_to_l2_cloud raises for ph kind ───────────────────────────────


def test_slice_to_l2_cloud_raises_for_ph_kind() -> None:
    """slice_to_l2_cloud must raise ValueError for ph (pulse-height) sequences."""
    ds_l0_ph = _make_synthetic_l0_ph(n=60)
    seq = _mock_seq(ds_l0_ph)
    model = MagicMock(name="CloudModel")

    with (
        patch("panoseti_analysis.io.zarr_compat.sequence_to_dataset", return_value=ds_l0_ph),
        pytest.raises(ValueError, match="img"),
    ):
        slice_to_l2_cloud(seq, model, ph_stride=5)


# ── tests: frame_range forwarding ─────────────────────────────────────────────


def test_slice_to_l1_frame_range_forwarded() -> None:
    """frame_range should be unpacked to start/stop kwargs of sequence_to_dataset."""
    ds_l0 = _make_synthetic_l0_img(n=20)
    seq = _mock_seq(ds_l0)

    with patch(
        "panoseti_analysis.io.zarr_compat.sequence_to_dataset", return_value=ds_l0
    ) as mock_s2d:
        slice_to_l1(seq, frame_range=(5, 15), img_stride=5, block=8)
        _, kwargs = mock_s2d.call_args
        assert kwargs["start"] == 5
        assert kwargs["stop"] == 15


def test_slice_to_l1_time_range_forwarded() -> None:
    """time_range should be unpacked to start_ns/stop_ns kwargs of sequence_to_dataset."""
    ds_l0 = _make_synthetic_l0_img(n=20)
    seq = _mock_seq(ds_l0)
    start_ns = _EPOCH_NS + 100_000
    stop_ns = _EPOCH_NS + 200_000

    with patch(
        "panoseti_analysis.io.zarr_compat.sequence_to_dataset", return_value=ds_l0
    ) as mock_s2d:
        slice_to_l1(seq, time_range=(start_ns, stop_ns), img_stride=5, block=8)
        _, kwargs = mock_s2d.call_args
        assert kwargs["start_ns"] == start_ns
        assert kwargs["stop_ns"] == stop_ns
