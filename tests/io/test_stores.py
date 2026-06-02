"""io.stores round-trip tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import zarr

from panoseti_analysis.config.models import ProcessingStep
from panoseti_analysis.io.stores import open_l0, open_store, write_store

# ── helper ────────────────────────────────────────────────────────────────────

def _make_step(**overrides: Any) -> ProcessingStep:
    defaults: dict[str, Any] = {
        "step_name": "convert",
        "step_version": "1.0.0",
        "params": {"chunk_frames": 512},
        "timestamp_utc": "2026-05-29T00:00:00Z",
        "software": {"git_sha": "abc123", "package_version": "0.1.0"},
    }
    defaults.update(overrides)
    return ProcessingStep(**defaults)


def test_write_then_open_round_trips(l0_ph_ds: xr.Dataset, tmp_path: Path) -> None:
    out = tmp_path / "s.zarr"
    nbytes = write_store(l0_ph_ds, out)
    assert nbytes > 0
    assert out.is_dir()

    back = open_store(out)
    np.testing.assert_array_equal(back["images"].values, l0_ph_ds["images"].values)
    np.testing.assert_array_equal(back["unix_t_ns"].values, l0_ph_ds["unix_t_ns"].values)
    assert back.attrs["data_product"] == "ph256"


def test_write_store_is_idempotent(l0_img_ds: xr.Dataset, tmp_path: Path) -> None:
    out = tmp_path / "s.zarr"
    write_store(l0_img_ds, out)
    # second write over an existing store must not raise and must still round-trip
    write_store(l0_img_ds, out)
    back = open_store(out)
    assert back.sizes["time"] == l0_img_ds.sizes["time"]


def test_open_l0_requires_int64_time(l0_ph_ds: xr.Dataset, tmp_path: Path) -> None:
    out = tmp_path / "s.zarr"
    write_store(l0_ph_ds, out)
    ds = open_l0(out)
    assert ds["unix_t_ns"].dtype == np.int64


def test_write_store_rejects_unknown_codec(l0_ph_ds: xr.Dataset, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported codec"):
        write_store(l0_ph_ds, tmp_path / "s.zarr", codec="lz4")


def test_write_store_handles_nonuniform_dask_chunks(tmp_path: Path) -> None:
    # last time-chunk (60) larger than the first (40) — invalid for Zarr unless rechunked
    import dask.array as da

    arr = da.zeros((100, 4, 4), chunks=((40, 60), 4, 4), dtype="float32")
    ds = xr.Dataset(
        {"images": (("time", "y", "x"), arr), "unix_t_ns": ("time", np.arange(100, dtype="int64"))}
    )
    write_store(ds, tmp_path / "s.zarr")  # must not raise
    assert open_store(tmp_path / "s.zarr").sizes["time"] == 100


# ── processing_history stamping ───────────────────────────────────────────────

class TestWriteStoreProcessingHistory:
    def test_stamps_processing_history_into_root_attrs(
        self, l0_ph_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """write_store with processing_history writes the key into Zarr root attrs."""
        out = tmp_path / "s.zarr"
        step = _make_step()
        write_store(l0_ph_ds, out, processing_history=[step])

        # Read back raw Zarr attrs to confirm the key is on disk
        z = zarr.open(str(out), mode="r")
        assert "processing_history" in z.attrs
        history = z.attrs["processing_history"]
        assert isinstance(history, list)
        assert len(history) == 1
        assert history[0]["step_name"] == "convert"
        assert history[0]["step_version"] == "1.0.0"

    def test_processing_history_round_trips_via_pydantic(
        self, l0_ph_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """Serialized history deserializes back to identical ProcessingStep objects."""
        out = tmp_path / "s.zarr"
        step = _make_step(step_name="calibrate_ph", params={"sigma_threshold": 5.0})
        write_store(l0_ph_ds, out, processing_history=[step])

        z = zarr.open(str(out), mode="r")
        raw = z.attrs["processing_history"]
        restored = [ProcessingStep.model_validate(d) for d in raw]
        assert restored[0] == step

    def test_no_processing_history_key_when_none(
        self, l0_ph_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """write_store without processing_history must NOT write the key (backward compat)."""
        out = tmp_path / "s.zarr"
        write_store(l0_ph_ds, out)  # no processing_history arg

        z = zarr.open(str(out), mode="r")
        assert "processing_history" not in z.attrs

    def test_no_processing_history_key_when_empty_list(
        self, l0_ph_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """write_store with processing_history=[] must NOT write the key (backward compat)."""
        out = tmp_path / "s.zarr"
        write_store(l0_ph_ds, out, processing_history=[])

        z = zarr.open(str(out), mode="r")
        assert "processing_history" not in z.attrs

    def test_processing_history_multiple_steps(
        self, l0_ph_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """Multiple steps all appear in the root attrs in order."""
        out = tmp_path / "s.zarr"
        steps = [
            _make_step(step_name="convert"),
            _make_step(step_name="calibrate_ph", params={"sigma_threshold": 5.0}),
        ]
        write_store(l0_ph_ds, out, processing_history=steps)

        z = zarr.open(str(out), mode="r")
        history = z.attrs["processing_history"]
        assert len(history) == 2
        assert history[0]["step_name"] == "convert"
        assert history[1]["step_name"] == "calibrate_ph"

    def test_does_not_mutate_input_dataset(
        self, l0_ph_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """write_store must not add processing_history to the caller's Dataset attrs."""
        out = tmp_path / "s.zarr"
        original_attrs = dict(l0_ph_ds.attrs)
        step = _make_step()
        write_store(l0_ph_ds, out, processing_history=[step])

        # The original Dataset's attrs must be unchanged
        assert l0_ph_ds.attrs == original_attrs
        assert "processing_history" not in l0_ph_ds.attrs

    def test_existing_attrs_preserved_alongside_history(
        self, l0_ph_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """Existing attrs like data_product survive alongside the stamped history."""
        out = tmp_path / "s.zarr"
        step = _make_step()
        write_store(l0_ph_ds, out, processing_history=[step])

        z = zarr.open(str(out), mode="r")
        assert z.attrs["data_product"] == "ph256"
        assert "processing_history" in z.attrs


class TestWriteStoreSharding:
    """shard_factor=N must pack N dask-chunks into one shard file."""

    def test_sharding_reduces_file_count(
        self, l0_img_ds: xr.Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """shard_factor=4 must produce fewer zarr files than shard_factor=0."""
        import panoseti_analysis.io.stores as _stores
        monkeypatch.setattr(_stores, "_TIME_CHUNK", 8)  # makes 60-frame fixture span multiple chunks
        out_un = tmp_path / "unsharded.zarr"
        out_sh = tmp_path / "sharded.zarr"
        write_store(l0_img_ds, out_un, shard_factor=0)
        write_store(l0_img_ds, out_sh, shard_factor=4)
        files_un = sum(1 for f in out_un.rglob("*") if f.is_file())
        files_sh = sum(1 for f in out_sh.rglob("*") if f.is_file())
        assert files_sh < files_un, (
            f"Sharded ({files_sh}) must have fewer files than unsharded ({files_un})"
        )

    def test_sharded_roundtrip(self, l0_img_ds: xr.Dataset, tmp_path: Path) -> None:
        """Data written with sharding must read back identically."""
        out = tmp_path / "s.zarr"
        write_store(l0_img_ds, out, shard_factor=4)
        back = open_store(out)
        np.testing.assert_array_equal(
            back["images"].values, l0_img_ds["images"].values
        )
        np.testing.assert_array_equal(
            back["unix_t_ns"].values, l0_img_ds["unix_t_ns"].values
        )

    def test_shard_factor_zero_no_sharding(self, l0_img_ds: xr.Dataset, tmp_path: Path) -> None:
        """shard_factor=0 must not introduce ShardingCodec."""
        out = tmp_path / "s.zarr"
        write_store(l0_img_ds, out, shard_factor=0)
        z = zarr.open(str(out), mode="r", zarr_format=3)
        assert z["images"].shards is None, "shard_factor=0 must not shard"

    def test_sharded_store_has_time_dim_sharding_only(
        self, l0_img_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """Spatial dims must not be sharded (shard H == chunk H, shard W == chunk W)."""
        out = tmp_path / "s.zarr"
        write_store(l0_img_ds, out, shard_factor=4)
        z = zarr.open(str(out), mode="r", zarr_format=3)
        img = z["images"]
        if img.shards is not None:
            # shards along time axis only; spatial dims unchanged
            assert img.shards[1] == img.chunks[1], "H must not be sharded"
            assert img.shards[2] == img.chunks[2], "W must not be sharded"
