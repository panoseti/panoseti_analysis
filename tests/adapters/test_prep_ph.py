"""Adapter tests for pa-prep-ph (L1 PH → BetaVAE feature cache)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from panoseti_analysis.adapters.ph_prep import _log_norm_to_unit, run_prep_ph
from panoseti_analysis.config.models import StoreLineage
from panoseti_analysis.io.provenance import read_history
from panoseti_analysis.io.stores import open_store

_RECIPE = Path(__file__).parents[2] / "recipes" / "vae_train_v1.yml"
_EPOCH_NS = 1_700_000_000_000_000_000


def _make_l1_ph_store(
    tmp_path: Path,
    *,
    n_frames: int = 30,
    size: int = 16,
    module: str = "1",
) -> Path:
    """Create a minimal synthetic L1 PH store with ``pedestal_subtracted``."""
    cadence_ns = 1_000_000  # 1 ms
    t_arr = _EPOCH_NS + np.arange(n_frames, dtype=np.int64) * cadence_ns
    rng = np.random.default_rng(42)
    ph = rng.uniform(0.5, 50.0, size=(n_frames, size, size)).astype(np.float32)

    store_path = tmp_path / f"obs_TEST.dp_ph256.module_{module}.L1.zarr"
    ds = xr.Dataset(
        data_vars={
            "pedestal_subtracted": (["time", "y", "x"], ph),
            "unix_t_ns": (["time"], t_arr),
        },
        attrs={
            "data_product": "ph256",
            "module": module,
            "data_level": "L1",
        },
    )
    ds.to_zarr(str(store_path), zarr_format=3)
    return store_path


# ── unit test for helper ──────────────────────────────────────────────────────


def test_log_norm_to_unit_mean_near_zero() -> None:
    """_log_norm_to_unit output must have approximately zero mean and be float32."""
    rng = np.random.default_rng(0)
    arr = rng.random((10, 16, 16)).astype(np.float32) * 100 + 1
    result = _log_norm_to_unit(arr)
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    assert abs(result.mean()) < 0.1, f"Mean {result.mean():.4f} not close to 0"


# ── adapter tests ─────────────────────────────────────────────────────────────


def test_prep_ph_produces_zarr(tmp_path: Path) -> None:
    """run_prep_ph must write a readable feature cache with X shape (N, 1, H, W)."""
    l1_ph = _make_l1_ph_store(tmp_path / "l1")
    out_path = tmp_path / "ph_features.zarr"

    run_prep_ph([l1_ph], out_path, _RECIPE)

    assert out_path.is_dir(), "Feature cache directory not created"
    ds = open_store(out_path)
    assert "X" in ds.data_vars, "'X' variable missing from feature cache"
    X = ds["X"].values
    assert X.ndim == 4, f"Expected 4-D array, got {X.ndim}-D"
    assert X.shape[1] == 1, f"Expected 1 channel, got {X.shape[1]}"
    assert X.shape[2] == 16, f"Expected H=16, got {X.shape[2]}"
    assert X.shape[3] == 16, f"Expected W=16, got {X.shape[3]}"
    assert X.shape[0] >= 1, "Feature cache is empty"
    assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"


def test_prep_ph_processing_history(tmp_path: Path) -> None:
    """Feature store must carry processing_history with a 'prep_ph' step as last entry."""
    l1_ph = _make_l1_ph_store(tmp_path / "l1")
    out_path = tmp_path / "ph_features.zarr"

    run_prep_ph([l1_ph], out_path, _RECIPE)

    ds = open_store(out_path)
    history = read_history(dict(ds.attrs))
    assert len(history) >= 1, "processing_history is empty"
    assert history[-1].step_name == "prep_ph", (
        f"Last step is {history[-1].step_name!r}, expected 'prep_ph'"
    )


def test_prep_ph_lineage_json(tmp_path: Path) -> None:
    """With lineage_out, JSON must be a valid list with a record containing 'checksum'."""
    l1_ph = _make_l1_ph_store(tmp_path / "l1")
    out_path = tmp_path / "ph_features.zarr"
    lineage_out = tmp_path / "ph.lineage.json"

    rec = run_prep_ph([l1_ph], out_path, _RECIPE, lineage_out=lineage_out)

    assert lineage_out.exists(), "Lineage JSON not written"
    arr = json.loads(lineage_out.read_text())
    assert isinstance(arr, list) and len(arr) == 1
    entry = arr[0]
    assert "checksum" in entry, "'checksum' key missing from lineage entry"
    assert entry["checksum"] is not None
    assert entry["checksum"].startswith("sha256:")

    # round-trip validation
    StoreLineage.model_validate(entry)

    # returned record also has checksum
    assert rec.checksum is not None
    assert rec.checksum.startswith("sha256:")


def test_prep_ph_raises_on_no_valid_stores(tmp_path: Path) -> None:
    """run_prep_ph must raise ValueError when no PH L1 stores are found."""
    # Create an img store (no pedestal_subtracted / baseline_subtracted / ph_counts)
    store_path = tmp_path / "not_a_ph_store.zarr"
    ds = xr.Dataset(
        data_vars={
            "median_subtracted": (["time", "y", "x"], np.zeros((10, 32, 32), dtype=np.float32))
        },
        attrs={"data_product": "img16", "module": "1", "data_level": "L1"},
    )
    ds.to_zarr(str(store_path), zarr_format=3)

    out_path = tmp_path / "ph_features.zarr"
    with pytest.raises(ValueError, match="No calibrated PH"):
        run_prep_ph([store_path], out_path, _RECIPE)
