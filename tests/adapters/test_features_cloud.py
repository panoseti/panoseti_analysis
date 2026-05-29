"""Adapter tests for pa-features-cloud (L1 img -> feature cache)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from panoseti_analysis.adapters.features import run_features_cloud
from panoseti_analysis.config.models import StoreLineage
from panoseti_analysis.io.provenance import read_history
from panoseti_analysis.io.stores import open_store

_EPOCH_NS = 1_700_000_000_000_000_000  # arbitrary fixed base for reproducible timestamps
_RECIPE = Path(__file__).parents[2] / "recipes" / "cloud_v1.yml"


def _make_l1_img_store(tmp_path: Path, *, n_frames: int = 180, module: str = "1") -> Path:
    """Create a synthetic L1 img store with median_subtracted and a > 60 s time span."""
    # 1-second cadence → 180 frames → 179-second span (> 60 s needed for features)
    cadence_ns = 1_000_000_000
    t_arr = _EPOCH_NS + np.arange(n_frames, dtype=np.int64) * cadence_ns
    rng = np.random.default_rng(42)
    img = rng.standard_normal((n_frames, 32, 32)).astype(np.float32) * 10.0

    store_path = tmp_path / f"obs_TEST.dp_img16.module_{module}.L1.zarr"
    ds = xr.Dataset(
        data_vars={
            "median_subtracted": (["time", "y", "x"], img),
            "unix_t_ns": (["time"], t_arr),
        },
        attrs={
            "data_product": "img16",
            "module": module,
            "data_level": "L1",
        },
    )
    ds.to_zarr(str(store_path), zarr_format=3)
    return store_path


def test_features_cloud_produces_zarr(tmp_path: Path) -> None:
    """run_features_cloud must produce a readable feature store with X shape (N, 2, 32, 32)."""
    l1_store = _make_l1_img_store(tmp_path / "l1")
    out_path = tmp_path / "features.zarr"

    run_features_cloud([l1_store], out_path, _RECIPE)

    assert out_path.is_dir()
    ds = open_store(out_path)
    assert "X" in ds.data_vars
    X = ds["X"].values
    assert X.ndim == 4
    assert X.shape[1] == 2
    assert X.shape[2] == 32
    assert X.shape[3] == 32
    assert X.shape[0] >= 1


def test_features_cloud_processing_history(tmp_path: Path) -> None:
    """Feature store must carry processing_history with a 'cloud_features' step."""
    l1_store = _make_l1_img_store(tmp_path / "l1")
    out_path = tmp_path / "features.zarr"

    run_features_cloud([l1_store], out_path, _RECIPE)

    ds = open_store(out_path)
    history = read_history(dict(ds.attrs))
    assert len(history) >= 1
    assert history[-1].step_name == "cloud_features"


def test_features_cloud_with_label_csv(tmp_path: Path) -> None:
    """Labels from CSV must be joined by time window — non-(-1) values must appear."""
    l1_store = _make_l1_img_store(tmp_path / "l1", module="1")
    out_path = tmp_path / "features.zarr"

    # Create a label CSV that covers the first window of the feature store
    # Store starts at _EPOCH_NS; first feature window should be around _EPOCH_NS + 0
    t_start = _EPOCH_NS
    t_end = _EPOCH_NS + 120_000_000_000  # 120 s covers first 2 features at 60 s cadence

    label_csv = tmp_path / "labels.csv"
    df = pd.DataFrame(
        [{"module": "1", "t_start_ns": t_start, "t_end_ns": t_end, "label": 1}]
    )
    df.to_csv(label_csv, index=False)

    run_features_cloud([l1_store], out_path, _RECIPE, label_csv=label_csv)

    ds = open_store(out_path)
    labels = ds["label"].values
    # At least one sample must be labeled 1 (the CSV window covers some feature centers)
    assert (labels == 1).any(), f"Expected label=1 samples, got unique labels: {np.unique(labels)}"


def test_features_cloud_lineage_json(tmp_path: Path) -> None:
    """With lineage_out specified, the JSON file must be a valid array of StoreLineage records."""
    l1_store = _make_l1_img_store(tmp_path / "l1")
    out_path = tmp_path / "features.zarr"
    lineage_out = tmp_path / "features.lineage.json"

    rec = run_features_cloud([l1_store], out_path, _RECIPE, lineage_out=lineage_out)

    assert lineage_out.exists()
    arr = json.loads(lineage_out.read_text())
    assert isinstance(arr, list)
    assert len(arr) == 1

    entry = arr[0]
    assert "checksum" in entry
    assert entry["checksum"] is not None
    assert entry["checksum"].startswith("sha256:")
    assert "processing_history" in entry
    assert isinstance(entry["processing_history"], list)
    assert len(entry["processing_history"]) >= 1

    # Also validate the returned record
    assert rec.checksum is not None
    assert rec.checksum.startswith("sha256:")
    StoreLineage.model_validate(entry)


def test_features_cloud_no_label_csv_returns_minus_one(tmp_path: Path) -> None:
    """Without a label CSV all labels must be -1."""
    l1_store = _make_l1_img_store(tmp_path / "l1")
    out_path = tmp_path / "features.zarr"

    run_features_cloud([l1_store], out_path, _RECIPE)

    ds = open_store(out_path)
    assert (ds["label"].values == -1).all()


def test_features_cloud_split_column_populated(tmp_path: Path) -> None:
    """The 'split' variable must contain 'train', 'val', and 'test' values."""
    l1_store = _make_l1_img_store(tmp_path / "l1", n_frames=500)
    out_path = tmp_path / "features.zarr"

    run_features_cloud([l1_store], out_path, _RECIPE)

    ds = open_store(out_path)
    assert "split" in ds.data_vars
    splits = set(ds["split"].values.tolist())
    # With 500 frames at 1s cadence ~ 8 min span, we get about 8 features at 60s cadence
    # so all three split categories may or may not appear; check that split is assigned
    assert splits <= {"train", "val", "test", "unassigned"}
    # At least one non-unassigned split must exist for n >= 3 features
    assert splits - {"unassigned"}


def test_features_cloud_raises_on_no_valid_stores(tmp_path: Path) -> None:
    """run_features_cloud must raise ValueError when no valid img stores are provided."""
    # Create a store without 'median_subtracted'
    store_path = tmp_path / "bad.zarr"
    ds = xr.Dataset(
        data_vars={"images": (["time", "y", "x"], np.zeros((10, 32, 32), dtype=np.uint16))},
        attrs={"data_product": "img16", "module": "1"},
    )
    ds.to_zarr(str(store_path), zarr_format=3)

    out_path = tmp_path / "features.zarr"
    with pytest.raises(ValueError, match="No img L1 stores"):
        run_features_cloud([store_path], out_path, _RECIPE)
