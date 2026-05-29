import json
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
import xarray as xr

from panoseti_analysis.adapters.nextflow.classify_cloud import run_classify as run_classify_cli
from panoseti_analysis.adapters.ray.classify_cloud import process_store
from panoseti_analysis.config.models import CloudInferParams


@pytest.fixture(scope="session")
def ray_local_mode():
    """Initializes a local Ray cluster for testing."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def dummy_l1_store(tmp_path: Path) -> Path:
    """Create a dummy L1 store on disk."""
    store_path = tmp_path / "obs_TEST.dp_img16.module_1.L1.zarr"
    
    n_frames = 60
    t0_ns = 1700000000000000000
    cadence_ns = 1_000_000_000
    
    t_arr = np.arange(t0_ns, t0_ns + n_frames * cadence_ns, cadence_ns, dtype=np.int64)
    img_data = np.random.randn(n_frames, 32, 32).astype(np.float32) * 10.0
    
    ds = xr.Dataset(
        data_vars={
            "median_subtracted": (["T", "H", "W"], img_data),
            "unix_t_ns": (["T"], t_arr)
        },
        attrs={"data_product": "img16", "module": "1"}
    )
    
    ds.to_zarr(store_path)
    return store_path


@pytest.fixture
def dummy_model_path(tmp_path: Path) -> Path:
    """Create a dummy saved model."""
    from panoseti_analysis.algorithms.cloud_detector import CloudDetection
    
    model = CloudDetection()
    model.eval()
    
    model_path = tmp_path / "dummy_model.pt"
    torch.save(model.state_dict(), model_path)
    
    json_path = model_path.with_suffix(".json")
    from panoseti_analysis.io.checksum import compute_sha256
    sha = compute_sha256(model_path)
    
    with json_path.open("w") as f:
        json.dump({
            "model_name": "dummy",
            "model_version": "1.0",
            "checksum": f"sha256:{sha}",
            "input_spec": {}
        }, f)
        
    return model_path


def test_ray_vs_cli_equivalence(
    ray_local_mode, 
    dummy_l1_store: Path, 
    dummy_model_path: Path, 
    tmp_path: Path
) -> None:
    """Test that the Nextflow-CLI and Ray adapters produce identical output."""
    torch.use_deterministic_algorithms(True)
    
    out_cli = tmp_path / "cli_out"
    out_cli.mkdir()
    l2_store_cli = out_cli / "obs_TEST.cloud.module_1.zarr"
    
    out_ray = tmp_path / "ray_out"
    out_ray.mkdir()
    
    # 1. Run CLI adapter
    run_classify_cli(
        l1_store=dummy_l1_store,
        l2_store=l2_store_cli,
        model_path=dummy_model_path,
        cadence_s=60.0,
        threshold=0.5
    )
    
    # 2. Run Ray adapter
    from panoseti_analysis.io.models import load_classifier
    model, bundle = load_classifier(dummy_model_path)
    
    params = CloudInferParams(cadence_s=60.0, threshold=0.5)
    
    # ray executes in background, get blocks
    future = process_store.remote(
        l1_store=dummy_l1_store,
        out_dir=out_ray,
        model=model,
        bundle_dict=bundle.model_dump(),
        params=params,
        codec="zstd",
        level=5
    )
    
    l2_store_ray, record_ray = ray.get(future)
    
    # 3. Compare outputs
    ds_cli = xr.open_zarr(l2_store_cli)
    ds_ray = xr.open_zarr(l2_store_ray)
    
    np.testing.assert_array_equal(ds_cli["cloud_score"].values, ds_ray["cloud_score"].values)
    np.testing.assert_array_equal(ds_cli["feature_raw_fft"].values, ds_ray["feature_raw_fft"].values)
    
    assert ds_cli.attrs["data_level"] == "L2"
    assert ds_ray.attrs["data_level"] == "L2"
