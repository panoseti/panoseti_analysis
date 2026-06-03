"""Smoke test for Ray adapter using a real local multi-process cluster (not local_mode).

Starts a Ray head on localhost, submits one classify task, verifies the result,
then shuts down. Validates that ray.init(address=...) attaches to an existing
cluster — the same pattern used by the standalone launcher path for RAL.

Marked with 'slow' so it can be excluded from quick CI runs: pytest -m "not slow".
"""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

RAY_TEST_PORT = 6399  # non-default to avoid conflicts with other tests/sessions


@pytest.fixture(scope="module")
def real_ray_cluster(tmp_path_factory):
    """Start a local Ray head cluster, yield the address, then stop."""
    pytest.importorskip("ray")
    subprocess.Popen(
        [
            "ray",
            "start",
            "--head",
            f"--port={RAY_TEST_PORT}",
            "--num-cpus=2",
            "--num-gpus=0",
            "--include-dashboard=false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(6)
    address = f"127.0.0.1:{RAY_TEST_PORT}"
    yield address
    subprocess.run(["ray", "stop", "--force"], check=False)


@pytest.fixture
def l1_store_for_cluster(tmp_path: Path) -> Path:
    store_path = tmp_path / "obs_CLUSTERTEST.dp_img16.module_2.L1.zarr"
    n_frames = 60
    t0_ns = 1_700_000_000_000_000_000
    t_arr = np.arange(t0_ns, t0_ns + n_frames * 1_000_000_000, 1_000_000_000, dtype=np.int64)
    rng = np.random.default_rng(7)
    img_data = rng.standard_normal((n_frames, 32, 32)).astype(np.float32)
    ds = xr.Dataset(
        {
            "median_subtracted": (["T", "H", "W"], img_data),
            "unix_t_ns": (["T"], t_arr),
        },
        attrs={"data_product": "img16", "module": "2", "run_id": "obs_CLUSTERTEST"},
    )
    ds.to_zarr(store_path)
    return store_path


@pytest.fixture
def model_path_for_cluster(tmp_path: Path) -> Path:
    from panoseti_analysis.algorithms.cloud_detector import CloudDetection
    from panoseti_analysis.io.checksum import compute_sha256

    model = CloudDetection()
    model_path = tmp_path / "cluster_model.pt"
    torch.save(model.state_dict(), model_path)
    sha = compute_sha256(model_path)
    json_path = model_path.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(
            {
                "model_name": "cluster_test",
                "model_version": "1.0",
                "checksum": f"sha256:{sha}",
                "input_spec": {},
            },
            f,
        )
    return model_path


@pytest.mark.slow
def test_ray_real_cluster_smoke(
    real_ray_cluster: str,
    l1_store_for_cluster: Path,
    model_path_for_cluster: Path,
    tmp_path: Path,
) -> None:
    """Verify process_store works attached to a real local Ray cluster (not local_mode)."""
    import ray

    os.environ["RAY_ADDRESS"] = real_ray_cluster
    try:
        ray.init(address=real_ray_cluster, ignore_reinit_error=True)

        from panoseti_analysis.adapters.ray.classify_cloud import process_store
        from panoseti_analysis.config.models import CloudInferParams
        from panoseti_analysis.io.models import load_classifier

        out_dir = tmp_path / "cluster_out"
        out_dir.mkdir()
        model, bundle = load_classifier(model_path_for_cluster)
        params = CloudInferParams()

        future = process_store.remote(
            l1_store=l1_store_for_cluster,
            out_dir=out_dir,
            model=model,
            bundle_dict=bundle.model_dump(),
            params=params,
            codec="zstd",
            level=5,
        )
        l2_store, record = ray.get(future)

        assert l2_store.exists(), f"L2 store not written: {l2_store}"
        assert record.dp == "img16", f"dp={record.dp!r}, expected 'img16'"
        assert record.module == "2", f"module={record.module!r}, expected '2'"
        assert record.level == "L2"

        ds_l2 = xr.open_zarr(l2_store, consolidated=False)
        assert "cloud_score" in ds_l2
        assert ds_l2.attrs.get("data_level") == "L2"
    finally:
        ray.shutdown()
        del os.environ["RAY_ADDRESS"]
