from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from panoseti_analysis.algorithms.cloud_detector import CloudDetection, predict_cloud_score
from panoseti_analysis.config.models import CloudInferParams

CHECKPOINT_PATH = Path(__file__).parents[2] / "assets/models/cloud_detector_v1.pt"


@pytest.fixture
def dummy_model() -> CloudDetection:
    """Provides a fresh, uninitialized (random weights) model for testing."""
    model = CloudDetection()
    model.eval()
    return model


@pytest.fixture
def synthetic_l1_dataset() -> xr.Dataset:
    """Provides a synthetic 60-second L1 img dataset at 1 Hz cadence."""
    n_frames = 60
    t0_ns = 1700000000000000000
    cadence_ns = 1_000_000_000

    t_arr = np.arange(t0_ns, t0_ns + n_frames * cadence_ns, cadence_ns, dtype=np.int64)
    # Random synthetic frames
    np.random.seed(42)
    img_data = np.random.randn(n_frames, 32, 32).astype(np.float32) * 10.0

    return xr.Dataset(
        data_vars={
            "median_subtracted": (["T", "H", "W"], img_data),
            "unix_t_ns": (["T"], t_arr)
        }
    )


def test_cloud_detector_shape(synthetic_l1_dataset: xr.Dataset, dummy_model: CloudDetection) -> None:
    """Test that the cloud detector produces the correct output dataset shape."""
    params = CloudInferParams(cadence_s=60.0, threshold=0.5)

    out_ds = predict_cloud_score(synthetic_l1_dataset, dummy_model, params)

    assert "cloud_score" in out_ds
    assert "cloud_label" in out_ds
    assert "feature_raw_fft" in out_ds
    assert "feature_deriv_fft" in out_ds

    # 60s input = 1 window
    assert out_ds.sizes["T_l2"] == 1

    assert out_ds["cloud_score"].dtype == np.float32
    assert out_ds["cloud_label"].dtype == np.uint8
    assert out_ds["feature_raw_fft"].shape == (1, 32, 32)
    assert out_ds["feature_deriv_fft"].shape == (1, 32, 32)


def test_cloud_detector_determinism(synthetic_l1_dataset: xr.Dataset, dummy_model: CloudDetection) -> None:
    """Ensure that consecutive runs produce exactly identical output."""
    torch.use_deterministic_algorithms(True)

    params = CloudInferParams(cadence_s=60.0, threshold=0.5)

    out_1 = predict_cloud_score(synthetic_l1_dataset, dummy_model, params)
    out_2 = predict_cloud_score(synthetic_l1_dataset, dummy_model, params)

    np.testing.assert_array_equal(out_1["cloud_score"].values, out_2["cloud_score"].values)
    np.testing.assert_array_equal(out_1["feature_raw_fft"].values, out_2["feature_raw_fft"].values)


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="checkpoint not present in repo")
def test_checkpoint_loads_cleanly() -> None:
    """Verify pretrained state_dict loads with no missing/unexpected keys."""
    model = CloudDetection()
    state_dict = torch.load(CHECKPOINT_PATH, weights_only=True, map_location="cpu")
    result = model.load_state_dict(state_dict, strict=True)
    assert result.missing_keys == [], f"Missing keys: {result.missing_keys}"
    assert result.unexpected_keys == [], f"Unexpected keys: {result.unexpected_keys}"


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="checkpoint not present in repo")
def test_checkpoint_param_count() -> None:
    """Param count must match the trained model_summary.txt value of 109,427."""
    model = CloudDetection()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True, map_location="cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 109_427, f"Param count mismatch: got {n_params}, expected 109,427"
