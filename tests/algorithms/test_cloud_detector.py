"""Cloud detector kernel tests.

CONTRACT tests: stable across model swaps — output schema, dtypes, shapes, monotonicity.
MODEL-SPECIFIC tests: tied to the current CloudDetection v1.0 architecture.
  Update these when retraining; they are intentionally brittle for that class of change.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from panoseti_analysis.algorithms.cloud_detector import CloudDetection, predict_cloud_score
from panoseti_analysis.config.models import CloudInferParams

CHECKPOINT_PATH = Path(__file__).parents[2] / "assets/models/cloud_detector_v1.pt"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(params=["random_weights"])
def model_under_test(request: pytest.FixtureRequest) -> CloudDetection:
    """Parametrize over model variants.

    Add 'pretrained' to params when a stable retrained checkpoint is ready.
    """
    model = CloudDetection()
    model.eval()
    return model


@pytest.fixture
def dummy_model() -> CloudDetection:
    """Alias for single-model tests that don't need parametrize."""
    model = CloudDetection()
    model.eval()
    return model


@pytest.fixture
def synthetic_l1_dataset() -> xr.Dataset:
    """60-second L1 img dataset at 1 Hz cadence → 1 window at cadence_s=60."""
    n_frames = 60
    t0_ns = 1_700_000_000_000_000_000
    t_arr = np.arange(t0_ns, t0_ns + n_frames * 1_000_000_000, 1_000_000_000, dtype=np.int64)
    np.random.seed(42)
    img_data = np.random.randn(n_frames, 32, 32).astype(np.float32) * 10.0
    return xr.Dataset(
        {
            "median_subtracted": (["T", "H", "W"], img_data),
            "unix_t_ns": (["T"], t_arr),
        }
    )


@pytest.fixture
def multi_window_l1_dataset() -> xr.Dataset:
    """120-second dataset → 2 windows at cadence_s=60."""
    n_frames = 120
    t0_ns = 1_700_000_000_000_000_000
    t_arr = np.arange(t0_ns, t0_ns + n_frames * 1_000_000_000, 1_000_000_000, dtype=np.int64)
    np.random.seed(99)
    img_data = np.random.randn(n_frames, 32, 32).astype(np.float32)
    return xr.Dataset(
        {
            "median_subtracted": (["T", "H", "W"], img_data),
            "unix_t_ns": (["T"], t_arr),
        }
    )


# ── CONTRACT TESTS (stable across model swaps) ────────────────────────────────


class TestOutputContract:
    """These tests must pass regardless of which model is loaded."""

    def test_output_schema(
        self, synthetic_l1_dataset: xr.Dataset, model_under_test: CloudDetection
    ) -> None:
        out = predict_cloud_score(synthetic_l1_dataset, model_under_test, CloudInferParams())
        for var in ["cloud_score", "cloud_label", "feature_raw_fft", "feature_deriv_fft", "unix_t_ns"]:
            assert var in out, f"Missing output variable: {var}"

    def test_output_dtypes(
        self, synthetic_l1_dataset: xr.Dataset, model_under_test: CloudDetection
    ) -> None:
        out = predict_cloud_score(synthetic_l1_dataset, model_under_test, CloudInferParams())
        assert out["cloud_score"].dtype == np.float32
        assert out["cloud_label"].dtype == np.uint8
        assert out["feature_raw_fft"].dtype == np.float32
        assert out["feature_deriv_fft"].dtype == np.float32
        assert out["unix_t_ns"].dtype == np.int64

    def test_output_shape_single_window(
        self, synthetic_l1_dataset: xr.Dataset, model_under_test: CloudDetection
    ) -> None:
        out = predict_cloud_score(synthetic_l1_dataset, model_under_test, CloudInferParams(cadence_s=60.0))
        assert out.sizes["T_l2"] == 1
        assert out["feature_raw_fft"].shape == (1, 32, 32)

    def test_unix_t_ns_monotonic(
        self, multi_window_l1_dataset: xr.Dataset, model_under_test: CloudDetection
    ) -> None:
        out = predict_cloud_score(multi_window_l1_dataset, model_under_test, CloudInferParams(cadence_s=60.0))
        t_ns = out["unix_t_ns"].values
        assert np.all(np.diff(t_ns) >= 0), "unix_t_ns must be monotonic-non-decreasing"

    def test_determinism(
        self, synthetic_l1_dataset: xr.Dataset, model_under_test: CloudDetection
    ) -> None:
        torch.use_deterministic_algorithms(True)
        params = CloudInferParams()
        out1 = predict_cloud_score(synthetic_l1_dataset, model_under_test, params)
        out2 = predict_cloud_score(synthetic_l1_dataset, model_under_test, params)
        np.testing.assert_array_equal(out1["cloud_score"].values, out2["cloud_score"].values)

    def test_requires_median_subtracted(
        self, synthetic_l1_dataset: xr.Dataset, model_under_test: CloudDetection
    ) -> None:
        bad_ds = synthetic_l1_dataset.drop_vars("median_subtracted")
        with pytest.raises(ValueError, match="median_subtracted"):
            predict_cloud_score(bad_ds, model_under_test, CloudInferParams())


# ── MODEL-SPECIFIC TESTS (tied to CloudDetection v1.0; update when retraining) ──


class TestModelArchitecture:
    """Brittle by design: catch accidental architecture drift before the checkpoint ships."""

    def test_input_shape(self) -> None:
        assert CloudDetection.input_shape == (2, 32, 32)

    def test_param_count(self) -> None:
        model = CloudDetection()
        n = sum(p.numel() for p in model.parameters())
        assert n == 109_427, f"Param count changed: {n} (expected 109,427 for v1.0)"

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="checkpoint not present in repo")
    def test_checkpoint_loads_cleanly(self) -> None:
        model = CloudDetection()
        sd = torch.load(CHECKPOINT_PATH, weights_only=True, map_location="cpu")
        result = model.load_state_dict(sd, strict=True)
        assert result.missing_keys == [], f"Missing keys: {result.missing_keys}"
        assert result.unexpected_keys == [], f"Unexpected keys: {result.unexpected_keys}"

    def test_batch_fft_numerics(self, multi_window_l1_dataset: xr.Dataset) -> None:
        """Vectorized FFT must be deterministic across runs (regression guard)."""
        torch.use_deterministic_algorithms(True)
        model = CloudDetection()
        model.eval()
        params = CloudInferParams(cadence_s=60.0)
        out1 = predict_cloud_score(multi_window_l1_dataset, model, params)
        out2 = predict_cloud_score(multi_window_l1_dataset, model, params)
        np.testing.assert_array_equal(out1["feature_raw_fft"].values, out2["feature_raw_fft"].values)
        np.testing.assert_array_equal(out1["feature_deriv_fft"].values, out2["feature_deriv_fft"].values)
        assert out1.sizes["T_l2"] == 2

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="checkpoint not present in repo")
    def test_cloud_score_range(self, synthetic_l1_dataset: xr.Dataset) -> None:
        """With the trained checkpoint, scores must be in [0, 1] (softmax guarantee on finite inputs)."""
        model = CloudDetection()
        model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True, map_location="cpu"))
        model.eval()
        out = predict_cloud_score(synthetic_l1_dataset, model, CloudInferParams())
        scores = out["cloud_score"].values
        assert not np.any(np.isnan(scores)), "Trained model produced NaN scores on valid data"
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="checkpoint not present in repo")
    def test_checkpoint_param_count(self) -> None:
        model = CloudDetection()
        model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True, map_location="cpu"))
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 109_427, f"Param count mismatch: got {n_params}, expected 109,427"
