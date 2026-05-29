"""Cloud detector kernel and CNN model definition.

Pure Layer A module: takes xarray/numpy/pydantic, returns xarray.
No Ray or Nextflow dependencies.
"""

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from panoseti_analysis.config.models import CloudInferParams


class CloudDetection(nn.Module):
    """The CNN architecture trained for cloud detection."""

    input_shape = (2, 32, 32)

    def __init__(self) -> None:
        super().__init__()

        conv1_groups = 2
        conv1_nker = 28
        conv1_kernel_size = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_shape[0], conv1_nker, conv1_kernel_size, stride=1, padding='same', groups=conv1_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(conv1_nker, conv1_nker, conv1_kernel_size, stride=1, padding='same', groups=conv1_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(conv1_nker, conv1_nker, conv1_kernel_size, stride=1, padding='same', groups=conv1_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(conv1_nker, conv1_nker, conv1_kernel_size, stride=1, padding='same', groups=conv1_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(p=0.5)
        )

        conv2_groups = 1
        conv2_nker = 27
        conv2_kernel_size = 5
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_nker, conv2_nker, conv2_kernel_size, stride=1, padding='same', groups=conv2_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv2_nker),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(conv2_nker, conv2_nker, conv2_kernel_size, stride=1, padding='same', groups=conv2_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv2_nker),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(conv2_nker, conv2_nker, conv2_kernel_size, stride=1, padding='same', groups=conv2_groups),
            nn.ReLU(),
            nn.BatchNorm2d(conv2_nker),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(p=0.5),
        )

        self.flatten = nn.Flatten()

        self.linear_stack = nn.Sequential(
            nn.Linear(243, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout1d(p=0.5),

            nn.Linear(128, 84),
            nn.ReLU(),
            nn.BatchNorm1d(84),
            nn.Dropout1d(p=0.5),

            nn.Linear(84, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.flatten(out)
        out = self.linear_stack(out)
        return out


def predict_cloud_score(
    ds: xr.Dataset,
    model: torch.nn.Module,
    params: CloudInferParams
) -> xr.Dataset:
    """Run cloud detection inference over an L1 movie-mode dataset.

    Args:
        ds: The L1 Dataset containing `median_subtracted` and `unix_t_ns`.
        model: The loaded CloudDetection PyTorch model.
        params: Inference parameters (cadence, thresholds).

    Returns:
        An L2 Dataset with dimensions T_l2 and cloud_score / cloud_label.
    """
    if "median_subtracted" not in ds.data_vars:
        raise ValueError("Cloud detection requires L1 movie-mode dataset with 'median_subtracted'")

    img = ds["median_subtracted"].values  # (T, H, W)
    unix_t_ns = ds["unix_t_ns"].values

    if len(unix_t_ns) == 0:
        return xr.Dataset()

    t_start = unix_t_ns[0]
    t_end = unix_t_ns[-1]

    # We step by cadence_ns. The integration window lookback is 60s for the derivative.
    cadence_ns = int(params.cadence_s * 1e9)
    window_ns = 60_000_000_000
    n_stack = 10 # 10 frames of 100us = 1ms stacked integration

    target_times = np.arange(t_start, t_end + 1, cadence_ns)

    # Pre-compute 2D Hann window
    hann_1d = np.hanning(32)
    hann_2d = np.outer(hann_1d, hann_1d)
    hann_3d = hann_2d[np.newaxis, :, :]  # (1, H, W) for broadcasting over batch

    def batch_apply_fft(batch: np.ndarray) -> np.ndarray:
        """batch: (N, H, W) float32 → (N, H, W) float32 log-magnitude FFT."""
        windowed = batch * hann_3d
        mag = np.abs(np.fft.fftn(windowed, axes=(-2, -1)))
        shifted = np.fft.fftshift(mag, axes=(-2, -1))
        with np.errstate(divide="ignore"):
            return np.log(shifted).astype(np.float32)

    # Vectorised index computation for all windows
    n_ts = len(unix_t_ns)
    idx_curr_arr = np.searchsorted(unix_t_ns, target_times).clip(0, n_ts - 1)
    t_prev_arr = np.maximum(t_start, target_times - window_ns)
    idx_prev_arr = np.searchsorted(unix_t_ns, t_prev_arr).clip(0, n_ts - 1)

    # Filter out degenerate windows (end <= start; can't happen with clip, but keep guard)
    valid = (idx_curr_arr + 1) <= n_ts  # always true after clip; kept for clarity
    idx_curr_arr = idx_curr_arr[valid]
    idx_prev_arr = idx_prev_arr[valid]
    t_centers = target_times[valid]

    if len(t_centers) == 0:
        return xr.Dataset({
            "cloud_score": (["T_l2"], np.array([], dtype=np.float32)),
            "cloud_label": (["T_l2"], np.array([], dtype=np.uint8)),
            "feature_raw_fft": (["T_l2", "H", "W"], np.empty((0, 32, 32), dtype=np.float32)),
            "feature_deriv_fft": (["T_l2", "H", "W"], np.empty((0, 32, 32), dtype=np.float32)),
            "unix_t_ns": (["T_l2"], np.array([], dtype=np.int64)),
        })

    # Stack n_stack frames for each window index; shape (N, H, W)
    def stack_windows(indices: np.ndarray) -> np.ndarray:
        n = len(indices)
        h, w = img.shape[1], img.shape[2]
        out = np.zeros((n, h, w), dtype=np.float64)
        for i, idx in enumerate(indices):
            end = min(idx + n_stack, n_ts)
            out[i] = img[idx:end].sum(axis=0)
        return out.astype(np.float32)

    curr_imgs = stack_windows(idx_curr_arr)   # (N, H, W)
    prev_imgs = stack_windows(idx_prev_arr)   # (N, H, W)
    diff_imgs = curr_imgs - prev_imgs         # (N, H, W)

    features_fft_arr = batch_apply_fft(curr_imgs)    # (N, H, W)
    features_deriv_fft_arr = batch_apply_fft(diff_imgs)  # (N, H, W)

    X = np.stack([features_deriv_fft_arr, features_fft_arr], axis=1)  # (N, 2, H, W)

    device = next(model.parameters()).device
    tensor_x = torch.from_numpy(X).to(device)

    model.eval()
    with torch.no_grad():
        out = model(tensor_x)
        probs = torch.nn.functional.softmax(out, dim=1)
        cloud_score = probs[:, 1].cpu().numpy()

    cloud_label = (cloud_score >= params.threshold).astype(np.uint8)

    ds_out = xr.Dataset(
        data_vars={
            "cloud_score": (["T_l2"], cloud_score.astype(np.float32)),
            "cloud_label": (["T_l2"], cloud_label),
            "feature_raw_fft": (["T_l2", "H", "W"], features_fft_arr),
            "feature_deriv_fft": (["T_l2", "H", "W"], features_deriv_fft_arr),
            "unix_t_ns": (["T_l2"], t_centers.astype(np.int64)),
        }
    )

    return ds_out
