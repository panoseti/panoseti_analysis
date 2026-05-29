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

    features_fft = []
    features_deriv_fft = []
    t_centers = []

    # Pre-compute 2D Hann window
    hann_1d = np.hanning(32)
    hann_2d = np.outer(hann_1d, hann_1d)

    def apply_fft(data: np.ndarray) -> np.ndarray:
        d = data * hann_2d
        d = np.abs(np.fft.fftn(d))
        d = np.fft.fftshift(d)
        with np.errstate(divide="ignore"):
            d = np.log(d)
        return d

    for t in target_times:
        # Find index for current time t
        idx_curr = np.searchsorted(unix_t_ns, t)
        idx_curr = min(idx_curr, len(unix_t_ns) - 1)

        # Find index for t - 60s
        t_prev = max(t_start, t - window_ns)
        idx_prev = np.searchsorted(unix_t_ns, t_prev)
        idx_prev = min(idx_prev, len(unix_t_ns) - 1)

        # Stack 10 frames (1ms) at idx_curr and idx_prev
        end_curr = min(idx_curr + n_stack, len(unix_t_ns))
        end_prev = min(idx_prev + n_stack, len(unix_t_ns))

        if end_curr <= idx_curr:
            continue

        curr_img = np.sum(img[idx_curr:end_curr], axis=0)

        prev_img = np.sum(img[idx_prev:end_prev], axis=0) if end_prev > idx_prev else curr_img

        diff_img = curr_img - prev_img

        fft_mag = apply_fft(curr_img)
        deriv_fft_mag = apply_fft(diff_img)

        features_fft.append(fft_mag)
        features_deriv_fft.append(deriv_fft_mag)
        t_centers.append(t)

    if not t_centers:
        return xr.Dataset({
            "cloud_score": (["T_l2"], np.array([], dtype=np.float32)),
            "cloud_label": (["T_l2"], np.array([], dtype=np.uint8)),
            "feature_raw_fft": (["T_l2", "H", "W"], np.empty((0, 32, 32), dtype=np.float32)),
            "feature_deriv_fft": (["T_l2", "H", "W"], np.empty((0, 32, 32), dtype=np.float32)),
            "unix_t_ns": (["T_l2"], np.array([], dtype=np.int64)),
        })

    X = np.stack([
        np.array(features_deriv_fft, dtype=np.float32),
        np.array(features_fft, dtype=np.float32)
    ], axis=1)

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
            "cloud_score": (["T_l2"], cloud_score),
            "cloud_label": (["T_l2"], cloud_label),
            "feature_raw_fft": (["T_l2", "H", "W"], np.array(features_fft, dtype=np.float32)),
            "feature_deriv_fft": (["T_l2", "H", "W"], np.array(features_deriv_fft, dtype=np.float32)),
            "unix_t_ns": (["T_l2"], np.array(t_centers, dtype=np.int64)),
        }
    )

    return ds_out
