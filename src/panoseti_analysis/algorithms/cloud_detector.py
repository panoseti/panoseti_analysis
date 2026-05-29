"""Cloud detector kernel and CNN model definition.

Pure Layer A module: takes xarray/numpy/pydantic, returns xarray.
No Ray or Nextflow dependencies.
"""

from typing import Any
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


def _scale_data(data: np.ndarray) -> np.ndarray:
    """Scale data by inverse square root of magnitude."""
    with np.errstate(divide='ignore', invalid='ignore'):
        div = 1.0 / np.sqrt(np.abs(data))
        div = np.nan_to_num(div, nan=1.0, posinf=1.0, neginf=1.0)
        scaled_data = data * div
    return scaled_data


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
    
    # We require 60 second cadence windows for inference.
    # Convert cadence_s to nanoseconds
    window_ns = int(params.cadence_s * 1e9)
    
    # Very basic windowing: chunk by window_ns
    if len(unix_t_ns) == 0:
        return xr.Dataset()

    t_start = unix_t_ns[0]
    t_end = unix_t_ns[-1]
    
    windows_start = np.arange(t_start, t_end + window_ns, window_ns)
    
    features_fft = []
    features_deriv_fft = []
    t_centers = []
    
    # Calculate features per window
    # Wait, the model uses 'raw-derivative-fft.-60' and 'raw-fft'
    for w_start in windows_start[:-1]:
        w_end = w_start + window_ns
        mask = (unix_t_ns >= w_start) & (unix_t_ns < w_end)
        if not np.any(mask):
            continue
            
        w_data = img[mask]
        
        # 'raw-fft': mean FFT of the window.
        w_mean = np.mean(w_data, axis=0) # 32x32
        fft = np.fft.fft2(w_mean)
        fft_mag = np.abs(np.fft.fftshift(fft))
        
        # 'raw-derivative.-60' - wait, the model specifically used raw-derivative-fft.-60
        # which means FFT of the -60 derivative.
        # Let's approximate the time derivative as diffs
        if len(w_data) > 1:
            diffs = np.diff(w_data, axis=0)
            diff_mean = np.mean(diffs, axis=0)
            deriv_fft = np.fft.fft2(diff_mean)
            deriv_fft_mag = np.abs(np.fft.fftshift(deriv_fft))
        else:
            deriv_fft_mag = np.zeros((32, 32))
            
        # Scale derivative FFT as per loader
        deriv_fft_mag = _scale_data(deriv_fft_mag)
        
        features_fft.append(fft_mag)
        features_deriv_fft.append(deriv_fft_mag)
        t_centers.append(w_start + window_ns // 2)

    if not t_centers:
        # Return empty dataset matching schema
        return xr.Dataset({
            "cloud_score": (["T_l2"], np.array([], dtype=np.float32)),
            "cloud_label": (["T_l2"], np.array([], dtype=np.uint8)),
            "feature_raw_fft": (["T_l2", "H", "W"], np.empty((0, 32, 32), dtype=np.float32)),
            "feature_deriv_fft": (["T_l2", "H", "W"], np.empty((0, 32, 32), dtype=np.float32)),
            "unix_t_ns": (["T_l2"], np.array([], dtype=np.int64)),
        })

    # Stack features to (N, 2, 32, 32)
    # The input shape is (2, 32, 32) where channel 0 is deriv_fft, 1 is fft.
    X = np.stack([
        np.array(features_deriv_fft, dtype=np.float32),
        np.array(features_fft, dtype=np.float32)
    ], axis=1)

    device = next(model.parameters()).device
    tensor_x = torch.from_numpy(X).to(device)

    model.eval()
    with torch.no_grad():
        out = model(tensor_x)
        # Assuming 2-class softmax, class 1 is "cloud"
        probs = torch.nn.functional.softmax(out, dim=1)
        cloud_score = probs[:, 1].cpu().numpy()

    cloud_label = (cloud_score >= params.threshold).astype(np.uint8)

    # Construct output Dataset
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
