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
            nn.Conv2d(
                self.input_shape[0],
                conv1_nker,
                conv1_kernel_size,
                stride=1,
                padding="same",
                groups=conv1_groups,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(
                conv1_nker,
                conv1_nker,
                conv1_kernel_size,
                stride=1,
                padding="same",
                groups=conv1_groups,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(
                conv1_nker,
                conv1_nker,
                conv1_kernel_size,
                stride=1,
                padding="same",
                groups=conv1_groups,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(
                conv1_nker,
                conv1_nker,
                conv1_kernel_size,
                stride=1,
                padding="same",
                groups=conv1_groups,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(conv1_nker),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(p=0.5),
        )

        conv2_groups = 1
        conv2_nker = 27
        conv2_kernel_size = 5
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                conv1_nker,
                conv2_nker,
                conv2_kernel_size,
                stride=1,
                padding="same",
                groups=conv2_groups,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(conv2_nker),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(
                conv2_nker,
                conv2_nker,
                conv2_kernel_size,
                stride=1,
                padding="same",
                groups=conv2_groups,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(conv2_nker),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(
                conv2_nker,
                conv2_nker,
                conv2_kernel_size,
                stride=1,
                padding="same",
                groups=conv2_groups,
            ),
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


def _batch_apply_fft(batch: np.ndarray, hann_3d: np.ndarray) -> np.ndarray:
    """batch: (N, H, W) float32 → (N, H, W) float32 log-magnitude FFT."""
    windowed = batch * hann_3d
    mag = np.abs(np.fft.fftn(windowed, axes=(-2, -1)))
    shifted = np.fft.fftshift(mag, axes=(-2, -1))
    with np.errstate(divide="ignore"):
        log_mag = np.log(shifted)
    # Real PANOSETI images (photon noise) never produce zero FFT magnitudes.
    # nan_to_num is a no-op for real data; it guards synthetic/pathological inputs.
    return np.nan_to_num(log_mag, neginf=0.0).astype(np.float32)


def _window_sums(
    sub: np.ndarray, needed: np.ndarray, indices: np.ndarray, n_stack: int, n_ts: int
) -> np.ndarray:
    """Sum ``n_stack`` frames per window from pre-gathered frames.

    ``sub`` holds exactly the frames at row indices ``needed`` (sorted, unique). For each
    window start ``idx`` in ``indices`` this returns ``sum(img[idx : min(idx+n_stack, n_ts)])``
    — bit-identical to the old contiguous slice-sum, but it only ever touches the frames the
    windows actually reference, so the caller never materialises the whole store.

    Windows are grouped by their valid-frame count ``k`` so each group sums *exactly* ``k``
    frames (no zero-padding): padding with zeros would change numpy's pairwise-summation tree
    for truncated tail windows and break bit-equivalence with the original kernel.
    """
    n = len(indices)
    h, w = sub.shape[1], sub.shape[2]
    out = np.empty((n, h, w), dtype=np.float32)
    rows = indices[:, None] + np.arange(n_stack)  # (N, n_stack) absolute frame indices
    counts = np.minimum(indices + n_stack, n_ts) - indices  # valid frames per window (>=1)
    for k in np.unique(counts):
        sel = counts == k
        pos = np.searchsorted(needed, rows[sel, : int(k)])  # (m, k) positions into ``sub``
        out[sel] = sub[pos].sum(axis=1).astype(np.float32)  # sum exactly k contiguous frames
    return out  # (N, H, W)


def extract_cloud_features(
    ds: xr.Dataset,
    params: CloudInferParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 2-channel log-FFT features for cloud detection.

    Returns:
        X: float32 array of shape (N, 2, H, W) — [deriv_fft, raw_fft] channels
        t_centers: int64 array of shape (N,) — center timestamps in nanoseconds
    """
    da = ds["median_subtracted"]  # lazy (T, H, W) — only the referenced frames are loaded below
    unix_t_ns = ds["unix_t_ns"].values

    t_start = unix_t_ns[0]
    t_end = unix_t_ns[-1]

    cadence_ns = int(params.cadence_s * 1e9)
    window_ns = int(params.window_s * 1e9)
    n_stack = params.n_stack

    target_times = np.arange(t_start, t_end + 1, cadence_ns)

    # Pre-compute 2D Hann window
    hann_1d = np.hanning(32)
    hann_2d = np.outer(hann_1d, hann_1d)
    hann_3d = hann_2d[np.newaxis, :, :]  # (1, H, W) for broadcasting over batch

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
        return np.empty((0, 2, 32, 32), dtype=np.float32), np.array([], dtype=np.int64)

    # Gather only the frames the windows actually reference, instead of loading the whole
    # (potentially multi-TB) store. Both curr and prev windows stack n_stack contiguous
    # frames from their start index (truncated at n_ts).
    rows_curr = idx_curr_arr[:, None] + np.arange(n_stack)
    rows_prev = idx_prev_arr[:, None] + np.arange(n_stack)
    needed = np.unique(np.concatenate([rows_curr.ravel(), rows_prev.ravel()]))
    needed = needed[needed < n_ts]
    time_dim = da.dims[0]  # positional time axis, matching the original .values semantics
    sub = da.isel(
        {time_dim: needed}
    ).values  # (len(needed), H, W) — native dtype, referenced frames only

    curr_imgs = _window_sums(sub, needed, idx_curr_arr, n_stack, n_ts)  # (N, H, W)
    prev_imgs = _window_sums(sub, needed, idx_prev_arr, n_stack, n_ts)  # (N, H, W)
    diff_imgs = curr_imgs - prev_imgs  # (N, H, W)

    features_fft_arr = _batch_apply_fft(curr_imgs, hann_3d)  # (N, H, W)
    features_deriv_fft_arr = _batch_apply_fft(diff_imgs, hann_3d)  # (N, H, W)

    X = np.stack([features_deriv_fft_arr, features_fft_arr], axis=1)  # (N, 2, H, W)

    return X, t_centers.astype(np.int64)


def predict_cloud_score(
    ds: xr.Dataset, model: torch.nn.Module, params: CloudInferParams
) -> xr.Dataset:
    """Run cloud detection inference over an L1 movie-mode dataset.

    Args:
        ds: The L1 Dataset containing `median_subtracted` and `unix_t_ns`.
        model: The loaded CloudDetection PyTorch model.
        params: Inference parameters (cadence, thresholds).

    Returns:
        An L2 Dataset with dimensions T_l2 and cloud_score / cloud_label.
    """
    ds.pano.validate(level="L1", kind="img")

    unix_t_ns = ds["unix_t_ns"].values

    if len(unix_t_ns) == 0:
        return xr.Dataset()

    X, t_centers = extract_cloud_features(ds, params)

    if len(t_centers) == 0:
        return xr.Dataset(
            {
                "cloud_score": (["T_l2"], np.array([], dtype=np.float32)),
                "cloud_label": (["T_l2"], np.array([], dtype=np.uint8)),
                "feature_raw_fft": (["T_l2", "H", "W"], np.empty((0, 32, 32), dtype=np.float32)),
                "feature_deriv_fft": (["T_l2", "H", "W"], np.empty((0, 32, 32), dtype=np.float32)),
                "unix_t_ns": (["T_l2"], np.array([], dtype=np.int64)),
            }
        )

    # X shape: (N, 2, H, W) — channel 0 = deriv_fft, channel 1 = raw_fft
    features_deriv_fft_arr = X[:, 0, :, :]  # (N, H, W)
    features_fft_arr = X[:, 1, :, :]  # (N, H, W)

    # device = next(model.parameters()).device
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else torch.device("cpu")
    )
    model = model.to(device)
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
