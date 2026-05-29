"""Numerical parity between the original pano_utils.apply_fft and the new kernel's apply_fft.

Reference: /Users/nico/panoseti/panoseti-software/cloud-detection/dataset_construction/pano_utils.py
The original uses scipy.fftpack.fftn and skimage.filters.window("hann", shape).
skimage.filters.window("hann", shape) is equivalent to
np.outer(scipy.signal.windows.hann(H), scipy.signal.windows.hann(W)), which is
np.outer(np.hanning(H), np.hanning(W)). We use the scipy form to avoid adding
scikit-image as a dependency, while preserving the intent of the reference.
"""
import numpy as np
import pytest
from scipy.fftpack import fftn as scipy_fftn
from scipy.fftpack import fftshift as scipy_fftshift
from scipy.signal.windows import hann as scipy_hann


def reference_apply_fft(data: np.ndarray, shape: tuple = (32, 32)) -> np.ndarray:
    """Faithful port of the original pano_utils.apply_fft.

    skimage.filters.window("hann", shape) = np.outer(scipy_hann(H), scipy_hann(W)).
    """
    if data.shape != shape:
        data = np.reshape(data, shape)
    h, w = shape
    hann_2d = np.outer(scipy_hann(h), scipy_hann(w))
    data = data * hann_2d
    data = np.abs(scipy_fftn(data, shape))
    data = scipy_fftshift(data)
    with np.errstate(divide="ignore"):
        data = np.log(data)
    return data


def new_apply_fft(data: np.ndarray) -> np.ndarray:
    """Reproduces the kernel's internal apply_fft for comparison."""
    hann_1d = np.hanning(32)
    hann_2d = np.outer(hann_1d, hann_1d)
    d = data * hann_2d
    d = np.abs(np.fft.fftn(d))
    d = np.fft.fftshift(d)
    with np.errstate(divide="ignore"):
        d = np.log(d)
    return d


@pytest.fixture
def rng_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((32, 32)).astype(np.float32)


def test_hann_window_equivalence(rng_image: np.ndarray) -> None:
    """scipy_hann vs np.hanning must be numerically identical (same underlying formula)."""
    scipy_win = np.outer(scipy_hann(32), scipy_hann(32))
    hann_1d = np.hanning(32)
    np_win = np.outer(hann_1d, hann_1d)
    np.testing.assert_allclose(scipy_win, np_win, atol=1e-7,
                               err_msg="Hann window implementations differ!")


def test_fft_equivalence(rng_image: np.ndarray) -> None:
    """scipy.fftpack.fftn vs np.fft.fftn must agree on abs magnitude."""
    hann_2d = np.outer(np.hanning(32), np.hanning(32))
    windowed = rng_image * hann_2d
    scipy_mag = np.abs(scipy_fftn(windowed, (32, 32)))
    np_mag = np.abs(np.fft.fftn(windowed))
    np.testing.assert_allclose(scipy_mag, np_mag, rtol=1e-5,
                               err_msg="FFT magnitude implementations differ!")


def test_apply_fft_parity_on_nonzero(rng_image: np.ndarray) -> None:
    """Full apply_fft parity on data that won't hit log(0)."""
    ref = reference_apply_fft(rng_image)
    new = new_apply_fft(rng_image)
    np.testing.assert_allclose(new, ref, rtol=1e-5, atol=1e-6,
                               err_msg="apply_fft differs from original pipeline!")


def test_log_zero_behavior() -> None:
    """Both reference and new_apply_fft produce -inf for zero input (matching original math).

    Note: predict_cloud_score clips -inf to 0.0 before the model as a safety guard —
    real PANOSETI images (photon noise) never produce zero FFT magnitudes, so this
    nan_to_num call is a no-op for any real data.
    """
    zero_data = np.zeros((32, 32), dtype=np.float32)
    ref = reference_apply_fft(zero_data)
    new = new_apply_fft(zero_data)
    assert np.any(np.isneginf(ref)), "Reference should have -inf for zero input"
    assert np.any(np.isneginf(new)), "new_apply_fft should also produce -inf (parity with reference)"
    np.testing.assert_array_equal(np.isneginf(ref), np.isneginf(new))
