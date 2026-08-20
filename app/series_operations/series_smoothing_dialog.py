"""Chart-series smoothing dialog.

This module is intended to be a drop-in companion to ``dialog_series_interpolate``.
It keeps the same high-level construction pattern::

    dialog = SeriesSmoothingDialog(repo, figure_id, parent=self)

The dialog itself shows the same selection concept as the interpolation dialog:
axis list on the left, series list below it, and method/settings on the right.
The selected ``figure_id`` is the data context; the selected axis only scopes the
series shown in the dialog and is returned in metadata/callbacks.
"""

from __future__ import annotations

import math
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from app.data.data_source import parse_roles, row_value
from app.widgets.axis_series_selector_widget import AxisSeriesSelector

from scipy import fft
from scipy.interpolate import (
    RBFInterpolator,
    RectBivariateSpline,
    SmoothBivariateSpline,
    UnivariateSpline,
)
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter
from scipy.signal import butter, medfilt, savgol_filter, wiener

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.series_operation_dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.styles.style import (
    create_card_widget,
    stdSizeAndlayout,
)
from app.utils.i18n import _

from scipy.signal import sosfiltfilt

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    from statsmodels.tsa.filters.hp_filter import hpfilter
except ImportError:  # pragma: no cover
    sm_lowess = None
    hpfilter = None

try:
    from skimage.restoration import denoise_tv_chambolle
except ImportError:  # pragma: no cover
    denoise_tv_chambolle = None

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None


def _require_dependency(dependency: Any, name: str) -> Any:
    if dependency is None:
        applogger.error(
            f"Smoothing method requires optional dependency '{name}', "
            "which is not installed in the current Python environment."
        )
    return dependency


# ---------------------------------------------------------------------------
# Method constants and documentation links
# ---------------------------------------------------------------------------

DIM_1D = "1D series"
DIM_2D = "2D surface / XYZ"
DIM_3D = "3D volume / XYZW"

SMOOTH_MOVING_AVERAGE = "Moving Average"
SMOOTH_SAVGOL = "Savitzky-Golay Filter (SciPy)"
SMOOTH_GAUSSIAN = "Gaussian Filter (SciPy)"
SMOOTH_MEDIAN = "Median Filter (SciPy)"
SMOOTH_WIENER = "Wiener Filter (SciPy)"
SMOOTH_SPLINE = "Smoothing Spline (SciPy)"
SMOOTH_LOWESS = "LOWESS / LOESS"
SMOOTH_KALMAN = "Kalman Smoother"
SMOOTH_FFT = "FFT Low-Pass Filter"
SMOOTH_BUTTERWORTH = "Butterworth Filter (SciPy)"
SMOOTH_WAVELET = "Wavelet Denoising"
SMOOTH_WHITTAKER = "Whittaker-Eilers Smoother"
SMOOTH_HP = "Hodrick-Prescott Filter"
SMOOTH_TV = "Total Variation Denoising"

SMOOTH2D_GAUSSIAN = "2D Gaussian Surface Filter"
SMOOTH2D_MEDIAN = "2D Median Surface Filter"
SMOOTH2D_SPLINE = "2D SmoothBivariateSpline"
SMOOTH2D_RECT_SPLINE = "2D RectBivariateSpline"
SMOOTH2D_RBF = "2D RBF Surface Smoothing"
SMOOTH2D_TV = "2D Total Variation Surface Denoising"

SMOOTH3D_GAUSSIAN = "3D Gaussian Volume Filter"
SMOOTH3D_MEDIAN = "3D Median Volume Filter"
SMOOTH3D_FFT = "3D FFT Low-Pass Volume Filter"
SMOOTH3D_RBF = "3D RBF Volume Smoothing"
SMOOTH3D_TV = "3D Total Variation Volume Denoising"



# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SeriesChoice:
    """Selectable series descriptor loaded from the repository."""

    name: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None
    values: np.ndarray | None = None
    source: Any | None = None


@dataclass(slots=True)
class SmoothResult:
    """Result returned by the smoothing dialog."""

    source_name: str
    result_name: str
    method: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None
    values: np.ndarray | None
    metadata: dict[str, Any]



# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------




def _odd_window(value: int, n_values: int, minimum: int = 3) -> int:
    """Return an odd integer window length within the available data size."""
    window = max(minimum, int(value))
    if window % 2 == 0:
        window += 1
    if window > n_values:
        window = n_values if n_values % 2 == 1 else n_values - 1
    return max(minimum, window)


def _moving_average(y_values: np.ndarray, window: int, centered: bool) -> np.ndarray:
    """Simple moving average with explicit ndarray casts for strict type checkers."""
    safe_window = max(1, int(window))
    kernel = np.ones(safe_window, dtype=float) / float(safe_window)

    if centered:
        return np.asarray(np.convolve(y_values, kernel, mode="same"), dtype=float)

    output = np.asarray(
        np.convolve(y_values, kernel, mode="full")[: y_values.size],
        dtype=float,
    )
    if safe_window > 1:
        output[: safe_window - 1] = output[min(safe_window - 1, output.size - 1)]
    return output


def _fft_low_pass_1d(y_values: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """Low-pass smooth a 1D signal in the frequency domain."""
    ratio = float(np.clip(cutoff_ratio, 0.001, 0.999))
    mean = float(np.nanmean(y_values))
    spectrum = np.asarray(cast(Any, fft.rfft)(y_values - mean), dtype=complex)
    frequencies = np.asarray(cast(Any, fft.rfftfreq)(y_values.size), dtype=float)
    spectrum[frequencies > ratio * float(np.max(frequencies))] = 0.0
    filtered = cast(Any, fft.irfft)(spectrum, n=y_values.size)
    return np.asarray(filtered, dtype=float) + mean


def _fft_low_pass_nd(values: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """Low-pass smooth an N-dimensional grid in the frequency domain."""
    data = np.asarray(values, dtype=float)
    ratio = float(np.clip(cutoff_ratio, 0.001, 0.999))
    mean = float(np.nanmean(data))
    spectrum = np.asarray(cast(Any, fft.fftn)(data - mean), dtype=complex)
    frequency_grids = cast(
        list[np.ndarray],
        np.meshgrid(
            *[np.asarray(cast(Any, fft.fftfreq)(length), dtype=float) for length in data.shape],
            indexing="ij",
        ),
    )
    radius = np.sqrt(sum(grid * grid for grid in frequency_grids))
    spectrum[radius > ratio * float(np.max(radius))] = 0.0
    inverse = cast(Any, fft.ifftn)(spectrum)
    return np.asarray(np.real(np.asarray(inverse, dtype=complex)), dtype=float) + mean


def _butterworth(
    y_values: np.ndarray,
    cutoff: float,
    fs: float,
    order: int,
    btype: str,
    high_cutoff: float,
) -> np.ndarray:
    """Apply a zero-phase Butterworth filter."""
    wn: float | list[float]
    if btype in {"bandpass", "bandstop"}:
        wn = [float(cutoff), float(high_cutoff)]
    else:
        wn = float(cutoff)

    sos = butter(
        max(1, int(order)),
        wn,
        btype=btype,
        fs=max(float(fs), 1e-12),
        output="sos",
    )
    return cast(Any, sosfiltfilt)(sos, y_values)


def _wavelet_denoise(
    y_values: np.ndarray,
    wavelet: str,
    level: int,
    threshold_factor: float,
    mode: str,
) -> np.ndarray:
    """Denoise using discrete wavelet thresholding."""
    pywt_module = _require_dependency(pywt, "PyWavelets")
    coeffs = pywt_module.wavedec(
        y_values,
        wavelet=wavelet,
        mode="symmetric",
        level=level or None,
    )
    detail_coeffs = coeffs[1:]
    if not detail_coeffs:
        return y_values.copy()

    sigma = 0.0
    if detail_coeffs[-1].size:
        sigma = float(np.median(np.abs(detail_coeffs[-1])) / 0.6745)
    threshold = threshold_factor * sigma * math.sqrt(2.0 * math.log(max(y_values.size, 2)))
    new_coeffs = [coeffs[0]] + [
        pywt_module.threshold(coeff, threshold, mode=mode) for coeff in detail_coeffs
    ]
    output = pywt_module.waverec(new_coeffs, wavelet=wavelet, mode="symmetric")
    return np.asarray(output[: y_values.size], dtype=float)


def _whittaker_eilers(
    y_values: np.ndarray,
    lam: float,
    diff_order: int,
) -> np.ndarray:
    """Dense Whittaker-Eilers smoother for chart-sized series."""
    n_values = y_values.size
    identity = np.eye(n_values)
    difference = np.diff(identity, n=max(1, int(diff_order)), axis=0)
    system = identity + max(float(lam), 0.0) * difference.T @ difference
    return np.linalg.solve(system, y_values)


def _kalman_smoother_1d(
    y_values: np.ndarray,
    process_variance: float,
    measurement_variance: float,
    initial_covariance: float,
) -> np.ndarray:
    """Smooth a 1D signal with a local-level Kalman filter plus RTS pass.

    The state is a single latent level.  The forward pass removes measurement
    noise; the backward Rauch-Tung-Striebel pass reduces lag and produces a
    true smoother rather than only a filter.
    """
    values = np.asarray(y_values, dtype=float).reshape(-1)
    if values.size == 0:
        return values.copy()

    q_value = max(float(process_variance), 1e-12)
    r_value = max(float(measurement_variance), 1e-12)
    p0_value = max(float(initial_covariance), 1e-12)

    filtered = np.empty_like(values, dtype=float)
    predicted = np.empty_like(values, dtype=float)
    filter_cov = np.empty_like(values, dtype=float)
    predict_cov = np.empty_like(values, dtype=float)

    state = float(values[0])
    covariance = p0_value

    for index, observation in enumerate(values):
        predicted[index] = state
        predict_cov[index] = covariance + q_value

        gain = predict_cov[index] / (predict_cov[index] + r_value)
        state = predicted[index] + gain * (float(observation) - predicted[index])
        covariance = (1.0 - gain) * predict_cov[index]

        filtered[index] = state
        filter_cov[index] = covariance

    smoothed = filtered.copy()
    for index in range(values.size - 2, -1, -1):
        gain = filter_cov[index] / max(predict_cov[index + 1], 1e-12)
        smoothed[index] = filtered[index] + gain * (
            smoothed[index + 1] - predicted[index + 1]
        )

    return smoothed


# ---------------------------------------------------------------------------
# Public smoothing functions
# ---------------------------------------------------------------------------

def smooth_1d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    method: str,
    params: Mapping[str, Any],
) -> np.ndarray|None:
    """Smooth one XY series and return smoothed Y values."""
    x_clean, y_clean = _clean_xy(x_values, y_values)
    n_values = y_clean.size

    if method == SMOOTH_MOVING_AVERAGE:
        return _moving_average(
            y_clean,
            int(params.get("window", 7)),
            bool(params.get("centered", True)),
        )

    if method == SMOOTH_SAVGOL:
        window = _odd_window(int(params.get("window", 7)), n_values)
        polyorder = min(int(params.get("polyorder", 2)), window - 1)
        return np.asarray(
            savgol_filter(
                y_clean,
                window_length=window,
                polyorder=polyorder,
                deriv=int(params.get("deriv", 0)),
                delta=float(params.get("delta", 1.0)),
                mode=str(params.get("savgol_mode", params.get("mode", "interp"))),
            ),
            dtype=float,
        )

    if method == SMOOTH_GAUSSIAN:
        return gaussian_filter1d(
            y_clean,
            sigma=float(params.get("sigma", 2.0)),
            mode=str(params.get("mode", "nearest")),
            truncate=float(params.get("truncate", 4.0)),
        )

    if method == SMOOTH_MEDIAN:
        kernel = _odd_window(int(params.get("kernel", 5)), n_values)
        return medfilt(y_clean, kernel_size=kernel)

    if method == SMOOTH_WIENER:
        noise = params.get("noise")
        return wiener(
            y_clean,
            mysize=_odd_window(int(params.get("window", 7)), n_values),
            noise=None if noise in (None, 0.0, "") else float(noise),
        )

    if method == SMOOTH_SPLINE:
        spline_order = int(np.clip(int(params.get("spline_k", 3)), 1, min(5, n_values - 1)))
        spline = UnivariateSpline(
            x_clean,
            y_clean,
            s=float(params.get("spline_s", max(n_values, 1))),
            k=spline_order,
        )
        return np.asarray(spline(x_clean), dtype=float)

    if method == SMOOTH_LOWESS:
        sm_lowess_fn = _require_dependency(sm_lowess, "statsmodels")
        return sm_lowess_fn(
            y_clean,
            x_clean,
            frac=float(params.get("lowess_frac", 0.25)),
            it=int(params.get("lowess_it", 3)),
            return_sorted=False,
        )

    if method == SMOOTH_KALMAN:
        return _kalman_smoother_1d(
            y_clean,
            process_variance=float(params.get("kalman_process_variance", 1e-4)),
            measurement_variance=float(params.get("kalman_measurement_variance", 1e-2)),
            initial_covariance=float(params.get("kalman_initial_covariance", 1.0)),
        )

    if method == SMOOTH_FFT:
        return _fft_low_pass_1d(y_clean, float(params.get("fft_cutoff_ratio", 0.20)))

    if method == SMOOTH_BUTTERWORTH:
        return _butterworth(
            y_clean,
            cutoff=float(params.get("butter_cutoff", 0.20)),
            fs=float(params.get("butter_fs", 1.0)),
            order=int(params.get("butter_order", 4)),
            btype=str(params.get("butter_type", "lowpass")),
            high_cutoff=float(params.get("butter_high_cutoff", 0.40)),
        )

    if method == SMOOTH_WAVELET:
        _require_dependency(pywt, "PyWavelets")
        return _wavelet_denoise(
            y_clean,
            wavelet=str(params.get("wavelet", "db4")),
            level=int(params.get("wavelet_level", 0)),
            threshold_factor=float(params.get("wavelet_threshold_factor", 1.0)),
            mode=str(params.get("wavelet_threshold_mode", "soft")),
        )

    if method == SMOOTH_WHITTAKER:
        return _whittaker_eilers(
            y_clean,
            lam=float(params.get("whittaker_lambda", 1000.0)),
            diff_order=int(params.get("whittaker_order", 2)),
        )

    if method == SMOOTH_HP:
        hpfilter_fn = _require_dependency(hpfilter, "statsmodels")
        _cycle, trend = cast(Any, hpfilter_fn)(y_clean, lamb=float(params.get("hp_lambda", 1600.0)))
        return np.asarray(trend, dtype=float)

    if method == SMOOTH_TV:
        denoise_fn = _require_dependency(denoise_tv_chambolle, "scikit-image")
        return np.asarray(
            denoise_fn(
                y_clean,
                weight=float(params.get("tv_weight", 0.15)),
            ),
            dtype=float,
        )

    applogger.error(f"Unsupported 1D smoothing method: {method}")


# Data Cleaners
@staticmethod
def _clean_xy(x_values: Any, y_values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite 1D XY arrays sorted by X."""
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)

    if x.size != y.size:
        applogger.error("X and Y must have the same length.")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 3:
        applogger.error("At least 3 finite points are required.")

    order = np.argsort(x)
    return x[order], y[order]

@staticmethod
def _clean_xyzw_volume(
    x_values: Any,
    y_values: Any,
    z_values: Any,
    raw_values: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return 3D volume/grid or scattered XYZW arrays."""
    if z_values is None or raw_values is None:
        applogger.error("3D smoothing requires X, Y, Z and values.")

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    z = np.asarray(z_values, dtype=float)
    values = np.asarray(raw_values, dtype=float)

    if values.ndim == 3:
        return x.reshape(-1), y.reshape(-1), z.reshape(-1), values

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    values = values.reshape(-1)

    if not (x.size == y.size == z.size == values.size):
        applogger.error("Scattered X, Y, Z and value arrays must match.")

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(values)
    return x[mask], y[mask], z[mask], values[mask]

@staticmethod
def _clean_xyz_surface(
    x_values: Any,
    y_values: Any,
    z_values: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 2D surface/grid or scattered XYZ arrays."""
    if z_values is None:
        applogger.error("2D smoothing requires Z values.")

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    z = np.asarray(z_values, dtype=float)

    if z.ndim == 2:
        return x.reshape(-1), y.reshape(-1), z

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    if not (x.size == y.size == z.size):
        applogger.error("Scattered X, Y and Z arrays must have the same length.")

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[mask], y[mask], z[mask]


def smooth_2d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    method: str,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]|None:
    """Smooth a 2D surface or scattered XYZ field."""
    x_clean, y_clean, z_clean = _clean_xyz_surface(x_values, y_values, z_values)

    if method == SMOOTH2D_GAUSSIAN:
        if z_clean.ndim != 2:
            applogger.error("2D Gaussian expects gridded Z data.")
            return None
        return x_clean, y_clean, gaussian_filter(
            z_clean,
            sigma=(
                float(params.get("sigma_y", 1.5)),
                float(params.get("sigma_x", 1.5)),
            ),
            mode=str(params.get("mode", "nearest")),
            truncate=float(params.get("truncate", 4.0)),
        )

    if method == SMOOTH2D_MEDIAN:
        if z_clean.ndim != 2:
            applogger.error("2D median expects gridded Z data.")
        kernel_y = _odd_window(int(params.get("kernel_y", 3)), z_clean.shape[0])
        kernel_x = _odd_window(int(params.get("kernel_x", 3)), z_clean.shape[1])
        return x_clean, y_clean, median_filter(
            z_clean,
            size=(kernel_y, kernel_x),
            mode=str(params.get("mode", "nearest")),
        )

    if method == SMOOTH2D_SPLINE:
        if z_clean.ndim == 2:
            xx_mesh, yy_mesh = np.meshgrid(x_clean, y_clean)
            xs = xx_mesh.reshape(-1)
            ys = yy_mesh.reshape(-1)
            zs = z_clean.reshape(-1)
        else:
            xs = x_clean.reshape(-1)
            ys = y_clean.reshape(-1)
            zs = z_clean.reshape(-1)
        grid_x = np.unique(xs)
        grid_y = np.unique(ys)
        spline = SmoothBivariateSpline(
            xs,
            ys,
            zs,
            s=float(params.get("spline_s", len(zs))),
            kx=int(params.get("kx", 3)),
            ky=int(params.get("ky", 3)),
        )
        return grid_x, grid_y, spline(grid_x, grid_y).T

    if method == SMOOTH2D_RECT_SPLINE:
        if z_clean.ndim != 2:
            applogger.error("RectBivariateSpline expects gridded data.")
        z_input = z_clean.T if z_clean.shape == (y_clean.size, x_clean.size) else z_clean
        spline = cast(Any, RectBivariateSpline)(
            x_clean,
            y_clean,
            z_input,
            kx=int(params.get("kx", 3)),
            ky=int(params.get("ky", 3)),
            s=float(params.get("spline_s", 0.0)),
        )
        return x_clean, y_clean, spline(x_clean, y_clean).T

    if method == SMOOTH2D_RBF:
        if z_clean.ndim == 2:
            xx_mesh, yy_mesh = np.meshgrid(x_clean, y_clean)
            points = np.column_stack([xx_mesh.reshape(-1), yy_mesh.reshape(-1)])
            values = z_clean.reshape(-1)
            grid_x = x_clean
            grid_y = y_clean
        else:
            points = np.column_stack([x_clean.reshape(-1), y_clean.reshape(-1)])
            values = z_clean.reshape(-1)
            grid_x = np.unique(x_clean)
            grid_y = np.unique(y_clean)
        epsilon = params.get("rbf_epsilon") or None
        neighbors = int(params.get("rbf_neighbors", 0)) or None
        rbf = RBFInterpolator(
            points,
            values,
            kernel=str(params.get("rbf_kernel", "thin_plate_spline")),
            smoothing=float(params.get("rbf_smoothing", 0.1)),
            epsilon=epsilon,
            neighbors=neighbors,
        )
        xx_eval, yy_eval = np.meshgrid(grid_x, grid_y)
        eval_points = np.column_stack([xx_eval.reshape(-1), yy_eval.reshape(-1)])
        return grid_x, grid_y, rbf(eval_points).reshape(xx_eval.shape)

    if method == SMOOTH2D_TV:
        if z_clean.ndim != 2:
            applogger.error("2D total variation expects gridded data.")
        denoise_fn = _require_dependency(denoise_tv_chambolle, "scikit-image")
        return x_clean, y_clean, np.asarray(
            denoise_fn(
                z_clean,
                weight=float(params.get("tv_weight", 0.15)),
            ),
            dtype=float,
        )

    applogger.error(f"Unsupported 2D smoothing method: {method}")


def smooth_3d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    raw_values: np.ndarray,
    method: str,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]|None:
    """Smooth a 3D volume or scattered XYZW field."""
    x_clean, y_clean, z_clean, values = _clean_xyzw_volume(
        x_values,
        y_values,
        z_values,
        raw_values,
    )

    if method == SMOOTH3D_GAUSSIAN:
        if values.ndim != 3:
            applogger.error("3D Gaussian expects gridded volume data.")
        return x_clean, y_clean, z_clean, gaussian_filter(
            values,
            sigma=(
                float(params.get("sigma_z", 1.5)),
                float(params.get("sigma_y", 1.5)),
                float(params.get("sigma_x", 1.5)),
            ),
            mode=str(params.get("mode", "nearest")),
            truncate=float(params.get("truncate", 4.0)),
        )

    if method == SMOOTH3D_MEDIAN:
        if values.ndim != 3:
            applogger.error("3D median expects gridded volume data.")
        kernel_z = _odd_window(int(params.get("kernel_z", 3)), values.shape[0])
        kernel_y = _odd_window(int(params.get("kernel_y", 3)), values.shape[1])
        kernel_x = _odd_window(int(params.get("kernel_x", 3)), values.shape[2])
        return x_clean, y_clean, z_clean, median_filter(
            values,
            size=(kernel_z, kernel_y, kernel_x),
            mode=str(params.get("mode", "nearest")),
        )

    if method == SMOOTH3D_FFT:
        if values.ndim != 3:
            applogger.error("3D FFT expects gridded volume data.")
        return x_clean, y_clean, z_clean, _fft_low_pass_nd(
            values,
            float(params.get("fft_cutoff_ratio", 0.20)),
        )

    if method == SMOOTH3D_RBF:
        if values.ndim == 3:
            zz_mesh, yy_mesh, xx_mesh = np.meshgrid(
                z_clean,
                y_clean,
                x_clean,
                indexing="ij",
            )
            points = np.column_stack(
                [xx_mesh.reshape(-1), yy_mesh.reshape(-1), zz_mesh.reshape(-1)]
            )
            scalar_values = values.reshape(-1)
            grid_x = x_clean
            grid_y = y_clean
            grid_z = z_clean
        else:
            points = np.column_stack(
                [x_clean.reshape(-1), y_clean.reshape(-1), z_clean.reshape(-1)]
            )
            scalar_values = values.reshape(-1)
            grid_x = np.unique(x_clean)
            grid_y = np.unique(y_clean)
            grid_z = np.unique(z_clean)
        epsilon = params.get("rbf_epsilon") or None
        neighbors = int(params.get("rbf_neighbors", 0)) or None
        rbf = RBFInterpolator(
            points,
            scalar_values,
            kernel=str(params.get("rbf_kernel", "thin_plate_spline")),
            smoothing=float(params.get("rbf_smoothing", 0.1)),
            epsilon=epsilon,
            neighbors=neighbors,
        )
        zz_eval, yy_eval, xx_eval = np.meshgrid(
            grid_z,
            grid_y,
            grid_x,
            indexing="ij",
        )
        eval_points = np.column_stack(
            [xx_eval.reshape(-1), yy_eval.reshape(-1), zz_eval.reshape(-1)]
        )
        output = rbf(eval_points).reshape(xx_eval.shape)
        return grid_x, grid_y, grid_z, output

    if method == SMOOTH3D_TV:
        if values.ndim != 3:
            applogger.error("3D total variation expects gridded volume data.")
        denoise_fn = _require_dependency(denoise_tv_chambolle, "scikit-image")
        return x_clean, y_clean, z_clean, np.asarray(
            denoise_fn(
                values,
                weight=float(params.get("tv_weight", 0.15)),
            ),
            dtype=float,
        )

    applogger.error(f"Unsupported 3D smoothing method: {method}")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ModelSpec:
    """UI and documentation metadata for one smoothing model."""

    dimension: str
    doc_title: str
    doc_url: str
    fields: frozenset[str]


def _field_set(*names: str) -> frozenset[str]:
    return frozenset(names)


_MODEL_DEFS: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    (SMOOTH_MOVING_AVERAGE, DIM_1D, "NumPy convolve", "https://numpy.org/doc/stable/reference/generated/numpy.convolve.html", ("window", "centered")),
    (SMOOTH_SAVGOL, DIM_1D, "SciPy savgol_filter", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html", ("window", "polyorder", "deriv", "delta", "savgol_mode")),
    (SMOOTH_GAUSSIAN, DIM_1D, "SciPy gaussian_filter1d", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html", ("sigma", "truncate", "mode")),
    (SMOOTH_MEDIAN, DIM_1D, "SciPy medfilt", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html", ("kernel",)),
    (SMOOTH_WIENER, DIM_1D, "SciPy wiener", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html", ("window", "noise")),
    (SMOOTH_SPLINE, DIM_1D, "SciPy UnivariateSpline", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html", ("spline_s", "spline_k")),
    (SMOOTH_LOWESS, DIM_1D, "Statsmodels LOWESS", "https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html", ("lowess_frac", "lowess_it")),
    (SMOOTH_KALMAN, DIM_1D, "Kalman smoothing", "https://en.wikipedia.org/wiki/Kalman_filter", ("kalman_process_variance", "kalman_measurement_variance", "kalman_initial_covariance")),
    (SMOOTH_FFT, DIM_1D, "SciPy FFT", "https://docs.scipy.org/doc/scipy/reference/fft.html", ("fft_cutoff_ratio",)),
    (SMOOTH_BUTTERWORTH, DIM_1D, "SciPy butter", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html", ("butter_fs", "butter_cutoff", "butter_high_cutoff", "butter_order", "butter_type")),
    (SMOOTH_WAVELET, DIM_1D, "PyWavelets", "https://pywavelets.readthedocs.io/", ("wavelet", "wavelet_level", "wavelet_threshold_factor", "wavelet_threshold_mode")),
    (SMOOTH_WHITTAKER, DIM_1D, "Whittaker smoothing", "https://pybaselines.readthedocs.io/", ("whittaker_lambda", "whittaker_order")),
    (SMOOTH_HP, DIM_1D, "Statsmodels hpfilter", "https://www.statsmodels.org/stable/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html", ("hp_lambda",)),
    (SMOOTH_TV, DIM_1D, "scikit-image total variation", "https://scikit-image.org/docs/stable/api/skimage.restoration.html", ("tv_weight",)),
    (SMOOTH2D_GAUSSIAN, DIM_2D, "SciPy gaussian_filter", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html", ("sigma_x", "sigma_y", "truncate", "mode")),
    (SMOOTH2D_MEDIAN, DIM_2D, "SciPy median_filter", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html", ("kernel_x", "kernel_y", "mode")),
    (SMOOTH2D_SPLINE, DIM_2D, "SciPy SmoothBivariateSpline", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.SmoothBivariateSpline.html", ("spline_s", "kx", "ky")),
    (SMOOTH2D_RECT_SPLINE, DIM_2D, "SciPy RectBivariateSpline", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html", ("spline_s", "kx", "ky")),
    (SMOOTH2D_RBF, DIM_2D, "SciPy RBFInterpolator", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html", ("rbf_kernel", "rbf_smoothing", "rbf_epsilon", "rbf_neighbors")),
    (SMOOTH2D_TV, DIM_2D, "scikit-image total variation", "https://scikit-image.org/docs/stable/api/skimage.restoration.html", ("tv_weight",)),
    (SMOOTH3D_GAUSSIAN, DIM_3D, "SciPy gaussian_filter", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html", ("sigma_x", "sigma_y", "sigma_z", "truncate", "mode")),
    (SMOOTH3D_MEDIAN, DIM_3D, "SciPy median_filter", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html", ("kernel_x", "kernel_y", "kernel_z", "mode")),
    (SMOOTH3D_FFT, DIM_3D, "SciPy FFT", "https://docs.scipy.org/doc/scipy/reference/fft.html", ("fft_cutoff_ratio",)),
    (SMOOTH3D_RBF, DIM_3D, "SciPy RBFInterpolator", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html", ("rbf_kernel", "rbf_smoothing", "rbf_epsilon", "rbf_neighbors")),
    (SMOOTH3D_TV, DIM_3D, "scikit-image total variation", "https://scikit-image.org/docs/stable/api/skimage.restoration.html", ("tv_weight",)),
)

def _available_model_defs() -> tuple[tuple[str, str, str, str, tuple[str, ...]], ...]:
    available: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    for name, dimension, doc_title, doc_url, fields in _MODEL_DEFS:
        if name == SMOOTH_LOWESS and sm_lowess is None:
            continue
        if name == SMOOTH_HP and hpfilter is None:
            continue
        if name in {SMOOTH_TV, SMOOTH2D_TV, SMOOTH3D_TV} and denoise_tv_chambolle is None:
            continue
        if name == SMOOTH_WAVELET and pywt is None:
            continue
        available.append((name, dimension, doc_title, doc_url, fields))
    return tuple(available)

MODEL_REGISTRY: dict[str, ModelSpec] = {
    name: ModelSpec(dimension, doc_title, doc_url, _field_set(*fields))
    for name, dimension, doc_title, doc_url, fields in _available_model_defs()
}


def models_for_dimension(dimension: str) -> list[str]:
    """Return model names shown in the method combo for one dimension."""
    return [name for name, spec in MODEL_REGISTRY.items() if spec.dimension == dimension]


def model_spec(method: str) ->ModelSpec |None:
    """Return the registry spec or raise a clean error for stale UI values."""
    try:
        return MODEL_REGISTRY[method]
    except KeyError:
        applogger.error(f"Unsupported smoothing method: {method}")
        return None

# ---------------------------------------------------------------------------
# PySide6 dialog
# ---------------------------------------------------------------------------

class SeriesSmoothingDialog(SeriesOperationDialogBase):
    """Smoothing dialog that follows the shared operation-dialog pattern."""
    Name: str = "Smoothing"
    Description = "Reduce noise"

    Icon = """
    <path d="M4 13c1.7-4 3.4 4 5.1 0s3.4-4 5.1 0 3.4 4 5.8-1"/>
    <path d="M4 17c3.3-2.3 6.7-2.3 10 0 2 1.3 4 1.3 6 0"/>
    """
    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        applied_callback: Callable[[], None] | None = None,
        table: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesSmoothingDialog requires a repository instance.")

        self._repo: Any = repo
        self._figure_id = int(figure_id)
        self._applied_callback = applied_callback
        self._initial_table = table
        self._last_results: list[SmoothResult] = []
        self._field_rows: dict[str, tuple[QWidget, QWidget]] = {}

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Smoothing",
            parent=parent,
            width=720,
            height=640,
        )
        self.model_combo.setVisible(False)
        self._refresh_methods()
        self._refresh_visibility()
        self.refresh_results()

    def create_axis_series_selector(self) -> AxisSeriesSelector:
        return AxisSeriesSelector(self._repo, self._figure_id, self)

    def init_operation_widgets(self) -> None:
        self._create_controls()

    def build_model_selector(self) -> QWidget:
        panel = create_card_widget(self, "smoothingModelCard")
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)

        self.dimension_combo = QComboBox(self)
        self.dimension_combo.addItems([DIM_1D, DIM_2D, DIM_3D])
        self.dimension_combo.setToolTip(_("Choose whether the source series is 1D, 2D, or 3D data."))

        self.method_combo = QComboBox(self)
        self.method_combo.setToolTip(_("Choose the smoothing/filtering method."))

        form_widget = QWidget(panel)
        form = QFormLayout(form_widget)
        stdSizeAndlayout(form)
        form.addRow(_("Data type:"), self.dimension_combo)
        form.addRow(_("Model:"), self.method_combo)

        self._doc_link = QLabel(self)
        self._doc_link.setOpenExternalLinks(True)
        self._doc_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self._doc_link.setToolTip(_("Open documentation for the selected method."))
        form.addRow(_("Docs:"), self._doc_link)
        layout.addWidget(form_widget)

        return panel

    def build_parameter_selector(self) -> QWidget:
        settings_widget = create_card_widget(self, "smoothingParamsCard")
        self.form = QFormLayout(settings_widget)
        stdSizeAndlayout(self.form)
        self._add_parameter_rows()

        scroll = QScrollArea(self)
        stdSizeAndlayout(scroll)
        scroll.setWidget(settings_widget)
        return scroll

    def build_results_pane(self) -> QWidget:
        return super().build_results_pane()

    def connect_operation_signals(self) -> None:
        self.series_selector.selection_changed.connect(lambda *_args: self.refresh_results())
        self.series_selector.axis_changed.connect(lambda *_args: self.refresh_results())
        self.dimension_combo.currentIndexChanged.connect(self._refresh_methods)
        self.dimension_combo.currentIndexChanged.connect(self.refresh_results)
        self.method_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.method_combo.currentIndexChanged.connect(self.refresh_results)
        self._doc_link.linkActivated.connect(self._open_description)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setDecimals(decimals)
        widget.setSingleStep(10 ** -min(decimals, 3))
        return widget

    def _create_controls(self) -> None:
        """Create all parameter widgets once; visibility is managed dynamically."""
        self.window_spin = self._spin(3, 9999, 7)
        self.centered_check = QCheckBox()
        self.centered_check.setChecked(True)
        self.polyorder_spin = self._spin(0, 10, 2)
        self.deriv_spin = self._spin(0, 5, 0)
        self.delta_spin = self._double_spin(1e-12, 1e12, 1.0, 6)

        self.sigma_spin = self._double_spin(0.001, 1e6, 2.0, 4)
        self.sigma_x_spin = self._double_spin(0.001, 1e6, 1.5, 4)
        self.sigma_y_spin = self._double_spin(0.001, 1e6, 1.5, 4)
        self.sigma_z_spin = self._double_spin(0.001, 1e6, 1.5, 4)
        self.truncate_spin = self._double_spin(0.1, 50.0, 4.0, 2)

        self.kernel_spin = self._spin(3, 9999, 5)
        self.kernel_x_spin = self._spin(3, 9999, 3)
        self.kernel_y_spin = self._spin(3, 9999, 3)
        self.kernel_z_spin = self._spin(3, 9999, 3)

        self.noise_spin = self._double_spin(0.0, 1e12, 0.0, 6)
        self.noise_spin.setSpecialValueText(_("auto"))

        self.spline_s_spin = self._double_spin(0.0, 1e18, 10.0, 4)
        self.spline_k_spin = self._spin(1, 5, 3)
        self.kx_spin = self._spin(1, 5, 3)
        self.ky_spin = self._spin(1, 5, 3)

        self.lowess_frac_spin = self._double_spin(0.01, 1.0, 0.25, 3)
        self.lowess_it_spin = self._spin(0, 20, 3)
        self.fft_cutoff_spin = self._double_spin(0.001, 0.999, 0.20, 3)

        self.butter_fs_spin = self._double_spin(1e-12, 1e12, 1.0, 6)
        self.butter_cutoff_spin = self._double_spin(1e-12, 1e12, 0.20, 6)
        self.butter_high_cutoff_spin = self._double_spin(1e-12, 1e12, 0.40, 6)
        self.butter_order_spin = self._spin(1, 20, 4)
        self.butter_type_combo = QComboBox()
        self.butter_type_combo.addItems(["lowpass", "highpass", "bandpass", "bandstop"])

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["nearest", "reflect", "mirror", "constant", "wrap"])
        self.savgol_mode_combo = QComboBox()
        self.savgol_mode_combo.addItems(["interp", "mirror", "nearest", "constant", "wrap"])

        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(["db2", "db4", "sym4", "coif1", "haar"])
        self.wavelet_level_spin = self._spin(0, 20, 0)
        self.wavelet_threshold_factor_spin = self._double_spin(0.01, 100.0, 1.0, 3)
        self.wavelet_threshold_mode_combo = QComboBox()
        self.wavelet_threshold_mode_combo.addItems(["soft", "hard"])

        self.whittaker_lambda_spin = self._double_spin(0.0, 1e12, 1000.0, 3)
        self.whittaker_order_spin = self._spin(1, 5, 2)
        self.hp_lambda_spin = self._double_spin(0.0, 1e12, 1600.0, 3)
        self.tv_weight_spin = self._double_spin(0.0, 1e6, 0.15, 4)

        self.kalman_process_variance_spin = self._double_spin(1e-12, 1e6, 1e-4, 8)
        self.kalman_measurement_variance_spin = self._double_spin(1e-12, 1e6, 1e-2, 8)
        self.kalman_initial_covariance_spin = self._double_spin(1e-12, 1e6, 1.0, 8)

        self.rbf_kernel_combo = QComboBox()
        self.rbf_kernel_combo.addItems(
            [
                "thin_plate_spline",
                "linear",
                "cubic",
                "quintic",
                "multiquadric",
                "inverse_multiquadric",
                "inverse_quadratic",
                "gaussian",
            ]
        )
        self.rbf_smoothing_spin = self._double_spin(0.0, 1e9, 0.1, 4)
        self.rbf_epsilon_spin = self._double_spin(0.0, 1e9, 0.0, 4)
        self.rbf_epsilon_spin.setSpecialValueText(_("auto"))
        self.rbf_neighbors_spin = self._spin(0, 1_000_000, 0)
        self.rbf_neighbors_spin.setSpecialValueText(_("all"))

        self.preview_check = QCheckBox(_("Replace previous smoothing preview in chart"))
        self.preview_check.setChecked(True)
        self.preview_check.setToolTip(
            _("When enabled, applying this dialog again replaces the previously generated preview series.")
        )

    _PARAMETER_TOOLTIPS: dict[str, str] = {
        "window": "Window size, in points, used by the smoothing filter.",
        "centered": "Center the window on each point instead of trailing it.",
        "polyorder": "Polynomial order fit inside each Savitzky-Golay window.",
        "deriv": "Order of the derivative to compute (0 = smoothed value only).",
        "delta": "Sample spacing used when computing derivatives.",
        "sigma": "Standard deviation of the Gaussian kernel.",
        "sigma_x": "Standard deviation of the Gaussian kernel along X.",
        "sigma_y": "Standard deviation of the Gaussian kernel along Y.",
        "sigma_z": "Standard deviation of the Gaussian kernel along Z.",
        "truncate": "Truncate the Gaussian kernel at this many standard deviations.",
        "kernel": "Size, in points, of the moving kernel window.",
        "kernel_x": "Kernel size along X.",
        "kernel_y": "Kernel size along Y.",
        "kernel_z": "Kernel size along Z.",
        "noise": "Noise level estimate; leave at 'auto' to estimate from the data.",
        "spline_s": "Smoothing factor: higher values produce a smoother spline.",
        "spline_k": "Degree of the smoothing spline.",
        "kx": "Spline degree along X.",
        "ky": "Spline degree along Y.",
        "lowess_frac": "Fraction of points used for each local LOWESS regression.",
        "lowess_it": "Number of robustifying iterations for LOWESS.",
        "fft_cutoff_ratio": "Fraction of frequency components kept by the FFT low-pass filter.",
        "butter_fs": "Sampling frequency of the input data.",
        "butter_cutoff": "Cutoff frequency of the Butterworth filter.",
        "butter_high_cutoff": "Upper cutoff frequency for bandpass/bandstop filters.",
        "butter_order": "Order of the Butterworth filter.",
        "butter_type": "Butterworth filter type.",
        "mode": "Boundary handling mode applied at the edges of the data.",
        "savgol_mode": "Edge handling mode for the Savitzky-Golay filter.",
        "wavelet": "Wavelet family used for wavelet denoising.",
        "wavelet_level": "Decomposition level; 0 lets the algorithm choose automatically.",
        "wavelet_threshold_factor": "Multiplier applied to the estimated noise threshold.",
        "wavelet_threshold_mode": "Soft or hard thresholding of wavelet coefficients.",
        "whittaker_lambda": "Smoothness penalty for the Whittaker smoother (higher = smoother).",
        "whittaker_order": "Order of the finite-difference penalty.",
        "hp_lambda": "Smoothness penalty for the Hodrick-Prescott filter.",
        "tv_weight": "Regularization weight for total-variation denoising.",
        "kalman_process_variance": "Expected variance of the underlying process (model uncertainty).",
        "kalman_measurement_variance": "Expected variance of the measurement noise.",
        "kalman_initial_covariance": "Initial state covariance for the Kalman filter.",
        "rbf_kernel": "Radial basis function kernel used for interpolation-based smoothing.",
        "rbf_smoothing": "Smoothing factor for the RBF interpolant (0 = exact interpolation).",
        "rbf_epsilon": "Shape parameter for the RBF kernel; leave at 'auto' to estimate it.",
        "rbf_neighbors": "Number of nearest neighbors used per point; 'all' uses the full dataset.",
    }

    def _add_parameter_rows(self) -> None:
        """Add all parameter rows to the form; hidden rows stay in layout."""
        rows: tuple[tuple[str, str, QWidget], ...] = (
            ("window", "Window size:", self.window_spin),
            ("centered", "Centered:", self.centered_check),
            ("polyorder", "Polynomial order:", self.polyorder_spin),
            ("deriv", "Derivative order:", self.deriv_spin),
            ("delta", "Delta:", self.delta_spin),
            ("sigma", "Sigma:", self.sigma_spin),
            ("sigma_x", "Sigma X:", self.sigma_x_spin),
            ("sigma_y", "Sigma Y:", self.sigma_y_spin),
            ("sigma_z", "Sigma Z:", self.sigma_z_spin),
            ("truncate", "Gaussian truncate:", self.truncate_spin),
            ("kernel", "Kernel size:", self.kernel_spin),
            ("kernel_x", "Kernel X:", self.kernel_x_spin),
            ("kernel_y", "Kernel Y:", self.kernel_y_spin),
            ("kernel_z", "Kernel Z:", self.kernel_z_spin),
            ("noise", "Noise estimate:", self.noise_spin),
            ("spline_s", "Smoothing factor:", self.spline_s_spin),
            ("spline_k", "Spline order:", self.spline_k_spin),
            ("kx", "Spline order X:", self.kx_spin),
            ("ky", "Spline order Y:", self.ky_spin),
            ("lowess_frac", "LOWESS fraction:", self.lowess_frac_spin),
            ("lowess_it", "LOWESS iterations:", self.lowess_it_spin),
            ("fft_cutoff_ratio", "FFT cutoff ratio:", self.fft_cutoff_spin),
            ("butter_fs", "Sampling frequency:", self.butter_fs_spin),
            ("butter_cutoff", "Cutoff frequency:", self.butter_cutoff_spin),
            ("butter_high_cutoff", "High cutoff:", self.butter_high_cutoff_spin),
            ("butter_order", "Filter order:", self.butter_order_spin),
            ("butter_type", "Filter type:", self.butter_type_combo),
            ("mode", "Boundary mode:", self.mode_combo),
            ("savgol_mode", "Edge mode:", self.savgol_mode_combo),
            ("wavelet", "Wavelet:", self.wavelet_combo),
            ("wavelet_level", "Wavelet level:", self.wavelet_level_spin),
            (
                "wavelet_threshold_factor",
                "Threshold factor:",
                self.wavelet_threshold_factor_spin,
            ),
            (
                "wavelet_threshold_mode",
                "Threshold mode:",
                self.wavelet_threshold_mode_combo,
            ),
            ("whittaker_lambda", "Whittaker lambda:", self.whittaker_lambda_spin),
            ("whittaker_order", "Difference order:", self.whittaker_order_spin),
            ("hp_lambda", "HP lambda:", self.hp_lambda_spin),
            ("tv_weight", "TV weight:", self.tv_weight_spin),
            (
                "kalman_process_variance",
                "Kalman process variance:",
                self.kalman_process_variance_spin,
            ),
            (
                "kalman_measurement_variance",
                "Kalman measurement variance:",
                self.kalman_measurement_variance_spin,
            ),
            (
                "kalman_initial_covariance",
                "Kalman initial covariance:",
                self.kalman_initial_covariance_spin,
            ),
            ("rbf_kernel", "RBF kernel:", self.rbf_kernel_combo),
            ("rbf_smoothing", "RBF smoothing:", self.rbf_smoothing_spin),
            ("rbf_epsilon", "RBF epsilon:", self.rbf_epsilon_spin),
            ("rbf_neighbors", "RBF neighbors:", self.rbf_neighbors_spin),
        )

        for key, label, widget in rows:
            self._add_row(key, label, widget)

    def _add_row(self, key: str, label: str, widget: QWidget) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        stdSizeAndlayout(row_layout)
        row_layout.addWidget(widget)
        row_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        label_widget = QLabel(label)
        tooltip = self._PARAMETER_TOOLTIPS.get(key)
        if tooltip:
            widget.setToolTip(tooltip)
            label_widget.setToolTip(tooltip)
        self.form.addRow(label_widget, row_widget)
        self._field_rows[key] = (label_widget, row_widget)

    def _current_axis_name(self) -> str:
        return self.series_selector.selected_axis_name()



    def _series_choice_from_row(self, row: Any) -> SeriesChoice|None:
        name = row_value(row, "name", "series_name", "label", "title", default="Series")
        sql_query = row_value(row, "sql_query", "query", "sql", default="")
        if not sql_query:
            applogger.error("Selected series has no SQL query.")
            return None

        frame = self._repo.query_df(str(sql_query))
        if frame.empty:
            applogger.error("Selected series query returned no rows.")
            return None

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = [str(column) for column in frame.columns]
        numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]

        x_col = str(roles.get("x") or "")
        y_col = str(roles.get("y") or "")
        z_col = str(roles.get("z") or "")
        values_col = str(roles.get("values") or roles.get("value") or "")

        if x_col not in columns:
            x_col = numeric[0] if numeric else columns[0] if columns else "x"
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else columns[1] if len(columns) > 1 else "y"
        if z_col and z_col not in columns:
            z_col = ""
        if values_col and values_col not in columns:
            values_col = ""

        x_values = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
        z_values = None
        values = None
        if z_col:
            z_values = pd.to_numeric(frame[z_col], errors="coerce").to_numpy(dtype=float)
        if values_col:
            values = pd.to_numeric(frame[values_col], errors="coerce").to_numpy(dtype=float)

        return SeriesChoice(
            name=str(name),
            x=np.asarray(x_values, dtype=float).reshape(-1),
            y=np.asarray(y_values, dtype=float).reshape(-1),
            z=np.asarray(z_values, dtype=float).reshape(-1) if z_values is not None else None,
            values=np.asarray(values, dtype=float).reshape(-1) if values is not None else None,
            source=row,
        )

    def _on_axis_changed(self, axis_name: str) -> None:
        del axis_name
        self.refresh_results()

    def _refresh_methods(self) -> None:
        """Rebuild method list from the registry for the selected dimension."""
        current_method = self.method_combo.currentText()
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        self.method_combo.addItems(models_for_dimension(self.dimension_combo.currentText()))

        old_index = self.method_combo.findText(current_method)
        self.method_combo.setCurrentIndex(old_index if old_index >= 0 else 0)
        self.method_combo.blockSignals(False)
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        """Show only the settings required by the selected registry model."""
        method = self.method_combo.currentText()
        m=model_spec(method)
        if m is None:
            return
        visible = m.fields if method else frozenset()

        for key, widgets in self._field_rows.items():
            is_visible = key in visible
            label_widget, row_widget = widgets
            label_widget.setVisible(is_visible)
            row_widget.setVisible(is_visible)

        self._update_description_link()

    def _update_description_link(self) -> None:
        """Update the inline documentation hyperlink for the current method."""
        method = self.method_combo.currentText()
        if not method:
            self._doc_link.clear()
            return

        spec = model_spec(method)
        if spec is not None:
            self.set_doc_link(spec.doc_title, spec.doc_url)

    def _open_description(self, _link: str = "") -> None:
        """Open the documentation URL from the selected registry entry."""
        method = self.method_combo.currentText()
        if not method:
            return

        spec = model_spec(method)
        if spec is not None:
            try:
                webbrowser.open(spec.doc_url)
            except Exception:
                show_message(
                    self,
                    "series.open_docs_failed",
                    title=spec.doc_title,
                    url=spec.doc_url,
                )

    def _params(self) -> dict[str, Any]:
        """Collect current UI settings into a plain dict for metadata/reuse."""
        return {
            "window": self.window_spin.value(),
            "centered": self.centered_check.isChecked(),
            "polyorder": self.polyorder_spin.value(),
            "deriv": self.deriv_spin.value(),
            "delta": self.delta_spin.value(),
            "sigma": self.sigma_spin.value(),
            "sigma_x": self.sigma_x_spin.value(),
            "sigma_y": self.sigma_y_spin.value(),
            "sigma_z": self.sigma_z_spin.value(),
            "truncate": self.truncate_spin.value(),
            "kernel": self.kernel_spin.value(),
            "kernel_x": self.kernel_x_spin.value(),
            "kernel_y": self.kernel_y_spin.value(),
            "kernel_z": self.kernel_z_spin.value(),
            "noise": None if self.noise_spin.value() == 0.0 else self.noise_spin.value(),
            "spline_s": self.spline_s_spin.value(),
            "spline_k": self.spline_k_spin.value(),
            "kx": self.kx_spin.value(),
            "ky": self.ky_spin.value(),
            "lowess_frac": self.lowess_frac_spin.value(),
            "lowess_it": self.lowess_it_spin.value(),
            "fft_cutoff_ratio": self.fft_cutoff_spin.value(),
            "butter_fs": self.butter_fs_spin.value(),
            "butter_cutoff": self.butter_cutoff_spin.value(),
            "butter_high_cutoff": self.butter_high_cutoff_spin.value(),
            "butter_order": self.butter_order_spin.value(),
            "butter_type": self.butter_type_combo.currentText(),
            "mode": self.mode_combo.currentText(),
            "savgol_mode": self.savgol_mode_combo.currentText(),
            "wavelet": self.wavelet_combo.currentText(),
            "wavelet_level": self.wavelet_level_spin.value(),
            "wavelet_threshold_factor": self.wavelet_threshold_factor_spin.value(),
            "wavelet_threshold_mode": self.wavelet_threshold_mode_combo.currentText(),
            "whittaker_lambda": self.whittaker_lambda_spin.value(),
            "whittaker_order": self.whittaker_order_spin.value(),
            "hp_lambda": self.hp_lambda_spin.value(),
            "tv_weight": self.tv_weight_spin.value(),
            "kalman_process_variance": self.kalman_process_variance_spin.value(),
            "kalman_measurement_variance": self.kalman_measurement_variance_spin.value(),
            "kalman_initial_covariance": self.kalman_initial_covariance_spin.value(),
            "rbf_kernel": self.rbf_kernel_combo.currentText(),
            "rbf_smoothing": self.rbf_smoothing_spin.value(),
            "rbf_epsilon": (
                None if self.rbf_epsilon_spin.value() == 0.0
                else self.rbf_epsilon_spin.value()
            ),
            "rbf_neighbors": self.rbf_neighbors_spin.value(),
            "replace_preview": self.preview_check.isChecked(),
        }

    def compute_results(self) -> list[SmoothResult]:
        """Compute smoothing results for the current selector state."""
        axis_name = self._current_axis_name()
        dimension = self.dimension_combo.currentText()
        method = self.method_combo.currentText()
        params = self._params()
        selected_rows = self.selected_series()
        if not selected_rows:
            return []

        results: list[SmoothResult] = []
        errors: list[str] = []

        for row in selected_rows:
            try:
                series = self._series_choice_from_row(row)
                if series is None:
                    continue
                metadata = {
                    **dict(params),
                    "figure_id": self._figure_id,
                    "axis_name": axis_name,
                    "source_series_id": self._source_series_id(series),
                }
                sm=self._smooth_one_series(series, dimension, method, params, metadata)
                if sm:
                    results.append(sm)
            except Exception as exc:
                errors.append(f"{self._series_display_name(row)}: {exc}")

        if errors and not results:
            applogger.error("\n".join(errors))
            return []
        if errors:
            show_message(
                self,
                "series.some_failed",
                title=self.operation_label,
                errors="\n".join(errors),
            )
        return results

    def _smooth_one_series(
        self,
        series: SeriesChoice,
        dimension: str,
        method: str,
        params: Mapping[str, Any],
        metadata: dict[str, Any],
    ) -> SmoothResult|None:
        """Smooth one materialized series through the registry callable."""
        spec = model_spec(method)
        if spec is None:
            return None
        if spec.dimension != dimension:
            applogger.error(
                f"Method '{method}' is registered for {spec.dimension}, "
                f"not {dimension}."
            )

        if dimension == DIM_3D:
            z_values = series.z
            raw_values = series.values
            if z_values is None or raw_values is None or series is None or method is None or params is None:
                applogger.error("Selected series has no 3D values.")
                return None
            sm=smooth_3d(series.x, series.y, z_values, raw_values, method, params)
            if not sm:
                return None
            x_out, y_out, z_out, values_out = sm
            return SmoothResult(
                source_name=series.name,
                result_name=f"{series.name} - {method}",
                method=method,
                x=np.asarray(x_out, dtype=float),
                y=np.asarray(y_out, dtype=float),
                z=np.asarray(z_out, dtype=float),
                values=np.asarray(values_out, dtype=float),
                metadata=metadata,
            )

        if dimension == DIM_2D:
            z_values = series.z
            if z_values is None:
                applogger.error("Selected series has no Z values.")
                return None
            sm= smooth_2d(series.x,series.y,z_values,method,params,)
            if not sm:
                return None
            x_out, y_out, z_out = sm
            return SmoothResult(
                source_name=series.name,
                result_name=f"{series.name} - {method}",
                method=method,
                x=np.asarray(x_out, dtype=float),
                y=np.asarray(y_out, dtype=float),
                z=np.asarray(z_out, dtype=float),
                values=None,
                metadata=metadata,
            )

        x_clean, y_clean = _clean_xy(series.x, series.y)
        y_out = smooth_1d(x_clean, y_clean, method, params)
        return SmoothResult(
            source_name=series.name,
            result_name=f"{series.name} - {method}",
            method=method,
            x=x_clean,
            y=np.asarray(y_out, dtype=float),
            z=None,
            values=None,
            metadata=metadata,
        )

    def write_result_table(self, table_name: str, result: SmoothResult) -> None:
        frame = self.result_to_frame(result)
        self._repo.import_dataframe(
            frame,
            table_name=table_name,
            normalize_columns=False,
        )

    def result_to_frame(self, result: SmoothResult) -> pd.DataFrame:
        return self.results_to_dataframe([result])

    def result_series_spec(self, axis_id: int, table_name: str, result: SmoothResult) -> ResultSeriesSpec:
        del axis_id
        if result.values is not None and result.z is not None:
            sql_query = f'SELECT x, y, z, value FROM "{table_name}" ORDER BY z, y, x'
            roles = {"x": "x", "y": "y", "z": "z", "values": "value"}
        elif result.z is not None:
            sql_query = f'SELECT x, y, z FROM "{table_name}" ORDER BY x, y'
            roles = {"x": "x", "y": "y", "z": "z"}
        else:
            sql_query = f'SELECT x, y FROM "{table_name}" ORDER BY x'
            roles = {"x": "x", "y": "y"}

        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=sql_query,
            roles=roles,
            style={
                "generated_smoothing": True,
                "smoothing_dialog": "series_smoothing",
                "source_name": result.source_name,
                "source_series_id": result.metadata.get("source_series_id"),
                "method": result.method,
                "linestyle": "-",
                "linewidth": 2.0,
                "marker": "",
            },
        )

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_smoothing": True, "smoothing_dialog": "series_smoothing"}

    def result_table_name(self, axis_id: int, result: SmoothResult) -> str:
        raw = f"Smoothing_axis{axis_id}_{result.source_name}_{result.method}"
        return generated_table_name(raw, fallback="Smoothing_Result")

    def format_results(self, results: Sequence[SmoothResult]) -> str:
        if not results:
            return ""
        return f"Preview for {len(results)} smoothed series"

    def refresh_results(self) -> None:
        try:
            results = self.compute_results()
        except Exception as exc:
            self._last_results = []
            self.set_results_text(f"Error:\n{exc}")
            return

        self._last_results = results
        self.set_results_text(self.format_results(results) if results else "Select one or more source series.")

    @staticmethod
    def _source_series_id(series: SeriesChoice) -> int | None:
        source = series.source
        if source is None:
            return None
        keys = source.keys() if hasattr(source, "keys") else []
        if "id" in keys:
            return int(source["id"])
        if isinstance(source, Mapping) and "id" in source:
            return int(source["id"])
        value = getattr(source, "id", None)
        return int(value) if value is not None else None

    def apply(self) -> bool:
        """Apply smoothing using the shared generated-series workflow."""
        success = super().apply()
        if success and self._applied_callback is not None:
            self._applied_callback()
        return success

    @staticmethod
    def results_to_dataframe(results: Sequence[SmoothResult]) -> pd.DataFrame:
        """Flatten smoothing results into a DataFrame for export/storage."""
        rows: list[dict[str, Any]] = []

        for result in results:
            if result.values is not None and result.z is not None:
                SeriesSmoothingDialog._append_3d_rows(rows, result)
            elif result.z is not None:
                SeriesSmoothingDialog._append_2d_rows(rows, result)
            else:
                for x_value, y_value in zip(result.x, result.y, strict=False):
                    rows.append(
                        {
                            "result_name": result.result_name,
                            "source_name": result.source_name,
                            "method": result.method,
                            "x": float(x_value),
                            "y": float(y_value),
                        }
                    )

        return pd.DataFrame(rows)

    @staticmethod
    def _append_2d_rows(rows: list[dict[str, Any]], result: SmoothResult) -> None:
        z_values = np.asarray(result.z, dtype=float)
        x_values = np.asarray(result.x, dtype=float).reshape(-1)
        y_values = np.asarray(result.y, dtype=float).reshape(-1)

        if z_values.ndim == 2:
            for y_index, y_value in enumerate(y_values):
                for x_index, x_value in enumerate(x_values):
                    rows.append(
                        {
                            "result_name": result.result_name,
                            "source_name": result.source_name,
                            "method": result.method,
                            "x": float(x_value),
                            "y": float(y_value),
                            "z": float(z_values[y_index, x_index]),
                        }
                    )
            return

        for x_value, y_value, z_value in zip(
            x_values,
            y_values,
            z_values.reshape(-1),
            strict=False,
        ):
            rows.append(
                {
                    "result_name": result.result_name,
                    "source_name": result.source_name,
                    "method": result.method,
                    "x": float(x_value),
                    "y": float(y_value),
                    "z": float(z_value),
                }
            )

    @staticmethod
    def _append_3d_rows(rows: list[dict[str, Any]], result: SmoothResult) -> None:
        values = np.asarray(result.values, dtype=float)
        x_values = np.asarray(result.x, dtype=float).reshape(-1)
        y_values = np.asarray(result.y, dtype=float).reshape(-1)
        z_values = np.asarray(result.z, dtype=float).reshape(-1)

        if values.ndim == 3:
            for z_index, z_value in enumerate(z_values):
                for y_index, y_value in enumerate(y_values):
                    for x_index, x_value in enumerate(x_values):
                        rows.append(
                            {
                                "result_name": result.result_name,
                                "source_name": result.source_name,
                                "method": result.method,
                                "x": float(x_value),
                                "y": float(y_value),
                                "z": float(z_value),
                                "value": float(values[z_index, y_index, x_index]),
                            }
                        )
            return

        for x_value, y_value, z_value, scalar_value in zip(
            x_values,
            y_values,
            z_values,
            values.reshape(-1),
            strict=False,
        ):
            rows.append(
                {
                    "result_name": result.result_name,
                    "source_name": result.source_name,
                    "method": result.method,
                    "x": float(x_value),
                    "y": float(y_value),
                    "z": float(z_value),
                    "value": float(scalar_value),
                }
            )