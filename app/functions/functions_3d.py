"""Two-input fit functions for 3D surfaces.

The classes mirror ``app.functions.functions``: metadata lives on the class,
``execute(xy, p)`` evaluates the model, and ``initial_guess(xy, z)`` optionally
estimates parameters from data.  ``xy`` accepts either a ``(2, N)`` array, an
``(N, 2)`` array, or a pair ``(x, y)``.  Every function returns an array with
the broadcast shape of x and y, which makes the same models usable for scattered
samples and mesh grids.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.functions.base import eps, pos, base_function

_EPS = 1.0e-12


def _xy(xy: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return broadcast-compatible x and y arrays from supported inputs."""
    if isinstance(xy, (tuple, list)) and len(xy) == 2:
        x = np.asarray(xy[0], dtype=float)
        y = np.asarray(xy[1], dtype=float)
    else:
        values = np.asarray(xy, dtype=float)
        if values.ndim < 2:
            raise ValueError("3D functions require x and y coordinates.")
        if values.shape[0] == 2:
            x, y = values[0], values[1]
        elif values.shape[-1] == 2:
            x, y = values[..., 0], values[..., 1]
        else:
            raise ValueError("Coordinates must have shape (2, N) or (N, 2).")
    x_broadcast, y_broadcast = np.broadcast_arrays(x, y)
    return x_broadcast, y_broadcast


def _finite_xyz(xy: Any, z: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = _xy(xy)
    z_array = np.asarray(z, dtype=float)
    x, y, z_array = np.broadcast_arrays(x, y, z_array)
    x, y, z_array = x.ravel(), y.ravel(), z_array.ravel()
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z_array)
    return x[keep], y[keep], z_array[keep]


def _least_squares_guess(xy: Any, z: Any, columns: list[np.ndarray]) -> list[float] | None:
    x, y, values = _finite_xyz(xy, z)
    if values.size < len(columns):
        return None
    design_columns = [np.asarray(column, dtype=float).ravel() for column in columns]
    size = min([values.size, *(column.size for column in design_columns)])
    design = np.column_stack([column[:size] for column in design_columns])
    values = values[:size]
    keep = np.all(np.isfinite(design), axis=1) & np.isfinite(values)
    if int(np.count_nonzero(keep)) < design.shape[1]:
        return None
    try:
        coefficients, *_unused = np.linalg.lstsq(design[keep], values[keep], rcond=None)
    except np.linalg.LinAlgError:
        return None
    return [float(value) for value in coefficients]


def _surface_peak_guess(xy: Any, z: Any) -> tuple[float, float, float, float, float, float, float] | None:
    """Estimate amplitude, center, widths, angle and offset for one surface peak."""
    x, y, values = _finite_xyz(xy, z)
    if values.size < 6:
        return None
    offset = float(np.percentile(values, 10.0))
    signal = values - offset
    inverted = abs(float(np.min(signal))) > abs(float(np.max(signal)))
    weights = np.clip(-signal if inverted else signal, 0.0, None)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return None
    x0 = float(np.sum(weights * x) / total)
    y0 = float(np.sum(weights * y) / total)
    dx, dy = x - x0, y - y0
    covariance = np.array(
        [
            [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
            [np.sum(weights * dx * dy), np.sum(weights * dy * dy)],
        ],
        dtype=float,
    ) / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, _EPS)
    order = np.argsort(eigenvalues)[::-1]
    major, minor = eigenvalues[order]
    vector = eigenvectors[:, order[0]]
    angle = float(np.arctan2(vector[1], vector[0]))
    amplitude = float(np.max(weights)) * (-1.0 if inverted else 1.0)
    return amplitude, x0, y0, float(np.sqrt(major)), float(np.sqrt(minor)), angle, offset


def _rotate(x: np.ndarray, y: np.ndarray, x0: float, y0: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx, dy = x - x0, y - y0
    return cos_a * dx + sin_a * dy, -sin_a * dx + cos_a * dy


class constant_surface(base_function):
    name = "Constant surface"
    category = "3D basic surfaces"
    description = "Constant Z plane."
    expression = "<b>Constant surface</b><br>z = C"
    p0 = [0.0]
    params = ["C"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, _y = _xy(xy)
        return np.full_like(x, p[0], dtype=float)

    @staticmethod
    def initial_guess(xy: np.ndarray, z: np.ndarray) -> list[float] | None:
        _x, _y, values = _finite_xyz(xy, z)
        return [float(np.mean(values))] if values.size else None


class plane(base_function):
    name = "Plane"
    category = "3D basic surfaces"
    description = "Plane with independent X and Y slopes."
    expression = "<b>Plane</b><br>z = C + ax + by"
    p0 = [0.0, 1.0, 1.0]
    params = ["offset C", "x slope a", "y slope b"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        return p[0] + p[1] * x + p[2] * y

    @staticmethod
    def initial_guess(xy: np.ndarray, z: np.ndarray) -> list[float] | None:
        x, y, values = _finite_xyz(xy, z)
        return _least_squares_guess((x, y), values, [np.ones_like(x), x, y])


class quadratic_surface(base_function):
    name = "Quadratic surface"
    category = "3D polynomial surfaces"
    description = "General second-order polynomial surface."
    expression = "<b>Quadratic surface</b><br>z = c0 + c1x + c2y + c3x² + c4xy + c5y²"
    p0 = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    params = ["c0", "x", "y", "x squared", "x y", "y squared"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        return p[0] + p[1] * x + p[2] * y + p[3] * x**2 + p[4] * x * y + p[5] * y**2

    @staticmethod
    def initial_guess(xy: np.ndarray, z: np.ndarray) -> list[float] | None:
        x, y, values = _finite_xyz(xy, z)
        return _least_squares_guess((x, y), values, [np.ones_like(x), x, y, x**2, x * y, y**2])


class paraboloid(base_function):
    name = "Elliptic paraboloid"
    category = "3D polynomial surfaces"
    description = "Axis-aligned bowl or dome with movable center."
    expression = "<b>Elliptic paraboloid</b><br>z = C + ax(x-x0)² + ay(y-y0)²"
    p0 = [0.0, 0.0, 1.0, 1.0, 0.0]
    params = ["center x0", "center y0", "x curvature", "y curvature", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        return p[4] + p[2] * (x - p[0]) ** 2 + p[3] * (y - p[1]) ** 2


class saddle(base_function):
    name = "Saddle"
    category = "3D polynomial surfaces"
    description = "Hyperbolic paraboloid."
    expression = "<b>Saddle</b><br>z = C + a(x-x0)² - b(y-y0)²"
    p0 = [0.0, 0.0, 1.0, 1.0, 0.0]
    params = ["center x0", "center y0", "x curvature", "y curvature", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        return p[4] + p[2] * (x - p[0]) ** 2 - p[3] * (y - p[1]) ** 2


class gaussian_surface(base_function):
    name = "Gaussian surface"
    category = "3D peak surfaces"
    description = "Rotated anisotropic Gaussian peak with offset."
    expression = "<b>Gaussian surface</b><br>z = C + A exp(-0.5[(u/sx)² + (v/sy)²])"
    p0 = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    params = ["amplitude A", "center x0", "center y0", "sigma x", "sigma y", "angle", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        u, v = _rotate(x, y, p[1], p[2], p[5])
        return p[6] + p[0] * np.exp(-0.5 * ((u / pos(p[3])) ** 2 + (v / pos(p[4])) ** 2))

    @staticmethod
    def initial_guess(xy: np.ndarray, z: np.ndarray) -> list[float] | None:
        guess = _surface_peak_guess(xy, z)
        return list(guess) if guess is not None else None


class lorentzian_surface(base_function):
    name = "Lorentzian surface"
    category = "3D peak surfaces"
    description = "Rotated anisotropic Lorentzian peak with offset."
    expression = "<b>Lorentzian surface</b><br>z = C + A / [1 + (u/wx)² + (v/wy)²]"
    p0 = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    params = ["amplitude A", "center x0", "center y0", "width x", "width y", "angle", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        u, v = _rotate(x, y, p[1], p[2], p[5])
        return p[6] + p[0] / (1.0 + (u / pos(p[3])) ** 2 + (v / pos(p[4])) ** 2)

    @staticmethod
    def initial_guess(xy: np.ndarray, z: np.ndarray) -> list[float] | None:
        guess = _surface_peak_guess(xy, z)
        return list(guess) if guess is not None else None


class exponential_surface(base_function):
    name = "Radial exponential surface"
    category = "3D radial surfaces"
    description = "Exponential decay from a movable center with elliptical scaling."
    expression = "<b>Radial exponential</b><br>z = C + A exp(-sqrt((dx/sx)² + (dy/sy)²))"
    p0 = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    params = ["amplitude A", "center x0", "center y0", "x scale", "y scale", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        radius = np.sqrt(((x - p[1]) / pos(p[3])) ** 2 + ((y - p[2]) / pos(p[4])) ** 2)
        return p[5] + p[0] * np.exp(-radius)


class cone(base_function):
    name = "Elliptic cone"
    category = "3D radial surfaces"
    description = "Linear radial rise or fall from a movable center."
    expression = "<b>Elliptic cone</b><br>z = C + A sqrt((dx/sx)² + (dy/sy)²)"
    p0 = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    params = ["slope A", "center x0", "center y0", "x scale", "y scale", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        radius = np.sqrt(((x - p[1]) / pos(p[3])) ** 2 + ((y - p[2]) / pos(p[4])) ** 2)
        return p[5] + p[0] * radius


class sinc_surface(base_function):
    name = "Radial sinc surface"
    category = "3D radial surfaces"
    description = "Oscillating radial sinc profile."
    expression = "<b>Radial sinc</b><br>z = C + A sinc(k r / pi)"
    p0 = [1.0, 1.0, 0.0, 0.0, 0.0]
    params = ["amplitude A", "radial frequency k", "center x0", "center y0", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        radius = np.hypot(x - p[2], y - p[3])
        return p[4] + p[0] * np.sinc(p[1] * radius / np.pi)


class radial_ripple(base_function):
    name = "Radial ripple"
    category = "3D periodic surfaces"
    description = "Cosine rings with optional exponential damping."
    expression = "<b>Radial ripple</b><br>z = C + A exp(-d r) cos(k r + phi)"
    p0 = [1.0, 6.283185307179586, 0.0, 0.1, 0.0, 0.0, 0.0]
    params = ["amplitude A", "radial frequency k", "phase", "damping", "center x0", "center y0", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        radius = np.hypot(x - p[4], y - p[5])
        return p[6] + p[0] * np.exp(-pos(p[3]) * radius) * np.cos(p[1] * radius + p[2])


class plane_wave(base_function):
    name = "Plane wave"
    category = "3D periodic surfaces"
    description = "Sinusoidal plane wave with independent X and Y wave numbers."
    expression = "<b>Plane wave</b><br>z = C + A sin(kx x + ky y + phi)"
    p0 = [1.0, 6.283185307179586, 0.0, 0.0, 0.0]
    params = ["amplitude A", "x wave number", "y wave number", "phase", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        return p[4] + p[0] * np.sin(p[1] * x + p[2] * y + p[3])


class egg_crate(base_function):
    name = "Egg-crate surface"
    category = "3D periodic surfaces"
    description = "Product of orthogonal cosine waves."
    expression = "<b>Egg-crate</b><br>z = C + A cos(kx x + phix) cos(ky y + phiy)"
    p0 = [1.0, 6.283185307179586, 6.283185307179586, 0.0, 0.0, 0.0]
    params = ["amplitude A", "x wave number", "y wave number", "x phase", "y phase", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        return p[5] + p[0] * np.cos(p[1] * x + p[3]) * np.cos(p[2] * y + p[4])


class sombrero(base_function):
    name = "Sombrero surface"
    category = "3D radial surfaces"
    description = "Mexican-hat or Ricker-like radial peak."
    expression = "<b>Sombrero</b><br>z = C + A (1-r²) exp(-r²/2)"
    p0 = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    params = ["amplitude A", "center x0", "center y0", "x scale", "y scale", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        r2 = ((x - p[1]) / pos(p[3])) ** 2 + ((y - p[2]) / pos(p[4])) ** 2
        return p[5] + p[0] * (1.0 - r2) * np.exp(-0.5 * r2)


class hyperbolic_surface(base_function):
    name = "Hyperbolic surface"
    category = "3D rational surfaces"
    description = "Reciprocal quadratic surface with offset."
    expression = "<b>Hyperbolic surface</b><br>z = C + A / (1 + ax² + by²)"
    p0 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    params = ["amplitude A", "x coefficient", "y coefficient", "center x0", "center y0", "offset"]

    @staticmethod
    def execute(xy: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, y = _xy(xy)
        denominator = 1.0 + p[1] * (x - p[3]) ** 2 + p[2] * (y - p[4]) ** 2
        return p[5] + p[0] / eps(denominator)
