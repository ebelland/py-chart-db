from __future__ import annotations

import numpy as np
import scipy.special as spsp
from app.functions.base import (
    _baseline,
    _eps,
    _exponential,
    _measured_fwhm,
    _peak_shape,
    _periodic_guess,
    _polynomial_guess,
    _pos,
    _positive_x,
    _safe_x,
    base_function,
)

# ---------------------------------------------------------------------------
# Basic functions
# ---------------------------------------------------------------------------

class constant(base_function):
    name = "Constant"
    category = "Basic functions"
    description = "Constant baseline."
    expression = "<b>Constant</b><br>y = C"
    p0 = [0.0]
    params = ["C"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.full_like(x, p[0], dtype=float)

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        del x
        return [float(np.mean(y))] if y.size else None

class linear(base_function):
    name = "Linear"
    category = "Basic functions"
    description = "Straight line with intercept and slope."
    expression = "<b>Linear</b><br>y = b + m x"
    p0 = [0.0, 1.0]
    params = ["intercept", "slope"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        return _polynomial_guess(x, y, 1)

class quadratic(base_function):
    name = "Quadratic"
    category = "Basic functions"
    description = "Second-order polynomial."
    expression = "<b>Quadratic</b><br>y = c0 + c1 x + c2 x²"
    p0 = [0.0, 1.0, 0.0]
    params = ["c0", "c1", "c2"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x + p[2] * x * x

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        return _polynomial_guess(x, y, 2)

class cubic(base_function):
    name = "Cubic"
    category = "Basic functions"
    description = "Third-order polynomial."
    expression = "<b>Cubic</b><br>y = c0 + c1 x + c2 x² + c3 x³"
    p0 = [0.0, 1.0, 0.0, 0.0]
    params = ["c0", "c1", "c2", "c3"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        return _polynomial_guess(x, y, 3)

class reciprocal(base_function):
    name = "Reciprocal"
    category = "Basic functions"
    description = "Inverse relation with offset."
    expression = "<b>Reciprocal</b><br>y = A / x + C"
    p0 = [1.0, 0.0]
    params = ["A", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] / _safe_x(x) + p[1]

class logarithmic(base_function):
    name = "Logarithmic"
    category = "Basic functions"
    description = "Scaled natural logarithm plus offset.  X is clipped positive for numerical safety."
    expression = "<b>Logarithmic</b><br>y = A ln(x - x0) + C"
    p0 = [1.0, 0.0, 0.0]
    params = ["A", "x shift", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.log(_positive_x(x - p[1])) + p[2]

class power_law(base_function):
    name = "Power law"
    category = "Basic functions"
    description = "Power relation with offset."
    expression = "<b>Power law</b><br>y = A xⁿ + C"
    p0 = [1.0, 1.0, 0.0]
    params = ["A", "exponent n", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.power(_positive_x(x), p[1]) + p[2]

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        """``A x^n + C``, via a straight line through log x against log y.

        Both offsets are tried and the better one kept.  A power law rises
        from its own offset, so the value at the low end of x is very nearly
        that offset - except when there is no offset at all, where the same
        reading subtracts a real part of the curve and the log-log line comes
        out through skewed data.  The two candidates cost one polyfit each and
        are told apart by the residual they leave, which is not something a
        rule of thumb can decide in advance.
        """
        candidates: list[list[float]] = []
        for offset in (_baseline(y), 0.0):
            above = y - offset
            usable = (x > 0.0) & (above > 0.0)
            if int(np.count_nonzero(usable)) < 2:
                continue
            try:
                exponent, intercept = np.polyfit(
                    np.log(x[usable]), np.log(above[usable]), 1
                )
            except Exception:
                continue
            if np.isfinite(exponent) and np.isfinite(intercept):
                candidates.append([float(np.exp(intercept)), float(exponent), float(offset)])

        if not candidates:
            return None

        def misfit(guess: list[float]) -> float:
            values = power_law.execute(x, np.asarray(guess, dtype=float))
            residual = y - values
            return float(np.sum(residual * residual)) if np.all(np.isfinite(residual)) else np.inf

        return min(candidates, key=misfit)

class rational_21(base_function):
    name = "Rational 2/1"
    category = "Basic functions"
    description = "Second-order numerator over first-order denominator."
    expression = "<b>Rational 2/1</b><br>y = (a0 + a1 x + a2 x²) / (1 + b1 x)"
    p0 = [0.0, 1.0, 0.0, 0.0]
    params = ["a0", "a1", "a2", "b1"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (p[0] + p[1] * x + p[2] * x**2) / _eps(1.0 + p[3] * x)

# ---------------------------------------------------------------------------
# Growth and saturation
# ---------------------------------------------------------------------------

class exponential_growth(base_function):
    name = "Exponential growth"
    category = "Growth and saturation"
    description = "Exponential increase with offset."
    expression = "<b>Exponential growth</b><br>y = A exp(k x) + C"
    p0 = [1.0, 0.1, 0.0]
    params = ["A", "rate k", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.exp(p[1] * x) + p[2]

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        return _exponential(x, y, True)

class exponential_decay(base_function):
    name = "Exponential decay"
    category = "Growth and saturation"
    description = "Single exponential decay with offset."
    expression = "<b>Exponential decay</b><br>y = A exp(-k x) + C"
    p0 = [1.0, 0.1, 0.0]
    params = ["A", "rate k", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.exp(-p[1] * x) + p[2]

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        return _exponential(x, y, False)

class double_exponential_decay(base_function):
    name = "Double exponential decay"
    category = "Growth and saturation"
    description = "Two exponential time constants plus offset."
    expression = "<b>Double exponential decay</b><br>y = A₁ exp(-k₁x) + A₂ exp(-k₂x) + C"
    p0 = [1.0, 0.2, 0.5, 0.02, 0.0]
    params = ["A1", "k1", "A2", "k2", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.exp(-p[1] * x) + p[2] * np.exp(-p[3] * x) + p[4]

class saturation_exponential(base_function):
    name = "Saturating exponential"
    category = "Growth and saturation"
    description = "First-order approach to a plateau."
    expression = "<b>Saturating exponential</b><br>y = C + A (1 - exp(-k x))"
    p0 = [1.0, 0.1, 0.0]
    params = ["plateau span", "rate k", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[2] + p[0] * (1.0 - np.exp(-p[1] * x))

class michaelis_menten(base_function):
    name = "Michaelis-Menten"
    category = "Growth and saturation"
    description = "Hyperbolic saturation."
    expression = "<b>Michaelis-Menten</b><br>y = C + Vmax x / (Km + x)"
    p0 = [1.0, 1.0, 0.0]
    params = ["Vmax", "Km", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[2] + p[0] * x / _eps(p[1] + x)

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        if x.size < 3:
            return None

        usable = (x > 0.0) & (y > 0.0)
        if int(np.count_nonzero(usable)) >= 2:
            try:
                slope, intercept = np.polyfit(x[usable], x[usable] / y[usable], 1)
            except Exception:
                slope = intercept = 0.0
            if np.isfinite(slope) and np.isfinite(intercept) and slope > 0.0:
                v_max = float(1.0 / slope)
                km = float(intercept * v_max)
                if km > 0.0 and np.isfinite(km):
                    # Returned here rather than falling through: the
                    # Lineweaver-Burk reading is the better of the two, and
                    # the direct one below would overwrite it.
                    return [v_max, km, 0.0]

        # The direct reading, for data the transform cannot take: every y at or
        # below zero, or a slope the wrong way.
        v_max = float(np.max(y))
        if v_max <= 0.0:
            return None
        km = float(x[int(np.argmin(np.abs(y - 0.5 * v_max)))])
        if km <= 0.0:
            km = float(np.median(x[x > 0.0])) if np.any(x > 0.0) else 1.0
        return [v_max, km, 0.0]

class hill(base_function):
    name = "Hill"
    category = "Growth and saturation"
    description = "Cooperative saturation curve."
    expression = "<b>Hill</b><br>y = C + Vmax xⁿ / (Kⁿ + xⁿ)"
    p0 = [1.0, 1.0, 2.0, 0.0]
    params = ["Vmax", "K half", "Hill coefficient", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        xp = _positive_x(x)
        n = _pos(p[2])
        return p[3] + p[0] * xp**n / _eps(_positive_x(p[1])**n + xp**n)

# ---------------------------------------------------------------------------
# Sigmoidal curves
# ---------------------------------------------------------------------------

class logistic4(base_function):
    name = "Logistic 4P"
    category = "Sigmoidal curves"
    description = "Four-parameter logistic curve."
    expression = "<b>Logistic 4P</b><br>y = bottom + span / (1 + exp(-k (x - x₀)))"
    p0 = [0.0, 1.0, 1.0, 0.0]
    params = ["bottom", "span", "steepness k", "midpoint x0"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] / (1.0 + np.exp(-p[2] * (x - p[3])))

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        if x.size < 4:
            return None

        edge = max(1, x.size // 10)
        bottom = float(np.mean(y[:edge]))
        top = float(np.mean(y[-edge:]))
        span = top - bottom
        if span == 0.0:
            return None

        half = bottom + 0.5 * span
        midpoint = float(x[int(np.argmin(np.abs(y - half)))])

        # A transition occupying about a tenth of the plotted range; the sign
        # follows the direction of travel because k carries it.
        x_span = float(np.ptp(x)) or 1.0
        steepness = (10.0 / x_span) * (1.0 if span > 0 else -1.0)
        return [bottom, span, steepness, midpoint]

class logistic5(base_function):
    name = "Logistic 5P asymmetric"
    category = "Sigmoidal curves"
    description = "Five-parameter logistic curve with asymmetry."
    expression = "<b>Logistic 5P</b><br>y = bottom + span / (1 + exp(-k (x - x₀)))ˢ"
    p0 = [0.0, 1.0, 1.0, 0.0, 1.0]
    params = ["bottom", "span", "steepness k", "midpoint x0", "asymmetry s"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] / np.power(1.0 + np.exp(-p[2] * (x - p[3])), _pos(p[4]))

class richards(base_function):
    name = "Richards generalized logistic"
    category = "Sigmoidal curves"
    description = "Generalized logistic with an additional shape parameter."
    expression = "<b>Richards</b><br>y = bottom + span / (1 + ν exp(-k (x - x₀)))^(1/ν)"
    p0 = [0.0, 1.0, 1.0, 0.0, 1.0]
    params = ["bottom", "span", "growth rate k", "midpoint x0", "shape nu"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        nu = _pos(p[4])
        return p[0] + p[1] / np.power(1.0 + nu * np.exp(-p[2] * (x - p[3])), 1.0 / nu)

class gompertz_growth(base_function):
    name = "Gompertz growth"
    category = "Sigmoidal curves"
    description = "Asymmetric growth curve."
    expression = "<b>Gompertz</b><br>y = C + A exp(-B exp(-k x))"
    p0 = [1.0, 1.0, 1.0, 0.0]
    params = ["A", "B", "growth rate k", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[3] + p[0] * np.exp(-p[1] * np.exp(-p[2] * x))

class erf_sigmoid(base_function):
    name = "Error-function sigmoid"
    category = "Sigmoidal curves"
    description = "Normal-CDF-like transition."
    expression = "<b>Error-function sigmoid</b><br>y = C + 0.5 A [1 + erf((x - x₀)/(√2 σ))]"
    p0 = [1.0, 0.0, 1.0, 0.0]
    params = ["height A", "midpoint x0", "sigma", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        sigma = _pos(p[2])
        return p[3] + 0.5 * p[0] * (1.0 + spsp.erf((x - p[1]) / (np.sqrt(2.0) * sigma)))

class tanh_sigmoid(base_function):
    name = "Tanh sigmoid"
    category = "Sigmoidal curves"
    description = "Hyperbolic tangent transition."
    expression = "<b>Tanh sigmoid</b><br>y = C + 0.5 A [1 + tanh(k (x - x₀))]"
    p0 = [1.0, 1.0, 0.0, 0.0]
    params = ["height A", "steepness k", "midpoint x0", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[3] + 0.5 * p[0] * (1.0 + np.tanh(p[1] * (x - p[2])))

class arctan_sigmoid(base_function):
    name = "Arctan sigmoid"
    category = "Sigmoidal curves"
    description = "Smooth transition with heavier tails than logistic."
    expression = "<b>Arctan sigmoid</b><br>y = C + A [0.5 + atan(k (x - x₀))/π]"
    p0 = [1.0, 1.0, 0.0, 0.0]
    params = ["height A", "steepness k", "midpoint x0", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[3] + p[0] * (0.5 + np.arctan(p[1] * (x - p[2])) / np.pi)

class double_logistic_pulse(base_function):
    name = "Double logistic pulse"
    category = "Sigmoidal curves"
    description = "Pulse with separate rising and falling logistic edges."
    expression = "<b>Double logistic pulse</b><br>y = C + A rise(x) fall(x)"
    p0 = [1.0, 1.0, -1.0, 1.0, 1.0, 0.0]
    params = ["height A", "rise steepness", "rise midpoint", "fall steepness", "fall midpoint", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        rise = 1.0 / (1.0 + np.exp(-p[1] * (x - p[2])))
        fall = 1.0 / (1.0 + np.exp(p[3] * (x - p[4])))
        return p[5] + p[0] * rise * fall
# ---------------------------------------------------------------------------
# Peak functions
# ---------------------------------------------------------------------------

class gaussian_peak(base_function):
    name = "Gaussian peak"
    category = "Peak functions"
    description = "Symmetric Gaussian peak with offset."
    expression = "<b>Gaussian</b><br>y = C + A exp(-(x - μ)² / (2 σ²))"
    p0 = [1.0, 0.0, 1.0, 0.0]
    params = ["amplitude A", "center mu", "sigma", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        sigma = _pos(p[2])
        return p[0] * np.exp(-((x - p[1]) ** 2) / (2.0 * sigma**2)) + p[3]

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        shape = _peak_shape(x, y)
        if shape is None:
            return None
        amplitude, centre, width, offset = shape
        return [amplitude, centre, max(width, 1e-9), offset]

class lorentzian_peak(base_function):
    name = "Lorentzian peak"
    category = "Peak functions"
    description = "Lorentzian peak with full width at half maximum."
    expression = "<b>Lorentzian</b><br>y = C + A (0.5 Γ)² / ((x - x₀)² + (0.5 Γ)²)"
    p0 = [1.0, 0.0, 1.0, 0.0]
    params = ["amplitude A", "center x0", "FWHM gamma", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        g2 = (0.5 * _pos(p[2])) ** 2
        return p[0] * g2 / (((x - p[1]) ** 2) + g2) + p[3]

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        shape = _peak_shape(x, y)
        if shape is None:
            return None
        amplitude, centre, moment_width, offset = shape

        width = _measured_fwhm(x, y, centre, amplitude, offset)
        if width is None:
            # Falling back to the moment is wrong by a factor for a Lorentzian,
            # but a too-wide peak in the right place still converges; no guess at
            # all does not.
            width = 2.355 * moment_width
        return [amplitude, centre, max(width, 1e-9), offset]

class pseudo_voigt_peak(base_function):
    name = "Pseudo-Voigt peak"
    category = "Peak functions"
    description = "Weighted blend of Gaussian and Lorentzian components."
    expression = "<b>Pseudo-Voigt</b><br>y = C + A [η L(x) + (1 - η) G(x)]"
    p0 = [1.0, 0.0, 1.0, 0.5, 0.0]
    params = ["amplitude A", "center", "width", "Lorentzian fraction eta", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        width = _pos(p[2])
        eta = np.clip(p[3], 0.0, 1.0)
        g = np.exp(-((x - p[1]) ** 2) / (2.0 * width**2))
        l = (0.5 * width) ** 2 / (((x - p[1]) ** 2) + (0.5 * width) ** 2)
        return p[0] * (eta * l + (1.0 - eta) * g) + p[4]

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        shape = _peak_shape(x, y)
        if shape is None:
            return None
        amplitude, centre, moment_width, offset = shape
        width = _measured_fwhm(x, y, centre, amplitude, offset) or (2.355 * moment_width)
        # eta 0.5: an even mix is the honest starting point, since nothing in the
        # data says how Gaussian or Lorentzian the peak is.
        return [amplitude, centre, max(width, 1e-9), 0.5, offset]


class voigt_peak(base_function):
    name = "Voigt peak"
    category = "Peak functions"
    description = "Convolution-like Gaussian/Lorentzian Voigt profile."
    expression = "<b>Voigt</b><br>y = C + A Re[wofz(z)] / (σ √(2π))"
    p0 = [1.0, 0.0, 1.0, 0.5, 0.0]
    params = ["amplitude A", "center", "Gaussian sigma", "Lorentzian gamma", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        sigma = _pos(p[2])
        gamma = _pos(p[3])
        z = ((x - p[1]) + 1j * gamma) / (sigma * np.sqrt(2.0))
        v = np.real(spsp.wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
        return p[0] * v + p[4]


class asymmetric_gaussian_peak(base_function):
    name = "Asymmetric Gaussian"
    category = "Peak functions"
    description = "Gaussian peak with different left and right widths."
    expression = "<b>Asymmetric Gaussian</b><br>σ = σL for x < μ, σR otherwise"
    p0 = [1.0, 0.0, 1.0, 2.0, 0.0]
    params = ["amplitude A", "center mu", "sigma left", "sigma right", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        sigma = np.where(x < p[1], _pos(p[2]), _pos(p[3]))
        return p[0] * np.exp(-((x - p[1]) ** 2) / (2.0 * sigma**2)) + p[4]


class pearson_vii_peak(base_function):
    name = "Pearson VII peak"
    category = "Peak functions"
    description = "Flexible peak shape between Gaussian-like and Lorentzian-like profiles."
    expression = "<b>Pearson VII</b><br>y = C + A [1 + ((x - x₀)/w)² / m]^(-m)"
    p0 = [1.0, 0.0, 1.0, 2.0, 0.0]
    params = ["amplitude A", "center x0", "width w", "shape m", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        w = _pos(p[2])
        m = _pos(p[3])
        return p[0] * np.power(1.0 + ((x - p[1]) / w) ** 2 / m, -m) + p[4]


# ---------------------------------------------------------------------------
# Periodic functions
# ---------------------------------------------------------------------------

class sine(base_function):
    name = "Sine"
    category = "Periodic functions"
    description = "Sinusoidal oscillation."
    expression = "<b>Sine</b><br>y = A sin(ωx + φ) + C"
    p0 = [1.0, 6.283185307179586, 0.0, 0.0]
    params = ["amplitude A", "angular frequency omega", "phase phi", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.sin(p[1] * x + p[2]) + p[3]

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        guess = _periodic_guess(x, y)
        if guess is None:
            return None
        amplitude, omega, offset = guess
        return [amplitude, omega, 0.0, offset]


class cosine(base_function):
    name = "Cosine"
    category = "Periodic functions"
    description = "Cosine oscillation."
    expression = "<b>Cosine</b><br>y = A cos(ωx + φ) + C"
    p0 = [1.0, 6.283185307179586, 0.0, 0.0]
    params = ["amplitude A", "angular frequency omega", "phase phi", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.cos(p[1] * x + p[2]) + p[3]

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        guess = _periodic_guess(x, y)
        if guess is None:
            return None
        amplitude, omega, offset = guess
        return [amplitude, omega, 0.0, offset]


class damped_sine(base_function):
    name = "Damped sine"
    category = "Periodic functions"
    description = "Exponentially damped sinusoid."
    expression = "<b>Damped sine</b><br>y = A exp(-d x) sin(ωx + φ) + C"
    p0 = [1.0, 6.283185307179586, 0.0, 0.1, 0.0]
    params = ["amplitude A", "omega", "phase phi", "damping d", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.exp(-p[3] * x) * np.sin(p[1] * x + p[2]) + p[4]


class chirp_linear(base_function):
    name = "Linear chirp"
    category = "Periodic functions"
    description = "Sine with linearly changing frequency."
    expression = "<b>Linear chirp</b><br>y = A sin(φ + 2π(f0 x + 0.5 k x²)) + C"
    p0 = [1.0, 1.0, 0.1, 0.0, 0.0]
    params = ["amplitude A", "f0", "chirp rate k", "phase phi", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.sin(p[3] + 2.0 * np.pi * (p[1] * x + 0.5 * p[2] * x**2)) + p[4]


# ---------------------------------------------------------------------------
# Statistical distributions
# ---------------------------------------------------------------------------

class normal_pdf(base_function):
    name = "Normal PDF"
    category = "Statistical distributions"
    description = "Scaled normal probability density plus offset."
    expression = "<b>Normal PDF</b><br>y = C + A exp(-0.5 z²) / (σ √(2π)), z=(x-μ)/σ"
    p0 = [1.0, 0.0, 1.0, 0.0]
    params = ["area A", "mean mu", "sigma", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        sigma = _pos(p[2])
        z = (x - p[1]) / sigma
        return p[0] * np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi)) + p[3]


class normal_cdf(base_function):
    name = "Normal CDF"
    category = "Statistical distributions"
    description = "Scaled normal cumulative distribution plus offset."
    expression = "<b>Normal CDF</b><br>y = C + 0.5 A [1 + erf((x-μ)/(σ√2))]"
    p0 = [1.0, 0.0, 1.0, 0.0]
    params = ["height A", "mean mu", "sigma", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        sigma = _pos(p[2])
        return p[0] * 0.5 * (1.0 + spsp.erf((x - p[1]) / (sigma * np.sqrt(2.0)))) + p[3]


class lognormal_pdf(base_function):
    name = "Lognormal PDF"
    category = "Statistical distributions"
    description = "Scaled lognormal density plus offset."
    expression = "<b>Lognormal PDF</b><br>y = C + A LogNormalPDF(x-location; μ, σ)"
    p0 = [1.0, 0.0, 0.5, 0.0, 0.0]
    params = ["area A", "log-mean mu", "log-sigma", "location", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        sigma = _pos(p[2])
        xp = _positive_x(x - p[3])
        z = (np.log(xp) - p[1]) / sigma
        return p[0] * np.exp(-0.5 * z * z) / (xp * sigma * np.sqrt(2.0 * np.pi)) + p[4]


class lognormal_cdf(base_function):
    name = "Lognormal CDF"
    category = "Statistical distributions"
    description = "Scaled lognormal cumulative distribution plus offset."
    expression = "<b>Lognormal CDF</b><br>y = C + A LogNormalCDF(x-location; μ, σ)"
    p0 = [1.0, 0.0, 0.5, 0.0, 0.0]
    params = ["height A", "log-mean mu", "log-sigma", "location", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        sigma = _pos(p[2])
        xp = _positive_x(x - p[3])
        return p[0] * 0.5 * (1.0 + spsp.erf((np.log(xp) - p[1]) / (sigma * np.sqrt(2.0)))) + p[4]


class weibull_pdf(base_function):
    name = "Weibull PDF"
    category = "Statistical distributions"
    description = "Scaled Weibull density plus offset."
    expression = "<b>Weibull PDF</b><br>y = C + A (k/λ) ((x-x0)/λ)^(k-1) exp(-((x-x0)/λ)^k)"
    p0 = [1.0, 2.0, 1.0, 0.0, 0.0]
    params = ["area A", "shape k", "scale lambda", "location x0", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        k = _pos(p[1]); lam = _pos(p[2]); xp = np.maximum(x - p[3], 0.0)
        y = (k / lam) * np.power(xp / lam, k - 1.0) * np.exp(-np.power(xp / lam, k))
        return p[0] * np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) + p[4]


class weibull_cdf(base_function):
    name = "Weibull CDF"
    category = "Statistical distributions"
    description = "Scaled Weibull cumulative distribution plus offset."
    expression = "<b>Weibull CDF</b><br>y = C + A [1 - exp(-((x-x0)/λ)^k)]"
    p0 = [1.0, 2.0, 1.0, 0.0, 0.0]
    params = ["height A", "shape k", "scale lambda", "location x0", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        k = _pos(p[1]); lam = _pos(p[2]); xp = np.maximum(x - p[3], 0.0)
        return p[0] * (1.0 - np.exp(-np.power(xp / lam, k))) + p[4]


class gamma_pdf(base_function):
    name = "Gamma PDF"
    category = "Statistical distributions"
    description = "Scaled gamma density plus offset."
    expression = "<b>Gamma PDF</b><br>y = C + A GammaPDF(x-location; α, θ)"
    p0 = [1.0, 2.0, 1.0, 0.0, 0.0]
    params = ["area A", "shape alpha", "scale theta", "location", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        a = _pos(p[1]); scale = _pos(p[2]); xp = _positive_x(x - p[3])
        log_pdf = (a - 1.0) * np.log(xp) - xp / scale - spsp.gammaln(a) - a * np.log(scale)
        return p[0] * np.exp(log_pdf) + p[4]


class gamma_cdf(base_function):
    name = "Gamma CDF"
    category = "Statistical distributions"
    description = "Scaled gamma cumulative distribution plus offset."
    expression = "<b>Gamma CDF</b><br>y = C + A P(α, (x-location)/θ)"
    p0 = [1.0, 2.0, 1.0, 0.0, 0.0]
    params = ["height A", "shape alpha", "scale theta", "location", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        a = _pos(p[1]); scale = _pos(p[2]); xp = np.maximum((x - p[3]) / scale, 0.0)
        return p[0] * spsp.gammainc(a, xp) + p[4]


class beta_pdf(base_function):
    name = "Beta PDF scaled"
    category = "Statistical distributions"
    description = "Scaled beta density over a finite x interval."
    expression = "<b>Beta PDF scaled</b><br>t=(x-low)/width; y = C + A t^(α-1)(1-t)^(β-1)/(B(α,β) width)"
    p0 = [1.0, 2.0, 2.0, 0.0, 1.0, 0.0]
    params = ["area A", "alpha", "beta", "lower x", "width", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        a = _pos(p[1]); b = _pos(p[2]); lo = p[3]; width = _pos(p[4])
        t = np.clip((x - lo) / width, 1e-12, 1.0 - 1e-12)
        log_beta = spsp.gammaln(a) + spsp.gammaln(b) - spsp.gammaln(a + b)
        y = np.exp((a - 1.0) * np.log(t) + (b - 1.0) * np.log1p(-t) - log_beta) / width
        y = np.where((x >= lo) & (x <= lo + width), y, 0.0)
        return p[0] * y + p[5]


class beta_cdf(base_function):
    name = "Beta CDF scaled"
    category = "Statistical distributions"
    description = "Scaled beta cumulative distribution over a finite interval."
    expression = "<b>Beta CDF scaled</b><br>t=(x-low)/width; y = C + A I_t(α,β)"
    p0 = [1.0, 2.0, 2.0, 0.0, 1.0, 0.0]
    params = ["height A", "alpha", "beta", "lower x", "width", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        a = _pos(p[1]); b = _pos(p[2]); lo = p[3]; width = _pos(p[4])
        t = np.clip((x - lo) / width, 0.0, 1.0)
        return p[0] * spsp.betainc(a, b, t) + p[5]


class cauchy_pdf(base_function):
    name = "Cauchy PDF"
    category = "Statistical distributions"
    description = "Scaled Cauchy density plus offset."
    expression = "<b>Cauchy PDF</b><br>y = C + A / (πγ [1 + ((x-x0)/γ)²])"
    p0 = [1.0, 0.0, 1.0, 0.0]
    params = ["area A", "location x0", "gamma", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        gamma = _pos(p[2]); z = (x - p[1]) / gamma
        return p[0] / (np.pi * gamma * (1.0 + z * z)) + p[3]


class cauchy_cdf(base_function):
    name = "Cauchy CDF"
    category = "Statistical distributions"
    description = "Scaled Cauchy cumulative distribution plus offset."
    expression = "<b>Cauchy CDF</b><br>y = C + A [0.5 + atan((x-x0)/γ)/π]"
    p0 = [1.0, 0.0, 1.0, 0.0]
    params = ["height A", "location x0", "gamma", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        gamma = _pos(p[2])
        return p[0] * (0.5 + np.arctan((x - p[1]) / gamma) / np.pi) + p[3]


# ---------------------------------------------------------------------------
# Reliability models
# ---------------------------------------------------------------------------

class exponential_survival(base_function):
    name = "Exponential survival"
    category = "Reliability models"
    description = "Survival/reliability curve for constant hazard rate."
    expression = "<b>Exponential survival</b><br>y = C + A exp(-λ x)"
    p0 = [1.0, 0.1, 0.0]
    params = ["A", "lambda", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[2] + p[0] * np.exp(-_pos(p[1]) * np.maximum(x, 0.0))


class weibull_survival(base_function):
    name = "Weibull survival"
    category = "Reliability models"
    description = "Weibull reliability/survival curve."
    expression = "<b>Weibull survival</b><br>y = C + A exp(-(x/η)^β)"
    p0 = [1.0, 1.5, 1.0, 0.0]
    params = ["A", "shape beta", "scale eta", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        beta = _pos(p[1]); eta = _pos(p[2]); xp = np.maximum(x, 0.0)
        return p[3] + p[0] * np.exp(-np.power(xp / eta, beta))


class weibull_hazard(base_function):
    name = "Weibull hazard"
    category = "Reliability models"
    description = "Weibull instantaneous hazard model."
    expression = "<b>Weibull hazard</b><br>y = C + A (β/η) (x/η)^(β-1)"
    p0 = [1.0, 1.5, 1.0, 0.0]
    params = ["A", "shape beta", "scale eta", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        beta = _pos(p[1]); eta = _pos(p[2]); xp = np.maximum(x, 1e-12)
        return p[3] + p[0] * (beta / eta) * np.power(xp / eta, beta - 1.0)


# ---------------------------------------------------------------------------
# Semiconductor / process models
# ---------------------------------------------------------------------------

class arrhenius_kelvin(base_function):
    name = "Arrhenius Kelvin"
    category = "Semiconductor and process"
    description = "Arrhenius-type temperature dependence. X is temperature in Kelvin."
    expression = "<b>Arrhenius</b><br>y = C + A exp(-Ea / (kB T))"
    p0 = [1.0, 0.7, 0.0]
    params = ["A", "Ea eV", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        kb_ev = 8.617333262145e-5
        return p[2] + p[0] * np.exp(-p[1] / (kb_ev * _positive_x(x)))


class inverse_temperature_linear(base_function):
    name = "Inverse temperature linear"
    category = "Semiconductor and process"
    description = "Linear relation versus inverse temperature."
    expression = "<b>Inverse temperature linear</b><br>y = C + A / T"
    p0 = [1.0, 0.0]
    params = ["A", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[1] + p[0] / _positive_x(x)


class gaussian_process_window(base_function):
    name = "Gaussian process window"
    category = "Semiconductor and process"
    description = "Process response with an optimum x value and symmetric fall-off."
    expression = "<b>Gaussian process window</b><br>y = C + A exp(-0.5 ((x - xopt)/w)²)"
    p0 = [1.0, 0.0, 1.0, 0.0]
    params = ["response A", "optimum x", "width", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[3] + p[0] * np.exp(-0.5 * ((x - p[1]) / _pos(p[2])) ** 2)


class dose_response_power(base_function):
    name = "Dose response power"
    category = "Semiconductor and process"
    description = "Power-law process response to dose or time."
    expression = "<b>Dose response power</b><br>y = C + A (x - x0)^n"
    p0 = [1.0, 0.0, 1.0, 0.0]
    params = ["A", "x shift", "exponent n", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[3] + p[0] * np.power(_positive_x(x - p[1]), p[2])


class poisson_yield(base_function):
    name = "Poisson yield"
    category = "Semiconductor and process"
    description = "Simple Poisson yield relation versus area or defect opportunity."
    expression = "<b>Poisson yield</b><br>y = C + Y0 exp(-D x)"
    p0 = [1.0, 0.01, 0.0]
    params = ["Y0", "defect density D", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[2] + p[0] * np.exp(-_pos(p[1]) * np.maximum(x, 0.0))


# ---------------------------------------------------------------------------
# Adsorption / surface models
# ---------------------------------------------------------------------------

class langmuir(base_function):
    name = "Langmuir"
    category = "Adsorption and surface"
    description = "Langmuir adsorption/saturation model."
    expression = "<b>Langmuir</b><br>y = C + Qmax K x / (1 + K x)"
    p0 = [1.0, 1.0, 0.0]
    params = ["Qmax", "K", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        kx = _pos(p[1]) * _positive_x(x)
        return p[2] + p[0] * kx / (1.0 + kx)


class freundlich(base_function):
    name = "Freundlich"
    category = "Adsorption and surface"
    description = "Empirical adsorption isotherm."
    expression = "<b>Freundlich</b><br>y = C + K x^(1/n)"
    p0 = [1.0, 2.0, 0.0]
    params = ["K", "n", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[2] + p[0] * np.power(_positive_x(x), 1.0 / _pos(p[1]))


class temkin(base_function):
    name = "Temkin"
    category = "Adsorption and surface"
    description = "Logarithmic adsorption-like response."
    expression = "<b>Temkin</b><br>y = C + B ln(A x)"
    p0 = [1.0, 1.0, 0.0]
    params = ["B", "A", "offset"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[2] + p[0] * np.log(_positive_x(_pos(p[1]) * x))
