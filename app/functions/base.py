"""What every fit function is, and the estimators they share.

A fit function is a class with two staticmethods: ``execute(x, p)`` evaluates
it, and ``initial_guess(x, y)`` reads a starting point off the data.  Neither
is ever called on an instance - ``FunctionScanner`` discovers the classes and
reads their attributes without constructing anything - which is why both are
static and why ``p0`` is a ClassVar.

``initial_guess`` is optional and returns None when it has nothing to say.
That is a different answer from a bad guess: the caller then searches for a
starting point instead (see ``app/functions/monte_carlo.py``), rather than
starting the optimiser from a number somebody invented.

The private helpers below are what those estimators are made of - a peak's
moments, a baseline, a log-linear exponential fit.  They live here rather than
in ``functions.py`` because half the library shares them and because they are
worth testing on their own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, List

import numpy as np
from scipy.signal import find_peaks

def _safe_x(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.where(np.abs(x) < eps, np.sign(x + eps) * eps, x)

def _eps(value: float | np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    return np.where(np.abs(arr) < eps, eps, arr)

def _pos(value: float | np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    return np.where(np.abs(arr) < eps, eps, np.abs(arr))

def _positive_x(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.maximum(x, eps)

def _finite(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the pairs where both coordinates are finite."""
    x_array = np.asarray(x, dtype=float).ravel()
    y_array = np.asarray(y, dtype=float).ravel()
    size = min(x_array.size, y_array.size)
    x_array, y_array = x_array[:size], y_array[:size]
    keep = np.isfinite(x_array) & np.isfinite(y_array)
    return x_array[keep], y_array[keep]

def _peak_shape(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return (amplitude, centre, width, offset) for a peak-like series.

    The offset is the lower of the two ends rather than the minimum: a peak
    sits on a baseline, and the ends are where the baseline shows. Using the
    global minimum would put the offset at the bottom of a noise spike.

    The width is the second moment of the baseline-subtracted signal, which is
    sigma for a Gaussian and within a small factor of the half-width for the
    other peak shapes - close enough that the optimiser finishes the job.
    """
    if x.size < 3:
        return None

    edge = max(1, x.size // 10)
    offset = float(min(np.mean(y[:edge]), np.mean(y[-edge:])))
    signal = y - offset

    # An inverted peak is a peak too; fit its magnitude and restore the sign.
    inverted = abs(float(np.min(signal))) > abs(float(np.max(signal)))
    if inverted:
        signal = -signal

    total = float(np.sum(signal))
    if not np.isfinite(total) or total <= 0.0:
        return None

    weights = np.clip(signal, 0.0, None)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return None

    centre = float(np.sum(x * weights) / weight_sum)
    variance = float(np.sum(weights * (x - centre) ** 2) / weight_sum)
    width = float(np.sqrt(variance)) if variance > 0.0 else float(np.ptp(x) / 10.0) or 1.0

    amplitude = float(np.max(signal))
    if inverted:
        amplitude = -amplitude

    return amplitude, centre, width, offset


def _baseline(y: np.ndarray) -> float:
    """Return the offset a decaying or saturating curve settles on.

    The smaller end rather than the global minimum: these curves approach
    their offset asymptotically, so the value at the far end is the offset,
    while the global minimum could be any noise trough along the way.
    """
    edge = max(1, y.size // 10)
    return float(min(np.mean(y[:edge]), np.mean(y[-edge:])))

def _measured_fwhm(
    x: np.ndarray,
    y: np.ndarray,
    centre: float,
    amplitude: float,
    offset: float,
) -> float | None:
    """Return the full width at half maximum, read off the data.

    Measured rather than derived from a second moment, because the moment is
    only proportional to the width for a Gaussian. A Lorentzian's second
    moment does not converge at all - its tails are too heavy - so scaling one
    by 2.355 gives a width several times too large, and the optimiser starts
    from a peak far broader than the data.
    """
    if amplitude == 0.0:
        return None

    half = offset + 0.5 * amplitude
    signal = y - half
    if amplitude < 0.0:
        signal = -signal

    # The crossings either side of the centre; the distance between them is
    # the width by definition.
    left = x[(x <= centre) & (signal <= 0.0)]
    right = x[(x >= centre) & (signal <= 0.0)]
    if left.size == 0 or right.size == 0:
        return None

    width = float(right.min() - left.max())
    return width if width > 0.0 else None

def _periodic_guess(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """Return (amplitude, angular frequency, offset) for an oscillating series.

    Amplitude is half the 5th-95th percentile spread rather than half of
    max-min: one noise spike should not get to set the whole fit's scale.
    Offset is the mean, the natural centre of an oscillation around a
    baseline.

    Frequency comes from the spacing between peaks, not from the shape of
    any single crest - a least-squares fit is bad at recovering the number
    of cycles per unit x on its own from a wrong starting guess (it is a
    steep, many-times-repeating minimum), which is exactly what this is for.
    Needs at least two peaks - one peak has no spacing to measure - so a
    series covering less than one full period falls back to None like any
    other guess this module cannot make.
    """
    if x.size < 4:
        return None

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    offset = float(np.mean(y_sorted))
    amplitude = float((np.percentile(y_sorted, 95) - np.percentile(y_sorted, 5)) / 2.0)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        return None

    # A minimum prominence keeps peak-finding from tripping on point-to-point
    # noise instead of real crests.
    peaks, _props = find_peaks(y_sorted, prominence=amplitude * 0.2)
    if peaks.size < 2:
        return None

    period = float(np.mean(np.diff(x_sorted[peaks])))
    if not np.isfinite(period) or period <= 0.0:
        return None

    return amplitude, 2.0 * np.pi / period, offset


def _exponential(x: np.ndarray, y: np.ndarray, growth:bool) ->  list[float]|None:
    # Just under the minimum, not the average of an end. An exponential
    # approaches its offset asymptotically, so the offset is below every
    # observed value; taking an end average puts it *inside* the data and
    # leaves almost nothing with a real logarithm. The margin keeps the
    # subtraction strictly positive so every point stays usable.
    span = float(np.ptp(y))
    offset = float(np.min(y)) - (0.02 * span if span > 0.0 else 1e-9)
    above = y - offset
    usable = above > 0.0
    if int(np.count_nonzero(usable)) < 2:
        return None
    try:
        slope, intercept = np.polyfit(x[usable], np.log(above[usable]), 1)
    except Exception:
        return None
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None

    amplitude = float(np.exp(intercept))
    rate = float(slope) if growth else float(-slope)
    return [amplitude, rate, offset]

def _polynomial_guess( x: np.ndarray,y: np.ndarray,degree) ->  List[float]|None:
    if x.size < degree + 1:
        return None
    try:
        coefficients = np.polyfit(x, y, degree)
    except Exception:
        return None
    return [float(value) for value in coefficients[::-1]]


@dataclass
class base_function:
    """Base metadata contract for scanned fit functions.

    ``category`` is the top-level tree grouping in the fit dialog.  There is no
    separate ``kind`` field on function classes: all subclasses are executable
    fit functions discovered by FunctionScanner.
    """

    name: ClassVar[str] = ""
    category: ClassVar[str] = "Functions"
    description: ClassVar[str] = ""
    expression: ClassVar[str] = ""
    #: Declared starting values, one per entry of ``params``.  A ClassVar
    #: because nothing instantiates these classes; a dataclass field with a
    #: list default is also refused outright by ``dataclasses``.
    p0: ClassVar[list[float]] = []
    params: ClassVar[list[str]] = []

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> List[float] | None:
        """Return a starting point read off the data, or None.

        None rather than ``p0`` so that "I have no estimate" is distinguishable
        from "my estimate is the declared default": the caller falls back to a
        random search for the first and would waste it on the second.

        Static, like ``execute``, because the scanner never builds an instance.
        A guess derived from the data cannot depend on the current parameter
        values either - if it could, it would not be an *initial* guess.
        """
        del x, y
        return None


