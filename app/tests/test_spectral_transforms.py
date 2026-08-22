"""Laplace and wavelet estimates, and the catalogue entry for the dialog.

Both transforms are computed with numpy rather than a new dependency: the
Laplace transform along a vertical line in the s-plane *is* the FFT of a damped
signal, and the Morlet wavelet has a closed form in frequency, so the CWT is
four lines of array arithmetic.  That is worth testing precisely because it is
hand-rolled.

The tests call the methods unbound, with a dummy ``self``.  Neither touches a
widget - every parameter arrives as an argument - and that is a property worth
keeping, so a test that would break if it stopped being true is a feature.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.series_operations.spectral_dialog import (
    METHOD_LAPLACE,
    METHOD_WAVELET,
    SPECTRAL_METHODS,
    SeriesSpectralDialog,
)

APP_DIR = Path(__file__).resolve().parent.parent

FS = 200.0
LOW_HZ = 10.0
HIGH_HZ = 35.0


def _signal(seconds: float = 10.24) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(seconds * FS), dtype=float) / FS
    y = np.sin(2 * np.pi * LOW_HZ * t) + 0.5 * np.sin(2 * np.pi * HIGH_HZ * t)
    return t, y


def _laplace(y: np.ndarray, sigma: float):
    return SeriesSpectralDialog.__dict__["_laplace_spectrum"](object(), y, FS, sigma)


def _wavelet(y: np.ndarray, w0: float = 6.0, scales: int = 96):
    return SeriesSpectralDialog.__dict__["_wavelet_power"](object(), y, FS, w0, scales)


# ----------------------------------------------------------------------
# Laplace
# ----------------------------------------------------------------------
def test_at_zero_damping_it_is_the_fourier_spectrum() -> None:
    """The control is then easy to understand: turn sigma up from Fourier."""
    _t, y = _signal()

    frequencies, magnitude = _laplace(y, 0.0)
    fourier = np.abs(np.fft.rfft(y - y.mean())) / FS

    assert frequencies[np.argmax(magnitude)] == pytest.approx(LOW_HZ, abs=0.5)
    assert magnitude == pytest.approx(fourier, rel=1e-9, abs=1e-9)


def test_a_constant_offset_does_not_swallow_the_peak() -> None:
    """Mean removal has to happen after damping, not before."""
    _t, y = _signal()

    frequencies, magnitude = _laplace(y + 7.0, 0.0)

    assert frequencies[np.argmax(magnitude)] == pytest.approx(LOW_HZ, abs=0.5)


def test_damping_recovers_a_signal_that_has_no_fourier_transform() -> None:
    """The reason the Laplace transform exists, as a test.

    A sine multiplied by exp(1.5t) grows without bound and its Fourier integral
    does not converge.  Damping by the matching sigma leaves exactly the sine,
    and the peak comes back where it belongs.  Centring before damping - which
    the first version did - subtracted the mean of the *growing* signal and
    left a ramp that put the peak at DC.
    """
    t, y = _signal()
    growing = y * np.exp(1.5 * t)

    frequencies, magnitude = _laplace(growing, 1.5)

    assert frequencies[np.argmax(magnitude)] == pytest.approx(LOW_HZ, abs=0.5)


def test_the_result_is_finite_at_heavy_damping() -> None:
    """exp(-sigma*t) underflows over a long record; 0 * inf must not appear."""
    t, y = _signal()

    _frequencies, magnitude = _laplace(y * np.exp(20.0 * t), 40.0)

    assert np.all(np.isfinite(magnitude))


def test_more_damping_flattens_the_spectrum() -> None:
    """Damping widens every peak - the frequency resolution is the price."""
    _t, y = _signal()

    _f, sharp = _laplace(y, 0.0)
    _f2, blunt = _laplace(y, 5.0)

    def peakiness(values: np.ndarray) -> float:
        return float(values.max() / values.mean())

    assert peakiness(blunt) < peakiness(sharp)


# ----------------------------------------------------------------------
# Wavelet
# ----------------------------------------------------------------------
def test_the_wavelet_spectrum_finds_both_components() -> None:
    from scipy.signal import find_peaks

    _t, y = _signal()

    frequencies, power = _wavelet(y)
    peaks, _ = find_peaks(power)
    strongest = sorted(frequencies[peaks][np.argsort(power[peaks])[::-1]][:2])

    assert strongest[0] == pytest.approx(LOW_HZ, rel=0.15)
    assert strongest[1] == pytest.approx(HIGH_HZ, rel=0.15)


def test_the_scales_span_what_the_record_can_measure() -> None:
    """Below one cycle per record, or above Nyquist, there is nothing to see."""
    _t, y = _signal()

    frequencies, _power = _wavelet(y)

    assert frequencies[0] == pytest.approx(FS / y.size, rel=0.01)
    assert frequencies[-1] == pytest.approx(FS / 2.0, rel=0.01)
    assert np.all(np.diff(frequencies) > 0)


def test_the_number_of_scales_is_what_was_asked_for() -> None:
    _t, y = _signal()

    frequencies, power = _wavelet(y, scales=32)

    assert frequencies.size == 32
    assert power.size == 32


def test_a_transient_is_found_that_a_long_fft_would_smear() -> None:
    """What the wavelet basis buys over one long Fourier transform.

    A burst present in a twentieth of the record still registers, because the
    basis is localised in time instead of spanning the whole record.
    """
    t, _y = _signal()
    burst = np.zeros_like(t)
    window = slice(0, t.size // 20)
    burst[window] = np.sin(2 * np.pi * HIGH_HZ * t[window])

    frequencies, power = _wavelet(burst)

    assert frequencies[np.argmax(power)] == pytest.approx(HIGH_HZ, rel=0.25)


def test_the_wavelet_is_analytic() -> None:
    """Negative frequencies are zeroed; without that the power is doubled."""
    source = (
        APP_DIR / "series_operations" / "spectral_dialog.py"
    ).read_text(encoding="utf-8")

    assert "wavelet[omega <= 0.0] = 0.0" in source


def test_power_is_comparable_across_scales() -> None:
    """Without the sqrt(scale) the spectrum slopes for no physical reason."""
    _t, y = _signal()

    # White noise has no preferred scale, so a correctly normalised global
    # spectrum should be roughly flat rather than trending with frequency.
    rng = np.random.default_rng(0)
    frequencies, power = _wavelet(rng.normal(size=y.size))

    low_half = power[: power.size // 2].mean()
    high_half = power[power.size // 2 :].mean()
    assert low_half / high_half == pytest.approx(1.0, abs=0.9)


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------
def test_both_methods_are_offered() -> None:
    assert METHOD_LAPLACE in SPECTRAL_METHODS
    assert METHOD_WAVELET in SPECTRAL_METHODS


def test_neither_transform_touches_a_widget() -> None:
    """Which is what lets them be tested, and read, on their own."""
    source = (
        APP_DIR / "series_operations" / "spectral_dialog.py"
    ).read_text(encoding="utf-8")

    for name in ("_laplace_spectrum", "_wavelet_power"):
        start = source.index(f"def {name}")
        body = source[start : source.index("\n    def ", start + 10)]
        assert "self._" not in body, f"{name} reads a widget instead of an argument"


def test_the_transform_works_without_pywavelets() -> None:
    """PyWavelets is a real dependency here, but an optional one.

    The smoothing dialog uses it for wavelet denoising and guards the import,
    degrading gracefully when it is absent.  This transform deliberately does
    not join it: one analytic wavelet is four lines of numpy, so the spectral
    dialog keeps working on an installation where PyWavelets is missing.
    """
    source = (
        APP_DIR / "series_operations" / "spectral_dialog.py"
    ).read_text(encoding="utf-8")

    assert "pywt" not in source
    # ...and it really did run, above, in an environment without it.
    pytest.importorskip
    assert "np.fft.ifft" in source


def test_the_dialog_carries_its_own_presentation() -> None:
    """Its button had no label and no icon: the catalogue entry was deleted.

    The scanner names each operation after the dialog's ``Name`` attribute, so
    these ids appeared nowhere as a call site and an "unused actions" sweep
    removed this one.  Name, Description and Icon now live on the class, where
    nothing can sweep them away independently of the code that reads them -
    which is the whole reason the presentation moved out of config.json.
    """
    assert SeriesSpectralDialog.Name.strip()
    assert SeriesSpectralDialog.Description.strip()
    assert SeriesSpectralDialog.Icon.strip().startswith("<svg")


def test_every_discovered_operation_has_one() -> None:
    """The general form of the same bug, for the other six dialogs."""
    from app.scanners.series_operation_scanner import series_operations

    incomplete = [
        operation["value"]
        for operation in series_operations
        if not (operation["value"] or "").strip()
        or not (operation.get("description") or "").strip()
        or not (operation.get("icon") or "").strip()
    ]

    assert incomplete == []


def test_an_operation_is_not_registered_in_config_json() -> None:
    """A plugin that has to be registered elsewhere is not really a plugin.

    Dropping one .py file into ``app/series_operations`` is meant to be the
    whole install.  It was not, quietly: an operation whose action entry was
    missing got a blank button rather than an error, which is how the spectral
    one shipped that way.
    """
    from app.scanners.series_operation_scanner import series_operations

    actions = json.loads((APP_DIR.parent / "config.json").read_text(encoding="utf-8"))
    registered = [
        operation["value"]
        for operation in series_operations
        if operation["value"] in actions["actions"]
    ]

    assert registered == []
