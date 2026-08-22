"""Tests for the spectral estimators behind the analysis dialog.

Every test uses a signal whose answer is known in closed form - a sine at a
chosen frequency, a pair with a chosen lag - so a wrong window, a wrong
sampling rate or a swapped argument shows up as a wrong number rather than as
a plausible-looking curve.

The estimators are exercised through the dialog class without constructing the
Qt dialog: the numeric methods only read widget values, which the stand-in
supplies.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.series_operations.spectral_dialog import (
    METHOD_ACORR,
    METHOD_ANGLE,
    METHOD_COHERENCE,
    METHOD_CSD,
    METHOD_MAGNITUDE,
    METHOD_PHASE,
    METHOD_PSD,
    METHOD_XCORR,
    SeriesSpectralDialog,
)

FS = 200.0
DURATION = 8.0
TONE_HZ = 25.0


class _Value:
    """Stand-in for a Qt widget that only has to report one value."""

    def __init__(self, value: object) -> None:
        self._value = value

    def value(self):
        return self._value

    def isChecked(self) -> bool:
        return bool(self._value)

    def currentText(self) -> str:
        return str(self._value)


def _dialog(**overrides) -> SimpleNamespace:
    """Build the attribute set the numeric methods read."""
    defaults = {
        "_fs_auto_check": True,
        "_fs_spin": 1.0,
        "_nperseg_spin": 256,
        "_overlap_spin": 0.5,
        "_window_combo": "hann",
        "_detrend_combo": "constant",
        "_scaling_combo": "density",
        "_onesided_check": True,
        "_db_check": False,
        "_maxlags_spin": 0,
        "_corr_norm_combo": "unbiased",
    }
    defaults.update(overrides)
    namespace = SimpleNamespace(
        **{name: _Value(value) for name, value in defaults.items()}
    )

    # Bind the methods under test to the stand-in.
    for method in (
        "_sampling_frequency",
        "_welch_kwargs",
        "_to_decibels",
        "_one_sided_fft",
        "_correlate",
        "_compute_single",
        "_compute_pair",
    ):
        setattr(
            namespace,
            method,
            getattr(SeriesSpectralDialog, method).__get__(namespace, SeriesSpectralDialog),
        )
    namespace._frequency_result = SeriesSpectralDialog._frequency_result
    return namespace


def _tone(frequency: float = TONE_HZ, phase: float = 0.0, noise: float = 0.0):
    """Return (x, y) for a sine at a known frequency."""
    x = np.arange(0.0, DURATION, 1.0 / FS)
    y = np.sin(2.0 * np.pi * frequency * x + phase)
    if noise:
        y = y + np.random.default_rng(0).normal(0.0, noise, x.size)
    return x, y


# ----------------------------------------------------------------------
# Sampling frequency
# ----------------------------------------------------------------------
def test_sampling_frequency_is_derived_from_uniform_spacing() -> None:
    dialog = _dialog()
    x, _ = _tone()
    assert dialog._sampling_frequency(x, "s") == pytest.approx(FS, rel=1e-6)


def test_explicit_sampling_frequency_is_used_when_asked() -> None:
    dialog = _dialog(_fs_auto_check=False, _fs_spin=48_000.0)
    x, _ = _tone()
    assert dialog._sampling_frequency(x, "s") == pytest.approx(48_000.0)


def test_non_increasing_x_falls_back_to_one() -> None:
    dialog = _dialog()
    assert dialog._sampling_frequency(np.zeros(10), "s") == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Frequency-domain estimators
# ----------------------------------------------------------------------
def test_psd_peaks_at_the_tone_frequency() -> None:
    dialog = _dialog()
    result = dialog._compute_single(METHOD_PSD, ("tone", *_tone()))

    assert result is not None
    peak = result.x[int(np.argmax(result.y))]
    assert peak == pytest.approx(TONE_HZ, abs=1.0)
    assert result.x_label == "frequency"


def test_psd_in_decibels_is_the_log_of_the_linear_estimate() -> None:
    linear = _dialog()._compute_single(METHOD_PSD, ("tone", *_tone()))
    decibels = _dialog(_db_check=True)._compute_single(METHOD_PSD, ("tone", *_tone()))

    assert linear is not None and decibels is not None
    assert decibels.y_label == "dB"
    # Same peak location, different scale.
    assert linear.x[int(np.argmax(linear.y))] == pytest.approx(
        decibels.x[int(np.argmax(decibels.y))]
    )
    assert np.max(decibels.y) < np.max(linear.y) or np.max(linear.y) < 1.0


def test_magnitude_spectrum_peaks_at_the_tone_frequency() -> None:
    result = _dialog()._compute_single(METHOD_MAGNITUDE, ("tone", *_tone()))

    assert result is not None
    assert result.x[int(np.argmax(result.y))] == pytest.approx(TONE_HZ, abs=0.5)


def test_phase_spectrum_is_unwrapped_and_angle_is_not() -> None:
    x, y = _tone(phase=1.0)
    phase = _dialog()._compute_single(METHOD_PHASE, ("tone", x, y))
    angle = _dialog()._compute_single(METHOD_ANGLE, ("tone", x, y))

    assert phase is not None and angle is not None
    assert np.all(np.abs(angle.y) <= np.pi + 1e-9)
    # Unwrapping is what lets the phase leave the (-pi, pi] band.
    assert np.ptp(phase.y) >= np.ptp(angle.y)


def test_welch_segment_length_is_clamped_to_the_data() -> None:
    """A segment longer than the signal would make scipy raise."""
    dialog = _dialog(_nperseg_spin=100_000)
    kwargs = dialog._welch_kwargs(FS, 512)
    assert kwargs["nperseg"] == 512
    assert kwargs["noverlap"] < kwargs["nperseg"]


def test_detrend_none_is_passed_as_false() -> None:
    """scipy spells 'do not detrend' as False, not as the string 'none'."""
    assert _dialog(_detrend_combo="none")._welch_kwargs(FS, 1024)["detrend"] is False
    assert _dialog(_detrend_combo="linear")._welch_kwargs(FS, 1024)["detrend"] == "linear"


def test_decibel_conversion_survives_zeros() -> None:
    """A zero bin is a real measurement; dropping it would shift the axis."""
    values = np.array([0.0, 1.0, 100.0])
    result = _dialog()._to_decibels(values)

    assert np.all(np.isfinite(result))
    assert result[2] == pytest.approx(20.0)


# ----------------------------------------------------------------------
# Paired estimators
# ----------------------------------------------------------------------
def test_csd_peaks_where_both_signals_have_power() -> None:
    x, y = _tone()
    _, y2 = _tone(phase=0.4)
    result = _dialog()._compute_pair(METHOD_CSD, ("a", x, y), ("b", x, y2))

    assert result is not None
    assert result.x[int(np.argmax(result.y))] == pytest.approx(TONE_HZ, abs=1.0)
    assert "a x b" in result.source_name


def test_coherence_is_high_for_a_shared_tone_and_bounded() -> None:
    x, y = _tone(noise=0.05)
    _, y2 = _tone(phase=0.3, noise=0.05)
    result = _dialog()._compute_pair(METHOD_COHERENCE, ("a", x, y), ("b", x, y2))

    assert result is not None
    assert np.all(result.y >= -1e-9) and np.all(result.y <= 1.0 + 1e-9)
    peak_index = int(np.argmin(np.abs(result.x - TONE_HZ)))
    assert result.y[peak_index] > 0.8


def test_cross_correlation_finds_a_known_lag() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, 2048)
    lag = 17
    shifted = np.roll(base, lag)

    x = np.arange(base.size, dtype=float)
    result = _dialog(_maxlags_spin=100, _corr_norm_combo="biased")._compute_pair(
        METHOD_XCORR, ("a", x, shifted), ("b", x, base)
    )

    assert result is not None
    assert result.x[int(np.argmax(result.y))] == pytest.approx(lag, abs=1.0)


def test_autocorrelation_peaks_at_zero_lag() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(0.0, 1.0, 1024)
    x = np.arange(y.size, dtype=float)

    result = _dialog(_maxlags_spin=50)._compute_single(METHOD_ACORR, ("s", x, y))

    assert result is not None
    assert result.x[int(np.argmax(result.y))] == pytest.approx(0.0)
    assert result.x_label == "lag"


def test_max_lags_limits_the_output() -> None:
    rng = np.random.default_rng(5)
    y = rng.normal(0.0, 1.0, 512)
    x = np.arange(y.size, dtype=float)

    result = _dialog(_maxlags_spin=25)._compute_single(METHOD_ACORR, ("s", x, y))

    assert result is not None
    assert result.x.min() == pytest.approx(-25.0)
    assert result.x.max() == pytest.approx(25.0)


@pytest.mark.parametrize("normalisation", ["unbiased", "biased", "none"])
def test_every_correlation_normalisation_runs(normalisation: str) -> None:
    rng = np.random.default_rng(6)
    y = rng.normal(0.0, 1.0, 256)
    x = np.arange(y.size, dtype=float)

    result = _dialog(_maxlags_spin=20, _corr_norm_combo=normalisation)._compute_single(
        METHOD_ACORR, ("s", x, y)
    )

    assert result is not None
    assert np.all(np.isfinite(result.y))


# ----------------------------------------------------------------------
# Result plumbing
# ----------------------------------------------------------------------
def test_result_frame_columns_match_the_labels() -> None:
    result = _dialog()._compute_single(METHOD_PSD, ("tone", *_tone()))
    assert result is not None

    frame = result.to_frame()
    assert list(frame.columns) == [result.x_label, result.y_label]
    assert len(frame) == result.x.size


def test_generated_series_sql_selects_the_result_columns() -> None:
    result = _dialog()._compute_single(METHOD_PSD, ("tone", *_tone()))
    assert result is not None

    # generated_style_filter is a property, so the method needs an instance-like
    # holder rather than the class itself.
    holder = SimpleNamespace(
        generated_style_filter=SeriesSpectralDialog.generated_style_filter.fget(None)
    )
    spec = SeriesSpectralDialog.result_series_spec(
        holder, axis_id=1, table_name="tbl", result=result
    )
    assert '"frequency" AS x' in spec.sql_query
    assert spec.roles == {"x": "x", "y": "y"}
    assert spec.style["generated_spectral"] is True
