"""Filtering: IIR/FIR filters, analytic-signal derivatives, detrend.

Every numeric test uses a signal whose answer is known: a filter isolating
one of two mixed tones is checked against the tone itself, an AM envelope
against its own modulation, an instantaneous frequency against the tone
that produced it, so a swapped argument or a wrong Nyquist scaling shows up
as a wrong number rather than a plausible-looking curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.utils.dialog_state import clear_state
from app.series_operations.filter_dialog import (
    ANALYTIC_ENVELOPE,
    ANALYTIC_FREQUENCY,
    DETREND_LINEAR,
    FILTER_ANALYTIC,
    FILTER_DETREND,
    FILTER_FIR,
    FILTER_IIR,
    RESP_BANDPASS,
    RESP_LOWPASS,
    SeriesFilterDialog,
    analytic_signal,
    apply_detrend,
    apply_fir_filter,
    apply_iir_filter,
)

FS = 200.0
T = np.arange(0.0, 10.0, 1.0 / FS)
LOW_TONE = np.sin(2 * np.pi * 2.0 * T)
HIGH_TONE = np.sin(2 * np.pi * 40.0 * T)
MIXED = LOW_TONE + HIGH_TONE
EDGE = slice(150, -150)  # away from the filters' settling transients


# ----------------------------------------------------------------------
# IIR
# ----------------------------------------------------------------------
def test_iir_lowpass_recovers_the_low_tone_from_a_mix() -> None:
    filtered = apply_iir_filter(
        MIXED, FS, family="butter", response=RESP_LOWPASS, order=4, cutoff=10.0
    )
    assert filtered[EDGE] == pytest.approx(LOW_TONE[EDGE], abs=0.05)


def test_iir_is_zero_phase() -> None:
    """sosfiltfilt filters forward and back, so there is no lag to correct."""
    filtered = apply_iir_filter(
        LOW_TONE, FS, family="butter", response=RESP_LOWPASS, order=4, cutoff=10.0
    )
    assert filtered[EDGE] == pytest.approx(LOW_TONE[EDGE], abs=0.02)


def test_iir_bandpass_needs_two_cutoffs_in_order() -> None:
    with pytest.raises(ValueError, match="second cutoff"):
        apply_iir_filter(
            MIXED, FS, family="butter", response=RESP_BANDPASS, order=4,
            cutoff=10.0, cutoff2=5.0,
        )


def test_iir_rejects_a_cutoff_past_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        apply_iir_filter(
            MIXED, FS, family="butter", response=RESP_LOWPASS, order=4, cutoff=FS,
        )


def test_chebyshev_and_elliptic_families_run() -> None:
    for family, kwargs in (
        ("cheby1", {"ripple": 1.0}),
        ("cheby2", {"atten": 40.0}),
        ("bessel", {}),
        ("ellip", {"ripple": 1.0, "atten": 40.0}),
    ):
        filtered = apply_iir_filter(
            MIXED, FS, family=family, response=RESP_LOWPASS, order=4,
            cutoff=10.0, **kwargs,
        )
        assert filtered.shape == MIXED.shape
        assert np.all(np.isfinite(filtered))


# ----------------------------------------------------------------------
# FIR
# ----------------------------------------------------------------------
def test_fir_lowpass_recovers_the_low_tone_from_a_mix() -> None:
    filtered = apply_fir_filter(
        MIXED, FS, numtaps=201, window="hamming", response=RESP_LOWPASS, cutoff=10.0
    )
    assert filtered[EDGE] == pytest.approx(LOW_TONE[EDGE], abs=0.05)


def test_fir_forces_an_odd_tap_count_for_a_response_needing_nyquist() -> None:
    """An even-length FIR highpass/bandstop has a null at Nyquist; firwin
    would raise - the wrapper bumps the count instead of surfacing that."""
    from app.series_operations.filter_dialog import RESP_HIGHPASS

    filtered = apply_fir_filter(
        MIXED, FS, numtaps=100, window="hamming", response=RESP_HIGHPASS, cutoff=10.0
    )
    assert filtered.shape == MIXED.shape


# ----------------------------------------------------------------------
# Analytic signal
# ----------------------------------------------------------------------
def test_the_envelope_of_an_am_signal_tracks_its_modulation() -> None:
    modulation = 1.0 + 0.5 * np.sin(2.0 * np.pi * 1.0 * T)
    carrier = np.sin(2.0 * np.pi * 20.0 * T)
    envelope = analytic_signal(modulation * carrier, FS, ANALYTIC_ENVELOPE)
    assert np.corrcoef(envelope[EDGE], modulation[EDGE])[0, 1] > 0.99


def test_instantaneous_frequency_of_a_pure_tone_is_its_own_frequency() -> None:
    tone = np.sin(2.0 * np.pi * 20.0 * T)
    frequency = analytic_signal(tone, FS, ANALYTIC_FREQUENCY)
    assert frequency.size == tone.size
    assert np.mean(frequency[EDGE]) == pytest.approx(20.0, abs=0.1)


# ----------------------------------------------------------------------
# Detrend
# ----------------------------------------------------------------------
def test_linear_detrend_removes_a_ramp() -> None:
    ramp = np.linspace(0.0, 5.0, T.size)
    residual = apply_detrend(ramp, DETREND_LINEAR)
    assert np.std(residual) < 1e-6


# ----------------------------------------------------------------------
# The dialog, end to end
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_persisted_dialog_state():
    """Dialog state (model/response/cutoffs) survives in the real config.json
    across runs - reset it so these tests never inherit another run's combo
    selection, and never leave one behind for the next."""
    clear_state("SeriesFilterDialog")
    yield
    clear_state("SeriesFilterDialog")


@pytest.fixture
def figure_with_mixed_signal(repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame({"t": T, "signal": MIXED}),
        table_name="sig", normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="mix", x_label="t", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="mix",
        sql_query="SELECT t, signal FROM sig",
        roles={"x": "t", "y": "signal"}, style={},
    )
    return figure_id, axis_id


def test_the_dialog_applies_an_iir_lowpass(
    qapp, repo: SqliteRepo, figure_with_mixed_signal
) -> None:
    figure_id, axis_id = figure_with_mixed_signal
    dialog = SeriesFilterDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(FILTER_IIR)
        dialog._cutoff1_spin.setValue(10.0)

        results = dialog.compute_results()
        assert len(results) == 1
        assert results[0].y[EDGE] == pytest.approx(LOW_TONE[EDGE], abs=0.05)

        assert dialog.apply() is True
    finally:
        dialog.close()

    series = repo.get_series(axis_id) or []
    names = {str(row["name"]) for row in series}
    assert any("IIR" in name for name in names)


def test_the_dialog_derives_an_envelope(
    qapp, repo: SqliteRepo, figure_with_mixed_signal
) -> None:
    figure_id, _axis_id = figure_with_mixed_signal
    dialog = SeriesFilterDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(FILTER_ANALYTIC)
        results = dialog.compute_results()
        assert len(results) == 1
        assert results[0].y.size == results[0].x.size
    finally:
        dialog.close()


def test_a_response_needing_a_second_cutoff_shows_that_row(
    qapp, repo: SqliteRepo, figure_with_mixed_signal
) -> None:
    figure_id, _axis_id = figure_with_mixed_signal
    dialog = SeriesFilterDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(FILTER_FIR)
        dialog._response_combo.setCurrentText("Lowpass")
        assert dialog._cutoff2_spin.isHidden()
        dialog._response_combo.setCurrentText("Bandpass")
        assert not dialog._cutoff2_spin.isHidden()
    finally:
        dialog.close()


def test_detrend_needs_neither_fs_nor_a_response(
    qapp, repo: SqliteRepo, figure_with_mixed_signal
) -> None:
    figure_id, _axis_id = figure_with_mixed_signal
    dialog = SeriesFilterDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(FILTER_DETREND)
        results = dialog.compute_results()
        assert len(results) == 1
        assert dialog._fs_auto_check.isHidden()
        assert dialog._response_combo.isHidden()
    finally:
        dialog.close()
