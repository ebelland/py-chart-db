"""Baseline correction: AsLS and rubber band.

Both estimators are checked against a series built from a known baseline
plus known peaks: the estimate must track the baseline where there is no
peak, and must not be pulled up towards a peak where there is one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.baseline_dialog import (
    BASELINE_ASLS,
    BASELINE_RUBBER,
    SeriesBaselineDialog,
    asls_baseline,
    rubber_band_baseline,
)
from app.utils.dialog_state import clear_state

X = np.linspace(0.0, 100.0, 400)
TRUE_BASELINE = 5.0 + 0.02 * X + 3.0 * np.sin(X / 30.0)
PEAKS = 10.0 * np.exp(-((X - 30.0) ** 2) / 4.0) + 15.0 * np.exp(-((X - 70.0) ** 2) / 8.0)
NO_PEAK = PEAKS < 0.05


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state():
    clear_state("SeriesBaselineDialog")
    yield
    clear_state("SeriesBaselineDialog")


# ----------------------------------------------------------------------
# AsLS
# ----------------------------------------------------------------------
def test_asls_tracks_a_wandering_baseline_away_from_peaks() -> None:
    rng = np.random.default_rng(0)
    y = TRUE_BASELINE + PEAKS + 0.05 * rng.standard_normal(X.size)
    fitted = asls_baseline(y, lam=1e5, p=0.01, iterations=10)
    assert fitted[NO_PEAK] == pytest.approx(TRUE_BASELINE[NO_PEAK], abs=0.3)


def test_asls_does_not_chase_the_peaks() -> None:
    y = TRUE_BASELINE + PEAKS
    fitted = asls_baseline(y, lam=1e5, p=0.01, iterations=10)
    peak_index = int(np.argmax(PEAKS))
    # The baseline under the tallest peak must stay far below the peak's
    # own height, not rise to meet it.
    assert fitted[peak_index] < y[peak_index] - 5.0


def test_asls_rejects_bad_parameters() -> None:
    y = TRUE_BASELINE
    with pytest.raises(ValueError, match="lambda"):
        asls_baseline(y, lam=0.0, p=0.01)
    with pytest.raises(ValueError, match="p must be"):
        asls_baseline(y, lam=1e5, p=1.5)
    with pytest.raises(ValueError, match="at least"):
        asls_baseline(np.array([1.0, 2.0]), lam=1e5, p=0.01)


# ----------------------------------------------------------------------
# Rubber band
# ----------------------------------------------------------------------
def test_rubber_band_matches_a_flat_baseline_exactly() -> None:
    flat = 5.0 + 0.02 * X  # itself convex-hull-flat, so the hull is exact
    y = flat + PEAKS
    baseline = rubber_band_baseline(X, y)
    assert baseline[NO_PEAK] == pytest.approx(flat[NO_PEAK], abs=1e-6)


def test_rubber_band_never_goes_above_the_data() -> None:
    y = TRUE_BASELINE + PEAKS
    baseline = rubber_band_baseline(X, y)
    assert np.all(baseline <= y + 1e-9)


def test_rubber_band_skips_a_single_upward_spike() -> None:
    """A lower hull is not fooled by one point sticking up - it draws the
    straight line under it, the way a rubber band actually would."""
    x = np.arange(10.0)
    y = np.full(10, 10.0)
    y[5] = 100.0  # a "peak" - the hull must not rise to meet it
    baseline = rubber_band_baseline(x, y)
    assert baseline[5] == pytest.approx(10.0)


def test_rubber_band_does_follow_a_downward_outlier() -> None:
    """The opposite case is not a limitation to work around: a lower hull
    passes through the global minimum by definition - a single point below
    everything else *is* part of the true lower envelope."""
    x = np.arange(10.0)
    y = np.full(10, 10.0)
    y[5] = -100.0
    baseline = rubber_band_baseline(x, y)
    assert baseline[5] == pytest.approx(-100.0)


# ----------------------------------------------------------------------
# The dialog, end to end
# ----------------------------------------------------------------------
@pytest.fixture
def figure_with_spectrum(repo: SqliteRepo):
    y = TRUE_BASELINE + PEAKS
    repo.import_dataframe(
        pd.DataFrame({"x": X, "y": y}), table_name="spec", normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="spectrum", x_label="x", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="spec",
        sql_query="SELECT x, y FROM spec",
        roles={"x": "x", "y": "y"}, style={},
    )
    return figure_id, axis_id


def test_the_dialog_applies_an_asls_correction(
    qapp, repo: SqliteRepo, figure_with_spectrum
) -> None:
    figure_id, axis_id = figure_with_spectrum
    dialog = SeriesBaselineDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(BASELINE_ASLS)
        results = dialog.compute_results()
        assert len(results) == 1
        assert results[0].corrected[NO_PEAK] == pytest.approx(
            np.zeros(int(NO_PEAK.sum())), abs=0.5
        )

        assert dialog.apply() is True
    finally:
        dialog.close()

    series = repo.get_series(axis_id) or []
    names = {str(row["name"]) for row in series}
    assert any("corrected" in name for name in names)
    assert any("baseline" in name for name in names)


def test_hiding_the_baseline_line_still_writes_the_corrected_series(
    qapp, repo: SqliteRepo, figure_with_spectrum
) -> None:
    figure_id, axis_id = figure_with_spectrum
    dialog = SeriesBaselineDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(BASELINE_RUBBER)
        dialog._draw_baseline_check.setChecked(False)
        assert dialog.apply() is True
    finally:
        dialog.close()

    series = repo.get_series(axis_id) or []
    names = {str(row["name"]) for row in series}
    assert any("corrected" in name for name in names)
    assert not any(name.endswith("- baseline") for name in names)


def test_rubber_band_hides_the_asls_only_parameters(
    qapp, repo: SqliteRepo, figure_with_spectrum
) -> None:
    figure_id, _axis_id = figure_with_spectrum
    dialog = SeriesBaselineDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(BASELINE_RUBBER)
        assert dialog._lambda_spin.isHidden()
        assert dialog._p_spin.isHidden()
    finally:
        dialog.close()
