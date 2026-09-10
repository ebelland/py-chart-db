"""Attribute control charts: p, np, c, u.

The numerics are pure module functions, tested against Montgomery's
textbook worked examples so a wrong constant or a swapped argument shows up
as a wrong limit rather than a plausible-looking band. One end-to-end test
drives the Qt dialog through an Apply to prove the shell, the sample-size
column picker and the multi-series result all hang together.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.attribute_chart_dialog import (
    ATTR_C,
    ATTR_NP,
    ATTR_P,
    ATTR_U,
    SeriesAttributeChartDialog,
    attribute_limits,
    attribute_violations,
)

# Montgomery, Introduction to Statistical Quality Control, Example 6.1:
# 20 samples of n = 50, defective counts, p-bar = 0.2313 in the text's
# trial limits; here the plain (unrevised) chart.
P_DEFECTIVES = np.array(
    [12, 15, 8, 10, 4, 7, 16, 9, 14, 10, 5, 6, 17, 12, 22, 8, 10, 5, 13, 11],
    dtype=float,
)
# Montgomery Example 6.3: 26 samples, defect counts, c-bar = 19.85.
C_DEFECTS = np.array(
    [21, 24, 16, 12, 15, 5, 28, 20, 31, 25, 20, 24, 16, 19, 10,
     17, 13, 22, 18, 39, 30, 24, 16, 19, 17, 15],
    dtype=float,
)


# ----------------------------------------------------------------------
# p chart
# ----------------------------------------------------------------------
def test_p_chart_limits_match_the_textbook() -> None:
    stat, center, upper, lower, meta = attribute_limits(
        ATTR_P, P_DEFECTIVES, np.full(20, 50.0), 3.0
    )
    assert meta["p-bar"] == pytest.approx(0.2140, abs=1e-4)
    assert center == pytest.approx(0.2140, abs=1e-4)
    assert upper[0] == pytest.approx(0.388, abs=1e-3)
    assert lower[0] == pytest.approx(0.040, abs=1e-3)
    # Sample 15 (0-based 14): 22/50 = 0.44, above the UCL.
    assert attribute_violations(stat, upper, lower, center, runs=False) == [14]


def test_p_chart_limits_vary_with_the_sample_size() -> None:
    counts = np.array([5, 5, 5, 5], dtype=float)
    sizes = np.array([50, 100, 200, 400], dtype=float)
    _stat, _center, upper, lower, _meta = attribute_limits(ATTR_P, counts, sizes, 3.0)
    # A bigger sample tightens the band.
    assert np.all(np.diff(upper) < 0)
    assert np.all(np.diff(lower[lower > 0]) >= 0) or np.all(lower == 0)


def test_p_chart_lower_limit_never_goes_below_zero() -> None:
    _stat, _center, _upper, lower, _meta = attribute_limits(
        ATTR_P, np.array([1, 0, 2, 1], dtype=float), np.full(4, 30.0), 3.0
    )
    assert np.all(lower >= 0.0)


# ----------------------------------------------------------------------
# np chart
# ----------------------------------------------------------------------
def test_np_chart_centre_is_n_times_pbar() -> None:
    _stat, center, upper, lower, meta = attribute_limits(
        ATTR_NP, P_DEFECTIVES, np.full(20, 50.0), 3.0
    )
    assert center == pytest.approx(50 * meta["p-bar"])
    assert np.ptp(upper) == 0.0 and np.ptp(lower) == 0.0  # constant limits
    assert lower[0] >= 0.0


# ----------------------------------------------------------------------
# c chart
# ----------------------------------------------------------------------
def test_c_chart_limits_match_the_textbook() -> None:
    stat, center, upper, lower, meta = attribute_limits(
        ATTR_C, C_DEFECTS, np.ones_like(C_DEFECTS), 3.0
    )
    assert meta["c-bar"] == pytest.approx(19.85, abs=0.01)
    assert upper[0] == pytest.approx(33.21, abs=0.01)
    assert lower[0] == pytest.approx(6.48, abs=0.01)
    # Montgomery: samples 6 and 20 (0-based 5 and 19) are out of control.
    assert attribute_violations(stat, upper, lower, center, runs=False) == [5, 19]


# ----------------------------------------------------------------------
# u chart
# ----------------------------------------------------------------------
def test_u_chart_is_defects_over_units_with_a_per_point_band() -> None:
    counts = np.array([5, 3, 8, 4, 6], dtype=float)
    sizes = np.array([10, 12, 11, 9, 10], dtype=float)
    stat, center, upper, lower, meta = attribute_limits(ATTR_U, counts, sizes, 3.0)
    assert stat == pytest.approx(counts / sizes)
    assert center == pytest.approx(counts.sum() / sizes.sum())
    assert meta["u-bar"] == pytest.approx(center)
    assert np.ptp(upper) > 0.0  # the band moves point to point


def test_the_runs_rule_flags_a_long_one_sided_stretch() -> None:
    # Eight in a row above the centre, all inside the limits.
    stat = np.array([0.30] * 8 + [0.10] * 4, dtype=float)
    upper = np.full(stat.size, 0.50)
    lower = np.zeros(stat.size)
    assert attribute_violations(stat, upper, lower, 0.20, runs=False) == []
    assert 7 in attribute_violations(stat, upper, lower, 0.20, runs=True)


# ----------------------------------------------------------------------
# The dialog, end to end
# ----------------------------------------------------------------------
@pytest.fixture
def figure_with_counts(repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame(
            {
                "sample": np.arange(1, 21, dtype=float),
                "defectives": P_DEFECTIVES,
                "inspected": np.full(20, 50.0),
            }
        ),
        table_name="qc",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="QC", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="defectives", x_label="sample", y_label="d", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="run",
        sql_query="SELECT sample, defectives, inspected FROM qc",
        roles={"x": "sample", "y": "defectives"}, style={},
    )
    return figure_id, axis_id


def test_the_dialog_applies_a_p_chart_with_the_picked_size_column(
    qapp, repo: SqliteRepo, figure_with_counts
) -> None:
    figure_id, axis_id = figure_with_counts
    dialog = SeriesAttributeChartDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(ATTR_P)
        dialog._refresh_size_columns()
        dialog._size_column_combo.setCurrentText("inspected")

        results = dialog.compute_results()
        assert len(results) == 1
        assert results[0].center == pytest.approx(0.214, abs=1e-3)
        assert results[0].violation_index == (14,)

        assert dialog.apply() is True
    finally:
        dialog.close()

    series = repo.get_series(axis_id) or []
    names = {str(row["name"]) for row in series}
    # the points, plus UCL / LCL / CL / signals
    assert any(name.endswith("- p") for name in names)
    assert any("UCL" in name for name in names)
    assert any("signals" in name for name in names)


def test_a_size_column_that_is_not_there_is_a_clear_error(
    qapp, repo: SqliteRepo, figure_with_counts
) -> None:
    figure_id, _axis_id = figure_with_counts
    dialog = SeriesAttributeChartDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(ATTR_P)
        dialog._size_column_combo.clear()  # nothing picked
        with pytest.raises(ValueError, match="sample size"):
            dialog.compute_results()
    finally:
        dialog.close()


def test_c_chart_needs_no_size_column(
    qapp, repo: SqliteRepo, figure_with_counts
) -> None:
    figure_id, _axis_id = figure_with_counts
    dialog = SeriesAttributeChartDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(ATTR_C)
        dialog._size_column_combo.clear()
        results = dialog.compute_results()
        assert len(results) == 1
        assert results[0].model == ATTR_C
    finally:
        dialog.close()
