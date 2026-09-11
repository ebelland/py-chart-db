"""Attribute control charts: p, np, c, u.

These live in ``control_chart_dialog`` alongside the variables charts (I-MR,
X-bar) - one dialog, one "Chart:" combo, because "control chart" is one
concept to a user regardless of which distribution the limits come from.
This file keeps the attribute-specific coverage: the pure ``attribute_limits``
numerics against Montgomery's textbook worked examples, and the dialog's own
sample-size-column picker end to end. Nelson-rule coverage (including which
rules a varying-limit chart is allowed to run) lives with the rest of
``_find_violations`` in test_new_operations.py, since that logic is shared
with the variables charts rather than duplicated for this family.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.control_chart_dialog import (
    CHART_C,
    CHART_NP,
    CHART_P,
    CHART_U,
    SeriesControlChartDialog,
    attribute_limits,
)
from app.utils.dialog_state import clear_state

# Montgomery, Introduction to Statistical Quality Control, Example 6.1:
# 20 samples of n = 50, defective counts, p-bar = 0.214.
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


@pytest.fixture(autouse=True)
def _no_persisted_dialog_state():
    clear_state("SeriesControlChartDialog")
    yield
    clear_state("SeriesControlChartDialog")


# ----------------------------------------------------------------------
# p chart
# ----------------------------------------------------------------------
def test_p_chart_limits_match_the_textbook() -> None:
    stat, center, upper, lower, meta = attribute_limits(
        CHART_P, P_DEFECTIVES, np.full(20, 50.0), 3.0
    )
    assert meta["p-bar"] == pytest.approx(0.2140, abs=1e-4)
    assert center == pytest.approx(0.2140, abs=1e-4)
    assert upper[0] == pytest.approx(0.388, abs=1e-3)
    assert lower[0] == pytest.approx(0.040, abs=1e-3)
    # Sample 15 (0-based 14): 22/50 = 0.44, above the UCL - the only point
    # beyond its own limit in this table.
    beyond = set(np.flatnonzero((stat > upper) | (stat < lower)).tolist())
    assert beyond == {14}


def test_p_chart_limits_vary_with_the_sample_size() -> None:
    counts = np.array([5, 5, 5, 5], dtype=float)
    sizes = np.array([50, 100, 200, 400], dtype=float)
    _stat, _center, upper, lower, _meta = attribute_limits(CHART_P, counts, sizes, 3.0)
    # A bigger sample tightens the band.
    assert np.all(np.diff(upper) < 0)
    assert np.all(np.diff(lower[lower > 0]) >= 0) or np.all(lower == 0)


def test_p_chart_lower_limit_never_goes_below_zero() -> None:
    _stat, _center, _upper, lower, _meta = attribute_limits(
        CHART_P, np.array([1, 0, 2, 1], dtype=float), np.full(4, 30.0), 3.0
    )
    assert np.all(lower >= 0.0)


# ----------------------------------------------------------------------
# np chart
# ----------------------------------------------------------------------
def test_np_chart_centre_is_n_times_pbar() -> None:
    _stat, center, upper, lower, meta = attribute_limits(
        CHART_NP, P_DEFECTIVES, np.full(20, 50.0), 3.0
    )
    assert center == pytest.approx(50 * meta["p-bar"])
    assert np.ptp(upper) == 0.0 and np.ptp(lower) == 0.0  # constant limits
    assert lower[0] >= 0.0


# ----------------------------------------------------------------------
# c chart
# ----------------------------------------------------------------------
def test_c_chart_limits_match_the_textbook() -> None:
    stat, center, upper, lower, meta = attribute_limits(
        CHART_C, C_DEFECTS, np.ones_like(C_DEFECTS), 3.0
    )
    assert meta["c-bar"] == pytest.approx(19.85, abs=0.01)
    assert upper[0] == pytest.approx(33.21, abs=0.01)
    assert lower[0] == pytest.approx(6.48, abs=0.01)
    # Montgomery: samples 6 and 20 (0-based 5 and 19) are out of control.
    beyond = set(np.flatnonzero((stat > upper) | (stat < lower)).tolist())
    assert beyond == {5, 19}


# ----------------------------------------------------------------------
# u chart
# ----------------------------------------------------------------------
def test_u_chart_is_defects_over_units_with_a_per_point_band() -> None:
    counts = np.array([5, 3, 8, 4, 6], dtype=float)
    sizes = np.array([10, 12, 11, 9, 10], dtype=float)
    stat, center, upper, lower, meta = attribute_limits(CHART_U, counts, sizes, 3.0)
    assert stat == pytest.approx(counts / sizes)
    assert center == pytest.approx(counts.sum() / sizes.sum())
    assert meta["u-bar"] == pytest.approx(center)
    assert np.ptp(upper) > 0.0  # the band moves point to point


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
    dialog = SeriesControlChartDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(CHART_P)
        dialog._refresh_size_columns()
        dialog._size_column_combo.setCurrentText("inspected")

        results = dialog.compute_results()
        assert len(results) == 1
        assert results[0].center == pytest.approx(0.214, abs=1e-3)
        assert [v.index for v in results[0].violations] == [14]

        assert dialog.apply() is True
    finally:
        dialog.close()

    series = repo.get_series(axis_id) or []
    names = {str(row["name"]) for row in series}
    # the points, plus UCL / LCL / CL / signals
    assert any(name.endswith(f"- {CHART_P.split(' ')[0]}") for name in names)
    assert any("UCL" in name for name in names)
    assert any("signals" in name for name in names)


def test_a_size_column_that_is_not_there_is_a_clear_error(
    qapp, repo: SqliteRepo, figure_with_counts
) -> None:
    figure_id, _axis_id = figure_with_counts
    dialog = SeriesControlChartDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(CHART_P)
        dialog._size_column_combo.clear()  # nothing picked
        with pytest.raises(ValueError, match="sample size"):
            dialog.compute_results()
    finally:
        dialog.close()


def test_c_chart_needs_no_size_column(
    qapp, repo: SqliteRepo, figure_with_counts
) -> None:
    figure_id, _axis_id = figure_with_counts
    dialog = SeriesControlChartDialog(repo=repo, figure_id=figure_id)
    try:
        dialog.model_combo.setCurrentText(CHART_C)
        dialog._size_column_combo.clear()
        results = dialog.compute_results()
        assert len(results) == 1
        assert results[0].chart == CHART_C
    finally:
        dialog.close()


def test_the_chart_combo_lists_variables_charts_then_attribute_charts(
    qapp, repo: SqliteRepo, figure_with_counts
) -> None:
    """One decision, not two: a user should not have to know in advance
    which family their data belongs to before they can find the tool."""
    from app.series_operations.control_chart_dialog import (
        ATTRIBUTE_CHARTS,
        VARIABLES_CHARTS,
    )

    figure_id, _axis_id = figure_with_counts
    dialog = SeriesControlChartDialog(repo=repo, figure_id=figure_id)
    try:
        labels = [
            dialog.model_combo.itemText(i) for i in range(dialog.model_combo.count())
        ]
        # itemText is "" for the separator QComboBox.insertSeparator adds.
        labels = [label for label in labels if label]
        assert labels == [*VARIABLES_CHARTS, *ATTRIBUTE_CHARTS]
    finally:
        dialog.close()
