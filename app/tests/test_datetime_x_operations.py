"""Series operations on a dated x axis.

``pd.to_numeric`` turns a timestamp column into all-NaN, so an operation
reading its x that way reports "0 usable points" about a series of a
hundred perfectly good ones. ``app/utils/coercion`` was written for
precisely this - its own docstring says so - and only the outlier dialog
was calling it. Everything else broke on every dated series, which is a
lot of series: a time axis is what half of this application draws.

Two halves are tested here, because fixing only the first produces a
result that is right and unreadable:

*Reading* - the operation computes, instead of refusing a full table.
*Writing back* - the result's x returns as timestamps, so it lands on the
axis it came from rather than at x = 1 700 000 000 next to a chart drawn
in dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.render_figure import render_figure_from_descriptor
from app.data.sqlite_repo import SqliteRepo
from app.utils.coercion import coerce_axis, parse_datetimes, to_numeric_axis

DAYS = 60


@pytest.fixture
def dated_figure(repo: SqliteRepo):
    """One figure, one axis, one series whose x is an ISO date column."""
    days = pd.date_range("2024-01-01", periods=DAYS, freq="D")
    repo.import_dataframe(
        pd.DataFrame(
            {
                "t": days.strftime("%Y-%m-%d %H:%M:%S"),
                "v": np.sin(np.arange(DAYS) / 5.0),
            }
        ),
        table_name="ts",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Time Series",
            title="ts", x_label="t", y_label="v", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="sig",
        sql_query="SELECT t AS x, v AS y FROM ts",
        roles={"x": "x", "y": "y"}, style={},
    )
    return figure_id, axis_id


# ----------------------------------------------------------------------
# The parsing underneath, which had its own bug
# ----------------------------------------------------------------------
def test_a_column_of_mixed_precision_timestamps_parses_completely() -> None:
    """pandas infers one format from the first entry and coerces the rest to
    NaT. Computed timestamps come back from SQLite with fractional seconds
    on some rows and not others, so this is not a hypothetical - it is what
    every operation result looked like."""
    values = pd.Series(
        ["2024-01-01 00:00:00", "2024-01-16 16:58:38.426808", "2024-02-01 09:59:19.5"]
    )

    parsed = parse_datetimes(values)

    assert parsed.notna().all()
    assert parsed.iloc[1].microsecond == 426808


@pytest.mark.parametrize(
    "values",
    [
        ["2024-01-01", "2024-01-02"],
        ["2024-01-01 00:00:00", "2024-01-02 12:30:00"],
        ["01/02/2024", "03/02/2024"],
        ["Jan 5 2024", "Feb 6 2024"],
        ["2024-01-01T00:00:00+01:00", "2024-01-02T00:00:00+01:00"],
    ],
)
def test_the_layouts_people_actually_have_all_parse(values: list[str]) -> None:
    """ISO8601 is tried first because it is both exact and fast; the
    guessing path is still there for the rest."""
    assert parse_datetimes(pd.Series(values)).notna().all()


def test_a_column_that_is_not_dates_at_all_stays_unparsed() -> None:
    assert parse_datetimes(pd.Series(["nope", "also nope"])).isna().all()


def test_a_numeric_axis_is_never_read_as_nanoseconds() -> None:
    """The reason coerce_axis tries numbers first: an x running 0..4000 read
    as timestamps is four microseconds of 1 January 1970."""
    coerced, is_temporal = coerce_axis(pd.Series([0.0, 1000.0, 4000.0]))

    assert is_temporal is False
    assert list(coerced) == [0.0, 1000.0, 4000.0]


def test_timestamps_become_seconds_and_keep_their_spacing() -> None:
    seconds = to_numeric_axis(pd.Series(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]))

    assert seconds[1] - seconds[0] == pytest.approx(3600.0)


# ----------------------------------------------------------------------
# Reading: the operations compute at all
# ----------------------------------------------------------------------
def _dialog(cls, repo: SqliteRepo, figure_id: int):
    dialog = cls(repo=repo, figure_id=figure_id)
    dialog.series_selector.reload(select_all_series=True)
    return dialog


def test_roots_finds_the_crossings_of_a_dated_series(
    qapp, repo: SqliteRepo, dated_figure
) -> None:
    from app.series_operations.roots_dialog import SeriesRootsDialog

    figure_id, _axis_id = dated_figure
    results = _dialog(SeriesRootsDialog, repo, figure_id).compute_results()

    # sin(n/5) over 60 daily samples crosses zero four times.
    assert len(results[0].roots) == 4
    # The x values are seconds since the epoch, in 2024.
    assert all(1.7e9 < root.x < 1.8e9 for root in results[0].roots)


def test_peaks_finds_the_maxima_of_a_dated_series(
    qapp, repo: SqliteRepo, dated_figure
) -> None:
    from app.series_operations.peaks_dialog import SeriesPeaksDialog

    figure_id, _axis_id = dated_figure
    results = _dialog(SeriesPeaksDialog, repo, figure_id).compute_results()

    assert results[0].peaks


def test_the_calculus_dialog_differentiates_a_dated_series(
    qapp, repo: SqliteRepo, dated_figure
) -> None:
    from app.series_operations.calculus_dialog import SeriesCalculusDialog

    figure_id, _axis_id = dated_figure
    results = _dialog(SeriesCalculusDialog, repo, figure_id).compute_results()

    assert len(results[0].y) == DAYS


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("control_chart_dialog", "SeriesControlChartDialog"),
        ("outlier_dialog", "SeriesOutlierDialog"),
        ("smoothing_dialog", "SeriesSmoothingDialog"),
        ("spectral_dialog", "SeriesSpectralDialog"),
        ("statistics_dialog", "SeriesStatisticsDialog"),
        ("cluster_dialog", "SeriesClusterDialog"),
    ],
)
def test_every_other_operation_produces_a_result_too(
    qapp, repo: SqliteRepo, dated_figure, module_name: str, class_name: str
) -> None:
    """Each of these read x with pd.to_numeric and refused the series."""
    import importlib

    figure_id, _axis_id = dated_figure
    cls = getattr(
        importlib.import_module(f"app.series_operations.{module_name}"), class_name
    )

    results = _dialog(cls, repo, figure_id).compute_results()

    assert results, f"{class_name} produced nothing from a dated series"


# ----------------------------------------------------------------------
# Writing back: the result lands on the axis it came from
# ----------------------------------------------------------------------
def test_a_result_computed_from_dates_is_written_back_as_dates(
    qapp, repo: SqliteRepo, dated_figure
) -> None:
    from app.series_operations.roots_dialog import SeriesRootsDialog

    figure_id, axis_id = dated_figure
    dialog = _dialog(SeriesRootsDialog, repo, figure_id)
    results = dialog.compute_results()

    dialog.apply_results_to_axis(axis_id, results)

    stored = repo.query_df(
        f'SELECT x FROM "{dialog.result_table_name(axis_id, results[0])}"'
    )
    parsed = parse_datetimes(stored["x"])
    assert parsed.notna().all(), "the x column did not come back as timestamps"
    assert parsed.iloc[0].year == 2024


def test_a_result_computed_from_numbers_stays_numeric(
    qapp, repo: SqliteRepo
) -> None:
    """The write-back is conditional on the source, not on the operation."""
    from app.series_operations.roots_dialog import SeriesRootsDialog

    repo.import_dataframe(
        pd.DataFrame({"n": np.arange(60.0), "v": np.sin(np.arange(60) / 5.0)}),
        table_name="plain",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="P", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="p", x_label="n", y_label="v", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="sig",
        sql_query="SELECT n AS x, v AS y FROM plain",
        roles={"x": "x", "y": "y"}, style={},
    )

    dialog = _dialog(SeriesRootsDialog, repo, figure_id)
    results = dialog.compute_results()
    dialog.apply_results_to_axis(axis_id, results)

    stored = repo.query_df(
        f'SELECT x FROM "{dialog.result_table_name(axis_id, results[0])}"'
    )
    assert pd.to_numeric(stored["x"], errors="coerce").notna().all()


def test_the_result_is_drawn_on_the_same_axis_as_its_source(
    qapp, repo: SqliteRepo, dated_figure
) -> None:
    """The point of the whole exercise: the markers sit on the curve, not at
    x = 1 700 000 000 somewhere off the side of a chart drawn in dates."""
    from app.series_operations.roots_dialog import SeriesRootsDialog

    figure_id, axis_id = dated_figure
    dialog = _dialog(SeriesRootsDialog, repo, figure_id)
    dialog.apply_results_to_axis(axis_id, dialog.compute_results())

    figure = Figure()
    render_figure_from_descriptor(
        figure=figure, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )
    axes = figure.axes[0]

    source = next(line for line in axes.lines if line.get_label() == "sig")
    roots = next(line for line in axes.lines if line.get_label() == "sig - roots")
    source_x = np.asarray(source.get_xdata(), dtype="datetime64[ns]")
    roots_x = np.asarray(roots.get_xdata(), dtype="datetime64[ns]")

    assert np.nanmin(roots_x) >= np.nanmin(source_x)
    assert np.nanmax(roots_x) <= np.nanmax(source_x)
