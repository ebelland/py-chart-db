"""Downsampling a large series before it ever reaches pandas.

Per todo.txt P2-5: above roughly 50k points, repaint cost dominates, not the
SQL/descriptor pipeline. Rather than a Python-side thinning pass (LTTB) run
after the full result is already a DataFrame - which pays the cost this is
meant to avoid - the row count and the decimation both happen inside
SQLite: one COUNT(*), and, only past the configured threshold, one row kept
out of every stride via ROW_NUMBER(), ordered by the query's own first
column (the x role, by this application's own SELECT x, y FROM ... convention).

A figure that names no threshold gets DEFAULT_DOWNSAMPLE_THRESHOLD (1,000):
enough to keep the shape of any ordinary curve, few enough that a redraw
stays quick on a huge series. An explicit 0 turns decimation off entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.render_figure import (
    DEFAULT_DOWNSAMPLE_THRESHOLD,
    OPT_DOWNSAMPLE_THRESHOLD,
    render_figure_from_descriptor,
)
from app.data.sqlite_repo import SqliteRepo

# `repo` (a fresh SqliteRepo at this test's tmp_db_path) comes from conftest.py.


def _import_points(repo: SqliteRepo, n: int, table: str = "big") -> None:
    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(n), "y": np.sin(np.arange(n) / 500.0)}),
        table_name=table,
        normalize_columns=False,
    )


# ----------------------------------------------------------------------
# SqliteRepo.series_row_count / downsampled_series_df
# ----------------------------------------------------------------------
def test_row_count_matches_the_query(repo: SqliteRepo) -> None:
    _import_points(repo, 1234)
    assert repo.series_row_count("SELECT x, y FROM big") == 1234


def test_a_query_under_threshold_is_returned_unchanged(repo: SqliteRepo) -> None:
    _import_points(repo, 500)
    out = repo.downsampled_series_df("SELECT x, y FROM big", threshold=1000)
    assert len(out) == 500


def test_a_query_over_threshold_is_decimated_to_roughly_it(repo: SqliteRepo) -> None:
    _import_points(repo, 250_000)
    out = repo.downsampled_series_df("SELECT x, y FROM big ORDER BY x", threshold=1000)

    assert len(out) == 1000
    assert list(out.columns) == ["x", "y"]
    assert out["x"].is_monotonic_increasing
    # The stride keeps the endpoints in range rather than clustering near
    # the start - a naive LIMIT 1000 would silently drop everything past
    # the first 1000 rows.
    assert out["x"].iloc[0] == 0
    assert out["x"].iloc[-1] > 240_000


def test_threshold_zero_or_negative_disables_downsampling(repo: SqliteRepo) -> None:
    _import_points(repo, 10_000)
    out = repo.downsampled_series_df("SELECT x, y FROM big", threshold=0)
    assert len(out) == 10_000


def test_row_count_cache_follows_data_changes(repo: SqliteRepo) -> None:
    _import_points(repo, 100)
    assert repo.series_row_count("SELECT x, y FROM big") == 100

    _import_points(repo, 101)  # re-imports the table with a different row count
    assert repo.series_row_count("SELECT x, y FROM big") == 101


# ----------------------------------------------------------------------
# The render pipeline: a figure's downsample_threshold option actually
# reaches the drawn artist.
# ----------------------------------------------------------------------
def _figure_with_big_series(
    repo: SqliteRepo, *, n: int, downsample_threshold: int | None = None
) -> int:
    """*downsample_threshold* None leaves the option unset; an int (0 included)
    writes it, so a test can exercise the unset default and an explicit 0."""
    _import_points(repo, n)
    figure_id = repo.create_figure_descriptor(
        name="F",
        nrows=1,
        ncols=1,
        options=(
            {OPT_DOWNSAMPLE_THRESHOLD: downsample_threshold}
            if downsample_threshold is not None
            else {}
        ),
    )
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Time Series",
        title="t", x_label="x", y_label="y", options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM big ORDER BY x", roles={"x": "x", "y": "y"},
        style={},
    )
    return int(figure_id)


def test_a_figure_with_no_threshold_uses_the_default(repo: SqliteRepo) -> None:
    figure_id = _figure_with_big_series(repo, n=5_000, downsample_threshold=None)
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )
    assert len(fig.axes[0].lines[0].get_xdata()) == DEFAULT_DOWNSAMPLE_THRESHOLD


def test_an_explicit_zero_threshold_draws_every_point(repo: SqliteRepo) -> None:
    figure_id = _figure_with_big_series(repo, n=5_000, downsample_threshold=0)
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )
    assert len(fig.axes[0].lines[0].get_xdata()) == 5_000


def test_a_configured_threshold_thins_the_drawn_line(repo: SqliteRepo) -> None:
    figure_id = _figure_with_big_series(repo, n=200_000, downsample_threshold=1_000)
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )
    assert len(fig.axes[0].lines[0].get_xdata()) == 1_000


def test_a_series_already_under_the_threshold_is_unaffected(repo: SqliteRepo) -> None:
    figure_id = _figure_with_big_series(repo, n=200, downsample_threshold=1_000)
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )
    assert len(fig.axes[0].lines[0].get_xdata()) == 200


# ----------------------------------------------------------------------
# FigurePropertiesWidget: the threshold combo saves and reloads
# ----------------------------------------------------------------------
def test_downsample_combo_defaults_to_the_default_threshold(qapp) -> None:
    from app.widgets.figure_properties import FigurePropertiesWidget

    widget = FigurePropertiesWidget()
    assert widget._downsample_combo.currentData() == DEFAULT_DOWNSAMPLE_THRESHOLD


def test_downsample_combo_round_trips_through_save_and_reload(
    qapp, repo: SqliteRepo
) -> None:
    from app.widgets.figure_properties import FigurePropertiesWidget

    _import_points(repo, 10)
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    widget = FigurePropertiesWidget()
    widget.set_connected_figure(repo, figure_id, Figure())

    index = widget._downsample_combo.findData(100_000)
    assert index >= 0
    widget._downsample_combo.setCurrentIndex(index)

    payload: dict = {}
    widget.figure_options_requested.connect(payload.update)
    widget._save_figure_options()

    assert payload[OPT_DOWNSAMPLE_THRESHOLD] == 100_000

    # Simulate persistence the way MainWindow's handler does, then reload.
    repo.set_figure_options(figure_id, payload)
    widget.clear_connected_figure()
    widget.set_connected_figure(repo, figure_id, Figure())

    assert widget._downsample_combo.currentData() == 100_000
