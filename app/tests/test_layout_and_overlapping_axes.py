"""Integration tests for the layout presets and the ``twin_of`` axis option
they can write, against the real SqliteRepo and render_figure.py - no Qt
needed, a bare Figure renders the same way ChartPanel's does.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts import layout_presets as lp
from app.charts.render_figure import render_figure_from_descriptor
from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    df = pd.DataFrame({"t": [1, 2, 3, 4], "temp": [10, 20, 15, 18], "pressure": [1, 3, 2, 4]})
    built.import_dataframe(df, table_name="readings", normalize_columns=False)
    yield built
    built.close()


def _add_axis(repo: SqliteRepo, figure_id: int, axis_index: int, y_col: str) -> int:
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=axis_index,
        chart_type="Scatter Plot",
        title="",
        x_label="",
        y_label="",
        options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name=y_col,
        sql_query=f'SELECT "t" AS x, "{y_col}" AS y FROM "readings"',
        roles={"x": "x", "y": "y"},
        style={},
    )
    return axis_id


def _render(repo: SqliteRepo, figure_id: int) -> Figure:
    descriptor = repo.load_figure_descriptor(figure_id)
    assert descriptor is not None
    fig = Figure()
    render_figure_from_descriptor(figure=fig, descriptor=descriptor, repo=repo)
    return fig


def _apply_preset(repo: SqliteRepo, figure_id: int, preset: str) -> None:
    axis_ids = [axis_id for axis_id, _index, _title in repo.list_axes_for_figure(figure_id)]
    plan = lp.plan_layout(preset, axis_ids)
    repo.apply_axis_layout(
        figure_id=figure_id,
        nrows=plan.nrows,
        ncols=plan.ncols,
        placements=[(p.axis_id, p.axis_index, p.options) for p in plan.axes],
    )


# ----------------------------------------------------------------------
# twin_of, rendered directly (no preset involved)
# ----------------------------------------------------------------------
def test_a_twin_axis_shares_its_targets_rect_and_x_but_not_its_y(repo: SqliteRepo) -> None:
    figure_id = repo.create_figure_descriptor(name="twin")
    target_id = _add_axis(repo, figure_id, 0, "temp")
    twin_id = _add_axis(repo, figure_id, 1, "pressure")
    repo.set_axis_options(twin_id, {"twin_of": target_id})

    fig = _render(repo, figure_id)

    assert len(fig.axes) == 2
    target_ax, twin_ax = fig.axes
    assert target_ax.get_position().bounds == twin_ax.get_position().bounds
    assert twin_ax.get_shared_x_axes().joined(target_ax, twin_ax)
    assert not twin_ax.get_shared_y_axes().joined(target_ax, twin_ax)


def test_a_twin_of_a_missing_axis_id_falls_back_to_the_grid(repo: SqliteRepo) -> None:
    """A stale twin_of - the target axis was deleted - must not vanish the
    axis it was set on: it renders as an ordinary grid axis instead."""
    figure_id = repo.create_figure_descriptor(name="stale twin")
    axis_id = _add_axis(repo, figure_id, 0, "temp")
    repo.set_axis_options(axis_id, {"twin_of": 999_999})

    fig = _render(repo, figure_id)

    assert len(fig.axes) == 1


def test_a_twin_of_a_twin_falls_back_to_the_grid_instead_of_chaining(repo: SqliteRepo) -> None:
    """render_figure only pairs an axis with a base axis - a chain (twin of a
    twin) is not something ax.twinx() can express, so the second twin renders
    as an ordinary grid axis rather than being silently dropped."""
    figure_id = repo.create_figure_descriptor(name="chain")
    a_id = _add_axis(repo, figure_id, 0, "temp")
    b_id = _add_axis(repo, figure_id, 1, "pressure")
    c_id = _add_axis(repo, figure_id, 2, "temp")
    repo.set_axis_options(b_id, {"twin_of": a_id})
    repo.set_axis_options(c_id, {"twin_of": b_id})

    fig = _render(repo, figure_id)

    # a+b form one overlapping pair (2 axes at one position); c falls back to
    # its own grid cell - three Axes objects in total.
    assert len(fig.axes) == 3


# ----------------------------------------------------------------------
# apply_axis_layout + each preset, end to end
# ----------------------------------------------------------------------
def test_shared_grid_preset_renders_with_shared_scales(repo: SqliteRepo) -> None:
    figure_id = repo.create_figure_descriptor(name="shared")
    _add_axis(repo, figure_id, 0, "temp")
    _add_axis(repo, figure_id, 1, "pressure")

    _apply_preset(repo, figure_id, lp.SHARED_GRID)
    fig = _render(repo, figure_id)

    assert len(fig.axes) == 2
    first, second = fig.axes
    assert second.get_shared_x_axes().joined(first, second)
    assert second.get_shared_y_axes().joined(first, second)


def test_main_and_secondary_preset_gives_the_first_axis_more_width(repo: SqliteRepo) -> None:
    figure_id = repo.create_figure_descriptor(name="main+secondary")
    _add_axis(repo, figure_id, 0, "temp")
    _add_axis(repo, figure_id, 1, "pressure")
    _add_axis(repo, figure_id, 2, "temp")

    _apply_preset(repo, figure_id, lp.MAIN_AND_SECONDARY)
    fig = _render(repo, figure_id)

    assert len(fig.axes) == 3
    main_ax = fig.axes[0]
    secondary_widths = [ax.get_position().width for ax in fig.axes[1:]]
    assert main_ax.get_position().width > max(secondary_widths)


def test_overlapping_preset_renders_a_dual_scale_pair(repo: SqliteRepo) -> None:
    figure_id = repo.create_figure_descriptor(name="overlap preset")
    _add_axis(repo, figure_id, 0, "temp")
    _add_axis(repo, figure_id, 1, "pressure")

    _apply_preset(repo, figure_id, lp.OVERLAPPING)
    fig = _render(repo, figure_id)

    assert len(fig.axes) == 2
    first, second = fig.axes
    assert first.get_position().bounds == second.get_position().bounds


def test_grid_preset_clears_a_previously_applied_preset(repo: SqliteRepo) -> None:
    """Re-picking Grid after Overlapping must not leave a stale twin_of
    behind - each preset owns and clears the same keys."""
    figure_id = repo.create_figure_descriptor(name="reset")
    _add_axis(repo, figure_id, 0, "temp")
    _add_axis(repo, figure_id, 1, "pressure")

    _apply_preset(repo, figure_id, lp.OVERLAPPING)
    _apply_preset(repo, figure_id, lp.GRID)

    fig = _render(repo, figure_id)
    assert len(fig.axes) == 2
    first, second = fig.axes
    assert first.get_position().bounds != second.get_position().bounds


def test_apply_axis_layout_persists_the_new_grid_size(repo: SqliteRepo) -> None:
    figure_id = repo.create_figure_descriptor(name="grid size")
    for index in range(5):
        _add_axis(repo, figure_id, index, "temp")

    _apply_preset(repo, figure_id, lp.GRID)

    descriptor = repo.load_figure_descriptor(figure_id)
    assert descriptor is not None
    assert (descriptor.nrows, descriptor.ncols) == (2, 3)
