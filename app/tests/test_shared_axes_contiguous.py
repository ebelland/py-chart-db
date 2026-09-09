"""A shared-scale grid should read as one instrument split into panels.

Three things the earlier label_outer() fix did not cover, each reported
after the previous one shipped:

* The panels still had Matplotlib's default gap between them - label_outer()
  only hides tick numbers, it says nothing about spacing, so two subplots
  sharing an axis stopped repeating labels but never actually touched.
* Each panel kept its own copy of the axis label *text* ("body mass (g)",
  "Count") - a separate artist from the tick numbers label_outer() strips,
  so the interior panels still repeated it even once the numbers were gone.
* Each panel also kept its own title, repeated down every row - unlike
  Matplotlib's own shared-axis gallery example, which has no per-panel
  title at all, one suptitle only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.render_figure import render_figure_from_descriptor
from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    built = SqliteRepo(db_path=tmp_db_path)
    built.import_dataframe(
        pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]}),
        table_name="t",
        normalize_columns=False,
    )
    yield built
    built.close()


def _shared_2x2_figure(repo: SqliteRepo, *, figure_options: dict | None = None) -> int:
    """A 2x2 grid where every axis shares x and y with the first one."""
    figure_id = repo.create_figure_descriptor(
        name="F", nrows=2, ncols=2, options=figure_options or {}
    )
    for index in range(4):
        axis_id = repo.create_axis_descriptor(
            figure_id=figure_id,
            axis_index=index,
            chart_type="Scatter Plot",
            title=f"ax{index}",
            x_label="X label",
            y_label="Y label",
            options={} if index == 0 else {"sharex": True, "sharey": True},
        )
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=0, name="s",
            sql_query="SELECT x, y FROM t", roles={"x": "x", "y": "y"}, style={},
        )
    return int(figure_id)


def test_shared_axes_close_the_grid_spacing(repo: SqliteRepo) -> None:
    figure_id = _shared_2x2_figure(repo, figure_options={"layout_mode": "constrained"})
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )

    gridspec = fig.axes[0].get_gridspec()
    assert gridspec.wspace == 0.0
    assert gridspec.hspace == 0.0


def test_sharing_between_non_adjacent_axes_does_not_close_unrelated_gaps(
    repo: SqliteRepo,
) -> None:
    """Two axes may share a scale without sitting next to each other - a
    row/col-spanning "Main + secondary" layout is exactly this. Gridspec has
    one gap for the whole grid, so closing it for a distant pair would also
    close it between two axes that share nothing at all.
    """
    figure_id = repo.create_figure_descriptor(
        name="F", nrows=2, ncols=2, options={"layout_mode": "constrained"}
    )
    # axis 0 (top-left) and axis 3 (bottom-right) share - diagonal, not
    # adjacent in either direction. axes 1 and 2 share nothing with anyone.
    specs = [
        (0, {}),
        (1, {}),
        (2, {}),
        (3, {"sharex": True, "sharey": True}),
    ]
    for index, options in specs:
        axis_id = repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=index, chart_type="Scatter Plot",
            title=f"ax{index}", x_label="X", y_label="Y", options=options,
        )
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=0, name="s",
            sql_query="SELECT x, y FROM t", roles={"x": "x", "y": "y"}, style={},
        )
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )

    gridspec = fig.axes[0].get_gridspec()
    assert gridspec.wspace != 0.0
    assert gridspec.hspace != 0.0


def test_a_figure_with_no_shared_axes_keeps_the_default_spacing(
    repo: SqliteRepo,
) -> None:
    """The zeroing is conditional on an actual sharex/sharey pairing - an
    ordinary grid of independent axes must render exactly as before."""
    figure_id = repo.create_figure_descriptor(
        name="F", nrows=1, ncols=2, options={"layout_mode": "constrained"}
    )
    for index in range(2):
        axis_id = repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=index, chart_type="Scatter Plot",
            title=f"ax{index}", x_label="X", y_label="Y", options={},
        )
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=0, name="s",
            sql_query="SELECT x, y FROM t", roles={"x": "x", "y": "y"}, style={},
        )
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )

    gridspec = fig.axes[0].get_gridspec()
    assert gridspec.wspace != 0.0


def test_only_the_edge_axis_keeps_its_axis_label_text(repo: SqliteRepo) -> None:
    figure_id = _shared_2x2_figure(repo, figure_options={"layout_mode": "constrained"})
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )

    # 2x2: axis 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right.
    top_left, top_right, bottom_left, bottom_right = fig.axes

    assert top_left.get_xlabel() == ""       # not last row
    assert top_left.get_ylabel() == "Y label"  # first col

    assert top_right.get_xlabel() == ""      # not last row
    assert top_right.get_ylabel() == ""      # not first col

    assert bottom_left.get_xlabel() == "X label"  # last row
    assert bottom_left.get_ylabel() == "Y label"  # first col

    assert bottom_right.get_xlabel() == "X label"  # last row
    assert bottom_right.get_ylabel() == ""         # not first col


def test_only_the_first_row_keeps_its_axis_title(repo: SqliteRepo) -> None:
    """Matplotlib's own shared-axis gallery example has no per-panel title
    at all, one suptitle only - a title sits at the top of a panel, so
    (unlike xlabel/ylabel, which follow the "outer edge") it is kept only
    on the first row of each column."""
    figure_id = _shared_2x2_figure(repo, figure_options={"layout_mode": "constrained"})
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )

    top_left, top_right, bottom_left, bottom_right = fig.axes

    assert top_left.get_title() == "ax0"      # first row
    assert top_right.get_title() == "ax1"     # first row
    assert bottom_left.get_title() == ""      # not first row
    assert bottom_right.get_title() == ""     # not first row


def test_tick_numbers_stay_hidden_on_the_interior_edges(repo: SqliteRepo) -> None:
    """Regression guard for the original label_outer() fix, unaffected by the
    two behaviours added on top of it here."""
    figure_id = _shared_2x2_figure(repo, figure_options={"layout_mode": "constrained"})
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )
    top_left, top_right, bottom_left, bottom_right = fig.axes

    assert not any(t.get_visible() for t in top_left.get_xticklabels())
    assert not any(t.get_visible() for t in top_right.get_xticklabels())
    assert not any(t.get_visible() for t in top_right.get_yticklabels())
    assert any(t.get_visible() for t in bottom_left.get_xticklabels())
    assert any(t.get_visible() for t in bottom_left.get_yticklabels())


def test_manual_layout_margins_still_override_the_automatic_zero(
    repo: SqliteRepo,
) -> None:
    """A figure the user has switched to Manual spacing keeps full control -
    the automatic zero is only ever a default for the automatic engines."""
    figure_id = _shared_2x2_figure(
        repo,
        figure_options={
            "layout_mode": "none",
            "margins": {
                "left": 0.1, "right": 0.95, "bottom": 0.1, "top": 0.9,
                "wspace": 0.3, "hspace": 0.25,
            },
        },
    )
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )

    assert fig.subplotpars.wspace == pytest.approx(0.3)
    assert fig.subplotpars.hspace == pytest.approx(0.25)
