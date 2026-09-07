"""Switching a figure's layout mode, on the one Matplotlib Figure a chart
tab keeps for its whole life.

A figure is re-rendered into the *same* Figure object every time - a fresh
one is never created for a redraw. Matplotlib remembers, on that object,
that a "constrained"/"compressed"/"tight" engine was ever active, and once
it has, ``subplots_adjust`` silently refuses to move anything on it again -
unless the engine is cleared with Python ``None`` rather than the string
"none", which merely swaps in a placeholder that carries the same
restriction forward. A user picking "Manual" after "Constrained" (the
default every new figure starts on) is exactly this sequence.
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


def _figure_with_axis(repo: SqliteRepo, *, layout_mode: str, margins: dict | None = None):
    options: dict[str, object] = {"layout_mode": layout_mode}
    if margins is not None:
        options["margins"] = margins
    figure_id = repo.create_figure_descriptor(name="F", nrows=1, ncols=1, options=options)
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
        title="t", x_label="x", y_label="y", options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM t", roles={"x": "x", "y": "y"}, style={},
    )
    return int(figure_id)


def test_manual_margins_apply_on_a_figure_never_touched_before(repo: SqliteRepo) -> None:
    figure_id = _figure_with_axis(
        repo, layout_mode="none",
        margins={"left": 0.33, "right": 0.77, "bottom": 0.06, "top": 0.97,
                 "wspace": 0.08, "hspace": 0.08},
    )
    fig = Figure()

    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )

    assert fig.get_layout_engine() is None
    assert fig.subplotpars.left == pytest.approx(0.33)
    assert fig.subplotpars.right == pytest.approx(0.77)


def test_manual_margins_still_apply_after_a_prior_constrained_render(
    repo: SqliteRepo,
) -> None:
    """The actual bug: switching Constrained -> Manual on the *same* Figure
    object, which is what every redraw of an existing chart tab does."""
    figure_id = _figure_with_axis(repo, layout_mode="constrained")
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )
    assert fig.get_layout_engine() is not None

    repo.set_figure_options(
        figure_id,
        {
            "layout_mode": "none",
            "margins": {"left": 0.33, "right": 0.77, "bottom": 0.06, "top": 0.97,
                        "wspace": 0.08, "hspace": 0.08},
        },
    )
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )

    assert fig.get_layout_engine() is None
    assert fig.subplotpars.left == pytest.approx(0.33)
    assert fig.subplotpars.right == pytest.approx(0.77)


def test_switching_back_to_constrained_still_works(repo: SqliteRepo) -> None:
    figure_id = _figure_with_axis(repo, layout_mode="none", margins={"left": 0.33})
    fig = Figure()
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )
    assert fig.get_layout_engine() is None

    repo.set_figure_options(figure_id, {"layout_mode": "constrained"})
    render_figure_from_descriptor(
        figure=fig, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )

    assert fig.get_layout_engine() is not None
