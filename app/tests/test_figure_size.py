"""The width and height typed into the figure properties.

They were ignored. In FIXED mode the panel sizes the figure from a baseline
it captured from the last render - and what it renders in FIXED mode *is* that
baseline, so the value fed on its own output: whatever the first render
produced became permanent. The configured size was written to rcParams,
applied to the figure, and then overwritten by the stale baseline before the
descriptor was drawn.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.utils.figure_metrics import CM_PER_INCH


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    repo = SqliteRepo(db_path=tmp_db_path)
    repo.import_dataframe(
        pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 4.0]}),
        table_name="points",
        normalize_columns=False,
    )
    yield repo
    repo.close()


def _figure_with_series(repo: SqliteRepo) -> int:
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id,
            axis_index=0,
            chart_type="Scatter Plot",
            title="",
            x_label="",
            y_label="",
            options={"grid": True},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="s",
        sql_query='SELECT x, y FROM "points"',
        roles={"x": "x", "y": "y"},
        style={"marker": "o"},
    )
    return figure_id


@pytest.fixture
def panel(qapp, repo: SqliteRepo):
    from app.widgets.chart_panel import ChartPanel

    figure_id = _figure_with_series(repo)
    built = ChartPanel(repo=repo, figure_id=figure_id, parent=None)
    built.set_resize_mode("FIXED", persist=False, redraw=False)
    return built


def _size_cm(panel) -> tuple[float, float]:
    width_in, height_in = panel._figure.get_size_inches()
    return round(float(width_in) * CM_PER_INCH, 2), round(float(height_in) * CM_PER_INCH, 2)


def _set_metrics(repo: SqliteRepo, figure_id: int, width_cm: float, height_cm: float) -> None:
    options = dict(repo.load_figure_descriptor(figure_id).options or {})
    options.update(
        {
            "figure_width_cm": width_cm,
            "figure_height_cm": height_cm,
            "figure_dpi": 100.0,
        }
    )
    repo.set_figure_options(figure_id, options)


def test_the_configured_size_reaches_the_figure(panel, repo: SqliteRepo) -> None:
    _set_metrics(repo, panel._figure_id, 25.0, 10.0)

    panel.reload()

    assert _size_cm(panel) == (25.0, 10.0)


def test_changing_it_again_takes_effect(panel, repo: SqliteRepo) -> None:
    """The bug itself: the first render's size became permanent."""
    _set_metrics(repo, panel._figure_id, 25.0, 10.0)
    panel.reload()

    _set_metrics(repo, panel._figure_id, 8.5, 6.0)
    panel.reload()

    assert _size_cm(panel) == (8.5, 6.0)


def test_the_baseline_follows_the_figure_not_the_last_render(panel, repo: SqliteRepo) -> None:
    """FIXED mode draws the baseline, so the baseline has to be the figure's
    own size rather than a memory of what was drawn before."""
    _set_metrics(repo, panel._figure_id, 12.0, 9.0)
    panel.reload()

    width_in, height_in = panel._fixed_figure_size_inches
    assert round(width_in * CM_PER_INCH, 2) == 12.0
    assert round(height_in * CM_PER_INCH, 2) == 9.0
    assert panel._fixed_figure_dpi == 100.0


def test_a_figure_with_no_metrics_is_left_alone(panel) -> None:
    """Figures saved before the metrics were per-figure keep rendering as they
    did, which is why "no metrics" is not the same as "zero"."""
    before = _size_cm(panel)

    panel.reload()

    assert _size_cm(panel) == before


def test_half_a_set_of_metrics_is_treated_as_none(panel, repo: SqliteRepo) -> None:
    """A width without its height would distort the figure rather than
    resize it."""
    options = dict(repo.load_figure_descriptor(panel._figure_id).options or {})
    options["figure_width_cm"] = 25.0
    repo.set_figure_options(panel._figure_id, options)
    before = _size_cm(panel)

    panel.reload()

    assert _size_cm(panel) == before
