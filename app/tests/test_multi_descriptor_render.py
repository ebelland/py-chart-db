"""Render several figures from one database and save the results."""
from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from app.data.sqlite_repo import SqliteRepo
from app.charts.render_figure import render_figure_from_descriptor
from app.tests._large_db_factory import create_large_db


def _save(fig: Figure, path: Path) -> None:
    """Save figure to disk (always creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def test_render_multi_descriptor_multiple_figures(
    tmp_db_path: Path,
    plots_dir: Path,
    show_plots: bool,
) -> None:
    create_large_db(tmp_db_path, n_ts=30_000, n_scatter=20_000, seed=1234)
    repo = SqliteRepo(db_path=tmp_db_path)

    descriptors = [
        repo.load_figure_descriptor( figure_id=1),
        repo.load_figure_descriptor( figure_id=2),
    ]

    for idx, descriptor in enumerate(descriptors, start=1):
        fig = Figure()
        if descriptor is None:
            continue
        render_figure_from_descriptor(figure=fig, repo=repo, descriptor=descriptor)
        assert len(fig.axes) > 0
        if show_plots:
            _save(fig, plots_dir / f"multi_{idx}.png")

