"""Render one figure per axis renderer and save the result as a PNG.

The saved images are the point of these tests as much as the assertions: an
assertion tells you *that* a renderer broke, the PNG in the plots directory
tells you *how*.  Files land in ``<artifacts>/plots/renderers/``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib.figure import Figure

from app.scanners.axis_renderer_scanner import get_renderer, renderers
from app.charts.render_figure import render_figure_from_descriptor
from app.data.sqlite_repo import SqliteRepo
from app.tests._figure_factory import SHOWCASE_CHART_TYPES, create_renderer_showcase_db


@pytest.fixture(scope="module")
def showcase(tmp_path_factory: pytest.TempPathFactory) -> tuple[SqliteRepo, dict[str, int]]:
    """Build the showcase database once for the whole module."""
    db_path = tmp_path_factory.mktemp("showcase") / "renderers.dhub"
    figure_ids = create_renderer_showcase_db(db_path, n_points=4_000)
    return SqliteRepo(db_path=db_path), figure_ids


def _save(fig: Figure, path: Path) -> None:
    """Save a figure, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")


def test_every_discovered_renderer_is_covered() -> None:
    """A new renderer must come with a showcase figure, or this fails."""
    discovered = {entry["value"] for entry in renderers}
    assert discovered == set(SHOWCASE_CHART_TYPES), (
        "renderers and showcase figures have drifted apart: "
        f"only in code={sorted(discovered - set(SHOWCASE_CHART_TYPES))}, "
        f"only in tests={sorted(set(SHOWCASE_CHART_TYPES) - discovered)}"
    )


@pytest.mark.parametrize("chart_type", SHOWCASE_CHART_TYPES)
def test_renderer_draws_and_saves(
    chart_type: str,
    showcase: tuple[SqliteRepo, dict[str, int]],
    plots_dir: Path,
    show_plots: bool,
) -> None:
    repo, figure_ids = showcase
    assert get_renderer(chart_type) is not None, f"no renderer named {chart_type!r}"

    descriptor = repo.load_figure_descriptor(figure_id=figure_ids[chart_type])
    assert descriptor is not None

    fig = Figure(figsize=(7.0, 4.5))
    render_figure_from_descriptor(figure=fig, descriptor=descriptor, repo=repo)

    # One plot area, not one Axes.  A renderer may add a twin - the Pareto
    # chart puts its cumulative percentage on one, because the bars are in the
    # data's units and the line is in percent - and a twin shares its parent's
    # position exactly.  What this guards against is a renderer scattering
    # extra subplots across the figure, which is still caught.
    positions = {tuple(round(value, 6) for value in axes.get_position().bounds)
                 for axes in fig.axes}
    assert len(positions) == 1, (
        f"{chart_type}: expected one plot area, got {len(positions)} "
        f"across {len(fig.axes)} axes"
    )
    axis = fig.axes[0]

    # "Something was actually drawn" - an empty axis is the failure mode that
    # slips through when only exceptions are checked.
    drawn = (
        len(axis.lines)
        + len(axis.collections)
        + len(axis.patches)
        + len(axis.containers)
        # Axes.table() tracks its own artist in ax.tables - none of the four
        # collections above, so the Table renderer needs this one counted
        # too or it always reads as an empty axis.
        + len(axis.tables)
    )
    assert drawn > 0, f"{chart_type}: renderer produced an empty axis"

    if show_plots:
        slug = chart_type.lower().replace(" ", "_")
        _save(fig, plots_dir / "renderers" / f"{slug}.png")
