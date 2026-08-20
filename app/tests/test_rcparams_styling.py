"""A figure's stored mplstyle must reach the rendered output - and stay there.

Two properties are checked: the style is actually applied while rendering, and
it is fully unwound afterwards (the style is applied through a context manager,
so a leak would silently restyle every other chart in the session).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pytest
from matplotlib.figure import Figure

from app.charts.render_figure import render_figure_from_descriptor
from app.data.sqlite_repo import SqliteRepo
from app.tests._figure_factory import create_renderer_showcase_db

STYLE_TEXT = """
figure.facecolor: 0.85
axes.facecolor: 0.95
axes.grid: True
axes.edgecolor: FF00FF
axes.linewidth: 2.5
lines.linewidth: 3.0
font.size: 14
"""

# Same style, plus entries that cannot work without a TeX installation.
STYLE_TEXT_WITH_LATEX = STYLE_TEXT + "text.usetex: True\ntext.latex.preamble: \\usepackage{amsmath}\n"


@pytest.fixture
def styled_repo(tmp_path: Path) -> tuple[SqliteRepo, dict[str, int]]:
    """A showcase database whose figures carry the style above."""
    db_path = tmp_path / "styled.dhub"
    figure_ids = create_renderer_showcase_db(
        db_path,
        n_points=1_500,
        figure_options={"mpl_style": STYLE_TEXT},
    )
    return SqliteRepo(db_path=db_path), figure_ids


def test_style_reaches_the_rendered_figure(
    styled_repo: tuple[SqliteRepo, dict[str, int]],
    plots_dir: Path,
    show_plots: bool,
) -> None:
    repo, figure_ids = styled_repo
    descriptor = repo.load_figure_descriptor(figure_id=figure_ids["Time Series"])
    assert descriptor is not None

    fig = Figure(figsize=(7.0, 4.5))
    render_figure_from_descriptor(figure=fig, descriptor=descriptor, repo=repo)

    axis = fig.axes[0]
    assert axis.patch.get_facecolor()[:3] == pytest.approx((0.95, 0.95, 0.95), abs=1e-3)
    assert fig.get_facecolor()[:3] == pytest.approx((0.85, 0.85, 0.85), abs=1e-3)
    assert axis.spines["bottom"].get_linewidth() == pytest.approx(2.5)
    assert axis.spines["bottom"].get_edgecolor()[:3] == pytest.approx((1.0, 0.0, 1.0), abs=1e-3)
    assert axis.xaxis.label.get_fontsize() == pytest.approx(14.0)

    # Documented precedence: a renderer kwarg with an explicit default wins over
    # the style file.  lines.linewidth is 3.0 in the style, but the time-series
    # renderer declares linewidth=1.6, so 1.6 is what gets drawn.
    assert axis.lines and axis.lines[0].get_linewidth() == pytest.approx(1.6)

    if show_plots:
        target = plots_dir / "rcparams" / "styled_time_series.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=110, bbox_inches="tight")


def test_style_does_not_leak_into_global_rcparams(
    styled_repo: tuple[SqliteRepo, dict[str, int]],
) -> None:
    repo, figure_ids = styled_repo
    before = dict(mpl.rcParams)

    descriptor = repo.load_figure_descriptor(figure_id=figure_ids["Scatter Plot"])
    assert descriptor is not None
    render_figure_from_descriptor(figure=Figure(), descriptor=descriptor, repo=repo)

    changed = {
        key
        for key in before
        if str(before[key]) != str(mpl.rcParams[key])
    }
    assert not changed, f"figure style leaked into global rcParams: {sorted(changed)}"


def test_latex_entries_do_not_break_rendering(tmp_path: Path, plots_dir: Path, show_plots: bool) -> None:
    """A style written on a LaTeX machine must still render everywhere else."""
    db_path = tmp_path / "latex_styled.dhub"
    figure_ids = create_renderer_showcase_db(
        db_path,
        n_points=800,
        figure_options={"mpl_style": STYLE_TEXT_WITH_LATEX},
    )
    repo = SqliteRepo(db_path=db_path)

    descriptor = repo.load_figure_descriptor(figure_id=figure_ids["Bar Chart"])
    assert descriptor is not None

    fig = Figure(figsize=(7.0, 4.5))
    render_figure_from_descriptor(figure=fig, descriptor=descriptor, repo=repo)

    assert fig.axes and fig.axes[0].patches, "figure with LaTeX style entries did not draw"

    if show_plots:
        target = plots_dir / "rcparams" / "latex_style_bar.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=110, bbox_inches="tight")
