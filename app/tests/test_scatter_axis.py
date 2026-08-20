"""Tests for the scatter renderer's legend-label handling.

Regression cover for a latent bug: the renderer popped "label" from its kwargs
without a default, which only ever worked because ``get_kwargs`` used to emit
every declared kwarg including the ones resolving to None.  Once None-valued
kwargs stopped being forwarded to Matplotlib, the unguarded pop raised
``KeyError: 'label'`` on any series that draws a line as well as markers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base_axis import SeriesData
from app.charts.scatter_axis import ScatterAxisRenderer

RNG = np.random.default_rng(5)


def _frame(n: int = 200) -> pd.DataFrame:
    """A small x/y frame."""
    return pd.DataFrame({"x": np.arange(n, dtype=float), "y": RNG.normal(0, 1, n)})


def _render(style: dict, options: dict | None = None):
    """Render one scatter series and return (figure, axis)."""
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ScatterAxisRenderer().render_axis(
        ax=ax,
        series=[SeriesData(name="points", df=_frame(), style=style)],
        options=options or {},
    )
    return fig, ax


@pytest.mark.parametrize(
    "style",
    [
        pytest.param({"marker": "o", "linestyle": "-"}, id="marker+line"),
        pytest.param({"linestyle": "--"}, id="line-only"),
        pytest.param({"marker": "o"}, id="marker-only"),
        pytest.param({"marker": "o", "show_in_legend": False}, id="no-legend"),
        pytest.param({}, id="no-style"),
    ],
)
def test_every_marker_line_combination_renders(style: dict) -> None:
    """None of these may raise, and all of them must draw something."""
    _, ax = _render(style)
    assert ax.collections or ax.lines, f"nothing drawn for style={style}"


def test_line_owns_the_legend_when_both_are_drawn() -> None:
    """With a line present the scatter must not add a second legend entry."""
    _, ax = _render({"marker": "o", "linestyle": "-", "label": "series A"})

    labels = [
        artist.get_label()
        for artist in list(ax.collections) + list(ax.lines)
        if isinstance(artist.get_label(), str)
        and not artist.get_label().startswith("_")
    ]
    assert labels == ["series A"], f"expected exactly one legend entry, got {labels}"


def test_marker_only_series_carries_the_label() -> None:
    _, ax = _render({"marker": "o", "label": "series B"})
    assert [c.get_label() for c in ax.collections] == ["series B"]


def test_explicit_label_overrides_the_series_name(plots_dir: Path, show_plots: bool) -> None:
    fig, ax = _render({"marker": "o", "label": "custom"})
    assert "points" not in [c.get_label() for c in ax.collections]

    if show_plots:
        target = plots_dir / "scatter" / "labelled.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=110, bbox_inches="tight")
