"""Tests for the multi-data-set histogram renderer.

The property that matters is that all data sets share one binning: that is what
makes the bars comparable, and it only holds if they go through a single
``ax.hist`` call.  Several tests below check it indirectly by comparing the bar
geometry across data sets.

Vertical is the default orientation, so the geometry assertions read bar x/width
unless a test explicitly asks for horizontal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base_axis import SeriesData
from app.charts.histogram_axis import HistogramAxisRenderer

RNG = np.random.default_rng(3)


def _render(series: list[SeriesData], options: dict | None = None):
    """Render the given series and return (figure, axis)."""
    fig = Figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    HistogramAxisRenderer().render_axis(
        ax=ax, series=series, options=options or {}
    )
    return fig, ax


def _grouped_frame(groups: int = 3, per_group: int = 400) -> pd.DataFrame:
    """One frame holding several data sets in a 'dataset' column."""
    rows = []
    for index in range(groups):
        values = RNG.normal(10.0 + 2.5 * index, 1.0 + 0.3 * index, per_group)
        rows.append(pd.DataFrame({"dataset": f"g{index}", "value": values}))
    return pd.concat(rows, ignore_index=True)


def test_dataset_column_produces_one_legend_entry_per_group() -> None:
    df = _grouped_frame(groups=3)
    _, ax = _render([SeriesData(name="s", df=df, style={})])

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["g0", "g1", "g2"]


def test_multiple_series_each_become_a_dataset() -> None:
    frames = [
        pd.DataFrame({"value": RNG.normal(loc, 1.0, 300)}) for loc in (5.0, 9.0)
    ]
    _, ax = _render(
        [
            SeriesData(name="left", df=frames[0], style={}),
            SeriesData(name="right", df=frames[1], style={}),
        ]
    )

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["left", "right"]


def test_default_orientation_is_vertical() -> None:
    """Vertical bars vary in height and sit at distinct x positions."""
    df = pd.DataFrame({"value": RNG.normal(0, 1, 500)})
    _, ax = _render([SeriesData(name="s", df=df, style={})], {"bins": 10})

    patches = [p for container in ax.containers for p in container.patches]
    widths = {round(float(p.get_width()), 6) for p in patches}
    heights = {round(float(p.get_height()), 6) for p in patches}

    assert len(widths) == 1, "vertical bars should all share one bin width"
    assert len(heights) > 1, "vertical bars should vary in height"


def test_bars_share_one_binning() -> None:
    df = _grouped_frame(groups=3, per_group=300)
    _, ax = _render([SeriesData(name="s", df=df, style={})], {"bins": 12})

    assert len(ax.containers) == 3, "expected one bar container per data set"

    # Vertical bars vary in height, not width; every bar has the same width
    # because the bins are shared across data sets.
    widths = {
        round(float(patch.get_width()), 6)
        for container in ax.containers
        for patch in container.patches
    }
    assert len(widths) == 1, f"bins are not shared across data sets: {widths}"

    x_positions = [
        sorted(round(float(patch.get_x()), 6) for patch in container.patches)
        for container in ax.containers
    ]
    assert x_positions[0] != x_positions[1], "grouped bars should be offset per data set"


def test_horizontal_orientation_flips_the_geometry() -> None:
    df = _grouped_frame(groups=3, per_group=300)
    _, ax = _render(
        [SeriesData(name="s", df=df, style={})],
        {"bins": 12, "orientation": "horizontal"},
    )

    heights = {
        round(float(patch.get_height()), 6)
        for container in ax.containers
        for patch in container.patches
    }
    assert len(heights) == 1, f"bins are not shared across data sets: {heights}"

    y_positions = [
        sorted(round(float(patch.get_y()), 6) for patch in container.patches)
        for container in ax.containers
    ]
    assert y_positions[0] != y_positions[1], "grouped bars should be offset per data set"


def test_unknown_orientation_falls_back_to_vertical() -> None:
    df = pd.DataFrame({"value": RNG.normal(0, 1, 200)})
    _, ax = _render(
        [SeriesData(name="s", df=df, style={})],
        {"bins": 8, "orientation": "sideways"},
    )
    assert ax.get_ylabel() == "Count"


def test_stacked_mode_shares_bar_positions() -> None:
    df = _grouped_frame(groups=2, per_group=300)
    _, ax = _render(
        [SeriesData(name="s", df=df, style={})],
        {"bins": 10, "stacked": True},
    )

    first, second = (
        sorted(round(float(patch.get_x()), 6) for patch in container.patches)
        for container in ax.containers
    )
    assert first == second, "stacked data sets must occupy the same bin positions"


def test_explicit_range_clips_the_axis() -> None:
    df = pd.DataFrame({"value": np.concatenate([RNG.normal(0, 1, 500), [500.0]])})
    _, ax = _render(
        [SeriesData(name="s", df=df, style={})],
        {"bins": 10, "range_min": -4.0, "range_max": 4.0},
    )

    edges = [float(patch.get_x()) for container in ax.containers for patch in container.patches]
    assert min(edges) == pytest.approx(-4.0, abs=1e-6)
    assert max(edges) < 4.0


def test_density_switches_the_count_axis_label() -> None:
    df = pd.DataFrame({"value": RNG.normal(0, 1, 400)})
    _, ax = _render([SeriesData(name="s", df=df, style={})], {"density": True})
    assert ax.get_ylabel() == "Density"

    _, ax_counts = _render([SeriesData(name="s", df=df, style={})])
    assert ax_counts.get_ylabel() == "Count"


def test_count_label_follows_the_orientation() -> None:
    """The count label belongs to whichever axis the bars grow along."""
    df = pd.DataFrame({"value": RNG.normal(0, 1, 200)})

    _, vertical = _render([SeriesData(name="s", df=df, style={})])
    assert vertical.get_ylabel() == "Count"
    assert vertical.get_xlabel() == ""

    _, horizontal = _render(
        [SeriesData(name="s", df=df, style={})], {"orientation": "horizontal"}
    )
    assert horizontal.get_xlabel() == "Count"
    assert horizontal.get_ylabel() == ""


def test_weights_are_honoured() -> None:
    df = pd.DataFrame({"value": [1.0] * 10, "weight": [3.0] * 10})
    _, ax = _render([SeriesData(name="s", df=df, style={})], {"bins": 1})

    heights = [float(patch.get_height()) for container in ax.containers for patch in container.patches]
    assert max(heights) == pytest.approx(30.0)


@pytest.mark.parametrize("histtype", ["bar", "barstacked", "step", "stepfilled"])
def test_every_histtype_draws(histtype: str, plots_dir: Path, show_plots: bool) -> None:
    df = _grouped_frame(groups=3, per_group=250)
    fig, ax = _render(
        [SeriesData(name="s", df=df, style={})],
        {"bins": 18, "histtype": histtype},
    )

    assert ax.containers or ax.patches or ax.lines, f"{histtype} drew nothing"

    if show_plots:
        target = plots_dir / "histogram" / f"{histtype}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=110, bbox_inches="tight")


def test_invisible_series_is_skipped() -> None:
    df = pd.DataFrame({"value": RNG.normal(0, 1, 100)})
    _, ax = _render([SeriesData(name="s", df=df, style={"visible": False})])
    assert not ax.containers and not ax.patches


def test_missing_value_column_is_reported_not_raised() -> None:
    df = pd.DataFrame({"wrong": [1.0, 2.0, 3.0]})
    _, ax = _render([SeriesData(name="s", df=df, style={})])
    assert not ax.containers
