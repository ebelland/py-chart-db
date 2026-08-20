"""Tests for symmetric and asymmetric error bars.

Matplotlib's ``xerr``/``yerr`` are *distances from the point*, not absolute
positions, and the asymmetric form is a ``(2, N)`` array of ``[lower, upper]``.
Getting that shape wrong produces a plot that looks plausible and is wrong, so
the shape is asserted directly rather than only through the drawn artists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base_axis import SeriesData
from app.charts.scatter_axis import ScatterAxisRenderer
from app.charts.time_series_axis import TimeSeriesAxisRenderer

RENDERERS = {
    "scatter": (ScatterAxisRenderer, {"marker": "o"}),
    "time_series": (TimeSeriesAxisRenderer, {"linestyle": "-"}),
}


def _frame(**extra: object) -> pd.DataFrame:
    base = {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 3.0, 5.0]}
    base.update(extra)
    return pd.DataFrame(base)


def _render(renderer_key: str, df: pd.DataFrame, options: dict | None = None):
    renderer_class, style = RENDERERS[renderer_key]
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    renderer_class().render_axis(
        ax=ax,
        series=[SeriesData(name="s", df=df, style=dict(style))],
        options=options or {},
    )
    return fig, ax


def _error_containers(ax) -> list:
    """Return the ErrorbarContainers on an axis."""
    return [c for c in ax.containers if hasattr(c, "has_xerr")]


# ----------------------------------------------------------------------
# Shape of the error argument
# ----------------------------------------------------------------------
def test_symmetric_error_is_one_value_per_point() -> None:
    renderer = ScatterAxisRenderer()
    df = _frame(yerr=[0.1, 0.2, 0.3, 0.4])

    values = renderer.error_values(df, "y")

    assert values is not None
    assert np.asarray(values).shape == (4,)
    assert np.allclose(values, [0.1, 0.2, 0.3, 0.4])


def test_asymmetric_error_is_two_rows_of_distances() -> None:
    renderer = ScatterAxisRenderer()
    df = _frame(yerr_low=[0.1, 0.1, 0.1, 0.1], yerr_high=[0.5, 0.5, 0.5, 0.5])

    values = renderer.error_values(df, "y")

    assert np.asarray(values).shape == (2, 4)
    assert np.allclose(values[0], 0.1)
    assert np.allclose(values[1], 0.5)


def test_asymmetric_pair_wins_over_the_symmetric_column() -> None:
    """Supplying both can only mean the pair is the more specific intent."""
    renderer = ScatterAxisRenderer()
    df = _frame(
        yerr=[9.0, 9.0, 9.0, 9.0],
        yerr_low=[0.1, 0.1, 0.1, 0.1],
        yerr_high=[0.2, 0.2, 0.2, 0.2],
    )

    values = renderer.error_values(df, "y")

    assert np.asarray(values).shape == (2, 4)
    assert not np.any(np.isclose(values, 9.0))


def test_one_sided_pair_fills_the_other_side_with_zero() -> None:
    renderer = ScatterAxisRenderer()
    df = _frame(yerr_high=[0.5, 0.5, 0.5, 0.5])

    values = renderer.error_values(df, "y")

    assert np.allclose(values[0], 0.0)
    assert np.allclose(values[1], 0.5)


def test_negative_and_missing_errors_are_clipped_to_zero() -> None:
    """A negative half-width is a data error, not a shorter bar."""
    renderer = ScatterAxisRenderer()
    df = _frame(yerr=[-1.0, np.nan, 0.3, 0.4])

    values = renderer.error_values(df, "y")

    assert np.all(np.asarray(values) >= 0.0)
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(0.0)


def test_no_error_columns_returns_none() -> None:
    renderer = ScatterAxisRenderer()
    assert renderer.error_values(_frame(), "y") is None
    assert renderer.error_values(_frame(), "x") is None
    assert renderer.has_error_roles(_frame()) is False


def test_mask_is_applied_to_the_error_columns() -> None:
    """Errors must stay aligned with the points that survived filtering."""
    renderer = ScatterAxisRenderer()
    df = _frame(yerr=[0.1, 0.2, 0.3, 0.4])
    mask = pd.Series([True, False, True, False])

    values = renderer.error_values(df, "y", mask)

    assert np.allclose(values, [0.1, 0.3])


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
@pytest.mark.parametrize("renderer_key", list(RENDERERS))
def test_error_bars_are_drawn_on_both_renderers(renderer_key: str) -> None:
    _, ax = _render(renderer_key, _frame(yerr=[0.2, 0.2, 0.2, 0.2]))
    assert _error_containers(ax), f"{renderer_key} drew no error bars"


@pytest.mark.parametrize("renderer_key", list(RENDERERS))
def test_x_and_y_errors_can_be_combined(renderer_key: str) -> None:
    _, ax = _render(
        renderer_key,
        _frame(xerr=[0.1] * 4, yerr_low=[0.2] * 4, yerr_high=[0.4] * 4),
    )

    containers = _error_containers(ax)
    assert containers
    assert containers[0].has_xerr
    assert containers[0].has_yerr


@pytest.mark.parametrize("renderer_key", list(RENDERERS))
def test_no_error_columns_means_no_error_artists(renderer_key: str) -> None:
    _, ax = _render(renderer_key, _frame())
    assert not _error_containers(ax)


def test_error_bars_stay_out_of_the_legend() -> None:
    """One legend entry per series, not two."""
    _, ax = _render("scatter", _frame(yerr=[0.2] * 4))

    labels = [
        artist.get_label()
        for artist in list(ax.collections) + list(ax.lines)
        if isinstance(artist.get_label(), str)
        and not artist.get_label().startswith("_")
    ]
    assert len(labels) == 1


def test_sorting_reorders_the_errors_with_the_points() -> None:
    """sort_x must not leave each bar attached to a different point."""
    df = pd.DataFrame(
        {"x": [4.0, 1.0, 3.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0], "yerr": [0.4, 0.1, 0.3, 0.2]}
    )
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ScatterAxisRenderer().render_axis(
        ax=ax,
        series=[SeriesData(name="s", df=df, style={"marker": "o", "sort_x": True})],
        options={},
    )

    container = _error_containers(ax)[0]
    segments = container.lines[2][0].get_segments()
    # After sorting by x, the half-widths must read 0.1, 0.2, 0.3, 0.4.
    half_widths = [abs(segment[1][1] - segment[0][1]) / 2.0 for segment in segments]
    assert half_widths == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_errorevery_thins_dense_bars(plots_dir: Path, show_plots: bool) -> None:
    count = 200
    df = pd.DataFrame(
        {
            "x": np.arange(count, dtype=float),
            "y": np.sin(np.arange(count) / 10.0),
            "yerr": np.full(count, 0.1),
        }
    )
    fig, ax = _render("scatter", df, {"errorevery": 10, "capsize": 3.0})

    container = _error_containers(ax)[0]
    drawn = len(container.lines[2][0].get_segments())
    assert drawn == count // 10

    if show_plots:
        target = plots_dir / "error_bars" / "errorevery.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=110, bbox_inches="tight")


def test_saved_figure_shows_symmetric_and_asymmetric(
    plots_dir: Path, show_plots: bool
) -> None:
    df = _frame(
        xerr=[0.15] * 4,
        yerr_low=[0.1, 0.2, 0.3, 0.4],
        yerr_high=[0.5, 0.4, 0.3, 0.2],
    )
    fig, ax = _render("scatter", df, {"capsize": 4.0, "elinewidth": 1.2})

    assert _error_containers(ax)

    if show_plots:
        target = plots_dir / "error_bars" / "asymmetric.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=110, bbox_inches="tight")
