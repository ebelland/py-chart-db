"""Tests for the axis options applied after a renderer has drawn.

These options are order-sensitive: ``set_xscale`` resets the view interval and
``set_xlim`` rewrites the direction, so scale, limits and inversion have to be
applied in that order or one of the three is silently lost.  The first tests
below pin exactly that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from app.data.descriptors import AxisDescriptor
from app.charts.render_figure import _apply_axis_runtime_options


def _axis(options: dict) -> AxisDescriptor:
    """Build an axis descriptor carrying only the options under test."""
    return AxisDescriptor(
        id=1,
        figure_id=1,
        axis_index=0,
        chart_type="Scatter Plot",
        title="",
        x_label="",
        y_label="",
        z_label="",
        options=options,
        series=[],
    )


def _drawn_axis(options: dict):
    """Return an axis with a line drawn on it and the options applied."""
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.linspace(1.0, 1000.0, 50), np.linspace(1.0, 500.0, 50))
    _apply_axis_runtime_options(ax=ax, axis_desc=_axis(options))
    return fig, ax


@pytest.mark.parametrize("scale", ["linear", "log", "symlog", "logit"])
def test_every_supported_scale_is_applied(scale: str) -> None:
    _, ax = _drawn_axis({"x_scale": scale, "y_scale": scale})
    assert ax.get_xscale() == scale
    assert ax.get_yscale() == scale


def test_log_base_is_honoured() -> None:
    _, ax = _drawn_axis({"y_scale": "log", "y_scale_base": 2.0})
    assert ax.get_yscale() == "log"
    assert ax.yaxis.get_transform().base == pytest.approx(2.0)


def test_inversion_survives_explicit_limits() -> None:
    """set_xlim rewrites the direction, so inversion must be applied after it."""
    _, ax = _drawn_axis({"invert_x": True, "xlim": [0.0, 100.0]})

    low, high = ax.get_xlim()
    assert low > high, f"x axis is not inverted: {(low, high)}"
    assert sorted((low, high)) == pytest.approx([0.0, 100.0])


def test_inversion_is_absolute_not_a_toggle() -> None:
    """Re-applying the same options must not flip the axis back."""
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([1, 2, 3], [1, 2, 3])
    options = {"invert_y": True}

    for _ in range(3):
        _apply_axis_runtime_options(ax=ax, axis_desc=_axis(options))

    low, high = ax.get_ylim()
    assert low > high, "repeated renders toggled the axis instead of setting it"


def test_inversion_can_be_turned_off_again() -> None:
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([1, 2, 3], [1, 2, 3])

    _apply_axis_runtime_options(ax=ax, axis_desc=_axis({"invert_x": True}))
    _apply_axis_runtime_options(ax=ax, axis_desc=_axis({"invert_x": False}))

    low, high = ax.get_xlim()
    assert low < high, "clearing invert_x did not restore the normal direction"


def test_scale_and_limits_and_inversion_together() -> None:
    _, ax = _drawn_axis(
        {"x_scale": "log", "xlim": [1.0, 1000.0], "invert_x": True}
    )
    low, high = ax.get_xlim()
    assert ax.get_xscale() == "log"
    assert low > high
    assert sorted((low, high)) == pytest.approx([1.0, 1000.0])


def test_grid_which_and_axis() -> None:
    _, ax = _drawn_axis({"grid": True, "grid_which": "both", "grid_axis": "x"})

    assert any(line.get_visible() for line in ax.xaxis.get_gridlines())
    # A minor grid is pointless without minor ticks, so asking for one must
    # switch them on.
    assert ax.xaxis.get_minorticklocs().size > 0


def test_ticks_direction_and_rotation() -> None:
    _, ax = _drawn_axis(
        {"minor_ticks": True, "tick_direction": "in", "x_tick_rotation": 45.0}
    )
    assert ax.xaxis.get_minorticklocs().size > 0
    assert all(
        label.get_rotation() == pytest.approx(45.0) for label in ax.get_xticklabels()
    )


def test_spines_can_be_hidden() -> None:
    _, ax = _drawn_axis({"hide_spine_top": True, "hide_spine_right": True})

    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.spines["bottom"].get_visible()
    assert ax.spines["left"].get_visible()


def test_unknown_values_are_logged_not_raised() -> None:
    """A bad option must degrade to the default rather than break the render."""
    _, ax = _drawn_axis(
        {
            "x_scale": "banana",
            "grid": True,
            "grid_which": "sometimes",
            "grid_axis": "diagonal",
            "tick_direction": "sideways",
            "x_tick_rotation": "a lot",
        }
    )
    assert ax.get_xscale() == "linear"


def test_saved_figure_shows_the_options(plots_dir: Path, show_plots: bool) -> None:
    fig, _ = _drawn_axis(
        {
            "x_scale": "log",
            "y_scale": "log",
            "invert_y": True,
            "grid": True,
            "grid_which": "both",
            "minor_ticks": True,
            "tick_direction": "in",
            "x_tick_rotation": 30.0,
            "hide_spine_top": True,
            "hide_spine_right": True,
        }
    )
    if show_plots:
        target = plots_dir / "axis_options" / "scales_and_decorations.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=110, bbox_inches="tight")
