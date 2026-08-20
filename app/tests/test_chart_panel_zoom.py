"""The wheel zoom, tested without a window.

The handler takes a Matplotlib event and moves axis limits, so it can be
exercised by handing it an event object directly - no Qt, no scroll area, and
no guessing about which widget had focus.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from matplotlib.figure import Figure

from app.widgets.chart_panel_widget import ChartPanel


def _panel_and_axes():
    """A ChartPanel with just enough of itself to run the handler."""
    panel = ChartPanel.__new__(ChartPanel)
    figure = Figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.set_xlim(0.0, 10.0)
    axes.set_ylim(0.0, 10.0)
    panel._canvas = SimpleNamespace(draw_idle=lambda: None)
    return panel, axes


def _scroll(axes, *, step: float, key: str | None, x: float = 5.0, y: float = 5.0):
    return SimpleNamespace(inaxes=axes, step=step, key=key, xdata=x, ydata=y)


def test_the_wheel_alone_does_nothing() -> None:
    """A bare wheel scrolls the view in FIXED mode; taking it over would make
    the chart behave unlike every other scrollable thing in the window."""
    panel, axes = _panel_and_axes()

    panel._on_scroll_zoom(_scroll(axes, step=1, key=None))

    assert axes.get_xlim() == (0.0, 10.0)


def test_ctrl_wheel_zooms_in_and_out() -> None:
    panel, axes = _panel_and_axes()

    panel._on_scroll_zoom(_scroll(axes, step=1, key="control"))
    zoomed_in = axes.get_xlim()
    assert zoomed_in[1] - zoomed_in[0] < 10.0

    panel._on_scroll_zoom(_scroll(axes, step=-1, key="control"))
    assert axes.get_xlim()[1] - axes.get_xlim()[0] == pytest.approx(10.0)


def test_the_point_under_the_cursor_stays_put() -> None:
    """The whole reason to zoom about the cursor: you walk into a feature by
    pointing at it, instead of alternating zoom and pan."""
    panel, axes = _panel_and_axes()

    for _ in range(4):
        panel._on_scroll_zoom(_scroll(axes, step=1, key="control", x=2.0, y=8.0))

    low, high = axes.get_xlim()
    assert low < 2.0 < high
    # The cursor keeps its position *within* the visible range, not just
    # inside it: that is what "fixed under the pointer" means.
    assert (2.0 - low) / (high - low) == pytest.approx(0.2, abs=0.01)


def test_a_scroll_outside_the_axes_is_ignored() -> None:
    panel, axes = _panel_and_axes()

    panel._on_scroll_zoom(_scroll(axes, step=1, key="control"))
    after_valid = axes.get_xlim()

    event = _scroll(axes, step=1, key="control")
    event.inaxes = None
    panel._on_scroll_zoom(event)

    assert axes.get_xlim() == after_valid


def test_a_non_finite_cursor_position_is_skipped_not_fatal() -> None:
    """Log axes report NaN for a cursor left of zero."""
    panel, axes = _panel_and_axes()

    panel._on_scroll_zoom(_scroll(axes, step=1, key="control", x=float("nan")))

    # x untouched, y still zoomed: one bad axis must not lose the other.
    assert axes.get_xlim() == (0.0, 10.0)
    assert axes.get_ylim()[1] - axes.get_ylim()[0] < 10.0
