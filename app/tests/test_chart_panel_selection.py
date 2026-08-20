"""The status-bar readout for a picked point, tested without a window.

Like the wheel zoom, the handler takes a Matplotlib event and produces a
string, so it can be driven by handing it an event built by hand - no Qt, no
canvas, and no clicking at pixel coordinates that depend on the figure size.

What the tests are guarding is mostly *formatting*: the readout is the only
place in the application where a stored value is shown back raw, and the
renderers store time as epoch seconds.  A readout that said ``1710000000``
where the axis said ``2024-03-09`` would be correct and useless.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.dates import ConciseDateFormatter, num2date
from matplotlib.figure import Figure

from app.widgets.chart_panel_widget import PICK_TOLERANCE_POINTS, ChartPanel, axis_text


def _panel():
    """A ChartPanel with just enough of itself to run the handler.

    ``selection_changed`` is a Qt signal on the class, so an instance made
    without ``__init__`` cannot emit it; the recorder below stands in for it
    and collects what would have reached the status bar.
    """
    panel = ChartPanel.__new__(ChartPanel)
    emitted: list[str] = []
    panel.selection_changed = SimpleNamespace(emit=emitted.append)
    return panel, emitted


def _scatter(x, y, label: str = "Batch A"):
    figure = Figure()
    axes = figure.add_subplot(1, 1, 1)
    return axes.scatter(x, y, label=label)


# ----------------------------------------------------------------------
# What a selection says
# ----------------------------------------------------------------------
def test_one_point_reads_out_its_series_and_its_values() -> None:
    panel, emitted = _panel()
    artist = _scatter([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])

    panel._on_pick(SimpleNamespace(artist=artist, ind=[1]))

    assert emitted == ["Batch A — x: 2.000, y: 20.000"]


def test_several_points_read_out_the_count_and_the_means() -> None:
    """The case a single-point readout cannot serve: overlapping markers.

    Matplotlib hands back every index under the cursor, so this is the same
    event as a single pick with a longer ``ind`` - which is why there is one
    handler and not two.
    """
    panel, emitted = _panel()
    artist = _scatter([1.0, 2.0, 3.0], [10.0, 20.0, 60.0])

    panel._on_pick(SimpleNamespace(artist=artist, ind=[0, 1, 2]))

    assert emitted == ["Batch A — 3 points, mean x: 2.000, mean y: 30.000"]


def test_a_line_is_read_from_its_data_not_from_offsets() -> None:
    """Lines and collections carry their points differently.

    A scatter is a PathCollection with Nx2 offsets; a plotted line has two
    parallel sequences and no ``get_offsets`` at all.  Both are armed for
    picking, so both have to be readable.
    """
    panel, emitted = _panel()
    figure = Figure()
    axes = figure.add_subplot(1, 1, 1)
    (line,) = axes.plot([0.0, 5.0, 10.0], [3.0, 4.0, 5.0], label="Trend")

    panel._on_pick(SimpleNamespace(artist=line, ind=[2]))

    assert emitted == ["Trend — x: 10.000, y: 5.000"]


def test_an_unnamed_artist_is_not_reported_as_child0() -> None:
    """Matplotlib names an artist that was not given a label ``_child0``.

    That is an internal name, and the underscore prefix is also how decoration
    is kept out of the legend, so an underscore means "no name" here.
    """
    panel, emitted = _panel()
    figure = Figure()
    axes = figure.add_subplot(1, 1, 1)
    (line,) = axes.plot([0.0, 1.0], [0.0, 1.0])

    panel._on_pick(SimpleNamespace(artist=line, ind=[0]))

    assert emitted == ["series — x: 0.000, y: 0.000"]


def test_a_pick_carrying_no_indices_says_nothing() -> None:
    """Clearing the status bar on a stray event would wipe a real readout."""
    panel, emitted = _panel()
    artist = _scatter([1.0], [1.0])

    panel._on_pick(SimpleNamespace(artist=artist, ind=[]))

    assert emitted == []


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------
def test_a_date_axis_reads_out_a_date_not_a_number() -> None:
    """The reason the axis formatter is asked instead of formatting the float.

    Time series are stored as epoch seconds and converted for plotting, so the
    number in the artist is meaningless to a reader.  Only the axis knows how
    to turn it back into the label it is drawn under.
    """
    figure = Figure()
    axes = figure.add_subplot(1, 1, 1)
    days = np.arange(19700.0, 19710.0)
    axes.plot(days, days)
    axes.xaxis.set_major_formatter(ConciseDateFormatter(axes.xaxis.get_major_locator()))

    text = axis_text(axes.xaxis, 19700.0)

    assert str(num2date(19700.0).year) in text
    assert "19700" not in text


def test_a_formatter_that_raises_falls_back_to_the_number() -> None:
    """Renderers may install any callable as a formatter.

    One that cannot handle the value should cost the nicety, not the readout.
    """

    class Exploding:
        def format_data_short(self, value: float) -> str:
            raise ValueError("no")

    axis = SimpleNamespace(get_major_formatter=Exploding)

    assert axis_text(axis, 1.5) == "1.5"


# ----------------------------------------------------------------------
# Arming
# ----------------------------------------------------------------------
def test_rendering_arms_lines_and_collections_but_not_patches() -> None:
    """Bars and wedges have no per-point index to report.

    A pick on one would come back as a single point and mean the whole shape,
    which reads as a value the chart does not contain.
    """
    panel = ChartPanel.__new__(ChartPanel)
    figure = Figure()
    axes = figure.add_subplot(1, 1, 1)
    (line,) = axes.plot([0.0, 1.0], [0.0, 1.0])
    points = axes.scatter([0.0, 1.0], [1.0, 0.0])
    bars = axes.bar([0.0, 1.0], [1.0, 2.0])
    panel._figure = figure

    panel._make_data_artists_pickable()

    assert line.get_picker() == pytest.approx(PICK_TOLERANCE_POINTS)
    assert points.get_picker() == pytest.approx(PICK_TOLERANCE_POINTS)
    assert all(bar.get_picker() is None for bar in bars)


def test_the_window_shows_the_readout_in_the_status_bar() -> None:
    """The signal exists to reach the status bar; a panel wired to nothing is
    a feature nobody can see."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "dialogs" / "main_window.py"
    ).read_text(encoding="utf-8")

    assert "panel.selection_changed.connect" in source
    assert "showMessage" in source[source.index("panel.selection_changed.connect") :]
