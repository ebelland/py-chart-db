"""Adding a reference line from the chart's own context menu (todo.txt N-8).

The storage and the drawing landed with the Overlay properties panel: a
"lines" entry on the axis, drawn by BaseAxisRenderer.apply_reference_lines.
What was missing is the gesture - right-click where the line goes - and it
is the one that matters, because a threshold is something you see on the
chart before you can type its value into a table.

Two things are worth pinning. The click has to reach the right axis, which
means Qt's top-left pixels turned into Matplotlib's bottom-left ones and
then into data coordinates; get that wrong and the line lands somewhere
plausible but not where it was asked for. And the entries only appear when
the click is inside an axes at all - "add a line here" over the figure
margin has no *here*.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QPoint

from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import ChartPanel


@pytest.fixture
def panel(qapp, repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0)}),
        table_name="w",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="t", x_label="x", y_label="y", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM w", roles={"x": "x", "y": "y"}, style={},
    )
    built = ChartPanel(repo, figure_id)
    built.resize(600, 400)
    built.show()
    qapp.processEvents()
    yield built, axis_id
    built.close()
    repo.undo_store.discard_file()


def _centre(panel: ChartPanel) -> QPoint:
    canvas = panel._canvas
    return QPoint(canvas.width() // 2, canvas.height() // 2)


def _stored_lines(repo: SqliteRepo, axis_id: int) -> list:
    return list((repo.get_axis_options(axis_id) or {}).get("lines") or [])


# ----------------------------------------------------------------------
# Where the click landed
# ----------------------------------------------------------------------
def test_a_click_inside_the_axes_finds_it_and_its_data_coordinates(panel) -> None:
    built, axis_id = panel

    found = built._axis_at(_centre(built))

    assert found is not None
    clicked_axis, _axes, x_value, y_value = found
    assert clicked_axis == axis_id
    # The data runs 0..19 on both axes, so the middle of the axes is the
    # middle of the data - and not, say, the middle in pixels.
    assert 7.0 < x_value < 12.0
    assert 7.0 < y_value < 12.0


def test_a_click_outside_every_axes_finds_nothing(panel) -> None:
    """The figure margin has no "here" to put a line at."""
    built, _axis_id = panel

    assert built._axis_at(QPoint(2, 2)) is None


def test_the_axis_comes_from_the_tag_the_renderer_left(panel) -> None:
    """render_figure tags each axes with _dhub_axis_id, so a click maps back
    to the descriptor without this panel keeping a second mapping that could
    fall out of step with the figure."""
    built, axis_id = panel

    assert getattr(built._figure.axes[0], "_dhub_axis_id", None) == axis_id


# ----------------------------------------------------------------------
# The menu
# ----------------------------------------------------------------------
def test_the_menu_offers_both_lines_and_names_their_values(panel) -> None:
    built, _axis_id = panel

    texts = [action.text() for action in built.context_menu_for(_centre(built)).actions()]

    assert any(text.startswith("Add vertical line at x = ") for text in texts)
    assert any(text.startswith("Add horizontal line at y = ") for text in texts)


def test_the_plain_menu_is_shown_outside_the_axes(panel) -> None:
    built, _axis_id = panel

    texts = [action.text() for action in built.context_menu_for(QPoint(2, 2)).actions()]

    assert not any("line" in text for text in texts)
    assert "Reload" in texts


def test_the_value_is_written_the_way_the_axis_labels_it(qapp, repo) -> None:
    """A dated axis reads 2024-03-07, not 1709769600 - the renderers store
    epoch seconds and only the axis formatter knows how to turn one back."""
    days = pd.date_range("2024-01-01", periods=20, freq="D")
    repo.import_dataframe(
        pd.DataFrame({"t": days.strftime("%Y-%m-%d"), "v": np.arange(20.0)}),
        table_name="ts",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Time Series",
            title="t", x_label="t", y_label="v", options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT t AS x, v AS y FROM ts",
        roles={"x": "x", "y": "y"}, style={},
    )
    built = ChartPanel(repo, figure_id)
    built.resize(600, 400)
    built.show()
    qapp.processEvents()
    try:
        texts = [
            action.text() for action in built.context_menu_for(_centre(built)).actions()
        ]
        vertical = next(text for text in texts if text.startswith("Add vertical"))
        assert "2024" in vertical, vertical
    finally:
        built.close()
        repo.undo_store.discard_file()


# ----------------------------------------------------------------------
# What it writes
# ----------------------------------------------------------------------
def test_adding_a_line_stores_it_on_the_axis(panel) -> None:
    built, axis_id = panel

    built._add_reference_line(axis_id, "vertical", 4.5)

    assert _stored_lines(built._repo, axis_id) == [
        {"orientation": "vertical", "value": 4.5, "kwargs": {}}
    ]


def test_a_second_line_joins_the_first(panel) -> None:
    built, axis_id = panel

    built._add_reference_line(axis_id, "vertical", 4.5)
    built._add_reference_line(axis_id, "horizontal", 2.0)

    assert [line["orientation"] for line in _stored_lines(built._repo, axis_id)] == [
        "vertical",
        "horizontal",
    ]


def test_the_line_is_drawn_after_it_is_added(panel) -> None:
    built, axis_id = panel

    built._add_reference_line(axis_id, "vertical", 4.5)

    axes = built._figure.axes[0]
    assert any(
        list(line.get_xdata()) == [4.5, 4.5] for line in axes.lines
    ), "the reload did not draw it"


def test_adding_a_line_can_be_undone(panel) -> None:
    """It writes to the axis descriptor like any other edit, so it takes the
    same route back."""
    built, axis_id = panel
    built._add_reference_line(axis_id, "vertical", 4.5)

    assert [entry.label for entry in built._repo.undo_entries()] == [
        "Add vertical line"
    ]

    built._repo.undo_last()

    assert _stored_lines(built._repo, axis_id) == []


def test_the_line_is_the_same_key_the_overlay_panel_edits(panel) -> None:
    """One storage, two ways in: dropped from the chart, restyled in the
    panel."""
    from app.widgets.overlay_properties import OverlayPropertiesWidget

    built, axis_id = panel
    built._add_reference_line(axis_id, "horizontal", 3.0)

    widget = OverlayPropertiesWidget()
    widget.set_connected_figure(built._repo, built._figure_id, built._figure)
    widget.set_axis(axis_id)

    assert widget._lines_table.rowCount() == 1
    assert widget._lines_table.item(0, 1).text() == "3.0"
