"""The Series Operations panel: sectioned tile grid, hint bar, reflow.

A user wanting a control chart should not first have to work out which of
several sections it lives in - these tests pin the actual grouping - and a
square tile has no room for the description the old one-row list showed
inline, which is why it moved to a hint bar these also cover.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtGui import QEnterEvent
from PySide6.QtCore import QPointF

from app.scanners.series_operation_scanner import series_operations
from app.widgets.base_properties import BaseProperties
from app.widgets.series_operation import (
    OperationSection,
    OperationTile,
    SeriesOperationWidget,
    _group_by_section,
)


# ----------------------------------------------------------------------
# Shares the properties panels' look
# ----------------------------------------------------------------------
def test_the_panel_is_a_base_properties_widget(qapp) -> None:
    """Same panel background/size-policy convention as Figure/Axis/Series/
    Overlay properties, rather than a plain QWidget of its own."""
    widget = SeriesOperationWidget(None)
    assert isinstance(widget, BaseProperties)
    assert widget.objectName() == "basePropertiesPanel"


# ----------------------------------------------------------------------
# Grouping
# ----------------------------------------------------------------------
def test_every_discovered_operation_lands_in_a_section() -> None:
    """A plugin nothing groups it into is still shown - in "Other" - never
    silently dropped from the panel."""
    grouped = _group_by_section(list(series_operations))
    grouped_names = {op.get("value") for _title, ops in grouped for op in ops}
    all_names = {op.get("value") for op in series_operations}
    assert grouped_names == all_names


def test_the_approved_groupings_are_exact() -> None:
    grouped = dict(_group_by_section(list(series_operations)))
    by_section = {
        title: sorted(op.get("value") for op in ops) for title, ops in grouped.items()
    }
    assert by_section.get("Analysis") == sorted(["Peaks", "Roots", "Calculus"])
    assert by_section.get("Statistics") == sorted(
        ["Statistics", "Outliers", "Clustering", "Control Chart"]
    )
    assert by_section.get("Signal Processing") == sorted(
        ["Smoothing", "Spectral Analysis", "Filtering", "Baseline Correction"]
    )
    assert by_section.get("Modeling") == sorted(["Fit", "Interpolation", "Function"])


def test_an_unmapped_operation_falls_back_to_other_rather_than_vanishing() -> None:
    fake = {"name": "SomeNewDialog", "value": "Something New", "description": "", "icon": ""}
    grouped = dict(_group_by_section([*series_operations, fake]))
    assert fake in grouped["Other"]


def test_a_section_with_nothing_discovered_is_left_out() -> None:
    """Not shown as an empty header over nothing."""
    grouped = dict(_group_by_section([]))
    assert grouped == {}


def test_sections_appear_in_declared_order() -> None:
    grouped = _group_by_section(list(series_operations))
    titles = [title for title, _ops in grouped]
    assert titles == ["Analysis", "Statistics", "Signal Processing", "Modeling"]


# ----------------------------------------------------------------------
# The tile: click, hover/focus, keyboard
# ----------------------------------------------------------------------
def test_clicking_a_tile_emits_the_operation(qapp) -> None:
    section = OperationSection(
        None, "Analysis",
        [op for op in series_operations if op.get("value") == "Peaks"],
    )
    received: list[dict] = []
    section.operation_clicked.connect(received.append)

    tile = section.findChild(OperationTile)
    tile.clicked.emit()

    assert received and received[0].get("value") == "Peaks"


def test_hovering_a_tile_reports_its_title_and_description(qapp) -> None:
    section = OperationSection(
        None, "Analysis",
        [op for op in series_operations if op.get("value") == "Peaks"],
    )
    seen: list[tuple[str, str]] = []
    left_count = {"n": 0}
    section.tile_hovered.connect(lambda title, desc: seen.append((title, desc)))
    section.tile_left.connect(lambda: left_count.__setitem__("n", left_count["n"] + 1))

    tile = section.findChild(OperationTile)
    enter = QEnterEvent(QPointF(1, 1), QPointF(1, 1), QPointF(1, 1))
    tile.enterEvent(enter)
    tile.leaveEvent(QEvent(QEvent.Type.Leave))

    assert seen == [("Peaks", "Find and measure peaks")]
    assert left_count["n"] == 1


def test_keyboard_focus_reports_the_same_thing_hover_does(qapp) -> None:
    """Someone tabbing through with a keyboard gets the same explanation a
    mouse hover does - not just visual affordances a screen reader misses."""
    section = OperationSection(
        None, "Analysis",
        [op for op in series_operations if op.get("value") == "Peaks"],
    )
    seen: list[tuple[str, str]] = []
    section.tile_hovered.connect(lambda title, desc: seen.append((title, desc)))

    tile = section.findChild(OperationTile)
    tile.focusInEvent(None)

    assert seen == [("Peaks", "Find and measure peaks")]


def test_enter_and_space_activate_a_focused_tile(qapp) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    section = OperationSection(
        None, "Analysis",
        [op for op in series_operations if op.get("value") == "Peaks"],
    )
    received: list[dict] = []
    section.operation_clicked.connect(received.append)
    tile = section.findChild(OperationTile)

    for key in (Qt.Key.Key_Return, Qt.Key.Key_Space):
        received.clear()
        tile.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))
        assert received, f"key {key} did not activate the tile"


# ----------------------------------------------------------------------
# Reflow
# ----------------------------------------------------------------------
def test_the_grid_reflows_its_column_count_with_the_available_width(qapp) -> None:
    section = OperationSection(
        None, "Analysis",
        [op for op in series_operations if op.get("value") in ("Peaks", "Roots", "Calculus")],
    )
    section.show()
    qapp.processEvents()

    section.resize(150, 400)
    qapp.processEvents()
    assert section._columns == 1

    section.resize(400, 400)
    qapp.processEvents()
    assert section._columns == 3, "3 tiles, plenty of width - never more columns than tiles"

    section.resize(1000, 400)
    qapp.processEvents()
    assert section._columns == 3, "still never more columns than there are tiles"
    section.close()


# ----------------------------------------------------------------------
# The panel as a whole
# ----------------------------------------------------------------------
def test_the_panel_builds_with_a_placeholder_hint(qapp) -> None:
    widget = SeriesOperationWidget(None)
    assert "operation" in widget._hint_label.text().lower()


def test_hovering_any_tile_updates_the_panels_hint_bar(qapp) -> None:
    widget = SeriesOperationWidget(None)
    tile = widget.findChild(OperationTile)
    assert tile is not None

    enter = QEnterEvent(QPointF(1, 1), QPointF(1, 1), QPointF(1, 1))
    tile.enterEvent(enter)
    assert tile._title in widget._hint_label.text()

    tile.leaveEvent(QEvent(QEvent.Type.Leave))
    assert widget._hint_label.text() != ""
    assert tile._title not in widget._hint_label.text()


def test_new_plot_is_a_tile_like_every_other_operation(qapp) -> None:
    """Plot used to keep an older one-row treatment; it is a section of one
    tile now, styled and behaving exactly like the rest of the grid."""
    widget = SeriesOperationWidget(None)
    plot_tiles = [
        tile for tile in widget.findChildren(OperationTile)
        if tile.accessibleName() == "Plot"
    ]
    assert len(plot_tiles) == 1
    assert plot_tiles[0].objectName() == "operationTile"


def test_plot_gets_its_own_named_section(qapp) -> None:
    from PySide6.QtWidgets import QLabel

    widget = SeriesOperationWidget(None)
    plot_section = next(
        section
        for section in widget.findChildren(OperationSection)
        if any(tile.accessibleName() == "Plot" for tile in section.findChildren(OperationTile))
    )
    titles = [
        label.text()
        for label in plot_section.findChildren(QLabel)
        if label.property("sectionTitle")
    ]
    assert titles == ["Plot"]
