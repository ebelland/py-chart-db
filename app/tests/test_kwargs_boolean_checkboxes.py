"""Booleans in the kwargs editor are checkboxes, always, on every platform.

They used to be text that turned into a checkbox only while the row was being
edited: two clicks to change one, and on macOS the editor's box was placed by
the delegate rather than by the row, so a click frequently missed it and the
setting appeared simply not to work.

A check state is drawn by the view itself, so the box is there whether or not
anything is being edited and one click toggles it. These tests pin the three
things that make that true: the box is drawn, no second one is built on top of
it, and a toggle reaches the value.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from app.widgets.dictionary_editor import DictEditorPanel

CONFIG = {
    "antialiased": {
        "default": True,
        "type": bool,
        "group": "Appearance",
        "description": "Antialias the edges",
    },
    "filled": {
        "default": False,
        "type": bool,
        "group": "Appearance",
        "description": "Fill the bands",
    },
    "alpha": {
        "default": 0.5,
        "type": float,
        "group": "Appearance",
        "description": "Opacity",
    },
    "cmap": {
        "default": "viridis",
        "type": str,
        "group": "Appearance",
        "description": "Colormap",
    },
}


@pytest.fixture
def panel(qapp) -> DictEditorPanel:
    built = DictEditorPanel(config=CONFIG)
    built.set_values({"antialiased": True, "filled": False, "alpha": 0.5, "cmap": "viridis"})
    return built


def _box_is_drawn(panel: DictEditorPanel, key: str) -> bool:
    """A checkbox is painted only where CheckStateRole is actually set."""
    item = panel._key_items[key]
    return item.data(1, Qt.ItemDataRole.CheckStateRole) is not None


def test_a_boolean_row_draws_its_checkbox_without_being_edited(panel) -> None:
    assert _box_is_drawn(panel, "antialiased")
    assert _box_is_drawn(panel, "filled")


def test_the_box_starts_on_the_value(panel) -> None:
    assert panel._key_items["antialiased"].checkState(1) == Qt.CheckState.Checked
    assert panel._key_items["filled"].checkState(1) == Qt.CheckState.Unchecked


def test_nothing_else_grows_a_checkbox(panel) -> None:
    assert not _box_is_drawn(panel, "alpha")
    assert not _box_is_drawn(panel, "cmap")


def test_a_boolean_row_is_not_also_editable(panel) -> None:
    """Editable and checkable together is two controls for one value."""
    item = panel._key_items["filled"]

    assert not bool(item.flags() & Qt.ItemFlag.ItemIsEditable)
    assert bool(item.flags() & Qt.ItemFlag.ItemIsUserCheckable)


def test_the_delegate_builds_no_second_box_on_top(panel) -> None:
    """The editor is what macOS placed somewhere the click could miss."""
    index = panel.tree.indexFromItem(panel._key_items["filled"], 1)
    delegate = panel.tree.itemDelegateForColumn(1)

    assert delegate.createEditor(panel.tree, None, index) is None


def test_other_kinds_still_get_their_editor(panel) -> None:
    from PySide6.QtWidgets import QDoubleSpinBox

    index = panel.tree.indexFromItem(panel._key_items["alpha"], 1)
    delegate = panel.tree.itemDelegateForColumn(1)

    assert isinstance(delegate.createEditor(panel.tree, None, index), QDoubleSpinBox)


def test_ticking_the_box_reaches_the_value(panel) -> None:
    panel._key_items["filled"].setCheckState(1, Qt.CheckState.Checked)

    assert panel.get_values()["filled"] is True


def test_unticking_the_box_reaches_the_value(panel) -> None:
    panel._key_items["antialiased"].setCheckState(1, Qt.CheckState.Unchecked)

    assert panel.get_values()["antialiased"] is False


def test_the_value_column_carries_no_second_answer(panel) -> None:
    """The box already says True or False; a label beside it would be a
    second answer to the same question, and one that can disagree."""
    assert panel._key_items["filled"].text(1) == ""


def test_setting_the_value_from_outside_moves_the_box(panel) -> None:
    """The panel is filled from the descriptor when an axis is selected, not
    only by clicking - a box that ignored that would show the previous axis'
    setting."""
    panel.set_values({"antialiased": False, "filled": True, "alpha": 0.5, "cmap": "viridis"})

    assert panel._key_items["antialiased"].checkState(1) == Qt.CheckState.Unchecked
    assert panel._key_items["filled"].checkState(1) == Qt.CheckState.Checked


def test_a_round_trip_through_the_box_keeps_a_real_bool(panel) -> None:
    """Not the string "True": the value is forwarded to Matplotlib."""
    panel._key_items["filled"].setCheckState(1, Qt.CheckState.Checked)

    assert isinstance(panel.get_values()["filled"], bool)
