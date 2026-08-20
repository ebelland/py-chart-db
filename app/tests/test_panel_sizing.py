"""Tests for the rules that decide how narrow a side panel can be.

The left panel could not be dragged below about 300 px, and then stopped
respecting even that: whichever pane kept a large implicit minimum - first a
24-character combo, then the chart toolbar - won the splitter negotiation and
the other pane was squeezed past the minimum it had asked for.

These tests pin the two rules that replaced that: an implicit floor can be
cleared, and a deliberate one cannot.
"""
from __future__ import annotations

import pytest

from app.styles import style

qt = pytest.importorskip("PySide6.QtWidgets")





@pytest.fixture
def panel(qapp):
    """A panel with the two things that used to pin the width."""
    widget = qt.QWidget()
    layout = qt.QVBoxLayout(widget)

    combo = qt.QComboBox(widget)
    combo.setMinimumContentsLength(24)
    combo.addItem("a very long renderer name indeed")
    layout.addWidget(combo)

    view = qt.QTableView(widget)
    layout.addWidget(view)
    return widget


def test_a_combo_no_longer_dictates_the_panel_width(panel) -> None:
    before = panel.minimumSizeHint().width()
    style.relax_minimum_width(panel)
    after = panel.minimumSizeHint().width()

    assert after < before, "the implicit floor is still there"


def test_the_combo_keeps_a_readable_width(panel) -> None:
    """Relaxing the floor must not collapse the control to nothing."""
    style.relax_minimum_width(panel)
    combo = panel.findChild(qt.QComboBox)

    assert combo.minimumContentsLength() == style.COMBO_MIN_CONTENTS_LENGTH


def test_a_deliberately_fixed_width_is_left_alone(qapp) -> None:
    """The activity rail is 48 px because someone decided so."""
    parent = qt.QWidget()
    layout = qt.QVBoxLayout(parent)
    rail = qt.QWidget(parent)
    rail.setFixedWidth(48)
    layout.addWidget(rail)

    style.relax_minimum_width(parent)

    assert rail.minimumWidth() == 48
    assert rail.maximumWidth() == 48


def test_the_full_value_stays_reachable_in_the_tooltip(qapp) -> None:
    """Elided text is only acceptable if the value can still be read."""
    combo = qt.QComboBox()
    combo.addItems(["short", "a very long renderer name indeed"])
    style.configure_combo_width(combo)

    combo.setCurrentIndex(1)
    assert combo.toolTip() == "a very long renderer name indeed"


def test_a_requested_width_is_capped_not_honoured(qapp) -> None:
    combo = qt.QComboBox()
    style.configure_combo_width(combo, minimum_contents_length=24)
    assert combo.minimumContentsLength() == style.COMBO_MIN_CONTENTS_LENGTH


def test_a_smaller_request_is_honoured(qapp) -> None:
    combo = qt.QComboBox()
    style.configure_combo_width(combo, minimum_contents_length=2)
    assert combo.minimumContentsLength() == 2
