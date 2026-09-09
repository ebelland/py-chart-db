"""TableListPanel and TablePreviewPanel force Fusion on their own view.

Both style QAbstractItemView selection (a rounded "capsule" on the sidebar,
a flat highlight on the file list) and header colours directly through
macos_native.qss / fluent_win11.qss, and the native macOS/Windows Qt styles
paint a fair amount of that themselves - a table's selected-row highlight
in particular - in ways a stylesheet cannot always override. Scoped to just
these two views (QWidget.setStyle), not the whole application - see
apply_platform_style's own note on why every other, standard control stays
on the native style.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QWidget

from app.data.sqlite_repo import SqliteRepo
from app.styles.style import apply_fusion_for_item_view_styling
from app.widgets.table_list import TableListPanel
from app.widgets.table_preview import TablePreviewPanel


def test_the_helper_sets_a_fusion_style(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    """Checked by spying on setStyle() itself, not by reading
    widget.style() back afterwards: once *any* app-wide stylesheet is
    active - which every other test in this session's shared qapp may
    already have installed - Qt wraps a widget's true style (Fusion, or
    whatever it actually is) in a private QStyleSheetStyle proxy, and
    every such proxy's metaObject().className() reads "QStyleSheetStyle"
    regardless of what it wraps. The wrapping does not undo the per-widget
    override underneath it; it just makes that override unobservable this
    way.
    """
    calls: list[object] = []
    widget = QWidget()
    monkeypatch.setattr(widget, "setStyle", lambda s: calls.append(s))

    apply_fusion_for_item_view_styling(widget)

    assert len(calls) == 1
    assert calls[0].metaObject().className() == "QFusionStyle"


def test_table_list_panel_applies_the_helper_to_its_view(
    qapp, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    styled: list[QWidget] = []
    monkeypatch.setattr(
        "app.widgets.table_list.apply_fusion_for_item_view_styling",
        styled.append,
    )

    panel = TableListPanel(repo, None)

    assert styled == [panel._view]


def test_table_preview_panel_applies_the_helper_to_its_view(
    qapp, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    styled: list[QWidget] = []
    monkeypatch.setattr(
        "app.widgets.table_preview.apply_fusion_for_item_view_styling",
        styled.append,
    )

    panel = TablePreviewPanel(None, repo)

    assert styled == [panel.view]


def test_the_scroll_bars_are_put_back_on_the_application_style(qapp) -> None:
    """A widget's style is inherited by its children, so Fusion on the view
    reached the two scroll bars as well - and Fusion with no stylesheet to
    follow draws a square black handle between two stepper arrows, beside
    the rounded native ones on every other scrolling widget in the window.

    Read back by identity rather than by class name: what matters is that
    the bars are on the *same* style object as the rest of the application,
    whatever that object happens to be wrapped in.
    """
    from PySide6.QtWidgets import QApplication, QTableView

    view = QTableView()
    apply_fusion_for_item_view_styling(view)

    assert view.verticalScrollBar().style() is QApplication.style()
    assert view.horizontalScrollBar().style() is QApplication.style()
    assert view.style() is not QApplication.style(), "the view itself is Fusion"


def test_both_panels_leave_their_scroll_bars_alone(
    qapp, repo: SqliteRepo
) -> None:
    """Through the panels themselves, since that is where it was seen."""
    from PySide6.QtWidgets import QApplication

    host = QWidget()
    for view in (
        TableListPanel(repo, host)._view,
        TablePreviewPanel(host, repo).view,
    ):
        assert view.verticalScrollBar().style() is QApplication.style()
        assert view.horizontalScrollBar().style() is QApplication.style()


# ----------------------------------------------------------------------
# QToolBox pages
# ----------------------------------------------------------------------
def test_a_toolbox_page_gets_a_ground_and_room_to_float_its_cards(qapp) -> None:
    """A page is a plain QWidget: a stylesheet background on one is parsed
    and never painted without WA_StyledBackground, so the white cards on it
    showed nothing but their own rounded corners against the white behind."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QToolBox, QVBoxLayout

    from app.styles.style import (
        MARGIN_TOOLBOX_PAGE,
        apply_toolbox_page_metrics,
    )

    toolbox = QToolBox()
    page = QWidget()
    QVBoxLayout(page).setContentsMargins(0, 0, 0, 0)
    toolbox.addItem(page, "Page")

    apply_toolbox_page_metrics(toolbox)

    assert page.property("toolboxPage") is True
    assert page.testAttribute(Qt.WidgetAttribute.WA_StyledBackground)
    assert page.layout().getContentsMargins() == MARGIN_TOOLBOX_PAGE


def test_a_page_that_chose_its_own_margins_keeps_them(qapp) -> None:
    """The properties panels pick MARGIN_PANEL; a helper called for the
    ground colour has no business overruling a margin somebody chose."""
    from PySide6.QtWidgets import QToolBox, QVBoxLayout

    from app.styles.style import apply_toolbox_page_metrics

    toolbox = QToolBox()
    page = QWidget()
    QVBoxLayout(page).setContentsMargins(6, 6, 6, 6)
    toolbox.addItem(page, "Page")

    apply_toolbox_page_metrics(toolbox)

    assert page.layout().getContentsMargins() == (6, 6, 6, 6)
    assert page.property("toolboxPage") is True, "the ground still applies"


def test_both_sheets_say_what_a_toolbox_page_is(qapp) -> None:
    """Opposite answers on purpose - grey on macOS, where white cards float
    on a settings pane; white on Windows, which puts outlined cards on one
    continuous page - and neither may be left to inheritance."""
    from pathlib import Path

    styles = Path(__file__).resolve().parent.parent / "styles"
    for sheet, expected in (
        ("macos_native.qss", "palette(window)"),
        ("fluent_win11.qss", "palette(base)"),
    ):
        text = (styles / sheet).read_text(encoding="utf-8")
        rule = text.split('QWidget[toolboxPage="true"]', 1)
        assert len(rule) == 2, f"{sheet} does not style a toolbox page"
        assert expected in rule[1].split("}", 1)[0], sheet
