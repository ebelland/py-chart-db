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
