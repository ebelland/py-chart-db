"""Regression test for the TablePreviewPanel right-click menu.

``_show_context_menu`` used to build its menu from ``_model()``, a helper
that only ever returns a ``LazyTableModel`` (the real-table model). The
guard at the top of the method, ``if model is None: return``, therefore made
the *entire* menu disappear whenever the panel was previewing a saved query
instead - queries are shown through the read-only ``DataFrameTableModel``,
so ``_model()`` returned None for them even though a table was clearly
loaded on screen. Right-clicking a query preview silently did nothing.

The fix keeps the table-writing items (delete column, hide rows, ensure
hide/cluster columns, ...) table-only, since a query has nothing in the repo
for them to write to, but the menu itself - with at least "Refresh data
table" - must still appear.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QWidget

from app.data.sqlite_repo import SqliteRepo
from app.widgets.table_preview_widget import TablePreviewPanel


def _repo_with_table_and_query(db_path: Path) -> SqliteRepo:
    repo = SqliteRepo(db_path=db_path)
    repo.import_dataframe(
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        table_name="t1",
        normalize_columns=False,
    )
    repo.save_query("q1", "SELECT * FROM t1")
    return repo


def test_context_menu_appears_for_a_real_table(qapp, tmp_db_path: Path) -> None:
    repo = _repo_with_table_and_query(tmp_db_path)
    host = QWidget()
    panel = TablePreviewPanel(parent=host, repo=repo)
    panel.set_context(repo, "t1")

    # _build_context_menu is the part of _show_context_menu that runs before
    # QMenu.exec() opens its own (blocking) local event loop, so it is the
    # part a headless test can call directly.
    menu = panel._build_context_menu(QPoint(0, 0))
    assert menu is not None

    texts = {action.text() for action in menu.actions() if not action.isSeparator()}
    assert "Refresh data table" in texts
    # Table-only actions are present for a real table.
    assert "Ensure Hide column" in texts


def test_context_menu_still_appears_for_a_saved_query(qapp, tmp_db_path: Path) -> None:
    """The bug: this used to return None (no menu shown at all)."""
    repo = _repo_with_table_and_query(tmp_db_path)
    host = QWidget()
    panel = TablePreviewPanel(parent=host, repo=repo)
    panel.set_context(repo, "q1")

    menu = panel._build_context_menu(QPoint(0, 0))

    assert menu is not None, "the context menu must not disappear for a saved query"
    texts = {action.text() for action in menu.actions() if not action.isSeparator()}
    assert "Refresh data table" in texts
    # Table-writing actions do not apply to a query: nothing in the repo
    # backs a "hide" or "cluster" column for it.
    assert "Ensure Hide column" not in texts
    assert "Add column from SQL expression..." not in texts
