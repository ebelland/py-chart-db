"""Deleting from the table list, where both kinds of source are listed.

A saved query and a table sit in the same list and look almost the same - a
"Q" badge apart - but they are deleted differently. The list used to run
DROP TABLE on whatever was selected, so deleting a query dropped nothing and
the row was still there when the list reloaded: no error, no message, and a
user pressing Delete a second time.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.widgets import table_list as table_list_module
from app.widgets.table_list import TableListPanel


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    repo = SqliteRepo(db_path=tmp_db_path)
    repo.import_dataframe(
        pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}),
        table_name="readings",
        normalize_columns=False,
    )
    repo.save_query("first_two", "SELECT x, y FROM readings LIMIT 2")
    yield repo
    repo.close()


@pytest.fixture
def panel(qapp, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch):
    """A list showing one table and one saved query, answering Yes to Delete."""
    asked: list[str] = []
    monkeypatch.setattr(
        table_list_module,
        "ask",
        lambda _parent, message_id, **_k: asked.append(message_id) or True,
    )

    built = TableListPanel(repo=repo, parent=None)
    built.reload()
    built.asked = asked  # type: ignore[attr-defined]
    return built


def _row_of(panel: TableListPanel, name: str) -> int:
    for row in range(panel._model.rowCount()):
        item = panel._model.item(row, panel.COL_TABLE)
        if item is not None and str(item.data(panel.ROLE_TABLE_NAME)) == name:
            return row
    raise AssertionError(f"{name} is not in the list")


def _select(panel: TableListPanel, *names: str) -> None:
    """Select whole rows by name, leaving the current index on the last one.

    Through the selection model rather than the view: QAbstractItemView.
    setCurrentIndex clears the selection it is given, so calling it inside the
    loop leaves exactly one row selected however many were asked for.
    """
    from PySide6.QtCore import QItemSelection, QItemSelectionModel

    selection = panel._view.selectionModel()
    selection.clearSelection()

    ranges = QItemSelection()
    last = None
    for name in names:
        row = _row_of(panel, name)
        last = panel._model.index(row, panel.COL_TABLE)
        ranges.select(last, panel._model.index(row, panel.COL_NOTES))

    selection.select(ranges, QItemSelectionModel.SelectionFlag.Select)
    if last is not None:
        selection.setCurrentIndex(last, QItemSelectionModel.SelectionFlag.NoUpdate)


# ----------------------------------------------------------------------
# What the list knows about a row
# ----------------------------------------------------------------------
def test_the_list_shows_tables_and_saved_queries_together(panel) -> None:
    names = {
        str(panel._model.item(row, panel.COL_TABLE).data(panel.ROLE_TABLE_NAME))
        for row in range(panel._model.rowCount())
    }

    assert {"readings", "first_two"} <= names


def test_a_selected_row_says_which_kind_it_is(panel) -> None:
    _select(panel, "first_two")
    assert panel.selected_sources() == [("first_two", True)]

    _select(panel, "readings")
    assert panel.selected_sources() == [("readings", False)]


# ----------------------------------------------------------------------
# Deleting
# ----------------------------------------------------------------------
def test_deleting_a_saved_query_removes_it(panel, repo: SqliteRepo) -> None:
    """The bug: DROP TABLE "first_two" matches nothing, so it survived."""
    _select(panel, "first_two")

    panel._delete_selected()

    assert repo.get_query("first_two") is None
    assert "readings" in repo.list_table_names()


def test_deleting_a_query_asks_the_question_about_queries(panel) -> None:
    """Its consequence is different: a chart built on it stops finding data."""
    _select(panel, "first_two")

    panel._delete_selected()

    assert panel.asked == ["query.confirm_delete"]


def test_deleting_a_table_still_deletes_the_table(panel, repo: SqliteRepo) -> None:
    _select(panel, "readings")

    panel._delete_selected()

    assert "readings" not in repo.list_table_names()
    assert repo.get_query("first_two") is not None
    assert panel.asked == ["table.confirm_delete"]


def test_a_mixed_selection_deletes_both_and_says_so(panel, repo: SqliteRepo) -> None:
    _select(panel, "readings", "first_two")

    panel._delete_selected()

    assert "readings" not in repo.list_table_names()
    assert repo.get_query("first_two") is None
    assert panel.asked == ["source.confirm_delete_many"]


def test_several_tables_keep_the_table_wording(panel, repo: SqliteRepo) -> None:
    repo.import_dataframe(
        pd.DataFrame({"a": [1.0]}), table_name="second", normalize_columns=False
    )
    panel.reload()
    _select(panel, "readings", "second")

    panel._delete_selected()

    assert panel.asked == ["table.confirm_delete_many"]
    assert repo.get_query("first_two") is not None


def test_saying_no_deletes_nothing(
    panel, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(table_list_module, "ask", lambda *_a, **_k: False)
    _select(panel, "first_two")

    panel._delete_selected()

    assert repo.get_query("first_two") is not None
