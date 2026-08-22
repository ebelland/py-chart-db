"""The query builder: starting a query, and building one from a field.

The dialog was rewritten around two ideas - a query is *named first* and
saved later, and the SQL buttons act on the table and field chosen beside
them rather than pasting a skeleton at the cursor. Both are easy to break in
ways nothing else notices: a draft that gets written to the database before it
runs leaves rows no chart can read, and a clause command that appends instead
of replacing produces a statement with two WHERE clauses that SQLite refuses.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.query_builder_dialog import QueryBuilderDialog


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("DROP TABLE IF EXISTS batch_yields")
    repo.query_df("DROP TABLE IF EXISTS batch_notes")
    repo.import_dataframe(
        pd.DataFrame({"batch": list("aabbb"), "yield_pct": [1.0, 2.0, 3.0, 4.0, 5.0]}),
        table_name="batch_yields",
        normalize_columns=False,
    )
    repo.import_dataframe(
        pd.DataFrame({"batch": list("ab"), "note": ["x", "y"]}),
        table_name="batch_notes",
        normalize_columns=False,
    )
    yield repo
    repo.close()


@pytest.fixture
def dialog(qapp, repo: SqliteRepo, monkeypatch: pytest.MonkeyPatch) -> QueryBuilderDialog:
    """A builder that neither reads nor writes the user's config.json.

    The dialog remembers its editor contents between sessions, so without
    this a test starts with whatever SQL was last typed in the real
    application - which is how this file first "failed": with a half-written
    JOIN from someone's session in the editor. It would also write the test's
    own state back into the file. (See P2-9: config.json mixes the action
    catalogue with per-session state.)
    """
    import app.dialogs.query_builder_dialog as module

    for name in (
        "restore_dialog_state",
        "restore_window_geometry",
        "save_dialog_state",
        "save_window_geometry",
    ):
        monkeypatch.setattr(module, name, lambda *_a, **_k: None)

    built = QueryBuilderDialog(repo)
    built._table_combo.setCurrentText("batch_yields")
    built._reload_fields()
    built._field_combo.setCurrentText("yield_pct")
    return built


def _new_query(dialog: QueryBuilderDialog, name: str, monkeypatch) -> None:
    """Press New and answer its name prompt with *name*."""
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *_a, **_k: (name, True))
    )
    dialog._new_query()


# ----------------------------------------------------------------------
# Starting a query
# ----------------------------------------------------------------------
def test_a_new_query_is_named_but_not_yet_stored(dialog, repo, monkeypatch) -> None:
    """A query earns its row by running. save_query refuses empty SQL and the
    builder refuses a statement that fails validation, so a row written here
    would be one no chart could read."""
    _new_query(dialog, "daily", monkeypatch)

    assert dialog.current_name == "daily"
    assert repo.get_query("daily") is None
    assert dialog.sql == ""


def test_cancelling_the_name_prompt_changes_nothing(dialog, repo, monkeypatch) -> None:
    from PySide6.QtWidgets import QInputDialog

    dialog._editor.setPlainText("SELECT 1 AS a")
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *_a, **_k: ("", False))
    )
    dialog._new_query()

    assert dialog.sql == "SELECT 1 AS a"


def test_a_named_draft_is_stored_on_save(dialog, repo, monkeypatch) -> None:
    _new_query(dialog, "daily", monkeypatch)
    dialog._apply_command("select")

    assert dialog.save() is True
    saved = repo.get_query("daily")
    assert saved is not None and "batch_yields" in saved.sql


def test_the_suggested_name_skips_the_one_just_saved(dialog, repo, monkeypatch) -> None:
    _new_query(dialog, "Query 1", monkeypatch)
    dialog._apply_command("select")
    dialog.save()

    assert repo.next_query_name() == "Query 2"


# ----------------------------------------------------------------------
# Building one
# ----------------------------------------------------------------------
def test_select_writes_a_statement_over_the_chosen_field(dialog) -> None:
    dialog._apply_command("select")

    assert dialog.sql == 'SELECT "yield_pct"\nFROM "batch_yields"'
    assert dialog.validate() is True


def test_a_clause_replaces_itself_rather_than_stacking(dialog) -> None:
    """Two ORDER BY clauses is not a query SQLite will run, and the second
    press of a button is the most likely press there is."""
    dialog._apply_command("select")
    dialog._apply_command("order")
    dialog._apply_command("order")

    assert dialog.sql.count("ORDER BY") == 1
    assert dialog.validate() is True


def test_a_filter_is_inserted_before_the_clauses_that_must_follow_it(dialog) -> None:
    """WHERE comes before ORDER BY; appending it would produce a statement
    that parses as neither."""
    dialog._apply_command("select")
    dialog._apply_command("order")
    dialog._apply_command("filter")

    assert dialog.sql.index("WHERE") < dialog.sql.index("ORDER BY")


def test_a_summary_of_a_numeric_field_aggregates_it(dialog) -> None:
    dialog._apply_command("summary")

    assert "AVG(" in dialog.sql
    assert dialog.validate() is True


def test_a_summary_of_a_text_field_counts_it_instead(dialog) -> None:
    """AVG of a batch label is a number with no meaning; the shape of the
    summary has to follow the column's type."""
    dialog._field_combo.setCurrentText("batch")
    dialog._apply_command("summary")

    assert "GROUP BY" in dialog.sql
    assert "AVG(" not in dialog.sql
    assert dialog.validate() is True


def test_a_command_with_no_field_chosen_says_so_and_writes_nothing(dialog) -> None:
    # setCurrentIndex(-1), not setCurrentText(""): a non-editable combo
    # ignores text it has no item for, and would silently stay as it was.
    dialog._field_combo.setCurrentIndex(-1)
    dialog._apply_command("select")

    assert dialog.sql == ""
    assert dialog._status.text()


def test_the_field_list_follows_the_table(dialog) -> None:
    dialog._table_combo.setCurrentText("batch_notes")
    dialog._reload_fields()

    fields = [dialog._field_combo.itemText(i) for i in range(dialog._field_combo.count())]
    assert "note" in fields
    assert "yield_pct" not in fields


def test_running_a_built_query_returns_rows(dialog) -> None:
    dialog._apply_command("select")

    assert dialog.run() is True
    assert "row(s)" in dialog._status.text()
