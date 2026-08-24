"""The query builder: starting a query, and building one from a field.

Two ideas, both easy to break in ways nothing else notices.

New query makes an empty draft and asks nothing. Naming it up front - which
is what it used to do - put two decisions about *storing* something in front
of the user before there was anything to store, including "replace the
existing query of that name?" about a statement that had not been written
yet, let alone run. The name is asked for once, at Save, after the statement
validates; a draft that never validates is never named and never written.

And the SQL buttons act on the table and field chosen beside them rather
than pasting a skeleton at the cursor: a clause command that appends instead
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
    # From nothing, every time. The test database is a file in the artifacts
    # directory and survives the run, so a leftover saved query from an
    # earlier one would be in the list this test counts - which is the kind of
    # failure that appears only in a full run and never on its own.
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

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


def _prompts(monkeypatch, answer: str, *, accepted: bool = True) -> list[str]:
    """Answer the save-time name prompt, recording what it was prefilled with."""
    from PySide6.QtWidgets import QInputDialog

    seen: list[str] = []

    def fake(_parent, _title, _label, text="", **_kwargs):
        seen.append(text)
        return answer, accepted

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(fake))
    return seen


def _save_as(dialog: QueryBuilderDialog, name: str, monkeypatch) -> bool:
    """Press Save and answer the name prompt with *name*."""
    _prompts(monkeypatch, name)
    return dialog.save()


def _silence_boxes(monkeypatch) -> list[str]:
    """Swallow the message boxes save() shows, recording their ids."""
    import app.dialogs.query_builder_dialog as module

    shown: list[str] = []
    monkeypatch.setattr(
        module, "show_message", lambda _p, message_id, **_k: shown.append(message_id)
    )
    monkeypatch.setattr(module, "ask", lambda _p, _message_id, **_k: True)
    return shown


# ----------------------------------------------------------------------
# Starting a query
# ----------------------------------------------------------------------
def test_new_query_makes_an_empty_draft_and_asks_nothing(
    dialog, repo, monkeypatch
) -> None:
    """The whole point of the rework: no name prompt, and no row."""
    seen = _prompts(monkeypatch, "never asked")
    before = {saved.name for saved in repo.list_queries()}

    dialog._new_query()

    assert seen == [], "New query must not ask for a name"
    assert dialog.sql == ""
    assert {saved.name for saved in repo.list_queries()} == before


def test_a_draft_carries_a_free_name_to_suggest_later(dialog, repo) -> None:
    """Named by the repository, so the suggestion cannot clash on arrival."""
    dialog._new_query()

    assert dialog.current_name
    assert repo.get_query(dialog.current_name) is None


def test_an_invalid_draft_is_never_named_and_never_stored(
    dialog, repo, monkeypatch
) -> None:
    """A query earns its row by running. Asking for a name and then refusing
    to save under it is the flow this replaced."""
    shown = _silence_boxes(monkeypatch)
    seen = _prompts(monkeypatch, "daily")
    dialog._new_query()
    dialog._editor.setPlainText("SELECT * FROM no_such_table")

    assert dialog.save() is False
    assert seen == [], "nothing to name: the statement does not run"
    assert shown == ["query.invalid"]
    assert repo.get_query("daily") is None


def test_a_valid_draft_is_named_once_and_stored(dialog, repo, monkeypatch) -> None:
    _silence_boxes(monkeypatch)
    dialog._new_query()
    dialog._apply_command("select")

    assert _save_as(dialog, "daily", monkeypatch) is True
    saved = repo.get_query("daily")
    assert saved is not None and "batch_yields" in saved.sql


def test_the_prompt_offers_the_draft_name_it_already_has(
    dialog, repo, monkeypatch
) -> None:
    _silence_boxes(monkeypatch)
    dialog._new_query()
    suggested = dialog.current_name
    dialog._apply_command("select")
    seen = _prompts(monkeypatch, "daily")

    dialog.save()

    assert seen == [suggested]


def test_saving_again_does_not_ask_a_second_time(dialog, repo, monkeypatch) -> None:
    """Apply is pressed repeatedly while a query is being worked on."""
    _silence_boxes(monkeypatch)
    dialog._new_query()
    dialog._apply_command("select")
    _save_as(dialog, "daily", monkeypatch)

    seen = _prompts(monkeypatch, "should not be used")
    dialog._editor.setPlainText("SELECT batch FROM batch_yields")

    assert dialog.save() is True
    assert seen == []
    saved = repo.get_query("daily")
    assert saved is not None and saved.sql == "SELECT batch FROM batch_yields"


def test_cancelling_the_name_prompt_stores_nothing_and_keeps_the_sql(
    dialog, repo, monkeypatch
) -> None:
    _silence_boxes(monkeypatch)
    dialog._new_query()
    dialog._editor.setPlainText("SELECT 1 AS a")
    _prompts(monkeypatch, "", accepted=False)

    assert dialog.save() is False
    assert dialog.sql == "SELECT 1 AS a"
    assert repo.list_queries() == []


def test_a_name_a_table_already_owns_is_refused_at_save(
    dialog, repo, monkeypatch
) -> None:
    """A saved query named after a table can never be selected back: the
    table always wins when the name is resolved."""
    shown = _silence_boxes(monkeypatch)
    dialog._new_query()
    dialog._apply_command("select")

    assert _save_as(dialog, "batch_yields", monkeypatch) is False
    assert shown == ["query.name_is_a_table"]
    assert repo.get_query("batch_yields") is None


def test_the_suggested_name_skips_the_one_just_saved(dialog, repo, monkeypatch) -> None:
    _silence_boxes(monkeypatch)
    dialog._new_query()
    dialog._apply_command("select")
    _save_as(dialog, "Query 1", monkeypatch)

    assert repo.next_query_name() == "Query 2"


def test_loading_a_saved_query_never_re_asks_for_its_name(
    dialog, repo, monkeypatch
) -> None:
    """Its name is already the user's; Apply should just write."""
    _silence_boxes(monkeypatch)
    repo.save_query("existing", "SELECT batch FROM batch_yields")
    dialog.load_query("existing")

    seen = _prompts(monkeypatch, "should not be used")
    dialog._editor.setPlainText("SELECT yield_pct FROM batch_yields")

    assert dialog.save() is True
    assert seen == []
    saved = repo.get_query("existing")
    assert saved is not None and saved.sql == "SELECT yield_pct FROM batch_yields"


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
