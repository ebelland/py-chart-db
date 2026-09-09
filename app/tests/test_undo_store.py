"""Undo, as table snapshots in a database of its own.

The alternative was a copy of the whole .dhub before every change, and it
is worse twice over: it costs the size of the project to take back a
change to one table, and it reverts everything else that happened since,
which is not what undo means. What is copied here is the table an action
is about to change - either kind, the imported data or the chart settings
in the ``__…__`` descriptor tables - into ``<project>.undo.db``.

The properties worth pinning are the ones that are silently wrong if the
implementation drifts: that a restored table has its *own* schema back and
not an approximation of it, that a table the action created is dropped
rather than left behind, and that one entry covers every table one action
touched.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.data.undo_store import MAX_UNDO_ENTRIES, UndoStore


@pytest.fixture
def connection(tmp_path: Path) -> sqlite3.Connection:
    """A project with one data table and one descriptor table."""
    con = sqlite3.connect(str(tmp_path / "project.dhub"), isolation_level=None)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, v REAL NOT NULL)")
    con.execute("CREATE INDEX idx_data_v ON data(v)")
    con.executemany("INSERT INTO data (id, v) VALUES (?, ?)", [(1, 1.5), (2, 2.5)])
    con.execute(
        "CREATE TABLE __series_descriptors__ ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)"
    )
    con.execute("INSERT INTO __series_descriptors__ (name) VALUES ('a')")
    yield con
    con.close()


@pytest.fixture
def store(tmp_path: Path) -> UndoStore:
    return UndoStore(tmp_path / "project.dhub")


def _tables(connection: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]


# ----------------------------------------------------------------------
# Taking a change back
# ----------------------------------------------------------------------
def test_a_dropped_table_comes_back_with_its_rows(connection, store) -> None:
    store.snapshot(connection, ["data"], label="Delete table 'data'")
    connection.execute("DROP TABLE data")

    store.undo(connection)

    assert connection.execute("SELECT * FROM data ORDER BY id").fetchall() == [
        (1, 1.5),
        (2, 2.5),
    ]


def test_it_comes_back_with_its_own_schema(connection, store) -> None:
    """CREATE TABLE ... AS SELECT would lose the primary key, the NOT NULL
    and the index - fine for holding rows, useless for putting a descriptor
    table back."""
    store.snapshot(connection, ["data"], label="Delete table 'data'")
    connection.execute("DROP TABLE data")

    store.undo(connection)

    ddl = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'data'"
    ).fetchone()[0]
    assert "INTEGER PRIMARY KEY" in ddl
    assert "NOT NULL" in ddl
    assert connection.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'idx_data_v'"
    ).fetchone() is not None


def test_a_deleted_column_comes_back(connection, store) -> None:
    store.snapshot(connection, ["data"], label="Delete column 'v'")
    # SQLite refuses to drop a column an index covers, so a caller has to
    # remove the index first - which is another reason the snapshot keeps
    # the index DDL and puts it back.
    connection.execute("DROP INDEX idx_data_v")
    connection.execute("ALTER TABLE data DROP COLUMN v")
    assert "v" not in [row[1] for row in connection.execute("PRAGMA table_info(data)")]

    store.undo(connection)

    assert [row[1] for row in connection.execute("PRAGMA table_info(data)")] == [
        "id",
        "v",
    ]


def test_a_chart_settings_table_is_restored_like_any_other(
    connection, store
) -> None:
    """A table here is the imported data or the descriptors that draw it,
    and undo cannot tell them apart - they are rows in the same file."""
    store.snapshot(connection, ["__series_descriptors__"], label="Delete series")
    connection.execute("DELETE FROM __series_descriptors__")

    store.undo(connection)

    assert connection.execute(
        "SELECT name FROM __series_descriptors__"
    ).fetchall() == [("a",)]


def test_a_table_the_action_created_is_dropped_again(connection, store) -> None:
    """Undoing an operation that *wrote* a result means removing it. Without
    this the commonest change of all would silently survive its own undo."""
    store.snapshot(connection, ["result"], label="Apply operation")
    connection.execute("CREATE TABLE result (x REAL)")
    connection.execute("INSERT INTO result VALUES (1.0)")

    store.undo(connection)

    assert "result" not in _tables(connection)


def test_one_entry_covers_every_table_one_action_touched(
    connection, store
) -> None:
    """Deleting a data table also deletes the descriptors that drew it, and
    both have to come back together or the project is left inconsistent."""
    store.snapshot(
        connection, ["data", "__series_descriptors__"], label="Delete table 'data'"
    )
    connection.execute("DROP TABLE data")
    connection.execute("DELETE FROM __series_descriptors__")

    store.undo(connection)

    assert connection.execute("SELECT count(*) FROM data").fetchone()[0] == 2
    assert (
        connection.execute("SELECT count(*) FROM __series_descriptors__").fetchone()[0]
        == 1
    )


# ----------------------------------------------------------------------
# The stack
# ----------------------------------------------------------------------
def test_the_most_recent_change_is_undone_first(connection, store) -> None:
    store.snapshot(connection, ["data"], label="First")
    connection.execute("DELETE FROM data WHERE id = 1")
    store.snapshot(connection, ["data"], label="Second")
    connection.execute("DELETE FROM data WHERE id = 2")

    assert [entry.label for entry in store.entries(connection)] == ["Second", "First"]

    store.undo(connection)
    assert connection.execute("SELECT count(*) FROM data").fetchone()[0] == 1

    store.undo(connection)
    assert connection.execute("SELECT count(*) FROM data").fetchone()[0] == 2


def test_an_undone_entry_is_gone_from_the_stack(connection, store) -> None:
    store.snapshot(connection, ["data"], label="Delete")
    connection.execute("DROP TABLE data")

    store.undo(connection)

    assert store.entries(connection) == []


def test_undoing_nothing_is_not_an_error(connection, store) -> None:
    assert store.undo(connection) is None


def test_the_stack_is_capped(connection, store) -> None:
    for index in range(MAX_UNDO_ENTRIES + 3):
        store.snapshot(connection, ["data"], label=f"Change {index}")

    entries = store.entries(connection)
    assert len(entries) == MAX_UNDO_ENTRIES
    assert entries[0].label == f"Change {MAX_UNDO_ENTRIES + 2}"


def test_a_pruned_entry_takes_its_snapshot_with_it(connection, store) -> None:
    """Otherwise the undo file grows by a copy of the table every time and
    never shrinks."""
    for index in range(MAX_UNDO_ENTRIES + 2):
        store.snapshot(connection, ["data"], label=f"Change {index}")

    undo = sqlite3.connect(str(store.path))
    try:
        snapshots = [
            row[0]
            for row in undo.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'snap_%'"
            )
        ]
    finally:
        undo.close()

    assert len(snapshots) == MAX_UNDO_ENTRIES


def test_clearing_forgets_everything(connection, store) -> None:
    store.snapshot(connection, ["data"], label="Change")

    store.clear(connection)

    assert store.entries(connection) == []


def test_an_entry_describes_itself_for_a_menu(connection, store) -> None:
    store.snapshot(connection, ["data"], label="Delete column 'v'")

    entry = store.entries(connection)[0]

    assert entry.describe() == "Delete column 'v' (data)"
    assert entry.created  # an ISO timestamp, for a history list later


# ----------------------------------------------------------------------
# Not costing the user their action
# ----------------------------------------------------------------------
def test_a_snapshot_inside_a_transaction_is_refused(connection, store) -> None:
    """ATTACH is not allowed mid-transaction, so a caller has to take the
    snapshot before opening one - and is told, rather than finding out from
    an empty undo stack later."""
    connection.execute("BEGIN")
    try:
        assert store.snapshot(connection, ["data"], label="Change") is None
    finally:
        connection.execute("ROLLBACK")


def test_snapshotting_nothing_records_nothing(connection, store) -> None:
    assert store.snapshot(connection, [], label="Change") is None
    assert store.entries(connection) == []


def test_the_undo_file_sits_beside_the_project(tmp_path: Path) -> None:
    store = UndoStore(tmp_path / "project.dhub")

    assert store.path.parent == tmp_path
    assert store.path.name == "project.dhub.undo.db"


# ----------------------------------------------------------------------
# Through the repository, where the call sites are
# ----------------------------------------------------------------------
@pytest.fixture
def repo_with_table(tmp_db_path: Path) -> SqliteRepo:
    built = SqliteRepo(db_path=tmp_db_path)
    built.import_dataframe(
        pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}),
        table_name="t1",
        normalize_columns=False,
    )
    yield built
    built.close()
    built.undo_store.discard_file()


def test_deleting_a_column_can_be_taken_back(repo_with_table: SqliteRepo) -> None:
    repo_with_table.delete_table_column("t1", "b")
    assert "b" not in repo_with_table.query_df("SELECT * FROM t1").columns

    entry = repo_with_table.undo_last()

    assert entry is not None
    assert "b" in repo_with_table.query_df("SELECT * FROM t1").columns


def test_deleting_a_table_can_be_taken_back(repo_with_table: SqliteRepo) -> None:
    repo_with_table.delete_table("t1")
    assert "t1" not in list(repo_with_table.list_user_tables().get("Table", []))

    entry = repo_with_table.undo_last()

    assert entry is not None
    assert entry.label == "Delete table 't1'"
    assert len(repo_with_table.query_df("SELECT * FROM t1")) == 3


def test_the_entry_names_the_change_for_the_menu(
    repo_with_table: SqliteRepo,
) -> None:
    repo_with_table.delete_table_column("t1", "b")

    entries = repo_with_table.undo_entries()

    assert entries[0].label == "Delete column 'b' from 't1'"


def test_undoing_drops_the_series_cache(repo_with_table: SqliteRepo) -> None:
    """The rows behind a series may have just been replaced wholesale, and a
    cache that survived that would serve the state the user has undone."""
    repo_with_table.series_df("SELECT a, b FROM t1")
    repo_with_table.delete_table_column("t1", "b")
    repo_with_table.undo_last()

    frame = repo_with_table.series_df("SELECT a, b FROM t1")
    assert list(frame.columns) == ["a", "b"]
