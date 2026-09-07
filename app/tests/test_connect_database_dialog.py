"""The connect-to-database dialog: engine switching, connecting, picking a table.

No real PostgreSQL or MySQL server is available to test against here, so the
engine readers themselves (list_postgres_tables, read_postgres_table, ...)
are exercised in test_data_sources.py against a stubbed DBAPI connection.
This file covers the dialog's own logic: which fields show for which engine,
what a successful/failed connect does to the table list, and what gets
handed back through connection/table.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.dialogs.connect_database_dialog import (
    ENGINE_MYSQL,
    ENGINE_POSTGRES,
    ENGINE_SQLITE,
    ConnectDatabaseDialog,
)


@pytest.fixture
def dialog(qapp) -> ConnectDatabaseDialog:
    return ConnectDatabaseDialog()


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Path:
    path = tmp_path / "other.dhub"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE readings (t REAL, v REAL)")
        conn.execute("INSERT INTO readings VALUES (1.0, 2.0)")
        conn.commit()
    finally:
        conn.close()
    return path


def _select_engine(dialog: ConnectDatabaseDialog, kind: str) -> None:
    index = dialog._engine.findData(kind)
    assert index >= 0, f"no engine choice for {kind!r}"
    dialog._engine.setCurrentIndex(index)


# ----------------------------------------------------------------------
# Engine switching
# ----------------------------------------------------------------------
def test_sqlite_is_the_default_engine(dialog: ConnectDatabaseDialog) -> None:
    assert dialog._engine.currentData() == ENGINE_SQLITE
    assert not dialog._sqlite_row.isHidden()
    assert dialog._host.isHidden()


def test_switching_to_a_server_engine_swaps_the_fields(
    dialog: ConnectDatabaseDialog,
) -> None:
    _select_engine(dialog, ENGINE_POSTGRES)

    assert dialog._sqlite_row.isHidden()
    assert not dialog._host.isHidden()
    assert not dialog._port.isHidden()
    assert not dialog._database.isHidden()
    assert not dialog._username.isHidden()
    assert not dialog._password.isHidden()


def test_switching_engine_fills_in_the_default_port(
    dialog: ConnectDatabaseDialog,
) -> None:
    _select_engine(dialog, ENGINE_POSTGRES)
    assert dialog._port.value() == 5432

    _select_engine(dialog, ENGINE_MYSQL)
    assert dialog._port.value() == 3306


def test_switching_engine_clears_the_table_list(dialog: ConnectDatabaseDialog) -> None:
    dialog._tables.addItems(["leftover"])

    _select_engine(dialog, ENGINE_POSTGRES)

    assert dialog._tables.count() == 0


# ----------------------------------------------------------------------
# Connecting - SQLite, the one engine testable end to end without a server
# ----------------------------------------------------------------------
def test_connecting_to_sqlite_lists_its_tables(
    dialog: ConnectDatabaseDialog, sqlite_db: Path
) -> None:
    dialog._sqlite_path.setText(str(sqlite_db))

    dialog._on_connect()

    assert [dialog._tables.item(i).text() for i in range(dialog._tables.count())] == [
        "readings"
    ]


def test_connecting_with_no_file_is_rejected(
    dialog: ConnectDatabaseDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.dialogs.connect_database_dialog as module

    shown: list[str] = []
    monkeypatch.setattr(
        module, "show_message", lambda _p, message_id, **_k: shown.append(message_id)
    )

    dialog._on_connect()

    assert shown == ["database.connection_missing_file"]
    assert dialog._tables.count() == 0


def test_connecting_to_a_bad_path_reports_the_failure(
    dialog: ConnectDatabaseDialog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.dialogs.connect_database_dialog as module

    # A file that exists but is not a database: sqlite3 fails on the first
    # real query against it, not at connect() time.
    not_a_db = tmp_path / "not_a_database.dhub"
    not_a_db.write_text("hello", encoding="utf-8")
    dialog._sqlite_path.setText(str(not_a_db))

    shown: list[str] = []
    monkeypatch.setattr(
        module, "show_message", lambda _p, message_id, **_k: shown.append(message_id)
    )

    dialog._on_connect()

    assert shown == ["import.database_failed"]


# ----------------------------------------------------------------------
# Confirming
# ----------------------------------------------------------------------
def test_confirming_with_no_table_selected_is_rejected(
    dialog: ConnectDatabaseDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.dialogs.connect_database_dialog as module

    shown: list[str] = []
    monkeypatch.setattr(
        module, "show_message", lambda _p, message_id, **_k: shown.append(message_id)
    )

    dialog._confirm()

    assert shown == ["import.database_no_table_selected"]
    assert dialog.connection is None
    assert dialog.table is None


def test_confirming_after_connecting_returns_the_chosen_table(
    dialog: ConnectDatabaseDialog, sqlite_db: Path
) -> None:
    dialog._sqlite_path.setText(str(sqlite_db))
    dialog._on_connect()

    dialog._confirm()

    assert dialog.connection is not None
    assert dialog.connection.kind == "sqlite"
    assert dialog.connection.path == str(sqlite_db)
    assert dialog.table == "readings"
    assert dialog.result() == ConnectDatabaseDialog.DialogCode.Accepted


def test_a_server_connection_carries_no_stray_sqlite_path(
    dialog: ConnectDatabaseDialog,
) -> None:
    _select_engine(dialog, ENGINE_POSTGRES)
    dialog._host.setText("db.example.com")
    dialog._database.setText("analytics")
    dialog._username.setText("reader")
    dialog._password.setText("hunter2")
    dialog._tables.addItems(["orders"])
    dialog._tables.setCurrentRow(0)

    dialog._confirm()

    assert dialog.connection is not None
    assert dialog.connection.path == ""
    assert dialog.connection.host == "db.example.com"
    assert dialog.connection.password == "hunter2"
    assert dialog.connection.to_link_settings() == {
        "kind": "postgres",
        "host": "db.example.com",
        "port": 5432,
        "database": "analytics",
        "username": "reader",
    }
