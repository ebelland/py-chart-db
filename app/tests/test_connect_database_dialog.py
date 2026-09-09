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


@pytest.fixture(autouse=True)
def remembered(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Give the dialog a settings section of its own, not the developer's.

    ``get_connect_database_config`` reads the real user.json, which on the
    machine running these tests holds whatever the last connection actually
    was - so without this, "the default engine is SQLite" would pass or fail
    depending on what its owner last connected to.
    """
    import app.dialogs.connect_database_dialog as module

    stored: dict = {}
    monkeypatch.setattr(module, "get_connect_database_config", lambda: dict(stored))
    monkeypatch.setattr(
        module,
        "set_connect_database_config",
        lambda payload: stored.update(payload),
    )
    return stored


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
    dialog._database.setCurrentText("analytics")
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


# ----------------------------------------------------------------------
# Listing the databases on a server
# ----------------------------------------------------------------------
def _stub_catalogue(
    monkeypatch: pytest.MonkeyPatch, databases: list[str], tables: list[str]
) -> dict:
    """Answer both server queries without a server, recording the arguments."""
    import app.dialogs.connect_database_dialog as module

    seen: dict = {"databases_for": [], "tables_for": []}

    def list_databases(conn) -> list[str]:
        seen["databases_for"].append(conn.host)
        return databases

    def list_tables(conn) -> list[str]:
        seen["tables_for"].append(conn.database)
        return tables

    monkeypatch.setattr(
        module, "SERVER_DATABASE_CATALOGUES", {"postgres": list_databases}
    )
    monkeypatch.setattr(module, "list_postgres_tables", list_tables)
    return seen


def _database_choices(dialog: ConnectDatabaseDialog) -> list[str]:
    return [dialog._database.itemText(i) for i in range(dialog._database.count())]


def test_connecting_to_a_server_lists_its_databases(
    dialog: ConnectDatabaseDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name used to have to be typed from memory before anything at all
    could be listed, and a typo answered with a connection error."""
    seen = _stub_catalogue(monkeypatch, ["analytics", "staging"], ["orders"])
    _select_engine(dialog, ENGINE_POSTGRES)
    dialog._host.setText("db.example.com")

    dialog._on_connect()

    assert _database_choices(dialog) == ["analytics", "staging"]
    assert seen["databases_for"] == ["db.example.com"]


def test_the_typed_database_survives_the_listing(
    dialog: ConnectDatabaseDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_catalogue(monkeypatch, ["analytics", "staging"], ["orders"])
    _select_engine(dialog, ENGINE_POSTGRES)
    dialog._database.setCurrentText("staging")

    dialog._on_connect()

    assert dialog._database.currentText() == "staging"


def test_connecting_lists_the_tables_of_the_chosen_database(
    dialog: ConnectDatabaseDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _stub_catalogue(monkeypatch, ["analytics", "staging"], ["orders", "items"])
    _select_engine(dialog, ENGINE_POSTGRES)

    dialog._on_connect()

    # No database was typed, so the first one listed is the one opened.
    assert seen["tables_for"] == ["analytics"]
    assert [dialog._tables.item(i).text() for i in range(dialog._tables.count())] == [
        "orders",
        "items",
    ]


def test_picking_another_database_relists_the_tables(
    dialog: ConnectDatabaseDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _stub_catalogue(monkeypatch, ["analytics", "staging"], ["orders"])
    _select_engine(dialog, ENGINE_POSTGRES)
    dialog._on_connect()

    dialog._database.setCurrentIndex(1)
    dialog._on_database_chosen()

    assert seen["tables_for"] == ["analytics", "staging"]


def test_a_server_with_no_databases_says_so_without_giving_up(
    dialog: ConnectDatabaseDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A login with rights to one database and no catalogue access: the name
    it typed still has to be listable."""
    import app.dialogs.connect_database_dialog as module

    seen = _stub_catalogue(monkeypatch, [], ["orders"])
    shown: list[str] = []
    monkeypatch.setattr(
        module, "show_message", lambda _p, message_id, **_k: shown.append(message_id)
    )
    _select_engine(dialog, ENGINE_POSTGRES)
    dialog._database.setCurrentText("theirs")

    dialog._on_connect()

    assert shown == ["import.database_no_databases"]
    assert seen["tables_for"] == ["theirs"]


def test_a_failed_catalogue_query_stops_before_the_tables(
    dialog: ConnectDatabaseDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.dialogs.connect_database_dialog as module

    def explode(_conn) -> list[str]:
        raise ConnectionRefusedError("nope")

    tables_called: list[str] = []
    monkeypatch.setattr(module, "SERVER_DATABASE_CATALOGUES", {"postgres": explode})
    monkeypatch.setattr(
        module, "list_postgres_tables", lambda conn: tables_called.append(conn.database)
    )
    shown: list[str] = []
    monkeypatch.setattr(
        module, "show_message", lambda _p, message_id, **_k: shown.append(message_id)
    )
    _select_engine(dialog, ENGINE_POSTGRES)

    dialog._on_connect()

    assert shown == ["import.database_failed"]
    assert tables_called == []


def test_switching_engine_drops_the_other_engines_databases(
    dialog: ConnectDatabaseDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_catalogue(monkeypatch, ["analytics"], ["orders"])
    _select_engine(dialog, ENGINE_POSTGRES)
    dialog._on_connect()

    _select_engine(dialog, ENGINE_MYSQL)

    assert _database_choices(dialog) == []


# ----------------------------------------------------------------------
# Remembering the last selection
# ----------------------------------------------------------------------
def test_confirming_remembers_the_connection_without_the_password(
    dialog: ConnectDatabaseDialog, remembered: dict
) -> None:
    _select_engine(dialog, ENGINE_POSTGRES)
    dialog._host.setText("db.example.com")
    dialog._database.setCurrentText("analytics")
    dialog._username.setText("reader")
    dialog._password.setText("hunter2")
    dialog._tables.addItems(["orders"])
    dialog._tables.setCurrentRow(0)

    dialog._confirm()

    assert remembered["engine"] == "postgres"
    assert remembered["host"] == "db.example.com"
    assert remembered["port"] == 5432
    assert remembered["database"] == "analytics"
    assert remembered["username"] == "reader"
    assert remembered["table"] == "orders"
    assert "hunter2" not in str(remembered)
    assert "password" not in remembered


def test_the_next_dialog_opens_on_the_remembered_connection(
    qapp, remembered: dict
) -> None:
    remembered.update(
        {
            "engine": "postgres",
            "path": "",
            "host": "db.example.com",
            "port": 6543,
            "database": "analytics",
            "username": "reader",
            "table": "orders",
        }
    )

    reopened = ConnectDatabaseDialog()

    assert reopened._engine.currentData() == ENGINE_POSTGRES
    assert reopened._host.text() == "db.example.com"
    assert reopened._port.value() == 6543  # not overwritten by the engine default
    assert reopened._database.currentText() == "analytics"
    assert reopened._username.text() == "reader"
    assert reopened._password.text() == "", "a password is never stored"


def test_the_remembered_table_is_reselected_when_it_is_still_there(
    qapp, remembered: dict, sqlite_db: Path
) -> None:
    """Not row 0: the point of remembering is to land back on the table the
    user was working with, which is rarely the alphabetically first."""
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(str(sqlite_db))
    try:
        conn.execute("CREATE TABLE alphabetically_first (x REAL)")
        conn.commit()
    finally:
        conn.close()

    remembered.update({"engine": "sqlite", "path": str(sqlite_db), "table": "readings"})
    reopened = ConnectDatabaseDialog()

    reopened._on_connect()

    assert reopened._tables.currentItem().text() == "readings"


def test_a_remembered_table_that_is_gone_falls_back_to_the_first(
    qapp, remembered: dict, sqlite_db: Path
) -> None:
    remembered.update({"engine": "sqlite", "path": str(sqlite_db), "table": "dropped"})
    reopened = ConnectDatabaseDialog()

    reopened._on_connect()

    assert reopened._tables.currentItem().text() == "readings"
