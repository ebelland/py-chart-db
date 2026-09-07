"""Tests for app.utils.data_sources: the reading logic behind both the live
import dialog and a saved link's "Update link".

No real PostgreSQL or MySQL server is available here, so those two engines
are exercised against a stubbed DBAPI connection - enough to check the SQL
this module builds and the connection lifecycle it manages, which is what is
actually this module's responsibility; pandas' own SQL execution is pandas'
own test suite's job.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from app.utils.data_sources import (
    DatabaseConnection,
    MissingPasswordError,
    _extension_for_web_source,
    filename_from_url,
    is_valid_web_url,
    list_mysql_tables,
    list_postgres_tables,
    list_sqlite_tables,
    read_from_link_source,
    read_mysql_table,
    read_postgres_table,
    read_sqlite_table,
)


# ----------------------------------------------------------------------
# DatabaseConnection
# ----------------------------------------------------------------------
def test_a_sqlite_connection_links_by_path_only() -> None:
    conn = DatabaseConnection(kind="sqlite", path="/tmp/other.dhub")

    assert conn.to_link_settings() == {"kind": "sqlite", "path": "/tmp/other.dhub"}


def test_a_server_connections_link_settings_carry_no_password() -> None:
    conn = DatabaseConnection(
        kind="postgres",
        host="db.example.com",
        port=5432,
        database="analytics",
        username="reader",
        password="hunter2",
    )

    settings = conn.to_link_settings()

    assert "password" not in settings
    assert settings == {
        "kind": "postgres",
        "host": "db.example.com",
        "port": 5432,
        "database": "analytics",
        "username": "reader",
    }


def test_a_connection_rebuilt_from_link_settings_takes_the_fresh_password() -> None:
    settings = {
        "kind": "mysql",
        "host": "db.example.com",
        "port": 3306,
        "database": "shop",
        "username": "reader",
    }

    conn = DatabaseConnection.from_link_settings(settings, password="typed-now")

    assert conn.password == "typed-now"
    assert conn.host == "db.example.com"
    assert conn.port == 3306


# ----------------------------------------------------------------------
# Another SQLite database
# ----------------------------------------------------------------------
@pytest.fixture
def other_db(tmp_path: Path) -> Path:
    path = tmp_path / "other.dhub"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE readings (t REAL, v REAL)")
        conn.execute("INSERT INTO readings VALUES (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)")
        conn.execute("CREATE TABLE __figure_descriptors__ (id INTEGER)")
        conn.commit()
    finally:
        conn.close()
    return path


def test_list_sqlite_tables_excludes_this_applications_own_bookkeeping(
    other_db: Path,
) -> None:
    assert list_sqlite_tables(str(other_db)) == ["readings"]


def test_read_sqlite_table_applies_skip_options(other_db: Path) -> None:
    df = read_sqlite_table(str(other_db), "readings", skiprows=1, skipfooter=1)

    assert list(df["t"]) == [2.0]


# ----------------------------------------------------------------------
# The web
# ----------------------------------------------------------------------
def test_web_urls_reject_every_scheme_but_http_and_https() -> None:
    assert is_valid_web_url("https://example.com/data.csv")
    assert is_valid_web_url("http://example.com/data.csv")
    assert not is_valid_web_url("file:///etc/passwd")
    assert not is_valid_web_url("ftp://example.com/data.csv")
    assert not is_valid_web_url("")


def test_filename_from_url_strips_the_query_string() -> None:
    assert filename_from_url("https://example.com/path/data.csv?x=1") == "data.csv"
    assert filename_from_url("https://example.com/") == "web_data"


def test_extension_falls_back_to_content_type() -> None:
    assert _extension_for_web_source("https://example.com/data.csv", None) == ".csv"
    assert (
        _extension_for_web_source("https://api.example.com/export", "application/json")
        == ".json"
    )
    assert _extension_for_web_source("https://api.example.com/export", None) == ".csv"


# ----------------------------------------------------------------------
# A stubbed server database, for PostgreSQL and MySQL
# ----------------------------------------------------------------------
class _FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
        self.last_sql: str = ""

    def execute(self, sql: str) -> None:
        self.last_sql = sql

    def fetchall(self) -> list[tuple]:
        return self._rows

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _FakeConnection:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._rows)

    def close(self) -> None:
        self.closed = True


def _postgres_connection(monkeypatch: pytest.MonkeyPatch, rows: list[tuple]) -> _FakeConnection:
    conn = _FakeConnection(rows)
    calls: dict[str, object] = {}

    def fake_connect(**kwargs: object) -> _FakeConnection:
        calls["kwargs"] = kwargs
        return conn

    monkeypatch.setattr("pg8000.dbapi.connect", fake_connect)
    conn.calls = calls  # type: ignore[attr-defined]
    return conn


def _mysql_connection(monkeypatch: pytest.MonkeyPatch, rows: list[tuple]) -> _FakeConnection:
    conn = _FakeConnection(rows)
    calls: dict[str, object] = {}

    def fake_connect(**kwargs: object) -> _FakeConnection:
        calls["kwargs"] = kwargs
        return conn

    monkeypatch.setattr("pymysql.connect", fake_connect)
    conn.calls = calls  # type: ignore[attr-defined]
    return conn


def test_list_postgres_tables_queries_information_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _postgres_connection(monkeypatch, [("orders",), ("customers",)])
    dbconn = DatabaseConnection(
        kind="postgres", host="h", port=5432, database="d", username="u", password="p"
    )

    tables = list_postgres_tables(dbconn)

    assert tables == ["orders", "customers"]
    assert conn.calls["kwargs"] == {  # type: ignore[attr-defined]
        "host": "h", "port": 5432, "database": "d", "user": "u", "password": "p",
    }
    assert conn.closed is True


def test_read_postgres_table_selects_the_named_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.utils.data_sources as module

    conn = _postgres_connection(monkeypatch, [])
    captured: dict[str, object] = {}

    def fake_read_sql_query(sql: str, connection: object) -> pd.DataFrame:
        captured["sql"] = sql
        captured["connection"] = connection
        return pd.DataFrame({"id": [1, 2]})

    monkeypatch.setattr(module.pd, "read_sql_query", fake_read_sql_query)
    dbconn = DatabaseConnection(
        kind="postgres", host="h", port=5432, database="d", username="u", password="p"
    )

    df = read_postgres_table(dbconn, "orders")

    assert captured["sql"] == 'SELECT * FROM "orders"'
    assert captured["connection"] is conn
    assert list(df["id"]) == [1, 2]
    assert conn.closed is True


def test_list_mysql_tables_queries_show_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _mysql_connection(monkeypatch, [("orders",)])
    dbconn = DatabaseConnection(
        kind="mysql", host="h", port=3306, database="d", username="u", password="p"
    )

    tables = list_mysql_tables(dbconn)

    assert tables == ["orders"]
    assert conn.closed is True


def test_read_mysql_table_uses_backtick_quoting(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.utils.data_sources as module

    conn = _mysql_connection(monkeypatch, [])
    captured: dict[str, object] = {}

    def fake_read_sql_query(sql: str, connection: object) -> pd.DataFrame:
        captured["sql"] = sql
        return pd.DataFrame({"id": [1]})

    monkeypatch.setattr(module.pd, "read_sql_query", fake_read_sql_query)
    dbconn = DatabaseConnection(
        kind="mysql", host="h", port=3306, database="d", username="u", password="p"
    )

    read_mysql_table(dbconn, "orders")

    assert captured["sql"] == "SELECT * FROM `orders`"
    assert conn.closed is True


# ----------------------------------------------------------------------
# read_from_link_source: the one dispatcher both the dialog and the link
# refresh call into
# ----------------------------------------------------------------------
def test_dispatches_a_file_source(tmp_path: Path) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    df = read_from_link_source(
        {"kind": "file", "path": str(csv_path), "sheet": None},
        {"skiprows": 0, "skip_last": 0, "header": True},
    )

    assert list(df.columns) == ["a", "b"]


def test_dispatches_a_sqlite_source(other_db: Path) -> None:
    df = read_from_link_source(
        {"kind": "sqlite", "path": str(other_db), "table": "readings"},
        {},
    )

    assert list(df.columns) == ["t", "v"]


def test_a_server_source_with_no_password_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _postgres_connection(monkeypatch, [])

    with pytest.raises(MissingPasswordError) as excinfo:
        read_from_link_source(
            {
                "kind": "postgres", "host": "h", "port": 5432,
                "database": "d", "username": "u", "table": "orders",
            },
            {},
        )

    assert excinfo.value.kind == "postgres"


def test_a_server_source_with_a_password_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.utils.data_sources as module

    _postgres_connection(monkeypatch, [])
    monkeypatch.setattr(
        module.pd, "read_sql_query", lambda sql, connection: pd.DataFrame({"id": [1]})
    )

    df = read_from_link_source(
        {
            "kind": "postgres", "host": "h", "port": 5432,
            "database": "d", "username": "u", "table": "orders",
        },
        {},
        password="hunter2",
    )

    assert list(df["id"]) == [1]


def test_an_unknown_source_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown source kind"):
        read_from_link_source({"kind": "ftp"}, {})
