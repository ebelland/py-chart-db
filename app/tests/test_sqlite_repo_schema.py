"""Tests that the repository creates its descriptor schema and hides system tables."""
from __future__ import annotations

import sqlite3

from app.data.sqlite_repo import SqliteRepo


def test_repo_creates_descriptor_schema(tmp_db_path) -> None:
    repo = SqliteRepo(db_path=tmp_db_path)

    # Trigger schema creation
    repo.query_df("SELECT 1")

    db_path = repo.ensure_dhub_extension(tmp_db_path)
    with sqlite3.connect(str(db_path)) as con:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        names = {str(r[0]) for r in cur.fetchall()}

    # Verify expected columns (new persisted options_json)
    cols_fig = [r[1] for r in con.execute('PRAGMA table_info(__figure_descriptors__)').fetchall()]
    assert 'options_json' in cols_fig
    cols_ax = [r[1] for r in con.execute('PRAGMA table_info(__axis_descriptors__)').fetchall()]
    assert 'options_json' in cols_ax
    cols_series = [r[1] for r in con.execute('PRAGMA table_info(__series_descriptors__)').fetchall()]
    assert 'style_json' in cols_series

    assert '__figure_descriptors__' in names
    assert '__axis_descriptors__' in names
    assert '__series_descriptors__' in names
    assert '__import_links__' in names


def test_list_user_tables_excludes_metadata(tmp_db_path) -> None:
    repo = SqliteRepo(db_path=tmp_db_path)

    # Create a user table and a descriptor-like table directly via sqlite3.
    db_path = repo.ensure_dhub_extension(tmp_db_path)
    with sqlite3.connect(str(db_path)) as con:
        con.execute('CREATE TABLE IF NOT EXISTS data_table (x INTEGER)')
        con.execute('CREATE TABLE IF NOT EXISTS __foo_descriptors__ (x INTEGER)')

    names = set(repo.list_user_tables()["Table"].tolist())
    assert 'data_table' in names
    assert '__import_links__' not in names
    assert '__foo_descriptors__' not in names

def test_create_empty_writes_the_file_and_its_system_tables(tmp_db_path) -> None:
    """Startup uses this when the remembered database has gone: the point is
    that the failure surfaces here, not on the first query."""
    missing = tmp_db_path.parent / "gone_and_recreated.dhub"
    if missing.exists():
        missing.unlink()

    created = SqliteRepo.create_empty(missing)

    assert created.exists()
    with sqlite3.connect(str(created)) as con:
        names = {
            str(row[0])
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert '__figure_descriptors__' in names
    assert '__import_links__' in names


def test_create_empty_adds_the_extension_when_it_is_missing(tmp_db_path) -> None:
    created = SqliteRepo.create_empty(tmp_db_path.parent / "no_extension")

    assert created.suffix == ".dhub"
    assert created.exists()


def test_create_empty_leaves_no_connection_open(tmp_db_path) -> None:
    """The caller opens its own; two connections to one file is one too many."""
    path = tmp_db_path.parent / "closed_again.dhub"
    if path.exists():
        path.unlink()

    SqliteRepo.create_empty(path)

    repo = SqliteRepo(db_path=path)
    assert repo.list_table_names() == []
    repo.close()
