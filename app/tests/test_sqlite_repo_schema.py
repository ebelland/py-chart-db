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