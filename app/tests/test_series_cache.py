"""Tests for the series DataFrame cache in SqliteRepo.

The cache is the one optimisation that can return *wrong* data if invalidation
misses a write, so every write shape gets its own test: same-connection DML,
DDL, and a commit from a second connection.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from app.data.sqlite_repo import SqliteRepo

SQL = "SELECT x, y FROM t_cache ORDER BY x"


def _make_repo(tmp_db_path: Path) -> SqliteRepo:
    """Return a repo with a small populated table to query."""
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("DROP TABLE IF EXISTS t_cache")
    repo.query_df("CREATE TABLE t_cache (x INTEGER, y REAL)")
    repo.query_df("INSERT INTO t_cache (x, y) VALUES (1, 1.0), (2, 2.0)")
    return repo


def test_second_read_is_served_from_cache(tmp_db_path: Path) -> None:
    repo = _make_repo(tmp_db_path)

    first = repo.series_df(SQL)
    hits_before = repo.series_cache_stats["hits"]
    second = repo.series_df(SQL)

    pd.testing.assert_frame_equal(first, second)
    assert repo.series_cache_stats["hits"] == hits_before + 1
    repo.close()


def test_returned_frame_is_a_shallow_copy(tmp_db_path: Path) -> None:
    """Adding a column to a returned frame must not corrupt the cache."""
    repo = _make_repo(tmp_db_path)

    first = repo.series_df(SQL)
    first["injected"] = 0

    second = repo.series_df(SQL)
    assert "injected" not in second.columns
    repo.close()


def test_insert_on_same_connection_invalidates(tmp_db_path: Path) -> None:
    """PRAGMA data_version does not move for same-connection writes."""
    repo = _make_repo(tmp_db_path)

    assert len(repo.series_df(SQL)) == 2
    repo.query_df("INSERT INTO t_cache (x, y) VALUES (3, 3.0)")
    assert len(repo.series_df(SQL)) == 3
    repo.close()


def test_ddl_invalidates(tmp_db_path: Path) -> None:
    """Pure DDL changes no rows, so only schema_version catches it."""
    repo = _make_repo(tmp_db_path)

    assert list(repo.series_df("SELECT * FROM t_cache").columns) == ["x", "y"]
    repo.query_df("ALTER TABLE t_cache ADD COLUMN z REAL")
    assert list(repo.series_df("SELECT * FROM t_cache").columns) == ["x", "y", "z"]
    repo.close()


def test_external_connection_write_invalidates(tmp_db_path: Path) -> None:
    """A commit from another connection must be picked up."""
    repo = _make_repo(tmp_db_path)
    assert len(repo.series_df(SQL)) == 2

    db_path = repo.ensure_dhub_extension(tmp_db_path)
    with sqlite3.connect(str(db_path)) as con:
        con.execute("INSERT INTO t_cache (x, y) VALUES (4, 4.0)")
        con.commit()

    assert len(repo.series_df(SQL)) == 3
    repo.close()


def test_cache_is_bounded(tmp_db_path: Path) -> None:
    """The LRU must never grow past max_entries."""
    repo = _make_repo(tmp_db_path)
    repo._series_cache_max_entries = 3

    for i in range(10):
        repo.series_df(f"SELECT x + {i} AS x FROM t_cache")

    assert repo.series_cache_stats["entries"] == 3
    repo.close()


def test_cache_can_be_disabled(tmp_db_path: Path) -> None:
    """With the cache off, every read is a straight SQL round trip."""
    repo = _make_repo(tmp_db_path)
    repo._series_cache_enabled = False

    repo.series_df(SQL)
    repo.series_df(SQL)

    assert repo.series_cache_stats["hits"] == 0
    assert repo.series_cache_stats["entries"] == 0
    repo.close()
