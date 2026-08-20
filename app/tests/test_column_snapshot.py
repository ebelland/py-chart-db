"""Tests for the column snapshot/restore used by the clustering preview.

Clustering does not create removable preview artifacts: it overwrites a column
in the user's own table.  The snapshot renames that column aside and puts it
back on Close, so the guarantee under test is that Close leaves the table
byte-for-byte as it was.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo

BACKUP = "__ClusterId_preview_backup__"


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    """A repo with one table carrying a populated ClusterId column."""
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("DROP TABLE IF EXISTS t_snap")
    repo.query_df('CREATE TABLE t_snap (x REAL, y REAL, "ClusterId" INTEGER)')
    repo.query_df(
        'INSERT INTO t_snap (x, y, "ClusterId") VALUES '
        "(1.0, 1.0, 7), (2.0, 2.0, 8), (3.0, 3.0, 9)"
    )
    yield repo
    repo.close()


def _cluster_ids(repo: SqliteRepo) -> list[int]:
    frame = repo.query_df('SELECT "ClusterId" FROM t_snap ORDER BY x')
    return [int(value) for value in frame["ClusterId"].tolist()]


def test_snapshot_moves_the_column_aside(repo: SqliteRepo) -> None:
    assert repo.snapshot_column("t_snap", "ClusterId", BACKUP) is True

    assert not repo.has_column("t_snap", "ClusterId")
    assert repo.has_column("t_snap", BACKUP)


def test_restore_puts_the_original_values_back(repo: SqliteRepo) -> None:
    original = _cluster_ids(repo)

    repo.snapshot_column("t_snap", "ClusterId", BACKUP)
    repo.ensure_cluster_column("t_snap")
    repo.query_df('UPDATE t_snap SET "ClusterId" = 99')
    assert _cluster_ids(repo) == [99, 99, 99]

    repo.restore_column_snapshot("t_snap", "ClusterId", BACKUP)

    assert _cluster_ids(repo) == original
    assert not repo.has_column("t_snap", BACKUP)


def test_snapshot_reports_a_missing_column(repo: SqliteRepo) -> None:
    """No column to save means restoring is 'drop whatever replaced it'."""
    repo.delete_table_column("t_snap", "ClusterId")

    assert repo.snapshot_column("t_snap", "ClusterId", BACKUP) is False
    assert not repo.has_column("t_snap", BACKUP)


def test_discard_drops_the_backup(repo: SqliteRepo) -> None:
    repo.snapshot_column("t_snap", "ClusterId", BACKUP)
    repo.ensure_cluster_column("t_snap")

    repo.discard_column_snapshot("t_snap", BACKUP)

    assert not repo.has_column("t_snap", BACKUP)
    assert repo.has_column("t_snap", "ClusterId")


def test_a_stale_backup_is_replaced_not_kept(repo: SqliteRepo) -> None:
    """A crash mid-preview leaves a backup; the live column is newer truth."""
    repo.snapshot_column("t_snap", "ClusterId", BACKUP)
    repo.ensure_cluster_column("t_snap")
    repo.query_df('UPDATE t_snap SET "ClusterId" = 42')

    # Second preview, without the first having cleaned up.
    assert repo.snapshot_column("t_snap", "ClusterId", BACKUP) is True

    repo.ensure_cluster_column("t_snap")
    repo.query_df('UPDATE t_snap SET "ClusterId" = 5')
    repo.restore_column_snapshot("t_snap", "ClusterId", BACKUP)

    assert _cluster_ids(repo) == [42, 42, 42]


def test_other_columns_are_untouched(repo: SqliteRepo) -> None:
    before = repo.query_df("SELECT x, y FROM t_snap ORDER BY x")

    repo.snapshot_column("t_snap", "ClusterId", BACKUP)
    repo.ensure_cluster_column("t_snap")
    repo.query_df('UPDATE t_snap SET "ClusterId" = 1')
    repo.restore_column_snapshot("t_snap", "ClusterId", BACKUP)

    after = repo.query_df("SELECT x, y FROM t_snap ORDER BY x")
    pd.testing.assert_frame_equal(before, after)


def test_series_sql_can_be_read_back_for_snapshotting(tmp_db_path: Path) -> None:
    """The clustering preview also rewrites series SQL, so it must be readable."""
    repo = SqliteRepo(db_path=tmp_db_path)
    figure_id = repo.create_figure_descriptor(name="f", nrows=1, ncols=1)
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="t",
        x_label="x",
        y_label="y",
        options={},
    )
    repo.query_df("CREATE TABLE IF NOT EXISTS t_sql (x REAL, y REAL)")
    series_id = repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="s",
        sql_query="SELECT x AS x, y AS y FROM t_sql",
        roles={"x": "x", "y": "y"},
    )

    stored = repo.get_series_sql_query(series_id)
    assert stored is not None and "t_sql" in stored
    assert repo.get_series_sql_query(999_999) is None

    repo.close()
