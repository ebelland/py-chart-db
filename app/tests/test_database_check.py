"""Tests for the database check that OptimizeDb runs.

The report separates findings that need action from findings that are merely
informational. That split is the thing under test as much as the detection
itself: if an unreferenced table counted as a problem, every healthy database
with data waiting to be plotted would report as unhealthy, and the user would
learn to ignore the report.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    """A repo holding one figure, one axis, one series over a real table."""
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("DROP TABLE IF EXISTS good_table")
    repo.query_df("CREATE TABLE good_table (x REAL, y REAL)")
    repo.query_df("INSERT INTO good_table (x, y) VALUES (1.0, 2.0)")

    figure_id = repo.create_figure_descriptor(name="fig", nrows=1, ncols=1)
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="t",
        x_label="x",
        y_label="y",
        options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="s",
        sql_query='SELECT x AS x, y AS y FROM "good_table"',
        roles={"x": "x", "y": "y"},
    )
    yield repo
    repo.close()


def test_a_healthy_database_reports_no_problems(repo: SqliteRepo) -> None:
    report = repo.check_database()

    assert report.is_healthy
    assert report.integrity_errors == []
    assert report.foreign_key_errors == []
    assert "passed" in report.summary()


def test_integrity_check_runs(repo: SqliteRepo) -> None:
    """PRAGMA integrity_check on an intact file must say 'ok' and nothing else."""
    report = repo.check_database()
    assert not report.integrity_errors


def test_a_series_reading_a_dropped_table_is_reported(repo: SqliteRepo) -> None:
    repo.query_df("DROP TABLE good_table")

    report = repo.check_database()

    assert not report.is_healthy
    assert any("good_table" in message for message in report.dangling_series)


def test_an_orphan_axis_is_reported(repo: SqliteRepo) -> None:
    """The schema declares the foreign key, but it is only enforced when the
    pragma was on for every writer that ever touched the file."""
    repo._con.execute("PRAGMA foreign_keys = OFF")
    repo.query_df("DELETE FROM __figure_descriptors__")

    report = repo.check_database()

    assert any("axis" in message for message in report.orphan_descriptors)
    assert not report.is_healthy


def test_an_orphan_series_is_reported(repo: SqliteRepo) -> None:
    repo._con.execute("PRAGMA foreign_keys = OFF")
    repo.query_df("DELETE FROM __axis_descriptors__")

    report = repo.check_database()

    assert any("series" in message for message in report.orphan_descriptors)


def test_an_import_link_to_a_dropped_table_is_reported(repo: SqliteRepo) -> None:
    repo.upsert_link(
        table_name="gone_table", source_path="/tmp/gone.csv", settings={}
    )

    report = repo.check_database()

    assert any("gone_table" in message for message in report.dangling_links)


def test_an_unreferenced_table_is_information_not_a_problem(repo: SqliteRepo) -> None:
    """Data waiting to be charted is not an error."""
    repo.query_df("CREATE TABLE IF NOT EXISTS spare (a INTEGER)")

    report = repo.check_database()

    assert "spare" in report.unreferenced_tables
    assert report.is_healthy
    assert "unreferenced" in report.summary()


def test_a_charted_table_is_not_listed_as_unreferenced(repo: SqliteRepo) -> None:
    report = repo.check_database()
    assert "good_table" not in report.unreferenced_tables


def test_a_linked_table_is_not_listed_as_unreferenced(repo: SqliteRepo) -> None:
    repo.query_df("CREATE TABLE IF NOT EXISTS imported (a INTEGER)")
    repo.upsert_link(table_name="imported", source_path="/tmp/x.csv", settings={})

    report = repo.check_database()

    assert "imported" not in report.unreferenced_tables


def test_system_tables_are_never_reported(repo: SqliteRepo) -> None:
    """The application's own tables are not the user's data."""
    report = repo.check_database()
    assert not any(name.startswith("__") for name in report.unreferenced_tables)


def test_problems_collects_every_actionable_finding(repo: SqliteRepo) -> None:
    repo.query_df("DROP TABLE good_table")
    repo.upsert_link(table_name="also_gone", source_path="/tmp/x.csv", settings={})

    report = repo.check_database()

    assert len(report.problems) >= 2
    assert "2 problem" in report.summary() or "problem(s)" in report.summary()


def test_optimize_db_returns_the_report_and_still_compacts(repo: SqliteRepo) -> None:
    repo.query_df("DROP TABLE good_table")

    report = repo.optimize_db()

    assert not report.is_healthy
    # The database is still usable after VACUUM/ANALYZE.
    assert repo.query_df("SELECT 1 AS a").iloc[0]["a"] == 1


def test_check_runs_before_vacuum(repo: SqliteRepo) -> None:
    """VACUUM rewrites the file; anything it drops must be reported first."""
    repo.query_df("CREATE TABLE IF NOT EXISTS spare (a INTEGER)")
    report = repo.optimize_db()
    assert "spare" in report.unreferenced_tables


def test_report_logging_does_not_raise(repo: SqliteRepo) -> None:
    repo.query_df("DROP TABLE good_table")
    repo.check_database().log()  # must not raise
