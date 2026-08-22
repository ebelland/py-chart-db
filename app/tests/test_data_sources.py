"""Tests for saved queries as first-class data sources.

The contract under test is that a caller can build ``SELECT <cols> FROM
{source.from_clause()}`` and stop caring whether it got a physical table or a
saved query - and that a saved query is *executed*, never materialised, so
editing it changes every chart built on it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.data_source import DataSource, quote_identifier
from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    """A repo with one table and one saved query over it."""
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("DROP TABLE IF EXISTS measurements")
    repo.query_df("CREATE TABLE measurements (x REAL, y REAL, grp TEXT)")
    repo.query_df(
        "INSERT INTO measurements (x, y, grp) VALUES "
        "(1.0, 10.0, 'a'), (2.0, 20.0, 'a'), (3.0, 30.0, 'b'), (4.0, 40.0, 'b')"
    )
    repo.save_query("group_a", "SELECT x, y FROM measurements WHERE grp = 'a'")
    yield repo
    repo.close()


# ----------------------------------------------------------------------
# The FROM fragment
# ----------------------------------------------------------------------
def test_table_from_clause_is_a_quoted_name() -> None:
    assert DataSource.table("my_table").from_clause() == '"my_table"'


def test_query_from_clause_is_an_aliased_subquery() -> None:
    clause = DataSource.query("q", "SELECT 1 AS a").from_clause()
    assert clause.startswith("(SELECT 1 AS a) AS ")


def test_trailing_semicolon_is_stripped_from_a_subquery() -> None:
    """A subquery cannot contain a statement terminator."""
    clause = DataSource.query("q", "SELECT 1 AS a ;  ").from_clause()
    assert ";" not in clause


def test_identifier_quoting_escapes_embedded_quotes() -> None:
    assert quote_identifier('we"ird') == '"we""ird"'


def test_select_projects_only_valid_identifiers() -> None:
    source = DataSource.table("t")
    assert source.select_sql(["a", "b"]) == 'SELECT "a", "b" FROM "t"'
    # A non-identifier cannot have come from the app's own column listing.
    assert source.select_sql(["a; DROP TABLE t"]) == 'SELECT * FROM "t"'
    assert source.select_sql([]) == 'SELECT * FROM "t"'


# ----------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------
def test_a_table_resolves_to_a_table_source(repo: SqliteRepo) -> None:
    source = repo.get_data_source("measurements")
    assert source is not None and not source.is_query


def test_a_saved_query_resolves_to_a_query_source(repo: SqliteRepo) -> None:
    source = repo.get_data_source("group_a")
    assert source is not None and source.is_query
    assert "measurements" in source.sql


def test_an_unknown_name_resolves_to_nothing(repo: SqliteRepo) -> None:
    assert repo.get_data_source("nope") is None
    assert repo.get_data_source("") is None


def test_a_table_wins_over_a_query_with_the_same_name(repo: SqliteRepo) -> None:
    """Otherwise the preview and a hand-written query would disagree."""
    repo.save_query("measurements", "SELECT 1 AS x")

    source = repo.get_data_source("measurements")
    assert source is not None and not source.is_query


# ----------------------------------------------------------------------
# Reading through a source
# ----------------------------------------------------------------------
def test_columns_are_the_same_shape_for_both_kinds(repo: SqliteRepo) -> None:
    table_columns = repo.data_source_columns(repo.get_data_source("measurements"))
    query_columns = repo.data_source_columns(repo.get_data_source("group_a"))

    assert table_columns == ["x", "y", "grp"]
    assert query_columns == ["x", "y"]


def test_row_count_runs_the_query(repo: SqliteRepo) -> None:
    assert repo.data_source_row_count(repo.get_data_source("measurements")) == 4
    assert repo.data_source_row_count(repo.get_data_source("group_a")) == 2


def test_paging_works_for_a_query(repo: SqliteRepo) -> None:
    source = repo.get_data_source("group_a")

    first = repo.data_source_page(source, limit=1, offset=0)
    second = repo.data_source_page(source, limit=1, offset=1)

    assert len(first) == 1 and len(second) == 1
    assert first.iloc[0]["x"] != second.iloc[0]["x"]


def test_a_query_is_executed_not_materialised(repo: SqliteRepo) -> None:
    """Editing the query must change what every reader sees, immediately."""
    assert repo.data_source_row_count(repo.get_data_source("group_a")) == 2

    repo.save_query("group_a", "SELECT x, y FROM measurements")

    assert repo.data_source_row_count(repo.get_data_source("group_a")) == 4
    # And no table or view was created for it.
    assert "group_a" not in repo.list_table_names()


def test_new_rows_appear_in_a_saved_query(repo: SqliteRepo) -> None:
    repo.query_df("INSERT INTO measurements (x, y, grp) VALUES (5.0, 50.0, 'a')")
    assert repo.data_source_row_count(repo.get_data_source("group_a")) == 3


# ----------------------------------------------------------------------
# Listing
# ----------------------------------------------------------------------
def test_listing_includes_both_kinds_with_a_kind_column(repo: SqliteRepo) -> None:
    frame = repo.list_data_sources()

    kinds = dict(zip(frame["Table"], frame["kind"]))
    assert kinds["measurements"] == "table"
    assert kinds["group_a"] == "query"


def test_listing_keeps_the_table_columns(repo: SqliteRepo) -> None:
    """The table list renders both kinds from one frame."""
    frame = repo.list_data_sources()
    assert {"Table", "has_link", "Notes", "source_path", "kind"} <= set(frame.columns)
    assert not bool(frame.loc[frame["Table"] == "group_a", "has_link"].iloc[0])


def test_listing_without_queries_still_works(tmp_db_path: Path) -> None:
    """Backward compatibility: a database with no saved queries."""
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("CREATE TABLE IF NOT EXISTS plain (a INTEGER)")

    frame = repo.list_data_sources()
    assert list(frame.loc[frame["Table"] == "plain", "kind"]) == ["table"]
    repo.close()


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_a_valid_query_reports_its_columns(repo: SqliteRepo) -> None:
    ok, message = repo.validate_query("SELECT x, y FROM measurements")
    assert ok
    assert "x" in message and "y" in message


def test_a_syntax_error_is_reported_not_raised(repo: SqliteRepo) -> None:
    ok, message = repo.validate_query("SELECT FROM WHERE")
    assert not ok and message


def test_an_unknown_table_is_reported(repo: SqliteRepo) -> None:
    ok, message = repo.validate_query("SELECT * FROM no_such_table")
    assert not ok
    assert "no_such_table" in message


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "DELETE FROM measurements",
        "UPDATE measurements SET x = 1",
        "DROP TABLE measurements",
    ],
)
def test_only_row_returning_statements_are_accepted(repo: SqliteRepo, sql: str) -> None:
    """A saved query is read on every render; a write would run every time."""
    ok, _message = repo.validate_query(sql)
    assert not ok


def test_validation_does_not_read_rows(repo: SqliteRepo) -> None:
    """LIMIT 0 is what makes validating a query over a huge table instant."""
    ok, _ = repo.validate_query("SELECT * FROM measurements")
    assert ok
    # The table is untouched.
    assert repo.data_source_row_count(repo.get_data_source("measurements")) == 4


def test_a_with_statement_is_accepted(repo: SqliteRepo) -> None:
    ok, _ = repo.validate_query(
        "WITH doubled AS (SELECT x * 2 AS x FROM measurements) SELECT * FROM doubled"
    )
    assert ok


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def test_a_saved_query_survives_a_reopen(tmp_db_path: Path) -> None:
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
    repo.save_query("q1", "SELECT a FROM t")
    repo.close()

    reopened = SqliteRepo(db_path=tmp_db_path)
    saved = reopened.get_query("q1")
    assert saved is not None and saved.sql == "SELECT a FROM t"
    reopened.close()


def test_saving_the_same_name_updates_it(repo: SqliteRepo) -> None:
    first_id = repo.save_query("dup", "SELECT 1 AS a")
    second_id = repo.save_query("dup", "SELECT 2 AS a")

    assert first_id == second_id
    saved = repo.get_query("dup")
    assert saved is not None and "2" in saved.sql
    assert len([q for q in repo.list_queries() if q.name == "dup"]) == 1


def test_deleting_a_query_removes_it_from_the_sources(repo: SqliteRepo) -> None:
    assert repo.delete_query("group_a") is True
    assert repo.get_data_source("group_a") is None
    assert "group_a" not in set(repo.list_data_sources()["Table"])


# ----------------------------------------------------------------------
# A new, unsaved query
# ----------------------------------------------------------------------
def test_a_new_query_is_a_draft_and_writes_nothing(repo: SqliteRepo) -> None:
    """The whole point: a query is worth a row once it runs, and this one
    does not yet - ``save_query`` refuses empty SQL, so a row written here
    would be one no chart can read."""
    draft = repo.new_query("daily")

    assert draft.name == "daily"
    assert draft.id is None
    assert repo.get_query("daily") is None
    assert "daily" not in {saved.name for saved in repo.list_queries()}


def test_a_new_query_seeds_a_select_over_the_table_it_is_given(repo: SqliteRepo) -> None:
    draft = repo.new_query("over_measurements", table="measurements")

    assert draft.sql == 'SELECT * FROM "measurements"'
    ok, _message = repo.validate_query(draft.sql)
    assert ok, "the seed must be a statement that runs"


def test_a_new_query_with_no_table_starts_empty(repo: SqliteRepo) -> None:
    """Nothing beyond the table is guessed: an invented WHERE or JOIN gives a
    query that runs and returns the wrong rows, which is worse than none."""
    assert repo.new_query("blank").sql == ""


def test_an_unnamed_new_query_is_given_a_free_name(repo: SqliteRepo) -> None:
    draft = repo.new_query("")

    assert draft.name == "Query 1"
    assert repo.get_query(draft.name) is None


def test_a_suggested_name_skips_the_ones_already_taken(repo: SqliteRepo) -> None:
    repo.save_query("Query 1", "SELECT 1 AS a")
    repo.save_query("Query 2", "SELECT 2 AS a")

    assert repo.next_query_name() == "Query 3"


def test_a_suggested_name_avoids_table_names_too(repo: SqliteRepo) -> None:
    """A query named after a table can never be selected, because the table
    always wins - so suggesting one offers a name Save is about to refuse."""
    # IF NOT EXISTS: the test database is a file in the artifacts directory
    # and survives the run, so a second run would meet its own leftovers.
    repo.query_df('CREATE TABLE IF NOT EXISTS "Draft 1" (a INTEGER)')

    assert repo.next_query_name("Draft") == "Draft 2"


def test_a_suggested_name_ignores_case(repo: SqliteRepo) -> None:
    """SQLite resolves a table name whatever its case, and two saved queries
    differing only in case are a trap for whoever reads the list."""
    repo.save_query("query 1", "SELECT 1 AS a")

    assert repo.next_query_name() == "Query 2"


def test_a_draft_may_take_the_name_of_an_existing_query(repo: SqliteRepo) -> None:
    """Overwriting is the caller's decision - the builder asks first - so the
    repository reports it and hands the draft back rather than refusing."""
    draft = repo.new_query("group_a")

    assert draft.name == "group_a"
    assert repo.get_query("group_a") is not None, "the stored one is untouched"
