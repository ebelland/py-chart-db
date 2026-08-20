"""Tests for the helpers that replaced per-module copies.

Each of these existed in four to six places with subtly different behaviour.
The tests pin the one behaviour that now applies everywhere, and the last two
guard against a seventh copy appearing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.data.data_source import parse_roles, quote_identifier, row_value,parse_roles

APP_DIR = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# quote_identifier
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("plain", '"plain"'),
        ('we"ird', '"we""ird"'),
        ("with space", '"with space"'),
        ("", '""'),
        (None, '""'),
        ('""', '""""""'),
    ],
)
def test_quoting_doubles_embedded_quotes(name, expected: str) -> None:
    assert quote_identifier(name) == expected


def test_a_quoted_name_survives_a_round_trip(tmp_db_path: Path) -> None:
    """The point of quoting is that SQLite accepts the result."""
    from app.data.sqlite_repo import SqliteRepo

    repo = SqliteRepo(db_path=tmp_db_path)
    awkward = 'odd "name"'
    repo.query_df(f"CREATE TABLE IF NOT EXISTS {quote_identifier(awkward)} (a INTEGER)")
    repo.query_df(f"INSERT INTO {quote_identifier(awkward)} (a) VALUES (1)")

    frame = repo.query_df(f"SELECT a FROM {quote_identifier(awkward)}")
    assert int(frame.iloc[0]["a"]) == 1
    repo.close()


# ----------------------------------------------------------------------
# parse_roles
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('{"x": "a", "y": "b"}', {"x": "a", "y": "b"}),
        ({"x": "a"}, {"x": "a"}),
        ("", {}),
        (None, {}),
        ("{}", {}),
        # Malformed input is a data problem, not a crash: every caller does
        # roles.get(...) immediately after.
        ("not json", {}),
        ("[1, 2, 3]", {}),
        ("null", {}),
    ],
)
def test_roles_always_parse_to_a_dict(value, expected: dict) -> None:
    assert parse_roles(value) == expected


def test_role_keys_are_strings_even_from_json_numbers() -> None:
    assert parse_roles('{"1": "a"}') == {"1": "a"}
    assert parse_roles({1: "a"}) == {"1": "a"}


# ----------------------------------------------------------------------
# row_value
# ----------------------------------------------------------------------
def test_the_first_present_name_wins() -> None:
    assert row_value({"b": 2, "a": 1}, "a", "b") == 1


def test_a_none_value_falls_through_to_the_next_name() -> None:
    """A column that exists but is NULL is not an answer."""
    assert row_value({"a": None, "b": 2}, "a", "b") == 2


def test_a_missing_name_falls_through() -> None:
    assert row_value({"b": 2}, "a", "b") == 2


def test_the_default_is_returned_when_nothing_matches() -> None:
    assert row_value({}, "a", "b", default="fallback") == "fallback"
    assert row_value(None, "a") is None


def test_sqlite_rows_are_supported(tmp_db_path: Path) -> None:
    """sqlite3.Row is not a Mapping, which is why the helper has two branches."""
    from app.data.sqlite_repo import SqliteRepo

    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("CREATE TABLE IF NOT EXISTS rows_t (a INTEGER, b INTEGER)")
    repo.query_df("INSERT INTO rows_t (a, b) VALUES (NULL, 7)")

    row = repo._con.execute("SELECT a, b FROM rows_t").fetchone()
    assert row_value(row, "a", "b") == 7
    assert row_value(row, "missing", default="d") == "d"
    repo.close()


# ----------------------------------------------------------------------
# The duplication does not come back
# ----------------------------------------------------------------------
def _sources() -> list[Path]:
    return [
        path
        for path in APP_DIR.rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    ]


def test_identifier_quoting_is_defined_once() -> None:
    """Six copies of this existed and they did not all agree."""
    pattern = re.compile(r"""replace\(\s*['"]"['"]\s*,\s*['"]""['"]\s*\)""")
    offenders = [
        str(path.relative_to(APP_DIR))
        for path in _sources()
        if path.name != "data_source.py" and pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        "these files quote identifiers by hand; use "
        f"data_source.quote_identifier instead: {offenders}"
    )


def test_doc_links_are_built_by_the_helper() -> None:
    """Four of the five hand-built links forgot to escape the URL."""
    pattern = re.compile(r"""setText\(\s*f?['"]<a href=""")
    offenders = [
        str(path.relative_to(APP_DIR))
        for path in _sources()
        if path.name != "style.py" and pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        f"these files build <a href> markup by hand; use set_doc_link: {offenders}"
    )
