"""SqliteRepo.query_arrays: query_df's numeric-only sibling.

query_df().to_numpy() was the pattern at the overwhelming majority of its
call sites - a DataFrame built, then immediately discarded for a plain
array a line or two later. query_arrays reads straight from the DB-API
cursor instead, and never builds the DataFrame at all. These tests pin it
against query_df's own output wherever the two should agree, so a future
change to either cannot let them quietly drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.repo.tables import QueryColumns
from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
def counts_table(repo: SqliteRepo) -> SqliteRepo:
    repo.import_dataframe(
        pd.DataFrame(
            {
                "sample": [1, 2, 3, 4],
                "defectives": [3, "5", None, 2],  # a mixed column, on purpose
                "note": ["a", "b", "c", "d"],
            }
        ),
        table_name="qc",
        normalize_columns=False,
    )
    return repo


# ----------------------------------------------------------------------
# Basic shape
# ----------------------------------------------------------------------
def test_the_column_names_match_the_select(counts_table: SqliteRepo) -> None:
    result = counts_table.query_arrays("SELECT sample, note FROM qc")
    assert result.columns == ("sample", "note")


def test_a_numeric_column_comes_back_as_float64(counts_table: SqliteRepo) -> None:
    result = counts_table.query_arrays(
        "SELECT sample FROM qc ORDER BY sample", numeric=("sample",)
    )
    assert result["sample"].dtype == np.float64
    assert list(result["sample"]) == [1.0, 2.0, 3.0, 4.0]


def test_a_column_not_named_numeric_keeps_its_raw_values(counts_table: SqliteRepo) -> None:
    result = counts_table.query_arrays("SELECT note FROM qc ORDER BY note")
    assert result["note"].dtype == object
    assert list(result["note"]) == ["a", "b", "c", "d"]


# ----------------------------------------------------------------------
# Coercion - matching pd.to_numeric(errors="coerce")
# ----------------------------------------------------------------------
def test_none_and_unparseable_values_become_nan(counts_table: SqliteRepo) -> None:
    result = counts_table.query_arrays(
        "SELECT sample, defectives FROM qc ORDER BY sample", numeric=("defectives",)
    )
    defectives = result["defectives"]
    assert defectives[0] == pytest.approx(3.0)   # int, stored as int
    assert defectives[1] == pytest.approx(5.0)   # "5", a numeric string
    assert np.isnan(defectives[2])               # None
    assert defectives[3] == pytest.approx(2.0)


def test_agrees_with_query_df_to_numpy_on_the_same_query(counts_table: SqliteRepo) -> None:
    """The whole point: the two paths must not answer differently."""
    via_frame = pd.to_numeric(
        counts_table.query_df("SELECT defectives FROM qc ORDER BY sample")["defectives"],
        errors="coerce",
    ).to_numpy(dtype=float)
    via_arrays = counts_table.query_arrays(
        "SELECT defectives FROM qc ORDER BY sample", numeric=("defectives",)
    )["defectives"]
    np.testing.assert_array_equal(via_arrays, via_frame)


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
def test_an_empty_result_still_has_the_right_columns_and_dtypes(
    counts_table: SqliteRepo,
) -> None:
    result = counts_table.query_arrays(
        "SELECT sample, note FROM qc WHERE sample > 100", numeric=("sample",)
    )
    assert result.empty
    assert len(result) == 0
    assert result.columns == ("sample", "note")
    assert result["sample"].dtype == np.float64
    assert result["note"].dtype == object


def test_a_non_select_statement_returns_an_empty_result_not_an_error(
    counts_table: SqliteRepo,
) -> None:
    result = counts_table.query_arrays("DELETE FROM qc WHERE sample = 1")
    assert result.empty
    assert result.columns == ()
    # The statement still ran.
    assert counts_table.query_df("SELECT COUNT(*) AS n FROM qc")["n"].iloc[0] == 3


def test_a_blank_query_returns_an_empty_result(counts_table: SqliteRepo) -> None:
    assert counts_table.query_arrays("").empty
    assert counts_table.query_arrays("   ").empty


# ----------------------------------------------------------------------
# QueryColumns itself
# ----------------------------------------------------------------------
def test_contains_and_get_read_like_a_mapping(counts_table: SqliteRepo) -> None:
    result = counts_table.query_arrays("SELECT sample, note FROM qc")
    assert "sample" in result
    assert "nope" not in result
    assert result.get("nope") is None
    assert result.get("nope", "fallback") == "fallback"
    assert list(result.get("sample")) == list(result["sample"])


def test_len_reflects_the_row_count(counts_table: SqliteRepo) -> None:
    result = counts_table.query_arrays("SELECT sample FROM qc")
    assert len(result) == 4


def test_an_empty_columns_dict_has_length_zero() -> None:
    """A QueryColumns with nothing in it (the non-SELECT/blank-query case)
    must not raise trying to measure its own length."""
    assert len(QueryColumns((), {})) == 0
