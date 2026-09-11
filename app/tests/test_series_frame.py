"""SeriesFrame: the DataFrame-shaped idioms renderers actually use (P2-17).

Not a full DataFrame reimplementation - only the handful of operations found
by grepping every renderer's ``sd.df`` (column access, ``.columns``,
``.loc[mask, col]``, ``.copy()``, column assignment, ``.empty``, ``len()``)
plus the escape hatch (``to_pandas``/``from_pandas``) the two genuinely
table-shaped renderers use instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.series_frame import SeriesFrame


@pytest.fixture
def frame() -> SeriesFrame:
    return SeriesFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0], "label": ["a", "b", "c"]})


# ----------------------------------------------------------------------
# Column access - what pd.to_numeric()/.to_numpy()/.tolist() need to work
# ----------------------------------------------------------------------
def test_a_column_comes_back_as_a_series(frame: SeriesFrame) -> None:
    column = frame["x"]
    assert isinstance(column, pd.Series)
    assert column.tolist() == [1, 2, 3]


def test_to_numeric_and_to_numpy_keep_working_on_a_column(frame: SeriesFrame) -> None:
    numeric = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
    np.testing.assert_array_equal(numeric, [10.0, 20.0, 30.0])


def test_membership_and_columns_read_like_a_dataframe(frame: SeriesFrame) -> None:
    assert "x" in frame.columns
    assert "x" in frame
    assert "nope" not in frame
    assert frame.columns == ("x", "y", "label")


def test_a_list_of_columns_selects_a_narrower_frame(frame: SeriesFrame) -> None:
    narrow = frame[["x", "y"]]
    assert narrow.columns == ("x", "y")
    assert len(narrow) == 3


# ----------------------------------------------------------------------
# .loc[mask, column] - pie.py, time_series.py
# ----------------------------------------------------------------------
def test_loc_filters_a_column_by_a_boolean_mask(frame: SeriesFrame) -> None:
    mask = frame["x"] > 1
    kept = frame.loc[mask, "label"]
    assert kept.tolist() == ["b", "c"]


def test_loc_accepts_a_plain_boolean_array(frame: SeriesFrame) -> None:
    mask = np.array([True, False, True])
    assert frame.loc[mask, "x"].tolist() == [1, 3]


# ----------------------------------------------------------------------
# empty / len
# ----------------------------------------------------------------------
def test_empty_and_len_agree_with_row_count(frame: SeriesFrame) -> None:
    assert not frame.empty
    assert len(frame) == 3
    assert SeriesFrame().empty
    assert len(SeriesFrame()) == 0


# ----------------------------------------------------------------------
# copy() / mutation - bar.py, pareto.py
# ----------------------------------------------------------------------
def test_copy_is_independent_of_the_original(frame: SeriesFrame) -> None:
    clone = frame.copy()
    clone["x"] = [9, 9, 9]
    assert frame["x"].tolist() == [1, 2, 3]


def test_assigning_a_new_column_extends_the_frame(frame: SeriesFrame) -> None:
    clone = frame.copy()
    clone["z"] = np.array([100, 200, 300])
    assert "z" in clone.columns
    assert clone["z"].tolist() == [100, 200, 300]


def test_assigning_a_scalar_broadcasts_to_every_row(frame: SeriesFrame) -> None:
    clone = frame.copy()
    clone["flag"] = 0
    assert clone["flag"].tolist() == [0, 0, 0]


def test_a_mismatched_column_length_is_refused(frame: SeriesFrame) -> None:
    with pytest.raises(ValueError):
        frame["oops"] = [1, 2]


def test_building_a_column_at_a_time_sets_the_frames_length() -> None:
    built = SeriesFrame()
    built["x"] = [1, 2, 3, 4]
    assert len(built) == 4


# ----------------------------------------------------------------------
# drop_column - downsampled_series_frame's row-number column
# ----------------------------------------------------------------------
def test_drop_column_removes_it_and_leaves_the_rest(frame: SeriesFrame) -> None:
    dropped = frame.drop_column("label")
    assert dropped.columns == ("x", "y")
    assert len(dropped) == 3


def test_drop_column_is_a_no_op_when_the_column_is_absent(frame: SeriesFrame) -> None:
    dropped = frame.drop_column("nope")
    assert dropped.columns == frame.columns


def test_drop_column_shares_arrays_while_copy_does_not(frame: SeriesFrame) -> None:
    """The two have different jobs, and the docstrings say which is which:
    copy() is what the cache hands out, drop_column() is a view.

    Checked by array identity rather than by writing through one of them:
    a column comes back as a pandas Series, and under copy-on-write its
    ``to_numpy()`` is read-only, so the public accessor cannot scribble on a
    shared array even when one is shared.
    """
    assert frame.drop_column("label")._data["x"] is frame._data["x"]
    assert frame.copy()._data["x"] is not frame._data["x"]


# ----------------------------------------------------------------------
# The escape hatch - pareto.py's groupby, table.py's row/column layout
# ----------------------------------------------------------------------
def test_to_pandas_round_trips_through_from_pandas(frame: SeriesFrame) -> None:
    as_pandas = frame.to_pandas()
    assert list(as_pandas.columns) == ["x", "y", "label"]
    assert len(as_pandas) == 3

    back = SeriesFrame.from_pandas(as_pandas)
    assert back.columns == frame.columns
    assert back["y"].tolist() == frame["y"].tolist()


def test_to_pandas_supports_a_real_groupby() -> None:
    grouped = SeriesFrame({"X": ["a", "a", "b"], "Y": [1, 2, 3]}).to_pandas()
    totals = grouped.groupby("X", as_index=False).agg({"Y": "sum"})
    assert dict(zip(totals["X"], totals["Y"])) == {"a": 3, "b": 3}


# ----------------------------------------------------------------------
# Straight from a sqlite3 cursor - what SqliteRepo builds its cache from
# ----------------------------------------------------------------------
def test_from_rows_types_an_all_numeric_column_as_float64() -> None:
    built = SeriesFrame.from_rows(("x",), [(1,), (2,), (None,)])
    assert built["x"].dtype == np.float64
    assert built["x"].tolist()[:2] == [1.0, 2.0]
    assert np.isnan(built["x"].tolist()[2])


def test_from_rows_keeps_a_text_column_as_text_not_a_number() -> None:
    built = SeriesFrame.from_rows(("label",), [("a",), ("b",)])
    assert built["label"].dtype != np.float64
    assert built["label"].tolist() == ["a", "b"]


def test_from_rows_handles_no_rows() -> None:
    built = SeriesFrame.from_rows(("x", "y"), [])
    assert built.empty
    assert built.columns == ("x", "y")
