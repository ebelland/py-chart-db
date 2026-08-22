"""Tests for turning a stored column into values an operation can use.

The bug this exists for: outlier detection ran ``pd.to_numeric`` over its x
column, which turns a timestamp into NaN.  Every row of a time series was
therefore "not finite", and the dialog reported *"At least 3 finite X/Y points
are required"* about a table with a million rows - an error that sends the
user looking at their data instead of at the code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.utils.coercion import coerce_axis, to_numeric_axis


# ----------------------------------------------------------------------
# coerce_axis
# ----------------------------------------------------------------------
def test_numbers_stay_numbers() -> None:
    values, is_temporal = coerce_axis(pd.Series([1.0, 2.0, 3.0]))

    assert not is_temporal
    assert list(values) == [1.0, 2.0, 3.0]


def test_a_numeric_axis_is_not_read_as_nanoseconds() -> None:
    """to_datetime first would draw 0..4000 as four microseconds of 1970."""
    values, is_temporal = coerce_axis(pd.Series(range(4000)))

    assert not is_temporal
    assert float(values.iloc[-1]) == 3999.0


def test_real_datetimes_pass_through() -> None:
    stamps = pd.to_datetime(["2026-01-01", "2026-01-02"])
    values, is_temporal = coerce_axis(pd.Series(stamps))

    assert is_temporal
    assert pd.api.types.is_datetime64_any_dtype(values)


def test_timestamp_strings_are_recognised() -> None:
    """The case that used to become all-NaN."""
    values, is_temporal = coerce_axis(
        pd.Series(["2026-01-01 10:00", "2026-01-01 11:00"])
    )

    assert is_temporal
    assert values.notna().all()


def test_numbers_stored_as_text_are_numbers() -> None:
    values, is_temporal = coerce_axis(pd.Series(["1.5", "2.5"]))

    assert not is_temporal
    assert list(values) == [1.5, 2.5]


# ----------------------------------------------------------------------
# to_numeric_axis
# ----------------------------------------------------------------------
def test_a_timestamp_column_yields_finite_numbers() -> None:
    """The whole point: an operation can subtract these."""
    values = to_numeric_axis(pd.Series(["2026-01-01 10:00", "2026-01-01 11:00"]))

    assert np.isfinite(values).all()
    assert values[1] - values[0] == pytest.approx(3600.0)


def test_the_spacing_is_faithful() -> None:
    stamps = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"])
    values = to_numeric_axis(pd.Series(stamps))

    day = 86_400.0
    assert values[1] - values[0] == pytest.approx(day)
    assert values[2] - values[1] == pytest.approx(2 * day)


def test_missing_timestamps_become_nan_not_a_huge_number() -> None:
    """NaT cast to int64 is -9223372036854775808, which would look finite."""
    values = to_numeric_axis(pd.Series(["2026-01-01", "not a date"]))

    assert np.isfinite(values[0])
    assert np.isnan(values[1])


def test_a_numeric_column_is_unchanged() -> None:
    values = to_numeric_axis(pd.Series([1.0, 2.0, np.nan]))

    assert values[0] == 1.0 and values[1] == 2.0
    assert np.isnan(values[2])


def test_a_text_column_is_all_nan_rather_than_an_exception() -> None:
    """Unusable data is a data problem; it must not raise mid-operation."""
    values = to_numeric_axis(pd.Series(["alpha", "beta"]))

    assert np.isnan(values).all()


# ----------------------------------------------------------------------
# The callers
# ----------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "relative",
    [
        "series_operations/outlier_dialog.py",
        "series_operations/dialog_base.py",
    ],
)
def test_the_operations_no_longer_coerce_x_by_hand(relative: str) -> None:
    """One rule, one place: a second copy is how the first one drifted."""
    source = (APP_DIR / relative).read_text(encoding="utf-8")

    assert "to_numeric_axis" in source
    assert 'pd.to_numeric(source_df["x"]' not in source


def test_the_time_series_renderer_uses_the_same_rule() -> None:
    source = (APP_DIR / "charts" / "time_series.py").read_text(encoding="utf-8")

    assert "coerce_axis" in source
    assert "def _coerce_x_axis" not in source


def test_the_outlier_error_says_what_it_found() -> None:
    """"3 points required" against a full table sends the user astray."""
    source = (APP_DIR / "series_operations" / "outlier_dialog.py").read_text(
        encoding="utf-8"
    )
    start = source.index("def _detect_outliers")
    body = source[start : source.index("\n    def ", start + 10)]

    assert "len(source_df)" in body
    assert "finite_count" in body
