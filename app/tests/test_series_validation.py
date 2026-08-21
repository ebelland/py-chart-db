"""What the pre-flight check catches, and what it must not reject.

The second half matters as much as the first: a validator that rejects data an
operation handles perfectly well is worse than none, because it stops real work
and teaches people to distrust it.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.utils.series_validation import (
    CONSTANT_Y,
    DUPLICATE_X,
    EMPTY,
    LENGTH_MISMATCH,
    NON_FINITE,
    NON_UNIFORM_X,
    TOO_FEW_POINTS,
    UNSORTED_X,
    clean_xy,
    errors,
    validate_xy,
)


def codes(issues) -> set[str]:
    return {issue.code for issue in issues}


# ----------------------------------------------------------------------
# Clean input is left alone
# ----------------------------------------------------------------------

def test_a_good_series_produces_no_issues() -> None:
    x = np.linspace(0.0, 10.0, 50)
    y = np.sin(x)
    assert validate_xy(x, y) == []


def test_the_strictest_settings_still_accept_a_good_series() -> None:
    """Every requirement switched on at once must not reject clean data."""
    x = np.linspace(0.0, 1.0, 21)
    issues = validate_xy(
        x,
        np.cos(x),
        require_sorted_x=True,
        require_unique_x=True,
        require_uniform_x=True,
        require_varying_y=True,
    )
    assert issues == []


def test_descending_x_is_accepted_when_order_is_not_required() -> None:
    """Only operations that treat the series as f(x) care about order."""
    x = np.linspace(10.0, 0.0, 20)
    assert validate_xy(x, x * 2.0) == []


# ----------------------------------------------------------------------
# The three that give wrong answers rather than errors
# ----------------------------------------------------------------------

def test_unsorted_x_is_reported_when_the_operation_needs_order() -> None:
    x = np.array([0.0, 3.0, 1.0, 2.0])
    issues = validate_xy(x, x.copy(), require_sorted_x=True)
    assert UNSORTED_X in codes(issues)


def test_duplicate_x_is_an_error_for_an_interpolating_operation() -> None:
    x = np.array([0.0, 1.0, 1.0, 2.0])
    issues = validate_xy(x, np.array([0.0, 5.0, 9.0, 1.0]), require_unique_x=True)
    assert DUPLICATE_X in codes(issues)
    assert errors(issues), "a spline through two y at one x has no solution"


def test_duplicate_x_is_ignored_when_the_operation_does_not_care() -> None:
    x = np.array([0.0, 1.0, 1.0, 2.0])
    assert validate_xy(x, x.copy()) == []


def test_non_finite_values_are_reported_but_do_not_block() -> None:
    x = np.arange(6.0)
    y = np.array([1.0, np.nan, 3.0, np.inf, 5.0, 6.0])
    issues = validate_xy(x, y)
    assert NON_FINITE in codes(issues)
    assert not errors(issues), "clean_xy can drop these, so they are a warning"


def test_the_non_finite_message_counts_them() -> None:
    y = np.array([1.0, np.nan, np.nan, 4.0])
    issue = next(i for i in validate_xy(np.arange(4.0), y) if i.code == NON_FINITE)
    assert "2 of 4" in issue.message


# ----------------------------------------------------------------------
# Shape and size
# ----------------------------------------------------------------------

@pytest.mark.parametrize("x, y", [([], []), (None, None), ([], [1.0])])
def test_an_empty_series_is_an_error(x, y) -> None:
    assert EMPTY in codes(validate_xy(x, y))


def test_mismatched_lengths_are_an_error() -> None:
    issues = validate_xy(np.arange(5.0), np.arange(3.0))
    assert LENGTH_MISMATCH in codes(issues)


def test_too_few_points_counts_only_the_usable_ones() -> None:
    """Three points of which two are NaN cannot satisfy a window of three."""
    x = np.arange(3.0)
    y = np.array([1.0, np.nan, np.nan])
    issues = validate_xy(x, y, minimum_points=3)
    assert TOO_FEW_POINTS in codes(issues)
    assert "1 usable point(s)" in str(issues[-1])


def test_shape_checks_stop_once_there_is_too_little_data() -> None:
    """No point complaining about spacing in a series of one."""
    issues = validate_xy([1.0], [1.0], minimum_points=5, require_uniform_x=True)
    assert codes(issues) == {TOO_FEW_POINTS}


# ----------------------------------------------------------------------
# Spacing and variation
# ----------------------------------------------------------------------

def test_non_uniform_x_is_reported_for_an_fft_style_operation() -> None:
    x = np.array([0.0, 1.0, 2.0, 9.0, 10.0])
    issues = validate_xy(x, x.copy(), require_uniform_x=True)
    assert NON_UNIFORM_X in codes(issues)


def test_real_measurements_are_not_rejected_as_non_uniform() -> None:
    """Instrument x is never exactly evenly spaced; the check is tolerant."""
    rng = np.random.default_rng(0)
    x = np.arange(100.0) + rng.normal(0.0, 1e-4, 100)
    assert validate_xy(x, np.sin(x), require_uniform_x=True) == []


def test_a_flat_series_is_an_error_when_variation_is_required() -> None:
    issues = validate_xy(np.arange(10.0), np.full(10, 3.5), require_varying_y=True)
    assert CONSTANT_Y in codes(issues)
    assert "3.5" in str(issues[0])


# ----------------------------------------------------------------------
# Labelling
# ----------------------------------------------------------------------

def test_the_label_names_the_offending_series() -> None:
    issues = validate_xy([], [], label="Detector B")
    assert str(issues[0]).startswith("Detector B: ")


# ----------------------------------------------------------------------
# Repair
# ----------------------------------------------------------------------

def test_clean_drops_non_finite_pairs_and_says_how_many() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, np.nan, 3.0, np.inf])
    x_out, y_out, report = clean_xy(x, y)
    assert x_out.tolist() == [0.0, 2.0]
    assert y_out.tolist() == [1.0, 3.0]
    assert report.dropped_non_finite == 2
    assert "dropped 2" in report.describe()


def test_clean_drops_a_pair_when_only_x_is_non_finite() -> None:
    x_out, y_out, _ = clean_xy([0.0, np.nan, 2.0], [1.0, 2.0, 3.0])
    assert x_out.tolist() == [0.0, 2.0]
    assert y_out.tolist() == [1.0, 3.0]


def test_clean_sorts_only_when_asked() -> None:
    x = np.array([2.0, 0.0, 1.0])
    y = np.array([20.0, 0.0, 10.0])

    untouched, _, report = clean_xy(x, y)
    assert untouched.tolist() == [2.0, 0.0, 1.0]
    assert not report.sorted_x

    x_out, y_out, report = clean_xy(x, y, sort_x=True)
    assert x_out.tolist() == [0.0, 1.0, 2.0]
    assert y_out.tolist() == [0.0, 10.0, 20.0], "y must follow its x"
    assert report.sorted_x


def test_clean_averages_duplicate_x() -> None:
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0, 30.0])
    x_out, y_out, report = clean_xy(x, y, merge_duplicate_x=True)
    assert x_out.tolist() == [0.0, 1.0, 2.0]
    assert y_out.tolist() == [0.0, 15.0, 30.0]
    assert report.merged_duplicates == 1


def test_merging_duplicates_reports_the_reordering_it_implies() -> None:
    """np.unique sorts, so the caller must be told the order changed."""
    x = np.array([2.0, 1.0, 1.0])
    _, _, report = clean_xy(x, np.array([9.0, 1.0, 3.0]), merge_duplicate_x=True)
    assert report.sorted_x


def test_clean_reports_nothing_when_the_series_was_already_good() -> None:
    x = np.linspace(0.0, 1.0, 10)
    _, _, report = clean_xy(x, x.copy(), sort_x=True, merge_duplicate_x=True)
    assert not report.changed
    assert report.describe() == ""


def test_clean_survives_mismatched_lengths() -> None:
    """validate_xy calls this an error, but clean must not raise on it."""
    x_out, y_out, _ = clean_xy(np.arange(5.0), np.arange(3.0))
    assert x_out.size == y_out.size == 3
