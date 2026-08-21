"""Check that a series is fit to be operated on, before an operation runs.

Series operations were handed whatever the query returned.  The failure modes
are not equally loud, and that is the problem: some raise somewhere deep in
SciPy with a message that names a matrix rather than a series, and some return
a perfectly plausible-looking array that is quietly wrong.

Three that produce wrong answers rather than errors:

* **Unsorted x.** ``savgol_filter``, ``np.gradient`` and every rolling window
  treat the array as a sequence, not as a function of x.  Feed them points in
  query order and they smooth across a fold in the data, returning numbers
  that look like a result.
* **Duplicate x.** An interpolating spline through two different y at one x has
  no solution; SciPy either raises about a singular matrix or silently returns
  a fit that passes through neither point.
* **Non-finite values.** One NaN propagates through a convolution to every
  point in its window, and through an FFT to the entire output.

So the rule here is: detect, say which series and what is wrong in the caller's
language, and offer to repair.  ``clean_xy`` does the repair and reports what it
changed, because dropping points silently is its own kind of wrong answer.

Pure numpy, no Qt and no pandas: this is the part that has to be testable
without a window server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

Severity = Literal["error", "warning"]

#: Issue codes.  Strings rather than an enum so a caller can name one in a
#: message catalogue without importing this module.
EMPTY = "empty"
LENGTH_MISMATCH = "length_mismatch"
TOO_FEW_POINTS = "too_few_points"
NON_FINITE = "non_finite"
UNSORTED_X = "unsorted_x"
DUPLICATE_X = "duplicate_x"
CONSTANT_Y = "constant_y"
NON_UNIFORM_X = "non_uniform_x"


@dataclass(frozen=True, slots=True)
class SeriesIssue:
    """One problem found in a series."""

    code: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class CleanReport:
    """What ``clean_xy`` changed, so the caller can say so."""

    dropped_non_finite: int = 0
    sorted_x: bool = False
    merged_duplicates: int = 0

    @property
    def changed(self) -> bool:
        return bool(
            self.dropped_non_finite or self.sorted_x or self.merged_duplicates
        )

    def describe(self) -> str:
        """Return a human-readable summary, or empty when nothing changed."""
        parts: list[str] = []
        if self.dropped_non_finite:
            parts.append(
                f"dropped {self.dropped_non_finite} non-finite point(s)"
            )
        if self.sorted_x:
            parts.append("sorted by x")
        if self.merged_duplicates:
            parts.append(f"merged {self.merged_duplicates} duplicate x value(s)")
        return "; ".join(parts)


def _as_float_array(values: Any) -> np.ndarray:
    """Return ``values`` as a 1-D float array, or an empty one."""
    if values is None:
        return np.empty(0, dtype=float)
    try:
        array = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.empty(0, dtype=float)
    return array


def errors(issues: Sequence[SeriesIssue]) -> list[SeriesIssue]:
    """Return only the issues that should stop the operation."""
    return [issue for issue in issues if issue.severity == "error"]


def validate_xy(
    x: Any,
    y: Any,
    *,
    minimum_points: int = 2,
    require_sorted_x: bool = False,
    require_unique_x: bool = False,
    require_uniform_x: bool = False,
    require_varying_y: bool = False,
    repairable: bool = False,
    label: str = "",
) -> list[SeriesIssue]:
    """Return every problem found in one x/y series.

    The caller says what the operation actually needs, because the answer
    differs per operation and guessing the strictest set would reject series
    that a given operation handles perfectly well.  An FFT needs uniform
    spacing; ``np.gradient`` does not.  A spline needs unique x; a scatter
    smoother does not care.

    ``repairable`` says the caller is about to run ``clean_xy`` with matching
    settings.  It only changes severity, never what is detected: duplicate x is
    fatal to a spline, but averaging the duplicates is a defensible repair, so
    it is an error when nobody is going to fix it and a warning when somebody
    is.  Reporting it either way is the point - the previous code averaged
    duplicates silently, so a series with two readings at one x produced a
    curve through neither, with nothing said.

    An empty result means the series is usable.  Order is stable and
    most-fundamental-first, so a caller showing only ``issues[0]`` shows the
    thing worth fixing first.
    """
    prefix = f"{label}: " if label else ""
    issues: list[SeriesIssue] = []

    x_array = _as_float_array(x)
    y_array = _as_float_array(y)

    if x_array.size == 0 or y_array.size == 0:
        issues.append(
            SeriesIssue(EMPTY, "error", f"{prefix}The series has no data.")
        )
        return issues

    if x_array.size != y_array.size:
        issues.append(
            SeriesIssue(
                LENGTH_MISMATCH,
                "error",
                f"{prefix}x has {x_array.size} value(s) but y has "
                f"{y_array.size}.",
            )
        )
        return issues

    finite = np.isfinite(x_array) & np.isfinite(y_array)
    non_finite_count = int((~finite).sum())
    if non_finite_count:
        issues.append(
            SeriesIssue(
                NON_FINITE,
                # A warning, not an error: clean_xy can drop these, and most
                # operations are happy once they are gone.
                "warning",
                f"{prefix}{non_finite_count} of {x_array.size} point(s) are "
                f"NaN or infinite and will be ignored.",
            )
        )

    usable = int(finite.sum())
    if usable < minimum_points:
        issues.append(
            SeriesIssue(
                TOO_FEW_POINTS,
                "error",
                f"{prefix}{usable} usable point(s); this operation needs at "
                f"least {minimum_points}.",
            )
        )
        # Everything below describes the shape of the data, which is not
        # meaningful once there is not enough of it.
        return issues

    x_finite = x_array[finite]
    y_finite = y_array[finite]

    if require_sorted_x and np.any(np.diff(x_finite) < 0):
        issues.append(
            SeriesIssue(
                UNSORTED_X,
                "warning",
                f"{prefix}x is not in increasing order; the points will be "
                f"sorted before the calculation."
                if repairable
                # Not every caller can reorder. The outlier detector maps its
                # result back to source rows by rowid, so sorting would move
                # the mark onto a different row - it has to say "fix your
                # data" rather than "we fixed it".
                else f"{prefix}x is not in increasing order, so this "
                f"operation will follow row order rather than x. Sort the "
                f"source data by x for a meaningful result.",
            )
        )

    if require_unique_x:
        duplicates = x_finite.size - np.unique(x_finite).size
        if duplicates:
            issues.append(
                SeriesIssue(
                    DUPLICATE_X,
                    "warning" if repairable else "error",
                    f"{prefix}{duplicates} duplicate x value(s); their y "
                    f"values will be averaged."
                    if repairable
                    else f"{prefix}{duplicates} duplicate x value(s). This "
                    f"operation needs one y per x - remove or average them "
                    f"first.",
                )
            )

    if require_uniform_x and x_finite.size > 2:
        steps = np.diff(np.sort(x_finite))
        # Relative tolerance against the mean step: floating-point x from a
        # real instrument is never exactly uniform, and demanding that it be
        # would reject every genuine measurement.
        mean_step = float(np.mean(steps))
        if mean_step > 0 and float(np.std(steps)) / mean_step > 0.01:
            issues.append(
                SeriesIssue(
                    NON_UNIFORM_X,
                    "warning",
                    f"{prefix}x is not evenly spaced. This operation assumes "
                    f"a constant step, so the result will be misleading - "
                    f"resample onto a uniform grid first.",
                )
            )

    if require_varying_y and y_finite.size and float(np.ptp(y_finite)) == 0.0:
        issues.append(
            SeriesIssue(
                CONSTANT_Y,
                "error",
                f"{prefix}Every y value is {float(y_finite[0]):g}; there is "
                f"nothing for this operation to work on.",
            )
        )

    return issues


def clean_xy(
    x: Any,
    y: Any,
    *,
    sort_x: bool = False,
    merge_duplicate_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, CleanReport]:
    """Return the series with non-finite points removed, optionally tidied.

    Always drops non-finite pairs, because no operation wants them.  Sorting
    and duplicate merging are opt-in: they change which x each y belongs to,
    and an operation that treats the series as an ordered sequence rather than
    as a function of x would be wrong to have them applied behind its back.

    Duplicates are merged by averaging their y, which is the only choice that
    does not silently prefer one measurement over another.
    """
    x_array = _as_float_array(x)
    y_array = _as_float_array(y)

    if x_array.size != y_array.size:
        size = min(x_array.size, y_array.size)
        x_array = x_array[:size]
        y_array = y_array[:size]

    finite = np.isfinite(x_array) & np.isfinite(y_array)
    dropped = int((~finite).sum())
    x_array = x_array[finite]
    y_array = y_array[finite]

    sorted_x = False
    if sort_x and x_array.size and np.any(np.diff(x_array) < 0):
        order = np.argsort(x_array, kind="stable")
        x_array = x_array[order]
        y_array = y_array[order]
        sorted_x = True

    merged = 0
    if merge_duplicate_x and x_array.size:
        # np.unique returns sorted values, so this implies sorted output. An
        # operation that asked to merge duplicates but not to sort would get a
        # reordered series it did not expect, so sorting is reported too.
        unique_x, inverse = np.unique(x_array, return_inverse=True)
        if unique_x.size != x_array.size:
            merged = int(x_array.size - unique_x.size)
            totals = np.bincount(inverse, weights=y_array)
            counts = np.bincount(inverse)
            y_array = totals / counts
            x_array = unique_x
            sorted_x = True

    return x_array, y_array, CleanReport(dropped, sorted_x, merged)
