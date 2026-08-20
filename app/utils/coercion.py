"""Turning a stored column into values an operation can work with.

One rule, in one place, because getting it wrong is invisible: a timestamp
column run through ``pd.to_numeric`` becomes all-NaN, and the operation that
receives it reports "not enough finite points" about a table with a million
rows.  That is exactly what outlier detection did to every time series.

Two functions, because two callers want different things from the same
column: a renderer wants datetimes so the axis can be formatted as dates, and
a statistical operation wants numbers so it can subtract them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The epoch and one second as numpy scalars.  Deliberately not a divisor:
# see the note in ``to_numeric_axis``.
_EPOCH = np.datetime64(0, "s")
_ONE_SECOND = np.timedelta64(1, "s")


def coerce_axis(values: pd.Series) -> tuple[pd.Series, bool]:
    """Return the column as datetimes when it is one, else as numbers.

    Returns ``(series, is_temporal)``.

    Order matters.  Trying ``to_datetime`` first would reinterpret a plain
    numeric axis as *nanoseconds since 1970*, drawing an x running 0..4000 as
    four microseconds of 1 January 1970; trying only ``to_numeric`` turns a
    genuine timestamp into NaN and loses the series entirely.  So: real
    datetimes pass through, anything numeric is numeric, and only what is
    neither is read as a timestamp.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return values, True

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric, False

    return pd.to_datetime(values, errors="coerce"), True


def to_numeric_axis(values: pd.Series) -> np.ndarray:
    """Return a float array usable for arithmetic, timestamps included.

    Timestamps become seconds since the epoch.  The unit is arbitrary but the
    spacing is faithful, which is all any of the callers need: outlier
    distances, cluster distances and sampling intervals are all differences,
    and a difference in seconds is as good as one in days.
    """
    coerced, is_temporal = coerce_axis(values)
    if not is_temporal:
        return coerced.to_numpy(dtype=float)

    # A timezone-aware column comes out of ``to_numpy`` as boxed Timestamps,
    # which the arithmetic below cannot do anything with.  The instants are
    # unchanged by dropping the offset and every caller takes differences, so
    # there is nothing here to preserve.
    if isinstance(coerced.dtype, pd.DatetimeTZDtype):
        coerced = coerced.dt.tz_convert("UTC").dt.tz_localize(None)

    # No fixed divisor.  Dividing int64 by 1e9 assumed nanoseconds, and pandas
    # 3 parses strings to datetime64[us] where pandas 2 gave [ns] - so an hour
    # came back as 3.6 instead of 3600, quietly, in every distance the callers
    # compute.  Subtracting a datetime64 epoch and dividing by a timedelta64
    # hands the unit arithmetic to numpy, which is right for [s], [ms], [us]
    # and [ns] alike and stays right if the default moves again.
    #
    # NaT divides to NaN, which is what the callers test for; it must not
    # arrive as int64's -9223372036854775808, which looks perfectly finite.
    seconds = (coerced.to_numpy() - _EPOCH) / _ONE_SECOND
    return np.asarray(seconds, dtype="float64")
