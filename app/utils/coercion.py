"""Turning a stored column into values an operation can work with.

One rule, in one place, because getting it wrong is invisible: a timestamp
column run through ``pd.to_numeric`` becomes all-NaN, and the operation that
receives it reports "not enough finite points" about a table with a million
rows.  That is exactly what outlier detection did to every time series.

Two functions, because two callers want different things from the same
column: a renderer wants datetimes so the axis can be formatted as dates, and
a statistical operation wants numbers so it can subtract them.  A third,
``parse_datetimes``, is the parsing both of them share - and the reason it
is a function rather than one call to ``pd.to_datetime`` is written on it.
"""
from __future__ import annotations

import warnings

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

    return parse_datetimes(values), True


def parse_datetimes(values: pd.Series) -> pd.Series:
    """Parse a column of timestamps, whatever shape its entries are in.

    ``pd.to_datetime`` on its own infers *one* format from the first entry
    and coerces everything that does not match it to NaT.  A column whose
    entries differ only in precision - "2024-01-01 00:00:00" beside
    "2024-01-16 16:58:38.426808", which is exactly what SQLite gives back
    after storing computed timestamps - therefore comes back as one date
    and a column of NaT, silently.

    So the formats are tried in order of how much they cost:

    1. ``ISO8601``, which accepts any amount of precision and is *faster*
       than the guessing path (12 ms against 21 for 200 000 rows) because
       it has nothing to guess.  This is the case for anything SQLite
       wrote and for most of what is imported.
    2. The guessing path, for the non-ISO layouts people really do have -
       "01/02/2024", "Jan 5 2024".
    3. ``mixed``, which parses each entry on its own.  Slowest, and the
       only thing that reads a column with genuinely different layouts in
       it.

    The first attempt that loses no non-null value wins; if none is clean,
    the one that lost the fewest does.
    """
    best: pd.Series | None = None
    best_lost = -1
    present = values.notna()

    for attempt in ({"format": "ISO8601"}, {}, {"format": "mixed"}):
        try:
            with warnings.catch_warnings():
                # "Could not infer format, so each element will be parsed
                # individually" - which is the guessing attempt doing exactly
                # what it is here to do. Warning about a fallback that was
                # asked for tells the user nothing they can act on.
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(values, errors="coerce", **attempt)
        except (TypeError, ValueError):
            continue

        lost = int((present & parsed.isna()).sum())
        if lost == 0:
            return parsed
        if best is None or lost < best_lost:
            best, best_lost = parsed, lost

    if best is None:  # pragma: no cover - to_datetime raising three times
        return pd.to_datetime(pd.Series([pd.NaT] * len(values), index=values.index))
    return best


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
