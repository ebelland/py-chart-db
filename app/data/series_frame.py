"""SeriesFrame: what ``SeriesData.df`` actually holds now (todo.txt P2-17).

A chart series used to travel SQLite -> ``pd.read_sql_query`` -> a cached
``pd.DataFrame`` -> every renderer's ``sd.df``. The DataFrame in the middle
bought type inference and alignment machinery that a chart never needed: a
renderer reads two or three named columns and draws them, it does not join
frames or align on an index. ``SeriesFrame`` is that plain shape - a dict of
numpy arrays - typed straight from the sqlite3 cursor instead of through
pandas.

Renderers keep writing ``sd.df["x"]``, ``"x" in sd.df.columns``, ``sd.df.copy()``
and ``sd.df.loc[mask, "color"]`` exactly as they did against a DataFrame.
Each column access still hands back a ``pandas.Series`` - a thin view over one
array, not a whole frame - which is what lets ``pd.to_numeric(sd.df["y"])``
and ``.to_numpy()`` keep working at every one of those call sites unexamined.
What is gone is the frame itself between the database and the renderer: no
shared block manager, no index-alignment joining unrelated columns, and no
``pd.read_sql_query`` parsing a whole result set before a renderer reads three
columns of it.

A renderer doing genuinely table-shaped work - Pareto's category
``groupby``, the Table renderer's row/column layout - calls :meth:`to_pandas`
for that one operation, the same way it would reach for numpy for a plain
average rather than inventing one. That is a deliberate, narrow exception,
not a fallback the rest of the type quietly depends on.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def _is_plain_number(value: Any) -> bool:
    """True for the values sqlite3 hands back for an INTEGER/REAL column."""
    return value is None or isinstance(value, (int, float))


def _typed_column(values: tuple[Any, ...]) -> np.ndarray:
    """Type one column the way ``pd.read_sql_query`` would, without pandas.

    sqlite3 already gives each value its native Python type - ``int``,
    ``float``, ``str``, ``bytes`` or ``None`` - straight from the column's
    storage, which is exactly what pandas' own sqlite reader types a column
    from. A column where every value is a number (or NULL) becomes a float64
    array with NaN for the NULLs; anything else - text, blobs, a column that
    genuinely mixes types - stays a plain object array of the raw values, the
    same as a pandas text column would.
    """
    if all(_is_plain_number(value) for value in values):
        return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    return np.array(values, dtype=object)


class _LocIndexer:
    """Backs ``frame.loc[mask, column]`` - the one DataFrame idiom that reads
    more naturally off its own object than as another ``SeriesFrame`` method."""

    __slots__ = ("_frame",)

    def __init__(self, frame: "SeriesFrame") -> None:
        self._frame = frame

    def __getitem__(self, key: tuple[Any, str]) -> pd.Series:
        mask, column = key
        return self._frame[column][mask]


class SeriesFrame:
    """Columns of numpy arrays, read and written the way a DataFrame was.

    Every column is the same length; assigning a new one shorter or longer
    than the columns already there is refused, the same protection a
    DataFrame gives by raising on a length mismatch.
    """

    __slots__ = ("_data", "_length")

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, np.ndarray] = {}
        self._length = 0
        for name, values in (data or {}).items():
            self[name] = values

    # ------------------------------------------------------------------
    # Construction / escape hatch
    # ------------------------------------------------------------------
    @classmethod
    def from_rows(cls, columns: tuple[str, ...], rows: list[tuple[Any, ...]]) -> "SeriesFrame":
        """Build straight from a sqlite3 cursor's column names and rows."""
        if not rows:
            return cls({name: np.array([], dtype=object) for name in columns})
        by_column = list(zip(*rows))
        return cls({name: _typed_column(values) for name, values in zip(columns, by_column)})

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "SeriesFrame":
        return cls({str(column): df[column].to_numpy() for column in df.columns})

    def to_pandas(self) -> pd.DataFrame:
        """The escape hatch for the few renderers doing real table work - a
        pivot, a groupby - that a columnar store has no business
        reimplementing in numpy just to stay pandas-free everywhere."""
        return pd.DataFrame(self._data)

    # ------------------------------------------------------------------
    # Mapping-ish interface
    # ------------------------------------------------------------------
    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._data.keys())

    @property
    def empty(self) -> bool:
        return len(self) == 0

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)

    def __len__(self) -> int:
        return self._length

    def __contains__(self, column: object) -> bool:
        return column in self._data

    def __getitem__(self, key: str | Iterable[str]) -> "pd.Series | SeriesFrame":
        if isinstance(key, str):
            return pd.Series(self._data[key], name=key)
        names = tuple(key)
        return SeriesFrame({name: self._data[name] for name in names})

    def __setitem__(self, column: str, values: Any) -> None:
        if isinstance(values, pd.Series):
            values = values.to_numpy()
        array = np.asarray(values)
        if array.ndim == 0:
            fill = self._length if self._length else 1
            array = np.full(fill, array.item())
        if self._data and array.size != self._length:
            raise ValueError(
                f"column {column!r} has {array.size} rows, frame has {self._length}"
            )
        self._data[column] = array
        self._length = array.size

    def get(self, column: str, default: Any = None) -> Any:
        if column not in self._data:
            return default
        return self[column]

    def copy(self) -> "SeriesFrame":
        clone = SeriesFrame.__new__(SeriesFrame)
        clone._data = {name: values.copy() for name, values in self._data.items()}
        clone._length = self._length
        return clone

    def drop_column(self, column: str) -> "SeriesFrame":
        """Return a copy without *column*; a no-op if it is not present."""
        if column not in self._data:
            return self.copy()
        return SeriesFrame({name: values for name, values in self._data.items() if name != column})

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SeriesFrame(columns={self.columns!r}, rows={len(self)})"
