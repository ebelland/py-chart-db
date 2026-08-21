"""One way to refer to "somewhere data comes from".

Charts, the preview pane and the table list all need the same thing: a name the
user picked, and a SQL fragment they can select from.  Before this existed the
answer was always a physical table name pasted straight into an f-string, which
is why saved queries could not be used anywhere.

A :class:`DataSource` answers both questions for either kind:

    table   -> ``"my_table"``
    query   -> ``(SELECT ...) AS _src``

Everything downstream builds ``SELECT <cols> FROM {source.from_clause()}`` and
stops caring which kind it got.  A saved query is therefore executed every time
it is read - it is never materialised into a table or a VIEW - so editing the
query updates every chart built on it.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

# Same identifier rule the repository uses; duplicated here rather than
# imported to keep this module free of repository imports.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Alias given to a saved query when it is used as a subquery.  Leading
# underscore so it cannot collide with a user column name.
QUERY_ALIAS: str = "_src"

SourceKind = Literal["table", "query"]


def quote_identifier(name: str) -> str:
    """Quote a SQLite identifier, doubling any embedded quotes.

    This is *the* quoting function for the whole application.  Six copies of it
    existed - in the repository, the preview widget, two operation dialogs and
    inline in the chart dialog - and they did not all agree: some coerced None
    to the string "None", some did not strip.  One implementation means one
    behaviour to reason about when a table name contains a quote.
    """
    return '"' + str(name or "").replace('"', '""') + '"'


def is_identifier(name: str) -> bool:
    """True when a name can be used unquoted as a SQLite identifier."""
    return bool(_IDENT_RE.match((name or "").strip()))


def parse_roles(value: object) -> dict[str, Any]:
    """Return a series' roles as a plain dict, from a mapping or JSON text.

    Roles arrive either already decoded (from a descriptor) or as the raw
    ``roles`` column (JSON text).  Four dialogs each had their own version of
    this, with four different behaviours for malformed input; this one always
    returns a dict, because every caller immediately does ``roles.get(...)``
    and a None there would only turn a data problem into an AttributeError.
    """
    if not value:
        return {}

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}

    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError):
        return {}

    if not isinstance(decoded, Mapping):
        return {}
    return {str(key): item for key, item in decoded.items()}


def row_value(row: object, *names: str, default: Any = None) -> Any:
    """Return the first present, non-None value among several column names.

    Rows reach the dialogs as ``sqlite3.Row``, as dicts, or as descriptor
    objects, and the same field has been spelled differently over time (``id``
    vs ``series_id``).  Accepting several names in one call is what lets a
    caller stay readable instead of chaining ``or`` across three lookups.
    """
    for name in names:
        try:
            if isinstance(row, Mapping):
                if name in row and row[name] is not None:
                    return row[name]
                continue
            keys = getattr(row, "keys", None)
            if callable(keys) and name in keys():  # type: ignore[operator]
                value = row[name]  # type: ignore[index]
                if value is not None:
                    return value
        except (TypeError, KeyError, IndexError):
            continue
    return default


@dataclass(frozen=True, slots=True)
class DataSource:
    """A physical table or a saved query, addressed the same way."""

    name: str
    kind: SourceKind = "table"
    sql: str = ""

    @property
    def is_query(self) -> bool:
        """True for a saved query, False for a physical table."""
        return self.kind == "query"

    @classmethod
    def table(cls, name: str) -> "DataSource":
        """Return a source for a physical table."""
        return cls(name=str(name), kind="table", sql="")

    @classmethod
    def query(cls, name: str, sql: str) -> "DataSource":
        """Return a source for a saved query."""
        return cls(name=str(name), kind="query", sql=str(sql))

    def from_clause(self, alias: str = QUERY_ALIAS) -> str:
        """Return the FROM fragment for this source.

        A saved query is wrapped in parentheses and aliased, which is what lets
        the caller keep writing ``SELECT <cols> FROM <this>`` unchanged.  The
        trailing semicolon is stripped because a subquery cannot contain one.
        """
        if not self.is_query:
            return quote_identifier(self.name)

        inner = self.sql.strip().rstrip(";").strip()
        return f"({inner}) AS {quote_identifier(alias)}"

    def select_sql(self, columns: list[str] | None = None) -> str:
        """Return a SELECT over this source, optionally projecting columns.

        Columns that are not valid identifiers are dropped rather than quoted
        blindly: they cannot have come from this application's own column
        listing, so passing them on would only turn a UI bug into a SQL error.
        """
        valid = [column for column in (columns or []) if is_identifier(column)]
        projection = ", ".join(quote_identifier(column) for column in valid) if valid else "*"
        return f"SELECT {projection} FROM {self.from_clause()}"

    def page_sql(self, *, limit: int, offset: int) -> str:
        """Return one page of rows.

        LIMIT/OFFSET rather than the repository's rowid paging: a query result
        has no rowid, and a preview never scrolls deep enough for the cost of
        OFFSET to matter.
        """
        return f"{self.select_sql()} LIMIT {int(limit)} OFFSET {int(offset)}"

    def count_sql(self) -> str:
        """Return a statement counting the rows this source yields."""
        return f"SELECT COUNT(*) AS n FROM {self.from_clause()}"

    def columns_sql(self) -> str:
        """Return a statement that yields the column names and no rows."""
        return f"{self.select_sql()} LIMIT 0"
