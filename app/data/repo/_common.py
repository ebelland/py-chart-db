"""The vocabulary the repository is built out of.

Split out of ``sqlite_repo`` so the pieces of that module - saved queries,
maintenance, and whatever follows them - can be modules of their own
without importing the class they are part of (todo.txt N-5). Nothing here
knows about SqliteRepo: it is the SQL-shaped helpers, the two small result
types, and the decorator every method that touches the connection wears.

``sqlite_repo`` re-exports all of it, so ``from app.data.sqlite_repo import
DatabaseReport`` keeps working - the split is meant to be invisible from
outside.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ClassVar, Mapping

from app.data.data_source import is_identifier, quote_identifier
from app.logs.logger import applogger


def ensure_connection_wrapper(func):
    ''' This wrapper ensures that database to be connected'''
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._is_connected or self._con is None:
            self._connect()
        if not self._is_connected or self._con is None:
            applogger.critical("No active connection")
        return func(self, *args, **kwargs)
    return wrapper

# Regex: SQL statement that returns rows (SELECT, WITH, PRAGMA, EXPLAIN)
_RETURNS_ROWS_RE = re.compile(r"^\s*(select|with|pragma|explain)\b", re.IGNORECASE)

# A saved query exists to put rows on a chart, so it must only ever read.
#
# "Returns rows" is a weaker property than "changes nothing", and the two are
# easy to confuse: PRAGMA returns rows and also writes (``PRAGMA user_version =
# 5``), and SQLite lets a WITH clause introduce a DELETE or an UPDATE, so
# ``WITH x AS (SELECT 1) DELETE FROM readings`` passes a leading-keyword test
# while emptying a table.  Saved queries therefore get this stricter check
# rather than _RETURNS_ROWS_RE.
_SAVED_QUERY_OPENER_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

#: Statement keywords that write.  Checked against the whole statement, after
#: comments and string literals are removed, because they can appear well past
#: the opening keyword.
#: ``replace`` is deliberately absent: it is also SQLite's string function, and
#: ``SELECT replace(name, 'a', 'b')`` is perfectly read-only.  The statement
#: form is caught by _REPLACE_INTO_RE instead.
_WRITE_KEYWORDS: frozenset[str] = frozenset(
    {
        "alter", "analyze", "attach", "begin", "commit", "create", "delete",
        "detach", "drop", "insert", "pragma", "reindex", "release", "rollback",
        "savepoint", "update", "vacuum",
    }
)

_REPLACE_INTO_RE = re.compile(r"\breplace\s+into\b", re.IGNORECASE)

_SQL_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_SQL_LITERAL_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|\[[^\]]*\]|`[^`]*`")
_SQL_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _sql_without_comments_and_literals(sql: str) -> str:
    """Return ``sql`` with comments, quoted strings and quoted names removed.

    Keyword matching has to happen on code, not on content: a perfectly
    read-only query can carry the word "delete" inside a string literal or a
    column alias, and refusing that would be a false alarm.  Removing both
    first means the keyword scan only ever sees SQL.
    """
    return _SQL_LITERAL_RE.sub(" ", _SQL_COMMENT_RE.sub(" ", sql))


def is_read_only_select(sql: str) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for whether ``sql`` only reads data.

    ``reason`` is empty when ok.  Trailing semicolons are tolerated, but an
    actual second statement is not: ``SELECT 1; DELETE FROM t`` is rejected
    rather than silently truncated to its harmless first half.
    """
    text = str(sql or "").strip()
    if not text:
        return False, "The query is empty."

    code = _sql_without_comments_and_literals(text).strip()

    # Tolerate trailing semicolons and whitespace, then refuse anything that
    # still has a statement separator with SQL after it.
    body = code.rstrip().rstrip(";").rstrip()
    if ";" in body:
        return False, "Only a single statement can be saved as a query."

    if not _SAVED_QUERY_OPENER_RE.match(body):
        return False, "A saved query must be a SELECT (or WITH) statement."

    found = {word.lower() for word in _SQL_WORD_RE.findall(body)} & _WRITE_KEYWORDS
    if _REPLACE_INTO_RE.search(body):
        found = found | {"replace"}

    if found:
        listed = ", ".join(sorted(found))
        return False, (
            f"A saved query must only read data, but this one uses: {listed}."
        )

    return True, ""


def _loads_json(text: str | None) -> dict[str, Any]:
    """Parse JSON to dict; return {} on empty/invalid/non-dict input."""
    try:
        return {} if not text else dict(json.loads(text))
    except Exception:
        return {}


def _dumps_json(obj: Mapping[str, Any] | None = None) -> str:
    """Serialize mapping to JSON string (UTF-8)."""
    return json.dumps(dict(obj or {}), ensure_ascii=False)


def _quote_ident(name: str) -> str:
    """Quote an identifier, logging anything that is not a plain name.

    The quoting itself is ``data_source.quote_identifier``; what this adds is
    the check, because a name reaching SQL from outside the application's own
    column listing is worth a log line even though quoting makes it safe.
    """
    if not is_identifier(name):
        applogger.error(f"Invalid table name: {name!r}")
    return quote_identifier((name or "").strip())


def _is_ident(name: str) -> bool:
    """Return True if name is a valid SQLite identifier."""
    return is_identifier(name)


@dataclass(slots=True)
class SavedQuery:
    """Stored SQL query descriptor."""

    id: int | None
    name: str
    sql: str
    settings: dict[str, Any]


@dataclass(slots=True)
class DatabaseReport:
    """What ``check_database`` found, grouped by how bad it is.

    The split matters: corruption and dangling references are problems the user
    must act on, while an unreferenced table is usually just data waiting to be
    charted.  Reporting them at the same severity would train the user to
    ignore the whole report.
    """

    integrity_errors: list[str] = field(default_factory=list)
    foreign_key_errors: list[str] = field(default_factory=list)
    dangling_series: list[str] = field(default_factory=list)
    dangling_links: list[str] = field(default_factory=list)
    orphan_descriptors: list[str] = field(default_factory=list)
    unreferenced_tables: list[str] = field(default_factory=list)

    @property
    def problems(self) -> list[str]:
        """Every finding that needs the user to do something."""
        return [
            *self.integrity_errors,
            *self.foreign_key_errors,
            *self.orphan_descriptors,
            *self.dangling_series,
            *self.dangling_links,
        ]

    @property
    def is_healthy(self) -> bool:
        """True when nothing actionable was found."""
        return not self.problems

    def summary(self) -> str:
        """Return a one-line summary suitable for a status bar."""
        if self.is_healthy:
            extra = (
                f" ({len(self.unreferenced_tables)} unreferenced table(s))"
                if self.unreferenced_tables
                else ""
            )
            return f"Database check passed{extra}."
        return f"Database check found {len(self.problems)} problem(s)."

    #: Every finding list, with the heading it is reported under.  One table,
    #: so the log, the summary and the detail pane cannot come to disagree
    #: about what was looked for - adding a check means adding it here.
    SECTIONS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("integrity_errors", "Integrity"),
        ("foreign_key_errors", "Foreign keys"),
        ("orphan_descriptors", "Orphan descriptors"),
        ("dangling_series", "Series reading a missing table"),
        ("dangling_links", "Import links to a missing table"),
        ("unreferenced_tables", "Unreferenced tables (not a problem)"),
    )

    def details(self) -> str:
        """Return every finding, grouped under its heading, one per line.

        The summary counts them; this says what they were.  Nothing is
        truncated: a check that reports "47 problems" and then shows twenty of
        them is a check the user cannot finish acting on, and the pane this
        goes into scrolls.

        Unreferenced tables are included even though they are not problems.
        "3 unreferenced table(s)" with no way to see which three is a line
        nobody can do anything with.
        """
        lines: list[str] = []
        for attribute, heading in self.SECTIONS:
            findings = getattr(self, attribute)
            if not findings:
                continue
            if lines:
                lines.append("")
            lines.append(f"{heading}:")
            lines.extend(f"  {finding}" for finding in findings)
        return "\n".join(lines)

    def log(self) -> None:
        """Write the whole report to the log, worst first."""
        for message in self.integrity_errors:
            applogger.error(
                "Database integrity: %s", message, show_dialog=False, raise_error=False
            )
        for message in self.foreign_key_errors:
            applogger.error(
                "Foreign key: %s", message, show_dialog=False, raise_error=False
            )
        for message in self.orphan_descriptors:
            applogger.warning(
                "Orphan descriptor: %s", message, show_dialog=False, raise_error=False
            )
        for message in self.dangling_series:
            applogger.warning(
                "Dangling reference: %s", message, show_dialog=False, raise_error=False
            )
        for message in self.dangling_links:
            applogger.warning(
                "Dangling reference: %s", message, show_dialog=False, raise_error=False
            )
        if self.unreferenced_tables:
            applogger.info(
                "Tables not used by any chart or import link: %s",
                ", ".join(self.unreferenced_tables),
            )
        applogger.info(self.summary())

