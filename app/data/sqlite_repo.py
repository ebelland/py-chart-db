"""SQLite repository for Data Hub.

Handles DataFrame queries, imports, and figure/axis/series descriptors.
Optimized for speed with connection pooling and efficient SQL generation.

It also owns the **series DataFrame cache**, the single biggest lever on render
time: without it every property tweak re-runs every series query against SQLite.
See ``series_df`` for the invalidation contract.
"""
from __future__ import annotations

import json
import re
import sqlite3
import struct
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas._typing import DtypeArg

import app.data.descriptors
from app.data.data_source import DataSource, is_identifier, quote_identifier
from app.logs.logger import applogger
from app.utils.config import load_config
from functools import wraps

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


# Series cache defaults, overridable from config.json under "series_cache".
_SERIES_CACHE_DEFAULT_ENABLED = True
_SERIES_CACHE_DEFAULT_MAX_ENTRIES = 64


@dataclass(slots=True)
class SqliteRepo:
    """SQLite repository for Data Hub.

    Manages:
      - DataFrame queries and imports
      - Saved import links (source path + settings)
      - Figure/axis/series descriptor schemas
      - The series DataFrame cache used by the render pipeline

    Optimizations:
      - Persistent connection with WAL mode
      - Memory-mapped I/O and large cache
      - Prepared statements (row_factory cached)
      - Bounded LRU cache of series DataFrames (see ``series_df``)
    """

    db_path: Path
    _con: sqlite3.Connection | None = None
    _is_connected: bool = False
    _preview_savepoint_name: str | None = None

    # --- series DataFrame cache -----------------------------------------
    _series_cache: OrderedDict[str, pd.DataFrame] = field(default_factory=OrderedDict)
    _series_cache_stamp: tuple[int, int, int] | None = None
    _series_cache_enabled: bool = _SERIES_CACHE_DEFAULT_ENABLED
    _series_cache_max_entries: int = _SERIES_CACHE_DEFAULT_MAX_ENTRIES
    _series_cache_hits: int = 0
    _series_cache_misses: int = 0


    # =====================================================================
    # Path helpers
    # =====================================================================

    @staticmethod
    def ensure_dhub_extension(path: Path) -> Path:
        """Enforce .dhub extension; use existing path if found, else add .dhub."""
        p = path.expanduser().resolve()
        if p.exists():
            return p
        return p if p.suffix.lower() == ".dhub" else p.with_suffix(".dhub")

    @classmethod
    def create_empty(cls, path: Path) -> Path:
        """Create an empty database at *path* and return the file written.

        Connecting is what creates a database - the file, the pragmas and the
        system tables all come from ``_connect`` - so this exists to make that
        happen *now*, at a point where a failure can still be handled, rather
        than on the first query.  Startup uses it when the remembered database
        has been moved or deleted: an empty project the user can import into
        beats a file dialog they have nothing to pick in.

        The connection is closed again, because the caller opens its own.
        """
        repo = cls(db_path=cls.ensure_dhub_extension(Path(path)))
        try:
            repo._connect()
        finally:
            repo.close()
        applogger.info("Created empty database: %s", repo.db_path)
        return repo.db_path

    # =====================================================================
    # Connection lifecycle
    # =====================================================================

    def _connect(self):
        """Establish SQLite connection with performance pragmas.
        
        Enables WAL mode, memory-mapped I/O, and large cache for speed.
        Creates required system tables if missing.
        """
        # Close existing connection if open
        if self._con is not None:
            self._con.close()
            del self._con

        db_path = self.ensure_dhub_extension(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Establish connection
        self._con = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode
        )
        self._is_connected = False

        try:
            # Performance pragmas (must be outside transaction)
            self._con.execute("PRAGMA journal_mode = WAL;")
            self._con.execute("PRAGMA synchronous = NORMAL;")  # Fast + safe with WAL
            self._con.execute("PRAGMA foreign_keys = ON;")
            self._con.execute("PRAGMA busy_timeout = 30000;")

            # In-transaction pragmas
            with self._con:
                self._con.execute("PRAGMA temp_store = MEMORY;")
                self._con.execute("PRAGMA cache_size = -65536;")  # 64 MiB
                self._con.execute("PRAGMA mmap_size = 268435456;")  # 256 MiB
                self._con.execute("PRAGMA wal_autocheckpoint = 1000;")
                self._con.execute("PRAGMA journal_size_limit = 67108864;")
                self._con.execute("PRAGMA optimize;")
        except Exception:
            applogger.error("Failed to apply performance pragmas", exc_info=True)
        finally:
            applogger.debug(f"Connected to {db_path}")

        self._is_connected = True
        # Cache row_factory for performance
        self._con.row_factory = sqlite3.Row

        # A new connection means new counters; nothing cached is trustworthy.
        self.invalidate_series_cache()
        self._load_series_cache_settings()

        # Create system tables
        self._create_system_tables()

    # =====================================================================
    # Series DataFrame cache
    # =====================================================================

    def _load_series_cache_settings(self) -> None:
        """Read the optional ``series_cache`` section from config.json.

        Example::

            "series_cache": { "enabled": true, "max_entries": 64 }

        Why: the cache is the one optimisation that can produce wrong output if
        invalidation ever misses a write, so it must be switchable off without
        touching code.
        """
        self._series_cache_enabled = _SERIES_CACHE_DEFAULT_ENABLED
        self._series_cache_max_entries = _SERIES_CACHE_DEFAULT_MAX_ENTRIES

        try:
            section = load_config().get("series_cache", {})
        except Exception:
            applogger.exception("Failed to read series_cache config; using defaults")
            return

        if not isinstance(section, Mapping):
            return

        self._series_cache_enabled = bool(section.get("enabled", _SERIES_CACHE_DEFAULT_ENABLED))
        try:
            self._series_cache_max_entries = max(1, int(section.get("max_entries", _SERIES_CACHE_DEFAULT_MAX_ENTRIES)))
        except (TypeError, ValueError):
            applogger.warning(
                "Invalid series_cache.max_entries=%r; using %d",
                section.get("max_entries"),
                _SERIES_CACHE_DEFAULT_MAX_ENTRIES,
                show_dialog=False,
                raise_error=False,
            )

    def _database_stamp(self) -> tuple[int, int, int]:
        """Return a triple that changes whenever query results could change.

        - ``Connection.total_changes`` counts every row inserted, updated or
          deleted through *this* connection, so it catches writes that
          ``PRAGMA data_version`` deliberately ignores.
        - ``PRAGMA data_version`` changes when *another* connection commits.
        - ``PRAGMA schema_version`` changes on any DDL (CREATE / DROP / ALTER),
          which neither of the other two reports.

        Together they need no manual bookkeeping on write paths, which removes
        the whole class of "invalidation missed a writer" bugs.  Cost is ~8 µs
        against the ~35 ms the cache saves per series.
        """
        con = self._con
        if con is None:
            return (0, 0, 0)

        data_version = int(con.execute("PRAGMA data_version").fetchone()[0])
        schema_version = int(con.execute("PRAGMA schema_version").fetchone()[0])
        return (int(con.total_changes), data_version, schema_version)

    def invalidate_series_cache(self) -> None:
        """Drop every cached series DataFrame."""
        self._series_cache.clear()
        self._series_cache_stamp = None

    @property
    def series_cache_stats(self) -> dict[str, int | bool]:
        """Return cache counters; used by tests and the benchmark."""
        return {
            "enabled": self._series_cache_enabled,
            "entries": len(self._series_cache),
            "max_entries": self._series_cache_max_entries,
            "hits": self._series_cache_hits,
            "misses": self._series_cache_misses,
        }

    @ensure_connection_wrapper
    def series_df(self, sql: str) -> pd.DataFrame:
        """Return the DataFrame for a series query, cached per database state.

        The cache is keyed by SQL text and wholesale invalidated whenever
        ``_database_stamp`` changes, so a hit can only ever be served for the
        exact database state that produced it.

        Contract: the returned frame is a shallow copy.  Adding or dropping
        columns on it is safe; mutating values in place is not, because the
        underlying blocks are shared with the cached frame.
        """
        sql_text = (sql or "").strip()
        if not sql_text:
            return pd.DataFrame()

        assert self._con is not None

        if not self._series_cache_enabled:
            return pd.read_sql_query(sql_text, self._con)

        stamp = self._database_stamp()
        if stamp != self._series_cache_stamp:
            self._series_cache.clear()
            self._series_cache_stamp = stamp

        cached = self._series_cache.get(sql_text)
        if cached is not None:
            self._series_cache_hits += 1
            self._series_cache.move_to_end(sql_text)
            return cached.copy(deep=False)

        self._series_cache_misses += 1
        frame = pd.read_sql_query(sql_text, self._con)
        self._series_cache[sql_text] = frame

        # Bound the cache; the oldest entry is the least recently used one.
        while len(self._series_cache) > self._series_cache_max_entries:
            self._series_cache.popitem(last=False)

        return frame.copy(deep=False)


    @ensure_connection_wrapper
    def _create_system_tables(self):
        """Create required system tables if missing."""

        # Import links: maps table names to source + settings
        assert self._con
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __import_links__ (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name    TEXT NOT NULL UNIQUE,
                source_path   TEXT NOT NULL,
                settings_json TEXT NOT NULL
            );
            """
        )

        # Figure descriptors: grid layout and options
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __figure_descriptors__ (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                nrows       INTEGER NOT NULL DEFAULT 1,
                ncols       INTEGER NOT NULL DEFAULT 1,
                options_json TEXT
            );
            """
        )

        # Ensure options_json exists on older databases (schema migration)
        cols = {
            row[1] for row in self._con.execute(
                "PRAGMA table_info(__figure_descriptors__)"
            ).fetchall()
        }
        if "options_json" not in cols:
            self._con.execute(
                "ALTER TABLE __figure_descriptors__ ADD COLUMN options_json TEXT"
            )

        # Axis descriptors: chart type and labels
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __axis_descriptors__ (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                figure_id     INTEGER NOT NULL,
                axis_index    INTEGER NOT NULL,
                chart_type    TEXT NOT NULL,
                title         TEXT,
                x_label       TEXT,
                y_label       TEXT,
                z_label       TEXT,
                options_json  TEXT,
                FOREIGN KEY(figure_id) REFERENCES __figure_descriptors__(id)
                    ON DELETE CASCADE,
                UNIQUE(figure_id, axis_index)
            );
            """
        )

        # Series descriptors: data queries and styling
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __series_descriptors__ (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                axis_id       INTEGER NOT NULL,
                series_index  INTEGER NOT NULL,
                name          TEXT,
                sql_query     TEXT NOT NULL,
                roles         TEXT,
                style_json    TEXT,
                FOREIGN KEY(axis_id) REFERENCES __axis_descriptors__(id)
                    ON DELETE CASCADE,
                UNIQUE(axis_id, series_index)
            );
            """
        )

        # Table descriptors: user table metadata
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __table_descriptors__ (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                notes       TEXT,
                info_json   TEXT
            );
            """
        )

        # Saved queries: named SQL snippets + optional UI/settings JSON
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __queries__ (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL UNIQUE,
                sql           TEXT NOT NULL,
                settings_json TEXT
            );
            """
        )

        # Core indexes for fast descriptor and saved-query lookups
        self._con.executescript(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_queries_name
                ON __queries__ (name);

            CREATE INDEX IF NOT EXISTS idx_import_links_table_name
                ON __import_links__ (table_name);

            CREATE INDEX IF NOT EXISTS idx_table_descriptors_name
                ON __table_descriptors__ (name);

            CREATE INDEX IF NOT EXISTS idx_figures_name
                ON __figure_descriptors__ (name);

            CREATE INDEX IF NOT EXISTS idx_axes_figure_id
                ON __axis_descriptors__ (figure_id);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_axes_figure_axis
                ON __axis_descriptors__ (figure_id, axis_index);

            CREATE INDEX IF NOT EXISTS idx_series_axis_id
                ON __series_descriptors__ (axis_id);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_series_axis_series
                ON __series_descriptors__ (axis_id, series_index);
            """
        )


    @contextmanager
    @ensure_connection_wrapper
    def connect(self):
        """Context manager yielding the shared SQLite connection."""
        yield self._con


    # =====================================================================
    # Preview savepoints
    # =====================================================================

    @property
    def preview_savepoint_active(self) -> bool:
        """True when a UI preview SAVEPOINT is currently open."""
        return bool(self._preview_savepoint_name)

    @ensure_connection_wrapper
    def begin_preview_savepoint(self, name: str | None = None) -> str:
        """Start a rollback-able SAVEPOINT for dialog Preview changes.

        Repository methods often call commit() after writes. During an active
        preview savepoint, those commits are intentionally suppressed by
        ``_commit()`` so Close/Cancel can roll back all preview changes at once.
        """
        assert self._con is not None
        if self._preview_savepoint_name:
            self.rollback_preview_savepoint()
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", str(name or f"preview_{uuid.uuid4().hex}"))
        if not safe_name or safe_name[0].isdigit():
            safe_name = f"preview_{safe_name}"
        self._con.execute(f"SAVEPOINT {safe_name}")
        self._preview_savepoint_name = safe_name
        applogger.info("Started preview savepoint: %s", safe_name)
        return safe_name

    @ensure_connection_wrapper
    def release_preview_savepoint(self) -> bool:
        """Commit the active Preview SAVEPOINT.

        Returns True when SQLite released the savepoint, False when Python state
        was stale and the savepoint was already gone.
        """
        assert self._con is not None
        sp = self._preview_savepoint_name
        if not sp:
            return False
        released = False
        try:
            self._con.execute(f"RELEASE SAVEPOINT {sp}")
            released = True
            applogger.info("Released preview savepoint: %s", sp)
        except sqlite3.OperationalError as exc:
            if "no such savepoint" not in str(exc).lower():
                raise
            applogger.warning(
                "Preview savepoint %s was already gone during release; clearing stale state.",
                sp,
            )
        finally:
            self._preview_savepoint_name = None
        if released:
            self._con.commit()
        return released

    @ensure_connection_wrapper
    def rollback_preview_savepoint(self) -> bool:
        """Undo the active Preview SAVEPOINT.

        Returns True when SQLite rolled back the savepoint, False when Python
        state was stale and the savepoint was already gone. Callers can use the
        False result to perform explicit temporary-artifact cleanup.
        """
        assert self._con is not None
        sp = self._preview_savepoint_name
        if not sp:
            return False
        rolled_back = False
        try:
            self._con.execute(f"ROLLBACK TO SAVEPOINT {sp}")
            self._con.execute(f"RELEASE SAVEPOINT {sp}")
            rolled_back = True
            applogger.info("Rolled back preview savepoint: %s", sp)
        except sqlite3.OperationalError as exc:
            if "no such savepoint" not in str(exc).lower():
                raise
            applogger.warning(
                "Preview savepoint %s was already gone during rollback; clearing stale state.",
                sp,
            )
        finally:
            self._preview_savepoint_name = None
        return rolled_back

    def _commit(self) -> None:
        """Commit unless a UI preview savepoint is open.

        Savepoint-backed Preview must survive repository helper methods that
        normally call commit() after each write. While preview is active, writes
        remain inside the savepoint and are either released on Apply/OK or rolled
        back on Close/Cancel.
        """
        if self._con is None:
            return
        if self._preview_savepoint_name:
            return
        self._con.commit()

    # =====================================================================
    # Transactions
    # =====================================================================

    @contextmanager
    @ensure_connection_wrapper
    def transaction(self, *, immediate: bool = False):
        """Context manager for transactions with SAVEPOINT support.
        
        - No active tx: start new transaction (optionally IMMEDIATE), then commit/rollback.
        - Active tx: use SAVEPOINT for nested safety.
        """

        if not self._con.in_transaction:  # pyright: ignore[reportOptionalMemberAccess]
            if immediate:
                self._con.execute("BEGIN IMMEDIATE")  # pyright: ignore[reportOptionalMemberAccess]
                try:
                    yield
                except Exception:
                    self._con.rollback()  # pyright: ignore[reportOptionalMemberAccess]
                    raise
                else:
                    self._commit()  # pyright: ignore[reportOptionalMemberAccess]
            else:
                with self._con:  # pyright: ignore[reportOptionalContextManager]
                    yield
            return

        # Nested transaction: use SAVEPOINT
        sp = f"sp_{uuid.uuid4().hex}"
        self._con.execute(f"SAVEPOINT {sp}")  # pyright: ignore[reportOptionalMemberAccess]
        try:
            yield
        except Exception:
            self._con.execute(f"ROLLBACK TO {sp}")  # pyright: ignore[reportOptionalMemberAccess]
            self._con.execute(f"RELEASE {sp}")  # pyright: ignore[reportOptionalMemberAccess]
            raise
        else:
            self._con.execute(f"RELEASE {sp}")  # pyright: ignore[reportOptionalMemberAccess]

    # =====================================================================
    # Table listing and introspection
    # =====================================================================


    def list_user_tables(self) -> pd.DataFrame:
        """List all user tables with link status and notes.
        
        Returns DataFrame with columns:
          - Table: table name
          - has_link: bool indicating if import link exists
          - Notes: user notes (or None)
          - source_path: import source path (or None)
        """
        sql = """
        SELECT
            sm.name AS "Table",
            (LENGTH(COALESCE(il.source_path, '')) > 0) AS has_link,
            td.notes AS "Notes",
            il.source_path AS "source_path"
        FROM sqlite_master sm
        LEFT JOIN __import_links__ il ON il.table_name = sm.name
        LEFT JOIN __table_descriptors__ td ON td.name = sm.name
        WHERE sm.type = 'table'
          AND sm.name NOT LIKE '__%__' ESCAPE '_'
          AND sm.name NOT LIKE 'sqlite_%'
        ORDER BY sm.name
        """
        df = self.query_df(sql)
        if df is None or df.empty:
            return pd.DataFrame(columns=["Table", "has_link", "Notes", "source_path"])
        df["has_link"] = df["has_link"].astype(bool)
        return df

    # =====================================================================
    # Data sources: physical tables and saved queries
    # =====================================================================

    def list_data_sources(self) -> pd.DataFrame:
        """List tables and saved queries as one addressable list.

        Same columns as ``list_user_tables`` plus ``kind``, so the table list
        can render both without a second code path.  Saved queries have no
        import link and no source file; their SQL goes in ``source_path`` so it
        can be shown as a tooltip.
        """
        tables = self.list_user_tables()
        tables = tables.assign(kind="table")

        queries = self.list_queries()
        if not queries:
            return tables

        query_frame = pd.DataFrame(
            {
                "Table": [query.name for query in queries],
                "has_link": [False] * len(queries),
                "Notes": [None] * len(queries),
                "source_path": [query.sql for query in queries],
                "kind": ["query"] * len(queries),
            }
        )

        combined = pd.concat([tables, query_frame], ignore_index=True)
        return combined.sort_values("Table", key=lambda s: s.str.lower(), ignore_index=True)

    @ensure_connection_wrapper
    def get_data_source(self, name: str) -> DataSource | None:
        """Resolve a name to a table or a saved query.

        Tables win over queries when a name is used twice: the physical object
        is the one the rest of SQLite would resolve, so shadowing it here would
        make the preview and a hand-written query disagree.
        """
        assert self._con is not None
        clean = str(name or "").strip()
        if not clean:
            return None

        if self.check_if_table_exists(clean):
            return DataSource.table(clean)

        saved = self.get_query(clean)
        if saved is not None:
            return DataSource.query(saved.name, saved.sql)

        return None

    @ensure_connection_wrapper
    def data_source_columns(self, source: DataSource) -> list[str]:
        """Return the column names a source yields, without reading any rows."""
        assert self._con is not None
        if not source.is_query:
            return self.get_columns(source.name)

        try:
            cursor = self._con.execute(source.columns_sql())
            if cursor.description is None:
                return []
            return [str(item[0]) for item in cursor.description if item and item[0]]
        except Exception:
            applogger.exception("Failed to read columns of query '%s'", source.name)
            return []

    @ensure_connection_wrapper
    def data_source_row_count(self, source: DataSource) -> int:
        """Return how many rows a source yields, or 0 when it cannot run."""
        assert self._con is not None
        try:
            row = self._con.execute(source.count_sql()).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            applogger.exception("Failed to count rows of source '%s'", source.name)
            return 0

    @ensure_connection_wrapper
    def data_source_page(self, source: DataSource, *, limit: int, offset: int) -> pd.DataFrame:
        """Return one page of a source's rows."""
        assert self._con is not None
        try:
            return pd.read_sql_query(source.page_sql(limit=limit, offset=offset), self._con)
        except Exception:
            applogger.exception("Failed to read source '%s'", source.name)
            return pd.DataFrame()

    def validate_query(self, sql: str) -> tuple[bool, str]:
        """Return (ok, message) for a candidate saved query.

        The statement is prepared and run with ``LIMIT 0``: that catches syntax
        errors, unknown tables and unknown columns without materialising a
        single row, so validating a query over a huge table is instant.
        """
        text = str(sql or "").strip().rstrip(";").strip()

        ok, reason = is_read_only_select(text)
        if not ok:
            return False, reason

        if self._con is None:
            self._connect()
        assert self._con is not None

        try:
            cursor = self._con.execute(f"SELECT * FROM ({text}) AS _probe LIMIT 0")
        except Exception as exc:
            return False, str(exc)

        columns = [str(item[0]) for item in (cursor.description or []) if item]
        if not columns:
            return False, "The query returns no columns."
        return True, f"{len(columns)} column(s): {', '.join(columns)}"

    @ensure_connection_wrapper
    def get_columns(self, table: str) -> list[str]:
        """Return column names for a table.
        
        Uses PRAGMA table_info to avoid reserved word issues.
        Returns empty list on error.
        """
        assert self._con is not None
        try:
            rows = self._con.execute(
                f"PRAGMA table_info({_quote_ident(table)})"
            ).fetchall()
            return [str(r[1]) for r in rows if len(r) > 1]
        except Exception:
            return []

    @ensure_connection_wrapper
    def table_info(self, table: str) -> list[sqlite3.Row]:
        """Return SQLite PRAGMA table_info rows for a user table.

        UI widgets use this for schema display. Keep the database access in the
        repository so callers do not execute PRAGMA statements directly.
        """
        assert self._con is not None
        try:
            return list(
                self._con.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            )
        except Exception:
            applogger.exception("Failed to read table schema for %s", table)
            return []

    @ensure_connection_wrapper
    def ensure_preview_state_columns(self, table_name: str) -> None:
        """Create/update temporary preview state columns for Hide preview.

        ``Hide`` is the editable runtime column used by chart filtering.
        ``__DataHubPreviewHide`` stores the pre-preview Hide state so Preview can
        be rolled back without relying on UI-side SQL or direct connection use.
        """
        assert self._con is not None
        table_sql = _quote_ident(table_name)
        preview_col = _quote_ident("__DataHubPreviewHide")
        self.ensure_hide_column(table_name)
        self.ensure_column(
            table_name=table_name,
            col_name="__DataHubPreviewHide",
            col_type="INTEGER",
        )
        self._con.execute(
            f'UPDATE {table_sql} SET {preview_col} = COALESCE("Hide", 0)'
        )
        self._commit()

    @ensure_connection_wrapper
    def restore_preview_state_columns(self, table_name: str) -> None:
        """Restore Hide values from the temporary preview state column."""
        assert self._con is not None
        table_sql = _quote_ident(table_name)
        preview_col = _quote_ident("__DataHubPreviewHide")
        columns = {str(row[1]) for row in self.table_info(table_name)}
        if "__DataHubPreviewHide" not in columns:
            return
        self.ensure_hide_column(table_name)
        self._con.execute(
            f'UPDATE {table_sql} SET "Hide" = COALESCE({preview_col}, 0)'
        )
        self._commit()

    @ensure_connection_wrapper
    def drop_preview_state_columns(self, table_name: str) -> None:
        """Drop temporary preview state columns created for Hide preview."""
        assert self._con is not None
        columns = {str(row[1]) for row in self.table_info(table_name)}
        if "__DataHubPreviewHide" not in columns:
            return
        self._con.execute(
            f'ALTER TABLE {_quote_ident(table_name)} '
            f'DROP COLUMN {_quote_ident("__DataHubPreviewHide")}'
        )
        self._commit()

    @ensure_connection_wrapper
    def set_table_notes(self, table: str, notes: str) -> None:
        """Set notes for a table in __table_descriptors__."""
        assert self._con is not None
        self._con.execute(
            """
            INSERT INTO __table_descriptors__ (name, notes)
            VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET notes = excluded.notes
            """,
            (table, notes),
        )
        self._commit()

    # =====================================================================
    # Query execution
    # =====================================================================
    @ensure_connection_wrapper
    def _get_sys_table_tuples(self, table_name: str) -> list[tuple[int, str]]:
        """Fetch (id, name) tuples from a system table. Helper for fast lookups."""
        assert self._con is not None
        try:
            rows = self._con.execute(
                f"SELECT id, name FROM {table_name} ORDER BY id"
            ).fetchall()
            return [(int(r[0]), str(r[1])) for r in rows] if rows else []
        except Exception:
            return []

    def get_figures(self) -> list[tuple[int, str]]:
        """Return figure descriptors as (id, name) tuples."""
        return self._get_sys_table_tuples("__figure_descriptors__")

    def get_links(self) -> list[tuple[int, str]]:
        """Return import links as (id, table_name) tuples."""
        return self._get_sys_table_tuples("__import_links__")
    
    @ensure_connection_wrapper
    def col_count(self, table: str) -> int:
        """Return column count for a table (or 0 on error)."""
        assert self._con is not None
        try:
            columns = self._con.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall()
            return len(columns)
        except Exception:
            return 0

    @ensure_connection_wrapper
    def row_count(self, table: str) -> int:
        """Return row count for a table (or 0 on error)."""
        assert self._con is not None
        try:
            row = self._con.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    @ensure_connection_wrapper
    def get_table_link(self, table: str) -> dict[str, Any] | None:
        """Get the import link for a table, or None if not found."""
        assert self._con is not None
        row = self._con.execute(
            "SELECT id, source_path, settings_json FROM __import_links__ WHERE table_name = ?",
            (table,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": int(row["id"]),
            "source_path": str(row["source_path"]),
            "settings": _loads_json(row["settings_json"]),
        }

    @ensure_connection_wrapper
    def table_has_link(self, table: str) -> bool:
        """Check if table has an associated import link."""
        assert self._con is not None
        try:
            row = self._con.execute(
                "SELECT 1 FROM __import_links__ WHERE table_name = ? LIMIT 1",
                (table,),
            ).fetchone()
            return row is not None
        except Exception:
            return False

    @ensure_connection_wrapper
    def query_df(self, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
        """Execute SQL and return DataFrame (or empty DF for non-SELECT).
        
        Detects SELECT/WITH/PRAGMA/EXPLAIN and returns data.
        For DDL/DML (CREATE/INSERT/UPDATE/DELETE), executes and returns empty DF.
        """
        sql_text = (sql or "").strip()
        if not sql_text or self._con is None:
            return pd.DataFrame()
        assert self._con is not None

        if _RETURNS_ROWS_RE.match(sql_text):
            return pd.read_sql_query(sql_text, self._con, params=params or ())
        else:
            self._con.execute(sql_text, params or ())
            return pd.DataFrame()

    # =====================================================================
    # DataFrame import
    # =====================================================================
    @ensure_connection_wrapper
    def import_dataframe(
        self,
        df: pd.DataFrame,
        *,
        table_name: str,
        normalize_columns: bool = True,
        dtype_overrides: DtypeArg | None = None,
    ) -> int:
        """Import DataFrame to SQLite table.
        
        Args:
            df: DataFrame to import
            table_name: destination table name
            normalize_columns: if True, convert column names to lowercase with underscores
            dtype_overrides: optional dtype mappings for columns
            
        Returns:
            Number of rows imported (or 0 on error)
        """
        table = table_name.strip()
        table_q = _quote_ident(table)

        # Normalize column names if requested
        if normalize_columns:
            df = df.copy()
            df.columns = [
                str(c).strip().lower().replace(" ", "_") for c in df.columns
            ]

        assert self._con is not None

        # Drop existing table to avoid conflicts
        self._con.execute(f"DROP TABLE IF EXISTS {table_q}")

        # Import using pandas.to_sql
        df.to_sql(
            table,
            self._con,
            if_exists="append",
            index=False,
            dtype=dtype_overrides,
        )
        applogger.info(f"Imported {df.shape[0]} rows to {table_name}")
        return int(df.shape[0])


    # =====================================================================
    # Import links management
    # =====================================================================
    @ensure_connection_wrapper
    def upsert_link(
        self, *, table_name: str, source_path: str, settings: dict[str, Any]
    ) -> int | None:
        """Insert or update import link; return link id (or None on error)."""
        assert self._con is not None

        self._con.execute(
            """
            INSERT INTO __import_links__ (table_name, source_path, settings_json)
            VALUES (?, ?, ?)
            ON CONFLICT(table_name) DO UPDATE SET
                source_path = excluded.source_path,
                settings_json = excluded.settings_json
            """,
            (table_name, source_path, _dumps_json(settings)),
        )
        row = self._con.execute(
            "SELECT id FROM __import_links__ WHERE table_name = ?",
            (table_name,),
        ).fetchone()

        if row is None:
            applogger.error("Failed to retrieve link after upsert")
            return None
        self._commit()
        return int(row["id"])

    @ensure_connection_wrapper
    def get_import_link(self, link_id: int) -> dict[str, Any]:
        """Fetch import link by id. Raises KeyError if not found."""
        assert self._con is not None

        row = self._con.execute(
            "SELECT * FROM __import_links__ WHERE id = ?",
            (int(link_id),),
        ).fetchone()
        if row is None:
            applogger.error(f"Link not found: {link_id}")
            return {}
        return {
            "id": int(row["id"]),
            "table_name": str(row["table_name"]),
            "source_path": str(row["source_path"]),
            "settings": _loads_json(row["settings_json"]),
        }

    @ensure_connection_wrapper
    def delete_link(self, link_id: int) -> None:
        """Delete import link by id."""
        assert self._con is not None
        self._con.execute("DELETE FROM __import_links__ WHERE id = ?", (link_id,))
        self._commit()

    # =====================================================================
    # Figure/axis/series deletion
    # =====================================================================
    @ensure_connection_wrapper
    def get_figure_descriptor(self, figure_id: int) -> app.data.descriptors.FigureDescriptor|None:
        """Get figure descriptor as dict (never None)."""
        assert self._con is not None
        row = self._con.execute(
            "SELECT * FROM __figure_descriptors__ WHERE id = ?",
            (int(figure_id),),
        ).fetchone()
        if row is None:
            return None
        return   app.data.descriptors.FigureDescriptor(
            id=int(row["id"]),
            name=str(row["name"]),
            nrows=int(row["nrows"]),
            ncols=int(row["ncols"]),
            options=_loads_json(row["options_json"]) if "options_json" in row.keys() else {},
            axes=[],
        )
    
    @ensure_connection_wrapper
    def get_figure_title(self, figure_id: int) -> str | None:
        """Get the name/title of a figure descriptor."""
        assert self._con is not None
        row = self._con.execute(
            "SELECT name FROM __figure_descriptors__ WHERE id = ?",
            (int(figure_id),),
        ).fetchone()
        return str(row[0]) if row else None

    @ensure_connection_wrapper
    def delete_figure(self, figure_id: int) -> None:
        """Delete figure and all associated axes and series (cascade)."""
        assert self._con is not None

        with self.transaction(immediate=True):
            # Foreign key constraint handles cascade automatically
            axes= self.get_axes(figure_id)
            if axes is not None:
                for ax in axes:
                    self.delete_axis(axis_id=int(ax[0]))
            self._con.execute(
                "DELETE FROM __figure_descriptors__ WHERE id = ?", (figure_id,)
            )

    @ensure_connection_wrapper
    def delete_axis(self, axis_id: int) -> None:
        """Delete axis and all associated series (cascade via FK)."""
        assert self._con is not None
        series=self.get_series(axis_id)
        if series is not None:
            for s in series:
                self.delete_series(s[0])
        self._con.execute(
            "DELETE FROM __axis_descriptors__ WHERE id = ?", (int(axis_id),)
        )
        self._commit()

    @ensure_connection_wrapper
    def delete_series(self, series_id: int) -> None:
        """Delete series from database."""
        assert self._con is not None
        self._con.execute("DELETE FROM __series_descriptors__ WHERE id = ?", (int(series_id),))
        self._commit()

    # =====================================================================
    # UI options persistence
    # =====================================================================
    @ensure_connection_wrapper
    def get_figure_options(self, figure_id: int) -> dict[str, Any]:
        """Return figure UI options as a dictionary."""
        assert self._con is not None

        row = self._con.execute(
            "SELECT options_json FROM __figure_descriptors__ WHERE id = ?",
            (int(figure_id),),
        ).fetchone()
        return _loads_json(row["options_json"]) if row else {}

    @ensure_connection_wrapper
    def set_figure_options(self, figure_id: int, options: dict[str, Any]) -> None:
        """Persist figure UI options."""
        assert self._con is not None

        self._con.execute(
            "UPDATE __figure_descriptors__ SET options_json = ? WHERE id = ?",
        (_dumps_json(options), int(figure_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    def set_figure_properties(self, figure_id: int, nrows:int, ncols:int,name:str, options: dict[str, Any]) -> None:
        """Persist figure UI options."""
        assert self._con is not None

        self._con.execute(
            "UPDATE __figure_descriptors__ SET options_json = ?, nrows = ?, ncols = ?, name= ? WHERE id = ?",
        (_dumps_json(options), nrows,ncols, name, figure_id),
        )
        self._commit()


    @ensure_connection_wrapper
    def get_axis_options(self, axis_id: int) -> dict[str, Any]:
        """Return axis UI options as a dictionary."""
        assert self._con is not None

        row = self._con.execute(
            "SELECT options_json FROM __axis_descriptors__ WHERE id = ?",
            (int(axis_id),),
        ).fetchone()
        return _loads_json(row["options_json"]) if row else {}

    @ensure_connection_wrapper
    def set_axis_options(self, axis_id: int, options: dict[str, Any]) -> None:
        """Persist axis UI options."""
        assert self._con is not None

        self._con.execute(
            "UPDATE __axis_descriptors__ SET options_json = ? WHERE id = ?",
            (_dumps_json(options), int(axis_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    def update_series_style(self, series_id: int, style: dict[str, Any]) -> None:
        """Persist series style_json."""
        assert self._con is not None

        self._con.execute(
            "UPDATE __series_descriptors__ SET style_json = ? WHERE id = ?",
            (_dumps_json(style), int(series_id)),
        )
        self._commit()
        
    @ensure_connection_wrapper
    def update_series_sql_query(self, series_id: int, sql_query: str) -> None:
        """Persist the SQL query for one series descriptor."""
        assert self._con is not None

        self._con.execute(
            """
            UPDATE __series_descriptors__
            SET sql_query = ?
            WHERE id = ?
            """,
            (str(sql_query or "").strip(), int(series_id)),
        )
        self._commit()

    # =====================================================================
    # Hide-column support for chart outlier filtering
    # =====================================================================
    
    def ensure_hide_column(self, table_name: str) -> None:
        """Ensure a boolean-compatible Hide column exists on a user data table."""
        self.ensure_column(table_name=table_name,col_name="Hide",col_type="INTEGER")
               
    def ensure_cluster_column(self, table_name: str) -> None:
        """Ensure a int-compatible ClusterId column exists on a user data table."""
        self.ensure_column(table_name=table_name,col_name="ClusterId",col_type="INTEGER")

    @ensure_connection_wrapper
    def ensure_column(self, table_name: str, col_name:str, col_type:str="INTEGER")-> None:
        """Ensure a int-compatible ClusterId column exists on a user data table."""
        assert self._con is not None
        table_sql = _quote_ident(table_name)
        columns = {
            str(row[1])
            for row in self._con.execute(f"PRAGMA table_info({table_sql})").fetchall()
        }
        if col_name not in columns:
            self._con.execute(
                f"ALTER TABLE {table_sql} ADD COLUMN {_quote_ident(col_name)} {col_type}" + " NOT NULL DEFAULT 0" if col_type=="INTEGER" else ""     
            )
            self._commit()      

    @ensure_connection_wrapper
    def get_series_sql_query(self, series_id: int) -> str | None:
        """Return the stored SQL of one series, or None when it does not exist."""
        assert self._con is not None
        row = self._con.execute(
            "SELECT sql_query FROM __series_descriptors__ WHERE id = ?",
            (int(series_id),),
        ).fetchone()
        return None if row is None else str(row["sql_query"] or "")

    @ensure_connection_wrapper
    def has_column(self, table_name: str, col_name: str) -> bool:
        """Return True when a table already has a column with this name."""
        assert self._con is not None
        return col_name in set(self.get_columns(table_name))

    @ensure_connection_wrapper
    def rename_table_column(self, table_name: str, old_name: str, new_name: str) -> None:
        """Rename one column of a user table."""
        assert self._con is not None
        self._con.execute(
            f"ALTER TABLE {_quote_ident(table_name)} "
            f"RENAME COLUMN {_quote_ident(old_name)} TO {_quote_ident(new_name)}"
        )
        self._commit()

    @ensure_connection_wrapper
    def snapshot_column(self, table_name: str, col_name: str, backup_name: str) -> bool:
        """Move a column aside under ``backup_name`` so it can be restored.

        Returns True when a snapshot was taken, False when the column did not
        exist (in which case restoring means simply dropping whatever replaced
        it).

        Why rename instead of copying the values: a rename is O(1) metadata and
        cannot run out of space or time on a large table, and it guarantees the
        restored column is byte-for-byte the original rather than a re-inserted
        approximation of it.
        """
        assert self._con is not None

        # A leftover backup means a previous preview never finished cleaning
        # up; the live column is the newer truth, so drop the stale copy.
        if self.has_column(table_name, backup_name):
            applogger.warning(
                "Dropping a stale column snapshot %s.%s left by an earlier preview.",
                table_name,
                backup_name,
                show_dialog=False,
                raise_error=False,
            )
            self.delete_table_column(table_name, backup_name)

        if not self.has_column(table_name, col_name):
            return False

        self.rename_table_column(table_name, col_name, backup_name)
        return True

    @ensure_connection_wrapper
    def restore_column_snapshot(
        self,
        table_name: str,
        col_name: str,
        backup_name: str,
    ) -> None:
        """Undo ``snapshot_column``: drop the live column, restore the backup."""
        assert self._con is not None

        if self.has_column(table_name, col_name):
            self.delete_table_column(table_name, col_name)

        if self.has_column(table_name, backup_name):
            self.rename_table_column(table_name, backup_name, col_name)

    @ensure_connection_wrapper
    def discard_column_snapshot(self, table_name: str, backup_name: str) -> None:
        """Drop a snapshot after the change it protected has been committed."""
        assert self._con is not None
        if self.has_column(table_name, backup_name):
            self.delete_table_column(table_name, backup_name)

    @ensure_connection_wrapper
    def clear_integer_column(self, table_name: str, col_name:str) -> None:
        """Reset all Col values to False/0 for one user data table."""
        assert self._con is not None
        self.ensure_column(table_name=table_name,col_name=col_name,col_type="INTEGER")
        self._con.execute(f"UPDATE {_quote_ident(table_name)} SET {_quote_ident(col_name)} = 0")
        self._commit()

    def clear_hide_column(self, table_name: str) -> None:
        """Reset all Hide values to False/0 for one user data table."""
        self.clear_integer_column(table_name=table_name,col_name="Hide")

    def clear_cluster_column(self, table_name: str) -> None:
        """Reset all ClusterId values to False/0 for one user data table."""
        self.clear_integer_column(table_name=table_name,col_name="ClusterId")

    @ensure_connection_wrapper
    def set_ClusterId(self,source_table,source_x_column, x_values,cluster_values) -> None:
        assert self._con is not None
        quoted_table = _quote_ident(source_table)
        quoted_x = _quote_ident(source_x_column)
        for x_value, cluster_value in zip(x_values, cluster_values, strict=False):
            if pd.notna(x_value) and pd.notna(cluster_value):
                cluster_id:int = int(cluster_value)
                self._con.execute(
                    f'UPDATE {quoted_table} SET "ClusterId" = ? WHERE {quoted_x} = ?',
                    (cluster_id, x_value),
                )
        self._commit()

    @ensure_connection_wrapper
    def mark_hide_points(
        self,
        *,
        table_name: str,
        x_column: str,
        y_column: str,
        points: Sequence[tuple[float, float]],
    ) -> int:
        """Set Hide=True/1 for rows matching supplied X/Y point pairs.

        This is a compatibility fallback.  Outlier apply should prefer
        mark_hide_rowids because it is exact and avoids float equality issues.
        """
        assert self._con is not None
        self.ensure_hide_column(table_name)
        rows = [(float(x), float(y)) for x, y in points]
        if not rows:
            return 0
        sql = (
            f"UPDATE {_quote_ident(table_name)} "
            f"SET \"Hide\" = 1 "
            f"WHERE {_quote_ident(x_column)} = ? AND {_quote_ident(y_column)} = ?"
        )
        updated_count = 0
        for x_value, y_value in rows:
            cur = self._con.execute(sql, (float(x_value), float(y_value)))
            if cur.rowcount and cur.rowcount > 0:
                updated_count += int(cur.rowcount)
        self._commit()
        hidden_count = self.count_hidden_rows(table_name)
        print(
            f"[Outlier] table={table_name!r} requested={len(rows)} "
            f"matched={updated_count} hidden_total={hidden_count}"
        )
        applogger.info(
            "Outlier Hide update table=%s requested=%d matched=%d hidden_total=%d",
            table_name,
            len(rows),
            updated_count,
            hidden_count,
        )
        return hidden_count

    @ensure_connection_wrapper
    def count_hidden_rows(self, table_name: str) -> int:
        """Return count of rows where Hide=1."""
        assert self._con is not None
        self.ensure_hide_column(table_name)
        row = self._con.execute(
            f'SELECT COUNT(*) FROM {_quote_ident(table_name)} WHERE "Hide" = 1'
        ).fetchone()
        return int(row[0]) if row else 0

    @ensure_connection_wrapper
    def query_series_frame_for_hide(
        self,
        *,
        sql_query: str,
        roles: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Return source rowid plus raw X/Y values for outlier detection."""
        assert self._con is not None
        table_name = self.query_source_table(sql_query)
        self.ensure_hide_column(table_name)
        x_col = str(roles.get("x", "")).strip()
        y_col = str(roles.get("y", "")).strip()
        if not x_col or not y_col:
            applogger.error("Series roles must contain x and y columns for outlier marking.")
        select_sql = (
            f"SELECT rowid AS __rowid__, "
            f"{_quote_ident(x_col)} AS x, "
            f"{_quote_ident(y_col)} AS y "
            f"FROM {_quote_ident(table_name)} "
            f"WHERE \"Hide\" = 0"
        )
        return pd.read_sql_query(select_sql, self._con)

    @ensure_connection_wrapper
    def mark_hide_rowids(
        self,
        *,
        table_name: str,
        rowids: Sequence[int],
        clear_existing: bool = False,
    ) -> int:
        """Set Hide=True/1 for exact SQLite rowids and report matched totals."""
        assert self._con is not None
        if clear_existing:
            self.clear_hide_column(table_name)
        else:
            self.ensure_hide_column(table_name)
        ids = [int(rowid) for rowid in rowids]
        if not ids:
            hidden_count = self.count_hidden_rows(table_name)
            print(f"[Outlier] table={table_name!r} requested=0 matched=0 hidden_total={hidden_count}")
            return hidden_count
        sql = f"UPDATE {_quote_ident(table_name)} SET \"Hide\" = 1 WHERE rowid = ?"
        updated_count = 0
        for rowid in ids:
            cur = self._con.execute(sql, (int(rowid),))
            if cur.rowcount and cur.rowcount > 0:
                updated_count += int(cur.rowcount)
        self._commit()
        hidden_count = self.count_hidden_rows(table_name)
        print(
            f"[Outlier] table={table_name!r} requested={len(ids)} "
            f"matched={updated_count} hidden_total={hidden_count}"
        )
        applogger.info(
            "Outlier Hide update table=%s requested=%d matched=%d hidden_total=%d",
            table_name,
            len(ids),
            updated_count,
            hidden_count,
        )
        return hidden_count


    @ensure_connection_wrapper
    def apply_outlier_hide_flags(
        self,
        *,
        table_name: str,
        x_column: str,
        y_column: str,
        points: Sequence[tuple[float, float]],
        clear_existing: bool = False,
    ) -> int:
        """Apply outlier flags to Hide column, optionally clearing old flags first."""
        if clear_existing:
            self.clear_hide_column(table_name)
        else:
            self.ensure_hide_column(table_name)
        return self.mark_hide_points(
            table_name=table_name,
            x_column=x_column,
            y_column=y_column,
            points=points,
        )

    def query_source_table(self, sql_query: str) -> str:
        """Best-effort extraction of the first table name after FROM."""
        match = re.search(
            r'\bfrom\s+(?:"([^"]+)"|\'([^\']+)\'|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))',
            sql_query,
            flags=re.IGNORECASE,
        )
        if match is None:
            applogger.error("Cannot identify source table from SQL query.")
            return ""
        mg=match.groups()
        if mg is None:
            return ""
        return next(part for part in match.groups() if part)

    @staticmethod
    def is_table_backed_sql(sql_query: str) -> bool:
        """True when a series query selects directly from a named table.

        The Hide machinery flips a flag on real rows and finds them by rowid,
        so it only applies to a plain table.  A series over a saved query reads
        from a subquery, which has neither a Hide column nor a rowid - adding
        the filter there produces "no such column: Hide" and loses the series.
        """
        sql = str(sql_query or "")
        match = re.search(r"\bfrom\s+(.)", sql, flags=re.IGNORECASE)
        return match is not None and match.group(1) != "("

    def sql_with_hide_filter(self, sql_query: str) -> str:
        """Return SQL with a Hide=False filter inserted, where that applies."""
        sql = str(sql_query or "").strip().rstrip(";")
        if not sql:
            return sql
        if not self.is_table_backed_sql(sql):
            return sql
        if re.search(r'\bhide\b\s*(?:=\s*0|is\s+false)', sql, flags=re.IGNORECASE):
            return sql
        clause = '"Hide" = 0'
        insert_before = re.search(
            r'\b(order\s+by|group\s+by|limit|offset)\b',
            sql,
            flags=re.IGNORECASE,
        )
        addition = (
            f" AND {clause}"
            if re.search(r'\bwhere\b', sql, flags=re.IGNORECASE)
            else f" WHERE {clause}"
        )
        if insert_before is None:
            return sql + addition
        index = insert_before.start()
        return sql[:index].rstrip() + addition + " " + sql[index:].lstrip()

    @ensure_connection_wrapper
    def update_series_hide_filter(self, series_id: int, sql_query: str | None = None) -> None:
        """Update a series descriptor query so it excludes Hide=True rows."""
        assert self._con is not None
        current_sql = sql_query
        if current_sql is None:
            row = self._con.execute(
                "SELECT sql_query FROM __series_descriptors__ WHERE id = ?",
                (int(series_id),),
            ).fetchone()
            if row is None:
                return
            current_sql = str(row["sql_query"] or "")
        filtered_sql = self.sql_with_hide_filter(current_sql)
        table_name = self.query_source_table(filtered_sql)
        self.ensure_hide_column(table_name)
        self.update_series_sql_query(int(series_id), filtered_sql)


    @ensure_connection_wrapper
    def set_figure_grid(self, figure_id: int, *, nrows: int, ncols: int) -> None:
        """Persist figure grid layout."""
        assert self._con is not None

        self._con.execute(
            "UPDATE __figure_descriptors__ SET nrows = ?, ncols = ? WHERE id = ?",
            (int(nrows), int(ncols), int(figure_id)),
        )
        self._commit()


    # =====================================================================
    # Series management
    # =====================================================================
    @ensure_connection_wrapper
    def list_series_dict(self) -> list[dict[str, Any]]:
        """Return all series descriptors as dicts."""
        assert self._con is not None
        rows = self._con.execute(
            "SELECT series_index, name, sql_query FROM __series_descriptors__"
        ).fetchall()
        if not rows:
            return []
        return [
            {
                "series_index": int(r[0]),
                "name": str(r[1]),
                "sql_query": str(r[2]),
            }
            for r in rows
        ]

    @ensure_connection_wrapper
    def get_series(self, axis_id: int) -> list[sqlite3.Row]:
        """Return every series descriptor row for one axis, ordered by index."""
        assert self._con is not None
        return self._con.execute(
            "SELECT * FROM __series_descriptors__ WHERE axis_id = ? ORDER BY series_index",
            (axis_id,),
        ).fetchall()

    @ensure_connection_wrapper
    def get_series_for_axes(self, axis_ids: Sequence[int]) -> dict[int, list[sqlite3.Row]]:
        """Return series rows for many axes in a single query, grouped by axis id.

        Why: loading a figure used to issue one ``get_series`` per axis, so the
        cost grew with the number of axes for no reason.  Axes with no series
        are still present in the result, mapped to an empty list.
        """
        assert self._con is not None

        ids = [int(axis_id) for axis_id in axis_ids]
        grouped: dict[int, list[sqlite3.Row]] = {axis_id: [] for axis_id in ids}
        if not ids:
            return grouped

        placeholders = ",".join("?" * len(ids))
        rows = self._con.execute(
            f"SELECT * FROM __series_descriptors__ "
            f"WHERE axis_id IN ({placeholders}) ORDER BY axis_id, series_index",
            tuple(ids),
        ).fetchall()

        for row in rows:
            grouped[int(row["axis_id"])].append(row)
        return grouped

    @ensure_connection_wrapper
    def query_columns_from_sql(self, sql: str) -> list[str]:
        """Extract column names from arbitrary SELECT query without fetching data.
        
        Wraps query with LIMIT 0 subquery to read cursor.description.
        Useful for dynamic plot builder dialogs.
        """
        assert self._con is not None

        q = (sql or "").strip().rstrip(";")
        if not q:
            return []

        try:
            wrapped = f"SELECT * FROM ({q}) AS _q LIMIT 0"
            cur = self._con.execute(wrapped)
            if cur.description is None:
                return []
            return [str(d[0]) for d in cur.description if d and d[0] is not None]
        except Exception:
            return []

    @ensure_connection_wrapper
    def next_axis_index(self, figure_id: int) -> int:
        """Get next available axis_index for a figure."""
        assert self._con is not None
        try:
            row = self._con.execute(
                "SELECT COALESCE(MAX(axis_index), -1) + 1 "
                "FROM __axis_descriptors__ WHERE figure_id = ?",
                (int(figure_id),),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        except Exception:
            return 0

    @ensure_connection_wrapper
    def next_series_index(self, axis_id: int) -> int:
        """Get next available series_index for an axis."""
        assert self._con is not None
        try:
            row = self._con.execute(
                "SELECT COALESCE(MAX(series_index), -1) + 1 "
                "FROM __series_descriptors__ WHERE axis_id = ?",
                (int(axis_id),),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        except Exception:
            return 0

    def build_series_select_sql(self, table: str, columns: list[str]) -> str:
        """Build safe SELECT statement with validated columns.
        
        Only includes columns passing identifier validation.
        Falls back to SELECT * if no valid columns.
        """
        cols = [c for c in columns if _is_ident(c)]
        if not cols:
            return f"SELECT * FROM {_quote_ident(table)}"
        cols_sql = ", ".join(_quote_ident(c) for c in cols)
        return f"SELECT {cols_sql} FROM {_quote_ident(table)}"

    # =====================================================================
    # Descriptor creation
    # =====================================================================

    @ensure_connection_wrapper
    def update_axis_chart_type(self, *, axis_id: int, chart_type: str) -> None:
        """Update chart_type for an existing axis."""
        assert self._con is not None
        self._con.execute(
            "UPDATE __axis_descriptors__ SET chart_type = ? WHERE id = ?",
            (str(chart_type), int(axis_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    def update_axis_descriptor(
        self,
        *,
        axis_id: int,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        z_label: str | None = None,
    ) -> None:
        """Update the labels of an existing axis.

        Only the fields that are given are written, so a caller that knows the
        y unit but not the title does not have to invent one.
        """
        assert self._con is not None

        updates = {
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "z_label": z_label,
        }
        assignments = {
            column: value for column, value in updates.items() if value is not None
        }
        if not assignments:
            return

        clause = ", ".join(f"{column} = ?" for column in assignments)
        self._con.execute(
            f"UPDATE __axis_descriptors__ SET {clause} WHERE id = ?",
            (*[str(value) for value in assignments.values()], int(axis_id)),
        )
        self._commit()

    @ensure_connection_wrapper
    def create_figure_descriptor(
        self,
        *,
        name: str,
        nrows: int = 1,
        ncols: int = 1,
        options: Mapping[str, Any] | None = None,
    ) -> int:
        """Create figure descriptor and return its id.
        
        Args:
            name: Figure name/title
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            layout: Optional layout configuration dict
            
        Returns:
            Primary key id of created figure (or 0 on error)
        """
        assert self._con is not None
        cur = self._con.execute(
            "INSERT INTO __figure_descriptors__ "
            "(name, nrows, ncols, options_json) "
            "VALUES (?, ?, ?, ?)",
            (
                str(name),
                int(nrows),
                int(ncols),
                _dumps_json(options or {}) if options else None,
            ),
        )
        self._commit()
        return int(cur.lastrowid or 0)

    def load_figures_from_db(self) -> list[tuple[int, str]]:
        """Load all figures from database as (id, name) tuples."""
        df = self.query_df("SELECT id, name FROM __figure_descriptors__ ORDER BY id")
        if df is None or df.empty:
            return []
        return [
            (int(fig_id), str(name))
            for fig_id, name in df.itertuples(index=False, name=None)
        ]
    
    @ensure_connection_wrapper
    def get_axes(self, figure_id:int)-> list[sqlite3.Row]|None:
        assert self._con is not None
        axis_rows = self._con.execute(
            "SELECT * FROM __axis_descriptors__ WHERE figure_id = ? ORDER BY axis_index",
            (figure_id,),
        ).fetchall()
        return axis_rows
    
    @ensure_connection_wrapper
    def list_axes_for_figure(self, figure_id: int) -> list[tuple[int, int, str]]:
        """Get axes for a figure as (axis_id, axis_index, title) tuples."""
        assert self._con is not None

        rows = self._con.execute(
            "SELECT id, axis_index, COALESCE(title, '') AS title "
            "FROM __axis_descriptors__ WHERE figure_id = ? ORDER BY axis_index",
            (int(figure_id),),
        ).fetchall()
        return [
            (int(r["id"]), int(r["axis_index"]), str(r["title"] or ""))
            for r in rows
        ]

    @ensure_connection_wrapper
    def create_axis_descriptor(
        self,
        *,
        figure_id: int,
        axis_index: int,
        chart_type: str,
        title: str,
        x_label: str,
        y_label: str,
        z_label: str | None = None,
        options: Mapping[str, Any] | None,
    ) -> int:
        """Create axis descriptor and return its id.
        
        Args:
            figure_id: Parent figure id
            axis_index: Position in subplot grid
            chart_type: Type of chart (scatter, line, etc.)
            title: Axis/subplot title
            x_label: X-axis label
            y_label: Y-axis label
            z_label: Z-axis label
            options: Optional configuration dict
            
        Returns:
            Primary key id of created axis (or 0 on error)
        """
        assert self._con is not None
        cur = self._con.execute(
            "INSERT INTO __axis_descriptors__ "
            "(figure_id, axis_index, chart_type, title, x_label, y_label, z_label, options_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(figure_id),
                int(axis_index),
                str(chart_type),
                str(title),
                str(x_label),
                str(y_label),
                str(z_label) if z_label is not None else "",
                _dumps_json(options or {}) if options else None,
            ),
        )
        self._commit()
        return int(cur.lastrowid or 0)

    @ensure_connection_wrapper
    def create_series_descriptor(
        self,
        *,
        axis_id: int,
        series_index: int,
        name: str,
        sql_query: str,
        roles: Mapping[str,str]|None = None,
        style: Mapping[str, Any] | None = None,
    ) -> int:
        """Create series descriptor and return its id.
        
        Args:
            axis_id: Parent axis id
            series_index: Position within axis
            name: Series name
            sql_query: SELECT query returning data for this series
            columns: column descriptios dic
            style: Style configuration dict (optional)
            
        Returns:
            Primary key id of created series (or 0 on error)
        """
        assert self._con is not None
        filtered_sql_query = self.sql_with_hide_filter(str(sql_query))

        # Only a table-backed series gets a Hide column; a series over a saved
        # query reads from a subquery that has no row identity to hide.
        if self.is_table_backed_sql(filtered_sql_query):
            source_table = self.query_source_table(filtered_sql_query)
            if source_table:
                self.ensure_hide_column(source_table)
        cur = self._con.execute(
            "INSERT INTO __series_descriptors__ "
            "(axis_id, series_index, name, sql_query, roles, style_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                int(axis_id),
                int(series_index),
                str(name),
                filtered_sql_query,
                _dumps_json(roles or {}) if roles else None,
                _dumps_json(style or {}) if style else None,
            ),
        )
        self._commit()
        return int(cur.lastrowid or 0)

    @ensure_connection_wrapper
    def ensure_figure_grid_capacity(self, *, figure_id: int, needed_axes: int) -> None:
        """Ensure figure grid has enough cells for needed_axes subplots.
        
        Dynamically increases nrows if necessary (keeps ncols fixed).
        """
        assert self._con is not None

        fig = self._con.execute(
            "SELECT nrows, ncols FROM __figure_descriptors__ WHERE id = ?",
            (int(figure_id),),
        ).fetchone()
        if fig is None:
            return

        nrows = max(1, int(fig["nrows"]))
        ncols = max(1, int(fig["ncols"]))
        cap = nrows * ncols

        if needed_axes <= cap:
            return

        # Increase nrows while keeping ncols fixed
        import math
        new_nrows = int(math.ceil(float(needed_axes) / float(ncols)))
        self._con.execute(
            "UPDATE __figure_descriptors__ SET nrows = ? WHERE id = ?",
            (int(new_nrows), int(figure_id)),
        )
        self._commit()

    # =====================================================================
    # Connection lifecycle
    # =====================================================================


    # =====================================================================
    # Saved queries and explicit indexing
    # =====================================================================

    @ensure_connection_wrapper
    def create_queries_table(self) -> None:
        """Create the saved-query table."""
        assert self._con is not None
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS __queries__ (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL UNIQUE,
                sql           TEXT NOT NULL,
                settings_json TEXT
            );
            """
        )
        self._con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_queries_name
                ON __queries__ (name)
            """
        )

    @ensure_connection_wrapper
    def create_index(
        self,
        table: str,
        columns: Sequence[str],
        *,
        name: str | None = None,
        unique: bool = False,
    ) -> None:
        """Create an index using validated identifiers only."""
        if not columns:
            applogger.error("At least one column is required")

        assert self._con is not None
        table_sql = _quote_ident(table)
        cols_sql = ", ".join(_quote_ident(col) for col in columns)
        index_name = name or f"idx_{table}_{'_'.join(columns)}"
        index_sql = _quote_ident(index_name)
        unique_sql = "UNIQUE " if unique else ""

        self._con.execute(
            f"CREATE {unique_sql}INDEX IF NOT EXISTS "
            f"{index_sql} ON {table_sql} ({cols_sql})"
        )

    @ensure_connection_wrapper
    def save_query(
        self,
        name: str,
        sql: str,
        settings: Mapping[str, Any] | None = None,
    ) -> int:
        """Insert or update a named SQL query and return its id."""
        assert self._con is not None
        self.create_queries_table()

        clean_name = name.strip()
        clean_sql = sql.strip()
        if not clean_name:
            applogger.error("Query name is required")
        if not clean_sql:
            applogger.error("Query SQL is required")

        self._con.execute(
            """
            INSERT INTO __queries__ (name, sql, settings_json)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                sql = excluded.sql,
                settings_json = excluded.settings_json
            """,
            (clean_name, clean_sql, _dumps_json(settings)),
        )
        row = self._con.execute(
            "SELECT id FROM __queries__ WHERE name = ?",
            (clean_name,),
        ).fetchone()
        if row is None:
            applogger.error(f"Failed to save query: {clean_name}")
            return 0
        return int(row["id"])

    @ensure_connection_wrapper
    def get_query(self, name: str) -> SavedQuery | None:
        """Return a saved query by name."""
        assert self._con is not None
        self.create_queries_table()

        row = self._con.execute(
            """
            SELECT id, name, sql, settings_json
            FROM __queries__
            WHERE name = ?
            """,
            (name,),
        ).fetchone()
        if row is None:
            return None
        return SavedQuery(
            id=int(row["id"]),
            name=str(row["name"]),
            sql=str(row["sql"]),
            settings=_loads_json(row["settings_json"]),
        )

    @ensure_connection_wrapper
    def get_query_by_id(self, query_id: int) -> SavedQuery | None:
        """Return a saved query by id."""
        assert self._con is not None
        self.create_queries_table()

        row = self._con.execute(
            """
            SELECT id, name, sql, settings_json
            FROM __queries__
            WHERE id = ?
            """,
            (int(query_id),),
        ).fetchone()
        if row is None:
            return None
        return SavedQuery(
            id=int(row["id"]),
            name=str(row["name"]),
            sql=str(row["sql"]),
            settings=_loads_json(row["settings_json"]),
        )

    @ensure_connection_wrapper
    def list_queries(self) -> list[SavedQuery]:
        """Return all saved queries ordered by name."""
        assert self._con is not None
        self.create_queries_table()

        rows = self._con.execute(
            """
            SELECT id, name, sql, settings_json
            FROM __queries__
            ORDER BY name COLLATE NOCASE
            """
        ).fetchall()
        return [
            SavedQuery(
                id=int(row["id"]),
                name=str(row["name"]),
                sql=str(row["sql"]),
                settings=_loads_json(row["settings_json"]),
            )
            for row in rows
        ]

    @ensure_connection_wrapper
    def new_query(
        self,
        name: str = "",
        *,
        table: str | None = None,
        sql: str = "",
    ) -> SavedQuery:
        """Return a new saved-query descriptor - named, seeded, not yet stored.

        Nothing is written.  A query is only worth a row once it runs, and a
        brand-new one does not: ``save_query`` refuses empty SQL and the
        builder refuses to save a statement that fails validation, so writing
        here would either need a second set of rules or leave rows behind that
        no chart can read.  The dialog holds the draft and stores it on Save,
        exactly like a new document in an editor.

        What the repository does own is the naming - an unnamed draft gets the
        first free ``Query <n>`` from :meth:`next_query_name` - and the seed
        statement, which is a plain SELECT over *table* when one is named and
        empty otherwise.  Nothing beyond the table is guessed: a WHERE or a
        JOIN invented here would produce a query that runs and returns the
        wrong rows, which is worse than one that does not run.

        The name is checked but not enforced: a name that is already a table's
        can never be selected (the table always wins) and a name already taken
        by a saved query will overwrite it on Save.  Both are the caller's
        decision - the builder asks about the second one - so this reports them
        to the log and hands the draft back either way.
        """
        clean_name = str(name or "").strip() or self.next_query_name()

        if self.check_if_table_exists(clean_name):
            applogger.warning(
                "New query %r has the name of a table; it could never be "
                "selected, because a table of the same name always wins.",
                clean_name,
            )
        elif self.get_query(clean_name) is not None:
            applogger.warning(
                "New query %r has the name of a saved query; saving it will "
                "replace that one.",
                clean_name,
            )

        clean_sql = str(sql or "").strip()
        if not clean_sql and table:
            clean_sql = f"SELECT * FROM {_quote_ident(table)}"

        # id None, not 0: None is what "no row yet" means everywhere else in
        # this module, and a 0 would compare equal to a real id in a falsy
        # test while looking like one in a log line.
        return SavedQuery(id=None, name=clean_name, sql=clean_sql, settings={})

    @ensure_connection_wrapper
    def next_query_name(self, base: str = "Query") -> str:
        """Return a free name for a new saved query, as ``"<base> <n>"``.

        Free of *both* saved queries and tables.  A query named after a table
        can never be selected - ``list_data_sources`` resolves the name against
        the schema first, so the table always wins - which is why the builder
        refuses such a name on Save; suggesting one here would be offering a
        name that is about to be rejected.

        Compared case-insensitively: SQLite resolves a table name whatever its
        case, so "Query 1" would still be shadowed by a table called "query 1",
        and two saved queries differing only in case are a trap for whoever has
        to tell them apart in the list.
        """
        stem = str(base or "Query").strip() or "Query"
        self.create_queries_table()

        taken = {saved.name.strip().lower() for saved in self.list_queries()}
        taken.update(name.strip().lower() for name in self.list_table_names())

        index = 1
        while f"{stem} {index}".lower() in taken:
            index += 1
        return f"{stem} {index}"

    @ensure_connection_wrapper
    def delete_query(self, name: str) -> bool:
        """Delete a saved query by name."""
        assert self._con is not None
        self.create_queries_table()

        cur = self._con.execute(
            "DELETE FROM __queries__ WHERE name = ?",
            (name,),
        )
        return cur.rowcount > 0

    @ensure_connection_wrapper
    def rename_query(self, old_name: str, new_name: str) -> None:
        """Rename a saved query."""
        assert self._con is not None
        self.create_queries_table()

        clean_name = new_name.strip()
        if not clean_name:
            applogger.error("New query name is required")

        self._con.execute(
            """
            UPDATE __queries__
            SET name = ?
            WHERE name = ?
            """,
            (clean_name, old_name),
        )

    @ensure_connection_wrapper
    def run_saved_query(
        self,
        name: str,
        params: tuple[Any, ...] | None = None,
    ) -> pd.DataFrame:
        """Execute a saved row-returning query."""
        query = self.get_query(name)
        if query is None or query.sql is None:
            applogger.error(f"Saved query not found: {name}")
            return pd.DataFrame()
        # Refuse rather than warn-and-run: this used to log "does not return
        # rows" and then hand the statement to query_df anyway, which executes
        # whatever it is given - so a saved query that had been edited into a
        # DELETE ran it.
        ok, reason = is_read_only_select(query.sql)
        if not ok:
            applogger.error("Saved query %r will not be run: %s", name, reason)
            return pd.DataFrame()

        return self.query_df(query.sql, params=params)

    @ensure_connection_wrapper
    def explain_query_plan(
        self,
        sql: str,
        params:  tuple[Any, ...] | None = None,
    ) -> pd.DataFrame:
        """Return SQLite query planner output for a statement."""
        return self.query_df(f"EXPLAIN QUERY PLAN {sql}", params=params)

    @ensure_connection_wrapper
    def optimize_db(self) -> DatabaseReport:
        """Check the database, report what is wrong, then compact it.

        Checks first, compaction second: VACUUM rewrites the file, so anything
        it silently drops would no longer be reportable afterwards.

        Inside an open transaction the compaction is skipped rather than
        attempted.  SQLite refuses to VACUUM there, and the resulting
        OperationalError used to travel up through whatever multi-step
        operation was running and roll the whole thing back - which is how a
        spectral analysis of five series ended up applying one.  A missed
        compaction costs some disk space; an aborted operation costs the
        user's work.

        Returns the report as well as logging it, so callers can show it.
        """
        assert self._con is not None

        report = self.check_database()
        report.log()

        if self._con.in_transaction:
            applogger.warning(
                "Skipping VACUUM: a transaction is open. Call optimize_db "
                "after committing.",
                show_dialog=False,
                raise_error=False,
            )
            return report

        self._con.execute("VACUUM")
        self._con.execute("ANALYZE")
        self._con.execute("PRAGMA optimize")
        applogger.info("Database optimized: VACUUM, ANALYZE, PRAGMA optimize.")
        return report

    @ensure_connection_wrapper
    def check_database(self) -> DatabaseReport:
        """Run integrity checks and look for orphaned descriptor rows/tables."""
        assert self._con is not None

        report = DatabaseReport()
        self._check_integrity(report)
        self._check_foreign_keys(report)
        self._check_zombies(report)
        return report

    def _check_integrity(self, report: DatabaseReport) -> None:
        """Run PRAGMA integrity_check and record anything but 'ok'."""
        assert self._con is not None
        try:
            rows = self._con.execute("PRAGMA integrity_check").fetchall()
        except Exception as exc:
            report.integrity_errors.append(f"integrity_check failed: {exc}")
            return

        for row in rows:
            message = str(row[0])
            if message.strip().lower() != "ok":
                report.integrity_errors.append(message)

    def _check_foreign_keys(self, report: DatabaseReport) -> None:
        """Record rows whose foreign key points at nothing."""
        assert self._con is not None
        try:
            rows = self._con.execute("PRAGMA foreign_key_check").fetchall()
        except Exception as exc:
            report.integrity_errors.append(f"foreign_key_check failed: {exc}")
            return

        for row in rows:
            # (table, rowid, parent table, fkid)
            report.foreign_key_errors.append(
                f"{row[0]} rowid={row[1]} references missing row in {row[2]}"
            )

    def _check_zombies(self, report: DatabaseReport) -> None:
        """Find descriptors pointing at nothing, and data nothing points at.

        Two directions, because they are different problems:

        * a **dangling reference** is a series or import link naming a table
          that no longer exists - the chart is broken and the user should know;
        * an **unreferenced table** is data no chart or link uses. That is not
          an error - it is often simply a table waiting to be plotted - so it
          is reported separately and only as information.
        """
        assert self._con is not None

        existing = set(self.list_table_names())

        # Series whose SQL selects from a table that is gone.
        for row in self._con.execute(
            "SELECT id, name, sql_query FROM __series_descriptors__"
        ).fetchall():
            sql = str(row["sql_query"] or "")
            # A series over a saved query reads from a subquery; its inner
            # tables are checked through that query, not through this name.
            if not self.is_table_backed_sql(sql):
                continue
            table = self.query_source_table(sql)
            if table and table not in existing:
                report.dangling_series.append(
                    f"series id={row['id']} '{row['name']}' reads from missing table '{table}'"
                )

        # Axes whose figure is gone, and series whose axis is gone.  The schema
        # declares these foreign keys, but only enforces them when the pragma
        # was on for every writer that ever touched the file.
        for label, child, parent, child_key in (
            ("axis", "__axis_descriptors__", "__figure_descriptors__", "figure_id"),
            ("series", "__series_descriptors__", "__axis_descriptors__", "axis_id"),
        ):
            for row in self._con.execute(
                f"SELECT c.id AS id, c.{child_key} AS parent_id FROM {child} AS c "
                f"LEFT JOIN {parent} AS p ON p.id = c.{child_key} "
                "WHERE p.id IS NULL"
            ).fetchall():
                report.orphan_descriptors.append(
                    f"{label} id={row['id']} belongs to missing parent id={row['parent_id']}"
                )

        # Import links whose destination table is gone.
        for row in self._con.execute(
            "SELECT id, table_name FROM __import_links__"
        ).fetchall():
            if str(row["table_name"]) not in existing:
                report.dangling_links.append(
                    f"import link id={row['id']} targets missing table '{row['table_name']}'"
                )

        # Tables no series and no link refers to.
        referenced: set[str] = set()
        for row in self._con.execute(
            "SELECT sql_query FROM __series_descriptors__"
        ).fetchall():
            sql = str(row["sql_query"] or "")
            if not self.is_table_backed_sql(sql):
                # Credit the tables the subquery mentions, so a table used only
                # through a saved query is not reported as unreferenced.
                referenced.update(re.findall(r'\bfrom\s+"?([A-Za-z_][A-Za-z0-9_]*)"?', sql, flags=re.IGNORECASE))
                continue
            table = self.query_source_table(sql)
            if table:
                referenced.add(table)
        for row in self._con.execute("SELECT table_name FROM __import_links__").fetchall():
            referenced.add(str(row["table_name"]))

        report.unreferenced_tables.extend(sorted(existing - referenced))

    @ensure_connection_wrapper
    def list_table_names(self) -> list[str]:
        """Return every user table name, excluding this application's own."""
        assert self._con is not None
        rows = self._con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' "
            "  AND name NOT LIKE '__%__' ESCAPE '_' "
            "  AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        return [str(row[0]) for row in rows]

    @ensure_connection_wrapper
    def swap_axis_indexes(
        self,
        *,
        figure_id: int,
        first_axis_id: int,
        second_axis_id: int,
    ) -> None:
        """Swap two axis_index values for axes in one figure.

        The temporary index avoids violating the UNIQUE(figure_id, axis_index)
        constraint while the two rows are exchanged.
        """
        assert self._con is not None
        rows = self.list_axes_for_figure(int(figure_id))
        by_id = {int(axis_id): int(axis_index) for axis_id, axis_index, _title in rows}
        first_id = int(first_axis_id)
        second_id = int(second_axis_id)
        if first_id not in by_id or second_id not in by_id:
            applogger.error("Cannot swap axis indexes: axis id not found.")
            return
        temporary_index = min(by_id.values()) - 1
        with self.transaction(immediate=True):
            self._con.execute(
                "UPDATE __axis_descriptors__ SET axis_index = ? WHERE id = ?",
                (temporary_index, first_id),
            )
            self._con.execute(
                "UPDATE __axis_descriptors__ SET axis_index = ? WHERE id = ?",
                (by_id[first_id], second_id),
            )
            self._con.execute(
                "UPDATE __axis_descriptors__ SET axis_index = ? WHERE id = ?",
                (by_id[second_id], first_id),
            )

    def close(self) -> None:
        """Close the database connection (call on app shutdown)."""
        con = self._con
        if con is not None:
            try:
                con.commit()
            except Exception:
                pass
            try:
                con.close()
            except Exception:
                pass
        self._con = None
        self._is_connected = False
        self.invalidate_series_cache()

    # =====================================================================
    # Table management (rename/delete with propagation)
    # =====================================================================
    @ensure_connection_wrapper
    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename a table and propagate references in series queries."""
        assert self._con is not None

        old = (old_name or "").strip()
        new = (new_name or "").strip()

        if not old or not new:
            applogger.error("Missing old_name/new_name")
        if old == new:
            return
        if not _is_ident(old) or not _is_ident(new):
            applogger.error(f"Invalid table name(s): {old!r} -> {new!r}")

        old_q = _quote_ident(old)
        new_q = _quote_ident(new)

        with self.transaction(immediate=True):
            # Verify source exists
            if not self._con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
                (old,),
            ).fetchone():
                applogger.error(f"Table not found for renaming: {old}")
                return

            # Verify target doesn't exist
            if self._con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
                (new,),
            ).fetchone():
                applogger.error(f"Target table already exists: {new}")
                return

            # Rename the table
            self._con.execute(f"ALTER TABLE {old_q} RENAME TO {new_q}")
            # Update all series that reference this table
            self._propagate_table_name(old, new, mode="rename")

    @ensure_connection_wrapper
    def check_if_table_exists(self, table_name: str) -> bool:
        """Check if a user table exists."""
        assert self._con is not None
        name = (table_name or "").strip()
        if not name:
            return False
        row = self._con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
            (name,),
        ).fetchone()
        return row is not None

    @ensure_connection_wrapper
    def delete_table(self, table_name: str) -> None:
        """Drop a user table and remove related series descriptors."""
        table = (table_name or "").strip()
        if not table:
            return
        assert self._con is not None

        with self.transaction(immediate=True):
            self._con.execute(f"DROP TABLE IF EXISTS {_quote_ident(table)}")
            self._propagate_table_name(table, None, mode="delete")

    def has_uncommitted_changes(self) -> bool:
        """Check if connection has an open transaction."""
        return bool(self._con is not None and self._con.in_transaction)

    @ensure_connection_wrapper
    def _propagate_table_name(
        self,
        table_name: str,
        new_name: str | None,
        mode: Literal["rename", "delete"],
    ) -> None:
        """Update series queries after table rename/delete.
        
        On rename: update FROM clauses in sql_query.
        On delete: remove affected series descriptors.
        
        Called within a transaction; no explicit commit.
        """
        assert self._con is not None

        for s in self.list_series_dict():
            sql = str(s.get("sql_query", "") or "")
            series_id = s.get("series_index")
            if series_id is None:
                continue

            # Extract table name from FROM clause
            parts = re.split(r"\bFROM\b", sql, flags=re.IGNORECASE)
            if len(parts) < 2:
                continue

            from_token = parts[1].strip().split()[0]
            from_token = from_token.strip('"`[]')

            if from_token != table_name:
                continue

            if mode == "rename":
                # Replace old table name with new
                repl = sql.replace(table_name, new_name or "")
                self._con.execute(
                    "UPDATE __series_descriptors__ SET sql_query = ? WHERE series_index = ?",
                    (repl, series_id),
                )
            else:
                # Delete series referencing deleted table
                self._con.execute(
                    "DELETE FROM __series_descriptors__ WHERE series_index = ?",
                    (series_id,),
                )

    # =====================================================================
    # Advanced data import (with type coercion)
    # =====================================================================

    @ensure_connection_wrapper
    def import_into_sqlite(
        self, table: str, df: pd.DataFrame, types: dict[str, str]
    ) -> None:
        """Import DataFrame with explicit type declarations.
        
        Uses executemany with value normalization to handle Excel bytes/memoryview.
        """
        if not table:
            return
        assert self._con is not None

        tq = _quote_ident(table)

        # Drop existing table if present
        if self._con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (table,),
        ).fetchone():
            self._con.execute(f"DROP TABLE {tq}")

        # Build CREATE TABLE statement with types
        cols_sql = [
            f"{_quote_ident(col)} {types.get(col, 'TEXT')}" for col in df.columns
        ]
        self._con.execute(f"CREATE TABLE {tq} ({', '.join(cols_sql)})")

        # Prepare batch insert
        cols = list(df.columns)
        placeholders = ", ".join(["?"] * len(cols))
        col_list = ", ".join(_quote_ident(c) for c in cols)
        sql = f"INSERT INTO {tq} ({col_list}) VALUES ({placeholders})"
        decls = [types.get(c, "TEXT") for c in cols]

        # Normalize values and insert
        base_iter = df.where(pd.notna(df), None).itertuples(
            index=False, name=None
        )
        records = (
            tuple(
                self._normalize_sqlite_value(row[i], decls[i])
                for i in range(len(cols))
            )
            for row in base_iter
        )
        self._con.executemany(sql, records)
        self._commit()

    @staticmethod
    def _normalize_sqlite_value(val: Any, decl: str) -> Any:
        """Normalize Python values for SQLite binding.
        
        Handles:
          - numpy scalars -> Python native
          - pandas Timestamp -> datetime
          - Excel bytes/memoryview -> numeric (IEEE754 unpack or decimal parse)
        """
        # None/NaN early return
        if val is None or pd.isna(val):
            return None

        # Convert numpy scalar to Python native
        if isinstance(val, np.generic):
            val = val.item()

        # Convert pandas Timestamp to datetime
        if isinstance(val, pd.Timestamp):
            return val.to_pydatetime()

        decl_u = (decl or "").upper()

        # Handle bytes/memoryview for numeric types
        if decl_u in ("REAL", "INTEGER", "NUMERIC", "DATE", "TIME", "DATETIME", "TEXT"):
            if isinstance(val, memoryview):
                val = val.tobytes()

            if isinstance(val, (bytes, bytearray)):
                b = bytes(val)

                # Try IEEE754 unpack (common Excel export format)
                try:
                    if len(b) == 8:
                        return float(struct.unpack("<d", b)[0])
                    if len(b) == 4:
                        return float(struct.unpack("<f", b)[0])
                except Exception:
                    pass

                # Try UTF-8 decode and numeric conversion
                try:
                    s = b.decode("utf-8", errors="strict").strip()
                    if decl_u == "INTEGER":
                        return int(float(s))
                    if decl_u in ("REAL", "NUMERIC"):
                        return float(s)
                    return s
                except Exception:
                    if decl_u in ("REAL", "INTEGER", "NUMERIC"):
                        return None  # Can't convert to number
                    return b.decode("utf-8", errors="replace")

        # BLOB: convert memoryview to bytes
        if decl_u == "BLOB":
            if isinstance(val, memoryview):
                return val.tobytes()
            return val

        return val
    
    def load_figure_descriptor(self, figure_id: int) -> app.data.descriptors.FigureDescriptor | None:
        """Load a full figure descriptor tree: figure, axes, and their series.

        Returns None only when the figure itself does not exist.  A figure with
        no axes is a valid, empty figure and is returned as such.
        """
        fig = self.get_figure_descriptor(figure_id)
        if fig is None:
            return None

        axis_rows = self.get_axes(figure_id) or []
        series_by_axis = self.get_series_for_axes([int(a["id"]) for a in axis_rows])

        for a in axis_rows:
            axis = app.data.descriptors.AxisDescriptor(
                id=int(a["id"]),
                figure_id=int(a["figure_id"]),
                axis_index=int(a["axis_index"]),
                chart_type=str(a["chart_type"]),
                title=str(a["title"] or ""),
                x_label=str(a["x_label"] or ""),
                y_label=str(a["y_label"] or ""),
                z_label=str(a["z_label"] or "") if "z_label" in a.keys() else "",
                options=json.loads(str(a["options_json"])) if a["options_json"] is not None else None,
                series=[
                    app.data.descriptors.SeriesDescriptor(
                        id=int(s["id"]),
                        axis_id=int(s["axis_id"]),
                        series_index=int(s["series_index"]),
                        name=str(s["name"] or f"Series {s['series_index']}"),
                        sql_query=str(s["sql_query"]),
                        roles=s["roles"],
                        style=json.loads(s["style_json"]) if s["style_json"] is not None else None,
                    )
                    for s in series_by_axis.get(int(a["id"]), [])
                ],
            )

            if fig.axes is None:
                fig.axes = []
            fig.axes.append(axis)

        # Apply persisted ordering.
        if fig.axes:
            fig_opts = fig.options if isinstance(fig.options, dict) else {}
            order_ids = [int(x) for x in fig_opts.get("axes_order", []) if str(x).isdigit()]
            if order_ids:
                by_id = {ax.id: ax for ax in fig.axes}
                ordered = [by_id[i] for i in order_ids if i in by_id]
                tail = [ax for ax in fig.axes if ax.id not in set(order_ids)]
                fig.axes = ordered + tail

            for ax in fig.axes:
                opts = ax.options if isinstance(ax.options, dict) else {}
                s_order = [int(x) for x in opts.get("series_order", []) if str(x).isdigit()]
                if s_order and ax.series is not None:
                    s_by = {s.id: s for s in ax.series}
                    s_ord = [s_by[i] for i in s_order if i in s_by]
                    s_tail = [s for s in ax.series if s.id not in set(s_order)]
                    ax.series = s_ord + s_tail

        return fig


    # =====================================================================
    # Runtime-attached table preview context-menu operations
    # =====================================================================
    # These are attached after the SqliteRepo class definition so the file remains
    # drop-in even if the class layout changes.

    def delete_table_column(self, table_name: str, column_name: str) -> None:
        """Delete a column from a user table."""
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        if column_name.lower() == "rowid":
            applogger.error("Cannot delete rowid.")
        if column_name == "Hide":
            applogger.error("Column 'Hide' is managed by Data Hub and cannot be deleted.")
        self._con.execute(
            f"ALTER TABLE {_quote_ident(table_name)} DROP COLUMN {_quote_ident(column_name)}"
        )
        self._commit()


    def reset_hide(self, table_name: str) -> int:
        """Ensure Hide exists and set all values to 0."""
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        self.ensure_hide_column(table_name)
        cur = self._con.execute(f'UPDATE {_quote_ident(table_name)} SET "Hide" = 0')
        self._commit()
        return int(cur.rowcount or 0)


    def invert_hide(self, table_name: str) -> int:
        """Ensure Hide exists and invert 0/1 values."""
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        self.ensure_hide_column(table_name)
        cur = self._con.execute(
            f'UPDATE {_quote_ident(table_name)} '
            f'SET "Hide" = CASE WHEN COALESCE("Hide", 0) = 0 THEN 1 ELSE 0 END'
        )
        self._commit()
        return int(cur.rowcount or 0)


    def hide_rows_by_value(
        self,
        table_name: str,
        column_name: str,
        operator: str,
        value: Any,
    ) -> int:
        """Set Hide=1 where column compares to a user-provided value."""
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        self.ensure_hide_column(table_name)
        op_map = {"=": "=", "!=": "!=", "<>": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}
        sql_op = op_map.get(str(operator).strip())
        if sql_op is None:
            applogger.error(f"Unsupported operator: {operator}")
        cur = self._con.execute(
            f'UPDATE {_quote_ident(table_name)} SET "Hide" = 1 WHERE {_quote_ident(column_name)} {sql_op} ?',
            (value,),
        )
        self._commit()
        return int(cur.rowcount or 0)


    def hide_rows_special(self, table_name: str, column_name: str, mode: str) -> int:
        """Set Hide=1 using a predefined special predicate."""
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        self.ensure_hide_column(table_name)
        column_sql = _quote_ident(column_name)
        if mode == "null_or_empty":
            predicate = f"{column_sql} IS NULL OR TRIM(CAST({column_sql} AS TEXT)) = ''"
        else:
            applogger.error(f"Unsupported hide mode: {mode}")
        cur = self._con.execute(
            f'UPDATE {_quote_ident(table_name)} SET "Hide" = 1 WHERE {predicate}'
        )
        self._commit()
        return int(cur.rowcount or 0)


    def add_column_from_expression(
        self,
        table_name: str,
        column_name: str,
        expression: str,
    ) -> None:
        """Add a column and populate it from a SQL expression evaluated per row."""
        if not self._is_connected or self._con is None:
            self._connect()
        assert self._con is not None
        expr = str(expression or "").strip()
        if not expr:
            applogger.error("SQL expression is required.")
        if column_name in set(self.get_columns(table_name)):
            applogger.error(f"Column already exists: {column_name}")
        table_sql = _quote_ident(table_name)
        column_sql = _quote_ident(column_name)
        self._con.execute(f"ALTER TABLE {table_sql} ADD COLUMN {column_sql}")
        self._con.execute(f"UPDATE {table_sql} SET {column_sql} = {expr}")
        self._commit()