"""SQLite repository for Data Hub: the connection, and everything on it.

``SqliteRepo`` is the one class the application talks to - 77 call sites
write ``from app.data.sqlite_repo import SqliteRepo`` and none of them
changed when this module went from 3 300 lines to a few hundred (todo.txt
N-5). What is left here is what the parts share and what has to be built
before any of them can run:

* the dataclass and its fields, including the connection itself;
* connecting, the pragmas, and closing;
* the **series DataFrame cache**, the single biggest lever on render time -
  without it every property tweak re-runs every series query against
  SQLite (see ``series_df`` for the invalidation contract);
* preview savepoints and the transaction helper every writer uses;
* the undo hooks, which need the connection to attach their own database.

The rest is in ``app/data/repo/``, mixed in below - user tables,
descriptors, saved queries, maintenance - split by subject rather than by
size, so that reading about one of them means reading one file.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pandas as pd

import app.data.descriptors
from app.data.undo_store import UndoEntry, UndoStore
from app.logs.logger import applogger
from app.utils.config import load_config

# Re-exported, not just imported: "from app.data.sqlite_repo import
# DatabaseReport" is what the rest of the application writes, and the split
# into app/data/repo/ is meant to be invisible from outside (todo.txt N-5).
from app.data.repo._common import (  # noqa: F401 - re-export
    DatabaseReport,
    SavedQuery,
    _dumps_json,
    _is_ident,
    _loads_json,
    _quote_ident,
    _RETURNS_ROWS_RE,
    ensure_connection_wrapper,
    is_read_only_select,
)
from app.data.repo.descriptors import DescriptorsMixin
from app.data.repo.maintenance import MaintenanceMixin
from app.data.repo.queries import QueriesMixin
from app.data.repo.tables import TablesMixin


# Series cache defaults, overridable from config.json under "series_cache".
_SERIES_CACHE_DEFAULT_ENABLED = True
_SERIES_CACHE_DEFAULT_MAX_ENTRIES = 64


@dataclass(slots=True)
class SqliteRepo(
    TablesMixin,
    DescriptorsMixin,
    QueriesMixin,
    MaintenanceMixin,
):
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

    # Row counts behind series_row_count(), invalidated the same way and at
    # the same time as _series_cache - see that field's docstring.
    _series_row_count_cache: dict[str, int] = field(default_factory=dict)

    # Built on first use rather than in __post_init__: a repository is
    # created for a great many things that never change a table, and the
    # store is only a path until one of them does.
    _undo_store: UndoStore | None = None


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
        self._series_row_count_cache.clear()
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
    def series_row_count(self, sql: str) -> int:
        """Return how many rows *sql* would produce, without building them.

        Downsampling needs the total row count before it can size a stride -
        a single ``COUNT(*)`` rather than reading the whole result into a
        DataFrame just to call ``len()`` on it, which would defeat the point
        of downsampling in the first place. Cached the same way and
        invalidated by the same database-state stamp as ``series_df``.
        """
        sql_text = (sql or "").strip()
        if not sql_text:
            return 0

        assert self._con is not None

        stamp = self._database_stamp()
        if stamp != self._series_cache_stamp:
            self._series_cache.clear()
            self._series_row_count_cache.clear()
            self._series_cache_stamp = stamp

        cached = self._series_row_count_cache.get(sql_text)
        if cached is not None:
            return cached

        row = self._con.execute(f"SELECT COUNT(*) FROM ({sql_text})").fetchone()
        count = int(row[0]) if row and row[0] is not None else 0
        self._series_row_count_cache[sql_text] = count
        return count

    @ensure_connection_wrapper
    def downsampled_series_df(self, sql: str, *, threshold: int) -> pd.DataFrame:
        """Return *sql*'s result, decimated to roughly ``threshold`` rows.

        Decided and applied entirely inside SQLite: one ``COUNT(*)``
        (``series_row_count``, itself cached) to size a stride, and - only
        when the query actually exceeds ``threshold`` - one row kept out of
        every stride via ``ROW_NUMBER()``, ordered by the query's own first
        column. Every point-series query in this application selects the x
        role first (``SELECT x, y FROM ...``), so ordinal position 1 names
        it without needing to know the column's actual alias. The full
        result is never read into pandas only to be thinned out afterwards -
        that would keep the exact cost this exists to avoid.

        ``threshold <= 0`` disables downsampling and reads the plain query.
        """
        sql_text = (sql or "").strip()
        if not sql_text or threshold <= 0:
            return self.series_df(sql_text)

        total = self.series_row_count(sql_text)
        if total <= threshold:
            return self.series_df(sql_text)

        stride = max(1, -(-total // threshold))  # ceil division
        wrapped = (
            "SELECT * FROM ("
            f"SELECT *, ROW_NUMBER() OVER (ORDER BY 1) AS __dhub_rn__ FROM ({sql_text})"
            f") WHERE (__dhub_rn__ - 1) % {stride} = 0"
        )
        frame = self.series_df(wrapped)
        return frame.drop(columns="__dhub_rn__", errors="ignore")


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
    # Undo
    # =====================================================================
    @property
    def undo_store(self) -> UndoStore:
        """The per-project snapshot store, made on first use.

        Keyed to this repository's own file, so two projects open one after
        the other never see each other's history.
        """
        if self._undo_store is None or self._undo_store.path.parent != Path(
            self.db_path
        ).parent:
            self._undo_store = UndoStore(Path(self.db_path))
        return self._undo_store

    #: The tables that hold what the properties panels and the operations
    #: edit: the figures, their axes, their series, and the per-table notes
    #: and links. Small enough that snapshotting all four is cheaper than
    #: working out which of them an edit will reach.
    DESCRIPTOR_TABLES: ClassVar[tuple[str, ...]] = (
        "__figure_descriptors__",
        "__axis_descriptors__",
        "__series_descriptors__",
        "__table_descriptors__",
    )

    def snapshot_for_undo(
        self,
        tables: Sequence[str],
        *,
        label: str,
        entry_id: int | None = None,
    ) -> int | None:
        """Record the state of *tables* before changing them.

        Call before opening the transaction that does the work: attaching
        the undo database is not allowed inside one. Returns None when
        nothing was recorded, which is not a failure the caller has to
        handle - the action goes ahead either way, it just cannot be taken
        back.

        Pass the returned id back as *entry_id* to add more tables to the
        same entry when what an action touches is only known in stages -
        an operation that creates the axis it then writes a result table
        to. One action, one entry, one step to undo it.
        """
        if self._con is None:
            return None
        return self.undo_store.snapshot(
            self._con, tables, label=label, entry_id=entry_id
        )

    def undo_entries(self) -> list[UndoEntry]:
        """What can be undone, most recent first."""
        if self._con is None:
            return []
        return self.undo_store.entries(self._con)

    def undo_last(self) -> UndoEntry | None:
        """Undo the most recent recorded action and return what it was.

        Everything cached about the database is dropped afterwards: the
        rows behind a series may have just been replaced wholesale, and a
        cache that survived that would serve the state the user has
        undone.
        """
        if self._con is None:
            return None
        entry = self.undo_store.undo(self._con)
        if entry is not None:
            self.invalidate_series_cache()
        return entry

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
