"""Undo for actions that change the database, one table at a time.

The obvious design is a copy of the whole ``.dhub`` before each action, and
it is the wrong one twice over: it costs the size of the *project* to undo a
change to one table - half a second and a hundred megabytes to delete a
column - and it reverts everything else that happened since, which is not
what "undo" means to the person who asked for it.

So what is copied is the table an action is about to change, into a
database of its own beside the project (``<project>.undo.db``). An entry is
one user action and may name several tables, because one action usually
touches several: deleting a data table also deletes the series descriptors
that drew it, and a table here is either kind - the imported data or the
chart settings, which live in ``__…__`` tables in the same file and are
restored the same way.

Three things are worth stating about how a table is captured, because each
is a decision that could have gone the other way:

*The schema is kept as SQL, not rebuilt.* A snapshot's rows go into a
plain ``CREATE TABLE … AS SELECT``, which loses declared types, keys and
indexes - fine for holding rows, useless for restoring a descriptor table
with a primary key and foreign keys pointing at it. The original DDL is
read out of ``sqlite_master`` and stored with the snapshot; restoring
recreates the table from that and pours the rows back in, so what comes
back is the table that was there, not an approximation of it.

*A table that did not exist is recorded as absent.* Undoing an action that
*created* a table means dropping it again. Without this the store would
have nothing to say about the commonest change of all - an operation
writing its result - and undo would silently leave it behind.

*Foreign keys are off during a restore.* Recreating three tables that
reference each other cannot be done in an order that satisfies every
constraint at every step, and the state at the end is the state that was
already consistent when it was captured.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from app.data.data_source import quote_identifier as _quote
from app.logs.logger import applogger

#: How many actions can be taken back. Ten is more than anyone reaches for
#: and small enough that the snapshots of ten changes to a big table do not
#: quietly become the largest file in the folder.
MAX_UNDO_ENTRIES: int = 10

#: The schema alias the project's own database is attached under while a
#: snapshot or a restore runs.
_UNDO_SCHEMA: str = "undo_store"

_ENTRIES_TABLE = "__undo_entries__"
_TABLES_TABLE = "__undo_tables__"


@dataclass(frozen=True, slots=True)
class UndoEntry:
    """One undoable action: what it was called, when, and what it touched."""

    entry_id: int
    label: str
    created: str
    tables: tuple[str, ...]

    def describe(self) -> str:
        """A menu-ready description: the action, and what it will bring back."""
        if not self.tables:
            return self.label
        return f"{self.label} ({', '.join(self.tables)})"


class UndoStore:
    """Snapshots of the tables an action is about to change.

    Holds no connection of its own to the project: it borrows the caller's,
    attaches its own file for the length of one snapshot or restore, and
    detaches again. Attaching permanently would make the undo file part of
    every backup, VACUUM and integrity check the project runs.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._undo_path = self._db_path.with_suffix(self._db_path.suffix + ".undo.db")

    @property
    def path(self) -> Path:
        """Where the snapshots are kept."""
        return self._undo_path

    # ------------------------------------------------------------------
    # The undo file
    # ------------------------------------------------------------------
    @contextmanager
    def _attached(self, connection: sqlite3.Connection) -> Iterator[None]:
        """Attach the undo database for the length of one operation.

        ATTACH is not allowed inside a transaction, which is the real reason
        this is a context manager rather than a permanent attachment: a
        caller taking a snapshot is by definition about to open one.
        """
        if connection.in_transaction:
            raise RuntimeError(
                "An undo snapshot has to be taken before the transaction opens."
            )

        self._undo_path.parent.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"ATTACH DATABASE ? AS {_UNDO_SCHEMA}", (str(self._undo_path),)
        )
        try:
            self._ensure_schema(connection)
            yield
        finally:
            try:
                connection.execute(f"DETACH DATABASE {_UNDO_SCHEMA}")
            except sqlite3.Error:
                applogger.exception("Could not detach the undo database.")

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            f"""CREATE TABLE IF NOT EXISTS {_UNDO_SCHEMA}.{_ENTRIES_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    created TEXT NOT NULL
                )"""
        )
        connection.execute(
            f"""CREATE TABLE IF NOT EXISTS {_UNDO_SCHEMA}.{_TABLES_TABLE} (
                    entry_id INTEGER NOT NULL,
                    table_name TEXT NOT NULL,
                    existed INTEGER NOT NULL,
                    ddl TEXT,
                    snapshot TEXT
                )"""
        )

    # ------------------------------------------------------------------
    # Taking a snapshot
    # ------------------------------------------------------------------
    def snapshot(
        self,
        connection: sqlite3.Connection,
        tables: Iterable[str],
        *,
        label: str,
        entry_id: int | None = None,
    ) -> int | None:
        """Record the current state of *tables*, and return the entry id.

        Call it *before* the change, outside any transaction. Returns None
        when nothing could be recorded - a snapshot that fails must not stop
        the action the user actually asked for, so the failure is logged and
        the action goes ahead without an undo entry rather than not at all.

        Pass *entry_id* to add tables to an entry already opened, when what
        an action will touch is only known in stages: applying an operation
        may create the axis it writes to, and the name of the result table
        depends on which axis that turned out to be. One action stays one
        entry, which is what makes it undo in one step.
        """
        wanted = [str(name).strip() for name in tables if str(name).strip()]
        if not wanted:
            return entry_id

        try:
            with self._attached(connection):
                if entry_id is None:
                    entry_id = self._write_entry(connection, wanted, label=label)
                else:
                    self._append_tables(connection, entry_id, wanted)
                self._prune(connection)
            return entry_id
        except Exception:  # noqa: BLE001 - never cost the user their action
            applogger.exception("Could not record an undo snapshot for %r.", label)
            return None

    def _write_entry(
        self,
        connection: sqlite3.Connection,
        tables: Sequence[str],
        *,
        label: str,
    ) -> int:
        cursor = connection.execute(
            f"INSERT INTO {_UNDO_SCHEMA}.{_ENTRIES_TABLE} (label, created) VALUES (?, ?)",
            (label, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        entry_id = int(cursor.lastrowid or 0)
        self._append_tables(connection, entry_id, tables)
        return entry_id

    def _append_tables(
        self,
        connection: sqlite3.Connection,
        entry_id: int,
        tables: Sequence[str],
    ) -> None:
        """Capture more tables into an entry, skipping any already in it."""
        already = {
            str(row[0])
            for row in connection.execute(
                f"SELECT table_name FROM {_UNDO_SCHEMA}.{_TABLES_TABLE} "
                "WHERE entry_id = ?",
                (entry_id,),
            ).fetchall()
        }

        for table in tables:
            if table in already:
                # The first capture is the one to keep: it is the state
                # before the action started, which is where undo goes back to.
                continue
            ddl = self._table_ddl(connection, table)
            if ddl is None:
                # The action is about to create it; undoing means dropping it.
                connection.execute(
                    f"INSERT INTO {_UNDO_SCHEMA}.{_TABLES_TABLE} "
                    "(entry_id, table_name, existed, ddl, snapshot) "
                    "VALUES (?, ?, 0, NULL, NULL)",
                    (entry_id, table),
                )
                continue

            snapshot = f"snap_{entry_id}_{abs(hash(table)) % 10**8}"
            connection.execute(
                f"DROP TABLE IF EXISTS {_UNDO_SCHEMA}.{_quote(snapshot)}"
            )
            connection.execute(
                f"CREATE TABLE {_UNDO_SCHEMA}.{_quote(snapshot)} AS "
                f"SELECT * FROM main.{_quote(table)}"
            )
            connection.execute(
                f"INSERT INTO {_UNDO_SCHEMA}.{_TABLES_TABLE} "
                "(entry_id, table_name, existed, ddl, snapshot) "
                "VALUES (?, ?, 1, ?, ?)",
                (entry_id, table, ddl, snapshot),
            )

    def _table_ddl(self, connection: sqlite3.Connection, table: str) -> str | None:
        """Return the statements that rebuild *table*, or None if it is absent.

        The table's own CREATE plus every index on it, in that order and
        separated by semicolons: an index is part of what the table was, and
        a restore that dropped them would leave a slower database behind
        every time it was used.
        """
        rows = connection.execute(
            "SELECT type, sql FROM main.sqlite_master "
            "WHERE tbl_name = ? AND sql IS NOT NULL "
            "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END",
            (table,),
        ).fetchall()
        statements = [str(row[1]) for row in rows if str(row[0]) in ("table", "index")]
        if not statements or not any(
            str(row[0]) == "table" for row in rows
        ):
            return None
        return ";\n".join(statements)

    def _prune(self, connection: sqlite3.Connection) -> None:
        """Drop the oldest entries past the cap, and their snapshots with them."""
        rows = connection.execute(
            f"SELECT id FROM {_UNDO_SCHEMA}.{_ENTRIES_TABLE} ORDER BY id DESC"
        ).fetchall()
        for row in rows[MAX_UNDO_ENTRIES:]:
            self._delete_entry(connection, int(row[0]))

    def _delete_entry(self, connection: sqlite3.Connection, entry_id: int) -> None:
        for row in connection.execute(
            f"SELECT snapshot FROM {_UNDO_SCHEMA}.{_TABLES_TABLE} WHERE entry_id = ?",
            (entry_id,),
        ).fetchall():
            snapshot = row[0]
            if snapshot:
                connection.execute(
                    f"DROP TABLE IF EXISTS {_UNDO_SCHEMA}.{_quote(str(snapshot))}"
                )
        connection.execute(
            f"DELETE FROM {_UNDO_SCHEMA}.{_TABLES_TABLE} WHERE entry_id = ?", (entry_id,)
        )
        connection.execute(
            f"DELETE FROM {_UNDO_SCHEMA}.{_ENTRIES_TABLE} WHERE id = ?", (entry_id,)
        )

    # ------------------------------------------------------------------
    # Reading the stack
    # ------------------------------------------------------------------
    def entries(self, connection: sqlite3.Connection) -> list[UndoEntry]:
        """Return what can be undone, most recent first."""
        if not self._undo_path.exists():
            return []
        try:
            with self._attached(connection):
                return self._read_entries(connection)
        except Exception:  # noqa: BLE001
            applogger.exception("Could not read the undo history.")
            return []

    def _read_entries(self, connection: sqlite3.Connection) -> list[UndoEntry]:
        found: list[UndoEntry] = []
        for row in connection.execute(
            f"SELECT id, label, created FROM {_UNDO_SCHEMA}.{_ENTRIES_TABLE} "
            "ORDER BY id DESC"
        ).fetchall():
            entry_id = int(row[0])
            tables = [
                str(name[0])
                for name in connection.execute(
                    f"SELECT table_name FROM {_UNDO_SCHEMA}.{_TABLES_TABLE} "
                    "WHERE entry_id = ? ORDER BY rowid",
                    (entry_id,),
                ).fetchall()
            ]
            found.append(
                UndoEntry(
                    entry_id=entry_id,
                    label=str(row[1]),
                    created=str(row[2]),
                    tables=tuple(tables),
                )
            )
        return found

    # ------------------------------------------------------------------
    # Undoing
    # ------------------------------------------------------------------
    def undo(
        self, connection: sqlite3.Connection, entry_id: int | None = None
    ) -> UndoEntry | None:
        """Put back what one entry recorded, and forget it. Most recent first.

        Everything the entry names is restored inside one transaction, so a
        half-restored action is not a state this can leave behind.
        """
        try:
            with self._attached(connection):
                entries = self._read_entries(connection)
                if not entries:
                    return None
                entry = (
                    entries[0]
                    if entry_id is None
                    else next((one for one in entries if one.entry_id == entry_id), None)
                )
                if entry is None:
                    return None

                # Off for the whole restore: three tables that reference each
                # other cannot be recreated in an order that satisfies every
                # constraint at every step. A pragma inside a transaction is
                # a no-op, so this has to be here, outside it.
                connection.execute("PRAGMA foreign_keys = OFF")
                try:
                    with connection:
                        self._restore_entry(connection, entry)
                        self._delete_entry(connection, entry.entry_id)
                finally:
                    connection.execute("PRAGMA foreign_keys = ON")
            return entry
        except Exception:  # noqa: BLE001
            applogger.exception("Could not undo the last change.")
            return None

    def _restore_entry(self, connection: sqlite3.Connection, entry: UndoEntry) -> None:
        for row in connection.execute(
            f"SELECT table_name, existed, ddl, snapshot FROM {_UNDO_SCHEMA}.{_TABLES_TABLE} "
            "WHERE entry_id = ? ORDER BY rowid",
            (entry.entry_id,),
        ).fetchall():
            table, existed, ddl, snapshot = (
                str(row[0]),
                bool(row[1]),
                row[2],
                row[3],
            )
            connection.execute(f"DROP TABLE IF EXISTS main.{_quote(table)}")
            if not existed:
                # It was created by the action being undone.
                continue

            for statement in str(ddl or "").split(";\n"):
                if statement.strip():
                    connection.execute(statement)
            if snapshot:
                connection.execute(
                    f"INSERT INTO main.{_quote(table)} "
                    f"SELECT * FROM {_UNDO_SCHEMA}.{_quote(str(snapshot))}"
                )

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def clear(self, connection: sqlite3.Connection) -> None:
        """Forget every entry. The project itself is untouched."""
        if not self._undo_path.exists():
            return
        try:
            with self._attached(connection):
                for entry in self._read_entries(connection):
                    self._delete_entry(connection, entry.entry_id)
        except Exception:  # noqa: BLE001
            applogger.exception("Could not clear the undo history.")

    def discard_file(self) -> None:
        """Delete the undo database itself - when its project is closed for good."""
        for path in (
            self._undo_path,
            self._undo_path.with_name(self._undo_path.name + "-wal"),
            self._undo_path.with_name(self._undo_path.name + "-shm"),
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                applogger.warning(
                    "Could not remove %s", path, show_dialog=False, raise_error=False
                )
