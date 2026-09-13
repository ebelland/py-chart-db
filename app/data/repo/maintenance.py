"""Keeping the file healthy: checks, compaction, saving a copy.

The three that have to be read together, because their order is the point:
``check_database`` reports what is wrong, ``optimize_db`` reports and then
compacts - checks first, because VACUUM rewrites the file and anything it
silently drops would no longer be reportable - and ``save_as`` copies
through VACUUM INTO rather than the filesystem, because in WAL mode the
.dhub on disk is not the whole database until its -wal is folded in.

Part of ``SqliteRepo``; see ``app/data/repo/__init__.py``.
"""
from __future__ import annotations

import re
from pathlib import Path

import app.data.descriptors
from app.data.repo._common import DatabaseReport, ensure_connection_wrapper
from app.logs.logger import applogger


class MaintenanceMixin:
    """Integrity checks, compaction, and saving the project elsewhere."""

    __slots__ = ()

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
    def checkpoint(self) -> None:
        """Fold the WAL into the .dhub file itself, on disk, right now.

        Every change is already committed through SQLite's WAL as it
        happens - there is no "unsaved" state for ``Save`` to flush the way
        there would be for a document editor. What TRUNCATE actually buys is
        the .dhub file being the *whole* database on its own again (WAL mode
        otherwise leaves recent writes in the -wal side file until SQLite
        checkpoints on its own schedule) and that side file shrunk back to
        empty - useful before copying the file, backing it up, or handing it
        to a tool that only reads the .dhub. Safe inside an open transaction,
        unlike VACUUM: a checkpoint does not rewrite the file, only merges
        into it.
        """
        assert self._con is not None
        self._con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        applogger.info("Checkpointed: WAL folded into the database file.")

    @ensure_connection_wrapper
    def save_as(self, target_path: Path) -> Path:
        """Write a compacted copy of this database to *target_path*.

        Uses ``VACUUM INTO`` rather than copying the file on disk: with WAL
        mode active (see __init__), the .dhub file is not the whole database
        on its own until its -wal side file is checkpointed into it, so a
        plain filesystem copy of just the .dhub file can silently miss
        recent writes. ``VACUUM INTO`` instead builds one self-contained,
        up-to-date file directly from the live connection.

        Refuses inside an open transaction, on the same principle as
        :meth:`optimize_db`: SQLite refuses to VACUUM there, and letting the
        error travel up would abort whatever multi-step operation triggered
        it. An existing file at *target_path* is removed first - the target
        came out of a save dialog that has already confirmed the overwrite -
        because VACUUM INTO refuses to write over one itself.
        """
        assert self._con is not None
        target = self.ensure_dhub_extension(target_path)

        if self._con.in_transaction:
            raise RuntimeError(
                "Cannot save as while a transaction is open. Commit first."
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()

        self._con.execute("VACUUM INTO ?", (str(target),))
        applogger.info("Saved a copy of the database to %s", target)
        return target

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

