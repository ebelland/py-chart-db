"""The data itself: tables, what is in them, and how it gets there.

Everything that reads or writes a *user* table rather than a descriptor -
listing them, describing their columns, running queries against them,
importing a DataFrame into one, the import links that say where a table
came from, the Hide column the chart filters on, and renaming or deleting
one (which has to reach into the series that draw it, hence
_propagate_table_name).

Part of ``SqliteRepo``; see ``app/data/repo/__init__.py``.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas._typing import DtypeArg

import app.data.descriptors
from app.data.data_source import DataSource
from app.data.repo._common import (
    _RETURNS_ROWS_RE,
    _dumps_json,
    _is_ident,
    _loads_json,
    _quote_ident,
    ensure_connection_wrapper,
    is_read_only_select,
)
from app.logs.logger import applogger


class TablesMixin:
    """User tables: listing, reading, importing, renaming, deleting."""

    __slots__ = ()

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

        # Before the transaction, not inside it: the undo store attaches its
        # own database, which SQLite does not allow mid-transaction. Both
        # tables, because this deletes both - the data, and the series
        # descriptors that drew it.
        self.snapshot_for_undo(
            [table, "__series_descriptors__"], label=f"Delete table '{table}'"
        )

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

        # A dropped column takes its data with it and SQLite has no way back,
        # which is what makes this worth a snapshot of the whole table.
        self.snapshot_for_undo(
            [table_name], label=f"Delete column '{column_name}' from '{table_name}'"
        )

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