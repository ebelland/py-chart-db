"""The chart itself, as rows: figures, axes, series, and their options.

What a .dhub holds besides the data - which figures exist, which axes each
has, which series each axis draws, and the options bags hanging off all
three. Descriptor rows are small and read constantly (every redraw walks
them), and they are the tables the undo store snapshots for a settings
edit.

The options are stored as JSON text in one column rather than as columns:
a renderer's kwargs are open-ended by design - see kwarg_spec - and a
schema that had to grow a column per Matplotlib keyword would be a
migration for every new one.

Part of ``SqliteRepo``; see ``app/data/repo/__init__.py``.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Mapping, Sequence

import app.data.descriptors
from app.data.repo._common import (
    _dumps_json,
    _is_ident,
    _loads_json,
    _quote_ident,
    ensure_connection_wrapper,
)


class DescriptorsMixin:
    """Figure, axis and series descriptors, and their stored options."""

    __slots__ = ()

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

    def apply_axis_layout(
        self,
        *,
        figure_id: int,
        nrows: int,
        ncols: int,
        placements: list[tuple[int, int, dict[str, Any]]],
    ) -> None:
        """Rewrite a figure's grid size and every axis's position/options at
        once - what a layout preset applies in one click.

        *placements* is ``(axis_id, axis_index, option_overrides)`` for every
        axis of the figure, in any order. An override value of ``None``
        removes that key from the axis's options rather than storing null,
        the same convention ``set_axis_options`` callers already follow
        elsewhere; anything else in the axis's existing options - a colour,
        a title - is left alone.

        Every axis_index is rewritten together, not one at a time: the
        UNIQUE(figure_id, axis_index) constraint would reject a plan that
        gives one axis the index another axis is about to give up, in
        whichever order a loop happened to touch them. Two passes side-step
        it the same way :meth:`swap_axis_indexes` does for two axes - here
        for however many a whole layout touches.
        """
        assert self._con is not None
        if not placements:
            return

        with self.transaction(immediate=True):
            self._con.execute(
                "UPDATE __figure_descriptors__ SET nrows = ?, ncols = ? WHERE id = ?",
                (int(nrows), int(ncols), int(figure_id)),
            )

            for offset, (axis_id, _axis_index, _overrides) in enumerate(placements):
                self._con.execute(
                    "UPDATE __axis_descriptors__ SET axis_index = ? WHERE id = ?",
                    (-(offset + 1), int(axis_id)),
                )

            for axis_id, axis_index, overrides in placements:
                row = self._con.execute(
                    "SELECT options_json FROM __axis_descriptors__ WHERE id = ?",
                    (int(axis_id),),
                ).fetchone()
                options = _loads_json(row["options_json"]) if row else {}
                for key, value in overrides.items():
                    if value is None:
                        options.pop(key, None)
                    else:
                        options[key] = value
                self._con.execute(
                    "UPDATE __axis_descriptors__ SET axis_index = ?, options_json = ? WHERE id = ?",
                    (int(axis_index), _dumps_json(options), int(axis_id)),
                )

