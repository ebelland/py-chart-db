"""Dialog for writing, validating and saving SQL queries.

A saved query is stored as text in ``__queries__`` and executed every time it is
read. It is deliberately *not* a SQLite VIEW: a view would freeze the statement
into the schema, would need a migration to change, and would make a query
indistinguishable from a table in ``sqlite_master`` - which is exactly the
distinction the table list has to show.

The dialog does three things and keeps them separate: build the text, prove it
runs, and give it a name. Validation is a ``LIMIT 0`` probe, so it is instant
even against a table with millions of rows; Run is what actually reads data,
and only on demand.
"""
from __future__ import annotations

import re
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QDialogButtonBox,
    QFormLayout,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.data.data_source import quote_identifier
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.styles.style import (
    apply_dialog_shell,
    create_card_widget,
    create_action_button,
    create_section_title,
    load_icon,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.dialog_state import (
    restore_dialog_state,
    restore_window_geometry,
    save_dialog_state,
    save_window_geometry,
)
from app.utils.messages import ask, show_message
from app.widgets.table_preview import DataFrameTableModel
from app.utils.i18n import _

# Rows fetched by Run.  Enough to see whether the query is right, few enough
# that pressing Run on a full-table select is not a mistake.
PREVIEW_ROW_LIMIT: int = 500

# config.json key for the remembered entries and geometry of this dialog.
STATE_KEY: str = "query_builder_dialog"

# Snippet id -> (button label, tooltip).  The order here is the order of the
# buttons, which runs from the most common to the least.
# Each builder receives the already-quoted table name.  Placeholders are
# written in <angle brackets>: they are invalid SQL, so a snippet left
# unfinished fails validation instead of running against the wrong column.
_SNIPPET_IDS = ("select", "join", "union", "summary", "order", "filter")


class QueryBuilderDialog(QDialog):
    """Create and edit saved queries, and run them to check the result."""

    def __init__(
        self,
        repo: SqliteRepo,
        *,
        query_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Open the builder, optionally on an existing saved query."""
        super().__init__(parent)
        self._repo = repo
        self._current_name: str | None = None

        self.setWindowTitle(_("Query Builder"))
        self.setWindowIcon(load_icon("data"))

        self._build_ui()
        self._reload_queries()
        self._reload_tables()

        # Restore after the lists are populated: a combo or a splitter cannot
        # be restored to a row that does not exist yet.
        restore_window_geometry(self, STATE_KEY)
        restore_dialog_state(self, STATE_KEY)

        if query_name:
            self.load_query(query_name)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Build the three-part shell: saved list, editor, result preview."""
        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="large")

        # Kept on self so that restore_dialog_state can remember its sizes.
        splitter = self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_side_panel())
        splitter.addWidget(self._build_editor_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 740])

        root.addWidget(splitter, 1)
        root.addLayout(self._build_action_row(), 0)

    def _build_side_panel(self) -> QWidget:
        """Build the settings panel shown on the left."""
        panel = create_card_widget(self, "querySettingsCard")
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)

        layout.addWidget(create_section_title(_("Saved queries:"), panel))
        self._query_combo = QComboBox(panel)
        self._query_combo.setToolTip(_("Pick a saved query."))
        self._query_combo.currentTextChanged.connect(self._on_saved_query_changed)
        layout.addWidget(self._query_combo, 0)

        add_row = QWidget(panel)
        add_layout = QHBoxLayout(add_row)
        stdSizeAndlayout(add_layout)
        create_action_button(parent=add_row, action_id="add",
                             action=self._new_query, layout=add_layout)
        add_layout.addStretch(1)
        layout.addWidget(add_row, 0)

        layout.addWidget(create_section_title(_("Tables:"), panel))
        self._table_combo = QComboBox(panel)
        self._table_combo.currentTextChanged.connect(self._reload_fields)
        layout.addWidget(self._table_combo, 0)

        layout.addWidget(create_section_title(_("Fields:"), panel))
        self._field_combo = QComboBox(panel)
        self._field_combo.setToolTip(_("The field the SQL buttons below act on."))
        layout.addWidget(self._field_combo, 0)

        layout.addWidget(create_section_title(_("Snippets:"), panel))
        layout.addWidget(self._build_snippet_rows(panel), 0)
        layout.addStretch(1)
        return panel

    def _build_snippet_rows(self, parent: QWidget) -> QWidget:
        """Build the six SQL command buttons in two rows of three."""
        container = QWidget(parent)
        outer = QVBoxLayout(container)
        stdSizeAndlayout(outer)
        for first in (0, 3):
            row = QWidget(container)
            row_layout = QHBoxLayout(row)
            stdSizeAndlayout(row_layout)
            for snippet_id in _SNIPPET_IDS[first:first + 3]:
                create_action_button(
                    parent=row,
                    action_id=f"sql_{snippet_id}",
                    action=lambda _checked=False, key=snippet_id: self._apply_command(key),
                    layout=row_layout,
                )
            row_layout.addStretch(1)
            outer.addWidget(row)
        return container

    def _build_editor_panel(self) -> QWidget:
        """SQL editor on top, result preview below."""
        panel = create_card_widget(self, "queryEditorCard")
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)

        layout.addWidget(create_section_title(_("SQL:"), panel))

        self._editor = QPlainTextEdit(panel)
        self._editor.setPlaceholderText(_("SELECT x, y FROM my_table WHERE ..."))
        self._editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._editor.textChanged.connect(self._on_text_changed)
        mark_editor_panel(self._editor)
        layout.addWidget(self._editor, 2)

        self._status = QLabel("", panel)
        self._status.setWordWrap(True)
        self._status.setProperty("muted", True)
        layout.addWidget(self._status, 0)

        layout.addWidget(create_section_title(_("Result preview:"), panel))
        self._result_view = QTableView(panel)
        self._result_view.setAlternatingRowColors(True)
        mark_editor_panel(self._result_view)
        layout.addWidget(self._result_view, 3)

        return panel

    def _build_action_row(self) -> QHBoxLayout:
        """Run on the left, Save / Close on the right."""
        row = QHBoxLayout()
        stdSizeAndlayout(row)

        # No separate Preview button: it only called validate(), which run()
        # already does before executing, so the two buttons differed only in
        # whether you also got the rows.
        create_action_button(
            parent=self,
            action_id="run",
            action=self.run,
            layout=row,
        )
        row.addStretch(1)
        create_action_button(
            parent=self,
            action_id="apply",
            action=self.save,
            layout=row,
        )
        create_action_button(
            parent=self,
            action_id="close",
            action=self.reject,
            layout=row,
        )
        return row

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def sql(self) -> str:
        """Return the SQL currently in the editor."""
        return self._editor.toPlainText().strip()

    @property
    def current_name(self) -> str | None:
        """Return the name of the loaded query, if any."""
        return self._current_name

    def _reload_queries(self) -> None:
        """Refill the saved-query drop-down without synthetic entries."""
        previous = self._current_name
        self._query_combo.blockSignals(True)
        try:
            self._query_combo.clear()
            selected = -1
            for saved in self._repo.list_queries():
                self._query_combo.addItem(saved.name, saved.name)
                index = self._query_combo.count() - 1
                self._query_combo.setItemData(index, saved.sql, Qt.ItemDataRole.ToolTipRole)
                if saved.name == previous:
                    selected = index
            self._query_combo.setCurrentIndex(selected)
        finally:
            self._query_combo.blockSignals(False)

    def _reload_tables(self) -> None:
        """Refill the table and field drop-downs."""
        self._table_combo.blockSignals(True)
        try:
            self._table_combo.clear()
            self._table_combo.addItems(self._repo.list_table_names())
        except Exception:
            applogger.exception("Failed to list tables for the query builder")
        finally:
            self._table_combo.blockSignals(False)
        self._reload_fields()

    def _reload_fields(self, _table: str = "") -> None:
        """Show the fields belonging to the selected table."""
        self._field_combo.clear()
        table = self._current_table()
        if table:
            self._field_combo.addItems(self._repo.get_columns(table))

    def load_query(self, name: str) -> None:
        """Load a saved query into the editor."""
        saved = self._repo.get_query(name)
        if saved is None:
            applogger.warning("Saved query '%s' not found.", name)
            return
        self._current_name = saved.name
        self._editor.setPlainText(saved.sql)
        self._reload_queries()
        self.validate()

    def _on_saved_query_changed(self, _text: str) -> None:
        name = self._query_combo.currentData()
        if name:
            self.load_query(str(name))

    def _new_query(self) -> None:
        """Create an unsaved named query descriptor through the repository."""
        name, accepted = QInputDialog.getText(
            self, _("New query"), _("Query name:")
        )
        if not accepted:
            return
        clean = name.strip()
        if not clean:
            show_message(self, "query.needs_a_name")
            return
        if self._repo.check_if_table_exists(clean):
            show_message(self, "query.name_is_a_table", name=clean)
            return
        if self._repo.get_query(clean) is not None:
            if not ask(self, "query.confirm_overwrite", name=clean):
                return

        query = self._repo.new_query(name=clean)
        self._current_name = query.name
        self._editor.setPlainText(query.sql)
        self._result_view.setModel(None)
        self._status.setText(_("New query, not saved yet."))
        self._query_combo.blockSignals(True)
        try:
            self._query_combo.setCurrentIndex(-1)
        finally:
            self._query_combo.blockSignals(False)
        self._editor.setFocus()

    def _on_text_changed(self) -> None:
        self._status.setText("")

    def _current_table(self) -> str:
        return self._table_combo.currentText().strip()

    def _current_field(self) -> str:
        return self._field_combo.currentText().strip()

    def _require_table_field(self) -> tuple[str, str] | None:
        table, field = self._current_table(), self._current_field()
        if not table or not field:
            self._status.setText(_("Pick a table and a field."))
            return None
        return table, field

    def _apply_command(self, command: str) -> None:
        handlers = {
            "select": self._select_field,
            "order": self._order_by_field,
            "filter": self._filter_by_field,
            "join": self._build_join,
            "union": self._build_union,
            "summary": self._build_summary,
        }
        handler = handlers.get(command)
        if handler:
            handler()

    def _select_field(self) -> None:
        selected = self._require_table_field()
        if not selected:
            return
        table, field = selected
        self._editor.setPlainText(
            f"SELECT {quote_identifier(field)}\nFROM {quote_identifier(table)}"
        )

    @staticmethod
    def _remove_clause(sql: str, clause: str, following: str) -> str:
        pattern = rf"(?is)\s+{clause}\b.*?(?=\s+(?:{following})\b|$)"
        return re.sub(pattern, "", sql).strip().rstrip(";")

    def _order_by_field(self) -> None:
        selected = self._require_table_field()
        if not selected:
            return
        _table, field = selected
        sql = self.sql or f"SELECT * FROM {quote_identifier(self._current_table())}"
        sql = self._remove_clause(sql, r"ORDER\s+BY", r"LIMIT|OFFSET")
        self._editor.setPlainText(f"{sql}\nORDER BY {quote_identifier(field)} ASC")

    def _filter_by_field(self) -> None:
        selected = self._require_table_field()
        if not selected:
            return
        table, field = selected
        sql = self.sql or f"SELECT * FROM {quote_identifier(table)}"
        sql = self._remove_clause(
            sql, "WHERE", r"GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET"
        )
        match = re.search(r"(?is)\s+(GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET)\b", sql)
        predicate = f"WHERE {quote_identifier(field)} = "
        if match:
            sql = f"{sql[:match.start()].rstrip()}\n{predicate}\n{sql[match.start():].lstrip()}"
        else:
            sql = f"{sql.rstrip()}\n{predicate}"
        self._editor.setPlainText(sql)

    def _build_join(self) -> None:
        tables = self._repo.list_table_names()
        if len(tables) < 2:
            self._status.setText(_("A JOIN needs at least two tables."))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(_("Build a JOIN"))
        form = QFormLayout(dialog)
        left_table, right_table = QComboBox(dialog), QComboBox(dialog)
        left_field, right_field = QComboBox(dialog), QComboBox(dialog)
        join_type = QComboBox(dialog)
        left_table.addItems(tables); right_table.addItems(tables)
        right_table.setCurrentIndex(1)
        join_type.addItems(["INNER", "LEFT", "RIGHT", "FULL OUTER", "CROSS"])

        def reload_left() -> None:
            left_field.clear(); left_field.addItems(self._repo.get_columns(left_table.currentText()))
        def reload_right() -> None:
            right_field.clear(); right_field.addItems(self._repo.get_columns(right_table.currentText()))
        left_table.currentTextChanged.connect(reload_left)
        right_table.currentTextChanged.connect(reload_right)
        reload_left(); reload_right()
        form.addRow(_("Main table:"), left_table)
        form.addRow(_("Table to join:"), right_table)
        form.addRow(_("Main field:"), left_field)
        form.addRow(_("Field to join on:"), right_field)
        form.addRow(_("JOIN type:"), join_type)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        lt, rt = left_table.currentText(), right_table.currentText()
        kind = join_type.currentText()
        sql = (
            f"SELECT a.*, b.*\nFROM {quote_identifier(lt)} AS a\n"
            f"{kind} JOIN {quote_identifier(rt)} AS b"
        )
        if kind != "CROSS":
            sql += (
                f"\n  ON a.{quote_identifier(left_field.currentText())} = "
                f"b.{quote_identifier(right_field.currentText())}"
            )
        self._editor.setPlainText(sql)

    def _build_union(self) -> None:
        tables = self._repo.list_table_names()
        if len(tables) < 2:
            self._status.setText(_("A UNION needs at least two tables."))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(_("Build a UNION"))
        form = QFormLayout(dialog)
        first, second, union_type = QComboBox(dialog), QComboBox(dialog), QComboBox(dialog)
        first.addItems(tables); second.addItems(tables); second.setCurrentIndex(1)
        union_type.addItems(["UNION ALL", "UNION"])
        form.addRow(_("First table:"), first)
        form.addRow(_("Second table:"), second)
        form.addRow(_("Type:"), union_type)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        first_cols = self._repo.get_columns(first.currentText())
        second_cols = set(self._repo.get_columns(second.currentText()))
        common = [col for col in first_cols if col in second_cols]
        if not common:
            self._status.setText(_("The tables have no fields in common to UNION."))
            return
        fields = ", ".join(quote_identifier(col) for col in common)
        self._editor.setPlainText(
            f"SELECT {fields}\nFROM {quote_identifier(first.currentText())}\n"
            f"{union_type.currentText()}\n"
            f"SELECT {fields}\nFROM {quote_identifier(second.currentText())}"
        )

    def _build_summary(self) -> None:
        selected = self._require_table_field()
        if not selected:
            return
        table, field = selected
        info = self._repo.table_info(table)
        declared = next((str(row[2]).upper() for row in info if str(row[1]) == field), "")
        numeric = any(token in declared for token in ("INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE"))
        qtable, qfield = quote_identifier(table), quote_identifier(field)
        if numeric:
            sql = (
                f"SELECT COUNT(*) AS n,\n"
                f"       SUM({qfield}) AS totale,\n"
                f"       AVG({qfield}) AS media,\n"
                f"       MIN({qfield}) AS minimo,\n"
                f"       MAX({qfield}) AS massimo\nFROM {qtable}"
            )
        else:
            sql = (
                f"SELECT {qfield}, COUNT(*) AS occorrenze\nFROM {qtable}\n"
                f"GROUP BY {qfield}\nORDER BY occorrenze DESC"
            )
        self._editor.setPlainText(sql)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def validate(self) -> bool:
        """Check the statement without reading rows; report either way."""
        ok, message = self._repo.validate_query(self.sql)
        self._status.setText(("OK - " if ok else "Error - ") + message)
        return ok

    def run(self) -> bool:
        """Execute the statement and show the first rows."""
        if not self.validate():
            return False

        try:
            frame = self._repo.query_df(
                f"SELECT * FROM ({self.sql.rstrip(';')}) AS _preview "
                f"LIMIT {PREVIEW_ROW_LIMIT}"
            )
        except Exception as exc:
            applogger.exception("Query preview failed")
            self._status.setText(f"Error - {exc}")
            return False

        self._result_view.setModel(DataFrameTableModel(frame, parent=self._result_view))
        self._status.setText(
            f"{len(frame)} row(s) shown"
            + (f" (limited to {PREVIEW_ROW_LIMIT})" if len(frame) >= PREVIEW_ROW_LIMIT else "")
        )
        return True

    def save(self) -> bool:
        """Validate and persist the current named query."""
        if not self.validate():
            show_message(self, "query.invalid", error=self._status.text())
            return False
        clean = (self._current_name or "").strip()
        if not clean:
            show_message(self, "query.needs_a_name")
            return False
        try:
            self._repo.save_query(clean, self.sql)
        except Exception as exc:
            applogger.exception("Failed to save query '%s'", clean)
            show_message(self, "query.save_failed", error=exc)
            return False
        self._reload_queries()
        self._status.setText(f"Saved as '{clean}'.")
        applogger.info("Saved query '%s'.", clean)
        return True

    def accept(self) -> None:  # noqa: D102 - Qt override
        self._remember_state()
        self.save()
        super().accept()

    def reject(self) -> None:  # noqa: D102 - Qt override
        self._remember_state()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802, D102 - Qt override
        self._remember_state()
        super().closeEvent(event)

    def _remember_state(self) -> None:
        """Persist the splitter, the selected table and the geometry."""
        save_dialog_state(self, STATE_KEY)
        save_window_geometry(self, STATE_KEY)

    def settings(self) -> dict[str, Any]:
        """Return the dialog state, for callers that want to reopen it."""
        return {"name": self._current_name, "sql": self.sql}
