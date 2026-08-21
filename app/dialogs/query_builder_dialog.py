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

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
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
from app.widgets.table_preview_widget import DataFrameTableModel
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
_SNIPPET_BUILDERS: dict[str, Any] = {
    "select": lambda table: f"SELECT * FROM {table}",
    "join": lambda table: (
        f"SELECT a.*, b.*\n"
        f"FROM {table} AS a\n"
        f'LEFT JOIN "<other_table>" AS b\n'
        f'  ON a."<key>" = b."<key>"'
    ),
    "union": lambda table: (
        f"SELECT * FROM {table}\n"
        "UNION ALL\n"
        'SELECT * FROM "<other_table>"'
    ),
    "summary": lambda table: (
        'SELECT "<group_column>",\n'
        "       COUNT(*) AS n,\n"
        '       AVG("<value_column>") AS mean,\n'
        '       MIN("<value_column>") AS min,\n'
        '       MAX("<value_column>") AS max\n'
        f"FROM {table}\n"
        'GROUP BY "<group_column>"'
    ),
    "order": lambda table: f'SELECT * FROM {table}\nORDER BY "<column>" ASC',
    "filter": lambda table: (
        f"SELECT * FROM {table}\n"
        'WHERE "<column>" > 0\n'
        '  AND "<column>" IS NOT NULL'
    ),
}


class QueryBuilderDialog(QDialog):
    """Create and edit saved queries, and run them to check the result."""

    #: Roughly eight rows.  Enough to pick from without the list dominating the
    #: side panel; it scrolls past that.
    SAVED_LIST_MAX_HEIGHT = 170

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
        """Saved queries on top, available tables below."""
        panel = create_card_widget(self, "queryListCard")
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)

        layout.addWidget(create_section_title(_("Saved queries"), panel))
        self._query_list = QListWidget(panel)
        mark_editor_panel(self._query_list)
        self._query_list.itemSelectionChanged.connect(self._on_saved_selected)
        # Stretch 0 with a capped height: the list is a picker, not the subject
        # of the dialog, and at stretch 2 it swallowed the space the tables and
        # snippets below it needed.
        self._query_list.setMaximumHeight(self.SAVED_LIST_MAX_HEIGHT)
        layout.addWidget(self._query_list, 0)

        layout.addWidget(create_section_title(_("Tables"), panel))
        self._table_combo = QComboBox(panel)
        self._table_combo.setToolTip(
            _("Insert a SELECT over this table at the cursor.")
        )
        stdSizeAndlayout(self._table_combo)
        layout.addWidget(self._table_combo, 0)

        row = QWidget(panel)
        row_layout = QHBoxLayout(row)
        stdSizeAndlayout(row_layout)
        create_action_button(
            parent=row,
            action_id="import",
            action=self._insert_table_select,
            layout=row_layout,
        )
        create_action_button(
            parent=row,
            action_id="delete",
            action=self._delete_selected_query,
            layout=row_layout,
        )
        row_layout.addStretch(1)
        layout.addWidget(row, 0)

        layout.addWidget(create_section_title(_("Snippets"), panel))
        layout.addWidget(self._build_snippet_row(panel), 0)

        return panel

    def _build_snippet_row(self, parent: QWidget) -> QWidget:
        """Buttons that write the shape of a statement into the editor.

        SQL is not hard, but its *shape* is easy to forget: which side of a
        LEFT JOIN keeps its rows, that UNION deduplicates and UNION ALL does
        not, that an aggregate needs its GROUP BY to list every non-aggregated
        column.  Each button inserts a skeleton with the current table already
        in it and the parts to replace spelled out, so the editor starts from
        something that runs.
        """
        row = QWidget(parent)
        row_layout = QHBoxLayout(row)
        stdSizeAndlayout(row_layout)

        for snippet_id in _SNIPPET_BUILDERS:
            create_action_button(
                parent=row,
                action_id=f"sql_{snippet_id}",
                action=lambda _checked=False, key=snippet_id: self._insert_snippet(key),
                layout=row_layout,
            )
        row_layout.addStretch(1)
        return row

    def _build_editor_panel(self) -> QWidget:
        """SQL editor on top, result preview below."""
        panel = create_card_widget(self, "queryEditorCard")
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)

        layout.addWidget(create_section_title(_("SQL"), panel))

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

        layout.addWidget(create_section_title(_("Result preview"), panel))
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
        """Refill the saved-query list, keeping the current selection."""
        previous = self._current_name
        self._query_list.blockSignals(True)
        try:
            self._query_list.clear()
            for saved in self._repo.list_queries():
                item = QListWidgetItem(saved.name)
                item.setToolTip(saved.sql)
                self._query_list.addItem(item)
                if saved.name == previous:
                    self._query_list.setCurrentItem(item)
        finally:
            self._query_list.blockSignals(False)

    def _reload_tables(self) -> None:
        """Refill the table combo used by Insert."""
        self._table_combo.clear()
        try:
            self._table_combo.addItems(self._repo.list_table_names())
        except Exception:
            applogger.exception("Failed to list tables for the query builder")

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

    def _on_saved_selected(self) -> None:
        item = self._query_list.currentItem()
        if item is not None:
            self.load_query(item.text())

    def _on_text_changed(self) -> None:
        """Clear the status: it describes text that no longer exists."""
        self._status.setText("")

    def _insert_table_select(self) -> None:
        """Insert a SELECT over the chosen table at the cursor."""
        table = self._current_table()
        if not table:
            return
        self._editor.insertPlainText(f"SELECT * FROM {quote_identifier(table)}")

    def _current_table(self) -> str:
        """Return the table chosen in the combo, or empty."""
        return self._table_combo.currentText().strip()

    def _insert_snippet(self, snippet_id: str) -> None:
        """Insert one snippet, with the selected table filled in.

        Nothing is guessed beyond the table name: a join needs to know which
        columns match, and inventing a guess there produces a query that runs
        and returns the wrong rows - worse than one that does not run.
        """
        builder = _SNIPPET_BUILDERS.get(snippet_id)
        if builder is None:
            applogger.warning("Unknown SQL snippet %r", snippet_id, show_dialog=False,
                              raise_error=False)
            return

        table = self._current_table() or "table_name"
        text = builder(quote_identifier(table))

        cursor = self._editor.textCursor()
        if cursor.position() > 0 and not self._editor.toPlainText().endswith("\n"):
            text = "\n" + text
        self._editor.insertPlainText(text)
        self._editor.setFocus()

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
        """Validate, ask for a name, and store the query."""
        if not self.validate():
            show_message(self, "query.invalid", error=self._status.text())
            return False

        name, accepted = QInputDialog.getText(
            self,
            _("Save query"),
            _("Query name:"),
            text=self._current_name or "",
        )
        if not accepted:
            return False

        clean = name.strip()
        if not clean:
            show_message(self, "query.needs_a_name")
            return False

        # A query named after a table would be shadowed by it everywhere, so
        # refuse the name instead of saving something unreachable.
        if self._repo.check_if_table_exists(clean):
            show_message(self, "query.name_is_a_table", name=clean)
            return False

        existing = self._repo.get_query(clean)
        if existing is not None and clean != self._current_name:
            if not ask(self, "query.confirm_overwrite", name=clean):
                return False

        try:
            self._repo.save_query(clean, self.sql)
        except Exception as exc:
            applogger.exception("Failed to save query '%s'", clean)
            show_message(self, "query.save_failed", error=exc)
            return False

        self._current_name = clean
        self._reload_queries()
        self._status.setText(f"Saved as '{clean}'.")
        applogger.info("Saved query '%s'.", clean)
        return True

    def _delete_selected_query(self) -> None:
        """Delete the selected saved query after confirmation."""
        item = self._query_list.currentItem()
        if item is None:
            return

        name = item.text()
        if not ask(self, "query.confirm_delete", name=name):
            return

        self._repo.delete_query(name)
        if self._current_name == name:
            self._current_name = None
            self._editor.clear()
        self._reload_queries()
        applogger.info("Deleted query '%s'.", name)

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
