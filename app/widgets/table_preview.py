"""Paged preview of a database table.

Rows are fetched in chunks through a lazy model rather than loaded up front, so
opening a table with millions of rows costs the same as opening a small one.
The panel also hosts the table context-menu operations (hide, filter, column
edit) that are implemented on the repository.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QObject, QPoint, Qt, Signal
from PySide6.QtWidgets import QWidget, QFrame, QTableView, QHeaderView, QInputDialog, QVBoxLayout, QLineEdit
from app.data.data_source import quote_identifier as _quote_table
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.styles.style import MenuItem, create_menu, create_menu_item
from app.utils.messages import ask, show_message
from app.utils.i18n import _


# Rows loaded when previewing a saved query.  A preview is for judging shape
# and content, not for scrolling a whole result set.
QUERY_PREVIEW_ROW_LIMIT: int = 1000


def _type_code(decl_type: str | None) -> str:
    match (decl_type or "").strip().upper():
        case 'BLOB':
            return "BLB"
        case '':
            return "?"
        case 'INT'|'INTEGER'|'TINYINT'|'SMALLINT'|'MEDIUMINT'|'BIGINT'|'UNSIGNED BIG INT'|'INT2'|'INT8':
            return 'I'
        case 'REAL'|'DOUBLE'|'DOUBLE PRECISION'|'FLOAT'|'NUMERIC'|'DECIMAL(10,5)':
            return 'F'
        case 'BOOLEAN':
            return 'B'
        case 'DATE'|'DATETIME':
            return 'DT'
        case _ :
            return 'S'


        


class DataFrameTableModel(QAbstractTableModel):
    """Read-only view over an already-materialised DataFrame.

    Used for saved queries and for the query builder's Run preview.  A query
    result has no rowid, so the chunked model's rowid paging does not apply to
    it; a bounded page loaded up front is simpler and, for a preview, cheaper
    than inventing OFFSET paging that nobody scrolls through.
    """

    def __init__(self, frame: Any, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._frame = frame
        self._columns = [str(column) for column in frame.columns]

    @property
    def frame(self) -> Any:
        """Return the underlying DataFrame."""
        return self._frame

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else int(len(self._frame))

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._columns)

    def column_name(self, index: int) -> str | None:
        """Return the column name at a position, mirroring LazyTableModel."""
        return self._columns[index] if 0 <= index < len(self._columns) else None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        try:
            value = self._frame.iat[index.row(), index.column()]
        except Exception:
            return None
        return "" if value is None else str(value)

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._columns[section] if section < len(self._columns) else None
        return section + 1


class TablePreviewPanel(QWidget):
    """Reusable table preview panel with a context-sensitive right-click menu."""
    refresh = Signal()

    def __init__(self, parent: QWidget, repo:SqliteRepo) -> None:
        super().__init__(parent)
        self._repo = repo
        self._table: str|None = None

        self.view = QTableView(self)
        self.view.setShowGrid(False)
        # Finder's list view: alternating rows carry the eye across a wide
        # row, and the frame is dropped so the list meets the sidebar
        # hairline instead of drawing a second border beside it.
        self.view.setAlternatingRowColors(True)
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view.setSortingEnabled(False)

        # QTableView is backed by a viewport. Right-click events may arrive on
        # the viewport, not on the view, so wire both and also filter events.
        for widget in (self.view, self.view.viewport()):
            widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            widget.installEventFilter(self)
        self.view.customContextMenuRequested.connect(self._show_context_menu)
        self.view.viewport().customContextMenuRequested.connect(self._show_context_menu)

        header = self.view.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setHighlightSections(False)

        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(32)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 1)
        self.setLayout(layout)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched in (self.view, self.view.viewport()):
            event_type = event.type()
            if event_type == QEvent.Type.ContextMenu:
                pos = event.pos()  # type: ignore[attr-defined]
                if watched is self.view:
                    pos = self.view.viewport().mapFrom(self.view, pos)
                self._show_context_menu(pos)
                event.accept()
                return True
            if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:  # type: ignore[attr-defined]
                pos = event.pos()  # type: ignore[attr-defined]
                if watched is self.view:
                    pos = self.view.viewport().mapFrom(self.view, pos)
                self._show_context_menu(pos)
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def clear(self) -> None:
        self._repo = None
        self._table = None
        self.view.setModel(None)

    def set_context(self, repo: Optional[SqliteRepo], table: Optional[str]) -> None:
        if repo is None or not table:
            self.clear()
            return
        self._repo = repo
        self._table = table
        self._reload_model()

    def set_model(self, model) -> None:
        self._repo = None
        self._table = None
        self.view.setModel(model)

    def _reload_model(self) -> None:
        """Show the current source, whether it is a table or a saved query.

        The source is resolved through the repository rather than assumed to be
        a table, which is what lets a saved query be previewed with no second
        code path here.  Queries get the DataFrame model because their result
        has no rowid for the chunked model to page on.
        """
        if self._repo is None or not self._table:
            self.view.setModel(None)
            return

        try:
            source = self._repo.get_data_source(self._table)
            if source is not None and source.is_query:
                frame = self._repo.data_source_page(
                    source, limit=QUERY_PREVIEW_ROW_LIMIT, offset=0
                )
                self.view.setModel(DataFrameTableModel(frame, parent=self.view))
                return

            self.view.setModel(LazyTableModel(self._repo, self._table, parent=self.view))
        except Exception as exc:
            applogger.exception("Preview model init failed for source=%s: %s", self._table, exc)
            self.view.setModel(None)

    def _model(self) -> "LazyTableModel | None":
        model = self.view.model()
        return model if isinstance(model, LazyTableModel) else None

    def _current_column_name(self) -> str | None:
        model = self._model()
        index = self.view.currentIndex()
        if model is None or not index.isValid():
            return None
        return model.column_name(index.column())

    def _show_context_menu(self, pos: QPoint) -> None:
        """Show the preview's right-click menu."""
        menu = self._build_context_menu(pos)
        if menu is None:
            return
        menu.exec(self.view.viewport().mapToGlobal(pos))

    def _build_context_menu(self, pos: QPoint) -> "QMenu | None":
        """Build the right-click menu for *pos*, or None to show nothing.

        Split out from _show_context_menu so a test can inspect the built
        menu's actions without going through QMenu.exec(), which opens a
        real (blocking) local event loop.

        The view is backed by one of two model kinds: LazyTableModel for a
        real table, or the read-only DataFrameTableModel for a saved query
        (see _reload_model). _model() only ever returns a LazyTableModel, so
        gating the whole menu on it being non-None made the menu disappear
        entirely while a query was on screen, instead of just hiding the
        items that write back to a table a query does not have. The guard
        below only requires some model to be loaded; lazy_model still gates
        the table-only items further down.
        """
        if self.view.model() is None:
            return None

        lazy_model = self._model()
        if lazy_model is not None:
            if self._repo is None:
                self._repo = lazy_model.repo
            if self._table is None:
                self._table = lazy_model.table
        if self._repo is None or not self._table:
            return None

        clicked = self.view.indexAt(pos)
        if clicked.isValid():
            self.view.setCurrentIndex(clicked)

        column = self._current_column_name() if lazy_model is not None else None
        items: list[MenuItem | None] = []
        if column:
            items.append(
                MenuItem(
                    text=_("Delete column '{column}'...").format(column=column),
                    tooltip=_("Delete the selected column"),
                    callback=lambda _=False, col=column: self._delete_column(col),
                    icon="delete",
                )
            )

        menu = create_menu(self, items)

        if column:
            hide_menu = menu.addMenu(_("Hide rows by selected column"))
            # The operator is the payload and stays as it is; only the label
            # is translated. Wrapping the operator too would hide rows by
            # comparing against a translated string instead of SQL.
            for label, operator in (
                (_("Equal to..."), "="),
                (_("Different from..."), "!="),
                (_("Lower than..."), "<"),
                (_("Lower or equal..."), "<="),
                (_("Higher than..."), ">"),
                (_("Higher or equal..."), ">="),
            ):
                create_menu_item(
                    parent=self,
                    menu=hide_menu,
                    icon=None,
                    checkable=False,
                    text=label,
                    tooltip=_("Hide rows where {column} {operator} value").format(
                        column=column, operator=operator
                    ),
                    key=None,
                    action=(
                        lambda _=False, col=column, op=operator:
                        self._hide_by_comparison(col, op)
                    ),
                )

            create_menu_item(
                parent=self,
                menu=hide_menu,
                icon=None,
                checkable=False,
                text=_("Hide NULL / empty values"),
                tooltip=_("Hide rows where the selected column is empty"),
                key=None,
                action=(
                    lambda _=False, col=column:
                    self._hide_special(col, "null_or_empty")
                ),
            )
            create_menu_item(
                parent=self,
                menu=hide_menu,
                icon=None,
                checkable=False,
                text=_("Hide rows equal to selected cell"),
                tooltip=_("Hide rows matching the selected cell value"),
                key=None,
                action=self._hide_selected_cell_value,
            )
            menu.addSeparator()

        # Table-writing actions need a real, LazyTableModel-backed table: a
        # saved query has no rowid and nothing in the repo to hide/cluster/add
        # a column to. A query preview keeps only the model-agnostic reload.
        trailing_items: tuple[MenuItem | None, ...] = (
            MenuItem(
                            _("Add column from SQL expression..."),
                            callback=self._add_column_from_expression,
                            icon="add",
                        ),
            None,
            MenuItem(_("Ensure Hide column"), callback=self._ensure_hide,icon="view-conceal"),
            MenuItem(_("Reset Hide to 0"), callback=self._reset_hide, icon=""),
            MenuItem(_("Invert Hide 0 <-> 1"), callback=self._invert_hide,icon="object-flip-vertical"),
            None,
            MenuItem(_("Ensure ClusterId column"), callback=self._ensure_cluster,icon="view-grid"),
            MenuItem(_("Reset Clusters"), callback=self._reset_cluster,icon=""),
            None,
            MenuItem(_("Refresh data table"), callback=self._reload_model, icon="reload"),
        ) if lazy_model is not None else (
            MenuItem(_("Refresh data table"), callback=self._reload_model, icon="reload"),
        )

        for item in trailing_items:
            if item is None:
                menu.addSeparator()
                continue
            create_menu_item(
                parent=self,
                menu=menu,
                icon=item.icon,
                checkable=item.checkable,
                text=item.text,
                tooltip=item.tooltip,
                key=item.shortcut,
                action=item.callback,
            )

        return menu

    def _delete_column(self, column: str) -> None:
        if self._repo is None or not self._table:
            return
        if not ask(self, "preview.confirm_delete_column", column=column):
            return
        try:
            self._repo.delete_table_column(self._table, column)
            self._reload_model()
        except Exception as exc:
            applogger.exception("Delete column failed: %s", exc)
            show_message(self, "preview.delete_column_failed", error=exc)

    def _reset_hide(self) -> None:
        if self._repo and self._table:
            self._repo.clear_hide_column(self._table)
            self._reload_model()

    def _reset_cluster(self) -> None:
        if self._repo and self._table:
            self._repo.clear_cluster_column(self._table)
            self._reload_model()

    def _invert_hide(self) -> None:
        if self._repo and self._table:
            self._repo.invert_hide(self._table)
            self._reload_model()

    def _ensure_hide(self) -> None:
        if self._repo and self._table:
            self._repo.ensure_hide_column(self._table)
            self._reload_model()

    def _ensure_cluster(self) -> None:
        if self._repo and self._table:
            self._repo.ensure_cluster_column(self._table)
            self._reload_model()
            self.refresh.emit()

    def _hide_by_comparison(self, column: str, operator: str) -> None:
        if self._repo is None or not self._table:
            return
        value, ok = QInputDialog.getText(
            self,
            _("Hide rows"),
            f"Hide rows where {column} {operator} value:",
            QLineEdit.EchoMode.Normal,
            "",
        )
        if not ok:
            return
        try:
            count = self._repo.hide_rows_by_value(self._table, column, operator, value)
            self._reload_model()
            show_message(self, "preview.rows_hidden", count=count)
        except Exception as exc:
            applogger.exception("Hide rows failed: %s", exc)
            show_message(self, "preview.hide_rows_failed", error=exc)

    def _hide_special(self, column: str, mode: str) -> None:
        if self._repo is None or not self._table:
            return
        try:
            count = self._repo.hide_rows_special(self._table, column, mode)
            self._reload_model()
            show_message(self, "preview.rows_hidden", count=count)
        except Exception as exc:
            applogger.exception("Hide rows failed: %s", exc)
            show_message(self, "preview.hide_rows_failed", error=exc)

    def _hide_selected_cell_value(self) -> None:
        index = self.view.currentIndex()
        column = self._current_column_name()
        if not index.isValid() or column is None or self._repo is None or not self._table:
            return
        value = index.data(Qt.ItemDataRole.DisplayRole)
        try:
            count = self._repo.hide_rows_by_value(self._table, column, "=", value)
            self._reload_model()
            show_message(self, "preview.rows_hidden", count=count)
        except Exception as exc:
            applogger.exception("Hide selected value failed: %s", exc)
            show_message(self, "preview.hide_rows_failed", error=exc)

    def _add_column_from_expression(self) -> None:
        if self._repo is None or not self._table:
            return
        column_name, ok = QInputDialog.getText(self, _("Add computed column"), _("New column name:"), QLineEdit.EchoMode.Normal, "new_column")
        if not ok:
            return
        column_name = (column_name or "").strip()
        if not column_name:
            return
        expression, ok = QInputDialog.getMultiLineText(
            self,
            "SQL expression",
            "Expression used in UPDATE, e.g. 2 * salary_eur or age / 10.0:",
            "",
        )
        if not ok:
            return
        expression = (expression or "").strip()
        if not expression:
            return
        try:
            self._repo.add_column_from_expression(self._table, column_name, expression)
            self._reload_model()
        except Exception as exc:
            applogger.exception("Add computed column failed: %s", exc)
            show_message(self, "preview.add_column_failed", error=exc)


class LazyTableModel(QAbstractTableModel):
    def __init__(self, repo: SqliteRepo, table: str, parent=None) -> None:
        super().__init__(parent)
        self._repo = repo
        self._table = table
        self._table_q = _quote_table(table)
        self._columns: list[str] = []
        self._codes: list[str] = []
        self._row_count: int = 0
        self._chunk_size = 5000
        self._cache: dict[int, list[list[Any]]] = {}
        self._chunk_rowid: dict[int, int] = {}
        self._select_sql = f"SELECT rowid, * FROM {self._table_q} WHERE rowid > ? ORDER BY rowid LIMIT ?"
        self._load_schema_and_count()

    @property
    def repo(self) -> SqliteRepo:
        return self._repo

    @property
    def table(self) -> str:
        return self._table

    def _load_schema_and_count(self) -> None:
        if self._repo._con is None:
            return
        rows = self._repo.table_info(self._table_q)
        self._columns = [str(row[1]) for row in rows]
        self._codes = [_type_code(str(row[2])) for row in rows]
        self._row_count = int(self._repo.row_count(self._table))

    def column_name(self, index: int) -> str | None:
        return self._columns[index] if 0 <= index < len(self._columns) else None

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self._row_count

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._columns)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._columns):
                return f"{self._columns[section]} ({self._codes[section]})"
            return None
        return str(section + 1)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None
        row_index = index.row()
        column_index = index.column()
        if row_index >= self._row_count or column_index >= len(self._columns):
            return None
        chunk = row_index // self._chunk_size
        offset = row_index % self._chunk_size
        rows = self._cache.get(chunk)
        if rows is None:
            rows = self._fetch_chunk(chunk)
            self._cache[chunk] = rows
        if offset >= len(rows):
            return None
        value = rows[offset][column_index]
        return "" if value is None else str(value)

    def _fetch_chunk(self, chunk_index: int) -> list[list[Any]]:
        last_rowid = self._chunk_rowid.get(chunk_index - 1, 0) if chunk_index > 0 else 0
        if self._repo._con is None:
            return []
        cursor = self._repo._con.execute(self._select_sql, (last_rowid, self._chunk_size))
        rows = cursor.fetchall()
        if not rows:
            return []
        self._chunk_rowid[chunk_index] = int(rows[-1][0])
        return [list(row[1:]) for row in rows]

    def clear_cache(self) -> None:
        self._cache.clear()
        self._chunk_rowid.clear()