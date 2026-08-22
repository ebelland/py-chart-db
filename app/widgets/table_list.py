"""Panel showing the list of user tables (Table / Link / File / Notes)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import isna

import PySide6.QtCore
from PySide6.QtGui import QColor, QPainter, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QStyledItemDelegate, QWidget, QStyleOptionViewItem, QVBoxLayout, QTableView, QAbstractItemView, QFrame, QHeaderView, QFileDialog, QLineEdit, QInputDialog

from app.styles.style import MenuItem, create_menu, load_icon
from app.utils.config import get_section, update_section
from app.data.sqlite_repo import SqliteRepo
from app.utils.import_runner import refresh_link
from app.logs.logger import applogger
from app.utils.messages import ask
from app.utils.i18n import _
from app.widgets.series_operation import SeriesOperationWidget



# Tables written by a series operation are prefixed, so the list can hide the
# whole class of them without keeping a registry of what it produced.
GENERATED_TABLE_PREFIX: str = "_"

CONFIG_SECTION: str = "table_list"
CONFIG_SHOW_GENERATED: str = "show_generated"


def _is_generated(name: str) -> bool:
    """Return True for a table written by a series operation."""
    return str(name).startswith(GENERATED_TABLE_PREFIX)


# ---------------------------------------------------------------------------
# Link badge delegate
# ---------------------------------------------------------------------------
class _LinkDelegate(QStyledItemDelegate):
    """Draws the source badge: a chain link for a linked table, "Q" for a query.

    One delegate for both because they occupy the same column and are mutually
    exclusive - a saved query is never an import target - so a second column
    would be empty in every row.
    """

    _BADGE_SIZE = 16
    _DOT_RADIUS = 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Loaded lazily (not as a class attribute) so the icon is only
        # touched once a QApplication instance is guaranteed to exist.
        self._icon = load_icon("link")

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: PySide6.QtCore.QModelIndex,
    ) -> None:
        super().paint(painter, option, index)

        rect: PySide6.QtCore.QRect = option.rect

        if bool(index.data(TableListPanel.ROLE_IS_QUERY)):
            painter.save()
            painter.setPen(QColor("#c2410c"))
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                rect,
                PySide6.QtCore.Qt.AlignmentFlag.AlignCenter,
                "Q",
            )
            painter.restore()
            return

        if not index.data(PySide6.QtCore.Qt.ItemDataRole.UserRole):
            return

        if not self._icon.isNull():
            size = self._BADGE_SIZE
            x = rect.x() + (rect.width() - size) // 2
            y = rect.y() + (rect.height() - size) // 2
            self._icon.paint(painter, PySide6.QtCore.QRect(x, y, size, size))
            return

        # Fallback: small highlight-coloured dot.
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(option.palette.highlight().color())
        painter.setPen(PySide6.QtCore.Qt.PenStyle.NoPen)
        cx = rect.x() + rect.width() // 2
        cy = rect.y() + rect.height() // 2
        painter.drawEllipse(cx - self._DOT_RADIUS, cy - self._DOT_RADIUS,
                             self._DOT_RADIUS * 2, self._DOT_RADIUS * 2)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: PySide6.QtCore.QModelIndex) -> PySide6.QtCore.QSize:
        return PySide6.QtCore.QSize(36, 32)


# ---------------------------------------------------------------------------
# Custom model – persists Notes edits
# ---------------------------------------------------------------------------
class _TableListModel(QStandardItemModel):
    """Model that writes Notes edits back to the repo as soon as they commit."""

    def __init__(self, panel: "TableListPanel") -> None:
        super().__init__(panel)
        self._panel = panel

    def setData(
        self,
        index: PySide6.QtCore.QModelIndex,
        value: object,
        role: int = PySide6.QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not super().setData(index, value, role):
            return False
        if role == PySide6.QtCore.Qt.ItemDataRole.EditRole and index.column() == self._panel.COL_NOTES:
            self._panel._persist_notes_for_row(index.row())
        return True


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
class TableListPanel(QWidget):
    """Panel: list of user tables with Table / Link / File / Notes columns."""

    tableSelected = PySide6.QtCore.Signal(str)
    actionRequested = PySide6.QtCore.Signal(str, str)

    COL_TABLE = 0
    COL_HAS_LINK = 1
    COL_FILE = 2
    COL_NOTES = 3

    ROLE_TABLE_NAME = PySide6.QtCore.Qt.ItemDataRole.UserRole + 1
    ROLE_HAS_LINK = PySide6.QtCore.Qt.ItemDataRole.UserRole
    ROLE_IS_QUERY = PySide6.QtCore.Qt.ItemDataRole.UserRole + 2

    def __init__(self, repo:SqliteRepo,parent: QWidget ) -> None:
        super().__init__(parent)
        self._repo=repo
        self._reloading = False
        # Generated tables are hidden by default: a project with a few fits has
        # more of them than imported tables, and they are not what the user is
        # looking for in this list.  The choice is remembered in config.json.
        self._show_generated = bool(
            get_section(CONFIG_SECTION).get(CONFIG_SHOW_GENERATED, False)
        )

        self._view = self._build_view()
        self._model = self._build_model()
        self._view.setModel(self._model)

        self._link_delegate = _LinkDelegate(self._view)
        self._view.setItemDelegateForColumn(self.COL_HAS_LINK, self._link_delegate)

        self._configure_headers()

        sel = self._view.selectionModel()
        if sel is not None:
            sel.selectionChanged.connect(self._on_selection_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view, 1)
        self._build_context_menu()
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_view(self) -> QTableView:
        """Build the source list as a Finder sidebar rather than a data grid.

        No frame and no alternating rows: a sidebar is a plain list on the
        sidebar surface, and the stripes that help a wide table be read across
        only add noise to a single column of names.
        """
        view = QTableView(self)
        view.setShowGrid(False)
        view.setFrameShape(QFrame.Shape.NoFrame)
        view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # No inline stylesheet here.  One that set only ``color`` on
        # ::item:selected was what made a selected row two-tone: naming the
        # subcontrol hands Qt the whole item-painting path, so the cell drew a
        # default blue while the view drew its own selection colour behind it.
        # The row's appearance belongs in the .qss with everything else.
        view.setAlternatingRowColors(False)
        view.setSortingEnabled(False)
        view.setContextMenuPolicy(PySide6.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        view.verticalHeader().setVisible(False)
        view.verticalHeader().setDefaultSectionSize(32)
        # Only the Notes column is meant to be editable.
        view.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        
        return view

    def _build_model(self) -> _TableListModel:
        model = _TableListModel(self)
        model.setColumnCount(4)
        model.setHorizontalHeaderLabels(["Table", "Link", "File", "Notes"])
        return model

    def _configure_headers(self) -> None:
        """Configure manually resizable column headers."""
        hh = self._view.horizontalHeader()
        hh.setSectionResizeMode(self.COL_TABLE, QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(self.COL_HAS_LINK, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(self.COL_FILE, QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(self.COL_NOTES, QHeaderView.ResizeMode.Interactive)
        hh.resizeSection(self.COL_TABLE, 180)
        hh.resizeSection(self.COL_HAS_LINK, 40)
        hh.resizeSection(self.COL_FILE, 240)
        hh.resizeSection(self.COL_NOTES, 260)
        hh.setMinimumSectionSize(36)
        hh.setSectionsMovable(False)
        hh.setHighlightSections(False)
        hh.setStretchLastSection(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_repo(self, repo: SqliteRepo) -> None:
        self._repo = repo

    def clear(self) -> None:
        self._reloading = True
        try:
            self._model.removeRows(0, self._model.rowCount())
        finally:
            self._reloading = False

    def reload(self) -> None:
        if self._repo is None:
            self.clear()
            return

        self._reloading = True
        selected_to_emit: str | None = None

        try:
            self._model.removeRows(0, self._model.rowCount())

            # Data sources, not tables: saved queries are listed alongside
            # physical tables and rendered by the same row builder.
            df: pd.DataFrame = self._repo.list_data_sources()
            for row in df.itertuples(index=False):
                if not self._show_generated and _is_generated(str(row.Table)):
                    continue
                self._model.appendRow(self._build_row(row))

            if self._model.rowCount() > 0:
                sm = self._view.selectionModel()
                if sm is None or not sm.hasSelection():
                    self._view.selectRow(0)
                first_idx = self._model.index(0, self.COL_TABLE)
                selected_to_emit = str(
                    self._model.data(first_idx, PySide6.QtCore.Qt.ItemDataRole.DisplayRole)
                )

        except Exception as exc:  # noqa: BLE001 - surfaced via logger, not re-raised
            applogger.exception("Table list failed: %s", exc)

        finally:
            self._reloading = False

        if selected_to_emit:
            self.tableSelected.emit(selected_to_emit)

    def _build_row(self, row) -> list[QStandardItem]:
        """Build the four QStandardItems for one row of the source dataframe."""
        table_name = str(row.Table)
        has_link = bool(row.has_link)
        is_query = str(getattr(row, "kind", "table")) == "query"
        source_path = "" if isna(row.source_path) else str(row.source_path)
        notes = "" if isna(row.Notes) else str(row.Notes)

        it_table = QStandardItem(table_name)
        it_table.setData(table_name, self.ROLE_TABLE_NAME)
        it_table.setData(is_query, self.ROLE_IS_QUERY)
        it_table.setEditable(False)

        # has_link is stored on UserRole; the delegate draws the icon/dot.
        it_link = QStandardItem()
        it_link.setData(has_link, self.ROLE_HAS_LINK)
        it_link.setData(is_query, self.ROLE_IS_QUERY)
        it_link.setToolTip("Saved query" if is_query else ("Imported from a file" if has_link else ""))
        it_link.setEditable(False)
        it_link.setFlags(PySide6.QtCore.Qt.ItemFlag.ItemIsEnabled | PySide6.QtCore.Qt.ItemFlag.ItemIsSelectable)

        # Show filename only; full path goes in the tooltip.
        filename = "" if is_query else (Path(source_path).name if source_path else "")
        it_file = QStandardItem(filename)
        # For a query, list_data_sources puts the SQL in source_path.
        it_file.setToolTip(source_path)
        it_file.setEditable(False)
        if not filename:
            # Dim placeholder so empty cells don't compete visually.
            it_file.setForeground(QColor(128, 128, 128, 80))

        it_notes = QStandardItem(notes)
        it_notes.setEditable(not is_query)

        for item in (it_table, it_link, it_file, it_notes):
            item.setFlags(item.flags() | PySide6.QtCore.Qt.ItemFlag.ItemIsSelectable | PySide6.QtCore.Qt.ItemFlag.ItemIsEnabled)

        return [it_table, it_link, it_file, it_notes]

    @property
    def current(self) -> str | None:
        """Return the currently selected table name, if any."""
        table, _has_link = self._selected_row_info()
        return table

    def current_table(self) -> str | None:
        """Compatibility helper for callers that prefer a method."""
        return self.current

    def select_table(self, table: str) -> None:
        table = (table or "").strip()
        if not table:
            return
        for r in range(self._model.rowCount()):
            idx = self._model.index(r, self.COL_TABLE)
            if str(self._model.data(idx, PySide6.QtCore.Qt.ItemDataRole.DisplayRole)) == table:
                self._view.selectRow(r)
                self._view.scrollTo(idx, QAbstractItemView.ScrollHint.PositionAtCenter)
                return

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist_notes_for_row(self, row: int) -> None:
        if self._reloading or self._repo is None:
            return
        table = self._model.item(row, self.COL_TABLE).data(self.ROLE_TABLE_NAME)
        notes = self._model.item(row, self.COL_NOTES).text()
        if table:
            self._repo.set_table_notes(str(table), notes)

    # ------------------------------------------------------------------
    # Selection & context menu
    # ------------------------------------------------------------------
    def selected_tables(self) -> list[str]:
        """Return all selected table names in visual row order."""
        sm = self._view.selectionModel()
        if sm is None:
            return []
        rows = sorted({index.row() for index in sm.selectedRows(self.COL_TABLE)})
        tables: list[str] = []
        for row in rows:
            item = self._model.item(row, self.COL_TABLE)
            if item is None:
                continue
            table = item.data(self.ROLE_TABLE_NAME)
            if table:
                tables.append(str(table))
        return tables

    def selected_sources(self) -> list[tuple[str, bool]]:
        """Return the selected rows as ``(name, is_query)``, in row order.

        The list holds both kinds and they are deleted differently - a saved
        query is a row in ``__queries__``, not a table, so DROP TABLE on its
        name matches nothing and leaves it exactly where it was.
        """
        selection = self._view.selectionModel()
        if selection is None:
            return []

        rows = sorted({index.row() for index in selection.selectedRows(self.COL_TABLE)})
        sources: list[tuple[str, bool]] = []
        for row in rows:
            item = self._model.item(row, self.COL_TABLE)
            if item is None:
                continue
            name = item.data(self.ROLE_TABLE_NAME)
            if name:
                sources.append((str(name), bool(item.data(self.ROLE_IS_QUERY))))
        return sources

    def _current_source(self) -> tuple[str, bool] | None:
        """Return the current row as ``(name, is_query)``, or None."""
        row = self._current_row()
        if row < 0:
            return None
        item = self._model.item(row, self.COL_TABLE)
        if item is None:
            return None
        name = item.data(self.ROLE_TABLE_NAME)
        if not name:
            return None
        return str(name), bool(item.data(self.ROLE_IS_QUERY))

    def _current_row(self) -> int:
        """Return the current row, falling back to the first selected row."""
        current = self._view.currentIndex()
        if current.isValid():
            return current.row()
        sm = self._view.selectionModel()
        if sm is None:
            return -1
        rows = sm.selectedRows(self.COL_TABLE)
        return rows[0].row() if rows else -1

    def _selected_row_info(self) -> tuple[str | None, bool]:
        """Return the current-row table info.

        Multi-selection is allowed, but Plot/Export/Rename/Link act on the
        current row. Delete uses selected_tables().
        """
        row = self._current_row()
        if row < 0:
            return None, False
        table_item = self._model.item(row, self.COL_TABLE)
        link_item = self._model.item(row, self.COL_HAS_LINK)
        if table_item is None or link_item is None:
            return None, False
        table = table_item.data(self.ROLE_TABLE_NAME)
        has_link = bool(link_item.data(self.ROLE_HAS_LINK))
        return (str(table) if table else None), has_link

    def _on_selection_changed(self, *_args: object) -> None:
        if self._reloading:
            return
        table, _unused = self._selected_row_info()
        if table:
            self.tableSelected.emit(table)

    def _build_context_menu(self) -> None:
        self._view.customContextMenuRequested.connect(
            self._show_context_menu
        )


    def _show_context_menu(self, pos: PySide6.QtCore.QPoint) -> None:
        clicked_index = self._view.indexAt(pos)
        if clicked_index.isValid():
            sm = self._view.selectionModel()
            if sm is not None:
                clicked_row = clicked_index.row()
                clicked_table_index = self._model.index(clicked_row, self.COL_TABLE)

                # Do not call selectRow() here. On a right-click it clears the
                # existing multi-selection. Keep the selected rows exactly as
                # they are and only move the current index used by Plot/Export.
                sm.setCurrentIndex(
                    clicked_table_index,
                    PySide6.QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                )

                # If the user opens the context menu with no selected rows,
                # create a single selection for the clicked row.
                if not sm.hasSelection():
                    sm.select(
                        clicked_table_index,
                        PySide6.QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
                        | PySide6.QtCore.QItemSelectionModel.SelectionFlag.Rows,
                    )

        table, has_link = self._selected_row_info()
        selected_tables = self.selected_tables()
        if (not table and not selected_tables) or self._repo is None:
            return
        parent = self._top_level_parent()
        if parent is None:
            return

        fn = callable(getattr(parent, "_on_new_plot_tab", None))
        # Named for what is actually selected: "Delete selected tables" over a
        # saved query is a menu entry that promises the wrong thing.
        selected_sources = self.selected_sources()
        has_query = any(is_query for _name, is_query in selected_sources)
        if len(selected_sources) > 1:
            delete_text = (
                _("Delete selected items…") if has_query else _("Delete selected tables…")
            )
            delete_tip = (
                _("Delete {count} selected items").format(count=len(selected_sources))
                if has_query
                else _("Delete {count} selected tables").format(
                    count=len(selected_sources)
                )
            )
        else:
            delete_text = _("Delete…")
            delete_tip = (
                _("Delete the selected saved query")
                if has_query
                else _("Delete the selected table")
            )

        items: list[MenuItem | None] = []
        if fn is not None:
            items.append(
                MenuItem(
                    text=_("Plot"),
                    tooltip=_("Create a new plot tab for the current table"),
                    callback=self._on_new_plot_tab,
                    # Built from the same source as the "Plot" row at the top
                    # of the Series Operations panel (SeriesOperationWidget.
                    # plot_operation -> NewPlotTabDialog.Icon), rather than the
                    # unrelated "plot" catalogue/file icon, so the same action
                    # carries the same glyph everywhere it appears.
                    icon=SeriesOperationWidget.plugin_icon(
                        SeriesOperationWidget.plot_operation()
                    ),
                )
            )
            items.append(None)

        items.append(
            MenuItem(
                text=_("Rename…"),
                tooltip=_("Rename current table"),
                callback=self._rename_table,
                icon="rename",
            )
        )
        if has_link:
            items.append(
                MenuItem(
                    text=_("Update link"),
                    tooltip=_("Update current table from its link"),
                    callback=self._refresh_link_for_table,
                    icon="link",
                )
            )
        items.extend(
            [
                MenuItem(
                    text=_("Export → CSV…"),
                    tooltip=_("Export current table as comma separated text file"),
                    callback=self._export_table_csv,
                    icon="export_csv",
                ),
                MenuItem(
                    text=_("Export → XLSX…"),
                    tooltip=_("Export current table as Excel file"),
                    callback=self._export_table_xlsx,
                    icon="export_xlsx",
                ),
                None,
                MenuItem(
                    text=delete_text,
                    tooltip=delete_tip,
                    callback=self._delete_selected,
                    icon="delete",
                ),
                None,
                MenuItem(
                    text=_("Show generated tables"),
                    tooltip=_(
                        "Show the tables written by fits, smoothing, outlier "
                        "and spectral operations. Their names start with an "
                        "underscore."
                    ),
                    callback=self.set_generated_visible,
                    icon="preview",
                    checkable=True,
                    checked=self._show_generated,
                ),
                MenuItem(
                    text=_("Reload tables"),
                    tooltip=_("Reload database tables"),
                    callback=self.reload,
                    icon="reload",
                ),
            ]
        )

        menu = create_menu(self, items)
        menu.exec(self._view.viewport().mapToGlobal(pos))

    def set_generated_visible(self, visible: bool) -> None:
        """Show or hide the tables written by series operations."""
        self._show_generated = bool(visible)
        update_section(CONFIG_SECTION, **{CONFIG_SHOW_GENERATED: self._show_generated})
        self.reload()

    def _on_new_plot_tab(self) -> None:
        """Emit a signal to request a new plot tab for the selected table."""
        parent = self._top_level_parent()
        if parent is None or self._repo is None:
            return
        fn = getattr(parent, "_on_new_plot_tab", None)
        if callable(fn):
            fn()

    def _rename_table(self) -> None:
        """Rename a table and refresh dependent UI."""
        table, _unused = self._selected_row_info()
        new, ok = QInputDialog.getText(self,_("Rename table"),f"New name for table '{table}':",QLineEdit.EchoMode.Normal,table or "",)
        if not ok:
            return
        new = (new or "").strip()
        if not new or new == table or  self._repo is None:
            return

        try:
            self._repo.rename_table(table or "", new)
            self.reload()
            self.update_parent()
            parent = self._top_level_parent()
            if parent is not None:
                parent._table_panel.select_table(new) # type: ignore

        except Exception as exc:  # noqa: BLE001
            applogger.exception("Rename failed: %s", exc)

    def _export_table_csv(self) -> None:
        """Export the selected table to CSV."""
        table, _unused = self._selected_row_info()
        file_path, _unused = QFileDialog.getSaveFileName(self,_("Export CSV"),f"{table}.csv","CSV files (*.csv)",)
        if  file_path and self._repo and table:
            try:
                df = self._repo.query_df(f'SELECT * FROM "{table}"')
                df.to_csv(file_path, index=False)
            except Exception as exc:  # noqa: BLE001
                applogger.exception(f"CSV export failed: {exc}")

    def _export_table_xlsx(self) -> None:
        """Export the selected table to XLSX."""
        table, _unused = self._selected_row_info()
        file_path, _unused = QFileDialog.getSaveFileName(self,_("Export XLSX"),f"{table}.xlsx","Excel files (*.xlsx)",)
        if  file_path and self._repo and table:
            try:
                df = self._repo.query_df(f'SELECT * FROM "{table}"')
                df.to_excel(file_path, index=False, engine="openpyxl")
            except Exception as exc:  # noqa: BLE001
                applogger.exception(f"XLSX export failed: {exc}")

    def _refresh_link_for_table(self) -> None:
        """Refresh the external link associated with a table."""
        table, _unused = self._selected_row_info()
        if self._repo is None or table is None:
            return
        link = self._repo.get_table_link(table)
        if not link:
            applogger.exception(f"No link for this table: {table}")
            return
        try:
            refresh_link(self._repo, link_id=int(link["id"]))
            self.update_parent()
        except Exception as exc:  
            applogger.exception(f"Link refresh failed: {exc}")

    def _delete_selected(self) -> None:
        """Delete the selected tables and saved queries, after confirmation.

        Both kinds appear in this list and each needs its own call: deleting a
        saved query used to run DROP TABLE on its name, which matches nothing,
        so the row was still there when the list reloaded.
        """
        if self._repo is None:
            return

        sources = self.selected_sources()
        if not sources:
            current = self._current_source()
            sources = [current] if current else []
        if not sources:
            return

        if not self._confirm_delete(sources):
            return

        try:
            for name, is_query in sources:
                if is_query:
                    self._repo.delete_query(name)
                    applogger.info("Deleted query '%s'.", name)
                else:
                    self._repo.delete_table(name)
            self.reload()
            self.update_parent()

            # Ensure dependent previews/listeners are notified after delete.
            current_table = self.current
            self.tableSelected.emit(current_table or "")
        except Exception as exc:  # noqa: BLE001
            applogger.exception(f"Delete failed: {exc}")

    def _confirm_delete(self, sources: list[tuple[str, bool]]) -> bool:
        """Ask before deleting, in the wording the selection deserves.

        A saved query gets its own question because its consequence is
        different: the table is gone either way, but a chart built on a query
        stops finding its data.
        """
        if len(sources) == 1:
            name, is_query = sources[0]
            if is_query:
                return ask(self, "query.confirm_delete", name=name)
            return ask(self, "table.confirm_delete", table=name)

        preview = "\n".join(
            f"- {name}" + (" (query)" if is_query else "")
            for name, is_query in sources[:12]
        )
        if len(sources) > 12:
            preview += f"\n... and {len(sources) - 12} more"

        if any(is_query for _name, is_query in sources):
            return ask(
                self, "source.confirm_delete_many", count=len(sources), sources=preview
            )
        return ask(
            self,
            "table.confirm_delete_many",
            count=len(sources),
            tables=preview,
        )
        
    def update_parent(self) -> None:
        """Update the parent window's table list and tabs, if they exist."""
        table, _unused = self._selected_row_info()
        parent = self._top_level_parent()
        if self._repo is None or parent is None:
            return
        if type(parent).__name__ == "MainWindow" :
            parent.refresh2() # type: ignore


    
    def _top_level_parent(self) -> QWidget | None:
        widget: QWidget | None = self

        while widget is not None:
            next_widget = widget.parentWidget()
            if next_widget is None:
                return widget
            widget = next_widget

        return None