"""Database Info: path, size, tables and import links, in one place.

Everything here already existed somewhere - TableListPanel's own context
menu already exports a table and refreshes its link - this is a read-only
overview that puts the whole database's shape (how many tables, how big,
which ones are fed by a link) on screen at once, with those same actions
one click away for whichever row is selected, rather than requiring a
right-click per table.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.styles.style import (
    apply_card_layout,
    apply_dialog_shell,
    create_action_button,
    create_card_widget,
    create_section_title,
    load_icon,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _
from app.utils.import_runner import refresh_link


def _human_size(num_bytes: int) -> str:
    """Return *num_bytes* as "12.3 MB" - the units a person actually reads."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


class DatabaseInfoDialog(QDialog):
    """Path, size, tables and import links for the connected database."""

    def __init__(self, repo: SqliteRepo, parent: QWidget) -> None:
        super().__init__(parent)
        self._repo = repo

        self.setWindowTitle(_("Database Info"))
        self.setWindowIcon(load_icon("database_info"))

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="medium")

        card = create_card_widget(self, "databaseInfoCard")
        card_layout = QVBoxLayout(card)
        apply_card_layout(card_layout)
        card_layout.addWidget(create_section_title(_("Database Info"), card))

        form = QFormLayout()
        form.addRow(_("Path:"), self._selectable_label(str(repo.db_path)))
        form.addRow(_("Size on disk:"), self._selectable_label(self._db_size_text()))
        card_layout.addLayout(form)

        self._table = QTableWidget(0, 3, card)
        self._table.setHorizontalHeaderLabels([_("Table"), _("Rows"), _("Linked")])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.itemSelectionChanged.connect(self._update_button_states)
        mark_editor_panel(self._table)
        card_layout.addWidget(self._table, 1)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        self._export_csv_button = create_action_button(
            parent=self,
            action_id="export_csv",
            action=self._export_selected_csv,
            layout=action_row,
        )
        self._export_xlsx_button = create_action_button(
            parent=self,
            action_id="export_xlsx",
            action=self._export_selected_xlsx,
            layout=action_row,
        )
        self._update_link_button = create_action_button(
            parent=self,
            action_id="update_link",
            action=self._update_selected_link,
            layout=action_row,
            presentation=(
                load_icon("reload"),
                _("Update link"),
                _("Refresh the selected table from its import link"),
            ),
        )
        action_row.addStretch(1)
        create_action_button(
            parent=self, action_id="close", action=self.accept, layout=action_row
        )
        card_layout.addLayout(action_row)

        root.addWidget(card, 1)

        self._reload_tables()
        self._update_button_states()

    def _selectable_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        return label

    def _db_size_text(self) -> str:
        try:
            return _human_size(self._repo.db_path.stat().st_size)
        except OSError:
            return _("unknown")

    def _reload_tables(self) -> None:
        """Repopulate the table list, keeping the current selection by name."""
        selected = self._selected_table()
        frame = self._repo.list_user_tables()

        self._table.setRowCount(len(frame.index))
        for row, record in enumerate(frame.to_dict("records")):
            name = str(record.get("Table", ""))
            has_link = bool(record.get("has_link", False))
            rows = self._repo.row_count(name)

            self._table.setItem(row, 0, QTableWidgetItem(name))
            rows_item = QTableWidgetItem(f"{rows:,}")
            rows_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, 1, rows_item)
            self._table.setItem(row, 2, QTableWidgetItem(_("Yes") if has_link else ""))

            if name == selected:
                self._table.selectRow(row)

    def _selected_table(self) -> str | None:
        row = self._table.currentRow()
        if row < 0:
            return None
        item = self._table.item(row, 0)
        return item.text() if item else None

    def _selected_table_has_link(self) -> bool:
        row = self._table.currentRow()
        if row < 0:
            return False
        item = self._table.item(row, 2)
        return bool(item and item.text())

    def _update_button_states(self) -> None:
        table = self._selected_table()
        self._export_csv_button.setEnabled(table is not None)
        self._export_xlsx_button.setEnabled(table is not None)
        self._update_link_button.setEnabled(
            table is not None and self._selected_table_has_link()
        )

    def _export_selected_csv(self) -> None:
        table = self._selected_table()
        if not table:
            return
        file_path, _unused = QFileDialog.getSaveFileName(
            self, _("Export CSV"), f"{table}.csv", "CSV files (*.csv)"
        )
        if not file_path:
            return
        try:
            self._repo.query_df(f'SELECT * FROM "{table}"').to_csv(
                file_path, index=False
            )
        except Exception as exc:  # noqa: BLE001
            applogger.exception("CSV export failed: %s", exc)

    def _export_selected_xlsx(self) -> None:
        table = self._selected_table()
        if not table:
            return
        file_path, _unused = QFileDialog.getSaveFileName(
            self, _("Export XLSX"), f"{table}.xlsx", "Excel files (*.xlsx)"
        )
        if not file_path:
            return
        try:
            self._repo.query_df(f'SELECT * FROM "{table}"').to_excel(
                file_path, index=False, engine="openpyxl"
            )
        except Exception as exc:  # noqa: BLE001
            applogger.exception("XLSX export failed: %s", exc)

    def _update_selected_link(self) -> None:
        table = self._selected_table()
        if not table:
            return
        link = self._repo.get_table_link(table)
        if not link:
            return

        source = (link.get("settings") or {}).get("source") or {}
        password: str | None = None
        if source.get("kind") in ("postgres", "mysql"):
            entered, ok = QInputDialog.getText(
                self,
                _("Update link"),
                _("Password for {username}@{host}:").format(
                    username=source.get("username", ""), host=source.get("host", "")
                ),
                QLineEdit.EchoMode.Password,
            )
            if not ok:
                return
            password = entered

        try:
            refresh_link(self._repo, link_id=int(link["id"]), password=password)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Link refresh failed: %s", exc)
        self._reload_tables()
