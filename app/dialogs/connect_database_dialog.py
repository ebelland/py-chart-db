"""Dialog for connecting to another database and picking one of its tables.

Three engines - a local SQLite file, PostgreSQL, MySQL - behind one choice:
connect, list the base tables, pick one. The import dialog on the other side
of this needs to know nothing about which engine was used, only the
:class:`~app.utils.data_sources.DatabaseConnection` and table name this
dialog hands back through ``connection``/``table`` once accepted.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.logs.logger import applogger
from app.styles.style import (
    apply_dialog_shell,
    create_action_button,
    create_card_widget,
    create_section_title,
    load_icon,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.data_sources import (
    DATABASE_FILE_FILTER,
    DEFAULT_PORTS,
    DatabaseConnection,
    list_mysql_tables,
    list_postgres_tables,
    list_sqlite_tables,
)
from app.utils.i18n import _
from app.utils.messages import show_message

ENGINE_SQLITE = "sqlite"
ENGINE_POSTGRES = "postgres"
ENGINE_MYSQL = "mysql"

#: (kind, display label). The label goes through tr() at the call site, not
#: here, so it is a plain literal that xgettext's sweep can still find.
_ENGINE_CHOICES: tuple[tuple[str, str], ...] = (
    (ENGINE_SQLITE, "SQLite file"),
    (ENGINE_POSTGRES, "PostgreSQL"),
    (ENGINE_MYSQL, "MySQL"),
)


class ConnectDatabaseDialog(QDialog):
    """Pick an engine, connect, and choose one of its tables.

    ``connection`` and ``table`` hold the result once accepted - read
    through those rather than a signal, the same way ``LoadDemoDialog.chosen``
    is: the caller wants one answer before it goes on to read the table, not
    an ongoing conversation.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Connect to database"))
        self.setWindowIcon(load_icon("import_database"))
        self.connection: DatabaseConnection | None = None
        self.table: str | None = None

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="small")

        card = create_card_widget(self, "connectDatabaseCard")
        card_layout = QVBoxLayout(card)
        stdSizeAndlayout(card_layout)
        card_layout.addWidget(create_section_title(_("Connect to database"), card))

        form = QFormLayout()
        stdSizeAndlayout(form)

        self._engine = QComboBox(card)
        for key, label in _ENGINE_CHOICES:
            self._engine.addItem(_(label), userData=key)
        self._engine.currentIndexChanged.connect(lambda _i=0: self._on_engine_changed())
        form.addRow(_("Engine"), self._engine)

        # SQLite: a file path plus Browse.
        self._sqlite_row = QWidget(card)
        sqlite_lay = QHBoxLayout(self._sqlite_row)
        stdSizeAndlayout(sqlite_lay)
        self._sqlite_path = QLineEdit(self._sqlite_row)
        sqlite_lay.addWidget(self._sqlite_path, 1)
        create_action_button(
            parent=self._sqlite_row,
            action_id="open",
            action=self._on_browse_sqlite,
            layout=sqlite_lay,
        )
        form.addRow(_("File"), self._sqlite_row)
        self._sqlite_label = form.labelForField(self._sqlite_row)

        # PostgreSQL / MySQL: host, port, database, username, password.
        self._host = QLineEdit(card)
        self._host.setText("localhost")
        stdSizeAndlayout(self._host)
        form.addRow(_("Host"), self._host)
        self._host_label = form.labelForField(self._host)

        self._port = QSpinBox(card)
        self._port.setRange(1, 65535)
        stdSizeAndlayout(self._port)
        form.addRow(_("Port"), self._port)
        self._port_label = form.labelForField(self._port)

        self._database = QLineEdit(card)
        stdSizeAndlayout(self._database)
        form.addRow(_("Database"), self._database)
        self._database_label = form.labelForField(self._database)

        self._username = QLineEdit(card)
        stdSizeAndlayout(self._username)
        form.addRow(_("Username"), self._username)
        self._username_label = form.labelForField(self._username)

        self._password = QLineEdit(card)
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        stdSizeAndlayout(self._password)
        form.addRow(_("Password"), self._password)
        self._password_label = form.labelForField(self._password)

        card_layout.addLayout(form)

        connect_row = QHBoxLayout()
        stdSizeAndlayout(connect_row)
        connect_row.addStretch(1)
        create_action_button(
            parent=card,
            action_id="connect_database",
            action=self._on_connect,
            layout=connect_row,
        )
        card_layout.addLayout(connect_row)

        card_layout.addWidget(create_section_title(_("Table"), card))
        self._tables = QListWidget(card)
        mark_editor_panel(self._tables)
        card_layout.addWidget(self._tables, 1)

        root.addWidget(card, 1)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        create_action_button(
            parent=self, action_id="apply", action=self._confirm, layout=action_row
        )
        create_action_button(
            parent=self, action_id="close", action=self.reject, layout=action_row
        )
        root.addLayout(action_row, 0)

        self._on_engine_changed()

    # ------------------------------------------------------------------
    # Engine switch
    # ------------------------------------------------------------------
    def _on_engine_changed(self) -> None:
        engine = self._engine.currentData()
        is_sqlite = engine == ENGINE_SQLITE
        self._sqlite_row.setVisible(is_sqlite)
        if self._sqlite_label is not None:
            self._sqlite_label.setVisible(is_sqlite)

        for field, label in (
            (self._host, self._host_label),
            (self._port, self._port_label),
            (self._database, self._database_label),
            (self._username, self._username_label),
            (self._password, self._password_label),
        ):
            field.setVisible(not is_sqlite)
            if label is not None:
                label.setVisible(not is_sqlite)

        if not is_sqlite:
            self._port.setValue(DEFAULT_PORTS.get(engine, 0))

        self._tables.clear()

    def _on_browse_sqlite(self) -> None:
        path, _unused = QFileDialog.getOpenFileName(
            self, _("Select database"), str(Path.home()), DATABASE_FILE_FILTER
        )
        if path:
            self._sqlite_path.setText(path)

    # ------------------------------------------------------------------
    # Connecting
    # ------------------------------------------------------------------
    def _current_connection(self) -> DatabaseConnection:
        engine = str(self._engine.currentData())
        if engine == ENGINE_SQLITE:
            return DatabaseConnection(kind="sqlite", path=self._sqlite_path.text().strip())
        return DatabaseConnection(
            kind=engine,
            host=self._host.text().strip(),
            port=int(self._port.value()),
            database=self._database.text().strip(),
            username=self._username.text().strip(),
            password=self._password.text(),
        )

    def _on_connect(self) -> None:
        conn = self._current_connection()

        if conn.kind == "sqlite" and not conn.path:
            show_message(self, "database.connection_missing_file")
            return

        try:
            if conn.kind == "sqlite":
                tables = list_sqlite_tables(conn.path)
            elif conn.kind == "postgres":
                tables = list_postgres_tables(conn)
            else:
                tables = list_mysql_tables(conn)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Could not connect to database: %s", exc)
            show_message(self, "import.database_failed", error=exc)
            return

        self._tables.clear()
        if not tables:
            show_message(self, "import.database_no_tables")
            return

        self._tables.addItems(tables)
        self._tables.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------
    def _confirm(self) -> None:
        item = self._tables.currentItem()
        if item is None:
            show_message(self, "import.database_no_table_selected")
            return
        self.connection = self._current_connection()
        self.table = item.text()
        self.accept()
