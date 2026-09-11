"""Dialog for connecting to another database and picking one of its tables.

Three engines - a local SQLite file, PostgreSQL, MySQL - behind one choice:
connect, list the base tables, pick one. The import dialog on the other side
of this needs to know nothing about which engine was used, only the
:class:`~app.utils.data_sources.DatabaseConnection` and table name this
dialog hands back through ``connection``/``table`` once accepted.

Three things about the shape of it:

*Two columns.* The connection details are a form that is done with once it
is filled in; the table list is what the user is actually here to read, and
a server with two hundred tables in a list six rows tall is not browsable.
So the form takes the left column at its natural width and the list takes
the whole right column and every pixel the dialog is given.

*Database is a list, not a blank.* On a server the database name had to be
typed from memory, exactly, before anything at all could be listed - and a
typo answered with a connection error rather than with the four names it
could have offered. Connect now asks the server what it has (see
``SERVER_DATABASE_CATALOGUES``) and fills the combo, and picking one lists
its tables. It stays editable: a login may be allowed to open a database it
is not allowed to see in the catalogue.

*It remembers.* The engine, host, port, database, username and table of the
last accepted connection come back the next time this opens, because they
are the same ones almost every time. The password does not, and is not
stored anywhere - see ``config.get_connect_database_config``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.logs.logger import applogger
from app.styles.style import (
    apply_dialog_shell,
    configure_combo_width,
    create_action_button,
    create_card_widget,
    create_compact_section_title,
    load_icon,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.config import (
    get_connect_database_config,
    set_connect_database_config,
)
from app.utils.data_sources import (
    DATABASE_FILE_FILTER,
    DEFAULT_PORTS,
    SERVER_DATABASE_CATALOGUES,
    DatabaseConnection,
    list_mysql_tables,
    list_postgres_tables,
    list_sqlite_tables,
)
from app.utils.i18n import _
from app.utils.messages import show_message

ENGINE_SQLITE = "sqlite"
ENGINE_DHUB = "dhub"
ENGINE_POSTGRES = "postgres"
ENGINE_MYSQL = "mysql"

#: A .dhub file *is* a SQLite file - this app's own format, nothing more -
#: so ENGINE_DHUB shares every connect/list-tables code path ENGINE_SQLITE
#: does (see _current_connection, _on_engine_changed). It exists as its own
#: choice, not a variant someone has to know to pick "SQLite file" for,
#: because "connect to another one of this app's own projects" is a
#: different question in a user's head than "connect to a SQLite file", even
#: though the two are the same thing underneath - and the file dialog it
#: opens defaults to *.dhub first rather than leaving it to guess among four
#: extensions.
#:
#: (kind, display label). The label goes through tr() at the call site, not
#: here, so it is a plain literal that xgettext's sweep can still find.
_ENGINE_CHOICES: tuple[tuple[str, str], ...] = (
    (ENGINE_SQLITE, "SQLite file"),
    (ENGINE_DHUB, "Another ChartLibre project (.dhub)"),
    (ENGINE_POSTGRES, "PostgreSQL"),
    (ENGINE_MYSQL, "MySQL"),
)
_SQLITE_LIKE_ENGINES: frozenset[str] = frozenset({ENGINE_SQLITE, ENGINE_DHUB})

#: The file dialog filter for ENGINE_DHUB - narrower than data_sources.
#: DATABASE_FILE_FILTER's four-extension SQLite filter, because picking this
#: engine already said the file being looked for is a ChartLibre project.
_DHUB_FILE_FILTER: str = "ChartLibre project (*.dhub);;All files (*.*)"

#: How wide the connection column is allowed to get. The form is fixed-length
#: content - a host, a port, a name - so anything past this is width the
#: table list could be using instead.
_FORM_COLUMN_MAX_WIDTH: int = 340


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

        #: The table named by the remembered connection, selected once a
        #: listing actually contains it. Kept as a field rather than applied
        #: at build time because there is no list to select it in until the
        #: user has connected.
        self._remembered_table: str = ""

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="medium")

        columns = QHBoxLayout()
        stdSizeAndlayout(columns)
        columns.addWidget(self._build_connection_card(), 0)
        columns.addWidget(self._build_table_card(), 1)
        root.addLayout(columns, 1)

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
        self._restore_remembered_connection()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_connection_card(self) -> QWidget:
        """The left column: everything needed to open the connection."""
        card = create_card_widget(self, "connectDatabaseCard")
        card.setMaximumWidth(_FORM_COLUMN_MAX_WIDTH)
        card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        card_layout = QVBoxLayout(card)
        stdSizeAndlayout(card_layout)
        card_layout.addWidget(create_compact_section_title(_("Connection"), card))

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

        # Editable: Connect fills it with what the server reports, but a
        # login may be allowed to open a database that the catalogue query
        # does not return, and typing one has to keep working.
        self._database = QComboBox(card)
        self._database.setEditable(True)
        self._database.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        configure_combo_width(self._database)
        self._database.activated.connect(lambda _i=0: self._on_database_chosen())
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
        card_layout.addStretch(1)
        return card

    def _build_table_card(self) -> QWidget:
        """The right column: the tables, given the room to be read in."""
        card = create_card_widget(self, "connectTablesCard")
        card_layout = QVBoxLayout(card)
        stdSizeAndlayout(card_layout)
        card_layout.addWidget(create_compact_section_title(_("Tables"), card))

        self._tables = QListWidget(card)
        mark_editor_panel(self._tables)
        # A tighter row height than QListWidget's default: what a user does
        # here is scan a lot of names, not read one at a time, and a server
        # with two hundred tables in six-row-tall entries defeats the whole
        # point of giving this list the entire right column.
        self._tables.setUniformItemSizes(True)
        self._tables.setSpacing(0)
        self._tables.setStyleSheet("QListWidget::item { padding: 2px 4px; }")
        # Double-click is the same answer as picking and pressing OK, and it
        # is the one a file-list gesture reaches for first.
        self._tables.itemDoubleClicked.connect(lambda _item: self._confirm())
        card_layout.addWidget(self._tables, 1)
        return card

    # ------------------------------------------------------------------
    # Engine switch
    # ------------------------------------------------------------------
    def _on_engine_changed(self) -> None:
        engine = self._engine.currentData()
        is_sqlite = engine in _SQLITE_LIKE_ENGINES
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

        # Both listings belong to the engine that was connected to, not to
        # the one now selected: a Postgres database name in a MySQL combo is
        # worse than an empty one.
        self._clear_database_choices()
        self._tables.clear()

    def _on_browse_sqlite(self) -> None:
        is_dhub = self._engine.currentData() == ENGINE_DHUB
        path, _unused = QFileDialog.getOpenFileName(
            self,
            _("Select project") if is_dhub else _("Select database"),
            str(Path.home()),
            _DHUB_FILE_FILTER if is_dhub else DATABASE_FILE_FILTER,
        )
        if path:
            self._sqlite_path.setText(path)

    # ------------------------------------------------------------------
    # Connecting
    # ------------------------------------------------------------------
    def _current_connection(self) -> DatabaseConnection:
        engine = str(self._engine.currentData())
        if engine in _SQLITE_LIKE_ENGINES:
            return DatabaseConnection(kind="sqlite", path=self._sqlite_path.text().strip())
        return DatabaseConnection(
            kind=engine,
            host=self._host.text().strip(),
            port=int(self._port.value()),
            database=self._database.currentText().strip(),
            username=self._username.text().strip(),
            password=self._password.text(),
        )

    def _on_connect(self) -> None:
        """Connect, and list whatever this engine has to be picked from.

        Two steps for a server - the databases, then the tables of the one
        selected - and one for SQLite, where the file already is the
        database.
        """
        conn = self._current_connection()

        if conn.kind == "sqlite":
            if not conn.path:
                show_message(self, "database.connection_missing_file")
                return
            self._list_tables(conn)
            return

        if not self._list_databases(conn):
            return
        # With the combo now filled, re-read it: the database being listed
        # may not be the one that was typed before connecting.
        self._list_tables(self._current_connection())

    def _list_databases(self, conn: DatabaseConnection) -> bool:
        """Fill the database combo from the server. True when it succeeded."""
        list_databases = SERVER_DATABASE_CATALOGUES.get(conn.kind)
        if list_databases is None:
            return True

        try:
            databases = list_databases(conn)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Could not list the databases: %s", exc)
            show_message(self, "import.database_failed", error=exc)
            return False

        if not databases:
            # Not a failure: a login with rights to exactly one database and
            # no catalogue access sees this, and typing the name still works.
            show_message(self, "import.database_no_databases")
            return True

        self._fill_database_choices(databases, keep=conn.database)
        return True

    def _list_tables(self, conn: DatabaseConnection) -> None:
        """Fill the table list for one database, reporting any failure."""
        if conn.kind != "sqlite" and not conn.database:
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
        self._select_remembered_table()

    def _on_database_chosen(self) -> None:
        """List the tables of the database just picked from the combo.

        ``activated`` rather than ``currentTextChanged``: the combo is
        editable, and a connection attempt per keystroke is not what typing a
        name should cost.
        """
        conn = self._current_connection()
        if conn.kind == "sqlite":
            return
        self._list_tables(conn)

    # ------------------------------------------------------------------
    # The database combo
    # ------------------------------------------------------------------
    def _clear_database_choices(self) -> None:
        """Empty the combo without losing what the user typed into it."""
        typed = self._database.currentText()
        blocked = self._database.blockSignals(True)
        try:
            self._database.clear()
            self._database.setCurrentText(typed)
        finally:
            self._database.blockSignals(blocked)

    def _fill_database_choices(self, databases: list[str], *, keep: str = "") -> None:
        """Show *databases*, staying on *keep* when the server still has it."""
        blocked = self._database.blockSignals(True)
        try:
            self._database.clear()
            self._database.addItems(databases)
            index = self._database.findText(keep) if keep else -1
            self._database.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self._database.blockSignals(blocked)

    def _select_remembered_table(self) -> None:
        """Land on the table this connection ended on last time, if it is here."""
        matches = self._tables.findItems(
            self._remembered_table, Qt.MatchFlag.MatchExactly
        ) if self._remembered_table else []
        self._tables.setCurrentRow(
            self._tables.row(matches[0]) if matches else 0
        )

    # ------------------------------------------------------------------
    # Remembering the last connection
    # ------------------------------------------------------------------
    def _restore_remembered_connection(self) -> None:
        """Fill the form from the last accepted connection, password aside.

        Nothing is connected to here: this is the form as the user left it,
        not a session resumed. Reaching a server takes a password they still
        have to type, and doing it unasked on the way to a dialog opening
        would hang it on a machine that is off the VPN.
        """
        remembered = get_connect_database_config()
        if not remembered:
            return

        engine = str(remembered.get("engine") or "")
        index = self._engine.findData(engine)
        if index >= 0:
            self._engine.setCurrentIndex(index)  # fires _on_engine_changed

        self._sqlite_path.setText(str(remembered.get("path") or ""))
        self._host.setText(str(remembered.get("host") or "localhost"))
        # After the engine switch, which resets the port to the engine default.
        try:
            port = int(remembered.get("port") or 0)
        except (TypeError, ValueError):
            port = 0
        if port:
            self._port.setValue(port)
        self._database.setCurrentText(str(remembered.get("database") or ""))
        self._username.setText(str(remembered.get("username") or ""))
        self._remembered_table = str(remembered.get("table") or "")

    def _remember_connection(self, conn: DatabaseConnection, table: str) -> None:
        """Store everything but the password, for the next time this opens."""
        payload: dict[str, Any] = {
            "engine": conn.kind,
            "path": conn.path,
            "host": conn.host,
            "port": int(conn.port),
            "database": conn.database,
            "username": conn.username,
            "table": table,
        }
        try:
            set_connect_database_config(payload)
        except Exception as exc:  # noqa: BLE001
            # A settings file that cannot be written is not a reason to lose
            # the connection the user just made.
            applogger.warning(
                "Could not remember the database connection: %s",
                exc,
                show_dialog=False,
                raise_error=False,
            )

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
        self._remember_connection(self.connection, self.table)
        self.accept()
