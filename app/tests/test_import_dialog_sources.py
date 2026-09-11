"""The import dialog's two sources, and the rule that only one is current.

Opening a file and pasting are alternatives, not layers. The dialog kept the
file's path, title and sheet list when data was pasted over it, and then - on
the last line of the paste handler - called the preview refresh, which read
``self.file_name`` and nothing else. So pasting after opening a file put the
file's rows back under the clipboard's table name, and pasting again changed
nothing at all.

These tests are mostly about which source is current after each move, because
that is the state the bug lived in.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.import_data_dialog import CLIPBOARD_SOURCE_NAME, ImportDataDialog


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "sales.csv"
    path.write_text("region,units\nnorth,10\nsouth,20\n", encoding="utf-8")
    return path


@pytest.fixture
def dialog(qapp, tmp_db_path: Path):
    repo = SqliteRepo(db_path=tmp_db_path)
    built = ImportDataDialog(repo)
    yield built
    repo.close()


def _paste(dialog, text: str) -> None:
    QApplication.clipboard().setText(text)
    dialog._on_load_clipboard()


def _columns(dialog) -> list[str]:
    assert dialog._df is not None
    return [str(column) for column in dialog._df.columns]


# ----------------------------------------------------------------------
# Replacing one source with the other
# ----------------------------------------------------------------------
def test_pasting_over_an_opened_file_replaces_it(dialog, csv_file: Path) -> None:
    """The bug, in one test: the file's rows came back after the paste."""
    dialog.load_file(csv_file)
    assert _columns(dialog) == ["region", "units"]

    _paste(dialog, "alpha\tbeta\n1\t2\n")

    assert dialog._source_mode == "clipboard"
    assert _columns(dialog) == ["alpha", "beta"]


def test_pasting_again_replaces_the_previous_paste(dialog) -> None:
    _paste(dialog, "alpha\tbeta\n1\t2\n")
    _paste(dialog, "x\ty\tz\n7\t8\t9\n")

    assert _columns(dialog) == ["x", "y", "z"]
    assert len(dialog._df) == 1


def test_pasting_forgets_the_file_it_replaced(dialog, csv_file: Path) -> None:
    """Not cosmetic: the remembered path is what the import link is written
    from, and the preview used to read it back."""
    dialog.load_file(csv_file)
    _paste(dialog, "alpha\tbeta\n1\t2\n")

    assert dialog.file_name == ""
    assert dialog._path == ""
    assert dialog._table.text() == CLIPBOARD_SOURCE_NAME


def test_a_pasted_import_creates_no_link(dialog) -> None:
    """There is nothing left to reread once the clipboard has moved on, so
    Update Link has nothing to offer a paste and none is created."""
    _paste(dialog, "alpha\tbeta\n1\t2\n")

    dialog._on_accept()

    assert dialog.result is not None
    assert dialog._repo.get_table_link(dialog.result.table_name) is None


def test_opening_a_file_after_pasting_replaces_the_paste(dialog, csv_file: Path) -> None:
    _paste(dialog, "alpha\tbeta\n1\t2\n")
    dialog.load_file(csv_file)

    assert dialog._source_mode == "file"
    assert _columns(dialog) == ["region", "units"]
    assert dialog._table.text() == "sales"


def test_a_refresh_after_pasting_re_reads_the_clipboard_not_a_file(
    dialog, csv_file: Path
) -> None:
    """_refresh_preview is called from the option timer and the sheet combo
    as well as by hand, so it has to be safe to call at any time."""
    dialog.load_file(csv_file)
    _paste(dialog, "alpha\tbeta\n1\t2\n")

    dialog._refresh_preview()

    assert _columns(dialog) == ["alpha", "beta"]


# ----------------------------------------------------------------------
# The options describe how to parse, not where it came from
# ----------------------------------------------------------------------
def test_the_header_option_re_applies_to_pasted_data(dialog) -> None:
    """Pasting used to be a one-shot parse: getting the options wrong meant
    pasting the whole thing again."""
    _paste(dialog, "alpha\tbeta\n1\t2\n")
    assert _columns(dialog) == ["alpha", "beta"]

    dialog._has_header.setChecked(False)
    dialog._refresh_preview()

    assert _columns(dialog) != ["alpha", "beta"]
    assert len(dialog._df) == 2, "the header row is data now"


def test_skip_rows_re_applies_to_pasted_data(dialog) -> None:
    _paste(dialog, "junk\nalpha\tbeta\n1\t2\n3\t4\n")

    dialog._skip_rows.setValue(1)
    dialog._refresh_preview()

    assert _columns(dialog) == ["alpha", "beta"]
    assert len(dialog._df) == 2


def test_an_option_change_schedules_a_refresh_for_pasted_data_too(dialog) -> None:
    """The timer used to return early unless the source was a file."""
    _paste(dialog, "alpha\tbeta\n1\t2\n")

    dialog._schedule_preview()

    assert dialog._preview_timer.isActive()


def test_nothing_is_scheduled_when_there_is_no_source(dialog) -> None:
    dialog._schedule_preview()

    assert not dialog._preview_timer.isActive()


# ----------------------------------------------------------------------
# Refusing a paste leaves what is loaded alone
# ----------------------------------------------------------------------
def test_an_empty_clipboard_does_not_clear_what_is_loaded(
    dialog, csv_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing the dialog and *then* saying why would be the worst of both."""
    import app.dialogs.import_data_dialog as module

    shown: list[str] = []
    monkeypatch.setattr(
        module, "show_message", lambda _p, message_id, **_k: shown.append(message_id)
    )

    dialog.load_file(csv_file)
    _paste(dialog, "   ")

    assert shown == ["import.clipboard_empty"]
    assert dialog._source_mode == "file"
    assert _columns(dialog) == ["region", "units"]


# ----------------------------------------------------------------------
# Another database joins the same source-switching contract
# ----------------------------------------------------------------------
@pytest.fixture
def other_db(tmp_path: Path) -> Path:
    """Another .dhub-shaped file: one real table, and one that looks like
    this application's own bookkeeping - the picker must skip the second."""
    import sqlite3

    path = tmp_path / "other.dhub"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE readings (t REAL, v REAL)")
        conn.execute("INSERT INTO readings VALUES (1.0, 2.0), (2.0, 4.0)")
        conn.execute("CREATE TABLE __figure_descriptors__ (id INTEGER)")
        conn.commit()
    finally:
        conn.close()
    return path


def _pick_database(
    dialog, path: Path, monkeypatch: pytest.MonkeyPatch, *, table: str = "readings"
) -> None:
    """Stand in for the connect dialog, the way LoadDemoDialog.exec is
    stood in for elsewhere: picking a database is its own dialog now, not a
    bare file picker, since PostgreSQL and MySQL need host/user/password
    fields a file dialog has no room for."""
    from app.dialogs.connect_database_dialog import ConnectDatabaseDialog
    from app.utils.data_sources import DatabaseConnection

    def fake_exec(self: ConnectDatabaseDialog) -> bool:
        self.connection = DatabaseConnection(kind="sqlite", path=str(path))
        self.table = table
        return True

    monkeypatch.setattr(ConnectDatabaseDialog, "exec", fake_exec)
    dialog._on_import_database()


def test_database_tables_exclude_this_applications_own_bookkeeping(
    other_db: Path,
) -> None:
    from app.utils.data_sources import list_sqlite_tables

    assert list_sqlite_tables(str(other_db)) == ["readings"]


def test_picking_a_database_loads_its_first_table(
    dialog, other_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pick_database(dialog, other_db, monkeypatch)

    assert dialog._source_mode == "database"
    assert _columns(dialog) == ["t", "v"]
    assert dialog._table.text() == "readings"
    assert dialog._db_connection is not None


def test_opening_a_file_after_a_database_forgets_the_connection(
    dialog, other_db: Path, csv_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pick_database(dialog, other_db, monkeypatch)

    dialog.load_file(csv_file)

    assert dialog._source_mode == "file"
    assert dialog._db_connection is None


def test_pasting_after_a_database_replaces_it(
    dialog, other_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pick_database(dialog, other_db, monkeypatch)

    _paste(dialog, "alpha\tbeta\n1\t2\n")

    assert dialog._source_mode == "clipboard"
    assert dialog._db_connection is None
    assert _columns(dialog) == ["alpha", "beta"]


def test_a_database_import_remembers_its_source(
    dialog, other_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Update link needs the connection back, minus any password - there is
    none here (sqlite), but the shape is what a server database also uses."""
    _pick_database(dialog, other_db, monkeypatch)

    dialog._on_accept()

    assert dialog.result is not None
    link = dialog._repo.get_table_link(dialog.result.table_name)
    assert link is not None
    source = link["settings"]["source"]
    assert source == {"kind": "sqlite", "path": str(other_db), "table": "readings"}


def test_a_server_database_link_never_carries_a_password(
    dialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.dialogs.connect_database_dialog import ConnectDatabaseDialog
    from app.utils.data_sources import DatabaseConnection

    conn = DatabaseConnection(
        kind="postgres",
        host="db.example.com",
        port=5432,
        database="analytics",
        username="reader",
        password="hunter2",
    )

    def fake_exec(self: ConnectDatabaseDialog) -> bool:
        self.connection = conn
        self.table = "orders"
        return True

    monkeypatch.setattr(ConnectDatabaseDialog, "exec", fake_exec)
    monkeypatch.setattr(
        "app.dialogs.import_data_dialog.SERVER_DATABASE_READERS",
        {
            "postgres": (
                lambda _conn: ["orders"],
                lambda _conn, _table, **_kw: __import__("pandas").DataFrame(
                    {"id": [1, 2], "total": [9.5, 4.0]}
                ),
            ),
        },
    )

    dialog._on_import_database()
    dialog._on_accept()

    assert dialog.result is not None
    link = dialog._repo.get_table_link(dialog.result.table_name)
    assert link is not None
    source = link["settings"]["source"]
    assert "password" not in source
    assert source == {
        "kind": "postgres",
        "host": "db.example.com",
        "port": 5432,
        "database": "analytics",
        "username": "reader",
        "table": "orders",
    }


# ----------------------------------------------------------------------
# The web source: fetched into the same preview/import path as a file
# ----------------------------------------------------------------------
class _FakeResponse:
    """A stand-in for ``http.client.HTTPResponse``, just enough of it."""

    def __init__(self, body: bytes, content_type: str) -> None:
        self._body = body
        self._content_type = content_type

    def read(self, size: int | None = None) -> bytes:
        return self._body if size is None else self._body[:size]

    @property
    def headers(self):
        content_type = self._content_type

        class _Headers:
            def get_content_type(self) -> str:
                return content_type

        return _Headers()

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _stub_url(monkeypatch: pytest.MonkeyPatch, body: bytes, content_type: str) -> None:
    # read_web_url lives in app.utils.data_sources, not in the dialog module -
    # the dialog only imports the function, not the urllib module behind it.
    import app.utils.data_sources as module

    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *_a, **_k: _FakeResponse(body, content_type),
    )


def _type_url(dialog, url: str) -> None:
    """Stand in for typing into the web-source row's URL field."""
    dialog._url.setText(url)


def test_web_urls_reject_every_scheme_but_http_and_https() -> None:
    """A "file://" URL here would read local disk through a box that looks
    like it only talks to the network."""
    from app.utils.data_sources import is_valid_web_url

    assert is_valid_web_url("https://example.com/data.csv")
    assert is_valid_web_url("http://example.com/data.csv")
    assert not is_valid_web_url("file:///etc/passwd")
    assert not is_valid_web_url("ftp://example.com/data.csv")
    assert not is_valid_web_url("not a url")
    assert not is_valid_web_url("")


def test_the_extension_falls_back_to_content_type_when_the_url_has_none() -> None:
    from app.utils.data_sources import _extension_for_web_source

    assert (
        _extension_for_web_source("https://api.example.com/export", "application/json")
        == ".json"
    )
    assert _extension_for_web_source("https://example.com/data.csv", None) == ".csv"
    assert _extension_for_web_source("https://api.example.com/export", None) == ".csv"


def test_certifis_bundle_is_used_when_it_is_installed() -> None:
    """The context this app actually fetches with, not just that a context
    of some kind exists - a None here silently falls back to urlopen's own
    default trust store, which is the failure this exists to route around."""
    import certifi

    from app.utils.data_sources import _web_fetch_ssl_context

    _web_fetch_ssl_context.cache_clear()
    context = _web_fetch_ssl_context()
    assert context is not None
    assert context.get_ca_certs(), "certifi's bundle should have loaded some CAs"
    del certifi  # imported only to prove it is actually installed here


def test_a_certificate_verification_failure_gets_a_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CERTIFICATE_VERIFY_FAILED reads exactly like a network problem; the
    dialog's error message should not leave the user guessing which one it
    is - see _web_fetch_ssl_context for why this happens on a fresh
    python.org macOS install with no other network issue at all."""
    import ssl
    import urllib.error

    import app.utils.data_sources as module

    def _raise_wrapped(*_args: object, **_kwargs: object):
        raise urllib.error.URLError(
            ssl.SSLCertVerificationError("certificate verify failed")
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", _raise_wrapped)

    with pytest.raises(ValueError, match="certificate"):
        module.read_web_url("https://example.com/data.csv")


def test_a_bare_ssl_verification_error_gets_the_same_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ssl

    import app.utils.data_sources as module

    def _raise_bare(*_args: object, **_kwargs: object):
        raise ssl.SSLCertVerificationError("certificate verify failed")

    monkeypatch.setattr(module.urllib.request, "urlopen", _raise_bare)

    with pytest.raises(ValueError, match="certificate"):
        module.read_web_url("https://example.com/data.csv")


def test_an_unrelated_url_error_is_not_mistaken_for_a_certificate_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error

    import app.utils.data_sources as module

    def _raise_other(*_args: object, **_kwargs: object):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(module.urllib.request, "urlopen", _raise_other)

    with pytest.raises(urllib.error.URLError, match="Connection refused"):
        module.read_web_url("https://example.com/data.csv")


def test_picking_a_url_fetches_and_parses_it(
    dialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_url(monkeypatch, b"region,units\nnorth,10\nsouth,20\n", "text/csv")
    _type_url(dialog, "https://example.com/sales.csv")

    dialog._on_fetch_url()

    assert dialog._source_mode == "web"
    assert _columns(dialog) == ["region", "units"]
    assert dialog._table.text() == "sales"


def test_a_web_import_remembers_its_url(dialog, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_url(monkeypatch, b"region,units\nnorth,10\n", "text/csv")
    _type_url(dialog, "https://example.com/sales.csv")
    dialog._on_fetch_url()

    dialog._on_accept()

    assert dialog.result is not None
    link = dialog._repo.get_table_link(dialog.result.table_name)
    assert link is not None
    assert link["settings"]["source"] == {
        "kind": "web",
        "url": "https://example.com/sales.csv",
    }


def test_an_empty_url_field_does_nothing(
    dialog, csv_file: Path
) -> None:
    dialog.load_file(csv_file)
    _type_url(dialog, "")

    dialog._on_fetch_url()

    assert dialog._source_mode == "file"
    assert _columns(dialog) == ["region", "units"]


def test_an_invalid_url_is_rejected_without_touching_the_source(
    dialog, csv_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.dialogs.import_data_dialog as module

    shown: list[str] = []
    monkeypatch.setattr(
        module, "show_message", lambda _p, message_id, **_k: shown.append(message_id)
    )

    dialog.load_file(csv_file)
    _type_url(dialog, "not-a-url")

    dialog._on_fetch_url()

    assert shown == ["import.web_invalid_url"]
    assert dialog._source_mode == "file"
    assert _columns(dialog) == ["region", "units"]


def test_picking_a_quick_source_fills_the_url_field(dialog) -> None:
    import app.dialogs.import_data_dialog as module

    source = module.WEB_DATA_SOURCES[0]
    dialog._on_web_source_picked(source)

    assert dialog._url.text() == source.url


def test_opening_a_file_after_a_url_hides_the_table_picker(
    dialog, csv_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_url(monkeypatch, b"region,units\nnorth,10\n", "text/csv")
    _type_url(dialog, "https://example.com/sales.csv")
    dialog._on_fetch_url()

    dialog.load_file(csv_file)

    assert dialog._source_mode == "file"
    assert dialog._path == str(csv_file)
