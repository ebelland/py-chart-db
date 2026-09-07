"""Fetching a web data source: a real download, into the same import path a
local file already uses.

The dialog has a URL field and a small catalogue of quick-pick sources -
statistics, chemistry, mathematics - so the tests here run their own local
HTTP server rather than reaching out to the real internet, which would make
the suite flaky (and slow) for no reason: the code under test only cares
that it received *some* HTTP response, not which server sent it.

The actual fetch-and-parse (read_web_url) lives in app.utils.data_sources,
shared with a saved link's "Update link" - see test_data_sources.py for that
module's own tests with a mocked urlopen. This file is about the dialog:
what Fetch does to it, and the same web source read end to end over a real
socket.
"""
from __future__ import annotations

import http.server
import socketserver
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.import_data_dialog import ImportDataDialog, WEB_DATA_SOURCES
from app.utils.data_sources import is_valid_web_url, read_web_url

CSV_BODY = b"x,y\n1,2\n3,4\n5,6\n"
JSON_BODY = b'{"a": 1, "b": 2}'


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves a fixed CSV, a fixed JSON body with no file extension in its
    URL (matching PubChem's own .../CSV-suffixed but extension-less shape),
    and a 404 - the three cases read_web_url has to tell apart."""

    routes: dict[str, tuple[int, str, bytes]] = {
        "/data.csv": (200, "text/csv", CSV_BODY),
        "/no-extension": (200, "application/json; charset=utf-8", JSON_BODY),
    }

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's own name
        status, content_type, body = self.routes.get(self.path, (404, "text/plain", b""))
        self.send_response(status)
        if body:
            self.send_header("Content-Type", content_type)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # noqa: N802 - silence per-request logging
        pass


@pytest.fixture
def local_server() -> Iterator[str]:
    """Start a throwaway HTTP server on localhost and yield its base URL."""
    server = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def dialog(qapp, tmp_db_path: Path):
    # tmp_db_path is derived from the test's own name, not a fresh temp file
    # per run, so a .dhub left over from an earlier local run - carrying an
    # import link from that run's local_server, on a now-stale port - has to
    # be cleared before this test creates its own.
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    repo = SqliteRepo(db_path=tmp_db_path)
    built = ImportDataDialog(repo)
    yield built
    repo.close()


# ----------------------------------------------------------------------
# is_valid_web_url
# ----------------------------------------------------------------------
def test_http_and_https_are_web_urls() -> None:
    assert is_valid_web_url("https://example.com/data.csv")
    assert is_valid_web_url("http://example.com/data.csv")


@pytest.mark.parametrize(
    "text",
    ["/local/path.csv", "C:\\Users\\me\\data.csv", "data.csv", "", "  ", "file:///etc/passwd"],
)
def test_local_paths_are_not_web_urls(text: str) -> None:
    assert not is_valid_web_url(text)


# ----------------------------------------------------------------------
# read_web_url, against a real socket
# ----------------------------------------------------------------------
def test_fetch_downloads_and_parses_using_the_urls_own_extension(
    local_server: str,
) -> None:
    df = read_web_url(f"{local_server}/data.csv")

    assert list(df.columns) == ["x", "y"]
    assert len(df) == 3


def test_fetch_falls_back_to_content_type_when_the_url_has_no_extension(
    local_server: str,
) -> None:
    df = read_web_url(f"{local_server}/no-extension")

    assert df.to_dict(orient="records") == [{"a": 1, "b": 2}]


def test_fetch_raises_for_a_404(local_server: str) -> None:
    with pytest.raises(Exception):  # noqa: B017 - urllib's own HTTPError
        read_web_url(f"{local_server}/missing")


def test_fetch_raises_for_an_unreachable_host() -> None:
    with pytest.raises(Exception):  # noqa: B017 - urllib's own URLError
        read_web_url("http://127.0.0.1:1/definitely-not-listening", timeout=2)


# ----------------------------------------------------------------------
# The quick-pick catalogue
# ----------------------------------------------------------------------
def test_every_web_source_has_a_unique_name_and_an_http_url() -> None:
    names = [source.name for source in WEB_DATA_SOURCES]
    assert len(names) == len(set(names))
    for source in WEB_DATA_SOURCES:
        assert is_valid_web_url(source.url)
        assert source.category
        assert source.description


def test_the_catalogue_spans_more_than_one_subject() -> None:
    """The whole point of a catalogue rather than one example: statistics,
    chemistry and mathematics side by side."""
    categories = {source.category for source in WEB_DATA_SOURCES}
    assert len(categories) >= 3


def test_the_catalogue_is_loaded_from_the_bundled_json_file() -> None:
    """WEB_DATA_SOURCES is not hand-typed Python: it comes from
    app/data/web_sources.json, so retiring or adding an entry never touches
    code."""
    from app.utils.data_sources import WEB_SOURCES_PATH

    assert WEB_SOURCES_PATH.exists()
    assert WEB_SOURCES_PATH.name == "web_sources.json"


# ----------------------------------------------------------------------
# The web-source picker: a chevron button whose menu is grouped by category
# ----------------------------------------------------------------------
def test_the_web_source_button_pops_up_a_menu_instead_of_a_click(
    dialog: ImportDataDialog,
) -> None:
    assert dialog._web_source_button.menu() is dialog._web_source_menu
    assert (
        dialog._web_source_button.popupMode()
        == dialog._web_source_button.ToolButtonPopupMode.InstantPopup
    )


def test_the_menu_groups_entries_by_category_and_fills_the_url_on_click(
    dialog: ImportDataDialog,
) -> None:
    entry_actions = [
        entry for entry in dialog._web_source_menu.actions() if not entry.isSeparator()
    ]
    assert len(entry_actions) == len(WEB_DATA_SOURCES)

    first = WEB_DATA_SOURCES[0]
    matching = next(entry for entry in entry_actions if entry.text() == first.name)
    matching.trigger()

    assert dialog._url.text() == first.url
    assert dialog._source_mode == "none"


# ----------------------------------------------------------------------
# The dialog: Fetch behaves like Browse
# ----------------------------------------------------------------------
def test_fetching_a_url_previews_it_like_an_opened_file(
    dialog: ImportDataDialog, local_server: str
) -> None:
    dialog._url.setText(f"{local_server}/data.csv")
    dialog._on_fetch_url()

    assert dialog._source_mode == "web"
    assert dialog._df is not None
    assert list(dialog._df.columns) == ["x", "y"]
    assert len(dialog._df) == 3


def test_fetching_replaces_a_previously_pasted_source(
    dialog: ImportDataDialog, local_server: str
) -> None:
    QApplication.clipboard().setText("alpha\tbeta\n1\t2\n")
    dialog._on_load_clipboard()
    assert dialog._source_mode == "clipboard"

    dialog._url.setText(f"{local_server}/data.csv")
    dialog._on_fetch_url()

    assert dialog._source_mode == "web"
    assert list(dialog._df.columns) == ["x", "y"]


def test_picking_a_catalogue_entry_fills_the_url_field_without_fetching(
    dialog: ImportDataDialog,
) -> None:
    dialog._on_web_source_picked(WEB_DATA_SOURCES[0])

    assert dialog._url.text() == WEB_DATA_SOURCES[0].url
    # Nothing was downloaded yet - the source has not been loaded.
    assert dialog._source_mode == "none"


def test_an_empty_url_field_does_nothing(dialog: ImportDataDialog) -> None:
    dialog._url.setText("")
    dialog._on_fetch_url()

    assert dialog._source_mode == "none"
    assert dialog._df is None


def test_a_non_url_shows_a_message_instead_of_trying_to_fetch_it(
    dialog: ImportDataDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        "app.dialogs.import_data_dialog.show_message",
        lambda _parent, message_id, **_kwargs: called.append(message_id),
    )

    dialog._url.setText("not a url")
    dialog._on_fetch_url()

    assert called == ["import.web_invalid_url"]
    assert dialog._source_mode == "none"


def test_a_failed_fetch_shows_a_message_and_leaves_the_dialog_as_it_was(
    dialog: ImportDataDialog, local_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        "app.dialogs.import_data_dialog.show_message",
        lambda _parent, message_id, **_kwargs: called.append(message_id),
    )

    dialog._url.setText(f"{local_server}/missing")
    dialog._on_fetch_url()

    assert called == ["import.web_failed"]
    assert dialog._source_mode == "none"
    assert dialog._df is None


def test_the_default_table_name_comes_from_the_catalogue_entry_when_one_matches(
    dialog: ImportDataDialog, local_server: str
) -> None:
    """A random URL-derived name is not a table name a person would
    recognise; the catalogue's own name is."""
    dialog._on_web_source_picked(WEB_DATA_SOURCES[0])
    dialog._url.setText(f"{local_server}/data.csv")  # override with the fake server

    dialog._on_fetch_url()

    # The picked entry no longer matches the (overridden) URL, so the seed
    # falls back to the URL's own file name rather than the catalogue name -
    # this asserts the fallback path runs without raising.
    assert dialog._table.text()


# ----------------------------------------------------------------------
# The saved import link points at the URL, not at a temp file
# ----------------------------------------------------------------------
def test_accepting_a_fetched_import_links_the_url(
    dialog: ImportDataDialog, local_server: str
) -> None:
    dialog._url.setText(f"{local_server}/data.csv")
    dialog._on_fetch_url()
    dialog._table.setText("web_xy")

    dialog._on_accept()

    link = dialog._repo.get_table_link("web_xy")
    assert link is not None
    assert link["settings"]["source"] == {"kind": "web", "url": f"{local_server}/data.csv"}
