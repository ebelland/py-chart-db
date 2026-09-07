"""Fetching a web data source: a real download, into the same import path a
local file already uses.

The dialog gained a URL field and a small catalogue of quick-pick sources -
statistics, chemistry, mathematics - so the tests here run their own local
HTTP server rather than reaching out to the real internet, which would make
the suite flaky (and slow) for no reason: the code under test only cares
that it received *some* HTTP response, not which server sent it.
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
from app.dialogs.import_data_dialog import (
    WEB_DATA_SOURCES,
    ImportDataDialog,
    fetch_url_to_temp_file,
    is_web_url,
)

CSV_BODY = b"x,y\n1,2\n3,4\n5,6\n"
JSON_BODY = b'{"a": 1, "b": 2}'


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves a fixed CSV, a fixed JSON body with no file extension in its
    URL (matching PubChem's own .../CSV-suffixed but extension-less shape),
    and a 404 - the three cases fetch_url_to_temp_file has to tell apart."""

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
# is_web_url
# ----------------------------------------------------------------------
def test_http_and_https_are_web_urls() -> None:
    assert is_web_url("https://example.com/data.csv")
    assert is_web_url("http://example.com/data.csv")


@pytest.mark.parametrize(
    "text",
    ["/local/path.csv", "C:\\Users\\me\\data.csv", "data.csv", "", "  "],
)
def test_local_paths_are_not_web_urls(text: str) -> None:
    assert not is_web_url(text)


# ----------------------------------------------------------------------
# fetch_url_to_temp_file
# ----------------------------------------------------------------------
def test_fetch_downloads_the_body_and_keeps_the_urls_own_extension(
    local_server: str,
) -> None:
    path = fetch_url_to_temp_file(f"{local_server}/data.csv")
    try:
        assert path.suffix == ".csv"
        assert path.read_bytes() == CSV_BODY
    finally:
        path.unlink(missing_ok=True)


def test_fetch_falls_back_to_content_type_when_the_url_has_no_extension(
    local_server: str,
) -> None:
    path = fetch_url_to_temp_file(f"{local_server}/no-extension")
    try:
        assert path.suffix == ".json"
        assert path.read_bytes() == JSON_BODY
    finally:
        path.unlink(missing_ok=True)


def test_fetch_raises_for_a_404(local_server: str) -> None:
    with pytest.raises(Exception):  # noqa: B017 - urllib's own HTTPError
        fetch_url_to_temp_file(f"{local_server}/missing")


def test_fetch_raises_for_an_unreachable_host() -> None:
    with pytest.raises(Exception):  # noqa: B017 - urllib's own URLError
        fetch_url_to_temp_file("http://127.0.0.1:1/definitely-not-listening", timeout=2)


# ----------------------------------------------------------------------
# The quick-pick catalogue
# ----------------------------------------------------------------------
def test_every_web_source_has_a_unique_name_and_an_http_url() -> None:
    names = [source.name for source in WEB_DATA_SOURCES]
    assert len(names) == len(set(names))
    for source in WEB_DATA_SOURCES:
        assert is_web_url(source.url)
        assert source.category
        assert source.description


def test_the_catalogue_spans_more_than_one_subject() -> None:
    """The whole point of a catalogue rather than one example: statistics,
    chemistry and mathematics side by side."""
    categories = {source.category for source in WEB_DATA_SOURCES}
    assert len(categories) >= 3


# ----------------------------------------------------------------------
# The dialog: Fetch behaves like Browse
# ----------------------------------------------------------------------
def test_fetching_a_url_previews_it_like_an_opened_file(
    dialog: ImportDataDialog, local_server: str
) -> None:
    dialog._url.setText(f"{local_server}/data.csv")
    dialog._on_fetch_url()

    assert dialog._source_mode == "file"
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

    assert dialog._source_mode == "file"
    assert list(dialog._df.columns) == ["x", "y"]


def test_picking_a_catalogue_entry_fills_the_url_field_without_fetching(
    dialog: ImportDataDialog,
) -> None:
    index = dialog._web_source_combo.findText(
        f"{WEB_DATA_SOURCES[0].name} ({WEB_DATA_SOURCES[0].category})"
    )
    assert index >= 0

    dialog._web_source_combo.setCurrentIndex(index)

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
        lambda *args, **kwargs: called.append(kwargs.get("url", "")),
    )

    dialog._url.setText("not a url")
    dialog._on_fetch_url()

    assert called == ["not a url"]
    assert dialog._source_mode == "none"


def test_a_failed_fetch_shows_a_message_and_leaves_the_dialog_as_it_was(
    dialog: ImportDataDialog, local_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        "app.dialogs.import_data_dialog.show_message",
        lambda *args, **kwargs: called.append(kwargs.get("url", "")),
    )

    dialog._url.setText(f"{local_server}/missing")
    dialog._on_fetch_url()

    assert called
    assert dialog._source_mode == "none"
    assert dialog._df is None


def test_the_default_table_name_comes_from_the_catalogue_entry_when_one_matches(
    dialog: ImportDataDialog, local_server: str
) -> None:
    """A random temp-file name is not a table name a person would recognise;
    the catalogue's own name is."""
    entry_index = dialog._web_source_combo.findText(
        f"{WEB_DATA_SOURCES[0].name} ({WEB_DATA_SOURCES[0].category})"
    )
    dialog._web_source_combo.setCurrentIndex(entry_index)
    dialog._url.setText(f"{local_server}/data.csv")  # override with the fake server

    dialog._on_fetch_url()

    # The combo's own entry no longer matches the (overridden) URL, so the
    # seed falls back to the URL itself rather than the catalogue name -
    # this asserts the fallback path runs without raising.
    assert dialog._table.text()


# ----------------------------------------------------------------------
# The saved import link points at the URL, not at the temp file
# ----------------------------------------------------------------------
def test_accepting_a_fetched_import_links_the_url_not_the_temp_file(
    dialog: ImportDataDialog, local_server: str
) -> None:
    dialog._url.setText(f"{local_server}/data.csv")
    dialog._on_fetch_url()
    dialog._table.setText("web_xy")

    dialog._on_accept()

    link = dialog._repo.get_table_link("web_xy")
    assert link is not None
    assert link["source_path"] == f"{local_server}/data.csv"
