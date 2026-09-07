"""Every way data reaches Data Hub, independent of any dialog.

A source is read the same way whether it is being previewed live in the
import dialog or replayed headlessly by a saved link's "Update link" - so the
reading itself lives here, with no Qt import anywhere in the module, and both
``app.dialogs.import_data_dialog`` (the live picker) and ``app.utils.import_runner``
(the link refresh) call into it rather than each keeping its own copy.

Five kinds of source, one ``kind`` string each, used as the discriminator in
a saved link's ``settings["source"]["kind"]``:

``file``
    A local CSV/TSV/TXT/XLSX/XLSM/XLS/JSON/XML file.
``sqlite``
    One table read out of another SQLite (or ``.dhub``) file.
``web``
    Whatever an http(s) URL returns, parsed the same way a file of the same
    kind would be.
``postgres`` / ``mysql``
    One table read out of a server database. The password is asked for every
    time - see ``DatabaseConnection`` - because a link is a plain JSON blob
    sitting in the project file, and a project file is exactly the kind of
    thing that gets emailed or committed without a second thought.
"""
from __future__ import annotations

import csv
import os
import sqlite3
import tempfile
import urllib.request
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

from app.logs.logger import applogger

# -----------------------------------------------------------------------------
# Plain files
# -----------------------------------------------------------------------------

#: The default table name for pasted data, and the stand-in for a file path
#: wherever one is expected.  Named once: it is both what the table is called
#: and what a caller has to recognise as "not a path".
CLIPBOARD_SOURCE_NAME: str = "from_clipboard"

#: What the readers below can open, and therefore what the file dialog offers
#: and what a drop onto the main window is accepted for.  One list: the dialog
#: filter and the drop test used to be two literals, and a format added to the
#: reader was a format the window still refused.
IMPORTABLE_SUFFIXES: tuple[str, ...] = (
    ".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls", ".json", ".xml",
)

#: The same list as a QFileDialog filter.
IMPORT_FILE_FILTER: str = (
    "Data files ("
    + " ".join(f"*{suffix}" for suffix in IMPORTABLE_SUFFIXES)
    + ");;All files (*.*)"
)


def is_importable(path: str | Path) -> bool:
    """True when this file is one the import dialog can read."""
    return (Path(path).suffix or "").lower() in IMPORTABLE_SUFFIXES


def sniff_delimiter(sample: str) -> str:
    """Best-effort delimiter detection."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;| ")
        return dialect.delimiter
    except Exception:  # noqa: BLE001
        candidates = [",", "\t", ";", "|", " "]
        counts: dict[str, int] = {d: sample.count(d) for d in candidates}
        return max(candidates, key=lambda d: counts[d])


def read_text_file(
    path: str,
    *,
    skiprows: int = 0,
    skipfooter: int = 0,
    header: bool = True,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> pd.DataFrame:
    encodings = [encoding] if encoding else ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    encodings = [e for e in encodings if e]

    with open(path, "rb") as f:
        raw = f.read(64 * 1024)

    sample = ""
    for enc in encodings:
        try:
            sample = raw.decode(enc)
            break
        except Exception:  # noqa: BLE001
            continue

    delim = delimiter or sniff_delimiter(sample)
    hdr = 0 if header else None

    last_exc: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                sep=delim,
                encoding=enc,
                skiprows=int(skiprows),
                skipfooter=int(skipfooter),
                engine="python" if skipfooter else "c",
                header=hdr,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    applogger.error(f"Failed reading text file: {last_exc}")
    return pd.DataFrame()


def read_clipboard_text(
    clipboard_text: str,
    *,
    skiprows: int = 0,
    skipfooter: int = 0,
    header: bool = True,
    delimiter: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    text = (clipboard_text or "").strip("﻿\n\r\t ")
    if not text:
        return None

    sample = text[:4096]
    delim = delimiter or sniff_delimiter(sample)
    hdr = 0 if header else None

    return pd.read_csv(
        StringIO(text),
        sep=delim,
        skiprows=int(skiprows),
        skipfooter=int(skipfooter),
        engine="python" if skipfooter else "c",
        header=hdr,
    )


def read_excel_file(
    path: str,
    *,
    skiprows: int = 0,
    skipfooter: int = 0,
    sheet_name: Optional[str] = None,
    header: bool = True,
) -> pd.DataFrame:
    hdr = 0 if header else None
    df = pd.read_excel(path, sheet_name=sheet_name or 0, skiprows=int(skiprows), header=hdr, engine="openpyxl")
    if skipfooter:
        df = df.iloc[: max(0, len(df) - int(skipfooter))]
    return df


def read_json_file(path: str, *, skiprows: int = 0, skipfooter: int = 0) -> pd.DataFrame:
    try:
        df = pd.read_json(path)
    except ValueError:
        df = pd.read_json(path, lines=True)

    if skiprows:
        df = df.iloc[int(skiprows) :]
    if skipfooter:
        df = df.iloc[: max(0, len(df) - int(skipfooter))]
    return df


def read_xml_file(path: str, *, skiprows: int = 0, skipfooter: int = 0) -> pd.DataFrame:
    df = pd.read_xml(path)
    if skiprows:
        df = df.iloc[int(skiprows) :]
    if skipfooter:
        df = df.iloc[: max(0, len(df) - int(skipfooter))]
    return df


def read_any_file(
    path: str,
    *,
    skiprows: int = 0,
    skipfooter: int = 0,
    header: bool = True,
    sheet: Optional[str] = None,
    delim: Optional[str] = None,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    ext = (Path(path).suffix or "").lower()
    if ext in (".csv", ".tsv", ".txt"):
        return read_text_file(
            path,
            skiprows=skiprows,
            skipfooter=skipfooter,
            header=header,
            encoding=encoding,
            delimiter=delim,
        )
    if ext in (".xlsx", ".xlsm", ".xls"):
        return read_excel_file(path, skiprows=skiprows, skipfooter=skipfooter, sheet_name=sheet, header=header)
    if ext in (".json",):
        return read_json_file(path, skiprows=skiprows, skipfooter=skipfooter)
    if ext in (".xml",):
        return read_xml_file(path, skiprows=skiprows, skipfooter=skipfooter)

    # fallback: try delimited text
    return read_text_file(
        path,
        skiprows=skiprows,
        skipfooter=skipfooter,
        header=header,
        encoding=encoding,
        delimiter=delim,
    )


# -----------------------------------------------------------------------------
# Another SQLite database
# -----------------------------------------------------------------------------

#: The file dialog filter for picking another SQLite database to import from.
DATABASE_FILE_FILTER: str = (
    "SQLite database (*.dhub *.db *.sqlite *.sqlite3);;All files (*.*)"
)


def list_sqlite_tables(path: str) -> list[str]:
    """Return the user tables in *path*, for the source-table picker.

    A plain connect, not a read-only URI one: everything run against it here
    is a SELECT, and a URI's escaping rules for spaces and backslashes are
    one more way for a path to fail on Windows for no reason a user would
    understand.
    """
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' "
            # Excludes this application's own bookkeeping tables when the
            # other database is itself a .dhub - see SqliteRepo.list_user_tables,
            # which filters its own list the same way for the same reason.
            "AND name NOT LIKE '__%__' ESCAPE '_' "
            "ORDER BY name"
        ).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        conn.close()


def read_sqlite_table(
    path: str,
    table: str,
    *,
    skiprows: int = 0,
    skipfooter: int = 0,
) -> pd.DataFrame:
    """Read one table out of another SQLite database."""
    conn = sqlite3.connect(str(path))
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    finally:
        conn.close()

    if skiprows:
        df = df.iloc[int(skiprows) :]
    if skipfooter:
        df = df.iloc[: max(0, len(df) - int(skipfooter))]
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# A server database: PostgreSQL or MySQL
# -----------------------------------------------------------------------------

#: Default port per engine, offered as a starting point in the connection
#: dialog and used whenever a saved link's settings do not carry one (an
#: older link, or one hand-edited without it).
DEFAULT_PORTS: dict[str, int] = {"postgres": 5432, "mysql": 3306}


@dataclass(slots=True)
class DatabaseConnection:
    """Everything needed to open one server database, in memory only.

    ``password`` is deliberately never written to :meth:`to_link_settings`:
    a saved link sits in the project file's own JSON, in plain sight of
    anyone who opens the file, copies it, or commits it - which a password
    is not safe to assume none of them do. "Update link" therefore has to
    ask again every time; see ``app.widgets.table_list._refresh_link_for_table``.
    """

    kind: str  # "sqlite" | "postgres" | "mysql"
    path: str = ""  # sqlite only
    host: str = ""
    port: int = 0
    database: str = ""
    username: str = ""
    password: str = ""

    def to_link_settings(self) -> dict[str, object]:
        """Return the connection, minus the password, for a saved link."""
        if self.kind == "sqlite":
            return {"kind": "sqlite", "path": self.path}
        return {
            "kind": self.kind,
            "host": self.host,
            "port": int(self.port),
            "database": self.database,
            "username": self.username,
        }

    def display_name(self) -> str:
        """A one-line label for this connection, e.g. as a window title."""
        if self.kind == "sqlite":
            return self.path
        return f"{self.kind}://{self.host}:{self.port}/{self.database}"

    @classmethod
    def from_link_settings(cls, settings: dict[str, object], *, password: str = "") -> "DatabaseConnection":
        """Rebuild a connection from a saved link, plus a freshly-typed password."""
        kind = str(settings.get("kind") or "sqlite")
        if kind == "sqlite":
            return cls(kind="sqlite", path=str(settings.get("path") or ""))
        return cls(
            kind=kind,
            host=str(settings.get("host") or ""),
            port=int(settings.get("port") or DEFAULT_PORTS.get(kind, 0)),
            database=str(settings.get("database") or ""),
            username=str(settings.get("username") or ""),
            password=password,
        )


def list_postgres_tables(conn: DatabaseConnection) -> list[str]:
    """Return the base tables in *conn*'s ``public`` schema."""
    import pg8000.dbapi

    connection = pg8000.dbapi.connect(
        host=conn.host,
        port=int(conn.port),
        database=conn.database,
        user=conn.username,
        password=conn.password,
    )
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
        return [str(row[0]) for row in cursor.fetchall()]
    finally:
        connection.close()


def read_postgres_table(
    conn: DatabaseConnection,
    table: str,
    *,
    skiprows: int = 0,
    skipfooter: int = 0,
) -> pd.DataFrame:
    """Read one table out of a PostgreSQL database."""
    import pg8000.dbapi

    connection = pg8000.dbapi.connect(
        host=conn.host,
        port=int(conn.port),
        database=conn.database,
        user=conn.username,
        password=conn.password,
    )
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
    finally:
        connection.close()

    if skiprows:
        df = df.iloc[int(skiprows) :]
    if skipfooter:
        df = df.iloc[: max(0, len(df) - int(skipfooter))]
    return df.reset_index(drop=True)


def list_mysql_tables(conn: DatabaseConnection) -> list[str]:
    """Return the base tables in *conn*'s database."""
    import pymysql

    connection = pymysql.connect(
        host=conn.host,
        port=int(conn.port),
        database=conn.database,
        user=conn.username,
        password=conn.password,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW FULL TABLES WHERE Table_type = 'BASE TABLE'")
            return [str(row[0]) for row in cursor.fetchall()]
    finally:
        connection.close()


def read_mysql_table(
    conn: DatabaseConnection,
    table: str,
    *,
    skiprows: int = 0,
    skipfooter: int = 0,
) -> pd.DataFrame:
    """Read one table out of a MySQL database."""
    import pymysql

    connection = pymysql.connect(
        host=conn.host,
        port=int(conn.port),
        database=conn.database,
        user=conn.username,
        password=conn.password,
    )
    try:
        df = pd.read_sql_query(f"SELECT * FROM `{table}`", connection)
    finally:
        connection.close()

    if skiprows:
        df = df.iloc[int(skiprows) :]
    if skipfooter:
        df = df.iloc[: max(0, len(df) - int(skipfooter))]
    return df.reset_index(drop=True)


#: kind -> (list_tables(conn), read_table(conn, table, **kwargs)), one shape
#: for every engine so a caller need not name sqlite/postgres/mysql one at a
#: time - read_sqlite_table alone keeps the plain-path signature that
#: list_sqlite_tables and its own direct callers already use, so it is
#: wrapped here rather than changed.
SERVER_DATABASE_READERS: dict[str, tuple] = {
    "sqlite": (
        lambda conn: list_sqlite_tables(conn.path),
        lambda conn, table, **kwargs: read_sqlite_table(conn.path, table, **kwargs),
    ),
    "postgres": (list_postgres_tables, read_postgres_table),
    "mysql": (list_mysql_tables, read_mysql_table),
}


# -----------------------------------------------------------------------------
# The web
# -----------------------------------------------------------------------------

#: Schemes this module will fetch. Deliberately not "file" - a "file://" URL
#: typed or pasted into a box that looks like it only talks to the network
#: would otherwise read local disk through it.
_ALLOWED_WEB_SCHEMES: tuple[str, ...] = ("http", "https")

#: Bytes read from a URL before this gives up. This is a dataset importer,
#: not a general-purpose downloader: refusing a multi-gigabyte reply is
#: safer than filling the machine's memory with one.
WEB_FETCH_MAX_BYTES: int = 200 * 1024 * 1024

#: Content-Type -> the extension read_any_file dispatches on, for a response
#: whose URL has no recognisable suffix of its own (an API endpoint, a
#: redirect, a query string with no path).
_CONTENT_TYPE_EXTENSIONS: dict[str, str] = {
    "text/csv": ".csv",
    "text/tab-separated-values": ".tsv",
    "application/json": ".json",
    "text/json": ".json",
    "application/xml": ".xml",
    "text/xml": ".xml",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
}


def is_valid_web_url(url: str) -> bool:
    """True when *url* is an http(s) URL this module is willing to fetch."""
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return False
    return parsed.scheme in _ALLOWED_WEB_SCHEMES and bool(parsed.netloc)


def _extension_for_web_source(url: str, content_type: Optional[str]) -> str:
    """Return the extension read_any_file should dispatch on for *url*."""
    ext = (Path(urlparse(url).path).suffix or "").lower()
    if ext in IMPORTABLE_SUFFIXES:
        return ext
    base = (content_type or "").split(";")[0].strip().lower()
    return _CONTENT_TYPE_EXTENSIONS.get(base, ".csv")


def filename_from_url(url: str) -> str:
    """Return a display/table-name base for *url*, query string stripped."""
    name = Path(urlparse(url).path).name.strip()
    return name or "web_data"


def read_web_url(
    url: str,
    *,
    skiprows: int = 0,
    skipfooter: int = 0,
    header: bool = True,
    sheet: Optional[str] = None,
    delim: Optional[str] = None,
    encoding: Optional[str] = None,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Download *url* and parse it exactly as a local file of the same kind.

    Downloaded to a temporary file rather than parsed from the response in
    memory so that every format - CSV, Excel, JSON, XML - goes through the
    same ``read_any_file`` a local source would, instead of a second reading
    path that could drift from the first.
    """
    if not is_valid_web_url(url):
        raise ValueError(f"Not an http(s) URL: {url}")

    request = urllib.request.Request(url, headers={"User-Agent": "Data Hub/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        content_type = response.headers.get_content_type()
        data = response.read(WEB_FETCH_MAX_BYTES + 1)

    if len(data) > WEB_FETCH_MAX_BYTES:
        raise ValueError(
            f"Response exceeds the {WEB_FETCH_MAX_BYTES // (1024 * 1024)} MB "
            "limit for a web import."
        )

    ext = _extension_for_web_source(url, content_type)
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(data)
        return read_any_file(
            tmp_path,
            skiprows=skiprows,
            skipfooter=skipfooter,
            header=header,
            sheet=sheet,
            delim=delim,
            encoding=encoding,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Reading a saved link's source, by kind - what "Update link" calls
# -----------------------------------------------------------------------------


class MissingPasswordError(ValueError):
    """Raised when refreshing a server-database link with no password given.

    Never a bug in the link itself: a PostgreSQL or MySQL link never stores
    one (see ``DatabaseConnection``), so every refresh needs a caller that
    can ask for it - this is that caller's cue to.
    """

    def __init__(self, kind: str) -> None:
        super().__init__(f"A password is required to connect to this {kind} database.")
        self.kind = kind


def read_from_link_source(
    source: dict[str, object],
    read: dict[str, object],
    *,
    password: Optional[str] = None,
) -> pd.DataFrame:
    """Read the data a saved link's ``source``/``read`` settings describe.

    The one place that turns ``source["kind"]`` into an actual read, so a
    link is refreshed exactly the way its source would be previewed live -
    ``app.dialogs.import_data_dialog`` and ``app.utils.import_runner`` both
    end up here rather than each guessing at the other's settings shape.
    """
    kind = str(source.get("kind") or "file")
    skiprows = int(read.get("skiprows", 0) or 0)  # type: ignore[arg-type]
    skip_last = int(read.get("skip_last", 0) or 0)  # type: ignore[arg-type]
    header = bool(read.get("header", True))
    encoding = read.get("encoding") or None
    delimiter = read.get("delimiter") or None

    if kind == "file":
        path = str(source.get("path") or "")
        if not path:
            raise ValueError("Missing source path")
        return read_any_file(
            path,
            skiprows=skiprows,
            skipfooter=skip_last,
            header=header,
            sheet=source.get("sheet"),  # type: ignore[arg-type]
            delim=delimiter,  # type: ignore[arg-type]
            encoding=encoding,  # type: ignore[arg-type]
        )

    if kind == "web":
        url = str(source.get("url") or "")
        if not url:
            raise ValueError("Missing source URL")
        return read_web_url(
            url,
            skiprows=skiprows,
            skipfooter=skip_last,
            header=header,
            sheet=source.get("sheet"),  # type: ignore[arg-type]
            delim=delimiter,  # type: ignore[arg-type]
            encoding=encoding,  # type: ignore[arg-type]
        )

    if kind in SERVER_DATABASE_READERS:
        table = str(source.get("table") or "")
        if not table:
            raise ValueError("Missing source table")

        if kind == "sqlite":
            conn = DatabaseConnection(kind="sqlite", path=str(source.get("path") or ""))
        else:
            if not password:
                raise MissingPasswordError(kind)
            conn = DatabaseConnection(
                kind=kind,
                host=str(source.get("host") or ""),
                port=int(source.get("port") or DEFAULT_PORTS.get(kind, 0)),  # type: ignore[arg-type]
                database=str(source.get("database") or ""),
                username=str(source.get("username") or ""),
                password=password,
            )

        _list_tables, read_table = SERVER_DATABASE_READERS[kind]
        return read_table(conn, table, skiprows=skiprows, skipfooter=skip_last)

    raise ValueError(f"Unknown source kind: {kind!r}")
