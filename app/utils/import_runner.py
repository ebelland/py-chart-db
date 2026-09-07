"""Execute imports and refresh saved import links.

``execute_import`` is the single path from any source - a file, another
database, a URL - into a database table: read, apply per-column type
overrides and ignores, then hand the frame to the repository. The reading
itself is ``app.utils.data_sources.read_from_link_source``, dispatching on
``settings["source"]["kind"]``, so a link is refreshed exactly the way its
source would be previewed live in the import dialog. ``refresh_link``
re-runs a previously saved import, always replacing the destination table so
a refresh cannot silently append duplicates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from pandas._typing import DtypeArg

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils.data_sources import read_from_link_source


@dataclass(slots=True)
class LinkRefreshResult:
    link_id: int
    table_name: str
    rows: int
    cols: int


def _convert_datetime(series: pd.Series, kind: str) -> pd.Series:
    """
    Convert a Series to DATE/TIME/DATETIME formatted TEXT (string dtype).

    Pylance/stubs fixes:
    - Normalize to pandas 'string' dtype before to_datetime (avoids Series[Any] overload issues).
    - Force the inferred result to be a Series so .dt is always valid (not a DatetimeIndex).
    """
    s = series.astype("string")
    dt = cast(pd.Series, pd.to_datetime(s, errors="coerce"))

    if kind == "DATE":
        return dt.dt.strftime("%Y-%m-%d").astype("string")
    if kind == "TIME":
        return dt.dt.strftime("%H:%M:%S").astype("string")
    return dt.dt.strftime("%Y-%m-%d %H:%M:%S").astype("string")


def execute_import(
    repo: SqliteRepo,
    *,
    settings: dict[str, Any],
    link_id: int | None = None,  # kept for compatibility with older callers
    password: str | None = None,
) -> tuple[int, int]:
    """Execute an import from a settings dict. Returns (rows, cols).

    ``password`` is only ever needed for a PostgreSQL or MySQL source - see
    ``app.utils.data_sources.DatabaseConnection`` for why it is never part of
    ``settings`` itself, and raises ``MissingPasswordError`` when the source
    needs one and none was given, for the caller to catch and ask for.
    """
    source = cast(dict[str, Any], settings.get("source", {}))
    read = cast(dict[str, Any], settings.get("read", {}))
    dest = cast(dict[str, Any], settings.get("destination", {}))
    columns = cast(dict[str, Any], settings.get("columns", {}))

    df = read_from_link_source(source, read, password=password)

    col_types = cast(dict[str, str], columns.get("types", {}))

    # Apply Ignore + DATE/TIME/DATETIME conversions
    ignore_cols = [c for c, t in col_types.items() if t == "Ignore"]
    if ignore_cols:
        df = df.drop(columns=[c for c in ignore_cols if c in df.columns])

    dtype_overrides: dict[str, str] = {}
    for col, t in col_types.items():
        if col not in df.columns:
            continue
        if t in {"DATE", "TIME", "DATETIME"}:
            df[col] = _convert_datetime(df[col], t)
            dtype_overrides[col] = "TEXT"
        elif t not in {"Auto", "Ignore"}:
            dtype_overrides[col] = t

    table = cast(str, dest.get("table"))
    if not table:
        applogger.error("Missing destination table")
        return 0, 0

    normalize_cols = bool(dest.get("normalize_columns", True))

    rows = repo.import_dataframe(
        df,
        table_name=table,
        normalize_columns=normalize_cols,
        dtype_overrides=cast(DtypeArg, dtype_overrides) if dtype_overrides else None,
    )

    # link_id is intentionally ignored (no last_run/status columns)
    return int(rows), int(df.shape[1])


def refresh_link(
    repo: SqliteRepo, *, link_id: int, password: str | None = None
) -> LinkRefreshResult:
    """Refresh a link by re-importing its stored settings.

    SqliteRepo.get_import_link() returns a dict with keys:
      - table_name: destination table name
      - source_path: a display string only - see app.data.sqlite_repo.upsert_link
      - settings: the structured JSON dict saved from ImportDataDialog

    A link saved before ``settings["source"]["kind"]`` existed carries the
    *flat* config dict ImportDataDialog used to save instead (table/header/
    skip_rows/skip_last/delim/encoding/sheet), with the file path in its own
    ``source_path`` column rather than inside ``settings`` - adapted to the
    structured shape here, once, rather than asking every older project's
    links to be rewritten.
    """
    link = repo.get_import_link(int(link_id))

    table_name = cast(str, link.get("table_name"))
    raw_settings = cast(dict[str, Any], link.get("settings", {}))

    if not table_name:
        applogger.error("Link has no table name")

    if "source" in raw_settings:
        settings: dict[str, Any] = dict(raw_settings)
        destination = dict(cast(dict[str, Any], settings.get("destination") or {}))
        destination.setdefault("table", table_name)
        destination.setdefault("normalize_columns", True)
        settings["destination"] = destination
    else:
        source_path = cast(str, link.get("source_path") or "")
        if not source_path:
            applogger.error("Link has no source path")
        settings = {
            "source": {
                "kind": "file",
                "path": source_path,
                "sheet": cast(str | None, raw_settings.get("sheet")) or None,
            },
            "read": {
                "skiprows": int(raw_settings.get("skip_rows", 0) or 0),
                "skip_last": int(raw_settings.get("skip_last", 0) or 0),
                "header": bool(raw_settings.get("header", True)),
                "encoding": cast(str | None, raw_settings.get("encoding")) or None,
                "delimiter": cast(str | None, raw_settings.get("delim")) or None,
            },
            "destination": {
                "table": table_name,
                "normalize_columns": True,
            },
            "columns": {"types": cast(dict[str, str], raw_settings.get("types", {}))},
        }

    rows, cols = execute_import(
        repo, settings=settings, link_id=int(link_id), password=password
    )

    return LinkRefreshResult(
        link_id=int(link_id),
        table_name=table_name,
        rows=int(rows),
        cols=int(cols),
    )
