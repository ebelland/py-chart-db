"""Execute file imports and refresh saved import links.

``execute_import`` is the single path from a file on disk into a database table:
read, apply per-column type overrides and ignores, then hand the frame to the
repository.  ``refresh_link`` re-runs a previously saved import, always
replacing the destination table so a refresh cannot silently append duplicates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pandas._typing import DtypeArg

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger


@dataclass(slots=True)
class LinkRefreshResult:
    link_id: int
    table_name: str
    rows: int
    cols: int


def read_any_file(
    path: str,
    *,
    sheet: str | None,
    skiprows: int,
    skip_last: int,
    header: bool,
    encoding: str | None,
    delimiter: str | None,
) -> pd.DataFrame:
    """Read a file into a DataFrame based on extension."""
    p = Path(path)
    suffix = p.suffix.lower()
    header_row = 0 if header else None

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(
            str(p),
            sheet_name=sheet if sheet else 0,
            skiprows=skiprows,
            header=header_row,
            engine=("openpyxl" if suffix in {".xlsx", ".xlsm"} else None),
        )
    else:
        df = pd.read_csv(
            str(p),
            sep=delimiter or ",",
            skiprows=skiprows,
            header=header_row,
            encoding=None if encoding is None or encoding== 'auto' else encoding,
            engine="c",
        )

    if not header:
        df.columns = [f"col_{i + 1}" for i in range(len(df.columns))]

    if skip_last > 0:
        df = df.iloc[:-skip_last] if len(df) > skip_last else df.iloc[0:0]

    return df


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
) -> tuple[int, int]:
    """Execute an import from a settings dict. Returns (rows, cols)."""
    source = cast(dict[str, Any], settings.get("source", {}))
    read = cast(dict[str, Any], settings.get("read", {}))
    dest = cast(dict[str, Any], settings.get("destination", {}))
    columns = cast(dict[str, Any], settings.get("columns", {}))

    path = cast(str | None, source.get("path"))
    if not path:
        applogger.error("Missing source path")
        return 0, 0

    df = read_any_file(
        path,
        sheet=cast(str | None, source.get("sheet")),
        skiprows=int(read.get("skiprows", 0)),
        skip_last=int(read.get("skip_last", 0)),
        header=bool(read.get("header", True)),
        encoding=cast(str | None, read.get("encoding")),
        delimiter=cast(str | None, read.get("delimiter")),
    )

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


def refresh_link(repo: SqliteRepo, *, link_id: int) -> LinkRefreshResult:
    """Refresh a link by re-importing its stored settings.

    SqliteRepo.get_import_link() returns a dict with keys:
      - table_name: destination table name
      - source_path: file path
      - settings: JSON dict saved from ImportDataDialog

    ImportDataDialog stores a *flat* config dict (table/header/skip_rows/skip_last/delim/encoding/sheet).
    execute_import() expects a structured dict. This function maps the flat config accordingly.
    """
    link = repo.get_import_link(int(link_id))

    table_name = cast(str, link.get("table_name"))
    source_path = cast(str, link.get("source_path"))
    cfg = cast(dict[str, Any], link.get("settings", {}))

    if not source_path:
        applogger.error("Link has no source path")
    if not table_name:
        applogger.error("Link has no table name")

    settings: dict[str, Any] = {
        "source": {
            "path": source_path,
            "sheet": cast(str | None, cfg.get("sheet")) or None,
        },
        "read": {
            "skiprows": int(cfg.get("skip_rows", 0) or 0),
            "skip_last": int(cfg.get("skip_last", 0) or 0),
            "header": bool(cfg.get("header", True)),
            "encoding": cast(str | None, cfg.get("encoding")) or None,
            "delimiter": cast(str | None, cfg.get("delim")) or None,
        },
        "destination": {
            "table": table_name,
            "normalize_columns": True,
        },
        "columns": cast(dict[str, Any], cfg.get("columns", {}))
        or {"types": cast(dict[str, str], cfg.get("types", {}))},
    }

    rows, cols = execute_import(repo, settings=settings, link_id=int(link_id))

    return LinkRefreshResult(
        link_id=int(link_id),
        table_name=table_name,
        rows=int(rows),
        cols=int(cols),
    )
