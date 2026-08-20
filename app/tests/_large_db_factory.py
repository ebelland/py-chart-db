"""Build large synthetic databases for volume and performance tests.

The descriptor schema here must stay identical to
``SqliteRepo._create_system_tables``; when they drift, the tests silently stop
exercising the real code path.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import numpy as np


def ensure_dhub_extension(path: Path) -> Path:
    """Ensure the DB file uses the .dhub extension."""
    p = path.expanduser().resolve()
    return p if p.suffix.lower() == ".dhub" else p.with_suffix(".dhub")


def _executemany(
    cur: sqlite3.Cursor,
    sql: str,
    rows: Iterable[Tuple[object, ...]],
    *,
    chunk_size: int = 50_000,
) -> None:
    """Execute many inserts in chunks to avoid large Python lists."""
    buf: list[Tuple[object, ...]] = []
    for row in rows:
        buf.append(row)
        if len(buf) >= chunk_size:
            cur.executemany(sql, buf)
            buf.clear()
    if buf:
        cur.executemany(sql, buf)


def _create_dataset_table(cur: sqlite3.Cursor, table_name: str) -> None:
    """Create a dataset table with a stable schema used by tests."""
    cur.execute(f'DROP TABLE IF EXISTS "{table_name}";')
    cur.execute(
        f"""
        CREATE TABLE "{table_name}" (
            x        REAL NOT NULL,
            y        REAL NOT NULL,
            color    REAL,
            size     REAL,
            category TEXT
        );
        """
    )


def create_large_db(
    db_path: Path,
    *,
    n_ts: int = 300_000,
    n_scatter: int = 200_000,
    seed: int = 1234,
) -> None:
    """Create and populate a LARGE sqlite DB for rendering tests.

    Tables created:
      - __import_links__ (minimal)
      - __figure_descriptors__ / __axis_descriptors__ / __series_descriptors__
      - dataset tables: ds_<key> for each series

    The descriptor schema mirrors ``SqliteRepo._create_system_tables`` exactly,
    including the single ``roles`` JSON column on series.

    Notes:
      - This factory is deterministic given the seed.
      - No backward compatibility: we DROP and recreate all descriptor tables.
    """
    db_path = ensure_dhub_extension(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rng = np.random.default_rng(seed)

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()

        # Speed up bulk inserts (OK for test DBs)
        cur.execute("PRAGMA journal_mode = OFF;")
        cur.execute("PRAGMA synchronous = OFF;")
        cur.execute("PRAGMA temp_store = MEMORY;")
        cur.execute("PRAGMA cache_size = -200000;")

        # --------------------------
        # Schema
        # --------------------------
        cur.executescript(
            """
            DROP TABLE IF EXISTS __import_links__;
            DROP TABLE IF EXISTS __figure_descriptors__;
            DROP TABLE IF EXISTS __axis_descriptors__;
            DROP TABLE IF EXISTS __series_descriptors__;

            CREATE TABLE __import_links__ (
                id            INTEGER PRIMARY KEY,
                table_name    TEXT NOT NULL UNIQUE,
                source_path   TEXT NOT NULL,
                settings_json TEXT NOT NULL
            );

            CREATE TABLE __figure_descriptors__ (
                id          INTEGER PRIMARY KEY,
                name        TEXT NOT NULL,
                nrows       INTEGER NOT NULL,
                ncols       INTEGER NOT NULL,
                options_json TEXT
            );

            CREATE TABLE __axis_descriptors__ (
                id            INTEGER PRIMARY KEY,
                figure_id     INTEGER NOT NULL,
                axis_index    INTEGER NOT NULL,
                chart_type    TEXT NOT NULL,
                title         TEXT,
                x_label       TEXT,
                y_label       TEXT,
                z_label       TEXT,
                options_json  TEXT
            );

            CREATE TABLE __series_descriptors__ (
                id            INTEGER PRIMARY KEY,
                axis_id       INTEGER NOT NULL,
                series_index  INTEGER NOT NULL,
                name          TEXT,
                sql_query     TEXT NOT NULL,
                roles         TEXT,
                style_json    TEXT
            );
            """
        )

        # --------------------------
        # Figures
        # --------------------------
        cur.executemany(
            """
            INSERT INTO __figure_descriptors__
                (id, name, nrows, ncols, options_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (1, "LargeDB Figure 1", 1, 2,  "{}"),
                (2, "LargeDB Figure 2", 1, 2,  "{}"),
            ],
        )

        # --------------------------
        # Axes
        # --------------------------
        cur.executemany(
            """
            INSERT INTO __axis_descriptors__
                (id, figure_id, axis_index, chart_type, title, x_label, y_label, z_label, options_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (11, 1, 0, "Time Series", "TS (Fig1)", "t", "value", "", "{}"),
                (12, 1, 1, "Scatter Plot", "SC (Fig1)", "x", "y", "", "{}"),
                (21, 2, 0, "Time Series", "TS (Fig2)", "t", "value", "", "{}"),
                (22, 2, 1, "Scatter Plot", "SC (Fig2)", "x", "y", "", "{}"),
            ],
        )

        # --------------------------
        # Dataset tables
        # --------------------------
        series_keys = [
            "f1_ts_a",
            "f1_ts_b",
            "f1_sc_a",
            "f1_sc_b",
            "f2_ts_a",
            "f2_ts_b",
            "f2_sc_a",
            "f2_sc_b",
        ]
        for key in series_keys:
            _create_dataset_table(cur, f"ds_{key}")

        # --------------------------
        # Series descriptors
        # --------------------------
        # Renderers resolve roles from DataFrame column names, so every query
        # aliases its output columns to the lowercase role names the renderers
        # expect ("x", "y", "color", "size").
        xy_roles = '{"x": "x", "y": "y"}'
        xycs_roles = '{"x": "x", "y": "y", "color": "color", "size": "size"}'

        series_rows = [
            (1101, 11, 0, "F1 TS A", "SELECT x AS x, y AS y FROM ds_f1_ts_a ORDER BY x", xy_roles, "{}"),
            (1102, 11, 1, "F1 TS B", "SELECT x AS x, y AS y FROM ds_f1_ts_b ORDER BY x", xy_roles, "{}"),
            (1201, 12, 0, "F1 SC A", "SELECT x AS x, y AS y FROM ds_f1_sc_a", xy_roles, "{}"),
            (1202, 12, 1, "F1 SC B", "SELECT x AS x, y AS y, color AS color, size AS size FROM ds_f1_sc_b", xycs_roles, "{}"),
            (2101, 21, 0, "F2 TS A", "SELECT x AS x, y AS y FROM ds_f2_ts_a ORDER BY x", xy_roles, "{}"),
            (2102, 21, 1, "F2 TS B", "SELECT x AS x, y AS y FROM ds_f2_ts_b ORDER BY x", xy_roles, "{}"),
            (2201, 22, 0, "F2 SC A", "SELECT x AS x, y AS y FROM ds_f2_sc_a", xy_roles, "{}"),
            (2202, 22, 1, "F2 SC B", "SELECT x AS x, y AS y FROM ds_f2_sc_b", xy_roles, "{}"),
        ]
        cur.executemany(
            """
            INSERT INTO __series_descriptors__
                (id, axis_id, series_index, name, sql_query, roles, style_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            series_rows,
        )

        # --------------------------
        # Data generation helpers
        # --------------------------
        x_ts = np.arange(n_ts, dtype=float)

        def rows_xy(x: np.ndarray, y: np.ndarray) -> Iterator[Tuple[object, ...]]:
            for xv, yv in zip(x, y):
                yield (float(xv), float(yv), None, None, None)

        def rows_scatter_xy(sc: np.ndarray) -> Iterator[Tuple[object, ...]]:
            for xv, yv in sc:
                yield (float(xv), float(yv), None, None, None)

        def rows_scatter_xycs(sc: np.ndarray, color: np.ndarray, size: np.ndarray) -> Iterator[Tuple[object, ...]]:
            for (xv, yv), col, sz in zip(sc, color, size):
                yield (float(xv), float(yv), float(col), float(sz), None)

        # --------------------------
        # Time-series data
        # --------------------------
        y_f1_a = 10 + 0.02 * x_ts + rng.normal(0, 0.8, n_ts)
        y_f1_b = 20 + 5 * np.sin(2 * np.pi * x_ts / 60) + rng.normal(0, 1.5, n_ts)
        y_f2_a = 5 + 0.01 * x_ts + rng.normal(0, 0.6, n_ts)
        y_f2_b = 50 + 2 * np.cos(2 * np.pi * x_ts / 120) + rng.normal(0, 1.0, n_ts)

        # --------------------------
        # Scatter data
        # --------------------------
        sc1 = rng.multivariate_normal(mean=[10, 10], cov=[[4, 1.5], [1.5, 3]], size=n_scatter)
        sc2 = rng.multivariate_normal(mean=[6, 12], cov=[[5, -1.2], [-1.2, 4]], size=n_scatter)
        c2 = rng.normal(0.5, 0.12, n_scatter).clip(0.0, 1.0)
        s2 = rng.normal(40, 12, n_scatter).clip(3.0, 200.0)

        x_g1 = rng.gamma(shape=2.0, scale=4.0, size=n_scatter)
        y_g1 = rng.gamma(shape=1.2, scale=6.0, size=n_scatter)
        x_g2 = rng.gamma(shape=2.8, scale=3.5, size=n_scatter)
        y_g2 = rng.gamma(shape=1.8, scale=4.5, size=n_scatter)

        # --------------------------
        # Inserts
        # --------------------------
        _executemany(cur, "INSERT INTO ds_f1_ts_a(x, y, color, size, category) VALUES (?, ?, ?, ?, ?)", rows_xy(x_ts, y_f1_a))
        _executemany(cur, "INSERT INTO ds_f1_ts_b(x, y, color, size, category) VALUES (?, ?, ?, ?, ?)", rows_xy(x_ts, y_f1_b))
        _executemany(cur, "INSERT INTO ds_f2_ts_a(x, y, color, size, category) VALUES (?, ?, ?, ?, ?)", rows_xy(x_ts, y_f2_a))
        _executemany(cur, "INSERT INTO ds_f2_ts_b(x, y, color, size, category) VALUES (?, ?, ?, ?, ?)", rows_xy(x_ts, y_f2_b))

        _executemany(cur, "INSERT INTO ds_f1_sc_a(x, y, color, size, category) VALUES (?, ?, ?, ?, ?)", rows_scatter_xy(sc1))
        _executemany(cur, "INSERT INTO ds_f1_sc_b(x, y, color, size, category) VALUES (?, ?, ?, ?, ?)", rows_scatter_xycs(sc2, c2, s2))
        _executemany(cur, "INSERT INTO ds_f2_sc_a(x, y, color, size, category) VALUES (?, ?, ?, ?, ?)", rows_xy(x_g1, y_g1))
        _executemany(cur, "INSERT INTO ds_f2_sc_b(x, y, color, size, category) VALUES (?, ?, ?, ?, ?)", rows_xy(x_g2, y_g2))

        conn.commit()
