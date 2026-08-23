"""Build databases that exercise every shipped axis renderer.

``_large_db_factory`` covers volume; this module covers *breadth*: one figure per
renderer, with the role columns that renderer actually reads, so a rendering
regression in any chart type shows up as a failed test and a saved PNG rather
than as an empty axis nobody looks at.

Role columns are lowercase for scatter/time series, capitalised for bar
(``X`` / ``Y`` / ``YError``) and named ``value`` / ``group`` for box plots -
that asymmetry is in the renderers, not here.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

# figure id -> (figure name, chart type, series specs)
# Each series spec is (series name, sql, style dict).
SHOWCASE_CHART_TYPES: tuple[str, ...] = (
    "Scatter Plot",
    "Time Series",
    "Bar Chart",
    "Horizontal Bar Chart",
    "Box Plot",
    "Histogram",
    "Violin Plot",
    "ECDF",
    "Pie Chart",
    "Pareto Chart",
    "Stack Plot",
    "Stairs",
    "Fill Between",
    "Wind Barbs",
    "Timeline",
    "Broken Bar",
    "Broken Bar (Vertical)",
    "Table",
    "Text",
    "Surface Plot",
    "Surface Plot (Scattered)",
)


def _create_source_tables(cur: sqlite3.Cursor, rng: np.random.Generator, n: int) -> None:
    """Create and populate the source tables the showcase figures query."""
    cur.executescript(
        """
        DROP TABLE IF EXISTS src_xy;
        DROP TABLE IF EXISTS src_series;
        DROP TABLE IF EXISTS src_categories;
        DROP TABLE IF EXISTS src_samples;
        DROP TABLE IF EXISTS src_field;
        DROP TABLE IF EXISTS src_events;

        CREATE TABLE src_xy       (x REAL, y REAL, color REAL, size REAL);
        CREATE TABLE src_series   (t REAL, v REAL, w REAL);
        CREATE TABLE src_categories (label TEXT, amount REAL, err REAL);
        CREATE TABLE src_samples  (grp TEXT, measurement REAL);
        -- A vector field on a grid, for the barb renderer: x/y are the
        -- positions and u/v the components at each of them.
        CREATE TABLE src_field    (x REAL, y REAL, u REAL, v REAL);
        -- Dated events, for the timeline: an ISO date and the text to write
        -- at the top of each stem.
        CREATE TABLE src_events   (happened TEXT, title TEXT);
        -- Interval rows for the broken-bar renderers: one row per interval,
        -- several rows may share a category (the "broken" part).
        CREATE TABLE src_intervals (task TEXT, start REAL, duration REAL);
        -- Labelled points for the text renderer.
        CREATE TABLE src_labels  (x REAL, y REAL, label TEXT);
        -- A regular x/y grid, for the gridded surface renderer.
        CREATE TABLE src_grid    (x REAL, y REAL, z REAL);
        -- The same surface, sampled at scattered (non-gridded) points, for
        -- the triangulated surface renderer.
        CREATE TABLE src_scatter3d (x REAL, y REAL, z REAL);
        """
    )

    # Scatter: two visually distinct clouds with continuous colour and size.
    cloud = rng.multivariate_normal([10, 10], [[4, 1.5], [1.5, 3]], size=n)
    colors = rng.normal(0.5, 0.15, n).clip(0.0, 1.0)
    sizes = rng.normal(28, 9, n).clip(4.0, 160.0)
    cur.executemany(
        "INSERT INTO src_xy (x, y, color, size) VALUES (?, ?, ?, ?)",
        [
            (float(px), float(py), float(c), float(s))
            for (px, py), c, s in zip(cloud, colors, sizes)
        ],
    )

    # Time series: a trend plus a seasonal component, both noisy.
    t = np.arange(n, dtype=float)
    v = 20.0 + 0.01 * t + 4.0 * np.sin(2 * np.pi * t / 90.0) + rng.normal(0, 0.7, n)
    w = 12.0 + 2.0 * np.cos(2 * np.pi * t / 45.0) + rng.normal(0, 0.5, n)
    cur.executemany(
        "INSERT INTO src_series (t, v, w) VALUES (?, ?, ?)",
        [(float(a), float(b), float(c)) for a, b, c in zip(t, v, w)],
    )

    # Bars: a small categorical table with error bars.
    labels = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    amounts = rng.uniform(5.0, 40.0, len(labels))
    errors = rng.uniform(0.5, 3.0, len(labels))
    cur.executemany(
        "INSERT INTO src_categories (label, amount, err) VALUES (?, ?, ?)",
        [
            (label, float(amount), float(err))
            for label, amount, err in zip(labels, amounts, errors)
        ],
    )

    # Box plot: four groups with deliberately different spreads.
    rows: list[tuple[str, float]] = []
    for index, name in enumerate(("g1", "g2", "g3", "g4")):
        values = rng.normal(10.0 + 2.0 * index, 1.0 + 0.6 * index, max(50, n // 8))
        rows.extend((name, float(value)) for value in values)
    cur.executemany(
        "INSERT INTO src_samples (grp, measurement) VALUES (?, ?)", rows
    )

    cur.executemany(
        "INSERT INTO src_events (happened, title) VALUES (?, ?)",
        [
            ("2026-01-14", "v1.0"), ("2026-02-02", "v1.1"), ("2026-03-19", "v1.2"),
            ("2026-04-07", "v2.0 beta"), ("2026-05-30", "v2.0"), ("2026-06-11", "v2.1"),
            ("2026-08-01", "v2.2"), ("2026-09-22", "v3.0 rc"), ("2026-11-05", "v3.0"),
        ],
    )

    # Broken bar: three tasks, one of them ("Design") busy in two separate
    # stretches - the case broken_barh exists for.
    cur.executemany(
        "INSERT INTO src_intervals (task, start, duration) VALUES (?, ?, ?)",
        [
            ("Design", 0.0, 3.0), ("Design", 5.0, 2.0),
            ("Build", 2.0, 4.0),
            ("Test", 6.0, 3.0),
        ],
    )

    # Text: a handful of labelled points, two deliberately close together so
    # the showcase is meaningful with adjustText installed and still correct
    # (just overlapping) without it.
    cur.executemany(
        "INSERT INTO src_labels (x, y, label) VALUES (?, ?, ?)",
        [
            (0.0, 0.0, "origin"), (3.0, 4.0, "north-east"),
            (3.1, 4.1, "north-east 2"), (-2.0, 1.0, "west"),
        ],
    )

    # Surface: a ripple, sampled on a regular grid and again at scattered
    # points - the same shape, the two input layouts the surface renderers
    # each need.
    grid_axis = np.linspace(-3.0, 3.0, 20)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    grid_z = np.sin(np.hypot(grid_x, grid_y))
    cur.executemany(
        "INSERT INTO src_grid (x, y, z) VALUES (?, ?, ?)",
        [
            (float(px), float(py), float(pz))
            for px, py, pz in zip(grid_x.ravel(), grid_y.ravel(), grid_z.ravel())
        ],
    )

    scatter_x = rng.uniform(-3.0, 3.0, 400)
    scatter_y = rng.uniform(-3.0, 3.0, 400)
    scatter_z = np.sin(np.hypot(scatter_x, scatter_y))
    cur.executemany(
        "INSERT INTO src_scatter3d (x, y, z) VALUES (?, ?, ?)",
        [
            (float(px), float(py), float(pz))
            for px, py, pz in zip(scatter_x, scatter_y, scatter_z)
        ],
    )

    # A smooth rotating field, so the barbs form a legible pattern rather than
    # noise: an unreadable showcase proves the renderer ran, not that it works.
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 6.0, 10), np.linspace(0.0, 6.0, 10))
    cur.executemany(
        "INSERT INTO src_field (x, y, u, v) VALUES (?, ?, ?, ?)",
        [
            (float(px), float(py), float(10.0 * np.sin(py)), float(10.0 * np.cos(px)))
            for px, py, in zip(grid_x.ravel(), grid_y.ravel())
        ],
    )


def _showcase_definitions() -> list[dict[str, Any]]:
    """Return the figure/axis/series definition of every showcase figure."""
    return [
        {
            "chart_type": "Scatter Plot",
            "name": "Scatter showcase",
            "labels": ("x", "y"),
            "axis_options": {"title": "Scatter Plot", "grid": True},
            "series": [
                (
                    "cloud",
                    "SELECT x AS x, y AS y, color AS color, size AS size FROM src_xy",
                    {"marker": "o", "alpha": 0.5},
                ),
            ],
        },
        {
            "chart_type": "Time Series",
            "name": "Time series showcase",
            "labels": ("t", "value"),
            "axis_options": {"title": "Time Series", "grid": True},
            "series": [
                ("trend", "SELECT t AS x, v AS y FROM src_series ORDER BY t", {"linestyle": "-"}),
                ("second", "SELECT t AS x, w AS y FROM src_series ORDER BY t", {"linestyle": "--"}),
            ],
        },
        {
            "chart_type": "Stack Plot",
            "name": "Stack showcase",
            "labels": ("t", "value"),
            "axis_options": {"title": "Stack Plot"},
            "series": [
                ("v", "SELECT t AS x, v AS y FROM src_series LIMIT 200", {}),
                ("w", "SELECT t AS x, w AS y FROM src_series LIMIT 200", {}),
            ],
        },
        {
            "chart_type": "Stairs",
            "name": "Stairs showcase",
            "labels": ("bin", "count"),
            "axis_options": {"title": "Stairs", "fill": True, "alpha": 0.6},
            "series": [
                ("counts", "SELECT t AS x, v AS y FROM src_series LIMIT 40", {}),
            ],
        },
        {
            "chart_type": "Fill Between",
            "name": "Fill showcase",
            "labels": ("t", "value"),
            "axis_options": {"title": "Fill Between", "alpha": 0.35},
            "series": [
                (
                    "band",
                    "SELECT t AS x, v + 1.5 AS y, v - 1.5 AS y2 FROM src_series LIMIT 200",
                    {},
                ),
            ],
        },
        {
            "chart_type": "Wind Barbs",
            "name": "Barbs showcase",
            "labels": ("x", "y"),
            "axis_options": {"title": "Wind Barbs"},
            "series": [
                ("field", "SELECT x, y, u, v FROM src_field", {}),
            ],
        },
        {
            "chart_type": "Timeline",
            "name": "Timeline showcase",
            "labels": ("date", ""),
            "axis_options": {"title": "Timeline"},
            "series": [
                (
                    "releases",
                    "SELECT happened AS x, title AS label FROM src_events",
                    {"marker": "o"},
                ),
            ],
        },
        {
            "chart_type": "Pareto Chart",
            "name": "Pareto showcase",
            "labels": ("category", "amount"),
            "axis_options": {"title": "Pareto Chart", "tick_label_rotation": "45"},
            "series": [
                (
                    "amounts",
                    "SELECT label AS X, amount AS Y FROM src_categories",
                    {},
                ),
            ],
        },
        {
            "chart_type": "Bar Chart",
            "name": "Bar showcase",
            "labels": ("category", "amount"),
            "axis_options": {"title": "Bar Chart"},
            "series": [
                (
                    "amounts",
                    "SELECT label AS X, amount AS Y, err AS YError FROM src_categories",
                    {},
                ),
            ],
        },
        {
            "chart_type": "Horizontal Bar Chart",
            "name": "Horizontal bar showcase",
            "labels": ("amount", "category"),
            "axis_options": {"title": "Horizontal Bar Chart"},
            "series": [
                (
                    "amounts",
                    "SELECT label AS X, amount AS Y, err AS XError FROM src_categories",
                    {},
                ),
            ],
        },
        {
            "chart_type": "Box Plot",
            "name": "Box plot showcase",
            "labels": ("group", "measurement"),
            "axis_options": {"title": "Box Plot"},
            "series": [
                (
                    "samples",
                    "SELECT grp AS \"group\", measurement AS value FROM src_samples",
                    {},
                ),
            ],
        },
        {
            "chart_type": "Violin Plot",
            "name": "Violin showcase",
            "labels": ("group", "measurement"),
            "axis_options": {
                "title": "Violin Plot",
                "showmedians": True,
                "quantiles": "0.25, 0.75",
            },
            "series": [
                (
                    "samples",
                    "SELECT grp AS \"group\", measurement AS value FROM src_samples",
                    {},
                ),
            ],
        },
        {
            # Two samples on one axis: the comparison an ECDF is for.
            "chart_type": "ECDF",
            "name": "ECDF showcase",
            "labels": ("measurement", ""),
            "axis_options": {"title": "ECDF", "grid": True, "alpha": 0.9},
            "series": [
                (
                    "g1",
                    "SELECT measurement AS value FROM src_samples WHERE grp = 'g1'",
                    {"marker": ".", "color": "#1f77b4"},
                ),
                (
                    "g4",
                    "SELECT measurement AS value FROM src_samples WHERE grp = 'g4'",
                    {"marker": ".", "color": "#d62728"},
                ),
            ],
        },
        {
            # A pie draws one series; the categories table is the natural one.
            "chart_type": "Pie Chart",
            "name": "Pie showcase",
            "labels": ("", ""),
            "axis_options": {
                "title": "Pie Chart",
                "wedge_width": 0.45,
                "autopct": "%1.0f%%",
                "colormap": "tab20",
            },
            "series": [
                (
                    "categories",
                    "SELECT label AS label, amount AS value FROM src_categories",
                    {},
                ),
            ],
        },
        {
            # One series split into four data sets by its "dataset" column,
            # which is the multi-histogram case the renderer exists for.
            "chart_type": "Histogram",
            "name": "Histogram showcase",
            "labels": ("", "measurement"),
            "axis_options": {
                "title": "Histogram",
                "bins": 25,
                "histtype": "bar",
                "alpha": 0.8,
            },
            "series": [
                (
                    "samples",
                    "SELECT grp AS dataset, measurement AS value FROM src_samples",
                    {},
                ),
            ],
        },
        {
            "chart_type": "Broken Bar",
            "name": "Broken bar showcase",
            "labels": ("", ""),
            "axis_options": {"title": "Broken Bar"},
            "series": [
                (
                    "tasks",
                    "SELECT task AS category, start, duration FROM src_intervals",
                    {},
                ),
            ],
        },
        {
            "chart_type": "Broken Bar (Vertical)",
            "name": "Broken bar vertical showcase",
            "labels": ("", ""),
            "axis_options": {"title": "Broken Bar (Vertical)"},
            "series": [
                (
                    "tasks",
                    "SELECT task AS category, start, duration FROM src_intervals",
                    {},
                ),
            ],
        },
        {
            "chart_type": "Table",
            "name": "Table showcase",
            "labels": ("", ""),
            # A table has no x/y scale to show a frame or ticks for; the
            # renderer's own ax.axis("off") is undone again by
            # _apply_axis_runtime_options's unconditional set_axis_on(),
            # which runs after every renderer - hide_axis is the one thing
            # that survives it.
            "axis_options": {"title": "Table", "hide_axis": True},
            "series": [
                (
                    "amounts",
                    "SELECT label AS \"label\", amount, err FROM src_categories",
                    {},
                ),
            ],
        },
        {
            "chart_type": "Text",
            "name": "Text showcase",
            "labels": ("", ""),
            "axis_options": {"title": "Text"},
            "series": [
                (
                    "labels",
                    "SELECT x, y, label AS text FROM src_labels",
                    {},
                ),
            ],
        },
        {
            "chart_type": "Surface Plot",
            "name": "Surface showcase",
            "labels": ("", ""),
            "axis_options": {"title": "Surface Plot", "projection": "3d"},
            "series": [
                ("ripple", "SELECT x, y, z FROM src_grid", {}),
            ],
        },
        {
            "chart_type": "Surface Plot (Scattered)",
            "name": "Scattered surface showcase",
            "labels": ("", ""),
            "axis_options": {"title": "Surface Plot (Scattered)", "projection": "3d"},
            "series": [
                ("ripple", "SELECT x, y, z FROM src_scatter3d", {}),
            ],
        },
    ]


def create_renderer_showcase_db(
    db_path: Path,
    *,
    n_points: int = 5_000,
    seed: int = 7,
    figure_options: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Create a database with one figure per renderer.

    Returns a mapping of chart type -> figure id so tests can address a figure
    by renderer name instead of by magic number.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rng = np.random.default_rng(seed)
    figure_ids: dict[str, int] = {}

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode = OFF;")
        cur.execute("PRAGMA synchronous = OFF;")

        cur.executescript(
            """
            DROP TABLE IF EXISTS __figure_descriptors__;
            DROP TABLE IF EXISTS __axis_descriptors__;
            DROP TABLE IF EXISTS __series_descriptors__;

            CREATE TABLE __figure_descriptors__ (
                id           INTEGER PRIMARY KEY,
                name         TEXT NOT NULL,
                nrows        INTEGER NOT NULL,
                ncols        INTEGER NOT NULL,
                options_json TEXT
            );

            CREATE TABLE __axis_descriptors__ (
                id           INTEGER PRIMARY KEY,
                figure_id    INTEGER NOT NULL,
                axis_index   INTEGER NOT NULL,
                chart_type   TEXT NOT NULL,
                title        TEXT,
                x_label      TEXT,
                y_label      TEXT,
                z_label      TEXT,
                options_json TEXT
            );

            CREATE TABLE __series_descriptors__ (
                id           INTEGER PRIMARY KEY,
                axis_id      INTEGER NOT NULL,
                series_index INTEGER NOT NULL,
                name         TEXT,
                sql_query    TEXT NOT NULL,
                roles        TEXT,
                style_json   TEXT
            );
            """
        )

        _create_source_tables(cur, rng, n_points)

        options_json = json.dumps(figure_options or {})
        series_id = 1

        for index, spec in enumerate(_showcase_definitions(), start=1):
            figure_id = index
            axis_id = 100 + index
            figure_ids[str(spec["chart_type"])] = figure_id

            cur.execute(
                "INSERT INTO __figure_descriptors__ (id, name, nrows, ncols, options_json) "
                "VALUES (?, ?, 1, 1, ?)",
                (figure_id, spec["name"], options_json),
            )
            x_label, y_label = spec["labels"]
            cur.execute(
                "INSERT INTO __axis_descriptors__ "
                "(id, figure_id, axis_index, chart_type, title, x_label, y_label, z_label, options_json) "
                "VALUES (?, ?, 0, ?, ?, ?, ?, '', ?)",
                (
                    axis_id,
                    figure_id,
                    spec["chart_type"],
                    spec["name"],
                    x_label,
                    y_label,
                    json.dumps(spec["axis_options"]),
                ),
            )

            for series_index, (name, sql, style) in enumerate(spec["series"]):
                cur.execute(
                    "INSERT INTO __series_descriptors__ "
                    "(id, axis_id, series_index, name, sql_query, roles, style_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        series_id,
                        axis_id,
                        series_index,
                        name,
                        sql,
                        "{}",
                        json.dumps(style),
                    ),
                )
                series_id += 1

        conn.commit()

    return figure_ids
