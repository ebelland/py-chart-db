"""End-to-end: import a large delimited file, chart it, and save the figure.

This is the path a user actually takes - file on disk, ImportRunner, series SQL,
renderer - rather than a hand-built database, so it catches breakage in the
seams between those stages.  Marked as a perf test because writing and reading
several hundred thousand rows is not something to pay for on every run.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from app.charts.render_figure import render_figure_from_descriptor
from app.data.sqlite_repo import SqliteRepo
from app.utils.import_runner import execute_import

ROWS = 250_000
TABLE = "big_import"


def _write_csv(path: Path, rows: int, seed: int = 11) -> None:
    """Write a deterministic CSV with numeric and categorical columns."""
    rng = np.random.default_rng(seed)
    x = np.arange(rows, dtype=float)
    y = 50.0 + 0.001 * x + 5.0 * np.sin(x / 500.0) + rng.normal(0, 1.2, rows)
    category = rng.integers(1, 5, rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.column_stack([x, y, category])
    header = "x,y,category"
    np.savetxt(path, frame, delimiter=",", header=header, comments="", fmt="%.6f")


def _import_settings(csv_path: Path) -> dict:
    """Return the ImportRunner settings dict for the generated CSV."""
    return {
        "source": {"kind": "file", "path": str(csv_path), "sheet": None},
        "read": {
            "skiprows": 0,
            "skip_last": 0,
            "header": True,
            "delimiter": ",",
            "encoding": "utf-8",
        },
        "destination": {
            "table": TABLE,
            "if_exists": "replace",
            "normalize_columns": True,
        },
        "columns": {"types": {"x": "Auto", "y": "Auto", "category": "Auto"}},
    }


def _create_descriptors(repo: SqliteRepo) -> int:
    """Create one scatter figure over the imported table and return its id."""
    figure_id = repo.create_figure_descriptor(name="Imported data", nrows=1, ncols=1)
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="Imported data",
        x_label="x",
        y_label="y",
        options={"title": "Imported data", "grid": True},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="imported",
        sql_query=(
            f'SELECT x AS x, y AS y, category AS color FROM "{TABLE}"'
        ),
        roles={"x": "x", "y": "y", "color": "category"},
        style={"marker": ".", "alpha": 0.35},
    )
    return int(figure_id)


def test_large_file_import_then_render(
    tmp_db_path: Path,
    test_results_dir: Path,
    plots_dir: Path,
    show_plots: bool,
    run_perf: bool,
) -> None:
    if not run_perf:
        pytest.skip("large import test disabled; pass --run-perf")

    csv_path = test_results_dir / "large_import_source.csv"
    _write_csv(csv_path, ROWS)

    repo = SqliteRepo(db_path=tmp_db_path)

    started = time.perf_counter()
    rows, cols = execute_import(repo, settings=_import_settings(csv_path))
    import_seconds = time.perf_counter() - started

    assert rows == ROWS, f"imported {rows} rows, expected {ROWS}"
    assert cols == 3

    # The imported table must be queryable through the normal repository path.
    db_path = repo.ensure_dhub_extension(tmp_db_path)
    with sqlite3.connect(str(db_path)) as con:
        stored = con.execute(f'SELECT COUNT(*) FROM "{TABLE}"').fetchone()[0]
    assert int(stored) == ROWS

    figure_id = _create_descriptors(repo)
    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    assert descriptor is not None

    fig = Figure(figsize=(8.0, 5.0))
    started = time.perf_counter()
    render_figure_from_descriptor(figure=fig, descriptor=descriptor, repo=repo)
    render_seconds = time.perf_counter() - started

    assert fig.axes, "no axis was created for the imported figure"
    assert fig.axes[0].collections or fig.axes[0].lines, "nothing was drawn"

    if show_plots:
        target = plots_dir / "large_import" / "scatter.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=110, bbox_inches="tight")
        (target.parent / "timings.json").write_text(
            json.dumps(
                {
                    "rows": ROWS,
                    "import_seconds": round(import_seconds, 3),
                    "first_render_seconds": round(render_seconds, 3),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    repo.close()
