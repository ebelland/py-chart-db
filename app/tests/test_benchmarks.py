"""A small benchmark: the operations that decide whether the app feels quick.

Not a profiler and not a competition with one.  Each test asserts a budget
generous enough that ordinary machine-to-machine variation never trips it,
while a change that makes something an order of magnitude slower does.  The
point is to catch the accidental quadratic, the cache that stopped caching, the
per-row database round trip - not to measure milliseconds.

Timings are printed, so ``pytest -s app/tests/test_benchmarks.py`` doubles as a
report even when everything passes.

Budgets assume a developer machine and are deliberately loose.  If one fails,
read the printed number first: a small overshoot on a slow or loaded machine is
noise, while a 10x overshoot is the thing this file exists to find.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _timed(label: str, work, repeats: int = 1) -> float:
    """Run *work* and report the best of *repeats* seconds elapsed.

    Best-of rather than mean: the interesting quantity is how fast the code can
    go, and on a shared machine the slow runs measure the neighbours.
    """
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        work()
        best = min(best, time.perf_counter() - started)

    print(f"\n  {label}: {best * 1000:.1f} ms")
    return best


@pytest.fixture
def repo(tmp_path):
    from app.data.sqlite_repo import SqliteRepo

    repo = SqliteRepo(db_path=tmp_path / "benchmark.dhub")
    yield repo
    repo.close()


def _frame(rows: int = 50_000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "x": x,
            "y": np.sin(x / 500.0) + rng.normal(scale=0.1, size=rows),
            "label": rng.choice(["a", "b", "c"], size=rows),
        }
    )


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def test_importing_fifty_thousand_rows(repo) -> None:
    """Import is one bulk insert; a per-row loop would show up here at once."""
    frame = _frame()

    elapsed = _timed(
        "import 50k rows",
        lambda: repo.import_dataframe(frame, table_name="bench", normalize_columns=False),
    )

    assert elapsed < 5.0


def test_reading_them_back(repo) -> None:
    repo.import_dataframe(_frame(), table_name="bench", normalize_columns=False)

    elapsed = _timed(
        "read 50k rows",
        lambda: repo.query_df("SELECT * FROM bench"),
        repeats=3,
    )

    assert elapsed < 2.0


def test_listing_data_sources_is_not_linear_in_row_count(repo) -> None:
    """It reads the catalogue, so 50k rows must cost the same as 50.

    A version that counted rows per table made the table list slower with every
    import, which is the sort of thing nobody notices until a project is large.
    """
    repo.import_dataframe(_frame(50), table_name="small", normalize_columns=False)
    small = _timed("list sources (50 rows)", repo.list_data_sources, repeats=5)

    repo.import_dataframe(_frame(50_000), table_name="large", normalize_columns=False)
    large = _timed("list sources (+50k rows)", repo.list_data_sources, repeats=5)

    assert large < max(small * 8.0, 0.25)


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def test_rendering_a_scatter_of_fifty_thousand_points(repo) -> None:
    """The whole path: descriptor, query, renderer, Agg."""
    from matplotlib.figure import Figure

    from app.charts.render_figure import render_figure_from_descriptor

    repo.import_dataframe(_frame(), table_name="bench", normalize_columns=False)
    figure_id = repo.create_figure_descriptor(name="bench")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="",
        x_label="x",
        y_label="y",
        options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="bench",
        sql_query="SELECT x, y FROM bench",
        roles={"x": "x", "y": "y"},
        style={"marker": ".", "linestyle": ""},
    )
    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    figure = Figure(figsize=(8, 6), dpi=100)

    def render() -> None:
        figure.clear()
        render_figure_from_descriptor(figure=figure, descriptor=descriptor, repo=repo)
        figure.canvas.draw()

    elapsed = _timed("render 50k-point scatter", render, repeats=2)

    assert elapsed < 15.0


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def test_the_wavelet_transform_on_a_long_record() -> None:
    """One FFT per scale: 64 scales over 16k samples should stay interactive.

    The naive implementation convolves in the time domain and is minutes, not
    milliseconds - this budget is what keeps it in the Fourier domain.
    """
    from app.series_operations.spectral_dialog import SeriesSpectralDialog

    fs = 200.0
    t = np.arange(16_384) / fs
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 35 * t)
    method = SeriesSpectralDialog.__dict__["_wavelet_power"]

    elapsed = _timed(
        "wavelet, 16k samples x 64 scales",
        lambda: method(object(), signal, fs, 6.0, 64),
        repeats=2,
    )

    assert elapsed < 5.0


def test_the_laplace_transform_is_one_fft() -> None:
    from app.series_operations.spectral_dialog import SeriesSpectralDialog

    signal = np.sin(np.arange(1_048_576) / 100.0)
    method = SeriesSpectralDialog.__dict__["_laplace_spectrum"]

    elapsed = _timed(
        "laplace, 1M samples",
        lambda: method(object(), signal, 200.0, 0.5),
        repeats=2,
    )

    assert elapsed < 3.0


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
def test_the_action_catalogue_is_cached(qapp) -> None:
    """Every menu and button asks for one while the window is being built.

    Re-reading config.json per lookup is invisible in a unit test and obvious
    when a menu with forty items takes a moment to open.
    """
    from app.styles import style

    style.reload_actions()
    cold = _timed("first action lookup", lambda: style.action("copy"))
    warm = _timed("10k cached lookups", lambda: [style.action("copy") for _ in range(10_000)])

    assert warm / 10_000 < cold
    assert warm < 1.0


def test_icons_are_cached_by_size_and_ratio(qapp) -> None:
    """load_icon runs for every button on every menu, repeatedly."""
    from app.styles import style

    # Import pyobjc before the clock starts.  The first SF Symbol pulls in
    # AppKit, which costs most of a second, and whichever test touches one
    # first pays it - this file, since it sorts ahead of the icon tests.  It
    # was invisible while pyobjc was merely declared and not installed, because
    # then the import failed immediately; once installed it swamped the
    # measurement.  Loading a module is not what this budget is about.
    style._sf_symbol_bridge()

    style._FLUENT_ICON_CACHE.clear()
    style._SF_SYMBOL_ICON_CACHE.clear()

    elapsed = _timed(
        "1k load_icon calls",
        lambda: [style.load_icon("copy") for _ in range(1_000)],
    )

    # This benchmark found a real one: 338 ms per thousand, because every call
    # stat-ed the icon directories and built a fresh QIcon.  Caching the lookup
    # and the icon took it to under 5 ms.  The budget is set to catch a return
    # to the old behaviour, not to police the new number.
    assert elapsed < 0.25


def test_the_renderer_scan_happens_once() -> None:
    """AST-parsing app/charts on every chart creation would be felt."""
    from app.scanners import axis_renderer_scanner

    elapsed = _timed(
        "10k renderer lookups",
        lambda: [axis_renderer_scanner.get_renderer("Scatter Plot") for _ in range(10_000)],
    )

    assert elapsed < 1.0


def test_the_suite_itself_stays_quick() -> None:
    """A guard on the guards: 600 tests are only useful if they are run.

    Not a timing - a count.  The suite is fast because almost nothing in it
    builds a window; this fails if that stops being true and someone has to be
    told why the suite got slow.
    """
    folder = Path(__file__).resolve().parent
    marker = "QApplication" + "([])"  # split so this file does not match itself
    offenders = [
        path.name
        for path in folder.glob("test_*.py")
        if marker in path.read_text(encoding="utf-8")
    ]

    assert offenders == [], "these build their own QApplication; use the qapp fixture"
