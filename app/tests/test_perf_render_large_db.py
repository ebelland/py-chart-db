"""Render-time budget for a large database (opt in with --run-perf)."""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from app.data.sqlite_repo import SqliteRepo
from app.charts.render_figure import render_figure_from_descriptor
from app.tests._large_db_factory import create_large_db


def _timeit(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def test_perf_render_large_db(tmp_db_path: Path, run_perf: bool) -> None:
    if not run_perf:
        pytest.skip("perf test disabled; pass --run-perf")

    create_large_db(tmp_db_path, n_ts=3000, n_scatter=2000, seed=1234)
    repo = SqliteRepo(db_path=tmp_db_path)
    descriptor = repo.load_figure_descriptor( figure_id=1)

    def _render_once() -> None:
        fig = Figure()
        if descriptor is not None:
            render_figure_from_descriptor(figure=fig, repo=repo, descriptor=descriptor)

    durations = [_timeit(_render_once) for _ in range(5)]
    mean = statistics.mean(durations)

    assert mean < 2.0, f"Mean render time too high: {mean:.2f} seconds"
