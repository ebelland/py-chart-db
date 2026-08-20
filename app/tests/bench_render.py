"""Standalone render benchmark for the figure pipeline.

Not collected by pytest (the file name does not match ``test_*``).  Run it
directly to reproduce the numbers recorded in ``AUDIT.md``::

    python -m app.tests.bench_render --points 150000 --repeat 5

It reports two numbers:

* **cold**   - the first render, where every series SQL must hit SQLite;
* **warm**   - subsequent renders, which is what a property tweak costs.

Why both: the series cache only removes the SQL cost from repeated renders, so
a single average would hide exactly the effect the benchmark exists to measure.
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from app.charts.render_figure import render_figure_from_descriptor  # noqa: E402
from app.data.sqlite_repo import SqliteRepo  # noqa: E402
from app.tests._large_db_factory import create_large_db  # noqa: E402


def _timeit(fn) -> float:
    """Return the wall-clock seconds taken by a zero-argument callable."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def run(points_per_series: int, repeat: int, *, use_cache: bool = True) -> dict[str, float]:
    """Build a synthetic database, render it repeatedly, and return timings.

    Pass ``use_cache=False`` to reproduce the pre-cache behaviour, which is what
    makes the before/after comparison an apples-to-apples one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bench.dhub"
        create_large_db(
            db_path,
            n_ts=points_per_series,
            n_scatter=points_per_series,
            seed=1234,
        )

        repo = SqliteRepo(db_path=db_path)
        descriptor = repo.load_figure_descriptor(figure_id=1)
        assert descriptor is not None, "benchmark figure descriptor is missing"

        # After the first query, not before: connecting re-reads the cache
        # settings from config.json and would undo this.
        repo._series_cache_enabled = use_cache
        repo.invalidate_series_cache()

        def _render_once() -> None:
            render_figure_from_descriptor(
                figure=Figure(),
                descriptor=descriptor,
                repo=repo,
            )

        cold = _timeit(_render_once)
        warm = [_timeit(_render_once) for _ in range(max(1, repeat))]
        repo.close()

    return {
        "cold_ms": cold * 1000.0,
        # Median, not mean: a shared machine produces occasional outliers that
        # move a mean by more than the effect being measured.
        "warm_median_ms": statistics.median(warm) * 1000.0,
        "warm_min_ms": min(warm) * 1000.0,
    }


def main() -> None:
    """Parse arguments, run the benchmark, and print a one-line summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=150_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the series cache to reproduce the pre-optimisation timings",
    )
    args = parser.parse_args()

    result = run(args.points, args.repeat, use_cache=not args.no_cache)
    print(
        f"points/series={args.points}  "
        f"cache={'off' if args.no_cache else 'on'}  "
        f"cold={result['cold_ms']:.1f} ms  "
        f"warm median={result['warm_median_ms']:.1f} ms  "
        f"warm min={result['warm_min_ms']:.1f} ms"
    )


if __name__ == "__main__":
    main()
