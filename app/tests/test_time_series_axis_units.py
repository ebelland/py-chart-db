"""The time-series renderer must handle numeric and timestamp x columns.

Regression cover for two bugs that produced plausible-looking but wrong output:
a real timestamp column was coerced to numbers first and therefore dropped
entirely, and a plain numeric axis was labelled with dates because the date
locator was installed unconditionally.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from matplotlib.dates import AutoDateLocator
from matplotlib.figure import Figure

from app.charts.base_axis import SeriesData
from app.charts.time_series_axis import TimeSeriesAxisRenderer


def _render(df: pd.DataFrame) -> tuple[Figure, object]:
    """Render a single series and return the figure and its axis."""
    fig = Figure(figsize=(7.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    TimeSeriesAxisRenderer().render_axis(
        ax=ax,
        series=[SeriesData(name="s", df=df, style={"linestyle": "-"})],
        options={},
    )
    return fig, ax


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")


def test_numeric_x_keeps_a_numeric_axis(plots_dir: Path, show_plots: bool) -> None:
    df = pd.DataFrame({"x": range(500), "y": [float(v) ** 0.5 for v in range(500)]})
    fig, ax = _render(df)

    assert ax.lines, "numeric series was not drawn"
    assert not isinstance(ax.xaxis.get_major_locator(), AutoDateLocator)

    # The data spans 0..499; allow for Matplotlib's default margins, but the
    # span must stay in the hundreds rather than jumping to date numbers.
    low, high = ax.get_xlim()
    assert -100 <= low < high <= 700, f"unexpected numeric x range: {(low, high)}"

    if show_plots:
        _save(fig, plots_dir / "time_series" / "numeric_x.png")


def test_timestamp_x_is_plotted_as_dates(plots_dir: Path, show_plots: bool) -> None:
    stamps = pd.date_range("2026-01-01", periods=500, freq="h")
    df = pd.DataFrame({"x": stamps.astype(str), "y": range(500)})
    fig, ax = _render(df)

    assert ax.lines, "timestamp series was dropped instead of plotted"
    assert isinstance(ax.xaxis.get_major_locator(), AutoDateLocator)

    if show_plots:
        _save(fig, plots_dir / "time_series" / "timestamp_x.png")


@pytest.mark.parametrize("threshold", ["2h", "not-a-threshold"])
def test_gap_threshold_never_breaks_a_numeric_axis(threshold: str) -> None:
    """An offset string means "no gaps" on a numeric axis, never an exception."""
    frame = pd.DataFrame({"x": [0.0, 1.0, 50.0], "y": [1.0, 2.0, 3.0]})
    result = TimeSeriesAxisRenderer._insert_gaps(frame, threshold, temporal=False)
    assert len(result) == len(frame)
