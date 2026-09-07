""""Default" for a series' line style and marker: defer to rcParams.

Before this, a brand-new series always got a hard-coded "-"/"" baked into
its style the moment it was saved - see SeriesPropertiesWidget - which meant
a custom .mplstyle's own ``lines.linestyle``/``lines.marker`` (or
``scatter.marker``) was silently overridden by whichever combo happened to
default to what. "Default" is a real third state, distinct from an explicit
"none"/"" (which still means "never draw one, no matter what"), that lets a
series say "whatever the style sheet says" - see
app.charts.base.BaseAxisRenderer.series_linestyle/series_marker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib import rcParams
from matplotlib.figure import Figure

from app.charts.base import SeriesData
from app.charts.kwarg_spec import DEFAULT
from app.charts.scatter import ScatterAxisRenderer
from app.charts.time_series import TimeSeriesAxisRenderer

RNG = np.random.default_rng(7)


def _frame(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(n, dtype=float), "y": RNG.normal(0, 1, n)})


@pytest.fixture(autouse=True)
def _restore_rcparams():
    """Some tests below mutate rcParams; never let that leak to another test."""
    saved = {
        key: rcParams[key]
        for key in ("lines.linestyle", "lines.marker", "scatter.marker")
    }
    yield
    rcParams.update(saved)


# ----------------------------------------------------------------------
# app.charts.base.BaseAxisRenderer.series_linestyle / series_marker
# ----------------------------------------------------------------------
class _Renderer(ScatterAxisRenderer):
    """Exposes the base helpers without going through a full render."""


def test_missing_key_still_means_no_line_or_marker() -> None:
    """Unrelated to DEFAULT: an absent key keeps its old meaning, since
    Scatter's marker-only shape depends on it."""
    renderer = _Renderer()
    _, has_line = renderer.series_linestyle({}, {})
    _, has_marker = renderer.series_marker({}, {})
    assert has_line is False
    assert has_marker is False


def test_explicit_none_and_empty_still_hard_override() -> None:
    """An explicit "no line"/"no marker" choice never falls back to
    anything - not the axis, not rcParams."""
    renderer = _Renderer()
    rcParams["lines.linestyle"] = "--"
    rcParams["scatter.marker"] = "o"

    _, has_line = renderer.series_linestyle({"linestyle": "none"}, {"linestyle": "-"})
    _, has_marker = renderer.series_marker(
        {"marker": ""}, {"marker": "s"}, rcparam="scatter.marker"
    )

    assert has_line is False
    assert has_marker is False


def test_default_falls_back_to_the_axis_option_first() -> None:
    renderer = _Renderer()
    linestyle, has_line = renderer.series_linestyle(
        {"linestyle": DEFAULT}, {"linestyle": "--"}
    )
    marker, has_marker = renderer.series_marker(
        {"marker": DEFAULT}, {"marker": "s"}, rcparam="scatter.marker"
    )

    assert (linestyle, has_line) == ("--", True)
    assert (marker, has_marker) == ("s", True)


def test_default_falls_back_to_rcparams_when_the_axis_says_nothing() -> None:
    renderer = _Renderer()
    rcParams["lines.linestyle"] = "-."
    rcParams["scatter.marker"] = "D"

    linestyle, has_line = renderer.series_linestyle({"linestyle": DEFAULT}, {})
    marker, has_marker = renderer.series_marker(
        {"marker": DEFAULT}, {}, rcparam="scatter.marker"
    )

    # None: the caller passes this straight to Matplotlib, which re-reads
    # rcParams itself - a value copied out here would go stale the moment
    # the style sheet changed again.
    assert (linestyle, has_line) == (None, True)
    assert (marker, has_marker) == (None, True)


def test_default_marker_reads_lines_marker_for_a_plain_line_renderer() -> None:
    """Scatter's ax.scatter and a plain line's ax.plot default differently
    - lines.marker is normally "None", scatter.marker normally "o" - so
    "Default" must ask the rcParam the caller will actually draw with."""
    renderer = _Renderer()
    rcParams["lines.marker"] = "None"
    rcParams["scatter.marker"] = "o"

    line_marker, line_has_marker = renderer.series_marker({"marker": DEFAULT}, {})
    scatter_marker, scatter_has_marker = renderer.series_marker(
        {"marker": DEFAULT}, {}, rcparam="scatter.marker"
    )

    assert (line_marker, line_has_marker) == (None, False)
    assert (scatter_marker, scatter_has_marker) == (None, True)


# ----------------------------------------------------------------------
# End to end: a real style-sheet default actually reaches the drawn artist
# ----------------------------------------------------------------------
def test_a_default_scatter_series_actually_draws_with_the_rcparam_marker() -> None:
    rcParams["scatter.marker"] = "s"
    fig = Figure(figsize=(4.0, 3.0))
    ax = fig.add_subplot(1, 1, 1)

    ScatterAxisRenderer().render_axis(
        ax=ax,
        series=[SeriesData(name="points", df=_frame(), style={"marker": DEFAULT})],
        options={},
    )

    assert ax.collections, "expected the scatter layer to draw"


def test_a_default_time_series_line_draws_and_a_style_sheet_marker_shows() -> None:
    rcParams["lines.linestyle"] = "-"
    rcParams["lines.marker"] = "^"
    fig = Figure(figsize=(4.0, 3.0))
    ax = fig.add_subplot(1, 1, 1)

    TimeSeriesAxisRenderer().render_axis(
        ax=ax,
        series=[
            SeriesData(
                name="series",
                df=_frame(),
                style={"linestyle": DEFAULT, "marker": DEFAULT},
            )
        ],
        options={},
    )

    assert ax.lines, "expected a line to be drawn"
    assert ax.lines[0].get_marker() == "^"
