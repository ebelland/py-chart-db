"""Tests for the ECDF and pie renderers.

The ECDF is checked as a function first - its y values are the whole point and
are easy to get subtly wrong at ties and at the ends - and only then as a
renderer.  The pie is checked for the two things it can silently get wrong:
drawing more than one series, and drawing values that mean nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base_axis import SeriesData
from app.charts.ecdf_axis import EcdfAxisRenderer, ecdf_points
from app.charts.pie_axis import PieAxisRenderer


def _axes():
    figure = Figure(figsize=(6.0, 4.0))
    return figure, figure.add_subplot(1, 1, 1)


# ----------------------------------------------------------------------
# ecdf_points
# ----------------------------------------------------------------------
def test_the_curve_ends_at_one() -> None:
    """The last point is the whole sample, so it must be exactly 1."""
    _x, y = ecdf_points(np.array([5.0, 1.0, 3.0, 2.0]))
    assert y[-1] == pytest.approx(1.0)
    assert y[0] == pytest.approx(0.25)


def test_the_values_are_sorted() -> None:
    x, _y = ecdf_points(np.array([5.0, 1.0, 3.0]))
    assert list(x) == sorted(x)


def test_ties_produce_one_point_at_the_top_of_the_jump() -> None:
    """A repeated value is one jump, not several steps of unknowable size."""
    x, y = ecdf_points(np.array([1.0, 2.0, 2.0, 2.0, 3.0]))

    assert list(x) == [1.0, 2.0, 3.0]
    assert y == pytest.approx([0.2, 0.8, 1.0])


def test_weights_reproduce_the_raw_sample() -> None:
    """Pre-aggregated data must give the same curve as the rows it counts."""
    raw = ecdf_points(np.array([1.0, 1.0, 1.0, 2.0]))
    weighted = ecdf_points(np.array([1.0, 2.0]), weights=np.array([3.0, 1.0]))

    assert list(raw[0]) == list(weighted[0])
    assert raw[1] == pytest.approx(weighted[1])


def test_the_complement_is_one_minus_the_curve() -> None:
    _x, plain = ecdf_points(np.array([1.0, 2.0, 3.0, 4.0]))
    _x, survival = ecdf_points(np.array([1.0, 2.0, 3.0, 4.0]), complementary=True)

    assert survival == pytest.approx(1.0 - plain)


def test_percent_is_the_same_curve_scaled() -> None:
    _x, fraction = ecdf_points(np.array([1.0, 2.0]))
    _x, percent = ecdf_points(np.array([1.0, 2.0]), as_percent=True)

    assert percent == pytest.approx(fraction * 100.0)


def test_non_finite_values_are_dropped_not_counted() -> None:
    """Counting a NaN would shift every point on the curve."""
    _x, clean = ecdf_points(np.array([1.0, 2.0]))
    _x, dirty = ecdf_points(np.array([1.0, 2.0, np.nan, np.inf]))

    assert dirty == pytest.approx(clean)


def test_an_empty_sample_is_empty_not_an_error() -> None:
    x, y = ecdf_points(np.array([]))
    assert x.size == 0 and y.size == 0


# ----------------------------------------------------------------------
# The ECDF renderer
# ----------------------------------------------------------------------
def test_each_series_becomes_one_curve() -> None:
    _figure, ax = _axes()
    rng = np.random.default_rng(0)
    series = [
        SeriesData("a", pd.DataFrame({"value": rng.normal(size=50)}), {"marker": "."}),
        SeriesData("b", pd.DataFrame({"value": rng.normal(size=50)}), {"marker": "."}),
    ]

    EcdfAxisRenderer().render_axis(ax, series, {})

    assert len(ax.collections) == 2


def test_the_renderer_options_do_not_reach_matplotlib() -> None:
    """complementary is this renderer's, not a scatter keyword."""
    _figure, ax = _axes()
    series = [SeriesData("a", pd.DataFrame({"value": [1.0, 2.0, 3.0]}), {"marker": "o"})]

    EcdfAxisRenderer().render_axis(
        ax, series, {"complementary": True, "as_percent": True, "alpha": 0.5}
    )

    assert len(ax.collections) == 1


def test_the_y_axis_says_what_it_shows() -> None:
    _figure, ax = _axes()
    series = [SeriesData("a", pd.DataFrame({"value": [1.0, 2.0]}), {"marker": "."})]

    EcdfAxisRenderer().render_axis(ax, series, {"complementary": True})

    assert ax.get_ylabel() == "1 - F(x)"


def test_a_series_without_the_value_role_is_skipped() -> None:
    _figure, ax = _axes()
    series = [SeriesData("a", pd.DataFrame({"x": [1.0], "y": [2.0]}), {"marker": "."})]

    EcdfAxisRenderer().render_axis(ax, series, {})

    assert not ax.collections


def test_saved_ecdf_figure(plots_dir: Path, show_plots: bool) -> None:
    figure, ax = _axes()
    rng = np.random.default_rng(1)
    series = [
        SeriesData(
            "normal",
            pd.DataFrame({"value": rng.normal(size=300)}),
            {"marker": ".", "label": "normal", "color": "#1f77b4"},
        ),
        SeriesData(
            "shifted",
            pd.DataFrame({"value": rng.normal(loc=1.5, size=300)}),
            {"marker": "", "linestyle": "-", "label": "shifted", "color": "#d62728"},
        ),
    ]
    EcdfAxisRenderer().render_axis(ax, series, {})

    if show_plots:
        target = plots_dir / "ecdf" / "two_samples.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=110, bbox_inches="tight")


# ----------------------------------------------------------------------
# The pie renderer
# ----------------------------------------------------------------------
def test_one_wedge_per_positive_value() -> None:
    _figure, ax = _axes()
    series = [SeriesData("p", pd.DataFrame({"value": [1.0, 2.0, 3.0]}), {})]

    PieAxisRenderer().render_axis(ax, series, {})

    assert len(ax.patches) == 3


def test_non_positive_values_are_dropped() -> None:
    """Matplotlib would draw a negative wedge as if it were positive."""
    _figure, ax = _axes()
    series = [SeriesData("p", pd.DataFrame({"value": [1.0, -2.0, np.nan, 3.0]}), {})]

    PieAxisRenderer().render_axis(ax, series, {})

    assert len(ax.patches) == 2


def test_only_the_first_series_is_drawn() -> None:
    """Two pies on one axis would overlap and mean nothing."""
    _figure, ax = _axes()
    series = [
        SeriesData("a", pd.DataFrame({"value": [1.0, 1.0]}), {}),
        SeriesData("b", pd.DataFrame({"value": [1.0, 1.0, 1.0]}), {}),
    ]

    PieAxisRenderer().render_axis(ax, series, {})

    assert len(ax.patches) == 2


def test_an_invisible_series_is_not_the_one_drawn() -> None:
    _figure, ax = _axes()
    series = [
        SeriesData("hidden", pd.DataFrame({"value": [1.0, 1.0]}), {"visible": False}),
        SeriesData("shown", pd.DataFrame({"value": [1.0, 1.0, 1.0]}), {}),
    ]

    PieAxisRenderer().render_axis(ax, series, {})

    assert len(ax.patches) == 3


def test_the_pie_is_round() -> None:
    """On non-square axes a pie is an ellipse, and an ellipse cannot be read."""
    _figure, ax = _axes()
    PieAxisRenderer().render_axis(ax, [SeriesData("p", pd.DataFrame({"value": [1.0]}), {})], {})

    assert ax.get_aspect() == 1.0


def test_labels_and_percentages_are_drawn() -> None:
    _figure, ax = _axes()
    series = [
        SeriesData("p", pd.DataFrame({"value": [1.0, 3.0], "label": ["a", "b"]}), {})
    ]

    PieAxisRenderer().render_axis(ax, series, {"autopct": "%1.0f%%"})

    texts = [text.get_text() for text in ax.texts]
    assert "a" in texts and "b" in texts
    assert "75%" in texts


def test_a_donut_is_a_pie_with_a_hole() -> None:
    _figure, ax = _axes()
    series = [SeriesData("p", pd.DataFrame({"value": [1.0, 1.0]}), {})]

    PieAxisRenderer().render_axis(ax, series, {"wedge_width": 0.4})

    assert all(wedge.width == pytest.approx(0.4) for wedge in ax.patches)


def test_an_unknown_colormap_does_not_stop_the_chart() -> None:
    _figure, ax = _axes()
    series = [SeriesData("p", pd.DataFrame({"value": [1.0, 1.0]}), {})]

    PieAxisRenderer().render_axis(ax, series, {"colormap": "not_a_colormap"})

    assert len(ax.patches) == 2


def test_saved_pie_figure(plots_dir: Path, show_plots: bool) -> None:
    figure, ax = _axes()
    frame = pd.DataFrame(
        {
            "value": [35.0, 25.0, 20.0, 12.0, 8.0],
            "label": ["Alpha", "Beta", "Gamma", "Delta", "Other"],
            "explode": [0.05, 0.0, 0.0, 0.0, 0.0],
        }
    )
    PieAxisRenderer().render_axis(
        ax,
        [SeriesData("share", frame, {})],
        {"wedge_width": 0.45, "colormap": "tab20", "autopct": "%1.0f%%"},
    )

    if show_plots:
        target = plots_dir / "pie" / "donut.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=110, bbox_inches="tight")
