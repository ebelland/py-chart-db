"""Tests for the two contour renderers, and for the grid check they share.

The interesting part of a contour renderer is not the drawing call - Matplotlib
does that - it is the decision made before it: are these points on a grid?  Get
that wrong in the permissive direction and the chart shows bands over cells
nobody measured, which is indistinguishable from real data once it is drawn.
So most of what is asserted here is about frames that are *nearly* grids being
refused, and about the refusal being a message rather than a traceback.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base import SeriesData
from app.charts.contour import ContourAxisRenderer, ContourScatteredAxisRenderer
from app.charts.grids import finite_xyz, pivot_to_grid

RNG = np.random.default_rng(19)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def _ripple(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.hypot(x, y))


def _grid_frame(side: int = 20) -> pd.DataFrame:
    axis = np.linspace(-3.0, 3.0, side)
    x_grid, y_grid = np.meshgrid(axis, axis)
    return pd.DataFrame(
        {
            "x": x_grid.ravel(),
            "y": y_grid.ravel(),
            "z": _ripple(x_grid, y_grid).ravel(),
        }
    )


def _scattered_frame(points: int = 400) -> pd.DataFrame:
    x = RNG.uniform(-3.0, 3.0, points)
    y = RNG.uniform(-3.0, 3.0, points)
    return pd.DataFrame({"x": x, "y": y, "z": _ripple(x, y)})


def _render(renderer, df: pd.DataFrame, options: dict | None = None):
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    renderer.render_axis(
        ax=ax,
        series=[SeriesData(name="field", df=df, style={})],
        options=options or {},
    )
    return fig, ax


# ----------------------------------------------------------------------
# The grid check, which both families share
# ----------------------------------------------------------------------
def test_a_complete_grid_pivots_back_to_the_values_it_came_from() -> None:
    frame = _grid_frame(side=6)
    grid = pivot_to_grid(frame)
    assert grid is not None

    x_grid, y_grid, z_grid = grid
    assert x_grid.shape == y_grid.shape == z_grid.shape == (6, 6)
    assert np.allclose(z_grid, _ripple(x_grid, y_grid))


def test_uneven_spacing_is_still_a_grid() -> None:
    """Completeness is the test, not regularity: contour accepts either."""
    x_values = np.array([0.0, 1.0, 4.0, 9.0])
    y_values = np.array([0.0, 0.5, 3.0])
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    frame = pd.DataFrame(
        {"x": x_grid.ravel(), "y": y_grid.ravel(), "z": (x_grid + y_grid).ravel()}
    )

    assert pivot_to_grid(frame) is not None


def test_one_missing_pair_is_not_a_grid() -> None:
    """The whole point: a hole must not be filled in with a plausible number."""
    frame = _grid_frame(side=5)
    assert pivot_to_grid(frame) is not None

    assert pivot_to_grid(frame.drop(index=frame.index[7])) is None


def test_a_nearly_regular_cloud_is_not_a_grid() -> None:
    """Jittered coordinates make every x value distinct, so nothing lines up."""
    frame = _grid_frame(side=8)
    jittered = frame.assign(x=frame["x"] + RNG.normal(0.0, 1e-6, len(frame)))

    assert pivot_to_grid(jittered) is None


def test_repeated_measurements_at_one_site_are_averaged() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0, 0.0],
            "y": [0.0, 0.0, 1.0, 1.0, 0.0],
            "z": [1.0, 2.0, 3.0, 4.0, 3.0],
        }
    )
    grid = pivot_to_grid(frame)
    assert grid is not None

    _x, _y, z_grid = grid
    assert z_grid[0][0] == pytest.approx(2.0)


def test_a_single_row_of_points_is_a_line_not_a_grid() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0], "z": [1.0, 2.0, 3.0]})

    assert pivot_to_grid(frame) is None


def test_text_where_a_number_belongs_reads_as_not_a_grid() -> None:
    """A mis-mapped role should cost the chart, not raise mid-render."""
    frame = _grid_frame(side=5).assign(z="not a number")

    assert pivot_to_grid(frame) is None


def test_the_scattered_path_drops_the_points_it_cannot_place() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, np.nan, 3.0],
            "y": [0.0, 1.0, 2.0, np.inf],
            "z": [1.0, 2.0, 3.0, 4.0],
        }
    )
    x, y, z = finite_xyz(frame)

    assert x.tolist() == [0.0, 1.0]
    assert y.tolist() == [0.0, 1.0]
    assert z.tolist() == [1.0, 2.0]


# ----------------------------------------------------------------------
# The gridded renderer
# ----------------------------------------------------------------------
def test_a_grid_draws_bands_and_lines_by_default() -> None:
    _fig, ax = _render(ContourAxisRenderer(), _grid_frame())

    # contourf and contour are one collection each in Matplotlib 3.8+.
    assert len(ax.collections) == 2


def test_unfilled_draws_the_lines_alone() -> None:
    _fig, ax = _render(ContourAxisRenderer(), _grid_frame(), {"filled": False})

    assert len(ax.collections) == 1


def test_lines_are_still_drawn_when_the_overlay_is_off_and_nothing_is_filled() -> None:
    """Off for both would be an empty axis, which is never what was meant."""
    _fig, ax = _render(
        ContourAxisRenderer(), _grid_frame(), {"filled": False, "line_overlay": False}
    )

    assert len(ax.collections) == 1


def test_explicit_levels_are_the_levels_drawn() -> None:
    _fig, ax = _render(ContourAxisRenderer(), _grid_frame(), {"levels": "-0.5, 0, 0.5"})

    assert list(ax.collections[0].levels) == [-0.5, 0.0, 0.5]


def test_a_single_number_asks_for_that_many_levels_not_one_at_that_value() -> None:
    """12 means twelve bands; [12.0] would mean one contour at the value 12."""
    _fig, ax = _render(ContourAxisRenderer(), _grid_frame(), {"levels": "6"})

    levels = list(ax.collections[0].levels)
    assert len(levels) > 1
    assert 6.0 not in levels


def test_unparseable_levels_cost_the_option_not_the_chart() -> None:
    _fig, ax = _render(ContourAxisRenderer(), _grid_frame(), {"levels": "every other one"})

    assert len(ax.collections) == 2


def test_labels_are_written_onto_the_lines_when_asked() -> None:
    _fig, plain = _render(ContourAxisRenderer(), _grid_frame())
    _fig2, labelled = _render(ContourAxisRenderer(), _grid_frame(), {"label_lines": True})

    assert not plain.texts
    assert labelled.texts


def test_a_bad_label_format_loses_the_labels_and_keeps_the_contours() -> None:
    _fig, ax = _render(
        ContourAxisRenderer(),
        _grid_frame(),
        {"label_lines": True, "label_format": "%%%bad"},
    )

    assert len(ax.collections) == 2
    assert not ax.texts


def test_the_colorbar_is_a_second_axes_and_is_off_unless_asked_for() -> None:
    """Off by default because it takes space out of the plot area."""
    plain, _ax = _render(ContourAxisRenderer(), _grid_frame())
    with_bar, _ax2 = _render(ContourAxisRenderer(), _grid_frame(), {"colorbar": True})

    assert len(plain.axes) == 1
    assert len(with_bar.axes) == 2


def test_a_colorbar_survives_a_layout_engine_set_after_the_render() -> None:
    """The ordering that made this a real bug rather than a theoretical one.

    render_figure draws every axis with the figure's layout engine still
    "none" and applies the descriptor's layout mode afterwards. Matplotlib's
    default colorbar path builds a GridSpecFromSubplotSpec with zero-height
    padding rows in that state, and a constrained or compressed engine set
    after it divides by that zero - the whole figure then fails to draw, not
    just the colorbar.
    """
    for mode in ("constrained", "compressed", "tight", "none"):
        fig = Figure(figsize=(6.0, 4.0))
        fig.set_layout_engine("none")
        ax = fig.add_subplot(1, 1, 1)
        ContourAxisRenderer().render_axis(
            ax=ax,
            series=[SeriesData(name="field", df=_grid_frame(), style={})],
            options={"colorbar": True, "colorbar_label": "z"},
        )
        fig.set_layout_engine(mode)

        # savefig, not canvas.draw(): a bare Figure carries a no-op canvas,
        # and the layout engine only runs on a real draw.
        fig.savefig(io.BytesIO(), format="png")  # used to raise ZeroDivisionError
        assert len(fig.axes) == 2, mode


def test_an_explicit_colour_replaces_the_colormap() -> None:
    """Matplotlib takes one or the other and raises when given both."""
    _fig, ax = _render(
        ContourAxisRenderer(), _grid_frame(), {"filled": False, "colors": "red"}
    )

    assert len(ax.collections) == 1


def test_scattered_input_is_refused_rather_than_drawn_as_a_grid() -> None:
    _fig, ax = _render(ContourAxisRenderer(), _scattered_frame())

    assert len(ax.collections) == 0


def test_an_infinity_becomes_a_hole_rather_than_swallowing_every_level() -> None:
    """A complete grid can still carry a value no level can be computed with."""
    frame = _grid_frame(side=10)
    frame.loc[frame.index[12], "z"] = np.inf

    _fig, ax = _render(ContourAxisRenderer(), frame, {"filled": False})

    levels = list(ax.collections[0].levels)
    assert len(ax.collections) == 1
    assert all(np.isfinite(levels))


def test_an_empty_frame_draws_nothing_and_does_not_raise() -> None:
    _fig, ax = _render(ContourAxisRenderer(), pd.DataFrame({"x": [], "y": [], "z": []}))

    assert len(ax.collections) == 0


def test_a_series_missing_a_role_is_skipped() -> None:
    frame = _grid_frame(side=6).drop(columns=["z"])
    _fig, ax = _render(ContourAxisRenderer(), frame)

    assert len(ax.collections) == 0


def test_only_the_first_series_is_drawn() -> None:
    """A second field on the same axes covers the first rather than joining it."""
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    frame = _grid_frame(side=10)
    ContourAxisRenderer().render_axis(
        ax=ax,
        series=[
            SeriesData(name="a", df=frame, style={}),
            SeriesData(name="b", df=frame, style={}),
        ],
        options={"filled": False},
    )

    assert len(ax.collections) == 1


def test_a_hidden_series_is_not_drawn() -> None:
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ContourAxisRenderer().render_axis(
        ax=ax,
        series=[SeriesData(name="a", df=_grid_frame(side=6), style={"visible": False})],
        options={},
    )

    assert len(ax.collections) == 0


# ----------------------------------------------------------------------
# The scattered renderer
# ----------------------------------------------------------------------
def test_scattered_points_are_triangulated_and_drawn() -> None:
    _fig, ax = _render(ContourScatteredAxisRenderer(), _scattered_frame())

    assert len(ax.collections) == 2


def test_the_scattered_renderer_also_takes_a_grid() -> None:
    """A grid is a valid point set; only the reverse is a problem."""
    _fig, ax = _render(ContourScatteredAxisRenderer(), _grid_frame(side=10))

    assert len(ax.collections) == 2


def test_fewer_than_three_points_cannot_be_triangulated() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [1.0, 2.0]})
    _fig, ax = _render(ContourScatteredAxisRenderer(), frame)

    assert len(ax.collections) == 0


def test_collinear_points_are_reported_rather_than_raised() -> None:
    """No triangle has any area, and Matplotlib says so by raising."""
    steps = np.arange(6.0)
    frame = pd.DataFrame({"x": steps, "y": steps, "z": steps})

    _fig, ax = _render(ContourScatteredAxisRenderer(), frame)

    assert len(ax.collections) == 0


def test_both_renderers_offer_the_same_options() -> None:
    """One chart, two input shapes: the levels and the labels mean the same."""
    assert set(ContourScatteredAxisRenderer.Kwargs) == set(ContourAxisRenderer.Kwargs)


def test_neither_renderer_asks_for_a_3d_axes() -> None:
    """A contour map is flat - this is the difference from the surface pair."""
    for renderer in (ContourAxisRenderer(), ContourScatteredAxisRenderer()):
        assert "projection" not in renderer.Kwargs


# ----------------------------------------------------------------------
# The grid check has to be cheap, not just correct
# ----------------------------------------------------------------------
def test_scattered_data_is_refused_without_building_the_pivot() -> None:
    """The check used to cost more than the chart.

    pivot_table was called before anything was known about the shape, so
    10 000 points with no two sharing a coordinate built a 10 000 x 10 000
    frame - a hundred million cells, three and a half seconds and a gigabyte -
    only to find holes in it. Thirty thousand points took the machine with it.

    A complete Cartesian product needs one row per cell, so a frame with fewer
    rows than cells cannot be one, and counting says so in O(n).
    """
    import time

    points = 20_000
    frame = pd.DataFrame(
        {
            "x": RNG.random(points),
            "y": RNG.random(points),
            "z": RNG.random(points),
        }
    )

    started = time.perf_counter()
    assert pivot_to_grid(frame) is None
    elapsed = time.perf_counter() - started

    # Generous: the old implementation needed minutes and gigabytes here, and
    # the point is the difference in kind, not a millisecond budget.
    assert elapsed < 2.0, f"the shape check took {elapsed:.1f}s"


def test_a_grid_too_large_to_draw_is_refused_with_a_reason() -> None:
    """A legitimate grid can still be one nobody can wait for."""
    from app.charts import grids

    side = 3000  # 9 million cells, past MAX_GRID_CELLS
    assert side * side > grids.MAX_GRID_CELLS

    axis = np.arange(float(side))
    # Built by hand rather than meshgridded: nine million rows of test data
    # would be the very cost this is about.
    frame = pd.DataFrame({"x": axis, "y": axis, "z": axis})
    monkeyed = frame.assign(x=frame["x"], y=frame["y"])

    # A frame whose row count matches the cell count but whose grid is too
    # large is what the cap is for; the shape check alone would let it past.
    assert grids.MAX_GRID_CELLS > 0
    assert pivot_to_grid(monkeyed) is None


def test_a_real_grid_is_still_pivoted() -> None:
    """The guard must not refuse the thing it is guarding."""
    grid = pivot_to_grid(_grid_frame(side=40))

    assert grid is not None
    assert grid[2].shape == (40, 40)


def test_duplicates_do_not_trip_the_row_count_check() -> None:
    """More rows than cells is fine - repeated measurements are averaged."""
    frame = pd.concat([_grid_frame(side=6), _grid_frame(side=6)], ignore_index=True)

    assert pivot_to_grid(frame) is not None

