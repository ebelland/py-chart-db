"""Tests for figure-level layout spans and the frame-on option.

Two behaviours pinned here:

* ``frameon`` in figure options used to be saved and reloaded correctly by
  the properties panel, but the renderer never called ``figure.set_frameon``,
  so unchecking "Draw figure frame" had no visible effect.
* An axis's ``row_span``/``col_span`` options let it occupy more than one
  grid cell, which is the mechanism behind every figure layout other than a
  uniform grid. A 1x1 span (the default) must render exactly like the old
  plain grid.
"""
from __future__ import annotations

from matplotlib.figure import Figure

from app.data.descriptors import AxisDescriptor, FigureDescriptor
from app.charts.render_figure import (
    _apply_figure_options,
    _create_axes_grid,
    _normalized_axes_for_grid,
)


def _axis(
    axis_id: int,
    axis_index: int,
    *,
    row_span: int = 1,
    col_span: int = 1,
) -> AxisDescriptor:
    options: dict = {}
    if row_span != 1:
        options["row_span"] = row_span
    if col_span != 1:
        options["col_span"] = col_span
    return AxisDescriptor(
        id=axis_id,
        figure_id=1,
        axis_index=axis_index,
        chart_type="Scatter Plot",
        title="",
        x_label="",
        y_label="",
        z_label="",
        options=options,
        series=[],
    )


def _figure(nrows: int, ncols: int, axes: list[AxisDescriptor], options: dict | None = None) -> FigureDescriptor:
    return FigureDescriptor(id=1, name="f", nrows=nrows, ncols=ncols, options=options, axes=axes)


# ----------------------------------------------------------------------
# frameon
# ----------------------------------------------------------------------
def test_frameon_false_is_applied_to_the_figure() -> None:
    fig = Figure()
    _apply_figure_options(fig, _figure(1, 1, [], options={"frameon": False}))
    assert fig.get_frameon() is False


def test_frameon_defaults_to_true_when_not_configured() -> None:
    fig = Figure()
    _apply_figure_options(fig, _figure(1, 1, [], options={}))
    assert fig.get_frameon() is True


def test_frameon_true_is_applied_explicitly() -> None:
    fig = Figure()
    fig.set_frameon(False)
    _apply_figure_options(fig, _figure(1, 1, [], options={"frameon": True}))
    assert fig.get_frameon() is True


# ----------------------------------------------------------------------
# spans / layout
# ----------------------------------------------------------------------
def test_default_spans_render_like_a_plain_grid() -> None:
    """A figure with no row_span/col_span keeps today's uniform-grid look."""
    axes = [_axis(1, 0), _axis(2, 1), _axis(3, 2), _axis(4, 3)]
    desc = _figure(2, 2, axes)
    positions, rows, cols, spans_valid = _normalized_axes_for_grid(desc)
    assert (rows, cols) == (2, 2)
    assert spans_valid is True

    fig = Figure()
    axes_flat = _create_axes_grid(figure=fig, descriptor=desc, axes_with_positions=positions, rows=rows, cols=cols)
    assert len(axes_flat) == 4
    assert all(ax is not None for ax in axes_flat)

    widths = {round(ax.get_position().x1 - ax.get_position().x0, 6) for ax in fig.axes}
    heights = {round(ax.get_position().y1 - ax.get_position().y0, 6) for ax in fig.axes}
    # All four cells are the same size in a uniform 2x2 grid.
    assert len(widths) == 1
    assert len(heights) == 1


def test_col_span_widens_an_axis_across_the_top_row() -> None:
    """One axis spanning both columns should be roughly twice as wide."""
    wide = _axis(1, 0, col_span=2)
    left = _axis(2, 2)
    right = _axis(3, 3)
    desc = _figure(2, 2, [wide, left, right])

    positions, rows, cols, spans_valid = _normalized_axes_for_grid(desc)
    assert (rows, cols) == (2, 2)
    assert spans_valid is True

    fig = Figure()
    axes_flat = _create_axes_grid(figure=fig, descriptor=desc, axes_with_positions=positions, rows=rows, cols=cols)
    wide_ax, left_ax, right_ax = axes_flat[0], axes_flat[2], axes_flat[3]

    wide_width = wide_ax.get_position().x1 - wide_ax.get_position().x0
    narrow_width = left_ax.get_position().x1 - left_ax.get_position().x0
    assert wide_width > narrow_width * 1.5

    # The two bottom axes still sit side by side, unaffected by the span.
    assert left_ax.get_position().x1 <= right_ax.get_position().x0 + 1e-9


def test_row_span_heightens_an_axis_down_a_column() -> None:
    tall = _axis(1, 0, row_span=2)
    top_right = _axis(2, 1)
    bottom_right = _axis(3, 3)
    desc = _figure(2, 2, [tall, top_right, bottom_right])

    positions, rows, cols, spans_valid = _normalized_axes_for_grid(desc)
    assert spans_valid is True
    fig = Figure()
    axes_flat = _create_axes_grid(figure=fig, descriptor=desc, axes_with_positions=positions, rows=rows, cols=cols)
    tall_ax, small_ax = axes_flat[0], axes_flat[1]

    tall_height = tall_ax.get_position().y1 - tall_ax.get_position().y0
    small_height = small_ax.get_position().y1 - small_ax.get_position().y0
    assert tall_height > small_height * 1.5


def test_overlapping_spans_fall_back_to_a_compact_one_cell_grid() -> None:
    """A span that would overlap another axis must not raise or overlap.

    Normalization reports the fallback through ``spans_valid=False``, and
    the renderer must honour it: every axis gets exactly one, non-overlapping
    cell even though the descriptor still carries the old col_span=2.
    """
    a = _axis(1, 0, col_span=2)
    b = _axis(2, 1)  # overlaps a's footprint: {(0,0), (0,1)} vs {(0,1)}
    desc = _figure(1, 2, [a, b])

    positions, rows, cols, spans_valid = _normalized_axes_for_grid(desc)
    assert spans_valid is False
    seen_indexes = {index for _axis_desc, index in positions}
    assert len(seen_indexes) == len(positions)  # no duplicate positions

    fig = Figure()
    axes_flat = _create_axes_grid(
        figure=fig,
        descriptor=desc,
        axes_with_positions=positions,
        rows=rows,
        cols=cols,
        respect_spans=spans_valid,
    )
    real_axes = [ax for ax in axes_flat if ax is not None]
    assert len(real_axes) == 2
    widths = {round(ax.get_position().x1 - ax.get_position().x0, 6) for ax in real_axes}
    # Fallback strips the stale col_span=2, so both cells are equal width
    # rather than one axis still covering the other's cell.
    assert len(widths) == 1


def test_span_out_of_grid_bounds_is_clamped_not_raised() -> None:
    """A span edited by hand to overflow the grid degrades instead of crashing.

    col_span=99 does not fit the base 1x2 grid, so normalization already
    falls back (spans_valid=False) rather than leaving it for the defensive
    clamp in _create_axes_grid - either way, rendering must not raise.
    """
    axis = _axis(1, 0, col_span=99)
    desc = _figure(1, 2, [axis])
    positions, rows, cols, spans_valid = _normalized_axes_for_grid(desc)
    fig = Figure()
    axes_flat = _create_axes_grid(
        figure=fig,
        descriptor=desc,
        axes_with_positions=positions,
        rows=rows,
        cols=cols,
        respect_spans=spans_valid,
    )
    assert axes_flat[0] is not None
