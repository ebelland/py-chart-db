"""app.charts.descriptor_grid: fixing a stale rows/cols pair before render.

Extracted from ChartPanel, which had no direct test of its own for this -
only end-to-end coverage through a live figure. These pin the pure logic on
its own: a descriptor tree in, a normalized one (or the same one) out.
"""
from __future__ import annotations

import pytest

from app.charts.descriptor_grid import (
    descriptor_axis_count,
    descriptor_prepared_for_render,
    is_axis_collection,
    minimum_grid_for_axis_count,
    normalize_descriptor_grids,
)


def _figure(nrows: int, ncols: int, n_axes: int) -> dict:
    return {
        "nrows": nrows,
        "ncols": ncols,
        "axes": [{"title": f"ax{i}"} for i in range(n_axes)],
    }


# ----------------------------------------------------------------------
# descriptor_axis_count
# ----------------------------------------------------------------------
def test_counts_the_descriptor_axes_not_a_stale_grid() -> None:
    assert descriptor_axis_count(_figure(2, 1, 1)) == 1


def test_recurses_into_nested_mappings_and_lists() -> None:
    nested = {"figures": [_figure(1, 1, 3)]}
    assert descriptor_axis_count(nested) == 3


def test_an_empty_axes_list_still_counts_as_zero_axes() -> None:
    assert descriptor_axis_count(_figure(1, 1, 0)) == 0


def test_a_list_of_plain_strings_is_not_an_axis_collection() -> None:
    """Coordinate-ish lists must not be mistaken for descriptor axes."""
    assert is_axis_collection(["x", "y"]) is False
    assert is_axis_collection([1.0, 2.0]) is False


def test_a_list_of_mappings_is_an_axis_collection() -> None:
    assert is_axis_collection([{"title": "a"}, {"title": "b"}]) is True


# ----------------------------------------------------------------------
# minimum_grid_for_axis_count
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "axis_count, old_rows, old_cols, expected",
    [
        (1, 2, 1, (1, 1)),
        (4, 2, 2, (2, 2)),  # already minimal and already the right shape
        (3, 1, 4, (1, 3)),  # a stale 1x4 shrinks to fit 3 axes
        (0, 3, 3, (1, 1)),
    ],
)
def test_minimum_grid_shrinks_to_the_smallest_cell_count(
    axis_count, old_rows, old_cols, expected
) -> None:
    assert minimum_grid_for_axis_count(
        axis_count=axis_count, old_rows=old_rows, old_cols=old_cols
    ) == expected


def test_minimum_grid_prefers_the_old_aspect_ratio_among_equal_cell_counts() -> None:
    """6 axes: 1x6, 2x3, 3x2 and 6x1 are all zero-waste. Keep the old shape."""
    assert minimum_grid_for_axis_count(axis_count=6, old_rows=2, old_cols=3) == (2, 3)
    assert minimum_grid_for_axis_count(axis_count=6, old_rows=3, old_cols=2) == (3, 2)


# ----------------------------------------------------------------------
# normalize_descriptor_grids / descriptor_prepared_for_render
# ----------------------------------------------------------------------
def test_a_stale_grid_bigger_than_the_axes_shrinks() -> None:
    figure = _figure(2, 2, 2)  # 4 cells declared, only 2 axes left
    changed = normalize_descriptor_grids(figure, axis_count=2)
    assert changed is True
    assert figure["nrows"] * figure["ncols"] == 2


def test_a_grid_that_already_matches_is_left_alone() -> None:
    figure = _figure(1, 2, 2)
    changed = normalize_descriptor_grids(figure, axis_count=2)
    assert changed is False
    assert (figure["nrows"], figure["ncols"]) == (1, 2)


def test_row_col_key_spelling_variants_are_all_recognised() -> None:
    figure = {"n_rows": 2, "num_cols": 2, "axes": [{}]}
    assert normalize_descriptor_grids(figure, axis_count=1) is True
    assert figure["n_rows"] * figure["num_cols"] == 1


def test_a_mapping_with_only_one_of_the_two_keys_is_left_alone() -> None:
    """Without both keys there is no grid pair to normalize - not an error."""
    figure = {"nrows": 2, "axes": [{}]}
    assert normalize_descriptor_grids(figure, axis_count=1) is False
    assert figure["nrows"] == 2


def test_prepared_for_render_does_not_mutate_the_original() -> None:
    original = _figure(2, 2, 2)
    prepared, changed = descriptor_prepared_for_render(original)
    assert changed is True
    assert original["nrows"] * original["ncols"] == 4  # untouched
    assert prepared["nrows"] * prepared["ncols"] == 2


def test_prepared_for_render_with_no_axes_returns_the_input_unchanged() -> None:
    prepared, changed = descriptor_prepared_for_render({"nrows": 1, "ncols": 1})
    assert changed is False
    assert prepared == {"nrows": 1, "ncols": 1}


def test_axis_count_override_skips_recomputing_it_from_the_descriptor() -> None:
    """Callers that already rendered pass the descriptor count explicitly,
    so a stale grid recomputed from *rendered* Matplotlib axes cannot become
    self-confirming."""
    figure = _figure(2, 2, 4)  # descriptor says 4 axes, override says 1
    prepared, changed = descriptor_prepared_for_render(figure, axis_count_override=1)
    assert changed is True
    assert prepared["nrows"] * prepared["ncols"] == 1
