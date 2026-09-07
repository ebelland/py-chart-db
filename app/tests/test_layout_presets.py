"""Pure arithmetic tests for app.charts.layout_presets.

No Qt, no SqliteRepo, no Figure: a preset is just "given these axis ids, what
grid size and options does each one get", which is exactly what is asserted
here. Rendering the result (spans, sharex/sharey, twin_of) is covered
separately, in test_layout_and_overlapping_axes.py and
test_figure_layout_and_frame.py, against the machinery that already reads
those same options for a hand-built layout.
"""
from __future__ import annotations

import pytest

from app.charts import layout_presets as lp


def _options_by_id(plan: lp.LayoutPlan) -> dict[int, dict[str, object]]:
    return {placement.axis_id: placement.options for placement in plan.axes}


def _index_by_id(plan: lp.LayoutPlan) -> dict[int, int]:
    return {placement.axis_id: placement.axis_index for placement in plan.axes}


# ----------------------------------------------------------------------
# Degenerate cases: every preset agrees on 0 or 1 axis
# ----------------------------------------------------------------------
@pytest.mark.parametrize("preset", lp.PRESETS)
def test_no_axes_is_a_1x1_grid_with_nothing_placed(preset: str) -> None:
    plan = lp.plan_layout(preset, [])
    assert (plan.nrows, plan.ncols) == (1, 1)
    assert plan.axes == ()


@pytest.mark.parametrize("preset", lp.PRESETS)
def test_one_axis_is_a_1x1_grid_with_every_owned_key_cleared(preset: str) -> None:
    plan = lp.plan_layout(preset, [42])
    assert (plan.nrows, plan.ncols) == (1, 1)
    assert len(plan.axes) == 1
    placement = plan.axes[0]
    assert placement.axis_id == 42
    assert placement.axis_index == 0
    assert all(value is None for value in placement.options.values())


def test_an_unknown_preset_raises() -> None:
    with pytest.raises(ValueError):
        lp.plan_layout("not-a-real-preset", [1, 2])


# ----------------------------------------------------------------------
# Every placement always carries every owned key, so applying a preset
# always replaces the last one's rather than merging with it.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("preset", lp.PRESETS)
def test_every_placement_carries_every_owned_key(preset: str) -> None:
    plan = lp.plan_layout(preset, [1, 2, 3, 4, 5])
    owned = {"row_span", "col_span", "sharex", "sharey", "twin_of"}
    for placement in plan.axes:
        assert set(placement.options.keys()) == owned


@pytest.mark.parametrize("preset", lp.PRESETS)
def test_every_axis_id_appears_exactly_once(preset: str) -> None:
    ids = [10, 20, 30, 40, 50, 60, 70]
    plan = lp.plan_layout(preset, ids)
    placed_ids = [placement.axis_id for placement in plan.axes]
    assert sorted(placed_ids) == sorted(ids)
    assert len(placed_ids) == len(set(placed_ids))


@pytest.mark.parametrize("preset", lp.PRESETS)
def test_every_axis_index_is_unique_within_the_figure(preset: str) -> None:
    """The UNIQUE(figure_id, axis_index) constraint SqliteRepo enforces."""
    plan = lp.plan_layout(preset, [10, 20, 30, 40, 50])
    indexes = [placement.axis_index for placement in plan.axes]
    assert len(indexes) == len(set(indexes))


# ----------------------------------------------------------------------
# GRID
# ----------------------------------------------------------------------
def test_grid_places_every_axis_in_a_compact_square_ish_grid() -> None:
    plan = lp.plan_layout(lp.GRID, [1, 2, 3, 4])
    assert (plan.nrows, plan.ncols) == (2, 2)
    assert _index_by_id(plan) == {1: 0, 2: 1, 3: 2, 4: 3}


def test_grid_clears_sharing_and_spans() -> None:
    plan = lp.plan_layout(lp.GRID, [1, 2, 3])
    for placement in plan.axes:
        assert placement.options == {
            "row_span": None,
            "col_span": None,
            "sharex": None,
            "sharey": None,
            "twin_of": None,
        }


# ----------------------------------------------------------------------
# SHARED_GRID
# ----------------------------------------------------------------------
def test_shared_grid_uses_the_same_grid_shape_as_plain_grid() -> None:
    ids = [1, 2, 3, 4, 5]
    assert (lp.plan_layout(lp.SHARED_GRID, ids).nrows, lp.plan_layout(lp.SHARED_GRID, ids).ncols) == (
        lp.plan_layout(lp.GRID, ids).nrows,
        lp.plan_layout(lp.GRID, ids).ncols,
    )


def test_shared_grid_leaves_the_first_axis_unshared() -> None:
    options = _options_by_id(lp.plan_layout(lp.SHARED_GRID, [1, 2, 3]))
    assert options[1]["sharex"] is None
    assert options[1]["sharey"] is None


def test_shared_grid_shares_x_and_y_with_the_rest() -> None:
    options = _options_by_id(lp.plan_layout(lp.SHARED_GRID, [1, 2, 3]))
    for axis_id in (2, 3):
        assert options[axis_id]["sharex"] is True
        assert options[axis_id]["sharey"] is True


# ----------------------------------------------------------------------
# MAIN_AND_SECONDARY
# ----------------------------------------------------------------------
def test_main_and_secondary_gives_the_first_axis_the_full_top_row() -> None:
    plan = lp.plan_layout(lp.MAIN_AND_SECONDARY, [1, 2, 3, 4, 5])
    options = _options_by_id(plan)
    index = _index_by_id(plan)
    assert index[1] == 0
    assert options[1]["col_span"] == plan.ncols


def test_main_and_secondary_fills_the_remaining_rows_with_the_rest() -> None:
    plan = lp.plan_layout(lp.MAIN_AND_SECONDARY, [1, 2, 3, 4, 5])
    # 4 secondary axes -> a 2x2 block under the main row.
    assert (plan.nrows, plan.ncols) == (3, 2)
    index = _index_by_id(plan)
    secondary_indexes = sorted(index[axis_id] for axis_id in (2, 3, 4, 5))
    assert secondary_indexes == [2, 3, 4, 5]


def test_main_and_secondary_with_one_secondary_axis_is_a_two_row_grid() -> None:
    plan = lp.plan_layout(lp.MAIN_AND_SECONDARY, [1, 2])
    assert (plan.nrows, plan.ncols) == (2, 1)
    options = _options_by_id(plan)
    assert options[1]["col_span"] == 1


# ----------------------------------------------------------------------
# OVERLAPPING
# ----------------------------------------------------------------------
def test_overlapping_pairs_axes_two_at_a_time() -> None:
    options = _options_by_id(lp.plan_layout(lp.OVERLAPPING, [1, 2, 3, 4]))
    assert options[1]["twin_of"] is None
    assert options[2]["twin_of"] == 1
    assert options[3]["twin_of"] is None
    assert options[4]["twin_of"] == 3


def test_overlapping_leaves_an_odd_axis_out_as_an_ordinary_grid_cell() -> None:
    plan = lp.plan_layout(lp.OVERLAPPING, [1, 2, 3])
    options = _options_by_id(plan)
    index = _index_by_id(plan)
    assert options[1]["twin_of"] is None
    assert options[2]["twin_of"] == 1
    assert options[3]["twin_of"] is None
    # The odd one out is a real grid cell, distinct from every other axis.
    assert index[3] not in {index[1], index[2]}


def test_overlapping_grid_only_counts_primary_axes() -> None:
    """Four axes, two overlapping pairs - the grid only needs to fit the two
    primaries; each twin shares its primary's cell rather than needing one
    of its own."""
    plan = lp.plan_layout(lp.OVERLAPPING, [1, 2, 3, 4])
    assert (plan.nrows, plan.ncols) == (1, 2)
