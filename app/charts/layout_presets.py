"""Figure layout presets: what to write into each axis for a one-click
arrangement, computed from nothing but the axis ids an existing figure
already has.

Deliberately apart from ``render_figure.py`` and from any Qt code: a preset
writes exactly the ``row_span``/``col_span``/``sharex``/``sharey``/``twin_of``
options render_figure.py already reads for a layout built by hand, one axis
at a time, so applying a preset is indistinguishable at render time from
someone having set those options themselves. Keeping the arithmetic here,
with no database and no figure, is what lets "where does axis 4 of 7 go"
be tested on its own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: A uniform grid, one axis per cell - the default a new figure already
#: renders as, offered here mainly so re-picking it clears whatever another
#: preset (or a hand edit) left behind.
GRID: str = "grid"

#: The same grid, but every axis after the first shares the first axis's x
#: and y scale - panning or zooming one moves them together.
SHARED_GRID: str = "shared_grid"

#: The first axis spans the full top row; the rest fill a compact grid
#: underneath it - one large chart with several smaller ones for context.
MAIN_AND_SECONDARY: str = "main_and_secondary"

#: Axes paired up two at a time, the second of each pair drawn on top of the
#: first with its own y-scale on the right (`ax.twinx()`) - a dual-scale
#: chart. An odd axis out is left as an ordinary grid cell of its own.
OVERLAPPING: str = "overlapping"

PRESETS: tuple[str, ...] = (GRID, SHARED_GRID, MAIN_AND_SECONDARY, OVERLAPPING)

#: Keys a preset owns outright. Every placement sets all five, clearing the
#: ones it does not use to ``None`` (removed by the caller rather than
#: stored as null - see SqliteRepo.apply_axis_layout), so re-picking a
#: preset - or picking a different one - replaces the last one's keys
#: instead of layering on top of them.
_OWNED_KEYS: tuple[str, ...] = ("row_span", "col_span", "sharex", "sharey", "twin_of")


@dataclass(frozen=True, slots=True)
class AxisPlacement:
    """One axis's new grid position and the options a preset writes for it."""

    axis_id: int
    axis_index: int
    options: dict[str, object]


@dataclass(frozen=True, slots=True)
class LayoutPlan:
    """A whole figure's new grid size and every axis's placement in it."""

    nrows: int
    ncols: int
    axes: tuple[AxisPlacement, ...]


def _compact_grid(count: int) -> tuple[int, int]:
    """Return a roughly-square (rows, cols) grid that holds *count* cells."""
    count = max(1, int(count))
    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))
    return max(1, rows), max(1, cols)


def _cleared(**overrides: object) -> dict[str, object]:
    """Every owned key set to None (cleared), then *overrides* applied."""
    options: dict[str, object] = dict.fromkeys(_OWNED_KEYS)
    options.update(overrides)
    return options


def plan_layout(preset: str, axis_ids: list[int]) -> LayoutPlan:
    """Compute the grid size and per-axis options for *preset*.

    *axis_ids* is every axis of the figure, in the order they should be
    placed - the axis list's own order, which is also the order a person
    reads it in. Zero or one axis always resolves to a plain 1x1 grid:
    there is nothing to share, split, or overlap with only one axis.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown layout preset: {preset!r}")

    ids = [int(axis_id) for axis_id in axis_ids]
    if len(ids) <= 1:
        placements = tuple(
            AxisPlacement(axis_id=axis_id, axis_index=0, options=_cleared())
            for axis_id in ids
        )
        return LayoutPlan(nrows=1, ncols=1, axes=placements)

    builder = _PRESET_BUILDERS[preset]
    return builder(ids)


def _plan_grid(ids: list[int]) -> LayoutPlan:
    rows, cols = _compact_grid(len(ids))
    placements = tuple(
        AxisPlacement(axis_id=axis_id, axis_index=index, options=_cleared())
        for index, axis_id in enumerate(ids)
    )
    return LayoutPlan(nrows=rows, ncols=cols, axes=placements)


def _plan_shared_grid(ids: list[int]) -> LayoutPlan:
    rows, cols = _compact_grid(len(ids))
    placements = [
        AxisPlacement(
            axis_id=axis_id,
            axis_index=index,
            options=_cleared() if index == 0 else _cleared(sharex=True, sharey=True),
        )
        for index, axis_id in enumerate(ids)
    ]
    return LayoutPlan(nrows=rows, ncols=cols, axes=tuple(placements))


def _plan_main_and_secondary(ids: list[int]) -> LayoutPlan:
    main_id, *secondary_ids = ids
    secondary_rows, secondary_cols = (
        _compact_grid(len(secondary_ids)) if secondary_ids else (0, 1)
    )
    cols = max(1, secondary_cols)
    rows = 1 + secondary_rows

    placements = [
        AxisPlacement(axis_id=main_id, axis_index=0, options=_cleared(col_span=cols))
    ]
    # The main axis occupies the whole first row (col_span=cols), so the
    # first secondary cell is the row directly beneath it.
    placements.extend(
        AxisPlacement(axis_id=axis_id, axis_index=cols + offset, options=_cleared())
        for offset, axis_id in enumerate(secondary_ids)
    )
    return LayoutPlan(nrows=rows, ncols=cols, axes=tuple(placements))


def _plan_overlapping(ids: list[int]) -> LayoutPlan:
    primaries = ids[0::2]
    twins = ids[1::2]  # twins[i] is drawn over primaries[i]

    rows, cols = _compact_grid(len(primaries))
    placements = [
        AxisPlacement(axis_id=axis_id, axis_index=index, options=_cleared())
        for index, axis_id in enumerate(primaries)
    ]
    # A twin never renders from its own axis_index - see render_figure's
    # twin handling - so any index past the primaries' own is fine; it only
    # has to be unique per figure.
    placements.extend(
        AxisPlacement(
            axis_id=twin_id,
            axis_index=len(primaries) + offset,
            options=_cleared(twin_of=primaries[offset]),
        )
        for offset, twin_id in enumerate(twins)
    )
    return LayoutPlan(nrows=rows, ncols=cols, axes=tuple(placements))


_PRESET_BUILDERS = {
    GRID: _plan_grid,
    SHARED_GRID: _plan_shared_grid,
    MAIN_AND_SECONDARY: _plan_main_and_secondary,
    OVERLAPPING: _plan_overlapping,
}
