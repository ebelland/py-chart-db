"""Fix a stale subplot grid before a chart descriptor is rendered.

Split out of ``ChartPanel``: some saved descriptors can keep an old
``rows``/``cols`` pair after axes are added or removed, and Matplotlib may
then create extra slots or compute the first layout against stale geometry.
Everything here is pure - a descriptor tree in, a normalized one out - so it
is exercised directly rather than through a Qt widget.
"""
from __future__ import annotations

import math
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Any, Final

GRID_ROW_KEYS: Final[tuple[str, ...]] = ("rows", "nrows", "n_rows", "row_count", "num_rows")
GRID_COL_KEYS: Final[tuple[str, ...]] = ("cols", "columns", "ncols", "n_cols", "col_count", "num_cols")
AXIS_COLLECTION_KEYS: Final[tuple[str, ...]] = ("axes", "subplots", "plots", "charts", "panels")


def descriptor_axis_count(value: Any) -> int:
    """Best-effort axis count from a descriptor-like nested structure.

    Important: this must count descriptor axes, not rendered Matplotlib axes.
    A stale 2x1 grid can make the renderer construct two Matplotlib Axes even
    when the descriptor contains only one real chart axis. If that rendered
    count is fed back into grid normalization, the stale 2x1 grid becomes
    self-confirming and never shrinks after a cancelled preview.
    """
    if isinstance(value, MutableMapping):
        for key in AXIS_COLLECTION_KEYS:
            collection = value.get(key)
            if collection is not None and is_axis_collection(collection):
                return len(collection)
        for child in value.values():
            count = descriptor_axis_count(child)
            if count > 0:
                return count
    elif isinstance(value, list):
        # Only recurse into generic lists. Do not treat every Sequence as an
        # axis collection; strings, tuples of coordinates, and other list-
        # like values can appear in chart descriptors but are not axes.
        for child in value:
            count = descriptor_axis_count(child)
            if count > 0:
                return count
    return 0


def is_axis_collection(value: Any) -> bool:
    """Return True for descriptor collections that represent chart axes.

    The old implementation returned True for any non-string Sequence. That
    was too broad: it could count arbitrary lists as axes and, together with
    rendered ``figure.axes`` counts, preserve stale grid sizes. Real axis
    collections in the descriptor are lists of mappings.
    """
    if not isinstance(value, list):
        return False
    if not value:
        return True
    return all(isinstance(item, MutableMapping) for item in value)


def first_present_key(mapping: MutableMapping[Any, Any], candidates: tuple[str, ...]) -> Any | None:
    """Return the concrete key matching one of the candidate names."""
    normalized = {str(key).strip().lower(): key for key in mapping.keys()}
    for candidate in candidates:
        key = normalized.get(candidate)
        if key is not None:
            return key
    return None


def positive_int(value: Any) -> int | None:
    """Parse a positive integer or return None."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def minimum_grid_for_axis_count(*, axis_count: int, old_rows: int, old_cols: int) -> tuple[int, int]:
    """Return the minimum-cell grid that best preserves the old shape."""
    if axis_count <= 0:
        return 1, 1

    old_ratio = float(old_cols) / float(old_rows) if old_rows > 0 else 1.0
    best_rows = 1
    best_cols = axis_count
    best_score: tuple[int, float, int] | None = None

    for rows in range(1, axis_count + 1):
        cols = int(math.ceil(axis_count / rows))
        cells = rows * cols
        ratio = float(cols) / float(rows)
        score = (cells, abs(ratio - old_ratio), abs(cols - old_cols) + abs(rows - old_rows))
        if best_score is None or score < best_score:
            best_score = score
            best_rows = rows
            best_cols = cols

    return best_rows, best_cols


def normalize_grid_mapping(mapping: MutableMapping[Any, Any], axis_count: int) -> bool:
    """Normalize one mapping containing row and column grid keys."""
    row_key = first_present_key(mapping, GRID_ROW_KEYS)
    col_key = first_present_key(mapping, GRID_COL_KEYS)
    if row_key is None or col_key is None:
        return False

    rows = positive_int(mapping.get(row_key))
    cols = positive_int(mapping.get(col_key))
    if rows is None or cols is None:
        return False

    if rows * cols == axis_count:
        return False

    new_rows, new_cols = minimum_grid_for_axis_count(
        axis_count=axis_count, old_rows=rows, old_cols=cols
    )
    mapping[row_key] = new_rows
    mapping[col_key] = new_cols
    return True


def normalize_descriptor_grids(value: Any, axis_count: int) -> bool:
    """Normalize every row/column grid pair found in a descriptor tree."""
    changed = False
    if isinstance(value, MutableMapping):
        changed = normalize_grid_mapping(value, axis_count) or changed
        for child in value.values():
            changed = normalize_descriptor_grids(child, axis_count) or changed
    elif isinstance(value, list):
        for child in value:
            changed = normalize_descriptor_grids(child, axis_count) or changed
    return changed


def descriptor_prepared_for_render(
    descriptor: Any, *, axis_count_override: int | None = None,
) -> tuple[Any, bool]:
    """Return ``(prepared_descriptor, changed)`` with stale grids fixed.

    Normalizes the descriptor before it reaches the renderer so the grid
    capacity is the smallest one that can hold the descriptor's axes.
    ``descriptor`` is never mutated - a deep copy is returned instead, since
    every caller so far has kept the original around (undo, a cancelled
    preview) and needs it untouched.
    """
    axis_count = (
        int(axis_count_override)
        if axis_count_override is not None
        else descriptor_axis_count(descriptor)
    )
    if axis_count <= 0:
        return descriptor, False

    prepared = deepcopy(descriptor)
    changed = normalize_descriptor_grids(prepared, axis_count)
    return prepared, changed
