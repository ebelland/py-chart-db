"""Deciding whether x/y/z rows are on a grid, and getting them out as arrays.

Several chart types draw the same quantity two different ways depending on how
it was sampled: values on a complete x/y grid go to one Matplotlib call, and
values at scattered points go to another that triangulates them.  Which of the
two is in front of you is the part worth getting right, and worth having in
one place: a set of points that is only *nearly* a grid must not be treated as
one, or the renderer quietly draws through cells nobody measured.

:func:`pivot_to_grid` is that decision.  ``SurfaceAxisRenderer`` and
``ContourAxisRenderer`` both ask it, and both refuse the frame and name their
scattered counterpart when it says no.  :func:`finite_xyz` is the other half:
the scattered path, which needs the raw points rather than a grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.logs.logger import applogger

#: The largest grid this will pivot, in cells.
#:
#: Four million is a 2000 x 2000 map, which contour draws in about a second.
#: Past that the cost is in Matplotlib rather than here, and a chart nobody
#: can wait for is not a chart - so the frame is refused with a message
#: naming the way out rather than drawn eventually.
MAX_GRID_CELLS: int = 4_000_000


def pivot_to_grid(
    df: pd.DataFrame,
    *,
    x_role: str = "x",
    y_role: str = "y",
    z_role: str = "z",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pivot x/y/z rows into ``(X, Y, Z)`` grids, or None if they are not a grid.

    "A grid" means a complete Cartesian product: every distinct x value paired
    with every distinct y value, exactly once.  Spacing is not part of the
    test - an unevenly sampled but complete grid is still a grid, and both
    ``plot_surface`` and ``contour`` accept one - but completeness is: a single
    missing pair leaves a cell with no measurement in it, and there is no
    honest value to put there.  None is returned in that case rather than a
    grid with a guess in it, so the caller can point at the scattered renderer
    instead.

    Rows whose x, y or z does not parse as a number are dropped first, so a
    text column mapped to the wrong role reads as "not a grid" rather than
    raising.  Duplicate (x, y) pairs are averaged: repeated measurements at one
    site are a real thing, and their mean is the one summary that does not
    depend on row order.
    """
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(df[x_role], errors="coerce"),
            "y": pd.to_numeric(df[y_role], errors="coerce"),
            "z": pd.to_numeric(df[z_role], errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return None

    x_values = np.sort(frame["x"].unique())
    y_values = np.sort(frame["y"].unique())
    # One row or one column is a line, not a surface, and neither
    # plot_surface nor contour can do anything with it.
    if x_values.size < 2 or y_values.size < 2:
        return None

    cells = int(x_values.size) * int(y_values.size)

    # Answer "is this a grid?" by counting, before allocating anything.
    #
    # This is the whole reason the function is usable on real data. A
    # complete Cartesian product of the distinct x and y values needs one row
    # per cell, so a frame with fewer rows than cells cannot be one - and
    # scattered data is exactly that case, badly: 10 000 points with no two
    # sharing a coordinate means 10 000 x 10 000, and pivot_table would build
    # all hundred million of them, spend three and a half seconds and a
    # gigabyte of memory, and then be told there are holes in it. Thirty
    # thousand points took the machine with it.
    #
    # Duplicates are averaged, so more rows than cells is fine; fewer never
    # is.
    if cells > len(frame):
        return None

    if cells > MAX_GRID_CELLS:
        applogger.warning(
            "This is a %d x %d grid - %s cells, past the %s this draws "
            "without stalling. Aggregate it in the series SQL (GROUP BY a "
            "rounded x and y) or plot a subset.",
            x_values.size,
            y_values.size,
            f"{cells:,}",
            f"{MAX_GRID_CELLS:,}",
            show_dialog=False,
            raise_error=False,
        )
        return None

    pivot = frame.pivot_table(index="y", columns="x", values="z", aggfunc="mean")
    pivot = pivot.reindex(index=y_values, columns=x_values)
    # Counting cannot catch every hole: duplicates at one site can make the
    # totals agree while some other pair is missing entirely.
    if pivot.isna().to_numpy().any():
        return None

    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return x_grid, y_grid, pivot.to_numpy(dtype=float)


def finite_xyz(
    df: pd.DataFrame,
    *,
    x_role: str = "x",
    y_role: str = "y",
    z_role: str = "z",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x/y/z as three float arrays with the non-finite rows removed.

    The scattered counterpart to :func:`pivot_to_grid`: no grid is required,
    only points.  NaN and infinity are dropped rather than passed on, because
    a triangulation cannot place a point it has no coordinate for and a
    non-finite z makes every contour level it participates in meaningless.
    """
    x = pd.to_numeric(df[x_role], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_role], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df[z_role], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[finite], y[finite], z[finite]
