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

    pivot = frame.pivot_table(index="y", columns="x", values="z", aggfunc="mean")
    pivot = pivot.reindex(index=y_values, columns=x_values)
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
