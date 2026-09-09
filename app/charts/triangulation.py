"""Scattered points, triangulated: the mesh itself, and the mesh coloured.

The counterpart to the two gridded renderers in ``app/charts/contour.py``,
and built the same way round: nothing is pivoted, because the whole point of
a triangulation is that the samples are *not* on a grid. Matplotlib's own
Delaunay triangulation joins them, exactly as ``TriSurfaceAxisRenderer`` does
in three dimensions - so a scattered survey can be drawn flat here or lifted
there without being resampled first.

``Triangular Mesh`` (``Axes.triplot``) draws the triangulation and nothing
else: no value column at all. That sounds like a diagnostic, and it is - it
answers "where are my samples, and what will an interpolating renderer
assume between them?", which is the question behind every surprising
contour map. It is also a legitimate chart in its own right for a finite
element mesh or a survey network.

``Triangular Color Mesh`` (``Axes.tripcolor``) colours each triangle - or,
with Gouraud shading, each vertex - by a z value, which is the scattered
answer to pcolormesh.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib import rcParams

from app.charts.base import (
    ARTIST_KWARGS,
    CMAP_KWARGS,
    BaseAxisRenderer,
    SeriesData,
    merge,
    pick,
)
from app.charts.grids import finite_xyz
from app.logs.logger import applogger

#: The fewest points a triangulation can be built from.
_MIN_TRIANGULATION_POINTS: int = 3


def _finite_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return x and y as float arrays with the non-finite rows removed.

    ``grids.finite_xyz`` without the z, for the one renderer here that has
    no value column to drop rows on.
    """
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def _too_few_points(name: str, count: int) -> bool:
    """Report and refuse a series with nothing to triangulate."""
    if count >= _MIN_TRIANGULATION_POINTS:
        return False
    applogger.info(
        "Series '%s' skipped: %d finite points, and a triangulation needs "
        "at least %d.",
        name,
        count,
        _MIN_TRIANGULATION_POINTS,
    )
    return True


class TriplotAxisRenderer(BaseAxisRenderer):
    """The Delaunay triangulation of scattered x/y points.

    Role columns:
        x, y   required, one point per row. No value column: this draws the
               mesh, not a field over it.
    """

    Name: str = "Triangular Mesh"
    Category: str = "Irregularly gridded data"
    Description: str = (
        "The triangulation of scattered points, drawn as edges and vertices. "
        "Shows where the samples are and what an interpolating renderer will "
        "assume between them."
    )
    Link: str = "https://matplotlib.org/stable/plot_types/unstructured/triplot.html"

    RequiredRoles: list[str] = ["x", "y"]
    OptionalRoles: list[str] = []

    Kwargs: dict[str, object] = merge(
        pick(ARTIST_KWARGS, "alpha", "label", "zorder", "visible", "rasterized"),
        {
            "color": {
                "default": None,
                "type": str,
                "kind": "color",
                "group": "Appearance",
                "description": "Edge colour. From the style's colour cycle when unset.",
            },
            "linewidth": {
                "default": 0.5,
                "type": float,
                "min": 0.0,
                "max": 5.0,
                "step": 0.1,
                "group": "Appearance",
                "description": (
                    "Edge width. Thin by default: a mesh over a few thousand "
                    "points is a solid block at the style's usual line width."
                ),
            },
            "marker": {
                "default": "",
                "type": str,
                "group": "Appearance",
                "description": (
                    "Marker drawn at each vertex. Empty draws the edges only, "
                    "which is what a dense mesh wants."
                ),
            },
            "markersize": {
                "default": 2.0,
                "type": float,
                "min": 0.0,
                "max": 30.0,
                "step": 0.5,
                "group": "Appearance",
                "description": "Vertex marker size, when one is set.",
            },
        },
    )

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}

        for index, sd in enumerate(series):
            style = dict(sd.style or {})
            if not style.get("visible", True) or not self.ensure_required_roles(sd.df):
                continue

            x, y = _finite_xy(sd.df)
            if _too_few_points(sd.name, x.size):
                continue

            kwargs = self.get_kwargs(self.merge_style(axis_options, style))
            if not str(kwargs.get("color", "") or "").strip():
                # One mesh per series, told apart by the cycle - the same
                # rule every other per-series renderer follows.
                kwargs["color"] = self.series_color(style, index)
            if not str(kwargs.get("marker", "") or "").strip():
                kwargs.pop("marker", None)
                kwargs.pop("markersize", None)

            try:
                ax.triplot(x, y, **kwargs)
            except Exception:
                applogger.exception(
                    "Triangular Mesh failed to draw series '%s'.", sd.name
                )

        self.apply_annotations(ax, axis_options)


class TripcolorAxisRenderer(BaseAxisRenderer):
    """Scattered x/y/z triangulated and coloured by z.

    Role columns:
        x, y, z   required, one sample per row - any layout, no grid needed.
    """

    Name: str = "Triangular Color Mesh"
    Category: str = "Irregularly gridded data"
    Description: str = (
        "Scattered samples triangulated and filled by value - the "
        "unstructured answer to a pcolormesh, for a field measured wherever "
        "it could be measured."
    )
    Link: str = "https://matplotlib.org/stable/plot_types/unstructured/tripcolor.html"

    RequiredRoles: list[str] = ["x", "y", "z"]
    OptionalRoles: list[str] = []

    #: Two filled meshes over one set of axes hide each other.
    MaxSeries: int | None = 1

    Kwargs: dict[str, object] = merge(
        pick(ARTIST_KWARGS, "alpha", "zorder", "visible", "rasterized"),
        pick(CMAP_KWARGS, "cmap", "norm", "vmin", "vmax"),
        {
            "cmap": {
                "description": (
                    "Colormap the values are mapped through. From the style's "
                    "image.cmap when unset - the mesh is always colour-mapped, "
                    "but which map is the style's to say."
                ),
            },
            "shading": {
                "default": "flat",
                "type": ["flat", "gouraud"],
                "group": "Appearance",
                "description": (
                    "\"flat\" gives each triangle the mean of its corners and "
                    "shows the triangulation; \"gouraud\" interpolates across "
                    "it and reads as a smooth field."
                ),
            },
            "edgecolors": {
                "default": None,
                "type": str,
                "kind": "color",
                "group": "Appearance",
                "description": (
                    "Triangle outline colour. Empty draws no outlines, which "
                    "is what a dense mesh wants."
                ),
            },
            "linewidth": {
                "default": 0.0,
                "type": float,
                "min": 0.0,
                "max": 5.0,
                "step": 0.1,
                "group": "Appearance",
                "description": "Triangle outline width. 0 draws none.",
            },
        },
    )

    #: Read here and never forwarded: a colorbar is a second Axes, not a
    #: keyword of the drawing call.
    Options: dict[str, object] = {
        "colorbar": {
            # Off by default, as on the contour renderers and for their
            # reason: it is a second Axes taken out of this one's space, and
            # a figure that was laid out without it should not silently
            # reflow because a mesh was added to one panel. The colours mean
            # little without it, so it is the first thing to turn on.
            "default": False,
            "type": bool,
            "group": "Colorbar",
            "description": (
                "Add a colorbar. It is a second Axes taken out of this one's "
                "space, so a constrained or compressed figure layout places "
                "it best."
            ),
        },
        "colorbar_label": {
            "default": None,
            "type": str,
            "group": "Colorbar",
            "description": "Label written alongside the colorbar.",
        },
    }

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        sd = self.single_series(
            series, reason="a second filled mesh would cover the first"
        )
        if sd is None:
            return
        x, y, z = finite_xyz(sd.df)
        if _too_few_points(sd.name, x.size):
            return

        merged = self.merge_style(axis_options, sd.style or {})
        kwargs = {
            key: value
            for key, value in self.get_kwargs(merged).items()
            if value is not None and value != ""
        }
        kwargs.setdefault("cmap", rcParams["image.cmap"])

        try:
            mesh = ax.tripcolor(x, y, z, **kwargs)
        except Exception:
            applogger.exception(
                "Triangular Color Mesh failed to draw series '%s'.", sd.name
            )
            return

        if bool(self.opt("colorbar", merged)):
            self._colorbar(ax, mesh, merged)

        self.apply_annotations(ax, axis_options)

    def _colorbar(self, ax: Any, mappable: Any, options: dict[str, Any]) -> None:
        """Add a colorbar beside *ax*.

        ``use_gridspec=False`` for the reason ContourAxisRenderer._colorbar
        sets it: renderers draw while the figure's layout engine is still
        "none", and the gridspec path taken in that state builds padding rows
        of zero height that a constrained engine then divides by.
        """
        figure = getattr(ax, "figure", None)
        if mappable is None or figure is None:
            return
        colorbar = figure.colorbar(mappable, ax=ax, use_gridspec=False)
        label = self.opt("colorbar_label", options)
        if label is not None and str(label).strip() != "":
            colorbar.set_label(str(label))
