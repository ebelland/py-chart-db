"""3D renderers: two surfaces and a point cloud.

Two shapes of input, two Matplotlib functions - the same split the contour
renderers (app/charts/contour.py) are built around, and for the same
reason: a grid and a scattered cloud need genuinely different drawing
calls, not one call fed differently-shaped arrays.  The decision itself -
is this actually a grid? - is shared with them, in app/charts/grids.py.

``SurfaceAxisRenderer`` pivots x/y/z rows into a regular grid and draws
``Axes.plot_surface``, for data sampled at every combination of some set of
x values and some set of y values - equispaced or not, as long as it is a
complete grid.  ``TriSurfaceAxisRenderer`` skips the pivot and triangulates
the raw points with ``Axes.plot_trisurf`` instead, which is what scattered,
non-gridded x/y/z needs - see
https://matplotlib.org/stable/gallery/mplot3d/surface3d.html and
https://matplotlib.org/stable/gallery/mplot3d/trisurf3d.html.

``Scatter3DAxisRenderer`` draws the points themselves rather than a skin
over them, for a cloud whose shape - clusters, a plane, an outlier - is the
reading, and for data too sparse or too noisy for a surface to be honest
about.

All three request a 3D axes through ``options["projection"] = "3d"``, returned by
create_chart_dialog.py's per-chart-type axis defaults;
render_figure.py's _subplot_kwargs_for_axis already reads a generic
"projection" option for every renderer, so nothing about axis creation had to
change for 3D support.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib import rcParams

from app.charts.base import (
    ARTIST_KWARGS,
    CMAP_KWARGS,
    VIEW_OPTIONS,
    BaseAxisRenderer,
    SeriesData,
    merge,
    pick,
)
from app.charts.grids import finite_xyz, pivot_to_grid
from app.logs.logger import applogger


def _surface_kwargs(renderer: BaseAxisRenderer, options: dict[str, Any]) -> dict[str, Any]:
    """Return the keywords to forward, with the style's colormap applied.

    Nothing to remove - Options never reaches here.  The colormap is the one
    decision made in code rather than in the schema: a surface is always
    colour-mapped, so an unset cmap falls back to the style's ``image.cmap``
    rather than leaving Matplotlib to draw one flat colour.  Which map is the
    style's to say; that there is one is the renderer's.
    """
    kwargs = renderer.get_kwargs(options)
    kwargs.setdefault("cmap", rcParams["image.cmap"])
    return kwargs


def _view_kwargs(options: dict[str, Any], renderer: BaseAxisRenderer) -> dict[str, float]:
    view: dict[str, float] = {}
    for name in ("elev", "azim", "roll"):
        value = renderer.opt(name, options)
        if value is not None and value != "":
            view[name] = float(str(value))
    return view


class SurfaceAxisRenderer(BaseAxisRenderer):
    """Surface plot over a regular x/y grid.

    Role columns:
        x, y, z   required.  Every (x, y) combination present must appear
                  exactly once - a complete Cartesian product of the
                  distinct x and y values, at any spacing. A row missing
                  from that product leaves a hole in the surface; renders
                  nothing and logs an error instead of guessing a value for
                  it, since a wrong-but-present point looks like real data.
    """

    Name: str = "Surface Plot"
    Category: str = "3D and volumetric data"
    Description: str = "3D surface over a regular x/y grid."
    Link: str = "https://matplotlib.org/stable/gallery/mplot3d/surface3d.html"

    RequiredRoles: list[str] = ["x", "y", "z"]
    OptionalRoles: list[str] = []

    #: Two surfaces sharing one set of axes occlude each other.
    MaxSeries: int | None = 1

    #: Forwarded verbatim to ``plot_surface``.
    Kwargs: dict[str, object] = merge(
        pick(CMAP_KWARGS, "cmap"),
        pick(ARTIST_KWARGS, "alpha"),
        {
            "cmap": {
                "description": (
                    "Colormap the surface height is mapped through. From the "
                    "style's image.cmap when unset - a surface is always "
                    "colour-mapped, but which map is the style's to say."
                ),
            },
            "alpha": {"description": "Surface opacity."},
            "edgecolor": {
                "default": None,
                "type": str,
                "kind": "color",
                "group": "Appearance",
                "description": "Face outline colour. Leave empty for no visible mesh.",
            },
            "linewidth": {
                # Deliberately 0.0 rather than left to patch.linewidth: a mesh
                # over a dense grid is a black rectangle, so this renderer
                # overrules the style on purpose and says so.
                "default": 0.0,
                "type": float,
                "min": 0.0,
                "max": 5.0,
                "step": 0.1,
                "group": "Appearance",
                "description": "Face outline width. 0 draws no mesh.",
            },
            "antialiased": {
                "default": True,
                "type": bool,
                "group": "Appearance",
                "description": "Antialias the surface edges.",
            },
            "rstride": {
                "default": 1,
                "type": int,
                "min": 1,
                "max": 100,
                "group": "Sampling",
                "description": "Draw every Nth row of the grid. Raise this on a large grid to keep rendering fast.",
            },
            "cstride": {
                "default": 1,
                "type": int,
                "min": 1,
                "max": 100,
                "group": "Sampling",
                "description": "Draw every Nth column of the grid.",
            },
        },
    )

    #: Read here and never forwarded: the mask reshapes the data before it is
    #: drawn, and the camera is set on the axes afterwards.
    Options: dict[str, object] = merge(
        VIEW_OPTIONS,
        {
            "circular_mask": {
                "default": False,
                "type": bool,
                "group": "Mask",
                "description": "Blank out the grid outside a circular boundary centred on the data - a wafer-shaped surface instead of a rectangular one.",
            },
            "mask_radius": {
                "default": None,
                "type": float,
                "min": 0.0,
                "group": "Mask",
                "description": "Radius of the circular mask, in x/y data units. Defaults to half the shorter grid extent when empty.",
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
        valid_series = [
            sd
            for sd in series
            if (sd.style or {}).get("visible", True)
            and self.ensure_required_roles(sd.df)
            and not sd.df.empty
        ]
        if not valid_series:
            return

        if len(valid_series) > 1:
            applogger.info(
                "Surface Plot renders one series; %d more selected on this "
                "axis were not drawn - two surfaces sharing one set of axes "
                "would occlude each other.",
                len(valid_series) - 1,
            )

        sd = valid_series[0]
        merged = self.merge_style(axis_options, sd.style or {})
        grid = pivot_to_grid(sd.df)
        if grid is None:
            applogger.error(
                "Surface Plot needs a complete grid: every x value paired "
                "with every y value exactly once. Use 'Surface Plot "
                "(Scattered)' for data that is not on a regular grid.",
                show_dialog=False,
                raise_error=False,
            )
            return

        x_grid, y_grid, z_grid = grid
        if bool(self.opt("circular_mask", merged)):
            z_grid = self._apply_circular_mask(x_grid, y_grid, z_grid, merged)

        kwargs = _surface_kwargs(self, merged)
        ax.plot_surface(x_grid, y_grid, z_grid, **kwargs)

        view = _view_kwargs(axis_options, self)
        if view:
            ax.view_init(**view)

        self.apply_annotations(ax, axis_options)

    def _apply_circular_mask(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
        options: dict[str, Any],
    ) -> np.ndarray:
        center_x = float((x_grid.min() + x_grid.max()) / 2.0)
        center_y = float((y_grid.min() + y_grid.max()) / 2.0)
        radius_option = self.opt("mask_radius", options)
        if radius_option is not None and radius_option != "":
            radius = float(str(radius_option))
        else:
            radius = min(
                (x_grid.max() - x_grid.min()) / 2.0,
                (y_grid.max() - y_grid.min()) / 2.0,
            )

        distance = np.hypot(x_grid - center_x, y_grid - center_y)
        masked = z_grid.copy()
        masked[distance > radius] = np.nan
        return masked



class TriSurfaceAxisRenderer(BaseAxisRenderer):
    """Surface plot over scattered, non-gridded x/y/z points.

    Role columns:
        x, y, z   required, one point per row - any layout, no grid needed.

    Triangulates the x/y points with Matplotlib's own Delaunay triangulation
    and shades each triangular face by z, which is honest about what is
    actually known between measured points in a way that interpolating onto
    a grid first would not be.
    """

    Name: str = "Surface Plot (Scattered)"
    Category: str = "3D and volumetric data"
    Description: str = "3D triangulated surface for scattered (non-gridded) x/y/z data."
    Link: str = "https://matplotlib.org/stable/gallery/mplot3d/trisurf3d.html"

    RequiredRoles: list[str] = ["x", "y", "z"]
    OptionalRoles: list[str] = []

    #: Two surfaces sharing one set of axes occlude each other.
    MaxSeries: int | None = 1

    #: The gridded renderer's, without the two grid-only sampling keywords:
    #: ``plot_trisurf`` walks triangles, not rows and columns.
    Kwargs: dict[str, object] = {
        key: value
        for key, value in SurfaceAxisRenderer.Kwargs.items()
        if key not in ("rstride", "cstride")
    }
    Kwargs["linewidth"] = {
        **dict(Kwargs["linewidth"]),  # pyright: ignore[reportArgumentType]
        # Not 0.0 here: the triangles of a scattered surface are the reading,
        # and a faint mesh is what shows where the data actually was.
        "default": 0.2,
        "description": "Triangle outline width.",
    }
    Kwargs["edgecolor"] = {
        **dict(Kwargs["edgecolor"]),  # pyright: ignore[reportArgumentType]
        "description": "Triangle outline colour. Leave empty for no visible mesh.",
    }

    #: No mask: it blanks cells of a grid, and there is no grid here.
    Options: dict[str, object] = dict(VIEW_OPTIONS)

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        valid_series = [
            sd
            for sd in series
            if (sd.style or {}).get("visible", True)
            and self.ensure_required_roles(sd.df)
            and not sd.df.empty
        ]
        if not valid_series:
            return

        if len(valid_series) > 1:
            applogger.info(
                "Surface Plot (Scattered) renders one series; %d more "
                "selected on this axis were not drawn.",
                len(valid_series) - 1,
            )

        sd = valid_series[0]
        merged = self.merge_style(axis_options, sd.style or {})
        x, y, z = finite_xyz(sd.df)
        if x.size < 3:
            applogger.error(
                "Surface Plot (Scattered) needs at least 3 points to "
                "triangulate.",
                show_dialog=False,
                raise_error=False,
            )
            return

        kwargs = _surface_kwargs(self, merged)
        ax.plot_trisurf(x, y, z, **kwargs)

        view = _view_kwargs(axis_options, self)
        if view:
            ax.view_init(**view)

        self.apply_annotations(ax, axis_options)


class Scatter3DAxisRenderer(BaseAxisRenderer):
    """Points in three dimensions, one mark per row.

    Role columns:
        x, y, z          required, one point per row.
        color, size      optional, read exactly as the 2D scatter reads them
                         - a numeric column mapped through the colormap, and
                         a numeric column as marker areas.

    Deliberately not a subclass of ``ScatterAxisRenderer``, even though the
    two draw the same thing. That renderer's ``render_axis`` reads two role
    columns, resolves a line style, sorts by x, fits trend lines and draws
    confidence ellipses - all of it two-dimensional, and none of it
    meaningful on a 3D axes, where "sorted by x" is not a property of
    anything and an ellipse has no plane to lie in. What is shared is the
    option vocabulary, which comes from ``kwarg_spec`` rather than from the
    other renderer.

    Like both surfaces, this asks for a 3D axes through
    ``options["projection"] = "3d"``, which create_chart_dialog sets when the
    chart type is created (CHART_TYPES_NEEDING_3D_AXES) and
    render_figure._subplot_kwargs_for_axis reads for every renderer.
    """

    Name: str = "Scatter Plot (3D)"
    Category: str = "3D and volumetric data"
    Description: str = (
        "Points in three dimensions, optionally coloured and sized by further "
        "columns - a cloud read for its shape rather than a surface."
    )
    Link: str = "https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html"

    RequiredRoles: list[str] = ["x", "y", "z"]
    OptionalRoles: list[str] = ["color", "size"]

    Kwargs: dict[str, object] = merge(
        pick(ARTIST_KWARGS, "alpha", "label", "zorder", "visible", "rasterized"),
        pick(CMAP_KWARGS, "cmap", "norm", "vmin", "vmax"),
        {
            "marker": {
                "default": "o",
                "type": str,
                "group": "Appearance",
                "description": "Marker shape, as a Matplotlib marker code.",
            },
            "s": {
                "default": None,
                "type": float,
                "min": 0.0,
                "max": 1000.0,
                "step": 5.0,
                "group": "Appearance",
                "description": (
                    "Marker area in points squared, for every point. Ignored "
                    "when a size role column is mapped."
                ),
            },
            "color": {
                "default": None,
                "type": str,
                "kind": "color",
                "group": "Appearance",
                "description": (
                    "One colour for every point. Ignored when a color role "
                    "column is mapped, which colours them individually."
                ),
            },
            "depthshade": {
                "default": True,
                "type": bool,
                "group": "Appearance",
                "description": (
                    "Fade the points that are further away. It is the main "
                    "depth cue a still 3D image has."
                ),
            },
            "edgecolor": {
                "default": None,
                "type": str,
                "kind": "color",
                "group": "Appearance",
                "description": "Marker outline colour.",
            },
        },
    )

    #: The camera, read here and applied to the axes rather than forwarded.
    Options: dict[str, object] = dict(VIEW_OPTIONS)

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

            x, y, z = finite_xyz(sd.df)
            if x.size == 0:
                applogger.info("Series '%s' skipped: no finite x/y/z rows.", sd.name)
                continue

            merged = self.merge_style(axis_options, style)
            kwargs = {
                key: value
                for key, value in self.get_kwargs(merged).items()
                if value is not None and value != ""
            }
            kwargs.setdefault("label", style.get("label") or sd.name)

            colors = self._aligned_column(sd, "color", x.size)
            if colors is not None:
                # c, not color: one is an array mapped through the colormap,
                # the other is a single colour, and passing both is refused.
                kwargs.pop("color", None)
                kwargs["c"] = colors
                kwargs.setdefault("cmap", rcParams["image.cmap"])
            else:
                for key in ("cmap", "norm", "vmin", "vmax"):
                    kwargs.pop(key, None)
                kwargs.setdefault("color", self.series_color(style, index))

            sizes = self._aligned_column(sd, "size", x.size)
            if sizes is not None:
                kwargs.pop("s", None)
                kwargs["s"] = sizes

            try:
                ax.scatter(x, y, z, **kwargs)
            except Exception:
                applogger.exception(
                    "Scatter Plot (3D) failed to draw series '%s'.", sd.name
                )

        view = _view_kwargs(axis_options, self)
        if view:
            ax.view_init(**view)

        self.apply_annotations(ax, axis_options)

    def _aligned_column(
        self, sd: SeriesData, role: str, expected: int
    ) -> np.ndarray | None:
        """Return one optional numeric role, aligned to the finite x/y/z rows.

        None when the column is absent, or when dropping the non-finite rows
        has left it a different length than the points: colouring or sizing
        the wrong points is worse than doing neither.
        """
        if role not in sd.df.columns:
            return None

        frame = sd.df
        finite = np.isfinite(
            np.column_stack(
                [
                    pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
                    for name in ("x", "y", "z")
                ]
            )
        ).all(axis=1)
        values = pd.to_numeric(frame[role], errors="coerce").to_numpy(dtype=float)
        values = values[finite]
        if values.size != expected or not np.isfinite(values).all():
            applogger.info(
                "Series '%s': the %s column does not line up with the points, "
                "so it is ignored.",
                sd.name,
                role,
            )
            return None
        return values
