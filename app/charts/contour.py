"""2D contour renderers, for data on a regular grid and for scattered points.

The same split the surface renderers are built around - and the same reason
for it - one dimension lower: a contour map is a scalar field drawn on the
x/y plane instead of lifted into a third axis, so the input is the same x/y/z
rows and the decision about their shape is the same decision.

``ContourAxisRenderer`` pivots x/y/z rows into a regular grid and draws
``Axes.contourf`` / ``Axes.contour``.  ``ContourScatteredAxisRenderer`` skips
the pivot and triangulates the raw points with ``Axes.tricontourf`` /
``Axes.tricontour``, which is what non-gridded x/y/z needs.  Interpolating
scattered points onto a grid first was the other option and is not taken:
the triangulation is honest about where there is no data, since it simply
does not cover the hull's outside, while a grid fills those cells with
numbers that look exactly like measurements.

Both are 2D: unlike the surface renderers they need no ``projection`` option
and nothing about axis creation changes for them.

  https://matplotlib.org/stable/plot_types/arrays/contourf.html
  https://matplotlib.org/stable/plot_types/unstructured/tricontourf.html
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.charts.base import BaseAxisRenderer, SeriesData
from app.charts.grids import finite_xyz, pivot_to_grid
from app.logs.logger import applogger

#: Options the renderer consumes itself.  Everything else in ``Kwargs`` is
#: forwarded to the Matplotlib contour call, so anything read here has to be
#: named here or it reaches ``contourf`` as an unknown keyword.
_RENDERER_ONLY_KWARGS: tuple[str, ...] = (
    "levels",
    "filled",
    "line_overlay",
    "line_color",
    "linewidths",
    "linestyles",
    "label_lines",
    "label_format",
    "label_fontsize",
    "label_inline",
    "colorbar",
    "colorbar_label",
)

#: Line colour used for the contour lines drawn over filled bands when none
#: was chosen.  Matplotlib would colour them from the same colormap as the
#: bands underneath, which makes them all but invisible - the one case where
#: the default has to differ from Matplotlib's own.
_OVERLAY_LINE_COLOR = "black"


class ContourAxisRenderer(BaseAxisRenderer):
    """Contour map over a regular x/y grid.

    Role columns:
        x, y, z   required.  Every (x, y) combination present must appear
                  exactly once - a complete Cartesian product of the distinct
                  x and y values, at any spacing.  A pair missing from that
                  product is a cell with no measurement in it; the renderer
                  draws nothing and says so rather than inventing one, and
                  points at the scattered variant, which needs no grid.
    """

    Name: str = "Contour Plot"
    Category: str = "Gridded data"
    Description: str = "Filled or line contours of z over a regular x/y grid."
    Link: str = "https://matplotlib.org/stable/plot_types/arrays/contourf.html"

    RequiredRoles: list[str] = ["x", "y", "z"]
    OptionalRoles: list[str] = []

    Kwargs: dict[str, object] = {
        "levels": {
            "default": "",
            "type": str,
            "group": "Levels",
            "description": (
                "How the value range is cut up. A single number asks for that "
                "many levels ('12'); a list sets them explicitly "
                "('0, 5, 10, 20'). Empty lets Matplotlib choose."
            ),
        },
        "extend": {
            "default": "neither",
            "type": ["neither", "both", "min", "max"],
            "group": "Levels",
            "description": (
                "What to do with values outside the levels. 'neither' leaves "
                "them uncoloured; the others colour them with the end band "
                "and put an arrow on that end of the colorbar."
            ),
        },
        "filled": {
            "default": True,
            "type": bool,
            "group": "Appearance",
            "description": (
                "Fill the bands between levels (contourf). Off draws contour "
                "lines only, over the axes background."
            ),
        },
        "cmap": {
            "default": "viridis",
            "type": str,
            "group": "Appearance",
            "description": "Colormap the value is mapped through, e.g. 'viridis' or 'coolwarm'.",
        },
        "alpha": {
            "default": None,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "group": "Appearance",
            "description": "Opacity of the bands.",
        },
        "colors": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": (
                "One colour for every level, instead of a colormap. Setting "
                "it overrides Colormap - Matplotlib accepts one or the other, "
                "never both."
            ),
        },
        "line_overlay": {
            "default": True,
            "type": bool,
            "group": "Lines",
            "description": (
                "Draw the contour lines on top of the filled bands. Ignored "
                "when Filled is off, where the lines are the whole plot."
            ),
        },
        "line_color": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Lines",
            "description": (
                "Contour line colour. Empty means black over filled bands, "
                "where a colormapped line would vanish into the band it sits "
                "on, and the colormap when nothing is filled."
            ),
        },
        "linewidths": {
            "default": 0.5,
            "type": float,
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,
            "group": "Lines",
            "description": "Contour line width.",
        },
        "linestyles": {
            "default": "solid",
            "type": ["solid", "dashed", "dashdot", "dotted"],
            "group": "Lines",
            "description": "Contour line style.",
        },
        "label_lines": {
            "default": False,
            "type": bool,
            "group": "Labels",
            "description": (
                "Write each contour's value onto its line. Needs lines: with "
                "Filled on, that means Line overlay on too."
            ),
        },
        "label_format": {
            "default": "%1.3g",
            "type": str,
            "group": "Labels",
            "description": "printf-style format for the contour labels, e.g. '%1.1f' or '%d'.",
        },
        "label_fontsize": {
            "default": 8.0,
            "type": float,
            "min": 1.0,
            "max": 72.0,
            "step": 0.5,
            "group": "Labels",
            "description": "Contour label font size, in points.",
        },
        "label_inline": {
            "default": True,
            "type": bool,
            "group": "Labels",
            "description": "Break the line where its label sits instead of writing over it.",
        },
        "colorbar": {
            "default": False,
            "type": bool,
            "group": "Colorbar",
            "description": (
                "Add a colorbar for the levels. It is a second Axes taken out "
                "of this one's space, so a constrained or compressed figure "
                "layout places it best."
            ),
        },
        "colorbar_label": {
            "default": None,
            "type": str,
            "group": "Colorbar",
            "description": "Label written alongside the colorbar. Empty leaves it unlabelled.",
        },
    }

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        sd = self._single_series(series)
        if sd is None:
            return

        grid = pivot_to_grid(sd.df)
        if grid is None:
            applogger.error(
                "Contour Plot needs a complete grid: every x value paired "
                "with every y value exactly once. Use 'Contour Plot "
                "(Scattered)' for data that is not on a regular grid.",
                show_dialog=False,
                raise_error=False,
            )
            return

        x_grid, y_grid, z_grid = grid
        # A grid can be complete and still carry an infinity - pivot_to_grid
        # only refuses cells that are *missing*.  Matplotlib spreads one
        # infinite value across every level it computes, so it is turned into
        # the hole it effectively is; contour simply leaves those cells blank.
        z_grid = np.where(np.isfinite(z_grid), z_grid, np.nan)

        self._draw_contours(
            ax,
            (x_grid, y_grid, z_grid),
            self._merge_options(axis_options, sd.style or {}),
        )
        self.apply_annotations(ax, axis_options)

    def _contour_functions(self, ax: Any) -> tuple[Any, Any]:
        """Return the (filled, line) contour calls this renderer draws with."""
        return ax.contourf, ax.contour

    def _draw_contours(
        self,
        ax: Any,
        data: tuple[Any, ...],
        options: dict[str, Any],
    ) -> None:
        """Draw the bands, the lines and the labels that were asked for.

        Shared with the scattered renderer: only *data* and the two functions
        from :meth:`_contour_functions` differ between the two, which is the
        whole difference between a grid and a triangulation once the shape of
        the input has been settled.
        """
        filled_contour, line_contour = self._contour_functions(ax)
        base = self._contour_kwargs(options)
        levels = self._levels(options)
        filled = bool(self.opt("filled", options))

        band_set = None
        if filled:
            band_set = filled_contour(*data, **self._with_levels(base, levels))

        line_set = None
        if not filled or bool(self.opt("line_overlay", options)):
            line_set = line_contour(
                *data,
                **self._line_kwargs(base, options, filled=filled),
                **self._with_levels({}, levels),
            )

        if bool(self.opt("label_lines", options)) and line_set is not None:
            self._label(ax, line_set, options)

        if bool(self.opt("colorbar", options)):
            self._colorbar(ax, band_set if band_set is not None else line_set, options)

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------
    def _contour_kwargs(self, options: dict[str, Any]) -> dict[str, Any]:
        """Keyword arguments common to the filled and the line call."""
        kwargs = self.get_kwargs(options)
        for key in _RENDERER_ONLY_KWARGS:
            kwargs.pop(key, None)
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if value is not None and value != ""
        }

        # Matplotlib refuses both at once, so an explicit colour is taken as
        # the more specific intent and the colormap steps aside.
        colors = self.opt("colors", options)
        if colors is not None and str(colors).strip() != "":
            kwargs["colors"] = colors
            kwargs.pop("cmap", None)

        return kwargs

    def _line_kwargs(
        self,
        base: dict[str, Any],
        options: dict[str, Any],
        *,
        filled: bool,
    ) -> dict[str, Any]:
        """The band keywords, adjusted for drawing lines rather than areas."""
        kwargs = dict(base)
        for name in ("linewidths", "linestyles"):
            value = self.opt(name, options)
            if value is not None and value != "":
                kwargs[name] = value

        line_color = self.opt("line_color", options)
        if line_color is not None and str(line_color).strip() != "":
            kwargs["colors"] = line_color
            kwargs.pop("cmap", None)
        elif filled and "colors" not in kwargs:
            kwargs["colors"] = _OVERLAY_LINE_COLOR
            kwargs.pop("cmap", None)

        # The bands underneath already carry the transparency; repeating it on
        # the lines only makes the separators harder to see.
        if filled:
            kwargs.pop("alpha", None)
        return kwargs

    def _with_levels(self, kwargs: dict[str, Any], levels: Any) -> dict[str, Any]:
        merged = dict(kwargs)
        if levels is not None:
            merged["levels"] = levels
        return merged

    def _levels(self, options: dict[str, Any]) -> int | list[float] | None:
        """Parse the levels option into what Matplotlib expects, or None.

        A single number is a *count* of levels and an int is what says so - 12
        asks for twelve bands, while [12.0] would ask for one contour at the
        value 12.  Anything unparseable is dropped with a log entry rather
        than raised: a typo in one option should cost the option, not the
        chart.
        """
        raw = self.opt("levels", options)
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            values = [str(item) for item in raw]
        else:
            text = str(raw).strip()
            if text == "":
                return None
            values = [part for part in text.replace(";", ",").split(",") if part.strip()]

        try:
            numbers = [float(part) for part in values]
        except (TypeError, ValueError):
            applogger.info(
                "Contour levels %r are neither a count nor a list of numbers; "
                "letting Matplotlib choose the levels instead.",
                raw,
            )
            return None

        if not numbers:
            return None
        if len(numbers) == 1:
            count = int(numbers[0])
            return count if count > 0 else None
        return sorted(numbers)

    # ------------------------------------------------------------------
    # Decoration
    # ------------------------------------------------------------------
    def _label(self, ax: Any, line_set: Any, options: dict[str, Any]) -> None:
        fmt = str(self.opt("label_format", options) or "%1.3g")
        try:
            ax.clabel(
                line_set,
                inline=bool(self.opt("label_inline", options)),
                fmt=fmt,
                fontsize=float(str(self.opt("label_fontsize", options) or 8.0)),
            )
        except (TypeError, ValueError):
            # A bad printf format is the likely cause, and it is not worth
            # losing the contours over.
            applogger.info(
                "Contour labels were skipped: %r is not a usable label format.",
                fmt,
            )

    def _colorbar(self, ax: Any, mappable: Any, options: dict[str, Any]) -> None:
        """Add a colorbar for *mappable* beside *ax*.

        ``use_gridspec=False`` is not cosmetic.  Renderers draw while the
        figure's layout engine is still "none" - render_figure only applies
        the descriptor's layout mode once every axis has been drawn - and the
        gridspec path Matplotlib takes by default in that state builds a
        ``GridSpecFromSubplotSpec`` with zero-height padding rows.  A
        constrained or compressed engine applied afterwards then divides by
        that zero and the whole figure fails to draw.  ``make_axes`` takes the
        space out of the parent axes instead and owns no gridspec, so the
        colorbar survives whichever engine is set after it.
        """
        figure = getattr(ax, "figure", None)
        if mappable is None or figure is None:
            return
        colorbar = figure.colorbar(mappable, ax=ax, use_gridspec=False)
        label = self.opt("colorbar_label", options)
        if label is not None and str(label).strip() != "":
            colorbar.set_label(str(label))

    # ------------------------------------------------------------------
    # Series and option plumbing
    # ------------------------------------------------------------------
    def _single_series(self, series: list[SeriesData]) -> SeriesData | None:
        """Return the one series to draw, or None when there is nothing to draw.

        A contour map is one scalar field over the plot area: a second one
        drawn on the same axes covers the first rather than being compared
        with it, so the extras are reported and skipped instead.
        """
        valid = [
            sd
            for sd in series
            if (sd.style or {}).get("visible", True)
            and self.ensure_required_roles(sd.df)
            and not sd.df.empty
        ]
        if not valid:
            return None

        if len(valid) > 1:
            applogger.info(
                "%s renders one series; %d more selected on this axis were "
                "not drawn - the second field would cover the first.",
                self.Name,
                len(valid) - 1,
            )
        return valid[0]

    def _merge_options(
        self,
        axis_options: dict[str, Any],
        style: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(axis_options or {})
        axis_kwargs = dict(merged.get("axis_kwargs", {}) or {})
        axis_kwargs.update(style.get("axis_kwargs", {}) or {})
        for key, value in style.items():
            if key != "axis_kwargs":
                merged[key] = value
        merged["axis_kwargs"] = axis_kwargs
        return merged


class ContourScatteredAxisRenderer(ContourAxisRenderer, BaseAxisRenderer):
    """Contour map over scattered, non-gridded x/y/z points.

    Role columns:
        x, y, z   required, one point per row - any layout, no grid needed.

    Matplotlib's own Delaunay triangulation joins the points and the contours
    are traced across the triangles, so the map covers exactly the convex hull
    of what was actually measured and stops there.  That is the difference
    from interpolating onto a grid first: the grid would extend the map into
    the corners as well, with values nothing supports.
    """

    Name: str = "Contour Plot (Scattered)"
    Category: str = "Irregularly gridded data"
    Description: str = "Filled or line contours for scattered (non-gridded) x/y/z data."
    Link: str = "https://matplotlib.org/stable/plot_types/unstructured/tricontourf.html"

    RequiredRoles: list[str] = ["x", "y", "z"]
    OptionalRoles: list[str] = []

    #: The options are the gridded renderer's, unchanged: everything from the
    #: levels to the labels means the same thing on a triangulation.
    Kwargs: dict[str, object] = dict(ContourAxisRenderer.Kwargs)

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        sd = self._single_series(series)
        if sd is None:
            return

        x, y, z = finite_xyz(sd.df)
        # Three points are one triangle, which is the smallest thing a
        # contour can cross.
        if x.size < 3:
            applogger.error(
                "Contour Plot (Scattered) needs at least 3 points to "
                "triangulate.",
                show_dialog=False,
                raise_error=False,
            )
            return

        try:
            self._draw_contours(
                ax,
                (x, y, z),
                self._merge_options(axis_options, sd.style or {}),
            )
        except (RuntimeError, ValueError) as exc:
            # Collinear points have no triangles to contour across, and
            # Matplotlib says so by raising.  That is a property of the data,
            # not a bug, so it is reported the way an unusable frame is.
            applogger.error(
                "Contour Plot (Scattered) could not triangulate this series "
                "(%s: %s). Points that all lie on one line enclose no area "
                "for a contour to cross.",
                type(exc).__name__,
                exc,
                show_dialog=False,
                raise_error=False,
            )
            return

        self.apply_annotations(ax, axis_options)

    def _contour_functions(self, ax: Any) -> tuple[Any, Any]:
        return ax.tricontourf, ax.tricontour
