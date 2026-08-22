"""Pie chart renderer.

A pie is the one chart in this application that cannot show several series at
once: the wedges of a pie must sum to the whole, so two overlaid pies would
either hide one another or stop meaning anything.  The renderer therefore draws
the **first visible series** and logs the others rather than silently drawing a
subset - a chart that quietly ignores half its data is worse than one that says
so.

Roles:
    value    required numeric size of each wedge; negatives are dropped
    label    optional wedge label, otherwise the value's position is used
    explode  optional radial offset per wedge, in fractions of the radius

``wedge_width`` below 1 turns the pie into a donut, which is the same data with
a hole - useful because the eye compares arc length better than area.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from app.charts.base import BaseAxisRenderer, SeriesData
from app.logs.logger import applogger

# Where the percentage labels are placed, as a fraction of the radius.
_DEFAULT_PCT_DISTANCE: float = 0.6
_DEFAULT_LABEL_DISTANCE: float = 1.1


class PieAxisRenderer(BaseAxisRenderer):
    """Renderer for pie and donut charts.

    Honors per-series style fields: ``label``, ``visible``, ``color``.
    """

    RequiredRoles: list[str] = ["value"]
    OptionalRoles: list[str] = ["label", "explode"]

    Name: str = "Pie Chart"
    Category: str = "Statistical distributions"
    Description: str = "Pie or donut chart of one series."
    Link: str = (
        "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pie.html"
    )

    Kwargs: dict[str, object] = {
        "autopct": {
            "default": "%1.1f%%",
            "type": str,
            "group": "Labels",
            "description": (
                "Format for the percentage drawn inside each wedge, e.g. "
                "'%1.1f%%'.  Leave empty to draw no percentages."
            ),
        },
        "startangle": {
            "default": 90.0,
            "type": float,
            "min": -360.0,
            "max": 360.0,
            "group": "Layout",
            "description": (
                "Angle of the first wedge, counter-clockwise from the x axis. "
                "90 puts it at the top, which is where a reader starts."
            ),
        },
        "counterclock": {
            "default": True,
            "type": bool,
            "group": "Layout",
            "description": "Lay the wedges out counter-clockwise.",
        },
        "wedge_width": {
            "default": 1.0,
            "type": float,
            "min": 0.05,
            "max": 1.0,
            "group": "Layout",
            "description": (
                "Wedge thickness as a fraction of the radius. Below 1 the pie "
                "becomes a donut."
            ),
        },
        "radius": {
            "default": 1.0,
            "type": float,
            "min": 0.1,
            "max": 3.0,
            "group": "Layout",
            "description": "Pie radius in axes units.",
        },
        "pctdistance": {
            "default": _DEFAULT_PCT_DISTANCE,
            "type": float,
            "min": 0.0,
            "max": 2.0,
            "group": "Labels",
            "description": "Distance of the percentage text from the centre.",
        },
        "labeldistance": {
            "default": _DEFAULT_LABEL_DISTANCE,
            "type": float,
            "min": 0.0,
            "max": 3.0,
            "group": "Labels",
            "description": "Distance of the wedge labels from the centre.",
        },
        "shadow": {
            "default": False,
            "type": bool,
            "group": "Appearance",
            "description": "Draw a shadow under the pie.",
        },
        "normalize": {
            "default": True,
            "type": bool,
            "group": "Layout",
            "description": (
                "Scale the values so they fill a full circle. Switch off to "
                "draw a partial pie when the values are already fractions of a "
                "whole that is not all present."
            ),
        },
        "colormap": {
            "default": None,
            "type": str,
            "group": "Appearance",
            "description": (
                "Matplotlib colormap used to colour the wedges, e.g. 'tab20'. "
                "Empty uses the current style's colour cycle."
            ),
        },
        "edge_color": {
            "default": "white",
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": (
                "Colour of the line between wedges. A light edge is what keeps "
                "adjacent wedges of similar colour readable."
            ),
        },
        "edge_width": {
            "default": 1.0,
            "type": float,
            "min": 0.0,
            "max": 10.0,
            "group": "Appearance",
            "description": "Width of the line between wedges.",
        },
        "show_legend": {
            "default": False,
            "type": bool,
            "group": "Appearance",
            "description": (
                "Show a legend instead of relying on the labels around the "
                "pie. Useful when the labels are long."
            ),
        },
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        """Draw the first visible series as a pie."""
        drawable = [
            sd
            for sd in series
            if (sd.style or {}).get("visible", True)
            and self.ensure_required_roles(sd.df)
            and not sd.df.empty
        ]
        if not drawable:
            return

        if len(drawable) > 1:
            applogger.warning(
                "Pie chart: %d series on this axis, only '%s' is drawn. A pie's "
                "wedges have to sum to one whole, so put the others on their "
                "own axis.",
                len(drawable),
                drawable[0].name,
                show_dialog=False,
                raise_error=False,
            )

        sd = drawable[0]
        values, labels, explode = self._wedges(sd)
        if values.size == 0:
            applogger.info("Series '%s' skipped: no positive values.", sd.name)
            return

        pie_kwargs = self._pie_kwargs(options, wedge_count=values.size)
        if labels is not None:
            pie_kwargs["labels"] = labels
        if explode is not None:
            pie_kwargs["explode"] = explode

        result = ax.pie(values, **pie_kwargs)
        wedges = result[0]

        # A pie drawn on non-square axes is an ellipse, and an ellipse cannot
        # be read: equal aspect is part of the chart, not a preference.
        ax.set_aspect("equal")

        if self.opt("show_legend", options) and labels is not None:
            ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
        # Draw descriptor annotations after renderer-owned artists.
        self.apply_annotations(ax, options or {})

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _wedges(
        self,
        sd: SeriesData,
    ) -> tuple[np.ndarray, list[str] | None, np.ndarray | None]:
        """Return the values, labels and explode offsets for one series.

        Non-finite and negative values are dropped: a negative wedge has no
        meaning in a pie, and Matplotlib would draw it as if it were positive.
        """
        values = pd.to_numeric(sd.df["value"], errors="coerce")
        keep = values.notna() & (values > 0)

        dropped = int((~keep).sum())
        if dropped:
            applogger.info(
                "Pie chart: %d non-positive or missing value(s) dropped from '%s'.",
                dropped,
                sd.name,
            )

        wedge_values = values[keep].to_numpy(dtype=float)

        labels: list[str] | None = None
        if "label" in sd.df.columns:
            labels = [str(value) for value in sd.df.loc[keep, "label"].tolist()]

        explode: np.ndarray | None = None
        if "explode" in sd.df.columns:
            explode = (
                pd.to_numeric(sd.df.loc[keep, "explode"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

        return wedge_values, labels, explode

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------
    def _float_option(self, name: str, options: dict[str, Any], default: float = 0.0) -> float:
        """Return an option as float without Pylance object-conversion warnings."""
        value = cast(Any, self.opt(name, options))
        if value in (None, ""):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            applogger.warning(
                "Invalid pie option %s=%r; using %r",
                name,
                value,
                default,
                show_dialog=False,
                raise_error=False,
            )
            return default

    def _pie_kwargs(self, options: dict, *, wedge_count: int) -> dict[str, Any]:
        """Translate this renderer's options into ``ax.pie`` keywords."""
        kwargs: dict[str, Any] = {
            "startangle": self._float_option("startangle", options, 0.0),
            "counterclock": bool(self.opt("counterclock", options)),
            "radius": self._float_option("radius", options, 1.0),
            "pctdistance": self._float_option("pctdistance", options, _DEFAULT_PCT_DISTANCE),
            "labeldistance": self._float_option("labeldistance", options, _DEFAULT_LABEL_DISTANCE),
            "shadow": bool(self.opt("shadow", options)),
            "normalize": bool(self.opt("normalize", options)),
        }

        autopct = str(self.opt("autopct", options) or "").strip()
        if autopct:
            kwargs["autopct"] = autopct

        # A donut is a pie whose wedges are drawn as annuli; anything below 1
        # has to reach Matplotlib through wedgeprops, not through a keyword.
        wedge_props: dict[str, Any] = {}
        width = self._float_option("wedge_width", options, 1.0)
        if 0.0 < width < 1.0:
            wedge_props["width"] = width * kwargs["radius"]

        edge_color = str(self.opt("edge_color", options) or "").strip()
        if edge_color:
            wedge_props["edgecolor"] = edge_color
            wedge_props["linewidth"] = self._float_option("edge_width", options, 0.0)
        if wedge_props:
            kwargs["wedgeprops"] = wedge_props

        colormap = str(self.opt("colormap", options) or "").strip()
        if colormap:
            kwargs["colors"] = self._colormap_colors(colormap, wedge_count)

        return kwargs

    @staticmethod
    def _colormap_colors(name: str, count: int) -> Any:
        """Sample *count* evenly spaced colours from a named colormap."""
        from matplotlib import colormaps

        try:
            colormap = colormaps[name]
        except KeyError:
            applogger.warning(
                "Unknown colormap %r; using the style's colour cycle.",
                name,
                show_dialog=False,
                raise_error=False,
            )
            return None

        if count <= 1:
            return colormap([0.5])
        return colormap(np.linspace(0.0, 1.0, count))
