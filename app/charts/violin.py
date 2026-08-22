"""Violin plot renderer.

A violin answers the same question as a box plot - how is this sample
distributed - but shows the estimated density instead of five summary numbers.
It therefore reuses the box renderer's grouping and layout wholesale: same
``value``/``group`` roles, same category-then-series arrangement, same widths.
Swapping a Box Plot axis to a Violin Plot axis keeps every item in exactly the
same place.

What is not shared is the drawing: ``ax.violinplot`` takes one call per body
and returns a dict of collections rather than the box artists, so the styling
and legend code is its own.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.patches import Patch

from app.charts.base import BaseAxisRenderer, SeriesData
from app.charts.box import BoxAxisRenderer
from app.logs.logger import applogger

# The three marker collections violinplot can return, and the option that asks
# for each.  Keeping them in one table is what makes the styling loop short.
_MARKER_PARTS: tuple[tuple[str, str], ...] = (
    ("cmeans", "showmeans"),
    ("cmedians", "showmedians"),
    ("cbars", "showextrema"),
    ("cmins", "showextrema"),
    ("cmaxes", "showextrema"),
)


class ViolinAxisRenderer(BoxAxisRenderer, BaseAxisRenderer):
    """Renderer for violin plots.

    BaseAxisRenderer is listed explicitly even though BoxAxisRenderer already
    provides it: the renderer scanner matches ``class X(BaseAxisRenderer)``
    statically by base name, so a renderer that only inherits it indirectly is
    never discovered.  HorizontalBarAxisRenderer does the same.

    Role columns:
        value    required numeric sample values
        group    optional category key; each distinct value becomes a category
        color    optional per-series colour

    Honors per-series style fields: label, visible, show_in_legend, color,
    alpha, zorder, linewidth.
    """

    RequiredRoles = ["value"]
    OptionalRoles = ["group", "color"]

    Name = "Violin Plot"
    Category: str = "Statistical distributions"
    Description = "Show the estimated distribution of one or more samples."
    Link: str = (
        "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.violinplot.html"
    )

    Kwargs: dict[str, object] = {
        "direction": {
            "default": "vertical",
            "type": ["vertical", "horizontal"],
            "group": "Layout",
            "description": "Draw the violins vertically or horizontally.",
        },
        "widths": {
            "default": 0.5,
            "type": float,
            "min": 0.05,
            "max": 1.0,
            "group": "Layout",
            "description": "Width of each violin as a fraction of the category spacing.",
        },
        "showmeans": {
            "default": False,
            "type": bool,
            "group": "Markers",
            "description": "Draw a marker at the mean of each sample.",
        },
        "showmedians": {
            "default": True,
            "type": bool,
            "group": "Markers",
            "description": "Draw a marker at the median of each sample.",
        },
        "showextrema": {
            "default": True,
            "type": bool,
            "group": "Markers",
            "description": "Draw the min/max whisker and its caps.",
        },
        "bw_method": {
            "default": None,
            "type": float,
            "min": 0.01,
            "max": 5.0,
            "group": "Density",
            "description": (
                "Bandwidth of the kernel density estimate. Smaller values "
                "follow the data more closely; leave empty for Scott's rule."
            ),
        },
        "points": {
            "default": 100,
            "type": int,
            "min": 10,
            "max": 5000,
            "group": "Density",
            "description": "Number of points at which the density is evaluated.",
        },
        "quantiles": {
            "default": "",
            "type": str,
            "group": "Markers",
            "description": (
                "Comma-separated quantiles to mark on every violin, "
                'e.g. "0.25, 0.5, 0.75". Leave empty for none.'
            ),
        },
        "label": {
            "default": None,
            "type": str,
            "description": "Legend label override. If empty, the series name is used.",
        },
        "alpha": {
            "default": 0.7,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "description": "Body opacity from 0.0 (transparent) to 1.0 (opaque).",
        },
        "linewidth": {
            "default": None,
            "type": float,
            "min": 0.0,
            "max": 10.0,
            "description": "Outline width of the violin body and markers.",
        },
        "zorder": {
            "default": None,
            "type": float,
            "min": -1000.0,
            "max": 1000.0,
            "description": "Drawing order; higher values are drawn on top.",
        },
    }

    # ------------------------------------------------------------------
    # Option parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_quantiles(raw: object) -> list[float] | None:
        """Return the quantile list, or None when nothing usable was given.

        Values outside (0, 1) are dropped rather than passed on: Matplotlib
        raises on them, which would lose the whole plot over one typo.
        """
        text = str(raw or "").strip()
        if not text:
            return None

        values: list[float] = []
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                value = float(part)
            except ValueError:
                applogger.warning(
                    "Ignoring invalid violin quantile %r.",
                    part,
                    show_dialog=False,
                    raise_error=False,
                )
                continue
            if 0.0 < value < 1.0:
                values.append(value)
            else:
                applogger.warning(
                    "Violin quantile %s is outside (0, 1) and was ignored.",
                    value,
                    show_dialog=False,
                    raise_error=False,
                )

        return sorted(set(values)) or None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        """Render every violin, one call per body so each can be styled."""
        base_kwargs = self.get_kwargs(options)
        vert = self.is_vertical(base_kwargs, options)
        requested_width = self._coerce_positive_float(base_kwargs.get("widths", 0.5), 0.5)

        grouped = self.collect_grouped_items(series, options)
        plot_items = grouped["items"]
        category_labels = grouped["category_labels"]
        series_show_in_legend = grouped["show_in_legend"]
        series_colors = grouped["colors"]

        if not plot_items:
            applogger.warning("Violin plot: no data to draw.")
            return

        layout = self.grouped_layout(
            items=plot_items,
            category_labels=category_labels,
            series_labels=grouped["series_labels"],
            requested_width=requested_width,
        )
        category_positions = layout["category_positions"]
        violin_width = layout["item_width"]

        quantiles = self._parse_quantiles(base_kwargs.get("quantiles"))
        legend_handles: list[Patch] = []
        legend_labels: list[str] = []
        legend_seen: set[str] = set()

        for item in plot_items:
            series_label = item["series_label"]
            style = item["style"]
            data = item["data"]

            # A kernel density estimate needs spread; a constant sample has
            # none and makes violinplot raise on a singular covariance.
            if data.size < 2 or float(np.ptp(data)) == 0.0:
                applogger.info(
                    "Violin '%s' skipped: needs at least two distinct values.",
                    item["category_label"],
                )
                continue

            position = self.item_position(layout, item)
            face_color = series_colors[series_label]

            violin_kwargs: dict[str, Any] = {
                "positions": [position],
                "widths": violin_width,
                "orientation": "vertical" if vert else "horizontal",
                "showmeans": bool(base_kwargs.get("showmeans", False)),
                "showmedians": bool(base_kwargs.get("showmedians", True)),
                "showextrema": bool(base_kwargs.get("showextrema", True)),
                "points": max(10, int(base_kwargs.get("points", 100) or 100)),
            }

            bandwidth = base_kwargs.get("bw_method")
            if bandwidth not in (None, ""):
                try:
                    violin_kwargs["bw_method"] = float(bandwidth)
                except (TypeError, ValueError):
                    applogger.warning("Invalid violin bw_method: %r", bandwidth)

            if quantiles:
                violin_kwargs["quantiles"] = [quantiles]

            try:
                result = ax.violinplot([data], **violin_kwargs)
            except Exception:
                applogger.exception(
                    "Failed to draw violin '%s' of series '%s'.",
                    item["category_label"],
                    series_label,
                )
                continue

            self._style_violin(
                result=result,
                face_color=face_color,
                style=style,
                base_kwargs=base_kwargs,
            )

            if (
                series_show_in_legend.get(series_label, True)
                and series_label not in legend_seen
            ):
                legend_handles.append(Patch(facecolor=face_color, label=series_label))
                legend_labels.append(series_label)
                legend_seen.add(series_label)

        self.apply_category_ticks(
            ax,
            category_labels=category_labels,
            category_positions=category_positions,
            vert=vert,
        )

        if legend_handles:
            ax.legend(handles=legend_handles, labels=legend_labels)
        # Draw descriptor annotations after renderer-owned artists.
        self.apply_annotations(ax, options or {})


    def _style_violin(
        self,
        *,
        result: dict[str, Any],
        face_color: Any,
        style: dict[str, Any],
        base_kwargs: dict[str, Any],
    ) -> None:
        """Apply the series colour, alpha, linewidth and zorder to one violin.

        ``violinplot`` returns the bodies under ``bodies`` and every marker as
        a LineCollection under its own key, so the two are styled separately:
        the body is filled, the markers only take the outline colour.
        """
        alpha = style.get("alpha", base_kwargs.get("alpha", 0.7))
        linewidth = style.get("linewidth", style.get("line_width", base_kwargs.get("linewidth")))
        zorder = style.get("zorder", base_kwargs.get("zorder"))

        for body in result.get("bodies", []):
            try:
                body.set_facecolor(face_color)
                body.set_edgecolor(face_color)
                if alpha not in (None, ""):
                    body.set_alpha(float(alpha))
                if linewidth not in (None, ""):
                    body.set_linewidth(float(linewidth))
                if zorder not in (None, ""):
                    body.set_zorder(float(zorder))
            except Exception:
                applogger.exception("Failed to style a violin body")

        for part_name, _option in _MARKER_PARTS:
            collection = result.get(part_name)
            if collection is None:
                continue
            try:
                collection.set_edgecolor(face_color)
                if linewidth not in (None, ""):
                    collection.set_linewidth(float(linewidth))
                if zorder not in (None, ""):
                    collection.set_zorder(float(zorder))
            except Exception:
                applogger.exception("Failed to style violin part %s", part_name)

        quantile_collection = result.get("cquantiles")
        if quantile_collection is not None:
            try:
                quantile_collection.set_edgecolor(face_color)
                quantile_collection.set_linestyle("--")
            except Exception:
                applogger.exception("Failed to style violin quantiles")
