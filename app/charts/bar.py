"""Vertical and horizontal bar renderers.

``HorizontalBarAxisRenderer`` subclasses the vertical one and only flips the
orientation, so the two chart types cannot drift apart.  Categories come from
the optional ``X`` role; without it the row index is used.
"""
from __future__ import annotations

from typing import Any, Literal

import math
import numpy as np
import pandas as pd

from app.charts.base import BaseAxisRenderer, SeriesData


Orientation = Literal["vertical", "horizontal"]


class BarAxisRenderer(BaseAxisRenderer):
    """Renderer for vertical bar charts.

    Role columns:
        Y       required numeric bar value
        X       optional category/position, row index is used when omitted
        color   optional per-bar color sequence
        YError  optional vertical error bar values
        XError  optional horizontal error bar values, used by barh
        Bottom  optional vertical baseline for bar
        Left    optional horizontal baseline for barh

    The Kwargs schema intentionally mirrors the richer ScatterAxisRenderer-style
    metadata pattern: each kwarg exposes at least a default and, where useful,
    type/min/max/kind/group/description metadata for the generic kwargs editor.
    """

    Name: str = "Bar Chart"
    Category: str = "Pairwise data"
    Description: str = "Vertical bar chart"
    Orientation: Orientation = "vertical"
    Link="https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html"

    RequiredRoles: list[str] = ["Y"]
    OptionalRoles: list[str] = ["X", "color", "YError", "XError", "Bottom", "Left"]

    Kwargs: dict[str, object] = {
        # Geometry / placement
        "width": {
            "default": 0.8,
            "type": float,
            "min": 0.0,
            "max": 10.0,
            "step": 0.05,
            "decimals": 4,
            "group": "Geometry",
            "description": "Total category width allocated to each vertical bar group.",
        },
        "height": {
            "default": 0.8,
            "type": float,
            "min": 0.0,
            "max": 10.0,
            "step": 0.05,
            "decimals": 4,
            "group": "Geometry",
            "description": "Total category height allocated to each horizontal bar group.",
        },
        "bottom": {
            "default": None,
            "type": float,
            "group": "Geometry",
            "description": "Scalar baseline for vertical bars when no Bottom role column is present.",
        },
        "left": {
            "default": None,
            "type": float,
            "group": "Geometry",
            "description": "Scalar baseline for horizontal bars when no Left role column is present.",
        },
        "align": {
            "default": "center",
            "type": ["center", "edge"],
            "group": "Geometry",
            "description": "Alignment of bars to category positions.",
        },

        # Colors / patch styling
        "color": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Bar face color. Overridden by the color role column when provided.",
        },
        "facecolor": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Explicit bar face color. Matplotlib gives this precedence over color.",
        },
        "edgecolor": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Patch",
            "description": "Bar edge color.",
        },
        "linewidth": {
            "default": None,
            "type": float,
            "min": 0.0,
            "max": 20.0,
            "step": 0.25,
            "decimals": 3,
            "group": "Patch",
            "description": "Bar edge line width. Use 0 to hide edges.",
        },
        "linestyle": {
            "default": None,
            "type": str,
            "kind": "linestyle",
            "group": "Patch",
            "description": "Bar edge line style.",
        },
        "hatch": {
            "default": None,
            "type": str,
            "group": "Patch",
            "description": "Matplotlib hatch pattern for bar patches.",
        },
        "alpha": {
            "default": 0.9,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "decimals": 3,
            "group": "Appearance",
            "description": "Bar transparency from 0.0 transparent to 1.0 opaque.",
        },
        "fill": {
            "default": True,
            "type": bool,
            "group": "Patch",
            "description": "Whether bar patches are filled.",
        },

        # Labels / error bars
        "label": {
            "default": None,
            "type": str,
            "group": "Text",
            "description": "Legend label override. If empty, the series name is used.",
        },
        "tick_label": {
            "default": None,
            "type": str,
            "group": "Text",
            "description": "Optional Matplotlib tick_label override. Usually leave empty.",
        },
        "xerr": {
            "default": None,
            "type": float,
            "min": 0.0,
            "group": "Error bars",
            "description": "Scalar horizontal error bar when no XError role column is present.",
        },
        "yerr": {
            "default": None,
            "type": float,
            "min": 0.0,
            "group": "Error bars",
            "description": "Scalar vertical error bar when no YError role column is present.",
        },
        "ecolor": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Error bars",
            "description": "Error bar line color.",
        },
        "capsize": {
            "default": 0.0,
            "type": float,
            "min": 0.0,
            "max": 50.0,
            "step": 0.5,
            "decimals": 2,
            "group": "Error bars",
            "description": "Length of error bar caps in points.",
        },
        "error_kw": {
            "default": None,
            "type": str,
            "group": "Error bars",
            "description": "Optional dictionary of keyword arguments passed to Matplotlib errorbar.",
        },
        "log": {
            "default": False,
            "type": bool,
            "group": "Behavior",
            "description": "Use logarithmic scale on the value axis.",
        },

        # Common Artist / Rectangle properties
        "animated": {"default": False, "type": bool, "group": "Behavior", "description": "Enable Matplotlib animation flag."},
        "clip_on": {"default": True, "type": bool, "group": "Behavior", "description": "Clip bars to the axes area."},
        "gid": {"default": None, "type": str, "group": "Behavior", "description": "Matplotlib artist group id."},
        "in_layout": {"default": True, "type": bool, "group": "Behavior", "description": "Include bar artists in layout calculations."},
        "picker": {"default": None, "type": float, "min": 0.0, "group": "Behavior", "description": "Pick radius or picker tolerance."},
        "rasterized": {"default": False, "type": bool, "group": "Behavior", "description": "Rasterize bar artists in vector outputs."},
        "snap": {"default": None, "type": bool, "group": "Behavior", "description": "Snap artist positions to pixels."},
        "url": {"default": None, "type": str, "group": "Text", "description": "URL associated with bar artists in supported backends."},
        "visible": {"default": True, "type": bool, "group": "Behavior", "description": "Show or hide bars."},
        "zorder": {"default": None, "type": float, "step": 1.0, "decimals": 2, "group": "Behavior", "description": "Drawing order of bar artists."},

        # Renderer-level behavior
        "show_legend": {"default": True, "type": bool, "group": "Behavior", "description": "Show the legend when labelled bars are present."},
        "max_tick_labels": {"default": 40, "type": int, "min": 1, "max": 1000, "step": 1, "group": "Text", "description": "Maximum number of category tick labels to display."},
        "tick_label_rotation": {"default": "auto", "type": ["auto", "0", "30", "45", "60", "90"], "group": "Text", "description": "Category label rotation. Auto rotates labels when there are many categories."},
        "tick_label_fontsize": {"default": "auto", "type": ["auto", "6", "7", "8", "9", "10", "11", "12"], "group": "Text", "description": "Category label font size. Auto reduces font when there are many labels."},
    }

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        """Draw every series as one group of bars per category.

        Categories are collected across all series first, so that a series
        missing a category still leaves its slot empty instead of shifting
        every later bar left by one.
        """
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

        categories = self._collect_categories(valid_series)
        category_pos = {value: index for index, value in enumerate(categories)}
        layer_count = max(1, len(valid_series))

        for layer_index, sd in enumerate(valid_series):
            self._render_series(
                ax=ax,
                sd=sd,
                axis_options=axis_options,
                category_pos=category_pos,
                layer_index=layer_index,
                layer_count=layer_count,
            )

        tick_options = self._merge_options(axis_options, valid_series[0].style or {})
        self._apply_category_ticks(ax, categories, tick_options)

        # after the bars/ticks so they sit over the finished chart.

        if self.opt("show_legend", tick_options):
            handles, _labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
        # Draw descriptor annotations after renderer-owned artists.
        self.apply_annotations(ax, options or {})

    def _render_series(
        self,
        *,
        ax: Any,
        sd: SeriesData,
        axis_options: dict[str, Any],
        category_pos: dict[Any, int],
        layer_index: int,
        layer_count: int,
    ) -> None:
        """Draw one series as the *layer_index*-th bar of each group.

        The whole group occupies the configured ``width`` (or ``height`` when
        horizontal) around the integer category position; each layer takes an
        equal share of it and is offset so the group stays centred whatever the
        number of series.

        Error bars come from a data column when the series provides one
        (``XError`` / ``YError``) and fall back to the axis option otherwise,
        which is what lets one axis mix measured and nominal series.
        """
        df = sd.df.copy()
        style = sd.style or {}
        merged_options = self._merge_options(axis_options, style)

        if "X" not in df.columns:
            df["X"] = list(range(len(df)))

        kwargs = self._bar_kwargs(merged_options)
        series_label = self._series_label(
            sd=sd,
            style=style,
            axis_options=axis_options,
        )
        show_in_legend = bool(style.get("show_in_legend", True))
        label = series_label if show_in_legend else "_nolegend_"
        self._apply_series_color(
            kwargs=kwargs,
            df=df,
            style=style,
            layer_index=layer_index,
        )

        if self.Orientation == "horizontal":
            thickness = float(str(self.opt("height", merged_options)))
        else:
            thickness = float(str(self.opt("width", merged_options)))

        bar_thickness = thickness / layer_count
        offset = -thickness / 2.0 + layer_index * bar_thickness + bar_thickness / 2.0
        positions = np.asarray(
            [float(category_pos[value]) for value in df["X"].tolist()],
            dtype=float,
        ) + offset
        values = pd.to_numeric(df["Y"], errors="coerce").to_numpy(dtype=float)

        if self.Orientation == "horizontal":
            left = df["Left"].to_numpy(dtype=float) if "Left" in df.columns else self.opt("left", merged_options)
            xerr = df["XError"].to_numpy(dtype=float) if "XError" in df.columns else self.opt("xerr", merged_options)
            ax.barh(positions, values, height=bar_thickness, left=left, xerr=xerr, label=label, **kwargs)
        else:
            bottom = df["Bottom"].to_numpy(dtype=float) if "Bottom" in df.columns else self.opt("bottom", merged_options)
            yerr = df["YError"].to_numpy(dtype=float) if "YError" in df.columns else self.opt("yerr", merged_options)
            ax.bar(positions, values, width=bar_thickness, bottom=bottom, yerr=yerr, label=label, **kwargs)


    def _series_label(
        self,
        *,
        sd: SeriesData,
        style: dict[str, Any],
        axis_options: dict[str, Any],
    ) -> str:
        """Resolve the legend label with the same rule as BoxAxisRenderer."""
        series_label = str(style.get("label", "") or "").strip()
        if series_label:
            return series_label

        axis_label = str(self.opt("label", axis_options) or "").strip()
        if axis_label:
            return axis_label

        fallback_label = sd.name.strip()
        if fallback_label:
            return fallback_label

        return "Series"

    def _apply_series_color(
        self,
        *,
        kwargs: dict[str, Any],
        df: pd.DataFrame,
        style: dict[str, Any],
        layer_index: int,
    ) -> None:
        """Apply bar face color from the lowercase color role or style.

        Integer color values are discrete category ids mapped to the rcParams
        palette, while continuous float values are mapped through a colormap.
        """
        kwargs.pop("label", None)
        fallback_color = self.series_color(style, layer_index)
        if "color" in df.columns:
            colors = self.color_sequence_from_values(
                df["color"],
                fallback_color=fallback_color,
            )
            if colors:
                kwargs["color"] = colors
                kwargs.pop("facecolor", None)
                return
        kwargs["color"] = fallback_color
        kwargs.pop("facecolor", None)

    def _collect_categories(self, series: list[SeriesData]) -> list[Any]:
        values: list[Any] = []
        for sd in series:
            raw_values = sd.df["X"].tolist() if "X" in sd.df.columns else list(range(len(sd.df)))
            for value in raw_values:
                if value not in values:
                    values.append(value)
        return values

    def _apply_category_ticks(self, ax: Any, categories: list[Any], options: dict[str, Any]) -> None:
        """Label the category axis, thinning and rotating labels as needed.

        With more categories than labels can fit, every step-th label is drawn
        rather than all of them shrunk: Matplotlib would otherwise overlap them
        into an unreadable band.  ``auto`` rotation and font size follow the
        category count; an explicit value always wins.
        """
        count = len(categories)
        labels = [str(value) for value in categories]
        positions = list(range(count))

        max_labels = self.opt("max_tick_labels", options)
        step = max(1, math.ceil(count / int(str(max_labels)))) if max_labels else 1
        tick_positions = positions[::step]
        tick_labels = labels[::step]

        rotation_option = self.opt("tick_label_rotation", options)
        if rotation_option == "auto":
            if count > 40:
                rotation_value = 90.0
            elif count > 12:
                rotation_value = 45.0
            else:
                rotation_value = 0.0
        else:
            rotation_value = float(str(rotation_option))

        fontsize_option = self.opt("tick_label_fontsize", options)
        if fontsize_option == "auto":
            if count > 80:
                fontsize_value: float | None = 6.0
            elif count > 40:
                fontsize_value = 7.0
            elif count > 20:
                fontsize_value = 8.0
            else:
                fontsize_value = None
        else:
            fontsize_value = float(str(fontsize_option))

        if self.Orientation == "horizontal":
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
            for tick_label in ax.get_yticklabels():
                if fontsize_value is not None:
                    tick_label.set_fontsize(fontsize_value)
        else:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.tick_params(axis="x", labelrotation=rotation_value)
            for tick_label in ax.get_xticklabels():
                tick_label.set_ha("right" if rotation_value else "center")
                if fontsize_value is not None:
                    tick_label.set_fontsize(fontsize_value)

    def _bar_kwargs(self, options: dict[str, Any]) -> dict[str, Any]:
        """Return the kwargs to forward to ``bar``/``barh``.

        The options this renderer consumes itself - bar thickness, the error
        columns, everything driving the ticks - are removed here, because
        passing them on would raise an unexpected-keyword error inside
        Matplotlib.
        """
        kwargs = self.get_kwargs(options)
        for key in [
            "width",
            "height",
            "bottom",
            "left",
            "xerr",
            "yerr",
            "tick_label",
            "show_legend",
            "max_tick_labels",
            "tick_label_rotation",
            "tick_label_fontsize",
        ]:
            kwargs.pop(key, None)

        clean_kwargs = {key: value for key, value in kwargs.items() if value is not None and value != ""}
        for key in ["alpha", "linewidth", "capsize", "zorder", "picker"]:
            if key in clean_kwargs:
                clean_kwargs[key] = float(str(clean_kwargs[key]))
        return clean_kwargs

    def _merge_options(self, axis_options: dict[str, Any], style: dict[str, Any]) -> dict[str, Any]:
        """Overlay one series' style on the axis options.

        ``axis_kwargs`` is merged key by key rather than replaced, so a series
        overriding its colour keeps the axis-wide alpha.
        """
        merged = dict(axis_options or {})
        axis_kwargs = dict(merged.get("axis_kwargs", {}) or {})
        axis_kwargs.update(style.get("axis_kwargs", {}) or {})
        for key, value in style.items():
            if key != "axis_kwargs":
                merged[key] = value
        merged["axis_kwargs"] = axis_kwargs
        return merged


class HorizontalBarAxisRenderer(BarAxisRenderer, BaseAxisRenderer):
    """Renderer for horizontal bar charts."""

    Name: str = "Horizontal Bar Chart"
    Category: str = "Pairwise data"
    Description: str = "Horizontal bar chart"
    Orientation: Orientation = "horizontal"
    Link="https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.barh.html"
