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

from app.charts import kwarg_spec
from app.charts.base import (
    ARTIST_ADVANCED_KWARGS,
    ARTIST_KWARGS,
    LEGEND_OPTIONS,
    LINE_KWARGS,
    PATCH_KWARGS,
    BaseAxisRenderer,
    SeriesData,
    SeriesFrame,
    merge,
    pick,
)


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

    #: Forwarded verbatim to ``bar``/``barh``.  Everything here is a keyword
    #: Matplotlib accepts, which is why there is nothing to remove before the
    #: call: the removal list this renderer used to keep - eleven names, kept
    #: in step with the schema by hand - is now the ``Options`` dict below.
    #:
    #: Note what is *not* here.  ``capthick``, ``elinewidth`` and
    #: ``errorevery`` are in the shared ERROR_BAR_KWARGS but ``bar`` does not
    #: take them: its error bars are configured through ``ecolor``,
    #: ``capsize`` and nothing else, and the extra three would reach
    #: ``Rectangle.set`` and raise.
    Kwargs: dict[str, object] = merge(
        # label is the renderer's own: it resolves the legend text from the
        # series and passes it separately, so it is an Option, not a keyword.
        pick(ARTIST_KWARGS, "alpha", "zorder", "visible", "rasterized", "picker"),
        ARTIST_ADVANCED_KWARGS,
        PATCH_KWARGS,
        pick(LINE_KWARGS, "color"),
        {
            "color": {
                "description": (
                    "Bar face colour. Overridden by the color role column "
                    "when the series provides one."
                ),
            },
            "align": {
                "default": "center",
                "type": ["center", "edge"],
                "group": "Geometry",
                "description": "Whether a bar is centred on its category position or starts at it.",
            },
            "ecolor": {
                "default": None,
                "type": str,
                "kind": "color",
                "group": "Error bars",
                "description": "Error bar colour. Defaults to the bar colour.",
            },
            "capsize": {
                "default": None,
                kwarg_spec.STYLE_DEFAULT: True,
                kwarg_spec.RCPARAM: "errorbar.capsize",
                "type": float,
                "min": 0.0,
                "max": 50.0,
                "step": 0.5,
                "decimals": 2,
                "group": "Error bars",
                "description": "Length of the error bar caps, in points. 0 draws no caps.",
            },
            "log": {
                "default": False,
                "type": bool,
                "group": "Behaviour",
                "description": "Use a logarithmic scale on the value axis.",
            },
        },
    )

    #: Read by this renderer and never forwarded.
    #:
    #: ``width``/``height`` look like the Matplotlib keywords of the same name
    #: and are not: the declared value is the width of the whole category
    #: group, which is divided by the number of series and passed as the width
    #: of one bar.  Forwarding it would make every series as wide as the group
    #: and draw them on top of each other.  That is the difference the single
    #: dict could not express.
    Options: dict[str, object] = merge(
        LEGEND_OPTIONS,
        {
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
                "description": "Scalar baseline for vertical bars, when no Bottom role column is present.",
            },
            "left": {
                "default": None,
                "type": float,
                "group": "Geometry",
                "description": "Scalar baseline for horizontal bars, when no Left role column is present.",
            },
            "xerr": {
                "default": None,
                "type": float,
                "min": 0.0,
                "group": "Error bars",
                "description": "Scalar horizontal error bar, when no XError role column is present.",
            },
            "yerr": {
                "default": None,
                "type": float,
                "min": 0.0,
                "group": "Error bars",
                "description": "Scalar vertical error bar, when no YError role column is present.",
            },
            "label": {
                "default": None,
                "type": str,
                "group": "Legend",
                "description": "Legend label override. The series name is used when empty.",
            },
            "max_tick_labels": {
                "default": 40,
                "type": int,
                "min": 1,
                "max": 1000,
                "step": 1,
                "group": "Text",
                "description": "Most category tick labels to draw. Every step-th one is kept beyond this.",
            },
            "tick_label_rotation": {
                "default": "auto",
                "type": ["auto", "0", "30", "45", "60", "90"],
                "group": "Text",
                "description": "Category label rotation. Auto rotates when there are many categories.",
            },
            "tick_label_fontsize": {
                "default": "auto",
                "type": ["auto", "6", "7", "8", "9", "10", "11", "12"],
                "group": "Text",
                "description": "Category label font size. Auto shrinks when there are many labels.",
            },
        },
    )

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
        valid_series = self.valid_series(series)
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

        tick_options = self.merge_style(axis_options, valid_series[0].style or {})
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
        merged_options = self.merge_style(axis_options, style)

        if "X" not in df.columns:
            df["X"] = list(range(len(df)))

        kwargs = self.get_kwargs(merged_options)
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
        df: SeriesFrame,
        style: dict[str, Any],
        layer_index: int,
    ) -> None:
        """Apply bar face color from the lowercase color role or style.

        Integer color values are discrete category ids mapped to the rcParams
        palette, while continuous float values are mapped through a colormap.
        """
        fallback_color = self.series_color(style, layer_index)
        if "color" in df.columns:
            colors = self.color_sequence_from_values(
                df["color"],
                fallback_color=fallback_color,
            )
            # len(), not truthiness: the continuous path returns an (n, 4)
            # RGBA array, and asking an array whether it is true raises.
            if len(colors):
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


class HorizontalBarAxisRenderer(BarAxisRenderer, BaseAxisRenderer):
    """Renderer for horizontal bar charts."""

    Name: str = "Horizontal Bar Chart"
    Category: str = "Pairwise data"
    Description: str = "Horizontal bar chart"
    Orientation: Orientation = "horizontal"
    Link="https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.barh.html"
