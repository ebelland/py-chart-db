"""Broken bar (Gantt-style interval) renderers, horizontal and vertical.

``BrokenBarAxisRenderer`` draws horizontal interval bars grouped by category -
Matplotlib's own ``broken_barh``, one call per category so every row's
intervals land on that row's band.  ``BrokenBarVerticalAxisRenderer``
subclasses it for the column orientation Matplotlib has no built-in for,
drawing the same intervals as Rectangle patches instead.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from app.charts.base import BaseAxisRenderer, SeriesData

Orientation = Literal["horizontal", "vertical"]


class BrokenBarAxisRenderer(BaseAxisRenderer):
    """Interval bars grouped by category - a Gantt chart, in effect.

    Role columns:
        category    required, the row/band each interval belongs to
        start       required, where the interval begins on the value axis
        duration    required, how far the interval extends
        color       optional per-row color

    Every row is one interval; a category with several rows gets several
    intervals on the same band - the "broken" in broken_barh.
    """

    Name: str = "Broken Bar"
    Category: str = "Pairwise data"
    Description: str = "Interval bars grouped by category (Gantt-style)."
    Orientation: Orientation = "horizontal"
    Link: str = "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.broken_barh.html"

    RequiredRoles: list[str] = ["category", "start", "duration"]
    OptionalRoles: list[str] = ["color"]

    Kwargs: dict[str, object] = {
        "band_height": {
            "default": 0.8,
            "type": float,
            "min": 0.05,
            "max": 1.0,
            "step": 0.05,
            "decimals": 3,
            "group": "Geometry",
            "description": "Band thickness as a fraction of the space between categories.",
        },
        "facecolor": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Interval fill color. Overridden by the color role when present.",
        },
        "edgecolor": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Interval outline color.",
        },
        "linewidth": {
            "default": None,
            "type": float,
            "min": 0.0,
            "max": 20.0,
            "step": 0.25,
            "decimals": 3,
            "group": "Appearance",
            "description": "Interval outline width. Use 0 to hide it.",
        },
        "alpha": {
            "default": None,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "group": "Appearance",
            "description": "Interval fill opacity.",
        },
    }

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

        categories = self._collect_categories(valid_series)
        category_pos = {value: index for index, value in enumerate(categories)}

        for layer_index, sd in enumerate(valid_series):
            merged = self._merge_options(axis_options, sd.style or {})
            self._render_series(
                ax=ax,
                sd=sd,
                merged_options=merged,
                category_pos=category_pos,
                layer_index=layer_index,
            )

        self._apply_category_ticks(ax, categories)
        self.apply_annotations(ax, axis_options)

    def _render_series(
        self,
        *,
        ax: Any,
        sd: SeriesData,
        merged_options: dict[str, Any],
        category_pos: dict[Any, int],
        layer_index: int,
    ) -> None:
        df = sd.df
        height = float(str(self.opt("band_height", merged_options)))
        kwargs = self._patch_kwargs(merged_options)
        fallback_color = self.series_color(sd.style or {}, layer_index)

        starts = pd.to_numeric(df["start"], errors="coerce").to_numpy(dtype=float)
        durations = pd.to_numeric(df["duration"], errors="coerce").to_numpy(dtype=float)
        colors = (
            self.color_sequence_from_values(df["color"], fallback_color=fallback_color)
            if "color" in df.columns
            else None
        )

        for category_value, group_index in self._grouped_row_indexes(df["category"]).items():
            if category_value not in category_pos:
                continue
            band_center = category_pos[category_value]
            xranges = [
                (float(starts[i]), float(durations[i]))
                for i in group_index
                if np.isfinite(starts[i]) and np.isfinite(durations[i])
            ]
            if not xranges:
                continue
            row_kwargs = dict(kwargs)
            if colors is not None:
                row_kwargs["facecolors"] = [colors[i] for i in group_index]
            self._draw_band(ax, xranges, band_center, height, row_kwargs)

    def _draw_band(
        self,
        ax: Any,
        xranges: list[tuple[float, float]],
        band_center: float,
        height: float,
        kwargs: dict[str, Any],
    ) -> None:
        ax.broken_barh(xranges, (band_center - height / 2.0, height), **kwargs)

    def _grouped_row_indexes(self, categories: pd.Series) -> dict[Any, list[int]]:
        grouped: dict[Any, list[int]] = {}
        for row_index, value in enumerate(categories.tolist()):
            grouped.setdefault(value, []).append(row_index)
        return grouped

    def _collect_categories(self, series: list[SeriesData]) -> list[Any]:
        values: list[Any] = []
        for sd in series:
            for value in sd.df["category"].tolist():
                if value not in values:
                    values.append(value)
        return values

    def _apply_category_ticks(self, ax: Any, categories: list[Any]) -> None:
        positions = list(range(len(categories)))
        labels = [str(value) for value in categories]
        if self.Orientation == "horizontal":
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)

    def _patch_kwargs(self, options: dict[str, Any]) -> dict[str, Any]:
        kwargs = self.get_kwargs(options)
        kwargs.pop("band_height", None)
        return {key: value for key, value in kwargs.items() if value is not None and value != ""}

    def _merge_options(self, axis_options: dict[str, Any], style: dict[str, Any]) -> dict[str, Any]:
        merged = dict(axis_options or {})
        axis_kwargs = dict(merged.get("axis_kwargs", {}) or {})
        axis_kwargs.update(style.get("axis_kwargs", {}) or {})
        for key, value in style.items():
            if key != "axis_kwargs":
                merged[key] = value
        merged["axis_kwargs"] = axis_kwargs
        return merged


class BrokenBarVerticalAxisRenderer(BrokenBarAxisRenderer, BaseAxisRenderer):
    """Column orientation of the same chart - Matplotlib has no built-in for it.

    Drawn as Rectangle patches rather than ``broken_barh``, which only ever
    draws horizontally; the grouping and category handling are otherwise
    identical to the horizontal renderer.
    """

    Name: str = "Broken Bar (Vertical)"
    Category: str = "Pairwise data"
    Description: str = "Interval bars grouped by category, stacked in columns."
    Orientation: Orientation = "vertical"

    def _draw_band(
        self,
        ax: Any,
        xranges: list[tuple[float, float]],
        band_center: float,
        height: float,
        kwargs: dict[str, Any],
    ) -> None:
        facecolors = kwargs.pop("facecolors", None)
        for index, (start, duration) in enumerate(xranges):
            patch_kwargs = dict(kwargs)
            if facecolors is not None:
                patch_kwargs["facecolor"] = facecolors[index]
            ax.add_patch(
                Rectangle(
                    (band_center - height / 2.0, start),
                    height,
                    duration,
                    **patch_kwargs,
                )
            )
        # add_patch() does not grow the view limits the way a plotting call
        # would, so without this the bars are drawn outside an axis that
        # never resized to show them.
        ax.relim()
        ax.autoscale_view()
