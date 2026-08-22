"""Box-and-whisker renderer with per-series styling.

Statistics are computed here and drawn through ``ax.bxp`` rather than delegated
to ``ax.boxplot``.  Why: ``boxplot`` draws all boxes in one call with one style,
while this renderer needs per-series colour, alpha, linewidth and zorder, which
requires one ``bxp`` call per box.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from app.charts.base import BaseAxisRenderer, SeriesData
from app.logs.logger import applogger


class BoxAxisRenderer(BaseAxisRenderer):
    """Renderer for box plots.

    Behavior:
    - If a 'group' column is present, each unique group becomes a category.
      Multiple series that share the same category are drawn side by side.
    - If no 'group' column is present, each series contributes one category.
    - Boxes are filled by series color and the legend contains one entry per
      series.
    - Box outline, whisker, cap, median, mean and flier line colors are left to
      matplotlib rcParams/theme. The renderer only applies the series color to
      the box face/fill.
    - Optional statistical annotations can be enabled through axis options.
    - Boxes are drawn one at a time (one ax.bxp call per box) so that
      per-series style (color, alpha, linewidth, zorder) can be honored
      independently, mirroring ScatterAxisRenderer's per-series drawing.
    - Box statistics (median, quartiles, whiskers, fliers) are computed
      manually so ax.bxp can be used instead of ax.boxplot, which is what
      allows per-box styling.

    Honors per-series style fields such as:
        - label
        - visible
        - show_in_legend
        - color
        - alpha
        - zorder
        - linewidth
    """

    RequiredRoles = ["value"]
    OptionalRoles = ["group", "color"]
    Name = "Box Plot"
    Category: str = "Statistical distributions"
    Description = "Draw a box and whisker plot."
    Link="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html"
    
    Kwargs: dict[str, object] = {
        "notch": {
            "default": False,
            "type": bool,
            "description": "Draw notched boxes indicating the confidence interval around the median.",
        },
        "sym": {
            "default": None,
            "type": str,
            "description": (
                "Outlier marker symbol (matplotlib marker string). "
                "Leave empty to use the default flier marker."
            ),
        },
        "whis": {
            "default": 1.5,
            "type": float,
            "min": 0.0,
            "max": 100.0,
            "description": (
                "Whisker reach as an IQR multiplier. Default extends to "
                "1.5x IQR beyond the first and third quartiles."
            ),
        },
        "widths": {
            "default": 0.5,
            "type": float,
            "min": 0.01,
            "max": 1.0,
            "description": "Width of each box.",
        },
        "direction": {
            "default": "vertical",
            "type": ["vertical", "horizontal"],
            "description": "Draw boxes vertically or horizontally.",
        },
        "patch_artist": {
            "default": True,
            "type": bool,
            "description": "Fill boxes with a face color instead of drawing them as outlines only.",
        },
        "showmeans": {
            "default": False,
            "type": bool,
            "description": "Show the arithmetic mean of each box.",
        },
        "meanline": {
            "default": False,
            "type": bool,
            "description": "Draw the mean as a line instead of a point (requires showmeans).",
        },
        "showcaps": {
            "default": True,
            "type": bool,
            "description": "Show the caps at the ends of the whiskers.",
        },
        "showbox": {
            "default": True,
            "type": bool,
            "description": "Show the box body.",
        },
        "showfliers": {
            "default": True,
            "type": bool,
            "description": "Show outlier points beyond the whiskers.",
        },
        "label": {
            "default": None,
            "type": str,
            "description": "Legend label override. If empty, the series name is used.",
        },
        "visible": {
            "default": True,
            "type": bool,
            "description": "Whether the box collection is visible.",
        },
        "zorder": {
            "default": None,
            "type": float,
            "min": -1000.0,
            "max": 1000.0,
            "description": "Drawing order; higher values are drawn on top.",
        },
        "show_stats": {
            "default": False,
            "type": bool,
            "description": "Display statistical annotations for each box.",
        },
        "stats_type": {
            "default": "median",
            "type": str,
            "description": (
                "Annotation content: median, mean, n, median+n, mean+n, "
                "or all."
            ),
        },
        "stats_format": {
            "default": "",
            "type": str,
            "description": (
                "Optional Python format string using fields: n, mean, median, "
                "q1, q3, whislo, whishi, std. Example: 'M={median:.2f}\\nn={n}'."
            ),
        },
        "stats_position": {
            "default": "outside",
            "type": str,
            "description": "Annotation position: outside, median, or mean.",
        },
        "stats_fontsize": {
            "default": 8,
            "type": int,
            "min": 1,
            "max": 72,
            "description": "Font size for statistical annotations.",
        },
        "stats_color_mode": {
            "default": "series",
            "type": str,
            "description": "Annotation color: series, black, rcparams, or any matplotlib color.",
        },
        "stats_offset_fraction": {
            "default": 0.04,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "description": (
                "Offset for outside statistical annotations, expressed as a "
                "fraction of the current data range."
            ),
        },
    }

    def collect_grouped_items(
        self,
        series: list[SeriesData],
        options: dict,
    ) -> dict[str, Any]:
        """Split every visible series into per-category data groups.

        Shared with the violin renderer, which arranges its bodies on exactly
        the same category-then-series grid.  Returns the plot items plus the
        label, colour and legend bookkeeping the caller needs to draw them.
        """
        plot_items: list[dict[str, Any]] = []
        category_labels: list[str] = []
        series_labels: list[str] = []
        series_show_in_legend: dict[str, bool] = {}
        series_colors: dict[str, Any] = {}

        for series_index, sd in enumerate(series):
            style = dict(sd.style or {})
            if not bool(style.get("visible", True)) or not self.ensure_required_roles(sd.df):
                continue

            value = pd.to_numeric(sd.df["value"], errors="coerce")
            if value is None:
                applogger.warning(
                    "Series '%s' skipped: required numeric 'value' data not found.",
                    sd.name,
                )
                continue

            series_label = str(style.get("label", "") or "").strip()
            if not series_label:
                axis_label = str(self.opt("label", options) or "").strip()
                series_label = axis_label if axis_label else sd.name.strip()
            if not series_label:
                series_label = f"Series {series_index + 1}"

            if series_label not in series_labels:
                series_labels.append(series_label)
                series_show_in_legend[series_label] = bool(style.get("show_in_legend", True))
                series_colors[series_label] = self._series_box_color(sd.df, style, len(series_labels) - 1)

            group = sd.df["group"] if "group" in sd.df.columns else None
            if group is not None:
                sub_frames = [
                    (str(group_value), value[group == group_value])
                    for group_value in pd.unique(group.dropna())
                ]
            else:
                sub_frames = [(series_label, value)]

            for category_label, sub_values in sub_frames:
                data = sub_values.dropna().to_numpy(dtype=float)
                if data.size == 0:
                    applogger.debug(
                        "Series '%s' group '%s' skipped: no finite values.",
                        sd.name,
                        category_label,
                    )
                    continue

                if category_label not in category_labels:
                    category_labels.append(category_label)

                plot_items.append(
                    {
                        "category_label": category_label,
                        "series_label": series_label,
                        "data": data,
                        "style": style,
                    }
                )

        return {
            "items": plot_items,
            "category_labels": category_labels,
            "series_labels": series_labels,
            "show_in_legend": series_show_in_legend,
            "colors": series_colors,
        }

    @staticmethod
    def grouped_layout(
        *,
        items: list[dict[str, Any]],
        category_labels: list[str],
        series_labels: list[str],
        requested_width: float,
    ) -> dict[str, Any]:
        """Return the drawing position of every item, plus the shared width.

        Everything in one category shares a band of 80 % of the tick spacing,
        split evenly between the series *that appear in that category*.  The
        qualifier matters: reserving a slot for every series in every category
        pushes a lone item off its own tick, which is what happened when four
        batches were four one-category series - each violin sat a quarter of a
        band away from the label underneath it.

        The width is computed from the busiest category so that items are the
        same size across the whole axis; only the offsets vary.

        Keeping this in one place is what makes a violin land exactly where the
        box it replaces would have been.
        """
        category_positions = {
            category_label: float(index)
            for index, category_label in enumerate(category_labels)
        }

        # Which series actually contribute to each category, in series order.
        order = {label: index for index, label in enumerate(series_labels)}
        present: dict[str, list[str]] = {}
        for item in items:
            bucket = present.setdefault(str(item["category_label"]), [])
            if item["series_label"] not in bucket:
                bucket.append(str(item["series_label"]))
        for bucket in present.values():
            bucket.sort(key=lambda label: order.get(label, 0))

        widest = max((len(bucket) for bucket in present.values()), default=1)
        slot_width = 0.8 / max(1, widest)
        item_width = (
            min(requested_width, slot_width * 0.85) if widest > 1 else requested_width
        )

        offsets: dict[tuple[str, str], float] = {}
        for category_label, bucket in present.items():
            count = len(bucket)
            for index, series_label in enumerate(bucket):
                offsets[(category_label, series_label)] = (
                    index - (count - 1) / 2.0
                ) * slot_width

        return {
            "category_positions": category_positions,
            "offsets": offsets,
            "series_count": widest,
            "slot_width": slot_width,
            "item_width": item_width,
        }

    @staticmethod
    def item_position(layout: dict[str, Any], item: dict[str, Any]) -> float:
        """Return the axis coordinate one item should be drawn at."""
        category_label = str(item["category_label"])
        key = (category_label, str(item["series_label"]))
        return layout["category_positions"][category_label] + layout["offsets"].get(key, 0.0)

    @staticmethod
    def is_vertical(base_kwargs: dict, options: dict) -> bool:
        """Return True when the plot is drawn vertically.

        ``direction`` replaced a boolean option - ``vert`` on the box plot,
        ``vertical`` on the violin - because Matplotlib 3.11 deprecated the
        boolean ``vert`` argument in favour of ``orientation``, and carrying
        two spellings of the same idea was how they drifted apart.

        The old keys are read from the raw options because ``get_kwargs``
        keeps only declared ones: a figure saved as horizontal before the
        rename would otherwise come back vertical, silently, with nothing to
        say why.

        The raw options are also where ``direction`` itself is looked for
        first.  ``base_kwargs`` cannot answer whether a figure *chose* a
        direction, since ``get_kwargs`` substitutes the declared default for
        anything absent - so consulting it first would make every old figure
        look like it had asked for vertical.
        """
        axis_kwargs = options.get("axis_kwargs")
        sources = [options, axis_kwargs if isinstance(axis_kwargs, dict) else {}]

        for source in sources:
            if source.get("direction") is not None:
                return str(source["direction"]).strip().lower() != "horizontal"

        for source in sources:
            for legacy in ("vert", "vertical"):
                if legacy in source:
                    return bool(source[legacy])

        return str(base_kwargs.get("direction", "vertical")).strip().lower() != "horizontal"

    def apply_category_ticks(
        self,
        ax: Any,
        *,
        category_labels: list[str],
        category_positions: dict[str, float],
        vert: bool,
    ) -> None:
        """Label the category axis, whichever way round the plot is drawn."""
        tick_positions = [category_positions[label] for label in category_labels]
        if vert:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(category_labels)
        else:
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(category_labels)

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        """Render all box series onto a single axis.

        Grouped data is arranged by category first, then by series. This means
        that when multiple series contain the same group/category values, their
        boxes are placed side by side within each category. Colors and legend
        entries are keyed by series, not by category.
        """
        base_kwargs = self.get_kwargs(options)
        vert = self.is_vertical(base_kwargs, options)
        requested_width = self._coerce_positive_float(base_kwargs.get("widths", 0.5), 0.5)

        grouped = self.collect_grouped_items(series, options)
        plot_items = grouped["items"]
        category_labels = grouped["category_labels"]
        series_show_in_legend = grouped["show_in_legend"]
        series_colors = grouped["colors"]

        if not plot_items:
            return

        layout = self.grouped_layout(
            items=plot_items,
            category_labels=category_labels,
            series_labels=grouped["series_labels"],
            requested_width=requested_width,
        )
        category_positions = layout["category_positions"]
        box_width = layout["item_width"]

        legend_handles: list[Patch] = []
        legend_labels: list[str] = []
        legend_seen: set[str] = set()
        annotation_requests: list[dict[str, Any]] = []

        for item in plot_items:
            series_label = item["series_label"]
            style = item["style"]
            data = item["data"]

            position = self.item_position(layout, item)
            flat_color = series_colors[series_label]
            zorder_value = style.get("zorder")
            linewidth_value = style.get("linewidth", style.get("line_width"))
            stats = self._bxp_stats(data, base_kwargs)

            box_result = ax.bxp(
                [stats],
                positions=[position],
                widths=box_width,
                orientation="vertical" if vert else "horizontal",
                patch_artist=base_kwargs.get("patch_artist", True),
                shownotches=bool(base_kwargs.get("notch", False)),
                showmeans=base_kwargs.get("showmeans", False),
                meanline=base_kwargs.get("meanline", False),
                showcaps=base_kwargs.get("showcaps", True),
                showbox=base_kwargs.get("showbox", True),
                showfliers=base_kwargs.get("showfliers", True),
            )

            self._style_box(
                box_result=box_result,
                face_color=flat_color,
                alpha=style.get("alpha") or 1,
                linewidth=linewidth_value,
                zorder=zorder_value,
                flier_marker=base_kwargs.get("sym"),
            )

            if bool(base_kwargs.get("show_stats", False)):
                annotation_requests.append(
                    {
                        "position": position,
                        "stats": stats,
                        "data": data,
                        "series_color": flat_color,
                    }
                )

            if (
                series_show_in_legend.get(series_label, True)
                and series_label not in legend_seen
            ):
                boxes = box_result.get("boxes", [])
                if boxes:
                    face_color, edge_color = self._legend_patch_colors(
                        box_artist=boxes[0],
                        fallback_face_color=flat_color,
                    )
                    legend_handles.append(
                        Patch(
                            facecolor=face_color,
                            edgecolor=edge_color,
                            label=series_label,
                        )
                    )
                    legend_labels.append(series_label)
                    legend_seen.add(series_label)

        self.apply_category_ticks(
            ax,
            category_labels=category_labels,
            category_positions=category_positions,
            vert=vert,
        )

        if annotation_requests:
            self._draw_stat_annotations(
                ax=ax,
                annotations=annotation_requests,
                box_kwargs=base_kwargs,
                vert=vert,
            )

        if legend_handles:
            ax.legend(handles=legend_handles, labels=legend_labels)
        # Draw descriptor annotations after renderer-owned artists.
        self.apply_annotations(ax, options or {})



    def _series_box_color(self, df: pd.DataFrame, style: dict[str, Any], layer_index: int) -> Any:
        fallback_color = self.series_color(style, layer_index)
        if "color" in df.columns:
            return self.first_color_from_values(df["color"], fallback_color=fallback_color)
        return fallback_color

    def _style_box(
        self,
        box_result: dict[str, list[Any]],
        face_color: str,
        alpha: Any,
        linewidth: Any,
        zorder: Any,
        flier_marker: Any,
    ) -> None:
        """Apply per-box style overrides to the artists returned by ax.bxp.

        Important styling rule:
        - The renderer applies series color only to the box face/fill.
        - Edge, whisker, cap, median, mean and flier line colors are not set
          here. They remain controlled by matplotlib rcParams/theme.
        """
        if face_color:
            for patch in box_result.get("boxes", []):
                patch.set_facecolor(face_color)

        if alpha not in (None, ""):
            for patch in box_result.get("boxes", []):
                patch.set_alpha(alpha)

        if isinstance(flier_marker, str) and flier_marker.strip():
            for flier in box_result.get("fliers", []):
                flier.set_marker(flier_marker.strip())

        # zorder/linewidth apply uniformly across every artist in the box
        # (box, whiskers, caps, median, fliers) so the whole box moves or
        # thickens together. Colors still come from rcParams unless they were
        # already set by matplotlib.
        all_artists = [
            artist for artist_list in box_result.values() for artist in artist_list
        ]

        if zorder not in (None, ""):
            for artist in all_artists:
                artist.set_zorder(zorder)

        if linewidth not in (None, ""):
            for artist in all_artists:
                artist.set_linewidth(linewidth)


    @staticmethod
    def _legend_patch_colors(
        *,
        box_artist: Any,
        fallback_face_color: str,
    ) -> tuple[Any, Any]:
        """Return legend face/edge colors for patch and line box artists."""
        if isinstance(box_artist, Patch):
            face_color = fallback_face_color or box_artist.get_facecolor()
            return face_color, box_artist.get_edgecolor()

        if isinstance(box_artist, Line2D):
            line_color = box_artist.get_color()
            face_color = fallback_face_color or line_color
            return face_color, line_color

        return fallback_face_color or "C0", "black"

    @classmethod
    def _draw_stat_annotations(
        cls,
        *,
        ax: Any,
        annotations: list[dict[str, Any]],
        box_kwargs: dict[str, Any],
        vert: bool,
    ) -> None:
        """Draw optional statistics labels for each box."""
        fontsize = int(cls._coerce_positive_float(box_kwargs.get("stats_fontsize", 8), 8.0))
        position_mode = str(box_kwargs.get("stats_position", "outside") or "outside").strip().lower()
        color_mode = str(box_kwargs.get("stats_color_mode", "series") or "series").strip().lower()
        offset_fraction = cls._coerce_positive_float(
            box_kwargs.get("stats_offset_fraction", 0.04),
            0.04,
        )

        if vert:
            axis_min, axis_max = ax.get_ylim()
        else:
            axis_min, axis_max = ax.get_xlim()
        axis_span = abs(float(axis_max) - float(axis_min)) or 1.0
        offset = axis_span * offset_fraction

        for annotation in annotations:
            position = float(annotation["position"])
            stats = annotation["stats"]
            data = annotation["data"]
            series_color = str(annotation.get("series_color", "") or "").strip()

            text = cls._format_stats_text(stats=stats, data=data, box_kwargs=box_kwargs)
            if not text:
                continue

            color = cls._resolve_stats_color(
                color_mode=color_mode,
                series_color=series_color,
            )

            if position_mode == "median":
                value_position = float(stats["med"])
                va = "center"
                ha = "center"
            elif position_mode == "mean":
                value_position = float(stats["mean"])
                va = "center"
                ha = "center"
            else:
                # For vertical boxes annotation goes above the upper whisker.
                # For horizontal boxes annotation goes to the right of the
                # upper whisker.
                value_position = float(stats["whishi"]) + offset
                va = "bottom" if vert else "center"
                ha = "center" if vert else "left"

            text_kwargs: dict[str, Any] = {
                "fontsize": fontsize,
                "ha": ha,
                "va": va,
            }
            if color is not None:
                text_kwargs["color"] = color

            if vert:
                ax.text(position, value_position, text, **text_kwargs)
            else:
                ax.text(value_position, position, text, **text_kwargs)

    @classmethod
    def _format_stats_text(
        cls,
        *,
        stats: dict[str, Any],
        data: np.ndarray,
        box_kwargs: dict[str, Any],
    ) -> str:
        """Build the statistical annotation text for one box."""
        n = int(data.size)
        values: dict[str, Any] = {
            "n": n,
            "mean": float(stats["mean"]),
            "median": float(stats["med"]),
            "med": float(stats["med"]),
            "q1": float(stats["q1"]),
            "q3": float(stats["q3"]),
            "whislo": float(stats["whislo"]),
            "whishi": float(stats["whishi"]),
            "std": float(np.std(data, ddof=1)) if n > 1 else 0.0,
        }

        custom_format = str(box_kwargs.get("stats_format", "") or "").strip()
        if custom_format:
            try:
                return custom_format.format(**values)
            except (KeyError, IndexError, ValueError) as exc:
                applogger.warning(
                    "Invalid box plot stats_format %r: %s. Falling back to stats_type.",
                    custom_format,
                    exc,
                )

        stats_type = str(box_kwargs.get("stats_type", "median") or "median").strip().lower()
        if stats_type == "median":
            return f"M={values['median']:.2f}"
        if stats_type == "mean":
            return f"mean={values['mean']:.2f}"
        if stats_type == "n":
            return f"n={values['n']}"
        if stats_type in {"median+n", "median_n"}:
            return f"M={values['median']:.2f}\nn={values['n']}"
        if stats_type in {"mean+n", "mean_n"}:
            return f"mean={values['mean']:.2f}\nn={values['n']}"
        if stats_type == "all":
            return (
                f"n={values['n']}\n"
                f"mean={values['mean']:.2f}\n"
                f"M={values['median']:.2f}\n"
                f"std={values['std']:.2f}"
            )

        return f"M={values['median']:.2f}"

    @staticmethod
    def _resolve_stats_color(*, color_mode: str, series_color: str) -> str | None:
        """Resolve statistical annotation color.

        Returning None leaves matplotlib text color controlled by rcParams.
        """
        if color_mode in {"", "rc", "rcparams", "default"}:
            return None
        if color_mode == "series":
            return series_color or None
        if color_mode == "black":
            return "black"
        return color_mode

    @staticmethod
    def _coerce_positive_float(value: Any, default: float) -> float:
        """Return a positive float, falling back to default for invalid input."""
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if result > 0 else default

    @staticmethod
    def _coerce_whis_multiplier(value: Any, default: float = 1.5) -> float:
        """Return a scalar whisker multiplier for non-percentile whis values."""
        if isinstance(value, (list, tuple)):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bxp_stats(data: np.ndarray, box_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Compute matplotlib bxp-compatible summary stats for one box.

        ax.bxp expects pre-computed stats (median, quartiles, whisker
        bounds, fliers) rather than raw data, unlike ax.boxplot.
        """
        whis = box_kwargs.get("whis", 1.5)
        show_fliers = box_kwargs.get("showfliers", True)

        q1, median, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1

        if isinstance(whis, (list, tuple)) and len(whis) == 2:
            lo, hi = np.percentile(data, [float(whis[0]), float(whis[1])])
        else:
            whis_multiplier = BoxAxisRenderer._coerce_whis_multiplier(whis)
            lo = q1 - whis_multiplier * iqr
            hi = q3 + whis_multiplier * iqr

        in_range = data[(data >= lo) & (data <= hi)]
        whislo = float(in_range.min()) if in_range.size else float(q1)
        whishi = float(in_range.max()) if in_range.size else float(q3)

        fliers = data[(data < whislo) | (data > whishi)] if show_fliers else np.array([])

        notch_half_width = 1.57 * iqr / np.sqrt(float(data.size))

        return {
            "med": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "whislo": whislo,
            "whishi": whishi,
            "mean": float(np.mean(data)),
            "fliers": fliers,
            "cilo": float(median - notch_half_width),
            "cihi": float(median + notch_half_width),
        }
