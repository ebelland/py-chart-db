"""Pareto chart: bars sorted by descending magnitude, with a cumulative line.

The chart Juran named after Pareto, and the one people mean by "the 80/20
rule": categories ordered largest first, so that the few that dominate are the
leftmost ones, and a cumulative percentage line that says where the running
total crosses a threshold.  Reading it is one question - how far along the axis
do I get to 80% - and the whole layout exists to make that question answerable
by eye.

Inherits the bar renderer rather than reimplementing it.  Everything about
drawing bars - grouping, error bars, colours, category tick thinning, the
horizontal variant - is already solved there, and a Pareto chart is that plus
an ordering and a second axis.  What this class overrides is exactly those two
things.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.charts.bar_axis import BarAxisRenderer
from app.charts.base_axis import BaseAxisRenderer, SeriesData
from app.logs.logger import applogger

#: Label used for the bar that absorbs everything past ``max_categories``.
OTHER_LABEL: str = "Other"


class ParetoAxisRenderer(BarAxisRenderer, BaseAxisRenderer):
    """Bars in descending order with a cumulative percentage overlay."""

    Name: str = "Pareto Chart"
    Category: str = "Statistical distributions"
    Description: str = (
        "Categories sorted by descending magnitude with a cumulative "
        "percentage line, for finding the few causes behind most of an effect."
    )
    Link: str = (
        "https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_label_demo.html"
    )

    RequiredRoles: list[str] = ["Y"]
    OptionalRoles: list[str] = ["X", "color", "YError", "XError", "Bottom", "Left"]

    Kwargs: dict[str, object] = {
        **BarAxisRenderer.Kwargs,
        "cumulative_line": {
            "default": True,
            "type": bool,
            "group": "Pareto",
            "description": (
                "Draw the running total as a percentage on a second axis. "
                "Without it this is a sorted bar chart."
            ),
        },
        "cumulative_color": {
            "default": "#D32F2F",
            "type": str,
            "kind": "color",
            "group": "Pareto",
            "description": "Colour of the cumulative line and its axis.",
        },
        "cumulative_label": {
            "default": "Cumulative %",
            "type": str,
            "group": "Pareto",
            "description": "Label for the cumulative axis on the right.",
        },
        "reference_percent": {
            "default": 80.0,
            "type": float,
            "min": 0.0,
            "max": 100.0,
            "group": "Pareto",
            "description": (
                "Horizontal reference line, as a percentage. 80 is the "
                "conventional one. Set to 0 to draw no reference."
            ),
        },
        "max_categories": {
            "default": 0,
            "type": int,
            "min": 0,
            "max": 1000,
            "group": "Pareto",
            "description": (
                "Keep only this many categories and total the rest into a "
                "single 'Other' bar. 0 keeps every category. The cumulative "
                "line still accounts for all of them, so the total stays 100%."
            ),
        },
        "ascending": {
            "default": False,
            "type": bool,
            "group": "Pareto",
            "description": (
                "Sort smallest first instead. The cumulative line then rises "
                "slowly and late, which is the wrong shape for the usual "
                "reading but right for a 'long tail' argument."
            ),
        },
    }

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------
    def _category_totals(self, series: list[SeriesData]) -> dict[Any, float]:
        """Total Y per category, across every visible series.

        Across series, not per series: the order has to be one order, or the
        bars in a group would each sit under a different category label.
        """
        totals: dict[Any, float] = {}
        for sd in series:
            frame = sd.df
            if "Y" not in frame.columns:
                continue
            categories = (
                frame["X"].tolist()
                if "X" in frame.columns
                else list(range(len(frame)))
            )
            values = pd.to_numeric(frame["Y"], errors="coerce").to_numpy(dtype=float)
            for category, value in zip(categories, values):
                if np.isfinite(value):
                    totals[category] = totals.get(category, 0.0) + float(value)
        return totals

    #: The order ``render_axis`` resolved, read back by the inherited drawing
    #: code.  An attribute rather than a parameter because the bar renderer
    #: calls ``_collect_categories`` itself, from inside the method being
    #: reused, and threading an argument through would mean overriding that
    #: method too.
    _pareto_order: list[Any] | None = None

    def _collect_categories(self, series: list[SeriesData]) -> list[Any]:
        """Return the categories in Pareto order, largest first.

        Overriding this is the whole of the sorting: the bar renderer places
        each bar at the index this list gives it, so returning a different
        order is enough to reorder the chart without touching how it draws.

        ``render_axis`` has already resolved the order - it needs it for the
        cumulative line as well - so this hands back that answer rather than
        sorting again.  Sorting twice would risk the bars and the line
        disagreeing, which is the one way this chart can lie.
        """
        if self._pareto_order is not None:
            return list(self._pareto_order)

        totals = self._category_totals(series)
        if not totals:
            return super()._collect_categories(series)
        return sorted(totals, key=lambda category: totals[category], reverse=True)

    #: The options this renderer consumes itself.  The bar renderer forwards
    #: everything it does not recognise straight to ``ax.bar``, which rejects
    #: unknown keywords - so anything added here has to be named here too.
    PARETO_ONLY: tuple[str, ...] = (
        "cumulative_line",
        "cumulative_color",
        "cumulative_label",
        "reference_percent",
        "max_categories",
        "ascending",
    )

    def _bar_kwargs(self, options: dict[str, Any]) -> dict[str, Any]:
        """Drop the Pareto options before the rest reach Matplotlib."""
        kwargs = super()._bar_kwargs(options)
        for key in self.PARETO_ONLY:
            kwargs.pop(key, None)
        return kwargs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        """Draw the sorted bars, then the cumulative percentage over them."""
        axis_options = options or {}
        visible = [
            sd
            for sd in series
            if (sd.style or {}).get("visible", True)
            and self.ensure_required_roles(sd.df)
            and not sd.df.empty
        ]
        if not visible:
            return

        merged = self._merge_options(axis_options, visible[0].style or {})
        ascending = bool(merged.get("ascending", False))
        limit = max(0, int(merged.get("max_categories", 0) or 0))

        totals = self._category_totals(visible)
        ordered = sorted(
            totals, key=lambda category: totals[category], reverse=not ascending
        )

        drawn_series, ordered = self._apply_category_limit(visible, ordered, limit)

        # The bars, drawn by the bar renderer in the order collected above.
        self._pareto_order = ordered
        try:
            super().render_axis(ax, drawn_series, axis_options)
        finally:
            self._pareto_order = None

        if bool(merged.get("cumulative_line", True)):
            self._draw_cumulative(ax, totals, ordered, merged)


    def _apply_category_limit(
        self, series: list[SeriesData], ordered: list[Any], limit: int
    ) -> tuple[list[SeriesData], list[Any]]:
        """Fold everything past *limit* into one 'Other' category.

        The tail is summed rather than dropped: a Pareto chart whose bars did
        not add up to the total would put the cumulative line somewhere below
        100% with nothing to say why.
        """
        if limit <= 0 or limit >= len(ordered):
            return series, ordered

        kept = set(ordered[:limit])
        folded: list[SeriesData] = []
        for sd in series:
            frame = sd.df.copy()
            if "X" not in frame.columns:
                frame["X"] = list(range(len(frame)))
            frame["X"] = [
                value if value in kept else OTHER_LABEL for value in frame["X"]
            ]
            frame = (
                frame.groupby("X", as_index=False, sort=False)
                .agg({column: "sum" for column in frame.columns if column != "X"})
            )
            folded.append(SeriesData(name=sd.name, df=frame, style=sd.style))

        applogger.info(
            "Pareto chart: %d categories folded into '%s'.",
            len(ordered) - limit,
            OTHER_LABEL,
        )
        return folded, [*ordered[:limit], OTHER_LABEL]

    def _draw_cumulative(
        self,
        ax: Any,
        totals: dict[Any, float],
        ordered: list[Any],
        options: dict[str, Any],
    ) -> None:
        """Draw the running total, as a percentage, on a twin axis.

        A twin rather than a second scale on the same axis: the bars are in the
        data's own units and the line is in percent, and forcing them onto one
        axis would make one of the two unreadable.  The twin is capped at 105
        so the line does not touch the frame at 100%.
        """
        grand_total = float(sum(value for value in totals.values() if value > 0))
        if grand_total <= 0:
            applogger.info(
                "Pareto chart: the totals are not positive, so no cumulative "
                "line is drawn."
            )
            return

        # 'Other' is not in totals - it is the sum of everything not kept.
        values: list[float] = []
        for category in ordered:
            if category == OTHER_LABEL and category not in totals:
                kept = [c for c in ordered if c != OTHER_LABEL]
                values.append(grand_total - sum(totals.get(c, 0.0) for c in kept))
            else:
                values.append(totals.get(category, 0.0))

        cumulative = np.cumsum(values) / grand_total * 100.0
        color = str(options.get("cumulative_color", "#D32F2F") or "#D32F2F")
        positions = list(range(len(ordered)))

        twin = ax.twinx()
        line = twin.plot(
            positions,
            cumulative,
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=str(options.get("cumulative_label", "Cumulative %")),
            zorder=5,
        )
        twin.set_ylim(0, 105)
        twin.set_ylabel(str(options.get("cumulative_label", "Cumulative %")), color=color)
        twin.tick_params(axis="y", colors=color)

        reference = float(options.get("reference_percent", 80.0) or 0.0)
        if 0.0 < reference <= 100.0:
            twin.axhline(
                reference,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                zorder=4,
            )
            # Where the running total first clears the threshold: the number
            # the chart exists to produce, so it is stated rather than left to
            # be counted off the axis.
            crossed = int(np.searchsorted(cumulative, reference) + 1)
            if crossed <= len(ordered):
                applogger.info(
                    "Pareto chart: %d of %d categories reach %g%%.",
                    crossed,
                    len(ordered),
                    reference,
                )

        # The twin is drawn over the bars, so its background has to be see
        # through or it hides them.
        twin.set_zorder(ax.get_zorder() + 1)
        ax.patch.set_visible(True)
        twin.patch.set_visible(False)

        # One legend covering both axes.  The bar renderer builds its legend
        # before this line exists, so left alone the chart would name the bars
        # and say nothing about the curve running across them - the half of the
        # picture a reader is actually here for.
        bar_handles, bar_labels = ax.get_legend_handles_labels()
        if bar_handles or line:
            existing = ax.get_legend()
            if existing is not None:
                existing.remove()
            twin.legend(
                [*bar_handles, *line],
                [*bar_labels, *(artist.get_label() for artist in line)],
                loc="center right",
            )
