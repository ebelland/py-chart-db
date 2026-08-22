"""Timeline: dated events on a baseline, each on its own stem.

https://matplotlib.org/stable/gallery/lines_bars_and_markers/timeline.html

This one *does* inherit ``ScatterAxisRenderer``, unlike the four in
``area_axis``, and the difference is the point.  A timeline is a scatter: one
mark per row at an x and a y, with the scatter's colours, sizes, alpha and
legend all meaning exactly what they usually mean.  What it adds is a y nobody
supplied - the alternating stem levels - a stem down to a baseline, and the
event text.  So the drawing is delegated upwards and this class only prepares
the frame and decorates afterwards, which is what inheriting is for.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib import dates as mdates

from app.charts.base import BaseAxisRenderer, SeriesData
from app.charts.scatter import ScatterAxisRenderer
from app.logs.logger import applogger
from app.utils.coercion import coerce_axis


class TimelineAxisRenderer(ScatterAxisRenderer, BaseAxisRenderer):
    """Events along a dated axis, stemmed alternately above and below.

    ``BaseAxisRenderer`` is named a second time on purpose.  The scanner
    discovers renderers by looking for classes that inherit it *directly*, so a
    subclass of another renderer is invisible without it - which is why
    ViolinAxisRenderer and HorizontalBarAxisRenderer list it too.
    """

    Name: str = "Timeline"
    Category: str = "Pairwise data"
    Description: str = (
        "Dated events on a baseline, each on a stem with its label, for "
        "release histories and anything else that is a list of moments."
    )
    Link: str = (
        "https://matplotlib.org/stable/gallery/lines_bars_and_markers/timeline.html"
    )

    #: Only the date is required.  The level is generated, and the text is
    #: optional - a timeline of unlabelled events is still a useful density
    #: picture of when things happened.
    RequiredRoles: list[str] = ["x"]
    OptionalRoles: list[str] = ["label", "y", "color", "size"]

    Kwargs: dict[str, object] = {
        **ScatterAxisRenderer.Kwargs,
        "levels": {
            "default": 6,
            "type": int,
            "min": 1,
            "max": 20,
            "group": "Timeline",
            "description": (
                "How many alternating stem heights to cycle through. More of "
                "them separates labels that would otherwise collide; the "
                "matplotlib example uses six."
            ),
        },
        "annotate": {
            "default": True,
            "type": bool,
            "group": "Timeline",
            "description": "Write each event's label at the top of its stem.",
        },
        "annotation_fontsize": {
            "default": 8.0,
            "type": float,
            "min": 4.0,
            "max": 20.0,
            "group": "Timeline",
            "description": "Size of the event labels.",
        },
        "stem_color": {
            "default": "#8C959B",
            "type": str,
            "kind": "color",
            "group": "Timeline",
            "description": "Colour of the stems and the baseline.",
        },
    }

    #: Consumed here rather than forwarded to ax.scatter.
    TIMELINE_ONLY: frozenset[str] = frozenset(
        {"levels", "annotate", "annotation_fontsize", "stem_color"}
    )

    def get_kwargs(self, options: dict) -> dict:
        """Drop the timeline's own options before the scatter forwards the rest.

        Filtering the incoming options is not enough: ``get_kwargs`` walks
        ``Kwargs`` and substitutes each declared *default* for anything absent,
        so ``levels`` would reach ``ax.scatter`` even when nobody set it.
        """
        kwargs = super().get_kwargs(options)
        for key in self.TIMELINE_ONLY:
            kwargs.pop(key, None)
        return kwargs

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        """Prepare stem levels, delegate the marks, then draw the stems."""
        level_count = max(1, int(self.opt("levels", options) or 6))
        annotate = bool(self.opt("annotate", options))
        fontsize = float(self.opt("annotation_fontsize", options) or 8.0)
        stem_color = str(self.opt("stem_color", options) or "#8C959B")

        prepared: list[SeriesData] = []
        annotations: list[tuple[np.ndarray, np.ndarray, list[str]]] = []
        temporal = False

        for sd in series:
            if not (sd.style or {}).get("visible", True) or "x" not in sd.df.columns:
                continue

            # A timeline's x is usually a date.  coerce_axis is the one place
            # that decides what a column is, so a date column here is read the
            # same way the time series renderer reads it - and a plain numeric
            # axis still works, which is what makes this usable for anything
            # ordered rather than only for dates.
            coerced, is_temporal = coerce_axis(sd.df["x"])
            temporal = temporal or is_temporal
            x = (
                mdates.date2num(coerced.to_numpy())
                if is_temporal
                else pd.to_numeric(coerced, errors="coerce").to_numpy(dtype=float)
            )

            if "y" in sd.df.columns:
                y = pd.to_numeric(sd.df["y"], errors="coerce").to_numpy(dtype=float)
            else:
                # The alternating heights of the matplotlib example: successive
                # events land on different levels, so neighbouring labels do
                # not overlap however close together the dates are.
                pattern = np.tile([-1, 1], level_count)[:level_count] * (
                    np.arange(level_count) // 2 + 1
                )
                y = pattern[np.arange(x.size) % level_count].astype(float)

            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                applogger.info("Series '%s' skipped: no dated events.", sd.name)
                continue

            frame = pd.DataFrame({"x": x[mask], "y": y[mask]})
            for role in ("color", "size"):
                if role in sd.df.columns:
                    frame[role] = sd.df[role].to_numpy()[mask]
            prepared.append(SeriesData(name=sd.name, df=frame, style=sd.style))

            if annotate and "label" in sd.df.columns:
                texts = [str(value) for value in sd.df["label"].to_numpy()[mask]]
                annotations.append((frame["x"].to_numpy(), frame["y"].to_numpy(), texts))

        if not prepared:
            return

        # The stems go on first so the markers sit over them.
        for sd in prepared:
            x = sd.df["x"].to_numpy()
            y = sd.df["y"].to_numpy()
            ax.vlines(x, 0.0, y, color=stem_color, linewidth=1.0, zorder=1)
        ax.axhline(0.0, color=stem_color, linewidth=1.2, zorder=1)

        scatter_options = {
            key: value
            for key, value in options.items()
            if key not in self.TIMELINE_ONLY
        }
        super().render_axis(ax, prepared, scatter_options)

        for x, y, texts in annotations:
            for position, level, text in zip(x, y, texts):
                ax.annotate(
                    text,
                    xy=(position, level),
                    xytext=(0, 6 if level >= 0 else -6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if level >= 0 else "top",
                    fontsize=fontsize,
                )

        # The stem levels are an arrangement, not a measurement, so a scale on
        # them would invite reading a value off the height.
        ax.yaxis.set_visible(False)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)

        if temporal:
            # Locator *and* formatter.  The scatter drew plain numbers, so the
            # axis still carries a numeric locator - handing that to a date
            # formatter labels ticks that were never placed on date boundaries,
            # which reads as plausible dates in the wrong places.
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
