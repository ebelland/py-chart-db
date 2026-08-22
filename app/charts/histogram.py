"""Histogram renderer with multiple data sets on one axis.

Modelled on Matplotlib's *histogram with multiple data sets* example
(https://matplotlib.org/stable/gallery/statistics/histogram_multihist.html).
Vertical by default; ``orientation`` switches the bars to horizontal.

The defining constraint is that all data sets must be passed to a **single**
``ax.hist`` call.  Only then does Matplotlib compute one shared set of bin edges
and lay the bars of each data set out side by side inside every bin; calling
``hist`` once per data set instead produces independently binned, mutually
overlapping bars - which is what makes hand-rolled "multi histograms" look
wrong.  Everything in this renderer is arranged to preserve that single call.

Data sets come from two places, and both can be mixed in one axis:

* every visible series contributes one data set;
* a series that carries a ``dataset`` column is split into one data set per
  distinct value in that column.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.charts.base import BaseAxisRenderer, SeriesData
from app.logs.logger import applogger
from app.utils.distribution_fit import (
    CURATED_DISTRIBUTIONS,
    curve_points,
    fits_for_spec,
)

# Matplotlib histtypes, split by whether they draw patches or a line outline.
_HISTTYPES: tuple[str, ...] = ("bar", "barstacked", "step", "stepfilled")
_ORIENTATIONS: tuple[str, ...] = ("vertical", "horizontal")


class HistogramAxisRenderer(BaseAxisRenderer):
    """Renderer for histograms over one or more data sets.

    Role columns:
        value    required numeric sample values
        dataset  optional grouping key; splits one series into several data sets
        weight   optional per-sample weight

    Honors per-series style fields:
        label, visible, show_in_legend, color, alpha, zorder
    """

    RequiredRoles: list[str] = ["value"]
    OptionalRoles: list[str] = ["dataset", "weight", "color"]

    Name: str = "Histogram"
    Category: str = "Statistical distributions"
    Description: str = "Histogram supporting multiple data sets."
    Link: str = (
        "https://matplotlib.org/stable/gallery/statistics/histogram_multihist.html"
    )

    Kwargs: dict[str, object] = {
        "orientation": {
            "default": "vertical",
            "type": list(_ORIENTATIONS),
            "group": "Appearance",
            "description": (
                "vertical: values on the x axis, counts upwards. "
                "horizontal: values on the y axis, counts to the right."
            ),
        },
        "bins": {
            "default": 20,
            "type": int,
            "min": 1,
            "max": 1000,
            "group": "Binning",
            "description": (
                "Number of equal-width bins shared by every data set. "
                "One shared binning is what allows the data sets to be compared."
            ),
        },
        "range_min": {
            "default": None,
            "type": float,
            "group": "Binning",
            "description": (
                "Lower edge of the binning range. Leave empty to use the "
                "smallest value across all data sets."
            ),
        },
        "range_max": {
            "default": None,
            "type": float,
            "group": "Binning",
            "description": (
                "Upper edge of the binning range. Leave empty to use the "
                "largest value across all data sets."
            ),
        },
        "histtype": {
            "default": "bar",
            "type": list(_HISTTYPES),
            "group": "Appearance",
            "description": (
                "bar: grouped bars side by side; barstacked: stacked bars; "
                "step: unfilled outline; stepfilled: filled outline."
            ),
        },
        "stacked": {
            "default": False,
            "type": bool,
            "group": "Appearance",
            "description": "Stack the data sets instead of placing them side by side.",
        },
        "density": {
            "default": False,
            "type": bool,
            "group": "Normalisation",
            "description": (
                "Normalise each data set to a probability density. Use this to "
                "compare data sets of different sizes."
            ),
        },
        "distribution_fit": {
            "default": "",
            "type": ["", "best", "top3", "top5", *CURATED_DISTRIBUTIONS],
            "group": "Distribution fit",
            "description": (
                "Fit continuous distributions to each data set and draw their "
                "densities over the bars. 'best' takes the top-ranked "
                "candidate and 'top3'/'top5' that many, which is how two "
                "close candidates are told apart by eye. A comma-separated "
                "list of scipy names draws exactly those. Setting this "
                "normalises the histogram to a density, since a probability "
                "density cannot be drawn over counts."
            ),
        },
        "distribution_fit_points": {
            "default": 256,
            "type": int,
            "min": 16,
            "max": 4096,
            "group": "Distribution fit",
            "description": (
                "How many points the fitted curve is drawn with. A density is "
                "a curve; sampling it once per bin would just redraw the bars."
            ),
        },
        "cumulative": {
            "default": False,
            "type": bool,
            "group": "Normalisation",
            "description": "Accumulate counts from the lowest bin upwards.",
        },
        "log": {
            "default": False,
            "type": bool,
            "group": "Normalisation",
            "description": "Use a logarithmic scale for the count axis.",
        },
        "rwidth": {
            "default": None,
            "type": float,
            "min": 0.05,
            "max": 1.0,
            "group": "Appearance",
            "description": "Bar thickness as a fraction of the bin size.",
        },
        "alpha": {
            "default": 0.75,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "group": "Appearance",
            "description": "Bar opacity from 0.0 (transparent) to 1.0 (opaque).",
        },
        "linewidth": {
            "default": None,
            "type": float,
            "min": 0.0,
            "max": 10.0,
            "group": "Appearance",
            "description": "Outline width of the bars.",
        },
        "edgecolor": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Bar outline colour. Leave empty for no explicit edge.",
        },
        "zorder": {
            "default": None,
            "type": float,
            "min": -1000.0,
            "max": 1000.0,
            "group": "Appearance",
            "description": "Drawing order; higher values are drawn on top.",
        },
        "show_legend": {
            "default": True,
            "type": bool,
            "group": "Legend",
            "description": "Show a legend entry for every data set.",
        },
        "count_label": {
            "default": "",
            "type": str,
            "group": "Legend",
            "description": (
                "Label for the count axis. Defaults to Count, or Density when "
                "density is enabled."
            ),
        },
    }

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _datasets_from_series(
        self,
        sd: SeriesData,
        layer_index: int,
    ) -> list[tuple[str, np.ndarray, np.ndarray | None, Any]]:
        """Return (label, values, weights, color) for each data set in a series.

        A ``dataset`` column splits the series; without it the series is one
        data set.  Groups are returned in sorted order so the legend and the
        colour assignment are stable across renders.
        """
        style = dict(sd.style or {})
        base_label = str(style.get("label", "") or "").strip() or sd.name.strip()
        color = self.series_color(style, layer_index)

        values = pd.to_numeric(sd.df["value"], errors="coerce")
        weights = (
            pd.to_numeric(sd.df["weight"], errors="coerce")
            if "weight" in sd.df.columns
            else None
        )

        finite = values.notna()
        if weights is not None:
            finite &= weights.notna()

        if "dataset" not in sd.df.columns:
            selected = values[finite].to_numpy(dtype=float)
            if selected.size == 0:
                return []
            picked_weights = (
                weights[finite].to_numpy(dtype=float) if weights is not None else None
            )
            return [(base_label, selected, picked_weights, color)]

        groups = sd.df["dataset"]
        output: list[tuple[str, np.ndarray, np.ndarray | None, Any]] = []

        for offset, key in enumerate(sorted(groups[finite].dropna().unique(), key=str)):
            mask = finite & (groups == key)
            selected = values[mask].to_numpy(dtype=float)
            if selected.size == 0:
                continue

            picked_weights = (
                weights[mask].to_numpy(dtype=float) if weights is not None else None
            )
            label = str(key) if len(sd.style or {}) == 0 else f"{base_label} · {key}"
            output.append(
                (label, selected, picked_weights, self.series_color(style, layer_index + offset))
            )

        return output

    @staticmethod
    def _bin_range(
        datasets: list[np.ndarray],
        options: dict[str, Any],
    ) -> tuple[float, float] | None:
        """Return an explicit (low, high) binning range, or None for automatic.

        A partially specified range is completed from the data rather than
        rejected, because "everything above 0" is a common and reasonable thing
        to ask for.
        """
        low = options.get("range_min")
        high = options.get("range_max")
        if low is None and high is None:
            return None

        combined = np.concatenate(datasets)
        try:
            resolved_low = float(low) if low is not None else float(np.min(combined))
            resolved_high = float(high) if high is not None else float(np.max(combined))
        except (TypeError, ValueError):
            applogger.warning(
                "Invalid histogram range (%r, %r); using the data range.",
                low,
                high,
                show_dialog=False,
                raise_error=False,
            )
            return None

        if resolved_low >= resolved_high:
            applogger.warning(
                "Histogram range_min (%s) is not below range_max (%s); using the data range.",
                resolved_low,
                resolved_high,
                show_dialog=False,
                raise_error=False,
            )
            return None

        return resolved_low, resolved_high

    # ------------------------------------------------------------------
    # Distribution fit
    # ------------------------------------------------------------------
    def _draw_distribution_fits(
        self,
        ax,
        *,
        datasets: list[np.ndarray],
        labels: list[str],
        colors: list[Any],
        fit_name: str,
        orientation: str,
        points: int,
        bin_range: tuple[float, float] | None,
    ) -> bool:
        """Draw one fitted density per data set.  Returns True if any was drawn.

        Per data set rather than over the pooled sample: the whole reason this
        renderer takes several is that they are different populations, and one
        curve across all of them would describe none of them.

        The curve is drawn in the data set's own colour so it reads as
        belonging to those bars, and is labelled with the distribution that was
        actually fitted - which for 'best' differs per data set, and is the
        thing the reader most needs told.
        """
        drawn = False
        for index, sample in enumerate(datasets):
            values = np.asarray(sample, dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 3:
                continue

            fits = fits_for_spec(values, fit_name)
            if not fits:
                applogger.info(
                    "No %s fit for data set %r; its density is not drawn.",
                    fit_name,
                    labels[index] if index < len(labels) else index,
                )
                continue

            low, high = (
                bin_range
                if bin_range is not None
                else (float(np.min(values)), float(np.max(values)))
            )
            base_color = colors[index] if index < len(colors) else None

            for order, fit in enumerate(fits):
                x, pdf = curve_points(fit, low, high, points)
                label = f"{fit.name} fit"
                if index < len(labels) and labels[index]:
                    label = f"{labels[index]}: {label}"

                # The first curve takes the data set's colour so it reads as
                # belonging to those bars; the rest are dashed and left to the
                # cycle, because several solid curves in one colour would be
                # indistinguishable from each other.
                style = {"color": base_color} if order == 0 else {"linestyle": "--"}

                # Orientation swaps which axis carries the values, so the curve
                # has to swap with the bars or it lands at right angles to them.
                if orientation == "horizontal":
                    ax.plot(pdf, x, linewidth=2.0, label=label, **style)
                else:
                    ax.plot(x, pdf, linewidth=2.0, label=label, **style)
                drawn = True

        return drawn

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_axis(self, ax, series: list[SeriesData], options: dict) -> None:
        """Draw every data set of every visible series in one hist call."""
        base_kwargs = self.get_kwargs(options)

        labels: list[str] = []
        datasets: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        colors: list[Any] = []
        any_weights = False
        legend_wanted = False

        for layer_index, sd in enumerate(series):
            style = dict(sd.style or {})
            if not bool(style.get("visible", True)):
                continue
            if not self.ensure_required_roles(sd.df):
                applogger.warning(
                    "Series '%s' skipped: required 'value' column not found.",
                    sd.name,
                )
                continue

            entries = self._datasets_from_series(sd, layer_index)
            if not entries:
                applogger.info("Series '%s' skipped: no finite values.", sd.name)
                continue

            if bool(style.get("show_in_legend", True)):
                legend_wanted = True

            for label, values, series_weights, color in entries:
                labels.append(label)
                datasets.append(values)
                colors.append(color)
                # Weights must line up positionally with the data sets, so an
                # unweighted data set contributes an all-ones array rather than
                # a gap in the list.
                if series_weights is None:
                    weights.append(np.ones_like(values))
                else:
                    weights.append(series_weights)
                    any_weights = True

        if not datasets:
            applogger.warning("Horizontal histogram: no data sets to draw.")
            return

        histtype = str(base_kwargs.get("histtype", "bar") or "bar").strip()
        if histtype not in _HISTTYPES:
            applogger.warning(
                "Unknown histtype %r; falling back to 'bar'.",
                histtype,
                show_dialog=False,
                raise_error=False,
            )
            histtype = "bar"

        orientation = str(base_kwargs.get("orientation", "vertical") or "vertical").strip().lower()
        if orientation not in _ORIENTATIONS:
            applogger.warning(
                "Unknown histogram orientation %r; falling back to 'vertical'.",
                orientation,
                show_dialog=False,
                raise_error=False,
            )
            orientation = "vertical"

        fit_name = str(base_kwargs.get("distribution_fit", "") or "").strip()
        density = bool(base_kwargs.get("density", False))
        if fit_name and not density:
            # Not an error and not worth a warning: a density over counts is
            # simply meaningless, and the alternative - drawing nothing - would
            # look like the option was ignored.
            density = True

        hist_kwargs: dict[str, Any] = {
            "bins": max(1, int(base_kwargs.get("bins", 20) or 20)),
            "orientation": orientation,
            "histtype": histtype,
            "stacked": bool(base_kwargs.get("stacked", False)),
            "density": density,
            "cumulative": bool(base_kwargs.get("cumulative", False)),
            "log": bool(base_kwargs.get("log", False)),
            "color": colors,
            "label": labels,
        }

        bin_range = self._bin_range(datasets, base_kwargs)
        if bin_range is not None:
            hist_kwargs["range"] = bin_range

        if any_weights:
            hist_kwargs["weights"] = weights

        for key in ("alpha", "rwidth", "linewidth", "edgecolor", "zorder"):
            value = base_kwargs.get(key)
            if value is not None and value != "":
                hist_kwargs[key] = value

        # rwidth is silently ignored by Matplotlib for step histtypes; drop it
        # rather than leave a control that appears to do nothing.
        if histtype in {"step", "stepfilled"}:
            hist_kwargs.pop("rwidth", None)

        try:
            ax.hist(datasets, **hist_kwargs)
        except Exception:
            applogger.exception("Horizontal histogram failed to draw.")
            return

        if fit_name:
            legend_wanted = (
                self._draw_distribution_fits(
                    ax,
                    datasets=datasets,
                    labels=labels,
                    colors=colors,
                    fit_name=fit_name,
                    orientation=orientation,
                    points=int(base_kwargs.get("distribution_fit_points", 256) or 256),
                    bin_range=bin_range,
                )
                or legend_wanted
            )

        # The count axis is whichever one the bars grow along, so the label
        # follows the orientation rather than being pinned to x.
        count_label = str(base_kwargs.get("count_label", "") or "").strip()
        if not count_label:
            count_label = "Density" if density else "Count"
        if orientation == "horizontal":
            ax.set_xlabel(count_label)
        else:
            ax.set_ylabel(count_label)

        if legend_wanted and bool(base_kwargs.get("show_legend", True)):
            ax.legend()
        # Draw descriptor annotations after renderer-owned artists.
        self.apply_annotations(ax, options or {})

