"""Empirical cumulative distribution function renderer.

An ECDF answers "what fraction of the sample is at most x?" without the one
choice a histogram forces on you - the bin width - which is why it is the
honest first look at a distribution: every sample is drawn, nothing is
smoothed, and two ECDFs can be compared by eye even when their sample sizes
differ.

Implemented as a transform on top of :class:`ScatterAxisRenderer` rather than
as a renderer of its own.  The ECDF *is* a scatter of (sorted value, running
fraction), so everything the scatter renderer already does - per-series colour,
marker, connecting line, legend handling, sorting, visibility - applies
unchanged, and this class only has to produce the y column.

``Matplotlib`` gained ``Axes.ecdf`` in 3.8, but it draws a step line and takes
none of the per-series styling this application stores, so it is not used here.

Roles:
    value    required numeric sample; the x axis of the result
    weight   optional per-sample weight, for pre-aggregated data

The classic staircase is a matter of style, not of renderer: set the series
line style to a solid line and its marker to none, and set ``drawstyle`` to
``steps-post``.  Points with markers are the better default for small samples,
where the staircase hides how few observations there are.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from app.charts.base_axis import BaseAxisRenderer, SeriesData
from app.charts.scatter_axis import ScatterAxisRenderer
from app.logs.logger import applogger
from app.utils.distribution_fit import (
    CURATED_DISTRIBUTIONS,
    curve_points,
    fits_for_spec,
)


def ecdf_points(
    values: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    complementary: bool = False,
    as_percent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (x, y) points of an ECDF.

    ``y[i]`` is the fraction of the sample at or below ``x[i]``, so the curve
    starts at 1/n and ends at exactly 1 - the convention that makes the last
    point land on the axis top instead of just short of it.

    Ties share the same y, the largest one: with duplicated values the ECDF
    jumps once by the whole weight of the tie, and drawing intermediate steps
    would claim a resolution the sample does not have.

    Weights let an already-aggregated table be plotted (one row per distinct
    value with its count) and give the same curve as the raw sample.
    """
    finite = np.isfinite(values)
    if weights is not None:
        finite &= np.isfinite(weights)

    x_values = values[finite]
    if x_values.size == 0:
        return np.empty(0), np.empty(0)

    order = np.argsort(x_values, kind="stable")
    x_values = x_values[order]

    if weights is None:
        cumulative = np.arange(1, x_values.size + 1, dtype=float)
        total = float(x_values.size)
    else:
        sorted_weights = np.asarray(weights, dtype=float)[finite][order]
        cumulative = np.cumsum(sorted_weights)
        total = float(cumulative[-1]) if cumulative.size else 0.0

    if total <= 0.0:
        return np.empty(0), np.empty(0)

    y_values = cumulative / total

    # Collapse ties onto their final value: np.unique with return_index on the
    # reversed array gives the last occurrence of each distinct x.
    unique_x, last_index = np.unique(x_values[::-1], return_index=True)
    keep = x_values.size - 1 - last_index
    x_values, y_values = unique_x, y_values[keep]

    if complementary:
        y_values = 1.0 - y_values
    if as_percent:
        y_values = y_values * 100.0

    return x_values, y_values


class EcdfAxisRenderer(ScatterAxisRenderer, BaseAxisRenderer):
    """Renderer for empirical cumulative distribution functions.

    ``BaseAxisRenderer`` is listed explicitly as a base even though
    ``ScatterAxisRenderer`` already provides it: the renderer scanner matches
    on the literal base name in the source, so a renderer that inherits from
    another one would otherwise not be discovered.
    """

    RequiredRoles: list[str] = ["value"]
    OptionalRoles: list[str] = ["weight"]

    Name: str = "ECDF"
    Category: str = "Statistical distributions"
    Description: str = "Empirical cumulative distribution function."
    Link: str = "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.ecdf.html"

    # Options this renderer consumes itself.  They must not reach ax.scatter,
    # which would reject them, and they are stripped before delegating.
    ECDF_ONLY_OPTIONS: frozenset[str] = frozenset(
        {"complementary", "as_percent", "distribution_fit", "distribution_fit_points"}
    )

    Kwargs: dict[str, object] = {
        "complementary": {
            "default": False,
            "type": bool,
            "group": "Distribution",
            "description": (
                "Draw the survival function 1-F(x) instead of F(x): the "
                "fraction *above* each value, which is the readable form when "
                "the interesting behaviour is in the upper tail."
            ),
        },
        "as_percent": {
            "default": False,
            "type": bool,
            "group": "Distribution",
            "description": "Label the y axis 0-100 instead of 0-1.",
        },
        "distribution_fit": {
            "default": "",
            "type": ["", "best", "top3", "top5", *CURATED_DISTRIBUTIONS],
            "group": "Distribution fit",
            "description": (
                "Fit continuous distributions to each series and draw their "
                "theoretical CDFs over the steps. This is the honest way to "
                "judge a fit: the ECDF has no bin width to choose, so the gap "
                "between the two curves is the KS statistic itself, visible."
            ),
        },
        "distribution_fit_points": {
            "default": 256,
            "type": int,
            "min": 16,
            "max": 4096,
            "group": "Distribution fit",
            "description": "How many points each fitted curve is drawn with.",
        },
        **{
            # The scatter renderer's own options apply unchanged: this
            # renderer only replaces the data, not the drawing.
            key: value
            for key, value in ScatterAxisRenderer.Kwargs.items()
        },
    }

    # One shared delegate: renderers are stateless, so an instance per
    # ECDF axis would buy nothing.
    _scatter = ScatterAxisRenderer()

    def _int_option(self, name: str, options: dict[str, Any], default: int = 0) -> int:
        """Return an option as int without Pylance object-conversion warnings."""
        value = cast(Any, self.opt(name, options))
        if value in (None, ""):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            applogger.warning(
                "Invalid ECDF option %s=%r; using %r",
                name,
                value,
                default,
                show_dialog=False,
                raise_error=False,
            )
            return default

    def _scatter_options(self, options: dict) -> dict:
        """Return *options* without the keys only this renderer understands."""
        cleaned = {
            key: value
            for key, value in options.items()
            if key not in self.ECDF_ONLY_OPTIONS
        }
        axis_kwargs = cleaned.get("axis_kwargs")
        if isinstance(axis_kwargs, dict):
            cleaned["axis_kwargs"] = {
                key: value
                for key, value in axis_kwargs.items()
                if key not in self.ECDF_ONLY_OPTIONS
            }
        return cleaned

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        """Convert every series to its ECDF, then draw it as a scatter."""
        complementary = bool(self.opt("complementary", options))
        as_percent = bool(self.opt("as_percent", options))

        transformed: list[SeriesData] = []
        for sd in series:
            frame = self._ecdf_frame(sd, complementary=complementary, as_percent=as_percent)
            if frame is None:
                continue
            transformed.append(SeriesData(name=sd.name, df=frame, style=sd.style))

        if not transformed:
            return

        # Drawn through a plain scatter renderer rather than through super().
        # Two things depend on it: ``ensure_required_roles`` has to check for
        # x and y (this class requires ``value``), and ``get_kwargs`` has to
        # iterate over the scatter options only, or ``complementary`` would be
        # forwarded to ax.scatter as a keyword and raise.
        self._scatter.render_axis(ax, transformed, self._scatter_options(options))

        fit_spec = str(self.opt("distribution_fit", options) or "").strip()
        if fit_spec:
            self._draw_fitted_cdfs(
                ax,
                series=series,
                fit_spec=fit_spec,
                complementary=complementary,
                as_percent=as_percent,
                points=self._int_option("distribution_fit_points", options, 256),
            )

        if not str(ax.get_ylabel() or "").strip():
            if complementary:
                ax.set_ylabel("1 - F(x) [%]" if as_percent else "1 - F(x)")
            else:
                ax.set_ylabel("F(x) [%]" if as_percent else "F(x)")

        # drawn afterward, so draw descriptor annotations last.

    def _draw_fitted_cdfs(
        self,
        ax: Any,
        *,
        series: list[SeriesData],
        fit_spec: str,
        complementary: bool,
        as_percent: bool,
        points: int,
    ) -> None:
        """Overlay the theoretical CDF of each fitted distribution.

        Drawn from the raw values rather than from the ECDF frame built above:
        the fit describes the sample, and the ECDF is already a transform of
        it.  Fitting the transform would be fitting the picture.

        The curve follows whatever the steps are doing - complemented and
        rescaled to percent alongside them - because a theoretical curve on a
        different vertical scale from the data it describes is worse than none.
        """
        for sd in series:
            if "value" not in sd.df.columns:
                continue
            values = pd.to_numeric(sd.df["value"], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 3:
                continue

            fits = fits_for_spec(values, fit_spec)
            if not fits:
                applogger.info(
                    "Series '%s': no %s fit, so no theoretical CDF is drawn.",
                    sd.name,
                    fit_spec,
                )
                continue

            low, high = float(np.min(values)), float(np.max(values))
            for order, fit in enumerate(fits):
                x, cdf = curve_points(fit, low, high, points, cumulative=True)
                if complementary:
                    cdf = 1.0 - cdf
                if as_percent:
                    cdf = cdf * 100.0
                ax.plot(
                    x,
                    cdf,
                    linewidth=1.8,
                    linestyle="-" if order == 0 else "--",
                    label=f"{sd.name}: {fit.name} fit",
                    zorder=4,
                )

    def _ecdf_frame(
        self,
        sd: SeriesData,
        *,
        complementary: bool,
        as_percent: bool,
    ) -> pd.DataFrame | None:
        """Return a two-column x/y frame for one series, or None if unusable."""
        if "value" not in sd.df.columns:
            applogger.warning(
                "Series '%s' skipped: the ECDF renderer needs a 'value' role.",
                sd.name,
                show_dialog=False,
                raise_error=False,
            )
            return None

        values = pd.to_numeric(sd.df["value"], errors="coerce").to_numpy(dtype=float)
        weights = None
        if "weight" in sd.df.columns:
            weights = pd.to_numeric(sd.df["weight"], errors="coerce").to_numpy(dtype=float)

        x_values, y_values = ecdf_points(
            values,
            weights=weights,
            complementary=complementary,
            as_percent=as_percent,
        )
        if x_values.size == 0:
            applogger.info("Series '%s' skipped: no finite values.", sd.name)
            return None

        return pd.DataFrame({"x": x_values, "y": y_values})
