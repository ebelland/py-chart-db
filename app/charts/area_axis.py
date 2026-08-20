"""Four renderers from Matplotlib's basic plot types: stackplot, stairs,
fill_between and barbs.

https://matplotlib.org/stable/plot_types/basic/index.html

None of them inherits ``ScatterAxisRenderer``, and the reason is worth stating
because inheriting it is the obvious first thought.  What a scatter does is
read an x and a y per series and draw one mark per row; every renderer here
breaks one half of that.  ``stackplot`` needs *every* series at once, because a
stacked band's position depends on the ones below it.  ``fill_between`` reads
two y columns and draws a region rather than marks.  ``stairs`` reads bin edges
and one value fewer than it has edges.  ``barbs`` reads four columns.  Nothing
of the scatter's drawing survives contact with any of them, so inheriting it
would mean overriding ``render_axis`` entirely - which is inheritance for the
sake of a shared ancestor and buys nothing.

What *is* shared is the appearance vocabulary - alpha, zorder, label, picker -
and that is reused directly out of ``ScatterAxisRenderer.Kwargs``, the same way
the ECDF renderer already does.  Sharing the options without sharing the
drawing is the honest version of the relationship.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.charts.base_axis import BaseAxisRenderer, SeriesData
from app.charts.scatter_axis import ScatterAxisRenderer
from app.logs.logger import applogger

#: The options every renderer here understands, borrowed rather than retyped.
#: Only the ones that describe an artist's appearance: the scatter's colour
#: mapping and error bars mean nothing to a filled region or a wind barb.
_SHARED_KWARGS: dict[str, object] = {
    key: value
    for key, value in ScatterAxisRenderer.Kwargs.items()
    if key in {"alpha", "label", "zorder", "picker", "rasterized", "visible"}
}


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Return one column as floats, with unparseable entries as NaN."""
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


class StackplotAxisRenderer(BaseAxisRenderer):
    """Series stacked on top of one another, as bands over a shared x.

    The one renderer here that is not per-series: a band's lower edge is the
    sum of everything under it, so the whole set has to be drawn in one call.
    """

    Name: str = "Stack Plot"
    Category: str = "Pairwise data"
    Description: str = (
        "Series stacked into filled bands over a shared x, for showing how a "
        "total divides between its parts."
    )
    Link: str = "https://matplotlib.org/stable/plot_types/basic/stackplot.html"

    RequiredRoles: list[str] = ["x", "y"]
    OptionalRoles: list[str] = []

    Kwargs: dict[str, object] = {
        **_SHARED_KWARGS,
        "baseline": {
            "default": "zero",
            "type": ["zero", "sym", "wiggle", "weighted_wiggle"],
            "group": "Layout",
            "description": (
                "Where the stack sits. zero is the usual one; sym centres it "
                "on zero, and the two wiggle forms minimise slope for a "
                "streamgraph."
            ),
        },
        "show_legend": {
            "default": True,
            "type": bool,
            "group": "Legend",
            "description": "Show the legend naming each band.",
        },
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        """Stack every visible series over the x of the first one."""
        base_kwargs = self.get_kwargs(options)
        show_legend = bool(base_kwargs.pop("show_legend", True))

        usable = [
            sd
            for sd in series
            if (sd.style or {}).get("visible", True)
            and self.ensure_required_roles(sd.df)
            and not sd.df.empty
        ]
        if not usable:
            return

        # One x for the whole stack, taken from the first series.  Stacking
        # presumes a shared axis: bands sampled at different x cannot be added
        # to each other, and interpolating them here would invent data.
        x = _numeric(usable[0].df, "x")
        columns: list[np.ndarray] = []
        labels: list[str] = []
        for sd in usable:
            y = _numeric(sd.df, "y")
            if y.size != x.size:
                applogger.warning(
                    "Stack plot: series '%s' has %d points against the stack's "
                    "%d and is skipped; every band must share one x.",
                    sd.name,
                    y.size,
                    x.size,
                    show_dialog=False,
                    raise_error=False,
                )
                continue
            # A gap in a band would break the sum, so missing values read as
            # zero contribution rather than removing the row from every band.
            columns.append(np.nan_to_num(y, nan=0.0))
            labels.append(str((sd.style or {}).get("label") or sd.name))

        if not columns:
            return

        try:
            ax.stackplot(x, *columns, labels=labels, **base_kwargs)
        except Exception:
            applogger.exception("Stack plot failed to draw.")
            return

        if show_legend:
            ax.legend()
        # Draw descriptor annotations after renderer-owned artists.
        self.apply_annotations(ax, options or {})


class StairsAxisRenderer(BaseAxisRenderer):
    """A step outline, drawn from values and the edges between them.

    The natural shape for anything already binned - a histogram computed
    elsewhere, a rate held constant between readings - where a line would imply
    a slope the data does not claim.
    """

    Name: str = "Stairs"
    Category: str = "Statistical distributions"
    Description: str = (
        "A step outline over bin edges, for values that hold constant between "
        "them rather than sloping."
    )
    Link: str = "https://matplotlib.org/stable/plot_types/basic/stairs.html"

    #: ``y`` is the value per step.  ``x`` is optional and read as the *edges*,
    #: which is why there is one more of them than of the values.
    RequiredRoles: list[str] = ["y"]
    OptionalRoles: list[str] = ["x"]

    Kwargs: dict[str, object] = {
        **_SHARED_KWARGS,
        "fill": {
            "default": False,
            "type": bool,
            "group": "Appearance",
            "description": "Fill under the steps instead of drawing an outline.",
        },
        "orientation": {
            "default": "vertical",
            "type": ["vertical", "horizontal"],
            "group": "Layout",
            "description": "Which axis the values grow along.",
        },
        "linewidth": {
            "default": 1.8,
            "type": float,
            "min": 0.1,
            "max": 10.0,
            "group": "Appearance",
            "description": "Width of the step outline.",
        },
        "color": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Outline colour. Leave empty to follow the cycle.",
        },
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        for sd in series:
            style = dict(sd.style or {})
            if not style.get("visible", True) or not self.ensure_required_roles(sd.df):
                continue

            values = _numeric(sd.df, "y")
            if values.size == 0:
                continue

            kwargs = self.get_kwargs({**options, **style})
            kwargs.setdefault("label", style.get("label") or sd.name)

            edges = None
            if "x" in sd.df.columns:
                edges = _numeric(sd.df, "x")
                # Matplotlib wants one more edge than value.  A column of the
                # same length is the common case - it is the x of a line chart
                # - so the last step is closed by extrapolating one width
                # rather than refusing to draw.
                if edges.size == values.size:
                    step = edges[-1] - edges[-2] if edges.size > 1 else 1.0
                    edges = np.append(edges, edges[-1] + step)
                elif edges.size != values.size + 1:
                    applogger.warning(
                        "Stairs: series '%s' has %d edges for %d values, which "
                        "is neither; the edges are ignored.",
                        sd.name,
                        edges.size,
                        values.size,
                        show_dialog=False,
                        raise_error=False,
                    )
                    edges = None

            try:
                ax.stairs(values, edges, **kwargs)
            except Exception:
                applogger.exception("Stairs failed to draw series '%s'.", sd.name)
        # Draw descriptor annotations after renderer-owned artists.
        self.apply_annotations(ax, options or {})


class FillBetweenAxisRenderer(BaseAxisRenderer):
    """A filled region between two curves, or between one curve and a level.

    What a confidence band is: this renderer exists mostly so the fit and
    smoothing operations have somewhere to draw their uncertainty.
    """

    Name: str = "Fill Between"
    Category: str = "Pairwise data"
    Description: str = (
        "The region between two y curves over a shared x, for confidence "
        "bands and tolerance limits."
    )
    Link: str = "https://matplotlib.org/stable/plot_types/basic/fill_between.html"

    RequiredRoles: list[str] = ["x", "y"]
    #: ``y2`` is the other edge.  Without it the fill runs to ``baseline``,
    #: which is how "everything above zero" is drawn.
    OptionalRoles: list[str] = ["y2"]

    Kwargs: dict[str, object] = {
        **_SHARED_KWARGS,
        "baseline": {
            "default": 0.0,
            "type": float,
            "group": "Layout",
            "description": (
                "The level the fill runs to when a series has no y2 column."
            ),
        },
        "color": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Fill colour. Leave empty to follow the cycle.",
        },
        "edge_line": {
            "default": True,
            "type": bool,
            "group": "Appearance",
            "description": (
                "Draw the upper curve as a line over the fill. A band on its "
                "own reads as an area; with the line it reads as an estimate "
                "and its uncertainty, which is usually what it is."
            ),
        },
        "show_legend": {
            "default": True,
            "type": bool,
            "group": "Legend",
            "description": "Show the legend naming each band.",
        },
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        base = self.get_kwargs(options)
        baseline = float(base.pop("baseline", 0.0) or 0.0)
        edge_line = bool(base.pop("edge_line", True))
        show_legend = bool(base.pop("show_legend", True))
        drawn = False

        for sd in series:
            style = dict(sd.style or {})
            if not style.get("visible", True) or not self.ensure_required_roles(sd.df):
                continue

            x = _numeric(sd.df, "x")
            y = _numeric(sd.df, "y")
            lower = _numeric(sd.df, "y2") if "y2" in sd.df.columns else np.full_like(x, baseline)

            # Rows where either edge is missing are dropped rather than filled
            # through: a band drawn across a gap claims coverage that was never
            # measured.
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(lower)
            if not mask.any():
                applogger.info("Series '%s' skipped: no finite band.", sd.name)
                continue

            label = str(style.get("label") or sd.name)
            kwargs = {**base, **{k: v for k, v in style.items() if k in base or k == "color"}}
            kwargs.pop("visible", None)

            patch = ax.fill_between(
                x[mask], y[mask], lower[mask], label=label, **kwargs
            )
            if edge_line:
                # Same colour as the band it belongs to, taken from the patch
                # so the two cannot drift apart when the cycle assigns it.
                ax.plot(x[mask], y[mask], color=patch.get_facecolor()[0][:3], linewidth=1.6)
            drawn = True

        if drawn and show_legend:
            ax.legend()
        # Draw descriptor annotations after renderer-owned artists.
        self.apply_annotations(ax, options or {})


class BarbsAxisRenderer(BaseAxisRenderer):
    """Wind barbs: a direction and a magnitude at each point.

    Barbs rather than arrows because a barb encodes speed in its flags, so it
    stays readable where a field of arrows of different lengths does not.
    """

    Name: str = "Wind Barbs"
    Category: str = "Pairwise data"
    Description: str = (
        "A barb per point showing the direction and strength of a vector "
        "field, read from u and v components."
    )
    Link: str = "https://matplotlib.org/stable/plot_types/basic/barbs.html"

    RequiredRoles: list[str] = ["x", "y", "u", "v"]
    OptionalRoles: list[str] = []

    Kwargs: dict[str, object] = {
        **_SHARED_KWARGS,
        "length": {
            "default": 7.0,
            "type": float,
            "min": 2.0,
            "max": 20.0,
            "group": "Appearance",
            "description": "Length of a barb in points.",
        },
        "barbcolor": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Colour of the barb shaft and flags.",
        },
        "flip_barb": {
            "default": False,
            "type": bool,
            "group": "Appearance",
            "description": (
                "Flip which side the flags sit on. The convention is "
                "hemisphere-dependent, which is the only reason this exists."
            ),
        },
    }

    def render_axis(self, ax: Any, series: list[SeriesData], options: dict) -> None:
        base_kwargs = self.get_kwargs(options)

        for sd in series:
            style = dict(sd.style or {})
            if not style.get("visible", True) or not self.ensure_required_roles(sd.df):
                continue

            columns = [_numeric(sd.df, role) for role in ("x", "y", "u", "v")]
            mask = np.logical_and.reduce([np.isfinite(column) for column in columns])
            if not mask.any():
                applogger.info("Series '%s' skipped: no finite barbs.", sd.name)
                continue

            try:
                ax.barbs(*(column[mask] for column in columns), **base_kwargs)
            except Exception:
                applogger.exception("Wind barbs failed to draw series '%s'.", sd.name)
        # Draw descriptor annotations after renderer-owned artists.
        self.apply_annotations(ax, options or {})
