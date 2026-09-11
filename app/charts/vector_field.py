"""Two ways to draw a vector field: arrows at the samples, or the flow.

Both read the same four roles - ``x``, ``y``, ``u``, ``v``: a position and a
vector at it, which is what a measured or simulated field is. The third
member of the family, ``BarbsAxisRenderer``, already lives in
``app/charts/area.py`` and reads them identically; barbs encode magnitude in
flags rather than in length, which is what meteorology wants and physics
usually does not.

The two here differ in what they demand of the sampling, and the difference
is not cosmetic:

``Quiver``
    One arrow per row, wherever the rows happen to be. Nothing is assumed
    about the layout, so a field measured at scattered stations draws.

``Stream Plot``
    Integrates streamlines *through* the field, which means Matplotlib has
    to interpolate between samples - and its implementation can only do that
    on a rectangular grid with evenly spaced x and y. Rows that are not such
    a grid are refused with a message naming Quiver, the same way the
    gridded contour and surface renderers name their scattered
    counterparts (see app/charts/grids.py). Anything else would mean
    inventing the values between the samples and drawing streamlines
    through data nobody measured.

Neither inherits ``ScatterAxisRenderer``: a scatter reads x and y and draws
one mark per row, and a field reads four columns and draws an arrow whose
*direction* is the data. What is shared is the appearance vocabulary, and
that comes from ``kwarg_spec`` directly rather than from another renderer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib import rcParams

from app.charts.base import (
    ARTIST_KWARGS,
    CMAP_KWARGS,
    BaseAxisRenderer,
    SeriesData,
    merge,
    pick,
)
from app.charts.grids import pivot_to_grid
from app.data.series_frame import SeriesFrame
from app.logs.logger import applogger

#: The appearance keywords every field renderer here forwards. Deliberately
#: not the scatter's whole set: a colour cycle and error bars mean nothing to
#: an arrow whose colour comes from its own magnitude.
_FIELD_KWARGS: dict[str, Any] = pick(
    ARTIST_KWARGS, "alpha", "label", "zorder", "visible", "rasterized", "picker"
)

#: Even spacing, judged relatively: floating-point coordinates that came out
#: of a database round-trip are never exactly equal, and 1e-6 of the mean
#: step is far tighter than any real sampling irregularity.
_EVEN_SPACING_TOLERANCE: float = 1e-6


def _field_columns(df: SeriesFrame) -> tuple[np.ndarray, ...]:
    """Return x, y, u, v as float arrays, unparseable entries as NaN."""
    return tuple(
        pd.to_numeric(df[role], errors="coerce").to_numpy(dtype=float)
        for role in ("x", "y", "u", "v")
    )


def _finite_field(df: SeriesFrame) -> tuple[np.ndarray, ...] | None:
    """Return the rows where all four of x, y, u and v are finite."""
    columns = _field_columns(df)
    mask = np.logical_and.reduce([np.isfinite(column) for column in columns])
    if not mask.any():
        return None
    return tuple(column[mask] for column in columns)


def _is_evenly_spaced(values: np.ndarray) -> bool:
    """True when *values* are a regular sequence, within a relative tolerance."""
    if values.size < 2:
        return False
    steps = np.diff(values)
    mean_step = float(np.mean(steps))
    if mean_step == 0.0:
        return False
    return bool(
        np.allclose(steps, mean_step, rtol=_EVEN_SPACING_TOLERANCE, atol=0.0)
    )


class QuiverAxisRenderer(BaseAxisRenderer):
    """An arrow per sample: direction as angle, magnitude as length.

    Role columns:
        x, y, u, v   required. Position, then the vector's two components at
                     it. Rows with a non-finite value in any of the four are
                     dropped - an arrow needs all four to exist.
    """

    Name: str = "Quiver"
    Category: str = "Gridded data"
    Description: str = (
        "An arrow at each sample showing a vector field's direction and "
        "magnitude, read from u and v components. Draws wherever the samples "
        "are; use Stream Plot for the flow through an even grid."
    )
    Link: str = "https://matplotlib.org/stable/plot_types/arrays/quiver.html"

    RequiredRoles: list[str] = ["x", "y", "u", "v"]
    #: A fifth column colours the arrows through the colormap - wind speed,
    #: vorticity, error magnitude - which is the usual reason a quiver plot
    #: is preferred to barbs.
    OptionalRoles: list[str] = ["color"]

    Kwargs: dict[str, object] = merge(
        _FIELD_KWARGS,
        pick(CMAP_KWARGS, "cmap", "norm", "vmin", "vmax"),
        {
            "scale": {
                "default": None,
                "type": float,
                "min": 0.0,
                "group": "Arrows",
                "description": (
                    "Data units per arrow length unit. Larger makes the "
                    "arrows shorter. Empty lets Matplotlib choose from the "
                    "data, which is right until two axes have to be compared."
                ),
            },
            "width": {
                "default": None,
                "type": float,
                "min": 0.0,
                "max": 1.0,
                "step": 0.001,
                "decimals": 4,
                "group": "Arrows",
                "description": "Shaft width, as a fraction of the axis width.",
            },
            "headwidth": {
                "default": 3.0,
                "type": float,
                "min": 0.0,
                "max": 20.0,
                "step": 0.5,
                "group": "Arrows",
                "description": "Head width, in multiples of the shaft width.",
            },
            "headlength": {
                "default": 5.0,
                "type": float,
                "min": 0.0,
                "max": 20.0,
                "step": 0.5,
                "group": "Arrows",
                "description": "Head length, in multiples of the shaft width.",
            },
            "pivot": {
                "default": "tail",
                "type": ["tail", "mid", "middle", "tip"],
                "group": "Arrows",
                "description": (
                    "Which part of the arrow sits on the sample point. "
                    "\"mid\" centres it, which reads better on a dense grid."
                ),
            },
            "color": {
                "default": None,
                "type": str,
                "kind": "color",
                "group": "Appearance",
                "description": (
                    "One colour for every arrow. Ignored when a color role "
                    "column is mapped, which colours them individually."
                ),
            },
        },
    )

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        base_kwargs = self.get_kwargs(axis_options)

        for sd in series:
            style = dict(sd.style or {})
            if not style.get("visible", True) or not self.ensure_required_roles(sd.df):
                continue

            finite = _finite_field(sd.df)
            if finite is None:
                applogger.info("Series '%s' skipped: no finite vectors.", sd.name)
                continue
            x, y, u, v = finite

            kwargs = dict(base_kwargs)
            magnitudes = self._magnitudes(sd, x.size)
            if magnitudes is not None:
                # Positional, not a keyword: quiver's fifth positional
                # argument *is* the colour array, and passing both it and
                # ``color`` is what Matplotlib refuses.
                kwargs.pop("color", None)

            try:
                if magnitudes is not None:
                    ax.quiver(x, y, u, v, magnitudes, **kwargs)
                else:
                    kwargs.pop("cmap", None)
                    kwargs.pop("norm", None)
                    kwargs.pop("vmin", None)
                    kwargs.pop("vmax", None)
                    ax.quiver(x, y, u, v, **kwargs)
            except Exception:
                applogger.exception("Quiver failed to draw series '%s'.", sd.name)

        self.apply_annotations(ax, axis_options)

    def _magnitudes(self, sd: SeriesData, expected: int) -> np.ndarray | None:
        """Return the mapped colour column, aligned to the finite rows.

        None when no colour column was mapped, or when it has been left with
        a different number of finite rows than the vectors have: colouring
        the wrong arrows is worse than colouring none of them.
        """
        if "color" not in sd.df.columns:
            return None

        columns = _field_columns(sd.df)
        mask = np.logical_and.reduce([np.isfinite(column) for column in columns])
        values = pd.to_numeric(sd.df["color"], errors="coerce").to_numpy(dtype=float)
        values = values[mask]
        if values.size != expected or not np.isfinite(values).all():
            applogger.info(
                "Series '%s': the color column does not line up with the "
                "vectors, so the arrows are drawn in one colour.",
                sd.name,
            )
            return None
        return values


class StreamAxisRenderer(BaseAxisRenderer):
    """Streamlines traced through a vector field sampled on an even grid.

    Role columns:
        x, y, u, v   required, and the x/y positions must form a rectangular
                     grid with even spacing in each direction - see the
                     module docstring for why that is a demand and not a
                     preference.
    """

    Name: str = "Stream Plot"
    Category: str = "Gridded data"
    Description: str = (
        "Streamlines traced through a vector field, for seeing where a flow "
        "goes rather than what it does at each sample. Needs an evenly "
        "spaced grid; use Quiver for scattered samples."
    )
    Link: str = "https://matplotlib.org/stable/plot_types/arrays/streamplot.html"

    RequiredRoles: list[str] = ["x", "y", "u", "v"]
    OptionalRoles: list[str] = []

    #: Streamlines from two fields over one set of axes cross each other and
    #: mean nothing; the second is reported and skipped.
    MaxSeries: int | None = 1

    Kwargs: dict[str, object] = merge(
        pick(_FIELD_KWARGS, "zorder"),
        pick(CMAP_KWARGS, "cmap", "norm"),
        {
            "density": {
                "default": 1.0,
                "type": float,
                "min": 0.1,
                "max": 10.0,
                "step": 0.1,
                "group": "Streamlines",
                "description": (
                    "How closely the streamlines are seeded. Raise it for a "
                    "denser picture of the flow, at the cost of drawing time."
                ),
            },
            "linewidth": {
                "default": None,
                "type": float,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
                "group": "Streamlines",
                "description": "Streamline width. Empty uses the style's lines.linewidth.",
            },
            "color": {
                "default": None,
                "type": str,
                "kind": "color",
                "group": "Appearance",
                "description": (
                    "One colour for every streamline. Empty colours them by "
                    "the field's own magnitude through the colormap, which is "
                    "what makes a stream plot readable."
                ),
            },
            "arrowsize": {
                "default": 1.0,
                "type": float,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
                "group": "Streamlines",
                "description": "Size of the direction arrows along each line.",
            },
            "broken_streamlines": {
                "default": True,
                "type": bool,
                "group": "Streamlines",
                "description": (
                    "Let a streamline stop when it comes too close to "
                    "another. Turning this off draws every line to the edge "
                    "of the field, which is denser and slower."
                ),
            },
        },
    )

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        sd = self.single_series(
            series, reason="streamlines from two fields cross and mean nothing"
        )
        if sd is None:
            return
        grid = self._field_grid(sd)
        if grid is None:
            return
        x_axis, y_axis, u_grid, v_grid = grid

        kwargs = self.get_kwargs(self.merge_style(axis_options, sd.style or {}))
        colour = str(kwargs.pop("color", "") or "").strip()
        if colour:
            kwargs["color"] = colour
            kwargs.pop("cmap", None)
            kwargs.pop("norm", None)
        else:
            # The field's own magnitude, which is what a stream plot is
            # usually read for: where it speeds up and where it stalls.
            kwargs["color"] = np.hypot(u_grid, v_grid)
            kwargs.setdefault("cmap", rcParams["image.cmap"])

        try:
            ax.streamplot(x_axis, y_axis, u_grid, v_grid, **kwargs)
        except Exception:
            applogger.exception("Stream Plot failed to draw series '%s'.", sd.name)

        self.apply_annotations(ax, axis_options)

    def _field_grid(
        self, sd: SeriesData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (x axis, y axis, U, V), or None with a message saying why.

        Pivoted twice rather than once because ``pivot_to_grid`` answers "are
        these rows a grid?" for one value column at a time; u and v are
        sampled at the same points, so the second pivot is a formality that
        keeps the grid test in one place instead of half here.
        """
        u_grid = pivot_to_grid(sd.df, z_role="u")
        v_grid = pivot_to_grid(sd.df, z_role="v")
        if u_grid is None or v_grid is None:
            applogger.error(
                "Stream Plot needs the samples on a complete x/y grid: every "
                "x value paired with every y value exactly once. Use "
                "'Quiver' for a field measured at scattered points.",
                show_dialog=False,
                raise_error=False,
            )
            return None

        x_mesh, y_mesh, u_values = u_grid
        _x, _y, v_values = v_grid
        x_axis = x_mesh[0, :]
        y_axis = y_mesh[:, 0]

        if not (_is_evenly_spaced(x_axis) and _is_evenly_spaced(y_axis)):
            applogger.error(
                "Stream Plot needs evenly spaced x and y values: it "
                "interpolates between the samples, and Matplotlib can only "
                "do that on a regular grid. Use 'Quiver' to draw the samples "
                "themselves.",
                show_dialog=False,
                raise_error=False,
            )
            return None

        # A NaN never reaches here - it leaves the rows one short of a
        # complete grid, and pivot_to_grid has already refused them above.
        # An infinity does: it is a number as far as the pivot is concerned,
        # and streamplot draws nothing at all when it meets one. Read as no
        # flow, which is the honest reading of a sample that overflowed.
        u_values = np.where(np.isfinite(u_values), u_values, 0.0)
        v_values = np.where(np.isfinite(v_values), v_values, 0.0)
        return x_axis, y_axis, u_values, v_values
