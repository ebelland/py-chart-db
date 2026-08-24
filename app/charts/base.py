"""Base protocol and shared helpers for axis renderers.

Every chart type in ``app/charts`` implements :class:`BaseAxisRenderer`.  The
protocol carries the metadata the AST scanner reads statically (``Name``,
``Description``, ``RequiredRoles``, ``OptionalRoles``, ``Kwargs``) and the
colour/kwarg helpers that keep renderers consistent with each other.

Roles are DataFrame column names, not positional arguments: a renderer declares
the columns it needs and the series SQL is responsible for aliasing to them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np
import pandas as pd
from matplotlib import colormaps, rcParams


@dataclass(slots=True)
class SeriesData:
    """One series as a renderer sees it: its name, its rows, and its styling.

    ``roles`` is the series descriptor's own role map - role name to the
    source column the SQL aliased to it.  The DataFrame carries the aliases,
    so a renderer that needs the *original* column name (the Table renderer
    labels its columns with them) has nowhere else to get it.  Defaulted, so
    the many places that build a SeriesData with three arguments still do.
    """

    name: str
    df: pd.DataFrame
    style: dict
    roles: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Error bars
# ----------------------------------------------------------------------
# Kept at module level, not only on the class: renderers are re-executed from
# disk by the renderer scanner while ``app.charts.base`` stays cached in
# sys.modules, so a renderer edited while the app runs can be newer than the
# base module it imports.  Importing a named symbol fails with an ImportError
# that says which symbol is missing; reading a class attribute fails with an
# AttributeError buried in a class body.

# Optional role columns understood by every renderer that draws error bars.
# A symmetric role is one column; an asymmetric pair is two, and takes
# precedence when both are present.
ERROR_BAR_ROLES: dict[str, tuple[str, str, str]] = {
    # axis: (symmetric role, lower role, upper role)
    "x": ("xerr", "xerr_low", "xerr_high"),
    "y": ("yerr", "yerr_low", "yerr_high"),
}

# Matplotlib errorbar keywords a renderer may forward, with their editor
# metadata.  Renderers merge this into their own Kwargs.
ERROR_BAR_KWARGS: dict[str, object] = {
    "capsize": {
        "default": 0.0,
        "type": float,
        "min": 0.0,
        "max": 50.0,
        "group": "Error bars",
        "description": "Length of the error bar caps, in points. 0 draws no caps.",
    },
    "capthick": {
        "default": None,
        "type": float,
        "min": 0.0,
        "max": 20.0,
        "group": "Error bars",
        "description": "Thickness of the caps. Defaults to the bar width.",
    },
    "elinewidth": {
        "default": None,
        "type": float,
        "min": 0.0,
        "max": 20.0,
        "group": "Error bars",
        "description": "Line width of the error bars themselves.",
    },
    "ecolor": {
        "default": None,
        "type": str,
        "kind": "color",
        "group": "Error bars",
        "description": "Error bar colour. Defaults to the series colour.",
    },
    "errorevery": {
        "default": 1,
        "type": int,
        "min": 1,
        "max": 100_000,
        "group": "Error bars",
        "description": (
            "Draw an error bar every N points. Use this to keep a dense "
            "series readable instead of a solid block of bars."
        ),
    },
}


class BaseAxisRenderer(Protocol):
    """Base class for every chart renderer: one chart type, one subclass.

    A renderer is discovered, described and driven entirely through its class
    attributes, so adding a chart type means adding a module under
    ``app/charts`` and nothing else:

    ``Name``
        Display name, and the ``chart_type`` stored in the database.  Renaming
        it orphans existing axes unless an entry is added to
        ``axis_renderer_scanner.CHART_TYPE_ALIASES``.
    ``Description`` / ``Link``
        Shown in the chart picker and in the axis properties panel.
    ``RequiredRoles`` / ``OptionalRoles``
        Column names the series query must produce.  Roles *are* DataFrame
        column names: a series aliases its SQL to match (``SELECT t AS x``).
    ``Kwargs``
        Editable Matplotlib keyword arguments, as ``{name: metadata}``.  The
        metadata drives the generated editor UI - ``default``, ``type``,
        ``min``/``max``, ``kind`` (``color``, ``marker``, ...), ``group`` and
        ``description`` - so a new option needs no widget code.

    Subclasses override :meth:`render_axis` and read their options through
    :meth:`opt` / :meth:`get_kwargs`, never from ``options`` directly, so that
    per-series overrides and defaults resolve the same way everywhere.

    The class is a ``Protocol`` so that renderers can be type-checked without
    importing it at runtime; the scanner nonetheless matches on the literal
    base name, so a renderer must list ``BaseAxisRenderer`` in its bases even
    when it inherits behaviour from another renderer.
    """

    Name: str = "Base"
    #: Which family of plot this is, following Matplotlib's own taxonomy at
    #: https://matplotlib.org/stable/plot_types/ - "Pairwise data",
    #: "Statistical distributions", "Gridded data", "Irregularly gridded data",
    #: "3D and volumetric data".
    #:
    #: Borrowed rather than invented so that a renderer's family is decided by
    #: what it draws, not by whoever added it, and so the chart list can be read
    #: alongside the Matplotlib documentation the renderers wrap.  Inherited, so
    #: a renderer that extends another - ECDF extends Scatter - keeps the
    #: family unless it says otherwise.
    Category: str = "Pairwise data"
    Description: str = "Base class"
    Link:str=""
    RequiredRoles: list[str] = []
    OptionalRoles: list[str] = []
    Kwargs: dict[str, object] = {}

    def render_axis(self, ax, series: list[SeriesData]) -> None:
        """Draw *series* onto the Matplotlib axes *ax*.

        Called once per render with the series already loaded and ordered.
        Implementations own the axes completely: labels, limits and scales are
        applied afterwards by ``render_figure._apply_axis_runtime_options``.
        """

    def opt(self, name: str, options: dict) -> object:
        """Resolve one option: per-axis ``axis_kwargs`` first, then the default.

        Kept separate from :meth:`get_kwargs` for options that are consumed by
        the renderer itself rather than forwarded to Matplotlib.
        """
        meta = self.Kwargs.get(name, {})
        axis_kwargs = options.get("axis_kwargs", {})
        if isinstance(axis_kwargs, dict) and name in axis_kwargs:
            return axis_kwargs[name]
        return options.get(name, meta.get("default"))  # pyright: ignore[reportAttributeAccessIssue]

    def ensure_required_roles(self,df:pd.DataFrame) ->bool:
        """Return True when *df* carries every column this renderer needs.

        Checked before drawing so a mis-mapped series is skipped with a log
        entry instead of raising a KeyError in the middle of a figure.
        """
        for role in self.RequiredRoles:
            if role not in df.columns:
                return False
        return True

    def get_kwargs(self, options: dict) -> dict:
        """Build the matplotlib keyword arguments for this renderer.

        Keys resolving to None are dropped rather than forwarded.  Why: for
        Matplotlib "absent" and "None" are not the same thing — ``picker=None``
        reaches ``Line2D.set_pickradius(None)`` and raises *"pick radius should
        be a distance"*, while omitting the key simply uses the default.
        """
        kwargs = {}
        axis_kwargs = options.get("axis_kwargs", {})
        if not isinstance(axis_kwargs, dict):
            axis_kwargs = {}

        for name, meta in self.Kwargs.items():
            if name in axis_kwargs:
                value = axis_kwargs[name]
            else:
                value = options.get(name, meta.get("default"))  # pyright: ignore[reportAttributeAccessIssue]
            if value is None:
                continue
            kwargs[name] = value

        return kwargs

    # ------------------------------------------------------------------
    # Axis annotations
    # ------------------------------------------------------------------
    def apply_annotations(self, ax: Any, options: dict[str, Any]) -> None:
        """Apply axis-level Matplotlib annotations stored in ``options``.

        ``options["annotations"]`` is expected to be a list of dictionaries with
        ``x``, ``y``, ``type``, ``text`` and optional ``kwargs`` keys.  The
        helper deliberately keeps kwargs open-ended because ``Axes.annotate`` and
        ``Axes.text`` expose many useful placement and styling options.
        """
        annotations = options.get("annotations", [])
        if not isinstance(annotations, list):
            return
        for annotation in annotations:
            if isinstance(annotation, dict):
                self.apply_annotation(ax, annotation)

    def apply_annotation(self, ax: Any, annotation: dict[str, Any]) -> None:
        """Apply one text, boxed text or arrow annotation to *ax*."""
        try:
            x = float(annotation.get("x", 0.0))
            y = float(annotation.get("y", 0.0))
        except (TypeError, ValueError):
            return

        annotation_type = str(annotation.get("type", "text") or "text").lower()
        text = str(annotation.get("text", "") or "")
        kwargs = annotation.get("kwargs", {})
        if not isinstance(kwargs, dict):
            kwargs = {}
        kwargs = dict(kwargs)

        if annotation_type == "arrow":
            xytext = kwargs.pop("xytext", (10, 10))
            textcoords = kwargs.pop("textcoords", "offset points")
            arrowprops = kwargs.pop("arrowprops", None)
            if not isinstance(arrowprops, dict):
                arrowprops = {"arrowstyle": "->"}
            ax.annotate(
                text,
                xy=(x, y),
                xytext=xytext,
                textcoords=textcoords,
                arrowprops=arrowprops,
                **kwargs,
            )
            return

        if annotation_type == "boxed text":
            kwargs.setdefault(
                "bbox",
                {"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )

        ax.text(x, y, text, **kwargs)

    # ------------------------------------------------------------------
    # Error bars
    # ------------------------------------------------------------------
    ERROR_ROLES = ERROR_BAR_ROLES
    ERROR_KWARGS = ERROR_BAR_KWARGS

    def has_error_roles(self, df: pd.DataFrame) -> bool:
        """True when the frame carries any error column this renderer reads."""
        return any(
            role in df.columns
            for roles in self.ERROR_ROLES.values()
            for role in roles
        )

    def error_values(
        self,
        df: pd.DataFrame,
        axis: str,
        mask: Any = None,
    ) -> Any:
        """Return the ``xerr``/``yerr`` argument for one axis, or None.

        Shapes follow Matplotlib exactly:

        * symmetric -> a 1-D array, one half-width per point;
        * asymmetric -> a (2, N) array of ``[lower, upper]`` *distances* from
          the data point, not absolute positions.

        The asymmetric pair wins when both are present, because supplying both
        can only mean the pair is the more specific intent.  Negative values are
        clipped to zero: Matplotlib draws them, but a negative half-width is a
        data error, not a shorter bar.
        """
        symmetric_role, low_role, high_role = self.ERROR_ROLES[axis]

        def _column(name: str) -> Any:
            if name not in df.columns:
                return None
            values = pd.to_numeric(df[name], errors="coerce")
            if mask is not None:
                values = values[mask]
            return np.clip(np.nan_to_num(values.to_numpy(dtype=float)), 0.0, None)

        low = _column(low_role)
        high = _column(high_role)
        if low is not None or high is not None:
            # One half of an asymmetric pair is still usable: the missing side
            # is zero, which draws a one-sided bar.
            if low is None:
                low = np.zeros_like(high)
            if high is None:
                high = np.zeros_like(low)
            return np.vstack([low, high])

        return _column(symmetric_role)

    def error_kwargs(self, base_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract the errorbar-specific keywords from a renderer's kwargs."""
        return {
            name: base_kwargs[name]
            for name in self.ERROR_KWARGS
            if name in base_kwargs and base_kwargs[name] not in (None, "")
        }

    def palette_colors(self) -> list[Any]:
        """Return the active Matplotlib property-cycle colors.

        Keep colors in their native Matplotlib representation. Do not convert
        RGBA tuples to strings, because strings like ``"(1.0, 0.5, 0.0)"``
        are not valid Matplotlib color specifications.
        """
        colors = rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
        return list(colors)

    def series_color(self, style: dict[str, Any], layer_index: int = 0) -> Any:
        """Return an explicit style color or the rcParams cycle color."""
        style_color = str(style.get("color", "") or "").strip()
        if style_color:
            return style_color
        palette = self.palette_colors()
        return palette[int(layer_index) % len(palette)]

    def is_discrete_integer_color(self, values: Any) -> bool:
        """True only when the color field dtype is integer.

        Integer color fields are treated as discrete category ids and mapped to
        the rcParams palette. Float color fields are continuous and must use the
        colormap path, even when their values happen to be whole numbers.
        """
        if isinstance(values, pd.Series):
            return bool(pd.api.types.is_integer_dtype(values.dtype))

        array = np.asarray(values)
        return bool(np.issubdtype(array.dtype, np.integer))

    def map_integer_colors_to_palette(self, values: Any, fallback_color: str = "") -> list[str]:
        """Map integer category ids to the active Matplotlib color cycle."""
        numeric = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
        palette = self.palette_colors()
        fallback = fallback_color or palette[0]
        mapped: list[str] = []
        for value in numeric:
            if np.isfinite(value):
                color_index = (int(round(float(value))) - 1) % len(palette)
                mapped.append(palette[color_index])
            else:
                mapped.append(fallback)
        return mapped

    def map_continuous_colors_to_cmap(
        self,
        values: Any,
        *,
        cmap_name: str = "viridis",
        fallback_color: str = "",
    ) -> list[Any]:
        """Map continuous numeric values to explicit colors from a colormap."""
        numeric = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
        finite = numeric[np.isfinite(numeric)]
        palette = self.palette_colors()
        fallback = fallback_color or palette[0]
        if finite.size == 0:
            return [fallback for _value in numeric]
        low = float(finite.min())
        high = float(finite.max())
        span = high - low
        cmap = colormaps.get_cmap(cmap_name)
        mapped: list[Any] = []
        for value in numeric:
            if np.isfinite(value):
                normalized = 0.5 if span == 0.0 else (float(value) - low) / span
                mapped.append(cmap(normalized))
            else:
                mapped.append(fallback)
        return mapped

    def color_sequence_from_values(
        self,
        values: Any,
        *,
        fallback_color: str = "",
        cmap_name: str = "viridis",
    ) -> list[Any]:
        """Return colors for a column: integers use palette, floats use cmap."""
        if self.is_discrete_integer_color(values):
            return self.map_integer_colors_to_palette(values, fallback_color)
        return self.map_continuous_colors_to_cmap(
            values,
            cmap_name=cmap_name,
            fallback_color=fallback_color,
        )

    def first_color_from_values(
        self,
        values: Any,
        *,
        fallback_color: str = "",
        cmap_name: str = "viridis",
    ) -> Any:
        """Return the first resolved color from a color column."""
        colors = self.color_sequence_from_values(
            values,
            fallback_color=fallback_color,
            cmap_name=cmap_name,
        )
        return colors[0] if colors else fallback_color