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
from typing import Any, ClassVar, Protocol
import numpy as np
import pandas as pd
from matplotlib import colormaps, rcParams
from matplotlib.colors import to_rgba

from app.charts import kwarg_spec
from app.data.series_frame import SeriesFrame
from app.charts.kwarg_spec import (
    ARTIST_ADVANCED_KWARGS,
    ARTIST_KWARGS,
    CMAP_KWARGS,
    DEFAULT,
    FIT_OPTIONS,
    LEGEND_OPTIONS,
    LINE_KWARGS,
    PATCH_KWARGS,
    RCPARAM,
    STYLE_DEFAULT,
    VIEW_OPTIONS,
    merge,
    pick,
)
from app.logs.logger import applogger

#: Kept as aliases of the shared vocabulary, which now owns the conversion.
_TRUE_WORDS = kwarg_spec.TRUE_WORDS
_FALSE_WORDS = kwarg_spec.FALSE_WORDS

# Re-exported deliberately.  Renderers are re-executed from disk by the
# renderer scanner while ``app.charts.base`` stays cached in sys.modules, so a
# renderer edited while the application runs can be newer than the base module
# it imports; a missing name then fails as an ImportError naming the name.
# Every renderer already imports from base, so the shared vocabulary is
# reachable the same way rather than through a second import line each of them
# would have to grow.
__all__ = [
    "ARTIST_ADVANCED_KWARGS",
    "ARTIST_KWARGS",
    "BaseAxisRenderer",
    "CMAP_KWARGS",
    "DEFAULT",
    "ERROR_BAR_KWARGS",
    "ERROR_BAR_ROLES",
    "FIT_OPTIONS",
    "LEGEND_OPTIONS",
    "LINE_KWARGS",
    "PATCH_KWARGS",
    "RCPARAM",
    "STYLE_DEFAULT",
    "SeriesData",
    "SeriesFrame",
    "VIEW_OPTIONS",
    "merge",
    "pick",
]


@dataclass(slots=True)
class SeriesData:
    """One series as a renderer sees it: its name, its rows, and its styling.

    ``df`` is a :class:`~app.data.series_frame.SeriesFrame` (todo.txt P2-17) -
    columns of numpy arrays, not a ``pd.DataFrame`` - but every renderer keeps
    reading it the same way: ``sd.df["x"]``, ``"x" in sd.df.columns``,
    ``sd.df.loc[mask, "color"]``. See that module's docstring for why the
    switch is safe to make without touching those call sites.

    ``roles`` is the series descriptor's own role map - role name to the
    source column the SQL aliased to it.  ``df`` carries the aliases, so a
    renderer that needs the *original* column name (the Table renderer labels
    its columns with them) has nowhere else to get it.  Defaulted, so the
    many places that build a SeriesData with three arguments still do.
    """

    name: str
    df: SeriesFrame
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
        Matplotlib keyword arguments forwarded verbatim to the plot call, as
        ``{name: metadata}``.  The metadata drives the generated editor UI -
        ``default``, ``type``, ``min``/``max``, ``kind`` (``color``,
        ``marker``, ...), ``group`` and ``description`` - so a new keyword
        needs no widget code.  A name in here *is* passed to Matplotlib, so
        anything Matplotlib would reject belongs in ``Options`` instead.
    ``Options``
        Settings the renderer consumes itself and never forwards: whether to
        draw a legend, how many tick labels to keep, which fit to overlay.
        Same metadata, same editor, opposite destination.
    ``MaxSeries``
        How many series this renderer can draw on one axes, or None for any
        number.  Read by the New plot dialog, which offers a new *axis* per
        series rather than a second series the renderer would drop.

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

    #: Settings the renderer reads itself, never forwarded to Matplotlib.
    #:
    #: The two used to be one dict, and the boundary lived in a removal list
    #: inside each renderer - ``bar`` dropped eleven names before calling
    #: ``bar()``, ``area`` popped ``show_legend``, ``broken_bar`` popped
    #: ``band_height``.  A name forgotten there did not fail in the editor
    #: that offered it; it failed inside Matplotlib, at draw time, as an
    #: unexpected keyword argument.  Declaring the destination makes it the
    #: schema's job: :meth:`get_kwargs` only ever reads ``Kwargs``, so an
    #: option cannot reach Matplotlib by being forgotten.
    #:
    #: A renderer still on the single dict keeps working - :meth:`opt` falls
    #: back to ``Kwargs`` - so the conversion is one renderer at a time.
    Options: dict[str, object] = {}

    #: Series this renderer can draw on one axes; None means any number.
    #:
    #: Five renderers already enforce a limit of one - a surface, a table, a
    #: pie and the two contour maps each keep the first drawable series and
    #: log the rest away - but they only find out at render time, by which
    #: point the series exist in the database and the person who added them
    #: has been shown a chart that quietly lost most of them.  Declaring the
    #: limit here lets the New plot dialog do the honest thing instead: put
    #: each series on an axis of its own.  The renderer-side guard stays as
    #: the last line of defence, because the dialog is not the only way a
    #: series reaches an axis - the fit and statistics dialogs add derived
    #: ones too.
    MaxSeries: int | None = None

    def render_axis(self, ax, series: list[SeriesData]) -> None:
        """Draw *series* onto the Matplotlib axes *ax*.

        Called once per render with the series already loaded and ordered.
        Implementations own the axes completely: labels, limits and scales are
        applied afterwards by ``render_figure._apply_axis_runtime_options``.
        """

    #: Where a value may be found, most specific first.
    #:
    #: ``axis_kwargs`` is where the properties panel writes everything it
    #: edits, options included, so it stays in the chain for both schemas -
    #: a figure saved before the split keeps the settings it had.  The flat
    #: options dict is the older location still, and is read last.
    VALUE_SOURCES: tuple[str, ...] = ("renderer_options", "axis_kwargs")

    def _sources(self, options: dict) -> list[dict]:
        """Return the places to look for a value, most specific first."""
        found = [
            options.get(name)
            for name in self.VALUE_SOURCES
            if isinstance(options.get(name), dict)
        ]
        found.append(options)
        return found  # pyright: ignore[reportReturnType]

    def opt(self, name: str, options: dict) -> object:
        """Resolve one setting the renderer consumes itself.

        Reads :attr:`Options` first and falls back to :attr:`Kwargs`, so a
        renderer that has not been split yet resolves exactly as before.
        Returns the value raw; :meth:`opt_typed` is the converting version.
        """
        spec = self.Options if name in self.Options else self.Kwargs
        return kwarg_spec.resolve_one(spec, name, self._sources(options))

    def merge_style(self, options: dict, style: dict) -> dict:
        """Overlay one series' style on the axis options.

        The sub-dicts are merged key by key rather than replaced, so a series
        overriding its colour keeps the axis-wide alpha.  Seven renderers had
        their own copy of this - ``bar``, ``contour``, ``broken_bar``,
        ``text``, ``table`` and both surfaces - identical down to the variable
        names.
        """
        merged = dict(options or {})
        for name in self.VALUE_SOURCES:
            nested = dict(merged.get(name, {}) or {})
            nested.update(style.get(name, {}) or {})
            if nested:
                merged[name] = nested
        for key, value in (style or {}).items():
            if key not in self.VALUE_SOURCES:
                merged[key] = value
        return merged


    def valid_series(
        self,
        series: list["SeriesData"],
        *,
        require_roles: bool = True,
        require_rows: bool = True,
    ) -> list["SeriesData"]:
        """Return the series worth drawing: visible, mapped, and not empty.

        The same three conditions were written out in fifteen renderers and
        then in six more as they were added (todo.txt P2-14), which is
        twenty-one places for one of them to be forgotten - and the one
        that gets forgotten is ``visible``, because a hidden series looks
        exactly like a series until you hide it.

        ``require_roles`` is off for a renderer that draws whatever columns
        it is given rather than named ones - the Table renderer - and
        ``require_rows`` for one that has something to say about an empty
        frame.
        """
        return [
            sd
            for sd in series
            if (sd.style or {}).get("visible", True)
            and (not require_roles or self.ensure_required_roles(sd.df))
            and (not require_rows or not sd.df.empty)
        ]

    def single_series(
        self,
        series: list["SeriesData"],
        *,
        reason: str = "",
        **kwargs: Any,
    ) -> "SeriesData | None":
        """Return the one series to draw, reporting the ones skipped.

        For the renderers that declare ``MaxSeries = 1``: a contour map, a
        surface, a pie, a filled mesh. A second one drawn over the first
        covers it rather than being compared with it, so the extras are
        named in the log instead of silently vanishing - "nothing happened
        when I ticked the second series" is otherwise a mystery from the
        outside.

        *reason* completes the sentence "the extras were not drawn - ...",
        because why differs: a surface occludes, a pie has one circle to
        divide, a stream plot's lines would cross.
        """
        valid = self.valid_series(series, **kwargs)
        if not valid:
            return None

        if len(valid) > 1:
            applogger.info(
                "%s draws one series; %d more on this axis were not drawn%s",
                self.Name,
                len(valid) - 1,
                f" - {reason}." if reason else ".",
            )
        return valid[0]

    def ensure_required_roles(self, df: SeriesFrame) -> bool:
        """Return True when *df* carries every column this renderer needs.

        Checked before drawing so a mis-mapped series is skipped with a log
        entry instead of raising a KeyError in the middle of a figure.
        """
        for role in self.RequiredRoles:
            if role not in df.columns:
                return False
        return True

    #: Returned by :meth:`_coerce_option` for a value that cannot be made
    #: into the type the option declares, and so must not be forwarded.
    _UNCONVERTIBLE: ClassVar[object] = kwarg_spec.UNCONVERTIBLE

    @classmethod
    def _coerce_option(cls, value: Any, meta: dict) -> Any:
        """Return *value* as the type the option declares.

        The conversion itself lives in ``kwarg_spec`` now, so that
        :func:`~app.charts.kwarg_spec.resolve` applies exactly the same rules
        to a forwarded keyword as this applies to one a renderer reads for
        itself - two implementations of "is this a number?" is how they drift.
        Kept as a method because renderers call it.
        """
        return kwarg_spec.coerce(value, meta if isinstance(meta, dict) else {})

    def opt_typed(self, name: str, options: dict) -> Any:
        """Like :meth:`opt`, but converted to the type the option declares.

        For a renderer-owned option that is still handed to Matplotlib.
        :meth:`opt` stays raw because several of them are deliberately
        free-form - contour ``levels`` is a count or a list, ``linestyles`` is
        one of four words - but a numeric one has to arrive as a number:
        ``linewidths="0.5"`` is a *string* to Matplotlib, which reads it as
        the sequence of characters ``0``, ``.``, ``5`` and draws three lines
        of nonsense widths.

        Returns None when the value cannot be converted, which every caller
        already treats as "not set".
        """
        spec = self.Options if name in self.Options else self.Kwargs
        return kwarg_spec.resolve_one(spec, name, self._sources(options), typed=True)

    def get_kwargs(self, options: dict) -> dict:
        """Build the Matplotlib keyword arguments for this renderer.

        Only :attr:`Kwargs` is read, so the result is forwardable as it
        stands: there is no removal list to keep, and an option the renderer
        consumes cannot arrive here by being forgotten.

        Keys resolving to None, to an empty string or to ``DEFAULT`` are
        dropped rather than forwarded.  For Matplotlib "absent" and "None" are
        not the same thing - ``picker=None`` reaches
        ``Line2D.set_pickradius(None)`` and raises *"pick radius should be a
        distance"*, while omitting the key uses the default - and dropping is
        also what lets the style sheet decide, which is the whole meaning of
        ``DEFAULT``.

        Values are coerced to the type each entry declares; one that cannot be
        is dropped with a log entry rather than forwarded, on the same
        principle as the rest of the option handling - a typo should cost the
        option, not the chart.
        """
        dropped: list[tuple[str, Any, Any]] = []
        kwargs = kwarg_spec.resolve(self.Kwargs, self._sources(options), dropped)
        for name, value, declared in dropped:
            applogger.debug(
                "%s: option %r=%r is not a %s; leaving it out of the chart.",
                type(self).__name__,
                name,
                value,
                getattr(declared, "__name__", "value"),
            )
        return kwargs

    # ------------------------------------------------------------------
    # Axis annotations
    # ------------------------------------------------------------------
    def apply_annotations(self, ax: Any, options: dict[str, Any]) -> None:
        """Draw the decorations the descriptor asks for: annotations, lines.

        ``options["annotations"]`` is expected to be a list of dictionaries with
        ``x``, ``y``, ``type``, ``text`` and optional ``kwargs`` keys.  The
        helper deliberately keeps kwargs open-ended because ``Axes.annotate`` and
        ``Axes.text`` expose many useful placement and styling options.

        Reference lines are drawn from here rather than from a call of their
        own: every renderer already ends with ``apply_annotations``, and a
        second method would have to be added to twenty-nine of them - one of
        which would be forgotten, and the lines would then be missing from
        that chart type for no reason anyone could see.
        """
        annotations = options.get("annotations", [])
        if isinstance(annotations, list):
            for annotation in annotations:
                if isinstance(annotation, dict):
                    self.apply_annotation(ax, annotation)

        self.apply_reference_lines(ax, options)

    def apply_reference_lines(self, ax: Any, options: dict[str, Any]) -> None:
        """Draw the full-width/height lines stored in ``options["lines"]``.

        Each entry is ``{"orientation": "vertical"|"horizontal", "value":
        float, "kwargs": {...}}``. A threshold, a specification limit, a
        target: things that belong to the *axes* rather than to any series,
        which is why they are stored on the axis and not drawn as a
        two-point series.

        The value is in data coordinates and the line spans the axes, so it
        stays put when the data changes underneath it - which is the whole
        difference from drawing it as data.
        """
        lines = options.get("lines", [])
        if not isinstance(lines, list):
            return
        for line in lines:
            if isinstance(line, dict):
                self.apply_reference_line(ax, line)

    def apply_reference_line(self, ax: Any, line: dict[str, Any]) -> None:
        """Draw one reference line, or nothing if it does not describe one."""
        try:
            value = float(line.get("value"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            applogger.warning(
                "Skipping a reference line with no usable value: %r",
                line.get("value"),
                show_dialog=False,
                raise_error=False,
            )
            return
        if not np.isfinite(value):
            return

        kwargs = line.get("kwargs", {})
        kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
        # Not "_nolegend_": a threshold with a label is worth a legend entry,
        # and one without a label already stays out of it. This only stops
        # Matplotlib from inventing "_child3".
        kwargs.setdefault("label", "_nolegend_" if not kwargs.get("label") else kwargs["label"])

        orientation = str(line.get("orientation", "vertical") or "vertical").lower()
        draw = ax.axhline if orientation.startswith("h") else ax.axvline
        try:
            draw(value, **kwargs)
        except Exception:  # noqa: BLE001 - a bad kwarg costs the line, not the chart
            applogger.exception("A reference line could not be drawn: %r", line)

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

        # Arrow-only placement kwargs. Popped unconditionally, not just in
        # the "arrow" branch below: the Overlay panel edits one kwargs dict
        # per annotation, so switching an annotation from arrow to text/boxed
        # text leaves these behind in it - and Axes.text() has no such
        # properties and raises AttributeError if one is still there.
        xytext = kwargs.pop("xytext", (10, 10))
        textcoords = kwargs.pop("textcoords", "offset points")
        arrowprops = kwargs.pop("arrowprops", None)

        if annotation_type == "arrow":
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

    def has_error_roles(self, df: SeriesFrame) -> bool:
        """True when the frame carries any error column this renderer reads."""
        return any(
            role in df.columns
            for roles in self.ERROR_ROLES.values()
            for role in roles
        )

    def error_values(
        self,
        df: SeriesFrame,
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

    def series_linestyle(
        self, style: dict[str, Any], options: dict[str, Any]
    ) -> tuple[str | None, bool]:
        """Resolve a series' effective line style: value, and whether to draw one.

        A concrete style value - including "none", the explicit no-line
        choice - wins outright and is returned unchanged, exactly as before
        this method existed. A key that is simply absent keeps meaning "no
        line" too, for the same reason: two chart types (Scatter's marker-only
        default, in particular) already depend on that as their shape, and a
        long-missing key is not a person asking for anything.

        Only the DEFAULT sentinel - the "Default" entry in the Line style
        combo - means "I want whatever the axis or the style sheet says".
        That falls back to the axis' own ``linestyle`` option (set by the
        Function/Fit-style series operations, which want every series they
        add using the same linestyle) and finally to the active style
        sheet's ``lines.linestyle``, returned as ``(None, ...)`` so the
        caller omits the keyword rather than restating whatever rcParams
        says right now - a later style change should still take effect.
        """
        raw = style.get("linestyle")
        if isinstance(raw, str) and raw.strip().lower() == DEFAULT:
            axis_value = options.get("linestyle")
            if isinstance(axis_value, str) and axis_value.strip():
                value = axis_value.strip()
                return value, value.lower() not in ("", "none")
            rc_value = str(rcParams.get("lines.linestyle", "-") or "").strip().lower()
            return None, rc_value not in ("", "none")

        value = str(raw or "").strip()
        return value, value.lower() not in ("", "none")

    def series_marker(
        self,
        style: dict[str, Any],
        options: dict[str, Any],
        *,
        rcparam: str = "lines.marker",
    ) -> tuple[str | None, bool]:
        """Resolve a series' effective marker: value, and whether to draw one.

        Mirrors :meth:`series_linestyle`. The Marker combo's "None" entry is
        the empty string, which is also what an absent key already reads as
        - both keep meaning "no marker", unchanged. Only the DEFAULT
        sentinel cascades through the axis' own ``marker`` option to the
        active style sheet.

        ``rcparam`` names which rcParam that last step reads: Matplotlib
        gives ``ax.scatter`` its own default, ``scatter.marker`` (usually
        ``"o"``), separate from the plain-line default ``lines.marker``
        (usually ``"None"``) - a caller drawing markers with ``scatter``
        rather than ``plot`` should pass ``"scatter.marker"`` so "Default"
        means what that call would draw on its own.
        """
        raw = style.get("marker")
        if isinstance(raw, str) and raw.strip().lower() == DEFAULT:
            axis_value = options.get("marker")
            if isinstance(axis_value, str) and axis_value.strip():
                return axis_value.strip(), True
            rc_value = str(rcParams.get(rcparam, "") or "").strip()
            return (None, False) if rc_value.lower() in ("", "none") else (None, True)

        value = str(raw or "").strip()
        return value, value != ""

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
        """Map integer category ids to the active Matplotlib color cycle.

        The gather is done by NumPy rather than by a Python loop over the
        points: the index arithmetic - round, minus one, modulo the palette
        length - is the same three operations whether it runs once per point
        in Python or once over the whole array in C, and at 300 000 points
        that is 144 ms against 6.

        The lookup table is filled entry by entry on purpose. ``np.asarray``
        of a palette whose entries are RGBA tuples builds an (n, 4) array,
        not the flat array of objects the gather needs, and a style sheet is
        free to write its cycle either way.
        """
        numeric = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
        palette = self.palette_colors()
        fallback = fallback_color or palette[0]

        # The fallback is the table's last entry rather than a masked
        # assignment afterwards: a colour is a string in one style sheet and
        # a three- or four-tuple in the next, and assigning a tuple into
        # masked slots of an object array broadcasts it element by element.
        # As a table entry it is one object, whatever it is made of.
        lookup = np.empty(len(palette) + 1, dtype=object)
        for index, color in enumerate(palette):
            lookup[index] = color
        lookup[len(palette)] = fallback

        finite = np.isfinite(numeric)
        indices = np.full(numeric.shape, len(palette), dtype=np.intp)
        indices[finite] = (np.rint(numeric[finite]).astype(np.intp) - 1) % len(palette)

        return lookup[indices].tolist()

    def map_continuous_colors_to_cmap(
        self,
        values: Any,
        *,
        cmap_name: str = "viridis",
        fallback_color: str = "",
    ) -> np.ndarray:
        """Map continuous numeric values to RGBA rows from a colormap.

        Returns an ``(n, 4)`` float array, which is what Matplotlib wants
        anywhere a per-point colour is accepted, and what a Colormap
        produces when called with an array. It used to call ``cmap(value)``
        once per point and build a Python list of RGBA tuples: the same
        normalisation and lookup, done n times in Python instead of once in
        C, at 2 076 ms per 300 000 points against 8.

        Non-finite values have no place on the scale and take
        *fallback_color* instead - as before, only resolved to RGBA so it
        can sit in the same array.
        """
        numeric = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
        palette = self.palette_colors()
        fallback = to_rgba(fallback_color or palette[0])
        finite = np.isfinite(numeric)

        if not finite.any():
            return np.tile(np.asarray(fallback, dtype=float), (numeric.size, 1))

        low = float(numeric[finite].min())
        high = float(numeric[finite].max())
        span = high - low

        # 0.5 for the non-finite entries too: they are overwritten below, and
        # a NaN reaching the colormap would come back as its "bad" colour
        # rather than as the fallback that was asked for.
        normalized = np.full(numeric.shape, 0.5, dtype=float)
        if span != 0.0:
            np.divide(numeric - low, span, out=normalized, where=finite)

        mapped = np.asarray(colormaps.get_cmap(cmap_name)(normalized), dtype=float)
        mapped[~finite] = fallback
        return mapped

    def color_sequence_from_values(
        self,
        values: Any,
        *,
        fallback_color: str = "",
        cmap_name: str = "viridis",
    ) -> Any:
        """Return one colour per row: integers use the palette, floats a cmap.

        Two representations, because the two paths have different natural
        ones: a list of the palette's own colour specs for category ids, and
        an ``(n, 4)`` RGBA array for continuous values. Matplotlib accepts
        either wherever a colour sequence goes.

        A caller must therefore not ask a result for its truth value -
        ``if colors:`` raises on an array - and must compare two colours
        with ``np.array_equal`` rather than ``!=``. Both are ``len()``-able
        and indexable, which is all the drawing code needs.
        """
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
        if not len(colors):  # len(), not truthiness: colors may be an array
            return fallback_color

        first = colors[0]
        # A tuple, not the array's row: this is a *single* colour, and it
        # travels to call sites that test it with `if color:` - which raises
        # on a four-element array. One row is one tuple, so keeping the type
        # the continuous path has always returned costs nothing.
        return tuple(float(channel) for channel in first) if isinstance(
            first, np.ndarray
        ) else first