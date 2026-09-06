"""What a renderer forwards to Matplotlib, and what it keeps for itself.

One vocabulary, read by the renderers that declare these schemas and by the
editor that renders them.  The sibling of ``axis_options``: no Qt and no
Matplotlib here, so the split can be tested on its own.

It exists because ``Kwargs`` meant two different things at once:

* keys forwarded to ``ax.bar`` / ``ax.scatter`` / ``ax.hist``;
* keys the renderer consumes itself - ``show_legend``, ``max_tick_labels``,
  ``trend_degree``, ``stats_position`` - which raise an unexpected-keyword
  error the moment they reach Matplotlib.

Nothing in the schema said which was which, so each renderer kept a hand
written removal list - ``BarAxisRenderer`` dropped eleven keys before calling
``bar`` - and a forgotten entry failed at render time rather than at edit
time.  A renderer now declares two schemas: ``Kwargs`` is forwarded verbatim
and ``Options`` never is, so :func:`resolve` cannot return an option to
Matplotlib because it is only ever given ``Kwargs``.

The second thing this fixes is defaults that quietly overrule the style sheet.
``alpha`` was declared five times with five defaults (0.9, 0.8, 0.75, 0.7,
None) and ``linewidth`` four (1.8, 1.6, 0.2, 0.0), so a .mplstyle setting
``lines.linewidth`` - nine of the shipped ones do - was overridden by whichever
renderer happened to be drawing.  :data:`DEFAULT` is the value that says
"leave this to the style sheet": it resolves to the key being *omitted*, which
is the only way to let rcParams decide, since Matplotlib treats an explicit
``None`` and an absent keyword differently.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

#: Leave this keyword to the active .mplstyle: the resolver drops it, and
#: Matplotlib falls back to rcParams or to its own signature default.
#:
#: A plain string rather than a sentinel object because axis options are
#: persisted as JSON, exactly as ``axis_options.AUTO`` is - a module-level
#: object would not survive the round trip through the database.
#:
#: The string is why :data:`STYLE_DEFAULT` is not offered on free-text
#: keywords: ``label`` may legitimately be the word "default", and a series
#: called that would silently lose its legend entry.  It is offered on
#: numbers, booleans, colours and enums, where the editor writes the value
#: rather than the person typing it.
DEFAULT: str = "default"

#: Metadata key marking a keyword that may be left to the style sheet.
#:
#: The bundles below spell that default as ``None`` rather than as
#: :data:`DEFAULT`, and deliberately: both are dropped by :func:`resolve`, but
#: ``None`` is what ``DictEditorPanel`` already round-trips.  The panel infers
#: a row's editor from the *type of the default* and then calls ``float()`` on
#: whatever text the row holds, so a numeric keyword defaulting to the string
#: "default" loses its spin box and raises inside the delegate.
#:
#: So this flag records which keywords want the state, ``None`` provides the
#: behaviour today, and :data:`DEFAULT` is what the panel will write once it
#: draws the third state - a first combo entry, a spin box special value, a
#: tri-state check.  Until then the intent is in the schema rather than in
#: nobody's head.
STYLE_DEFAULT: str = "style_default"

#: Metadata key naming the rcParam a keyword shadows, e.g. ``lines.linewidth``.
#: The editor reads it to show the effective value as placeholder text, which
#: is what makes "from the style" mean something to the person looking at it;
#: a test reads it to assert such a keyword defaults to :data:`DEFAULT`.
RCPARAM: str = "rcparam"


def is_default(value: object) -> bool:
    """True when *value* means "leave this to the style sheet"."""
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() == DEFAULT


# ----------------------------------------------------------------------
# Building schemas
# ----------------------------------------------------------------------
def merge(*specs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Combine schemas, later ones refining earlier ones key by key.

    Per key rather than per schema, so a renderer wanting a different default
    for one shared keyword writes just the default::

        Kwargs = merge(PATCH_KWARGS, {"alpha": {"default": 0.5}})

    and keeps the shared type, range, group and description instead of
    restating them - which is how the current copies drifted apart.
    """
    merged: dict[str, dict[str, Any]] = {}
    for spec in specs:
        for name, meta in spec.items():
            merged.setdefault(name, {}).update(as_meta(meta))
    return merged


def pick(spec: Mapping[str, Any], *names: str) -> dict[str, dict[str, Any]]:
    """Return the named entries of *spec*, in the order given.

    Replaces the comprehension in ``area`` that reaches into
    ``ScatterAxisRenderer.Kwargs`` for six keys: a renderer should not have to
    import another renderer to describe ``alpha``.
    """
    return {name: dict(as_meta(spec[name])) for name in names if name in spec}


def as_meta(meta: object) -> dict[str, Any]:
    """Accept both ``{"alpha": {...}}`` and the bare-default shorthand."""
    if isinstance(meta, Mapping):
        return dict(meta)
    return {"default": meta}


# ----------------------------------------------------------------------
# Resolving values
# ----------------------------------------------------------------------
#: What a checkbox, a config file and a hand-typed option may each call a
#: boolean. Compared lowercased and stripped; anything else is not one.
TRUE_WORDS: frozenset[str] = frozenset({"1", "true", "yes", "on"})
FALSE_WORDS: frozenset[str] = frozenset({"0", "false", "no", "off"})

#: Returned by :func:`coerce` for a value that cannot be made into the type
#: its schema entry declares, and so must not be forwarded.
UNCONVERTIBLE: object = object()


def coerce(value: object, meta: Mapping[str, Any]) -> Any:
    """Return *value* as the type its schema entry declares.

    Options reach a renderer as text far more often than not: the axis options
    editor stores what was typed, and so does a saved descriptor, so
    ``rstride`` arrives as ``"2"`` rather than ``2``.  Matplotlib does not
    coerce - ``plot_surface`` computes ``(rows - 1) % rstride`` and raises
    *unsupported operand type(s) for %: 'int' and 'str'*, and a string
    ``linewidth`` reaches the C++ layer and fails there instead, with a
    message that names nothing the person typed.

    Only int, float and bool are converted.  A renderer-owned option that is
    free-form is parsed by the renderer that owns it: contour ``levels`` is
    deliberately either a count or a comma-separated list, which no scalar
    conversion could express.
    """
    declared = meta.get("type")
    if declared not in (int, float, bool) or value is None:
        return value

    if declared is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in TRUE_WORDS:
            return True
        if text in FALSE_WORDS:
            return False
        return UNCONVERTIBLE

    # A bool where a number was declared is left alone: it is almost certainly
    # a mis-declared option rather than the number 0 or 1, and silently
    # turning it into one would hide that.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return declared(value)

    text = str(value).strip()
    if text == "":
        # Empty means "not set" to every caller downstream, which drops it.
        return value
    try:
        return int(float(text)) if declared is int else float(text)
    except (TypeError, ValueError):
        return UNCONVERTIBLE


def resolve(
    spec: Mapping[str, Any],
    sources: Iterable[Mapping[str, Any]],
    dropped: list[tuple[str, Any, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve every key in *spec* against *sources*, most specific first.

    One chain, written down once::

        series style  >  axis value  >  schema default  >  DEFAULT (omitted)

    A key resolving to :data:`DEFAULT`, to None or to an empty string is left
    out entirely rather than forwarded, because for Matplotlib absent and None
    are not the same thing: ``picker=None`` reaches
    ``Line2D.set_pickradius(None)`` and raises *"pick radius should be a
    distance"*, while omitting the key simply uses the default.

    A value that cannot be made into its declared type is dropped too, and
    appended to *dropped* as ``(name, value, declared_type)`` when a list is
    given: this module has no logger of its own, and the caller wants to name
    the renderer in the message anyway.  A typo should cost the option, not
    the chart.
    """
    ordered = [source for source in sources if isinstance(source, Mapping)]
    resolved: dict[str, Any] = {}
    for name, raw_meta in spec.items():
        meta = as_meta(raw_meta)
        value = _first(name, ordered, meta.get("default"))
        if is_default(value) or value == "":
            continue
        coerced = coerce(value, meta)
        if coerced is UNCONVERTIBLE:
            if dropped is not None:
                dropped.append((name, value, meta.get("type")))
            continue
        resolved[name] = coerced
    return resolved


def resolve_one(
    spec: Mapping[str, Any],
    name: str,
    sources: Iterable[Mapping[str, Any]],
    *,
    typed: bool = False,
) -> Any:
    """Resolve one key, returning None when it is unset.

    Kept separate from :func:`resolve` because a renderer reading its own
    option wants the value whatever it is - including the schema default -
    while a forwarded keyword wants to be absent.

    Raw by default, and deliberately: several renderer-owned options are
    free-form, and contour ``levels`` is a count or a list.  Pass
    ``typed=True`` for one that is handed to Matplotlib after all, where
    ``linewidths="0.5"`` is a *string* Matplotlib reads as the characters
    ``0``, ``.``, ``5`` and draws three lines of nonsense widths.

    An empty string is *not* collapsed to None here, though :func:`resolve`
    drops it: emptiness means something to some renderers - ``contour``
    defaults ``levels`` to ``""`` and reads it as "let Matplotlib choose the
    levels" - and only the renderer knows which.
    """
    meta = as_meta(spec.get(name, {}))
    value = _first(
        name,
        [source for source in sources if isinstance(source, Mapping)],
        meta.get("default"),
    )
    if is_default(value):
        return None
    if not typed:
        return value
    coerced = coerce(value, meta)
    return None if coerced is UNCONVERTIBLE else coerced


def _first(name: str, sources: list[Mapping[str, Any]], fallback: object) -> object:
    for source in sources:
        if name in source:
            return source[name]
    return fallback


# ----------------------------------------------------------------------
# Keywords every renderer shares
# ----------------------------------------------------------------------
#: Artist-level keywords accepted by every Matplotlib drawing call, declared
#: nine times between the renderers today with defaults that no longer agree.
ARTIST_KWARGS: dict[str, dict[str, Any]] = {
    "alpha": {
        "default": None,
        STYLE_DEFAULT: True,
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "decimals": 3,
        "group": "Appearance",
        "description": "Opacity, from 0.0 transparent to 1.0 opaque.",
    },
    "zorder": {
        "default": None,
        STYLE_DEFAULT: True,
        "type": float,
        "min": -1000.0,
        "max": 1000.0,
        "step": 1.0,
        "decimals": 1,
        "group": "Behaviour",
        "description": "Drawing order. Higher values are drawn on top.",
    },
    "visible": {
        "default": True,
        "type": bool,
        "group": "Behaviour",
        "description": "Draw this artist at all.",
    },
    "label": {
        # No style default: this is free text, and a series legitimately named
        # "default" must not be read as "leave it to the style".
        "default": None,
        "type": str,
        "group": "Legend",
        "description": "Legend entry. Defaults to the series name.",
    },
    "rasterized": {
        "default": False,
        "type": bool,
        "group": "Behaviour",
        "description": (
            "Draw as pixels rather than vectors. Keeps a dense series from "
            "making a vector export unopenable."
        ),
    },
    "picker": {
        "default": None,
        STYLE_DEFAULT: True,
        "type": float,
        "min": 0.0,
        "max": 100.0,
        "step": 0.5,
        "decimals": 2,
        "group": "Behaviour",
        "description": "Distance in points within which a click selects this artist.",
    },
}

#: The rest of the Artist properties: real, forwardable, and reached for once
#: in a hundred figures.  A separate bundle so that a renderer adopting
#: ARTIST_KWARGS does not grow six editor rows nobody asked for.
ARTIST_ADVANCED_KWARGS: dict[str, dict[str, Any]] = {
    "animated": {
        "default": False,
        "type": bool,
        "group": "Behaviour",
        "description": "Matplotlib animation flag.",
    },
    "clip_on": {
        "default": True,
        "type": bool,
        "group": "Behaviour",
        "description": "Clip the artist to the axes area.",
    },
    "in_layout": {
        "default": True,
        "type": bool,
        "group": "Behaviour",
        "description": "Count this artist when the layout engine measures the axes.",
    },
    "snap": {
        # Matplotlib's third state - auto - which a two-state
        # checkbox cannot show; unchecked here means auto, not "never".
        "default": None,
        "type": bool,
        "group": "Behaviour",
        "description": "Snap positions to whole pixels. Unset lets Matplotlib decide.",
    },
    "gid": {
        "default": None,
        "type": str,
        "group": "Behaviour",
        "description": "Artist group id, used by the SVG backend.",
    },
    "url": {
        "default": None,
        "type": str,
        "group": "Behaviour",
        "description": "Link target in backends that support one.",
    },
}

#: Line keywords, for renderers drawing ``Line2D`` artists.
LINE_KWARGS: dict[str, dict[str, Any]] = {
    "color": {
        "default": None,
        STYLE_DEFAULT: True,
        RCPARAM: "axes.prop_cycle",
        "type": str,
        "kind": "color",
        "group": "Appearance",
        "description": "Line colour. From the style's colour cycle when unset.",
    },
    "linewidth": {
        "default": None,
        STYLE_DEFAULT: True,
        RCPARAM: "lines.linewidth",
        "type": float,
        "min": 0.0,
        "max": 20.0,
        "step": 0.25,
        "decimals": 3,
        "group": "Line",
        "description": "Line width in points. 0 hides the line.",
    },
    "linestyle": {
        "default": None,
        STYLE_DEFAULT: True,
        RCPARAM: "lines.linestyle",
        "type": str,
        "kind": "linestyle",
        "group": "Line",
        "description": "Solid, dashed, dotted or dash-dot.",
    },
}

#: Patch keywords, for renderers drawing bars, bands, wedges and boxes.
PATCH_KWARGS: dict[str, dict[str, Any]] = {
    "facecolor": {
        "default": None,
        STYLE_DEFAULT: True,
        RCPARAM: "patch.facecolor",
        "type": str,
        "kind": "color",
        "group": "Appearance",
        "description": "Fill colour. Matplotlib gives this precedence over color.",
    },
    "edgecolor": {
        "default": None,
        STYLE_DEFAULT: True,
        RCPARAM: "patch.edgecolor",
        "type": str,
        "kind": "color",
        "group": "Patch",
        "description": "Outline colour.",
    },
    "linewidth": {
        "default": None,
        STYLE_DEFAULT: True,
        RCPARAM: "patch.linewidth",
        "type": float,
        "min": 0.0,
        "max": 20.0,
        "step": 0.25,
        "decimals": 3,
        "group": "Patch",
        "description": "Outline width in points. 0 hides the outline.",
    },
    "linestyle": {
        # No rcParam: Matplotlib has patch.linewidth and patch.edgecolor, but
        # no patch.linestyle, so there is nothing for the style sheet to say.
        "default": None,
        STYLE_DEFAULT: True,
        "type": str,
        "kind": "linestyle",
        "group": "Patch",
        "description": "Outline style.",
    },
    "hatch": {
        "default": None,
        "type": str,
        "group": "Patch",
        "description": "Hatch pattern, e.g. //, xx, ..",
    },
    "fill": {
        "default": True,
        "type": bool,
        "group": "Patch",
        "description": "Fill the patch, or draw its outline only.",
    },
}

#: Colour-mapping keywords, for renderers mapping a value to a colour.
CMAP_KWARGS: dict[str, dict[str, Any]] = {
    "cmap": {
        # Hard-coded to "viridis" in three renderers today, which is a colour
        # decision taken away from the style sheet.
        "default": None,
        STYLE_DEFAULT: True,
        RCPARAM: "image.cmap",
        "type": str,
        "kind": "colormap",
        "group": "Colour mapping",
        "description": "Colormap used for the colour role.",
    },
    "norm": {
        "default": None,
        "type": ["linear", "log", "symlog", "logit"],
        "group": "Colour mapping",
        "description": "How values are normalised before the colormap is applied.",
    },
    "vmin": {
        "default": None,
        "type": float,
        "group": "Colour mapping",
        "description": "Value mapped to the low end. Unset uses the data minimum.",
    },
    "vmax": {
        "default": None,
        "type": float,
        "group": "Colour mapping",
        "description": "Value mapped to the high end. Unset uses the data maximum.",
    },
}

# ----------------------------------------------------------------------
# Options every renderer shares
# ----------------------------------------------------------------------
#: Declared four times as True and once as False today, and removed from the
#: kwargs by hand in each renderer that has it.
LEGEND_OPTIONS: dict[str, dict[str, Any]] = {
    "show_legend": {
        "default": True,
        "type": bool,
        "group": "Legend",
        "description": "Draw a legend naming each series.",
    },
}

#: Shared by the histogram and ECDF renderers, which both overlay a fit.
FIT_OPTIONS: dict[str, dict[str, Any]] = {
    "distribution_fit": {
        "default": "none",
        "type": ["none", "normal", "lognormal", "exponential", "gamma", "weibull"],
        "group": "Fit",
        "description": "Overlay a fitted distribution.",
    },
    "distribution_fit_points": {
        "default": 200,
        "type": int,
        "min": 10,
        "max": 10_000,
        "group": "Fit",
        "description": "Points used to draw the fitted curve.",
    },
}


#: Where the camera sits, for the renderers drawing on a 3D axes.
#:
#: Options rather than keywords, and that is the whole distinction in one
#: example: these names are not accepted by ``plot_surface`` at all.  They are
#: read back out afterwards and passed to ``ax.view_init``, which is why both
#: surfaces had to remove them by hand before drawing.
VIEW_OPTIONS: dict[str, dict[str, Any]] = {
    "elev": {
        "default": None,
        "type": float,
        "min": -180.0,
        "max": 180.0,
        "group": "View",
        "description": "Camera elevation angle, in degrees.",
    },
    "azim": {
        "default": None,
        "type": float,
        "min": -180.0,
        "max": 180.0,
        "group": "View",
        "description": "Camera azimuth angle, in degrees.",
    },
    "roll": {
        "default": None,
        "type": float,
        "min": -180.0,
        "max": 180.0,
        "group": "View",
        "description": "Camera roll angle, in degrees.",
    },
}
