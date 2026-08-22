"""What an axis's grid, ticks and limits were asked for.

One vocabulary, read by the properties widget that writes these keys and by
the renderer that applies them.  It exists because the two used to disagree in
a way nobody could see:

* ``grid: False`` meant "do nothing" rather than "no grid", so unticking Show
  grid left the grid exactly where it was whenever the active .mplstyle set
  ``axes.grid: True`` - which the shipped styles do;
* the grid was one setting for the whole axis (which ticks, which direction),
  so "x major only" was expressible and "x major and y minor" was not;
* minor ticks were a single switch that only ever turned them *on*, and
  turning it off did nothing at all.

So a grid or a tick setting is now per axis (x, y) and per tick class (major,
minor), and each one distinguishes three things a boolean cannot: leave it to
the style sheet, force it off, force it on.

No Qt and no Matplotlib here - the widget and the renderer both import it, and
the legacy translation is the part worth testing on its own.
"""
from __future__ import annotations

from typing import Any

#: Leave this setting to the active .mplstyle.  The default everywhere, and
#: the honest reading of an option nobody has set.
AUTO: str = "auto"

#: Force it off, whatever the style sheet says. This is the value that did not
#: exist before, and the reason Show grid appeared not to work.
OFF: str = "off"

#: Force it on, in the style sheet's own appearance.
ON: str = "on"

#: Grid choices, as (stored value, label).  The line styles force an
#: appearance as well as presence, which is what makes a minor grid usable:
#: dotted minor against solid major reads as two levels rather than as noise.
GRID_CHOICES: tuple[tuple[str, str], ...] = (
    (AUTO, "From the style"),
    (OFF, "Off"),
    (ON, "On"),
    ("-", "Solid"),
    ("--", "Dashed"),
    (":", "Dotted"),
    ("-.", "Dash-dot"),
)

#: Tick choices.  Direction doubles as "on", because a tick that is shown is
#: shown somewhere: inside the axes, outside, or across.
TICK_CHOICES: tuple[tuple[str, str], ...] = (
    (AUTO, "From the style"),
    (OFF, "Off"),
    (ON, "On"),
    ("in", "Inside"),
    ("out", "Outside"),
    ("inout", "Both sides"),
)

#: The axes and tick classes every one of those settings is written for.
AXES: tuple[str, ...] = ("x", "y")
WHICH: tuple[str, ...] = ("major", "minor")

#: How the view range is decided.
LIMITS_AUTO: str = "auto"
LIMITS_AUTO_X: str = "auto_x"
LIMITS_AUTO_Y: str = "auto_y"
LIMITS_MANUAL: str = "manual"

LIMIT_CHOICES: tuple[tuple[str, str], ...] = (
    (LIMITS_AUTO, "Automatic (both axes)"),
    (LIMITS_AUTO_X, "Automatic x, manual y"),
    (LIMITS_AUTO_Y, "Automatic y, manual x"),
    (LIMITS_MANUAL, "Manual (both axes)"),
)

#: Which axis each mode leaves to Matplotlib.
_AUTOMATIC_AXES: dict[str, frozenset[str]] = {
    LIMITS_AUTO: frozenset({"x", "y"}),
    LIMITS_AUTO_X: frozenset({"x"}),
    LIMITS_AUTO_Y: frozenset({"y"}),
    LIMITS_MANUAL: frozenset(),
}


def grid_key(axis: str, which: str) -> str:
    """Return the option key for one grid setting."""
    return f"grid_{axis}_{which}"


def tick_key(axis: str, which: str) -> str:
    """Return the option key for one tick setting."""
    return f"ticks_{axis}_{which}"


def limit_key(axis: str, edge: str) -> str:
    """Return the option key for one manual limit, e.g. ``x_min``."""
    return f"{axis}_{edge}"


def grid_setting(options: dict[str, Any], axis: str, which: str) -> str:
    """Return one grid setting, translating a pre-per-axis figure if needed."""
    value = options.get(grid_key(axis, which))
    if value is not None:
        return _clean(value, GRID_CHOICES)
    return _legacy_grid(options, axis, which)


def tick_setting(options: dict[str, Any], axis: str, which: str) -> str:
    """Return one tick setting, translating a pre-per-axis figure if needed."""
    value = options.get(tick_key(axis, which))
    if value is not None:
        return _clean(value, TICK_CHOICES)
    return _legacy_ticks(options, which)


def limits_mode(options: dict[str, Any]) -> str:
    """Return which axes are automatic; automatic for both by default."""
    value = str(options.get("limits_mode", LIMITS_AUTO) or LIMITS_AUTO).strip().lower()
    return value if value in dict(LIMIT_CHOICES) else LIMITS_AUTO


def is_automatic(options: dict[str, Any], axis: str) -> bool:
    """True when *axis* is left to Matplotlib to scale."""
    return axis in _AUTOMATIC_AXES[limits_mode(options)]


def manual_limit(options: dict[str, Any], axis: str, edge: str) -> float | None:
    """Return one manual limit, or None when it is not a usable number.

    None rather than a default: an unset edge means "this end stays
    automatic", which ``set_xlim`` understands directly, and a zero invented
    here would be a limit the user never typed.
    """
    value = options.get(limit_key(axis, edge))
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _clean(value: Any, choices: tuple[tuple[str, str], ...]) -> str:
    """Return a known value, falling back to AUTO rather than raising."""
    text = str(value or "").strip().lower()
    return text if text in dict(choices) else AUTO


def _legacy_grid(options: dict[str, Any], axis: str, which: str) -> str:
    """Translate ``grid`` / ``grid_which`` / ``grid_axis`` into one setting.

    A figure saved before this existed keeps the appearance it had, which is
    why "off" is not the answer for an untouched combination: ``grid(True,
    which="major", axis="x")`` never said anything about the y grid, and the
    style sheet decided it. Only what the old options actually turned on
    becomes ON.
    """
    if not bool(options.get("grid", False)):
        return AUTO

    old_which = str(options.get("grid_which", "major") or "major").strip().lower()
    old_axis = str(options.get("grid_axis", "both") or "both").strip().lower()

    covers_which = old_which in {which, "both"}
    covers_axis = old_axis in {axis, "both"}
    return ON if covers_which and covers_axis else AUTO


def _legacy_ticks(options: dict[str, Any], which: str) -> str:
    """Translate ``minor_ticks`` / ``tick_direction`` into one setting.

    The old direction applied to every tick on the axes, and the old switch
    only ever turned minor ticks on - there was no way to turn them off, so
    "not asked for" translates to AUTO rather than to OFF.
    """
    direction = str(options.get("tick_direction", "") or "").strip().lower()
    if which == "major":
        return direction if direction in dict(TICK_CHOICES) else AUTO

    if bool(options.get("minor_ticks", False)):
        return direction if direction in dict(TICK_CHOICES) else ON
    return AUTO


def defaults() -> dict[str, Any]:
    """Return every grid, tick and limit key at its default.

    Used by the properties widget to reset, and as the shape of the payload:
    a key that is always present cannot be one the renderer silently misses.
    """
    payload: dict[str, Any] = {}
    for axis in AXES:
        for which in WHICH:
            payload[grid_key(axis, which)] = AUTO
            payload[tick_key(axis, which)] = AUTO
        for edge in ("min", "max"):
            payload[limit_key(axis, edge)] = None
    payload["limits_mode"] = LIMITS_AUTO
    return payload
