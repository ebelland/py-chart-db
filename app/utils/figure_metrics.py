"""Physical figure metrics: width, height and dpi, stored per figure.

These three values used to live in ``config.json`` under ``chart_panel``, which
made them application-wide: opening a second figure applied the first one's
size to it, and editing either one changed both.  They belong to the figure -
a slide wants 25cm wide, a journal column wants 8.5 - so they are stored in the
figure descriptor's ``options`` alongside ``frameon`` and ``layout_mode``.

This module is deliberately free of Qt and Matplotlib imports so that both the
chart panel and the properties widget can read the vocabulary without either
importing the other.
"""

from __future__ import annotations

from typing import Any

CM_PER_INCH = 2.54

#: Fallback metrics, used when a figure's options say nothing.  Matches the
#: Matplotlib defaults so an untouched figure looks the way Matplotlib draws it.
DEFAULT_FIGURE_DPI = 100.0
DEFAULT_FIGURE_SIZE_IN = (6.4, 4.8)

OPT_FIGURE_WIDTH_CM = "figure_width_cm"
OPT_FIGURE_HEIGHT_CM = "figure_height_cm"
OPT_FIGURE_DPI = "figure_dpi"


def _positive_float(value: Any) -> float | None:
    """Return ``value`` as a positive float, or None when it is neither."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0.0 else None


def figure_metrics_from_options(
    options: dict[str, Any] | None,
) -> tuple[float, float, float] | None:
    """Return ``(width_cm, height_cm, dpi)`` from figure options, or None.

    None means "this figure has no metrics of its own", which is different from
    "this figure is 0cm wide": the caller should leave rcParams alone rather
    than reset it to a default, because a figure saved before this was
    per-figure has no keys at all and should keep rendering as it did.

    All three keys must be present and positive.  A half-set of metrics is
    treated as absent - applying a width without its matching height would
    distort the figure rather than resize it.
    """
    if not isinstance(options, dict):
        return None

    width_cm = _positive_float(options.get(OPT_FIGURE_WIDTH_CM))
    height_cm = _positive_float(options.get(OPT_FIGURE_HEIGHT_CM))
    dpi = _positive_float(options.get(OPT_FIGURE_DPI))

    if width_cm is None or height_cm is None or dpi is None:
        return None

    return width_cm, height_cm, dpi
