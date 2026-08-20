"""Conversions between Qt's logical pixels and Matplotlib's figure inches.

Two scales meet at the canvas and they are easy to mix up:

* Qt widget geometry is in **logical** pixels - the units a layout works in;
* a Matplotlib figure is inches x dpi, and ``FigureCanvasQT`` keeps the
  figure's dpi in **device** pixels, because ``_set_device_pixel_ratio``
  multiplies the configured dpi by the display's ratio.

So ``figure.dpi`` is not the dpi anyone configured.  On a 2x screen a 100 dpi
figure reports 200, and a widget 800 logical pixels wide needs a figure 8
inches wide, not 4.  Getting that wrong by a factor of the ratio makes Agg
paint a quarter of the widget and leave the rest as it found it, which is what
the artefacts on macOS were; on Windows the ratio is 1 and the mistake cancels
out exactly, which is why it looked like a platform bug.

Everything that converts between the two scales goes through this module, so
the ratio is applied in one place.  It was previously applied in three places
and omitted in a fourth, and the fourth quietly undid the other three.

The rule this module exists to enforce: **never store a device-scaled dpi.**
Configuration, rcParams and saved state all hold the dpi the user asked for;
the ratio is applied only at the moment of writing it onto a figure.
"""
from __future__ import annotations

from typing import Any


def canvas_pixel_ratio(canvas: Any | None) -> float:
    """Return the canvas's device pixel ratio, never zero.

    Read from the canvas rather than the screen: Matplotlib keeps its own copy
    there and keeps ``figure.dpi`` consistent with it, so this is the value the
    figure was actually scaled by - which is what the conversions need, even in
    the moment after a window has moved between displays.

    ``None`` is answered with 1.0 rather than refused.  A widget prepares its
    figure from configuration before constructing the canvas around it, and at
    that point no scaling is known yet; ``FigureCanvasQT.__init__`` then applies
    the ratio to whatever dpi it finds, so the figure ends up correct anyway.
    """
    if canvas is None:
        return 1.0

    try:
        return float(getattr(canvas, "device_pixel_ratio", 1.0)) or 1.0
    except (TypeError, ValueError):
        return 1.0


def apply_configured_dpi(figure: Any, canvas: Any, configured_dpi: float) -> float:
    """Write a configured dpi onto *figure*, scaled for the display.

    Use this for every dpi that came from config.json, rcParams or a spin box.
    Writing such a value with ``figure.set_dpi`` directly breaks the invariant
    the Qt backend maintains, and the figure is then mis-sized against its
    canvas for the rest of the session - reloading a chart or editing a figure
    property was enough to bring the artefacts back.

    Returns the device dpi actually set, for callers that want to log it.
    """
    device_dpi = max(1.0, float(configured_dpi) * canvas_pixel_ratio(canvas))
    figure.set_dpi(device_dpi)
    return device_dpi


def configured_dpi(figure: Any, canvas: Any) -> float:
    """Return *figure*'s dpi with the display's ratio taken back out.

    The inverse of :func:`apply_configured_dpi`, for the places that need to
    show or persist a dpi: what belongs in config.json is the number the user
    chose, not one that doubles on a Retina screen and doubles again next time
    it is read back.
    """
    return max(1.0, float(figure.get_dpi()) / canvas_pixel_ratio(canvas))


def logical_to_inches(logical_px: float, figure: Any, canvas: Any) -> float:
    """Convert a widget dimension in logical pixels to figure inches.

    The same conversion Matplotlib performs in ``FigureCanvasQT.resizeEvent``::

        w = event.size().width() * self.device_pixel_ratio
        winch = w / self.figure.dpi
    """
    dpi = float(figure.get_dpi()) or 100.0
    return max(1.0, float(logical_px)) * canvas_pixel_ratio(canvas) / dpi


def inches_to_logical(inches: float, configured_dpi_value: float, zoom: float = 1.0) -> float:
    """Convert figure inches to logical pixels, for ``setFixedSize`` and friends.

    Takes the *configured* dpi, not the figure's: the ratio cancels out of this
    direction entirely, which is the clearest sign that stored state should
    never carry it.  A figure 8 inches wide at 100 dpi occupies 800 logical
    pixels whatever the display does; only the number of device pixels behind
    them changes.
    """
    return max(1.0, float(inches) * float(configured_dpi_value) * float(zoom))
