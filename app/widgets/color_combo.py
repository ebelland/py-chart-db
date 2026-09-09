# -*- coding: utf-8 -*-
"""app.widgets.color_combo

MatplotlibColorCombo

A PySide6 QComboBox that lists Matplotlib-known colors and shows a small
color swatch icon next to each entry. The dropdown popup auto-sizes to fit
the longest color name (e.g. long ``xkcd:`` names), independent of how
narrow the combo box itself is.

Notes for strict PySide6 typing / Pylance:
- Use Qt.ItemDataRole.UserRole (not Qt.UserRole)
- Use Qt.GlobalColor.transparent (not Qt.transparent)

This module is intentionally standalone and PEP8 compliant.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen
from PySide6.QtWidgets import QWidget

from matplotlib import rcParams
from matplotlib.colors import (
    BASE_COLORS,
    CSS4_COLORS,
    TABLEAU_COLORS,
    XKCD_COLORS,
    to_hex,
)

from app.widgets.icon_combo import ComboEntry, IconComboBox
from app.styles.style import create_hidpi_pixmap

#: The first entry's stored value. An empty string, which every consumer
#: already reads as "no explicit color": the renderer answers it with the
#: style sheet's own colour cycle (BaseAxisRenderer.series_color), and the
#: kwargs editor answers it by leaving the key out. Only the label above it
#: differs - see ``none_label``.
_NONE_LABEL = "(none)"
_NONE_VALUE = ""

#: The label for that entry where an empty value means "the style sheet
#: decides" rather than "cleared". Spelled exactly like the first entry of
#: the Line style and Marker combos next to it, and left untranslated for
#: the same reason theirs are: the three read as one row of defaults, and
#: half a row in Italian would read as three unrelated widgets.
DEFAULT_COLOR_LABEL = "Default"
_SWATCH_SIZE = 14
_BORDER_COLOR = "#5a5a5a"

#: How many of the active cycle's colours the "no explicit colour" swatch
#: shows. Four is enough to read as "a palette decides this" at 14 pixels;
#: ten would be stripes too thin to tell apart.
_CYCLE_SWATCH_COLORS = 4


def _iter_named_colors() -> Iterator[ComboEntry]:
    """Yield (name, hex_color) pairs from every Matplotlib color table."""
    for mapping in (BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS):
        for name, value in mapping.items():
            try:
                yield name, to_hex(value)
            except ValueError:
                continue  # Skip malformed table entries.


def _cycle_colors() -> tuple[str, ...]:
    """Return the first few colours of the active style's property cycle.

    As hex, not as stored: a cycle may hold RGB tuples as readily as hex
    strings (Matplotlib's own default style does), and ``QColor`` renders
    the string form of a tuple as black.
    """
    try:
        colors = rcParams["axes.prop_cycle"].by_key().get("color", [])
    except (KeyError, AttributeError, TypeError):
        colors = []

    resolved: list[str] = []
    for color in colors[:_CYCLE_SWATCH_COLORS]:
        try:
            resolved.append(to_hex(color))
        except ValueError:
            continue
    return tuple(resolved) or ("#1f77b4",)


@lru_cache(maxsize=8)
def _cycle_swatch_icon(colors: tuple[str, ...]) -> QIcon:
    """Build (and cache) the swatch for "no explicit colour": the cycle.

    A blank square was what this entry used to show, and blank reads as
    "no colour at all" - which is not what happens. Leaving the colour
    unset hands the series to the style sheet's ``axes.prop_cycle``, so
    the swatch shows that cycle: a few of its colours side by side, which
    no single-colour entry below can be confused with.
    """
    pixmap = create_hidpi_pixmap(_SWATCH_SIZE, _SWATCH_SIZE)

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        inner = QRect(1, 1, _SWATCH_SIZE - 2, _SWATCH_SIZE - 2)
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        width = inner.width() / len(colors)
        for index, color in enumerate(colors):
            painter.setBrush(QBrush(QColor(color)))
            left = inner.left() + round(index * width)
            right = inner.left() + round((index + 1) * width)
            painter.drawRect(QRect(left, inner.top(), right - left, inner.height()))

        painter.setPen(QPen(QColor(_BORDER_COLOR)))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawRect(inner)
    finally:
        painter.end()

    return QIcon(pixmap)


def _entry_icon(hex_color: str) -> QIcon:
    """The icon for one entry: a swatch, or the cycle for the empty value.

    Not itself cached - it is the one place that reads the *live* cycle, so
    a combo built after a style sheet is loaded shows that sheet's colours
    rather than the ones the first combo of the session was built with.
    """
    if not hex_color:
        return _cycle_swatch_icon(_cycle_colors())
    return _swatch_icon(hex_color)


@lru_cache(maxsize=None)
def _swatch_icon(hex_color: str) -> QIcon:
    """Build (and cache) a small square swatch icon for *hex_color*."""
    if not hex_color:
        return QIcon()

    # Allocated at the display's pixel density: the coordinates below stay
    # logical, but the bitmap has the pixels to be sharp on a Retina screen.
    pixmap = create_hidpi_pixmap(_SWATCH_SIZE, _SWATCH_SIZE)

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = QRect(1, 1, _SWATCH_SIZE - 2, _SWATCH_SIZE - 2)
        painter.setPen(QPen(QColor(_BORDER_COLOR)))
        painter.setBrush(QBrush(QColor(hex_color)))
        painter.drawRect(rect)
    finally:
        painter.end()

    return QIcon(pixmap)


def _color_entries(include_none: bool, none_label: str = _NONE_LABEL) -> list[ComboEntry]:
    """Build the full, sorted (name, hex) entry list for the combo."""
    entries: list[ComboEntry] = []
    if include_none:
        entries.append((none_label, _NONE_VALUE))
    entries.extend(sorted(_iter_named_colors(), key=lambda item: item[0].lower()))
    return entries


class MatplotlibColorCombo(IconComboBox):
    """Combo box listing Matplotlib colors with a swatch icon.

    Parameters
    ----------
    include_none:
        If True, inserts a first entry whose stored value is an empty
        string. This keeps the widget compatible with editors that need
        to clear a color value.
    none_label:
        What to call that entry. It is "(none)" where an empty value
        clears a setting (the kwargs editor), and "Default" where an
        empty value means "let the style sheet's colour cycle decide" -
        which is what a series does with it, and which "(none)" read as
        "this series has no colour".

    Notes
    -----
    The canonical hex string (``#RRGGBB``) is stored as each entry's value
    and is what :meth:`current_hex` / :meth:`set_current_hex` operate on.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        include_none: bool = True,
        none_label: str = _NONE_LABEL,
    ) -> None:
        super().__init__(
            parent,
            entries=_color_entries(include_none, none_label),
            icon_for=_entry_icon,
            icon_size=QSize(_SWATCH_SIZE, _SWATCH_SIZE),
        )

    def current_hex(self) -> str:
        """Return the selected color as '#RRGGBB' (or '' for '(none)')."""
        return self.current_value()

    def set_current_hex(self, hex_color: str) -> bool:
        """Select the entry matching *hex_color* (with or without '#')."""
        raw = (hex_color or "").strip().lower()
        if not raw:
            return False
        target = raw if raw.startswith("#") else f"#{raw}"
        return self.set_current_value(target)

    def set_current_name(self, name: str) -> bool:
        """Select the first entry whose visible text matches *name*."""
        target = (name or "").strip().lower()
        if not target:
            return False
        for i in range(self.count()):
            if self.itemText(i).strip().lower() == target:
                self.setCurrentIndex(i)
                return True
        return False
