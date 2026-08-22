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

from PySide6.QtCore import QRect, QSize
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen
from PySide6.QtWidgets import QWidget

from matplotlib.colors import (
    BASE_COLORS,
    CSS4_COLORS,
    TABLEAU_COLORS,
    XKCD_COLORS,
    to_hex,
)

from app.widgets.icon_combo import ComboEntry, IconComboBox
from app.styles.style import create_hidpi_pixmap

_NONE_LABEL = "(none)"
_NONE_VALUE = ""
_SWATCH_SIZE = 14
_BORDER_COLOR = "#5a5a5a"


def _iter_named_colors() -> Iterator[ComboEntry]:
    """Yield (name, hex_color) pairs from every Matplotlib color table."""
    for mapping in (BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS):
        for name, value in mapping.items():
            try:
                yield name, to_hex(value)
            except ValueError:
                continue  # Skip malformed table entries.


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


def _color_entries(include_none: bool) -> list[ComboEntry]:
    """Build the full, sorted (name, hex) entry list for the combo."""
    entries: list[ComboEntry] = []
    if include_none:
        entries.append((_NONE_LABEL, _NONE_VALUE))
    entries.extend(sorted(_iter_named_colors(), key=lambda item: item[0].lower()))
    return entries


class MatplotlibColorCombo(IconComboBox):
    """Combo box listing Matplotlib colors with a swatch icon.

    Parameters
    ----------
    include_none:
        If True, inserts a '(none)' entry whose stored value is an empty
        string. This keeps the widget compatible with editors that need
        to clear a color value.

    Notes
    -----
    The canonical hex string (``#RRGGBB``) is stored as each entry's value
    and is what :meth:`current_hex` / :meth:`set_current_hex` operate on.
    """

    def __init__(self, parent: QWidget | None = None, *, include_none: bool = True) -> None:
        super().__init__(
            parent,
            entries=_color_entries(include_none),
            icon_for=_swatch_icon,
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
