# -*- coding: utf-8 -*-
"""app.utils.class_line_combo

LineStyleCombo

A PySide6 QComboBox that lists Matplotlib line styles and shows a small
line-preview icon next to each entry, mirroring ``MatplotlibColorCombo``.
The dropdown popup auto-sizes to fit the longest label, independent of how
narrow the combo box itself is.

Notes for strict PySide6 typing / Pylance:
- Use Qt.PenStyle.SolidLine (not Qt.SolidLine)
- Use Qt.GlobalColor.transparent (not Qt.transparent)

This module is intentionally standalone and PEP8 compliant.
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.styles.style import create_hidpi_pixmap
from app.utils.combo_base import ComboEntry, IconComboBox

_ICON_W, _ICON_H = 28, 14
_LINE_COLOR = "#3a3a3a"
_LINE_WIDTH = 2

#: (label, Matplotlib linestyle code) — same entries as the original
#: hand-rolled combo, now paired with a preview icon.
_ENTRIES: tuple[ComboEntry, ...] = (
    ("Solid", "-"),
    ("Dashed", "--"),
    ("Dash-dot", "-."),
    ("Dotted", ":"),
    ("None", "none"),
)

#: Matplotlib linestyle code -> Qt pen style. ``None`` means "draw nothing"
#: (used for the "None" / no-line entry).
_PEN_STYLES: dict[str, Qt.PenStyle | None] = {
    "-": Qt.PenStyle.SolidLine,
    "--": Qt.PenStyle.DashLine,
    "-.": Qt.PenStyle.DashDotLine,
    ":": Qt.PenStyle.DotLine,
    "none": None,
}


@lru_cache(maxsize=None)
def _line_icon(linestyle: str) -> QIcon:
    """Build (and cache) a preview icon for one Matplotlib linestyle."""
    # Allocated at the display's pixel density: the coordinates below stay
    # logical, but the bitmap has the pixels to be sharp on a Retina screen.
    pixmap = create_hidpi_pixmap(_ICON_W, _ICON_H)

    pen_style = _PEN_STYLES.get(linestyle)
    if pen_style is not None:
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(QColor(_LINE_COLOR))
            pen.setWidth(_LINE_WIDTH)
            pen.setStyle(pen_style)
            painter.setPen(pen)
            mid_y = _ICON_H // 2
            painter.drawLine(2, mid_y, _ICON_W - 2, mid_y)
        finally:
            painter.end()

    return QIcon(pixmap)


class LineStyleCombo(IconComboBox):
    """Combo box listing Matplotlib line styles with a preview icon.

    The stored value is the Matplotlib linestyle code (``"-"``, ``"--"``,
    ``"-."``, ``":"`` or ``"none"``), matching what was previously hand-
    rolled inline.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            entries=_ENTRIES,
            icon_for=_line_icon,
            icon_size=QSize(_ICON_W, _ICON_H),
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed,)

    def current_linestyle(self) -> str:
        """Return the selected Matplotlib linestyle code."""
        return self.current_value()

    def set_current_linestyle(self, linestyle: str) -> bool:
        """Select the entry matching *linestyle* (e.g. ``"--"``)."""
        return self.set_current_value((linestyle or "").strip())
