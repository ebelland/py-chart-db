# -*- coding: utf-8 -*-
"""app.widgets.marker_combo

MarkerStyleCombo

A PySide6 QComboBox that lists Matplotlib marker styles and shows a small
shape-preview icon next to each entry, mirroring ``MatplotlibColorCombo``.

This version includes the standard Matplotlib marker symbols from:
https://matplotlib.org/stable/api/markers_api.html
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from app.widgets.icon_combo import ComboEntry, IconComboBox
from app.styles.style import create_hidpi_pixmap


_ICON_SIZE = 16
_MARKER_COLOR = "#3a3a3a"
_PEN_WIDTH = 1.4


# Matplotlib marker entries.
# Stored value is the exact marker code passed to Matplotlib.
_ENTRIES: tuple[ComboEntry, ...] = (
    ("None", ""),
    ("Point .", "."),
    ("Pixel ,", ","),
    ("Circle o", "o"),
    ("Triangle down v", "v"),
    ("Triangle up ^", "^"),
    ("Triangle left <", "<"),
    ("Triangle right >", ">"),
    ("Tri down 1", "1"),
    ("Tri up 2", "2"),
    ("Tri left 3", "3"),
    ("Tri right 4", "4"),
    ("Octagon 8", "8"),
    ("Square s", "s"),
    ("Pentagon p", "p"),
    ("Plus filled P", "P"),
    ("Star *", "*"),
    ("Hexagon 1 h", "h"),
    ("Hexagon 2 H", "H"),
    ("Plus +", "+"),
    ("X x", "x"),
    ("X filled X", "X"),
    ("Diamond D", "D"),
    ("Thin diamond d", "d"),
    ("Vertical line |", "|"),
    ("Horizontal line _", "_"),
    ("Tick left 0", "0"),
    ("Tick right 1", "1"),
    ("Tick up 2", "2"),
    ("Tick down 3", "3"),
    ("Caret left 4", "4"),
    ("Caret right 5", "5"),
    ("Caret up 6", "6"),
    ("Caret down 7", "7"),
    ("Caret left base 8", "8"),
    ("Caret right base 9", "9"),
    ("Caret up base 10", "10"),
    ("Caret down base 11", "11"),
)


def _regular_polygon(
    cx: float,
    cy: float,
    radius: float,
    sides: int,
    rotation: float = -math.pi / 2,
) -> QPolygonF:
    """Build a regular polygon."""
    poly = QPolygonF()
    for i in range(sides):
        angle = rotation + (2 * math.pi * i / sides)
        poly.append(
            QPointF(
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle),
            )
        )
    return poly


def _star_polygon(cx: float, cy: float, outer_r: float, inner_r: float) -> QPolygonF:
    """Build a 5-point star polygon centered at cx/cy."""
    poly = QPolygonF()
    step = math.pi / 5
    start = -math.pi / 2
    for i in range(10):
        radius = outer_r if i % 2 == 0 else inner_r
        angle = start + i * step
        poly.append(
            QPointF(
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle),
            )
        )
    return poly


def _triangle_polygon(cx: float, cy: float, r: float, direction: str) -> QPolygonF:
    """Build a triangle pointing up/down/left/right."""
    if direction == "up":
        points = [
            QPointF(cx, cy - r),
            QPointF(cx + r, cy + r * 0.8),
            QPointF(cx - r, cy + r * 0.8),
        ]
    elif direction == "down":
        points = [
            QPointF(cx, cy + r),
            QPointF(cx + r, cy - r * 0.8),
            QPointF(cx - r, cy - r * 0.8),
        ]
    elif direction == "left":
        points = [
            QPointF(cx - r, cy),
            QPointF(cx + r * 0.8, cy - r),
            QPointF(cx + r * 0.8, cy + r),
        ]
    else:
        points = [
            QPointF(cx + r, cy),
            QPointF(cx - r * 0.8, cy - r),
            QPointF(cx - r * 0.8, cy + r),
        ]

    return QPolygonF(points)


# ----------------------------------------------------------------------
# Marker painters
#
# One function per Matplotlib marker code, all sharing the signature
# ``(painter, cx, cy, r)`` so that _DRAW_FUNCS can dispatch on the code
# alone.  The painter arrives with the pen and brush already set by
# _marker_icon, so a painter only describes the *shape*: never change colour,
# width or antialiasing here, or the preview stops matching the plot.
#
# ``r`` is the marker's nominal radius in device pixels, not a bounding box:
# shapes are drawn centred on (cx, cy) and are free to extend slightly past r
# where the Matplotlib glyph does (the star's outer points, for instance).
# ----------------------------------------------------------------------
def _draw_point(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawEllipse(QPointF(cx, cy), r * 0.35, r * 0.35)


def _draw_pixel(p: QPainter, cx: float, cy: float, r: float) -> None:
    side = max(2.0, r * 0.55)
    p.drawRect(QRectF(cx - side / 2, cy - side / 2, side, side))


def _draw_circle(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawEllipse(QPointF(cx, cy), r, r)


def _draw_triangle_up(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolygon(_triangle_polygon(cx, cy, r, "up"))


def _draw_triangle_down(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolygon(_triangle_polygon(cx, cy, r, "down"))


def _draw_triangle_left(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolygon(_triangle_polygon(cx, cy, r, "left"))


def _draw_triangle_right(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolygon(_triangle_polygon(cx, cy, r, "right"))


def _draw_square(p: QPainter, cx: float, cy: float, r: float) -> None:
    side = r * 1.7
    p.drawRect(QRectF(cx - side / 2, cy - side / 2, side, side))


def _draw_diamond(p: QPainter, cx: float, cy: float, r: float) -> None:
    poly = QPolygonF(
        [
            QPointF(cx, cy - r),
            QPointF(cx + r, cy),
            QPointF(cx, cy + r),
            QPointF(cx - r, cy),
        ]
    )
    p.drawPolygon(poly)


def _draw_thin_diamond(p: QPainter, cx: float, cy: float, r: float) -> None:
    poly = QPolygonF(
        [
            QPointF(cx, cy - r),
            QPointF(cx + r * 0.65, cy),
            QPointF(cx, cy + r),
            QPointF(cx - r * 0.65, cy),
        ]
    )
    p.drawPolygon(poly)


def _draw_octagon(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolygon(_regular_polygon(cx, cy, r, 8, rotation=math.pi / 8))


def _draw_pentagon(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolygon(_regular_polygon(cx, cy, r, 5))


def _draw_hexagon1(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolygon(_regular_polygon(cx, cy, r, 6, rotation=math.pi / 6))


def _draw_hexagon2(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolygon(_regular_polygon(cx, cy, r, 6, rotation=0.0))


def _draw_star(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolygon(_star_polygon(cx, cy, r, r * 0.45))


def _draw_plus(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))
    p.drawLine(QPointF(cx, cy - r), QPointF(cx, cy + r))


def _draw_x(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawLine(QPointF(cx - r, cy - r), QPointF(cx + r, cy + r))
    p.drawLine(QPointF(cx - r, cy + r), QPointF(cx + r, cy - r))


def _draw_filled_plus(p: QPainter, cx: float, cy: float, r: float) -> None:
    """Draw marker ``P``: a cross traced as one closed 12-point outline.

    Two overlapping rectangles would show their seam once the pen is visible,
    so the arm corners are walked in order instead.
    """
    w = r * 0.45
    poly = QPolygonF(
        [
            QPointF(cx - w, cy - r),
            QPointF(cx + w, cy - r),
            QPointF(cx + w, cy - w),
            QPointF(cx + r, cy - w),
            QPointF(cx + r, cy + w),
            QPointF(cx + w, cy + w),
            QPointF(cx + w, cy + r),
            QPointF(cx - w, cy + r),
            QPointF(cx - w, cy + w),
            QPointF(cx - r, cy + w),
            QPointF(cx - r, cy - w),
            QPointF(cx - w, cy - w),
        ]
    )
    p.drawPolygon(poly)


def _draw_filled_x(p: QPainter, cx: float, cy: float, r: float) -> None:
    # Approximation of Matplotlib's filled X marker.
    p.save()
    p.translate(cx, cy)
    p.rotate(45)
    _draw_filled_plus(p, 0.0, 0.0, r)
    p.restore()


def _draw_vertical_line(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawLine(QPointF(cx, cy - r), QPointF(cx, cy + r))


def _draw_horizontal_line(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))


def _draw_tri_down(p: QPainter, cx: float, cy: float, r: float) -> None:
    _draw_triangle_down(p, cx, cy, r * 0.78)


def _draw_tri_up(p: QPainter, cx: float, cy: float, r: float) -> None:
    _draw_triangle_up(p, cx, cy, r * 0.78)


def _draw_tri_left(p: QPainter, cx: float, cy: float, r: float) -> None:
    _draw_triangle_left(p, cx, cy, r * 0.78)


def _draw_tri_right(p: QPainter, cx: float, cy: float, r: float) -> None:
    _draw_triangle_right(p, cx, cy, r * 0.78)


def _draw_tick_left(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawLine(QPointF(cx - r, cy), QPointF(cx + r * 0.45, cy))


def _draw_caret_left(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolyline(
        QPolygonF(
            [
                QPointF(cx + r * 0.55, cy - r),
                QPointF(cx - r * 0.55, cy),
                QPointF(cx + r * 0.55, cy + r),
            ]
        )
    )


def _draw_caret_right(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolyline(
        QPolygonF(
            [
                QPointF(cx - r * 0.55, cy - r),
                QPointF(cx + r * 0.55, cy),
                QPointF(cx - r * 0.55, cy + r),
            ]
        )
    )


def _draw_caret_up(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolyline(
        QPolygonF(
            [
                QPointF(cx - r, cy + r * 0.55),
                QPointF(cx, cy - r * 0.55),
                QPointF(cx + r, cy + r * 0.55),
            ]
        )
    )


def _draw_caret_down(p: QPainter, cx: float, cy: float, r: float) -> None:
    p.drawPolyline(
        QPolygonF(
            [
                QPointF(cx - r, cy - r * 0.55),
                QPointF(cx, cy + r * 0.55),
                QPointF(cx + r, cy - r * 0.55),
            ]
        )
    )


def _draw_text_marker(p: QPainter, cx: float, cy: float, _r: float, text: str) -> None:
    """Fallback painter: render the marker code itself as centred text.

    Used for codes with no dedicated painter - notably ``$...$`` mathtext and
    any marker Matplotlib gains after this module was written - so an unknown
    marker still previews as something identifiable rather than a blank icon.
    """
    font = QFont()
    font.setPointSize(8)
    font.setBold(True)
    p.setFont(font)
    rect = QRectF(0, 0, _ICON_SIZE, _ICON_SIZE)
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


# The "_base" carets are Matplotlib's CARET*BASE markers: the same caret plus
# the baseline it sits against, drawn on the side the caret points away from.
# Only 9, 10 and 11 exist here.  CARETLEFTBASE is marker 8, and these keys are
# strings, where "8" already means the octagon - so it is unreachable and the
# function that drew it has been removed rather than left looking available.
def _draw_caret_right_base(p: QPainter, cx: float, cy: float, r: float) -> None:
    _draw_caret_right(p, cx, cy, r)
    p.drawLine(QPointF(cx - r * 0.65, cy - r), QPointF(cx - r * 0.65, cy + r))


def _draw_caret_up_base(p: QPainter, cx: float, cy: float, r: float) -> None:
    _draw_caret_up(p, cx, cy, r)
    p.drawLine(QPointF(cx - r, cy + r * 0.65), QPointF(cx + r, cy + r * 0.65))


def _draw_caret_down_base(p: QPainter, cx: float, cy: float, r: float) -> None:
    _draw_caret_down(p, cx, cy, r)
    p.drawLine(QPointF(cx - r, cy - r * 0.65), QPointF(cx + r, cy - r * 0.65))


_DRAW_FUNCS: dict[str, Callable[[QPainter, float, float, float], None]] = {
    ".": _draw_point,
    ",": _draw_pixel,
    "o": _draw_circle,
    "v": _draw_triangle_down,
    "^": _draw_triangle_up,
    "<": _draw_triangle_left,
    ">": _draw_triangle_right,
    "1": _draw_tri_down,
    "2": _draw_tri_up,
    "3": _draw_tri_left,
    "4": _draw_tri_right,
    "8": _draw_octagon,
    "s": _draw_square,
    "p": _draw_pentagon,
    "P": _draw_filled_plus,
    "*": _draw_star,
    "h": _draw_hexagon1,
    "H": _draw_hexagon2,
    "+": _draw_plus,
    "x": _draw_x,
    "X": _draw_filled_x,
    "D": _draw_diamond,
    "d": _draw_thin_diamond,
    "|": _draw_vertical_line,
    "_": _draw_horizontal_line,
    # Matplotlib's tick markers are the *integers* 0-3, while the strings
    # "1".."4" already mean tri_down, tri_up, tri_left and tri_right.  These
    # keys are strings, so only TICKLEFT survives the collision; the functions
    # that drew the other three were unreachable and have been removed.
    "0": _draw_tick_left,
    "5": _draw_caret_right,
    "6": _draw_caret_up,
    "7": _draw_caret_down,
    "9": _draw_caret_right_base,
    "10": _draw_caret_up_base,
    "11": _draw_caret_down_base,
}


_STROKE_ONLY = {
    "+",
    "x",
    "|",
    "_",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "9",
    "10",
    "11",
}


@lru_cache(maxsize=None)
def _marker_icon(marker: str) -> QIcon:
    """Build and cache a preview icon for one Matplotlib marker."""
    # Allocated at the display's pixel density: the coordinates below stay
    # logical, but the bitmap has the pixels to be sharp on a Retina screen.
    pixmap = create_hidpi_pixmap(_ICON_SIZE, _ICON_SIZE)

    if not marker:
        return QIcon(pixmap)

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        pen = QPen(QColor(_MARKER_COLOR))
        pen.setWidthF(_PEN_WIDTH)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)

        if marker in _STROKE_ONLY:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        else:
            painter.setBrush(QBrush(QColor(_MARKER_COLOR)))

        center = _ICON_SIZE / 2
        radius = _ICON_SIZE * 0.32

        draw = _DRAW_FUNCS.get(marker)
        if draw is not None:
            draw(painter, center, center, radius)
        else:
            _draw_text_marker(painter, center, center, radius, marker)

    finally:
        painter.end()

    return QIcon(pixmap)


class MarkerStyleCombo(IconComboBox):
    """Combo box listing Matplotlib markers with a shape-preview icon.

    The stored value is the Matplotlib marker code.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            entries=_ENTRIES,
            icon_for=_marker_icon,
            icon_size=QSize(_ICON_SIZE, _ICON_SIZE),
        )

    def current_marker(self) -> str:
        """Return the selected Matplotlib marker code."""
        return self.current_value()

    def set_current_marker(self, marker: str) -> bool:
        """Select the entry matching *marker*."""
        return self.set_current_value((marker or "").strip())