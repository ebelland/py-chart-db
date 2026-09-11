"""The Series Operations panel: New plot, then every operation, all tiles.

Grouped into the same five sections the layout proposal settled on -
Plot on its own, then Analysis (Peaks, Roots, Calculus), Statistics
(Statistics, Outliers, Clustering, Control Chart), Signal Processing
(Smoothing, Spectral Analysis, Filtering, Baseline Correction), Modeling
(Fit, Interpolation, Function) - because "I want a control chart" is one
decision, not "which of five sections is a control chart in, then which
dialog in that section". Plot is a section of one for the same reason the
others exist: naming what it is, above the button, rather than folding it
wordlessly into "Analysis".

Every operation - Plot included - is a square tile: an icon over its name,
the same size and behaviour whichever section it is in, rather than the
one-row-per-operation list an earlier version drew Plot as (a "this one is
different" distinction the grid layout does not need: a tile that makes a
place to put a series is not read differently from one that analyses what
is already there). Square tiles have no room for a description, so it
moves to a bar fixed at the bottom of the panel (below the scroll area,
not inside it) that hover *or* keyboard focus fills in - not a tooltip,
which would vanish the moment the pointer left and say nothing to someone
tabbing through with a keyboard instead of a mouse.
"""
from __future__ import annotations

from html import unescape

from PySide6.QtCore import QEvent, QSize, Signal, Qt
from PySide6.QtGui import (
    QEnterEvent,
    QFocusEvent,
    QIcon,
    QKeyEvent,
    QMouseEvent,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.dialogs.create_chart_dialog import NewPlotTabDialog
from app.scanners.series_operation_scanner import series_operations
from app.styles.style import (
    create_card_widget,
    create_compact_section_title,
    icon_from_svg_source,
    stdSizeAndlayout,
)
from app.utils.i18n import _, tr
from app.widgets.base_properties import BaseProperties

_ACCENT = "#2563EB"

#: A tile plus its spacing needs about this much width to stay readable;
#: OperationSection recomputes its column count from the width it is
#: actually given divided by this, which is what lets two columns at the
#: panel's usual width become three if it is widened.
_TILE_MIN_WIDTH = 92
_TILE_SPACING = 6
_TILE_HEIGHT = 84

#: Section name -> the operation Name strings (SeriesOperationDialogBase.Name)
#: it holds, in display order. An operation whose Name is not listed here
#: - a new plugin dropped in without this being updated - lands in
#: _FALLBACK_SECTION rather than disappearing, so being forgotten here costs
#: it a good home, not a listing at all.
_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (_("Analysis"), ("Peaks", "Roots", "Calculus")),
    (_("Statistics"), ("Statistics", "Outliers", "Clustering", "Control Chart")),
    (
        _("Signal Processing"),
        ("Smoothing", "Spectral Analysis", "Filtering", "Baseline Correction"),
    ),
    (_("Modeling"), ("Fit", "Interpolation", "Function")),
)
_FALLBACK_SECTION = _("Other")


def _operation_action_id(operation: dict) -> str:
    return str(operation.get("value") or operation.get("name") or "")


def _group_by_section(operations: list[dict]) -> list[tuple[str, list[dict]]]:
    """Sort *operations* into ``_SECTIONS``' order, dropping empty sections.

    A section with none of its operations discovered - every one of them
    failed to import, say - is left out rather than shown as an empty
    header with nothing under it.
    """
    by_name = {_operation_action_id(op): op for op in operations}
    grouped: list[tuple[str, list[dict]]] = []
    placed: set[str] = set()

    for title, names in _SECTIONS:
        items = [by_name[name] for name in names if name in by_name]
        placed.update(names)
        if items:
            grouped.append((title, items))

    leftover = [op for op in operations if _operation_action_id(op) not in placed]
    if leftover:
        grouped.append((_FALLBACK_SECTION, leftover))
    return grouped


class OperationTile(QFrame):
    """One square operation tile: an icon over its name.

    ``hovered``/``left`` carry the description to whatever is showing it -
    SeriesOperationWidget's hint bar - because a tooltip cannot hold a
    paragraph and disappears the instant the pointer moves, which is wrong
    for text someone is in the middle of reading. Keyboard focus fires the
    same two signals as hover, so tabbing through the grid explains each
    tile exactly as pointing at it does.
    """

    clicked = Signal()
    hovered = Signal(str, str)
    left = Signal()
    ICON_SIZE = 26

    def __init__(self, *, parent: QWidget, icon: QIcon, title: str, description: str) -> None:
        super().__init__(parent)
        self.setObjectName("operationTile")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFixedHeight(_TILE_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._title = title
        self._description = description

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        icon_label = QLabel(self)
        icon_label.setFixedSize(self.ICON_SIZE, self.ICON_SIZE)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setPixmap(icon.pixmap(QSize(self.ICON_SIZE, self.ICON_SIZE)))
        layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignHCenter)

        title_label = QLabel(title, self)
        title_label.setProperty("operationTileTitle", True)
        title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        title_label.setWordWrap(True)
        title_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        layout.addWidget(title_label, 1)

        self.setAccessibleName(title)
        self.setAccessibleDescription(description)

    def enterEvent(self, event: QEnterEvent) -> None:
        self.hovered.emit(self._title, self._description)
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self.left.emit()
        super().leaveEvent(event)

    def focusInEvent(self, event: QFocusEvent) -> None:
        self.hovered.emit(self._title, self._description)
        super().focusInEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        self.left.emit()
        super().focusOutEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class OperationSection(QWidget):
    """One section header plus its tiles, in a grid that recomputes its
    own column count from the width it is given (see ``resizeEvent``)."""

    operation_clicked = Signal(dict)
    tile_hovered = Signal(str, str)
    tile_left = Signal()

    def __init__(self, parent: QWidget, title: str, operations: list[dict]) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(create_compact_section_title(title, self))

        self._grid_host = QWidget(self)
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(_TILE_SPACING)
        layout.addWidget(self._grid_host)

        self._tiles: list[OperationTile] = []
        self._columns = 0
        for operation in operations:
            action_id = _operation_action_id(operation)
            tile = OperationTile(
                parent=self._grid_host,
                icon=SeriesOperationWidget.plugin_icon(operation),
                title=tr(action_id),
                description=tr(str(operation.get("description") or "")),
            )
            tile.clicked.connect(lambda op=operation: self.operation_clicked.emit(op))
            tile.hovered.connect(self.tile_hovered)
            tile.left.connect(self.tile_left)
            self._tiles.append(tile)

        self._relayout(2)

    def _relayout(self, columns: int) -> None:
        columns = max(1, columns)
        if columns == self._columns:
            return
        self._columns = columns
        while self._grid.count():
            self._grid.takeAt(0)
        for index, tile in enumerate(self._tiles):
            row, col = divmod(index, columns)
            self._grid.addWidget(tile, row, col)
        for col in range(columns):
            self._grid.setColumnStretch(col, 1)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        columns = max(1, min(len(self._tiles), self.width() // _TILE_MIN_WIDTH))
        self._relayout(columns)


class SeriesOperationWidget(BaseProperties):
    operation_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        root_layout = QVBoxLayout(self)
        stdSizeAndlayout(root_layout)

        page = create_card_widget(self, "seriesOperationsPageCard")
        page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root_layout.addWidget(page, 1)

        page_layout = QVBoxLayout(page)
        stdSizeAndlayout(page_layout)

        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        page_layout.addWidget(scroll, 1)

        content = QWidget(scroll)
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        stdSizeAndlayout(layout)
        layout.setSpacing(2)

        # Plot is a section of one, built the same way and out of the same
        # tile as every operation below it - see the module docstring for
        # why it no longer gets the old wide-row treatment.
        self._add_section(content, layout, _("Plot"), [self.plot_operation()])
        layout.addSpacing(16)
        for title, operations in _group_by_section(list(series_operations)):
            self._add_section(content, layout, title, operations)
            layout.addSpacing(12)

        layout.addStretch(1)

        # Fixed at the panel's bottom, outside the scroll area: a tile's
        # description belongs somewhere that stays put while it is read,
        # not scrolled away with whichever section happened to be hovered.
        self._hint_label = QLabel(_("Point at an operation for details"), page)
        self._hint_label.setObjectName("operationHint")
        self._hint_label.setProperty("muted", True)
        self._hint_label.setWordWrap(True)
        page_layout.addWidget(self._hint_label, 0)

    def _add_section(
        self, content: QWidget, layout: QVBoxLayout, title: str, operations: list[dict]
    ) -> OperationSection:
        section = OperationSection(content, title, operations)
        section.operation_clicked.connect(self.operation_requested)
        section.tile_hovered.connect(self._show_hint)
        section.tile_left.connect(self._clear_hint)
        layout.addWidget(section)
        return section

    def _reload_from_descriptor(self) -> None:
        """No-op: this panel lists operations, it does not edit a descriptor.

        Required by BaseProperties; nothing here reads from a connected
        figure, so there is nothing to reload when one changes.
        """

    def _set_enabled_state(self, enabled: bool) -> None:
        """No-op for the same reason: every tile stays clickable regardless.

        Whether an operation can actually run is decided where it is opened
        (main_window._open_series_operation refuses without a current
        chart), not by disabling the button that asks for one.
        """

    def _show_hint(self, title: str, description: str) -> None:
        self._hint_label.setProperty("muted", False)
        text = f"<b>{title}</b>"
        if description:
            text += f" — {description}"
        self._hint_label.setText(text)
        self._hint_label.style().unpolish(self._hint_label)
        self._hint_label.style().polish(self._hint_label)

    def _clear_hint(self) -> None:
        self._hint_label.setProperty("muted", True)
        self._hint_label.setText(_("Point at an operation for details"))
        self._hint_label.style().unpolish(self._hint_label)
        self._hint_label.style().polish(self._hint_label)

    @staticmethod
    def plot_operation() -> dict:
        return {
            "name": "NewPlotTabDialog",
            "value": "Plot",
            "description": getattr(NewPlotTabDialog, "Description", "Create a new plot"),
            "icon": NewPlotTabDialog.Icon,
            "builtin": True,
        }

    @staticmethod
    def plugin_icon(operation: dict) -> QIcon:
        svg_source = operation.get("icon") or operation.get("Icon") or ""
        svg_source = unescape(str(svg_source)).strip()
        if not svg_source:
            return QIcon()
        # The wrapping lives in style.icon_from_svg_source now; the accent
        # colour is this list's own, so it is passed rather than assumed.
        return icon_from_svg_source(svg_source, color=_ACCENT)
