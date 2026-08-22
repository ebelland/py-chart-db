from __future__ import annotations

from html import unescape

from PySide6.QtCore import QSize, Signal, Qt
from PySide6.QtGui import QFont, QIcon, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.dialogs.create_chart_dialog import NewPlotTabDialog
from app.scanners.series_operation_scanner import series_operations
from app.styles.style import create_card_widget, icon_from_svg_source, stdSizeAndlayout
from app.utils.i18n import _, tr

_ACCENT = "#2563EB"


class OperationRow(QFrame):
    clicked = Signal()
    ICON_SIZE = 20

    def __init__(self, *, parent: QWidget, icon: QIcon, title: str, description: str) -> None:
        super().__init__(parent)
        self.setObjectName("operationRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        row_layout = QHBoxLayout(self)
        row_layout.setContentsMargins(10, 8, 10, 8)
        row_layout.setSpacing(10)

        icon_label = QLabel(self)
        icon_label.setObjectName("operationIcon")
        icon_label.setProperty("fluentIcon", True)
        icon_label.setFixedSize(self.ICON_SIZE, self.ICON_SIZE)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setPixmap(icon.pixmap(QSize(self.ICON_SIZE, self.ICON_SIZE)))
        row_layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)

        title_label = QLabel(title, self)
        title_label.setProperty("operationTitle", True)
        title_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        title_font = QFont(title_label.font())
        title_point_size = title_font.pointSize()
        if title_point_size > 0:
            title_font.setPointSize(max(title_point_size, 10))
        else:
            title_font.setPixelSize(13)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_label.setFont(title_font)

        description_label = QLabel(description, self)
        description_label.setProperty("operationDescription", True)
        description_label.setWordWrap(False)
        description_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        description_font = QFont(description_label.font())
        description_point_size = description_font.pointSize()
        if description_point_size > 0:
            description_font.setPointSize(max(description_point_size - 1, 9))
        else:
            description_font.setPixelSize(12)

        description_label.setFont(description_font)

        text_layout.addWidget(title_label)
        text_layout.addWidget(description_label)
        row_layout.addLayout(text_layout, 1)

        self.setAccessibleName(title)
        self.setAccessibleDescription(description)

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


class SeriesOperationWidget(QWidget):
    operation_requested = Signal(dict)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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

        self._add_section_title(layout, _("Plot"))
        self._add_operation_item(layout=layout, operation=self.plot_operation())

        layout.addSpacing(16)
        self._add_section_title(layout, _("Series Operations"))
        for operation in series_operations:
            self._add_operation_item(layout=layout, operation=operation)

        layout.addStretch(1)

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

    def _add_section_title(self, layout: QVBoxLayout, title: str) -> None:
        label = QLabel(title, self)
        label.setProperty("sectionTitle", True)
        section_font = QFont(label.font())
        section_point_size = section_font.pointSize()
        if section_point_size > 0:
            section_font.setPointSize(max(section_point_size - 1, 9))
        else:
            section_font.setPixelSize(12)
        section_font.setWeight(QFont.Weight.DemiBold)
        label.setFont(section_font)
        layout.addWidget(label)

    def _add_operation_item(self, *, layout: QVBoxLayout, operation: dict) -> None:
        action_id = str(operation.get("value") or operation.get("name") or "")
        title = tr(action_id)
        description = tr(str(operation.get("description") or "Open operation"))
        row = OperationRow(
            parent=self,
            icon=self.plugin_icon(operation),
            title=title,
            description=description,
        )
        row.clicked.connect(lambda op=operation: self.operation_requested.emit(op))
        layout.addWidget(row)
