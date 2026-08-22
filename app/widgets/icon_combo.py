# -*- coding: utf-8 -*-
"""app.utils.combo_base

IconComboBox

Shared base for QComboBox subclasses that pair every entry with a small
preview icon plus a string value (e.g. a color swatch, a line-style
preview, a marker-shape preview). Concrete widgets only need to supply the
list of (label, value) entries and a function that renders the icon for a
given value.

Notes for strict PySide6 typing / Pylance:
- Use Qt.ItemDataRole.UserRole (not Qt.UserRole)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QComboBox, QWidget

#: One dropdown entry: (visible label, stored value).
ComboEntry = tuple[str, str]

#: Default square icon size used when a subclass doesn't override it.
_DEFAULT_ICON_SIZE = QSize(16, 16)

#: Extra horizontal slack added when auto-sizing the dropdown popup.
_POPUP_PADDING = 24


class IconComboBox(QComboBox):
    """QComboBox where every entry carries an icon plus a string value.

    Subclasses populate the list via the constructor; this base handles
    item creation, value lookup/selection, and an auto-sized dropdown
    popup so long labels are never elided regardless of the combo's own
    (possibly narrow) width.
    """

    #: Role used to store each entry's string value (set via addItem's
    #: userData argument, which Qt stores at Qt.ItemDataRole.UserRole).
    ROLE_VALUE = Qt.ItemDataRole.UserRole

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        entries: Iterable[ComboEntry],
        icon_for: Callable[[str], QIcon],
        icon_size: QSize = _DEFAULT_ICON_SIZE,
    ) -> None:
        super().__init__(parent)
        self.setIconSize(icon_size)
        for label, value in entries:
            self.addItem(icon_for(value), label, value)

    # ------------------------------------------------------------------
    # Value access
    # ------------------------------------------------------------------
    def current_value(self) -> str:
        """Return the selected entry's stored value ('' if none)."""
        data = self.currentData(self.ROLE_VALUE)
        return str(data) if data is not None else ""

    def set_current_value(self, value: str) -> bool:
        """Select the first entry whose stored value matches *value*."""
        index = self.findData(value, self.ROLE_VALUE)
        if index < 0:
            return False
        self.setCurrentIndex(index)
        return True

    # ------------------------------------------------------------------
    # Auto-width dropdown
    # ------------------------------------------------------------------
    def showPopup(self) -> None:  # noqa: N802 - Qt override signature
        self._size_popup_to_contents()
        super().showPopup()

    def _size_popup_to_contents(self) -> None:
        """Widen the popup view so the longest label is never elided."""
        fm = self.fontMetrics()
        text_width = max(
            (fm.horizontalAdvance(self.itemText(i)) for i in range(self.count())),
            default=0,
        )
        icon_width = self.iconSize().width() + 10
        scrollbar_width = self.view().verticalScrollBar().sizeHint().width()
        width = text_width + icon_width + scrollbar_width + _POPUP_PADDING
        self.view().setMinimumWidth(max(width, self.width()))
