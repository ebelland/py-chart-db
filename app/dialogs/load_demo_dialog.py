"""Pick one of the shipped demo projects, and build it on request.

The set itself - what each project shows, which figures and tables it needs -
is app/data/demo_project.py's; this dialog only presents the choice and asks
where to save it, the same way "New" asks where a blank database goes.

This is the only way in. A first run used to offer the whole set before the
window was even up; it now starts on an empty database and says nothing, so
the demo is something you go and get rather than something you decline.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.demo_project import DEMO_PROJECTS, DemoProject
from app.styles.style import (
    apply_dialog_shell,
    create_action_button,
    create_card_widget,
    create_section_title,
    load_icon,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _


class CreateDemoDialog(QDialog):
    """Let the user choose one demo project; ``chosen`` holds the result.

    Modal, and read through ``chosen`` rather than a signal: the caller wants
    one answer before it goes on to ask where to save it, not an ongoing
    conversation.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle(_("Create demo"))
        self.setWindowIcon(load_icon("plot"))
        self.chosen: DemoProject | None = None

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="small")

        card = create_card_widget(self, "createDemoCard")
        card_layout = QVBoxLayout(card)
        stdSizeAndlayout(card_layout)
        card_layout.addWidget(create_section_title(_("Create demo"), card))

        self._list = QListWidget(card)
        mark_editor_panel(self._list)
        for demo in DEMO_PROJECTS:
            item = QListWidgetItem(_(demo.file_name), self._list)
            item.setToolTip(_(demo.summary))
            item.setData(Qt.ItemDataRole.UserRole, demo)
        self._list.setCurrentRow(0)
        self._list.currentRowChanged.connect(self._update_summary)
        card_layout.addWidget(self._list, 1)

        self._summary = QLabel("", card)
        self._summary.setWordWrap(True)
        self._summary.setProperty("muted", True)
        card_layout.addWidget(self._summary, 0)
        self._update_summary(0)

        root.addWidget(card, 1)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        create_action_button(
            parent=self, action_id="apply", action=self._confirm, layout=action_row
        )
        create_action_button(
            parent=self, action_id="close", action=self.reject, layout=action_row
        )
        root.addLayout(action_row, 0)

    def _update_summary(self, row: int) -> None:
        item = self._list.item(row)
        self._summary.setText(_(item.data(Qt.ItemDataRole.UserRole).summary) if item else "")

    def _confirm(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        self.chosen = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

