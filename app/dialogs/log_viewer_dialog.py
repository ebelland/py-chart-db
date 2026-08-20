from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import (
    QCloseEvent,
    QFontDatabase,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.logs.logger import LOG_FILE
from app.styles.style import (
    apply_card_layout,
    apply_dialog_shell,
    create_action_button,
    create_card_widget,
    create_section_title,
    load_icon,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _


class LogViewerDialog(QDialog):
    def __init__(self, parent: QWidget):
        self._parent = parent

        super().__init__(parent)

        self.setWindowTitle(_("Application Log"))
        self.setWindowIcon(load_icon("log_viewer"))

        try:
            log_path = Path(LOG_FILE)

            if log_path.exists():
                log_text = log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

                line_count = log_text.count("\n")
                if log_text and not log_text.endswith("\n"):
                    line_count += 1
            else:
                log_text = f"Log file not found:\n{LOG_FILE}"
                line_count = 0

        except OSError as exc:
            log_text = f"Unable to read log file:\n\n{exc}"
            line_count = 0

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="medium")

        card = create_card_widget(self, "logViewerCard")
        card_layout = QVBoxLayout(card)
        apply_card_layout(card_layout)

        card_layout.addWidget(
            create_section_title(_("Application Log"), card)
        )

        viewer = QPlainTextEdit(card)
        viewer.setObjectName("logViewerText")
        viewer.setReadOnly(True)
        viewer.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        viewer.setFont(
            QFontDatabase.systemFont(
                QFontDatabase.SystemFont.FixedFont
            )
        )
        viewer.setPlainText(log_text)
        viewer.moveCursor(
            viewer.textCursor().MoveOperation.End
        )

        mark_editor_panel(viewer)
        card_layout.addWidget(viewer, 1)

        caption = QLabel(
            f"{line_count} lines loaded from {LOG_FILE}",
            card,
        )
        caption.setProperty("muted", True)
        caption.setWordWrap(True)
        card_layout.addWidget(caption, 0)

        root.addWidget(card, 1)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)

        create_action_button(
            parent=self,
            action_id="copy",
            action=lambda: QApplication.clipboard().setText(viewer.toPlainText()),
            layout=action_row,
        )

        create_action_button(
            parent=self,
            action_id="close",
            action=self.accept,
            layout=action_row,
        )

        root.addLayout(action_row, 0)

    def closeEvent(self, event: QCloseEvent) -> None:
        super().closeEvent(event)
