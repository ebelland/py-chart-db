"""Read-only HTML results pane used by the series-operation dialogs.

Displays formatted operation output and supports copying it to the clipboard as
HTML, which makes results pasteable into reports.

The widget deliberately mirrors part of the QLabel API (``setText``,
``setWordWrap``, ``setAlignment``, ``setTextInteractionFlags``) so it can be
dropped in wherever a label used to be, without every caller learning a new
interface. The layout-only setters are accepted and ignored.

Important behavior:
- ``setText(...)`` always treats content as plain text and escapes it.
- ``setHtml(...)`` always treats content as HTML markup.
- ``setContent(...)`` auto-detects whether the input looks like HTML and routes
  it to ``setHtml(...)`` or ``setText(...)``.
"""
from __future__ import annotations

import html
import re
from typing import cast

from PySide6.QtCore import QMimeData, QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.styles.style import (
    action_menu_item,
    create_menu,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _


_HTML_TAG_RE = re.compile(
    r"</?\s*("
    r"html|head|body|style|div|section|article|main|header|footer|"
    r"span|p|br|hr|"
    r"table|thead|tbody|tfoot|tr|td|th|caption|colgroup|col|"
    r"ul|ol|li|"
    r"h[1-6]|"
    r"strong|b|em|i|u|small|code|pre|blockquote|"
    r"a|img"
    r")\b",
    re.IGNORECASE,
)

_HTML_ENTITY_RE = re.compile(
    r"&(?:nbsp|amp|lt|gt|quot|apos|#[0-9]+|#x[0-9a-fA-F]+);"
)


def looks_like_html(value: str) -> bool:
    """Return True when text appears to contain real HTML markup.

    The test intentionally checks for known tags rather than any ``<...>``
    sequence, because plain text such as ``x < 5`` should remain plain text.
    HTML entities are treated as HTML too, because many report builders emit
    escaped content inside otherwise minimal markup.
    """
    text = str(value or "")
    return bool(_HTML_TAG_RE.search(text) or _HTML_ENTITY_RE.search(text))


def plain_to_html(text: str) -> str:
    """Escape plain text and wrap it in a minimal HTML document.

    Public because anything that has to hand plain text to something expecting
    markup needs exactly this, and doing it by hand is how unescaped ``<`` ends
    up truncating a report at the first angle bracket.
    """
    escaped = html.escape(str(text or ""), quote=True).replace("\n", "<br>")
    return (
        "<!doctype html>"
        "<html>"
        "<head>"
        "<meta charset='utf-8'>"
        "<style>"
        "body { font-family: Segoe UI, Arial, sans-serif; font-size: 10pt; }"
        "table { border-collapse: collapse; }"
        "th, td { padding: 4px 8px; border: 1px solid #d0d0d0; }"
        "th { font-weight: 600; }"
        "</style>"
        "</head>"
        f"<body>{escaped}</body>"
        "</html>"
    )


class HtmlResultsView(QWidget):
    """Reusable HTML results pane with Copy-as-HTML support."""

    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the pane: the viewer and a small custom context menu."""
        super().__init__(parent)
        self.setMinimumHeight(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self._html = ""
        self._plain_text = ""

        root = QVBoxLayout(self)
        stdSizeAndlayout(root)

        self._viewer = self._create_viewer()
        mark_editor_panel(self._viewer)
        root.addWidget(self._viewer, 1)

        self._install_context_menu()

    def _install_context_menu(self) -> None:
        """Install the same custom menu on the wrapper and the concrete viewer.

        QWebEngineView and QTextBrowser both provide their own standard context
        menu. Setting CustomContextMenu on the actual viewer suppresses that
        standard menu and guarantees that users only see Save, Clear and Copy.
        """
        for widget in (self, self._viewer):
            widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            widget.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, pos: QPoint) -> None:
        """Show the HTML results context menu."""
        source = self.sender()
        anchor = source if isinstance(source, QWidget) else self

        has_content = bool(self._html.strip() or self._plain_text.strip())
        menu = create_menu(
            self,
            [
                action_menu_item(
                    "save",
                    self.save_html_to_file,
                    enabled=has_content,
                    shortcut=None,
                ),
                action_menu_item(
                    "clear",
                    self.request_clear,
                    enabled=has_content,
                    shortcut=None,
                ),
                action_menu_item(
                    "copy",
                    self.copy_html_to_clipboard,
                    enabled=has_content,
                    shortcut=None,
                ),
            ],
        )
        menu.exec(anchor.mapToGlobal(pos))

    def _create_viewer(self) -> QWidget:
        """Return the best available HTML viewer.

        QtWebEngine renders CSS and tables properly but is a large optional
        dependency, so QTextBrowser is the fallback. QTextBrowser understands
        enough of the HTML subset for these operation result reports.
        """
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView

            return cast(QWidget, QWebEngineView(self))
        except Exception:
            pass

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)
        browser.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        return browser

    def setText(self, text: str) -> None:
        """Show plain text, escaped and line-broken into HTML."""
        self._plain_text = str(text or "")
        self.setHtml(plain_to_html(self._plain_text), plain_source=self._plain_text)

    def setHtml(self, markup: str, *, plain_source: str = "") -> None:
        """Show HTML markup directly."""
        self._html = str(markup or "")
        self._plain_text = str(plain_source or "")

        setter = getattr(self._viewer, "setHtml", None) or getattr(
            self._viewer,
            "setText",
            None,
        )
        if callable(setter):
            setter(self._html)

    def setContent(self, content: str) -> None:
        """Render HTML if content looks like markup, otherwise render as text.

        Use this from shared dialog code when callers may pass either plain text
        summaries or rich HTML reports.
        """
        value = str(content or "")
        if looks_like_html(value):
            self.setHtml(value)
        else:
            self.setText(value)

    def text(self) -> str:
        """Return the plain-text form, or HTML when HTML was set directly."""
        return self._plain_text if self._plain_text else self._html

    def toHtml(self) -> str:
        """Return the current markup."""
        return self._html

    def setWordWrap(self, _enabled: bool) -> None:
        """Accepted for QLabel compatibility; the viewer always wraps."""
        return None

    def setAlignment(self, _alignment: Qt.AlignmentFlag) -> None:
        """Accepted for QLabel compatibility; alignment comes from the markup."""
        return None

    def setTextInteractionFlags(self, _flags: Qt.TextInteractionFlag) -> None:
        """Accepted for QLabel compatibility; the viewer sets its own flags."""
        return None

    def request_clear(self) -> None:
        """Clear locally and notify an owning panel that it may collapse us."""
        self.clear()
        self.clear_requested.emit()

    def clear(self) -> None:
        """Clear the current HTML and plain-text content."""
        self._plain_text = ""
        self.setHtml("")

    def save_html_to_file(self) -> None:
        """Save the current HTML content to an .html file."""
        if not (self._html.strip() or self._plain_text.strip()):
            return
        file_path, _unused = QFileDialog.getSaveFileName(
            self,
            _("Save HTML"),
            "results.html",
            "HTML files (*.html *.htm);;Text files (*.txt);;All files (*)",
        )
        if not file_path:
            return
        path = str(file_path)
        content = self._html if self._html.strip() else plain_to_html(self._plain_text)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    def copy_html_to_clipboard(self) -> None:
        """Put both an HTML and a plain-text flavour on the clipboard.

        Rich-text targets such as Word or Outlook take the HTML. Plain-text
        targets get readable text instead of raw markup when possible.
        """
        mime = QMimeData()
        mime.setHtml(self._html)
        mime.setText(self._plain_text or self._html)
        QApplication.clipboard().setMimeData(mime)

    _plain_to_html = staticmethod(plain_to_html)
    looks_like_html = staticmethod(looks_like_html)
