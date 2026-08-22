"""Who made this, and what it is made of.

Short on purpose: the page is assembled from ``app/utils/credits``, which
reads the dependency list and the installed packages' own metadata, so this
module only decides what the page looks like.

It is also a licence statement, not only a thank-you - every library here is
someone's work, under a licence this application relies on - which is why the
licence column is not decoration and why the version shown is the one actually
installed rather than the one pinned.
"""
from __future__ import annotations

import html

from PySide6.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QWidget

from app import APP_NAME, APP_VERSION
from app.styles.style import (
    apply_dialog_shell,
    create_action_button,
    create_card_widget,
    create_section_title,
    load_icon,
    stdSizeAndlayout,
)
from app.utils import credits, report_html
from app.utils.i18n import _
from app.widgets.html_results import HtmlResultsView


class CreditsDialog(QDialog):
    """Show the author, the assistants and every dependency with its licence."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle(_("Credits"))
        self.setWindowIcon(load_icon("info"))

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="small")

        card = create_card_widget(self, "creditsCard")
        card_layout = QVBoxLayout(card)
        stdSizeAndlayout(card_layout)
        card_layout.addWidget(create_section_title(_("Credits"), card))

        # The same pane the operation reports use, so the credits can be
        # copied as HTML like everything else in the application.
        self._view = HtmlResultsView(card)
        self._view.setContent(credits_html())
        card_layout.addWidget(self._view, 1)
        root.addWidget(card, 1)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        # "dismiss", not "close": the catalogue's close button says Cancel,
        # and there is nothing here to cancel.
        create_action_button(
            parent=self, action_id="dismiss", action=self.reject, layout=action_row
        )
        root.addLayout(action_row, 0)


def credits_html() -> str:
    """Return the credits page, in the house report style."""
    return report_html.document(
        _("Credits"),
        f"{APP_NAME} {APP_VERSION}",
        report_html.section(_("Written by"), _people_table()),
        report_html.section(_("Libraries"), _libraries_table()),
        report_html.section(
            _("Licences"),
            report_html.note(
                _(
                    "Each library is used under its own licence, named above "
                    "and shipped with the package it belongs to. Versions are "
                    "the ones installed here, not the ones pinned."
                )
            ),
        ),
    )


def _people_table() -> str:
    """Return the author and the assistants, with what each contributed.

    Two columns rather than three: the role belongs under the name it
    describes, and a third column of one word each only makes the column that
    carries the sentences narrower.
    """
    rows = [
        (_named(credits.AUTHOR, _("Author")), html.escape(
            _("Design, direction, and the domain this exists for.")
        ))
    ]
    rows.extend(
        (_named(name, maker), html.escape(_(did)))
        for name, maker, did in credits.ASSISTANTS
    )

    return report_html.table(
        (_("Name"), _("Contribution")),
        rows,
        align=("left", "left"),
    )


def _libraries_table() -> str:
    """Return every declared dependency, with its version and licence.

    What a library is for goes under its name rather than in a fourth column:
    a summary is a sentence, and a sentence in a column beside three short
    ones is the column that gets cut off.
    """
    rows = []
    for package in credits.packages():
        purpose = _shorten(package.summary)
        if package.marker:
            # "macOS only" is part of what this dependency *is*, so it belongs
            # in the row rather than in a footnote nobody reads.
            purpose = f"{purpose} ({package.marker})" if purpose else package.marker
        rows.append(
            (
                _named(package.name, purpose),
                html.escape(package.installed or _("not installed")),
                html.escape(package.license or "-"),
            )
        )

    return report_html.table(
        (_("Library"), _("Version"), _("Licence")),
        rows,
        align=("left", "left", "left"),
        empty_message=_("The dependency list is not available in this build."),
    )


def _named(name: str, note: str) -> str:
    """Return a bold name with a muted second line under it."""
    if not note:
        return f"<b>{html.escape(name)}</b>"
    return (
        f"<b>{html.escape(name)}</b><br>"
        f"<span style='color:#6b7280;'>{html.escape(note)}</span>"
    )


def _shorten(text: str, limit: int = 52) -> str:
    """Return a one-line summary that fits a table cell.

    A package's own Summary runs to whatever length its author liked, and one
    long line makes every other column narrower or puts a horizontal scrollbar
    under the whole page. Cut at a word rather than mid-syllable.
    """
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    cut = clean[:limit].rsplit(" ", 1)[0]
    return f"{cut or clean[:limit]}\u2026"
