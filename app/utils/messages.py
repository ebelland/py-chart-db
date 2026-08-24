"""Every message box the application shows, in one catalogue.

Fifty-eight ``QMessageBox`` calls were scattered across the dialogs, each with
its title and its wording written inline.  That made two things impossible:
translating them, and keeping them consistent - the same situation was
announced as "Import" in one place and "Import failed" three lines later.

The catalogue lives in ``config.json`` under ``"messages"``, and only there -
this module holds no copy of it::

    "messages": {
      "import.nothing_to_import": {
        "level": "info",
        "title": "Import",
        "text": "Nothing to import."
      },
      ...
    }

One source, deliberately.  A default table in Python next to the JSON would
mean two catalogues that agree only until someone edits one of them, and the
disagreement would be invisible: the wrong wording would simply appear in a
box.  The file is versioned with the code, so it is as reviewable as the table
would have been, and editing a message no longer means touching a .py.

Call sites name the situation and supply the values, never the wording::

    show_message(self, "import.no_table_name")
    show_message(self, "preview.delete_column_failed", error=exc)
    if ask(self, "preview.confirm_delete_column", column=name):
        ...

``text`` may contain ``{name}`` placeholders filled from the keyword
arguments.  Named, not positional, so a translation is free to reorder them -
Italian frequently must.  Title and text are translated through
:func:`app.utils.i18n.tr`, so the English text in the JSON is the gettext
message id.

``level`` picks the icon and the buttons: ``info``, ``warning``, ``error``, or
``question`` (Yes/No, answered through :func:`ask`).

An id with no entry does not raise: the situation being reported is real, so
the box is shown with the id as its title and the miss is logged.  The test
suite fails on any id the code uses and the file does not define, which is
what makes that fallback a safety net rather than a way to lose messages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QWidget

from app.logs.logger import applogger
from app.utils.config import get_section
from app.utils.i18n import tr

CONFIG_SECTION = "messages"

Level = Literal["info", "warning", "error", "question"]

_ICONS: dict[str, QMessageBox.Icon] = {
    "info": QMessageBox.Icon.Information,
    "warning": QMessageBox.Icon.Warning,
    "error": QMessageBox.Icon.Critical,
    "question": QMessageBox.Icon.Question,
}


@dataclass(frozen=True, slots=True)
class MessageSpec:
    """One situation the application needs to report."""

    message_id: str
    level: Level
    title: str
    text: str

    @classmethod
    def from_config(cls, message_id: str, entry: dict[str, Any]) -> MessageSpec:
        """Build a spec from one ``config.json`` entry, tolerating gaps."""
        level = str(entry.get("level") or "info")
        return cls(
            message_id=message_id,
            level=level if level in _ICONS else "info",  # type: ignore[arg-type]
            title=str(entry.get("title") or ""),
            text=str(entry.get("text") or ""),
        )

    def to_config(self) -> dict[str, Any]:
        """Return the JSON form of this spec, as ``config.json`` stores it."""
        return {"level": self.level, "title": self.title, "text": self.text}

    @staticmethod
    def _normalize_format_fields(fields: dict[str, Any]) -> dict[str, Any]:
        """Return the mapping passed to ``str.format(**mapping)``.

        ``ask`` and ``show_message`` receive placeholder values through
        ``**fields``.  That means the correct call is::

            ask(parent, "axis.confirm_delete", axis="name")

        Some call sites already have a dictionary and pass it as one keyword::

            values = {"axis": "name"}
            ask(parent, "axis.confirm_delete", fields=values)

        In that case the collected ``fields`` value is
        ``{"fields": {"axis": "name"}}``.  ``str.format`` cannot fill
        ``{axis}`` from that shape, so this helper unwraps the nested dictionary
        before formatting.  Direct keyword arguments still win if both forms are
        present.
        """
        nested = fields.get("fields")
        if not isinstance(nested, dict):
            return dict(fields)

        normalized: dict[str, Any] = dict(nested)
        normalized.update(
            {key: value for key, value in fields.items() if key != "fields"}
        )
        return normalized

    def _format_template(self, template: str, fields: dict[str, Any]) -> str:
        """Format one translated message template with caller fields.

        The normal syntax is::

            ask(parent, "axis.confirm_delete", axis="Axis 1")

        and catalogue text uses ``{axis}``.  If a caller already has a dict, it
        must unpack it at the call site, for example ``ask(..., **values)``.

        As a compatibility guard, a single nested ``fields={...}`` dictionary is
        unwrapped too.  Missing placeholders are logged and left visible rather
        than raising inside a message box.
        """
        if not template:
            return template

        values = self._normalize_format_fields(fields) if fields else {}
        if not values:
            return template

        normalized = template
        for key in values:
            normalized = normalized.replace("{{" + key + "}}", "{" + key + "}")

        try:
            return normalized.format(**values)
        except (KeyError, IndexError, ValueError) as exc:
            applogger.warning(
                "Message '%s' could not be formatted: %r fields=%r error=%r",
                self.message_id,
                normalized,
                values,
                exc,
                show_dialog=False,
                raise_error=False,
            )
            return normalized


    def translated(self, **fields: Any) -> tuple[str, str]:
        """Return the translated ``(title, text)`` with placeholders filled.

        Preferred usage passes placeholders as normal keywords::

            ask(self, "axis.confirm_delete", axis="name")

        If a wrapper already has a dictionary, it can pass it as
        ``fields=that_dict`` and it will be unwrapped before formatting.
        """
        title = tr(self.title) if self.title else ""
        text = tr(self.text) if self.text else ""

        if not fields:
            return title, text

        return (
            self._format_template(title, dict(fields)),
            self._format_template(text, dict(fields)),
        )

# Read once from config.json and kept, because every message box would
# otherwise re-read and re-parse the whole file.  reload_messages() drops it.
_cache: dict[str, MessageSpec] | None = None


def catalog() -> dict[str, MessageSpec]:
    """Return the whole catalogue, read from ``config.json`` on first use.

    Nothing is written back: the file is the catalogue, not a cache of one.  A
    malformed entry is skipped with a log line rather than taking the rest of
    the file with it - one bad edit should cost one message, not all of them.
    """
    global _cache
    if _cache is not None:
        return _cache

    entries = get_section(CONFIG_SECTION)
    if not entries:
        applogger.error(
            "No '%s' section in config.json; every message box will show its "
            "id instead of its text.",
            CONFIG_SECTION,
            show_dialog=False,
            raise_error=False,
        )

    built: dict[str, MessageSpec] = {}
    for message_id, entry in entries.items():
        if not isinstance(entry, dict):
            applogger.warning(
                "Message '%s' in config.json is not an object; ignoring it.",
                message_id,
                show_dialog=False,
                raise_error=False,
            )
            continue
        built[str(message_id)] = MessageSpec.from_config(str(message_id), entry)

    _cache = built
    return _cache


def reload_messages() -> None:
    """Forget the cached catalogue, so an edited config.json is picked up."""
    global _cache
    _cache = None


def message(message_id: str) -> MessageSpec:
    """Return the spec for a message id.

    An unknown id must still produce a visible box - the situation it was
    reporting is real - so it degrades to the id itself and is logged.  The
    test suite fails on any id the code uses and config.json does not define,
    so this path means a hand-edited file, not a typo that shipped.
    """
    spec = catalog().get(message_id)
    if spec is not None:
        return spec

    applogger.error(
        "Unknown message id %r; showing it verbatim.",
        message_id,
        show_dialog=False,
        raise_error=False,
    )
    return MessageSpec(message_id, "info", message_id, "")


def show_message(
    parent: QWidget | None,
    message_id: str,
    *,
    title: str | None = None,
    details: str | None = None,
    **fields: Any,
) -> None:
    """Show one catalogued message.

    ``title`` overrides the catalogue for the dialogs that name themselves at
    runtime (a series operation titles its boxes after the operation).

    ``details`` is for findings rather than prose - the list of problems a
    database check turned up, say.  It goes behind Show Details, which is
    Qt's own scrollable, selectable, copy-and-pasteable pane: a list of
    twenty is unreadable as twenty more lines in the box, and a list of two
    hundred makes the box taller than the screen.  The catalogue text still
    says what happened; this says what exactly.

    Both are shown as plain text.  Qt's default is AutoText, which guesses
    at rich text and, having guessed, treats every newline as a space - so
    one table called ``<old>`` would collapse a whole report onto one line.
    Nothing in the catalogue is written as HTML, so there is nothing to lose
    by saying so.
    """
    spec = message(message_id)
    catalog_title, text = spec.translated(**fields)

    box = QMessageBox(parent)
    box.setIcon(_ICONS.get(spec.level, QMessageBox.Icon.Information))
    box.setWindowTitle(title or catalog_title)
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setText(text)
    if details and details.strip():
        box.setDetailedText(details)
    box.exec()


def ask(
    parent: QWidget | None,
    message_id: str,
    *,
    title: str | None = None,
    default_yes: bool = False,
    **fields: Any,
) -> bool:
    """Ask a catalogued yes/no question; returns True for Yes.

    ``default_yes`` is off by default because every question in this
    application asks about something destructive, and Return should not be the
    fast path to it.
    """
    spec = message(message_id)
    catalog_title, text = spec.translated(**fields)

    box = QMessageBox(parent)
    box.setIcon(_ICONS.get(spec.level, QMessageBox.Icon.Question))
    box.setWindowTitle(title or catalog_title)
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setText(text)
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.setDefaultButton(
        QMessageBox.StandardButton.Yes if default_yes else QMessageBox.StandardButton.No
    )
    return box.exec() == QMessageBox.StandardButton.Yes
