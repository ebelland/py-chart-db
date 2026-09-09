"""Application preferences, in one place the user can reach.

These values all lived in ``config.json`` and could only be changed by editing
it by hand - which meant that in practice they were never changed, and that a
typo in one of them showed up as an unstyled window or an untranslated menu
with nothing to point at.

Everything here is machine-level and so belongs in config.json rather than in
a .dhub file: it describes this installation, not any particular figure.

Nothing here applies live, and the dialog says so rather than leaving the user
to wonder.

The style used to preview itself as you moved through the list, because seeing
a stylesheet is the point of choosing it.  That had to go: applying a style
re-polishes every widget in the application, and ``QApplication.setStyle`` for
a Qt plugin destroys and rebuilds the QStyle underneath them all.  When one of
those widgets is a ``QWebEngineView`` - the HTML results pane, when PySide6's
addons are installed - the repolish reaches into Chromium's own delegate and
takes the process down with a segmentation fault, in ``CrBrowserMain``, below
anything Python can catch.

The language never applied live either, for a duller reason: menus and labels
read the catalogue when they are built, so switching mid-session translates
whatever is constructed next and leaves the rest in the old language.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.logs.logger import applogger
from app.styles.style import (
    CONFIG_APP_STYLE,
    available_app_styles,
    apply_card_layout,
    apply_dialog_shell,
    configure_combo_width,
    create_action_button,
    create_card_widget,
    create_section_title,
    load_icon,
    qss_file_style_key,
    qss_file_style_label,
    resolve_app_style,
    stdSizeAndlayout,
)
from app.utils.config import get_language, get_value, set_value
from app.utils.i18n import (
    AUTO_LANGUAGE,
    available_languages,
    platform_language,
    tr,
)

#: Config key and choices for the default chart export format.  The chart
#: panel's save dialog offers all of them whatever is chosen here; this only
#: decides which one it starts on, which is the one most saves use.
CONFIG_SAVE_FORMAT: str = "save_format"

#: Extension and file-dialog filter per format, in the order the save dialog
#: lists them.  One table, so the dialog's choices and the save dialog's
#: filters cannot come to disagree about what is supported.
SAVE_FORMAT_FILTERS: dict[str, tuple[str, str]] = {
    "PNG": ("png", "PNG Image (*.png)"),
    "JPEG": ("jpg", "JPEG Image (*.jpg *.jpeg)"),
    "SVG": ("svg", "SVG Vector (*.svg)"),
    "PDF": ("pdf", "PDF Document (*.pdf)"),
}
SAVE_FORMATS: tuple[str, ...] = tuple(SAVE_FORMAT_FILTERS)
DEFAULT_SAVE_FORMAT: str = "PNG"

#: Language codes have no display names in the catalogue, so they are named
#: here.  A code with no entry shows as itself rather than being hidden: a new
#: locale folder should appear in this list the moment it is added, named or
#: not.
#:
#: Deliberately not translated: a language is listed under its own name, so
#: that someone who has landed in a language they cannot read can still find
#: the one they can.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "it": "Italiano",
}


def language_choices() -> list[tuple[str, str]]:
    """Return the (stored value, label) pairs for the language combo.

    Auto comes first and is the honest default: it stores a sentinel rather
    than the resolved code, so a machine that later changes its locale follows
    it instead of staying on whatever it happened to be the day the setting
    was saved.  The label names the language it currently resolves to, because
    "Auto" on its own does not tell the user what they are about to get.
    """
    resolved = platform_language()
    auto_label = tr("Auto") + f" ({LANGUAGE_NAMES.get(resolved, resolved)})"
    return [
        (AUTO_LANGUAGE, auto_label),
        *((code, LANGUAGE_NAMES.get(code, code)) for code in available_languages()),
    ]


def normalized_save_format(value: object) -> str:
    """Return a supported export format, defaulting rather than raising."""
    clean = str(value or "").strip().upper()
    if clean in ("JPG", "JPEG"):
        return "JPEG"
    return clean if clean in SAVE_FORMATS else DEFAULT_SAVE_FORMAT


class SettingsDialog(QDialog):
    """Read and write the machine-level preferences in config.json.

    Nothing is written until Save, and nothing is applied until the next
    start: a dialog that persisted on every keystroke would make Cancel a lie,
    and one that applied a style live crashed the application - see the module
    docstring.
    """

    #: Sentinel combo entry that opens a file picker instead of naming a
    #: style, the same way FigurePropertiesWidget's Style combo browses for
    #: a .mplstyle. Never stored: picking it either adds a real ``file:``
    #: entry or puts the combo back where it was.
    _BROWSE_SENTINEL = "__browse_qss__"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle(tr("Settings"))
        self.setWindowIcon(load_icon("settings"))

        # Only to preselect the combo: nothing here changes the running app.
        self._entry_style = resolve_app_style()
        # The stored setting, not the resolved language: a user who chose Auto
        # must find Auto selected when they reopen this, not the code it
        # happened to resolve to on this machine.
        self._entry_language = get_language()

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="small")

        card = create_card_widget(self, "settingsCard")
        card_layout = QVBoxLayout(card)
        apply_card_layout(card_layout)
        # One QFormLayout for every row, section headings included.  Two of
        # them would each size their own label column, so the combos in the
        # second group would not line up with the first - visible, and the
        # kind of thing that reads as carelessness.
        form = QFormLayout()
        stdSizeAndlayout(form)
        form.addRow(create_section_title(tr("Appearance"), card))

        # Built from what is actually applicable: the themes this app ships,
        # plus whichever Qt style plugins are installed.  Breeze, Oxygen and
        # QtCurve appear here on a machine that has them and are absent on one
        # that does not, rather than being listed and doing nothing.
        self._style_combo = self._combo(
            card,
            [
                *[(key, tr(label)) for key, label in available_app_styles()],
                # Last, and a sentinel rather than a style: the dropdown can
                # only ever list what this installation ships or has
                # installed, and a sheet kept anywhere else has no other way
                # in. Same idea as the Style combo in Figure properties,
                # which browses for a .mplstyle the same way.
                (self._BROWSE_SENTINEL, tr("Browse…")),
            ],
            self._entry_style,
        )
        self._last_valid_style_index = self._style_combo.currentIndex()
        self._style_combo.currentIndexChanged.connect(self._on_style_selected)
        form.addRow(tr("App style"), self._style_combo)

        self._language_combo = self._combo(
            card,
            language_choices(),
            self._entry_language,
        )
        form.addRow(tr("Language"), self._language_combo)

        form.addRow(create_section_title(tr("Charts"), card))

        self._format_combo = self._combo(
            card,
            [(name, name) for name in SAVE_FORMATS],
            normalized_save_format(get_value(CONFIG_SAVE_FORMAT)),
        )
        form.addRow(tr("Default export format"), self._format_combo)

        card_layout.addLayout(form)

        note = QLabel(
            tr("The style and the language are applied the next time the app starts."),
            card,
        )
        note.setProperty("muted", True)
        note.setWordWrap(True)
        card_layout.addWidget(note, 0)

        card_layout.addStretch(1)
        root.addWidget(card, 1)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        action_row.addStretch(1)
        create_action_button(
            parent=self, action_id="save", action=self._save, layout=action_row
        )
        create_action_button(
            parent=self, action_id="close", action=self.reject, layout=action_row
        )
        root.addLayout(action_row, 0)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _combo(
        parent: QWidget, choices: list[tuple[str, str]], current: str
    ) -> QComboBox:
        """Return a combo of (stored value, shown label), on *current*.

        The stored value is carried as item data rather than being recovered
        from the label, so translating a label cannot change what is written
        to config.json.
        """
        combo = QComboBox(parent)
        for value, label in choices:
            combo.addItem(label, value)

        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        configure_combo_width(combo)
        return combo

    @staticmethod
    def _value(combo: QComboBox) -> str:
        return str(combo.currentData() or "")

    # ------------------------------------------------------------------
    # A stylesheet of the user's own
    # ------------------------------------------------------------------
    def _on_style_selected(self, _index: int) -> None:
        if self._value(self._style_combo) != self._BROWSE_SENTINEL:
            self._last_valid_style_index = self._style_combo.currentIndex()
            return
        self._browse_for_stylesheet()

    def _browse_for_stylesheet(self) -> None:
        """Pick a ``.qss`` file and select it as the app style.

        Nothing is applied or written here - this dialog saves on Save and
        the style takes effect on the next start, which the note at the
        bottom of it says. Cancelling puts the combo back where it was
        rather than leaving "Browse…" selected.
        """
        path_str, _unused = QFileDialog.getOpenFileName(
            self,
            tr("Select a stylesheet"),
            "",
            tr("Qt Stylesheet (*.qss);;All Files (*)"),
        )
        if not path_str:
            self._select_style_index(self._last_valid_style_index)
            return

        key = qss_file_style_key(Path(path_str))
        index = self._style_combo.findData(key)
        if index < 0:
            # Before the Browse… row, which stays last.
            index = self._style_combo.count() - 1
            self._style_combo.insertItem(index, qss_file_style_label(key), key)
        self._select_style_index(index)

    def _select_style_index(self, index: int) -> None:
        """Move the combo without re-entering the browse handler."""
        self._style_combo.blockSignals(True)
        try:
            self._style_combo.setCurrentIndex(index)
        finally:
            self._style_combo.blockSignals(False)
        self._last_valid_style_index = self._style_combo.currentIndex()

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def _save(self) -> None:
        """Write every preference, then close.

        Written one key at a time through ``set_value``, which re-reads and
        re-writes the whole document each time.  That is slower than assembling
        one write and entirely deliberate: this dialog is not the only writer
        of config.json - dialog geometry is saved on close - and a single
        blind write would drop whatever else changed while it was open.
        """
        # Never the sentinel: browsing either selects a real entry or puts
        # the combo back, so this is belt and braces rather than a case
        # that is reachable today.
        style_key = self._value(self._style_combo)
        if style_key == self._BROWSE_SENTINEL:
            style_key = self._entry_style
        set_value(CONFIG_APP_STYLE, style_key)
        set_value("language", self._value(self._language_combo))
        set_value(CONFIG_SAVE_FORMAT, self._value(self._format_combo))

        applogger.info(
            "Settings saved: style=%s language=%s format=%s",
            style_key,
            self._value(self._language_combo),
            self._value(self._format_combo),
        )
        self.accept()
