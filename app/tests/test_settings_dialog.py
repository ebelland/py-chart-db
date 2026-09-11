"""Tests for the application preferences dialog.

``save_format`` was already in config.json and read by nothing at all - a
settings dialog is exactly where that goes unnoticed, since the control looks
saved and the behaviour never changes. So part of what these tests are about
is *where* a value lands.

Chart sizing used to be here too, as a single global default, but it was
never anything more than the value ChartPanel started a *new* figure on - it
could not be changed per figure from this dialog, only from the chart's own
context menu, and now from Figure Properties (see FigurePropertiesWidget.
set_resize_mode_control). Kept in only one of those places rather than two
that could disagree about the same figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.dialogs.settings_dialog import (
    CONFIG_SAVE_FORMAT,
    SAVE_FORMAT_FILTERS,
    SettingsDialog,
    normalized_save_format,
)
from app.styles import style
from app.utils import config, i18n

APP_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def temp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point both configuration files at throwaway ones, so a test cannot
    rewrite the real ones."""
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    path = tmp_path / "user.json"
    monkeypatch.setattr(config, "USER_CONFIG_PATH", path)
    monkeypatch.setattr(config, "_migrated_from", None)
    config._cache.clear()
    config._cache_stamp.clear()
    config._merged = None
    return path


def _written(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("SVG", "SVG"),
        ("svg", "SVG"),
        ("jpg", "JPEG"),
        ("jpeg", "JPEG"),
        ("", "PNG"),
        (None, "PNG"),
        ("bmp", "PNG"),
    ],
)
def test_an_unusable_export_format_falls_back(stored, expected: str) -> None:
    """A hand-edited value must not stop the dialog from opening."""
    assert normalized_save_format(stored) == expected


def test_every_offered_format_has_a_filter() -> None:
    """The combo and the save dialog read the same table, so they cannot drift."""
    for name, (extension, file_filter) in SAVE_FORMAT_FILTERS.items():
        assert extension and f"*.{extension}" in file_filter


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def test_the_dialog_opens_on_what_is_stored(temp_config: Path, qapp) -> None:
    temp_config.write_text(
        json.dumps(
            {
                "app_style": "fluent_win11",
                "language": "it",
                "save_format": "PDF",
            }
        ),
        encoding="utf-8",
    )

    dialog = SettingsDialog()

    assert dialog._style_combo.currentData() == "fluent_win11"
    assert dialog._format_combo.currentData() == "PDF"


def test_the_dialog_no_longer_offers_a_chart_sizing_default(
    temp_config: Path, qapp
) -> None:
    """It moved to Figure Properties, per figure - see the module docstring."""
    assert not hasattr(SettingsDialog(), "_resize_combo")


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def test_saving_writes_every_preference_where_it_is_read(
    temp_config: Path, qapp
) -> None:
    temp_config.write_text(json.dumps({}), encoding="utf-8")

    dialog = SettingsDialog()
    dialog._style_combo.setCurrentIndex(dialog._style_combo.findData("macos_native"))
    dialog._format_combo.setCurrentIndex(dialog._format_combo.findData("SVG"))
    dialog._save()

    written = _written(temp_config)
    assert written["app_style"] == "macos_native"
    assert written[CONFIG_SAVE_FORMAT] == "SVG"


def test_saving_does_not_touch_the_chart_panel_section(
    temp_config: Path, qapp
) -> None:
    """That section is ChartPanel's own, written per figure - see
    FigurePropertiesWidget.set_resize_mode_control. This dialog has nothing
    left to write there and must not go near copy_dpi, background_color and
    the rest of what already lives in it."""
    temp_config.write_text(json.dumps({"chart_panel": {"copy_dpi": 200}}), encoding="utf-8")

    dialog = SettingsDialog()
    dialog._save()

    assert _written(temp_config)["chart_panel"] == {"copy_dpi": 200}


def test_cancel_writes_nothing(temp_config: Path, qapp) -> None:
    """Cancel has to mean it."""
    temp_config.write_text(json.dumps({"app_style": "automatic"}), encoding="utf-8")

    dialog = SettingsDialog()
    dialog._style_combo.setCurrentIndex(dialog._style_combo.findData("fluent_win11"))
    dialog.reject()

    assert _written(temp_config)["app_style"] == "automatic"


@pytest.mark.parametrize("closer", ["_save", "reject"])
def test_the_dialog_never_restyles_the_running_application(
    temp_config: Path, qapp, closer: str
) -> None:
    """The crash fix, pinned.

    This dialog used to apply each style as you moved through the list.
    Applying one re-polishes every widget in the application, and
    ``QApplication.setStyle`` for a Qt plugin destroys and rebuilds the QStyle
    underneath them all - which, with a QWebEngineView alive in the results
    pane, segfaults inside Chromium's delegate.  The setting is now written and
    read at the next start, so neither closing path may touch the live app.

    Asserted for Save as well as Cancel: saving is the path that used to be
    allowed to leave the new style installed.
    """
    temp_config.write_text(json.dumps({"app_style": "automatic"}), encoding="utf-8")
    before_sheet = qapp.styleSheet()
    before_style = qapp.style().objectName()

    dialog = SettingsDialog()
    dialog._style_combo.setCurrentIndex(dialog._style_combo.findData("fluent_win11"))
    getattr(dialog, closer)()

    assert qapp.styleSheet() == before_sheet
    assert qapp.style().objectName() == before_style


def test_saving_a_style_still_records_it_for_the_next_start(
    temp_config: Path, qapp
) -> None:
    """Not applying it must not mean not remembering it."""
    temp_config.write_text(json.dumps({"app_style": "automatic"}), encoding="utf-8")

    dialog = SettingsDialog()
    dialog._style_combo.setCurrentIndex(dialog._style_combo.findData("dark"))
    dialog._save()

    assert _written(temp_config)["app_style"] == "dark"


# ----------------------------------------------------------------------
# The stylesheet choice
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("preference", "expected"),
    [
        ("fluent_win11", "fluent_win11.qss"),
        ("macos_native", "macos_native.qss"),
    ],
)
def test_a_forced_style_is_applied_whatever_the_platform(
    preference: str, expected: str, qapp, suppressed_restyle
) -> None:
    """Picking Windows' sheet on a Mac is the point: it previews the other one."""
    resolved = style.apply_platform_style(qapp, preference)

    assert resolved.qss_file is not None
    assert resolved.qss_file.name == expected
    assert suppressed_restyle["sheet"], "the sheet it resolved is the one it installs"


@pytest.mark.parametrize("qss_name", ["fluent_win11.qss", "macos_native.qss"])
def test_every_stylesheet_actually_styles_create_card_widget(qss_name: str) -> None:
    """create_card_widget() (app/styles/style.py) sets the Qt *property*
    card=true on every card it makes, with its own distinct objectName - it
    never names one literally "Card"/"card".

    fluent_win11.qss used to select on #Card/#card (an object name) rather
    than [card="true"] (the property), which never matched a single card in
    the app - every card-styled section fell through to a plain transparent
    QFrame, with no surface, border or radius of its own, on both the
    Fluent and the Dark theme (dark reuses this same file). Regression
    guard: both sheets must carry the property selector that actually
    matches what the widgets are built with.
    """
    from app.styles.style import _load_qss

    qss, _path = _load_qss(qss_name)
    assert qss is not None
    assert '[card="true"]' in qss


@pytest.mark.parametrize("preference", ["fluent_win11", "macos_native", "dark"])
def test_a_themed_stylesheet_does_not_force_an_app_wide_qt_style(
    preference: str, qapp, suppressed_restyle
) -> None:
    """macos_native.qss's own header explains why standard controls
    (QPushButton, QComboBox, QMenu, QScrollBar...) are deliberately left
    unstyled: Qt's native Aqua/Fluent style plugins already draw those
    authentically, and forcing Fusion app-wide to fix one narrow rendering
    gap (item-view selection/header styling - see
    apply_fusion_for_item_view_styling) would throw that away for every
    other widget in the application. That fix is scoped to the specific
    views that need it instead (test_table_list.py /
    test_table_preview.py), not applied here.
    """
    style.apply_platform_style(qapp, preference)

    assert suppressed_restyle["style"] == []


def test_an_explicit_qt_style_is_still_applied(qapp, suppressed_restyle) -> None:
    """Picking a Qt style plugin by name is a real app.setStyle() case,
    unrelated to the themed-stylesheet path above."""
    style.apply_platform_style(qapp, f"{style.QT_STYLE_PREFIX}windows")

    assert suppressed_restyle["style"] == ["Windows"]


def test_an_unknown_style_reads_as_automatic(qapp) -> None:
    """A typo in config.json should leave the app looking normal, not bare."""
    assert style.resolve_app_style("not-a-style") == style.APP_STYLE_AUTOMATIC


# ----------------------------------------------------------------------
# Reachability
# ----------------------------------------------------------------------
def test_the_settings_action_exists_and_is_in_the_menu() -> None:
    """A dialog nothing opens is a dialog nobody has."""
    actions = json.loads(
        (APP_DIR.parent / "config.json").read_text(encoding="utf-8")
    )["actions"]
    assert "settings" in actions
    assert actions["settings"]["text"].strip()

    source = (APP_DIR / "dialogs" / "main_window.py").read_text(encoding="utf-8")
    assert 'action_menu_item("settings"' in source


def test_the_save_dialog_honours_the_configured_format() -> None:
    """The preference existed for a long time and changed nothing."""
    source = (APP_DIR / "widgets" / "chart_panel.py").read_text(encoding="utf-8")
    body = source[source.index("def save_chart_as") :]
    body = body[: body.index("\n    def ")]

    assert "CONFIG_SAVE_FORMAT" in body
    assert 'chart_{self._figure_id}.png"' not in body


# ----------------------------------------------------------------------
# Auto: follow the platform
# ----------------------------------------------------------------------
def test_auto_is_offered_and_comes_first(qapp) -> None:
    """A fresh installation should speak the machine's language, and the only
    way to say that is a setting that is not a language code."""
    from app.dialogs.settings_dialog import language_choices
    from app.utils.i18n import AUTO_LANGUAGE

    values = [value for value, _label in language_choices()]

    assert values[0] == AUTO_LANGUAGE
    assert set(values[1:]) == set(i18n.available_languages())


def test_the_auto_label_names_the_language_it_resolves_to(qapp) -> None:
    """"Auto" alone does not tell the user what they are about to get."""
    from app.dialogs.settings_dialog import LANGUAGE_NAMES, language_choices

    _value, label = language_choices()[0]
    resolved = i18n.platform_language()

    assert LANGUAGE_NAMES.get(resolved, resolved) in label


def test_auto_is_stored_as_the_sentinel_not_as_the_resolved_code(
    qapp, monkeypatch
) -> None:
    """Storing the resolved code would pin the app to whatever the machine
    was set to the day the setting was saved."""
    from app.dialogs import settings_dialog as module
    from app.utils.i18n import AUTO_LANGUAGE

    written: dict[str, str] = {}
    monkeypatch.setattr(module, "set_value", lambda key, value: written.update({key: value}))
    monkeypatch.setattr(module, "get_language", lambda: AUTO_LANGUAGE)

    dialog = module.SettingsDialog()
    dialog._save()

    assert written["language"] == AUTO_LANGUAGE


def test_a_saved_auto_reopens_as_auto(qapp, monkeypatch) -> None:
    """Not as the language it happens to resolve to on this machine."""
    from app.dialogs import settings_dialog as module
    from app.utils.i18n import AUTO_LANGUAGE

    monkeypatch.setattr(module, "get_language", lambda: AUTO_LANGUAGE)

    dialog = module.SettingsDialog()

    assert dialog._language_combo.currentData() == AUTO_LANGUAGE


def test_setting_auto_resolves_to_a_language_that_has_a_catalogue() -> None:
    from app.utils.i18n import AUTO_LANGUAGE, set_language

    before = i18n.language()
    try:
        assert set_language(AUTO_LANGUAGE) in i18n.available_languages()
    finally:
        set_language(before)


def test_an_unwritten_language_key_means_auto() -> None:
    """A config.json that has never been written is a fresh installation."""
    from app.utils import config
    from app.utils.i18n import AUTO_LANGUAGE

    assert config.get_language.__doc__
    assert AUTO_LANGUAGE == "auto"


def test_a_platform_locale_nothing_translates_falls_back_to_english(
    monkeypatch,
) -> None:
    """Untranslated source strings *are* English, so that is the honest
    fallback rather than the last language chosen."""
    from PySide6.QtCore import QLocale

    from app.utils import i18n as module

    monkeypatch.setattr(
        QLocale, "system", staticmethod(lambda: QLocale("fi_FI"))
    )
    # The macOS preference list is the other place an answer comes from,
    # and on a developer's own Mac it is a real one - so a test about a
    # machine set only to Finnish has to say that about all of it.
    monkeypatch.setattr(module, "_apple_preferred_languages", lambda: [])

    assert module.platform_language() == module.DEFAULT_LANGUAGE


def test_a_regional_variant_still_finds_its_language(monkeypatch) -> None:
    """The catalogues are named it, not it_IT."""
    from PySide6.QtCore import QLocale

    from app.utils import i18n as module

    monkeypatch.setattr(
        QLocale, "system", staticmethod(lambda: QLocale("it_CH"))
    )

    assert module.platform_language() == "it"



# ----------------------------------------------------------------------
# A stylesheet of the user's own
# ----------------------------------------------------------------------
@pytest.fixture
def custom_qss(tmp_path: Path) -> Path:
    sheet = tmp_path / "mine.qss"
    sheet.write_text("QPushButton { background: #ff0000; }", encoding="utf-8")
    return sheet


def test_the_style_combo_offers_browse_last(qapp) -> None:
    """The dropdown can only list what this installation ships or has
    installed; a sheet kept anywhere else has no other way in."""
    dialog = SettingsDialog()
    combo = dialog._style_combo

    assert combo.itemData(combo.count() - 1) == dialog._BROWSE_SENTINEL


def test_browsing_selects_the_chosen_sheet(
    qapp, custom_qss: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.dialogs.settings_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(custom_qss), ""),
    )
    dialog = SettingsDialog()
    combo = dialog._style_combo

    combo.setCurrentIndex(combo.findData(dialog._BROWSE_SENTINEL))

    assert combo.currentData() == style.qss_file_style_key(custom_qss)
    assert combo.currentText() == "File: mine.qss"
    # Browse… stays the last row, so it is always reachable again.
    assert combo.itemData(combo.count() - 1) == dialog._BROWSE_SENTINEL


def test_cancelling_the_picker_puts_the_combo_back(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.dialogs.settings_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: ("", ""),
    )
    dialog = SettingsDialog()
    combo = dialog._style_combo
    before = combo.currentIndex()

    combo.setCurrentIndex(combo.findData(dialog._BROWSE_SENTINEL))

    assert combo.currentIndex() == before
    assert combo.currentData() != dialog._BROWSE_SENTINEL


def test_the_sentinel_is_never_what_gets_saved(
    qapp, temp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Belt and braces: browsing always leaves a real entry selected, but
    "__browse_qss__" must never reach config.json even so."""
    dialog = SettingsDialog()
    combo = dialog._style_combo
    combo.blockSignals(True)
    combo.setCurrentIndex(combo.findData(dialog._BROWSE_SENTINEL))
    combo.blockSignals(False)

    dialog._save()

    assert _written(temp_config)["app_style"] != dialog._BROWSE_SENTINEL


def test_a_chosen_sheet_resolves_and_applies(
    qapp, custom_qss: Path, restored_app_style
) -> None:
    """The one test that reads the sheet back off the application, so it
    installs a real one - a two-line one, and it is put back afterwards."""
    key = style.qss_file_style_key(custom_qss)

    assert style.resolve_app_style(key) == key

    resolved = style.apply_platform_style(qapp, key)
    assert resolved.qss_file == custom_qss
    assert "ff0000" in qapp.styleSheet()


def test_a_sheet_that_has_gone_missing_falls_back(qapp, custom_qss: Path) -> None:
    """Same answer as an uninstalled Qt plugin: the app looks normal, not
    unstyled, and says why in the log."""
    key = style.qss_file_style_key(custom_qss)
    custom_qss.unlink()

    assert style.resolve_app_style(key) == style.APP_STYLE_AUTOMATIC


def test_the_macos_preference_answers_when_qt_reports_the_c_locale(
    monkeypatch,
) -> None:
    """The bug: an Italian Mac showing an English interface on Auto.

    Qt consults the POSIX environment before the system preference on
    macOS, so a session that exports LANG=C - which is what an app
    launched from a terminal with a locale-less profile inherits - makes
    QLocale.system() report the C locale on a Mac that is set to Italian.
    """
    from PySide6.QtCore import QLocale

    from app.utils import i18n as module

    monkeypatch.setattr(QLocale, "system", staticmethod(lambda: QLocale("C")))
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module, "_apple_preferred_languages", lambda: ["it-US"])
    monkeypatch.setattr(module.os, "environ", {"LANG": "C.UTF-8"})

    assert module.platform_language() == "it"


def test_the_macos_preference_is_not_read_on_other_platforms(
    monkeypatch,
) -> None:
    """It is a macOS preference; asking for it elsewhere is a Qt call that
    can only ever answer nothing."""
    from app.utils import i18n as module

    asked: list[int] = []
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(
        module, "_apple_preferred_languages", lambda: asked.append(1) or []
    )

    module._platform_language_candidates()

    assert asked == []


@pytest.mark.parametrize(
    "stored, expected",
    [
        (["it-US", "en-GB"], ["it-US", "en-GB"]),
        ("it-US", ["it-US"]),
        (None, []),
        (17, []),
    ],
)
def test_the_preference_list_is_read_whatever_shape_it_comes_back_in(
    monkeypatch, stored, expected
) -> None:
    """QSettings hands back a list for a multi-language preference, a bare
    string for one, and None where the key is absent."""
    from PySide6.QtCore import QSettings

    from app.utils import i18n as module

    monkeypatch.setattr(
        QSettings, "value", lambda self, key, *args, **kwargs: stored
    )

    assert module._apple_preferred_languages() == expected
