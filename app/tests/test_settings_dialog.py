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
from app.utils import config

APP_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def temp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point config.json at a throwaway file so a test cannot rewrite the real one."""
    path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
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
    preference: str, expected: str, qapp
) -> None:
    """Picking Windows' sheet on a Mac is the point: it previews the other one."""
    resolved = style.apply_platform_style(qapp, preference)

    assert resolved.qss_file is not None
    assert resolved.qss_file.name == expected


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
