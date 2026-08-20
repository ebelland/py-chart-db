"""Tests for the application preferences dialog.

Two of the four preferences it exposes were already in config.json and read by
nothing at all - ``save_format`` and a top-level ``resize_mode``.  A settings
dialog is exactly where that goes unnoticed: the control looks saved, the
behaviour never changes, and the disagreement is invisible until someone reads
both files.  So the tests here are mostly about *where* a value lands.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.dialogs.settings_dialog import (
    CONFIG_CHART_SECTION,
    CONFIG_RESIZE_MODE,
    CONFIG_SAVE_FORMAT,
    SAVE_FORMAT_FILTERS,
    SettingsDialog,
    normalized_resize_mode,
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


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("FIT", "FIT"),
        ("fixed", "FIXED"),
        ("", "FIT_PROPORTIONAL"),
        ("nonsense", "FIT_PROPORTIONAL"),
    ],
)
def test_an_unusable_resize_mode_falls_back(stored, expected: str) -> None:
    assert normalized_resize_mode(stored) == expected


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
                "chart_panel": {"resize_mode": "FIXED"},
            }
        ),
        encoding="utf-8",
    )

    dialog = SettingsDialog()

    assert dialog._style_combo.currentData() == "fluent_win11"
    assert dialog._format_combo.currentData() == "PDF"
    assert dialog._resize_combo.currentData() == "FIXED"


def test_the_resize_mode_is_read_from_the_section_the_chart_panel_uses(
    temp_config: Path, qapp
) -> None:
    """ChartPanel reads chart_panel.resize_mode; the top-level key is a ghost.

    Reading the wrong one would show the user a value that has not applied to
    anything since it was written.
    """
    temp_config.write_text(
        json.dumps({"resize_mode": "FIT", "chart_panel": {"resize_mode": "FIXED"}}),
        encoding="utf-8",
    )

    assert SettingsDialog()._resize_combo.currentData() == "FIXED"


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def test_saving_writes_every_preference_where_it_is_read(
    temp_config: Path, qapp
) -> None:
    temp_config.write_text(json.dumps({"chart_panel": {"copy_dpi": 200}}), encoding="utf-8")

    dialog = SettingsDialog()
    dialog._style_combo.setCurrentIndex(dialog._style_combo.findData("macos_native"))
    dialog._format_combo.setCurrentIndex(dialog._format_combo.findData("SVG"))
    dialog._resize_combo.setCurrentIndex(dialog._resize_combo.findData("FIT"))
    dialog._save()

    written = _written(temp_config)
    assert written["app_style"] == "macos_native"
    assert written[CONFIG_SAVE_FORMAT] == "SVG"
    assert written[CONFIG_CHART_SECTION][CONFIG_RESIZE_MODE] == "FIT"
    # Merged into the section rather than replacing it.
    assert written[CONFIG_CHART_SECTION]["copy_dpi"] == 200


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
    source = (APP_DIR / "widgets" / "chart_panel_widget.py").read_text(encoding="utf-8")
    body = source[source.index("def save_chart_as") :]
    body = body[: body.index("\n    def ")]

    assert "CONFIG_SAVE_FORMAT" in body
    assert 'chart_{self._figure_id}.png"' not in body
