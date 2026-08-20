"""Tests for remembered dialog entries and window geometry.

``import_data_dialog`` persisted its fields by hand, one accessor pair per
key.  These tests pin the generic replacement: a dialog's own widget
attributes are read and written by name, so a new dialog remembers its entries
without writing any persistence code at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.utils import config
from app.utils import dialog_state


@pytest.fixture(autouse=True)
def temp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point config.json at a throwaway file for every test in this module."""
    path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    return path


# ----------------------------------------------------------------------
# config.py
# ----------------------------------------------------------------------
def test_a_missing_file_reads_as_empty(temp_config: Path) -> None:
    assert not temp_config.exists()
    assert config.load_config() == {}
    assert config.get_section("anything") == {}
    assert config.get_value("anything", "fallback") == "fallback"


def test_a_corrupt_file_reads_as_empty(temp_config: Path) -> None:
    """A hand-edited file with a stray comma must not stop the application."""
    temp_config.write_text("{ this is not json", encoding="utf-8")
    assert config.load_config() == {}


def test_a_non_object_section_reads_as_empty(temp_config: Path) -> None:
    temp_config.write_text(json.dumps({"actions": "oops"}), encoding="utf-8")
    assert config.get_section("actions") == {}


def test_writing_one_section_keeps_the_others(temp_config: Path) -> None:
    """The read-modify-write dance is exactly what used to drop sections."""
    config.set_value("last_database", "/tmp/a.dhub")
    config.set_section("dialog_state", {"x": {"y": 1}})
    config.update_section("chart_panel", copy_dpi=300)

    saved = json.loads(temp_config.read_text(encoding="utf-8"))
    assert saved["last_database"] == "/tmp/a.dhub"
    assert saved["dialog_state"] == {"x": {"y": 1}}
    assert saved["chart_panel"] == {"copy_dpi": 300}


def test_the_last_database_is_forgotten_once_it_is_gone(tmp_path: Path) -> None:
    """Reopening a database that was deleted would fail at startup."""
    missing = tmp_path / "gone.dhub"
    config.set_last_database(missing)
    assert config.get_last_database() is None

    missing.write_text("", encoding="utf-8")
    assert config.get_last_database() == missing


# ----------------------------------------------------------------------
# dialog_state.py
# ----------------------------------------------------------------------
qt = pytest.importorskip("PySide6.QtWidgets")





class _Dialog(qt.QWidget):
    """A stand-in with one widget of each remembered kind."""

    def __init__(self) -> None:
        super().__init__()
        self._name = qt.QLineEdit(self)
        self._table = qt.QComboBox(self)
        self._table.addItems(["alpha", "beta"])
        self._rows = qt.QSpinBox(self)
        self._rows.setMaximum(1000)
        self._ratio = qt.QDoubleSpinBox(self)
        self._header = qt.QCheckBox(self)
        self._sql = qt.QPlainTextEdit(self)
        # Not persisted: no accessor is registered for a table view.
        self._view = qt.QTableView(self)


def test_every_supported_widget_is_remembered(qapp) -> None:
    dialog = _Dialog()
    dialog._name.setText("run 3")
    dialog._table.setCurrentIndex(1)
    dialog._rows.setValue(12)
    dialog._ratio.setValue(0.25)
    dialog._header.setChecked(True)
    dialog._sql.setPlainText("SELECT 1")

    stored = dialog_state.save_dialog_state(dialog, "demo")

    assert stored == {
        "name": "run 3",
        "table": "beta",
        "rows": 12,
        "ratio": 0.25,
        "header": True,
        "sql": "SELECT 1",
    }
    assert "view" not in stored


def test_the_entries_come_back_into_a_fresh_dialog(qapp) -> None:
    first = _Dialog()
    first._name.setText("run 3")
    first._table.setCurrentIndex(1)
    first._rows.setValue(12)
    first._header.setChecked(True)
    dialog_state.save_dialog_state(first, "demo")

    second = _Dialog()
    dialog_state.restore_dialog_state(second, "demo")

    assert second._name.text() == "run 3"
    assert second._table.currentText() == "beta"
    assert second._rows.value() == 12
    assert second._header.isChecked() is True


def test_a_combo_entry_that_no_longer_exists_is_skipped(qapp) -> None:
    """A remembered table can be dropped between two runs."""
    dialog_state.save_dialog_state(_Dialog(), "demo")
    section = config.get_section(dialog_state.CONFIG_SECTION)
    section["demo"]["table"] = "a table that was deleted"
    config.set_section(dialog_state.CONFIG_SECTION, section)

    dialog = _Dialog()
    dialog_state.restore_dialog_state(dialog, "demo")

    assert dialog._table.currentText() == "alpha"


def test_a_field_that_no_longer_exists_is_ignored(qapp) -> None:
    """Renaming an attribute forgets its value; it must not raise."""
    config.set_section(
        dialog_state.CONFIG_SECTION,
        {"demo": {"name": "kept", "was_renamed": 7}},
    )

    dialog = _Dialog()
    dialog_state.restore_dialog_state(dialog, "demo")
    assert dialog._name.text() == "kept"


def test_a_value_of_the_wrong_type_is_logged_not_raised(qapp) -> None:
    config.set_section(dialog_state.CONFIG_SECTION, {"demo": {"rows": "not a number"}})

    dialog = _Dialog()
    dialog_state.restore_dialog_state(dialog, "demo")
    assert dialog._rows.value() == 0


def test_restoring_does_not_fire_change_signals(qapp) -> None:
    """A restored value must not look like the user just edited the field."""
    first = _Dialog()
    first._name.setText("run 3")
    dialog_state.save_dialog_state(first, "demo")

    second = _Dialog()
    seen: list[str] = []
    second._name.textChanged.connect(seen.append)
    dialog_state.restore_dialog_state(second, "demo")

    assert seen == []


def test_nothing_stored_means_nothing_changes(qapp) -> None:
    dialog = _Dialog()
    dialog_state.restore_dialog_state(dialog, "never_saved")
    assert dialog._name.text() == ""


def test_two_dialogs_do_not_share_a_slot(qapp) -> None:
    first, second = _Dialog(), _Dialog()
    first._name.setText("one")
    second._name.setText("two")
    dialog_state.save_dialog_state(first, "first")
    dialog_state.save_dialog_state(second, "second")

    fresh = _Dialog()
    dialog_state.restore_dialog_state(fresh, "first")
    assert fresh._name.text() == "one"


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def test_geometry_is_stored_as_readable_numbers(qapp) -> None:
    """Qt's saveGeometry() blob would be unreadable in config.json."""
    window = qt.QWidget()
    window.resize(640, 480)

    dialog_state.save_window_geometry(window, "demo")
    stored = config.get_section(dialog_state.GEOMETRY_SECTION)["demo"]

    assert stored["width"] == 640 and stored["height"] == 480
    assert stored["maximized"] is False
    assert set(stored) == {"x", "y", "width", "height", "maximized"}


def test_the_size_comes_back(qapp) -> None:
    first = qt.QWidget()
    first.resize(640, 480)
    dialog_state.save_window_geometry(first, "demo")

    second = qt.QWidget()
    assert dialog_state.restore_window_geometry(second, "demo") is True
    assert (second.width(), second.height()) == (640, 480)


def test_no_saved_geometry_is_reported(qapp) -> None:
    assert dialog_state.restore_window_geometry(qt.QWidget(), "never_saved") is False


def test_a_position_off_every_screen_is_ignored(qapp) -> None:
    """A window restored onto an unplugged monitor would be unreachable."""
    window = qt.QWidget()
    config.set_section(
        dialog_state.GEOMETRY_SECTION,
        {"demo": {"x": -99000, "y": -99000, "width": 300, "height": 200}},
    )

    dialog_state.restore_window_geometry(window, "demo")

    assert (window.width(), window.height()) == (300, 200)
    assert window.pos().x() != -99000


def test_clearing_forgets_both_entries_and_geometry(qapp) -> None:
    window = _Dialog()
    dialog_state.save_dialog_state(window, "demo")
    dialog_state.save_window_geometry(window, "demo")

    dialog_state.clear_state("demo")

    assert "demo" not in config.get_section(dialog_state.CONFIG_SECTION)
    assert "demo" not in config.get_section(dialog_state.GEOMETRY_SECTION)
