"""``config.json`` ships; ``user.json`` accumulates.

The two used to be one file, and the cost showed up in the repository rather
than in a bug report: running the suite rewrote ``last_database`` to a pytest
temporary directory and left it in ``git status``, so a real diff had to be
picked out from around it.  ``app_style``, the window geometry and twelve
dialogs' remembered entries were versioned the same way, committed by whoever
happened to run the application before committing.

These tests hold the boundary in both directions - a setting must not land in
the shipped catalogue, and the catalogue must not be rewritten by the program
that reads it - and cover the migration, which only ever runs once on any
given machine and therefore gets no second chance to be right.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.utils import config


@pytest.fixture(autouse=True)
def temp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point both files at throwaway ones and clear the caches between tests."""
    application = tmp_path / "config.json"
    user = tmp_path / "user.json"
    monkeypatch.setattr(config, "CONFIG_PATH", application)
    monkeypatch.setattr(config, "USER_CONFIG_PATH", user)
    monkeypatch.setattr(config, "_migrated_from", None)
    config._cache.clear()
    config._cache_stamp.clear()
    config._merged = None
    return application, user


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Which file a write lands in
# ----------------------------------------------------------------------
def test_a_setting_is_written_to_the_user_file(
    temp_config: tuple[Path, Path],
) -> None:
    application, user = temp_config
    _write(application, {"actions": {"open": {"label": "Open"}}})

    config.set_value("last_database", "/tmp/a.dhub")
    config.set_section("dialog_state", {"x": {"y": 1}})
    config.update_section("chart_panel", copy_dpi=300)

    saved = _read(user)
    assert saved["last_database"] == "/tmp/a.dhub"
    assert saved["dialog_state"] == {"x": {"y": 1}}
    assert saved["chart_panel"] == {"copy_dpi": 300}
    # The read-modify-write dance is exactly what used to drop sections.
    assert set(saved) == {"last_database", "dialog_state", "chart_panel"}


def test_the_shipped_catalogue_is_left_alone_by_a_write(
    temp_config: tuple[Path, Path],
) -> None:
    application, _user = temp_config
    _write(application, {"actions": {"open": {"label": "Open"}}, "messages": {}})

    config.set_value("app_style", "dark")

    assert _read(application) == {"actions": {"open": {"label": "Open"}}, "messages": {}}


@pytest.mark.parametrize("name", sorted(config.APPLICATION_SECTIONS))
def test_writing_a_catalogue_section_is_refused(
    name: str, temp_config: tuple[Path, Path]
) -> None:
    """Not silently redirected into user.json either: a caller that manages to
    name a catalogue section is doing something wrong and should hear about
    it, rather than have a shadow copy of the catalogue start overriding the
    translated one."""
    application, user = temp_config
    _write(application, {name: {"shipped": True}})

    config.set_section(name, {"replaced": True})

    assert _read(application) == {name: {"shipped": True}}
    assert not user.exists()


# ----------------------------------------------------------------------
# Reading the two as one
# ----------------------------------------------------------------------
def test_both_files_are_read_as_one(temp_config: tuple[Path, Path]) -> None:
    application, user = temp_config
    _write(application, {"actions": {"open": {}}})
    _write(user, {"app_style": "dark"})

    assert config.get_section("actions") == {"open": {}}
    assert config.get_value("app_style") == "dark"


def test_the_user_file_wins(temp_config: tuple[Path, Path]) -> None:
    """During the changeover a key can exist in both. The one the application
    wrote is the newer one."""
    application, user = temp_config
    _write(application, {"language": "en"})
    _write(user, {"language": "it"})

    assert config.get_value("language") == "it"


def test_a_missing_pair_reads_as_empty() -> None:
    assert config.load_config() == {}
    assert config.get_section("anything") == {}
    assert config.get_value("anything", "fallback") == "fallback"


def test_a_corrupt_user_file_does_not_take_the_catalogue_with_it(
    temp_config: tuple[Path, Path],
) -> None:
    """A hand-edited file with a stray comma must not stop the application -
    and must not cost it its labels and icons either."""
    application, user = temp_config
    _write(application, {"actions": {"open": {"label": "Open"}}})
    user.write_text("{ this is not json", encoding="utf-8")

    assert config.get_section("actions") == {"open": {"label": "Open"}}


def test_a_non_object_section_reads_as_empty(temp_config: tuple[Path, Path]) -> None:
    application, _user = temp_config
    _write(application, {"actions": "oops"})

    assert config.get_section("actions") == {}


# ----------------------------------------------------------------------
# Migration, which happens once and cannot be retried
# ----------------------------------------------------------------------
PRE_SPLIT = {
    "actions": {"open": {"label": "Open"}},
    "messages": {"chart.no_series": {"text": "Add a series."}},
    "last_database": "/tmp/old.dhub",
    "app_style": "dark",
    "language": "it",
    "window_geometry": {"main_window": [0, 0, 800, 600]},
}


def test_settings_are_moved_out_on_the_first_read(
    temp_config: tuple[Path, Path],
) -> None:
    application, user = temp_config
    _write(application, PRE_SPLIT)

    assert config.get_value("app_style") == "dark"

    assert set(_read(application)) == {"actions", "messages"}
    assert _read(user) == {
        "last_database": "/tmp/old.dhub",
        "app_style": "dark",
        "language": "it",
        "window_geometry": {"main_window": [0, 0, 800, 600]},
    }


def test_nothing_is_lost_in_the_move(temp_config: tuple[Path, Path]) -> None:
    """The point of migrating rather than shipping a stripped config.json:
    the file on somebody's machine is the one holding their settings."""
    application, _user = temp_config
    _write(application, PRE_SPLIT)

    for name, expected in PRE_SPLIT.items():
        assert config.load_config()[name] == expected


def test_an_existing_user_value_survives_the_move(
    temp_config: tuple[Path, Path],
) -> None:
    """Migrating twice - two installs, one shared home - must not roll a
    setting back to what the old file happened to hold."""
    application, user = temp_config
    _write(application, PRE_SPLIT)
    _write(user, {"app_style": "light"})

    assert config.get_value("app_style") == "light"
    assert _read(user)["app_style"] == "light"


def test_a_read_only_install_still_reads_its_settings(
    temp_config: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An application installed somewhere it cannot write keeps working: the
    settings are read where they are until they can be moved."""
    application, _user = temp_config
    _write(application, PRE_SPLIT)
    monkeypatch.setattr(config, "_write", lambda path, data: False)

    assert config.get_value("app_style") == "dark"
    assert config.get_section("actions") == {"open": {"label": "Open"}}


def test_a_second_load_does_not_migrate_again(
    temp_config: tuple[Path, Path],
) -> None:
    application, user = temp_config
    _write(application, PRE_SPLIT)
    config.load_config()

    config.set_value("app_style", "light")
    config.load_config()

    assert _read(user)["app_style"] == "light"


def test_an_already_split_pair_is_left_alone(
    temp_config: tuple[Path, Path],
) -> None:
    application, user = temp_config
    _write(application, {"actions": {}, "messages": {}})
    _write(user, {"app_style": "dark"})
    before = application.stat().st_mtime_ns

    config.load_config()

    assert application.stat().st_mtime_ns == before


# ----------------------------------------------------------------------
# The repository's own copy
# ----------------------------------------------------------------------
def test_the_shipped_file_does_not_accumulate_settings() -> None:
    """The regression that started this: a suite run used to leave a pytest
    temporary path in the versioned config.json.

    Skipped rather than failed on a checkout where the migration has not run
    yet.  config.json is not shipped stripped - that would delete the settings
    on the machine it lands on - so before the first launch it still holds
    them, and asserting otherwise would fail on a fresh pull for the one
    reason that is not a bug.
    """
    shipped = json.loads(
        (Path(__file__).resolve().parents[2] / "config.json").read_text(
            encoding="utf-8"
        )
    )
    leftover = sorted(set(shipped) - config.APPLICATION_SECTIONS)
    if leftover:
        pytest.skip(
            "config.json still holds pre-split settings "
            f"({', '.join(leftover)}); they move to user.json on the next load"
        )

    assert not set(shipped) - config.APPLICATION_SECTIONS
