"""The Open recent menu, and the list in user.json behind it.

The list is written by ``set_last_database``, which every path that makes
a database current already calls - opening one, creating one, saving one
under a new name, loading a demo. Two functions to call would eventually
be one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.utils import config


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Give the recent list a settings file of its own, not the user's."""
    values: dict = {}
    monkeypatch.setattr(
        config, "get_value", lambda name, default=None: values.get(name, default)
    )
    monkeypatch.setattr(
        config, "set_value", lambda name, value: values.__setitem__(name, value)
    )
    return values


@pytest.fixture
def databases(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / f"project_{index}.dhub" for index in range(3)]
    for path in paths:
        path.write_text("x", encoding="utf-8")
    return paths


# ----------------------------------------------------------------------
# The list
# ----------------------------------------------------------------------
def test_the_most_recent_database_is_first(store, databases) -> None:
    for path in databases:
        config.remember_recent_database(path)

    assert config.get_recent_databases() == list(reversed(databases))


def test_reopening_one_moves_it_up_rather_than_repeating_it(
    store, databases
) -> None:
    for path in databases:
        config.remember_recent_database(path)
    config.remember_recent_database(databases[0])

    recent = config.get_recent_databases()
    assert recent[0] == databases[0]
    assert len(recent) == len(databases)


def test_a_file_that_is_gone_is_not_offered(store, databases) -> None:
    """A menu entry that can only fail is worse than a shorter menu."""
    for path in databases:
        config.remember_recent_database(path)
    databases[1].unlink()

    assert databases[1] not in config.get_recent_databases()
    assert len(config.get_recent_databases()) == 2


def test_the_list_is_capped(store, tmp_path: Path) -> None:
    paths = []
    for index in range(config.MAX_RECENT_DATABASES + 5):
        path = tmp_path / f"many_{index}.dhub"
        path.write_text("x", encoding="utf-8")
        paths.append(path)
        config.remember_recent_database(path)

    recent = config.get_recent_databases()
    assert len(recent) == config.MAX_RECENT_DATABASES
    assert recent[0] == paths[-1]


def test_opening_a_database_records_it(store, databases) -> None:
    """set_last_database is the one call every open path already makes."""
    config.set_last_database(databases[0])

    assert config.get_recent_databases() == [databases[0]]


def test_clearing_forgets_everything(store, databases) -> None:
    for path in databases:
        config.remember_recent_database(path)

    config.clear_recent_databases()

    assert config.get_recent_databases() == []


def test_a_hand_edited_list_of_the_wrong_shape_reads_as_empty(store) -> None:
    store["recent_databases"] = "not a list"

    assert config.get_recent_databases() == []


# ----------------------------------------------------------------------
# The menu
# ----------------------------------------------------------------------
@pytest.fixture
def window(qapp, repo, tmp_db_path, store):
    from app.dialogs.main_window import MainWindow
    from app.logs.logger import applogger

    built = MainWindow(repo=repo, db_path=tmp_db_path)
    yield built
    built.close()
    applogger.set_status_bar(None)


def _recent_submenu(window):
    for action in window._app_menu.actions():
        if action.menu() is not None:
            return action
    return None


def test_the_menu_has_a_recent_submenu(window) -> None:
    action = _recent_submenu(window)

    assert action is not None
    assert action.text() == "Open recent"


def test_an_empty_list_leaves_the_entry_disabled_rather_than_missing(
    window,
) -> None:
    """The menu keeps its shape between the first launch and the second."""
    action = _recent_submenu(window)

    assert not action.isEnabled()
    assert action.menu().actions() == []


def test_each_remembered_database_is_one_entry(window, databases) -> None:
    for path in databases:
        config.remember_recent_database(path)
    window._build_app_menu()

    action = _recent_submenu(window)
    entries = [item.text() for item in action.menu().actions() if not item.isSeparator()]

    assert entries == [path.name for path in reversed(databases)] + ["Clear menu"]
    assert action.isEnabled()


def test_an_entry_carries_the_folder_in_its_tooltip(window, databases) -> None:
    """Two projects called analysis.dhub in different folders is the normal
    case, and a menu of identical names is a menu of guesses."""
    config.remember_recent_database(databases[0])
    window._build_app_menu()

    entry = _recent_submenu(window).menu().actions()[0]

    assert entry.toolTip() == str(databases[0].parent)


def test_clearing_from_the_menu_empties_it(window, databases) -> None:
    for path in databases:
        config.remember_recent_database(path)
    window._build_app_menu()

    window._on_clear_recent()

    assert config.get_recent_databases() == []
    assert not _recent_submenu(window).isEnabled()


def test_opening_a_recent_database_switches_to_it(
    window, databases, monkeypatch
) -> None:
    switched: list[Path] = []
    monkeypatch.setattr(
        type(window), "_switch_database", lambda self, path: switched.append(path)
    )

    window._on_open_recent(databases[0])

    assert switched == [databases[0]]


def test_an_entry_whose_file_vanished_says_so_instead_of_switching(
    window, databases, monkeypatch
) -> None:
    """Between building the menu and clicking it - or a volume unmounted."""
    import app.dialogs.main_window as module

    shown: list[str] = []
    monkeypatch.setattr(
        module, "show_message", lambda _p, message_id, **_k: shown.append(message_id)
    )
    switched: list[Path] = []
    monkeypatch.setattr(
        type(window), "_switch_database", lambda self, path: switched.append(path)
    )
    databases[0].unlink()

    window._on_open_recent(databases[0])

    assert shown == ["database.open_failed"]
    assert switched == []
