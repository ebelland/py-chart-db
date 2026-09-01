"""The app menu, on macOS - and everywhere else.

There was no real menu bar at all: with none built, Cocoa still draws the bar
next to the apple, showing only the running process's own name ("Python",
since this is not a signed .app bundle) and nothing beneath it. "Menu" lived
only as the activity rail's popup button - not visible near the apple at all,
and not where a Mac user looks for it.

On macOS this now also fills a real QMenuBar; the popup keeps working
everywhere else. Settings and Credits carry explicit MenuRole values, which
is what actually moves them into the native application menu next to the
apple - Cocoa relocates a role-marked action wherever in the menu bar it was
declared, so they read the way "Preferences..." and "About" read in every
other Mac app, in the one slot this application's menu structure cannot
otherwise reach.

Qt object lifetime note for whoever edits this file: every intermediate
QAction/QMenu returned along a chain (menuBar().actions()[0].menu(), and so
on) has to be kept in a named variable for as long as it is used. Losing the
reference to an intermediate wrapper - even briefly, inside one chained
expression - lets PySide6 tear down the C++ object under it, which surfaces
later as "Internal C++ object already deleted" on a completely unrelated
line. Every test below keeps each step named for exactly this reason.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtGui import QAction

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
import app.dialogs.main_window as main_window_module
from app.dialogs.main_window import MainWindow


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    yield built
    built.close()


@pytest.fixture
def make_window(qapp, repo: SqliteRepo):
    """Build a MainWindow, and guarantee applogger loses track of its status
    bar again afterwards.

    MainWindow.__init__ points the (process-wide singleton) logger at
    self.statusBar(). Nothing it does undoes that on close, so a later test -
    any later test, anywhere in the suite, that happens to log a message at a
    level the status bar shows - was handed a reference to an already-deleted
    C++ QStatusBar and crashed with "Internal C++ object already deleted", on
    a line that has nothing to do with this file. One MainWindow built here
    without this fixture was enough to take the rest of the suite down.
    """
    built_windows: list[MainWindow] = []

    def factory() -> MainWindow:
        built = MainWindow(repo=repo)
        built_windows.append(built)
        return built

    yield factory

    for built in built_windows:
        built.close()
    applogger.set_status_bar(None)


@pytest.fixture
def window(make_window):
    return make_window()


def _rail_tooltips(window: MainWindow) -> list[str]:
    buttons = window._left_rail.findChildren(type(window._nav_buttons[0]))
    return [button.toolTip() for button in buttons]


# ----------------------------------------------------------------------
# The macOS menu bar
# ----------------------------------------------------------------------
def test_the_menu_bar_carries_one_top_level_menu(window: MainWindow) -> None:
    menu_bar = window.menuBar()
    top_actions = menu_bar.actions()

    assert len(top_actions) == 1
    # "File", not "Menu": a Mac user looks for New/Open/Import under File,
    # and the rail button this replaced is not on screen to be echoed.
    assert top_actions[0].text() == "File"


def test_the_menu_holds_every_item_the_popup_has(window: MainWindow) -> None:
    menu_bar = window.menuBar()
    top_actions = menu_bar.actions()
    top_action = top_actions[0]
    menu = top_action.menu()
    items = menu.actions()

    ids = [action.data() for action in items if not action.isSeparator()]
    assert ids == [
        "new",
        "open",
        "import",
        "query_builder",
        "create_demo",
        "optimize_db",
        "settings",
        "log_viewer",
        "credits",
    ]


def test_settings_stays_in_the_file_menu(window: MainWindow) -> None:
    """It used to carry PreferencesRole, and that is what broke it.

    Cocoa relocates a role-marked action into the application menu next to
    the apple, wherever it was declared. _build_app_menu rebuilds the whole
    bar - after Settings closes, among other times - and menuBar().clear()
    then leaves that native item referring to no live QAction, so choosing
    Preferences did nothing at all. Without the role it stays in the menu
    that is rebuilt with it.
    """
    menu_bar = window.menuBar()
    top_actions = menu_bar.actions()
    top_action = top_actions[0]
    menu = top_action.menu()
    items = menu.actions()

    settings_action = next(action for action in items if action.data() == "settings")
    assert settings_action.menuRole() == QAction.MenuRole.NoRole
    assert settings_action.isEnabled()


def test_credits_carries_the_about_role(window: MainWindow) -> None:
    menu_bar = window.menuBar()
    top_actions = menu_bar.actions()
    top_action = top_actions[0]
    menu = top_action.menu()
    items = menu.actions()

    credits_action = next(action for action in items if action.data() == "credits")
    assert credits_action.menuRole() == QAction.MenuRole.AboutRole


def test_nothing_else_is_given_a_role_it_did_not_ask_for(window: MainWindow) -> None:
    """Only Credits is meant to be pulled out of this menu."""
    menu_bar = window.menuBar()
    top_actions = menu_bar.actions()
    top_action = top_actions[0]
    menu = top_action.menu()
    items = menu.actions()

    ordinary = [
        action for action in items
        if not action.isSeparator() and action.data() not in ("settings", "credits")
    ]
    assert ordinary
    assert all(
        action.menuRole() == QAction.MenuRole.TextHeuristicRole for action in ordinary
    )


# ----------------------------------------------------------------------
# The rail no longer offers a redundant popup
# ----------------------------------------------------------------------
def test_the_rail_has_no_menu_button_on_macos(window: MainWindow) -> None:
    assert "Show main menu" not in _rail_tooltips(window)


def test_the_rail_still_has_the_three_real_pages(window: MainWindow) -> None:
    assert _rail_tooltips(window) == [
        "Show data tables",
        "Show chart options",
        "Show series operations",
    ]


def test_page_switching_no_longer_needs_an_offset(window: MainWindow) -> None:
    """The button group used to reserve id 0 for the popup and shift every
    real page by one; with the popup gone from the group entirely, id and
    stack-page index are the same number."""
    window._set_nav_index(0)
    assert window._left_stack.currentIndex() == 0
    assert window._nav_buttons[0].isChecked()

    window._set_nav_index(2)
    assert window._left_stack.currentIndex() == 2
    assert window._nav_buttons[2].isChecked()


def test_startup_selects_and_highlights_the_first_page(window: MainWindow) -> None:
    """Under the old offset, _set_nav_index(0) - called once at startup to
    select the default page - addressed the popup's own reserved slot and
    did nothing: the data page showed only because QStackedWidget already
    defaults to index 0, and its rail button was never actually checked."""
    assert window._left_stack.currentIndex() == 0
    assert window._nav_buttons[0].isChecked()


# ----------------------------------------------------------------------
# Everywhere else, unchanged
# ----------------------------------------------------------------------
def test_off_macos_the_rail_keeps_the_popup_button(
    make_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_window_module, "IS_MACOS", False)

    built = make_window()

    tooltips = _rail_tooltips(built)
    assert "Show main menu" in tooltips
    assert len(tooltips) == 4


def test_off_macos_the_menu_button_opens_the_popup(
    make_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_window_module, "IS_MACOS", False)

    built = make_window()

    buttons = built._left_rail.findChildren(type(built._nav_buttons[0]))
    menu_button = next(
        button for button in buttons if button.toolTip() == "Show main menu"
    )
    assert menu_button.menu() is built._app_menu
    assert built._app_menu is not None


def test_off_macos_page_switching_is_unaffected(
    make_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_window_module, "IS_MACOS", False)

    built = make_window()

    built._set_nav_index(1)
    assert built._left_stack.currentIndex() == 1
    assert built._nav_buttons[1].isChecked()


# ----------------------------------------------------------------------
# One list feeds both
# ----------------------------------------------------------------------
def test_the_popup_and_the_menu_bar_share_one_item_list(window: MainWindow) -> None:
    """They cannot drift apart the way two hand-kept copies would."""
    popup_ids = [
        action.data()
        for action in window._app_menu.actions()
        if not action.isSeparator()
    ]

    menu_bar = window.menuBar()
    top_actions = menu_bar.actions()
    top_action = top_actions[0]
    bar_menu = top_action.menu()
    bar_ids = [
        action.data() for action in bar_menu.actions() if not action.isSeparator()
    ]

    assert popup_ids == bar_ids


def test_rebuilding_the_menu_does_not_duplicate_the_bar(window: MainWindow) -> None:
    """_build_app_menu runs again after Settings closes - language or theme
    may have changed. The menu bar must be replaced, not added to."""
    window._build_app_menu()
    window._build_app_menu()

    menu_bar = window.menuBar()
    assert len(menu_bar.actions()) == 1
