"""The demo picker, and the menu action that opens it.

First run offers the whole set automatically and opens the complete one (see
app/utils/startup.py). This is the way back to that choice afterwards -
rebuilding one demo that got edited, or looking at a subject that was skipped
the first time - without starting the application over.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.demo_project import DEMO_PROJECTS
from app.dialogs.create_demo_dialog import CreateDemoDialog


@pytest.fixture
def dialog(qapp) -> CreateDemoDialog:
    return CreateDemoDialog()


# ----------------------------------------------------------------------
# The picker itself
# ----------------------------------------------------------------------
def test_every_demo_project_is_offered(dialog: CreateDemoDialog) -> None:
    assert dialog._list.count() == len(DEMO_PROJECTS)


def test_the_complete_project_is_selected_by_default(dialog: CreateDemoDialog) -> None:
    """DEMO_PROJECTS[0] is "Getting started" - the one a first run opens, and
    the reasonable thing to land on here too."""
    assert dialog._list.currentRow() == 0
    assert dialog._list.currentItem().data(_user_role()) is DEMO_PROJECTS[0]


def test_the_summary_follows_the_selection(dialog: CreateDemoDialog) -> None:
    dialog._list.setCurrentRow(2)

    assert dialog._summary.text() == DEMO_PROJECTS[2].summary


def test_closing_without_choosing_leaves_nothing_chosen(dialog: CreateDemoDialog) -> None:
    dialog.reject()

    assert dialog.chosen is None


def test_confirming_records_the_selected_project(dialog: CreateDemoDialog) -> None:
    dialog._list.setCurrentRow(3)

    dialog._confirm()

    assert dialog.chosen is DEMO_PROJECTS[3]
    assert dialog.result() == CreateDemoDialog.DialogCode.Accepted


def _user_role():
    from PySide6.QtCore import Qt

    return Qt.ItemDataRole.UserRole


# ----------------------------------------------------------------------
# The menu action, end to end
# ----------------------------------------------------------------------
@pytest.fixture
def repo(tmp_db_path: Path):
    from app.data.sqlite_repo import SqliteRepo

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
def window(qapp, repo):
    from app.logs.logger import applogger
    from app.dialogs.main_window import MainWindow

    built = MainWindow(repo=repo)
    yield built
    built.close()
    applogger.set_status_bar(None)


def test_choosing_a_demo_builds_it_and_opens_it(
    window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path: pick one, say where, and it becomes the open database."""
    from PySide6.QtWidgets import QFileDialog

    target = tmp_path / "peak scan demo.dhub"

    monkeypatch.setattr(
        CreateDemoDialog, "exec", lambda self: (setattr(self, "chosen", DEMO_PROJECTS[9]) or True)
    )
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *_a, **_k: (str(target), "")),
    )

    window._on_create_demo()

    assert target.exists()
    assert window._repo.db_path == target


def test_declining_the_save_dialog_builds_nothing(
    window, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(
        CreateDemoDialog, "exec", lambda self: (setattr(self, "chosen", DEMO_PROJECTS[0]) or True)
    )
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *_a, **_k: ("", ""))
    )

    original_db_path = window._repo.db_path
    window._on_create_demo()

    assert window._repo.db_path == original_db_path


def test_cancelling_the_picker_never_opens_a_save_dialog(
    window, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(CreateDemoDialog, "exec", lambda self: False)
    asked = []
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *_a, **_k: asked.append(1) or ("", "")),
    )

    window._on_create_demo()

    assert asked == []


def test_the_menu_offers_create_demo(window) -> None:
    ids = [
        action.data()
        for action in window._app_menu.actions()
        if not action.isSeparator()
    ]
    assert "create_demo" in ids
