"""The demo picker, and the menu action that opens it.

First run offers the whole set automatically and opens the complete one (see
app/utils/startup.py). This is the way back to that choice afterwards -
rebuilding one demo that got edited, or looking at a subject that was skipped
the first time - without starting the application over.

Loading a demo used to ask where to save it first, exactly like "New" does.
It no longer does: a demo is a fixed, disposable file by name, copied into
``projects/`` beside the application (demo_project.PROJECTS_DIR) and opened -
the same reasoning that lets a first run open with no dialog at all (see
app.utils.startup.DEFAULT_DATABASE_NAME).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data import demo_project
from app.data.demo_project import DEMO_PROJECTS, build_demo_project
from app.dialogs import main_window as main_window_module
from app.dialogs.load_demo_dialog import LoadDemoDialog


@pytest.fixture
def dialog(qapp) -> LoadDemoDialog:
    return LoadDemoDialog()


# ----------------------------------------------------------------------
# The picker itself
# ----------------------------------------------------------------------
def test_every_demo_project_is_offered(dialog: LoadDemoDialog) -> None:
    assert dialog._list.count() == len(DEMO_PROJECTS)


def test_the_complete_project_is_selected_by_default(dialog: LoadDemoDialog) -> None:
    """DEMO_PROJECTS[0] is "Getting started" - the one a first run opens, and
    the reasonable thing to land on here too."""
    assert dialog._list.currentRow() == 0
    assert dialog._list.currentItem().data(_user_role()) is DEMO_PROJECTS[0]


def test_the_summary_follows_the_selection(dialog: LoadDemoDialog) -> None:
    dialog._list.setCurrentRow(2)

    assert dialog._summary.text() == DEMO_PROJECTS[2].summary


def test_closing_without_choosing_leaves_nothing_chosen(dialog: LoadDemoDialog) -> None:
    dialog.reject()

    assert dialog.chosen is None


def test_confirming_records_the_selected_project(dialog: LoadDemoDialog) -> None:
    dialog._list.setCurrentRow(3)

    dialog._confirm()

    assert dialog.chosen is DEMO_PROJECTS[3]
    assert dialog.result() == LoadDemoDialog.DialogCode.Accepted


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
def window(qapp, repo, tmp_db_path: Path):
    from app.logs.logger import applogger
    from app.dialogs.main_window import MainWindow

    built = MainWindow(repo=repo, db_path=tmp_db_path)
    yield built
    built.close()
    applogger.set_status_bar(None)


def test_choosing_a_demo_loads_it_with_no_save_dialog(
    window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path: pick one, and it becomes the open database - nothing
    else asked.

    "Load demo" copies a pre-built file rather than building on the spot -
    see app.data.demo_project.copy_demo_project - so the test pre-builds the
    chosen project into a temp DEMO_DIR instead of relying on the real one,
    which is not version-controlled and may not exist in a fresh checkout.
    """
    chosen = DEMO_PROJECTS[9]
    build_demo_project(tmp_path / "source" / chosen.path_name, chosen.figures)
    monkeypatch.setattr(demo_project, "DEMO_DIR", tmp_path / "source")
    monkeypatch.setattr(main_window_module, "PROJECTS_DIR", tmp_path / "projects")

    monkeypatch.setattr(
        LoadDemoDialog, "exec", lambda self: (setattr(self, "chosen", chosen) or True)
    )

    window._on_load_demo()

    target = tmp_path / "projects" / chosen.path_name
    assert target.exists()
    assert window._db_path == target


def test_loading_the_same_demo_again_overwrites_its_previous_copy(
    window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading it again is how you get back the pristine version, not a
    second file to clean up."""
    chosen = DEMO_PROJECTS[0]
    build_demo_project(tmp_path / "source" / chosen.path_name, chosen.figures)
    monkeypatch.setattr(demo_project, "DEMO_DIR", tmp_path / "source")
    monkeypatch.setattr(main_window_module, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(
        LoadDemoDialog, "exec", lambda self: (setattr(self, "chosen", chosen) or True)
    )

    window._on_load_demo()
    target = tmp_path / "projects" / chosen.path_name
    edited_marker = b"not a real database, just proving the file got replaced"
    target.write_bytes(edited_marker)

    window._on_load_demo()

    assert target.read_bytes() != edited_marker


def test_cancelling_the_picker_loads_nothing(
    window, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(LoadDemoDialog, "exec", lambda self: False)

    original_db_path = window._db_path
    window._on_load_demo()

    assert window._db_path == original_db_path


def test_the_menu_offers_load_demo(window) -> None:
    ids = [
        action.data()
        for action in window._app_menu.actions()
        if not action.isSeparator()
    ]
    assert "load_demo" in ids
