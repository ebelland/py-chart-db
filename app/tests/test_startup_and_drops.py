"""Two ways in: what opens at startup, and what a drop on the window does.

Both used to have the same shape of bug - a path that only *looked* handled.
Startup fell through to an open dialog whenever the remembered database was
gone, which offers a list to someone who has nothing to pick from; and the
window refused every drop, so a file dragged onto it did nothing at all and
said nothing about why.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.import_data_dialog import IMPORTABLE_SUFFIXES, is_importable
from app.dialogs.main_window import MainWindow
from app.utils import startup


# ----------------------------------------------------------------------
# Which database opens
# ----------------------------------------------------------------------
@pytest.fixture
def no_dialogs(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Answer every box and dialog from the test, and record what was asked."""
    asked: dict = {"ask": [], "message": [], "browse": 0, "demo": []}

    monkeypatch.setattr(
        startup, "ask", lambda _parent, message_id, **_k: asked["ask"].append(message_id) or False
    )
    monkeypatch.setattr(
        startup,
        "show_message",
        lambda _parent, message_id, **_k: asked["message"].append(message_id),
    )
    monkeypatch.setattr(
        startup.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *_a, **_k: (asked.update(browse=asked["browse"] + 1) or "", "")),
    )
    return asked


def test_an_argument_wins_over_everything(no_dialogs, tmp_path: Path) -> None:
    """Opening a file from the shell must not be second-guessed."""
    chosen = startup.select_database(str(tmp_path / "explicit.dhub"))

    assert chosen == (tmp_path / "explicit.dhub").resolve()
    assert no_dialogs["ask"] == [], "nothing to ask: the path was given"


def test_the_remembered_database_opens_without_a_question(
    no_dialogs, monkeypatch: pytest.MonkeyPatch, tmp_db_path: Path
) -> None:
    """The common case is opening the app on yesterday's work."""
    SqliteRepo.create_empty(tmp_db_path)
    monkeypatch.setattr(startup, "get_last_database", lambda: str(tmp_db_path))

    assert startup.select_database() == tmp_db_path.resolve()
    assert no_dialogs["browse"] == 0


def test_a_remembered_database_that_is_gone_is_recreated_empty(
    no_dialogs, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The file dialog can only *open* something. Offering it to someone whose
    only project just disappeared is offering an empty list, and cancelling it
    exits the application."""
    missing = tmp_path / "moved_away.dhub"
    monkeypatch.setattr(startup, "get_last_database", lambda: str(missing))

    chosen = startup.select_database()

    assert chosen == missing
    assert chosen.exists()
    assert no_dialogs["browse"] == 0


def test_a_first_run_is_offered_the_demo_projects(
    no_dialogs, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """And the complete one is the one that opens."""
    built = [tmp_path / "Getting started.dhub", tmp_path / "One subject.dhub"]
    monkeypatch.setattr(startup, "get_last_database", lambda: "")
    monkeypatch.setattr(startup, "ask", lambda *_a, **_k: True)
    monkeypatch.setattr(startup, "build_demo_projects", lambda _target: built)

    assert startup.select_database() == built[0]
    assert no_dialogs["browse"] == 0, "the demo answered the question"


def test_declining_the_demo_falls_back_to_the_open_dialog(
    no_dialogs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(startup, "get_last_database", lambda: "")

    assert startup.select_database() is None
    assert no_dialogs["ask"] == ["startup.offer_demo"]
    assert no_dialogs["browse"] == 1


def test_a_demo_that_cannot_be_built_says_so_and_still_offers_the_dialog(
    no_dialogs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure to write the example must not be a failure to start."""
    monkeypatch.setattr(startup, "get_last_database", lambda: "")
    monkeypatch.setattr(startup, "ask", lambda *_a, **_k: True)

    def explode(_target):
        raise OSError("read-only volume")

    monkeypatch.setattr(startup, "build_demo_projects", explode)

    assert startup.select_database() is None
    assert no_dialogs["message"] == ["startup.demo_failed"]
    assert no_dialogs["browse"] == 1


# ----------------------------------------------------------------------
# What a drop does
# ----------------------------------------------------------------------
def test_a_dropped_project_is_opened() -> None:
    kind, paths = MainWindow.accepted_drop([Path("/tmp/Project.dhub")])

    assert kind == "database"
    assert paths == [Path("/tmp/Project.dhub")]


@pytest.mark.parametrize("suffix", IMPORTABLE_SUFFIXES)
def test_every_format_the_readers_know_is_accepted(suffix: str) -> None:
    """The drop test and the file dialog filter read the same list, so a
    format added to the readers cannot be one the window still refuses."""
    kind, paths = MainWindow.accepted_drop([Path(f"/tmp/data{suffix}")])

    assert kind == "import"
    assert len(paths) == 1


def test_the_extension_is_matched_whatever_its_case() -> None:
    assert is_importable("/tmp/READINGS.CSV")


def test_a_file_nothing_can_read_is_refused() -> None:
    """Refused before the cursor changes: the window says no by not offering
    to accept, rather than by a box after the drop."""
    kind, paths = MainWindow.accepted_drop([Path("/tmp/photo.png")])

    assert kind == ""
    assert paths == []


def test_several_data_files_are_all_accepted() -> None:
    kind, paths = MainWindow.accepted_drop(
        [Path("/tmp/a.csv"), Path("/tmp/b.xlsx"), Path("/tmp/notes.png")]
    )

    assert kind == "import"
    assert paths == [Path("/tmp/a.csv"), Path("/tmp/b.xlsx")]


def test_two_projects_at_once_are_refused() -> None:
    """Opening two projects has no meaning, and picking one for the user is
    picking wrong half the time."""
    kind, _paths = MainWindow.accepted_drop(
        [Path("/tmp/one.dhub"), Path("/tmp/two.dhub")]
    )

    assert kind == ""


def test_a_project_wins_over_data_dropped_with_it() -> None:
    """Importing into a database that is about to be closed is worse than
    ignoring the file dropped with it."""
    kind, paths = MainWindow.accepted_drop([Path("/tmp/p.dhub"), Path("/tmp/a.csv")])

    assert kind == "database"
    assert paths == [Path("/tmp/p.dhub")]


def test_an_empty_drop_is_refused() -> None:
    assert MainWindow.accepted_drop([]) == ("", [])
