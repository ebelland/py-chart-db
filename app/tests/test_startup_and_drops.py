"""Two ways in: what opens at startup, and what a drop on the window does.

Both used to have the same shape of bug - a path that only *looked* handled.
Startup fell through to an open dialog whenever the remembered database was
gone, which offers a list to someone who has nothing to pick from; and the
window refused every drop, so a file dragged onto it did nothing at all and
said nothing about why.

Startup now asks nothing at all in the ordinary cases. It used to offer to
build the demo set on a first run and fall back to that same empty open
dialog when the offer was declined; these tests are largely about there being
no question left to answer.
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

    # startup asks nothing any more; the list stays in the fixture so a
    # question added back here fails a test instead of appearing silently.
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


def test_a_first_run_opens_an_empty_database_without_asking_anything(
    no_dialogs, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No demo offer, no open dialog: a project to start working in."""
    monkeypatch.setattr(startup, "get_last_database", lambda: "")
    monkeypatch.setattr(startup.Path, "home", staticmethod(lambda: tmp_path))

    chosen = startup.select_database()

    assert chosen == tmp_path / startup.DEFAULT_DATABASE_NAME
    assert chosen.exists()
    assert no_dialogs["ask"] == [], "a first run is asked nothing"
    assert no_dialogs["browse"] == 0


def test_a_first_run_keeps_a_database_that_is_already_there(
    no_dialogs, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A reset config.json must not blank the work it forgot about."""
    existing = tmp_path / startup.DEFAULT_DATABASE_NAME
    repo = SqliteRepo(db_path=existing)
    repo.import_dataframe(
        __import__("pandas").DataFrame({"a": [1, 2, 3]}), table_name="kept"
    )
    repo.close()

    monkeypatch.setattr(startup, "get_last_database", lambda: "")
    monkeypatch.setattr(startup.Path, "home", staticmethod(lambda: tmp_path))

    assert startup.select_database() == existing

    reopened = SqliteRepo(db_path=existing)
    try:
        assert "kept" in reopened.list_table_names()
    finally:
        reopened.close()


def test_an_unwritable_home_says_so_and_falls_back_to_the_open_dialog(
    no_dialogs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one case a file cannot solve: there is nowhere to write one."""
    monkeypatch.setattr(startup, "get_last_database", lambda: "")

    def explode(_path):
        raise OSError("read-only volume")

    monkeypatch.setattr(startup.SqliteRepo, "create_empty", staticmethod(explode))

    assert startup.select_database() is None
    assert no_dialogs["message"] == ["startup.default_database_failed"]
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
