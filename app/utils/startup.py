"""What has to be decided before the main window can be built.

One question: which database is this run about?  Three answers, in order -
the one named on the command line, the one the last run left behind, or one
the user is asked for.

It lives here rather than in ``main.py`` because two of the three answers show
a box, and every box in this application comes from the catalogue in
config.json (see ``app/utils/messages.py``).  A message id used only from the
root script would sit outside the sweep that proves every id is defined and
every defined id is used.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QWidget

from app.data.demo_project import DEMO_PROJECTS, build_demo_projects
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils.config import get_last_database
from app.utils.i18n import _
from app.utils.messages import ask, show_message

#: Where a first run writes the demo projects.  One folder in the home
#: directory, because the demo is a *set* now - one file per subject, each
#: named for what it shows - and ten loose .dhub files in someone's home is
#: not a gift.
DEMO_FOLDER_NAME: str = "Data Hub demo projects"

#: The open dialog's filter.  ``.dhub`` first so the file the user is looking
#: for is the one they see.
DATABASE_FILTER: str = "Data Hub DB (*.dhub);;All files (*.*)"


def select_database(
    argument: str | None = None,
    *,
    parent: QWidget | None = None,
) -> Path | None:
    """Return the database this run should open, or None to exit.

    The order is deliberate: an explicit argument beats a remembered path,
    and a remembered path beats asking - because the common case is opening
    the app on yesterday's work, and a dialog in front of that every morning
    is a dialog nobody reads.
    """
    if argument and argument.strip():
        return Path(argument).expanduser().resolve()

    remembered = _remembered_database()
    if remembered is not None:
        return remembered

    # Nothing remembered: this is a first run, or config.json was reset.
    demo = _offer_the_demo_project(parent)
    if demo is not None:
        return demo

    return _ask_for_a_database(parent)


def _remembered_database() -> Path | None:
    """Return the database the last run left behind, creating it if gone."""
    last_database = get_last_database()
    if not last_database:
        return None

    path = Path(last_database).expanduser().resolve()
    if path.exists():
        return path

    # Moved, renamed or deleted.  An empty database in its place rather than
    # a file dialog: the dialog can only *open* something, so the user whose
    # only project just disappeared is shown a list with nothing in it, and
    # cancelling it exits the application.
    applogger.warning(
        "Last database not found: %s. Creating an empty one in its place.",
        path,
    )
    try:
        return SqliteRepo.create_empty(path)
    except Exception as exc:  # noqa: BLE001
        # Unwritable directory, read-only volume, a name that is now a folder.
        applogger.exception("Could not create a database at %s: %s", path, exc)
        return None


def _offer_the_demo_project(parent: QWidget | None) -> Path | None:
    """Ask whether to build the demo project, and build it if so.

    An empty database is a legitimate way to start and a terrible way to be
    introduced: every panel is empty, and nothing in an empty window says what
    the application is *for*.  The demo is a set of real projects built through
    the same repository the app uses, one per subject and each named for what
    it shows.

    The complete one is what opens; the rest sit beside it in the folder,
    which is what makes them findable - and they can be dropped straight onto
    the window.
    """
    if not ask(parent, "startup.offer_demo"):
        return None

    target = Path.home() / DEMO_FOLDER_NAME
    applogger.info("Building %d demo projects in %s", len(DEMO_PROJECTS), target)
    try:
        written = build_demo_projects(target)
    except Exception as exc:  # noqa: BLE001
        applogger.exception("Could not build the demo projects: %s", exc)
        show_message(parent, "startup.demo_failed", error=exc)
        return None

    # The complete project first: see DEMO_PROJECTS.
    return written[0] if written else None


def _ask_for_a_database(parent: QWidget | None) -> Path | None:
    """Return a database chosen in the open dialog, or None if cancelled."""
    file_path, _unused = QFileDialog.getOpenFileName(
        parent,
        _("Open database"),
        str(Path.home()),
        DATABASE_FILTER,
    )
    if not file_path:
        return None
    return Path(file_path).expanduser().resolve()
