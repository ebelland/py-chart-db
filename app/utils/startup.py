"""What has to be decided before the main window can be built.

One question: which database is this run about?  Three answers, in order -
the one named on the command line, the one the last run left behind, or a new
empty one in the home directory.

No question is asked for any of them.  A first run used to offer to build the
demo projects and, when that was declined, fall back to an open dialog: two
boxes in front of someone who has not seen the application yet, the second of
which can only open a file they do not have.  Starting on an empty project is
the answer that needs no explaining, and the demo set is still one menu item
away once the window is up (File, Create demo).

It lives here rather than in ``main.py`` because the remaining fallback shows
a box, and every box in this application comes from the catalogue in
config.json (see ``app/utils/messages.py``).  A message id used only from the
root script would sit outside the sweep that proves every id is defined and
every defined id is used.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QWidget

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils.config import get_last_database
from app.utils.i18n import _
from app.utils.messages import show_message

#: What a run with nothing remembered opens.  The home directory, because it
#: is the one place every desktop platform agrees the user can write to, and a
#: fixed name, because the point is to have something to open rather than to
#: name it well - Save as renames it later.
DEFAULT_DATABASE_NAME: str = "Data Hub.dhub"

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
    and a remembered path beats both - because the common case is opening the
    app on yesterday's work, and a dialog in front of that every morning is a
    dialog nobody reads.

    None is returned only when there is nowhere to write and the user then
    cancels the open dialog, which is the one case where the application
    genuinely has no database to be about.
    """
    if argument and argument.strip():
        return Path(argument).expanduser().resolve()

    remembered = _remembered_database()
    if remembered is not None:
        return remembered

    # Nothing remembered: a first run, or config.json was reset.
    return _default_database(parent)


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


def _default_database(parent: QWidget | None) -> Path | None:
    """Return an empty database in the home directory, created if it is not there.

    Nothing is asked, which is the whole point: the alternative was offering
    to build the demo set and then, on "no", an open dialog with nothing in it
    to open.  An empty project is a legitimate way to start, and the demo is
    still reachable from File, Create demo once the window is up.

    An existing file at that path is opened rather than replaced.
    ``create_empty`` only connects, and connecting to a database that already
    has tables in it leaves them exactly as they are - so running twice with a
    reset config.json returns the user to their work rather than to a blank
    file where it used to be.

    The open dialog survives as the fallback for the one case that cannot be
    solved by writing a file: a home directory that is not writable.
    """
    path = Path.home() / DEFAULT_DATABASE_NAME
    applogger.info("No database remembered; starting on %s", path)
    try:
        return SqliteRepo.create_empty(path)
    except Exception as exc:  # noqa: BLE001
        # Read-only home, a full disk, or that name taken by a folder.
        applogger.exception("Could not create a database at %s: %s", path, exc)
        show_message(parent, "startup.default_database_failed", error=exc)
        return _ask_for_a_database(parent)


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
