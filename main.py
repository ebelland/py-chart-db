from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6 import __file__ as PYSIDE6_FILE

# Ensure Qt can locate its platform plugins.
os.environ.setdefault(
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    os.path.join(os.path.dirname(PYSIDE6_FILE), "Qt", "plugins"),
)
# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
from app.logs.logger import AppLogger, applogger
AppLogger.configure()
applogger.debug("Application starting...")
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow

from app.styles.style import apply_platform_style
from app.utils.config import get_language, get_last_database, set_last_database
from app.utils.i18n import set_language


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _select_database() -> Path | None:
    """Return a database path from argv, last config, or an open-file dialog."""
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return Path(sys.argv[1]).expanduser().resolve()

    last_database = get_last_database()
    if last_database:
        path = Path(last_database).expanduser().resolve()
        if path.exists():
            return path

        # The remembered database has been moved, renamed or deleted.  Create
        # an empty one in its place rather than opening a file dialog: the
        # dialog can only *open* something, so a user whose only project just
        # disappeared is offered a list they have nothing to pick from, and
        # cancelling it exits the application.  An empty project starts.
        applogger.warning(
            "Last database not found: %s. Creating an empty one in its place.",
            path,
        )
        try:
            return SqliteRepo.create_empty(path)
        except Exception as exc:  # noqa: BLE001
            # Unwritable directory, read-only volume, a name that is now a
            # folder: fall through to the dialog rather than refusing to start.
            applogger.exception("Could not create a database at %s: %s", path, exc)

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Open database",
        str(Path.home()),
        "Data Hub DB (*.dhub);;All files (*.*)",
    )
    if not file_path:
        return None

    return Path(file_path).expanduser().resolve()


# ----------------------------------------------------------------------
# CLI / bootstrap
# ----------------------------------------------------------------------


def run_app() -> int:
    app = QApplication(sys.argv)

    # Language before any widget is built: menus read their labels through the
    # translator when they are constructed.
    applogger.info("Interface language: %s", set_language(get_language()))

    apply_platform_style(app)

    try:
        db_path = _select_database()
        if db_path is None:
            applogger.info("No database selected. Application exiting.")
            return 0

        repo = SqliteRepo(db_path=db_path)
        set_last_database(db_path)

        window = MainWindow(repo=repo, db_path=db_path)
        window.show()

        return app.exec()

    except Exception as exc:  # noqa: BLE001
        applogger.exception("Application startup failed: %s", exc)
        QMessageBox.critical(None, "Data Hub", f"Application startup failed:\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(run_app())
