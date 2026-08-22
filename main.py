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
from PySide6.QtWidgets import QApplication, QMessageBox

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow

from app.styles.style import apply_platform_style
from app.utils.config import get_language, set_last_database
from app.utils.i18n import set_language
from app.utils.startup import select_database


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
        # Which database this run is about, and what to do when there is
        # none, lives in app/utils/startup.py - it shows boxes, and those
        # come from the catalogue every other box in the app comes from.
        db_path = select_database(sys.argv[1] if len(sys.argv) > 1 else None)
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
