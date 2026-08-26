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

from app import APP_NAME, APP_VERSION
from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow

from app.styles.style import apply_platform_style, ensure_icon_theme
from app.utils.config import get_language, set_last_database
from app.utils.i18n import install_qt_translations, set_language
from app.utils.startup import select_database


# ----------------------------------------------------------------------
# CLI / bootstrap
# ----------------------------------------------------------------------


def run_app() -> int:
    # Before the QApplication is built: on macOS this is what Cocoa's native
    # Application menu reads for the two entries it labels itself rather than
    # from any QAction's own text - "About {name}" for the one carrying
    # AboutRole, "Quit {name}" for the one every Qt app gets whether it adds
    # one or not. Unset, both read the running executable's name instead -
    # "About Python", "Quit Python" - because this is not a signed .app
    # bundle with its own Info.plist to name it. Nothing about the two
    # entries themselves can be renamed independently of the app's own name;
    # that pairing is Cocoa's, not this application's.
    QApplication.setApplicationName(APP_NAME)
    QApplication.setApplicationVersion(APP_VERSION)
    QApplication.setOrganizationName(APP_NAME)

    app = QApplication(sys.argv)

    # Language before any widget is built: menus read their labels through the
    # translator when they are constructed.
    applogger.info("Interface language: %s", set_language(get_language()))

    # Qt's own strings, which ours cannot reach: QMessageBox builds its Yes
    # and No from Qt's catalogue, so every confirmation in the app asked in
    # the user's language and answered in English until this was installed.
    install_qt_translations(app)

    apply_platform_style(app)

    # Icons come from the desktop's own theme wherever there is one. GNOME and
    # KDE name it themselves; a bare window manager does not, and without this
    # that whole class of desktop shows no icons at all - there is no SVG left
    # underneath to fall back to.
    applogger.info("Icon theme: %s", ensure_icon_theme() or "none installed")

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

        window = MainWindow(repo=repo)
        window.show()

        return app.exec()

    except Exception as exc:  # noqa: BLE001
        applogger.exception("Application startup failed: %s", exc)
        QMessageBox.critical(None, "Data Hub", f"Application startup failed:\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(run_app())
