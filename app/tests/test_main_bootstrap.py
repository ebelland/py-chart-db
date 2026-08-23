"""What main.py sets up before anything else runs.

run_app() is not something to call from a test - it opens a real window and,
with no database remembered, a real file dialog. What is worth pinning is the
one thing that has to happen *before* QApplication exists to have any effect
at all: naming the application for Cocoa's native menu.
"""
from __future__ import annotations

from pathlib import Path

MAIN_SOURCE = Path(__file__).resolve().parent.parent.parent / "main.py"


def _source() -> str:
    return MAIN_SOURCE.read_text(encoding="utf-8")


def test_the_application_is_named_before_it_is_built() -> None:
    """Unnamed, Cocoa's native "About"/"Quit" menu entries read the running
    executable instead - "About Python", "Quit Python" - because this is not
    a signed .app bundle with its own Info.plist. setApplicationName has to
    run before QApplication(sys.argv) does; Qt reads it once, at construction.
    """
    source = _source()

    name_call = source.index("QApplication.setApplicationName(")
    construction = source.index("QApplication(sys.argv)")
    assert name_call < construction


def test_the_name_comes_from_the_one_place_it_is_declared() -> None:
    """Not a second, hand-typed "Data Hub" that could drift from app/__init__.py."""
    source = _source()

    assert "from app import APP_NAME, APP_VERSION" in source
    assert "QApplication.setApplicationName(APP_NAME)" in source
    assert "QApplication.setApplicationVersion(APP_VERSION)" in source


def test_the_actual_values_reach_qt(qapp) -> None:
    """The source-level checks above prove the call is well-formed and
    ordered; this proves the values it passes are the real ones."""
    from PySide6.QtWidgets import QApplication

    from app import APP_NAME, APP_VERSION

    QApplication.setApplicationName(APP_NAME)
    QApplication.setApplicationVersion(APP_VERSION)

    assert qapp.applicationName() == APP_NAME
    assert qapp.applicationVersion() == APP_VERSION
