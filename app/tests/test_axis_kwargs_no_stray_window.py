"""The kwargs editor must not leave stray top-level windows behind.

With the "Axis drawing (kwargs)" tab open, switching chart panels rebuilds
the editor. The old one used to be detached with ``setParent(None)``, which
does not merely unparent a widget - it promotes it to a *top-level window*.
Qt was then free to show it, and did: a fully-populated kwargs panel floating
over the application with its own title bar and traffic lights.

``takeAt`` has already removed the widget from the layout, so the parent can
simply be left alone and the widget hidden and deleteLater()'d. Keeping the
parent matters even though the widget is being deleted: between deleteLater()
and the deletion actually happening, an unparented widget is a real window
that Qt can put on screen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, QEvent

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow
from app.logs.logger import applogger
from app.widgets.dictionary_editor import DictEditorPanel

#: Index of the "Axis drawing (kwargs)" tab in AxisPropertiesWidget.
KWARGS_TAB_INDEX = 1


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    built.import_dataframe(
        pd.DataFrame({"x": np.arange(10), "y": np.arange(10) ** 2}),
        table_name="src",
        normalize_columns=False,
    )
    # Two figures, so there is something to switch between.
    for name in ("fig one", "fig two"):
        figure_id = built.create_figure_descriptor(name=name)
        axis_id = built.create_axis_descriptor(
            figure_id=figure_id,
            axis_index=0,
            chart_type="Scatter Plot",
            title=name,
            x_label="",
            y_label="",
            options={},
        )
        built.create_series_descriptor(
            axis_id=axis_id,
            series_index=0,
            name="s",
            sql_query="SELECT x, y FROM src",
            roles={"x": "x", "y": "y"},
            style={},
        )
    yield built
    built.close()


@pytest.fixture
def window(qapp, repo: SqliteRepo, tmp_db_path: Path):
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    yield built
    built.close()
    applogger.set_status_bar(None)


def _flush_deletions(qapp) -> None:
    """Run the deferred deletions deleteLater() queued.

    processEvents() alone does not deliver DeferredDelete - it is held until
    the event loop that queued it unwinds, which never happens in a test.
    """
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _panels(qapp) -> list[DictEditorPanel]:
    return [w for w in qapp.allWidgets() if isinstance(w, DictEditorPanel)]


def _panels_of(qapp, window: MainWindow) -> list[DictEditorPanel]:
    """Only the editors belonging to *window*.

    Scoped rather than counted app-wide: a MainWindow closed by an earlier
    test in this module is not necessarily deleted yet, and its own editor
    would otherwise be counted against this one.
    """
    return [panel for panel in _panels(qapp) if panel.window() is window]


def test_switching_panels_leaves_no_orphan_window(qapp, window: MainWindow) -> None:
    """The bug itself: every switch used to leave one more stray window."""
    window._axis_widget._tabs.setCurrentIndex(KWARGS_TAB_INDEX)
    qapp.processEvents()

    for index in range(8):
        window._tabs.setCurrentIndex(index % 2)
        qapp.processEvents()
    _flush_deletions(qapp)

    orphans = [panel for panel in _panels(qapp) if panel.isWindow()]
    assert orphans == []


def test_switching_panels_does_not_accumulate_editors(qapp, window: MainWindow) -> None:
    """One live editor, not one per switch - the leak behind the stray window."""
    window._axis_widget._tabs.setCurrentIndex(KWARGS_TAB_INDEX)
    qapp.processEvents()

    for index in range(8):
        window._tabs.setCurrentIndex(index % 2)
        qapp.processEvents()
    _flush_deletions(qapp)

    assert len(_panels_of(qapp, window)) <= 1


def test_the_editor_still_works_after_switching(qapp, window: MainWindow) -> None:
    """Deleting the old editor must not leave the panel empty."""
    window._axis_widget._tabs.setCurrentIndex(KWARGS_TAB_INDEX)
    qapp.processEvents()

    window._tabs.setCurrentIndex(1)
    qapp.processEvents()
    _flush_deletions(qapp)

    editor = window._axis_widget._kwargs_editor
    assert editor is not None
    assert editor.tree.topLevelItemCount() > 0


def test_no_visible_window_besides_the_main_one(qapp, window: MainWindow) -> None:
    """What the user actually saw: a second window on screen."""
    window.show()
    window._axis_widget._tabs.setCurrentIndex(KWARGS_TAB_INDEX)
    qapp.processEvents()

    for index in range(4):
        window._tabs.setCurrentIndex(index % 2)
        qapp.processEvents()
    _flush_deletions(qapp)

    visible = [w for w in qapp.topLevelWidgets() if w.isVisible()]
    assert visible == [window]
