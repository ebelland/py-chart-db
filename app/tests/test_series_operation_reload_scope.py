"""_open_series_operation used to redraw every chart, not just the one it ran on.

Every Apply/Preview/Close of a series-operation dialog called
``self._reload_tabs()`` unconditionally afterwards, on the theory that the
operation might have created a figure of its own (Fit's "new figure"
destination, for instance) that needs a tab of its own. But ``_reload_tabs``
does not check for that case narrowly - it clears the whole tab bar and
builds a brand-new ``ChartPanel`` (a full render) for every figure in the
database, every time. So running Fit or any other operation on one chart in
a database of twenty re-rendered all twenty, every single time.

The fix compares the figure count before and after the dialog runs, and only
pays for ``_reload_tabs`` when a figure was actually added.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Signal

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.main_window import MainWindow
from app.logs.logger import applogger


def _add_figure(repo: SqliteRepo, name: str) -> int:
    figure_id = repo.create_figure_descriptor(name=name)
    repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="",
        x_label="",
        y_label="",
        options={},
    )
    return figure_id


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    _add_figure(built, "first")
    _add_figure(built, "second")
    yield built
    built.close()


@pytest.fixture
def window(qapp, repo: SqliteRepo, tmp_db_path: Path):
    built = MainWindow(repo=repo, db_path=tmp_db_path)
    yield built
    built.close()
    applogger.set_status_bar(None)


class _FakeDialog(QObject):
    """Stands in for a real series-operation dialog: same signals and shape,
    no UI, and an ``exec`` that optionally creates a figure like Fit's "new
    figure" destination does."""

    applied = Signal()
    results_published = Signal(str)
    creates_figure = False

    def __init__(self, *, repo: SqliteRepo, figure_id: int, parent=None) -> None:
        super().__init__(parent)
        self._repo = repo
        self._figure_id = figure_id

    def setWindowIcon(self, icon) -> None:
        pass

    def exec(self) -> bool:
        if self.creates_figure:
            _add_figure(self._repo, "operation result")
        return True


def _dialog_class(*, creates_figure: bool) -> type:
    return type(
        "FakeSeriesOperationDialog",
        (_FakeDialog,),
        {"creates_figure": creates_figure},
    )


def test_no_new_figure_skips_the_full_tab_rebuild(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common case - Preview, Apply on the same axis, Cancel - only has to
    refresh the panel it ran on, not tear down and re-render every other
    chart in the database."""
    rebuilds: list[int] = []
    monkeypatch.setattr(MainWindow, "_reload_tabs", lambda self: rebuilds.append(1))
    panel_before = window._tabs.widget(0)

    window._open_series_operation(_dialog_class(creates_figure=False))

    assert rebuilds == []
    assert window._tabs.widget(0) is panel_before


def test_a_new_figure_still_gets_a_tab(window: MainWindow) -> None:
    """The one case _reload_tabs exists for - a genuinely new figure - still
    works, with the real (not monkeypatched) rebuild."""
    figure_count_before = window._tabs.count()

    window._open_series_operation(_dialog_class(creates_figure=True))

    assert window._tabs.count() == figure_count_before + 1


def test_no_new_figure_still_reloads_the_current_panel(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipping the full rebuild must not skip refreshing the chart the
    operation actually ran on."""
    panel = window._tabs.widget(0)
    reloaded: list[int] = []
    monkeypatch.setattr(type(panel), "reload", lambda self: reloaded.append(1))

    window._open_series_operation(_dialog_class(creates_figure=False))

    assert reloaded == [1]
