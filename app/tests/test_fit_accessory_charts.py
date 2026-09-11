"""The Fit dialog's optional residual / measured-vs-fit charts.

The first version of this feature created its axes from inside
``preview_results_to_axis`` - that is, inside the preview savepoint - and it
deleted the user's data.

Writing a result table goes through pandas' ``to_sql``, which commits, and a
commit invalidates the open SAVEPOINT. Every repository write after that has
its own commit suppressed (``SqliteRepo._commit`` skips while a preview
savepoint name is set) and so accumulates in an implicit transaction that
Close/Cancel then discards - taking with it the series the user already had
on *other* axes, which nothing in the fit dialog ever touched.

``_run_operation`` calls ``resolve_target_axis_id`` before
``begin_preview_transaction()``, which is why the spectral and statistics
dialogs both create their result axis there. This one now does too, and the
first test below is the one that fails if it ever moves back.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.fit_dialog import SeriesFitDialog


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    x = np.linspace(0.0, 10.0, 40)
    built.import_dataframe(
        pd.DataFrame({"x": x, "y": 2.0 * x + 1.0}),
        table_name="src",
        normalize_columns=False,
    )
    yield built
    built.close()


@pytest.fixture
def figure_id(repo: SqliteRepo) -> int:
    """A figure with the fit's source series, plus a second axis of its own.

    The second axis is the point: it is data the fit dialog has no business
    touching, and it is what the savepoint bug destroyed.
    """
    figure_id = repo.create_figure_descriptor(name="fig")
    source_axis = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="main",
        x_label="",
        y_label="",
        options={},
    )
    repo.create_series_descriptor(
        axis_id=source_axis,
        series_index=0,
        name="DATA",
        sql_query='SELECT x AS x, y AS y FROM src',
        roles={"x": "x", "y": "y"},
        style={},
    )
    other_axis = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=1,
        chart_type="Scatter Plot",
        title="untouched",
        x_label="",
        y_label="",
        options={},
    )
    repo.create_series_descriptor(
        axis_id=other_axis,
        series_index=0,
        name="UNRELATED",
        sql_query='SELECT x AS x, y AS y FROM src',
        roles={"x": "x", "y": "y"},
        style={},
    )
    return figure_id


@pytest.fixture
def dialog(qapp, repo: SqliteRepo, figure_id: int) -> SeriesFitDialog:
    built = SeriesFitDialog(repo=repo, figure_id=figure_id)
    built.series_selector.select_all_series()
    built._select_first_model()
    # Both boxes are restored from config.json like every other entry in this
    # dialog (see SeriesOperationDialogBase.restore_dialog_state), so whatever
    # the last real run left checked would otherwise decide what these tests
    # see. Each test says what it wants.
    built._residual_chart_check.setChecked(False)
    built._fit_vs_measured_chart_check.setChecked(False)
    return built


def _series_names(repo: SqliteRepo, figure_id: int) -> dict[str, list[str]]:
    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    return {
        str(axis.title): [str(series.name) for series in axis.series]
        for axis in descriptor.axes
    }


def _axis_titles(repo: SqliteRepo, figure_id: int) -> list[str]:
    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    return [str(axis.title) for axis in descriptor.axes]


# ----------------------------------------------------------------------
# The data-loss regression
# ----------------------------------------------------------------------
def test_cancelling_a_preview_keeps_every_series_the_user_had(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    """The bug, exactly: both real series vanished from the figure.

    Not only the fit's own source axis - the unrelated axis lost its series
    too, which is what made this data loss rather than an untidy preview.
    """
    dialog._residual_chart_check.setChecked(True)
    dialog._fit_vs_measured_chart_check.setChecked(True)

    dialog.preview()
    dialog.cancel_operation_changes()

    names = _series_names(repo, figure_id)
    assert "DATA" in names["main"]
    assert "UNRELATED" in names["untouched"]


def test_the_accessory_axes_are_built_before_the_savepoint_opens(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    """resolve_target_axis_id is the safe hook; preview_results_to_axis is not.

    Asserted on the class rather than by observing a run: the distinction is
    *which hook* does the work, and a version that got the timing wrong could
    still look right in a single happy-path preview.
    """
    assert "_sync_accessory_charts" in SeriesFitDialog.resolve_target_axis_id.__code__.co_names
    assert "preview_results_to_axis" not in vars(SeriesFitDialog)


def test_the_fit_still_lands_on_its_own_source_axis(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    """The overlay belongs with the data it was computed from."""
    dialog.apply()

    assert any(
        name.startswith("Fit:") for name in _series_names(repo, figure_id)["main"]
    )


# ----------------------------------------------------------------------
# What the checkboxes do
# ----------------------------------------------------------------------
def test_no_checkbox_adds_no_axis(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    dialog.preview()

    assert _axis_titles(repo, figure_id) == ["main", "untouched"]


def test_each_checkbox_adds_its_own_axis(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    dialog._residual_chart_check.setChecked(True)
    dialog._fit_vs_measured_chart_check.setChecked(True)

    dialog.preview()

    titles = _axis_titles(repo, figure_id)
    assert "Residuals" in titles
    assert "Measured vs. fit" in titles


def test_previewing_repeatedly_does_not_stack_axes(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    """Each accessory axis is created once and reused."""
    dialog._residual_chart_check.setChecked(True)

    dialog.preview()
    dialog.preview()
    dialog.preview()

    assert _axis_titles(repo, figure_id).count("Residuals") == 1


def test_unchecking_removes_that_axis_and_leaves_the_other(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    dialog._residual_chart_check.setChecked(True)
    dialog._fit_vs_measured_chart_check.setChecked(True)
    dialog.preview()

    dialog._fit_vs_measured_chart_check.setChecked(False)
    dialog.preview()

    titles = _axis_titles(repo, figure_id)
    assert "Residuals" in titles
    assert "Measured vs. fit" not in titles


def test_closing_without_apply_removes_the_accessory_axes(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    dialog._residual_chart_check.setChecked(True)
    dialog._fit_vs_measured_chart_check.setChecked(True)
    dialog.preview()

    dialog.reject()

    assert _axis_titles(repo, figure_id) == ["main", "untouched"]


def test_applying_keeps_them(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    dialog._residual_chart_check.setChecked(True)
    dialog._fit_vs_measured_chart_check.setChecked(True)

    dialog.apply()

    titles = _axis_titles(repo, figure_id)
    assert "Residuals" in titles
    assert "Measured vs. fit" in titles


def test_the_accessory_charts_read_the_fit_output_table(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    """Residual and measured-vs-fit both come off columns the fit already
    writes, so neither recomputes anything."""
    dialog._residual_chart_check.setChecked(True)
    dialog._fit_vs_measured_chart_check.setChecked(True)
    dialog.apply()

    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    queries = {
        str(axis.title): str(axis.series[0].sql_query)
        for axis in descriptor.axes
        if str(axis.title) in ("Residuals", "Measured vs. fit") and axis.series
    }

    assert "residual" in queries["Residuals"]
    assert "y_fit" in queries["Measured vs. fit"]


# ----------------------------------------------------------------------
# Selecting more than one source series
# ----------------------------------------------------------------------
#
# Fit optimises one parametric model against one series; there is no sense
# in which two unrelated series share one set of parameters, so
# _selected_series_row (dialog_base.py) picks the first selected series and
# ignores the rest. That used to be a debug-log line that named neither how
# many series nor which one won, so the accessory charts silently described
# whichever series happened to be first, with nothing on screen explaining
# why reordering the series changed what a residual chart showed.
#
# The warning is deliberately NOT show_dialog=True (a nearby, tempting fix):
# this runs on every compute_results() - every Preview, every parameter
# tweak - and a modal QMessageBox.exec() on each of those would freeze the
# dialog until dismissed, again and again; worse, AppLogger._show_message_box
# only skips the dialog when there is no QApplication at all, so it
# exec()s and quietly returns under the offscreen QPA platform pytest runs
# on, passing green there while genuinely hanging the first real desktop
# run that reaches this path. The status bar already surfaces every
# WARNING-or-louder message on its own (AppLogger._log_with_policy), which
# is why naming the series and the count is the whole fix.
def test_selecting_two_series_names_which_one_is_used(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.series_operations.dialog_base as module

    repo.create_series_descriptor(
        axis_id=repo.load_figure_descriptor(figure_id=figure_id).axes[0].id,
        series_index=1,
        name="SECOND",
        sql_query="SELECT x AS x, y AS y FROM src",
        roles={"x": "x", "y": "y"},
        style={},
    )
    dialog.series_selector.reload(select_all_series=True)

    seen: list[tuple[str, bool | None]] = []
    monkeypatch.setattr(
        module.applogger,
        "warning",
        lambda msg, *args, show_dialog=None, **kwargs: seen.append(
            (str(msg) % args if args else str(msg), show_dialog)
        ),
    )

    dialog._selected_series_row()

    assert seen, "selecting two series raised no warning at all"
    message, show_dialog = seen[-1]
    assert show_dialog is None, "must stay at the default - see the note above"
    assert "DATA" in message or "SECOND" in message
    assert "2" in message


def test_only_the_first_selected_series_gets_a_residual_chart(
    dialog: SeriesFitDialog, repo: SqliteRepo, figure_id: int
) -> None:
    """Pinning today's behaviour, not endorsing it: the accessory charts
    describe exactly one series - whichever _selected_series_row picked -
    never a silent mix of two."""
    repo.create_series_descriptor(
        axis_id=repo.load_figure_descriptor(figure_id=figure_id).axes[0].id,
        series_index=1,
        name="SECOND",
        sql_query="SELECT x AS x, y AS y FROM src",
        roles={"x": "x", "y": "y"},
        style={},
    )
    dialog.series_selector.reload(select_all_series=True)
    dialog._select_first_model()
    dialog._residual_chart_check.setChecked(True)

    dialog.apply()

    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    residual_axes = [axis for axis in descriptor.axes if str(axis.title) == "Residuals"]
    assert len(residual_axes) == 1
    assert len(residual_axes[0].series) == 1
