"""Tests for applying a multi-result series operation.

A spectral analysis of five series produced one series on the chart.  The cause
was not in the loop that writes the results - that was always correct - but in
what the loop ended with: ``optimize_db()``, whose VACUUM cannot run inside a
transaction.  The OperationalError escaped through ``_run_operation``, which
rolled the savepoint back, so everything after the first flush to disk was
undone.

These tests pin the two halves of the fix without needing a dialog: the
repository refuses to VACUUM inside a transaction instead of raising, and a
batch of results all survive a transaction that also touches the tables they
are written to.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    repo = SqliteRepo(db_path=tmp_db_path)
    yield repo
    repo.close()


def _frame(size: int = 32) -> pd.DataFrame:
    x = np.arange(size, dtype=float)
    return pd.DataFrame({"frequency": x, "power": np.sin(x)})


# ----------------------------------------------------------------------
# optimize_db inside a transaction
# ----------------------------------------------------------------------
def test_optimize_db_does_not_raise_inside_a_transaction(repo: SqliteRepo) -> None:
    """It used to, and the exception discarded whatever operation was running."""
    repo.query_df("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
    repo._con.execute("SAVEPOINT batch")
    try:
        report = repo.optimize_db()
        assert report is not None
    finally:
        repo._con.execute("RELEASE batch")


def test_the_checks_still_run_inside_a_transaction(repo: SqliteRepo) -> None:
    """Only the compaction is skipped; the report is the point of the call."""
    repo.query_df("CREATE TABLE IF NOT EXISTS orphan (a INTEGER)")
    repo._con.execute("SAVEPOINT batch")
    try:
        report = repo.optimize_db()
        assert report.summary()
    finally:
        repo._con.execute("RELEASE batch")


def test_optimize_db_still_compacts_outside_a_transaction(repo: SqliteRepo) -> None:
    repo.query_df("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
    assert repo._con.in_transaction is False

    report = repo.optimize_db()

    assert report is not None


# ----------------------------------------------------------------------
# The batch survives
# ----------------------------------------------------------------------
def test_writing_a_table_ends_the_caller_s_transaction(repo: SqliteRepo) -> None:
    """The sharp edge underneath the bug, pinned so it is not forgotten.

    ``import_dataframe`` goes through pandas' ``to_sql``, which commits.  A
    savepoint opened around a batch of result tables therefore does not survive
    the first one, so the batch is *not* atomic however it looks in the code -
    which is why an exception later in the loop left some results behind
    instead of cleanly undoing all of them.
    """
    repo.query_df("SELECT 1")
    repo._con.execute("SAVEPOINT batch")

    repo.import_dataframe(_frame(), table_name="result_0", normalize_columns=False)

    with pytest.raises(Exception):
        repo._con.execute("RELEASE batch")


def test_every_result_of_a_batch_is_written(repo: SqliteRepo) -> None:
    """Five results in, five tables out - the spectral bug in miniature.

    The loop ends with the optimize call that used to raise; if it raised
    again, the exception would reach here.
    """
    for index in range(5):
        repo.import_dataframe(
            _frame(), table_name=f"result_{index}", normalize_columns=False
        )
    repo.optimize_db()

    tables = set(repo.list_table_names())
    assert {f"result_{index}" for index in range(5)} <= tables


def test_the_apply_loop_no_longer_optimizes(repo: SqliteRepo) -> None:
    """Guard: the compaction belongs after the commit, not inside the loop.

    Checked by reading the source because the alternative is instantiating a
    QDialog, and the point of the fix is that the loop contains no such call.
    """
    source = (
        Path(__file__).resolve().parent.parent
        / "series_operations"
        / "series_operation_dialog_base.py"
    ).read_text(encoding="utf-8")

    start = source.index("def apply_results_to_axis")
    end = source.index("def ", start + 10)
    body = source[start:end]

    assert "self._repo.optimize_db()" not in body, (
        "apply_results_to_axis optimizes inside the caller's transaction again; "
        "VACUUM cannot run there and the whole batch is rolled back"
    )


def test_the_preview_loop_no_longer_optimizes(repo: SqliteRepo) -> None:
    source = (
        Path(__file__).resolve().parent.parent
        / "series_operations"
        / "series_operation_dialog_base.py"
    ).read_text(encoding="utf-8")

    start = source.index("def preview_results_to_axis")
    end = source.index("def ", start + 10)
    body = source[start:end]

    assert "self._repo.optimize_db()" not in body


# ----------------------------------------------------------------------
# The spectral results get their own chart
# ----------------------------------------------------------------------
def test_axis_labels_can_be_updated(repo: SqliteRepo) -> None:
    """A spectrum's axes are named after what it produced, not the source."""
    figure_id = repo.create_figure_descriptor(name="spectral")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="",
        x_label="",
        y_label="",
        options={},
    )

    repo.update_axis_descriptor(
        axis_id=axis_id, title="PSD", x_label="frequency [1/x]", y_label="dB"
    )

    descriptor = repo.load_figure_descriptor(figure_id=figure_id)
    axis = descriptor.axes[0]
    assert (axis.title, axis.x_label, axis.y_label) == ("PSD", "frequency [1/x]", "dB")


def test_updating_one_label_leaves_the_others(repo: SqliteRepo) -> None:
    figure_id = repo.create_figure_descriptor(name="spectral")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="keep me",
        x_label="keep me too",
        y_label="",
        options={},
    )

    repo.update_axis_descriptor(axis_id=axis_id, y_label="coherence")

    axis = repo.load_figure_descriptor(figure_id=figure_id).axes[0]
    assert (axis.title, axis.x_label, axis.y_label) == ("keep me", "keep me too", "coherence")


def test_deleting_the_figure_removes_its_axes(repo: SqliteRepo) -> None:
    """Close without Apply has to leave no trace of the chart it built."""
    figure_id = repo.create_figure_descriptor(name="spectral")
    repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="",
        x_label="",
        y_label="",
        options={},
    )

    repo.delete_figure(figure_id)

    assert repo.load_figure_descriptor(figure_id=figure_id) is None
    assert figure_id not in [fid for fid, _name in repo.load_figures_from_db()]


def test_the_spectral_dialog_targets_its_own_axis() -> None:
    """Guard on the two halves that cannot be exercised without a widget.

    The results go to a new axis on the *current figure*, not to a new figure:
    a spectrum is a second view of this chart's data, and a separate tab would
    put it where nobody compares it with the original.
    """
    source = (
        Path(__file__).resolve().parent.parent
        / "series_operations"
        / "series_spectral_dialog.py"
    ).read_text(encoding="utf-8")

    assert "def resolve_target_axis_id" in source
    assert "self.create_result_axis(" in source
    assert "create_figure_descriptor" not in source
    # Closing without Apply must remove it again.
    assert "def discard_operation_artifacts" in source
    assert "delete_axis" in source


def test_a_result_axis_grows_the_figure_grid(repo: SqliteRepo) -> None:
    """An axis outside the grid is never drawn, so the grid has to grow."""
    source = (
        Path(__file__).resolve().parent.parent
        / "series_operations"
        / "series_operation_dialog_base.py"
    ).read_text(encoding="utf-8")
    start = source.index("def create_result_axis")
    body = source[start : source.index("\n    def ", start + 10)]

    assert "next_axis_index" in body
    assert "set_figure_grid" in body


def test_the_statistics_dialog_charts_what_it_measured() -> None:
    """Each model offers the renderers that show what that model is about.

    Asserted against the table rather than the source text, so adding a chart
    to a model does not break this - only removing the guarantee does: every
    model that charts at all must name a renderer that exists, and "all" must
    stay empty because no single picture illustrates every section at once.
    """
    from app.scanners.axis_renderer_scanner import renderers
    from app.series_operations.series_statistics_dialog import SeriesStatisticsDialog

    source = (
        Path(__file__).resolve().parent.parent
        / "series_operations"
        / "series_statistics_dialog.py"
    ).read_text(encoding="utf-8")

    known = {entry["value"] for entry in renderers}
    charts = SeriesStatisticsDialog.CHARTS_BY_MODEL

    assert charts["all"] == (), "the combined model must not draw a chart"
    for model, entries in charts.items():
        for chart_type, title in entries:
            assert chart_type in known, f"{model} names a renderer that does not exist"
            assert title.strip(), f"{model} has an untitled axis"

    # The two models that compare samples chart one against the other.
    assert charts["correlation"][0][0] == "Scatter Plot"
    assert charts["paired"][0][0] == "Scatter Plot"
    assert "_attach_series_pair" in source
    # Preview draws it too, like every other operation dialog; Close removes
    # it again when Apply never happened.
    start = source.index("def _run_statistics")
    body = source[start : source.index("\n    def ", start + 10)]
    assert "_create_model_axis" in body
    assert "create_chart_check.isChecked()" in body
    assert "def discard_operation_artifacts" in source
    assert "_remove_result_axis" in source
