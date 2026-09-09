"""The Statistics dialog measures the series that are checked. Only those.

It used to substitute *every visible series on the axis* whenever exactly
one row was checked and the model was one that needs pairs - reasoning
that the dialog is often opened from a single active series while the
requested test needs two. The visible result was that unchecking two of
three series changed nothing: all three were still measured, still
reported, and the checkbox that had just been cleared was contradicted on
screen.

A guess about intent does not get to overrule an explicit action. The
one-series case has an honest answer already - the report says the paired
and correlation sections need a second series - and that is a thing the
user can act on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.statistics_dialog import SeriesStatisticsDialog

NAMES = ("Neomycin", "Penicillin", "Streptomycin")


@pytest.fixture
def dialog(qapp, repo: SqliteRepo) -> SeriesStatisticsDialog:
    """Three series on one axis, as in the report that prompted this."""
    rng = np.random.default_rng(1)
    repo.import_dataframe(
        pd.DataFrame(
            {
                "b": [f"b{index}" for index in range(16)],
                "neo": rng.normal(4.0, 2.0, 16),
                "pen": rng.normal(200.0, 50.0, 16),
                "strep": rng.normal(3.0, 1.0, 16),
            }
        ),
        table_name="ab",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="A", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="potency", x_label="b", y_label="v", options={},
        )
    )
    for index, (name, column) in enumerate(zip(NAMES, ("neo", "pen", "strep"))):
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=index, name=name,
            sql_query=f"SELECT b AS x, {column} AS y FROM ab",
            roles={"x": "x", "y": "y"}, style={},
        )
    return SeriesStatisticsDialog(repo=repo, figure_id=figure_id)


def _check_only(dialog: SeriesStatisticsDialog, wanted: str) -> None:
    series_list = dialog.series_selector.series_list
    for index in range(series_list.count()):
        item = series_list.item(index)
        item.setCheckState(
            Qt.CheckState.Checked
            if item.text() == wanted
            else Qt.CheckState.Unchecked
        )


def _use_model(dialog: SeriesStatisticsDialog, model: str) -> None:
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData(model))


def test_every_series_is_measured_when_every_one_is_checked(
    dialog: SeriesStatisticsDialog,
) -> None:
    assert [sample.name for sample in dialog._selected_samples()] == list(NAMES)


@pytest.mark.parametrize("model", ["all", "paired", "correlation", "descriptive"])
def test_only_the_checked_series_is_measured(
    dialog: SeriesStatisticsDialog, model: str
) -> None:
    """Whatever the model: the three that used to substitute the axis are
    exactly the three parametrised here alongside one that never did."""
    _check_only(dialog, "Streptomycin")
    _use_model(dialog, model)

    assert [sample.name for sample in dialog._selected_samples()] == ["Streptomycin"]


def test_the_report_names_only_the_checked_series(
    dialog: SeriesStatisticsDialog,
) -> None:
    """What was actually seen: a report of three series with one ticked."""
    _check_only(dialog, "Streptomycin")
    _use_model(dialog, "all")

    report = dialog.format_results(dialog.compute_results())

    assert "Streptomycin" in report
    assert "Neomycin" not in report
    assert "Penicillin" not in report


def test_one_checked_series_says_why_the_paired_sections_are_missing(
    dialog: SeriesStatisticsDialog,
) -> None:
    """The honest answer to "this test needs two samples and has one"."""
    _check_only(dialog, "Streptomycin")
    _use_model(dialog, "all")

    report = dialog.format_results(dialog.compute_results())

    assert "need at least two" in report


def test_two_checked_series_get_their_paired_tests(
    dialog: SeriesStatisticsDialog,
) -> None:
    """The case the substitution existed to serve, reached by checking a
    second box rather than by ignoring the first."""
    series_list = dialog.series_selector.series_list
    for index in range(series_list.count()):
        item = series_list.item(index)
        item.setCheckState(
            Qt.CheckState.Checked
            if item.text() in ("Neomycin", "Streptomycin")
            else Qt.CheckState.Unchecked
        )
    _use_model(dialog, "paired")

    report = dialog.format_results(dialog.compute_results())

    assert "Neomycin vs Streptomycin" in report
    assert "Penicillin" not in report


def test_nothing_checked_is_an_error_rather_than_everything(
    dialog: SeriesStatisticsDialog,
) -> None:
    series_list = dialog.series_selector.series_list
    for index in range(series_list.count()):
        series_list.item(index).setCheckState(Qt.CheckState.Unchecked)

    with pytest.raises(ValueError, match="at least one source series"):
        dialog.compute_results()
