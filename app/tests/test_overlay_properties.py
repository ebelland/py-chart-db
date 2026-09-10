"""The Overlay properties panel: annotations moved here, lines added.

Two things are worth pinning beyond "the table has rows". The first is the
*move*: a figure annotated before this panel existed has to load into it
unchanged, and the Axis properties panel must not clear those annotations
now that it no longer edits them - which it would, if it still sent an
empty list in its payload.

The second is the new key. A reference line is stored on the axis and drawn
with axhline/axvline, so it belongs to the axes rather than to any series:
it stays where it was put when the data changes underneath it, which is the
whole reason not to draw it as a two-point series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base import SeriesData
from app.charts.render_figure import render_figure_from_descriptor
from app.charts.scatter import ScatterAxisRenderer
from app.data.sqlite_repo import SqliteRepo
from app.widgets.overlay_properties import OverlayPropertiesWidget

ANNOTATION = {
    "x": 3.0,
    "y": 1.0,
    "type": "arrow",
    "text": "peak",
    "kwargs": {"xytext": [10, 10]},
}
LINE = {
    "orientation": "horizontal",
    "value": 0.5,
    "kwargs": {"color": "red", "linestyle": "--"},
}
#: What LINE round-trips to once its colour has been through the colour
#: combo, which normalises every named colour to its canonical hex - the
#: same normalisation the Series colour combo already does.
LINE_ROUND_TRIPPED = {
    "orientation": "horizontal",
    "value": 0.5,
    "kwargs": {"color": "#ff0000", "linestyle": "--"},
}


@pytest.fixture
def figure_with_axis(repo: SqliteRepo):
    """A one-axis figure carrying one annotation and one line."""
    repo.import_dataframe(
        pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)}),
        table_name="w",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id,
            axis_index=0,
            chart_type="Scatter Plot",
            title="wave",
            x_label="x",
            y_label="y",
            options={"annotations": [dict(ANNOTATION)], "lines": [dict(LINE)]},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="s",
        sql_query="SELECT x, y FROM w",
        roles={"x": "x", "y": "y"},
        style={"marker": "o"},
    )
    return figure_id, axis_id


@pytest.fixture
def widget(qapp, repo: SqliteRepo, figure_with_axis) -> OverlayPropertiesWidget:
    figure_id, axis_id = figure_with_axis
    built = OverlayPropertiesWidget()
    built.set_connected_figure(repo, figure_id, Figure())
    built.set_axis(axis_id)
    return built


# ----------------------------------------------------------------------
# The panel
# ----------------------------------------------------------------------
def test_it_hosts_one_tab_per_kind_of_overlay(
    widget: OverlayPropertiesWidget,
) -> None:
    titles = [widget._tabs.tabText(i) for i in range(widget._tabs.count())]

    assert titles == ["Annotations", "Lines"]


def test_it_names_the_axis_it_is_editing(widget: OverlayPropertiesWidget) -> None:
    """There is no axis selector here - the Axis properties panel owns that -
    so the panel has to say which axis it is pointed at."""
    assert widget._axis_label.text() == "wave"


def test_it_follows_the_axis_it_is_told_to(
    qapp, repo: SqliteRepo, figure_with_axis
) -> None:
    figure_id, axis_id = figure_with_axis
    second = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=1, chart_type="Scatter Plot",
            title="second", x_label="x", y_label="y", options={},
        )
    )
    widget = OverlayPropertiesWidget()
    widget.set_connected_figure(repo, figure_id, Figure())

    widget.set_axis(axis_id)
    assert widget._annotations_table.rowCount() == 1

    widget.set_axis(second)
    assert widget._axis_label.text() == "second"
    assert widget._annotations_table.rowCount() == 0
    assert widget._lines_table.rowCount() == 0


def test_nothing_is_editable_until_an_axis_is_selected(
    qapp, repo: SqliteRepo, figure_with_axis
) -> None:
    figure_id, _axis_id = figure_with_axis
    widget = OverlayPropertiesWidget()
    widget.set_connected_figure(repo, figure_id, Figure())

    assert not widget._annotations_table.isEnabled()
    assert not widget._btn_add_line.isEnabled()
    assert widget._axis_label.text() == "No axis selected"


# ----------------------------------------------------------------------
# Annotations: the move
# ----------------------------------------------------------------------
def test_an_annotation_written_before_this_panel_existed_loads_into_it(
    widget: OverlayPropertiesWidget,
) -> None:
    table = widget._annotations_table

    assert table.rowCount() == 1
    assert table.item(0, 0).text() == "3.0"
    assert table.item(0, 3).text() == "peak"
    assert table.cellWidget(0, 2).currentData() == "arrow"
    assert "xytext" in table.item(0, 7).text()


def test_the_annotations_round_trip_through_the_payload(
    widget: OverlayPropertiesWidget,
) -> None:
    sent: list[dict] = []
    widget.overlay_options_requested.connect(sent.append)

    widget._emit_overlay_options_requested()

    assert sent[0]["annotations"] == [ANNOTATION]


def test_the_axis_panel_no_longer_sends_annotations(
    qapp, repo: SqliteRepo, figure_with_axis
) -> None:
    """It stopped editing them, so it must stop sending them: the window
    merges the payload, and an empty list would erase what this panel just
    wrote."""
    from app.widgets.axis_properties import AxisPropertiesWidget

    figure_id, _axis_id = figure_with_axis
    axis_panel = AxisPropertiesWidget()
    axis_panel.set_connected_figure(repo, figure_id, Figure())

    sent: list[dict] = []
    axis_panel.axis_options_requested.connect(sent.append)
    axis_panel._emit_axis_options_requested()

    assert "annotations" not in sent[0]


# ----------------------------------------------------------------------
# Lines: the new key
# ----------------------------------------------------------------------
def test_a_stored_line_loads_into_the_table(
    widget: OverlayPropertiesWidget,
) -> None:
    table = widget._lines_table

    assert table.rowCount() == 1
    assert table.cellWidget(0, 0).currentData() == "horizontal"
    assert table.item(0, 1).text() == "0.5"
    assert table.cellWidget(0, 2).current_hex() == "#ff0000"
    assert table.cellWidget(0, 3).current_linestyle() == "--"


def test_a_new_line_reaches_the_payload(widget: OverlayPropertiesWidget) -> None:
    sent: list[dict] = []
    widget.overlay_options_requested.connect(sent.append)

    widget._add_line_row({"orientation": "vertical", "value": 7.0, "kwargs": {}})
    widget._emit_overlay_options_requested()

    assert sent[0]["lines"] == [
        LINE_ROUND_TRIPPED,
        {"orientation": "vertical", "value": 7.0, "kwargs": {}},
    ]


def test_a_row_with_no_usable_value_is_skipped_not_stored(
    widget: OverlayPropertiesWidget,
) -> None:
    """A half-typed row should cost itself, not the rows around it."""
    widget._add_line_row({"orientation": "vertical", "value": "", "kwargs": {}})
    sent: list[dict] = []
    widget.overlay_options_requested.connect(sent.append)

    widget._emit_overlay_options_requested()

    assert sent[0]["lines"] == [LINE_ROUND_TRIPPED]


def test_invalid_kwargs_json_costs_the_kwargs_not_the_line(
    widget: OverlayPropertiesWidget,
) -> None:
    widget._add_line_row({"orientation": "vertical", "value": 2.0, "kwargs": {}})
    widget._lines_table.setItem(1, 4, widget._item("{not json"))
    sent: list[dict] = []
    widget.overlay_options_requested.connect(sent.append)

    widget._emit_overlay_options_requested()

    assert sent[0]["lines"][1] == {
        "orientation": "vertical",
        "value": 2.0,
        "kwargs": {},
    }


# ----------------------------------------------------------------------
# Visual editors: colour, line style, font, size
# ----------------------------------------------------------------------
def test_there_is_no_apply_button_every_edit_auto_applies(
    widget: OverlayPropertiesWidget,
) -> None:
    assert not hasattr(widget, "_btn_apply")


def test_changing_a_cell_widget_queues_an_auto_apply(
    widget: OverlayPropertiesWidget,
) -> None:
    assert not widget._auto_apply_timer.isActive()

    widget._lines_table.cellWidget(0, 3).set_current_linestyle(":")

    assert widget._auto_apply_timer.isActive()


def test_a_stored_line_splits_colour_and_style_out_of_the_json_cell(
    widget: OverlayPropertiesWidget,
) -> None:
    """The JSON column shows only what the dedicated widgets do not cover."""
    assert widget._lines_table.item(0, 4).text() == ""


def test_the_line_widgets_write_back_into_kwargs(
    widget: OverlayPropertiesWidget,
) -> None:
    widget._add_line_row({"orientation": "vertical", "value": 4.0, "kwargs": {}})
    widget._lines_table.cellWidget(1, 2).set_current_name("blue")
    widget._lines_table.cellWidget(1, 3).set_current_linestyle(":")
    sent: list[dict] = []
    widget.overlay_options_requested.connect(sent.append)

    widget._emit_overlay_options_requested()

    assert sent[0]["lines"][1]["kwargs"] == {"color": "#0000ff", "linestyle": ":"}


def test_line_json_kwargs_still_reach_the_payload_alongside_the_widgets(
    widget: OverlayPropertiesWidget,
) -> None:
    widget._add_line_row(
        {"orientation": "vertical", "value": 4.0, "kwargs": {"linewidth": 3}}
    )
    widget._lines_table.cellWidget(1, 2).set_current_name("green")

    sent: list[dict] = []
    widget.overlay_options_requested.connect(sent.append)
    widget._emit_overlay_options_requested()

    assert sent[0]["lines"][1]["kwargs"] == {"linewidth": 3, "color": "#008000"}


def test_a_stored_annotation_splits_colour_font_and_size_into_widgets(
    qapp, repo: SqliteRepo, figure_with_axis
) -> None:
    figure_id, axis_id = figure_with_axis
    options = repo.get_axis_options(axis_id) or {}
    options["annotations"] = [
        {
            "x": 1.0,
            "y": 2.0,
            "type": "text",
            "text": "hi",
            "kwargs": {"color": "red", "fontfamily": "monospace",
                       "fontsize": 13, "rotation": 45},
        }
    ]
    repo.set_axis_options(axis_id, options)

    widget = OverlayPropertiesWidget()
    widget.set_connected_figure(repo, figure_id, Figure())
    widget.set_axis(axis_id)

    table = widget._annotations_table
    assert table.cellWidget(0, 4).current_hex() == "#ff0000"
    assert table.cellWidget(0, 5).currentData() == "monospace"
    assert table.cellWidget(0, 6).value() == 13
    # Only the leftover kwarg stays in the JSON cell.
    assert "rotation" in table.item(0, 7).text()
    assert "fontsize" not in table.item(0, 7).text()

    sent: list[dict] = []
    widget.overlay_options_requested.connect(sent.append)
    widget._emit_overlay_options_requested()

    assert sent[0]["annotations"][0]["kwargs"] == {
        "rotation": 45,
        "color": "#ff0000",
        "fontfamily": "monospace",
        "fontsize": 13,
    }


def test_an_arbitrary_hex_colour_stays_in_the_json_cell(
    qapp, repo: SqliteRepo, figure_with_axis
) -> None:
    """The colour combo only knows Matplotlib's named colours, so a hex it
    cannot show must keep being edited as JSON rather than disappear."""
    figure_id, axis_id = figure_with_axis
    options = repo.get_axis_options(axis_id) or {}
    options["lines"] = [
        {"orientation": "vertical", "value": 1.0, "kwargs": {"color": "#123456"}}
    ]
    repo.set_axis_options(axis_id, options)

    widget = OverlayPropertiesWidget()
    widget.set_connected_figure(repo, figure_id, Figure())
    widget.set_axis(axis_id)

    assert widget._lines_table.cellWidget(0, 2).current_hex() == ""
    assert "#123456" in widget._lines_table.item(0, 4).text()

    sent: list[dict] = []
    widget.overlay_options_requested.connect(sent.append)
    widget._emit_overlay_options_requested()
    assert sent[0]["lines"][0]["kwargs"] == {"color": "#123456"}


def test_a_non_numeric_font_size_is_left_in_the_json_cell(
    qapp, repo: SqliteRepo, figure_with_axis
) -> None:
    """Matplotlib accepts fontsize="small"; the size spin cannot, so that
    value must stay where the JSON column can still edit it."""
    figure_id, axis_id = figure_with_axis
    options = repo.get_axis_options(axis_id) or {}
    options["annotations"] = [
        {"x": 0.0, "y": 0.0, "type": "text", "text": "t",
         "kwargs": {"fontsize": "small"}}
    ]
    repo.set_axis_options(axis_id, options)

    widget = OverlayPropertiesWidget()
    widget.set_connected_figure(repo, figure_id, Figure())
    widget.set_axis(axis_id)

    table = widget._annotations_table
    assert table.cellWidget(0, 6).value() == 0
    assert "small" in table.item(0, 7).text()


# ----------------------------------------------------------------------
# What the renderer does with them
# ----------------------------------------------------------------------
def _drawn(options: dict) -> tuple:
    figure = Figure()
    axes = figure.add_subplot(1, 1, 1)
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 4.0]})
    ScatterAxisRenderer().render_axis(
        axes, [SeriesData(name="s", df=frame, style={"marker": "o"})], options
    )
    return figure, axes


def test_a_line_is_drawn_across_the_whole_axes() -> None:
    """axhline, not a two-point series: the line spans the axes and does not
    move when the data underneath it changes."""
    _figure, axes = _drawn({"lines": [dict(LINE)]})

    assert len(axes.lines) == 1
    drawn = axes.lines[0]
    assert drawn.get_color() == "red"
    assert drawn.get_linestyle() == "--"
    assert list(drawn.get_ydata()) == [0.5, 0.5]


def test_a_vertical_line_is_the_other_orientation() -> None:
    _figure, axes = _drawn(
        {"lines": [{"orientation": "vertical", "value": 1.5, "kwargs": {}}]}
    )

    assert list(axes.lines[0].get_xdata()) == [1.5, 1.5]


def test_a_labelled_line_appears_in_the_legend_and_an_unlabelled_one_does_not() -> None:
    _figure, axes = _drawn(
        {
            "lines": [
                {"orientation": "vertical", "value": 1.0, "kwargs": {"label": "limit"}},
                {"orientation": "vertical", "value": 1.5, "kwargs": {}},
            ]
        }
    )

    _handles, labels = axes.get_legend_handles_labels()
    assert "limit" in labels
    assert len(labels) == 2, "the unlabelled line should not be listed"


def test_a_line_with_an_unusable_value_costs_only_itself() -> None:
    _figure, axes = _drawn(
        {
            "lines": [
                {"orientation": "vertical", "value": "nonsense"},
                {"orientation": "vertical", "value": 1.0},
            ]
        }
    )

    assert len(axes.lines) == 1


def test_a_line_with_an_unusable_kwarg_costs_only_itself() -> None:
    _figure, axes = _drawn(
        {
            "lines": [
                {"orientation": "vertical", "value": 1.0, "kwargs": {"nope": 1}},
                {"orientation": "vertical", "value": 2.0},
            ]
        }
    )

    assert len(axes.lines) == 1


def test_every_renderer_draws_lines_because_annotations_do(
    repo: SqliteRepo, figure_with_axis
) -> None:
    """They are drawn from apply_annotations, which every renderer already
    calls - rather than from a second call that would have to be added to
    twenty-nine renderers and forgotten in one."""
    figure_id, _axis_id = figure_with_axis
    figure = Figure()

    render_figure_from_descriptor(
        figure=figure, descriptor=repo.load_figure_descriptor(figure_id), repo=repo
    )

    axes = figure.axes[0]
    assert [text.get_text() for text in axes.texts] == ["peak"]
    assert any(list(line.get_ydata()) == [0.5, 0.5] for line in axes.lines)


# ----------------------------------------------------------------------
# Persistence, through the window that writes it
# ----------------------------------------------------------------------
def test_the_window_writes_both_keys_and_leaves_the_rest_of_the_axis_alone(
    qapp, repo: SqliteRepo, figure_with_axis, tmp_db_path
) -> None:
    from app.dialogs.main_window import MainWindow
    from app.logs.logger import applogger

    figure_id, axis_id = figure_with_axis
    options = repo.get_axis_options(axis_id) or {}
    options["x_scale"] = "log"
    repo.set_axis_options(axis_id, options)

    window = MainWindow(repo=repo, db_path=tmp_db_path)
    try:
        window._properties_figure_id = figure_id
        window._on_overlay_options_requested(
            {
                "axis_id": axis_id,
                "annotations": [],
                "lines": [{"orientation": "vertical", "value": 3.0, "kwargs": {}}],
            }
        )
    finally:
        window.close()
        applogger.set_status_bar(None)

    stored = repo.get_axis_options(axis_id) or {}
    assert stored["lines"] == [
        {"orientation": "vertical", "value": 3.0, "kwargs": {}}
    ]
    # Deleting the last annotation removes the key rather than storing [].
    assert "annotations" not in stored
    assert stored["x_scale"] == "log", "the rest of the axis was rewritten"
