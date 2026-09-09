"""The Axis properties tab strip, after the long options page was split.

One "Axis options" page held every axis setting there is: four labels, a
projection, sharing, a span, three scales with three bases, a symlog
threshold, three inverts, eight grid/tick combos, a tick length, a rotation,
four spines, a limits mode and three ranges - thirty-odd controls that only
a scroll bar divided into sections. Now: Labels, Scale, Ticks.

Pick radius is gone with it. The Kwargs page has offered ``picker`` - the
same distance, on the artists that are actually clicked - since the schema
grew ARTIST_KWARGS, and of two fields for one idea the one that loses is
whichever the user did not think to set.
"""
from __future__ import annotations

import pytest
from matplotlib.figure import Figure

from app.data.sqlite_repo import SqliteRepo
from app.widgets.axis_properties import AxisPropertiesWidget


@pytest.fixture
def widget(qapp, repo: SqliteRepo) -> AxisPropertiesWidget:
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="ax",
        x_label="x",
        y_label="y",
        options={},
    )
    built = AxisPropertiesWidget()
    built.set_connected_figure(repo, figure_id, Figure())
    return built


def _tab_titles(widget: AxisPropertiesWidget) -> list[str]:
    return [widget._tabs.tabText(i) for i in range(widget._tabs.count())]


def test_the_options_page_is_split_by_what_it_sets(
    widget: AxisPropertiesWidget,
) -> None:
    assert _tab_titles(widget) == [
        "Labels",
        "Scale",
        "Ticks",
        "Kwargs",
        "Annotations",
    ]


def test_the_kwargs_index_constant_points_at_the_kwargs_page(
    widget: AxisPropertiesWidget,
) -> None:
    """Callers reach that page by index; the constant is what they use."""
    assert (
        _tab_titles(widget)[AxisPropertiesWidget.KWARGS_TAB_INDEX] == "Kwargs"
    )


def test_every_control_still_exists_somewhere(
    widget: AxisPropertiesWidget,
) -> None:
    """Splitting a page is a move, not a cull - Pick radius aside."""
    for name in (
        "_axis_label_edit", "_x_label_edit", "_y_label_edit", "_z_label_edit",
        "_projection_combo", "_sharex_check", "_sharey_check",
        "_hide_axis_check", "_row_span_spin", "_col_span_spin",
        "_x_scale_combo", "_y_scale_combo", "_z_scale_combo",
        "_x_scale_base_spin", "_linthresh_spin", "_invert_x_check",
        "_limits_mode_combo", "_tick_length_spin", "_x_tick_rotation_spin",
        "_hide_spine_top_check",
    ):
        assert hasattr(widget, name), f"{name} was lost in the split"
    assert len(widget._grid_combos) == 6
    assert len(widget._tick_combos) == 6
    assert len(widget._limit_spins) == 6


def test_pick_radius_is_gone_from_the_form_and_the_payload(
    widget: AxisPropertiesWidget,
) -> None:
    assert not hasattr(widget, "_pickradius_spin")

    sent: list[dict] = []
    widget.axis_options_requested.connect(sent.append)
    widget._emit_axis_options_requested()

    assert sent, "no payload was emitted"
    assert "pickradius" not in sent[0]


def test_an_axis_that_already_has_a_pickradius_keeps_it(
    qapp, repo: SqliteRepo, tmp_db_path
) -> None:
    """Through the window that actually writes it: the payload is *merged*
    into the stored options, so a key the panel no longer sends is left
    alone rather than cleared out of every axis that had one."""
    from app.dialogs.main_window import MainWindow
    from app.logs.logger import applogger

    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id, axis_index=0, chart_type="Scatter Plot",
            title="ax", x_label="x", y_label="y", options={"pickradius": 4.0},
        )
    )

    window = MainWindow(repo=repo, db_path=tmp_db_path)
    try:
        window._properties_figure_id = figure_id
        window._on_axis_options_requested({"axis_id": axis_id, "title": "renamed"})
    finally:
        window.close()
        applogger.set_status_bar(None)

    stored = repo.get_axis_options(axis_id) or {}
    assert stored["title"] == "renamed"
    assert stored["pickradius"] == 4.0


def test_the_grid_combos_are_given_room_to_show_their_labels(
    widget: AxisPropertiesWidget,
) -> None:
    """They used to sit three-across next to a form label column, each at its
    80px minimum, showing "Fr" of "From the style"."""
    from app.charts import axis_options

    combo = widget._grid_combos[("x", "major")]
    assert combo.itemText(0) == "Default"
    assert axis_options.GRID_CHOICES[0][0] == axis_options.AUTO
    assert axis_options.TICK_CHOICES[0][1] == "Default"
