"""The layout-preset picker in Figure properties.

It briefly lost its whole combo when the "Apply layout" button next to it
was reported as not working and removed - nothing here tested the picker,
so nothing objected. The button is gone for real now; the presets are not,
and picking one *is* the apply.

Why picking applies immediately rather than riding on the panel's Apply
button: the combo has no stored value to reload (a preset is a one-shot
rearrangement, not a figure property), so it always comes back showing its
placeholder - an Apply pressed for some unrelated edit would otherwise
re-run whatever preset happened to still be selected, and "Grid" flattens
a hand-built layout back to a uniform one.
"""
from __future__ import annotations

import pytest
from matplotlib.figure import Figure

from app.charts import layout_presets
from app.data.sqlite_repo import SqliteRepo
from app.widgets.figure_properties import FigurePropertiesWidget


@pytest.fixture
def widget(qapp, repo: SqliteRepo) -> FigurePropertiesWidget:
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    built = FigurePropertiesWidget()
    built.set_connected_figure(repo, figure_id, Figure())
    return built


def test_every_preset_is_offered_behind_a_placeholder(
    widget: FigurePropertiesWidget,
) -> None:
    combo = widget._layout_preset_combo

    assert combo.itemData(0) is None, "first row carries no preset"
    offered = [combo.itemData(i) for i in range(1, combo.count())]
    assert offered == [preset for _label, preset in widget.LAYOUT_PRESETS]
    assert set(offered) == set(layout_presets.PRESETS)


def test_there_is_no_apply_layout_button_any_more(
    widget: FigurePropertiesWidget,
) -> None:
    """The one thing that actually got removed."""
    assert not hasattr(widget, "_btn_apply_layout_preset")


def test_picking_a_preset_requests_it(widget: FigurePropertiesWidget) -> None:
    requested: list[str] = []
    widget.layout_preset_requested.connect(requested.append)

    combo = widget._layout_preset_combo
    combo.setCurrentIndex(combo.findData(layout_presets.OVERLAPPING))

    assert requested == [layout_presets.OVERLAPPING]


def test_picking_the_placeholder_requests_nothing(
    widget: FigurePropertiesWidget,
) -> None:
    combo = widget._layout_preset_combo
    combo.setCurrentIndex(combo.findData(layout_presets.SHARED_GRID))

    requested: list[str] = []
    widget.layout_preset_requested.connect(requested.append)
    combo.setCurrentIndex(0)

    assert requested == []


def test_the_same_preset_can_be_picked_again(widget: FigurePropertiesWidget) -> None:
    """"Grid" is how another preset gets undone, and re-running one after a
    hand edit has to work - which it only does because the combo goes back
    to its placeholder, making the next pick a change of index again."""
    requested: list[str] = []
    widget.layout_preset_requested.connect(requested.append)
    combo = widget._layout_preset_combo

    for _ in range(2):
        combo.setCurrentIndex(combo.findData(layout_presets.GRID))
        widget._reset_layout_preset_combo()  # what the reload after an apply does

    assert requested == [layout_presets.GRID, layout_presets.GRID]


def test_reloading_puts_the_combo_back_without_requesting_anything(
    widget: FigurePropertiesWidget, repo: SqliteRepo
) -> None:
    """The dangerous case: the reload that follows every apply (and every
    chart switch) must not read as a fresh pick."""
    combo = widget._layout_preset_combo
    combo.setCurrentIndex(combo.findData(layout_presets.MAIN_AND_SECONDARY))

    requested: list[str] = []
    widget.layout_preset_requested.connect(requested.append)
    widget._reload_from_descriptor()

    assert combo.currentIndex() == 0
    assert requested == []


def test_a_disconnected_panel_requests_nothing(qapp) -> None:
    """clear_connected_figure resets the combo too; with no figure behind
    it, that reset must not rearrange whichever figure comes next."""
    widget = FigurePropertiesWidget()
    requested: list[str] = []
    widget.layout_preset_requested.connect(requested.append)

    combo = widget._layout_preset_combo
    combo.setCurrentIndex(combo.findData(layout_presets.SHARED_GRID))

    assert requested == []


# ----------------------------------------------------------------------
# End to end, through MainWindow's handler
# ----------------------------------------------------------------------
def test_picking_a_preset_actually_rearranges_the_figure(
    qapp, repo: SqliteRepo
) -> None:
    """MainWindow._on_layout_preset_requested is the other half: the panel
    only names a preset, the window knows which axes the figure has."""
    import pandas as pd

    from app.dialogs.main_window import MainWindow
    from app.logs.logger import applogger

    repo.import_dataframe(
        pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]}),
        table_name="t",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=4))
    for index in range(4):
        axis_id = int(
            repo.create_axis_descriptor(
                figure_id=figure_id, axis_index=index, chart_type="Scatter Plot",
                title=f"ax{index}", x_label="x", y_label="y", options={},
            )
        )
        repo.create_series_descriptor(
            axis_id=axis_id, series_index=0, name="s",
            sql_query="SELECT x, y FROM t", roles={"x": "x", "y": "y"}, style={},
        )

    window = MainWindow(repo=repo, db_path=repo.db_path)
    try:
        window._set_nav_index(1)
        combo = window._figure_widget._layout_preset_combo
        combo.setCurrentIndex(combo.findData(layout_presets.SHARED_GRID))

        descriptor = repo.load_figure_descriptor(figure_id)
        # 4 axes, shared grid: a compact 2x2, everything after the first
        # sharing the first's scale.
        assert (descriptor.nrows, descriptor.ncols) == (2, 2)
        by_index = sorted(descriptor.axes, key=lambda a: a.axis_index)
        assert not (by_index[0].options or {}).get("sharex")
        assert all((axis.options or {}).get("sharex") for axis in by_index[1:])
    finally:
        window.close()
        applogger.set_status_bar(None)
