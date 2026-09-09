""""Reset to style" for the axis kwargs editor.

DictEditorPanel.reset_to_defaults() already existed - every kwarg's schema
default *is* kwarg_spec.DEFAULT, the "leave this to the style sheet"
sentinel - but nothing in axis_properties.py ever called it, so the only way
back to "stop overriding this" was retyping "default" into each row by hand.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.widgets.axis_properties import AxisPropertiesWidget


@pytest.fixture
def widget(qapp, repo: SqliteRepo):
    repo.import_dataframe(
        pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]}),
        table_name="t",
        normalize_columns=False,
    )
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id,
            axis_index=0,
            chart_type="Scatter Plot",
            title="t",
            x_label="x",
            y_label="y",
            options={},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id, series_index=0, name="s",
        sql_query="SELECT x, y FROM t", roles={"x": "x", "y": "y"}, style={},
    )

    built = AxisPropertiesWidget()
    built.set_connected_figure(repo, figure_id, figure=None)
    built.rebuild_kwargs_editor(axis_id)
    return built


def test_the_reset_button_exists_and_is_wired(widget: AxisPropertiesWidget) -> None:
    assert widget._btn_reset_kwargs.isEnabled()

    editor = widget._kwargs_editor
    assert editor is not None
    a_key = next(iter(editor.config))
    editor.set_value_for_key(a_key, "a value nothing defaults to")

    widget._btn_reset_kwargs.click()

    assert editor.get_values()[a_key] == editor.config[a_key].get("default")


def test_reset_puts_every_row_back_to_its_schema_default(
    widget: AxisPropertiesWidget,
) -> None:
    editor = widget._kwargs_editor
    assert editor is not None

    # Change something away from its default first, or the test could pass
    # by never having moved anything in the first place.
    a_key = next(iter(editor.config))
    editor.set_value_for_key(a_key, "a value nothing defaults to")
    assert editor.get_values()[a_key] != editor.config[a_key].get("default")

    widget._reset_kwargs_to_defaults()

    for key, meta in editor.config.items():
        assert editor.get_values()[key] == meta.get("default")


def test_reset_does_not_touch_the_stored_axis_until_apply(
    widget: AxisPropertiesWidget, repo: SqliteRepo
) -> None:
    """Same rule as any other edit in this panel: nothing is persisted
    until Apply is pressed, so a reset can still be backed out of."""
    axis_id = widget.current_axis_id()
    before = dict(repo.get_axis_options(axis_id) or {})

    widget._reset_kwargs_to_defaults()

    after = dict(repo.get_axis_options(axis_id) or {})
    assert after == before


def test_resetting_with_no_editor_loaded_does_not_raise(qapp) -> None:
    """No axis connected yet - clicking Reset must be a no-op, not a crash."""
    built = AxisPropertiesWidget()
    built._reset_kwargs_to_defaults()


def test_the_button_shares_the_editor_s_search_row(widget: AxisPropertiesWidget) -> None:
    """One strip of chrome over the tree, not two: the button sits beside
    "Search properties...", not on a row of its own above it."""
    editor = widget._kwargs_editor
    assert editor is not None

    row = widget._btn_reset_kwargs.parentWidget()
    assert row is editor.search_edit.parentWidget()
    assert row.layout().indexOf(widget._btn_reset_kwargs) > row.layout().indexOf(
        editor.search_edit
    )


def test_the_button_is_icon_only(widget: AxisPropertiesWidget) -> None:
    """Beside a field rather than in a row of labelled buttons - the
    tooltip carries what the words were carrying."""
    button = widget._btn_reset_kwargs

    assert button.text() == ""
    assert button.property("iconOnly") is True
    assert not button.icon().isNull()
    assert button.toolTip()


def test_rebuilding_the_editor_replaces_the_button_with_it(
    widget: AxisPropertiesWidget,
) -> None:
    """The button is owned by the editor now, so it must be rebuilt with
    it - a kept reference would be a pointer into a deleted panel."""
    first = widget._btn_reset_kwargs
    widget.rebuild_kwargs_editor(widget.current_axis_id())

    assert widget._btn_reset_kwargs is not first
    assert widget._btn_reset_kwargs.parentWidget() is not None
    widget._btn_reset_kwargs.click()  # still wired, and does not crash
