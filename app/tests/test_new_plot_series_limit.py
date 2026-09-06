"""A chart type that draws one series per axes gets one axis per series.

Five renderers - the two surfaces, the two contour maps and the table - keep
the first drawable series and log the others away, because a second surface
occludes the first and a second table is drawn on top of one.  Until now the
New plot dialog let a person add four of them to one axis and told them
nothing: the figure came back with one surface and three series that had been
saved, were listed in the properties panel, and drew nothing.

The dialog now reads ``MaxSeries`` off the renderer and splits the list into
one axis per series instead.  These tests are about that split and about the
one case it cannot cover - an existing axis, which is one axis by definition.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.create_chart_dialog import NewPlotTabDialog


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    """A repo with one gridded table a surface can be drawn from."""
    repo = SqliteRepo(db_path=tmp_db_path)
    repo.query_df("DROP TABLE IF EXISTS field")
    repo.query_df("CREATE TABLE field (x REAL, y REAL, z REAL)")
    repo.query_df(
        "INSERT INTO field (x, y, z) VALUES "
        "(0.0, 0.0, 1.0), (0.0, 1.0, 2.0), (1.0, 0.0, 3.0), (1.0, 1.0, 4.0)"
    )
    yield repo
    repo.close()


def _dialog(qapp, repo: SqliteRepo, chart_type: str) -> NewPlotTabDialog:
    """Open the dialog on *chart_type* with its one default series."""
    dialog = NewPlotTabDialog(repo, current_table="field")
    assert dialog._select_renderer_by_name(chart_type), chart_type
    return dialog


def _add_series(dialog: NewPlotTabDialog, count: int) -> None:
    for _index in range(count):
        dialog._on_add_series()


# ----------------------------------------------------------------------
# What the renderers declare
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "chart_type",
    ["Surface Plot", "Surface Plot (Scattered)", "Contour Plot", "Pie Chart", "Table"],
)
def test_single_series_renderers_say_so(qapp, repo: SqliteRepo, chart_type: str) -> None:
    """The limit is read from the renderer, not from a list kept in the dialog."""
    dialog = _dialog(qapp, repo, chart_type)

    assert dialog._max_series_per_axis() == 1


def test_an_ordinary_renderer_has_no_limit(qapp, repo: SqliteRepo) -> None:
    dialog = _dialog(qapp, repo, "Scatter Plot")

    assert dialog._max_series_per_axis() is None


# ----------------------------------------------------------------------
# Splitting the list
# ----------------------------------------------------------------------
def test_series_are_grouped_one_per_axis(qapp, repo: SqliteRepo) -> None:
    dialog = _dialog(qapp, repo, "Surface Plot")
    _add_series(dialog, 2)  # three including the one the dialog starts with

    groups = dialog._axis_groups(dialog._ordered_drafts())

    assert [len(group) for group in groups] == [1, 1, 1]


def test_an_unlimited_renderer_keeps_one_group(qapp, repo: SqliteRepo) -> None:
    dialog = _dialog(qapp, repo, "Scatter Plot")
    _add_series(dialog, 2)

    assert [len(group) for group in dialog._axis_groups(dialog._ordered_drafts())] == [3]


def test_the_split_is_announced_before_create_is_pressed(qapp, repo: SqliteRepo) -> None:
    """The note is the whole point: the person has to know before, not after."""
    dialog = _dialog(qapp, repo, "Surface Plot")
    assert not dialog._series_split_note.isVisible() or not dialog._series_split_note.text()

    _add_series(dialog, 1)

    assert dialog._series_split_note.text()


def test_a_split_withdraws_the_existing_axis_choice(qapp, repo: SqliteRepo) -> None:
    """One axis cannot hold a set of series meant for several."""
    figure_id = repo.create_figure_descriptor(name="fig", nrows=1, ncols=1)
    repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Surface Plot",
        title="first",
        x_label="x",
        y_label="y",
        z_label="z",
        options={"projection": "3d"},
    )

    dialog = NewPlotTabDialog(repo, current_figure_id=figure_id, current_table="field")
    assert dialog._select_renderer_by_name("Surface Plot")
    dialog._rb_current_figure.setChecked(True)
    assert dialog._rb_existing_axis.isEnabled()

    _add_series(dialog, 1)

    assert not dialog._rb_existing_axis.isEnabled()


# ----------------------------------------------------------------------
# What lands in the database
# ----------------------------------------------------------------------
def _accept_with_roles(dialog: NewPlotTabDialog) -> None:
    """Map every required role to the column of the same name, then create."""
    for role, combo in dialog._role_combos.items():
        index = combo.findData(role)
        if index >= 0:
            combo.setCurrentIndex(index)
    for item_id, draft in dialog._series_by_item.items():
        draft.roles = {role: role for role in dialog._all_renderer_role_names()}
    dialog._on_accept()


def test_three_surfaces_become_three_axes(qapp, repo: SqliteRepo) -> None:
    dialog = _dialog(qapp, repo, "Surface Plot")
    dialog._edit_figure_name.setText("surfaces")
    _add_series(dialog, 2)

    _accept_with_roles(dialog)

    result = dialog.result
    assert result is not None
    assert result.series_count == 3
    axes = repo.list_axes_for_figure(result.figure_id)
    assert len(axes) == 3
    for axis_id, _index, _title in axes:
        assert repo.next_series_index(int(axis_id)) == 1


def test_each_split_axis_keeps_the_3d_projection(qapp, repo: SqliteRepo) -> None:
    """A surface needs a 3D axes, and the second one needs it as much as the
    first - the option is applied per axis, not once for the figure."""
    dialog = _dialog(qapp, repo, "Surface Plot")
    dialog._edit_figure_name.setText("surfaces")
    _add_series(dialog, 1)

    _accept_with_roles(dialog)

    result = dialog.result
    assert result is not None
    for axis_id, _index, _title in repo.list_axes_for_figure(result.figure_id):
        assert (repo.get_axis_options(int(axis_id)) or {}).get("projection") == "3d"


def test_each_split_axis_is_named_after_its_own_series(qapp, repo: SqliteRepo) -> None:
    dialog = _dialog(qapp, repo, "Surface Plot")
    dialog._edit_figure_name.setText("surfaces")
    _add_series(dialog, 1)
    for row, name in enumerate(("north", "south")):
        dialog._series_list.setCurrentRow(row)
        dialog._edit_series_name.setText(name)
        dialog._on_series_name_changed(name)

    _accept_with_roles(dialog)

    result = dialog.result
    assert result is not None
    titles = [title for _id, _index, title in repo.list_axes_for_figure(result.figure_id)]
    assert titles == ["north", "south"]


def test_three_scatters_stay_on_one_axis(qapp, repo: SqliteRepo) -> None:
    """The unlimited case is unchanged, which is most of the application."""
    dialog = _dialog(qapp, repo, "Scatter Plot")
    dialog._edit_figure_name.setText("scatters")
    _add_series(dialog, 2)

    _accept_with_roles(dialog)

    result = dialog.result
    assert result is not None
    assert len(repo.list_axes_for_figure(result.figure_id)) == 1
    assert repo.next_series_index(result.axis_id) == 3
