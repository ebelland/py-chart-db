"""The fit mode, moved from ChartPanel's context menu to Figure Properties.

It used to be three checkable menu entries kept in sync by comparing an
action's *label* against the mode - and the label was ``_("Fit")`` /
``_("Fit proportional")`` / ``_("Fixed")``, translated, while the comparison
was against the English literal. So on any language but English the
checkmarks never matched what was actually selected, silently. Moving it to
one combo, whose stored value is the mode key rather than the label, removes
the comparison rather than fixing it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import RESIZE_MODE_CHOICES, RESIZE_MODES


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
        pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 4.0]}),
        table_name="points",
        normalize_columns=False,
    )
    yield built
    built.close()


@pytest.fixture
def figure_id(repo: SqliteRepo) -> int:
    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    axis_id = int(
        repo.create_axis_descriptor(
            figure_id=figure_id,
            axis_index=0,
            chart_type="Scatter Plot",
            title="",
            x_label="",
            y_label="",
            options={"grid": True},
        )
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="s",
        sql_query='SELECT x, y FROM "points"',
        roles={"x": "x", "y": "y"},
        style={"marker": "o"},
    )
    return figure_id


@pytest.fixture
def panel(qapp, repo: SqliteRepo, figure_id: int):
    from app.widgets.chart_panel import ChartPanel

    return ChartPanel(repo=repo, figure_id=figure_id, parent=None)


@pytest.fixture
def widget(qapp):
    from app.widgets.figure_properties import FigurePropertiesWidget

    return FigurePropertiesWidget()


# ----------------------------------------------------------------------
# The menu no longer offers it
# ----------------------------------------------------------------------
def test_the_context_menu_no_longer_offers_a_fit_mode(panel) -> None:
    menu = panel._build_actions_menu()
    labels = {action.text() for action in menu.actions() if not action.isSeparator()}

    assert labels.isdisjoint({"Fit", "Fit proportional", "Fixed"})


def test_removing_the_entries_left_one_separator_not_two(panel) -> None:
    """Three items plus their two flanking Nones came out; the surviving
    separator either side of them should not have doubled up."""
    menu = panel._build_actions_menu()
    actions = menu.actions()

    consecutive_separators = any(
        first.isSeparator() and second.isSeparator()
        for first, second in zip(actions, actions[1:])
    )
    assert not consecutive_separators


def test_the_menu_keeps_everything_that_was_not_the_fit_mode(panel) -> None:
    menu = panel._build_actions_menu()
    labels = [action.text() for action in menu.actions() if not action.isSeparator()]

    assert labels == ["Reload", "Copy", "Save", "Delete"]


# ----------------------------------------------------------------------
# The panel still owns the state; it is just reached differently
# ----------------------------------------------------------------------
def test_the_panel_exposes_its_own_mode(panel) -> None:
    panel.set_resize_mode("FIT", persist=False, redraw=False)

    assert panel.resize_mode == "FIT"


@pytest.mark.parametrize("mode", RESIZE_MODES)
def test_set_resize_mode_still_works_the_same_way(panel, mode: str) -> None:
    """Nothing about how a mode is applied changed - only how it is chosen."""
    panel.set_resize_mode(mode, persist=False, redraw=False)

    assert panel.resize_mode == mode


# ----------------------------------------------------------------------
# The combo in Figure Properties
# ----------------------------------------------------------------------
def test_the_combo_offers_every_mode(widget) -> None:
    combo = widget._resize_mode_combo
    stored = {combo.itemData(index) for index in range(combo.count())}

    assert stored == set(RESIZE_MODES)


def test_every_choice_has_a_translated_tooltip(widget) -> None:
    from PySide6.QtCore import Qt

    combo = widget._resize_mode_combo
    for index in range(combo.count()):
        tooltip = combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
        assert tooltip


def test_the_combo_is_disabled_until_wired(widget, panel) -> None:
    """set_connected_figure alone must not leave a combo that looks
    interactive but is not attached to anything."""
    widget.set_connected_figure(
        repo=panel._repo, figure_id=panel.figure_id, figure=panel.figure
    )

    assert widget._resize_mode_combo.isEnabled() is False


def test_wiring_shows_the_panel_s_actual_mode(widget, panel) -> None:
    panel.set_resize_mode("FIXED", persist=False, redraw=False)

    widget.set_resize_mode_control(panel.resize_mode, panel.set_resize_mode)

    assert widget._resize_mode_combo.currentData() == "FIXED"
    assert widget._resize_mode_combo.isEnabled() is True


def test_choosing_a_mode_in_the_combo_changes_the_panel(widget, panel) -> None:
    panel.set_resize_mode("FIXED", persist=False, redraw=False)
    widget.set_resize_mode_control(panel.resize_mode, panel.set_resize_mode)

    combo = widget._resize_mode_combo
    combo.setCurrentIndex(combo.findData("FIT_PROPORTIONAL"))

    assert panel.resize_mode == "FIT_PROPORTIONAL"


def test_choosing_a_mode_persists_it_to_the_descriptor(widget, panel, repo: SqliteRepo) -> None:
    widget.set_resize_mode_control(panel.resize_mode, panel.set_resize_mode)
    combo = widget._resize_mode_combo
    combo.setCurrentIndex(combo.findData("FIXED"))

    saved = repo.load_figure_descriptor(panel.figure_id).options.get("view", {})
    assert saved.get("resize_mode") == "FIXED"


def test_reconnecting_shows_the_mode_the_panel_actually_has(widget, panel) -> None:
    """Switching chart tabs and back must not show a stale value."""
    panel.set_resize_mode("FIT", persist=False, redraw=False)
    widget.set_resize_mode_control(panel.resize_mode, panel.set_resize_mode)

    panel.set_resize_mode("FIXED", persist=False, redraw=False)
    widget.set_resize_mode_control(panel.resize_mode, panel.set_resize_mode)

    assert widget._resize_mode_combo.currentData() == "FIXED"


def test_clearing_disables_the_combo_and_drops_the_wiring(widget, panel) -> None:
    """Otherwise the old panel's setter keeps firing after its tab closed."""
    widget.set_resize_mode_control(panel.resize_mode, panel.set_resize_mode)

    widget.clear_connected_figure()

    assert widget._resize_mode_combo.isEnabled() is False
    assert widget._on_resize_mode_changed is None


def test_the_labels_and_the_context_menu_named_the_same_thing(widget) -> None:
    """The combo is the only picker left; its wording is what the removed
    menu items used, so nobody has to relearn what "Fit" meant."""
    combo_labels = {value: combo_label for value, combo_label, _tip in RESIZE_MODE_CHOICES}
    assert combo_labels == {
        "FIT_PROPORTIONAL": "Fit proportional",
        "FIT": "Fit",
        "FIXED": "Fixed",
    }
