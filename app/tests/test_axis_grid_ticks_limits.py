"""Grid, ticks and view range: the three settings that did not do what they said.

Show grid could only ever add a grid - it returned early when the box was
unticked - so against a style sheet with ``axes.grid: True``, which the shipped
ones have, unticking it changed nothing. Minor ticks was a switch that only
turned them on. And there was no way at all to say where an axis should start
and stop.

The settings are now one per axis and per tick class, and each distinguishes
three things a boolean cannot: leave it to the style, force it off, force it
on. These tests are mostly about "off", because that is the half that never
worked.
"""
from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from app.charts import axis_options  # noqa: E402
from app.charts.render_figure import (  # noqa: E402
    _apply_grid,
    _apply_limits,
    _apply_ticks,
)


def _axes(grid_in_style: bool = False, **kwargs):
    """Return drawn axes with *options* applied, as the renderer applies them."""
    with plt.rc_context({"axes.grid": grid_in_style}):
        figure = Figure()
        ax = figure.add_subplot(111)
        ax.plot([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        _apply_ticks(ax, kwargs)
        _apply_grid(ax, kwargs)
        figure.canvas.draw()
        return ax


def _major_grid_visible(ax, axis: str) -> bool:
    lines = (ax.xaxis if axis == "x" else ax.yaxis).get_gridlines()
    return bool(lines) and all(line.get_visible() for line in lines)


def _minor_grid(ax, axis: str) -> list:
    ticks = (ax.xaxis if axis == "x" else ax.yaxis).get_minor_ticks()
    return [tick.gridline for tick in ticks]


def _minor_tick_count(ax, axis: str) -> int:
    return len((ax.xaxis if axis == "x" else ax.yaxis).get_minorticklocs())


# ----------------------------------------------------------------------
# The grid, including off
# ----------------------------------------------------------------------
def test_a_grid_can_be_turned_off_against_a_style_that_turns_it_on() -> None:
    """The reported bug, in one assertion."""
    ax = _axes(grid_in_style=True, grid_x_major=axis_options.OFF)

    assert not _major_grid_visible(ax, "x")


def test_turning_one_grid_off_leaves_the_other_alone() -> None:
    ax = _axes(grid_in_style=True, grid_x_major=axis_options.OFF)

    assert _major_grid_visible(ax, "y")


def test_auto_leaves_the_style_sheet_in_charge() -> None:
    """The default, and what an option nobody set has to mean."""
    on = _axes(grid_in_style=True, grid_x_major=axis_options.AUTO)
    off = _axes(grid_in_style=False, grid_x_major=axis_options.AUTO)

    assert _major_grid_visible(on, "x")
    assert not _major_grid_visible(off, "x")


def test_a_grid_can_be_turned_on_against_a_style_that_leaves_it_off() -> None:
    ax = _axes(grid_in_style=False, grid_y_major=axis_options.ON)

    assert _major_grid_visible(ax, "y")
    assert not _major_grid_visible(ax, "x")


def test_a_line_style_applies_to_that_grid_only() -> None:
    """Dotted minor under solid major is what makes two levels readable."""
    ax = _axes(grid_y_minor=":", grid_y_major=axis_options.ON)

    minor = _minor_grid(ax, "y")
    assert minor and all(line.get_visible() for line in minor)
    assert all(line.get_linestyle() == ":" for line in minor)
    assert all(not line.get_visible() for line in _minor_grid(ax, "x"))


def test_a_minor_grid_brings_the_minor_ticks_it_needs() -> None:
    """A grid on ticks that do not exist draws nothing, which reads as broken."""
    ax = _axes(grid_y_minor="--")

    assert _minor_tick_count(ax, "y") > 0


# ----------------------------------------------------------------------
# Ticks
# ----------------------------------------------------------------------
def test_minor_ticks_can_be_turned_on_for_one_axis() -> None:
    ax = _axes(ticks_x_minor=axis_options.ON)

    assert _minor_tick_count(ax, "x") > 0


def test_minor_ticks_can_be_turned_off() -> None:
    """The half that was missing: the old switch only ever turned them on."""
    ax = _axes(ticks_x_minor=axis_options.ON, ticks_y_minor=axis_options.OFF)

    assert _minor_tick_count(ax, "x") > 0
    assert _minor_tick_count(ax, "y") == 0


def test_a_direction_applies_to_the_named_ticks_only() -> None:
    ax = _axes(ticks_x_major="out", ticks_y_major="in")

    assert ax.xaxis.get_major_ticks()[0]._tickdir == "out"
    assert ax.yaxis.get_major_ticks()[0]._tickdir == "in"


def test_major_ticks_off_keeps_the_labels() -> None:
    """"No ticks" is about the marks; a chart with no numbers is a different
    request, and one nobody made here."""
    ax = _axes(ticks_x_major=axis_options.OFF)

    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 0
    assert any(label.get_text() for label in ax.get_xticklabels())


def test_minor_ticks_on_a_log_axis_use_the_log_locator() -> None:
    """AutoMinorLocator is simply wrong on a log scale, which is why this goes
    through minorticks_on rather than setting a locator directly."""
    with plt.rc_context({"axes.grid": False}):
        figure = Figure()
        ax = figure.add_subplot(111)
        ax.plot([1.0, 10.0, 100.0], [1.0, 2.0, 3.0])
        ax.set_xscale("log")
        _apply_ticks(ax, {"ticks_x_minor": axis_options.ON})
        figure.canvas.draw()

    assert _minor_tick_count(ax, "x") > 0
    assert "Log" in type(ax.xaxis.get_minor_locator()).__name__


# ----------------------------------------------------------------------
# Figures saved before any of this existed
# ----------------------------------------------------------------------
def test_an_old_figure_keeps_the_grid_it_had() -> None:
    """grid(True, which="both", axis="y") never said anything about x, so x
    stays with the style sheet rather than being turned off."""
    ax = _axes(
        grid_in_style=True, grid=True, grid_which="both", grid_axis="y"
    )

    assert _major_grid_visible(ax, "y")
    assert _major_grid_visible(ax, "x"), "the style sheet still decides x"
    assert all(line.get_visible() for line in _minor_grid(ax, "y"))


def test_an_old_figure_without_a_grid_is_left_to_the_style() -> None:
    """"grid: False" meant "do nothing", and translating it to "off" would
    change how every old figure looks."""
    ax = _axes(grid_in_style=True, grid=False)

    assert _major_grid_visible(ax, "x")
    assert _major_grid_visible(ax, "y")


def test_an_old_minor_ticks_switch_still_turns_them_on() -> None:
    ax = _axes(minor_ticks=True)

    assert _minor_tick_count(ax, "x") > 0
    assert _minor_tick_count(ax, "y") > 0


def test_an_old_tick_direction_still_applies_to_both_axes() -> None:
    ax = _axes(tick_direction="in")

    assert ax.xaxis.get_major_ticks()[0]._tickdir == "in"
    assert ax.yaxis.get_major_ticks()[0]._tickdir == "in"


# ----------------------------------------------------------------------
# Limits
# ----------------------------------------------------------------------
def _limited(**options) -> tuple:
    figure = Figure()
    ax = figure.add_subplot(111)
    ax.plot([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    _apply_limits(ax, options)
    return ax.get_xlim(), ax.get_ylim()


def test_automatic_leaves_both_axes_to_the_data() -> None:
    (x_low, x_high), (y_low, y_high) = _limited(
        limits_mode=axis_options.LIMITS_AUTO, x_min=-50.0, x_max=50.0
    )

    assert x_low > -50.0 and x_high < 50.0
    assert y_low < 0.0 < 4.0 < y_high or y_high >= 4.0


def test_manual_sets_both_axes() -> None:
    limits = _limited(
        limits_mode=axis_options.LIMITS_MANUAL,
        x_min=-1.0,
        x_max=3.0,
        y_min=-2.0,
        y_max=8.0,
    )

    assert limits == ((-1.0, 3.0), (-2.0, 8.0))


def test_automatic_x_leaves_x_alone_and_fixes_y() -> None:
    (x_low, x_high), y_limits = _limited(
        limits_mode=axis_options.LIMITS_AUTO_X,
        x_min=-99.0,
        x_max=99.0,
        y_min=-2.0,
        y_max=8.0,
    )

    assert (x_low, x_high) != (-99.0, 99.0)
    assert y_limits == (-2.0, 8.0)


def test_one_empty_end_stays_automatic() -> None:
    """An axis can be pinned at the bottom and left to the data at the top."""
    _x, (y_low, y_high) = _limited(limits_mode=axis_options.LIMITS_MANUAL, y_min=0.0)

    assert y_low == 0.0
    assert y_high > 4.0


def test_reversed_limits_are_put_in_order() -> None:
    """Inverting through the limits would fight the axis direction option,
    which is applied after them."""
    (x_low, x_high), _y = _limited(
        limits_mode=axis_options.LIMITS_MANUAL, x_min=5.0, x_max=1.0
    )

    assert (x_low, x_high) == (1.0, 5.0)


def test_two_identical_ends_are_ignored_rather_than_collapsing_the_axis() -> None:
    (x_low, x_high), _y = _limited(
        limits_mode=axis_options.LIMITS_MANUAL, x_min=2.0, x_max=2.0
    )

    assert x_low != x_high


@pytest.mark.parametrize(
    "mode, automatic",
    [
        (axis_options.LIMITS_AUTO, {"x", "y"}),
        (axis_options.LIMITS_AUTO_X, {"x"}),
        (axis_options.LIMITS_AUTO_Y, {"y"}),
        (axis_options.LIMITS_MANUAL, set()),
    ],
)
def test_each_mode_says_which_axis_is_automatic(mode: str, automatic: set) -> None:
    options = {"limits_mode": mode}

    assert {
        axis for axis in axis_options.AXES if axis_options.is_automatic(options, axis)
    } == automatic


def test_an_unknown_mode_falls_back_to_automatic() -> None:
    """A figure naming a mode this build does not have still opens."""
    assert axis_options.limits_mode({"limits_mode": "sideways"}) == axis_options.LIMITS_AUTO


# ----------------------------------------------------------------------
# The controls that write them
# ----------------------------------------------------------------------
@pytest.fixture
def widget(qapp):
    from app.widgets.axis_properties import AxisPropertiesWidget

    return AxisPropertiesWidget()


def test_the_widget_offers_one_control_per_axis_and_tick_class(widget) -> None:
    """Four and four, which is what "x major and y minor" needs to be sayable."""
    expected = {(axis, which) for axis in axis_options.AXES for which in axis_options.WHICH}

    assert set(widget._grid_combos) == expected
    assert set(widget._tick_combos) == expected


def test_every_choice_reaches_the_control(widget) -> None:
    combo = widget._grid_combos[("x", "major")]
    stored = {combo.itemData(index) for index in range(combo.count())}

    assert stored == {value for value, _label in axis_options.GRID_CHOICES}


def test_what_is_chosen_is_what_the_renderer_reads(widget) -> None:
    """The widget writes the keys axis_options names, so a setting cannot be
    saved under a name nothing applies."""
    combo = widget._grid_combos[("y", "minor")]
    combo.setCurrentIndex(combo.findData(":"))

    payload = widget._extended_axis_options_payload()

    assert payload[axis_options.grid_key("y", "minor")] == ":"
    assert axis_options.grid_setting(payload, "y", "minor") == ":"


def test_an_old_figure_opens_with_the_settings_it_had(widget) -> None:
    widget._load_extended_axis_options(
        {"grid": True, "grid_which": "major", "grid_axis": "x", "minor_ticks": True}
    )

    assert widget._grid_combos[("x", "major")].currentData() == axis_options.ON
    assert widget._grid_combos[("y", "major")].currentData() == axis_options.AUTO
    assert widget._tick_combos[("x", "minor")].currentData() == axis_options.ON


def test_the_stale_legacy_keys_are_rewritten_on_save(widget) -> None:
    """Otherwise a build that still reads "grid" would draw a grid the user
    has just switched off in these controls."""
    widget._load_extended_axis_options({"grid": True, "grid_axis": "both"})
    for combo in widget._grid_combos.values():
        combo.setCurrentIndex(combo.findData(axis_options.OFF))

    payload = widget._extended_axis_options_payload()

    assert payload["grid"] is False


def test_the_limit_boxes_follow_the_mode(widget) -> None:
    """A box that does nothing in the chosen mode is a box that lies.

    Asked as WA_ForceDisabled rather than isEnabled(): the whole properties
    panel is disabled until an axis is selected, so isEnabled() is False for
    every control here and would answer a different question.
    """
    from PySide6.QtCore import Qt

    combo = widget._limits_mode_combo
    combo.setCurrentIndex(combo.findData(axis_options.LIMITS_AUTO_X))

    def switched_off(axis: str) -> bool:
        return widget._limit_spins[(axis, "min")].testAttribute(
            Qt.WidgetAttribute.WA_ForceDisabled
        )

    assert switched_off("x") is True, "x is automatic in this mode"
    assert switched_off("y") is False


def test_an_untouched_limit_is_saved_as_unset(widget) -> None:
    """Not as the bottom of the spin box's range, which would pin the axis to
    minus a quadrillion."""
    combo = widget._limits_mode_combo
    combo.setCurrentIndex(combo.findData(axis_options.LIMITS_MANUAL))

    payload = widget._extended_axis_options_payload()

    assert payload["x_min"] is None
    assert payload["y_max"] is None


def test_a_typed_limit_survives_the_round_trip(widget) -> None:
    combo = widget._limits_mode_combo
    combo.setCurrentIndex(combo.findData(axis_options.LIMITS_MANUAL))
    widget._limit_spins[("y", "max")].setValue(12.5)

    payload = widget._extended_axis_options_payload()

    assert payload["y_max"] == 12.5
    assert axis_options.manual_limit(payload, "y", "max") == 12.5

    widget._load_extended_axis_options(payload)
    assert widget._limit_spins[("y", "max")].value() == 12.5
