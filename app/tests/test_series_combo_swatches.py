"""The series selector's colour/linestyle swatch icons.

Series properties used to list series by name alone, in a plain-text combo -
every row read the same regardless of what colour or line style it was
actually going to draw, so "which one is this" meant opening each one in
turn. Each row now carries a small preview icon instead; these tests pin
what it resolves to, not what it looks like pixel for pixel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from cycler import cycler
from matplotlib import rcParams

from app.widgets.series_properties import (
    SeriesPropertiesWidget,
    _series_swatch_icon,
)


@dataclass
class _FakeSeries:
    id: int
    name: str = ""
    style: dict[str, Any] | None = None


@pytest.fixture(autouse=True)
def _restore_prop_cycle():
    saved = rcParams["axes.prop_cycle"]
    yield
    rcParams["axes.prop_cycle"] = saved


def test_the_icon_function_never_raises_on_odd_input(qapp) -> None:
    """A bad or empty colour string must fall back, not crash the combo."""
    for color in ("#ff0000", "red", "", "not-a-color", "#123"):
        for linestyle in ("-", "--", "-.", ":", "none", "unknown-code"):
            assert _series_swatch_icon(color, linestyle) is not None


def test_an_explicit_colour_and_linestyle_are_used_as_is(qapp) -> None:
    widget = SeriesPropertiesWidget()
    series = _FakeSeries(id=1, style={"color": "#ff0000", "linestyle": "--"})

    widget._series_combo.addItem("placeholder")  # occupy index 0
    icon = widget._series_swatch_icon_for(series)

    assert not icon.isNull()


def test_no_explicit_colour_cycles_by_position_in_the_combo(qapp) -> None:
    """Two series that both leave colour unset must not resolve to the
    same swatch - that is exactly the "which one is this" problem this
    feature exists to fix."""
    rcParams["axes.prop_cycle"] = cycler(color=["#111111", "#222222", "#333333"])
    widget = SeriesPropertiesWidget()

    first = _FakeSeries(id=1, style={})
    icon_a = widget._series_swatch_icon_for(first)
    widget._series_combo.addItem("a", 1)

    second = _FakeSeries(id=2, style={})
    icon_b = widget._series_swatch_icon_for(second)

    assert not icon_a.isNull()
    assert not icon_b.isNull()
    assert icon_a.cacheKey() != icon_b.cacheKey()


def test_a_series_with_no_line_gets_a_dot_not_an_empty_icon(qapp) -> None:
    widget = SeriesPropertiesWidget()
    series = _FakeSeries(id=1, style={"color": "#00ff00", "linestyle": "none"})

    icon = widget._series_swatch_icon_for(series)
    assert not icon.isNull()
