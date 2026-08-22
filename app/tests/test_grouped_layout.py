"""Tests for the category/series layout shared by box and violin plots.

The rule under test: a series only takes a slot in the categories it actually
appears in.  Reserving a slot everywhere pushed a lone item a quarter of a band
away from the tick labelled underneath it, which is what four one-category
series - four batches, each its own series - looked like.
"""
from __future__ import annotations

import pytest

from app.charts.box import BoxAxisRenderer


def _item(category: str, series: str) -> dict:
    return {"category_label": category, "series_label": series, "data": [], "style": {}}


def _layout(items: list[dict], series_labels: list[str], width: float = 0.5) -> dict:
    categories: list[str] = []
    for item in items:
        if item["category_label"] not in categories:
            categories.append(item["category_label"])
    return BoxAxisRenderer.grouped_layout(
        items=items,
        category_labels=categories,
        series_labels=series_labels,
        requested_width=width,
    )


def test_a_lone_item_sits_on_its_tick() -> None:
    """One series per category: every item is centred on its own label."""
    items = [_item(name, f"Batch {name}") for name in ("A", "B", "C", "D")]
    layout = _layout(items, [f"Batch {name}" for name in ("A", "B", "C", "D")])

    for index, item in enumerate(items):
        assert BoxAxisRenderer.item_position(layout, item) == pytest.approx(float(index))


def test_two_series_in_one_category_are_offset_symmetrically() -> None:
    items = [_item("A", "left"), _item("A", "right")]
    layout = _layout(items, ["left", "right"])

    positions = [BoxAxisRenderer.item_position(layout, item) for item in items]
    assert positions[0] < 0.0 < positions[1]
    assert positions[0] == pytest.approx(-positions[1])


def test_a_missing_series_does_not_reserve_a_slot() -> None:
    """Category B has one series, so that item stays on B's tick."""
    items = [_item("A", "left"), _item("A", "right"), _item("B", "left")]
    layout = _layout(items, ["left", "right"])

    assert BoxAxisRenderer.item_position(layout, items[2]) == pytest.approx(1.0)


def test_width_comes_from_the_busiest_category() -> None:
    """Items are the same size across the axis; only the offsets vary."""
    crowded = [_item("A", f"s{i}") for i in range(4)] + [_item("B", "s0")]
    layout = _layout(crowded, [f"s{i}" for i in range(4)])

    assert layout["series_count"] == 4
    assert layout["item_width"] < 0.5


def test_a_single_series_keeps_the_requested_width() -> None:
    layout = _layout([_item("A", "only"), _item("B", "only")], ["only"], width=0.6)
    assert layout["item_width"] == pytest.approx(0.6)


def test_offsets_follow_the_series_order_not_the_item_order() -> None:
    """Otherwise the same series would jump sides between categories."""
    items = [_item("A", "second"), _item("A", "first"), _item("B", "second"), _item("B", "first")]
    layout = _layout(items, ["first", "second"])

    first_a = BoxAxisRenderer.item_position(layout, _item("A", "first"))
    second_a = BoxAxisRenderer.item_position(layout, _item("A", "second"))
    first_b = BoxAxisRenderer.item_position(layout, _item("B", "first"))
    second_b = BoxAxisRenderer.item_position(layout, _item("B", "second"))

    assert first_a < second_a
    assert first_b < second_b


def test_ticks_are_one_unit_apart() -> None:
    items = [_item(name, "s") for name in ("A", "B", "C")]
    layout = _layout(items, ["s"])
    assert list(layout["category_positions"].values()) == [0.0, 1.0, 2.0]


def test_an_empty_item_list_is_survivable() -> None:
    layout = BoxAxisRenderer.grouped_layout(
        items=[], category_labels=[], series_labels=[], requested_width=0.5
    )
    assert layout["series_count"] == 1
    assert layout["item_width"] == pytest.approx(0.5)
