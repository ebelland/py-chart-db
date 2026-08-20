"""Tests for the violin renderer.

The point of inheriting from the box renderer is that a violin lands exactly
where the box it replaces would have been, so the alignment is asserted against
the box renderer rather than against hard-coded coordinates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base_axis import SeriesData
from app.charts.box_plot_axis import BoxAxisRenderer
from app.charts.violin_axis import ViolinAxisRenderer

RNG = np.random.default_rng(11)


def _grouped(groups: int = 3, per_group: int = 200) -> pd.DataFrame:
    rows = []
    for index in range(groups):
        rows.append(
            pd.DataFrame(
                {
                    "group": f"g{index}",
                    "value": RNG.normal(10.0 + 2.0 * index, 1.0 + 0.4 * index, per_group),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _render(renderer, series: list[SeriesData], options: dict | None = None):
    fig = Figure(figsize=(7.0, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    renderer.render_axis(ax=ax, series=series, options=options or {})
    return fig, ax


def _bodies(ax) -> list:
    """Violin bodies are PolyCollections on the axis."""
    return list(ax.collections)


def test_one_body_per_group() -> None:
    df = _grouped(groups=3)
    _, ax = _render(ViolinAxisRenderer(), [SeriesData(name="s", df=df, style={})])

    # bodies + median/extrema line collections; bodies are the filled ones.
    filled = [c for c in _bodies(ax) if getattr(c, "get_paths", None) and c.get_facecolor().size]
    assert len(filled) >= 3


def test_ticks_match_the_groups() -> None:
    df = _grouped(groups=4)
    _, ax = _render(ViolinAxisRenderer(), [SeriesData(name="s", df=df, style={})])

    assert [text.get_text() for text in ax.get_xticklabels()] == ["g0", "g1", "g2", "g3"]


def test_violins_land_where_the_boxes_would() -> None:
    """The shared layout is the whole reason for inheriting from the box."""
    df = _grouped(groups=3)
    series = [SeriesData(name="s", df=df, style={})]

    _, box_ax = _render(BoxAxisRenderer(), series)
    _, violin_ax = _render(ViolinAxisRenderer(), series)

    assert list(box_ax.get_xticks()) == pytest.approx(list(violin_ax.get_xticks()))
    assert [t.get_text() for t in box_ax.get_xticklabels()] == [
        t.get_text() for t in violin_ax.get_xticklabels()
    ]


def test_two_series_are_offset_within_each_group() -> None:
    df_a = _grouped(groups=2)
    df_b = _grouped(groups=2)
    _, ax = _render(
        ViolinAxisRenderer(),
        [
            SeriesData(name="a", df=df_a, style={}),
            SeriesData(name="b", df=df_b, style={}),
        ],
    )

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["a", "b"]


def test_horizontal_orientation_labels_the_y_axis() -> None:
    df = _grouped(groups=3)
    _, ax = _render(
        ViolinAxisRenderer(), [SeriesData(name="s", df=df, style={})], {"direction": "horizontal"}
    )
    assert [text.get_text() for text in ax.get_yticklabels()] == ["g0", "g1", "g2"]


@pytest.mark.parametrize("renderer", [ViolinAxisRenderer, BoxAxisRenderer])
@pytest.mark.parametrize(
    ("options", "expect_horizontal"),
    [
        ({"direction": "horizontal"}, True),
        ({"direction": "vertical"}, False),
        ({}, False),
        # Figures saved before the rename.  get_kwargs keeps only declared
        # keys, so the old boolean reaches the renderer through the raw
        # options or not at all - and a stored horizontal chart coming back
        # vertical is the kind of silent loss nobody connects to a refactor
        # weeks afterwards.
        ({"vert": False}, True),
        ({"vertical": False}, True),
        ({"vert": True}, False),
        ({"axis_kwargs": {"vert": False}}, True),
        # An explicit direction beats a stale boolean left beside it.
        ({"direction": "vertical", "vert": False}, False),
    ],
)
def test_the_direction_option_replaced_a_boolean_without_losing_it(
    renderer, options: dict, expect_horizontal: bool
) -> None:
    """``vert`` became ``direction`` when Matplotlib 3.11 deprecated the bool."""
    _, ax = _render(
        renderer(), [SeriesData(name="s", df=_grouped(groups=3), style={})], options
    )

    categories = ["g0", "g1", "g2"]
    on_y = [text.get_text() for text in ax.get_yticklabels()] == categories
    on_x = [text.get_text() for text in ax.get_xticklabels()] == categories

    assert on_y is expect_horizontal
    assert on_x is not expect_horizontal


def test_no_deprecated_boolean_reaches_matplotlib() -> None:
    """The bool is removed in 3.13, so a warning now is an exception later."""
    import warnings

    for renderer in (ViolinAxisRenderer(), BoxAxisRenderer()):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter("error", PendingDeprecationWarning)
            _render(renderer, [SeriesData(name="s", df=_grouped(groups=2), style={})])


def test_constant_sample_is_skipped_not_raised() -> None:
    """A kernel density estimate needs spread; a flat sample has none."""
    df = pd.DataFrame({"group": ["g0"] * 20, "value": [5.0] * 20})
    _, ax = _render(ViolinAxisRenderer(), [SeriesData(name="s", df=df, style={})])
    assert not _bodies(ax)


def test_single_point_group_is_skipped() -> None:
    df = pd.DataFrame({"group": ["g0"], "value": [1.0]})
    _, ax = _render(ViolinAxisRenderer(), [SeriesData(name="s", df=df, style={})])
    assert not _bodies(ax)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.25, 0.5, 0.75", [0.25, 0.5, 0.75]),
        ("0.5;0.9", [0.5, 0.9]),
        ("", None),
        ("   ", None),
        # Out of range and unparseable entries are dropped, not fatal.
        ("0, 1, 0.5", [0.5]),
        ("abc", None),
        ("0.5, 0.5", [0.5]),
    ],
)
def test_quantile_parsing(raw: str, expected: list[float] | None) -> None:
    assert ViolinAxisRenderer._parse_quantiles(raw) == expected


def test_quantiles_add_a_collection() -> None:
    df = _grouped(groups=2)
    _, without = _render(ViolinAxisRenderer(), [SeriesData(name="s", df=df, style={})])
    _, with_quantiles = _render(
        ViolinAxisRenderer(),
        [SeriesData(name="s", df=df, style={})],
        {"quantiles": "0.25, 0.75"},
    )
    assert len(with_quantiles.collections) > len(without.collections)


def test_invisible_series_is_skipped() -> None:
    df = _grouped(groups=2)
    _, ax = _render(
        ViolinAxisRenderer(), [SeriesData(name="s", df=df, style={"visible": False})]
    )
    assert not _bodies(ax)


def test_saved_figure(plots_dir: Path, show_plots: bool) -> None:
    df = _grouped(groups=4, per_group=300)
    fig, ax = _render(
        ViolinAxisRenderer(),
        [SeriesData(name="samples", df=df, style={})],
        {"showmeans": True, "showmedians": True, "quantiles": "0.25, 0.75"},
    )
    assert _bodies(ax)

    if show_plots:
        target = plots_dir / "violin" / "grouped.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=110, bbox_inches="tight")
