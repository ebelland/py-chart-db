"""A renderer's ``Kwargs`` go to Matplotlib and its ``Options`` do not.

The two used to be one dict, and which was which lived in a removal list
inside each renderer: ``bar`` dropped eleven names before calling ``bar()``,
``area`` popped ``show_legend``, ``broken_bar`` popped ``band_height``.  A name
forgotten there did not fail in the editor that offered it - it failed inside
Matplotlib at draw time, as an unexpected keyword argument, on somebody's
figure.

These tests are the guard that makes the split a rule rather than a
convention: the two schemas may not overlap, and every keyword a converted
renderer declares has to be one its Matplotlib call actually accepts.  The
second half is the one that would have caught ``capthick`` on a bar chart.
"""
from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.charts import kwarg_spec  # noqa: E402
from app.charts.bar import BarAxisRenderer, HorizontalBarAxisRenderer  # noqa: E402
from app.charts.pareto import ParetoAxisRenderer  # noqa: E402
from app.charts.pie import PieAxisRenderer  # noqa: E402
from app.charts.surface import (  # noqa: E402
    SurfaceAxisRenderer,
    TriSurfaceAxisRenderer,
)
from app.charts.table import TableAxisRenderer  # noqa: E402
from app.charts.base import BaseAxisRenderer  # noqa: E402
from app.scanners.axis_renderer_scanner import (  # noqa: E402
    get_renderer,
    import_class_from_file,
    renderers,
)


def _renderer_classes() -> list[type]:
    """Every discovered renderer class, loaded the way the app loads them."""
    classes = []
    for entry in renderers:
        renderer = get_renderer(str(entry["value"]))
        assert renderer is not None, entry
        loaded = import_class_from_file(renderer)
        if loaded is not None:
            classes.append(loaded)
    return classes


ALL_RENDERERS = _renderer_classes()


# ----------------------------------------------------------------------
# The rule
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "renderer_class", ALL_RENDERERS, ids=lambda cls: str(cls.Name)
)
def test_the_two_schemas_never_overlap(renderer_class: type) -> None:
    """One name, one destination. A key in both would resolve twice."""
    overlap = set(getattr(renderer_class, "Kwargs", {})) & set(
        getattr(renderer_class, "Options", {})
    )

    assert not overlap, f"{renderer_class.Name} declares {sorted(overlap)} twice"


def test_an_unconverted_renderer_still_resolves(  # noqa: D103
) -> None:
    """The conversion is one renderer at a time, so the single-dict form has
    to keep working: opt() falls back to Kwargs when Options has no entry."""
    unconverted = [cls for cls in ALL_RENDERERS if not getattr(cls, "Options", {})]
    assert unconverted, "nothing left to check - update this test"

    renderer = next(cls for cls in unconverted if cls.Kwargs)()
    name, meta = next(iter(renderer.Kwargs.items()))

    expected = kwarg_spec.as_meta(meta).get("default")
    resolved = renderer.opt(name, {})
    assert resolved == expected or (
        kwarg_spec.is_default(expected) and resolved is None
    )


# ----------------------------------------------------------------------
# Every converted renderer, asked of Matplotlib itself
# ----------------------------------------------------------------------
#: One usable value per declared keyword, and the draw call to feed it to.
#:
#: The table is the test.  A keyword added to a converted renderer without a
#: probe fails ``test_every_declared_kwarg_has_a_probe`` rather than sliding
#: through untested, and a keyword Matplotlib does not accept fails the
#: parametrized check below - at the schema, not on somebody's figure.
def _bar(ax: Any, kwargs: dict[str, Any]) -> None:
    ax.bar([0, 1], [1.0, 2.0], **kwargs)


def _pie(ax: Any, kwargs: dict[str, Any]) -> None:
    ax.pie([1.0, 2.0, 3.0], **kwargs)


def _table(ax: Any, kwargs: dict[str, Any]) -> None:
    ax.table(cellText=[["a"]], colLabels=["c"], **kwargs)


def _surface(ax: Any, kwargs: dict[str, Any]) -> None:
    grid = np.arange(3.0)
    x, y = np.meshgrid(grid, grid)
    ax.plot_surface(x, y, x + y, **kwargs)


def _trisurface(ax: Any, kwargs: dict[str, Any]) -> None:
    grid = np.arange(3.0)
    x, y = np.meshgrid(grid, grid)
    ax.plot_trisurf(x.ravel(), y.ravel(), (x + y).ravel(), **kwargs)


ARTIST_PROBES: dict[str, Any] = {
    "alpha": 0.5,
    "zorder": 3.0,
    "visible": True,
    "rasterized": False,
    "picker": 2.0,
    "animated": False,
    "clip_on": True,
    "in_layout": True,
    "snap": True,
    "gid": "group-id",
    "url": "https://example.invalid",
}

CONVERTED: dict[str, tuple[type, dict[str, Any], Any, bool]] = {
    "Bar Chart": (
        BarAxisRenderer,
        {
            **ARTIST_PROBES,
            "facecolor": "red", "edgecolor": "blue", "linewidth": 1.0,
            "linestyle": "--", "hatch": "//", "fill": True, "color": "green",
            "align": "center", "ecolor": "black", "capsize": 3.0, "log": False,
        },
        _bar,
        False,
    ),
    "Pareto Chart": (
        ParetoAxisRenderer,
        {
            **ARTIST_PROBES,
            "facecolor": "red", "edgecolor": "blue", "linewidth": 1.0,
            "linestyle": "--", "hatch": "//", "fill": True, "color": "green",
            "align": "center", "ecolor": "black", "capsize": 3.0, "log": False,
        },
        _bar,
        False,
    ),
    "Pie Chart": (
        PieAxisRenderer,
        {
            "autopct": "%1.1f%%", "startangle": 90.0, "counterclock": True,
            "radius": 1.0, "pctdistance": 0.6, "labeldistance": 1.1,
            "shadow": False, "normalize": True,
        },
        _pie,
        False,
    ),
    "Table": (
        TableAxisRenderer,
        {"loc": "center", "cellLoc": "center", "colLoc": "center"},
        _table,
        False,
    ),
    "Surface Plot": (
        SurfaceAxisRenderer,
        {
            "cmap": "viridis", "alpha": 0.5, "edgecolor": "k",
            "linewidth": 0.0, "antialiased": True, "rstride": 1, "cstride": 1,
        },
        _surface,
        True,
    ),
    "Surface Plot (Scattered)": (
        TriSurfaceAxisRenderer,
        {
            "cmap": "viridis", "alpha": 0.5, "edgecolor": "k",
            "linewidth": 0.2, "antialiased": True,
        },
        _trisurface,
        True,
    ),
}


@pytest.mark.parametrize("chart_type", sorted(CONVERTED))
def test_every_declared_kwarg_has_a_probe(chart_type: str) -> None:
    renderer_class, probes, _draw, _is_3d = CONVERTED[chart_type]

    assert set(renderer_class.Kwargs) == set(probes)


@pytest.mark.parametrize(
    ("chart_type", "name"),
    [
        (chart_type, name)
        for chart_type, (_cls, probes, _draw, _3d) in sorted(CONVERTED.items())
        for name in sorted(probes)
    ],
)
def test_matplotlib_accepts_every_declared_kwarg(chart_type: str, name: str) -> None:
    """The check a removal list could not do: ask Matplotlib."""
    _cls, probes, draw, is_3d = CONVERTED[chart_type]
    figure = plt.figure()
    try:
        ax = figure.add_subplot(projection="3d") if is_3d else figure.add_subplot()
        draw(ax, {name: probes[name]})
    finally:
        plt.close(figure)


@pytest.mark.parametrize("chart_type", sorted(CONVERTED))
def test_no_option_is_a_matplotlib_keyword(chart_type: str) -> None:
    """The point of the split, stated once per converted renderer."""
    renderer_class, _probes, _draw, _3d = CONVERTED[chart_type]
    renderer = renderer_class()
    options = {"axis_kwargs": {name: 1 for name in renderer.Options}}

    assert not set(renderer.get_kwargs(options)) & set(renderer.Options)


# ----------------------------------------------------------------------
# Bar, in more detail
# ----------------------------------------------------------------------
BAR_KWARG_PROBES: dict[str, Any] = CONVERTED["Bar Chart"][1]


@pytest.mark.parametrize("name", ["capthick", "elinewidth", "errorevery"])
def test_the_shared_error_bar_keywords_would_have_broken_bar(name: str) -> None:
    """Why bar declares ecolor and capsize by hand instead of taking the
    shared ERROR_BAR_KWARGS wholesale: ``bar`` is not ``errorbar``."""
    figure, ax = plt.subplots()
    try:
        with pytest.raises((AttributeError, TypeError)):
            ax.bar([0, 1], [1.0, 2.0], **{name: 1.0})
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name", ["alpha", "linewidth", "color", "edgecolor", "hatch"])
def test_an_untouched_axis_leaves_the_styled_keywords_alone(name: str) -> None:
    """``alpha`` used to be 0.9 here, so every bar chart was 90% opaque
    whatever the .mplstyle said, and ``linewidth`` 1.8 in stairs and 1.6 in
    time series against nine shipped styles that set lines.linewidth."""
    assert name not in BarAxisRenderer().get_kwargs({})


def test_the_booleans_keep_their_matplotlib_default() -> None:
    """Not left unset, deliberately: a checkbox has two states, and one that
    reads False while the artist behaves as True is a lie. They go back to
    being unset the day the panel can draw a third state."""
    forwarded = BarAxisRenderer().get_kwargs({})

    assert forwarded["fill"] is True
    assert forwarded["visible"] is True
    assert forwarded["animated"] is False


def test_a_surface_takes_its_colormap_from_the_style() -> None:
    """Hard-coded to viridis in three renderers, which is a colour decision
    taken away from the style sheet - but a surface has to have *some* map,
    so the fallback is the style's rather than Matplotlib's flat colour."""
    from matplotlib import rcParams

    from app.charts.surface import _surface_kwargs

    assert "cmap" not in SurfaceAxisRenderer().get_kwargs({})
    assert _surface_kwargs(SurfaceAxisRenderer(), {})["cmap"] == rcParams["image.cmap"]


def test_a_pie_keeps_the_defaults_it_means_to_override() -> None:
    """Matplotlib starts at 3 o'clock and draws no percentages; both are
    deliberate deviations here, so both stay literal in the schema."""
    forwarded = PieAxisRenderer().get_kwargs({})

    assert forwarded["startangle"] == 90.0
    assert forwarded["autopct"] == "%1.1f%%"


def test_the_horizontal_bar_inherits_both_schemas() -> None:
    assert HorizontalBarAxisRenderer.Kwargs == BarAxisRenderer.Kwargs
    assert HorizontalBarAxisRenderer.Options == BarAxisRenderer.Options


# ----------------------------------------------------------------------
# The resolution chain
# ----------------------------------------------------------------------
def test_a_series_style_beats_the_axis() -> None:
    renderer = BarAxisRenderer()
    merged = renderer.merge_style(
        {"axis_kwargs": {"alpha": 0.2, "hatch": "//"}},
        {"axis_kwargs": {"alpha": 0.9}},
    )

    kwargs = renderer.get_kwargs(merged)
    assert kwargs["alpha"] == 0.9
    # Merged key by key: the series overriding alpha keeps the axis hatch.
    assert kwargs["hatch"] == "//"


def test_the_flat_legacy_location_is_still_read() -> None:
    """Figures saved before axis_kwargs existed keep their appearance."""
    assert BarAxisRenderer().get_kwargs({"hatch": "xx"})["hatch"] == "xx"


def test_edited_text_becomes_the_number_the_schema_declares() -> None:
    """The panel writes text; five renderers converted it back by hand."""
    kwargs = BarAxisRenderer().get_kwargs({"axis_kwargs": {"linewidth": "2.5"}})

    assert kwargs["linewidth"] == 2.5
    assert isinstance(kwargs["linewidth"], float)


def test_a_value_that_will_not_convert_is_dropped() -> None:
    """Not forwarded raw for Matplotlib to choke on: a typo should cost the
    option, not the chart. The dropped name is logged at debug with the
    renderer that declared it."""
    assert "alpha" not in BarAxisRenderer().get_kwargs({"axis_kwargs": {"alpha": "wide"}})


@pytest.mark.parametrize(
    ("text", "expected"),
    [("yes", True), ("on", True), ("1", True), ("no", False), ("off", False)],
)
def test_a_boolean_written_as_a_word_is_understood(text: str, expected: bool) -> None:
    """A checkbox, a config file and a hand-typed option each spell a boolean
    differently, and all three end up in the same schema entry."""
    forwarded = BarAxisRenderer().get_kwargs({"axis_kwargs": {"fill": text}})

    assert forwarded["fill"] is expected


def test_a_number_written_as_a_word_is_not_a_boolean() -> None:
    """``bool("wide")`` is True, which is how a typo becomes a silent setting."""
    assert "fill" not in BarAxisRenderer().get_kwargs({"axis_kwargs": {"fill": "wide"}})


@pytest.mark.parametrize("empty", [None, "", kwarg_spec.DEFAULT])
def test_an_unset_keyword_is_omitted_rather_than_forwarded(empty: object) -> None:
    """picker=None reaches set_pickradius(None) and raises; absent does not."""
    assert "picker" not in BarAxisRenderer().get_kwargs({"axis_kwargs": {"picker": empty}})


def test_base_declares_both_schemas_empty() -> None:
    assert BaseAxisRenderer.Kwargs == {}
    assert BaseAxisRenderer.Options == {}
