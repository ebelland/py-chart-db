"""Options declare a type, and ``get_kwargs`` is where it is honoured.

Every option in a renderer's ``Kwargs`` carries a ``type``, but the value that
arrives almost never has it: the axis options editor stores what was typed and
a saved descriptor stores what it was given, so ``rstride`` reaches the
renderer as ``"2"`` rather than ``2``.

Matplotlib does not coerce. ``plot_surface`` computes ``(rows - 1) % rstride``
and raises *unsupported operand type(s) for %: 'int' and 'str'*; a string
``linewidth`` gets further, into the C++ layer, and fails there with a message
naming nothing the user typed. Both were reachable by typing a number into a
box that asked for one.

The declared type is applied only to what is forwarded to Matplotlib -
renderer-owned options are parsed by the renderer that owns them, and contour
``levels`` is the reason why: it is deliberately either a count or a list.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.base import BaseAxisRenderer, SeriesData
from app.charts.contour import ContourAxisRenderer
from app.charts.surface import SurfaceAxisRenderer


def _grid_frame(side: int = 12) -> pd.DataFrame:
    axis = np.linspace(-2.0, 2.0, side)
    x_grid, y_grid = np.meshgrid(axis, axis)
    return pd.DataFrame(
        {
            "x": x_grid.ravel(),
            "y": y_grid.ravel(),
            "z": np.sin(x_grid).ravel() + np.cos(y_grid).ravel(),
        }
    )


# ----------------------------------------------------------------------
# The conversion itself
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "declared", "expected"),
    [
        ("2", int, 2),
        ("2.7", int, 2),
        (2.7, int, 2),
        ("0", float, 0.0),
        ("1.5", float, 1.5),
        (3, float, 3.0),
        ("true", bool, True),
        ("False", bool, False),
        ("on", bool, True),
        ("0", bool, False),
    ],
)
def test_a_typed_string_becomes_the_declared_type(value, declared, expected) -> None:
    assert BaseAxisRenderer._coerce_option(value, {"type": declared}) == expected


def test_a_value_of_the_right_type_is_left_alone() -> None:
    assert BaseAxisRenderer._coerce_option(True, {"type": bool}) is True
    assert BaseAxisRenderer._coerce_option(4, {"type": int}) == 4


def test_a_bool_where_a_number_was_declared_is_not_turned_into_one() -> None:
    """Almost certainly a mis-declared option, and worth leaving visible."""
    assert BaseAxisRenderer._coerce_option(True, {"type": float}) is True


def test_an_undeclared_type_passes_through_untouched() -> None:
    assert BaseAxisRenderer._coerce_option("viridis", {"type": str}) == "viridis"
    assert BaseAxisRenderer._coerce_option("anything", {}) == "anything"


def test_nonsense_is_reported_as_unconvertible() -> None:
    assert (
        BaseAxisRenderer._coerce_option("abc", {"type": int})
        is BaseAxisRenderer._UNCONVERTIBLE
    )


def test_empty_is_not_nonsense() -> None:
    """Empty means "not set" to every caller downstream, which drops it."""
    assert BaseAxisRenderer._coerce_option("", {"type": float}) == ""


# ----------------------------------------------------------------------
# What reaches Matplotlib
# ----------------------------------------------------------------------
def test_get_kwargs_converts_what_it_forwards() -> None:
    kwargs = SurfaceAxisRenderer().get_kwargs(
        {"rstride": "2", "cstride": "3", "linewidth": "0", "antialiased": "true"}
    )

    assert kwargs["rstride"] == 2
    assert kwargs["cstride"] == 3
    assert kwargs["linewidth"] == 0.0
    assert kwargs["antialiased"] is True


def test_an_unconvertible_option_is_dropped_rather_than_forwarded() -> None:
    """A typo should cost the option, not the chart."""
    kwargs = SurfaceAxisRenderer().get_kwargs({"rstride": "every other one"})

    assert "rstride" not in kwargs


def test_a_renderer_owned_option_is_still_read_raw() -> None:
    """Contour levels is a count *or* a list, which no scalar type describes.

    ``opt`` is deliberately not coerced for exactly this: the renderer that
    owns an option is the one that knows how to read it.
    """
    renderer = ContourAxisRenderer()
    raw = renderer.opt("levels", {"levels": "0.5, 2, 5, 15"})

    assert raw == "0.5, 2, 5, 15"


# ----------------------------------------------------------------------
# The crashes this prevents
# ----------------------------------------------------------------------
def test_a_surface_renders_with_its_numbers_typed_as_text() -> None:
    """The original failure: TypeError on (rows - 1) % rstride."""
    figure = Figure(figsize=(5.0, 4.0))
    ax = figure.add_subplot(1, 1, 1, projection="3d")

    SurfaceAxisRenderer().render_axis(
        ax=ax,
        series=[SeriesData(name="field", df=_grid_frame(), style={})],
        options={
            "projection": "3d",
            "rstride": "2",
            "cstride": "2",
            "linewidth": "0",
            "elev": "40",
            "azim": "-60",
        },
    )
    figure.savefig(io.BytesIO(), format="png")

    assert ax.has_data()


def test_a_contour_renders_with_its_numbers_typed_as_text() -> None:
    figure = Figure(figsize=(5.0, 4.0))
    ax = figure.add_subplot(1, 1, 1)

    ContourAxisRenderer().render_axis(
        ax=ax,
        series=[SeriesData(name="field", df=_grid_frame(), style={})],
        options={"levels": "8", "linewidths": "0.5", "filled": True},
    )
    figure.savefig(io.BytesIO(), format="png")

    assert ax.has_data()
