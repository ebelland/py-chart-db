"""Colour columns are mapped in one NumPy call, not one call per point.

todo.txt P0-11: ``map_continuous_colors_to_cmap`` called ``cmap(value)``
once per point and built a Python list of RGBA tuples, and
``map_integer_colors_to_palette`` did the index arithmetic per point in
Python. Measured on this machine before the change: 300 000 points took
2 076 ms and 144 ms; after, 3.9 ms and 3.8 ms.

Timing is not what is asserted here - a wall-clock threshold fails on a
loaded machine and passes on a fast one whatever the code does. The
property that made it fast is asserted instead: the colormap is called
*once*, whatever the row count. The rest of the file pins the behaviour
that had to stay identical while the implementation changed, and the two
return types the drawing code now relies on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from app.charts.scatter import ScatterAxisRenderer


@pytest.fixture
def renderer() -> ScatterAxisRenderer:
    return ScatterAxisRenderer()


# ----------------------------------------------------------------------
# The property that makes it fast
# ----------------------------------------------------------------------
def test_the_colormap_is_called_once_for_the_whole_column(
    renderer: ScatterAxisRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once, not once per point - which is the whole of P0-11."""
    import app.charts.base as base

    calls: list[int] = []
    real = base.colormaps.get_cmap("viridis")

    def counting_cmap(values):
        calls.append(np.size(values))
        return real(values)

    monkeypatch.setattr(base.colormaps, "get_cmap", lambda _name: counting_cmap)

    renderer.map_continuous_colors_to_cmap(pd.Series(np.linspace(0.0, 1.0, 5_000)))

    assert calls == [5_000]


def test_the_palette_lookup_does_no_python_arithmetic_per_point(
    renderer: ScatterAxisRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The integer path's equivalent: the palette is read once, and the
    index arithmetic happens over the array rather than per element."""
    reads: list[int] = []
    original = ScatterAxisRenderer.palette_colors

    def counting_palette(self):
        reads.append(1)
        return original(self)

    monkeypatch.setattr(ScatterAxisRenderer, "palette_colors", counting_palette)

    renderer.map_integer_colors_to_palette(pd.Series(np.arange(5_000) % 7 + 1))

    assert reads == [1]


# ----------------------------------------------------------------------
# What each path returns
# ----------------------------------------------------------------------
def test_the_continuous_path_returns_an_rgba_array(
    renderer: ScatterAxisRenderer,
) -> None:
    """(n, 4) floats: what a Colormap produces for an array, and what
    Matplotlib wants wherever a per-point colour is accepted."""
    mapped = renderer.map_continuous_colors_to_cmap(pd.Series([0.0, 0.5, 1.0]))

    assert isinstance(mapped, np.ndarray)
    assert mapped.shape == (3, 4)
    assert mapped.dtype == float


def test_the_discrete_path_returns_the_palettes_own_colours(
    renderer: ScatterAxisRenderer,
) -> None:
    """A list, not an array: these are the style sheet's colour specs, which
    may be strings, and turning them into RGBA would lose that."""
    palette = renderer.palette_colors()

    mapped = renderer.map_integer_colors_to_palette(pd.Series([1, 2, 3]))

    assert isinstance(mapped, list)
    assert mapped == [palette[0], palette[1], palette[2]]


def test_one_colour_comes_back_as_a_tuple_not_a_row(
    renderer: ScatterAxisRenderer,
) -> None:
    """first_color_from_values is a scalar accessor, and its result reaches
    call sites that test it with ``if color:`` - which raises on an array."""
    color = renderer.first_color_from_values(pd.Series([1.5, 2.5]))

    assert isinstance(color, tuple)
    assert len(color) == 4
    assert bool(color) is True


def test_no_rows_gives_the_fallback_back(renderer: ScatterAxisRenderer) -> None:
    color = renderer.first_color_from_values(
        pd.Series([], dtype=float), fallback_color="#00ff00"
    )

    assert color == "#00ff00"


# ----------------------------------------------------------------------
# Behaviour that had to stay identical
# ----------------------------------------------------------------------
def test_the_ends_of_the_range_are_the_ends_of_the_colormap(
    renderer: ScatterAxisRenderer,
) -> None:
    mapped = renderer.map_continuous_colors_to_cmap(pd.Series([10.0, 15.0, 20.0]))

    import matplotlib

    viridis = matplotlib.colormaps.get_cmap("viridis")
    assert np.allclose(mapped[0], viridis(0.0))
    assert np.allclose(mapped[1], viridis(0.5))
    assert np.allclose(mapped[2], viridis(1.0))


def test_a_column_with_no_spread_lands_in_the_middle(
    renderer: ScatterAxisRenderer,
) -> None:
    """Dividing by a zero span would be a NaN; the midpoint is the answer
    the per-point version gave and there is no better one."""
    mapped = renderer.map_continuous_colors_to_cmap(pd.Series([5.0, 5.0, 5.0]))

    import matplotlib

    middle = matplotlib.colormaps.get_cmap("viridis")(0.5)
    assert np.allclose(mapped, np.tile(middle, (3, 1)))


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_a_value_off_the_scale_takes_the_fallback(
    renderer: ScatterAxisRenderer, missing: float
) -> None:
    """Not the colormap's "bad" colour, which is what a NaN reaching it
    would produce."""
    mapped = renderer.map_continuous_colors_to_cmap(
        pd.Series([0.0, missing, 1.0]), fallback_color="#ff0000"
    )

    assert np.allclose(mapped[1], to_rgba("#ff0000"))


def test_a_column_that_is_entirely_unusable_is_all_fallback(
    renderer: ScatterAxisRenderer,
) -> None:
    mapped = renderer.map_continuous_colors_to_cmap(
        pd.Series([np.nan, np.nan]), fallback_color="#ff0000"
    )

    assert mapped.shape == (2, 4)
    assert np.allclose(mapped, np.tile(to_rgba("#ff0000"), (2, 1)))


def test_category_ids_cycle_through_the_palette(
    renderer: ScatterAxisRenderer,
) -> None:
    """1-based, and wrapping: id 1 is the first colour, and the id after the
    last one starts again."""
    palette = renderer.palette_colors()
    ids = pd.Series([1, len(palette), len(palette) + 1])

    mapped = renderer.map_integer_colors_to_palette(ids)

    assert mapped == [palette[0], palette[-1], palette[0]]


def test_a_missing_category_id_takes_the_fallback(
    renderer: ScatterAxisRenderer,
) -> None:
    mapped = renderer.map_integer_colors_to_palette(
        pd.Series([1.0, np.nan, 2.0]), "#ff0000"
    )

    assert mapped[1] == "#ff0000"


def test_a_fallback_that_is_a_tuple_survives_the_lookup_table(
    renderer: ScatterAxisRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A style sheet may write its cycle as RGB tuples, and the fallback
    then is one. Assigning a tuple into masked slots of an object array
    broadcasts it element by element, which is why it is a table entry."""
    monkeypatch.setattr(
        ScatterAxisRenderer,
        "palette_colors",
        lambda self: [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
    )

    mapped = renderer.map_integer_colors_to_palette(pd.Series([1.0, np.nan]))

    assert mapped == [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]


def test_an_empty_column_maps_to_nothing(renderer: ScatterAxisRenderer) -> None:
    assert renderer.map_integer_colors_to_palette(pd.Series([], dtype=float)) == []
    assert renderer.map_continuous_colors_to_cmap(
        pd.Series([], dtype=float)
    ).shape == (0, 4)


# ----------------------------------------------------------------------
# The dispatcher
# ----------------------------------------------------------------------
def test_integers_take_the_palette_and_floats_the_colormap(
    renderer: ScatterAxisRenderer,
) -> None:
    """Whole numbers stored as floats are still continuous - the dtype is
    the decision, not the values."""
    discrete = renderer.color_sequence_from_values(pd.Series([1, 2, 3]))
    continuous = renderer.color_sequence_from_values(pd.Series([1.0, 2.0, 3.0]))

    assert isinstance(discrete, list)
    assert isinstance(continuous, np.ndarray)
