"""Six renderers added in one go: stem, quiver, streamplot, triplot,
tripcolor and 3D scatter.

``test_render_all_renderers.py`` already draws every renderer once through
the full descriptor pipeline and saves the picture. What that cannot say is
what each of these does with input that is *not* the showcase's: a field
that is not on a grid, a colour column that no longer lines up with its
points, three points where a triangulation needs three. Those are the paths
worth pinning, because each of them is a choice - refuse and say why, or
draw something the data does not support.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from app.charts.area import StemAxisRenderer
from app.charts.base import SeriesData
from app.charts.surface import Scatter3DAxisRenderer
from app.charts.triangulation import TriplotAxisRenderer, TripcolorAxisRenderer
from app.charts.vector_field import QuiverAxisRenderer, StreamAxisRenderer


def _series(df: pd.DataFrame, name: str = "s", **style: object) -> SeriesData:
    return SeriesData(name=name, df=df, style=dict(style))


def _axes(*, projection: str | None = None):
    figure = Figure()
    return figure, figure.add_subplot(projection=projection)


@pytest.fixture
def field() -> pd.DataFrame:
    """A rotating vector field on an even 10x10 grid."""
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 6.0, 10), np.linspace(0.0, 6.0, 10))
    return pd.DataFrame(
        {
            "x": grid_x.ravel(),
            "y": grid_y.ravel(),
            "u": np.sin(grid_y).ravel(),
            "v": np.cos(grid_x).ravel(),
        }
    )


@pytest.fixture
def cloud() -> pd.DataFrame:
    """Scattered points sampling the same ripple the showcase uses."""
    rng = np.random.default_rng(11)
    x = rng.uniform(-3.0, 3.0, 200)
    y = rng.uniform(-3.0, 3.0, 200)
    return pd.DataFrame({"x": x, "y": y, "z": np.sin(np.hypot(x, y))})


# ----------------------------------------------------------------------
# Stem
# ----------------------------------------------------------------------
def test_stem_draws_a_container_per_series() -> None:
    _figure, ax = _axes()
    frame = pd.DataFrame({"x": np.arange(10.0), "y": np.sin(np.arange(10.0))})

    StemAxisRenderer().render_axis(ax, [_series(frame), _series(frame, label="b")], {})

    assert len(ax.containers) == 2


def test_stem_forwards_no_keyword_matplotlib_would_refuse() -> None:
    """``ax.stem`` is a helper with a fixed signature, not an artist taking
    the usual properties - alpha or zorder would raise, so they are not
    offered."""
    offered = set(StemAxisRenderer.Kwargs)

    assert offered == {
        "label", "linefmt", "markerfmt", "basefmt", "bottom", "orientation"
    }


def test_stem_drops_empty_format_strings() -> None:
    """"" is a format meaning "no colour, no style, no marker" and draws
    nothing; absent means "use the defaults"."""
    _figure, ax = _axes()
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})

    StemAxisRenderer().render_axis(
        ax, [_series(frame)], {"linefmt": "", "markerfmt": "", "basefmt": ""}
    )

    assert len(ax.containers) == 1


def test_stem_skips_a_series_with_no_finite_pairs() -> None:
    _figure, ax = _axes()
    frame = pd.DataFrame({"x": [np.nan, np.nan], "y": [1.0, 2.0]})

    StemAxisRenderer().render_axis(ax, [_series(frame)], {})

    assert len(ax.containers) == 0


# ----------------------------------------------------------------------
# Quiver
# ----------------------------------------------------------------------
def test_quiver_draws_the_field(field: pd.DataFrame) -> None:
    _figure, ax = _axes()

    QuiverAxisRenderer().render_axis(ax, [_series(field)], {})

    assert len(ax.collections) == 1


def test_quiver_does_not_need_a_grid(field: pd.DataFrame) -> None:
    """The difference from Stream Plot, and the reason both exist."""
    scattered = field.sample(frac=0.5, random_state=2)
    _figure, ax = _axes()

    QuiverAxisRenderer().render_axis(ax, [_series(scattered)], {})

    assert len(ax.collections) == 1


def test_quiver_colours_by_a_mapped_column(field: pd.DataFrame) -> None:
    coloured = field.assign(color=np.hypot(field["u"], field["v"]))
    _figure, ax = _axes()

    QuiverAxisRenderer().render_axis(ax, [_series(coloured)], {})

    quiver = ax.collections[0]
    assert quiver.get_array() is not None
    assert len(quiver.get_array()) == len(field)


def test_quiver_ignores_a_colour_column_that_does_not_line_up(
    field: pd.DataFrame,
) -> None:
    """Colouring the wrong arrows is worse than colouring none of them."""
    ragged = field.assign(color=np.nan)
    _figure, ax = _axes()

    QuiverAxisRenderer().render_axis(ax, [_series(ragged)], {})

    assert ax.collections[0].get_array() is None


def test_quiver_skips_a_series_with_no_finite_vectors() -> None:
    _figure, ax = _axes()
    frame = pd.DataFrame(
        {"x": [0.0], "y": [0.0], "u": [np.nan], "v": [1.0]}
    )

    QuiverAxisRenderer().render_axis(ax, [_series(frame)], {})

    assert len(ax.collections) == 0


# ----------------------------------------------------------------------
# Stream plot
# ----------------------------------------------------------------------
def test_streamplot_draws_lines_and_arrows(field: pd.DataFrame) -> None:
    _figure, ax = _axes()

    StreamAxisRenderer().render_axis(ax, [_series(field)], {})

    # One LineCollection for the streamlines, one arrow patch per line.
    assert len(ax.collections) == 1
    assert ax.patches


def test_streamplot_refuses_scattered_samples(field: pd.DataFrame) -> None:
    """It interpolates *between* samples, so a non-grid would mean drawing
    streamlines through data nobody measured."""
    scattered = field.sample(frac=0.5, random_state=5)
    _figure, ax = _axes()

    StreamAxisRenderer().render_axis(ax, [_series(scattered)], {})

    assert len(ax.collections) == 0
    assert len(ax.patches) == 0


def test_streamplot_refuses_an_unevenly_spaced_grid(field: pd.DataFrame) -> None:
    """A complete grid is not enough: Matplotlib's implementation can only
    interpolate on a regular one."""
    uneven = field.copy()
    uneven["x"] = uneven["x"] ** 2  # still a complete grid, no longer even
    _figure, ax = _axes()

    StreamAxisRenderer().render_axis(ax, [_series(uneven)], {})

    assert len(ax.collections) == 0


def test_streamplot_colours_by_magnitude_when_no_colour_is_set(
    field: pd.DataFrame,
) -> None:
    _figure, ax = _axes()

    StreamAxisRenderer().render_axis(ax, [_series(field)], {})

    assert ax.collections[0].get_array() is not None


def test_an_explicit_colour_turns_the_magnitude_mapping_off(
    field: pd.DataFrame,
) -> None:
    _figure, ax = _axes()

    StreamAxisRenderer().render_axis(ax, [_series(field)], {"color": "#cc0000"})

    assert ax.collections[0].get_array() is None


def test_streamplot_draws_one_field_only(field: pd.DataFrame) -> None:
    _figure, ax = _axes()

    StreamAxisRenderer().render_axis(ax, [_series(field), _series(field, )], {})

    assert len(ax.collections) == 1


def test_an_infinite_sample_is_read_as_no_flow(field: pd.DataFrame) -> None:
    """streamplot draws nothing at all when it meets a non-finite value, and
    an infinity survives the grid test - unlike a NaN, which makes the rows
    an incomplete grid and is refused before it gets here."""
    spoiled = field.copy()
    spoiled.loc[spoiled.index[40], "u"] = np.inf
    _figure, ax = _axes()

    StreamAxisRenderer().render_axis(ax, [_series(spoiled)], {})

    assert len(ax.collections) == 1


def test_a_missing_sample_is_refused_as_a_hole_in_the_grid(
    field: pd.DataFrame,
) -> None:
    """A NaN leaves the rows one short of a complete grid, which is what
    "not a grid" means - the renderer says so and names Quiver."""
    holed = field.copy()
    holed.loc[holed.index[40], "u"] = np.nan
    _figure, ax = _axes()

    StreamAxisRenderer().render_axis(ax, [_series(holed)], {})

    assert len(ax.collections) == 0


# ----------------------------------------------------------------------
# Triangulation
# ----------------------------------------------------------------------
def test_triplot_draws_the_mesh(cloud: pd.DataFrame) -> None:
    _figure, ax = _axes()

    TriplotAxisRenderer().render_axis(ax, [_series(cloud)], {})

    # triplot returns the edges and the vertex markers as two Line2D.
    assert len(ax.lines) == 2


def test_triplot_needs_no_value_column(cloud: pd.DataFrame) -> None:
    _figure, ax = _axes()

    TriplotAxisRenderer().render_axis(ax, [_series(cloud.drop(columns=["z"]))], {})

    assert ax.lines


def test_triplot_refuses_fewer_than_three_points() -> None:
    _figure, ax = _axes()
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})

    TriplotAxisRenderer().render_axis(ax, [_series(frame)], {})

    assert len(ax.lines) == 0


def test_each_triplot_series_takes_the_next_cycle_colour(
    cloud: pd.DataFrame,
) -> None:
    _figure, ax = _axes()

    TriplotAxisRenderer().render_axis(
        ax, [_series(cloud, name="a"), _series(cloud.iloc[::2], name="b")], {}
    )

    colours = {line.get_color() for line in ax.lines}
    assert len(colours) > 1, "both meshes were drawn in the same colour"


def test_tripcolor_fills_the_mesh(cloud: pd.DataFrame) -> None:
    _figure, ax = _axes()

    TripcolorAxisRenderer().render_axis(ax, [_series(cloud)], {})

    assert len(ax.collections) == 1


def test_tripcolor_adds_no_colorbar_unless_asked(cloud: pd.DataFrame) -> None:
    """It is a second Axes taken out of this one's space, so a figure laid
    out without one should not reflow because a mesh was added."""
    figure, ax = _axes()

    TripcolorAxisRenderer().render_axis(ax, [_series(cloud)], {})

    assert len(figure.axes) == 1


def test_tripcolor_labels_the_colorbar_it_was_asked_for(
    cloud: pd.DataFrame,
) -> None:
    figure, ax = _axes()

    TripcolorAxisRenderer().render_axis(
        ax, [_series(cloud)], {"colorbar": True, "colorbar_label": "height"}
    )

    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylabel() == "height"


def test_tripcolor_draws_one_field_only(cloud: pd.DataFrame) -> None:
    _figure, ax = _axes()

    TripcolorAxisRenderer().render_axis(ax, [_series(cloud), _series(cloud)], {})

    assert len(ax.collections) == 1


# ----------------------------------------------------------------------
# 3D scatter
# ----------------------------------------------------------------------
def test_scatter3d_draws_the_cloud(cloud: pd.DataFrame) -> None:
    _figure, ax = _axes(projection="3d")

    Scatter3DAxisRenderer().render_axis(ax, [_series(cloud)], {})

    assert len(ax.collections) == 1


def test_scatter3d_maps_a_colour_column_through_the_colormap(
    cloud: pd.DataFrame,
) -> None:
    _figure, ax = _axes(projection="3d")

    Scatter3DAxisRenderer().render_axis(
        ax, [_series(cloud.assign(color=cloud["z"]))], {}
    )

    assert ax.collections[0].get_array() is not None


def test_scatter3d_ignores_a_colour_column_that_does_not_line_up(
    cloud: pd.DataFrame,
) -> None:
    _figure, ax = _axes(projection="3d")

    Scatter3DAxisRenderer().render_axis(
        ax, [_series(cloud.assign(color=np.nan))], {}
    )

    assert ax.collections[0].get_array() is None


def test_scatter3d_drops_the_rows_it_has_no_coordinates_for(
    cloud: pd.DataFrame,
) -> None:
    holed = cloud.copy()
    holed.loc[holed.index[:20], "z"] = np.nan
    _figure, ax = _axes(projection="3d")

    Scatter3DAxisRenderer().render_axis(ax, [_series(holed)], {})

    assert len(ax.collections[0].get_offsets()) == len(cloud) - 20


def test_scatter3d_applies_the_camera_it_was_given(cloud: pd.DataFrame) -> None:
    _figure, ax = _axes(projection="3d")

    Scatter3DAxisRenderer().render_axis(ax, [_series(cloud)], {"elev": 12.0, "azim": 45.0})

    assert round(float(ax.elev)) == 12
    assert round(float(ax.azim)) == 45


def test_scatter3d_asks_for_a_3d_axes_when_the_chart_is_created() -> None:
    """The renderer cannot create its own axes - render_figure reads the
    projection off the descriptor, which the create dialog writes."""
    from app.dialogs.create_chart_dialog import NewPlotTabDialog

    assert "Scatter Plot (3D)" in NewPlotTabDialog.CHART_TYPES_NEEDING_3D_AXES
