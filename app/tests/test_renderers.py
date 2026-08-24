"""Tests for the static AST-based axis-renderer scanner."""
from __future__ import annotations

from app.scanners.axis_renderer_scanner import (
    CHART_TYPE_ALIASES,
    _discover_axis_renderers,
    get_renderer,
    import_class_from_file,
    renderers,
)


def test_discovery_finds_renderers() -> None:
    """The scanner must find the renderers shipped in app/charts."""
    found = _discover_axis_renderers()
    assert found, "no axis renderers discovered"

    values = {entry["value"] for entry in found}
    assert {"Scatter Plot", "Time Series"} <= values


def test_discovery_matches_module_level_cache() -> None:
    """The module-level `renderers` list must match a fresh scan."""
    fresh = sorted(entry["value"] for entry in _discover_axis_renderers())
    cached = sorted(entry["value"] for entry in renderers)
    assert fresh == cached


def test_get_renderer_returns_none_for_unknown() -> None:
    """Unknown chart types resolve to None rather than raising."""
    assert get_renderer("No Such Chart Type") is None


def test_renamed_chart_types_still_resolve() -> None:
    """chart_type is stored in the database, so a rename must not orphan figures."""
    for old_name, new_name in CHART_TYPE_ALIASES.items():
        entry = get_renderer(old_name)
        assert entry is not None, f"{old_name!r} no longer resolves to a renderer"
        assert entry["value"] == new_name


def test_aliases_point_at_renderers_that_exist() -> None:
    """An alias to a name nobody implements would be a silent dead end."""
    known = {entry["value"] for entry in renderers}
    for old_name, new_name in CHART_TYPE_ALIASES.items():
        assert new_name in known, f"alias {old_name!r} -> {new_name!r} has no renderer"
        assert old_name not in known, f"alias {old_name!r} shadows a real renderer"


def test_import_class_from_file_is_cached() -> None:
    """Repeated imports must return the very same class object.

    Why: identity stability is what makes `isinstance` checks against a
    previously obtained renderer class work.
    """
    entry = get_renderer("Scatter Plot")
    assert entry is not None

    first = import_class_from_file(entry)
    second = import_class_from_file(entry)

    assert first is not None
    assert first is second
    assert first.Name == "Scatter Plot"


# ----------------------------------------------------------------------
# The Table renderer's column roles
# ----------------------------------------------------------------------
def test_the_table_renderer_declares_column_roles() -> None:
    """It used to declare none, and the chart picker's role panel then said
    "No renderer roles declared." - which reads as a renderer that is not
    finished rather than one that draws every column."""
    entry = get_renderer("Table")
    assert entry is not None
    assert entry["optional"], "the picker builds one chooser per declared role"
    assert entry["required"] == []


def test_the_declared_roles_are_the_ones_the_renderer_reads() -> None:
    from app.charts.table import COLUMN_ROLES, TableAxisRenderer

    assert TableAxisRenderer.OptionalRoles == COLUMN_ROLES


def test_mapped_columns_keep_their_source_names_as_headers() -> None:
    """A role aliases its column in the SQL, so the frame arrives carrying
    column_1 where it carried region - and a table headed "column_1" is a
    table nobody can read."""
    import pandas as pd
    from matplotlib.figure import Figure

    from app.charts.base import SeriesData
    from app.charts.table import TableAxisRenderer

    figure = Figure()
    ax = figure.add_subplot(111)
    TableAxisRenderer().render_axis(
        ax=ax,
        series=[
            SeriesData(
                name="s",
                df=pd.DataFrame({"column_1": ["north"], "column_2": [10]}),
                style={},
                roles={"column_1": "region", "column_2": "units"},
            )
        ],
        options={},
    )

    headers = [ax.tables[0][(0, column)].get_text().get_text() for column in range(2)]
    assert headers == ["region", "units"]


def test_no_roles_mapped_still_draws_every_column() -> None:
    """SELECT * is what a Table did before the slots existed, and what every
    figure already saved with one still carries."""
    import pandas as pd
    from matplotlib.figure import Figure

    from app.charts.base import SeriesData
    from app.charts.table import TableAxisRenderer

    figure = Figure()
    ax = figure.add_subplot(111)
    TableAxisRenderer().render_axis(
        ax=ax,
        series=[
            SeriesData(
                name="s",
                df=pd.DataFrame({"region": ["north"], "units": [10], "note": ["ok"]}),
                style={},
            )
        ],
        options={},
    )

    headers = [ax.tables[0][(0, column)].get_text().get_text() for column in range(3)]
    assert headers == ["region", "units", "note"]


def test_a_slot_naming_a_column_the_query_lost_is_skipped() -> None:
    """A blank column under a real heading looks like missing data; a mapping
    that has gone stale should simply not be drawn."""
    import pandas as pd
    from matplotlib.figure import Figure

    from app.charts.base import SeriesData
    from app.charts.table import TableAxisRenderer

    figure = Figure()
    ax = figure.add_subplot(111)
    TableAxisRenderer().render_axis(
        ax=ax,
        series=[
            SeriesData(
                name="s",
                df=pd.DataFrame({"column_1": ["north"]}),
                style={},
                roles={"column_1": "region", "column_3": "gone"},
            )
        ],
        options={},
    )

    cells = ax.tables[0].get_celld()
    assert [key for key in cells if key[0] == 0] == [(0, 0)]


def test_series_data_carries_the_role_map_the_headers_come_from() -> None:
    """Defaulted, so the many three-argument constructions still work."""
    import pandas as pd

    from app.charts.base import SeriesData

    assert SeriesData(name="s", df=pd.DataFrame(), style={}).roles == {}

