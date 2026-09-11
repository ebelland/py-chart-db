"""The demo set: real datasets, each named for what it shows.

A demo is the first thing a new installation opens, so its failures are the
ones nobody reports - a chart with no data behind it, a table the figures do
not read, a file whose name says nothing. These tests build the whole set and
open every file, because that is the only way to know the demo demonstrates
anything.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.charts import layout_presets
from app.data.demo_project import (
    DEMO_PROJECTS,
    QUERY_SOURCES,
    TABLE_SOURCES,
    build_demo_project,
    build_demo_projects,
    copy_demo_project,
    _figure_specs,
    _multi_axis_figure_specs,
)
from app.data.sqlite_repo import SqliteRepo


def _all_specs():
    """Every figure spec, single- and multi-axis alike."""
    return [*_figure_specs(), *_multi_axis_figure_specs()]


@pytest.fixture(scope="module")
def demo_set(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """The whole set, built once from the real sample data."""
    return build_demo_projects(tmp_path_factory.mktemp("demo_set"))


# ----------------------------------------------------------------------
# What the set is
# ----------------------------------------------------------------------
def test_the_set_has_one_file_per_subject(demo_set: list[Path]) -> None:
    assert len(demo_set) == len(DEMO_PROJECTS) >= 8
    assert len({path.name for path in demo_set}) == len(demo_set)


def test_every_name_says_what_the_file_shows() -> None:
    """The file name is the documentation: someone with a dozen .dhub files
    in a folder should be able to open the one that answers their question."""
    for demo in DEMO_PROJECTS:
        assert " - " in demo.file_name, demo.file_name
        subject, shows = demo.file_name.split(" - ", 1)
        assert len(subject.split()) >= 1
        assert len(shows.split()) >= 2, f"{demo.file_name}: says too little"
        assert demo.summary.strip()


def test_no_name_is_a_number(demo_set: list[Path]) -> None:
    """"Demo 1.dhub" tells nobody anything a week later."""
    for path in demo_set:
        assert not path.stem.rstrip("0123456789 ").endswith("Demo")


def test_the_complete_project_comes_first() -> None:
    """Startup opens it, so it has to be the one with everything in it."""
    first = DEMO_PROJECTS[0]

    assert first.figures == (), "an empty selection means every figure"
    assert "everything" in first.summary.lower() or "every" in first.summary.lower()


def test_every_demo_names_figures_that_exist() -> None:
    """A typo in a figure key would give a demo file with no charts at all."""
    known = {spec.key for spec in _all_specs()}

    for demo in DEMO_PROJECTS:
        unknown = sorted(set(demo.figures) - known)
        assert unknown == [], f"{demo.file_name}: {unknown}"


def test_every_figure_appears_in_some_demo() -> None:
    """A figure nothing ships is a figure nobody sees."""
    shipped = {key for demo in DEMO_PROJECTS for key in demo.figures}
    defined = {spec.key for spec in _all_specs()}

    assert defined - shipped == set()


def test_every_figure_declares_the_tables_it_reads() -> None:
    """That declaration is what lets a one-subject file carry one table."""
    for spec in _all_specs():
        assert spec.key, spec.name
        assert spec.tables, spec.name
        unknown = sorted(set(spec.tables) - set(TABLE_SOURCES))
        assert unknown == [], f"{spec.name}: {unknown}"


# ----------------------------------------------------------------------
# What is in each file
# ----------------------------------------------------------------------
def _open(path: Path) -> tuple[list[tuple[int, str]], list[str], list[str]]:
    repo = SqliteRepo(db_path=path)
    try:
        return (
            list(repo.get_figures()),
            list(repo.list_table_names()),
            [saved.name for saved in repo.list_queries()],
        )
    finally:
        repo.close()


def test_every_file_opens_and_has_charts(demo_set: list[Path]) -> None:
    for path in demo_set:
        figures, tables, _queries = _open(path)
        assert figures, f"{path.name} has no figures"
        assert tables, f"{path.name} has no tables"


def test_every_series_in_every_file_returns_rows(demo_set: list[Path]) -> None:
    """The failure this catches is the quiet one: a chart drawn from a query
    that matches nothing looks like an empty axis and reads as a broken app."""
    empty: list[str] = []

    for path in demo_set:
        repo = SqliteRepo(db_path=path)
        try:
            for figure_id, _name in repo.get_figures():
                descriptor = repo.load_figure_descriptor(figure_id=int(figure_id))
                for axis in descriptor.axes:
                    for series in axis.series:
                        frame = repo.query_df(series.sql_query)
                        if frame.empty:
                            empty.append(f"{path.stem}: {series.name}")
        finally:
            repo.close()

    assert empty == []


def test_a_single_subject_file_carries_only_what_it_needs(tmp_path: Path) -> None:
    """Twelve tables in a file about one force curve is eleven things to
    explain that have nothing to do with it."""
    path = build_demo_project(tmp_path / "dlvo.dhub", ("dlvo_force",))
    figures, tables, queries = _open(path)

    assert len(figures) == 1
    assert tables == ["dlvo_curve"]
    assert queries == []


def test_a_saved_query_brings_its_source_table_with_it(tmp_path: Path) -> None:
    """The query is executed on every read, so the table it selects from has
    to be in the file - no figure reads it directly."""
    path = build_demo_project(tmp_path / "query.dhub", ("transactions_query",))
    _figures, tables, queries = _open(path)

    assert "transactions" in tables
    assert "avg_amount_by_channel" in queries
    assert set(queries) <= set(QUERY_SOURCES)


def test_the_complete_project_carries_every_table_and_query(tmp_path: Path) -> None:
    path = build_demo_project(tmp_path / "all.dhub")
    figures, tables, queries = _open(path)

    assert len(figures) == len(_all_specs())
    assert set(tables) == set(TABLE_SOURCES)
    assert set(queries) == set(QUERY_SOURCES)


def test_building_twice_gives_the_same_data(tmp_path: Path) -> None:
    """The source files are static, so nothing here should vary between
    builds - unlike the old synthetic generator, there is no seed to pin."""
    first = build_demo_project(tmp_path / "one.dhub", ("dlvo_force",))
    second = build_demo_project(tmp_path / "two.dhub", ("dlvo_force",))

    def forces(path: Path) -> list[float]:
        repo = SqliteRepo(db_path=path)
        try:
            return repo.query_df(
                "SELECT F_total_nN FROM dlvo_curve ORDER BY separation_nm"
            )["F_total_nN"].tolist()
        finally:
            repo.close()

    assert forces(first) == forces(second)


# ----------------------------------------------------------------------
# The datasets shaped for a specific operation
# ----------------------------------------------------------------------
def test_the_dlvo_curve_has_a_repulsive_and_an_attractive_regime(
    tmp_path: Path,
) -> None:
    """The point of shipping this curve: a Fit needs a sign change to be
    worth fitting at all, and Calculus needs a real feature to differentiate."""
    path = build_demo_project(tmp_path / "dlvo.dhub", ("dlvo_force",))
    repo = SqliteRepo(db_path=path)
    try:
        frame = repo.query_df(
            "SELECT separation_nm, F_total_nN FROM dlvo_curve ORDER BY separation_nm"
        )
    finally:
        repo.close()

    force = frame["F_total_nN"].to_numpy(dtype=float)
    assert force.max() > 0.05, "no clear repulsive peak"
    assert force.min() < -0.05, "no clear attractive regime"


def test_the_employee_dataset_has_a_real_outlier(tmp_path: Path) -> None:
    """The point of shipping this dataset: the Outlier operation needs
    something to find, and the box plot's own whiskers should already show it."""
    path = build_demo_project(tmp_path / "employees.dhub", ("employee_box",))
    repo = SqliteRepo(db_path=path)
    try:
        salaries = repo.query_df(
            "SELECT salary_eur FROM employee_compensation"
        )["salary_eur"].to_numpy(dtype=float)
    finally:
        repo.close()

    median = float(np.median(salaries))
    assert salaries.max() > median * 4, "no salary far enough above the median"


def test_the_attribute_counts_reproduce_montgomerys_p_chart(tmp_path: Path) -> None:
    """The point of shipping this table: running a p-chart over it should
    give the textbook's own numbers, not just some plausible-looking chart."""
    from app.series_operations.control_chart_dialog import CHART_P, attribute_limits

    path = build_demo_project(tmp_path / "attributes.dhub", ("attribute_counts",))
    repo = SqliteRepo(db_path=path)
    try:
        frame = repo.query_df(
            "SELECT defectives, inspected FROM attribute_chart_counts ORDER BY sample"
        )
    finally:
        repo.close()

    _stat, center, upper, lower, meta = attribute_limits(
        CHART_P,
        frame["defectives"].to_numpy(dtype=float),
        frame["inspected"].to_numpy(dtype=float),
        3.0,
    )
    assert meta["p-bar"] == pytest.approx(0.214, abs=1e-3)
    assert center == pytest.approx(0.214, abs=1e-3)
    assert upper[0] == pytest.approx(0.388, abs=1e-3)
    assert lower[0] == pytest.approx(0.040, abs=1e-3)


def test_the_attribute_chart_annotation_and_line_carry_an_explicit_colour(
    tmp_path: Path,
) -> None:
    """The point of the annotation/line on this figure: opening Overlay
    properties on it should show the colour (and the annotation's font and
    size) combos already populated, not at "(none)"/"Default"."""
    path = build_demo_project(tmp_path / "attributes.dhub", ("attribute_counts",))
    descriptor = _load_axes(
        path, "23 · Defectives per sample - ready for the Control Chart operation"
    )
    options = descriptor.axes[0].options
    annotation = options["annotations"][0]
    assert annotation["kwargs"]["color"]
    assert annotation["kwargs"]["fontfamily"]
    assert annotation["kwargs"]["fontsize"]
    assert options["lines"][0]["kwargs"]["color"]


def test_the_baseline_spectrum_has_a_wandering_background_under_real_peaks(
    tmp_path: Path,
) -> None:
    """The point of shipping this table: AsLS should track the background
    away from the peaks and stay well below the peaks themselves - a rubber
    band, being convex-only, is not what this table is meant to challenge."""
    from app.series_operations.baseline_dialog import asls_baseline

    path = build_demo_project(tmp_path / "baseline.dhub", ("baseline_spectrum",))
    repo = SqliteRepo(db_path=path)
    try:
        frame = repo.query_df("SELECT x, intensity FROM baseline_spectrum ORDER BY x")
    finally:
        repo.close()

    x = frame["x"].to_numpy(dtype=float)
    y = frame["intensity"].to_numpy(dtype=float)
    baseline = asls_baseline(y, lam=1e5, p=0.01, iterations=10)
    corrected = y - baseline

    no_peak = (np.abs(x - 30.0) > 5.0) & (np.abs(x - 70.0) > 8.0)
    assert corrected[no_peak] == pytest.approx(np.zeros(int(no_peak.sum())), abs=1.0)
    assert corrected.max() > 5.0, "no peak survives the correction"


# ----------------------------------------------------------------------
# Multi-axis figures: one demo per layout preset
# ----------------------------------------------------------------------
def _load_axes(path: Path, figure_name: str):
    repo = SqliteRepo(db_path=path)
    try:
        figure_id = next(fid for fid, name in repo.get_figures() if name == figure_name)
        return repo.load_figure_descriptor(figure_id=int(figure_id))
    finally:
        repo.close()


def test_every_multi_axis_figure_actually_has_more_than_one_axis() -> None:
    """The whole point: a demo of a multi-axis layout with one axis in it
    would not show what the layout does at all."""
    for spec in _multi_axis_figure_specs():
        assert len(spec.axes) > 1, spec.name
        assert spec.layout in layout_presets.PRESETS, spec.name


def test_the_main_and_secondary_figure_spans_the_main_axis(tmp_path: Path) -> None:
    path = build_demo_project(tmp_path / "layouts.dhub", ("stock_main_secondary",))
    descriptor = _load_axes(path, "17 · Stock prices - main and secondary")

    assert descriptor.nrows > 1
    main_axis = next(a for a in descriptor.axes if a.title.startswith("AAPL"))
    assert main_axis.options.get("col_span") == descriptor.ncols


def test_the_shared_grid_figure_shares_scale_past_the_first_axis(tmp_path: Path) -> None:
    path = build_demo_project(tmp_path / "layouts.dhub", ("penguin_shared_grid",))
    descriptor = _load_axes(path, "18 · Penguins - shared scale grid")

    axes_by_index = sorted(descriptor.axes, key=lambda a: a.axis_index)
    assert not axes_by_index[0].options.get("sharex")
    for axis in axes_by_index[1:]:
        assert axis.options.get("sharex")
        assert axis.options.get("sharey")


def test_the_overlapping_figure_twins_its_second_axis(tmp_path: Path) -> None:
    path = build_demo_project(tmp_path / "layouts.dhub", ("dlvo_overlapping",))
    descriptor = _load_axes(path, "19 · DLVO force and potential - overlapping axes")

    assert len(descriptor.axes) == 2
    primary = next(a for a in descriptor.axes if not a.options.get("twin_of"))
    twin = next(a for a in descriptor.axes if a.options.get("twin_of"))
    assert twin.options["twin_of"] == primary.id


def test_anscombe_quartet_has_the_same_summary_statistics_on_every_dataset() -> None:
    """The whole point of the dataset: identical mean, variance and
    correlation on all four, despite four very different shapes."""
    frame = TABLE_SOURCES["anscombe"]()
    stats = frame.groupby("dataset").agg(
        mean_x=("x", "mean"), mean_y=("y", "mean"),
        var_x=("x", "var"), var_y=("y", "var"),
        corr=("x", lambda s: s.corr(frame.loc[s.index, "y"])),
    )
    assert stats["mean_x"].round(1).nunique() == 1
    assert stats["mean_y"].round(1).nunique() == 1
    assert stats["var_x"].round(1).nunique() == 1
    assert stats["var_y"].round(1).nunique() == 1
    assert (stats["corr"].round(2) == 0.82).all()


def test_the_anscombe_figure_shares_scale_past_the_first_axis(tmp_path: Path) -> None:
    path = build_demo_project(tmp_path / "layouts.dhub", ("anscombe_quartet",))
    descriptor = _load_axes(path, "20 · Anscombe's quartet - the matplotlib classic")

    axes_by_index = sorted(descriptor.axes, key=lambda a: a.axis_index)
    assert len(axes_by_index) == 4
    assert not axes_by_index[0].options.get("sharex")
    for axis in axes_by_index[1:]:
        assert axis.options.get("sharex")
        assert axis.options.get("sharey")


def test_lissajous_curves_stay_inside_their_own_box() -> None:
    """x=sin(...), y=sin(...): every point of every curve is bounded in
    [-1, 1] by construction - if this ever fails, the generator changed."""
    frame = TABLE_SOURCES["lissajous"]()
    assert frame["x"].between(-1.0, 1.0).all()
    assert frame["y"].between(-1.0, 1.0).all()
    assert frame["curve"].nunique() == 4


def test_the_lissajous_figure_keeps_equal_aspect_per_axis(tmp_path: Path) -> None:
    """Plain GRID, not SHARED_GRID: Matplotlib refuses equal aspect on axes
    that share both x and y (see the axis_options comment in
    _multi_axis_figure_specs), so each of these must be independent."""
    path = build_demo_project(tmp_path / "layouts.dhub", ("lissajous_grid",))
    descriptor = _load_axes(path, "21 · Lissajous curves - equal-aspect grid")

    assert len(descriptor.axes) == 4
    for axis in descriptor.axes:
        assert axis.options.get("aspect") == "equal"
        assert axis.options.get("adjustable") == "datalim"
        assert not axis.options.get("sharex")
        assert not axis.options.get("sharey")


def test_the_signal_spectrum_recovers_its_three_injected_tones() -> None:
    """The signal is built from exactly three tones at 5, 20 and 50 Hz - the
    paired spectrum table has to actually recover them, or pairing the two
    tables demonstrates nothing."""
    spectrum = TABLE_SOURCES["signal_spectrum"]()
    peaks = spectrum.nlargest(3, "magnitude").sort_values("freq_hz")
    assert peaks["freq_hz"].round(0).tolist() == [5.0, 20.0, 50.0]
    # Amplitudes were injected as 1.0, 0.6, 0.3; the FFT normalisation
    # should recover each to within a few percent.
    for magnitude, expected in zip(peaks["magnitude"], (1.0, 0.6, 0.3)):
        assert magnitude == pytest.approx(expected, abs=0.05)


def test_the_signal_figure_is_a_plain_two_axis_grid(tmp_path: Path) -> None:
    path = build_demo_project(tmp_path / "layouts.dhub", ("signal_time_and_frequency",))
    descriptor = _load_axes(path, "22 · Signal analysis - time and frequency domains")

    assert len(descriptor.axes) == 2
    for axis in descriptor.axes:
        assert not axis.options.get("sharex")
        assert not axis.options.get("sharey")
        assert not axis.options.get("twin_of")
        # A raw signal and its spectrum are each the point; a rolling mean
        # drawn over either would hide it.
        assert axis.options.get("show_rolling") is False


# ----------------------------------------------------------------------
# What the running application does with a pre-built file
# ----------------------------------------------------------------------
def test_copying_a_demo_writes_an_independent_file(tmp_path: Path) -> None:
    """The application copies a pre-built file rather than rebuilding it -
    the point of pre-building datasets too large to regenerate on a click."""
    demo = DEMO_PROJECTS[1]
    source_dir = tmp_path / "source"
    built = build_demo_project(source_dir / demo.path_name, demo.figures)

    class _Stub:
        file_name = demo.file_name
        source_path = built

    target = tmp_path / "copy.dhub"
    result = copy_demo_project(_Stub(), target)

    assert result == target
    assert target.is_file()
    assert _open(target)[0]  # has figures


def test_copying_a_missing_demo_raises(tmp_path: Path) -> None:
    class _Stub:
        file_name = "Nonexistent"
        source_path = tmp_path / "does-not-exist.dhub"

    with pytest.raises(FileNotFoundError):
        copy_demo_project(_Stub(), tmp_path / "target.dhub")
