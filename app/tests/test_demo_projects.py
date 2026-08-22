"""The demo set: ten projects, each named for what it shows.

A demo is the first thing a new installation opens, so its failures are the
ones nobody reports - a chart with no data behind it, a table the figures do
not read, a file whose name says nothing. These tests build the whole set and
open every file, because that is the only way to know the demo demonstrates
anything.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.demo_project import (
    DEMO_PROJECTS,
    QUERY_SOURCES,
    TABLE_BUILDERS,
    build_demo_project,
    build_demo_projects,
    _figure_specs,
)
from app.data.sqlite_repo import SqliteRepo


@pytest.fixture(scope="module")
def demo_set(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """The whole set, built once - it takes about a third of a second."""
    return build_demo_projects(tmp_path_factory.mktemp("demo_set"))


# ----------------------------------------------------------------------
# What the set is
# ----------------------------------------------------------------------
def test_the_set_has_one_file_per_subject(demo_set: list[Path]) -> None:
    assert len(demo_set) == len(DEMO_PROJECTS) >= 8
    assert len({path.name for path in demo_set}) == len(demo_set)


def test_every_name_says_what_the_file_shows() -> None:
    """The file name is the documentation: someone with ten .dhub files in a
    folder should be able to open the one that answers their question."""
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
    known = {spec.key for spec in _figure_specs()}

    for demo in DEMO_PROJECTS:
        unknown = sorted(set(demo.figures) - known)
        assert unknown == [], f"{demo.file_name}: {unknown}"


def test_every_figure_appears_in_some_demo() -> None:
    """A figure nothing ships is a figure nobody sees."""
    shipped = {key for demo in DEMO_PROJECTS for key in demo.figures}
    defined = {spec.key for spec in _figure_specs()}

    assert defined - shipped == set()


def test_every_figure_declares_the_tables_it_reads() -> None:
    """That declaration is what lets a one-subject file carry one table."""
    for spec in _figure_specs():
        assert spec.key, spec.name
        assert spec.tables, spec.name
        unknown = sorted(set(spec.tables) - set(TABLE_BUILDERS))
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
    """Six tables in a file about one peak is six things to explain."""
    path = build_demo_project(tmp_path / "peak.dhub", ("peak_scan",))
    figures, tables, queries = _open(path)

    assert len(figures) == 1
    assert tables == ["peak_scan"]
    assert queries == []


def test_a_saved_query_brings_its_source_table_with_it(tmp_path: Path) -> None:
    """The query is executed on every read, so the table it selects from has
    to be in the file - no figure reads it directly."""
    path = build_demo_project(tmp_path / "query.dhub", ("saved_query",))
    _figures, tables, queries = _open(path)

    assert "sensor_readings" in tables
    assert "daily_mean_temperature" in queries
    assert set(queries) <= set(QUERY_SOURCES)


def test_the_complete_project_carries_every_table_and_query(tmp_path: Path) -> None:
    path = build_demo_project(tmp_path / "all.dhub")
    figures, tables, queries = _open(path)

    assert len(figures) == len(_figure_specs())
    assert set(tables) == set(TABLE_BUILDERS)
    assert set(queries) == set(QUERY_SOURCES)


# ----------------------------------------------------------------------
# The two datasets shaped for an operation
# ----------------------------------------------------------------------
def test_the_process_run_actually_shifts(tmp_path: Path) -> None:
    """A control chart demo of a stable process demonstrates nothing: the
    point of the operation is that within-subgroup sigma catches a shift the
    overall spread would swallow."""
    import numpy as np

    path = build_demo_project(tmp_path / "run.dhub", ("process_run",))
    repo = SqliteRepo(db_path=path)
    try:
        values = repo.query_df("SELECT measurement FROM process_run ORDER BY sample")
    finally:
        repo.close()

    measurements = values["measurement"].to_numpy(dtype=float)
    before, after = measurements[:80], measurements[80:]
    assert float(np.mean(after) - np.mean(before)) > 1.5


def test_the_peak_scan_has_a_peak_away_from_the_origin(tmp_path: Path) -> None:
    """Where a starting point read off the data earns its keep."""
    import numpy as np

    path = build_demo_project(tmp_path / "scan.dhub", ("peak_scan",))
    repo = SqliteRepo(db_path=path)
    try:
        frame = repo.query_df(
            "SELECT wavelength_nm, intensity FROM peak_scan ORDER BY wavelength_nm"
        )
    finally:
        repo.close()

    x = frame["wavelength_nm"].to_numpy(dtype=float)
    y = frame["intensity"].to_numpy(dtype=float)
    assert 7.0 < float(x[int(np.argmax(y))]) < 9.5
    assert float(np.max(y) - np.median(y)) > 2.0


def test_the_same_seed_gives_the_same_demo(tmp_path: Path) -> None:
    """A demo that changed on every build could not be described in a manual."""
    first = build_demo_project(tmp_path / "one.dhub", ("process_run",))
    second = build_demo_project(tmp_path / "two.dhub", ("process_run",))

    def measurements(path: Path) -> list[float]:
        repo = SqliteRepo(db_path=path)
        try:
            return repo.query_df(
                "SELECT measurement FROM process_run ORDER BY sample"
            )["measurement"].tolist()
        finally:
            repo.close()

    assert measurements(first) == measurements(second)
