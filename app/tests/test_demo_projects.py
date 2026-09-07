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

from app.data.demo_project import (
    DEMO_PROJECTS,
    QUERY_SOURCES,
    TABLE_SOURCES,
    build_demo_project,
    build_demo_projects,
    copy_demo_project,
    _figure_specs,
)
from app.data.sqlite_repo import SqliteRepo


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

    assert len(figures) == len(_figure_specs())
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
