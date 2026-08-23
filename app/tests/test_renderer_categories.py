"""Renderers carry a category, and the plot dialog groups by it.

The chart picker was a flat list of nine names in discovery order, which is
neither alphabetical nor meaningful - it is whatever order the scanner walked
the folder in.  Renderers now declare a ``Category`` taken from Matplotlib's
own taxonomy (https://matplotlib.org/stable/plot_types/) rather than one
invented here, so a renderer's family is decided by what it draws and the
picker can be read alongside the documentation the renderers wrap.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.charts.base import BaseAxisRenderer
from app.scanners.axis_renderer_scanner import renderers

APP_DIR = Path(__file__).resolve().parent.parent

# The five headings on the Matplotlib page, verbatim.
MATPLOTLIB_CATEGORIES = {
    "Pairwise data",
    "Statistical distributions",
    "Gridded data",
    "Irregularly gridded data",
    "3D and volumetric data",
}


# ----------------------------------------------------------------------
# The property
# ----------------------------------------------------------------------
def test_the_base_renderer_declares_a_category() -> None:
    """Inherited, so a renderer extending another keeps its family."""
    assert isinstance(BaseAxisRenderer.Category, str)
    assert BaseAxisRenderer.Category in MATPLOTLIB_CATEGORIES


def test_every_renderer_has_a_category() -> None:
    missing = [r["value"] for r in renderers if not str(r.get("category") or "").strip()]
    assert missing == []


def test_no_category_is_invented() -> None:
    """The point of borrowing the taxonomy is not adding to it.

    A category outside this set still works - the dialog puts it in its own
    section at the end - but it should be a decision, not a typo.
    """
    unknown = {
        str(r.get("category"))
        for r in renderers
        if str(r.get("category")) not in MATPLOTLIB_CATEGORIES
    }
    assert unknown == set()


@pytest.mark.parametrize(
    ("chart_type", "category"),
    [
        ("Scatter Plot", "Pairwise data"),
        ("Time Series", "Pairwise data"),
        ("Bar Chart", "Pairwise data"),
        ("Horizontal Bar Chart", "Pairwise data"),
        ("Histogram", "Statistical distributions"),
        ("Box Plot", "Statistical distributions"),
        ("Violin Plot", "Statistical distributions"),
        ("ECDF", "Statistical distributions"),
        ("Pie Chart", "Statistical distributions"),
    ],
)
def test_each_renderer_lands_where_matplotlib_puts_it(
    chart_type: str, category: str
) -> None:
    """Checked one by one, because these are judgements, not a rule.

    hist, boxplot, violinplot, pie and ecdf are all listed under Statistical
    distributions on the Matplotlib page - pie included, which is the one that
    surprises people.
    """
    match = next(r for r in renderers if r["value"] == chart_type)
    assert match["category"] == category


def test_the_scanner_reads_the_category_statically() -> None:
    """Discovery is by AST, so a new attribute has to be listed to be seen."""
    source = (APP_DIR / "scanners" / "axis_renderer_scanner.py").read_text(
        encoding="utf-8"
    )
    assert '"Category"' in source


def test_an_extending_renderer_inherits_the_family_it_declares() -> None:
    """ECDF extends Scatter but is a distribution, and says so itself."""
    from app.charts.ecdf import EcdfAxisRenderer
    from app.charts.scatter import ScatterAxisRenderer

    assert ScatterAxisRenderer.Category == "Pairwise data"
    assert EcdfAxisRenderer.Category == "Statistical distributions"


# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------
@pytest.fixture
def dialog(qapp, tmp_path):
    from app.data.sqlite_repo import SqliteRepo
    from app.dialogs.create_chart_dialog import NewPlotTabDialog

    repo = SqliteRepo(db_path=tmp_path / "categories.dhub")
    yield NewPlotTabDialog(repo)
    repo.close()


def test_the_picker_is_an_accordion_of_categories(dialog) -> None:
    """Same widget as the properties panels, so both sides behave alike."""
    toolbox = dialog._types_toolbox
    headings = [toolbox.itemText(i) for i in range(toolbox.count())]

    assert headings == [
        "Pairwise data",
        "Statistical distributions",
        "3D and volumetric data",
    ]


def test_matplotlibs_order_is_kept_not_alphabetical(dialog) -> None:
    """"Pairwise data" first: those are the plots people reach for."""
    toolbox = dialog._types_toolbox

    assert toolbox.itemText(0) == "Pairwise data"


def test_every_renderer_is_reachable(dialog) -> None:
    """Grouping must not lose one."""
    from PySide6.QtCore import Qt

    listed = {
        renderer_list.item(row).data(Qt.ItemDataRole.UserRole)
        for renderer_list in dialog._lists_by_category.values()
        for row in range(renderer_list.count())
    }

    assert listed == {r["value"] for r in renderers}


def test_a_renderer_is_selected_to_begin_with(dialog) -> None:
    assert dialog._current_item() is not None
    assert dialog._selected_chart_type() in {r["value"] for r in renderers}


def test_the_name_comes_from_the_item_data_not_its_text(dialog) -> None:
    """The label may be decorated; the id is what the dialog acts on."""
    assert dialog._current_renderer_name() == dialog._selected_chart_type()


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------
def test_searching_a_name_selects_it(dialog) -> None:
    dialog._filter_renderer_toolbox("viol")

    assert dialog._selected_chart_type() == "Violin Plot"


def test_searching_a_category_keeps_the_whole_family(dialog) -> None:
    """Typing "stat" should bring up the statistical plots, not nothing.

    The categories are half of why the list is worth searching at all.
    """
    dialog._filter_renderer_toolbox("stat")

    stats = dialog._lists_by_category["Statistical distributions"]
    hidden = [stats.item(r).isHidden() for r in range(stats.count())]
    assert not any(hidden)

    pairwise = dialog._lists_by_category["Pairwise data"]
    assert all(pairwise.item(r).isHidden() for r in range(pairwise.count()))


def test_a_section_with_no_match_is_disabled_not_removed(dialog) -> None:
    """Removing pages renumbers the ones after it, and the index is the key."""
    toolbox = dialog._types_toolbox
    before = toolbox.count()

    dialog._filter_renderer_toolbox("viol")

    assert toolbox.count() == before
    assert not toolbox.isItemEnabled(0)
    assert toolbox.isItemEnabled(1)


def test_the_heading_counts_the_matches_while_searching(dialog) -> None:
    """The count follows the renderers, so it is derived rather than written.

    Hard-coding it meant every new renderer in the section broke this test for
    no reason other than the number changing.
    """
    from app.scanners.axis_renderer_scanner import renderers

    expected = sum(
        1
        for entry in renderers
        if "stat" in f"{entry['value']} {entry.get('category', '')}".lower()
    )
    dialog._filter_renderer_toolbox("stat")

    assert dialog._types_toolbox.itemText(1).endswith(f"({expected})")
    assert expected > 1


def test_clearing_the_search_restores_the_headings(dialog) -> None:
    """The count is not part of the name; the next filter must not key off it."""
    dialog._filter_renderer_toolbox("stat")
    dialog._filter_renderer_toolbox("")

    toolbox = dialog._types_toolbox
    assert [toolbox.itemText(i) for i in range(toolbox.count())] == [
        "Pairwise data",
        "Statistical distributions",
        "3D and volumetric data",
    ]


def test_a_search_matching_nothing_leaves_the_previous_choice(dialog) -> None:
    """Better than clearing it: the dialog still has a valid chart type."""
    dialog._filter_renderer_toolbox("pie")
    dialog._filter_renderer_toolbox("zzzz")

    assert dialog._selected_chart_type() == "Pie Chart"


def test_the_selection_stays_exclusive_across_sections(dialog) -> None:
    """Two highlighted rows would read as two chart types chosen."""
    dialog._filter_renderer_toolbox("")
    pairwise = dialog._lists_by_category["Pairwise data"]
    stats = dialog._lists_by_category["Statistical distributions"]

    pairwise.setCurrentRow(0)
    dialog._clear_other_selections(pairwise)
    stats.setCurrentRow(0)
    dialog._clear_other_selections(stats)

    assert not pairwise.selectedItems()
    assert len(stats.selectedItems()) == 1
