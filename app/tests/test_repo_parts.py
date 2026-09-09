"""SqliteRepo is one class assembled from parts (todo.txt N-5).

3 300 lines in one module is not something anyone reads, and the seams
between its subjects were already marked by section comments. They are
modules now - user tables, descriptors, saved queries, maintenance - and
the class is still one class: ``from app.data.sqlite_repo import
SqliteRepo``, as 77 call sites already wrote it.

What is worth guarding is not where a method sits but the two properties
that make the split invisible: the concrete class keeps the slots its
dataclass declares, and nothing that used to be importable from
``sqlite_repo`` stopped being so.
"""
from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest

from app.data import sqlite_repo
from app.data.repo.descriptors import DescriptorsMixin
from app.data.repo.maintenance import MaintenanceMixin
from app.data.repo.queries import QueriesMixin
from app.data.repo.tables import TablesMixin
from app.data.sqlite_repo import SqliteRepo

MIXINS = (TablesMixin, DescriptorsMixin, QueriesMixin, MaintenanceMixin)


def test_the_class_is_assembled_from_the_parts() -> None:
    assert set(MIXINS) <= set(SqliteRepo.__mro__)


@pytest.mark.parametrize("mixin", MIXINS, ids=lambda cls: cls.__name__)
def test_every_part_declares_empty_slots(mixin: type) -> None:
    """Without ``__slots__ = ()`` a mixin gives the concrete class a
    __dict__, and a typo in an attribute name would quietly create one
    instead of raising - which is the whole reason the dataclass is
    slotted."""
    assert mixin.__dict__.get("__slots__") == ()


def test_an_unknown_attribute_still_raises(tmp_db_path: Path) -> None:
    """The property the slots exist for, checked through the class rather
    than through its declaration."""
    repo = SqliteRepo(db_path=tmp_db_path)
    try:
        with pytest.raises(AttributeError):
            repo._is_conected = True  # noqa: B018 - the typo is the point
    finally:
        repo.close()


@pytest.mark.parametrize("mixin", MIXINS, ids=lambda cls: cls.__name__)
def test_no_part_carries_state_of_its_own(mixin: type) -> None:
    """Every field belongs to SqliteRepo: the parts are behaviour, and the
    connection, the caches and the transaction state are what they share."""
    annotations = getattr(mixin, "__annotations__", {})

    assert annotations == {}


def test_every_field_is_declared_on_the_concrete_class() -> None:
    own = set(SqliteRepo.__dict__.get("__annotations__", {}))
    fields = {field.name for field in dataclasses.fields(SqliteRepo)}

    assert fields <= own


@pytest.mark.parametrize(
    "name",
    ["SqliteRepo", "DatabaseReport", "SavedQuery", "ensure_connection_wrapper",
     "is_read_only_select"],
)
def test_what_was_importable_from_sqlite_repo_still_is(name: str) -> None:
    """The split is meant to be invisible from outside; a re-export is what
    makes that true rather than merely intended."""
    assert hasattr(sqlite_repo, name)


def test_the_connection_and_its_caches_stayed_together() -> None:
    """The todo's own instruction: the dataclass with its OrderedDict field
    and the series cache belong with the connection, not with the
    descriptors."""
    for name in ("_connect", "close", "transaction", "series_df", "undo_store"):
        assert name in SqliteRepo.__dict__, f"{name} left the connection module"


@pytest.mark.parametrize(
    "method, mixin",
    [
        ("import_dataframe", TablesMixin),
        ("delete_table", TablesMixin),
        ("get_figure_descriptor", DescriptorsMixin),
        ("get_axis_options", DescriptorsMixin),
        ("save_query", QueriesMixin),
        ("optimize_db", MaintenanceMixin),
        ("save_as", MaintenanceMixin),
    ],
)
def test_each_method_lives_with_its_subject(method: str, mixin: type) -> None:
    assert method in mixin.__dict__


def test_the_modules_are_a_size_a_person_reads() -> None:
    """Not a style rule: this is the item. 3 300 lines was the complaint,
    and a part that grows back past the biggest of them is the same
    complaint returning."""
    root = Path(sqlite_repo.__file__).parent
    sizes = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in [root / "sqlite_repo.py", *sorted((root / "repo").glob("*.py"))]
    }

    assert max(sizes.values()) < 1500, sizes
