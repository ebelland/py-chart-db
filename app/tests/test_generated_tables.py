"""Tests for the tables a series operation writes.

They used to be named like anything else, so a project with a few fits and a
spectral analysis showed more generated tables in the source list than imported
ones - and the imported ones are what the user came to find.  Every generated
table now starts with an underscore, which is enough for the list to hide the
whole class of them without keeping a registry.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.series_operations.dialog_base import (
    GENERATED_TABLE_PREFIX,
    generated_table_name,
)
from app.widgets.table_list import _is_generated

APP_DIR = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# The name
# ----------------------------------------------------------------------
def test_a_generated_name_is_prefixed() -> None:
    assert generated_table_name("Fit_series_1").startswith(GENERATED_TABLE_PREFIX)


def test_unsafe_characters_are_replaced() -> None:
    """The name goes into SQL, and a series can be called anything."""
    assert generated_table_name("Fit: my series (2)") == "_Fit_my_series_2"


def test_an_empty_name_falls_back() -> None:
    assert generated_table_name("", fallback="Result") == "_Result"
    assert generated_table_name("!!!", fallback="Result") == "_Result"


def test_the_prefix_is_not_doubled_by_stripping() -> None:
    """A name that already starts with _ keeps exactly one."""
    assert generated_table_name("_Fit_1") == "_Fit_1"


def test_the_list_recognises_what_the_operations_write() -> None:
    """The two halves have to agree or the filter hides nothing."""
    assert _is_generated(generated_table_name("Smoothing_axis1"))
    assert not _is_generated("measurements")


# ----------------------------------------------------------------------
# Every operation uses it
# ----------------------------------------------------------------------
def test_no_operation_composes_a_table_name_without_the_helper() -> None:
    """One helper, or the prefix is forgotten in exactly one dialog.

    A method that only forwards - ``return result.table_name`` - is fine: the
    name it forwards was built through the helper somewhere else.  What must
    not happen is a method *composing* a name from string literals on its own.
    """
    offenders = []
    for path in (APP_DIR / "series_operations").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"def (result_table_name|_output_table_name)\b", source):
            start = match.start()
            body = source[start : source.index("\n    def ", start + 10)]
            composes = '"' in body or "'" in body
            if composes and "generated_table_name" not in body:
                offenders.append(f"{path.name}:{match.group(1)}")
    assert offenders == [], f"these build a table name without the prefix: {offenders}"


# ----------------------------------------------------------------------
# The toggle
# ----------------------------------------------------------------------
def test_the_toggle_is_remembered() -> None:
    source = (APP_DIR / "widgets" / "table_list.py").read_text(encoding="utf-8")
    start = source.index("def set_generated_visible")
    body = source[start : source.index("\n    def ", start + 10)]

    assert "update_section" in body
    assert "self.reload()" in body


def test_generated_tables_are_hidden_by_default() -> None:
    source = (APP_DIR / "widgets" / "table_list.py").read_text(encoding="utf-8")
    assert "CONFIG_SHOW_GENERATED, False" in source


def test_the_filter_is_applied_while_the_list_is_built() -> None:
    source = (APP_DIR / "widgets" / "table_list.py").read_text(encoding="utf-8")
    start = source.index("def reload")
    body = source[start : source.index("\n    def ", start + 10)]

    assert "_is_generated" in body
    assert "self._show_generated" in body
