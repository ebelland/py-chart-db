"""Tests for the shared report style.

Five dialogs produced HTML and each invented its own: three shades of header
grey, two table borders, three ways of writing a p-value, and a fit report that
was a paragraph while the statistics report was a table.  Read one after
another they looked like five different applications.

The conventions pinned here are JMP's, because that is what this output gets
compared against.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.widgets import report_html

APP_DIR = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# p-values
# ----------------------------------------------------------------------
@pytest.mark.parametrize("value", [0.0, 1e-12, 9.9e-5])
def test_a_small_p_value_is_written_as_a_bound(value: float) -> None:
    """3e-12 and 8e-5 are the same statement; their digits are noise."""
    assert report_html.format_p_value(value) == "&lt;.0001"


def test_an_ordinary_p_value_keeps_four_decimals() -> None:
    assert report_html.format_p_value(0.0342) == ".0342"
    assert report_html.format_p_value(0.5) == ".5000"


def test_the_leading_zero_is_dropped() -> None:
    """JMP's notation, and it keeps the column aligned on the decimal point."""
    assert not report_html.format_p_value(0.25).startswith("0")


def test_one_is_written_in_full() -> None:
    assert report_html.format_p_value(1.0) == "1.0000"


def test_the_boundary_is_not_reported_as_a_bound() -> None:
    assert report_html.format_p_value(0.0001) == ".0001"


@pytest.mark.parametrize("value", [None, "", "not a number", float("nan"), float("inf")])
def test_a_missing_p_value_is_blank(value) -> None:
    """An absent statistic reads as an absence; "nan" reads as a failure."""
    assert report_html.format_p_value(value) == ""


# ----------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------
def test_a_whole_number_has_no_decimal_point() -> None:
    assert report_html.format_number(42.0) == "42"


def test_a_fraction_keeps_significant_digits() -> None:
    assert report_html.format_number(1 / 3, digits=4) == "0.3333"


def test_a_non_finite_number_is_blank() -> None:
    assert report_html.format_number(float("nan")) == ""
    assert report_html.format_number(float("inf")) == ""


def test_text_passes_through_escaped() -> None:
    assert report_html.format_number("a <b>") == "a &lt;b&gt;"


# ----------------------------------------------------------------------
# The blocks
# ----------------------------------------------------------------------
def test_an_empty_table_says_so_instead_of_drawing_a_header() -> None:
    markup = report_html.table(["a", "b"], [], empty_message="Nothing here.")
    assert "Nothing here." in markup
    assert "<table" not in markup


def test_the_first_column_is_left_aligned_and_the_rest_right() -> None:
    """Labels read from the left; numbers compare down the right."""
    markup = report_html.table(["Test", "Value"], [("t", "1")])
    assert markup.index("text-align:left") < markup.index("text-align:right")


def test_alignment_can_be_given_per_column() -> None:
    markup = report_html.table(["a", "b"], [("1", "2")], align=["right", "left"])
    assert markup.index("text-align:right") < markup.index("text-align:left")


def test_summary_values_are_escaped() -> None:
    markup = report_html.summary_table([("Source", "a <b> table")])
    assert "&lt;b&gt;" in markup and "<b>" not in markup


def test_an_empty_summary_value_is_dropped() -> None:
    """A row reading "Target:" with nothing after it is worse than no row."""
    markup = report_html.summary_table([("Kept", "yes"), ("Dropped", "")])
    assert "Kept" in markup and "Dropped" not in markup


def test_a_section_carries_its_title() -> None:
    assert "Parameter estimates" in report_html.section("Parameter estimates", "<p></p>")


def test_a_document_is_a_complete_html_page() -> None:
    markup = report_html.document("Fit", "Gaussian", report_html.section("A", "<p></p>"))
    assert markup.startswith("<html>") and markup.endswith("</body></html>")
    assert "Fit" in markup and "Gaussian" in markup


# ----------------------------------------------------------------------
# Every operation uses it
# ----------------------------------------------------------------------
def test_no_operation_hand_rolls_a_table() -> None:
    """One style, or the reports drift apart again."""
    offenders = []
    for path in (APP_DIR / "series_operations").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"<t(able|head|body)\b", source) and "report_html" not in source:
            offenders.append(path.name)
    assert offenders == [], f"these build tables without the shared style: {offenders}"


def test_no_operation_formats_a_p_value_by_hand() -> None:
    offenders = []
    for path in (APP_DIR / "series_operations").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"def _format_pvalue\b", source):
            start = match.start()
            body = source[start : source.index("\n    def ", start + 10)]
            if "report_html" not in body:
                offenders.append(path.name)
    assert offenders == []
