"""What an operation's report is made of, and whether it says so.

A dialog builds its report with the shared ``report_html`` helpers and hands
it to the base, which decides whether to render it as markup or escape it -
from one class attribute. Get that attribute wrong and the report still
appears, in full, as text: the user reads ``<html><body style='font-family:
Segoe UI...`` under their chart.

It is invisible in the dialog itself, which is why it survived: the results
pane there sniffs the content and renders markup either way. Only the chart's
notes pane, which trusts the flag, shows the difference.
"""
from __future__ import annotations

import inspect

import pytest

from app.scanners.series_operation_scanner import (
    _discover_series_operations,
    import_class_from_file,
)


def _operations() -> list[tuple[str, type]]:
    found: list[tuple[str, type]] = []
    for operation in _discover_series_operations():
        cls = import_class_from_file(operation)
        if cls is not None:
            found.append((str(operation.get("value") or cls.__name__), cls))
    return found


def _builds_markup(cls: type) -> bool:
    """True when this operation's format_results assembles HTML.

    Read from the source: report_html.* is the shared builder, and a private
    ``*_html`` helper is the other way these dialogs do it.
    """
    try:
        source = inspect.getsource(cls.format_results)
    except (OSError, TypeError):
        return False
    return "report_html." in source or "_html(" in source


def test_the_operations_are_discovered() -> None:
    assert len(_operations()) >= 10


@pytest.mark.parametrize("name, cls", _operations(), ids=lambda value: getattr(value, "__name__", value))
def test_a_report_built_as_markup_says_it_is_markup(name: str, cls: type) -> None:
    """The one line that decides whether the notes pane renders or escapes it.

    Spectral Analysis and Statistics both failed this: they returned a full
    HTML document and left the flag at its default.
    """
    assert bool(getattr(cls, "RESULTS_ARE_HTML", False)) == _builds_markup(cls), (
        f"{name}: RESULTS_ARE_HTML does not match what format_results returns"
    )


def test_the_report_reaches_the_notes_pane_unescaped() -> None:
    """End to end, on the base's own wrapper: markup in, markup out."""
    from app.series_operations.dialog_base import SeriesOperationDialogBase

    class _Markup(SeriesOperationDialogBase):
        RESULTS_ARE_HTML = True

        @property
        def operation_label(self) -> str:
            return "Test"

    wrapped = SeriesOperationDialogBase.results_report_html(
        _Markup.__new__(_Markup), "<table><tr><td>3.14</td></tr></table>", []
    )

    assert "<table>" in wrapped
    assert "&lt;table&gt;" not in wrapped


def test_a_plain_report_is_still_escaped() -> None:
    """The flag has to work in both directions: a plain-text report that
    happens to contain a < must not be read as a tag."""
    from app.series_operations.dialog_base import SeriesOperationDialogBase

    class _Plain(SeriesOperationDialogBase):
        RESULTS_ARE_HTML = False

        @property
        def operation_label(self) -> str:
            return "Test"

    wrapped = SeriesOperationDialogBase.results_report_html(
        _Plain.__new__(_Plain), "p < 0.05", []
    )

    assert "p &lt; 0.05" in wrapped


def test_every_report_carries_the_operation_it_came_from() -> None:
    """Two appended reports with no headings are one wall of text."""
    from app.series_operations.dialog_base import SeriesOperationDialogBase

    class _Named(SeriesOperationDialogBase):
        RESULTS_ARE_HTML = True

        @property
        def operation_label(self) -> str:
            return "Spectral Analysis"

    wrapped = SeriesOperationDialogBase.results_report_html(
        _Named.__new__(_Named), "<p>body</p>", [1, 2]
    )

    assert "Spectral Analysis" in wrapped
    assert "2 result(s)" in wrapped
