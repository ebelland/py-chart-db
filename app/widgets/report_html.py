"""One house style for every series-operation report.

Five dialogs produced HTML and each invented its own: three shades of header
grey, two table borders, three ways of writing a p-value, and a fit report that
was a paragraph while the statistics report was a table.  Read one after
another they looked like five different applications.

The style is JMP's, because JMP is what this kind of output is compared
against and because its conventions are good ones:

* one titled panel per section, stacked;
* a plain two-column table for a summary, a headed table for a list;
* numbers right-aligned, labels left-aligned, so a column of values can be
  scanned down;
* p-values written as ``<.0001`` rather than as ``9.7e-05``.  The exponent is
  noise: below a ten-thousandth the only thing anyone reads is "smaller than
  anything I care about", and the digits invite a precision the test does not
  have.

Everything here returns a string.  Nothing touches Qt, so a report can be
built and checked without a widget.
"""
from __future__ import annotations

import html
import math
from collections.abc import Iterable, Sequence
from typing import Any

# JMP's threshold, and its notation: no leading zero, four decimals.
P_VALUE_FLOOR: float = 1e-4

# The house palette.  Grey headers, hairline rules, no colour except for the
# accent on a section title - a report is read, not admired.
_TEXT = "#1f2937"
_MUTED = "#6b7280"
_RULE = "#d7dde5"
_HEADER_BG = "#eef2f7"
_STRIPE_BG = "#f8fafc"

_FONT = "font-family:Segoe UI,-apple-system,Helvetica,Arial,sans-serif;font-size:10pt;"


def format_p_value(value: Any) -> str:
    """Return a p-value the JMP way: ``<.0001`` when it is small enough.

    Four decimals, no leading zero.  A p of 3e-12 and one of 8e-5 are the same
    statement - "not by chance" - and printing their digits suggests the test
    resolves a difference between them, which it does not.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if number < P_VALUE_FLOOR:
        return "&lt;.0001"
    # ".0342", not "0.0342": the leading zero carries no information and
    # misaligns the column.
    return f"{number:.4f}".lstrip("0") or "0"


def format_number(value: Any, *, digits: int = 6) -> str:
    """Return a number for a report cell, or an empty cell when it is not one.

    Non-finite values become blank rather than ``nan``: an absent statistic
    reads as an absence, while ``nan`` reads as a failure.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value)) if value not in (None, "") else ""
    if not math.isfinite(number):
        return ""
    if number == int(number) and abs(number) < 1e15:
        return f"{int(number)}"
    return f"{number:.{digits}g}"


def _cell(content: str, *, align: str, header: bool, stripe: bool) -> str:
    tag = "th" if header else "td"
    background = _HEADER_BG if header else (_STRIPE_BG if stripe else "#ffffff")
    weight = "600" if header else "400"
    return (
        f"<{tag} style='background:{background};color:{_TEXT};font-weight:{weight};"
        f"text-align:{align};padding:4px 10px;border-bottom:1px solid {_RULE};"
        f"white-space:nowrap;'>{content}</{tag}>"
    )


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    align: Sequence[str] | None = None,
    empty_message: str = "No results for this selection.",
) -> str:
    """Return a headed table in the house style.

    ``align`` is one of ``"left"``/``"right"`` per column; by default the first
    column is left-aligned - it is the label - and the rest are right-aligned,
    which is what makes a column of numbers readable down the page.

    Cells are inserted as given, so a caller may pass markup produced by
    :func:`format_p_value`; anything else must be escaped by the caller.
    """
    rows = list(rows)
    if not rows:
        return note(empty_message)

    columns = len(headers)
    alignment = list(align) if align else ["left"] + ["right"] * (columns - 1)

    head = "".join(
        _cell(html.escape(str(header)), align=alignment[index], header=True, stripe=False)
        for index, header in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            _cell(str(cell), align=alignment[index] if index < len(alignment) else "right",
                  header=False, stripe=row_index % 2 == 1)
            for index, cell in enumerate(row)
        )
        + "</tr>"
        for row_index, row in enumerate(rows)
    )
    return (
        "<table cellspacing='0' cellpadding='0' "
        f"style='border-collapse:collapse;margin:0 0 10px 0;{_FONT}'>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def summary_table(pairs: Iterable[tuple[str, Any]]) -> str:
    """Return the label/value block that opens a report.

    No header row: the left column *is* the header, which is JMP's convention
    for "what was run" as opposed to "what came out".
    """
    rows = "".join(
        f"<tr><th style='text-align:left;color:{_MUTED};font-weight:400;"
        f"padding:2px 12px 2px 0;white-space:nowrap;'>{html.escape(str(label))}</th>"
        f"<td style='padding:2px 0;color:{_TEXT};'>{html.escape(str(value))}</td></tr>"
        for label, value in pairs
        if str(value).strip()
    )
    return f"<table cellspacing='0' style='margin:0 0 10px 0;{_FONT}'>{rows}</table>"


def section(title: str, *blocks: str) -> str:
    """Return one titled section: a heading rule, then its blocks."""
    body = "".join(block for block in blocks if block)
    return (
        f"<div style='margin:0 0 14px 0;'>"
        f"<div style='font-weight:600;color:{_TEXT};border-bottom:2px solid {_RULE};"
        f"padding:0 0 3px 0;margin:0 0 8px 0;'>{html.escape(str(title))}</div>"
        f"{body}</div>"
    )


def note(text: str) -> str:
    """Return a muted line, for "nothing to report" and for caveats."""
    return (
        f"<div style='color:{_MUTED};padding:2px 0 8px 0;{_FONT}'>"
        f"{html.escape(str(text))}</div>"
    )


def document(title: str, subtitle: str = "", *sections: str) -> str:
    """Wrap sections in the report shell: title, optional subtitle, body."""
    head = f"<div style='font-size:12pt;font-weight:600;color:{_TEXT};'>{html.escape(title)}</div>"
    if subtitle:
        head += f"<div style='color:{_MUTED};margin:2px 0 12px 0;'>{html.escape(subtitle)}</div>"
    else:
        head += "<div style='margin:0 0 12px 0;'></div>"

    return (
        f"<html><body style='{_FONT}color:{_TEXT};'>"
        f"{head}{''.join(sections)}</body></html>"
    )
