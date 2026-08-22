"""Tests for the report that a series operation leaves on its chart.

An operation's numbers used to live in a dialog and disappear with it.  They
now go to a pane below the chart they describe, so the two can be read
together and both are saved in the .dhub.

The wiring is: the dialog emits ``results_published`` on Apply, the main window
forwards it to the panel the dialog was opened on, and the panel stores the
markup in its figure's view state.  Widgets cannot be built in every
environment, so the parts that can be checked without one - the markup, the
signal, the persistence key - are checked here.
"""
from __future__ import annotations

import re
from pathlib import Path


from app.widgets.html_results import plain_to_html

APP_DIR = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# plain_to_html
# ----------------------------------------------------------------------
def test_markup_in_plain_text_is_escaped() -> None:
    """An unescaped '<' truncates the report at the first angle bracket."""
    result = plain_to_html("a < b & c > d")

    assert "&lt;" in result and "&amp;" in result and "&gt;" in result
    assert "<b>" not in result


def test_line_breaks_survive() -> None:
    assert "<br>" in plain_to_html("one\ntwo")


# ----------------------------------------------------------------------
# The report block
# ----------------------------------------------------------------------
def test_the_report_is_titled_and_dated() -> None:
    """Several appended reports are unreadable without a heading each."""
    source = (APP_DIR / "series_operations" / "dialog_base.py").read_text(
        encoding="utf-8"
    )
    start = source.index("def results_report_html")
    body = source[start : source.index("\n    def ", start + 10)]

    assert "operation_label" in body
    assert "strftime" in body
    assert "len(results)" in body


def test_plain_results_are_escaped_before_they_become_markup() -> None:
    """A dialog whose format_results is plain text must not inject markup."""
    source = (APP_DIR / "series_operations" / "dialog_base.py").read_text(
        encoding="utf-8"
    )
    start = source.index("def results_report_html")
    body = source[start : source.index("\n    def ", start + 10)]

    assert "plain_to_html(formatted)" in body


def test_only_apply_publishes() -> None:
    """A preview is undone on Close; a note about undone results is a lie."""
    source = (APP_DIR / "series_operations" / "dialog_base.py").read_text(
        encoding="utf-8"
    )
    start = source.index("def _finalize_successful_operation")
    body = source[start : source.index("\n    def ", start + 10)]

    assert 'verb == "Applied"' in body
    assert "results_published.emit" in body


# ----------------------------------------------------------------------
# The wiring
# ----------------------------------------------------------------------
def test_every_series_dialog_is_opened_through_one_helper() -> None:
    """Six copies of the wiring meant six chances to forget the new signal."""
    source = (APP_DIR / "dialogs" / "main_window.py").read_text(encoding="utf-8")

    assert source.count("results_published.connect") == 1
    # One helper, reached from one place: the operations list emits a request
    # and the window opens whatever class the scanner resolved.  It used to be
    # six hand-written call sites, which is what this count guarded.
    assert source.count("_open_series_operation(") >= 2  # the call and the def
    assert "operation_requested.connect" in source


def test_the_report_goes_to_the_panel_it_was_opened_on() -> None:
    """Binding to "the current panel" would be wrong once tabs are switched."""
    source = (APP_DIR / "dialogs" / "main_window.py").read_text(encoding="utf-8")
    start = source.index("def _open_series_operation")
    body = source[start : source.index("\n    def ", start + 10)]

    assert "target=panel" in body
    assert "append=True" in body


# ----------------------------------------------------------------------
# The panel
# ----------------------------------------------------------------------
def test_the_notes_pane_is_part_of_the_saved_view_state() -> None:
    """A report about a figure belongs in the .dhub with the figure."""
    source = (APP_DIR / "widgets" / "chart_panel.py").read_text(encoding="utf-8")
    start = source.index("def _persist_view_state")
    body = source[start : source.index("\n    def ", start + 10)]

    assert "notes_html" in body
    assert "notes_split" in body


def test_the_chart_keeps_a_floor_and_the_notes_open_at_a_usable_size() -> None:
    """The notes pane may be dragged shut, but never *opens* shut.

    NOTES_MIN_HEIGHT is 0 on purpose - collapsing the pane is a thing the user
    is allowed to do - so the guarantee moved to first open instead.
    """
    from app.widgets import chart_panel

    assert chart_panel.CHART_AREA_MIN_HEIGHT > 0
    assert chart_panel.NOTES_FIRST_OPEN_MIN_HEIGHT > 0
    assert 0.0 < chart_panel.NOTES_INITIAL_FRACTION < 1.0


def test_the_canvas_is_reparented_to_the_chart_area() -> None:
    """The splitter's children must stay put while the canvas moves around.

    Mode switching reparents the canvas between the layout and the scroll
    area.  If it were reparented to the panel again it would escape the
    splitter and cover the notes pane.
    """
    source = (APP_DIR / "widgets" / "chart_panel.py").read_text(encoding="utf-8")

    assert "self._canvas.setParent(self._chart_area)" in source
    assert not re.search(r"self\._canvas\.setParent\(self\)", source)


# ----------------------------------------------------------------------
# The macOS background fix
# ----------------------------------------------------------------------
# The guards for how the background is painted live in
# test_chart_panel_buffer.py, next to the manual check that reproduces the
# artefacts.  Two of them used to be duplicated here and went stale when the
# viewport stopped being painted by hand: they asserted the presence of the
# code that turned out to be the bug.
def test_a_resize_repaints_the_strip_the_canvas_vacates() -> None:
    source = (APP_DIR / "widgets" / "chart_panel.py").read_text(encoding="utf-8")
    start = source.index("def resizeEvent")
    body = source[start : source.index("\n    def ", start + 10)]

    assert "self.update()" in body
