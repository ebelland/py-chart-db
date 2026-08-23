"""Guards on the two attributes that decide whether the canvas is cleared.

The chart went wrong only on macOS, and only after several redraws: axes drawn
over axes, text growing bolder, ghosts of the previous curve.  Nothing was
wrong with the figure - saving it to PNG produced a clean image every time,
because that path goes back through Agg and never touches the widget.

The cause was two contradictory promises on one widget.  ``FigureCanvasQTAgg``
sets ``WA_OpaquePaintEvent``, meaning "I paint every pixel of my rect, do not
erase it first".  ChartPanel additionally set ``WA_TranslucentBackground``,
meaning "composite me as transparent".  Nothing then clears the rect while Agg
blits an RGBA buffer into it: Windows returns a clean surface by luck, macOS
keeps whatever was there.

These tests are cheap and they pin the arrangement, because the symptom needs a
real window server to reproduce and would otherwise be caught only by a user.
``supporting apps/_testChart.py`` shows it on a Mac.
"""
from __future__ import annotations

from pathlib import Path

import pytest

WIDGET_SOURCE = (
    Path(__file__).resolve().parent.parent / "widgets" / "chart_panel.py"
).read_text(encoding="utf-8")


def _method(name: str) -> str:
    start = WIDGET_SOURCE.index(f"def {name}")
    return WIDGET_SOURCE[start : WIDGET_SOURCE.index("\n    def ", start + 10)]


# ----------------------------------------------------------------------
# The canvas
# ----------------------------------------------------------------------
def test_the_canvas_is_not_translucent() -> None:
    """The bug in one line: a transparent widget that is never erased."""
    body = _method("_configure_canvas")

    assert "WA_TranslucentBackground, True" not in body
    assert "WA_TranslucentBackground, False" in body


def test_the_canvas_has_no_transparent_stylesheet_either() -> None:
    """``background: transparent`` reaches the same surface by another route."""
    assert "background: transparent" not in _method("_configure_canvas")


def test_the_canvas_still_does_not_autofill() -> None:
    """Qt filling it as well would waste a full-rect paint under every frame."""
    assert "setAutoFillBackground(False)" in _method("_configure_canvas")


def test_the_reason_is_written_next_to_the_line() -> None:
    """Someone will read this line and think the False is redundant."""
    body = _method("_configure_canvas")

    assert "WA_OpaquePaintEvent" in body
    assert "_testChart.py" in body


# ----------------------------------------------------------------------
# What paints the area the canvas does not
# ----------------------------------------------------------------------
def test_the_panel_paints_its_own_background() -> None:
    """In FIT_PROPORTIONAL the canvas is centred and smaller than the panel."""
    assert "def paintEvent" in WIDGET_SOURCE
    assert "_fill_background(self)" in _method("paintEvent")


def test_the_viewport_paint_is_not_intercepted() -> None:
    """Intercepting it was the bug, not the fix.

    The filter caught the viewport's Paint, filled the background by hand and
    returned True.  That paints a widget from outside its own paintEvent, and
    the True cancelled the paint Qt was about to run - so the viewport was
    never marked clean and stale content survived anywhere in the window.
    """
    assert "QEvent.Type.Paint" not in WIDGET_SOURCE


def test_the_viewport_erases_itself_instead() -> None:
    """Which it was already configured to do; the filter was cancelling it."""
    body = _method("_apply_background_color_to_widgets")

    assert "setAutoFillBackground(True)" in body
    assert "self._fixed_scroll_area.viewport()" in _method("_background_painted_widgets")


def test_resizing_asks_for_a_repaint() -> None:
    """The strip the canvas vacates is not invalidated by every platform."""
    assert "self.update()" in _method("resizeEvent")


# ----------------------------------------------------------------------
# Live: the canvas really does come back opaque
# ----------------------------------------------------------------------
def test_a_configured_canvas_reports_itself_opaque(qapp) -> None:
    """Read from a real widget, not from the source, so the attribute is real."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PySide6.QtCore import Qt

    canvas = FigureCanvasQTAgg(Figure(figsize=(2.0, 1.5)))
    canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

    assert canvas.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
    assert not canvas.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


def test_redrawing_does_not_accumulate_artists(qapp) -> None:
    """The other half of "ghosts": figures that are never cleared.

    A redraw that adds an axis instead of replacing one produces the same
    stacked look as a stale buffer, and is much easier to do by accident.
    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    canvas = FigureCanvasQTAgg(Figure(figsize=(2.0, 1.5)))

    for _ in range(5):
        canvas.figure.clear()
        canvas.figure.add_subplot(111).plot([0, 1], [0, 1])

    assert len(canvas.figure.axes) == 1


@pytest.mark.parametrize("name", ["_configure_canvas", "paintEvent", "resizeEvent"])
def test_the_painting_methods_are_all_still_here(name: str) -> None:
    """These were lost once already in a rewrite of this widget."""
    assert f"def {name}" in WIDGET_SOURCE


# ----------------------------------------------------------------------
# Logical pixels vs device pixels
# ----------------------------------------------------------------------
# The arithmetic lives in app/utils/hidpi, so it is tested there as functions
# rather than by matching on the source of its callers.  What is checked here
# is that ChartPanel goes through it and keeps nothing scaled in its own state.
def test_the_panel_converts_through_the_shared_module() -> None:
    """Four files did this by hand; one of them omitted the ratio."""
    body = _method("_apply_figure_size_from_canvas")

    assert "logical_to_inches" in body


def test_only_one_place_in_the_app_calls_set_dpi() -> None:
    """Four files used to set a figure's dpi and only three did it correctly.

    The fourth - the figure properties panel - wrote the configured dpi onto
    the very figure ChartPanel was showing, so editing any figure property
    re-broke the invariant ChartPanel had just established.  That is what made
    FIXED zoom wrong again after the resize fix.
    """
    app_dir = Path(__file__).resolve().parent.parent
    offenders = {
        path.relative_to(app_dir).as_posix()
        for path in app_dir.rglob("*.py")
        if "tests" not in path.parts
        and "set_dpi(" in path.read_text(encoding="utf-8")
    }

    assert offenders == {"utils/hidpi.py"}


def test_the_fixed_baseline_is_the_configured_dpi() -> None:
    """Reading it back from the figure is what made zoom compound.

    By the time the metrics are captured the dpi has been multiplied twice:
    once by the display's ratio, once by the zoom.  Storing that as the
    baseline means the next zoom multiplies an already-zoomed number.
    """
    body = _method("_capture_fixed_metrics_from_rendered_figure")

    assert 'rcParams.get("figure.dpi"' in body
    assert "self._figure.get_dpi()" not in body


def test_the_fixed_canvas_size_carries_no_pixel_ratio() -> None:
    """It cancels out: logical pixels are inches x configured dpi x zoom."""
    # Everything after the closing docstring quotes: the code, not the prose
    # explaining why the ratio is absent from it.
    body = _method("_current_figure_pixel_size").split('"""')[-1]

    assert "inches_to_logical" in body
    assert "ratio" not in body


def test_zoom_is_applied_to_the_dpi_not_the_size() -> None:
    """So the figure keeps its physical size and only its resolution changes.

    Zoom multiplies FIXED_MODE_SCREEN_DPI, the fixed on-screen reference,
    rather than the figure's own configured dpi. Using the configured dpi
    here made a figure at a fixed Width/Height grow or shrink on screen
    purely because its print/export dpi changed - see the module note on
    FIXED_MODE_SCREEN_DPI in chart_panel.py.
    """
    body = _method("_apply_fixed_mode_pixel_size")

    assert "_set_configured_dpi(FIXED_MODE_SCREEN_DPI * zoom)" in body
    assert "self._fixed_figure_dpi" not in body


# ----------------------------------------------------------------------
# The conversions themselves
# ----------------------------------------------------------------------
class _FakeCanvas:
    def __init__(self, ratio: float) -> None:
        self.device_pixel_ratio = ratio


@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0, 3.0])
def test_a_configured_dpi_gains_the_ratio_and_gives_it_back(ratio: float) -> None:
    """The invariant in one line: stored values never carry the ratio."""
    from matplotlib.figure import Figure

    from app.utils import hidpi

    figure = Figure()
    canvas = _FakeCanvas(ratio)

    hidpi.apply_configured_dpi(figure, canvas, 100.0)

    assert figure.get_dpi() == pytest.approx(100.0 * ratio)
    assert hidpi.configured_dpi(figure, canvas) == pytest.approx(100.0)


@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0, 3.0])
def test_logical_pixels_survive_the_round_trip(ratio: float) -> None:
    """Logical -> inches -> logical, which both directions must agree on."""
    from matplotlib.figure import Figure

    from app.utils import hidpi

    figure = Figure()
    canvas = _FakeCanvas(ratio)
    hidpi.apply_configured_dpi(figure, canvas, 100.0)

    for logical_px in (320, 778, 1200):
        inches = hidpi.logical_to_inches(logical_px, figure, canvas)
        back = hidpi.inches_to_logical(inches, 100.0)

        assert back == pytest.approx(logical_px)
        # And the Agg buffer really is the widget's full device size.
        assert inches * figure.get_dpi() == pytest.approx(logical_px * ratio)


def test_dropping_the_ratio_would_paint_a_quarter_of_the_widget() -> None:
    """The failure, stated as a number a reader can check."""
    ratio = 2.0
    figure_dpi = 100.0 * ratio
    logical_px = 800

    wrong_inches = logical_px / figure_dpi
    right_inches = logical_px * ratio / figure_dpi

    assert right_inches == 2 * wrong_inches
    assert (wrong_inches / right_inches) ** 2 == 0.25


@pytest.mark.parametrize("ratio", [1.0, 2.0])
def test_zoom_does_not_compound(ratio: float) -> None:
    """Repeated zooms are recomputed from the baseline, not from each other.

    The baseline used to be re-read from a figure that already had the zoom in
    it, so 200% twice gave 400%, then 800%.  The on-screen size must depend
    only on the current zoom, and must not depend on the display at all.
    """
    from matplotlib.figure import Figure

    from app.utils import hidpi

    figure = Figure()
    canvas = _FakeCanvas(ratio)
    configured = 100.0
    inches = 16.0

    sizes = []
    for zoom in (2.0, 2.0, 1.0, 2.5, 2.5):
        hidpi.apply_configured_dpi(figure, canvas, configured * zoom)
        sizes.append(round(hidpi.inches_to_logical(inches, configured, zoom)))
        # The baseline is untouched by what was just done to the figure.
        assert configured == 100.0

    assert sizes == [3200, 3200, 1600, 4000, 4000]


def test_the_dpi_helpers_survive_a_missing_canvas(qapp) -> None:
    """ChartPanel prepares its figure from rcParams before building the canvas.

    ``_reset_figure_metrics_from_rcparams_for_reload`` runs from the
    constructor, at a point where ``self._canvas`` does not exist yet.  Reaching
    for it raised AttributeError on every panel, and the try/except around the
    caller turned that into a logged failure rather than a crash - so the dpi
    was simply never applied.
    """
    from matplotlib.figure import Figure

    from app.utils import hidpi

    figure = Figure()

    assert hidpi.canvas_pixel_ratio(None) == 1.0
    assert hidpi.apply_configured_dpi(figure, None, 120.0) == pytest.approx(120.0)
    assert hidpi.configured_dpi(figure, None) == pytest.approx(120.0)


def test_the_panel_reaches_for_the_canvas_defensively() -> None:
    """Both helpers run before the canvas is constructed."""
    for name in ("_device_pixel_ratio", "_set_configured_dpi"):
        assert 'getattr(self, "_canvas", None)' in _method(name), name


def test_a_panel_can_be_constructed(qapp, tmp_path) -> None:
    """The end of it: build one and read back what the arithmetic produced.

    No test built a ChartPanel, which is why an AttributeError in its
    constructor reached the user instead of the suite.
    """
    from app.data.sqlite_repo import SqliteRepo
    from app.widgets.chart_panel import ChartPanel

    repo = SqliteRepo(db_path=tmp_path / "panel.dhub")
    try:
        figure_id = repo.create_figure_descriptor(name="panel")
        panel = ChartPanel(repo, figure_id)

        # The configured dpi is stored unscaled; the figure carries the ratio,
        # and in FIXED mode the zoom as well.  That is the whole invariant:
        # every factor is applied to the figure and none is stored.
        ratio = panel._device_pixel_ratio()
        zoom = panel._zoom_percent / 100.0 if panel._resize_mode == "FIXED" else 1.0

        assert panel._fixed_figure_dpi > 0
        assert panel._figure.get_dpi() == pytest.approx(
            panel._fixed_figure_dpi * ratio * zoom
        )
    finally:
        repo.close()


def test_zooming_a_real_panel_does_not_compound(qapp, tmp_path) -> None:
    """Set the same zoom twice and the canvas must not grow."""
    from app.data.sqlite_repo import SqliteRepo
    from app.widgets.chart_panel import ChartPanel

    repo = SqliteRepo(db_path=tmp_path / "zoom.dhub")
    try:
        panel = ChartPanel(repo, repo.create_figure_descriptor(name="zoom"))
        panel.set_resize_mode("FIXED", persist=False, redraw=False)

        panel.set_zoom_percent(200)
        first = panel._current_figure_pixel_size(apply_zoom=True)
        panel.set_zoom_percent(200)
        again = panel._current_figure_pixel_size(apply_zoom=True)

        assert first == again

        panel.set_zoom_percent(100)
        unzoomed = panel._current_figure_pixel_size(apply_zoom=True)
        assert first.width() == pytest.approx(unzoomed.width() * 2, abs=2)
    finally:
        repo.close()
