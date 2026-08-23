# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnnecessaryComparison=false
from __future__ import annotations

"""ChartPanel with direct Matplotlib canvas sizing.

This version deliberately avoids an intermediate host widget for FIT and
FIT_PROPORTIONAL modes.  The QScrollArea owns the visible chart viewport and the
FigureCanvasQTAgg is the scroll area's widget directly.
"""

from copy import deepcopy
from io import BytesIO
import math
from collections.abc import MutableMapping
from typing import Any, Final

from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Qt, QTimer, Signal
import PySide6.QtGui
from PySide6.QtWidgets import QApplication, QAbstractScrollArea, QFileDialog, QFrame, QHBoxLayout, QLabel, QMenu, QScrollArea, QSizePolicy, QSlider, QSplitter, QToolButton, QVBoxLayout, QWidget

import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle, Wedge

from app.charts.render_figure import render_figure_from_descriptor
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils.messages import ask, show_message
from app.styles.style import SPLITTER_HANDLE_WIDTH, MenuItem, create_menu, create_toolbar_button
from app.utils.config import get_value, load_config, save_config
from app.utils.figure_metrics import CM_PER_INCH, figure_metrics_from_options
from app.utils.hidpi import (
    apply_configured_dpi,
    canvas_pixel_ratio,
    inches_to_logical,
    logical_to_inches,
)
from app.widgets.html_results import HtmlResultsView
from app.utils.i18n import _


TOOLBAR_ICON_SIZE: Final[QSize] = QSize(24, 24)
TOOLBAR_ACTIONS_TO_REMOVE: Final[set[str]] = {"Save","Subplots", "Customize", "Configure subplots", "Edit axis", "Edit colors"}
DEFAULT_ASPECT_RATIO: Final[float] = 16.0 / 9.0
CONFIG_SECTION: Final[str] = "chart_panel"
CONFIG_RESIZE_MODE: Final[str] = "resize_mode"
CONFIG_MIN_ZOOM: Final[str] = "min_zoom_percent"
CONFIG_MAX_ZOOM: Final[str] = "max_zoom_percent"
CONFIG_INITIAL_ZOOM: Final[str] = "initial_zoom_percent"
CONFIG_BACKGROUND_COLOR: Final[str] = "background_color"
CONFIG_COPY_DPI: Final[str] = "copy_dpi"
FIGURE_VIEW_OPTIONS_KEY: Final[str] = "view"
RESIZE_MODES: Final[tuple[str, ...]] = ("FIT", "FIT_PROPORTIONAL", "FIXED")

#: (key, label, tooltip), in the order every picker offers them. Shared so the
#: Figure Properties combo - the only place this choice is made now - cannot
#: drift from the mode names ChartPanel itself understands.
RESIZE_MODE_CHOICES: Final[tuple[tuple[str, str, str], ...]] = (
    ("FIT_PROPORTIONAL", "Fit proportional", "Fit chart keeping proportions"),
    ("FIT", "Fit", "Fit chart without keeping proportions"),
    ("FIXED", "Fixed", "Fixed size with zoom and scrollbars"),
)
CHART_AREA_MIN_HEIGHT: Final[int] = 160
NOTES_MIN_HEIGHT: Final[int] = 0
NOTES_FIRST_OPEN_MIN_HEIGHT: Final[int] = 80
NOTES_INITIAL_FRACTION: Final[float] = 0.3
GRID_ROW_KEYS: Final[tuple[str, ...]] = ("rows", "nrows", "n_rows", "row_count", "num_rows")
GRID_COL_KEYS: Final[tuple[str, ...]] = ("cols", "columns", "ncols", "n_cols", "col_count", "num_cols")
AXIS_COLLECTION_KEYS: Final[tuple[str, ...]] = ("axes", "subplots", "plots", "charts", "panels")

#: How near the cursor has to be, in points, for a marker to count as picked.
#: Five is about a default marker's own radius: close enough that a click has
#: to be deliberate, wide enough that it does not have to be exact.
PICK_TOLERANCE_POINTS: Final[float] = 5.0

#: FIXED mode's on-screen reference: the dpi used to turn a figure's physical
#: size in inches into logical widget pixels, at 100% zoom.  Deliberately not
#: the figure's own configured dpi - dpi is a print/export resolution, and
#: using it for on-screen sizing too meant a figure at a fixed 20x15cm looked
#: three times as big on nothing more than its dpi going from 100 to 300,
#: although Width and Height never changed.  100 matches Matplotlib's own
#: default figure.dpi, so a figure saved before per-figure dpi existed keeps
#: the on-screen size it always had.
FIXED_MODE_SCREEN_DPI: Final[float] = 100.0


def axis_text(axis: Any, value: float) -> str:
    """Format *value* the way *axis* would label it.

    Asking the axis rather than formatting the float ourselves is what makes a
    time series read ``2024-03-07`` instead of ``19789``: the renderers convert
    timestamps to epoch seconds before plotting, and only the axis formatter
    knows how to turn one back. The same call gives sensible precision on a log
    axis and on a percentage axis for free, and it is the call Matplotlib's own
    navigation toolbar makes for its cursor readout - so a picked point and the
    corner of the toolbar agree about how a number is written, including the
    trailing zeros ScalarFormatter pads out ("2.000", not "2").

    Formatters are third-party code as far as this panel is concerned - a
    renderer may install any callable - so a formatter that raises degrades to
    the plain number rather than losing the whole readout.
    """
    try:
        text = axis.get_major_formatter().format_data_short(value).strip()
    except Exception:
        text = ""
    return text or f"{value:g}"


class ChartPanel(QFrame):
    """Embed a repository-backed Matplotlib chart.

    Important sizing rule:
        FIT and FIT_PROPORTIONAL do not use any host panel. The Matplotlib
        canvas is installed directly into the QScrollArea.
    """

    delete_requested = Signal(int)

    #: Emitted with a one-line description of whatever the user just clicked on
    #: in the chart. The panel has no status bar of its own, and deliberately
    #: does not go looking for the window's: it says what was selected and lets
    #: whoever owns the window decide where that belongs.
    selection_changed = Signal(str)

    def __init__(
        self,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        """Initialise the panel and render its first chart state."""
        super().__init__(parent)

        self._repo = repo
        self._figure_id = int(figure_id)
        self._deleted = False
        self._pending_canvas_sync = False
        self._pending_canvas_redraw = False
        self._last_canvas_size = QSize()

        self._canvas_sync_timer = QTimer(self)
        self._canvas_sync_timer.setSingleShot(True)
        self._canvas_sync_timer.timeout.connect(self._run_pending_canvas_geometry_sync)

        # A short second pass catches tab/page activation and final layout
        # geometry without changing the old gwXG resize architecture.
        self._canvas_late_sync_timer = QTimer(self)
        self._canvas_late_sync_timer.setSingleShot(True)
        self._canvas_late_sync_timer.timeout.connect(self._run_pending_canvas_geometry_sync)

        self._config = load_config()
        
        chart_config_obj = self._config.setdefault(CONFIG_SECTION, {})
        if not isinstance(chart_config_obj, dict):
            chart_config_obj = {}
            self._config[CONFIG_SECTION] = chart_config_obj
        chart_config: dict[str, Any] = chart_config_obj

        self._min_zoom_percent:int = int(
            chart_config.get(CONFIG_MIN_ZOOM, 25)
        )
        self._max_zoom_percent:int = int(
            chart_config.get(CONFIG_MAX_ZOOM, 250)
        )
        if self._max_zoom_percent < self._min_zoom_percent:
            self._max_zoom_percent = self._min_zoom_percent

        self._zoom_percent:int = self._clamp_zoom(
            int(chart_config.get(CONFIG_INITIAL_ZOOM, 64))
        )
        self._resize_mode = self._normalized_resize_mode(
            chart_config.get(CONFIG_RESIZE_MODE, "FIT")
        )
        self._background_color: str = chart_config.get(CONFIG_BACKGROUND_COLOR, "#ffffff")

        # The config value is only the default for newly-created figures.
        # If this figure has its own view state, it wins.
        self._apply_persisted_view_state()

        self._apply_persisted_figure_metrics_to_rcparams()

        self._figure = Figure()
        self._reset_figure_metrics_from_rcparams_for_reload()
        self._fixed_figure_size_inches = tuple(
            float(v) for v in self._figure.get_size_inches()
        )
        # The configured dpi, never the figure's: see
        # _capture_fixed_metrics_from_rendered_figure.
        self._fixed_figure_dpi = float(rcParams.get("figure.dpi", 100.0) or 100.0)

        # Legend handle -> the artists it stands for. Rebuilt by every render,
        # bound here so a pick arriving before the first one cannot fail.
        self._legend_targets: dict[Any, list[Any]] = {}

        self._canvas = FigureCanvasQTAgg(self._figure)
        self._init_hover_readout()
        self._toolbar = NavigationToolbar2QT(
            self._canvas,
            parent=self,
            coordinates=True,
        )

        self._init_ui()
        self.set_resize_mode(self._resize_mode, persist=False, redraw=False)
        self.reload()

    # ------------------------------------------------------------------
    # UI initialisation
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        """Build widgets with direct-canvas FIT modes and scrollable FIXED mode.

        Display policy:
            * FIT: the Matplotlib canvas is placed directly in ChartPanel's
              main layout below the toolbar row and fills that client area.
            * FIT_PROPORTIONAL: the same direct canvas placement is used,
              but the canvas width/height are fixed to preserve the natural
              figure aspect ratio.
            * FIXED: the canvas is moved into a QScrollArea.  The scroll-area
              viewport shows the configured chart panel background color.
        """
        self.setObjectName("ChartPanel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(0, 0)

        self._configure_canvas()
        self._configure_toolbar()
        self._actions_menu = self._build_actions_menu()
        self._top_row = self._build_top_row()

        self._fixed_scroll_area = QScrollArea(self)
        self._fixed_scroll_area.setObjectName("chartScrollArea")
        self._fixed_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._fixed_scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fixed_scroll_area.setWidgetResizable(False)
        self._fixed_scroll_area.viewport().installEventFilter(self)
        self._configure_fixed_scroll_area()

        # Keep the old gwXG chart layout intact, but host it in a stable
        # splitter child so the HTML results pane can be resized and collapsed
        # independently from the chart display.
        self._chart_area = QWidget(self)
        self._chart_area.setMinimumHeight(CHART_AREA_MIN_HEIGHT)
        self._chart_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._main_layout = QVBoxLayout(self._chart_area)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)
        self._main_layout.addWidget(self._top_row, 0)
        self._main_layout.addWidget(self._canvas, 1)

        self._notes_html = ""
        self._notes_view = HtmlResultsView(self)
        self._notes_view.setObjectName("chartNotesView")
        self._notes_view.setMinimumHeight(NOTES_MIN_HEIGHT)
        self._notes_view.clear_requested.connect(self.clear_notes)

        self._content_split = QSplitter(Qt.Orientation.Vertical, self)
        self._content_split.setObjectName("chartContentSplit")
        self._content_split.setChildrenCollapsible(True)
        self._content_split.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        self._content_split.addWidget(self._chart_area)
        self._content_split.addWidget(self._notes_view)
        self._content_split.setCollapsible(0, False)
        self._content_split.setCollapsible(1, True)
        self._content_split.setStretchFactor(0, 1)
        self._content_split.setStretchFactor(1, 0)

        panel_layout = QVBoxLayout(self)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        panel_layout.addWidget(self._content_split, 1)

        # Empty until an operation publishes something.
        self._notes_view.setVisible(False)
        self._restore_notes_state()

        self._active_chart_widget: QWidget = self._canvas
        self._apply_background_color_to_widgets()

    # ------------------------------------------------------------------
    # Notes / HTML results pane
    # ------------------------------------------------------------------
    def set_notes_html(self, markup: str, *, append: bool = False) -> None:
        """Show HTML results below the chart, revealing the pane if needed."""
        markup = str(markup or "")
        if not markup.strip():
            return

        if append and self._notes_html.strip():
            self._notes_html = f"{self._notes_html}<hr>{markup}"
        else:
            self._notes_html = markup

        self._notes_view.setHtml(self._notes_html)
        self._set_notes_visible(True)
        self._persist_view_state()

    def notes_html(self) -> str:
        """Return the markup currently shown below the chart."""
        return self._notes_html

    def clear_notes(self) -> None:
        """Empty and fully collapse the notes pane."""
        self._notes_html = ""
        self._notes_view.clear()
        self._set_notes_visible(False)
        self._persist_view_state()

    def _restore_notes_state(self) -> None:
        """Re-show persisted notes and splitter position for this figure."""
        markup = getattr(self, "_pending_notes_html", "")
        if not markup:
            return

        self._notes_html = markup
        self._notes_view.setHtml(markup)
        self._notes_view.setVisible(True)

        sizes = getattr(self, "_pending_notes_sizes", None)
        if sizes:
            self._content_split.setSizes(sizes)

    def _set_notes_visible(self, visible: bool) -> None:
        """Show/hide notes and let users collapse the pane completely."""
        was_visible = self._notes_view.isVisible()
        self._notes_view.setVisible(visible)
        if not visible:
            sizes = self._content_split.sizes()
            chart_height = max(CHART_AREA_MIN_HEIGHT, sum(sizes) if sizes else self.height())
            self._content_split.setSizes([chart_height, 0])
            self._schedule_canvas_geometry_sync(redraw=True, late=True)
            return
        if was_visible:
            return

        total = max(self.height(), CHART_AREA_MIN_HEIGHT + NOTES_FIRST_OPEN_MIN_HEIGHT)
        notes_height = max(NOTES_FIRST_OPEN_MIN_HEIGHT, int(total * NOTES_INITIAL_FRACTION))
        self._content_split.setSizes([total - notes_height, notes_height])
        self._schedule_canvas_geometry_sync(redraw=True, late=True)

    def _on_scroll_zoom(self, event: Any) -> None:
        """Zoom about the cursor on Ctrl/Cmd + wheel.

        Modifier-gated deliberately.  In FIXED mode the canvas lives in a
        QScrollArea and a bare wheel scrolls it, which is what a bare wheel
        should do; taking that over would make the chart behave differently
        from every other scrollable thing in the window.  Requiring the
        modifier also means the two gestures never fight, in any mode.

        Zooming about the cursor rather than about the centre is the whole
        point: it is what lets you walk into a feature by pointing at it,
        instead of alternating zoom and pan.
        """
        if event.inaxes is None or not event.step:
            return
        # Matplotlib reports held modifiers in event.key; on macOS Qt maps the
        # Command key to Control, so this is Cmd there without special-casing.
        if "control" not in str(event.key or ""):
            return

        # One notch in is a smaller window, one notch out a larger one.
        factor = self.ZOOM_STEP if event.step > 0 else 1.0 / self.ZOOM_STEP

        axes = event.inaxes
        for get_limits, set_limits, position in (
            (axes.get_xlim, axes.set_xlim, event.xdata),
            (axes.get_ylim, axes.set_ylim, event.ydata),
        ):
            low, high = get_limits()
            if position is None or not math.isfinite(position):
                continue
            # Each side is scaled by its own distance to the cursor, which is
            # what keeps the point under the pointer fixed while the rest of
            # the range closes in on it.
            set_limits(
                position - (position - low) * factor,
                position + (high - position) * factor,
            )

        self._canvas.draw_idle()

    def _make_data_artists_pickable(self) -> None:
        """Arm picking on the lines and marker collections a render produced.

        Done here rather than by the renderers because it is a *viewer*
        decision, not a chart property: every renderer would otherwise have to
        remember to opt in, and a figure saved by one version would come back
        unpickable in another. Setting it after the render also covers the
        renderers that draw through Matplotlib helpers we do not construct
        ourselves.

        Lines and collections report a point. Patches - bars and wedges - are
        armed too but read out differently: a patch has no per-point index, so
        the old code left them unpickable entirely and clicking a bar chart did
        nothing at all. The fix is not to force them through the point path,
        which would say "1 point" and mean the whole shape, but to report what
        is actually meaningful about a bar: its category and its value. See
        _describe_patch.

        The legend is armed last, and for a different purpose: clicking a
        legend entry toggles its series rather than reading anything out.
        """
        for axes in self._figure.axes:
            for artist in (*axes.lines, *axes.collections):
                artist.set_picker(PICK_TOLERANCE_POINTS)

            # True, not a tolerance in points: a patch is a filled area, so
            # "inside the shape" is the hit test, and a distance from its edge
            # would make the middle of a tall bar unclickable.
            for patch in axes.patches:
                patch.set_picker(True)

        self._make_legend_pickable()

    def _make_legend_pickable(self) -> None:
        """Let a click on a legend entry hide and show its series.

        Wired here rather than at render time for the same reason as the data
        artists: the legend is rebuilt by every render, so the connection has
        to be re-established against the new handles each time.
        """
        self._legend_targets = {}

        for axes in self._figure.axes:
            legend = axes.get_legend()
            if legend is None:
                continue

            handles = list(getattr(legend, "legend_handles", None) or [])
            labels = [text.get_text() for text in legend.get_texts()]

            # Map each legend handle back to the artists it stands for. Matched
            # by label rather than by position: a renderer that draws a series
            # as several artists - a line plus its error bars - produces one
            # legend entry for the group, and toggling only the first would
            # leave the error bars floating on their own.
            by_label: dict[str, list[Any]] = {}
            for artist in (*axes.lines, *axes.collections, *axes.patches):
                label = str(artist.get_label() or "")
                if label and not label.startswith("_"):
                    by_label.setdefault(label, []).append(artist)

            for handle, label in zip(handles, labels):
                targets = by_label.get(str(label), [])
                if not targets:
                    continue
                handle.set_picker(PICK_TOLERANCE_POINTS)
                self._legend_targets[handle] = targets

            for text in legend.get_texts():
                # The label text is the larger target and the one people
                # actually aim at; the swatch alone is a few pixels.
                targets = by_label.get(str(text.get_text()), [])
                if targets:
                    text.set_picker(True)
                    self._legend_targets[text] = targets

    # Ctrl + left click used to append an annotation to the axis options here
    # and reload. It is gone: the text was the fixed word "Annotation", there
    # was no way to edit it, and nothing on the chart deleted it - the only
    # way back was the annotations table in the axis properties. A dialog that
    # asks for the text at the point clicked replaces it (see N-7 in
    # todo.txt); the axes-to-descriptor mapping it will need is in this file's
    # history.

    def _on_pick(self, event: Any) -> None:
        """Describe the clicked point, or the clicked group of points.

        Matplotlib reports a pick as an artist plus ``ind``, a list of the
        indices under the cursor - one entry for a marker clicked on its own,
        several where markers overlap or where a rubber-band selection covers
        them. That single mechanism covers both cases the readout has to
        distinguish, so there is no separate "multiple selection" path here:
        one index reads out the value, several read out the summary.
        """
        artist = event.artist

        # Legend first: a legend handle is also a line or a patch, so testing
        # it after the data cases would read out the swatch's own coordinates
        # instead of toggling the series.
        if artist in getattr(self, "_legend_targets", {}):
            self._toggle_series_visibility(artist)
            return

        # A patch carries no index, so it never reaches the point path below.
        if isinstance(artist, Patch):
            self._describe_patch(artist)
            return

        indices = [int(index) for index in getattr(event, "ind", []) or []]
        if not indices:
            return

        # A scatter is a PathCollection and carries its points as Nx2 offsets;
        # a line carries two parallel sequences. Nothing else is made pickable.
        if hasattr(artist, "get_offsets"):
            points = artist.get_offsets()
            pairs = [(float(points[i][0]), float(points[i][1])) for i in indices]
        else:
            x_data, y_data = artist.get_data()
            pairs = [(float(x_data[i]), float(y_data[i])) for i in indices]

        axes = artist.axes
        # Matplotlib invents a label for an artist that was not given one -
        # "_child0", "_collection2" - and marks decoration as "_nolegend_".
        # Both start with an underscore by that convention, and neither is
        # worth showing, so an underscore means "no name" here.
        series = str(artist.get_label() or "").strip()
        if not series or series.startswith("_"):
            series = _("series")

        if len(pairs) == 1:
            x_value, y_value = pairs[0]
            self.selection_changed.emit(
                _("{series} — x: {x}, y: {y}").format(
                    series=series,
                    x=axis_text(axes.xaxis, x_value),
                    y=axis_text(axes.yaxis, y_value),
                )
            )
            return

        self.selection_changed.emit(
            _("{series} — {count} points, mean x: {x}, mean y: {y}").format(
                series=series,
                count=len(pairs),
                x=axis_text(axes.xaxis, sum(x for x, _unused in pairs) / len(pairs)),
                y=axis_text(axes.yaxis, sum(y for _unused, y in pairs) / len(pairs)),
            )
        )

    def _toggle_series_visibility(self, handle: Any) -> None:
        """Show or hide the series a legend entry stands for.

        The legend entry itself is dimmed rather than removed, so a hidden
        series still has something to click to bring it back - a legend that
        deleted its own entry would be a one-way door.
        """
        targets = self._legend_targets.get(handle) or []
        if not targets:
            return

        visible = not targets[0].get_visible()
        for artist in targets:
            artist.set_visible(visible)

        handle.set_alpha(1.0 if visible else 0.25)

        label = str(targets[0].get_label() or "").strip() or _("series")
        self.selection_changed.emit(
            _("{series} shown").format(series=label)
            if visible
            else _("{series} hidden").format(series=label)
        )
        self._canvas.draw_idle()

    def _describe_patch(self, patch: Any) -> None:
        """Read out a bar or a wedge by what it means, not by index.

        A patch has no per-point index, which is why these were left
        unpickable and clicking a bar chart did nothing. What is meaningful
        about a bar is its category and its height, and both can be recovered
        from the geometry: the tick label under its centre names it, and the
        rectangle's height is the value.

        A wedge carries its share in its angles rather than in a height, so it
        reports a percentage instead.
        """
        axes = patch.axes
        if axes is None:
            return

        series = str(patch.get_label() or "").strip()
        if not series or series.startswith("_"):
            series = ""

        if isinstance(patch, Wedge):
            share = abs(patch.theta2 - patch.theta1) / 360.0
            name = series or _("slice")
            self.selection_changed.emit(
                _("{series} — {percent:.1f}% of the total").format(
                    series=name, percent=share * 100.0
                )
            )
            return

        if isinstance(patch, Rectangle):
            width = float(patch.get_width())
            height = float(patch.get_height())
            if self._bar_is_horizontal(axes, patch):
                value, value_axis = width, axes.xaxis
                centre, category_axis = patch.get_y() + height / 2.0, axes.yaxis
            else:
                value, value_axis = height, axes.yaxis
                centre, category_axis = patch.get_x() + width / 2.0, axes.xaxis

            category = self._category_at(category_axis, centre)
            label = " — ".join(part for part in (series, category) if part)
            self.selection_changed.emit(
                _("{label}: {value}").format(
                    label=label or _("bar"),
                    value=axis_text(value_axis, value),
                )
            )
            return

        if series:
            self.selection_changed.emit(series)

    @staticmethod
    def _bar_is_horizontal(axes: Any, patch: Any) -> bool:
        """Say whether a bar grows along x rather than along y.

        Read from the BarContainer that owns the patch, which records the
        orientation ax.bar/ax.barh gave it. Guessing from the rectangle's own
        geometry does not work: a tall thin horizontal bar and a tall thin
        vertical bar have the same shape, and only the container knows which
        dimension is the value.

        The geometric fallback is for a patch drawn directly rather than
        through the bar helpers, where there is no container to ask.
        """
        for container in getattr(axes, "containers", []) or []:
            if patch in list(getattr(container, "patches", []) or []):
                return str(getattr(container, "orientation", "vertical")) == "horizontal"

        # No container: a bar sits on its baseline, so the coordinate that is
        # pinned to zero tells us which way it grows.
        return float(patch.get_x()) == 0.0 and float(patch.get_y()) != 0.0

    @staticmethod
    def _category_at(axis: Any, position: float) -> str:
        """Return the tick label at a position, or the position itself.

        A categorical bar chart puts the category names in the tick labels and
        the bars at integer positions, so the label under the bar's centre is
        its name. A numeric bar chart has no such names, and there the position
        formatted by the axis is the honest answer.
        """
        try:
            locations = list(axis.get_ticklocs())
            labels = [text.get_text() for text in axis.get_ticklabels()]
            for location, label in zip(locations, labels):
                if abs(float(location) - position) < 0.5 and label:
                    return str(label)
        except Exception:
            pass
        return axis_text(axis, position)

    # ------------------------------------------------------------------
    # Hover readout
    # ------------------------------------------------------------------

    #: How long the pointer must be still before the readout is recomputed.
    #: Mouse motion arrives far faster than a hit test plus a blit can be done,
    #: so handling every event makes the chart lag behind the cursor. 40ms is
    #: below the ~100ms at which a response stops feeling immediate, and it
    #: collapses a fast sweep across the plot into a handful of tests instead
    #: of hundreds.
    HOVER_INTERVAL_MS: Final[int] = 40

    #: Pixels. Wider than the click tolerance: hovering is a coarse gesture and
    #: nobody positions the pointer precisely before reading a value.
    HOVER_TOLERANCE_PIXELS: Final[float] = 12.0

    def _init_hover_readout(self) -> None:
        """Prepare the throttle timer and the blitting state."""
        self._hover_event: Any | None = None
        self._hover_annotation: Any | None = None
        self._hover_background: Any | None = None
        self._hover_axes: Any | None = None

        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(self.HOVER_INTERVAL_MS)
        self._hover_timer.timeout.connect(self._update_hover_readout)

    def _on_motion(self, event: Any) -> None:
        """Record the pointer and let the timer decide when to act.

        Deliberately does no work: this runs on every mouse move, and a hit
        test here would be the thing that makes the chart feel slow.
        """
        self._hover_event = event
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _invalidate_hover_background(self) -> None:
        """Drop the cached pixels after anything that repaints the canvas.

        The background is a snapshot of the chart without the annotation on it.
        Zooming, panning and resizing all invalidate it, and blitting a stale
        one paints the old chart back over the new one.

        Only the bitmap is dropped. The annotation artist itself survives a
        repaint, and clearing _hover_axes here would make every hover rebuild
        it and force the full draw that blitting exists to avoid - the blit
        path would then never be reached at all.
        """
        self._hover_background = None

    def _discard_hover_annotation(self) -> None:
        """Forget the annotation artist itself, after the figure is cleared.

        figure.clear() destroys every artist including this one, so the
        reference left behind belongs to no axes and cannot be drawn.
        """
        self._hover_annotation = None
        self._hover_axes = None
        self._hover_background = None

    def _update_hover_readout(self) -> None:
        """Find the point under the pointer and draw its value beside it."""
        event = self._hover_event
        if event is None or getattr(event, "inaxes", None) is None:
            self._hide_hover_annotation()
            return

        hit = self._nearest_point_to(event)
        if hit is None:
            self._hide_hover_annotation()
            return

        artist, x_value, y_value = hit
        axes = event.inaxes

        series = str(artist.get_label() or "").strip()
        if not series or series.startswith("_"):
            series = _("series")

        text = _("{series}\nx: {x}\ny: {y}").format(
            series=series,
            x=axis_text(axes.xaxis, x_value),
            y=axis_text(axes.yaxis, y_value),
        )

        annotation = self._hover_annotation_for(axes)
        annotation.xy = (x_value, y_value)
        annotation.set_text(text)
        annotation.set_visible(True)
        self._blit_hover(axes, annotation)

    def _nearest_point_to(self, event: Any) -> tuple[Any, float, float] | None:
        """Return (artist, x, y) for the closest data point, or None.

        Distance is measured in display pixels, not in data units: the two axes
        rarely share a scale, and a chart of millivolts against seconds would
        otherwise treat a step along x as thousands of times nearer than a step
        along y.
        """
        axes = event.inaxes
        best: tuple[float, Any, float, float] | None = None

        for artist in (*axes.lines, *axes.collections):
            if not artist.get_visible():
                continue

            try:
                if hasattr(artist, "get_offsets"):
                    points = np.asarray(artist.get_offsets(), dtype=float)
                    if points.size == 0:
                        continue
                    xs, ys = points[:, 0], points[:, 1]
                else:
                    xs, ys = (np.asarray(a, dtype=float) for a in artist.get_data())
                if xs.size == 0:
                    continue

                pixels = axes.transData.transform(np.column_stack([xs, ys]))
            except Exception:
                continue

            offsets = pixels - np.array([event.x, event.y], dtype=float)
            distances = np.hypot(offsets[:, 0], offsets[:, 1])
            index = int(np.nanargmin(distances))
            distance = float(distances[index])

            if distance <= self.HOVER_TOLERANCE_PIXELS and (
                best is None or distance < best[0]
            ):
                best = (distance, artist, float(xs[index]), float(ys[index]))

        return None if best is None else (best[1], best[2], best[3])

    def _hover_annotation_for(self, axes: Any) -> Any:
        """Return the annotation artist, moving it to *axes* when needed.

        One artist reused rather than one per axes: it is only ever visible in
        the axes under the pointer, and a per-axes cache would have to be
        invalidated on every re-render alongside everything else.
        """
        if self._hover_annotation is not None and self._hover_axes is axes:
            return self._hover_annotation

        if self._hover_annotation is not None:
            try:
                self._hover_annotation.remove()
            except Exception:
                pass
            # The annotation was part of the old axes' contents, so the cached
            # background no longer matches what is on screen.
            self._hover_background = None

        self._hover_annotation = axes.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            zorder=10_000,
            bbox={"boxstyle": "round,pad=0.4", "fc": "#ffffe0", "ec": "#808080", "alpha": 0.95},
            arrowprops={"arrowstyle": "-", "color": "#808080", "linewidth": 0.8},
            annotation_clip=False,
        )
        self._hover_annotation.set_visible(False)
        self._hover_axes = axes
        return self._hover_annotation

    def _blit_hover(self, axes: Any, annotation: Any) -> None:
        """Repaint only the annotation, over a cached copy of the chart.

        This is the whole reason hovering is affordable. A full draw() re-runs
        every renderer for every series; blitting restores a bitmap and draws
        one text box, so the cost does not grow with the size of the data.
        """
        canvas = self._canvas
        try:
            if self._hover_background is None:
                # Capture without the annotation, or it would be baked into
                # the background and smear across the plot as the pointer moves.
                annotation.set_visible(False)
                canvas.draw()
                self._hover_background = canvas.copy_from_bbox(self._figure.bbox)
                annotation.set_visible(True)

            canvas.restore_region(self._hover_background)
            axes.draw_artist(annotation)
            canvas.blit(self._figure.bbox)
        except Exception:
            # Any backend that cannot blit still gets a correct, slower chart.
            self._hover_background = None
            canvas.draw_idle()

    def _hide_hover_annotation(self) -> None:
        """Take the readout away once the pointer leaves every point."""
        annotation = self._hover_annotation
        if annotation is None or not annotation.get_visible():
            return

        annotation.set_visible(False)
        try:
            if self._hover_background is not None:
                self._canvas.restore_region(self._hover_background)
                self._canvas.blit(self._figure.bbox)
                return
        except Exception:
            self._hover_background = None
        self._canvas.draw_idle()

    def _configure_fixed_scroll_area(self) -> None:
        """Configure the scroll area used only by FIXED mode."""
        self._fixed_scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        self._fixed_scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._fixed_scroll_area.setMinimumSize(0, 0)
        self._fixed_scroll_area.setMaximumSize(QSize(16777215, 16777215))
        self._fixed_scroll_area.viewport().setMinimumSize(0, 0)

    #: How much one wheel notch changes the visible range.  0.8 zooms in by a
    #: fifth per notch: small enough to stop where you meant to, large enough
    #: that crossing two orders of magnitude does not need forty notches.
    ZOOM_STEP: Final[float] = 0.8

    def _configure_canvas(self) -> None:
        """Configure the Matplotlib canvas for direct scroll-area use."""
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Connected through Matplotlib rather than through a Qt event filter.
        # The handler needs the cursor position in *data* coordinates to zoom
        # about the point under it, and mpl_connect delivers exactly that;
        # a Qt wheel event carries pixels, which would have to be transformed
        # back by hand for each axis.
        self._canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        # Picking is likewise Matplotlib's own: it already knows which artist
        # owns each pixel and which of its points are under the cursor, and
        # redoing that hit test against Qt coordinates would mean reimplementing
        # marker sizes, transforms and axis scales for every renderer.
        self._canvas.mpl_connect("pick_event", self._on_pick)
        # Ctrl + left click creates an axis-level annotation at the clicked
        # data coordinates.  This is deliberately a mouse-button event, not
        # a pick event, so it works even when there is no marker exactly
        # under the cursor.
        # Hover readout. Throttled and blitted; see _on_motion.
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        # Any of these repaints the canvas, so the cached background that
        # blitting restores is no longer what is underneath.
        self._canvas.mpl_connect("draw_event", lambda _e: self._invalidate_hover_background())
        self._canvas.mpl_connect("resize_event", lambda _e: self._invalidate_hover_background())
        self._canvas.mpl_connect("axes_leave_event", lambda _e: self._hide_hover_annotation())
        self._canvas.setMinimumSize(0, 0)
        self._canvas.setMaximumSize(QSize(16777215, 16777215))
        self._canvas.setAutoFillBackground(False)
        # Deliberately *not* translucent.  FigureCanvasQTAgg sets
        # WA_OpaquePaintEvent - "I paint every pixel, do not erase me" - and
        # WA_TranslucentBackground is the opposite promise, that the background
        # shows through.  With both set nothing ever clears the canvas rect
        # while Agg blits an RGBA buffer over it: Windows happens to hand back
        # a clean surface, macOS keeps the previous frame, and successive draws
        # pile up as ghost axes and thickening text.
        #
        # There is nothing to see through it in any case: paintEvent fills the
        # panel and the viewport filter fills the strip the canvas does not
        # cover, so the transparency only ever exposed stale pixels.
        #
        # _testChart.py, beside main.py, shows the difference on a Mac.
        self._canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self._canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._canvas.installEventFilter(self)
        self._canvas.customContextMenuRequested.connect(self._show_context_menu)

    def _configure_toolbar(self) -> None:
        """Make the Matplotlib toolbar compact and QSS-addressable."""
        self._toolbar.setObjectName("chartToolbar")
        self._toolbar.setMovable(False)
        self._toolbar.setFloatable(False)
        self._toolbar.setIconSize(TOOLBAR_ICON_SIZE)
        self._toolbar.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        self._toolbar.setMinimumWidth(0)
        self._toolbar.setFixedHeight(self._toolbar.sizeHint().height())
        self._remove_toolbar_actions(TOOLBAR_ACTIONS_TO_REMOVE)

        for button in self._toolbar.findChildren(QToolButton):
            button.setAutoRaise(True)
            button.setObjectName("chartToolButton")

    def _build_top_row(self) -> QWidget:
        """Create a fixed-height top control row."""
        top_row = QWidget(self)
        top_row.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )

        layout = QHBoxLayout(top_row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._toolbar, 0)
        layout.addStretch(1)

        self._zoom_label = QLabel(f"{self._zoom_percent}%", top_row)
        self._zoom_label.setObjectName("chartZoomLabel")

        self._zoom_fit_button = create_toolbar_button(self, "zoom_fit", self._zoom_best_fit, layout)

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal, top_row)
        self._zoom_slider.setObjectName("chartZoomSlider")
        self._zoom_slider.setMinimum(self._min_zoom_percent)
        self._zoom_slider.setMaximum(self._max_zoom_percent)
        self._zoom_slider.setValue(self._zoom_percent)
        self._zoom_slider.setSingleStep(5)
        self._zoom_slider.setPageStep(25)
        self._zoom_slider.setMinimumWidth(60)
        self._zoom_slider.setMaximumWidth(140)
        self._zoom_slider.setTracking(False)
        self._zoom_slider.valueChanged.connect(self.set_zoom_percent)

   
        layout.addWidget(self._zoom_slider, 0)
        layout.addWidget(self._zoom_label, 0)

        return top_row

    def _zoom_best_fit(self) -> None:
        """PowerPoint-style fit to window.

        Fill the available viewport width. Vertical scrollbars are acceptable.
        """
        if self._resize_mode != "FIXED":
            return

        viewport = self._fixed_scroll_area.viewport().size()
        if viewport.width() <= 0:
            return

        base_size = self._current_figure_pixel_size(apply_zoom=False)
        if base_size.width() <= 0:
            return

        available_width = max(1, viewport.width())

        zoom_percent = int(
            round(
                available_width
                * 100.0
                / float(base_size.width())
            )
        )

        self.set_zoom_percent(
            self._clamp_zoom(zoom_percent)
        )

    def _remove_toolbar_actions(self, action_names: set[str]) -> None:
        """Remove selected built-in Matplotlib toolbar actions by label."""
        for action in list(self._toolbar.actions()):
            text = (action.text() or "").strip()
            if text in action_names:
                self._toolbar.removeAction(action)
                action.deleteLater()

    def _build_actions_menu(self) -> QMenu:
        """Create the chart actions menu shared by button and canvas.

        Fit / Fit proportional / Fixed used to live here as three checkable
        entries kept in sync by comparing an action's (translated) label
        against the resize mode - which meant the checkmarks stopped
        matching anything the moment the interface language changed. The
        choice is made from the Figure Properties panel now, as a combo
        box next to the figure's other display settings; see
        FigurePropertiesWidget.set_resize_mode_control.
        """
        menu = create_menu(
            self,
            [
                MenuItem(_("Reload"), _("Reload chart"), None, self.reload,False,"reload"),
                MenuItem(_("Copy"),_("Copy figure"),PySide6.QtGui.QKeySequence.StandardKey.Copy,self.copy_chart_to_clipboard,False,"copy"),
                MenuItem(_("Save"),_("Save as picture"), PySide6.QtGui.QKeySequence.StandardKey.SaveAs,self.save_chart_as,False,"save"),
                None,
                MenuItem(_("Delete"),_("Delete this figure"),PySide6.QtGui.QKeySequence.StandardKey.Delete,self.delete_chart,False,"delete"),
            ],
        )
        return menu

    def _show_context_menu(self, pos: QPoint) -> None:
        """Show the cached actions menu."""
        self._actions_menu.exec(self._canvas.mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Resize / zoom / background helpers
    # ------------------------------------------------------------------
    def minimumSizeHint(self) -> QSize:
        """Keep parent layouts from using the chart content as minimum size."""
        return QSize(80, 60)

    def sizeHint(self) -> QSize:
        """Return a moderate preferred size, independent from canvas content."""
        return QSize(800, 450)

    def showEvent(self, event: PySide6.QtGui.QShowEvent) -> None:
        """Refresh geometry when a hidden chart tab becomes visible."""
        super().showEvent(event)
        self._schedule_canvas_geometry_sync(redraw=True, late=True)

    def event(self, event: QEvent) -> bool:
        """Catch tab/page layout events that are not resize events."""
        handled = super().event(event)
        try:
            event_type = event.type()
        except Exception:
            return handled
        if event_type in (QEvent.Type.Show, QEvent.Type.LayoutRequest, QEvent.Type.PolishRequest):
            self._schedule_canvas_geometry_sync(redraw=True, late=True)
        return handled

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Re-sync the canvas geometry after resize events.

        Paint events are deliberately *not* handled here.

        They used to be: the viewport's Paint was intercepted, the background
        filled by hand, and True returned.  Two things were wrong with that.
        It painted a widget from outside its own paintEvent, and returning True
        cancelled the paint Qt was about to perform - so the viewport was never
        marked clean, and the compositor was free to leave stale content
        anywhere in the window.  Tiles of unrelated widgets repeating down the
        window are what that looks like on macOS.

        Nothing had to replace it.  ``_apply_background_color_to_widgets``
        already gives the viewport ``autoFillBackground`` and the chart
        background colour, so Qt erases it on every paint through the normal
        path and the canvas draws on top.  The filter was suppressing the very
        paint that would have cleared the area it was trying to clear.
        """
        try:
            event_type = event.type()
        except Exception:
            return False

        if event_type != QEvent.Type.Resize:
            return super().eventFilter(watched, event)

        if watched is self._canvas:
            self._schedule_canvas_geometry_sync(redraw=True)
            return super().eventFilter(watched, event)

        if watched is self._fixed_scroll_area.viewport():
            self._schedule_canvas_geometry_sync(redraw=True)

        return super().eventFilter(watched, event)

    def _fill_background(self, widget: QWidget) -> None:
        """Fill a widget's whole rect with the configured background colour."""
        color = PySide6.QtGui.QColor(self._background_color)
        if not color.isValid():
            color = PySide6.QtGui.QColor("#ffffff")

        painter = PySide6.QtGui.QPainter(widget)
        try:
            painter.fillRect(widget.rect(), color)
        finally:
            painter.end()

    def paintEvent(self, event: PySide6.QtGui.QPaintEvent) -> None:
        """Paint the panel background before the frame and children.

        Why this is not left to the stylesheet: in FIT_PROPORTIONAL the canvas
        is centred and smaller than the panel, so a strip of the panel is left
        uncovered.  With an opaque-painting child on top, macOS does not erase
        that strip and it keeps the previous frame's pixels.  Filling it
        explicitly costs one rect fill and removes the dependency on
        platform-specific erase behaviour.
        """
        self._fill_background(self)
        super().paintEvent(event)

    def resizeEvent(self, event: PySide6.QtGui.QResizeEvent) -> None:
        """Update canvas geometry when ChartPanel is resized."""
        super().resizeEvent(event)
        # The strip the canvas vacates belongs to the panel; ask for a repaint
        # explicitly rather than trusting the platform to invalidate the newly
        # exposed area (see paintEvent).
        self.update()
        self._schedule_canvas_geometry_sync(redraw=True)


    def set_resize_mode(
        self,
        resize_mode: str,
        *,
        persist: bool = True,
        redraw: bool = True,
    ) -> None:
        """Set resize behaviour and persist it for this figure by default."""
        self._resize_mode = self._normalized_resize_mode(resize_mode)
        is_fixed = self._resize_mode == "FIXED"

        self._zoom_fit_button.setVisible(is_fixed)
        self._zoom_slider.setVisible(is_fixed)
        self._zoom_label.setVisible(is_fixed)

        if is_fixed:
            self._move_canvas_to_fixed_page()
            self._apply_fixed_mode_pixel_size()
        else:
            self._move_canvas_to_direct_layout(
                centered=self._resize_mode == "FIT_PROPORTIONAL"
            )
            self._restore_unzoomed_figure_metrics()

        if persist:
            self._persist_chart_panel_config()
            self._persist_view_state()
        if redraw:
            self._schedule_canvas_geometry_sync(redraw=True)

    def _remove_active_chart_widget_from_layout(self) -> None:
        """Detach chart display widgets from the main layout.

        Qt layouts tolerate removeWidget() for widgets that are not currently
        managed by the layout.  Removing both possible chart widgets makes mode
        switching deterministic and prevents a hidden scroll area from remaining
        as a layout item.
        """
        self._main_layout.removeWidget(self._canvas)
        self._main_layout.removeWidget(self._fixed_scroll_area)

    def _move_canvas_to_direct_layout(self, *, centered: bool) -> None:
        """Put the canvas directly in the ChartPanel layout for FIT modes.

        FIT and FIT_PROPORTIONAL intentionally share the same parent/layout
        path.  FIT lets the layout expand the canvas.  FIT_PROPORTIONAL keeps
        the same layout path, but gives the canvas a fixed proportional size
        and lets the layout center it in the chart client area.
        """
        self._remove_active_chart_widget_from_layout()

        if self._canvas.parent() is self._fixed_scroll_area.viewport():
            self._fixed_scroll_area.takeWidget()

        self._fixed_scroll_area.setVisible(False)
        self._fixed_scroll_area.lower()
        self._canvas.setParent(self._chart_area)
        self._canvas.show()
        self._canvas.raise_()

        if centered:
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            self._main_layout.addWidget(
                self._canvas,
                1,
                Qt.AlignmentFlag.AlignCenter,
            )
        else:
            # Clear any fixed-size constraint left by FIT_PROPORTIONAL/FIXED.
            self._canvas.setMinimumSize(0, 0)
            self._canvas.setMaximumSize(QSize(16777215, 16777215))
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self._main_layout.addWidget(self._canvas, 1)

        self._active_chart_widget = self._canvas

    def _move_canvas_to_fixed_page(self) -> None:
        """Put the canvas into the scroll area for FIXED mode."""
        self._remove_active_chart_widget_from_layout()

        if self._canvas.parent() is not self._fixed_scroll_area.viewport():
            self._fixed_scroll_area.takeWidget()
            self._fixed_scroll_area.setWidget(self._canvas)

        self._main_layout.addWidget(self._fixed_scroll_area, 1)
        self._fixed_scroll_area.setVisible(True)
        self._fixed_scroll_area.raise_()
        self._active_chart_widget = self._fixed_scroll_area

    def set_zoom_percent(self, percent: int) -> None:
        """Set FIXED-mode zoom and persist it to config.json."""
        new_value = self._clamp_zoom(percent)
        if new_value == self._zoom_percent:
            return

        self._zoom_percent = new_value
        self._zoom_label.setText(f"{self._zoom_percent}%")

        if self._zoom_slider.value() != self._zoom_percent:
            self._zoom_slider.blockSignals(True)
            self._zoom_slider.setValue(self._zoom_percent)
            self._zoom_slider.blockSignals(False)

        if self._resize_mode == "FIXED":
            self._apply_fixed_mode_pixel_size()
            self._schedule_canvas_geometry_sync(redraw=True)

        self._persist_chart_panel_config()
        self._persist_view_state()

    def background_color(self) -> str:
        """Return the background colour used around the chart canvas."""
        return self._background_color

    def set_background_color(self, color: str, *, persist: bool = True) -> None:
        """Set the panel, viewport, and figure background colour."""
        qcolor = PySide6.QtGui.QColor(str(color))
        if not qcolor.isValid():
            applogger.warning("Invalid chart panel background color: %s", color)
            return

        self._background_color = qcolor.name()
        self._apply_background_color_to_widgets()

        if persist:
            self._persist_chart_panel_config()
            self._persist_view_state()

    def _apply_background_color_to_widgets(self) -> None:
        """Apply the panel background without changing the Matplotlib figure.

        The background belongs to the Qt client area around the canvas.  It must
        be visible in FIT_PROPORTIONAL margins and in the FIXED scroll viewport,
        but it must not alter Matplotlib figure or axes face colors.

        ``autoFillBackground`` here is what erases the scroll viewport before
        each paint, which matters because the canvas sets WA_OpaquePaintEvent
        and so paints only its own rect.  Do not add a Paint handler for the
        viewport on top of this: one existed, returned True, and cancelled this
        erase - see eventFilter.
        """
        qcolor = PySide6.QtGui.QColor(self._background_color) 
        if not qcolor.isValid():
            qcolor = PySide6.QtGui.QColor("#ffffff")
            self._background_color = qcolor.name()
            applogger.warning("Invalid background color")

        color_name = qcolor.name()

        for widget in self._background_painted_widgets():
            widget.setAutoFillBackground(True)
            widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            palette = widget.palette()
            palette.setColor(PySide6.QtGui.QPalette.ColorRole.Window, qcolor)
            palette.setColor(PySide6.QtGui.QPalette.ColorRole.Base, qcolor)
            widget.setPalette(palette)

        # Style sheets make the background deterministic even when the app-wide
        # QSS disables palette painting for QFrame/QScrollArea descendants.
        self.setStyleSheet(f"QFrame#ChartPanel {{ background-color: {color_name}; }}")
        self._fixed_scroll_area.setStyleSheet("QScrollArea#chartScrollArea { "
            f"background-color: {color_name}; border: none; "
            "}"
        )
        self._fixed_scroll_area.viewport().setStyleSheet(f"background-color: {color_name};")


    def _background_painted_widgets(self) -> tuple[QWidget, ...]:
        """Return widgets that should show the chosen background colour."""
        return (
            self,
            self._chart_area,
            self._notes_view,
            self._fixed_scroll_area,
            self._fixed_scroll_area.viewport(),
        )

    def _schedule_canvas_geometry_sync(self, *, redraw: bool, late: bool = False) -> None:
        """Coalesce resize-driven canvas updates using owned timers.

        The old gwXG sizing path is kept intact.  The only change here is that
        the callback is owned by the widget, so it cannot fire after close, and
        a late pass is available for tab activation/final layout geometry.
        """
        self._pending_canvas_redraw = self._pending_canvas_redraw or bool(redraw)

        if not self._pending_canvas_sync:
            self._pending_canvas_sync = True
            self._canvas_sync_timer.start(0)

        if late:
            self._canvas_late_sync_timer.start(80)

    def _run_pending_canvas_geometry_sync(self) -> None:
        """Run the pending geometry sync with the merged redraw flag."""
        if self._deleted:
            return
        redraw = self._pending_canvas_redraw
        self._pending_canvas_redraw = False
        self._sync_canvas_geometry(redraw=redraw)

    def _sync_canvas_geometry(self, *, redraw: bool) -> None:
        """Synchronize canvas and Matplotlib figure geometry.

        FIT is pure Qt layout management.  The canvas receives its size from
        the ChartPanel layout and this method only redraws when that size
        changes.

        FIT_PROPORTIONAL stays as close as possible to FIT: the canvas is still
        a direct widget in the ChartPanel layout.  The only difference from FIT
        is that the canvas width and height are fixed to the largest size that
        preserves the natural figure aspect ratio inside the same chart client
        area used by FIT.

        FIXED moves the canvas into _fixed_scroll_area and gives it the zoomed,
        fixed figure pixel size.
        """
        self._pending_canvas_sync = False

        if self._resize_mode == "FIXED":
            target_size = self._current_figure_pixel_size(apply_zoom=True)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            if self._canvas.size() != target_size:
                self._canvas.setFixedSize(target_size)
                self._fixed_scroll_area.viewport().update()
            self._apply_figure_size_from_canvas(target_size)

        elif self._resize_mode == "FIT_PROPORTIONAL":
            area_size = self._direct_chart_client_size()
            if area_size.width() <= 0 or area_size.height() <= 0:
                return

            target_size = self._proportional_size_for_viewport(area_size)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            if self._canvas.size() != target_size:
                self._canvas.setFixedSize(target_size)
                self._main_layout.invalidate()

            self._apply_figure_size_from_canvas(target_size)

        else:
            # FIT: the layout owns the widget geometry.  Do not call
            # setFixedSize(), resize(), or updateGeometry() here, otherwise Qt
            # and the manual synchronizer can enter a resize feedback loop.
            # Still update Matplotlib's logical figure size from the canvas that
            # Qt has assigned.  Without this, tab switches and dialog closes can
            # leave the drawing using the previous panel's inches/DPI even
            # though the QWidget has a new pixel size.
            target_size = QSize(self._canvas.size())
            if target_size.width() <= 0 or target_size.height() <= 0:
                return
            self._apply_figure_size_from_canvas(target_size)

        if redraw or target_size != self._last_canvas_size:
            self._last_canvas_size = QSize(target_size)
            self._canvas.draw_idle()

    def _direct_chart_client_size(self) -> QSize:
        """Return the chart area available below the toolbar row."""
        contents = self._chart_area.contentsRect()
        top_row_height = self._top_row.height() if self._top_row.isVisible() else 0
        layout_spacing = self._main_layout.spacing() if top_row_height > 0 else 0
        width = max(0, contents.width())
        height = max(0, contents.height() - top_row_height - layout_spacing)
        return QSize(width, height)

    def _device_pixel_ratio(self) -> float:
        """Return the canvas's pixel ratio, never zero.

        ``getattr``: the constructor touches the figure before it builds the
        canvas, and a missing canvas means "no scaling known yet", not a crash.
        """
        return canvas_pixel_ratio(getattr(self, "_canvas", None))

    def _apply_figure_size_from_canvas(self, canvas_size: QSize) -> None:
        """Make Matplotlib's figure size match the canvas.

        ``canvas_size`` is in logical pixels; ``figure.dpi`` counts device
        pixels, because ``FigureCanvasQT._update_pixel_ratio`` multiplied it by
        the display's ratio.  The two have to be reconciled, and this is the
        conversion Matplotlib itself performs in ``FigureCanvasQT.resizeEvent``:

            w = event.size().width() * self.device_pixel_ratio
            winch = w / self.figure.dpi

        Dropping the ratio - which this method used to do - makes the figure
        half the size of the widget on a 2x screen.  Agg then paints a quarter
        of the widget's area and the remainder keeps whatever the backing store
        held, which is the field of artefacts reported on macOS and never on
        Windows, where the ratio is 1 and the mistake cancels out.
        """
        if canvas_size.width() <= 0 or canvas_size.height() <= 0:
            return

        if self._resize_mode == "FIXED":
            width_in, height_in = self._fixed_figure_size_inches
            self._figure.set_size_inches(width_in, height_in, forward=False)
            return

        width_in = logical_to_inches(canvas_size.width(), self._figure, self._canvas)
        height_in = logical_to_inches(canvas_size.height(), self._figure, self._canvas)
        self._figure.set_size_inches(width_in, height_in, forward=False)

    def _proportional_size_for_viewport(self, viewport_size: QSize) -> QSize:
        """Return largest size inside viewport preserving current aspect ratio."""
        viewport_width = max(1, viewport_size.width())
        viewport_height = max(1, viewport_size.height())
        ratio = self._current_figure_aspect_ratio()

        width_from_height = int(round(float(viewport_height) * ratio))
        if width_from_height <= viewport_width:
            return QSize(max(1, width_from_height), viewport_height)

        height_from_width = int(round(float(viewport_width) / ratio))
        return QSize(viewport_width, max(1, height_from_width))

    def _clamp_zoom(self, percent: int) -> int:
        return max(self._min_zoom_percent, min(self._max_zoom_percent, int(percent)))

    def _current_figure_aspect_ratio(self) -> float:
        """Return the unzoomed rendered figure aspect ratio.

        FIT_PROPORTIONAL must preserve the chart's natural rendered aspect, not
        the current canvas aspect.  In FIT mode the Matplotlib figure is resized
        to the available client rectangle, so using figure.get_size_inches()
        here would make proportional mode inherit the last FIT rectangle and
        look identical to FIT.  The captured fixed metrics are the stable source
        of truth for proportional sizing.
        """
        try:
            width_in, height_in = self._fixed_figure_size_inches
            width_value = float(width_in)
            height_value = float(height_in)
        except Exception:
            return DEFAULT_ASPECT_RATIO

        if width_value <= 0.0 or height_value <= 0.0:
            return DEFAULT_ASPECT_RATIO

        return width_value / height_value

    # ------------------------------------------------------------------
    # Descriptor and pre-render geometry normalization
    # ------------------------------------------------------------------
    def _descriptor_prepared_for_render(
        self,
        descriptor: Any,
        *,
        axis_count_override: int | None = None,
    ) -> Any:
        """Return a render descriptor with oversized/inconsistent grids fixed.

        Some saved descriptors can keep an old subplot grid after axes are
        added or removed.  Matplotlib may then create extra slots or compute the
        first layout against stale geometry.  Normalize the descriptor before it
        reaches the renderer so the grid capacity is the smallest one that can
        hold the descriptor's axes.
        """
        axis_count = (
            int(axis_count_override)
            if axis_count_override is not None
            else self._descriptor_axis_count(descriptor)
        )
        if axis_count <= 0:
            return descriptor

        prepared = deepcopy(descriptor)
        changed = self._normalize_descriptor_grids(prepared, axis_count)
        if changed:
            applogger.debug(
                "Normalized chart grid before render (figure_id=%s, axes=%s)",
                self._figure_id,
                axis_count,
            )
        return prepared

    def _descriptor_axis_count(self, value: Any) -> int:
        """Best-effort axis count from a descriptor-like nested structure.

        Important: this must count descriptor axes, not rendered Matplotlib axes.
        A stale 2x1 grid can make the renderer construct two Matplotlib Axes even
        when the descriptor contains only one real chart axis. If that rendered
        count is fed back into grid normalization, the stale 2x1 grid becomes
        self-confirming and never shrinks after a cancelled preview.
        """
        if isinstance(value, MutableMapping):
            for key in AXIS_COLLECTION_KEYS:
                collection = value.get(key)
                if collection is not None and self._is_axis_collection(collection):
                    return len(collection)
            for child in value.values():
                count = self._descriptor_axis_count(child)
                if count > 0:
                    return count
        elif isinstance(value, list):
            # Only recurse into generic lists. Do not treat every Sequence as an
            # axis collection; strings, tuples of coordinates, and other list-
            # like values can appear in chart descriptors but are not axes.
            for child in value:
                count = self._descriptor_axis_count(child)
                if count > 0:
                    return count
        return 0

    @staticmethod
    def _is_axis_collection(value: Any) -> bool:
        """Return True for descriptor collections that represent chart axes.

        The old implementation returned True for any non-string Sequence. That
        was too broad: it could count arbitrary lists as axes and, together with
        rendered ``figure.axes`` counts, preserve stale grid sizes. Real axis
        collections in the descriptor are lists of mappings.
        """
        if not isinstance(value, list):
            return False
        if not value:
            return True
        return all(isinstance(item, MutableMapping) for item in value)

    def _normalize_descriptor_grids(self, value: Any, axis_count: int) -> bool:
        """Normalize every row/column grid pair found in a descriptor tree."""
        changed = False
        if isinstance(value, MutableMapping):
            changed = self._normalize_grid_mapping(value, axis_count) or changed
            for child in value.values():
                changed = self._normalize_descriptor_grids(child, axis_count) or changed
        elif isinstance(value, list):
            for child in value:
                changed = self._normalize_descriptor_grids(child, axis_count) or changed
        return changed

    def _normalize_grid_mapping(self, mapping: MutableMapping[Any, Any], axis_count: int) -> bool:
        """Normalize one mapping containing row and column grid keys."""
        row_key = self._first_present_key(mapping, GRID_ROW_KEYS)
        col_key = self._first_present_key(mapping, GRID_COL_KEYS)
        if row_key is None or col_key is None:
            return False

        rows = self._positive_int(mapping.get(row_key))
        cols = self._positive_int(mapping.get(col_key))
        if rows is None or cols is None:
            return False

        if rows * cols == axis_count:
            return False

        new_rows, new_cols = self._minimum_grid_for_axis_count(
            axis_count=axis_count,
            old_rows=rows,
            old_cols=cols,
        )
        mapping[row_key] = new_rows
        mapping[col_key] = new_cols
        return True

    @staticmethod
    def _first_present_key(mapping: MutableMapping[Any, Any], candidates: tuple[str, ...]) -> Any | None:
        """Return the concrete key matching one of the candidate names."""
        normalized = {str(key).strip().lower(): key for key in mapping.keys()}
        for candidate in candidates:
            key = normalized.get(candidate)
            if key is not None:
                return key
        return None

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        """Parse a positive integer or return None."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _minimum_grid_for_axis_count(*, axis_count: int, old_rows: int, old_cols: int) -> tuple[int, int]:
        """Return the minimum-cell grid that best preserves the old shape."""
        if axis_count <= 0:
            return 1, 1

        old_ratio = float(old_cols) / float(old_rows) if old_rows > 0 else 1.0
        best_rows = 1
        best_cols = axis_count
        best_score: tuple[int, float, int] | None = None

        for rows in range(1, axis_count + 1):
            cols = int(math.ceil(axis_count / rows))
            cells = rows * cols
            ratio = float(cols) / float(rows)
            score = (cells, abs(ratio - old_ratio), abs(cols - old_cols) + abs(rows - old_rows))
            if best_score is None or score < best_score:
                best_score = score
                best_rows = rows
                best_cols = cols

        return best_rows, best_cols

    def _prepare_canvas_geometry_before_render(self) -> None:
        """Synchronize Qt and Matplotlib sizes before rendering.

        ``reload()`` may be called immediately after a modal dialog closes, while
        Qt still has stale splitter/tab geometry cached.  For FIT modes the
        Matplotlib figure size is derived from the *current* chart client area,
        so force the owning window layouts to settle, read the visible size, and
        apply that size before the descriptor is rendered.
        """
        self._activate_current_layouts()

        if self._resize_mode == "FIXED":
            self._apply_fixed_mode_pixel_size()
            self._apply_figure_size_from_canvas(self._current_figure_pixel_size(apply_zoom=True))
            return

        area_size = self._current_fit_client_size()
        if area_size.width() <= 0 or area_size.height() <= 0:
            return

        if self._resize_mode == "FIT_PROPORTIONAL":
            target_size = self._proportional_size_for_viewport(area_size)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
            self._canvas.setFixedSize(target_size)
            self._main_layout.invalidate()
        else:
            target_size = area_size
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self._canvas.setMinimumSize(0, 0)
            self._canvas.setMaximumSize(16777215, 16777215)
            self._canvas.resize(target_size)

        self._apply_figure_size_from_canvas(target_size)
        self._last_canvas_size = QSize(target_size)

    def _current_fit_client_size(self) -> QSize:
        """Return the best current canvas area for FIT-style reloads.

        The direct chart client rectangle is authoritative once layouts are
        settled.  If Qt still reports an empty size, fall back to the visible
        window dimensions so reload can still render a sensible fitted figure.
        """
        area_size = self._direct_chart_client_size()
        if area_size.width() > 0 and area_size.height() > 0:
            return area_size
        return self._fallback_chart_client_size_from_window()

    def _activate_current_layouts(self) -> None:
        """Ask Qt layouts to resolve the current window/widget geometry now."""
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

        for widget in (self.window(), self, self._content_split, self._chart_area):
            if isinstance(widget, QWidget):
                widget.updateGeometry()
            layout = widget.layout() if isinstance(widget, QWidget) else None
            if layout is not None:
                layout.activate()

        self._fixed_scroll_area.updateGeometry()
        self._fixed_scroll_area.viewport().updateGeometry()

    def _fallback_chart_client_size_from_window(self) -> QSize:
        """Estimate chart client size from the visible window when layout is stale."""
        window = self.window()
        width = max(self._chart_area.width(), self.width())
        height = max(self._chart_area.height(), self.height())

        if isinstance(window, QWidget):
            window_size = window.size()
            width = max(width, window_size.width())
            height = max(height, window_size.height())

        top_row_height = self._top_row.sizeHint().height() if self._top_row.isVisible() else 0
        notes_height = self._notes_view.height() if self._notes_view.isVisible() else 0
        height = max(0, height - top_row_height - notes_height)
        return QSize(max(0, width), max(0, height))

    # ------------------------------------------------------------------
    # Config and figure metrics
    # ------------------------------------------------------------------
    def _chart_config(self) -> dict[str, Any]:
        """Return chart_panel config as a real dict."""
        chart_config_obj = self._config.setdefault(CONFIG_SECTION, {})
        if isinstance(chart_config_obj, dict):
            return chart_config_obj
        chart_config: dict[str, Any] = {}
        self._config[CONFIG_SECTION] = chart_config
        return chart_config

    def _refresh_config_from_source(self) -> None:
        """Reload config through the app.utils.config API."""
        self._config = load_config()

    def _apply_persisted_figure_metrics_to_rcparams(self) -> None:
        """Apply this figure's own width, height and DPI to rcParams.

        Read from the figure descriptor rather than from config.json: the
        metrics belong to the figure, so every panel has to set them before it
        renders instead of inheriting whatever the previously-opened figure
        left in this process-wide rcParams.

        A figure with no metrics of its own leaves rcParams untouched, which
        keeps figures saved before this change rendering exactly as they did.
        """
        try:
            descriptor = self._repo.load_figure_descriptor(self._figure_id)
            options = getattr(descriptor, "options", None) if descriptor else None
            metrics = figure_metrics_from_options(
                options if isinstance(options, dict) else None
            )
            if metrics is None:
                return

            width_cm, height_cm, dpi = metrics
            width_in = width_cm / CM_PER_INCH
            height_in = height_cm / CM_PER_INCH
            rcParams["figure.figsize"] = [width_in, height_in]
            rcParams["figure.dpi"] = dpi

            # And the FIXED baseline, which is the same statement: this is the
            # size the figure is configured to be.
            #
            # Without this the baseline was only ever *captured from the last
            # render* - and in FIXED mode what gets rendered is the baseline
            # itself, so it fed on its own output. Whatever the first render
            # produced became permanent, and the width and height typed into
            # the figure properties were applied to rcParams, applied to the
            # figure, and then overwritten by the stale baseline before the
            # descriptor was drawn. Which is exactly how they "were ignored".
            self._fixed_figure_size_inches = (width_in, height_in)
            self._fixed_figure_dpi = float(dpi)
        except Exception:
            applogger.exception(
                "Failed to apply figure metrics (figure_id=%s)", self._figure_id
            )

    def _reset_figure_metrics_from_rcparams_for_reload(self) -> None:
        """Reset logical figure size and base DPI from current rcParams."""
        try:
            figsize = rcParams.get("figure.figsize", None)
            dpi = rcParams.get("figure.dpi", None)

            if figsize is not None and len(figsize) >= 2:
                width_in = float(figsize[0])
                height_in = float(figsize[1])
                if width_in > 0.0 and height_in > 0.0:
                    self._figure.set_size_inches(width_in, height_in, forward=False)

            if dpi is not None:
                dpi_value = float(dpi)
                if dpi_value > 0.0:
                    self._set_configured_dpi(dpi_value)
        except Exception:
            applogger.exception(
                "Failed to reset figure metrics from rcParams (figure_id=%s)",
                self._figure_id,
            )

    def _set_configured_dpi(self, dpi: float) -> None:
        """Set the figure's dpi from a value the user configured.

        ``figure.dpi`` is expected to be in device pixels everywhere else in
        this class, because that is the invariant the Qt backend maintains:
        ``_set_device_pixel_ratio`` multiplies the dpi by the display's ratio
        and only revisits it when the ratio itself changes.

        rcParams and config.json hold the dpi the user asked for, with no
        ratio in it.  Writing that straight onto the figure - which this used
        to do on every reload - silently breaks the invariant for the rest of
        the session, and the figure comes out at the wrong size against the
        canvas again.  Reloading a chart was enough to bring the artefacts back
        on a Retina screen even after the resize path was fixed.

        ``_canvas`` may not exist yet: the constructor prepares the figure from
        rcParams before building the canvas around it.  That is not a problem
        to work around, because ``FigureCanvasQT.__init__`` applies the ratio to
        whatever dpi it finds - so the figure ends up correctly scaled either
        way, and the only thing needed here is not to raise.
        """
        apply_configured_dpi(self._figure, getattr(self, "_canvas", None), dpi)

    def _capture_fixed_metrics_from_rendered_figure(self) -> None:
        """Capture rendered figure metrics for FIXED mode."""
        try:
            width_in, height_in = self._figure.get_size_inches()

            if float(width_in) > 0.0 and float(height_in) > 0.0:
                self._fixed_figure_size_inches = (float(width_in), float(height_in))

            # The dpi is *not* read back from the figure.
            #
            # Two things have been done to it by the time this runs: the Qt
            # backend multiplied it by the display's pixel ratio, and FIXED
            # mode multiplied it again by the zoom.  Capturing that as the base
            # dpi folds the zoom into the baseline, so the next zoom multiplies
            # an already-zoomed value and the chart grows without bound - which
            # is exactly how FIXED zoom stopped being correct.
            #
            # rcParams holds the dpi the user configured, with neither factor
            # in it.  That is the only stable baseline.
            self._fixed_figure_dpi = float(rcParams.get("figure.dpi", 100.0) or 100.0)
        except Exception:
            applogger.exception(
                "Failed to capture rendered figure metrics (figure_id=%s)",
                self._figure_id,
            )

    def _current_figure_pixel_size(self, *, apply_zoom: bool) -> QSize:
        """Return the FIXED-mode canvas size in *logical* pixels.

        The result goes to ``QWidget.setFixedSize``, which speaks logical
        pixels.  Sized from ``FIXED_MODE_SCREEN_DPI``, a fixed on-screen
        reference, rather than from the figure's own configured dpi - dpi is
        a print/export resolution and must not also change how big the
        figure looks on screen.  A 16 inch figure occupies 1600 logical
        pixels at 100% zoom on every display and at every configured dpi;
        only the number of device pixels behind them differs, and that is
        the backend's business rather than this method's.
        """
        try:
            width_in, height_in = self._fixed_figure_size_inches
        except Exception:
            width_in, height_in = 16.0, 9.0

        zoom = self._zoom_percent / 100.0 if apply_zoom else 1.0
        return QSize(
            int(round(inches_to_logical(width_in, FIXED_MODE_SCREEN_DPI, zoom))),
            int(round(inches_to_logical(height_in, FIXED_MODE_SCREEN_DPI, zoom))),
        )

    def _restore_unzoomed_figure_metrics(self) -> None:
        """Restore captured logical size and base DPI after leaving FIXED mode."""
        try:
            width_in, height_in = self._fixed_figure_size_inches
            # Configured dpi in, device dpi onto the figure.
            self._set_configured_dpi(self._fixed_figure_dpi)
            self._figure.set_size_inches(width_in, height_in, forward=False)
        except Exception:
            applogger.exception(
                "Failed to restore unzoomed figure metrics (figure_id=%s)",
                self._figure_id,
            )

    def _apply_fixed_mode_pixel_size(self) -> None:
        """Size the canvas for FIXED mode at the current zoom.

        Zoom is applied to ``FIXED_MODE_SCREEN_DPI``, the same on-screen
        reference ``_current_figure_pixel_size`` sizes the widget from - not
        to the figure's own configured dpi, which must stay a pure
        print/export setting.  The figure is rendered at exactly the dpi the
        widget's logical size needs (times the display's own pixel ratio, via
        ``_set_configured_dpi``); using the configured dpi here instead was
        what made the on-screen figure grow or shrink with it.  Width, height
        and the physical print/export size stay in inches throughout - only
        the number of on-screen pixels changes with zoom.

        Neither factor is stored: ``zoom`` from the slider, the display's
        ratio by ``_set_configured_dpi``.  Storing either one is what made
        zoom compound.
        """
        if self._resize_mode != "FIXED":
            return

        try:
            zoom = self._zoom_percent / 100.0
            width_in, height_in = self._fixed_figure_size_inches
            self._set_configured_dpi(FIXED_MODE_SCREEN_DPI * zoom)
            self._figure.set_size_inches(width_in, height_in, forward=False)
            self._canvas.setFixedSize(self._current_figure_pixel_size(apply_zoom=True))
            self._figure.set_size_inches(width_in, height_in, forward=False)
        except Exception:
            applogger.exception(
                "Failed to apply fixed-mode pixel size (figure_id=%s)",
                self._figure_id,
            )

    # ------------------------------------------------------------------
    # Local actions
    # ------------------------------------------------------------------
    def _canvas_to_qimage(self) -> PySide6.QtGui.QImage | None:
        """Render the current Matplotlib figure into a detached QImage."""
        self._canvas.draw()
        buffer = BytesIO()
        copy_dpi:float = self._config.get(CONFIG_COPY_DPI,600.0)
        
        try:
            self._figure.savefig(
                buffer,
                format="png",
                dpi=copy_dpi,
                bbox_inches=None,
                facecolor=self._figure.get_facecolor(),
                edgecolor=self._figure.get_edgecolor(),
            )
            data = buffer.getvalue()
        finally:
            buffer.close()

        image = PySide6.QtGui.QImage.fromData(data)
        if image.isNull():
            applogger.error("Failed to build QImage from in-memory PNG data.")
            return None
        return image

    def copy_chart_to_clipboard(self) -> None:
        """Copy the current chart image to the system clipboard."""
        try:
            image = self._canvas_to_qimage()
            if image is None:
                return
            PySide6.QtGui.QGuiApplication.clipboard().setImage(image)
            applogger.info("Chart copied to clipboard (figure_id=%s)", self._figure_id)
        except Exception:
            applogger.exception(
                "Failed to copy chart to clipboard (figure_id=%s)",
                self._figure_id,
            )

    def save_chart_as(self) -> None:
        """Save the chart to an image or vector format via a file dialog."""
        # The configured default decides the suggested extension and which
        # filter the dialog opens on.  Both were hard-coded to PNG while
        # config.json carried a ``save_format`` that nothing read, so the
        # preference existed and did nothing.
        from app.dialogs.settings_dialog import (
            CONFIG_SAVE_FORMAT,
            SAVE_FORMAT_FILTERS,
            normalized_save_format,
        )

        preferred = normalized_save_format(get_value(CONFIG_SAVE_FORMAT))
        extension, preferred_filter = SAVE_FORMAT_FILTERS[preferred]
        default_name = f"chart_{self._figure_id}.{extension}"

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            _("Save chart as"),
            default_name,
            ";;".join(label for _unused, label in SAVE_FORMAT_FILTERS.values()),
            preferred_filter,
        )
        if not file_path:
            return

        try:
            # Not self._figure.dpi: in FIXED mode that now holds the on-screen
            # rendering dpi (FIXED_MODE_SCREEN_DPI times zoom), which has
            # nothing to do with export quality.  _fixed_figure_dpi is always
            # the dpi actually configured for this figure, screen state aside.
            self._figure.savefig(
                file_path, dpi=self._fixed_figure_dpi, bbox_inches="tight"
            )
            applogger.info(
                "Chart saved (figure_id=%s, path=%s, filter=%s)",
                self._figure_id,
                file_path,
                selected_filter,
            )
        except Exception:
            applogger.exception("Failed to save chart (figure_id=%s)", self._figure_id)
            show_message(self, "chart.save_failed")

    def delete_chart(self) -> None:
        """Ask for confirmation and close/delete the current chart panel."""
        if ask(self, "chart.confirm_delete"):
            self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    @property
    def figure_id(self) -> int:
        """Return the repository figure id rendered by this panel."""
        return self._figure_id

    @property
    def figure(self) -> Figure:
        """Return the Matplotlib figure object."""
        return self._figure

    @property
    def canvas(self) -> FigureCanvasQTAgg:
        """Return the Matplotlib canvas widget."""
        return self._canvas

    @property
    def resize_mode(self) -> str:
        """Return this panel's current fit mode, for the properties combo."""
        return self._resize_mode

    def close(self) -> bool:
        """Delete the figure from the repository and close the panel widget."""
        if self._deleted:
            return super().close()

        self._canvas_sync_timer.stop()
        self._canvas_late_sync_timer.stop()

        try:
            self._repo.delete_figure(figure_id=self._figure_id)
            self._deleted = True
            self.delete_requested.emit(self._figure_id)
        except Exception:
            applogger.exception(
                "Failed to delete figure (figure_id=%s)",
                self._figure_id,
            )
            return False

        return super().close()

    def closeEvent(self, event: PySide6.QtGui.QCloseEvent) -> None:
        """Keep default Qt close handling explicit for strict typing tools."""
        super().closeEvent(event)

    def reload(self) -> None:
        """Reload the latest descriptor from the repository and re-render it."""
        self._figure.clear()
        self._discard_hover_annotation()
        self._refresh_config_from_source()
        self._apply_persisted_figure_metrics_to_rcparams()
        self._reset_figure_metrics_from_rcparams_for_reload()

        descriptor = self._repo.load_figure_descriptor(self._figure_id)
        if descriptor is None:
            applogger.warning("No descriptor found for figure_id=%s", self._figure_id)
            self._capture_fixed_metrics_from_rendered_figure()
            self._schedule_canvas_geometry_sync(redraw=True)
            return

        try:
            self._prepare_canvas_geometry_before_render()
            original_descriptor = descriptor
            descriptor = self._descriptor_prepared_for_render(original_descriptor)
            render_figure_from_descriptor(
                figure=self._figure,
                descriptor=descriptor,
                repo=self._repo,
            )
            self._make_data_artists_pickable()

            # Do not normalize again from len(self._figure.axes). Matplotlib can
            # contain extra Axes created by a stale grid, colorbars, twinx axes,
            # or other renderer-side helpers. The descriptor is the authoritative
            # source for real chart-axis count, and it has already been
            # normalized before rendering. Re-normalizing from rendered axes is
            # what kept a 2x1 layout after a preview axis was deleted.
        except Exception:
            applogger.exception("Failed to render figure_id=%s", self._figure_id)
            self._figure.clear()

        self._capture_fixed_metrics_from_rendered_figure()
        if self._resize_mode == "FIXED":
            self._apply_fixed_mode_pixel_size()

        self._schedule_canvas_geometry_sync(redraw=True, late=True)

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _normalized_resize_mode(value: object) -> str:
        """Return a valid resize mode, falling back to FIT."""
        mode = str(value or "").strip().upper()
        return mode if mode in RESIZE_MODES else "FIT"

    def _apply_persisted_view_state(self) -> None:
        """Restore this figure's own resize mode and zoom when present."""
        try:
            options = self._repo.get_figure_options(self._figure_id)
        except Exception:
            applogger.exception(
                "Failed to read view state for figure_id=%s", self._figure_id
            )
            return

        view = options.get(FIGURE_VIEW_OPTIONS_KEY)
        if not isinstance(view, dict):
            return

        if CONFIG_RESIZE_MODE in view:
            self._resize_mode = self._normalized_resize_mode(view[CONFIG_RESIZE_MODE])

        self._pending_notes_html = str(view.get("notes_html", "") or "")
        sizes = view.get("notes_split")
        self._pending_notes_sizes = (
            [int(value) for value in sizes]
            if isinstance(sizes, list) and len(sizes) == 2
            else None
        )

        try:
            if "zoom_percent" in view:
                self._zoom_percent = self._clamp_zoom(int(view["zoom_percent"]))
        except (TypeError, ValueError):
            applogger.warning(
                "Ignoring invalid persisted zoom for figure_id=%s: %r",
                self._figure_id,
                view.get("zoom_percent"),
                show_dialog=False,
                raise_error=False,
            )

    def _persist_view_state(self) -> None:
        """Write this figure's resize mode and zoom into descriptor options."""
        if self._deleted:
            return
        try:
            options = self._repo.get_figure_options(self._figure_id)
            old_view = options.get(FIGURE_VIEW_OPTIONS_KEY)
            view = dict(old_view) if isinstance(old_view, dict) else {}
            new_view = {
                CONFIG_RESIZE_MODE: self._resize_mode,
                "zoom_percent": int(self._zoom_percent),
            }
            if self._notes_html:
                new_view["notes_html"] = self._notes_html
                new_view["notes_split"] = [int(size) for size in self._content_split.sizes()]
            if view == new_view:
                return
            options[FIGURE_VIEW_OPTIONS_KEY] = new_view
            self._repo.set_figure_options(self._figure_id, options)
        except Exception:
            applogger.exception(
                "Failed to persist view state for figure_id=%s", self._figure_id
            )

    def _persist_chart_panel_config(self) -> None:
        """Persist global defaults for newly-created figures."""
        chart_config = self._chart_config()
        chart_config[CONFIG_RESIZE_MODE] = self._resize_mode
        chart_config[CONFIG_INITIAL_ZOOM] = int(self._zoom_percent)
        chart_config[CONFIG_BACKGROUND_COLOR] = self._background_color
        save_config(self._config)
