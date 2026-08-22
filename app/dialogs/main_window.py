"""Main application window.

Layout: an activity rail on the left switches a QStackedWidget between the data
page (table list plus preview) and the charts page (one ChartPanel per figure in
a tab widget).  A properties QToolBox on the right edits the figure, axis, and
series of whichever chart tab is active.

Chart reloads requested by the property pages are debounced, because each one
re-renders the whole figure.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, cast

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import (
    QCloseEvent,
    QIcon,
)
from app.dialogs.log_viewer_dialog import LogViewerDialog
from app.data.sqlite_repo import SqliteRepo
from app.widgets.chart_panel import ChartPanel
from app.dialogs.create_chart_dialog import NewPlotTabDialog
from app.dialogs.import_data_dialog import ImportDataDialog, is_importable
from app.dialogs.credits_dialog import CreditsDialog
from app.dialogs.query_builder_dialog import QueryBuilderDialog
from app.widgets.axis_properties import AxisPropertiesWidget
from app.widgets.figure_properties import FigurePropertiesWidget
from app.widgets.series_properties import SeriesPropertiesWidget
from app.widgets.series_operation import SeriesOperationWidget
from app.scanners.series_operation_scanner import import_class_from_file
from app.styles.style import (
    PANEL_MIN_WIDTH,
    action_menu_item,
    action_presentation,
    SPLITTER_HANDLE_WIDTH,
    apply_toolbox_header_metrics,
    create_card_widget,
    create_menu,
    load_icon,
    relax_minimum_width,
    stdSizeAndlayout,
)
from app.widgets.table_list import  TableListPanel
from app.widgets.table_preview import TablePreviewPanel
from app.utils.config import get_section, set_last_database, set_section
from app.utils.dialog_state import restore_window_geometry, save_window_geometry
from app.utils.messages import show_message
from app.logs.logger import applogger
from app.utils.i18n import _
from PySide6.QtWidgets import QApplication, QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QMainWindow, QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QTabWidget, QToolBox, QToolButton, QVBoxLayout, QWidget

# Coalescing window for property-driven chart reloads, in milliseconds.
# Long enough to swallow a spinbox drag, short enough to feel immediate.
PROPERTIES_REDRAW_DEBOUNCE_MS: int = 120

# config.json keys for the remembered window layout.
STATE_KEY: str = "main_window"
SPLITTERS_SECTION: str = "main_window_splitters"
LOG_VIEWER_STATE_KEY: str = "log_viewer"

# Narrowest useful chart pane.  Explicit, because the alternative is whatever
# the chart toolbar happens to add up to - and that number silently wins the
# splitter negotiation against the left panel.
CHART_PANE_MIN_WIDTH: int = 260

# How long a picked-point readout stays in the status bar, in milliseconds.
# Long enough to read and write down, short enough that it is gone before it
# can be mistaken for a description of some later chart.
CHART_SELECTION_TIMEOUT_MS: int = 15_000

class MainWindow(QMainWindow):
    """Main window with custom activity rail and chart tabs.

    The top-level window must remain shrinkable. To avoid child widgets
    propagating a large minimum height upward, the central layout explicitly
    uses zero minimum sizes and a scrollable properties page.
    """

    def __init__(self, repo: SqliteRepo, db_path: Path) -> None:
        super().__init__()
        self._repo = repo
        self._db_path = db_path
        applogger.set_status_bar(self.statusBar())
        applogger.debug(f"Initializing main window for database: {db_path}")

        self.setWindowTitle(_("Data Hub"))
        self.setWindowIcon(load_icon("new_plot"))
        self.resize(1200, 800)

        # Debounce for property-driven chart reloads (see _redraw_properties_chart).
        self._properties_redraw_callback: Any | None = None
        # The panel itself, kept alongside the raw Figure the other property
        # widgets get: the fit-mode combo calls panel.set_resize_mode and
        # panel.resize_mode directly, since that state is not a descriptor
        # key routed through figure_options_requested like everything else
        # the figure widget writes.
        self._properties_panel: ChartPanel | None = None
        self._properties_redraw_timer = QTimer(self)
        self._properties_redraw_timer.setSingleShot(True)
        self._properties_redraw_timer.timeout.connect(self._flush_properties_chart_redraw)

        self.statusBar().showMessage("Ready")

        # Files can be dropped on the window: see dropEvent.
        self.setAcceptDrops(True)

        # Keep the whole window shrinkable.
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)

        # Right side: chart tabs.
        self._tabs = QTabWidget(self)
        self._tabs.currentChanged.connect(self._on_chart_tab_changed)
        self._configure_tabs()

        # Left side data widgets.
        self._table_panel = TableListPanel(parent=self, repo=self._repo)
        self._table_panel.tableSelected.connect(self._on_table_selected)
        self._preview = TablePreviewPanel(parent=self, repo=self._repo)
        self._preview.refresh.connect(self.refresh)
        # Left-side pages.
        self._data_page = self._create_data_page()
        self._properties_control = self._create_properties_control()
        self._configure_properties_control()

        # Build VS Code-like rail + stacked pages.
        self._build_app_menu()
        self._left_stack = self._create_left_stack()
        self._left_rail = self._create_activity_rail()
        self._left_panel = self._create_left_panel()
        self._configure_left_panel()

        # Main split: left panel + chart tabs.
        self._main_split = self._create_main_split()

        # Wrap the splitter in a plain central widget with a zero-minimum layout.
        self._central_host = self._create_central_host()
        self.setCentralWidget(self._central_host)

        # Default page.
        self._set_nav_index(0)
        self._table_panel.reload()
        self._reload_tabs()
        self._update_properties_for_current_chart()

        # Last: the saved geometry must win over the resize() above and over
        # any size hint the freshly populated panels have just produced.
        self._restore_layout()

        applogger.debug("Main window initialized")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _create_properties_control(self) -> QToolBox:
        """Create the properties QToolBox directly in the main window."""
        self._figure_widget = FigurePropertiesWidget(self)
        self._axis_widget = AxisPropertiesWidget(self)
        self._series_widget = SeriesPropertiesWidget(self)

        control = QToolBox(self)
        control.setObjectName("propertiesToolBox")
        self._figure_properties_index = control.addItem(
            self._figure_widget,
            _("Figure properties"),
        )
        self._axis_properties_index = control.addItem(
            self._axis_widget,
            _("Axis properties"),
        )
        self._series_properties_index = control.addItem(
            self._series_widget,
            _("Series properties"),
        )
        control.setCurrentIndex(self._figure_properties_index)
        # Section headers are sized from font metrics; QSS padding alone leaves
        # the labels clipped (see apply_toolbox_header_metrics).
        apply_toolbox_header_metrics(control)
        self._connect_property_signals()
        self._clear_property_widgets()
        return control

    def _connect_property_signals(self) -> None:
        """Connect property widgets to main-window persistence handlers."""
        self._figure_widget.style_changed.connect(self._on_figure_style_changed)
        self._figure_widget.grid_layout_requested.connect(self._on_grid_layout_requested)
        self._figure_widget.figure_options_requested.connect(self._on_figure_options_requested)
        self._axis_widget.axis_selected.connect(self._on_axis_selected)
        self._axis_widget.renderer_changed.connect(self._on_axis_renderer_changed)
        self._axis_widget.axis_options_requested.connect(self._on_axis_options_requested)
        self._axis_widget.axis_action_requested.connect(self._on_axis_action_requested)
        self._series_widget.series_options_requested.connect(self._on_series_options_requested)
        self._series_widget.series_order_requested.connect(self._on_series_order_requested)
        self._series_widget.series_delete_requested.connect(self._on_series_delete_requested)

    def _configure_tabs(self) -> None:
        """Make the chart tabs shrink-friendly in both directions.

        A QTabWidget is as wide as the wider of its tab bar and its current
        page, and both push back by default: the bar lays every tab out at its
        full title width, and the page reports the chart toolbar's width.  With
        that floor in place the splitter has no room left to give the left
        panel, which is why the left panel appeared to ignore its own minimum.
        Eliding the titles and scrolling the bar removes the first half; the
        second is handled inside ChartPanel.
        """
        self._tabs.setObjectName("chartTabs")
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(False)
        self._tabs.setTabsClosable(False)
        self._tabs.setMinimumSize(0, 0)
        self._tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        tab_bar = self._tabs.tabBar()
        tab_bar.setUsesScrollButtons(True)
        tab_bar.setExpanding(False)
        tab_bar.setElideMode(Qt.TextElideMode.ElideNone)
        #tab_bar.setExpanding(False)
        
        # Without this the bar still asks for the full width of every title.
        tab_bar.setMinimumWidth(300)
        tab_bar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

    def _configure_properties_control(self) -> None:
        """Keep the properties control shrink-friendly.

        The properties pane is later wrapped in a scroll area so it can exceed
        the available height without forcing the whole main window taller.
        """
        self._properties_control.setMinimumSize(0, 0)
        self._properties_control.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Ignored,
        )

    def _configure_left_panel(self) -> None:
        """Configure the composite left panel to avoid height lock-up."""
        self._left_panel.setMinimumSize(0, 0)
        self._left_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )

    def _create_central_host(self) -> QWidget:
        """Wrap the main splitter in a neutral central widget.

        This prevents the top-level QMainWindow from taking an over-constrained
        size hint directly from the splitter tree.
        """
        host = QWidget(self)
        host.setMinimumSize(0, 0)
        host.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(host)
        stdSizeAndlayout(layout)
        layout.addWidget(self._main_split, 1)

        return host

    def _on_settings(self) -> None:
        """Open the application preferences.

        The menus are rebuilt afterwards because the stylesheet may have
        changed underneath them, and because a saved language has to reach the
        one part of the interface that is cheap to rebuild - the rest of it
        picks the new catalogue up at the next start, which the dialog says.
        """
        from app.dialogs.settings_dialog import SettingsDialog

        dialog = SettingsDialog(self)
        if dialog.exec():
            self._build_app_menu()

    def _show_log_viewer(self) -> None:
        """Open a read-only viewer for recent in-memory log records.

        Laid out like every other dialog in the app - shell margins, a titled
        card around the content, a button row at the bottom - rather than a
        bare text box filling the window to its edges.  The records are
        monospaced and not wrapped: a log line is columns (time, level, caller,
        message), and wrapping destroys the alignment that makes it scannable.
        """
        dialog = LogViewerDialog(self)
        restore_window_geometry(dialog, LOG_VIEWER_STATE_KEY)
        dialog.exec()
        save_window_geometry(dialog, LOG_VIEWER_STATE_KEY)

    # ------------------------------------------------------------------
    # Left pages
    # ------------------------------------------------------------------
    def _create_data_page(self) -> QWidget:
        """Create the Data page with table list and preview splitter."""
        page = create_card_widget(self, "dataPageCard")
        page.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout = QVBoxLayout(page)
        stdSizeAndlayout(layout)
        split = self._data_split = QSplitter(Qt.Orientation.Vertical, page)
        split.setChildrenCollapsible(True)
        split.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        split.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._table_panel.setMinimumSize(0, 0)
        self._table_panel.setContentsMargins(5,5,5,5)
        self._table_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._preview.setContentsMargins(5,5,5,5)
        self._preview.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        split.addWidget(self._table_panel)
        split.addWidget(self._preview)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)

        layout.addWidget(split, 1)
        return page

    def _create_series_operations_page(self) -> QWidget:
        """Create the series operations page."""
        widget = SeriesOperationWidget(self)
        widget.operation_requested.connect(self._on_series_operation_requested)
        return widget

    def _on_series_operation_requested(self, operation: dict) -> None:
        """Handle a built-in or runtime series-operation request."""
        icon = SeriesOperationWidget.plugin_icon(operation)

        if operation.get("name") == "NewPlotTabDialog":
            self._on_new_plot_tab(icon)
            return

        dialog_class = import_class_from_file(operation)
        if dialog_class is None:
            applogger.error(
                "Could not load series operation class: %r",
                operation.get("name"),
            )
            return

        self._open_series_operation(dialog_class, icon)

    def _create_left_stack(self) -> QStackedWidget:
        """Create the stacked pages shown next to the activity rail.

        The properties page is wrapped in a QScrollArea so it scrolls vertically
        instead of forcing the whole main window to keep a large minimum height.
        """
        stack = QStackedWidget(self)
        stack.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding,)
        stack.addWidget(self._data_page)
        
        properties_scroll = QScrollArea(self)
        stdSizeAndlayout(properties_scroll)
        properties_scroll.setMinimumSize(0, 0)
        properties_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        properties_scroll.setWidgetResizable(True)
        properties_scroll.setWidget(self._properties_control)
        stack.addWidget(properties_scroll)
        stack.addWidget(self._create_series_operations_page())
        return stack

    # ------------------------------------------------------------------
    # Activity rail
    # ------------------------------------------------------------------
    def _build_app_menu(self) -> None:
        """Create the compact menus shown from the activity rail.

        Items come from the action catalogue, so each one's icon, label and
        tooltip are defined together in a single place and are translated for
        the active language.  Rebuilding these menus is all that switching
        language requires.
        """
        self._app_menu = create_menu(
            self,
            [
                action_menu_item("new", self._on_new_file),
                action_menu_item("open", self._on_open_database),
                action_menu_item("import", self._on_import_data),
                action_menu_item("query_builder", self._on_query_builder),
                None,
                action_menu_item("optimize_db", self._on_optimize_db),
                None,
                action_menu_item("settings", self._on_settings),
                action_menu_item("log_viewer", self._show_log_viewer),
                action_menu_item("credits", self._on_credits),
            ],
        )


    def _create_activity_rail(self) -> QFrame:
        """Create a VS Code-like vertical activity rail."""
        rail = QFrame(self)
        rail.setObjectName("activityRail")
        rail.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        rail.setFixedWidth(48)
        rail.setMinimumHeight(0)
        rail.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(rail)
        stdSizeAndlayout(layout)

        # Navigation buttons.
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_group.idClicked.connect(self._set_nav_index)
        self._nav_buttons: list[QToolButton] = []
        nav_actions = (
            "nav_menu",
            "nav_data",
            "nav_chart_options",
            "nav_series_operations",
        )
        for index, action_id in enumerate(nav_actions):
            button = self._create_activity_button(action_id=action_id)
            self._nav_group.addButton(button, index)
            self._nav_buttons.append(button)
            layout.addWidget( button,0,Qt.AlignmentFlag.AlignHCenter,)

        layout.addStretch(1)

        
        return rail

    def _create_activity_button(self, *, action_id: str) -> QToolButton:
        """Create one icon-only activity rail button from its catalogue entry.

        Icon-only, so the label is dropped and the description carries the
        whole meaning of the button - which is why these four actions are the
        ones where a missing description would be most felt.
        """
        icon, _text, tooltip = action_presentation(action_id)

        button = QToolButton(self)
        button.setObjectName("activityButton")
        button.setAutoRaise(True)
        button.setIcon(icon)
        button.setIconSize(QSize(20, 20))
        button.setToolTip(tooltip)
        button.setStatusTip(tooltip)
        button.setFixedSize(32, 32)
        button.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        if action_id == "nav_menu":
            button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            button.setMenu(self._app_menu)
        else:
            button.setCheckable(True)
        return button

    def _create_left_panel(self) -> QWidget:
        """Create the left-side area: activity rail + stacked content."""
        panel = create_card_widget(self, "leftPanelCard")
        panel.setMinimumSize(0, 0)
        panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )

        layout = QHBoxLayout(panel)
        stdSizeAndlayout(layout)
        layout.addWidget(self._left_rail, 0)
        layout.addWidget(self._left_stack, 1)

        return panel

    def _set_nav_index(self, index: int) -> None:
        """Switch the active left-side page from the activity rail."""
        if index>0:
            self._left_stack.setCurrentIndex(index-1)
            self._nav_buttons[index].setChecked(True)
            applogger.debug("Left navigation page changed to index %s", index)


    # ------------------------------------------------------------------
    # Main layout
    # ------------------------------------------------------------------
    def _create_main_split(self) -> QSplitter:
        """Create the main horizontal splitter.

        Three things make the handle track the mouse instead of snapping:

        * the left panel's children have their implicit minimum widths removed
          (``relax_minimum_width``), so the only floor is the one chosen here -
          previously a 24-character combo held the panel at about 300 px and
          the handle simply stopped;
        * neither pane may collapse, so there is no jump to zero at the end of
          the travel;
        * the chart pane is the only stretchy one, so growing the window does
          not silently re-widen the panel the user just narrowed.
        """
        split = QSplitter(Qt.Orientation.Horizontal, self)
        split.setChildrenCollapsible(False)
        split.setOpaqueResize(True)
        split.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        split.setMinimumSize(0, 0)
        split.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        relax_minimum_width(self._left_panel)
        # The rail is fixed-width and always visible, so the floor applies to
        # the content next to it, not to the panel as a whole.
        self._left_panel.setMinimumWidth(PANEL_MIN_WIDTH + self._left_rail.minimumWidth())

        # The chart pane needs the same treatment, and for the same reason:
        # whichever pane keeps a large implicit minimum wins the whole
        # negotiation, and the other one is squeezed past the minimum it asked
        # for.  Both floors are now explicit and comparable.
        relax_minimum_width(self._tabs)
        self._tabs.setMinimumWidth(CHART_PANE_MIN_WIDTH)

        split.addWidget(self._left_panel)
        split.addWidget(self._tabs)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 840])

        return split

    # ------------------------------------------------------------------
    # Database switching / reload
    # ------------------------------------------------------------------
    def _switch_database(self, db_path: Path) -> None:
        """Switch the UI to another database and rebind all dependent widgets."""
        applogger.info("Switching database to %s", db_path)
        self.setUpdatesEnabled(False)
        try:
            self._tabs.clear()
            self._preview.clear()
            QApplication.processEvents()

            self._repo.close()
            gc.collect()

            QApplication.processEvents()

            self._repo = SqliteRepo(db_path=db_path)
            self._db_path = db_path
            self._table_panel.set_repo(self._repo)
            self.setWindowTitle(f"Data hub: {self._db_path}")
            set_last_database(db_path)

            self._table_panel.reload()
            self._reload_tabs()
            self._update_properties_for_current_chart()
        finally:
            self.setUpdatesEnabled(True)


    def _reload_tabs(self) -> None:
        """Reload chart tabs from the current repository."""
        applogger.debug("Reloading chart tabs")
        current_index = self._tabs.currentIndex()

        self._tabs.blockSignals(True)
        try:
            self._tabs.clear()
            for fig_id, name in self._repo.load_figures_from_db():
                panel = ChartPanel(self._repo, fig_id, parent=self._tabs)
                panel.setMinimumSize(0, 0)
                panel.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Expanding,
                )
                panel.delete_requested.connect(self._on_chart_panel_deleted)
                # Clicking a point reports it in the status bar, which is the
                # only surface in the window that can carry a transient line
                # without moving anything else. It times out rather than
                # staying: the bar is shared with the logger, and a readout
                # that never cleared would sit there describing a chart the
                # user has since left.
                panel.selection_changed.connect(
                    lambda text: self.statusBar().showMessage(
                        text, CHART_SELECTION_TIMEOUT_MS
                    )
                )
                self._tabs.addTab(panel, name)
        finally:
            self._tabs.blockSignals(False)

        if self._tabs.count() == 0:
            self._clear_property_widgets()
            return

        if 0 <= current_index < self._tabs.count():
            self._tabs.setCurrentIndex(current_index)
        else:
            self._tabs.setCurrentIndex(0)

        self._update_properties_for_current_chart()

    def _on_chart_panel_deleted(self, figure_id: int) -> None:
        """Refresh chart tabs after a chart is deleted from a local panel menu."""
        applogger.info("Chart deleted from panel menu (figure_id=%s)", figure_id)
        self._reload_tabs()

    def _on_chart_tab_changed(self, index: int) -> None:
        """Rebind the properties control when the selected chart tab changes."""
        applogger.debug("Chart tab changed to index %s", index)
        self._update_properties_for_current_chart()

    def _update_properties_for_current_chart(self) -> None:
        """Connect the properties control to the currently selected chart."""
        panel = self._current_chart_panel()
        if panel is None :
            self._clear_property_widgets()
            return

        self._set_property_widgets_connected(
            figure_id=int(panel.figure_id),
            figure=panel.figure,
            redraw_callback=panel.reload,
            panel=panel,
        )

    def _set_property_widgets_connected(
        self,
        *,
        figure_id: int,
        figure: Any,
        redraw_callback: Any | None = None,
        panel: ChartPanel | None = None,
    ) -> None:
        """Load the direct QToolBox property pages for one chart.

        Why the property widgets get ``_redraw_properties_chart`` rather than the
        panel's own ``reload``: a full rebuild costs a complete re-render, so
        every redraw request from a spinbox or colour picker has to go through
        the same debounce as the ones raised here.
        """
        self._properties_figure_id = int(figure_id)
        self._properties_figure = figure
        self._properties_redraw_callback = redraw_callback
        self._properties_panel = panel
        for widget in (self._figure_widget, self._axis_widget, self._series_widget):
            widget.set_connected_figure(
                repo=self._repo,
                figure_id=figure_id,
                figure=figure,
                redraw_callback=self._redraw_properties_chart,
            )
        if panel is not None:
            self._figure_widget.set_resize_mode_control(
                panel.resize_mode, panel.set_resize_mode
            )
        current_axis_id = self._axis_widget.current_axis_id()
        self._series_widget.set_current_axis_id(current_axis_id)
        self._axis_widget.rebuild_kwargs_editor(current_axis_id)

    def _clear_property_widgets(self) -> None:
        """Clear all direct property pages."""
        # Drop any pending redraw: its target chart is going away.
        self._properties_redraw_timer.stop()
        self._properties_figure_id = None
        self._properties_figure = None
        self._properties_redraw_callback = None
        self._properties_panel = None
        self._figure_widget.clear_connected_figure()
        self._axis_widget.clear_connected_figure()
        self._series_widget.clear_connected_figure()
        self._axis_widget.rebuild_kwargs_editor(None)

    def _redraw_properties_chart(self) -> None:
        """Request a reload of the chart connected to the property pages.

        The request is debounced: dragging a spinbox emits a change per step and
        each reload re-renders the whole figure, so without coalescing the UI
        thread stalls for the duration of every intermediate value.  Same
        pattern as the style editor's 300 ms timer, tightened to keep the chart
        feeling live.
        """
        self._properties_redraw_timer.start(PROPERTIES_REDRAW_DEBOUNCE_MS)

    def _flush_properties_chart_redraw(self) -> None:
        """Run the debounced chart reload."""
        if self._properties_redraw_callback is not None:
            self._properties_redraw_callback()

    def _reload_property_widgets(self) -> None:
        """Reload direct property pages while preserving selected axis."""
        if self._properties_figure_id is None or self._properties_figure is None:
            return
        current_axis_id = self._axis_widget.current_axis_id()
        for widget in (self._figure_widget, self._axis_widget, self._series_widget):
            widget.set_connected_figure(
                repo=self._repo,
                figure_id=self._properties_figure_id,
                figure=self._properties_figure,
                redraw_callback=self._properties_redraw_callback,
            )
        if self._properties_panel is not None:
            self._figure_widget.set_resize_mode_control(
                self._properties_panel.resize_mode, self._properties_panel.set_resize_mode
            )
        self._series_widget.set_current_axis_id(current_axis_id)
        self._axis_widget.rebuild_kwargs_editor(current_axis_id)

    def _properties_figure_options(self) -> dict[str, Any]:
        """Return mutable figure options for the active property figure."""
        if self._properties_figure_id is None:
            return {}
        desc = self._repo.load_figure_descriptor(self._properties_figure_id)
        if desc is None:
            return {}
        if desc.options is None:
            return {}
        if isinstance(desc.options, dict):
            return dict(desc.options)
        applogger.error("Figure id=%r has invalid options.", desc.id)
        return {}

    def _properties_series_descriptor(self, series_id: int | None) -> Any | None:
        """Return one series descriptor by id from the active figure."""
        if self._properties_figure_id is None or series_id is None:
            return None
        desc = self._repo.load_figure_descriptor(self._properties_figure_id)
        if desc is None or desc.axes is None:
            return None
        for axis_desc in desc.axes:
            for series_desc in list(axis_desc.series or []):
                if int(series_desc.id) == int(series_id):
                    return series_desc
        return None

    def _axis_rows(self) -> list[tuple[int, int, str]]:
        """Return axes for the active property figure."""
        if self._properties_figure_id is None:
            return []
        return [
            (int(axis_id), int(axis_index), str(title or ""))
            for axis_id, axis_index, title in self._repo.list_axes_for_figure(
                int(self._properties_figure_id)
            )
        ]

    def _move_axis_descriptor(self, axis_id: int, delta: int) -> bool:
        """Move one axis through repository-managed index swapping."""
        rows = self._axis_rows()
        ordered_ids = [axis_id_value for axis_id_value, _index, _title in rows]
        if not ordered_ids or self._properties_figure_id is None:
            return False
        try:
            current_index = ordered_ids.index(int(axis_id))
        except ValueError:
            return False
        target_index = current_index + int(delta)
        if target_index < 0 or target_index >= len(ordered_ids):
            return False
        self._repo.swap_axis_indexes(
            figure_id=int(self._properties_figure_id),
            first_axis_id=ordered_ids[current_index],
            second_axis_id=ordered_ids[target_index],
        )
        return True

    def _delete_axis_descriptor(self, axis_id: int) -> bool:
        """Delete one axis through the repository."""
        existing_axis_ids = {axis_id_value for axis_id_value, _index, _title in self._axis_rows()}
        if int(axis_id) not in existing_axis_ids:
            return False
        self._repo.delete_axis(int(axis_id))
        return True

    def _on_figure_style_changed(self, style_text: str) -> None:
        if self._properties_figure_id is None:
            return
        options = self._properties_figure_options()
        style = str(style_text or "")
        if style:
            options["mpl_style"] = style
        else:
            options.pop("mpl_style", None)
        self._repo.set_figure_options(self._properties_figure_id, options)
        self._redraw_properties_chart()

    def _on_grid_layout_requested(self, nrows: int, ncols: int) -> None:
        if self._properties_figure_id is None:
            return
        self._repo.set_figure_grid(
            self._properties_figure_id,
            nrows=int(nrows),
            ncols=int(ncols),
        )
        self._redraw_properties_chart()

    # Figure payload keys handled outside the generic copy below.
    _FIGURE_PAYLOAD_NON_OPTIONS: frozenset[str] = frozenset({"name", "layout"})

    def _on_figure_options_requested(self, payload: dict[str, Any]) -> None:
        """Persist the figure options the properties widget just emitted.

        Same rule as the axis handler: store the whole payload rather than a
        hand-listed subset, so a control added to the widget cannot silently
        fail to persist. None removes the key.
        """
        if self._properties_figure_id is None:
            return

        options = self._properties_figure_options()

        for key, value in payload.items():
            if key in self._FIGURE_PAYLOAD_NON_OPTIONS:
                continue
            if value is None or (key == "mpl_style" and not str(value)):
                options.pop(key, None)
            else:
                options[key] = value

        # "layout" is the older spelling; keep layout_mode authoritative.
        options["layout_mode"] = str(
            payload.get("layout_mode", payload.get("layout", "constrained"))
            or "constrained"
        )

        self._repo.set_figure_options(self._properties_figure_id, options)
        self._rename_figure_if_requested(payload)
        self._redraw_properties_chart()

    def _rename_figure_if_requested(self, payload: dict[str, Any]) -> None:
        """Rename the figure and its chart tab when the payload carries a name."""
        if self._properties_figure_id is None or "name" not in payload:
            return

        name = str(payload.get("name") or "").strip()
        if not name:
            return

        figure_id = int(self._properties_figure_id)
        try:
            descriptor = self._repo.load_figure_descriptor(figure_id)
            if descriptor is None or str(descriptor.name) == name:
                return
            self._repo.set_figure_properties(
                figure_id,
                nrows=int(descriptor.nrows or 1),
                ncols=int(descriptor.ncols or 1),
                name=name,
                options=descriptor.options or {},
            )
        except Exception:
            applogger.exception("Failed to rename figure_id=%s", figure_id)
            return

        self._update_tab_title(figure_id, name)

    def _update_tab_title(self, figure_id: int, name: str) -> None:
        """Retitle the chart tab that shows a given figure."""
        for index in range(self._tabs.count()):
            widget = self._tabs.widget(index)
            if isinstance(widget, ChartPanel) and int(widget.figure_id) == figure_id:
                self._tabs.setTabText(index, name)
                self._tabs.setTabToolTip(index, name)
                return

    def _on_axis_selected(self, axis_id: int) -> None:
        self._series_widget.set_current_axis_id(axis_id)
        self._axis_widget.rebuild_kwargs_editor(axis_id)

    def _on_axis_renderer_changed(self, _renderer_name: str) -> None:
        self._axis_widget.rebuild_kwargs_editor(self._axis_widget.current_axis_id())

    def _on_axis_action_requested(self, payload: dict[str, Any]) -> None:
        action = str(payload.get("action", "") or "").strip()
        axis_id_value = payload.get("axis_id")
        if axis_id_value is None:
            return
        axis_id = int(axis_id_value)
        changed = False
        if action == "move_up":
            changed = self._move_axis_descriptor(axis_id, -1)
        elif action == "move_down":
            changed = self._move_axis_descriptor(axis_id, 1)
        elif action == "delete":
            changed = self._delete_axis_descriptor(axis_id)
        if changed:
            self._reload_property_widgets()
            self._redraw_properties_chart()

    # Payload keys that are transport, not axis options: they are either
    # addressing (axis_id) or handled explicitly below.
    _AXIS_PAYLOAD_NON_OPTIONS: frozenset[str] = frozenset({"axis_id", "renderer"})

    def _on_axis_options_requested(self, payload: dict[str, Any]) -> None:
        """Persist the axis options the properties widget just emitted.

        Everything in the payload is stored, rather than a hand-listed subset.
        Why: the previous version copied ~10 named keys, so every control added
        to the widget was silently dropped here - it looked like the setting
        did nothing and reset itself on the next panel switch, because it was
        never written. A whitelist that has to be edited in a second file to
        add a control is a bug waiting to happen twice.

        None means "not configured" and is removed rather than persisted, so a
        disabled control does not overwrite a real value with null.
        """
        axis_id_value = payload.get("axis_id")
        if axis_id_value is None:
            return
        axis_id = int(axis_id_value)
        options = self._repo.get_axis_options(axis_id) or {}

        for key, value in payload.items():
            if key in self._AXIS_PAYLOAD_NON_OPTIONS:
                continue
            if value is None:
                options.pop(key, None)
            else:
                options[key] = value

        # "hidden" is the legacy spelling the renderer still reads.
        options["hidden"] = bool(payload.get("hide_axis", False))

        renderer_name = str(payload.get("renderer", "") or "").strip()
        if renderer_name:
            options["renderer"] = renderer_name
            options["renderer_name"] = renderer_name
        else:
            options.pop("renderer", None)
            options.pop("renderer_name", None)
        axis_kwargs = self._axis_widget.clean_kwargs()
        if axis_kwargs:
            options["axis_kwargs"] = axis_kwargs
        else:
            options.pop("axis_kwargs", None)
        self._repo.set_axis_options(axis_id, options)
        self._redraw_properties_chart()

    def _on_series_order_requested(self, ordered_ids: list[int]) -> None:
        axis_id_value = self._series_widget.current_axis_id()
        if axis_id_value is None:
            return
        axis_id = int(axis_id_value)
        options = self._repo.get_axis_options(axis_id) or {}
        options["series_order"] = [int(series_id) for series_id in ordered_ids]
        self._repo.set_axis_options(axis_id, options)
        self._reload_property_widgets()
        self._redraw_properties_chart()

    def _on_series_delete_requested(self, series_id: int) -> None:
        self._repo.delete_series(int(series_id))
        self._reload_property_widgets()
        self._redraw_properties_chart()

    def _on_series_options_requested(self, payload: dict[str, Any]) -> None:
        series_id_value = payload.get("series_id")
        if series_id_value is None:
            return
        series_id = int(series_id_value)
        raw_series_desc = self._properties_series_descriptor(series_id)
        if raw_series_desc is None:
            applogger.error("Series descriptor id=%r not found.", series_id)
            return
        series_desc = cast(Any, raw_series_desc)
        if series_desc.style is None:
            style: dict[str, Any] = {}
        elif isinstance(series_desc.style, dict):
            style = dict(series_desc.style)
        else:
            applogger.error("Series id=%r has invalid style.", series_desc.id)
            return
        # Same rule as the axis and figure handlers: everything the widget
        # sends is stored, minus the addressing keys.
        for key, value in payload.items():
            if key in {"series_id", "sql_query"}:
                continue
            if value is None:
                style.pop(key, None)
            else:
                style[key] = value

        sql_query = str(payload.get("sql_query", "") or "").strip()
        self._repo.update_series_style(series_id, style)
        self._repo.update_series_sql_query(series_id, sql_query)
        self._reload_property_widgets()
        self._redraw_properties_chart()

    # ------------------------------------------------------------------
    # Table panel actions
    # ------------------------------------------------------------------
    def _on_table_selected(self, table: str) -> None:
        """Update the preview panel when a table is selected."""
        applogger.debug("Selected table: %s", table)
        self._preview.set_context(self._repo, table)


    # ------------------------------------------------------------------
    # App menu actions
    # ------------------------------------------------------------------
        
    def _on_new_file(self) -> None:
        """Create a new database, safely replacing an existing file if possible."""
        base_dir = str(self._db_path.parent) if self._db_path else ""
        file_path, _unused = QFileDialog.getSaveFileName(
            self,
            _("New database"),
            base_dir,
            "Data Hub DB (*.dhub)",
        )
        if not file_path:
            return

        db_path = SqliteRepo.ensure_dhub_extension(Path(file_path))
        applogger.info("Creating new database: %s", db_path)

        self._tabs.clear()
        self._preview.clear()
        QApplication.processEvents()

        if self._repo:
            self._repo.close()

        gc.collect()
        QApplication.processEvents()

        try:
            if db_path.exists():
                db_path.unlink()

            self._repo = SqliteRepo(db_path=db_path)
            self._db_path = db_path
            self._table_panel.set_repo(self._repo)
            self.setWindowTitle(f"Data hub: {self._db_path}")
            set_last_database(db_path)
            self._table_panel.reload()
            self._reload_tabs()
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to create database: %s", exc)
            show_message(self, "database.create_failed", error=exc)

    def _on_open_database(self) -> None:
        """Open an existing database and switch the current UI."""
        base_dir = str(self._db_path.parent) if self._db_path else ""
        file_path, _unused = QFileDialog.getOpenFileName(
            self,
            _("Open database"),
            base_dir,
            "Data Hub DB (*.dhub)",
        )
        if not file_path:
            return

        db_path = Path(file_path)
        applogger.info("Opening database: %s", db_path)

        try:
            self._switch_database(db_path)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Failed to open database: %s", exc)
            show_message(self, "database.open_failed", error=exc)

    def _on_import_data(self, source_path: Path | None = None) -> None:
        """Open the import dialog and refresh UI if import succeeds.

        ``source_path`` is the file a drop arrived with; the dialog then opens
        already showing it, with its preview and its default table name, which
        is the whole point of dropping it rather than browsing for it.
        """
        dlg = ImportDataDialog(self._repo, parent=self)
        if source_path is not None:
            dlg.load_file(source_path)
        if dlg.exec():
            self._table_panel.reload()
            self._reload_tabs()


    def _on_new_plot_tab(self, icon: QIcon | None = None) -> None:
        """Create a chart tab, optionally using the operation-list icon.

        ``TableListPanel`` invokes this callback without arguments, while the
        series-operation page supplies the Plot operation icon.
        """
        panel = self._current_chart_panel()
        dialog = NewPlotTabDialog(
            self._repo,
            current_figure_id=(
                int(panel.figure_id) if panel is not None else None
            ),
            current_table=self._table_panel.current,
            parent=self,
        )

        if icon is None or icon.isNull():
            icon = SeriesOperationWidget.plugin_icon(
                {
                    "name": "NewPlotTabDialog",
                    "value": "Plot",
                    "icon": NewPlotTabDialog.Icon,
                    "builtin": True,
                }
            )

        if not icon.isNull():
            dialog.setWindowIcon(icon)

        if dialog.exec():
            self._table_panel.reload()
            self._reload_tabs()

    def _get_current_figure_id(self) -> tuple[ChartPanel | None, int | None]:
        """Return the figure ID of the currently selected chart tab, if any."""
        panel = self._current_chart_panel()
        if panel is None:
            return None, None
        id=panel.figure_id
        return (panel, id)
    
    def refresh(self):
        panel, id = self._get_current_figure_id()
        if id is not None and panel is not None:
            panel.reload()
            self._table_panel.reload()

    def refresh2(self):
        """Refresh active chart and data panes after series-operation Preview/Apply.

        Series operation dialogs emit ``applied`` for three cases:
        - Preview wrote temporary chart/data changes.
        - Apply committed final changes.
        - Close/Cancel rolled preview changes back.

        The previous implementation only rebound the properties pane, so the
        chart canvas and selected dataset preview could remain stale.
        """
        panel, _figure_id = self._get_current_figure_id()
        if panel is not None:
            panel.reload()

        try:
            self._table_panel.reload()
            if self._table_panel.current:
                self._preview.set_context(self._repo, str(self._table_panel.current))
        except Exception:
            applogger.exception("Failed to refresh data panes after chart operation.")

        self._update_properties_for_current_chart()

    def _open_series_operation(
        self,
        dialog_class: type,
        icon: QIcon | None = None,
    ) -> None:
        """Open one runtime series-operation dialog on the current chart."""
        panel, figure_id = self._get_current_figure_id()
        if figure_id is None or panel is None:
            return

        dialog = dialog_class(
            repo=self._repo,
            figure_id=figure_id,
            parent=self,
        )

        if icon is not None and not icon.isNull():
            dialog.setWindowIcon(icon)

        dialog.applied.connect(self.refresh2)
        # Bind results to the panel active when the dialog was opened.
        dialog.results_published.connect(
            lambda markup, target=panel: target.set_notes_html(
                markup,
                append=True,
            )
        )
        dialog.exec()
        panel.reload()
        self._table_panel.reload()
        # An operation may have created a figure of its own, which is a new
        # chart tab rather than a change to this one - reloading only the panel
        # left it invisible until the next restart.
        self._reload_tabs()
        self._update_properties_for_current_chart()
   

    def _on_optimize_db(self) -> None:
        """Check the database, report what it found, then compact it."""
        report = self._repo.optimize_db()
        self.statusBar().showMessage(report.summary(), 10_000)

        if report.is_healthy:
            return

        show_message(
            self,
            "database.check_found_problems",
            report=(
                report.summary()
                + "\n\n"
                + "\n".join(report.problems[:20])
                + ("\n..." if len(report.problems) > 20 else "")
            ),
        )
    def _on_credits(self) -> None:
        """Show who made this and what it is made of."""
        CreditsDialog(parent=self).exec()

    def _on_query_builder(self) -> None:
        """Open the query builder and refresh the source lists afterwards."""
        dialog = QueryBuilderDialog(self._repo, parent=self)
        dialog.exec()
        self._table_panel.reload()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_chart_panel(self) -> ChartPanel | None:
        """Return the currently selected chart panel, if any."""
        widget = self._tabs.currentWidget()
        if isinstance(widget, ChartPanel):
            return widget
        return None

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------
    @staticmethod
    def dropped_paths(event: Any) -> list[Path]:
        """Return the local files carried by a drag, in the order dropped.

        Local files only: a drag from a browser carries a URL that names a
        file on a web server, and ``toLocalFile`` returns an empty string for
        it rather than something ``open()`` would fail on later.
        """
        data = event.mimeData()
        if data is None or not data.hasUrls():
            return []
        return [
            Path(local)
            for url in data.urls()
            if (local := url.toLocalFile())
        ]

    @classmethod
    def accepted_drop(cls, paths: list[Path]) -> tuple[str, list[Path]]:
        """Classify a drop as ``("database"|"import"|"", paths)``.

        A ``.dhub`` is a project to open, anything the import readers know is
        data to import, and everything else is refused - refused *before* the
        cursor changes, so the window says no by not offering to accept it
        rather than by a box after the fact.

        A database wins over data files dropped with it, and only one at a
        time: opening two projects at once has no meaning, and importing into
        a database that is about to be closed has less.
        """
        databases = [path for path in paths if path.suffix.lower() == ".dhub"]
        if len(databases) == 1:
            return "database", databases

        importable = [path for path in paths if is_importable(path)]
        if importable and not databases:
            return "import", importable

        return "", []

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802
        """Accept a drag only when the drop would actually do something."""
        kind, _paths = self.accepted_drop(self.dropped_paths(event))
        if kind:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: Any) -> None:  # noqa: N802
        """Keep the acceptance while the cursor moves over the window."""
        self.dragEnterEvent(event)

    def dropEvent(self, event: Any) -> None:  # noqa: N802
        """Open a dropped project, or import dropped data files.

        Several data files are handled one dialog at a time, in the order they
        were dropped, and cancelling one stops the rest: cancel means "not
        this", and the natural reading of it on the second of four files is
        "stop", not "carry on with the next one".
        """
        kind, paths = self.accepted_drop(self.dropped_paths(event))
        if not kind:
            event.ignore()
            return

        event.acceptProposedAction()

        if kind == "database":
            applogger.info("Opening dropped database: %s", paths[0])
            try:
                self._switch_database(paths[0])
            except Exception as exc:  # noqa: BLE001
                applogger.exception("Failed to open the dropped database: %s", exc)
                show_message(self, "database.open_failed", error=exc)
            return

        for path in paths:
            applogger.info("Importing dropped file: %s", path)
            if not self._import_one_dropped_file(path):
                break

    def _import_one_dropped_file(self, path: Path) -> bool:
        """Import one dropped file; False when the user cancelled."""
        dialog = ImportDataDialog(self._repo, parent=self)
        dialog.load_file(path)
        if not dialog.exec():
            return False
        self._table_panel.reload()
        self._reload_tabs()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Persist the window layout, then collect resources and close."""
        applogger.info("Closing main window")
        try:
            self._remember_layout()
            gc.collect()
        finally:
            super().closeEvent(event)

    # ------------------------------------------------------------------
    # Window layout persistence
    # ------------------------------------------------------------------
    def _remember_layout(self) -> None:
        """Write size, position and splitter sizes to config.json.

        Saved on close rather than on every resize: a splitter drag emits
        hundreds of events and each one would rewrite the file.
        """
        save_window_geometry(self, STATE_KEY)
        set_section(
            SPLITTERS_SECTION,
            {
                "main": self._main_split.sizes(),
                "data_page": self._data_split.sizes(),
            },
        )

    def _restore_layout(self) -> None:
        """Re-apply the saved size, position and splitter sizes.

        A stale entry - a splitter that has gained a pane since, a window saved
        on a monitor that is no longer attached - is ignored by the helpers
        rather than applied blindly.
        """
        restore_window_geometry(self, STATE_KEY)
        sizes = get_section(SPLITTERS_SECTION)
        for splitter, key in ((self._main_split, "main"), (self._data_split, "data_page")):
            saved = sizes.get(key)
            if isinstance(saved, list) and len(saved) == splitter.count():
                splitter.setSizes([int(value) for value in saved])