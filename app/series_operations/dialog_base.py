"""Shared base dialog for chart-series operation panels.

The base class owns the common shell used by smoothing, interpolation, outlier
removal and fitting dialogs:

- left side: QToolBox with Axis/Series selector, model selector and parameters
- right side: results pane and log output
- bottom action buttons

Subclasses provide operation-specific controls by overriding the builder hooks.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import html
import json
import re
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout, QHBoxLayout, QSizePolicy, QSplitter, QToolBox, QVBoxLayout, QWidget
import numpy as np

from app.data.sqlite_repo import SqliteRepo
from app.widgets.axis_series_selector import AxisSeriesSelector
from app.styles.style import (
    apply_dialog_shell,
    icon_from_svg_source,
    set_doc_link,
    apply_toolbox_header_metrics,
    create_card_widget,
    create_action_button,
    create_section_title,
    stdSizeAndlayout,
)
from app.logs.logger import applogger
from app.utils.coercion import to_numeric_axis
from app.utils.series_validation import (
    SeriesIssue,
    clean_xy,
    errors,
    validate_xy,
)
from app.series_operations.parameter_form import ParameterForm
from app.series_operations.parameter_spec import ChoiceParam, Param, defaults
from app.utils.messages import show_message
from app.utils.dialog_state import (
    restore_dialog_state,
    restore_window_geometry,
    save_dialog_state,
    save_window_geometry,
)
from app.widgets.html_results import HtmlResultsView, looks_like_html, plain_to_html
from app.utils.i18n import _

_TABLE_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")

# Every table an operation writes starts with this.  One character, and the
# source list can hide the whole class of them: a project with six fits and a
# spectral analysis has more generated tables than imported ones, and the
# imported ones are what the user is looking for.
GENERATED_TABLE_PREFIX: str = "_"


def generated_table_name(raw: str, *, fallback: str = "Result") -> str:
    """Return a safe generated-table name, prefixed and never bare.

    Callers pass whatever they assembled - an axis id, a series name, a model -
    and get back something SQLite accepts and the source list can recognise.
    """
    safe = _TABLE_SAFE_RE.sub("_", str(raw).strip()).strip("_") or fallback
    return f"{GENERATED_TABLE_PREFIX}{safe}"


@dataclass(slots=True)
class ResultSeriesSpec:
    """Repository descriptor for one generated/updated chart series."""
    name: str
    sql_query: str
    roles: Mapping[str, Any]
    style: Mapping[str, Any]

class SeriesOperationDialogBase(QDialog):
    """Abstract shell for all chart-series operation dialogs.

    Subclasses should:
    - create model controls in ``build_model_selector``;
    - create parameter controls in ``build_parameter_selector``;
    - optionally override ``build_results_pane`` when a QLabel is not enough;
    - implement ``compute_results`` and ``refresh_results``;
    - provide generated-series metadata through the apply hooks below;
    - connect operation-specific signals in ``connect_operation_signals``.
    """
    #: How the operation presents itself.  All three live on the class rather
    #: than in config.json because an operation is a plugin: dropping one .py
    #: file into ``app/series_operations`` is meant to be the whole install,
    #: and presentation that had to be registered elsewhere made that untrue.
    #: The scanner reads them straight out of the source, so none of it costs
    #: an import.
    Name: str = ""
    Description: str = ""

    #: The operation's icon, as an SVG document.  Inline rather than a path to
    #: one: a file beside the module is a second thing to copy and a second
    #: thing to lose.  ``style.icon_from_svg_source`` renders it; an operation
    #: that leaves this empty gets no icon and is otherwise unaffected.

    Icon = """
    <rect x="5" y="5" width="14" height="14" rx="3"/>
    <path d="M8.5 12h7"/>
    <path d="M12 8.5v7"/>
    """

    applied = Signal()

    #: Emitted with the run's report, as HTML, once Apply has committed.
    #: The main window forwards it to the chart panel the operation ran on, so
    #: the numbers end up beside the chart instead of disappearing with the
    #: dialog that produced them.
    results_published = Signal(str)

    def __init__(
        self,
        repo: SqliteRepo,
        figure_id: int,
        title: str="Series Operation",
        parent: QWidget | None = None,
        width: int = 900,
        height: int = 640
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle(title)
        self.resize(width, height)

        # The window icon is the operation's own artwork, set once here rather
        # than by each subclass.  Two of them had got it wrong exactly because
        # it was copied: the spectral dialog asked for "fit", and the
        # statistics dialog for "statistics", which is not a file - so that
        # window simply had no icon.  An id that has to match a filename is a
        # thing to get wrong; ``Icon`` cannot be.
        if self.Icon:
            self.setWindowIcon(icon_from_svg_source(self.Icon, size=32))

        self._repo = repo
        self._figure_id = int(figure_id)
        self._original_grid: tuple[int, int] | None = self._capture_current_grid()

        self._results_label = HtmlResultsView(self)
        self._results_view = self._results_label
        self._preview_axis_ids: set[int] = set()
        self._preview_table_names: set[str] = set()
        self._preview_active = False
        self.series_selector = AxisSeriesSelector(self._repo, self._figure_id, self)
        # Every operation is opened on one figure and keeps it: create_result_
        # axis adds to self._figure_id, so a combo that could change figures
        # would show one figure's axes while writing to another's.
        self.series_selector.set_figure_locked(self.LOCK_FIGURE_SELECTION)
        self.series_selector.set_series_visible(self.SHOWS_SERIES_SELECTOR)
        # The Axis / Series page lives inside the QToolBox left panel. It must
        # be vertically expanding; otherwise its internal series_list can expand
        # only inside the selector's fixed size and the empty space remains in
        # the QToolBox page below it. Model and Parameters already behave this
        # way because they are not forced to QSizePolicy.Fixed here.
        self.series_selector.setMinimumHeight(0)
        self.series_selector.setMaximumHeight(16777215)
        self.series_selector.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # Shared model combo used by operation-specific model selectors.
        self.model_combo = QComboBox(self)

        # Subclasses create widgets here before build_* hooks run.
        self.init_operation_widgets()

        self._model_selector_widget = self.build_model_selector()
        self._parameter_selector_widget = self.build_parameter_selector()
        self._results_widget = self.build_results_pane()

        self._build_common_ui()
        self.connect_common_signals()
        self.connect_operation_signals()

        # Every subclass gets remembered entries and geometry for free; the
        # storage key is the class name, so two dialogs never share a slot.
        self._state_key = type(self).__name__
        restore_window_geometry(self, self._state_key)
        restore_dialog_state(self, self._state_key)

    def _remember_state(self) -> None:
        """Persist entries and geometry to config.json.

        Called from every exit path - Apply, Close, the window button - because
        the useful moment to record a choice is when the user leaves, not when
        they make it.
        """
        save_dialog_state(self, self._state_key)
        save_window_geometry(self, self._state_key)


    def _series_display_name(self, row: Any) -> str:
        return str(row["name"])

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------


    def set_row_visible(self, field_widget: QWidget, visible: bool, *, form: QFormLayout | None = None) -> None:
        """Show or hide a form row together with its label.

        Hiding only the field leaves its label behind, which reads as a bug.
        Two dialogs had their own copy of this; the ``form`` argument defaults
        to ``self._parameter_form``, which is the one every subclass uses.
        """
        layout = form if form is not None else getattr(self, "_parameter_form", None)
        if layout is None:
            return

        field_widget.setVisible(visible)
        label = layout.labelForField(field_widget)
        if label is not None:
            label.setVisible(visible)

    def set_doc_link(self, title: str, url: str) -> None:
        """Point this dialog's documentation label at a method's docs."""
        label = getattr(self, "_doc_link", None)
        if label is not None:
            set_doc_link(label, title, url)

    # ------------------------------------------------------------------
    # Builder hooks
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        """Create subclass widgets before build_model_selector/build_parameter_selector.

        The base constructor calls the builder hooks, so subclasses must create
        controls used by those hooks here rather than after super().__init__().
        """
        return None

    def build_model_selector(self) -> QWidget:
        """Default model selector using the shared model_combo."""
        panel = create_card_widget(self, "operationModelCard")
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)
        stdSizeAndlayout(self.model_combo)
        layout.addWidget(self.model_combo)
        return panel

    
    #: Whether the figure combo is disabled. True for every operation so far;
    #: an operation that genuinely means to move between figures would have to
    #: keep self._figure_id in step with it, which none currently does.
    LOCK_FIGURE_SELECTION: bool = True

    #: Whether the series list is shown. An operation that generates a series
    #: rather than transforming one has nothing to select, and a picker it
    #: never reads is worse than no picker.
    SHOWS_SERIES_SELECTOR: bool = True

    #: Whether the Axis / Series page appears in the toolbox at all. The
    #: selector itself is still built and still answers selected_axis_id(),
    #: because _run_operation needs an axis to write to - this only decides
    #: whether the user is shown a page they have no decision to make on.
    SHOWS_AXIS_SERIES_PAGE: bool = True

    #: Declared parameters.  An operation that sets this gets its parameter
    #: form built, wired and read back for free; see ``parameter_spec``.
    #: Leaving it empty keeps the old behaviour, where the subclass overrides
    #: ``build_parameter_selector`` and builds the form by hand - which is
    #: still the right answer for a genuinely unusual control.
    PARAMS: tuple[Param, ...] = ()

    def build_parameter_selector(self) -> QWidget:
        """Return the parameter editor widget for the left panel.

        Builds itself from ``PARAMS`` when the operation declares any.  An
        operation that declares none must override this - hence the empty
        widget rather than NotImplementedError, which is what the dialogs
        written before ``PARAMS`` existed rely on.
        """
        if not self.PARAMS:
            return QWidget(self)

        self._parameter_form_spec = ParameterForm(
            self.PARAMS,
            self,
            on_change=self.refresh_results,
            context=self.parameter_context,
        )
        # Kept under the name the base already uses for the hand-built form, so
        # set_row_visible and the state helpers keep working unchanged.
        self._parameter_form = self._parameter_form_spec.layout
        return self._parameter_form_spec.widget

    def parameter_context(self) -> Mapping[str, Any]:
        """Return values a ``visible_for`` rule may reference but not own.

        The model combo by default, under the name ``model``: nearly every
        operation's parameter visibility depends on which model is selected,
        and the combo belongs to this base rather than to ``PARAMS``.
        """
        combo = getattr(self, "model_combo", None)
        if combo is None:
            return {}
        return {"model": combo.currentText()}

    def parameter_values(self) -> dict[str, Any]:
        """Return the declared parameters' current values, keyed by name.

        Falls back to the declared defaults when there is no form yet, so an
        operation can call this from ``compute_results`` during construction
        without a special case.
        """
        form = getattr(self, "_parameter_form_spec", None)
        if form is None:
            return defaults(self.PARAMS)
        return form.values()

    def set_parameter_values(self, values: Mapping[str, Any]) -> None:
        """Restore declared parameters from saved state."""
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.set_values(values)

    def build_results_pane(self) -> QWidget:
        """Return the shared HTML results widget."""
        return self._results_view

    def build_extra_action_buttons(self, layout: QHBoxLayout) -> None:
        """Add operation-specific buttons to the bottom row, after Preview.

        Only one operation needs this so far - fitting, where optimising the
        parameters is a separate act from drawing them - but putting the button
        in the shared row is what keeps it in the same place as every other
        dialog's buttons.
        """
        del layout

    def connect_common_signals(self) -> None:
        """Wire signals common to selector-based operation dialogs."""
        selection_changed = getattr(self.series_selector, "selection_changed", None)
        if selection_changed is not None:
            selection_changed.connect(lambda *_args: self.refresh_results())
    
    def connect_operation_signals(self) -> None:
        """Subclasses connect model/parameter widgets here."""
        return None

    # ------------------------------------------------------------------
    # Common layout
    # ------------------------------------------------------------------
    def _build_common_ui(self) -> None:
        """Build the common shell with a QToolBox left panel."""
        left_toolbox = QToolBox(self)
        left_toolbox.setObjectName("operationToolBox")
        left_toolbox.setMinimumWidth(320)
        left_toolbox.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )

        self.axis_series_panel = self.series_selector
        self.model_panel = self._model_selector_widget
        self.parameters_panel = self._parameter_selector_widget

        # QToolBox gives the current page the available page area, but the page
        # itself still follows its own size policy. Keep all three pages capable
        # of vertical expansion so the active page fills the left panel.
        for panel in (
            self.axis_series_panel,
            self.model_panel,
            self.parameters_panel,
        ):
            panel.setMinimumHeight(0)
            panel.setMaximumHeight(16777215)
            panel.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )

        # The Axis / Series page is omitted entirely for an operation that
        # selects nothing. Hiding the page's widget is not enough - a QToolBox
        # keeps the header button for a hidden page, leaving a section that
        # opens onto nothing.
        if self.SHOWS_AXIS_SERIES_PAGE:
            left_toolbox.addItem(self.axis_series_panel, _("Axis / Series"))
        left_toolbox.addItem(self.model_panel, _("Model"))
        left_toolbox.addItem(self.parameters_panel, _("Parameters"))
        left_toolbox.setCurrentIndex(0)
        apply_toolbox_header_metrics(left_toolbox)
        self.left_toolbox = left_toolbox

        right = create_card_widget(self, "operationResultsCard")
        right_layout = QVBoxLayout(right)
        stdSizeAndlayout(right_layout)

        results_title = create_section_title(_("Results"), right)
        right_layout.addWidget(results_title, 0)
        right_layout.addWidget(self._results_widget, 3)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(left_toolbox)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        right.setMinimumWidth(380)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)

        self.preview_button = create_action_button(
                                  parent=self,
                                  action_id="preview",
                                  action=self.preview,
                                  layout=action_row,
                              )

        # Operation-specific buttons go next to Preview, on the left: they act
        # on the dialog's own state, unlike Apply/Close which end it.
        self.build_extra_action_buttons(action_row)

        action_row.addStretch(1)

        self.copy_html_button = create_action_button(
                                    parent=self,
                                    action_id="copy",
                                    action=self._results_label.copy_html_to_clipboard,
                                    layout=action_row,
                                )

        self.apply_button = create_action_button(
                                parent=self,
                                action_id="apply",
                                action=self.ok,
                                layout=action_row,
                            )

        self.close_button = create_action_button(
                                parent=self,
                                action_id="close",
                                action=self.reject,
                                layout=action_row,
                            )

        root = QVBoxLayout(self)
        # One helper owns dialog padding and size for every dialog in the app.
        apply_dialog_shell(self, root, size=None)
        root.addWidget(splitter, 1)
        root.addLayout(action_row, 0)


    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def selected_series(self) -> list[Any]:
        return self.series_selector.selected_series()

    def _selected_series_row(self) -> Any:
        rows = self.series_selector.selected_series()

        if not rows:
            message = "Select one source series in the Axis / Series panel."
            applogger.error(message)
            raise ValueError(message)

        if len(rows) > 1:
            applogger.warning(
                "Multiple source series selected; using the first selected series."
            )

        return rows[0]
    

    #: Whether ``format_results`` returns markup rather than plain text.
    #: A dialog that builds a table has to set this, or its markup is escaped
    #: and the user reads ``<tr><td>`` instead of a table - which is what the
    #: spectral dialog did, and why the cluster dialog gave up on HTML.
    RESULTS_ARE_HTML: bool = False

    def set_results_text(self, text: str) -> None:
        """Show results in the results pane.

        Several operation dialogs return generated HTML reports while older
        callers still pass plain-text summaries through this same method.
        Route HTML-looking content to ``setHtml`` so tags render as markup,
        and route everything else through ``setText`` so plain text remains
        escaped and safe.
        """
        content = str(text or "")

        if hasattr(self._results_label, "setContent"):
            self._results_label.setContent(content)
            return

        if looks_like_html(content):
            self._results_label.setHtml(content)
        else:
            self._results_label.setText(content)

    def set_results_html(self, markup: str) -> None:
        """Show markup in the results pane."""
        self._results_label.setHtml(str(markup or ""))

    def publish_results(self, formatted: str) -> None:
        """Show ``format_results`` output, as text or as markup.

        One entry point so every code path - preview, apply, refresh - honours
        RESULTS_ARE_HTML.  Three of them used to call set_results_text
        directly, so a dialog that returned HTML had it escaped in some places
        and rendered in others.
        """
        if self.RESULTS_ARE_HTML:
            self.set_results_html(formatted)
        else:
            self.set_results_text(formatted)

    def refresh_results(self) -> None:
        """Recompute/refresh the right-side results pane."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared generated-series apply pipeline
    # ------------------------------------------------------------------
    @property
    def operation_label(self) -> str:
        """Human readable operation name used in messages/logs."""
        return self.windowTitle() or self.__class__.__name__

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        """Style key/value pairs identifying series generated by this dialog.

        Example::

            {"generated_smoothing": True, "smoothing_dialog": "series_smoothing"}

        ``remove_previous_generated_series`` deletes only descriptors whose
        ``style_json`` contains all these key/value pairs.
        """
        raise NotImplementedError

    def compute_results(self) -> Sequence[Any]:
        """Return operation-specific result objects for the selected series."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    #: What this operation needs of its input.  Subclasses override the ones
    #: that matter to them; the defaults are the weakest useful set, so an
    #: operation that says nothing still gets the empty/length/NaN checks
    #: without having its input rejected for a shape it handles fine.
    #:
    #: See ``app/utils/series_validation.py`` for what each one catches.
    INPUT_MINIMUM_POINTS: int = 2
    INPUT_REQUIRES_SORTED_X: bool = False
    INPUT_REQUIRES_UNIQUE_X: bool = False
    INPUT_REQUIRES_UNIFORM_X: bool = False
    INPUT_REQUIRES_VARYING_Y: bool = False

    def validate_input_xy(
        self,
        x: Any,
        y: Any,
        *,
        label: str = "",
        raise_on_error: bool = True,
        repairable: bool = False,
    ) -> list[SeriesIssue]:
        """Check one series against this operation's requirements.

        Warnings are logged and the caller continues; errors are logged and,
        by default, raised - so an operation that does not check the return
        value still cannot run on data that would give a wrong answer.

        Pass ``raise_on_error=False`` to collect problems across several series
        and report them together, which is what the multi-series dialogs want:
        one bad series should not hide the six good ones.
        """
        issues = validate_xy(
            x,
            y,
            minimum_points=self.INPUT_MINIMUM_POINTS,
            require_sorted_x=self.INPUT_REQUIRES_SORTED_X,
            require_unique_x=self.INPUT_REQUIRES_UNIQUE_X,
            require_uniform_x=self.INPUT_REQUIRES_UNIFORM_X,
            require_varying_y=self.INPUT_REQUIRES_VARYING_Y,
            repairable=repairable,
            label=label,
        )

        for issue in issues:
            if issue.severity == "warning":
                applogger.warning(
                    issue.message, show_dialog=False, raise_error=False
                )

        blocking = errors(issues)
        if blocking and raise_on_error:
            # One message, not one per issue: a dialog that pops three boxes
            # for one bad series trains the user to dismiss them unread.
            applogger.error(
                " ".join(issue.message for issue in blocking),
                show_dialog=True,
                raise_error=True,
            )

        return issues

    def prepare_input_xy(
        self,
        x: Any,
        y: Any,
        *,
        label: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate, then return the series cleaned to this operation's needs.

        The repair follows the declared requirements: an operation that needs
        sorted x gets sorted x, one that needs unique x gets duplicates
        averaged, and every operation gets non-finite points dropped.  Anything
        changed is reported, because quietly discarding a user's points is its
        own kind of wrong answer.
        """
        # repairable=True: clean_xy below fixes exactly what the declared
        # requirements ask for, so a problem it is about to solve is a warning
        # rather than a reason to stop.
        self.validate_input_xy(x, y, label=label, repairable=True)

        x_clean, y_clean, report = clean_xy(
            x,
            y,
            sort_x=self.INPUT_REQUIRES_SORTED_X,
            merge_duplicate_x=self.INPUT_REQUIRES_UNIQUE_X,
        )

        if report.changed:
            prefix = f"{label}: " if label else ""
            applogger.warning(
                f"{prefix}{report.describe()}.",
                show_dialog=False,
                raise_error=False,
            )

        return x_clean, y_clean

    def result_to_frame(self, result: Any) -> Any:
        """Return the pandas DataFrame to save for one result."""
        raise NotImplementedError

    def result_series_spec(self, axis_id: int, table_name: str, result: Any) -> ResultSeriesSpec:
        """Build the chart-series descriptor for one saved result table."""
        raise NotImplementedError

    def result_series_specs(
        self,
        axis_id: int,
        table_name: str,
        result: Any,
    ) -> Sequence[ResultSeriesSpec]:
        """Return every series one result should draw.

        One by default, which is what almost every operation wants and what
        ``result_series_spec`` already answers. A control chart is the
        exception: its result table carries the plotted values, the centre
        line and the control limits, and drawing only the first of those
        leaves the chart without the lines that make it a control chart.

        Overriding this rather than calling create_series_descriptor directly
        keeps preview and apply on the same path - a spec listed here is
        previewed, cleaned up and applied exactly like any other.
        """
        return [self.result_series_spec(axis_id, table_name, result)]

    def result_table_name(self, axis_id: int, result: Any) -> str:
        """Return a safe default table name for one result.

        Subclasses should override when they already have stable table naming
        semantics.  The default uses common attributes when available.
        """
        source_name = str(getattr(result, "source_name", "Series") or "Series")
        raw = f"{self.__class__.__name__}_axis{axis_id}_{source_name}_{result.model}"
        return generated_table_name(raw, fallback="Series_Result")

    def format_results(self, results: Sequence[Any]) -> str:
        """Return text for the results pane after apply.

        Subclasses may override to reuse their richer preview formatting.
        """
        return f"{len(results)} result(s)"

    def load_cached_results(self) -> Sequence[Any] | None:
        """Return cached preview results when a subclass keeps them.

        Existing dialogs commonly store ``self._last_results``; the base uses
        that convention when present.
        """
        return getattr(self, "_last_results", None)


    def store_cached_results(self, results: Sequence[Any]) -> None:
        """Store latest results using the existing dialog convention."""
        if hasattr(self, "_last_results"):
            setattr(self, "_last_results", list(results))

    def write_result_table(self, table_name: str, result: Any) -> None:
        """Persist one operation result to a normal SQLite table."""
        frame = self.result_to_frame(result)
        self._repo.import_dataframe(frame, table_name=table_name, normalize_columns=False)

    def remove_previous_generated_series(self, axis_id: int) -> int:
        """Delete generated descriptors for this operation on the selected axis."""
        removed = 0
        expected = dict(self.generated_style_filter)
        for row in self._repo.get_series(axis_id) or []:
            try:
                style = json.loads(str(row["style_json"] or "{}"))
            except Exception:
                style = {}
            if all(style.get(key) == value for key, value in expected.items()):
                self._repo.delete_series(int(row["id"]))
                removed += 1
        if removed:
            applogger.info(f"Removed {removed} previous {self.operation_label} generated series.")
        return removed

    def create_result_series(self, axis_id: int, table_name: str, result: Any) -> int | None:
        """Create every chart-series descriptor for a saved result table.

        Returns the first series id, for callers that only ever make one.
        """
        first_id: int | None = None
        for spec in self.result_series_specs(axis_id, table_name, result):
            series_id = self._repo.create_series_descriptor(
                axis_id=axis_id,
                series_index=self._repo.next_series_index(axis_id),
                name=spec.name,
                sql_query=spec.sql_query,
                roles=dict(spec.roles),
                style=dict(spec.style),
            )
            applogger.info(f"Created series: {spec.name}")
            if first_id is None and series_id is not None:
                first_id = int(series_id)
        return first_id

    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        """Return the axis the results should be written to.

        The selected axis by default: a smoothed series belongs next to the
        series it smoothed.  A transform that changes the meaning of the x
        axis - a spectrum's x is frequency, not the source's x - overrides this
        to build a chart of its own, because putting frequencies on a time axis
        produces a picture that is simply wrong.
        """
        del results
        return selected_axis_id

    def create_result_axis(
        self,
        *,
        chart_type: str,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        options: Mapping[str, Any] | None = None,
    ) -> int:
        """Add an axis to the current figure and return its id.

        For operations whose output cannot share the source's axes.  A spectrum
        turns a time axis into a frequency one, and a box plot of a sample has
        no x in common with the series it summarises - putting either on top of
        its input produces a picture that is simply wrong.

        A new axis rather than a new figure: the result belongs beside what it
        was computed from, in the same tab, and a chart tab per estimate would
        bury the original.  The figure's grid grows by one row to make room,
        because an axis outside the grid is never drawn.
        """
        figure_id = self._figure_id
        axis_index = self._repo.next_axis_index(figure_id)

        descriptor = self._repo.load_figure_descriptor(figure_id=figure_id)
        rows = int(getattr(descriptor, "nrows", 1) or 1)
        cols = int(getattr(descriptor, "ncols", 1) or 1)
        if axis_index >= rows * cols:
            self._repo.set_figure_grid(figure_id, nrows=axis_index + 1, ncols=cols)

        axis_id = int(
            self._repo.create_axis_descriptor(
                figure_id=figure_id,
                axis_index=axis_index,
                chart_type=chart_type,
                title=title,
                x_label=x_label,
                y_label=y_label,
                options=dict(options or {"grid": True}),
            )
        )
        applogger.info(
            "%s: created axis %s (%s) on figure %s.",
            self.operation_label,
            axis_id,
            chart_type,
            figure_id,
        )
        return axis_id

    # ------------------------------------------------------------------
    # Where results are drawn
    # ------------------------------------------------------------------

    #: The three places a generated result can go. Shared because the choice
    #: is the same wherever it appears, and two operations offering the same
    #: decision under different words would be worse than either.
    DEST_SAME_AXIS: str = "same_axis"
    DEST_NEW_AXIS: str = "new_axis"
    DEST_NEW_FIGURE: str = "new_figure"

    @classmethod
    def destination_param(
        cls,
        *,
        name: str = "destination",
        label: str = "Draw on:",
        tooltip: str = "",
        default: str | None = None,
    ) -> Param:
        """Return the declared destination choice, worded once.

        Operations differ in *why* they default one way or another - a
        derivative needs its own scale, a drawn function usually belongs
        beside the data it is being compared with - so the default and the
        tooltip are arguments, while the options themselves are not.
        """
        return ChoiceParam(
            name,
            label,
            tooltip=tooltip,
            choices=(
                ("New axis in this figure", cls.DEST_NEW_AXIS),
                ("Same axis as the source", cls.DEST_SAME_AXIS),
                ("New figure", cls.DEST_NEW_FIGURE),
            ),
            default_value=default or cls.DEST_NEW_AXIS,
        )

    def resolve_destination_axis(
        self,
        selected_axis_id: int,
        *,
        chart_type: str,
        title: str = "",
        figure_name: str = "",
        options: Mapping[str, Any] | None = None,
        parameter: str = "destination",
    ) -> int:
        """Return the axis to draw on, honouring the declared destination.

        Creates the axis or figure once and reuses it, so adjusting a
        parameter repeatedly does not leave a trail of empty axes behind. The
        ids are remembered on the dialog so ``discard_result_target`` can undo
        them if Apply never happens.
        """
        destination = str(self.parameter_values().get(parameter, self.DEST_NEW_AXIS))
        if destination == self.DEST_SAME_AXIS:
            return selected_axis_id

        if getattr(self, "_result_axis_id", None) is None:
            if destination == self.DEST_NEW_FIGURE:
                self._result_figure_id, self._result_axis_id = self.create_result_figure(
                    name=figure_name or title or self.operation_label,
                    chart_type=chart_type,
                    title=title,
                    options=options,
                )
            else:
                self._result_figure_id = None
                self._result_axis_id = self.create_result_axis(
                    chart_type=chart_type,
                    title=title,
                    options=options,
                )

        # Falls back to the selected axis when creation failed: an
        # operation that cannot make its own axis should still draw
        # somewhere rather than raise on the way to the chart.
        return (
            int(self._result_axis_id)
            if self._result_axis_id is not None
            else selected_axis_id
        )

    def discard_result_target(self) -> None:
        """Remove an axis or figure this dialog made, when Apply never ran.

        A whole figure when that is what was made: deleting only its axis
        would leave an empty chart tab behind, which is worse than the axis it
        was meant to clean up.
        """
        if getattr(self, "_applied", False):
            return

        axis_id = getattr(self, "_result_axis_id", None)
        figure_id = getattr(self, "_result_figure_id", None)
        if axis_id is None and figure_id is None:
            return

        self._result_axis_id = None
        self._result_figure_id = None
        try:
            if figure_id is not None:
                self._repo.delete_figure(int(figure_id))
                applogger.info("Discarded the unapplied figure %s.", figure_id)
            elif axis_id is not None:
                self._repo.delete_axis(int(axis_id))
                applogger.info("Discarded the unapplied axis %s.", axis_id)
        except Exception:
            applogger.exception("Failed to discard the unapplied result target")

    def create_result_figure(
        self,
        *,
        name: str,
        chart_type: str,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        options: Mapping[str, Any] | None = None,
    ) -> tuple[int, int]:
        """Create a figure with one axis and return ``(figure_id, axis_id)``.

        For results that do not belong in the source chart at all.  A new axis
        (``create_result_axis``) keeps the result beside its input, which is
        right when the two are meant to be compared; a new figure is right when
        they are not - a derivative kept for its own sake, or a result destined
        for a different report.

        The dialog does NOT follow the new figure: ``self._figure_id`` stays
        put, because the selector is locked to it and the axis combo would
        otherwise start listing axes of a figure the user cannot see. The main
        window picks the new figure up when it reloads its tabs.
        """
        figure_id = int(self._repo.create_figure_descriptor(name=str(name), nrows=1, ncols=1))
        axis_id = int(
            self._repo.create_axis_descriptor(
                figure_id=figure_id,
                axis_index=0,
                chart_type=chart_type,
                title=title,
                x_label=x_label,
                y_label=y_label,
                options=dict(options or {"grid": True}),
            )
        )
        applogger.info(
            "%s: created figure %s with axis %s (%s).",
            self.operation_label,
            figure_id,
            axis_id,
            chart_type,
        )
        return figure_id, axis_id

    def discard_operation_artifacts(self) -> None:
        """Undo anything the operation created outside the preview savepoint.

        Called from Close/Cancel.  The savepoint covers descriptor rows, but
        writing a result table commits (pandas' to_sql does), so a dialog that
        creates its own figure has to remove it here or an abandoned chart is
        left behind.
        """
        return None

    def apply_results_to_axis(self, axis_id: int, results: Sequence[Any]) -> None:
        """Write every result as a table plus a chart series on *axis_id*.

        Every result, not the first: this runs inside the savepoint opened by
        ``_run_operation``, so anything raised here discards the whole batch.
        It used to end with ``optimize_db()``, whose VACUUM cannot run inside a
        transaction - the OperationalError aborted the operation part-way and
        left only the results already flushed to disk.  The caller optimizes
        after committing instead.
        """
        self.remove_previous_generated_series(axis_id)
        for result in results:
            table_name = self.result_table_name(axis_id, result)
            self.write_result_table(table_name, result)
            applogger.info(f"Saved result table: {table_name}")
            self.create_result_series(axis_id, table_name, result)

    @property
    def preview_style_filter(self) -> Mapping[str, Any]:
        return {**dict(self.generated_style_filter), "operation_preview": True, "preview_owner": self.__class__.__name__}

    def remove_previous_preview_series(self, axis_id: int) -> int:
        removed = 0
        expected = dict(self.preview_style_filter)
        for row in self._repo.get_series(axis_id) or []:
            try:
                style = json.loads(str(row["style_json"] or "{}"))
            except Exception:
                style = {}
            if all(style.get(key) == value for key, value in expected.items()):
                self._repo.delete_series(int(row["id"]))
                removed += 1
        return removed

    def preview_table_name(self, axis_id: int, result: Any) -> str:
        """Return the table name a preview writes to.

        Derived from the final name so the two are recognisably a pair, and
        prefixed like it: a preview is even more clearly not the user's data.
        """
        base = self.result_table_name(axis_id, result)
        return generated_table_name(f"{base}_Preview", fallback="Series_Preview")

    def _delete_preview_table(self, table_name: str) -> None:
        self._repo.delete_table(str(table_name))

    def remove_preview_tables(self, table_names: Sequence[str] | None = None) -> None:
        names = {str(name) for name in (table_names or self._preview_table_names) if str(name).strip()}
        for table_name in names:
            self._delete_preview_table(table_name)
            applogger.info(f"Deleted preview table: {table_name}")
        self._preview_table_names.difference_update(names)

    def remove_preview_artifacts(self, axis_id: int | None = None) -> None:
        if axis_id is None:
            for preview_axis_id in list(self._preview_axis_ids):
                self.remove_previous_preview_series(int(preview_axis_id))
        else:
            self.remove_previous_preview_series(int(axis_id))
        self.remove_preview_tables()

    def create_preview_series(self, axis_id: int, table_name: str, result: Any) -> int | None:
        first_id: int | None = None
        for spec in self.result_series_specs(axis_id, table_name, result):
            style = dict(spec.style)
            style.update(self.preview_style_filter)
            series_id = self._repo.create_series_descriptor(
                axis_id=axis_id,
                series_index=self._repo.next_series_index(axis_id),
                name=f"Preview: {spec.name}",
                sql_query=spec.sql_query,
                roles=dict(spec.roles),
                style=style,
            )
            if first_id is None and series_id is not None:
                first_id = int(series_id)
        return first_id

    def preview_results_to_axis(self, axis_id: int, results: Sequence[Any]) -> None:
        self.remove_preview_artifacts(axis_id)
        for result in results:
            table_name = self.preview_table_name(axis_id, result)
            self.write_result_table(table_name, result)
            self._preview_table_names.add(table_name)
            self.create_preview_series(axis_id, table_name, result)
        # No optimize_db() here: a preview also runs inside the savepoint, and
        # VACUUM cannot run in a transaction (see apply_results_to_axis).
    # TO IMPLEMENT!!!!
    def get_data(
        self,
        sql_query: str,
        roles: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return finite X/Y values and source rowids for Hide-aware queries.

        ``query_series_frame_for_hide`` exposes normalized ``x``/``y`` columns
        plus ``__rowid__``.  Do not read ``z`` here: the outlier dialog is a
        1D X/Y operation, and normal series frames often have no Z column.
        """
        source_df = self._repo.query_series_frame_for_hide(
            sql_query=sql_query,
            roles=roles,
        )
        # Not pd.to_numeric: a timestamp x would become all-NaN and the
        # caller would see an empty series (see utils.coercion).
        raw_x = to_numeric_axis(source_df["x"])
        raw_y = to_numeric_axis(source_df["y"])
        raw_rowids = source_df["__rowid__"].to_numpy(dtype=int)

        finite_mask = np.isfinite(raw_x) & np.isfinite(raw_y)
        return raw_x[finite_mask], raw_y[finite_mask], raw_rowids[finite_mask]


    def _refresh_after_preview_state_change(self) -> None:
        """Notify the owner window that chart/data panes must refresh."""
        try:
            self.applied.emit()
        except Exception:
            applogger.exception("Failed to emit preview refresh signal.")

    def _reset_preview_state_flags(self) -> None:
        """Clear in-memory preview bookkeeping after commit or rollback."""
        self._preview_active = False
        self._preview_axis_ids.clear()
        self._preview_table_names.clear()

    def _capture_current_grid(self) -> tuple[int, int] | None:
        """Return the figure grid as it was before this dialog previews.

        Preview-only operations can temporarily create extra axes.  Some repo
        paths update the figure grid separately from the axis descriptors, so a
        rollback/cleanup must restore both pieces of state.
        """
        try:
            descriptor = self._repo.load_figure_descriptor(figure_id=self._figure_id)
            rows = int(getattr(descriptor, "nrows", 1) or 1)
            cols = int(getattr(descriptor, "ncols", 1) or 1)
            return max(1, rows), max(1, cols)
        except Exception:
            applogger.exception("Failed to capture original figure grid.")
            return None

    def _restore_original_grid(self) -> None:
        """Restore the figure grid captured when the dialog opened."""
        if self._original_grid is None:
            return
        rows, cols = self._original_grid
        try:
            descriptor = self._repo.load_figure_descriptor(figure_id=self._figure_id)
            current_rows = int(getattr(descriptor, "nrows", 1) or 1)
            current_cols = int(getattr(descriptor, "ncols", 1) or 1)
            if (current_rows, current_cols) != (rows, cols):
                self._repo.set_figure_grid(self._figure_id, nrows=rows, ncols=cols)
        except Exception:
            applogger.exception("Failed to restore original figure grid.")

    # ------------------------------------------------------------------
    # Preview transaction
    # ------------------------------------------------------------------
    def begin_preview_transaction(self) -> None:
        """Open the SAVEPOINT that makes a Preview undoable.

        Why a database savepoint and not just artifact bookkeeping: removing the
        preview series and temp tables only undoes what the *generated-table*
        operations create.  Clustering writes a ClusterId column into the user's
        own source table and rewrites the selected series' SQL, and outlier
        removal flips hide flags on real rows - none of which the artifact
        cleanup can reach.  With every preview write inside one savepoint,
        Close rolls back whatever the operation did, whatever that was.

        SqliteRepo._commit() suppresses commits while the savepoint is open, so
        repository helpers that normally commit after each write stay inside it.
        """
        if self._repo.preview_savepoint_active:
            return
        self._repo.begin_preview_savepoint(
            f"preview_{self.__class__.__name__}"
        )

    def rollback_preview_transaction(self) -> bool:
        """Undo every database change made since the preview started."""
        if not self._repo.preview_savepoint_active:
            return False
        try:
            return bool(self._repo.rollback_preview_savepoint())
        except Exception:
            applogger.exception("Failed to roll back the preview savepoint")
            return False

    def commit_preview_transaction(self) -> None:
        """Make the preview changes permanent."""
        if not self._repo.preview_savepoint_active:
            return
        try:
            self._repo.release_preview_savepoint()
        except Exception:
            applogger.exception("Failed to commit the preview savepoint")

    def _finalize_successful_operation(
        self,
        *,
        results: Sequence[Any],
        verb: str,
    ) -> None:
        """Refresh UI and update the results pane after Preview/Apply."""
        self.store_cached_results(results)
        self._refresh_after_preview_state_change()
        formatted = self.format_results(results) or f"{verb}: {len(results)} result(s)."
        self.publish_results(formatted)

        # Applying is the point at which the report stops being scratch work,
        # so that is when it is handed to the chart it describes.  A preview is
        # deliberately excluded: it is undone on Close, and a note about
        # results that no longer exist is worse than no note.
        if verb == "Applied":
            self.results_published.emit(self.results_report_html(formatted, results))

    def results_report_html(self, formatted: str, results: Sequence[Any]) -> str:
        """Wrap one run's output in a titled block for the chart's notes pane.

        The heading is what makes several appended reports readable: without
        it, two runs of two different operations are one undifferentiated wall
        of text.
        """
        from datetime import datetime

        body = formatted if self.RESULTS_ARE_HTML else plain_to_html(formatted)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"<h3 style='margin:0 0 4px 0;'>{html.escape(self.operation_label)}</h3>"
            f"<p style='margin:0 0 8px 0;color:#666;font-size:9pt;'>"
            f"{stamp} - {len(results)} result(s)</p>"
            f"{body}"
        )

    def _run_operation(self, *, commit: bool) -> bool:
        """Run Preview or Apply through one simple pipeline.

        Preview first executes the same cleanup used by Close/Cancel, then writes
        new preview artifacts. Apply first removes any preview artifacts, then
        writes final artifacts and commits/exits through ``ok``.
        """
        action = "Apply" if commit else "Preview"
        try:
            axis_id_value = self.series_selector.selected_axis_id()
            if axis_id_value is None:
                # The catalogue holds the wording; the operation names the box.
                show_message(self, "series.no_axis_selected", title=self.operation_label)
                return False
            results = list(self.compute_results())
            if not results:
                show_message(self, "series.no_series_selected", title=self.operation_label)
                return False

            # Most operations write back onto the axis their inputs came from.
            # One does not: see resolve_target_axis_id.
            axis_id = self.resolve_target_axis_id(int(axis_id_value), results)

            if commit:
                self.cancel_operation_changes(refresh=False)
                self.begin_preview_transaction()
                self.apply_results_to_axis(axis_id, results)
                # Release before optimize_db: VACUUM cannot run inside an open
                # transaction.
                self.commit_preview_transaction()
                self._repo.optimize_db()
                self._reset_preview_state_flags()
                self._finalize_successful_operation(
                    results=results,
                    verb="Applied",
                )
                return True

            self.cancel_operation_changes(refresh=False)
            self.begin_preview_transaction()
            self.preview_results_to_axis(axis_id, results)
            self._preview_axis_ids.add(axis_id)
            self._preview_active = True
            self.store_cached_results(results)
            self._refresh_after_preview_state_change()
            self.publish_results(
                self.format_results(results)
                or f"Preview updated: {len(results)} result(s)."
            )
            return True

        except Exception as exc:
            self.cancel_operation_changes(refresh=True)
            if commit:
                applogger.critical(f"{action} failed: {exc}", show_dialog=True)
            else:
                applogger.error(f"{action} failed: {exc}", show_dialog=True)
            return False

    def preview(self) -> bool:
        applogger.debug("Preview")
        return self._run_operation(commit=False)

    def ok(self) -> None:
        if self.apply():
            self.accept()

    def cancel(self) -> None:
        self.cancel_operation_changes()
        super().reject()

    def cancel_operation_changes(self, *, refresh: bool = True) -> None:
        """Remove temporary Preview artifacts.

        This is intentionally the same cleanup used before creating a new
        Preview, so clicking Preview repeatedly behaves like Close followed by a
        fresh Preview.
        """
        rolled_back = self.rollback_preview_transaction()

        has_preview = bool(
            self._preview_active
            or self._preview_axis_ids
            or self._preview_table_names
        )
        if not rolled_back and not has_preview:
            return

        # The savepoint already undid everything it covered.  The artifact
        # sweep still runs when it did not, so a stale savepoint (or a preview
        # written before this dialog opened one) is not left behind.
        if not rolled_back:
            self.remove_preview_artifacts()

        # Keep the grid consistent with the rolled-back axis descriptors.
        # This fixes preview flows that grow the grid (for example 1x1 -> 2x1)
        # and then cancel back to the original axis count.
        self._restore_original_grid()

        self._reset_preview_state_flags()

        if refresh:
            self._refresh_after_preview_state_change()

    def apply(self) -> bool:
        """Run the operation and commit it; remembers the entries either way."""
        self._remember_state()
        return self._run_operation(commit=True)

    def reject(self) -> None:
        """Close/Cancel rejects temporary Preview changes."""
        self._remember_state()
        self.cancel_operation_changes()
        self.discard_operation_artifacts()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Window close button also rejects temporary Preview changes."""
        self._remember_state()
        self.cancel_operation_changes()
        self.discard_operation_artifacts()
        super().closeEvent(event)
