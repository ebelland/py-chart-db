"""Dialog to fit SQLite tables without a plot preview.

The dialog is intentionally table-oriented:
- 1D fitting: target = f(x)
- Scanned function catalog, parameter table, bounds, fixed parameters
- Explicit error handling when selected source columns are missing/stale
- Save fitted values/residuals back to a normal SQLite table
"""

from __future__ import annotations

from html import escape as html_escape
from collections.abc import Mapping
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence, cast

import numpy as np
import pandas as pd
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from scipy.optimize import least_squares

from app.data.data_source import row_value , parse_roles
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.series_operation_dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.widgets import report_html

from app.styles.style import (
    create_action_button,
    create_card_widget,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _
from app.scanners.functions_scanner import FunctionScanner


def _primary_x(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] < 1:
            applogger.error("2D model input has no columns.")
        return arr[:, 0]
    return arr

def _split_xy(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        applogger.error("This model requires 2D input with x and y independent columns.")
    return arr[:, 0], arr[:, 1]

@dataclass(slots=True)
class SeriesFitResult:
    source_table: str
    x_col: str
    target_col: str
    x2_col: str | None
    fit_mode: str
    model_name: str
    params: np.ndarray
    param_std: np.ndarray
    param_corr: np.ndarray
    param_names: list[str]
    expression: str
    evaluated_expression: str
    metrics: dict[str, float]
    output_table: str
    frame: pd.DataFrame
    message: str


class SeriesFitDialog(SeriesOperationDialogBase):
    """Fit a model to a series, or draw the parameters as they stand.

    Two verbs, deliberately separate:

    ``Fit``
        Optimises, writes the optimum into the parameter table, reports it,
        and previews the result.
    ``Preview`` / ``Apply``
        Draw the parameters currently in the table, whatever their origin - a
        fit, a hand-typed guess, or a value copied from a paper.

    Preview used to re-fit, which made a starting guess impossible to see: the
    optimiser replaced it before anything reached the chart.
    """
    Name: str  = "Fit"
    Description = "Fit data models"

    # least_squares needs residuals that actually vary: a flat y makes the
    # Jacobian singular, and the optimiser returns the starting guess with a
    # success flag rather than reporting that there was nothing to fit.
    #
    # Sorting and duplicate merging are declared because this dialog has always
    # done both - it sorted by x and averaged repeated x before fitting. A fit
    # does not strictly need either, but the declaration has to describe what
    # the code actually does to the data, or the report is a lie.
    INPUT_REQUIRES_VARYING_Y = True
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    INPUT_MINIMUM_POINTS = 2

    Icon = """
    <path d="M4 18.5h16"/>
    <path d="M4.5 18V5"/>
    <path d="M6.5 15.5c2.2-5.6 5.2-7.8 11-7.6"/>
    <circle cx="7" cy="15" r="1.2"/>
    <circle cx="11" cy="10.8" r="1.2"/>
    <circle cx="16.5" cy="8" r="1.2"/>
    """
    saved = Signal(str)

    # format_results returns a table.
    RESULTS_ARE_HTML: bool = True

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        applied_callback: Callable[[], None] | None = None,
        table: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Fit",
            parent=parent,
            width=900,
            height=560
        )
        self.setModal(True)
        self._applied_callback = applied_callback
        self._initial_table = table
        self._refresh_default_output_name(force=True)
        self.model_combo.setVisible(False)

    def init_operation_widgets(self) -> None:
        """Create fit controls before base builder hooks run.

        The base dialog invokes build_model_selector() and
        build_parameter_selector() during super().__init__().  Therefore every
        widget used by those builders must be created here.
        """
        self._applied_callback: Callable[[], None] | None = None
        self._initial_table: str | None = None
        self._selected_model: dict[str, Any] = {}
        self._function_scanner = FunctionScanner()
        self._param_defaults: list[float] = [0.0, 1.0]
        self._last_result: SeriesFitResult | None = None
        self._source_name = "Selected series"
        self._source_x_col = "x"
        self._source_y_col = "y"

        self._model_search = QLineEdit(self)
        self._models_tree = QTreeWidget(self)
        self._function_expression_html = QLabel("", self)
        self._function_expression_html.setTextFormat(Qt.TextFormat.RichText)
        self._function_expression_html.setWordWrap(True)
        self._function_expression_html.setOpenExternalLinks(False)
        self._function_expression_html.setProperty("muted", True)
        # What each p[i] means for the selected model, shown under the formula.
        self._param_names: list[str] = []
        self._param_legend = QLabel("", self)
        self._param_legend.setWordWrap(True)
        self._param_legend.setProperty("muted", True)
        self._param_legend.hide()

        self._multi_family_combo = QComboBox(self)
        self._multi_family_combo.addItems(["Gaussian", "Lorentzian", "Pseudo-Voigt"])
        self._multi_count_spin = QSpinBox(self)
        self._multi_count_spin.setRange(1, 12)
        self._multi_count_spin.setValue(2)
        self._multi_tie_width = QCheckBox(_("Tie width"), self)
        self._multi_tie_eta = QCheckBox(_("Tie eta"), self)
        self._multi_apply_btn = QPushButton(_("Apply multi-peak"), self)

        self._spin_params = QSpinBox(self)
        self._params_table = QTableWidget(0, 5, self)
        self._params_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._params_table.customContextMenuRequested.connect(self._show_params_context_menu)
        self._results_table = QTableWidget(0, 3, self)
        # Legacy/internal table used only as a data cache for compatibility.
        # It is no longer part of the visible layout; hide it immediately so it
        # does not appear as an orphan widget in the dialog's top-left corner.
        self._results_table.hide()
        self._results_table.setVisible(False)
        self._output_table_edit = QLineEdit(self)
        self._max_nfev_edit = QLineEdit("800", self)
        self._weighted_check = QCheckBox(_("Weighted RMSE/R²"), self)
        self._btn_use_fit_params = create_action_button(
                                       parent=self,
                                       action_id="copy",
                                       action=self.on_use_fit_results_as_initial,
                                   )
        self._btn_use_fit_params.setEnabled(False)
        self._btn_use_fit_params.hide()
        self._lbl_degree = QLabel(_("Degree:"), self)
        self._spin_degree = QSpinBox(self)
        self._lbl_knots = QLabel(_("Knots:"), self)
        self._spin_knots = QSpinBox(self)
        self._lbl_spacing = QLabel(_("Spacing:"), self)
        self._combo_knot_spacing = QComboBox(self)

    def build_model_selector(self) -> QWidget:
        panel = create_card_widget(self, "fitModelCard")
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)
        self._model_search.setPlaceholderText(_("Search models..."))
        self._model_search.setToolTip(_("Filter the model catalog by name."))
        layout.addWidget(self._model_search)
        self._models_tree.setHeaderHidden(True)
        self._models_tree.setMinimumHeight(170)
        self._models_tree.setToolTip(_("Pick a scanned function model to populate the formula and parameters."))
        self._models_tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        mark_editor_panel(self._models_tree)
        layout.addWidget(self._models_tree)

        multi_box = QWidget(panel)
        multi_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        multi_layout = QVBoxLayout(multi_box)
        stdSizeAndlayout(multi_layout)
        multi_title = QLabel(_("Multi-peak builder"), multi_box)
        multi_title.setProperty("muted", True)
        multi_layout.addWidget(multi_title)

        family_row = QHBoxLayout()
        stdSizeAndlayout(family_row)
        family_row.addWidget(QLabel(_("Family:"), multi_box))
        family_row.addWidget(self._multi_family_combo)
        family_row.addWidget(QLabel(_("Peaks:"), multi_box))
        family_row.addWidget(self._multi_count_spin)
        family_row.addStretch(1)
        multi_layout.addLayout(family_row)

        tie_row = QHBoxLayout()
        stdSizeAndlayout(tie_row)
        tie_row.addWidget(self._multi_tie_width)
        tie_row.addWidget(self._multi_tie_eta)
        tie_row.addStretch(1)
        multi_layout.addLayout(tie_row)

        apply_row = QHBoxLayout()
        stdSizeAndlayout(apply_row)
        apply_row.addWidget(self._multi_apply_btn)
        apply_row.addStretch(1)
        multi_layout.addLayout(apply_row)

        layout.addWidget(multi_box)
        layout.setStretchFactor(self._models_tree, 1)
        layout.setStretchFactor(multi_box, 0)
        return panel

    def build_parameter_selector(self) -> QWidget:
        return self._build_parameters_panel()

    def connect_operation_signals(self) -> None:
        self.series_selector.selection_changed.connect(
            lambda *_args: self._refresh_default_output_name(force=True)
        )
        self._model_search.textChanged.connect(self._filter_model_catalog)
        self._multi_apply_btn.clicked.connect(self.on_multi_peak)
        self._models_tree.itemSelectionChanged.connect(self._on_model_tree_selection)
        self._spin_degree.valueChanged.connect(self._on_degree_changed)
        self._spin_knots.valueChanged.connect(self._on_knots_changed)
        self._build_model_catalog()
        self._select_first_model()

    def _build_parameters_panel(self) -> QWidget:
        """Build the main Parameters panel without nested collapsible panels.

        The shared Axis / Series collapsible panel is the data source selector.
        This panel only contains expression/model options, parameter editor,
        and output/fit options.
        """
        outer = create_card_widget(self, "fitParamsCard")
        layout = QVBoxLayout(outer)
        stdSizeAndlayout(layout)

        # Expression / model options.
        option_row = QWidget(outer)
        option_layout = QHBoxLayout(option_row)
        stdSizeAndlayout(option_layout)

        self._spin_degree.setRange(1, 48)
        self._spin_degree.setValue(4)
        self._spin_degree.setToolTip(_("Polynomial degree (for polynomial-family models)."))
        self._spin_knots.setRange(2, 128)
        self._spin_knots.setValue(5)
        self._spin_knots.setToolTip(_("Number of knots (for knot-based spline models)."))
        self._combo_knot_spacing.addItems(["Quantiles", "Linear"])
        self._combo_knot_spacing.setToolTip(_("How knots are distributed along X."))

        for widget in (
            self._lbl_degree,
            self._spin_degree,
            self._lbl_knots,
            self._spin_knots,
            self._lbl_spacing,
            self._combo_knot_spacing,
        ):
            option_layout.addWidget(widget)
        option_layout.addStretch(1)
        layout.addWidget(option_row)

        layout.addWidget(QLabel(_("Expression:"), outer))
        layout.addWidget(self._function_expression_html)
        layout.addWidget(self._param_legend)

        # Parameter editor. Actions are available from the table context menu.
        self._spin_params.setRange(1, 256)
        self._spin_params.hide()

        mark_editor_panel(self._params_table)
        self._params_table.setHorizontalHeaderLabels(["Parameter", "Initial", "Lower", "Upper", "Fix"])
        self._params_table.horizontalHeader().setStretchLastSection(True)
        self._params_table.setMinimumHeight(160)
        self._params_table.setToolTip(
            _("Initial value, bounds and fixed flag for each function parameter. Apply runs the fit with these settings.")
        )
        layout.addWidget(self._params_table, 1)

        # Output / fit options.
        fit_options = QWidget(outer)
        fit_options_layout = QHBoxLayout(fit_options)
        stdSizeAndlayout(fit_options_layout)
        fit_options_layout.addWidget(QLabel(_("Max evals:")))
        self._max_nfev_edit.setMaximumWidth(90)
        self._max_nfev_edit.setToolTip(_("Maximum number of function evaluations for the optimizer."))
        fit_options_layout.addWidget(self._max_nfev_edit)
        self._weighted_check.setToolTip(_("Weight residuals by magnitude when computing RMSE/R²."))
        fit_options_layout.addWidget(self._weighted_check)
        fit_options_layout.addStretch(1)
        layout.addWidget(fit_options)

        layout.addWidget(QLabel(_("Output table:"), outer))
        self._output_table_edit.setToolTip(_("Name of the SQLite table where fitted values/residuals are saved."))
        layout.addWidget(self._output_table_edit)

        return outer

    def _catalog_data(self) -> dict[str, list[dict[str, Any]]]:
        """Return scanned fit functions grouped by category.

        The previous JSON catalog is intentionally removed.  All fit models are
        now function classes discovered from app/functions/functions.py and
        app/functions/user_functions.py through FunctionScanner.
        """
        return self._function_scanner.catalog()

    def _build_model_catalog(self) -> None:
        self._models_tree.clear()
        for cat_name, models in self._catalog_data().items():
            cat = QTreeWidgetItem([cat_name])
            self._models_tree.addTopLevelItem(cat)
            for payload in models:
                item = QTreeWidgetItem([str(payload["name"])])
                item.setData(0, Qt.ItemDataRole.UserRole, payload)
                cat.addChild(item)
            cat.setExpanded(False)

    def _filter_model_catalog(self, text: str) -> None:
        needle = text.lower().strip()
        for i in range(self._models_tree.topLevelItemCount()):
            cat = self._models_tree.topLevelItem(i)
            any_visible = False
            if cat is None:
                continue
            for j in range(cat.childCount()):
                child = cat.child(j)
                match = needle in child.text(0).lower()
                child.setHidden(not match)
                any_visible = any_visible or match
            cat.setHidden(not any_visible)
            cat.setExpanded(bool(needle) and any_visible)

    def _select_first_model(self) -> None:
        """Select Linear by default when available."""
        fallback: QTreeWidgetItem | None = None
        for i in range(self._models_tree.topLevelItemCount()):
            cat = self._models_tree.topLevelItem(i)
            if cat is None:
                continue
            for j in range(cat.childCount()):
                child = cat.child(j)
                if fallback is None:
                    fallback = child
                payload = child.data(0, Qt.ItemDataRole.UserRole)
                name = str(payload.get("name", child.text(0)) if isinstance(payload, dict) else child.text(0)).strip().lower()
                if name == "linear":
                    cat.setExpanded(True)
                    self._models_tree.setCurrentItem(child)
                    return
        if fallback is not None:
            parent = fallback.parent()
            if parent is not None:
                parent.setExpanded(True)
            self._models_tree.setCurrentItem(fallback)

    def _on_model_tree_selection(self) -> None:
        item = self._models_tree.currentItem()
        if item is None:
            return
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict):
            self._apply_model_choice_from_payload(payload)

    def _hide_model_options(self) -> None:
        for w in (self._lbl_degree, self._spin_degree, self._lbl_knots, self._spin_knots, self._lbl_spacing, self._combo_knot_spacing):
            w.hide()

    def _current_table(self) -> str:
        """Return a stable source table label for results/reporting."""
        return self._initial_table or self._source_name or "selected_series"

    def _is_2d_fit(self) -> bool:
        """The scanned function dialog currently fits 1D target = f(x)."""
        return False

    def _selected_column_names(self) -> tuple[str, str | None, str]:
        """Return source X, optional X2, and target column names."""
        return self._source_x_col or "x", None, self._source_y_col or "target"

    def _apply_model_choice_from_payload(self, payload: dict[str, Any]) -> None:
        """Apply one scanned function class to the dialog.

        All selectable models now come from ``FunctionScanner``.  There are no
        expression, rational, orthopoly, spline, or JSON catalogue branches here.
        Function metadata is the single source of truth.
        """
        self._hide_model_options()
        self._selected_model = dict(payload)
        self._param_names = [str(name) for name in payload.get("params", [])]
        self._refresh_param_legend()

        p0 = [float(v) for v in payload.get("p0", [1.0, 1.0])]
        self._param_defaults = p0
        self._spin_params.setValue(len(p0))
        self._ensure_params_rows(len(p0))
        self._populate_params_defaults()

        expression_html = str(payload.get("expression", "")).strip()
        description = str(payload.get("description", "")).strip()
        if description:
            description_html = f'<div style="color:#666; margin-top:6px;">{html_escape(description)}</div>'
            expression_html = f"{expression_html}{description_html}" if expression_html else description_html

        if expression_html:
            self._function_expression_html.setText(expression_html)
            self._function_expression_html.show()
        else:
            self._function_expression_html.clear()
            self._function_expression_html.hide()

        self._refresh_default_output_name()

    def _on_degree_changed(self, val: int) -> None:
        del val

    def _on_knots_changed(self, val: int) -> None:
        del val

    def _param_label(self, row: int) -> str:
        """Return the human-readable parameter name for the table/report."""
        name = self._param_names[row] if row < len(self._param_names) else ""
        return name if name else f"p[{row}]"

    def _refresh_param_legend(self) -> None:
        """Show the parameter meanings under the expression."""
        if not self._param_names:
            self._param_legend.clear()
            self._param_legend.hide()
            return

        self._param_legend.setText("   ".join(self._param_names))
        self._param_legend.show()

    def _ensure_params_rows(self, count: int) -> None:
        """Grow or shrink the parameter table, keeping the values already typed."""
        self._params_table.setRowCount(int(count))
        for row in range(int(count)):
            # Column 0 is rewritten every time rather than only when missing:
            # it carries the parameter's meaning, which changes with the model.
            self._params_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            for col, text in ((1, "0.0"), (2, "-inf"), (3, "inf")):
                if self._params_table.item(row, col) is None:
                    self._params_table.setItem(row, col, QTableWidgetItem(text))
            if self._params_table.cellWidget(row, 4) is None:
                self._params_table.setCellWidget(row, 4, QCheckBox(self._params_table))
        self._params_table.resizeColumnsToContents()

    def _populate_params_defaults(self) -> None:
        self._ensure_params_rows(len(self._param_defaults))
        for row, value in enumerate(self._param_defaults):
            self._params_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            self._params_table.setItem(row, 1, QTableWidgetItem(str(float(value))))
            self._params_table.setItem(row, 2, QTableWidgetItem("-inf"))
            self._params_table.setItem(row, 3, QTableWidgetItem("inf"))
            cb = self._get_fix_checkbox(row)
            cb.setChecked(False)

    def _get_fix_checkbox(self, row: int) -> QCheckBox:
        widget = self._params_table.cellWidget(row, 4)
        if not isinstance(widget, QCheckBox):
            widget = QCheckBox(self._params_table)
            self._params_table.setCellWidget(row, 4, widget)
        return widget

    @staticmethod
    def _unmatched_bracket_positions(expr: str) -> list[tuple[int, str, str]]:
        """Return unmatched bracket positions while ignoring strings and comments.

        The returned tuples are (absolute_position, character, reason).
        Brackets covered: (), [], {}. The UI message says parentheses because
        that is the most common user-facing case, but highlighting covers all
        expression grouping delimiters.
        """
        opens = {"(": ")", "[": "]", "{": "}"}
        closes = {")": "(", "]": "[", "}": "{"}
        stack: list[tuple[str, int]] = []
        errors: list[tuple[int, str, str]] = []
        quote: str | None = None
        triple_quote: str | None = None
        escape = False
        index = 0
        n = len(expr)

        while index < n:
            ch = expr[index]
            nxt3 = expr[index : index + 3]

            if quote is not None:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif triple_quote is not None and nxt3 == triple_quote:
                    quote = None
                    triple_quote = None
                    index += 2
                elif triple_quote is None and ch == quote:
                    quote = None
                index += 1
                continue

            if ch == "#":
                newline = expr.find("\n", index)
                if newline < 0:
                    break
                index = newline + 1
                continue

            if nxt3 == "'''" or nxt3 == '"""':
                quote = nxt3[0]
                triple_quote = nxt3
                index += 3
                continue

            if ch in ("'", '"'):
                quote = ch
                triple_quote = None
                index += 1
                continue

            if ch in opens:
                stack.append((ch, index))
            elif ch in closes:
                if not stack:
                    errors.append((index, ch, f"unmatched closing '{ch}'"))
                else:
                    open_ch, open_pos = stack.pop()
                    if open_ch != closes[ch]:
                        errors.append((open_pos, open_ch, f"expected '{opens[open_ch]}' before '{ch}'"))
                        errors.append((index, ch, f"unmatched closing '{ch}'"))
            index += 1

        for open_ch, open_pos in stack:
            errors.append((open_pos, open_ch, f"unmatched opening '{open_ch}'"))
        return sorted(errors, key=lambda item: item[0])


    def on_multi_peak(self) -> None:
        """Create a configurable multi-peak model from the inline Model frame."""
        family = self._multi_family_combo.currentText()
        count = int(self._multi_count_spin.value())
        tie_width = bool(self._multi_tie_width.isChecked())
        tie_eta = bool(self._multi_tie_eta.isChecked())

        params: list[str] = []
        p0: list[float] = []
        for i in range(count):
            params.extend([f"amp {i + 1}", f"center {i + 1}"])
            p0.extend([1.0, float(i)])
            if not tie_width:
                params.append(f"width {i + 1}")
                p0.append(1.0)
            if family == "Pseudo-Voigt" and not tie_eta:
                params.append(f"eta {i + 1}")
                p0.append(0.5)
        if tie_width:
            params.append("shared width")
            p0.append(1.0)
        if family == "Pseudo-Voigt" and tie_eta:
            params.append("shared eta")
            p0.append(0.5)
        params.append("offset")
        p0.append(0.0)

        formula = {
            "Gaussian": "Σ Aᵢ exp(-(x-cᵢ)²/(2wᵢ²)) + C",
            "Lorentzian": "Σ Aᵢ (0.5wᵢ)² / ((x-cᵢ)² + (0.5wᵢ)²) + C",
            "Pseudo-Voigt": "Σ Aᵢ [ηᵢ Lᵢ(x) + (1-ηᵢ) Gᵢ(x)] + C",
        }.get(family, "multi-peak")

        self._selected_model = {
            "name": f"Multi {family} ({count})",
            "_multi_peak": True,
            "family": family,
            "count": count,
            "tie_width": tie_width,
            "tie_eta": tie_eta,
            "params": params,
            "p0": p0,
            "expression": f"<b>Multi {family}</b><br>{formula}",
            "description": "Configurable multi-peak model built from the inline controls.",
        }
        self._param_names = params
        self._refresh_param_legend()
        self._param_defaults = p0
        self._spin_params.setValue(len(p0))
        self._ensure_params_rows(len(p0))
        self._populate_params_defaults()
        self._function_expression_html.setText(str(self._selected_model["expression"]))
        self._function_expression_html.show()
        self._refresh_default_output_name(force=True)


    def _load_fit_data(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        row = self._selected_series_row()
        source_name = str(parse_roles(row["roles"]).get("name", "Series"))
        sql_query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        if not sql_query:
            applogger.error("Selected series has no SQL query.")

        frame = self._repo.query_df(sql_query)
        if frame.empty:
            applogger.error("Selected series query returned no rows.")

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = [str(column) for column in frame.columns]
        numeric = [str(column) for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]

        x_col = str(roles.get("x", ""))
        y_col = str(roles.get("y", ""))
        if x_col not in columns:
            x_col = numeric[0] if numeric else ""
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else ""

        if not x_col or not y_col:
            applogger.error("Selected series query must expose at least two numeric columns.")
        if x_col == y_col:
            applogger.error("Selected series X and Y columns must be different.")

        # Same three repairs as before - drop non-finite, sort by x, average
        # repeated x - but reported rather than silent.
        x_prepared, target_prepared = self.prepare_input_xy(
            pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float),
            label=source_name,
        )

        clean_frame = cast(
            pd.DataFrame,
            pd.DataFrame({"x": x_prepared, "target": target_prepared}),
        )

        self._source_name = source_name
        self._source_x_col = x_col
        self._source_y_col = y_col
        self._refresh_default_output_name(force=True)

        x_data = clean_frame["x"].to_numpy(dtype=float)
        target_data = clean_frame["target"].to_numpy(dtype=float)
        return x_data, target_data, clean_frame

    def _model_name(self) -> str:
        item = self._models_tree.currentItem()
        if item is not None and item.parent() is not None:
            return item.text(0)
        return str(self._selected_model.get("name", "Custom"))

    def _make_multi_peak_model(self, payload: dict[str, Any]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Return a callable for the inline multi-peak model."""
        family = str(payload.get("family", "Gaussian"))
        count = int(payload.get("count", 1))
        tie_width = bool(payload.get("tie_width", False))
        tie_eta = bool(payload.get("tie_eta", False))

        def model(x_or_xy: np.ndarray, p: np.ndarray) -> np.ndarray:
            x = _primary_x(x_or_xy)
            values = np.asarray(p, dtype=float)
            idx = 0
            peaks: list[tuple[float, float, float, float]] = []
            for _i in range(count):
                amp = float(values[idx]); idx += 1
                center = float(values[idx]); idx += 1
                width_value = 1.0
                eta_value = 0.5
                if not tie_width:
                    width_value = max(abs(float(values[idx])), 1e-12); idx += 1
                if family == "Pseudo-Voigt" and not tie_eta:
                    eta_value = float(np.clip(values[idx], 0.0, 1.0)); idx += 1
                peaks.append((amp, center, width_value, eta_value))

            shared_width = 1.0
            if tie_width:
                shared_width = max(abs(float(values[idx])), 1e-12)
                idx += 1
            shared_eta = 0.5
            if family == "Pseudo-Voigt" and tie_eta:
                shared_eta = float(np.clip(values[idx], 0.0, 1.0))
                idx += 1
            offset = float(values[idx]) if idx < values.size else 0.0

            y = np.full_like(x, offset, dtype=float)
            for amp, center, width_value, eta_value in peaks:
                w = shared_width if tie_width else width_value
                w = max(w, 1e-12)
                if family == "Gaussian":
                    y += amp * np.exp(-((x - center) ** 2) / (2.0 * w * w))
                elif family == "Lorentzian":
                    g2 = (0.5 * w) ** 2
                    y += amp * g2 / (((x - center) ** 2) + g2)
                else:
                    e = shared_eta if tie_eta else eta_value
                    e = float(np.clip(e, 0.0, 1.0))
                    g = np.exp(-((x - center) ** 2) / (2.0 * w * w))
                    l = (0.5 * w) ** 2 / (((x - center) ** 2) + (0.5 * w) ** 2)
                    y += amp * (e * l + (1.0 - e) * g)
            return y

        return model

    def _build_model(self, x_data: np.ndarray, target_data: np.ndarray) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], np.ndarray] | None:
        """Build the selected scanned function model."""
        del x_data, target_data  # not needed for class-backed function models
        p0, _unused, _unused, _unused = self._collect_params_from_table()
        if self._selected_model.get("_multi_peak"):
            return self._make_multi_peak_model(self._selected_model), p0
        try:
            return self._function_scanner.make_model(self._selected_model), p0
        except Exception:
            applogger.exception(
                "Failed to build scanned fit function: %s",
                self._selected_model.get("name", ""),
            )
            return None

    def _float_item(self, row: int, col: int, default: float) -> float:
        item = self._params_table.item(row, col)
        text = item.text().strip() if item is not None else ""
        if not text:
            return default
        low = text.lower()
        if low in ("inf", "+inf", "infinity", "+infinity"):
            return math.inf
        if low in ("-inf", "-infinity"):
            return -math.inf
        return float(text)

    def _collect_params_from_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self._params_table.rowCount()
        p0 = np.zeros(n, dtype=float)
        lb = np.full(n, -np.inf, dtype=float)
        ub = np.full(n, np.inf, dtype=float)
        fixed = np.zeros(n, dtype=bool)
        for row in range(n):
            p0[row] = self._float_item(row, 1, 0.0)
            lb[row] = self._float_item(row, 2, -np.inf)
            ub[row] = self._float_item(row, 3, np.inf)
            fixed[row] = self._get_fix_checkbox(row).isChecked()
            if lb[row] > ub[row]:
                applogger.error(f"Lower bound is greater than upper bound at p[{row}].")
            p0[row] = min(max(p0[row], lb[row]), ub[row])
        return p0, lb, ub, fixed

    def _set_initial_params(self, values: np.ndarray) -> None:
        for row, value in enumerate(np.asarray(values, dtype=float)):
            if row < self._params_table.rowCount():
                self._params_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.12g}"))

    def on_fit(self) -> None:
        """Optimise the parameters, show the result, and preview it.

        Fit and Preview are deliberately different acts.  Preview draws the
        parameters currently in the table - which is how you try a starting
        guess, or hand-tune one, and see it immediately.  Fit is what changes
        those parameters: it optimises, writes the optimum back into the table
        so the table always says what is drawn, and then previews that.
        """
        self._evaluate(optimise=True)

        result = self._last_result
        if result is None:
            return

        # The table is the single source of truth for what gets drawn, so the
        # optimum has to land in it rather than only in the report.
        self._set_initial_params(np.asarray(result.params, dtype=float))
        self._fill_results_table(result.params, result.param_std)
        self._btn_use_fit_params.setEnabled(True)
        self.publish_results(self.format_results([result]))

        # A fit nobody can see is only half an answer.
        self.preview()

    def _evaluate(self, *, optimise: bool) -> None:
        """Run the model once and store the outcome in ``_last_result``.

        ``optimise=False`` evaluates the model at the parameters as they are,
        which is what Preview and Apply do; the code path is otherwise shared
        with the fit, so the residuals, metrics and output frame are computed
        exactly the same way in both cases.
        """
        try:
            x_data, target_data, clean = self._load_fit_data()
            if x_data is None or target_data is None:
                return
            built = self._build_model(x_data, target_data)
            if built is None:
                return
            model, _model_p0 = built

            p0, lb, ub, fixed = self._collect_params_from_table()


            free = ~fixed if optimise else np.zeros_like(fixed, dtype=bool)
            if not np.any(free):
                p_opt = p0.copy()
                residual = target_data - model(x_data, p_opt)
                jac_free = np.empty((target_data.size, 0))
                success = True
                message = (
                    "Evaluated at the current parameters."
                    if not optimise
                    else "All parameters fixed; evaluated model only."
                )
            else:
                sigma = self._weights_sigma(target_data) if self._weighted_check.isChecked() else None

                def residual_fun(p_free: np.ndarray) -> np.ndarray:
                    p = p0.copy()
                    p[free] = p_free
                    r = target_data - model(x_data, p)
                    if sigma is not None:
                        r = r / sigma
                    return np.asarray(r, dtype=float)

                max_nfev = max(1, int(float(self._max_nfev_edit.text().strip() or "800")))
                res = least_squares(residual_fun, p0[free], bounds=(lb[free], ub[free]), max_nfev=max_nfev)
                p_opt = p0.copy()
                p_opt[free] = res.x
                residual = target_data - model(x_data, p_opt)
                jac_free = np.asarray(res.jac, dtype=float)
                success = bool(res.success)
                message = str(res.message)
            fit_values = model(x_data, p_opt)
            uncertainty_free = free.copy()
            if not optimise and p0.size:
                jac_free = self._numerical_jacobian(model, x_data, p_opt)
                uncertainty_free = np.ones_like(fixed, dtype=bool)
            metrics = self._metrics(target_data, fit_values, int(np.count_nonzero(uncertainty_free)))
            std, _cov, corr = self._param_uncertainty(jac_free, residual, uncertainty_free, len(p0))
            frame = self._build_output_frame(clean, fit_values, residual)
            x_col, x2_col, target_col = self._selected_column_names()
            self._last_result = SeriesFitResult(
                source_table=self._current_table(),
                x_col=x_col,
                x2_col=x2_col,
                target_col=target_col,
                fit_mode="2D" if self._is_2d_fit() else "1D",
                model_name=self._model_name(),
                params=p_opt,
                param_std=std,
                param_corr=corr,
                param_names=[self._param_label(row) for row in range(len(p_opt))],
                expression=str(self._selected_model.get("expression", "")).strip(),
                evaluated_expression=self._evaluated_expression_html(p_opt),
                metrics=metrics,
                output_table=self._output_table_name(),
                frame=frame,
                message=message,
            )
            applogger.info(
                "%s %s: %s",
                "Fit" if optimise else "Evaluation",
                "success" if success else "warning",
                message,
            )
            applogger.info(self._format_metrics(metrics))
        except Exception as exc:
            applogger.exception("Table fit failed")
            self._last_result = None
            applogger.warning(f"Fit failed: {exc}")
            raise

    def build_extra_action_buttons(self, layout) -> None:
        """Add the Fit button next to Preview."""
        self.fit_button = create_action_button(
                              parent=self,
                              action_id="run_fit",
                              action=self.on_fit,
                              layout=layout,
                          )

    def _weights_sigma(self, data: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(data), np.nanmedian(np.abs(data)) * 1e-6 + 1e-12).astype(float)

    def _metrics(self, target: np.ndarray, fit: np.ndarray, p_count: int) -> dict[str, float]:
        residual = target - fit
        n = int(target.size)
        ss_res = float(np.nansum(np.square(residual)))
        ss_tot = float(np.nansum(np.square(target - np.nanmean(target))))
        return {
            "rmse": float(np.sqrt(np.nanmean(np.square(residual)))),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan,
            "ss_res": ss_res,
            "aic": n * math.log(max(ss_res / max(n, 1), 1e-300)) + 2.0 * p_count,
            "bic": n * math.log(max(ss_res / max(n, 1), 1e-300)) + p_count * math.log(max(n, 1)),
        }

    def _numerical_jacobian(
        self,
        model: Callable[[np.ndarray, np.ndarray], np.ndarray],
        x_data: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Central-difference Jacobian of model residuals with respect to all parameters."""
        params = np.asarray(p, dtype=float)
        base = np.asarray(model(x_data, params), dtype=float)
        jac = np.empty((base.size, params.size), dtype=float)
        for col in range(params.size):
            step = 1e-6 * max(1.0, abs(float(params[col])))
            p_plus = params.copy(); p_plus[col] += step
            p_minus = params.copy(); p_minus[col] -= step
            y_plus = np.asarray(model(x_data, p_plus), dtype=float)
            y_minus = np.asarray(model(x_data, p_minus), dtype=float)
            jac[:, col] = -(y_plus - y_minus) / (2.0 * step)
        return jac

    def _param_uncertainty(
        self,
        jac_free: np.ndarray,
        residual: np.ndarray,
        free: np.ndarray,
        n_params: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return parameter standard errors, covariance and correlation matrix."""
        std = np.full(n_params, np.nan, dtype=float)
        cov_full = np.full((n_params, n_params), np.nan, dtype=float)
        corr_full = np.full((n_params, n_params), np.nan, dtype=float)
        if jac_free.size == 0 or jac_free.shape[1] == 0:
            return std, cov_full, corr_full
        try:
            dof = max(1, residual.size - jac_free.shape[1])
            cov = np.linalg.pinv(jac_free.T @ jac_free) * float(np.dot(residual, residual) / dof)
            free_idx = np.flatnonzero(free)
            for i, row in enumerate(free_idx):
                for j, col in enumerate(free_idx):
                    cov_full[row, col] = cov[i, j]
            diag = np.maximum(np.diag(cov), 0.0)
            std_free = np.sqrt(diag)
            std[free_idx] = std_free
            denom = np.outer(std_free, std_free)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0.0)
            for i, row in enumerate(free_idx):
                for j, col in enumerate(free_idx):
                    corr_full[row, col] = corr[i, j]
        except Exception:
            pass
        return std, cov_full, corr_full

    def _build_output_frame(self, clean: pd.DataFrame, fit_values: np.ndarray, residual: np.ndarray) -> pd.DataFrame:
        x_col, x2_col, target_col = self._selected_column_names()
        data: dict[str, Any] = {
            "x": clean["x"].to_numpy(float),
            "target": clean["target"].to_numpy(float),
            "fit": np.asarray(fit_values, dtype=float),
            "residual": np.asarray(residual, dtype=float),
            "source_table": self._current_table(),
            "source_x_col": x_col,
            "source_target_col": target_col,
            "fit_mode": "2D" if self._is_2d_fit() else "1D",
            "model": self._model_name(),
        }
        if self._is_2d_fit():
            data["y"] = clean["y"].to_numpy(float)
            data["z"] = data["target"]
            data["z_fit"] = data["fit"]
            data["source_y_col"] = x2_col or ""
        else:
            data["y"] = data["target"]
            data["y_fit"] = data["fit"]
            data["source_y_col"] = target_col
        return pd.DataFrame(data)

    def _fill_results_table(self, params: np.ndarray, std: np.ndarray) -> None:
        """Fill the parameter grid in the left panel."""
        self._results_table.setRowCount(len(params))
        for row, value in enumerate(params):
            self._results_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            self._results_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.12g}"))
            self._results_table.setItem(
                row,
                2,
                QTableWidgetItem("" if not np.isfinite(std[row]) else f"{float(std[row]):.6g}"),
            )
        self._results_table.resizeColumnsToContents()

    def _evaluated_expression_html(self, params: np.ndarray) -> str:
        """Return the expression plus evaluated parameter values."""
        expression = str(self._selected_model.get("expression", "")).strip()
        if not expression:
            expression = html_escape(self._model_name())
        rows = []
        for row, value in enumerate(np.asarray(params, dtype=float)):
            rows.append(f"{html_escape(self._param_label(row))} = {float(value):.8g}")
        values = "<br>".join(rows)
        if not values:
            return expression
        return f"{expression}<br><br><b>Evaluated parameters</b><br>{values}"

    def _results_html(self, result: SeriesFitResult) -> str:
        """Return the fit report in the shared house style."""
        params = np.asarray(result.params, dtype=float)
        std = np.asarray(result.param_std, dtype=float)
        names = result.param_names or [self._param_label(row) for row in range(params.size)]

        parameter_rows = [
            (
                html_escape(names[row] if row < len(names) else f"p[{row}]"),
                report_html.format_number(value, digits=8),
                self._std_text(std, row),
            )
            for row, value in enumerate(params)
        ]

        metric_labels = {
            "r2": "R&sup2;",
            "rmse": "RMSE",
            "ss_res": "SS residual",
            "aic": "AIC",
            "bic": "BIC",
        }
        metric_rows = [
            (metric_labels[key], report_html.format_number(result.metrics[key]))
            for key in ("r2", "rmse", "ss_res", "aic", "bic")
            if key in result.metrics and np.isfinite(result.metrics[key])
        ]

        corr = np.asarray(result.param_corr, dtype=float)
        corr_rows: list[tuple[Any, ...]] = []
        if corr.size:
            for i, name in enumerate(names):
                row_values = [html_escape(name)]
                for j in range(len(names)):
                    value = corr[i, j] if i < corr.shape[0] and j < corr.shape[1] else math.nan
                    row_values.append("" if not np.isfinite(value) else f"{float(value):.4g}")
                corr_rows.append(tuple(row_values))

        return report_html.document(
            "Fit",
            result.model_name,
            report_html.section(
                _("Curve"),
                report_html.summary_table(
                    [
                        ("Mode", result.fit_mode),
                        ("Source", result.source_table),
                        ("Target", result.target_col or ""),
                        ("Status", result.message),
                    ]
                ),
            ),
            report_html.section(
                _("Function expression"),
                result.expression or "",
            ),
            report_html.section(
                _("Function expression with evaluated parameters"),
                result.evaluated_expression or "",
            ),
            report_html.section(
                _("Parameter estimates"),
                report_html.table(
                    ["Parameter", "Estimate", "Std. error"],
                    parameter_rows,
                ),
            ),
            report_html.section(
                _("Correlation matrix"),
                report_html.table(
                    ["Parameter", *[html_escape(name) for name in names]],
                    corr_rows,
                    empty_message="No correlation matrix available.",
                ),
            ),
            report_html.section(
                _("Goodness of fit"),
                report_html.table(
                    ["Measure", "Value"],
                    metric_rows,
                    empty_message="No metrics available.",
                ),
            ),
        )

    @staticmethod
    def _std_text(std: np.ndarray, row: int) -> str:
        """Return the standard error of one parameter, blank when unknown.

        A fixed parameter, or one the optimiser could not resolve, has no
        meaningful error; printing ``nan`` there would look like a failure
        rather than like an absence.
        """
        if row >= std.size or not np.isfinite(std[row]):
            return ""
        return f"{float(std[row]):.6g}"

    def _format_metrics(self, metrics: dict[str, float]) -> str:
        labels = {"r2": "R^2", "rmse": "RMSE", "ss_res": "SS_RES", "aic": "AIC", "bic": "BIC"}
        parts = []
        for key in ("r2", "rmse", "ss_res", "aic", "bic"):
            value = metrics.get(key)
            if value is not None and np.isfinite(value):
                parts.append(f"{labels[key]}={value:.6g}")
        return ", ".join(parts)

    def _default_output_table_name(self) -> str:
        table = self._current_table() or "table"
        target = self._source_y_col or "target"
        model = self._model_name() or "fit"
        return generated_table_name(f"Fit_{table}_{target}_{model}", fallback="Fit_Result")

    def _refresh_default_output_name(self, force: bool = False) -> None:
        if force or not self._output_table_edit.text().strip():
            self._output_table_edit.setText(self._default_output_table_name())

    def _output_table_name(self) -> str:
        """Return the output table name, prefixed however the user typed it.

        The field is editable, so the prefix is applied here rather than only
        to the default: a hand-typed name is still a generated table.
        """
        return generated_table_name(
            self._output_table_edit.text().strip() or self._default_output_table_name(),
            fallback="Fit_Result",
        )

    # ------------------------------------------------------------------
    # SeriesOperationDialogBase hooks
    # ------------------------------------------------------------------

    def refresh_results(self) -> None:
        """Re-show the last outcome after a selection or parameter change."""
        if self._last_result is not None:
            self._fill_results_table(self._last_result.params, self._last_result.param_std)
            self.publish_results(self.format_results([self._last_result]))

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_fit": True, "fit_dialog": "series_fit"}

    def compute_results(self) -> Sequence[SeriesFitResult]:
        """Evaluate the model at the parameters currently in the table.

        Preview and Apply both come through here, and neither optimises: that
        is what the Fit button is for.  Preview used to re-fit, which meant a
        hand-edited starting guess could never be seen - the optimiser
        overwrote it before anything was drawn.
        """
        self._evaluate(optimise=False)
        return [self._last_result] if self._last_result is not None else []

    def result_to_frame(self, result: SeriesFitResult) -> pd.DataFrame:
        return result.frame

    def result_table_name(self, axis_id: int, result: SeriesFitResult) -> str:
        return self._output_table_name()

    def result_series_spec(self, axis_id: int, table_name: str, result: SeriesFitResult) -> ResultSeriesSpec:
        del axis_id
        sql_query = f'SELECT x, y_fit AS y FROM "{table_name}" ORDER BY x'
        roles = {"x": "x", "y": "y"}
        return ResultSeriesSpec(
            name=f"Fit: {result.source_table} [{result.model_name}]",
            sql_query=sql_query,
            roles=roles,
            style={
                "linestyle": "--",
                "linewidth": 2.0,
                "marker": "",
                "source_series": result.source_table,
                "source_x_col": result.x_col,
                "source_y_col": result.target_col,
                "fit_model": result.model_name,
                "fit_mode": "1D",
                "generated_fit": True,
                "fit_dialog": "series_fit",
            },
        )

    def format_results(self, results: Sequence[SeriesFitResult]) -> str:
        """Return the fit report as an HTML table."""
        if not results:
            return "<p>Run Fit, or press Preview to draw the current parameters.</p>"
        return self._results_html(results[0])

    def on_use_fit_results_as_initial(self) -> None:
        """Copy latest optimized fit parameters back to the Initial column."""
        if self._last_result is None:
            show_message(self, "series.no_fit_results")
            return
        params = np.asarray(self._last_result.params, dtype=float)
        if params.size == 0:
            show_message(self, "series.no_parameters")
            return
        self._spin_params.blockSignals(True)
        try:
            self._spin_params.setValue(int(params.size))
            self._ensure_params_rows(int(params.size))
        finally:
            self._spin_params.blockSignals(False)
        for row, value in enumerate(params):
            self._params_table.setItem(row, 0, QTableWidgetItem(self._param_label(row)))
            self._params_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.12g}"))
            if self._params_table.item(row, 2) is None:
                self._params_table.setItem(row, 2, QTableWidgetItem("-inf"))
            if self._params_table.item(row, 3) is None:
                self._params_table.setItem(row, 3, QTableWidgetItem("inf"))
            self._get_fix_checkbox(row).setChecked(False)
        self._params_table.resizeColumnsToContents()
        applogger.info("Initial parameters updated from latest fit results.")

    def _cell_text(self, row: int, col: int, default: str = "") -> str:
        item = self._params_table.item(row, col)
        if item is None:
            return default
        return item.text()

    def _params_as_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for row in range(self._params_table.rowCount()):
            fix_widget = self._params_table.cellWidget(row, 4)
            fixed = bool(fix_widget.isChecked()) if isinstance(fix_widget, QCheckBox) else False
            rows.append(
                {
                    "parameter": self._cell_text(row, 0, f"p[{row}]"),
                    "initial": self._cell_text(row, 1),
                    "lower": self._cell_text(row, 2),
                    "upper": self._cell_text(row, 3),
                    "fixed": fixed,
                }
            )
        return pd.DataFrame(rows)

    def _copy_params_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self._params_as_frame().to_csv(index=False, sep="\t"))
        applogger.info("Parameter table copied to clipboard.")

    def _save_params_as_csv(self) -> None:
        default_name = f"{self._model_name().replace(' ', '_').lower()}_parameters.csv"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            _("Save parameters as CSV"),
            default_name,
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        self._params_as_frame().to_csv(path, index=False)
        applogger.info("Parameter table saved to %s", path)

    def _show_params_context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self._params_table)
        copy_action = QAction(_("Copy parameters"), menu)
        copy_action.triggered.connect(self._copy_params_to_clipboard)
        menu.addAction(copy_action)

        use_fit_action = QAction(_("Use latest fitted parameters"), menu)
        use_fit_action.setEnabled(self._last_result is not None)
        use_fit_action.triggered.connect(self.on_use_fit_results_as_initial)
        menu.addAction(use_fit_action)

        reset_action = QAction(_("Reset parameters"), menu)
        reset_action.triggered.connect(self.on_reset_params)
        menu.addAction(reset_action)

        menu.addSeparator()
        save_csv_action = QAction(_("Save as CSV"), menu)
        save_csv_action.triggered.connect(self._save_params_as_csv)
        menu.addAction(save_csv_action)

        menu.exec(self._params_table.viewport().mapToGlobal(pos))

    def on_reset_params(self) -> None:
        self._populate_params_defaults()
        applogger.info("Parameter table reset to model defaults.")
