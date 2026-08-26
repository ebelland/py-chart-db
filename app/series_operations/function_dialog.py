"""Plot scanned 1D functions and 3D surface functions without fitting.

One-input functions generate ordinary XY series. Two-input surface functions
(such as those in app.functions.functions_3d) are evaluated on an X/Y mesh and
stored as an XYZ series suitable for a 3D surface plot.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.scanners.functions_scanner import FunctionScanner
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.series_operations.parameter_spec import ChoiceParam, FloatParam, IntParam
from app.styles.style import create_doc_link, mark_editor_panel, set_doc_link
from app.utils import report_html
from app.utils.i18n import _

SPACING_LINEAR = "linear"
SPACING_LOG = "log"


@dataclass(slots=True)
class FunctionResult:
    """One evaluated curve or surface."""

    function_name: str
    result_name: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None
    params: dict[str, float]
    expression: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_surface(self) -> bool:
        return self.z is not None

    def to_frame(self) -> pd.DataFrame:
        if self.z is None:
            return pd.DataFrame({"x": self.x.reshape(-1), "y": self.y.reshape(-1)})
        broadcast = np.broadcast_arrays(self.x, self.y, self.z)
        x_values = np.asarray(broadcast[0], dtype=float)
        y_values = np.asarray(broadcast[1], dtype=float)
        z_values = np.asarray(broadcast[2], dtype=float)
        return pd.DataFrame(
            {
                "x": x_values.reshape(-1),
                "y": y_values.reshape(-1),
                "z": z_values.reshape(-1),
            }
        )


class SeriesFunctionDialog(SeriesOperationDialogBase):
    """Evaluate scanned curve and surface functions and add them to a chart."""

    Name: str = "Function"
    Description = "Plot a function"
    INPUT_MINIMUM_POINTS = 0
    SHOWS_SERIES_SELECTOR = False
    SHOWS_AXIS_SERIES_PAGE = False

    PARAMS = (
        FloatParam(
            "start", "From x:", tooltip="Start of the X range.",
            default_value=0.0, minimum=-1.0e12, maximum=1.0e12,
            decimals=6, step=1.0,
        ),
        FloatParam(
            "stop", "To x:", tooltip="End of the X range.",
            default_value=10.0, minimum=-1.0e12, maximum=1.0e12,
            decimals=6, step=1.0,
        ),
        IntParam(
            "points", "X points:", tooltip="Number of X samples.",
            default_value=200, minimum=2, maximum=1_000_000, step=50,
        ),
        FloatParam(
            "y_start", "From y:", tooltip="Start of the Y range for a surface.",
            default_value=0.0, minimum=-1.0e12, maximum=1.0e12,
            decimals=6, step=1.0,
        ),
        FloatParam(
            "y_stop", "To y:", tooltip="End of the Y range for a surface.",
            default_value=10.0, minimum=-1.0e12, maximum=1.0e12,
            decimals=6, step=1.0,
        ),
        IntParam(
            "y_points", "Y points:", tooltip="Number of Y samples for a surface.",
            default_value=100, minimum=2, maximum=10_000, step=10,
        ),
        SeriesOperationDialogBase.destination_param(
            tooltip="Choose where the generated curve or surface is drawn.",
            default=SeriesOperationDialogBase.DEST_SAME_AXIS,
        ),
        ChoiceParam(
            "spacing", "Spacing:",
            tooltip="Linear or logarithmic spacing for both generated axes.",
            choices=(("Linear", SPACING_LINEAR), ("Logarithmic", SPACING_LOG)),
        ),
    )

    Icon = """
    <path d="M4 20c4 0 4-16 8-16"/>
    <path d="M12 4c4 0 4 16 8 16"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            raise ValueError("SeriesFunctionDialog requires a repository instance.")
        self._last_results: list[FunctionResult] = []
        self._parameter_form: QFormLayout | None = None
        self._scanner = FunctionScanner()
        self._selected_function: dict[str, Any] = {}
        self._result_axis_id: int | None = None
        self._result_figure_id: int | None = None
        self._applied = False
        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Plot Function",
            parent=parent,
            width=860,
            height=700,
        )
        self.axis_series_panel.setVisible(False)
        self._reload_function_tree()
        self.refresh_results()

    def init_operation_widgets(self) -> None:
        self._doc_link = create_doc_link(self)
        self._function_tree = QTreeWidget(self)
        self._params_table = QTableWidget(0, 2, self)
        self._expression_label = QLabel("", self)
        self._parameter_form = None

    def build_model_selector(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.model_combo.setVisible(False)
        self._function_tree.setHeaderLabels([_("Function"), _("Description")])
        self._function_tree.setMinimumHeight(200)
        self._function_tree.currentItemChanged.connect(self._on_function_changed)
        mark_editor_panel(self._function_tree)
        layout.addWidget(self._function_tree, 1)
        self._expression_label.setWordWrap(True)
        self._expression_label.setTextFormat(Qt.TextFormat.RichText)
        self._expression_label.setProperty("muted", True)
        layout.addWidget(self._expression_label)
        doc_container = QWidget(panel)
        form = QFormLayout(doc_container)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(_("Docs:"), self._doc_link)
        layout.addWidget(doc_container)
        return panel

    def build_parameter_selector(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(super().build_parameter_selector(), 0)
        self._params_table.setHorizontalHeaderLabels([_("Parameter"), _("Value")])
        self._params_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._params_table.verticalHeader().setVisible(False)
        self._params_table.setMinimumHeight(140)
        self._params_table.itemChanged.connect(lambda *_args: self.refresh_results())
        mark_editor_panel(self._params_table)
        layout.addWidget(self._params_table, 1)
        return widget

    def connect_operation_signals(self) -> None:
        return None

    def _reload_function_tree(self) -> None:
        self._function_tree.blockSignals(True)
        try:
            self._function_tree.clear()
            for category, functions in self._scanner.catalog().items():
                parent = QTreeWidgetItem([category, ""])
                parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                for payload in functions:
                    child = QTreeWidgetItem(
                        [str(payload.get("name", "")), str(payload.get("description", ""))]
                    )
                    child.setData(0, Qt.ItemDataRole.UserRole, payload)
                    parent.addChild(child)
                self._function_tree.addTopLevelItem(parent)
            self._function_tree.expandAll()
            self._function_tree.resizeColumnToContents(0)
        finally:
            self._function_tree.blockSignals(False)
        first = self._first_function_item()
        if first is not None:
            self._function_tree.setCurrentItem(first)

    def _first_function_item(self) -> QTreeWidgetItem | None:
        for index in range(self._function_tree.topLevelItemCount()):
            parent = self._function_tree.topLevelItem(index)
            if parent is not None and parent.childCount():
                return parent.child(0)
        return None

    @staticmethod
    def _payload_is_surface(payload: Mapping[str, Any]) -> bool:
        """Recognize a two-input surface without requiring one scanner version."""
        dimension = payload.get("dimensions", payload.get("input_dimensions", 1))
        try:
            if int(dimension) >= 2:
                return True
        except (TypeError, ValueError):
            pass
        category = str(payload.get("category", "")).strip().lower()
        module = str(payload.get("module", payload.get("module_name", ""))).lower()
        return category.startswith("3d ") or "functions_3d" in module

    def _on_function_changed(self, current: Any, _previous: Any = None) -> None:
        payload = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else None
        if not isinstance(payload, dict):
            return
        self._selected_function = payload
        self._rebuild_params_table(payload)
        self._expression_label.setText(str(payload.get("expression", "") or ""))
        set_doc_link(
            self._doc_link,
            str(payload.get("name", "")),
            str(payload.get("doc_url", "") or ""),
        )
        self.refresh_results()

    def _rebuild_params_table(self, payload: Mapping[str, Any]) -> None:
        names = [str(name) for name in (payload.get("params") or [])]
        defaults = [float(value) for value in (payload.get("p0") or [])]
        defaults += [1.0] * max(0, len(names) - len(defaults))
        self._params_table.blockSignals(True)
        try:
            self._params_table.setRowCount(len(names))
            for row, name in enumerate(names):
                label = QTableWidgetItem(name)
                label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._params_table.setItem(row, 0, label)
                self._params_table.setItem(row, 1, QTableWidgetItem(f"{defaults[row]:g}"))
        finally:
            self._params_table.blockSignals(False)

    def _function_params(self) -> tuple[list[str], np.ndarray]:
        names: list[str] = []
        values: list[float] = []
        for row in range(self._params_table.rowCount()):
            label = self._params_table.item(row, 0)
            cell = self._params_table.item(row, 1)
            names.append(label.text() if label else f"p{row}")
            try:
                values.append(float(cell.text()) if cell else 0.0)
            except ValueError:
                values.append(0.0)
        return names, np.asarray(values, dtype=float)

    def refresh_results(self) -> None:
        try:
            results = self.compute_results()
        except Exception as exc:
            self._last_results = []
            self.set_results_text(f"Error:\n{exc}")
            return
        self._last_results = list(results)
        self.set_results_text(self.format_results(results) if results else _("Select a function."))

    @staticmethod
    def _build_range(
        start: float,
        stop: float,
        count: int,
        spacing: str,
        axis_name: str,
    ) -> np.ndarray:
        if start == stop:
            raise ValueError(f"the {axis_name} range is empty")
        if spacing == SPACING_LOG:
            if start <= 0.0 or stop <= 0.0:
                raise ValueError(f"logarithmic {axis_name} spacing requires positive limits")
            return np.logspace(np.log10(start), np.log10(stop), max(2, count))
        return np.linspace(start, stop, max(2, count))

    def compute_results(self) -> list[FunctionResult]:
        payload = self._selected_function
        if not payload:
            return []
        settings = self.parameter_values()
        spacing = str(settings.get("spacing", SPACING_LINEAR))
        x_values = self._build_range(
            float(settings.get("start", 0.0)),
            float(settings.get("stop", 10.0)),
            int(settings.get("points", 200)),
            spacing,
            "X",
        )
        names, values = self._function_params()
        model = self._scanner.make_model(dict(payload))
        is_surface = self._payload_is_surface(payload)

        if is_surface:
            y_axis = self._build_range(
                float(settings.get("y_start", 0.0)),
                float(settings.get("y_stop", 10.0)),
                int(settings.get("y_points", 100)),
                spacing,
                "Y",
            )
            x_mesh, y_mesh = np.meshgrid(
                x_values,
                y_axis,
                indexing="xy",
            )

            xy_values = np.column_stack(
                (
                    x_mesh.reshape(-1),
                    y_mesh.reshape(-1),
                )
            )

            z_values = np.asarray(
                model(xy_values, values),
                dtype=float,
            ).reshape(x_mesh.shape)

            if z_values.shape != x_mesh.shape:
                raise ValueError(
                    f"the surface returned shape {z_values.shape}; "
                    f"expected {x_mesh.shape}"
                )
            finite = int(np.count_nonzero(np.isfinite(z_values)))
            if finite == 0:
                raise ValueError("the surface is undefined everywhere in this X/Y range")
            x_out, y_out, z_out = x_mesh, y_mesh, z_values
            metadata: dict[str, Any] = {
                "dimension": 3,
                "spacing": spacing,
                "x_range": f"{x_values[0]:g} .. {x_values[-1]:g}",
                "y_range": f"{y_axis[0]:g} .. {y_axis[-1]:g}",
                "grid": f"{x_values.size} x {y_axis.size}",
            }
            if finite < z_values.size:
                metadata["undefined"] = z_values.size - finite
        else:
            y_values = np.asarray(model(x_values, values), dtype=float)
            if y_values.shape != x_values.shape:
                raise ValueError(
                    f"the function returned {y_values.size} values for {x_values.size} inputs"
                )
            finite = int(np.count_nonzero(np.isfinite(y_values)))
            if finite == 0:
                raise ValueError("the function is undefined everywhere in this range")
            x_out, y_out, z_out = x_values, y_values, None
            metadata = {
                "dimension": 2,
                "spacing": spacing,
                "range": f"{x_values[0]:g} .. {x_values[-1]:g}",
            }
            if finite < y_values.size:
                metadata["undefined"] = y_values.size - finite

        name = str(payload.get("name", "function"))
        return [
            FunctionResult(
                function_name=name,
                result_name=name,
                x=x_out,
                y=y_out,
                z=z_out,
                params=dict(zip(names, (float(value) for value in values))),
                expression=str(payload.get("expression", "") or ""),
                metadata=metadata,
            )
        ]

    def resolve_target_axis_id(
        self,
        selected_axis_id: int,
        results: Sequence[Any],
    ) -> int:
        result = results[0] if results else None
        is_surface = isinstance(result, FunctionResult) and result.is_surface
        name = result.function_name if isinstance(result, FunctionResult) else "Function"
        return self.resolve_destination_axis(
            selected_axis_id,
            chart_type="Surface Plot" if is_surface else "Scatter Plot",
            title=name,
            figure_name=name,
            options=(
                {"grid": True, "surface": True, "cmap": "viridis"}
                if is_surface
                else {"grid": True, "linestyle": "-", "marker": ""}
            ),
        )

    def discard_operation_artifacts(self) -> None:
        self.discard_result_target()

    def selected_series(self) -> list[Any]:
        return []

    def apply(self) -> bool:
        applied = super().apply()
        self._applied = self._applied or applied
        return applied

    def result_to_frame(self, result: FunctionResult) -> pd.DataFrame:
        return result.to_frame()

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: FunctionResult,
    ) -> ResultSeriesSpec:
        del axis_id
        if result.is_surface:
            query = f'SELECT x, y, z FROM "{table_name}" ORDER BY y, x'
            roles = {"x": "x", "y": "y", "z": "z"}
            style: dict[str, Any] = {
                "generated_function": True,
                "function_dialog": "series_function",
                "function": result.function_name,
                "surface": True,
                "cmap": "viridis",
            }
        else:
            query = f'SELECT x, y FROM "{table_name}" ORDER BY x'
            roles = {"x": "x", "y": "y"}
            style = {
                "generated_function": True,
                "function_dialog": "series_function",
                "function": result.function_name,
                "linestyle": "-",
                "linewidth": 1.6,
                "marker": "",
            }
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=query,
            roles=roles,
            style=style,
        )

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_function": True, "function_dialog": "series_function"}

    def result_table_name(self, axis_id: int, result: FunctionResult) -> str:
        suffix = "surface" if result.is_surface else "curve"
        return generated_table_name(
            f"Function_axis{axis_id}_{result.function_name}_{suffix}",
            fallback="Function_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Function"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[FunctionResult]) -> str:
        if not results:
            return report_html.note(_("No results."))
        sections: list[str] = []
        for result in results:
            values = result.z if result.z is not None else result.y
            finite = np.isfinite(values)
            summary_rows: list[tuple[str, Any]] = [
                (_("Function"), result.function_name),
                (_("Plot type"), _("3D surface") if result.is_surface else _("2D curve")),
                (_("Points"), values.size),
                (_("Spacing"), result.metadata.get("spacing", "")),
            ]
            if result.is_surface:
                summary_rows.extend(
                    [
                        (_("X range"), result.metadata.get("x_range", "")),
                        (_("Y range"), result.metadata.get("y_range", "")),
                        (_("Grid"), result.metadata.get("grid", "")),
                    ]
                )
            else:
                summary_rows.append((_("Range"), result.metadata.get("range", "")))
            if finite.any():
                label = _("Z range") if result.is_surface else _("Y range")
                summary_rows.append(
                    (
                        label,
                        f"{report_html.format_number(float(np.min(values[finite])))} .. "
                        f"{report_html.format_number(float(np.max(values[finite])))}",
                    )
                )
            if "undefined" in result.metadata:
                summary_rows.append((_("Undefined points"), result.metadata["undefined"]))
            blocks = [report_html.summary_table(summary_rows)]
            if result.expression:
                blocks.append(report_html.raw_note(result.expression))
            if result.params:
                blocks.append(
                    report_html.table(
                        (_("Parameter"), _("Value")),
                        [(name, report_html.format_number(value)) for name, value in result.params.items()],
                        align=("left", "right"),
                        empty_message=_("This function takes no parameters."),
                    )
                )
            sections.append(report_html.section(result.function_name, *blocks))
        return report_html.document(_("Function"), "", *sections)
