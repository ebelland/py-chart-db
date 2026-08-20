"""Compact chart-series series  interpolation dialog.

Features:
- Select one chart axis and one or more series.
- Fit/interpolate selected series with NumPy/SciPy models.
- Dynamic settings visibility for compact PySide6 strict UI.
- Evaluate generated Y values at multiple X spacing modes/custom X values.
- Apply generated series to the selected axis without closing the dialog.
- Re-apply removes previously generated series by this dialog and replaces them.
- Save generated datapoints as a normal SQLite table.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from app.data.data_source import parse_roles
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.series_operation_dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.logs.logger import applogger
from app.utils.i18n import _

from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
    UnivariateSpline,
    make_interp_spline,
)
from scipy.optimize import curve_fit


MODEL_POLYNOMIAL: Final[str] = "Polynomial"
MODEL_LINEAR: Final[str] = "Linear"
MODEL_EXPONENTIAL: Final[str] = "Exponential"
MODEL_LOGARITHMIC: Final[str] = "Logarithmic"
MODEL_POWER: Final[str] = "Power"
MODEL_GAUSSIAN: Final[str] = "Gaussian"
MODEL_SIGMOID: Final[str] = "Sigmoid"
MODEL_NUMPY_INTERP: Final[str] = "NumPy interp"
MODEL_SCIPY_PCHIP: Final[str] = "SciPy PCHIP"
MODEL_SCIPY_CUBIC: Final[str] = "SciPy CubicSpline"
MODEL_SCIPY_SPLINE: Final[str] = "SciPy spline family"
MODEL_SCIPY_AKIMA: Final[str] = "SciPy Akima"

MODEL_NAMES: Final[tuple[str, ...]] = (
    MODEL_POLYNOMIAL,
    MODEL_LINEAR,
    MODEL_EXPONENTIAL,
    MODEL_LOGARITHMIC,
    MODEL_POWER,
    MODEL_GAUSSIAN,
    MODEL_SIGMOID,
    MODEL_NUMPY_INTERP,
    MODEL_SCIPY_PCHIP,
    MODEL_SCIPY_AKIMA,
    MODEL_SCIPY_CUBIC,
    MODEL_SCIPY_SPLINE,
)

INTERPOLATION_MODELS: Final[set[str]] = {
    MODEL_NUMPY_INTERP,
    MODEL_SCIPY_PCHIP,
    MODEL_SCIPY_AKIMA,
    MODEL_SCIPY_CUBIC,
    MODEL_SCIPY_SPLINE,
}

CURVE_FIT_MODELS: Final[set[str]] = {
    MODEL_EXPONENTIAL,
    MODEL_LOGARITHMIC,
    MODEL_POWER,
    MODEL_GAUSSIAN,
    MODEL_SIGMOID,
}

MODEL_DOCS: Final[dict[str, tuple[str, str]]] = {
    MODEL_POLYNOMIAL: (
        "NumPy polyfit",
        "https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html",
    ),
    MODEL_LINEAR: (
        "NumPy polyfit",
        "https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html",
    ),
    MODEL_NUMPY_INTERP: (
        "NumPy interp",
        "https://numpy.org/doc/stable/reference/generated/numpy.interp.html",
    ),
    MODEL_SCIPY_PCHIP: (
        "SciPy PchipInterpolator",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html",
    ),
    MODEL_SCIPY_AKIMA: (
        "SciPy Akima1DInterpolator",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html",
    ),
    MODEL_SCIPY_CUBIC: (
        "SciPy CubicSpline",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html",
    ),
    MODEL_SCIPY_SPLINE: (
        "SciPy interpolation",
        "https://docs.scipy.org/doc/scipy/tutorial/interpolate.html",
    ),
    MODEL_EXPONENTIAL: (
        "SciPy curve_fit",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html",
    ),
    MODEL_LOGARITHMIC: (
        "SciPy curve_fit",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html",
    ),
    MODEL_POWER: (
        "SciPy curve_fit",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html",
    ),
    MODEL_GAUSSIAN: (
        "SciPy curve_fit",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html",
    ),
    MODEL_SIGMOID: (
        "SciPy curve_fit",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html",
    ),
}

_TABLE_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(slots=True)
class SeriesChoice:
    """Selectable chart series descriptor."""

    series_id: int
    series_index: int
    name: str
    sql_query: str
    roles: dict[str, Any]

    def label(self) -> str:
        name = self.name.strip() or f"Series {self.series_index}"
        return f"{self.series_index}: {name}"


@dataclass(slots=True)
class FitResult:
    """interpolation output for one source series."""

    source: SeriesChoice
    model: str
    table_name: str
    output_name: str
    x_eval: np.ndarray
    y_eval: np.ndarray
    params: dict[str, float]
    metrics: dict[str, float]
    message: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source_series_id": self.source.series_id,
                "source_series_name": self.source.name,
                "model": self.model,
                "x": self.x_eval,
                "y": self.y_eval,
            }
        )


class SeriesInterpolateDialog(SeriesOperationDialogBase):
    """Compact series fitting/interpolation dialog for one chart panel."""
    Name: str  = "Interpolation"
    Description = "Fill missing values"

    Icon = """
    <path d="M4 18.5h16"/>
    <path d="M4.5 18V5"/>
    <path d="M6.5 15.5l4.2-4.2 3.2 2.4 4.2-6.2"/>
    <circle cx="6.5" cy="15.5" r="1"/>
    <circle cx="10.7" cy="11.3" r="1"/>
    <circle cx="13.9" cy="13.7" r="1"/>
    <circle cx="18.1" cy="7.5" r="1"/>
    """
    applied = Signal()

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QMainWindow,
    ) -> None:
        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Interpolation",
            parent=parent,
            width=720,
            height=640
        )
        self._refresh_model_defaults()
        self.model_combo.setVisible(True)

    def init_operation_widgets(self) -> None:
        """Create interpolation controls before base builder hooks run."""

        self._series_choices: list[SeriesChoice] = []
        self._last_results: list[FitResult] = []
        self._settings_form: QFormLayout | None = None

        self._doc_link = QLabel(self)
        self._degree_spin = QSpinBox(self)
        self._points_spin = QSpinBox(self)
        self._spacing_combo = QComboBox(self)
        self._range_edit = QLineEdit(self)
        self._integer_step_spin = QDoubleSpinBox(self)
        self._custom_x_edit = QLineEdit(self)
        self._extend_spin = QDoubleSpinBox(self)
        self._extrap_check = QCheckBox(_("Allow extrapolation"), self)
        self._spline_type_combo = QComboBox(self)
        self._spline_degree_spin = QSpinBox(self)
        self._cubic_bc_combo = QComboBox(self)
        self._smoothing_spin = QDoubleSpinBox(self)
        self._guess_check = QCheckBox(_("Guess starting parameters"), self)
        self._params_label = QLabel(_("Start params:"), self)
        self._params_edit = QPlainTextEdit(self)
        #self.series_selector.reload(select_all_series=False)


    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build_model_selector(self) -> QWidget:
        panel = QWidget(self)
        form = QFormLayout(panel)
        form.setContentsMargins(4, 4, 4, 4)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.model_combo.addItems(MODEL_NAMES)
        form.addRow(_("Model:"), self.model_combo)
        self._doc_link.setOpenExternalLinks(True)
        self._doc_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        form.addRow(_("Docs:"), self._doc_link)
        return panel

    def build_parameter_selector(self) -> QWidget:
        settings = QWidget(self)
        form = QFormLayout(settings)
        self._settings_form = form
        form.setContentsMargins(4, 4, 4, 4)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._degree_spin.setRange(1, 12)
        self._degree_spin.setValue(2)
        form.addRow(_("Degree:"), self._degree_spin)

        self._points_spin.setRange(10, 200_000)
        self._points_spin.setValue(600)
        form.addRow(_("Points:"), self._points_spin)

        self._spacing_combo.addItems([
            "linspace",
            "logspace",
            "geomspace",
            "original data X",
            "integer step",
            "chebyshev nodes",
            "custom X values",
        ])
        form.addRow(_("X spacing:"), self._spacing_combo)

        self._range_edit.setPlaceholderText(_("auto, or start, stop"))
        form.addRow(_("X range:"), self._range_edit)

        self._integer_step_spin.setRange(1e-12, 1e12)
        self._integer_step_spin.setDecimals(6)
        self._integer_step_spin.setValue(1.0)
        form.addRow(_("Step:"), self._integer_step_spin)

        self._custom_x_edit.setPlaceholderText(_("1, 2.5, 10 or one per line"))
        form.addRow(_("Eval X:"), self._custom_x_edit)

        self._extend_spin.setRange(0.0, 500.0)
        self._extend_spin.setDecimals(1)
        self._extend_spin.setSuffix(" %")
        form.addRow(_("Extend:"), self._extend_spin)

        self._extrap_check.setChecked(True)
        form.addRow("", self._extrap_check)

        self._spline_type_combo.addItems([
            "UnivariateSpline",
            "CubicSpline",
            "PCHIP",
            "Akima1D",
            "B-spline",
        ])
        form.addRow(_("Spline:"), self._spline_type_combo)

        self._spline_degree_spin.setRange(1, 5)
        self._spline_degree_spin.setValue(3)
        form.addRow(_("Spline k:"), self._spline_degree_spin)

        self._cubic_bc_combo.addItems(["not-a-knot", "natural", "clamped", "periodic"])
        form.addRow(_("Boundary:"), self._cubic_bc_combo)

        self._smoothing_spin.setRange(0.0, 1_000_000.0)
        self._smoothing_spin.setDecimals(3)
        form.addRow(_("Smooth s:"), self._smoothing_spin)

        self._guess_check.setChecked(True)
        form.addRow("", self._guess_check)

        self._params_edit.setMaximumHeight(64)
        self._params_edit.setPlaceholderText(_('Example: {"a": 1.0, "b": 0.1}'))
        form.addRow(self._params_label, self._params_edit)

        for widget in (
            self._range_edit,
            self._custom_x_edit,
            self._spacing_combo,
            self._spline_type_combo,
            self._cubic_bc_combo,
            self._params_edit,
        ):
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        scroll = QScrollArea(self)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings)
        self._set_setting_tooltips()
        return scroll

    def connect_operation_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_model_defaults)
        self._degree_spin.valueChanged.connect(self._refresh_model_defaults)
        self._points_spin.valueChanged.connect(self.refresh_results)
        self._spacing_combo.currentIndexChanged.connect(self._refresh_model_defaults)
        self._range_edit.textChanged.connect(self.refresh_results)
        self._integer_step_spin.valueChanged.connect(self.refresh_results)
        self._custom_x_edit.textChanged.connect(self.refresh_results)
        self._extend_spin.valueChanged.connect(self.refresh_results)
        self._extrap_check.stateChanged.connect(self.refresh_results)
        self._spline_type_combo.currentIndexChanged.connect(self._refresh_model_defaults)
        self._spline_degree_spin.valueChanged.connect(self.refresh_results)
        self._cubic_bc_combo.currentIndexChanged.connect(self.refresh_results)
        self._smoothing_spin.valueChanged.connect(self.refresh_results)
        self._guess_check.stateChanged.connect(self._refresh_model_defaults)
        self._params_edit.textChanged.connect(self.refresh_results)

    def _set_setting_tooltips(self) -> None:
        self.model_combo.setToolTip(_("Choose the fitting/interpolation model."))
        self._doc_link.setToolTip(_("Open NumPy/SciPy documentation for the model."))
        self._degree_spin.setToolTip(_("Polynomial degree for NumPy polyfit."))
        self._points_spin.setToolTip(_("Number of generated points for continuous spacing modes."))
        self._spacing_combo.setToolTip(_("Choose how generated X values are built."))
        self._range_edit.setToolTip(_("Optional range as 'start, stop'. Empty = data range plus Extend."))
        self._integer_step_spin.setToolTip(_("Step size used by integer/fixed-step spacing."))
        self._custom_x_edit.setToolTip(_("Explicit X values where Y is evaluated."))
        self._extend_spin.setToolTip(_("Extend automatic data range by this percent on both sides."))
        self._extrap_check.setToolTip(_("Allow evaluation outside source data X range."))
        self._spline_type_combo.setToolTip(_("Spline algorithm for SciPy spline family."))
        self._spline_degree_spin.setToolTip(_("Spline degree k for UnivariateSpline/B-spline."))
        self._cubic_bc_combo.setToolTip(_("Spline boundary condition."))
        self._smoothing_spin.setToolTip(_("UnivariateSpline smoothing factor s."))
        self._guess_check.setToolTip(_("Guess starting parameters for nonlinear models."))
        self._params_edit.setToolTip(_("JSON starting parameters for nonlinear series_fit models."))

    def _set_form_row_visible(self, field: QWidget, visible: bool) -> None:
        field.setVisible(visible)
        form = self._settings_form
        if form is None:
            return
        label = form.labelForField(field)
        if label is not None:
            label.setVisible(visible)

    # ------------------------------------------------------------------
    # Descriptor loading
    # ------------------------------------------------------------------



    @staticmethod
    def _series_label_from_row(row: Any) -> str:
        keys = row.keys() if hasattr(row, "keys") else []
        index = row["series_index"] if "series_index" in keys else ""
        name = row["name"] if "name" in keys else f"Series {index}"
        return f"{index}: {name}" if index != "" else str(name)

    def _series_choice_from_row(self, row: Any) -> SeriesChoice:
        series_index = int(row["series_index"])
        return SeriesChoice(
            series_id=int(row["id"]),
            series_index=series_index,
            name=str(row["name"] or f"Series {series_index}"),
            sql_query=str(row["sql_query"]),
            roles=parse_roles(row["roles"]),
        )

    # ------------------------------------------------------------------
    # Dynamic settings
    # ------------------------------------------------------------------

    def _model_name(self) -> str:
        return str(self.model_combo.currentText())

    def _refresh_model_defaults(self) -> None:
        model = self._model_name()
        spacing = self._spacing_combo.currentText()
        spline_type = self._spline_type_combo.currentText()

        uses_interpolation = model in INTERPOLATION_MODELS
        uses_poly_degree = model == MODEL_POLYNOMIAL
        uses_spline_menu = model == MODEL_SCIPY_SPLINE
        uses_spline_k = uses_spline_menu and spline_type in {
            "UnivariateSpline",
            "B-spline",
        }
        uses_smoothing = uses_spline_menu and spline_type == "UnivariateSpline"
        uses_boundary = model == MODEL_SCIPY_CUBIC or (
            uses_spline_menu and spline_type in {"CubicSpline", "B-spline"}
        )

        uses_custom_x = spacing == "custom X values"
        uses_original_x = spacing == "original data X"
        uses_step = spacing == "integer step"
        uses_points = spacing not in {
            "custom X values",
            "original data X",
            "integer step",
        }
        uses_range = spacing not in {"custom X values", "original data X"}
        uses_extrap = uses_range or model in {
            MODEL_SCIPY_PCHIP,
            MODEL_SCIPY_AKIMA,
            MODEL_SCIPY_CUBIC,
            MODEL_SCIPY_SPLINE,
        }

        self._set_form_row_visible(self._degree_spin, uses_poly_degree)
        self._set_form_row_visible(self._points_spin, uses_points)
        self._set_form_row_visible(self._range_edit, uses_range)
        self._set_form_row_visible(self._integer_step_spin, uses_step)
        self._set_form_row_visible(self._custom_x_edit, uses_custom_x)
        self._set_form_row_visible(self._extend_spin, uses_range)
        self._set_form_row_visible(self._extrap_check, uses_extrap)
        self._set_form_row_visible(self._spline_type_combo, uses_spline_menu)
        self._set_form_row_visible(self._spline_degree_spin, uses_spline_k)
        self._set_form_row_visible(self._cubic_bc_combo, uses_boundary)
        self._set_form_row_visible(self._smoothing_spin, uses_smoothing)
        self._set_form_row_visible(self._guess_check, uses_interpolation)
        self._set_form_row_visible(self._params_edit, uses_interpolation)
        self._params_label.setVisible(uses_interpolation)

        title, url = MODEL_DOCS.get(
            model,
            ("SciPy interpolation", "https://docs.scipy.org/doc/scipy/tutorial/interpolate.html"),
        )
        self.set_doc_link(title, url)

        if uses_interpolation and self._guess_check.isChecked():
            self._params_edit.blockSignals(True)
            self._params_edit.setPlainText(
                json.dumps(self._default_params_for_model(model), indent=2)
            )
            self._params_edit.blockSignals(False)

        # Suppress unused variable warning while keeping logic readable.
        _unused = uses_original_x
        self.refresh_results()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _start_params(self) -> dict[str, float]:
        if self._model_name() not in INTERPOLATION_MODELS:
            return {}
        text = self._params_edit.toPlainText().strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            applogger.error(f"Invalid parameter JSON: {exc}") 
        if not isinstance(value, dict):
            applogger.error("Start parameters must be a JSON object.")
        return {str(k): float(v) for k, v in value.items()}

    @staticmethod
    def _default_params_for_model(model: str) -> dict[str, float]:
        if model == MODEL_EXPONENTIAL:
            return {"a": 1.0, "b": 0.1, "c": 0.0}
        if model == MODEL_LOGARITHMIC:
            return {"a": 1.0, "b": 0.0}
        if model == MODEL_POWER:
            return {"a": 1.0, "b": 1.0, "c": 0.0}
        if model == MODEL_GAUSSIAN:
            return {"a": 1.0, "mu": 0.0, "sigma": 1.0, "c": 0.0}
        if model == MODEL_SIGMOID:
            return {"a": 1.0, "x0": 0.0, "k": 1.0, "c": 0.0}
        return {}

    def refresh_results(self) -> None:
        try:
            results = self.compute_results()
        except Exception as exc:
            self._last_results = []
            self._results_label.setText(f"Error:\n{exc}")
            return
        self._last_results = results
        self._results_label.setText(
            self.format_results(results) if results else "Select one or more source series."
        )

    def compute_results(self) -> list[FitResult]:
        """Build interpolation results for selected SQLite series rows."""
        selected_rows = self.selected_series()
        choices = [self._series_choice_from_row(row) for row in selected_rows]
        return [self._interpolate_one_series(choice) for choice in choices]

    def _interpolate_one_series(self, series: SeriesChoice) -> FitResult:
        df = self._repo.query_df(series.sql_query)
        x_col, y_col = self._xy_columns(df, series.roles)

        clean = self._clean_xy(df, x_col, y_col)
        if clean.shape[0] < 2:
            applogger.error(f"{series.name}: at least two valid points are required.")

        x_data = clean[x_col].to_numpy(dtype=float)
        y_data = clean[y_col].to_numpy(dtype=float)
        x_data, y_data = self._sort_unique_xy(x_data, y_data)

        if x_data.size < 2:
            applogger.error(f"{series.name}: at least two unique valid X values are required.")

        x_eval = self._x_eval(x_data)
        model = self._model_name()

        y_eval, params, message = self._evaluate_model(
            model=model,
            x_data=x_data,
            y_data=y_data,
            x_eval=x_eval,
            start_params=self._start_params(),
        )

        metrics = self._metrics(x_data, y_data, model, params)

        safe_model_name = _TABLE_SAFE_RE.sub(
            "_",
            model.strip().lower(),
        ).strip("_") or "series"

        table_name = generated_table_name(
            f"Interpolation_axis{self.series_selector.selected_axis_id()}"
            f"_series{series.series_id}_{safe_model_name}",
            fallback="Interpolation_Result",
        )

        return FitResult(
            source=series,
            model=model,
            table_name=table_name,
            output_name=f"Interpolate: {series.name} [{model}]",
            x_eval=x_eval,
            y_eval=y_eval,
            params=params,
            metrics=metrics,
            message=message,
        )

    @staticmethod
    def _xy_columns(df: pd.DataFrame, roles: Mapping[str, Any]) -> tuple[str, str]:
        if df.empty:
            applogger.error("Series query returned no rows.")

        columns = [str(col) for col in df.columns]
        x_name = str(roles.get("x", ""))
        y_name = str(roles.get("y", ""))

        if x_name in columns and y_name in columns:
            return x_name, y_name

        numeric = [
            str(col)
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        if len(numeric) < 2:
            applogger.error("Series query must expose at least two numeric columns.")

        return numeric[0], numeric[1]

    @staticmethod
    def _clean_xy(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        clean = df[[x_col, y_col]].copy()
        clean[x_col] = pd.to_numeric(clean[x_col], errors="coerce")
        clean[y_col] = pd.to_numeric(clean[y_col], errors="coerce")
        return clean.replace([np.inf, -np.inf], np.nan).dropna()


    @staticmethod
    def _sort_unique_xy(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(x_data)
        grouped = pd.DataFrame({"x": x_data[order], "y": y_data[order]}).groupby(
            "x",
            as_index=False,
        )["y"].mean()
        return grouped["x"].to_numpy(float), grouped["y"].to_numpy(float)

    def _x_eval(self, x_data: np.ndarray) -> np.ndarray:
        spacing = self._spacing_combo.currentText()
        if spacing == "custom X values":
            values = self._parse_values(self._custom_x_edit.text())
            if values.size == 0:
                applogger.error("Enter at least one Eval X value.")
            return np.sort(values.astype(float))

        if spacing == "original data X":
            return np.sort(np.asarray(x_data, dtype=float))

        start, stop = self._x_range(x_data)
        count = int(self._points_spin.value())
        if spacing == "logspace":
            if start <= 0.0 or stop <= 0.0:
                applogger.error("logspace requires a positive X range.")
            return np.logspace(np.log10(start), np.log10(stop), count)
        if spacing == "geomspace":
            if start <= 0.0 or stop <= 0.0:
                applogger.error("geomspace requires a positive X range.")
            return np.geomspace(start, stop, count)
        if spacing == "integer step":
            step = float(self._integer_step_spin.value())
            first = math.ceil(start / step) * step
            values = np.arange(first, stop + step * 0.5, step, dtype=float)
            if values.size == 0:
                applogger.error("Step spacing produced no X values.")
            return values
        if spacing == "chebyshev nodes":
            k = np.arange(count, dtype=float)
            nodes = np.cos((2.0 * k + 1.0) * np.pi / (2.0 * count))
            scaled = 0.5 * (start + stop) + 0.5 * (stop - start) * nodes
            return np.sort(scaled)
        return np.linspace(start, stop, count)

    def _x_range(self, x_data: np.ndarray) -> tuple[float, float]:
        text = self._range_edit.text().strip()
        if text:
            values = self._parse_values(text)
            if values.size < 2:
                applogger.error("X range must contain start and stop.")
            start = float(values[0])
            stop = float(values[1])
        else:
            x_min = float(np.min(x_data))
            x_max = float(np.max(x_data))
            span = x_max - x_min
            if span <= 0.0:
                applogger.error("X data must contain more than one unique value.")
            pad = span * float(self._extend_spin.value()) / 100.0
            if not self._extrap_check.isChecked():
                pad = 0.0
            start = x_min - pad
            stop = x_max + pad
        if start == stop:
            applogger.error("X range start and stop must differ.")
        if start > stop:
            start, stop = stop, start
        return start, stop

    @staticmethod
    def _parse_values(text: str) -> np.ndarray:
        raw = text.replace(";", ",").replace("\n", ",")
        tokens: list[str] = []
        for chunk in raw.split(","):
            tokens.extend(part for part in chunk.split(" ") if part.strip())
        return np.asarray([float(token) for token in tokens], dtype=float)

    def _evaluate_model(
        self,
        *,
        model: str,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_eval: np.ndarray,
        start_params: dict[str, float],
    ) -> tuple[np.ndarray, dict[str, float], str]:
        if model == MODEL_POLYNOMIAL:
            degree = min(int(self._degree_spin.value()), max(1, x_data.size - 1))
            coeff = np.polyfit(x_data, y_data, degree)
            return (
                np.polyval(coeff, x_eval),
                {f"c{i}": float(v) for i, v in enumerate(coeff)},
                f"NumPy polyfit degree={degree}",
            )
        if model == MODEL_LINEAR:
            coeff = np.polyfit(x_data, y_data, 1)
            return np.polyval(coeff, x_eval), {"m": float(coeff[0]), "b": float(coeff[1])}, "Linear fit"
        if model == MODEL_NUMPY_INTERP:
            return np.interp(x_eval, x_data, y_data), {}, "NumPy interpolation"
        if model == MODEL_SCIPY_PCHIP:
            pchip_cls = PchipInterpolator
            if pchip_cls is None:
                applogger.error("PchipInterpolator is not available. Install scipy.")
            interp = pchip_cls(x_data, y_data, extrapolate=self._extrap_check.isChecked())
            return cast(np.ndarray, interp(x_eval)), {}, "SciPy PCHIP"
        if model == MODEL_SCIPY_AKIMA:
            akima_cls = Akima1DInterpolator
            if akima_cls is None:
                applogger.error("Akima1DInterpolator is not available. Install scipy.")
            interp = akima_cls(x_data, y_data)
            y_eval = cast(np.ndarray, interp(x_eval))
            if self._extrap_check.isChecked():
                return y_eval, {}, "SciPy Akima1D"
            return self._mask_extrapolated(x_data, x_eval, y_eval), {}, "SciPy Akima1D"
        if model == MODEL_SCIPY_CUBIC:
            cubic_cls = CubicSpline
            if cubic_cls is None:
                applogger.error("CubicSpline is not available. Install scipy.")
            interp = cubic_cls(
                x_data,
                y_data,
                bc_type=self._cubic_bc_combo.currentText(),
                extrapolate=self._extrap_check.isChecked(),
            )
            return cast(np.ndarray, interp(x_eval)), {}, "SciPy CubicSpline"
        if model == MODEL_SCIPY_SPLINE:
            return self._evaluate_spline_family(x_data, y_data, x_eval)
        return self._series_interpolate_model(model, x_data, y_data, x_eval, start_params)

    def _evaluate_spline_family(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_eval: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float], str]:
        spline_type = self._spline_type_combo.currentText()
        extrapolate = self._extrap_check.isChecked()

        if spline_type == "CubicSpline":
            cubic_cls = CubicSpline
            if cubic_cls is None:
                applogger.error("CubicSpline is not available. Install scipy.")
            interp = cubic_cls(
                x_data,
                y_data,
                bc_type=self._cubic_bc_combo.currentText(),
                extrapolate=extrapolate,
            )
            return cast(np.ndarray, interp(x_eval)), {}, "SciPy CubicSpline"

        if spline_type == "PCHIP":
            pchip_cls = PchipInterpolator
            if pchip_cls is None:
                applogger.error("PchipInterpolator is not available. Install scipy.")
            interp = pchip_cls(x_data, y_data, extrapolate=extrapolate)
            return cast(np.ndarray, interp(x_eval)), {}, "SciPy PCHIP"

        if spline_type == "Akima1D":
            akima_cls = Akima1DInterpolator
            if akima_cls is None:
                applogger.error("Akima1DInterpolator is not available. Install scipy.")
            interp = akima_cls(x_data, y_data)
            y_eval = cast(np.ndarray, interp(x_eval))
            return self._mask_extrapolated(x_data, x_eval, y_eval), {}, "SciPy Akima1D"

        if spline_type == "B-spline":
            make_spline = make_interp_spline
            if make_spline is None:
                applogger.error("make_interp_spline is not available. Install scipy.")
            k = min(int(self._spline_degree_spin.value()), max(1, x_data.size - 1))
            interp = make_spline(
                x_data,
                y_data,
                k=k,
                bc_type=self._cubic_bc_combo.currentText(),
            )
            y_eval = cast(np.ndarray, interp(x_eval))
            return self._mask_extrapolated(x_data, x_eval, y_eval), {"k": float(k)}, "SciPy B-spline"

        spline_cls = UnivariateSpline
        if spline_cls is None:
            applogger.error("UnivariateSpline is not available. Install scipy.")
        k = min(int(self._spline_degree_spin.value()), max(1, x_data.size - 1))
        interp = spline_cls(
            x_data,
            y_data,
            s=float(self._smoothing_spin.value()),
            k=k,
        )
        y_eval = cast(np.ndarray, interp(x_eval))
        return self._mask_extrapolated(x_data, x_eval, y_eval), {"k": float(k)}, "SciPy UnivariateSpline"

    def _mask_extrapolated(
        self,
        x_data: np.ndarray,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
    ) -> np.ndarray:
        if self._extrap_check.isChecked():
            return y_eval
        mask = (x_eval < float(np.min(x_data))) | (x_eval > float(np.max(x_data)))
        masked = y_eval.astype(float, copy=True)
        masked[mask] = np.nan
        return masked

    def _series_interpolate_model(
        self,
        model: str,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_eval: np.ndarray,
        start_params: dict[str, float],
    ) -> tuple[np.ndarray, dict[str, float], str]:
        series_interp_func = curve_fit
        if series_interp_func is None:
            applogger.error("series_fit is not available. Install scipy.")
            return np.zeros(0),{},""
        res=self._model_function(model, x_data, y_data, start_params)
        if not res:
            return np.zeros(0),{},""
        func, names, guess = res
        popt, _unused = series_interp_func(
            func,
            x_data,
            y_data,
            p0=[guess[name] for name in names],
            maxfev=50_000,
        )
        params = {name: float(value) for name, value in zip(names, popt)}
        return func(x_eval, *popt), params, f"SciPy series_fit {model}"

    def _model_function(
        self,
        model: str,
        x_data: np.ndarray,
        y_data: np.ndarray,
        start_params: dict[str, float],
    ) -> tuple[Callable[..., np.ndarray], list[str], dict[str, float]]|None:
        guess = self._guess_params(model, x_data, y_data)
        guess.update(start_params)
        if model == MODEL_EXPONENTIAL:
            return lambda x, a, b, c: a * np.exp(b * x) + c, ["a", "b", "c"], guess
        if model == MODEL_LOGARITHMIC:
            return lambda x, a, b: a * np.log(x) + b, ["a", "b"], guess
        if model == MODEL_POWER:
            return lambda x, a, b, c: a * np.power(x, b) + c, ["a", "b", "c"], guess
        if model == MODEL_GAUSSIAN:
            return (
                lambda x, a, mu, sigma, c: a * np.exp(-0.5 * np.square((x - mu) / sigma)) + c,
                ["a", "mu", "sigma", "c"],
                guess,
            )
        if model == MODEL_SIGMOID:
            return lambda x, a, x0, k, c: a / (1.0 + np.exp(-k * (x - x0))) + c, ["a", "x0", "k", "c"], guess
        applogger.error(f"Unsupported model: {model}")

    @staticmethod
    def _guess_params(model: str, x_data: np.ndarray, y_data: np.ndarray) -> dict[str, float]:
        x_min = float(np.min(x_data))
        x_max = float(np.max(x_data))
        y_min = float(np.min(y_data))
        y_max = float(np.max(y_data))
        y_span = y_max - y_min if y_max != y_min else 1.0
        x_mid = float(np.median(x_data))
        if model == MODEL_EXPONENTIAL:
            return {"a": y_span, "b": 1.0 / max(abs(x_max - x_min), 1.0), "c": y_min}
        if model == MODEL_LOGARITHMIC:
            return {"a": y_span, "b": y_min}
        if model == MODEL_POWER:
            return {"a": 1.0, "b": 1.0, "c": y_min}
        if model == MODEL_GAUSSIAN:
            return {"a": y_span, "mu": x_mid, "sigma": max((x_max - x_min) / 6.0, 1e-9), "c": y_min}
        if model == MODEL_SIGMOID:
            return {"a": y_span, "x0": x_mid, "k": 1.0 / max(abs(x_max - x_min), 1.0), "c": y_min}
        return {}

    def _metrics(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        model: str,
        params: Mapping[str, float],
    ) -> dict[str, float]:
        try:
            y_hat, _unused, _unused = self._evaluate_model(
                model=model,
                x_data=x_data,
                y_data=y_data,
                x_eval=x_data,
                start_params=dict(params),
            )
        except Exception:
            return {}
        residual = y_data - y_hat
        ss_res = float(np.nansum(np.square(residual)))
        ss_tot = float(np.nansum(np.square(y_data - np.nanmean(y_data))))
        return {
            "rmse": float(np.sqrt(np.nanmean(np.square(residual)))),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan,
        }

    @staticmethod
    def format_results(results: Sequence[FitResult]) -> str:
        lines: list[str] = []
        for result in results:
            lines.append(result.source.name)
            if result.metrics:
                r2 = result.metrics.get("r2")
                rmse = result.metrics.get("rmse")
                if r2 is not None and np.isfinite(r2):
                    lines.append(f"R² = {r2:.5g}")
                if rmse is not None and np.isfinite(rmse):
                    lines.append(f"RMSE = {rmse:.5g}")
            else:
                lines.append(f"Generated {len(result.x_eval)} points")
            lines.append("")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # SeriesOperationDialogBase hooks
    # ------------------------------------------------------------------


    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated": True, "dialog": "series_interpolation"}

    def result_to_frame(self, result: FitResult) -> pd.DataFrame:
        return result.to_frame()

    def result_table_name(self, axis_id: int, result: FitResult) -> str:
        return result.table_name

    def result_series_spec(self, axis_id: int, table_name: str, result: FitResult) -> ResultSeriesSpec:
        return ResultSeriesSpec(
            name=result.output_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                "linestyle": "--",
                "linewidth": 2.0,
                "marker": "",
                "source_series_id": result.source.series_id,
                "model": result.model,
                "generated": True,
                "dialog": "series_interpolation",
            },
        )
