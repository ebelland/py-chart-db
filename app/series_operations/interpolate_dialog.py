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
from app.series_operations.dialog_base import (
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
    LinearNDInterpolator,
    RBFInterpolator,
    RegularGridInterpolator,
)
from scipy.spatial import Delaunay
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
MODEL_3D_REGULAR: Final[str] = "3D regular grid"
MODEL_3D_SCATTERED: Final[str] = "3D uneven/scattered"
MODEL_3D_RBF: Final[str] = "3D RBF to grid"

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
    MODEL_3D_REGULAR,
    MODEL_3D_SCATTERED,
    MODEL_3D_RBF,
)

INTERPOLATION_MODELS: Final[set[str]] = {
    MODEL_NUMPY_INTERP,
    MODEL_SCIPY_PCHIP,
    MODEL_SCIPY_AKIMA,
    MODEL_SCIPY_CUBIC,
    MODEL_SCIPY_SPLINE,
}

SURFACE_MODELS: Final[set[str]] = {
    MODEL_3D_REGULAR, MODEL_3D_SCATTERED, MODEL_3D_RBF,
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
    MODEL_3D_REGULAR: ("SciPy RegularGridInterpolator", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html"),
    MODEL_3D_SCATTERED: ("SciPy LinearNDInterpolator", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html"),
    MODEL_3D_RBF: ("SciPy RBFInterpolator", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html"),
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
    z_eval: np.ndarray | None
    params: dict[str, float]
    metrics: dict[str, float]
    message: str

    def to_frame(self) -> pd.DataFrame:
        data: dict[str, object] = {
            "source_series_id": self.source.series_id,
            "source_series_name": self.source.name,
            "model": self.model,
            "x": self.x_eval,
            "y": self.y_eval,
        }
        if self.z_eval is not None:
            data["z"] = self.z_eval
        return pd.DataFrame(data)


class SeriesInterpolateDialog(SeriesOperationDialogBase):
    """Compact series fitting/interpolation dialog for one chart panel."""
    Name: str  = "Interpolation"
    Description = "Fill missing values"

    # Every model here interpolates y as a function of x, so x has to be
    # ordered and single-valued: a spline through two different y at one x has
    # no solution, and SciPy reports that as a singular matrix rather than as
    # a problem with the data.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True

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
        self._grid_y_points_spin = QSpinBox(self)
        self._y_range_edit = QLineEdit(self)
        self._rbf_kernel_combo = QComboBox(self)
        self._rbf_epsilon_spin = QDoubleSpinBox(self)
        self._rbf_smoothing_spin = QDoubleSpinBox(self)
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

        self._grid_y_points_spin.setRange(2, 2000)
        self._grid_y_points_spin.setValue(100)
        form.addRow(_("Grid Y points:"), self._grid_y_points_spin)

        self._y_range_edit.setPlaceholderText(_("auto, or start, stop"))
        form.addRow(_("Y range:"), self._y_range_edit)

        self._rbf_kernel_combo.addItems([
            "thin_plate_spline", "linear", "cubic", "quintic",
            "multiquadric", "inverse_multiquadric", "inverse_quadratic", "gaussian",
        ])
        form.addRow(_("RBF kernel:"), self._rbf_kernel_combo)

        self._rbf_epsilon_spin.setRange(1e-12, 1e12)
        self._rbf_epsilon_spin.setDecimals(6)
        self._rbf_epsilon_spin.setValue(1.0)
        form.addRow(_("RBF epsilon:"), self._rbf_epsilon_spin)

        self._rbf_smoothing_spin.setRange(0.0, 1e12)
        self._rbf_smoothing_spin.setDecimals(6)
        form.addRow(_("RBF smoothing:"), self._rbf_smoothing_spin)

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
            self._y_range_edit,
            self._rbf_kernel_combo,
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
        self._grid_y_points_spin.valueChanged.connect(self.refresh_results)
        self._y_range_edit.textChanged.connect(self.refresh_results)
        self._rbf_kernel_combo.currentIndexChanged.connect(self._refresh_model_defaults)
        self._rbf_epsilon_spin.valueChanged.connect(self.refresh_results)
        self._rbf_smoothing_spin.valueChanged.connect(self.refresh_results)
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

        uses_surface = model in SURFACE_MODELS
        uses_rbf = model == MODEL_3D_RBF
        uses_interpolation = model in INTERPOLATION_MODELS or uses_surface
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
        uses_points = uses_surface or spacing not in {
            "custom X values",
            "original data X",
            "integer step",
        }
        uses_range = uses_surface or spacing not in {"custom X values", "original data X"}
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
        self._set_form_row_visible(self._grid_y_points_spin, uses_surface)
        self._set_form_row_visible(self._y_range_edit, uses_surface)
        self._set_form_row_visible(self._rbf_kernel_combo, uses_rbf)
        self._set_form_row_visible(self._rbf_epsilon_spin, uses_rbf)
        self._set_form_row_visible(self._rbf_smoothing_spin, uses_rbf)
        self._set_form_row_visible(self._spacing_combo, not uses_surface)
        self._set_form_row_visible(self._integer_step_spin, uses_step and not uses_surface)
        self._set_form_row_visible(self._custom_x_edit, uses_custom_x and not uses_surface)
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
        if self._model_name() in SURFACE_MODELS and len(choices) > 1:
            raise ValueError(
                "Select one source series for a 3D grid interpolation. "
                "Preview or OK creates a dedicated Contour Plot axis."
            )
        return [self._interpolate_one_series(choice) for choice in choices]

    @staticmethod
    def _has_grid_results(results: Sequence[FitResult]) -> bool:
        return any(result.model in SURFACE_MODELS for result in results)

    def _create_contour_axis(self, results: Sequence[FitResult]) -> int:
        first = results[0]
        return self.create_result_axis(
            chart_type="Contour Plot",
            title=f"Interpolation: {first.source.name}",
            x_label="X",
            y_label="Y",
            options={"grid": True},
        )

    def preview_results_to_axis(
        self, axis_id: int, results: Sequence[FitResult]
    ) -> None:
        """Write Preview to a new contour axis inside the savepoint."""
        target_axis_id = axis_id
        if self._has_grid_results(results):
            target_axis_id = self._create_contour_axis(results)
            self._preview_axis_ids.add(target_axis_id)
        super().preview_results_to_axis(target_axis_id, results)

    def apply_results_to_axis(
        self, axis_id: int, results: Sequence[FitResult]
    ) -> None:
        """Write OK output to a new contour axis inside the transaction."""
        target_axis_id = axis_id
        if self._has_grid_results(results):
            target_axis_id = self._create_contour_axis(results)
        super().apply_results_to_axis(target_axis_id, results)

    def _interpolate_one_series(self, series: SeriesChoice) -> FitResult:
        df = self._repo.query_df(series.sql_query)
        x_col, y_col, z_col = self._series_columns(df, series.roles)

        x_raw = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        y_raw = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        z_raw = (
            pd.to_numeric(df[z_col], errors="coerce").to_numpy(dtype=float)
            if z_col is not None else None
        )
        model = self._model_name()
        if model in SURFACE_MODELS:
            if z_raw is None:
                raise ValueError(f"{series.name}: the selected 3D model requires X, Y and Z roles.")
            x_data, y_data, z_data = self._prepare_surface_xyz(x_raw, y_raw, z_raw, label=series.name)
            x_eval, y_eval, z_eval, params, metrics, message = self._evaluate_surface(
                model, x_data, y_data, z_data
            )
            return self._make_result(series, model, x_eval, y_eval, z_eval, params, metrics, message)

        if z_raw is None:
            x_data, y_data = self.prepare_input_xy(x_raw, y_raw, label=series.name)
            z_data = None
        else:
            x_data, y_data, z_data = self._prepare_input_xyz(
                x_raw, y_raw, z_raw, label=series.name
            )

        x_eval = self._x_eval(x_data)

        start_params = self._start_params()
        y_eval, params, message = self._evaluate_model(
            model=model,
            x_data=x_data,
            y_data=y_data,
            x_eval=x_eval,
            start_params=start_params,
        )

        metrics = self._metrics(x_data, y_data, model, params)
        z_eval: np.ndarray | None = None
        if z_data is not None:
            z_eval, z_params, z_message = self._evaluate_model(
                model=model,
                x_data=x_data,
                y_data=z_data,
                x_eval=x_eval,
                start_params=start_params,
            )
            z_metrics = self._metrics(x_data, z_data, model, z_params)
            metrics.update({f"z_{key}": value for key, value in z_metrics.items()})
            params.update({f"z_{key}": value for key, value in z_params.items()})
            message = f"Y: {message}; Z: {z_message}"

        return self._make_result(
            series, model, x_eval, y_eval, z_eval, params, metrics, message
        )

    def _make_result(
        self, series: SeriesChoice, model: str, x_eval: np.ndarray,
        y_eval: np.ndarray, z_eval: np.ndarray | None, params: dict[str, float],
        metrics: dict[str, float], message: str,
    ) -> FitResult:
        safe_model_name = _TABLE_SAFE_RE.sub("_", model.strip().lower()).strip("_") or "series"
        table_name = generated_table_name(
            f"Interpolation_axis{self.series_selector.selected_axis_id()}"
            f"_series{series.series_id}_{safe_model_name}",
            fallback="Interpolation_Result",
        )
        return FitResult(
            source=series, model=model, table_name=table_name,
            output_name=f"Interpolate: {series.name} [{model}]",
            x_eval=x_eval, y_eval=y_eval, z_eval=z_eval, params=params,
            metrics=metrics, message=message,
        )

    @staticmethod
    def _series_columns(
        df: pd.DataFrame,
        roles: Mapping[str, Any],
    ) -> tuple[str, str, str | None]:
        """Resolve X/Y and the optional Z role for 2D or 3D series."""
        if df.empty:
            raise ValueError("Series query returned no rows.")

        columns = [str(col) for col in df.columns]
        x_name = str(roles.get("x", ""))
        y_name = str(roles.get("y", ""))
        z_name = str(roles.get("z", ""))

        if x_name in columns and y_name in columns:
            return x_name, y_name, z_name if z_name in columns else None

        numeric = [
            str(col)
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        if len(numeric) < 2:
            raise ValueError("Series query must expose at least two numeric columns.")
        fallback_z = numeric[2] if len(numeric) >= 3 and z_name else None
        return numeric[0], numeric[1], fallback_z

    @staticmethod
    def _xy_columns(df: pd.DataFrame, roles: Mapping[str, Any]) -> tuple[str, str]:
        """Backward-compatible 2D column resolver."""
        x_name, y_name, _z_name = SeriesInterpolateDialog._series_columns(df, roles)
        return x_name, y_name

    @staticmethod
    def _prepare_surface_xyz(
        x: np.ndarray, y: np.ndarray, z: np.ndarray, *, label: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not (x.size == y.size == z.size):
            raise ValueError(f"{label}: X, Y and Z must have equal lengths.")
        frame = pd.DataFrame({"x": x, "y": y, "z": z}).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        frame = frame.groupby(["x", "y"], as_index=False, sort=True)["z"].mean()
        if len(frame) < 3:
            raise ValueError(f"{label}: at least three distinct X/Y points are required.")
        return tuple(frame[c].to_numpy(dtype=float) for c in ("x", "y", "z"))

    def _axis_grid(self, values: np.ndarray, count: int, range_text: str) -> np.ndarray:
        if range_text.strip():
            limits = self._parse_values(range_text)
            if limits.size < 2:
                raise ValueError("Grid range must contain start and stop.")
            start, stop = float(limits[0]), float(limits[1])
        else:
            start, stop = float(np.min(values)), float(np.max(values))
            if self._extrap_check.isChecked():
                pad = (stop - start) * float(self._extend_spin.value()) / 100.0
                start, stop = start - pad, stop + pad
        if start > stop:
            start, stop = stop, start
        if start == stop:
            raise ValueError("Grid range endpoints must differ.")
        return np.linspace(start, stop, count)

    def _evaluate_surface(
        self, model: str, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, float], str]:
        x_axis = self._axis_grid(x, self._points_spin.value(), self._range_edit.text())
        y_axis = self._axis_grid(y, self._grid_y_points_spin.value(), self._y_range_edit.text())
        xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
        targets = np.column_stack((xx.ravel(), yy.ravel()))
        source = np.column_stack((x, y))
        params: dict[str, float] = {"nx": float(x_axis.size), "ny": float(y_axis.size)}

        if model == MODEL_3D_REGULAR:
            ux, uy = np.unique(x), np.unique(y)
            if ux.size * uy.size != z.size:
                raise ValueError("Regular-grid interpolation requires one Z value for every X/Y grid pair.")
            for axis, name in ((ux, "X"), (uy, "Y")):
                if axis.size > 2 and not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-6, atol=1e-12):
                    raise ValueError(f"{name} values are not equally spaced; use 3D uneven/scattered or 3D RBF.")
            grid = pd.DataFrame({"x": x, "y": y, "z": z}).pivot(index="x", columns="y", values="z")
            interpolator = RegularGridInterpolator(
                (grid.index.to_numpy(float), grid.columns.to_numpy(float)),
                grid.to_numpy(float), bounds_error=False, fill_value=None if self._extrap_check.isChecked() else np.nan,  # type: ignore[arg-type]
            )
            z_grid = interpolator(targets)
            z_source = interpolator(source)
            message = f"Regular-grid surface converted to {x_axis.size} x {y_axis.size} grid"
        elif model == MODEL_3D_SCATTERED:
            interpolator = LinearNDInterpolator(source, z, fill_value=np.nan)
            z_grid = np.asarray(interpolator(targets), dtype=float)
            z_source = np.asarray(interpolator(source), dtype=float)
            message = f"Uneven/scattered surface converted to {x_axis.size} x {y_axis.size} grid"
        else:
            kernel = self._rbf_kernel_combo.currentText()
            smoothing = float(self._rbf_smoothing_spin.value())
            epsilon_kernels = {
                "multiquadric", "inverse_multiquadric",
                "inverse_quadratic", "gaussian",
            }
            if kernel in epsilon_kernels:
                interpolator = RBFInterpolator(
                    source, z, kernel=kernel, smoothing=smoothing,
                    epsilon=float(self._rbf_epsilon_spin.value()),
                )
            else:
                interpolator = RBFInterpolator(
                    source, z, kernel=kernel, smoothing=smoothing,
                )
            z_grid = np.asarray(interpolator(targets), dtype=float)
            z_source = np.asarray(interpolator(source), dtype=float)
            params.update({"epsilon": float(self._rbf_epsilon_spin.value()), "smoothing": float(self._rbf_smoothing_spin.value())})
            if not self._extrap_check.isChecked():
                inside = Delaunay(source).find_simplex(targets) >= 0
                z_grid[~inside] = np.nan
            message = f"RBF ({kernel}) surface converted to {x_axis.size} x {y_axis.size} grid"

        finite = np.isfinite(z_source) & np.isfinite(z)
        metrics: dict[str, float] = {}
        if finite.any():
            residual = z[finite] - z_source[finite]
            metrics["z_rmse"] = float(np.sqrt(np.mean(residual ** 2)))
            total = float(np.sum((z[finite] - np.mean(z[finite])) ** 2))
            metrics["z_r2"] = 1.0 - float(np.sum(residual ** 2)) / total if total > 0 else float("nan")
        return xx.ravel(), yy.ravel(), z_grid, params, metrics, message

    def _prepare_input_xyz(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Clean a 3D line series while preserving aligned X/Y/Z rows."""
        if not (x.size == y.size == z.size):
            raise ValueError(f"{label}: X, Y and Z must have equal lengths.")
        frame = pd.DataFrame({"x": x, "y": y, "z": z}).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if frame.empty:
            raise ValueError(f"{label}: no finite X/Y/Z points are available.")
        frame = frame.groupby("x", as_index=False, sort=True)[["y", "z"]].mean()
        if len(frame) < 2:
            raise ValueError(f"{label}: at least two distinct X values are required.")
        return (
            frame["x"].to_numpy(dtype=float),
            frame["y"].to_numpy(dtype=float),
            frame["z"].to_numpy(dtype=float),
        )

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
                    lines.append(f"Y RMSE = {rmse:.5g}" if result.z_eval is not None else f"RMSE = {rmse:.5g}")
                z_r2 = result.metrics.get("z_r2")
                z_rmse = result.metrics.get("z_rmse")
                if z_r2 is not None and np.isfinite(z_r2):
                    lines.append(f"Z R² = {z_r2:.5g}")
                if z_rmse is not None and np.isfinite(z_rmse):
                    lines.append(f"Z RMSE = {z_rmse:.5g}")
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
        """Return a generated name tied to the actual destination axis.

        Never reuse the source table name. Preview adds its own suffix in the
        base class, and all operation-owned tables retain the generated prefix.
        """
        safe_model = _TABLE_SAFE_RE.sub(
            "_", result.model.strip().lower()
        ).strip("_") or "series"
        return generated_table_name(
            f"Interpolation_axis{axis_id}"
            f"_series{result.source.series_id}_{safe_model}",
            fallback="Interpolation_Result",
        )

    def result_series_spec(self, axis_id: int, table_name: str, result: FitResult) -> ResultSeriesSpec:
        is_3d = result.z_eval is not None
        columns = "x, y, z" if is_3d else "x, y"
        roles = {"x": "x", "y": "y", "z": "z"} if is_3d else {"x": "x", "y": "y"}
        return ResultSeriesSpec(
            name=result.output_name,
            sql_query=f'SELECT {columns} FROM "{table_name}" ORDER BY x',
            roles=roles,
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
