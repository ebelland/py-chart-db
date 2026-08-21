"""Numerical derivative and integral of a chart series.

Both halves of calculus in one dialog, because they are the same conversation
about the same series and neither is much use without the other: you
differentiate to find where a signal turns, and integrate to find how much of
it there was.

Two things are deliberately built in rather than left as steps the user is
expected to remember first.

**Smoothing, inside the derivative.**  Differentiation amplifies noise - that
is not a caveat, it is the whole difficulty of doing it numerically.  A
finite difference divides by a small dx, so noise of size e on neighbouring
points becomes noise of size 2e/dx on the result; halve the sampling interval
and the noise doubles.  Offering a raw ``np.gradient`` and trusting people to
smooth first produces a spiky mess that looks like data.  Savitzky-Golay is
therefore the default: it fits a low-order polynomial over a window and
differentiates *that*, so smoothing and differentiating happen in one step
rather than as two the user can get the order of wrong.

**Baseline subtraction, inside the integral.**  The area under a peak means
nothing if the signal does not return to zero: a constant offset contributes
offset x width to the result, which for a broad peak on a raised baseline is
most of the answer.  So the integral offers to remove a baseline first.

Peak finding lives next door in ``series_peaks_dialog``; find a peak there,
read its bounds, integrate between them here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget
from scipy.integrate import cumulative_trapezoid, simpson, trapezoid
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from app.data.data_source import parse_roles, row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.parameter_spec import BoolParam, ChoiceParam, IntParam
from app.series_operations.series_operation_dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.styles.style import create_doc_link, set_doc_link
from app.widgets import report_html
from app.utils.i18n import _

# --- Models -----------------------------------------------------------

DERIV_SAVGOL = "Derivative (Savitzky-Golay)"
DERIV_GRADIENT = "Derivative (finite difference)"
DERIV_SPLINE = "Derivative (spline)"
INTEGRAL_CUMULATIVE = "Integral (cumulative)"
INTEGRAL_DEFINITE = "Integral (total area)"

CALCULUS_MODELS = (
    DERIV_SAVGOL,
    DERIV_GRADIENT,
    DERIV_SPLINE,
    INTEGRAL_CUMULATIVE,
    INTEGRAL_DEFINITE,
)

DERIVATIVES = (DERIV_SAVGOL, DERIV_GRADIENT, DERIV_SPLINE)
INTEGRALS = (INTEGRAL_CUMULATIVE, INTEGRAL_DEFINITE)

CALCULUS_DOCS = {
    DERIV_SAVGOL: (
        "Savitzky-Golay filter",
        "https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter",
    ),
    DERIV_GRADIENT: (
        "Finite difference",
        "https://en.wikipedia.org/wiki/Finite_difference",
    ),
    DERIV_SPLINE: (
        "Smoothing spline",
        "https://en.wikipedia.org/wiki/Smoothing_spline",
    ),
    INTEGRAL_CUMULATIVE: (
        "Trapezoidal rule",
        "https://en.wikipedia.org/wiki/Trapezoidal_rule",
    ),
    INTEGRAL_DEFINITE: (
        "Simpson's rule",
        "https://en.wikipedia.org/wiki/Simpson%27s_rule",
    ),
}

BASELINE_NONE = "none"
BASELINE_MINIMUM = "minimum"
BASELINE_ENDPOINTS = "endpoints"


@dataclass(slots=True)
class CalculusResult:
    """Derivative or integral of one source series."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    y: np.ndarray
    #: Set only for the definite integral, which produces one number rather
    #: than a curve. Kept separate from ``y`` so the report can show it without
    #: the chart having to special-case a one-point series.
    total: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"x": self.x, "y": self.y})


class SeriesCalculusDialog(SeriesOperationDialogBase):
    """Differentiate or integrate a chart series."""

    Name: str = "Calculus"
    Description = "Differentiate or integrate"

    # Every model here treats the series as y = f(x), so the points have to be
    # in x order and single-valued. Both are repaired and reported.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    # A derivative needs a point either side of the one it is estimating.
    INPUT_MINIMUM_POINTS = 3

    PARAMS = (
        IntParam(
            "window",
            "Smoothing window:",
            tooltip=(
                "Points per polynomial fit. Larger smooths more and rounds "
                "sharp features; must exceed the polynomial order."
            ),
            default_value=11,
            minimum=3,
            maximum=9999,
            odd_only=True,
            visible_for={"model": (DERIV_SAVGOL,)},
        ),
        IntParam(
            "polyorder",
            "Polynomial order:",
            tooltip="Order of the polynomial fitted over each window.",
            default_value=3,
            minimum=1,
            maximum=9,
            visible_for={"model": (DERIV_SAVGOL,)},
        ),
        IntParam(
            "order",
            "Derivative order:",
            tooltip="1 for the slope, 2 for the curvature.",
            default_value=1,
            minimum=1,
            maximum=2,
            visible_for={"model": (DERIV_SAVGOL, DERIV_SPLINE)},
        ),
        IntParam(
            "smoothing",
            "Spline smoothing:",
            tooltip=(
                "0 interpolates every point exactly; larger values allow the "
                "spline to depart from noisy data."
            ),
            default_value=0,
            minimum=0,
            maximum=1_000_000,
            visible_for={"model": (DERIV_SPLINE,)},
        ),
        ChoiceParam(
            "baseline",
            "Baseline:",
            tooltip=(
                "Removed before integrating. Without this, a raised baseline "
                "contributes its offset times the width to the area."
            ),
            choices=(
                ("None", BASELINE_NONE),
                ("Subtract minimum", BASELINE_MINIMUM),
                ("Straight line between endpoints", BASELINE_ENDPOINTS),
            ),
            visible_for={"model": INTEGRALS},
        ),
        BoolParam(
            "new_axis",
            "Draw on a new axis:",
            tooltip=(
                "A derivative or an integral rarely shares a scale with the "
                "series it came from - d/dx of a slow drift is near zero "
                "beside data in the thousands. On by default for that reason; "
                "turn it off to overlay when the ranges happen to be "
                "comparable."
            ),
            default_value=True,
        ),
        BoolParam(
            "simpson",
            "Use Simpson's rule:",
            tooltip=(
                "More accurate on smooth data than the trapezoidal rule. "
                "Assumes the curve is well approximated by parabolas."
            ),
            default_value=False,
            visible_for={"model": (INTEGRAL_DEFINITE,)},
        ),
    )

    Icon = """
    <path d="M7 19c0-6 2-14 5-14"/>
    <path d="M5 12h9"/>
    <path d="M14 19h5"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesCalculusDialog requires a repository instance.")

        self._last_results: list[CalculusResult] = []
        self._parameter_form: QFormLayout | None = None
        # Created lazily on the first Preview/Apply that asks for it, and
        # reused after, so adjusting the window repeatedly does not leave a
        # trail of empty axes behind.
        self._result_axis_id: int | None = None
        self._applied = False

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Calculus",
            parent=parent,
            width=760,
            height=640,
        )
        self.series_selector.reload(select_all_series=True)
        self._refresh_visibility()
        self.refresh_results()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_operation_widgets(self) -> None:
        self._doc_link = create_doc_link(self)
        self._parameter_form = None

    def build_model_selector(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        container = QWidget(panel)
        form = QFormLayout(container)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo.addItems(CALCULUS_MODELS)
        self.model_combo.setToolTip(_("Choose the calculation."))
        form.addRow(_("Model:"), self.model_combo)
        form.addRow(_("Docs:"), self._doc_link)

        layout.addWidget(container)
        return panel

    def connect_operation_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.model_combo.currentIndexChanged.connect(self.refresh_results)

    def _refresh_visibility(self) -> None:
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()
        title, url = CALCULUS_DOCS[self._model()]
        set_doc_link(self._doc_link, title, url)

    def _model(self) -> str:
        return self.model_combo.currentText() or DERIV_SAVGOL

    def refresh_results(self) -> None:
        try:
            results = self.compute_results()
        except Exception as exc:
            self._last_results = []
            self.set_results_text(f"Error:\n{exc}")
            return

        self._last_results = list(results)
        self.set_results_text(
            self.format_results(results)
            if results
            else _("Select one or more source series.")
        )

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute_results(self) -> list[CalculusResult]:
        model = self._model()
        params = self.parameter_values()

        results: list[CalculusResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self._series_xy(row, name)
                results.append(self._compute_one(name, x_values, y_values, model, params))
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message, show_dialog=False, raise_error=False)

        return results

    def _series_xy(self, row: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Materialize one series, validated and repaired."""
        sql_query = str(row_value(row, "sql_query", "query", "sql", default="")).strip()
        if not sql_query:
            raise ValueError("the series has no SQL query")

        frame = self._repo.query_df(sql_query)
        if frame.empty:
            raise ValueError("the series query returned no rows")

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = [str(column) for column in frame.columns]
        numeric = [
            str(column)
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column])
        ]

        x_col = str(roles.get("x") or "")
        y_col = str(roles.get("y") or "")
        if x_col not in columns:
            x_col = numeric[0] if numeric else columns[0]
        if y_col not in columns:
            y_col = numeric[1] if len(numeric) > 1 else x_col

        return self.prepare_input_xy(
            pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float),
            label=name,
        )

    def _compute_one(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> CalculusResult:
        if model in DERIVATIVES:
            return self._differentiate(name, x_values, y_values, model, params)
        return self._integrate(name, x_values, y_values, model, params)

    # --- Derivatives ---------------------------------------------------

    def _differentiate(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> CalculusResult:
        order = int(params.get("order", 1))

        if model == DERIV_GRADIENT:
            # np.gradient, not np.diff: it takes x explicitly, so it is correct
            # on unevenly sampled data, and it returns one value per input
            # point rather than n-1, so the result still lines up with the
            # source series on the same axis.
            derivative = np.gradient(y_values, x_values, edge_order=2)
            detail = "central difference"

        elif model == DERIV_SPLINE:
            spline = UnivariateSpline(
                x_values,
                y_values,
                k=min(5, max(order + 1, 3)),
                s=float(params.get("smoothing", 0)),
            )
            derivative = spline.derivative(n=order)(x_values)
            detail = f"spline, s={params.get('smoothing', 0)}"

        else:
            window, polyorder = self._savgol_window(x_values.size, params)
            spacing = self._uniform_spacing(x_values, name)
            # delta scales the result into units of y per unit of x. Without
            # it savgol returns a derivative per sample index, which is off by
            # a factor of the sampling interval - silently right only when the
            # step happens to be 1.
            derivative = savgol_filter(
                y_values,
                window_length=window,
                polyorder=polyorder,
                deriv=order,
                delta=spacing,
            )
            detail = f"window {window}, order {polyorder}"

        return CalculusResult(
            source_name=name,
            result_name=f"{name} - d{'²' if order == 2 else ''}y/dx{'²' if order == 2 else ''}",
            model=model,
            x=x_values,
            y=derivative,
            metadata={"order": order, "detail": detail},
        )

    def _savgol_window(
        self,
        available: int,
        params: Mapping[str, Any],
    ) -> tuple[int, int]:
        """Return a (window, polyorder) savgol_filter will actually accept.

        Both of its constraints are reported from inside SciPy in terms of
        array shapes rather than of the controls the user moved, so they are
        resolved here: the window cannot exceed the series, and the polynomial
        order must be below the window.
        """
        window = int(params.get("window", 11))
        polyorder = int(params.get("polyorder", 3))

        if window > available:
            window = available if available % 2 == 1 else available - 1
        window = max(3, window)

        if polyorder >= window:
            polyorder = window - 1
        return window, max(1, polyorder)

    def _uniform_spacing(self, x_values: np.ndarray, name: str) -> float:
        """Return the sample spacing savgol should assume.

        savgol_filter takes a single delta, so it can only be right for evenly
        spaced data. The median step is the best single answer for data that
        is nearly even; validate_input_xy has already warned when it is not.
        """
        steps = np.diff(x_values)
        if steps.size == 0:
            return 1.0
        spacing = float(np.median(steps))
        if spacing <= 0.0:
            applogger.warning(
                f"{name}: could not determine a sample spacing; assuming 1.",
                show_dialog=False,
                raise_error=False,
            )
            return 1.0
        return spacing

    # --- Integrals -----------------------------------------------------

    def _integrate(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> CalculusResult:
        corrected, baseline_detail = self._subtract_baseline(
            x_values, y_values, str(params.get("baseline", BASELINE_NONE))
        )

        if model == INTEGRAL_CUMULATIVE:
            # initial=0 so the result has one value per input point and starts
            # at zero, which is what makes it plottable against the source.
            running = cumulative_trapezoid(corrected, x_values, initial=0.0)
            return CalculusResult(
                source_name=name,
                result_name=f"{name} - ∫y dx",
                model=model,
                x=x_values,
                y=running,
                total=float(running[-1]) if running.size else 0.0,
                metadata={"baseline": baseline_detail},
            )

        use_simpson = bool(params.get("simpson", False))
        if use_simpson and x_values.size % 2 == 0:
            # Simpson's rule pairs intervals, so it needs an odd number of
            # points. SciPy silently changes method on an even sample rather
            # than saying so, which makes the reported rule wrong.
            applogger.warning(
                f"{name}: Simpson's rule needs an odd number of points; "
                f"the series has {x_values.size}, so the trapezoidal rule was "
                f"used instead.",
                show_dialog=False,
                raise_error=False,
            )
            use_simpson = False

        total = (
            float(simpson(corrected, x=x_values))
            if use_simpson
            else float(trapezoid(corrected, x_values))
        )

        return CalculusResult(
            source_name=name,
            result_name=f"{name} - area",
            model=model,
            # A single number still has to be a series to be stored and drawn,
            # so it is reported across the range it was computed over.
            x=np.array([float(x_values[0]), float(x_values[-1])]),
            y=np.array([total, total]),
            total=total,
            metadata={
                "baseline": baseline_detail,
                "rule": "Simpson" if use_simpson else "trapezoidal",
            },
        )

    @staticmethod
    def _subtract_baseline(
        x_values: np.ndarray,
        y_values: np.ndarray,
        mode: str,
    ) -> tuple[np.ndarray, str]:
        """Return the signal with its baseline removed, and what was done."""
        if mode == BASELINE_MINIMUM:
            floor = float(np.min(y_values))
            return y_values - floor, f"minimum ({floor:g}) subtracted"

        if mode == BASELINE_ENDPOINTS:
            if x_values.size < 2:
                return y_values, "none"
            # The straight line through the first and last points, which is
            # the usual approximation for a peak sitting on a sloping
            # background.
            slope = (y_values[-1] - y_values[0]) / (x_values[-1] - x_values[0])
            line = y_values[0] + slope * (x_values - x_values[0])
            return y_values - line, "endpoint line subtracted"

        return y_values, "none"

    # ------------------------------------------------------------------
    # Where the results are drawn
    # ------------------------------------------------------------------

    def apply(self) -> bool:
        """Apply, and keep any axis this dialog created."""
        applied = super().apply()
        # Only now is the new axis the user's rather than this dialog's.
        self._applied = self._applied or applied
        return applied

    def resolve_target_axis_id(
        self,
        selected_axis_id: int,
        results: Sequence[Any],
    ) -> int:
        """Return the axis to draw on: a new one unless asked otherwise.

        A derivative and its source almost never share a scale. Differentiating
        divides by dx, so a slow drift over thousands of seconds has a
        derivative near zero; drawn on the source's axis the result is a flat
        line on the zero gridline while the source fills the plot. An integral
        goes the other way and grows without bound. Either way both curves
        become unreadable, which is why this defaults to a new axis rather than
        following the usual "write back where the input came from" rule.

        A new axis on the same figure, not a new tab: the result is a second
        view of this chart's data and belongs beside it.
        """
        if not bool(self.parameter_values().get("new_axis", True)):
            return selected_axis_id

        if self._result_axis_id is None:
            self._result_axis_id = self.create_result_axis(
                chart_type="Scatter Plot",
                title=self._model(),
                options={"grid": True, "linestyle": "-", "marker": ""},
            )

        self._label_result_axis(results)
        return self._result_axis_id

    def _label_result_axis(self, results: Sequence[Any]) -> None:
        """Name the axis after what was actually computed.

        From the results rather than from the model name, because the same
        model produces different quantities: a first derivative and a second
        are both "Savitzky-Golay".
        """
        if self._result_axis_id is None or not results:
            return

        first = results[0]
        order = int(first.metadata.get("order", 1) or 1)
        if first.model in DERIVATIVES:
            y_label = "d\u00b2y/dx\u00b2" if order == 2 else "dy/dx"
        else:
            y_label = "\u222by dx"

        try:
            self._repo.update_axis_descriptor(
                axis_id=self._result_axis_id,
                title=str(first.model or ""),
                x_label="x",
                y_label=y_label,
            )
        except Exception:
            applogger.exception("Failed to label the calculus result axis")

    def discard_operation_artifacts(self) -> None:
        """Delete the axis this dialog created, when Apply never happened.

        Creating an axis commits, so it is not covered by the preview
        savepoint: closing without applying has to remove it by hand or an
        empty axis is left behind.
        """
        if self._applied or self._result_axis_id is None:
            return

        axis_id = self._result_axis_id
        self._result_axis_id = None
        try:
            self._repo.delete_axis(axis_id)
            applogger.info("Discarded the unapplied calculus axis %s.", axis_id)
        except Exception:
            applogger.exception("Failed to discard calculus axis %s", axis_id)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_to_frame(self, result: CalculusResult) -> pd.DataFrame:
        return result.to_frame()

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: CalculusResult,
    ) -> ResultSeriesSpec:
        del axis_id
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                "generated_calculus": True,
                "calculus_dialog": "series_calculus",
                "source_name": result.source_name,
                "model": result.model,
                "linestyle": "-",
                "linewidth": 1.6,
                "marker": "",
            },
        )

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_calculus": True, "calculus_dialog": "series_calculus"}

    def result_table_name(self, axis_id: int, result: CalculusResult) -> str:
        return generated_table_name(
            f"Calculus_axis{axis_id}_{result.source_name}_{result.model}",
            fallback="Calculus_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Calculus"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[CalculusResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        rows = [
            (
                result.source_name,
                result.model,
                report_html.format_number(result.y.size, digits=0),
                report_html.format_number(result.total)
                if result.total is not None
                else "&mdash;",
                ", ".join(f"{key}: {value}" for key, value in result.metadata.items()),
            )
            for result in results
        ]

        return report_html.document(
            _("Calculus"),
            self._model(),
            report_html.section(
                _("Results"),
                report_html.table(
                    (_("Series"), _("Model"), _("Points"), _("Total"), _("Detail")),
                    rows,
                    align=("left", "left", "right", "right", "left"),
                    empty_message=_("No results for this selection."),
                ),
            ),
        )
