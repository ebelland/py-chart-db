"""Attribute control charts: p, np, c and u.

The other half of SPC, the one ``control_chart_dialog`` does not cover. Those
charts plot a *measurement* and estimate sigma from within-subgroup variation;
these plot a *count* - defectives, or defects - and the limits come from the
distribution the count follows:

* **p**  - fraction defective, ``d / n``. Binomial. Centre ``p̄ = Σd / Σn``,
  limits ``p̄ ± L·√(p̄(1-p̄)/nᵢ)``: when the sample size varies, so does the
  limit, and the chart draws a different band at every point.
* **np** - number defective, ``d``. Binomial with a fixed ``n``. Centre
  ``n·p̄``, limits ``n·p̄ ± L·√(n·p̄(1-p̄))``.
* **c**  - number of defects in a unit of constant size. Poisson. Centre
  ``c̄``, limits ``c̄ ± L·√c̄``.
* **u**  - defects per unit, ``c / n``, where ``n`` is the number of units
  inspected. Poisson. Centre ``ū = Σc / Σn``, limits ``ū ± L·√(ū/nᵢ)`` -
  again per point when ``n`` varies.

The lower limit is clipped at zero: a count cannot be negative, and an
unclipped LCL below zero never signals.

Only Nelson rule 1 - a point outside its own limits - is applied by default.
The run rules are phrased in equal A/B/C zones, which a varying-limit chart
does not have; an optional "runs" check adds the one that survives (a long
stretch on one side of the centre line).

The count column is the series' ``y`` role. The sample-size column is picked
in the parameters pane (or taken from an ``n`` / ``sample_size`` role if the
series carries one); the ``c`` chart needs neither.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from app.data.data_source import parse_roles, row_value
from app.logs.logger import applogger
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.data.sqlite_repo import SqliteRepo
from app.styles.style import create_doc_link, set_doc_link
from app.utils import report_html
from app.utils.i18n import _

ATTR_P = "p (fraction defective)"
ATTR_NP = "np (count defective)"
ATTR_C = "c (defects per unit)"
ATTR_U = "u (defects per unit, variable size)"

ATTRIBUTE_CHARTS = (ATTR_P, ATTR_NP, ATTR_C, ATTR_U)

#: Charts that read a sample-size column. ``c`` is the exception - it assumes
#: a constant area of opportunity, so there is nothing to divide by.
NEEDS_SIZE = frozenset({ATTR_P, ATTR_NP, ATTR_U})
#: Charts whose limits move with the sample size, point by point.
VARIABLE_LIMITS = frozenset({ATTR_P, ATTR_U})
#: Charts on the binomial (a proportion, bounded at 1) rather than Poisson.
BINOMIAL = frozenset({ATTR_P, ATTR_NP})

ATTRIBUTE_DOCS = {
    ATTR_P: ("p-chart", "https://en.wikipedia.org/wiki/P-chart"),
    ATTR_NP: ("np-chart", "https://en.wikipedia.org/wiki/Np-chart"),
    ATTR_C: ("c-chart", "https://en.wikipedia.org/wiki/C-chart"),
    ATTR_U: ("u-chart", "https://en.wikipedia.org/wiki/U-chart"),
}

#: Below this the limits are too soft to trust; the report says so.
RECOMMENDED_SUBGROUPS = 20


@dataclass(slots=True)
class AttributeChartResult:
    """One attribute control chart for one source series."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    y: np.ndarray
    center: float
    upper: np.ndarray
    lower: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    violation_index: tuple[int, ...] = ()

    def to_frame(self) -> pd.DataFrame:
        """Every line the chart draws, as columns of one table.

        ``ucl``/``lcl`` are per point: for p and u they genuinely vary, and
        for np and c the column just repeats the constant so the same query
        draws either kind.
        """
        flagged = set(self.violation_index)
        return pd.DataFrame(
            {
                "x": self.x,
                "y": self.y,
                "center": np.full(self.x.size, self.center),
                "ucl": self.upper,
                "lcl": self.lower,
                "violation": [int(i in flagged) for i in range(self.x.size)],
                "violation_y": [
                    float(value) if index in flagged else None
                    for index, value in enumerate(self.y)
                ],
            }
        )


# ----------------------------------------------------------------------
# The numerics - pure, no Qt, so a test can call them directly
# ----------------------------------------------------------------------
def attribute_limits(
    model: str,
    counts: np.ndarray,
    sizes: np.ndarray,
    sigma_limit: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(plotted_statistic, centre, upper, lower, metadata)``.

    ``counts`` is the series' y (defectives for p/np, defects for c/u).
    ``sizes`` is the sample size per point; for the c chart it is ignored.
    """
    counts = np.asarray(counts, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    point_count = counts.size
    meta: dict[str, Any] = {}

    if model == ATTR_P:
        total_n = float(sizes.sum())
        pbar = float(counts.sum() / total_n) if total_n else 0.0
        statistic = np.divide(counts, sizes, out=np.zeros_like(counts), where=sizes > 0)
        spread = sigma_limit * np.sqrt(
            np.divide(pbar * (1.0 - pbar), sizes, out=np.zeros_like(sizes), where=sizes > 0)
        )
        center, upper, lower = pbar, pbar + spread, pbar - spread
        meta["p-bar"] = pbar

    elif model == ATTR_NP:
        n = float(np.mean(sizes)) if sizes.size else 0.0
        pbar = float(counts.sum() / (n * point_count)) if n and point_count else 0.0
        statistic = counts
        spread = sigma_limit * np.sqrt(max(n * pbar * (1.0 - pbar), 0.0))
        center = n * pbar
        upper = np.full(point_count, center + spread)
        lower = np.full(point_count, center - spread)
        meta["p-bar"] = pbar
        meta["sample size"] = n

    elif model == ATTR_C:
        cbar = float(np.mean(counts)) if counts.size else 0.0
        statistic = counts
        spread = sigma_limit * np.sqrt(max(cbar, 0.0))
        center = cbar
        upper = np.full(point_count, cbar + spread)
        lower = np.full(point_count, cbar - spread)
        meta["c-bar"] = cbar

    elif model == ATTR_U:
        total_n = float(sizes.sum())
        ubar = float(counts.sum() / total_n) if total_n else 0.0
        statistic = np.divide(counts, sizes, out=np.zeros_like(counts), where=sizes > 0)
        spread = sigma_limit * np.sqrt(
            np.divide(ubar, sizes, out=np.zeros_like(sizes), where=sizes > 0)
        )
        center, upper, lower = ubar, ubar + spread, ubar - spread
        meta["u-bar"] = ubar

    else:  # pragma: no cover - the combo cannot hold anything else
        raise ValueError(f"unknown attribute chart {model!r}")

    upper = np.broadcast_to(np.asarray(upper, dtype=float), (point_count,)).copy()
    lower = np.broadcast_to(np.asarray(lower, dtype=float), (point_count,)).copy()
    # A count is non-negative; an LCL below zero would never signal.
    np.clip(lower, 0.0, None, out=lower)
    # A proportion is bounded above by 1.
    if model == ATTR_P:
        np.clip(upper, None, 1.0, out=upper)
    return np.asarray(statistic, dtype=float), float(center), upper, lower, meta


def attribute_violations(
    statistic: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    center: float,
    *,
    runs: bool,
) -> list[int]:
    """Indices that signal: rule 1 always, the long-run rule when asked."""
    flagged: set[int] = set(
        np.flatnonzero((statistic > upper) | (statistic < lower)).tolist()
    )

    if runs and statistic.size >= 8:
        above = statistic > center
        below = statistic < center
        for start in range(statistic.size - 7):
            window = slice(start, start + 8)
            if above[window].all() or below[window].all():
                flagged.add(start + 7)

    return sorted(flagged)


# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------
class SeriesAttributeChartDialog(SeriesOperationDialogBase):
    """Draw a p, np, c or u control chart for a chart series."""

    Name: str = "Attribute Chart"
    Description = "Control chart for counts (p, np, c, u)"

    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    INPUT_MINIMUM_POINTS = 3

    RESULTS_ARE_HTML = True

    Icon = """
    <path d="M3 17h18"/>
    <path d="M4 13l4 1 4-6 4 3 4-4"/>
    <path d="M3 8h18" stroke-dasharray="2 2"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesAttributeChartDialog requires a repository instance.")

        self._last_results: list[AttributeChartResult] = []
        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Attribute Chart",
            parent=parent,
            width=800,
            height=640,
        )
        self.series_selector.reload(select_all_series=True)
        self._refresh_size_columns()
        self._refresh_visibility()
        self.refresh_results()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        self._doc_link = create_doc_link(self)
        self._size_column_combo = QComboBox(self)
        self._sigma_spin = QDoubleSpinBox(self)
        self._center_check = QCheckBox("", self)
        self._limits_check = QCheckBox("", self)
        self._signals_check = QCheckBox("", self)
        self._runs_check = QCheckBox("", self)

    def build_model_selector(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo.addItems(ATTRIBUTE_CHARTS)
        self.model_combo.setToolTip(_("Choose the attribute chart."))
        form.addRow(_("Chart:"), self.model_combo)
        form.addRow(_("Docs:"), self._doc_link)

        layout.addLayout(form)
        return panel

    def build_parameter_selector(self) -> QWidget:
        widget = QWidget(self)
        self._parameter_form = QFormLayout(widget)
        self._parameter_form.setContentsMargins(0, 0, 0, 0)
        self._parameter_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self._size_column_combo.setToolTip(
            _("The column holding the sample size (units inspected) at each point.")
        )
        self._parameter_form.addRow(_("Sample size column:"), self._size_column_combo)

        self._sigma_spin.setRange(1.0, 6.0)
        self._sigma_spin.setDecimals(2)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setValue(3.0)
        self._sigma_spin.setToolTip(
            _("3 is the Shewhart convention (about one false alarm per 370 points).")
        )
        self._parameter_form.addRow(_("Limits at sigma:"), self._sigma_spin)

        self._runs_check.setChecked(False)
        self._runs_check.setToolTip(
            _("Also flag eight points in a row on one side of the centre line.")
        )
        self._parameter_form.addRow(_("Flag long runs:"), self._runs_check)

        self._center_check.setChecked(True)
        self._center_check.setToolTip(_("Draw the centre line."))
        self._parameter_form.addRow(_("Draw the centre line:"), self._center_check)

        self._limits_check.setChecked(True)
        self._limits_check.setToolTip(_("Draw the upper and lower control limits."))
        self._parameter_form.addRow(_("Draw the control limits:"), self._limits_check)

        self._signals_check.setChecked(True)
        self._signals_check.setToolTip(_("Redraw the signalling points as markers."))
        self._parameter_form.addRow(_("Highlight the flagged points:"), self._signals_check)

        return widget

    def connect_common_signals(self) -> None:
        # Not super(): the base connects selection_changed straight to
        # refresh_results, but the size-column combo has to be repopulated
        # from the new selection *before* the results are recomputed.
        changed = getattr(self.series_selector, "selection_changed", None)
        if changed is not None:
            changed.connect(self._on_selection_changed)

    def _on_selection_changed(self, *_args: Any) -> None:
        self._refresh_size_columns()
        self.refresh_results()

    def connect_operation_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.model_combo.currentIndexChanged.connect(self.refresh_results)
        self._size_column_combo.currentIndexChanged.connect(self.refresh_results)
        self._sigma_spin.valueChanged.connect(self.refresh_results)
        for check in (
            self._runs_check, self._center_check,
            self._limits_check, self._signals_check,
        ):
            check.toggled.connect(self.refresh_results)

    def _model(self) -> str:
        return self.model_combo.currentText() or ATTR_P

    def _refresh_visibility(self) -> None:
        needs_size = self._model() in NEEDS_SIZE
        self.set_row_visible(self._size_column_combo, needs_size)
        title, url = ATTRIBUTE_DOCS.get(self._model(), ("", ""))
        set_doc_link(self._doc_link, title, url)

    def _refresh_size_columns(self) -> None:
        """Fill the sample-size combo with the selected series' columns."""
        wanted = self._size_column_combo.currentText()
        columns: list[str] = []
        for row in self.selected_series():
            frame = self._safe_frame(row)
            if frame is None:
                continue
            for column in frame.columns:
                if str(column) not in columns:
                    columns.append(str(column))

        self._size_column_combo.blockSignals(True)
        self._size_column_combo.clear()
        self._size_column_combo.addItems(columns)
        if wanted in columns:
            self._size_column_combo.setCurrentText(wanted)
        elif "n" in columns:
            self._size_column_combo.setCurrentText("n")
        self._size_column_combo.blockSignals(False)

    def refresh_results(self) -> None:
        try:
            results = self.compute_results()
        except Exception as exc:  # noqa: BLE001 - shown in the results pane
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
    # Input
    # ------------------------------------------------------------------
    def _safe_frame(self, row: Any) -> pd.DataFrame | None:
        query = str(row_value(row, "sql_query", "query", default="")).strip()
        if not query:
            return None
        try:
            return self._repo.query_df(query)
        except Exception:  # noqa: BLE001 - a bad query is reported at compute time
            return None

    def _series_counts(
        self, row: Any, name: str, model: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(x, counts, sizes)`` for one series, aligned and finite."""
        frame = self._safe_frame(row)
        if frame is None or frame.empty:
            raise ValueError("the series query returned no rows")

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = [str(column) for column in frame.columns]

        y_col = str(roles.get("y") or "y")
        if y_col not in columns:
            numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
            y_col = numeric[-1] if numeric else columns[-1]
        counts = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)

        x_col = str(roles.get("x") or "x")
        x_values = (
            self.numeric_x(frame[x_col], name)
            if x_col in columns
            else np.arange(counts.size, dtype=float)
        )

        if model in NEEDS_SIZE:
            size_col = self._size_column_combo.currentText().strip()
            size_col = size_col or str(roles.get("n") or roles.get("sample_size") or "")
            if size_col not in columns:
                raise ValueError(
                    "pick the column holding the sample size in the parameters pane"
                )
            sizes = pd.to_numeric(frame[size_col], errors="coerce").to_numpy(dtype=float)
        else:
            sizes = np.ones(counts.size, dtype=float)

        finite = np.isfinite(x_values) & np.isfinite(counts) & np.isfinite(sizes)
        finite &= sizes > 0
        x_values, counts, sizes = x_values[finite], counts[finite], sizes[finite]
        if x_values.size < self.INPUT_MINIMUM_POINTS:
            raise ValueError(
                f"only {x_values.size} usable point(s); at least "
                f"{self.INPUT_MINIMUM_POINTS} are needed"
            )
        if np.any(counts < 0):
            raise ValueError("the count column has negative values")
        if model in BINOMIAL and np.any(counts > sizes):
            raise ValueError("a subgroup has more defectives than its sample size")

        order = np.argsort(x_values, kind="stable")
        return x_values[order], counts[order], sizes[order]

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def compute_results(self) -> list[AttributeChartResult]:
        model = self._model()
        sigma_limit = float(self._sigma_spin.value())
        runs = self._runs_check.isChecked()

        results: list[AttributeChartResult] = []
        errors: list[str] = []
        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, counts, sizes = self._series_counts(row, name, model)
                statistic, center, upper, lower, meta = attribute_limits(
                    model, counts, sizes, sigma_limit
                )
                flagged = attribute_violations(
                    statistic, upper, lower, center, runs=runs
                )
                if x_values.size < RECOMMENDED_SUBGROUPS:
                    meta["note"] = (
                        f"only {x_values.size} subgroups; "
                        f"{RECOMMENDED_SUBGROUPS}+ give trustworthy limits"
                    )
                if model in NEEDS_SIZE:
                    meta["mean sample size"] = float(np.mean(sizes))
                    if float(sizes.min()) != float(sizes.max()):
                        meta["limits"] = "vary with the sample size"
                results.append(
                    AttributeChartResult(
                        source_name=name,
                        result_name=f"{name} - {model.split(' ')[0]}",
                        model=model,
                        x=x_values,
                        y=statistic,
                        center=center,
                        upper=upper,
                        lower=lower,
                        metadata=meta,
                        violation_index=tuple(flagged),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - collected, then reported
                errors.append(f"{name}: {exc}")

        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message, show_dialog=False, raise_error=False)
        return results

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def result_to_frame(self, result: AttributeChartResult) -> pd.DataFrame:
        return result.to_frame()

    def _series_style(self, result: AttributeChartResult, **overrides: Any) -> dict[str, Any]:
        style: dict[str, Any] = {
            "generated_attribute_chart": True,
            "attribute_chart_dialog": "series_attribute_chart",
            "source_name": result.source_name,
            "chart": result.model,
        }
        style.update(overrides)
        return style

    def result_series_spec(
        self, axis_id: int, table_name: str, result: AttributeChartResult,
    ) -> ResultSeriesSpec:
        del axis_id
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style=self._series_style(
                result, linestyle="-", linewidth=1.2, marker="o", markersize=4.0
            ),
        )

    def result_series_specs(
        self, axis_id: int, table_name: str, result: AttributeChartResult,
    ) -> Sequence[ResultSeriesSpec]:
        """The points, plus whichever reference lines are switched on.

        The limits are drawn as ordinary connected lines. For p and u they
        step between points because the limit itself does; a plain polyline
        through the per-point values is the honest picture of that.
        """
        specs: list[ResultSeriesSpec] = []

        def line(column: str, label: str, **style: Any) -> ResultSeriesSpec:
            return ResultSeriesSpec(
                name=f"{result.result_name} - {label}",
                sql_query=f'SELECT x, {column} AS y FROM "{table_name}" ORDER BY x',
                roles={"x": "x", "y": "y"},
                style=self._series_style(result, marker="", **style),
            )

        if self._limits_check.isChecked():
            specs.append(line("ucl", "UCL", linestyle="--", linewidth=1.1, color="#c62828"))
            specs.append(line("lcl", "LCL", linestyle="--", linewidth=1.1, color="#c62828"))
        if self._center_check.isChecked():
            specs.append(line("center", "CL", linestyle="-", linewidth=1.0, color="#2e7d32"))

        specs.append(self.result_series_spec(axis_id, table_name, result))

        if self._signals_check.isChecked() and result.violation_index:
            specs.append(
                ResultSeriesSpec(
                    name=f"{result.result_name} - signals",
                    sql_query=(
                        f'SELECT x, violation_y AS y FROM "{table_name}" '
                        f"WHERE violation_y IS NOT NULL ORDER BY x"
                    ),
                    roles={"x": "x", "y": "y"},
                    style=self._series_style(
                        result, linestyle="", marker="o", markersize=9.0, color="#c62828"
                    ),
                )
            )
        return specs

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {
            "generated_attribute_chart": True,
            "attribute_chart_dialog": "series_attribute_chart",
        }

    def result_table_name(self, axis_id: int, result: AttributeChartResult) -> str:
        return generated_table_name(
            f"AttributeChart_axis{axis_id}_{result.source_name}_{result.model.split(' ')[0]}",
            fallback="AttributeChart_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Attribute Chart"

    def format_results(self, results: Sequence[AttributeChartResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        sections: list[str] = []
        for result in results:
            mean_limit = (
                report_html.format_number(float(np.mean(result.upper)))
                if result.model in VARIABLE_LIMITS
                else report_html.format_number(float(result.upper[0]) if result.upper.size else 0.0)
            )
            pairs = [
                (_("Chart"), result.model),
                (_("Points plotted"), result.y.size),
                (_("Centre line"), report_html.format_number(result.center)),
                (_("Mean upper limit") if result.model in VARIABLE_LIMITS
                 else _("Upper control limit"), mean_limit),
            ]
            pairs += [(str(key), report_html.format_number(value)
                       if isinstance(value, float) else value)
                      for key, value in result.metadata.items()]

            if result.violation_index:
                rows = [
                    (
                        str(index + 1),
                        report_html.format_number(float(result.x[index])),
                        report_html.format_number(float(result.y[index])),
                        _("above UCL") if result.y[index] > result.upper[index]
                        else _("below LCL") if result.y[index] < result.lower[index]
                        else _("run"),
                    )
                    for index in result.violation_index
                ]
                table = report_html.table(
                    (_("Point"), "x", "y", _("Signal")), rows,
                    align=("right", "right", "right", "left"),
                )
                verdict = report_html.note(
                    _("{count} of {total} points signal. The process is not in "
                      "statistical control.").format(
                        count=len(result.violation_index), total=result.y.size
                    )
                )
            else:
                table = ""
                verdict = report_html.note(
                    _("No points signal. The process is in statistical control.")
                )

            sections.append(
                report_html.section(
                    result.source_name,
                    report_html.summary_table(pairs),
                    verdict,
                    table,
                )
            )
        return report_html.document(_("Attribute Chart"), self._model(), *sections)
