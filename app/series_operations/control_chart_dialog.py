"""Shewhart control charts for a chart series.

A control chart asks one question: is this process varying the way a stable
process varies, or has something changed?  It answers it by drawing limits at
plus and minus three sigma of the process's *own* short-term variation and
flagging the points that fall outside.

The thing that makes it a control chart rather than a scatter plot with error
bars is where that sigma comes from.  It is **never** the standard deviation
of all the data.  A process that has drifted has a large overall standard
deviation precisely *because* it drifted, so limits built from it are wide
enough to contain the drift and the chart declares the process fine.  Sigma is
estimated instead from variation *within* subgroups - the average moving range
for individual measurements, the average range or standard deviation within
subgroups otherwise - which is unaffected by shifts between them.  That is the
whole idea, and it is the one thing easy to get wrong.

The estimators need the unbiasing constants d2, d3, c4, A2, D3, D4, B3 and B4.
They are tabulated below rather than computed: c4 has a closed form in gamma
functions, but d2 and d3 are integrals over the range distribution with no
elementary form, and every SPC text ships the same table.  Using anything else
would put this chart's limits at odds with every other tool's.

Violations are reported by the Nelson rules, which catch the patterns that
stay inside the limits - a run on one side, a trend, a hug of the centre line -
and are what a chart is for beyond spotting the obvious outlier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget

from app.data.data_source import parse_roles, row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.parameter_spec import BoolParam, FloatParam, IntParam
from app.series_operations.series_operation_dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.styles.style import create_doc_link, set_doc_link
from app.utils.i18n import _
from app.widgets import report_html

CHART_INDIVIDUALS = "Individuals (I-MR)"
CHART_MOVING_RANGE = "Moving range (MR)"
CHART_XBAR_R = "X-bar and R"
CHART_XBAR_S = "X-bar and S"

CONTROL_CHARTS = (
    CHART_INDIVIDUALS,
    CHART_MOVING_RANGE,
    CHART_XBAR_R,
    CHART_XBAR_S,
)

SUBGROUPED = (CHART_XBAR_R, CHART_XBAR_S)

CONTROL_DOCS = {
    CHART_INDIVIDUALS: (
        "Individuals control chart",
        "https://en.wikipedia.org/wiki/Shewhart_individuals_control_chart",
    ),
    CHART_MOVING_RANGE: (
        "Moving range",
        "https://en.wikipedia.org/wiki/Shewhart_individuals_control_chart",
    ),
    CHART_XBAR_R: (
        "X-bar and R chart",
        "https://en.wikipedia.org/wiki/X%CC%84_and_R_chart",
    ),
    CHART_XBAR_S: (
        "X-bar and s chart",
        "https://en.wikipedia.org/wiki/X%CC%84_and_s_chart",
    ),
}

#: Unbiasing constants by subgroup size, from the standard SPC tables.
#: n -> (d2, d3, c4, A2, D3, D4, B3, B4)
#:
#: d2/d3 relate the mean range to sigma; c4 does the same for the mean standard
#: deviation. A2, D3, D4, B3, B4 are the shortcuts that fold those into limit
#: formulas directly, and are what the published tables give.
SPC_CONSTANTS: dict[int, tuple[float, float, float, float, float, float, float, float]] = {
    2:  (1.128, 0.853, 0.7979, 1.880, 0.000, 3.267, 0.000, 3.267),
    3:  (1.693, 0.888, 0.8862, 1.023, 0.000, 2.574, 0.000, 2.568),
    4:  (2.059, 0.880, 0.9213, 0.729, 0.000, 2.282, 0.000, 2.266),
    5:  (2.326, 0.864, 0.9400, 0.577, 0.000, 2.114, 0.000, 2.089),
    6:  (2.534, 0.848, 0.9515, 0.483, 0.000, 2.004, 0.030, 1.970),
    7:  (2.704, 0.833, 0.9594, 0.419, 0.076, 1.924, 0.118, 1.882),
    8:  (2.847, 0.820, 0.9650, 0.373, 0.136, 1.864, 0.185, 1.815),
    9:  (2.970, 0.808, 0.9693, 0.337, 0.184, 1.816, 0.239, 1.761),
    10: (3.078, 0.797, 0.9727, 0.308, 0.223, 1.777, 0.284, 1.716),
    11: (3.173, 0.787, 0.9754, 0.285, 0.256, 1.744, 0.321, 1.679),
    12: (3.258, 0.778, 0.9776, 0.266, 0.283, 1.717, 0.354, 1.646),
    13: (3.336, 0.770, 0.9794, 0.249, 0.307, 1.693, 0.382, 1.618),
    14: (3.407, 0.763, 0.9810, 0.235, 0.328, 1.672, 0.406, 1.594),
    15: (3.472, 0.756, 0.9823, 0.223, 0.347, 1.653, 0.428, 1.572),
    20: (3.735, 0.729, 0.9869, 0.180, 0.415, 1.585, 0.510, 1.490),
    25: (3.931, 0.709, 0.9896, 0.153, 0.459, 1.541, 0.565, 1.435),
}

#: Applied when the subgroup size is outside the table. c4 has a closed form,
#: and d2 tends to a slow logarithmic growth; a large-n approximation is
#: better than refusing to draw the chart, but it is reported as approximate.
LARGEST_TABULATED = max(SPC_CONSTANTS)


@dataclass(slots=True)
class Violation:
    """Every rule broken at one point.

    All of them, not just the first: the rule numbers are historical, not a
    severity ranking, so picking one to report means picking arbitrarily. A
    stretch of points that both sits on one side of the centre (rule 2) and
    hugs it (rule 7) is telling you two different things, and reporting only
    the lower-numbered one hides the more interesting half.
    """

    index: int
    x: float
    y: float
    rules: tuple[int, ...]
    descriptions: tuple[str, ...]

    @property
    def rule(self) -> int:
        """The lowest rule number, for sorting and for a compact display."""
        return min(self.rules) if self.rules else 0


@dataclass(slots=True)
class ControlChartResult:
    """Control chart for one source series."""

    source_name: str
    result_name: str
    chart: str
    x: np.ndarray
    y: np.ndarray
    center: float
    upper: float
    lower: float
    sigma: float
    subgroup_size: int
    violations: list[Violation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Every line the chart may draw, as columns of one table.

        Written whether or not the corresponding checkbox is ticked: the
        columns cost almost nothing, and holding them means turning a line on
        later is a descriptor change rather than a recomputation.

        The zone columns are the one- and two-sigma lines, which is what
        divides a Shewhart chart into the A/B/C zones the run rules are
        phrased in - "two of three beyond two sigma" is a statement about a
        line the reader should be able to see.
        """
        flagged = {violation.index for violation in self.violations}
        size = self.x.size
        return pd.DataFrame(
            {
                "x": self.x,
                "y": self.y,
                "center": np.full(size, self.center),
                "ucl": np.full(size, self.upper),
                "lcl": np.full(size, self.lower),
                "zone_2_upper": np.full(size, self.center + 2.0 * self.sigma),
                "zone_2_lower": np.full(size, self.center - 2.0 * self.sigma),
                "zone_1_upper": np.full(size, self.center + self.sigma),
                "zone_1_lower": np.full(size, self.center - self.sigma),
                "violation": [int(i in flagged) for i in range(size)],
                # The flagged points as their own column, NULL elsewhere, so a
                # series can plot them alone. A WHERE clause would work for
                # apply, but the preview writes the same table and a second
                # query over it is a second scan for no gain.
                "violation_y": [
                    float(value) if index in flagged else None
                    for index, value in enumerate(self.y)
                ],
            }
        )


class SeriesControlChartDialog(SeriesOperationDialogBase):
    """Draw a Shewhart control chart for a chart series."""

    Name: str = "Control Chart"
    Description = "Monitor process stability"

    # The points are a time order, so they must be in x order: every estimator
    # here reads consecutive differences, and a shuffled series produces a
    # moving range that describes the sort order rather than the process.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    # A moving range needs a predecessor, and two points give one range - too
    # few to estimate anything, but enough not to crash. Nelson's run rules
    # want far more; the report says so when the series is short.
    INPUT_MINIMUM_POINTS = 3

    PARAMS = (
        IntParam(
            "subgroup",
            "Subgroup size:",
            tooltip=(
                "Consecutive points averaged into each plotted subgroup. "
                "Rational subgrouping: choose it so variation within a "
                "subgroup is only common-cause noise."
            ),
            default_value=5,
            minimum=2,
            maximum=25,
            visible_for={"model": SUBGROUPED},
        ),
        FloatParam(
            "sigma_limit",
            "Limits at sigma:",
            tooltip=(
                "3 is the Shewhart convention: on a stable normal process it "
                "gives about one false alarm per 370 points, which balances "
                "missed signals against chasing noise."
            ),
            default_value=3.0,
            minimum=1.0,
            maximum=6.0,
            decimals=2,
            step=0.5,
        ),
        BoolParam(
            "nelson",
            "Apply Nelson rules:",
            tooltip=(
                "Flags runs, trends and other patterns that stay inside the "
                "limits - the signals a limits-only chart misses."
            ),
            default_value=True,
        ),
        BoolParam(
            "draw_center",
            "Draw the centre line:",
            tooltip="The process mean the limits are built around.",
            default_value=True,
        ),
        BoolParam(
            "draw_limits",
            "Draw the control limits:",
            tooltip=(
                "The upper and lower limits. Without them the chart is a run "
                "chart - the points, but nothing to judge them against."
            ),
            default_value=True,
        ),
        BoolParam(
            "draw_zones",
            "Draw the one and two sigma lines:",
            tooltip=(
                "The A/B/C zones the run rules are phrased in: \"two of three "
                "beyond two sigma\" is a statement about a line worth seeing."
            ),
            default_value=False,
        ),
        BoolParam(
            "draw_violations",
            "Highlight the flagged points:",
            tooltip="Draws the signalling points again as separate markers.",
            default_value=True,
        ),
        BoolParam(
            "exclude_violations",
            "Exclude flagged points from the limits:",
            tooltip=(
                "Recomputes the limits without the points they flagged. Use "
                "only when the flagged points have an assigned cause you have "
                "removed; otherwise it hides the problem."
            ),
            default_value=False,
        ),
    )

    Icon = """
    <path d="M3 8h18"/>
    <path d="M3 16h18"/>
    <path d="M3 12h18" stroke-dasharray="2 2"/>
    <path d="M5 13l3-2 3 3 3-5 3 4 3-2"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesControlChartDialog requires a repository instance.")

        self._last_results: list[ControlChartResult] = []
        self._parameter_form: QFormLayout | None = None

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Control Chart",
            parent=parent,
            width=800,
            height=660,
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

        self.model_combo.addItems(CONTROL_CHARTS)
        self.model_combo.setToolTip(_("Choose the control chart."))
        form.addRow(_("Chart:"), self.model_combo)
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
        title, url = CONTROL_DOCS[self._chart()]
        set_doc_link(self._doc_link, title, url)

    def _chart(self) -> str:
        return self.model_combo.currentText() or CHART_INDIVIDUALS

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

    def compute_results(self) -> list[ControlChartResult]:
        chart = self._chart()
        params = self.parameter_values()

        results: list[ControlChartResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self._series_xy(row, name)
                results.append(self._build_chart(name, x_values, y_values, chart, params))
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message, show_dialog=False, raise_error=False)

        return results

    def _series_xy(self, row: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def _constants(size: int) -> tuple[tuple[float, ...], bool]:
        """Return the SPC constants for a subgroup size, and whether exact.

        Sizes between tabulated entries take the nearest smaller row rather
        than interpolating: the tables are what every other tool uses, and an
        interpolated d2 would put these limits subtly at odds with them.
        """
        if size in SPC_CONSTANTS:
            return SPC_CONSTANTS[size], True

        candidates = [n for n in SPC_CONSTANTS if n <= size]
        nearest = max(candidates) if candidates else min(SPC_CONSTANTS)
        return SPC_CONSTANTS[nearest], False

    def _build_chart(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        chart: str,
        params: Mapping[str, Any],
    ) -> ControlChartResult:
        sigma_limit = float(params.get("sigma_limit", 3.0))
        use_nelson = bool(params.get("nelson", True))

        if chart in SUBGROUPED:
            built = self._subgrouped_chart(x_values, y_values, chart, params, sigma_limit)
        elif chart == CHART_MOVING_RANGE:
            built = self._moving_range_chart(x_values, y_values, sigma_limit)
        else:
            built = self._individuals_chart(x_values, y_values, sigma_limit)

        plot_x, plot_y, center, upper, lower, sigma, subgroup, meta = built

        violations = self._find_violations(
            plot_y, plot_x, center, sigma, upper, lower, use_nelson
        )

        if bool(params.get("exclude_violations", False)) and violations:
            keep = np.ones(plot_y.size, dtype=bool)
            keep[[violation.index for violation in violations]] = False
            if int(keep.sum()) >= 3:
                # Recompute from the surviving points only. Deliberately a
                # second pass over the same estimator rather than a trimmed
                # sigma: the point is to exclude assignable causes, not to
                # make the estimator robust to them.
                sub_x, sub_y = plot_x[keep], plot_y[keep]
                if chart in SUBGROUPED:
                    center, upper, lower, sigma = self._limits_from_subgroup_stats(
                        sub_y, meta.get("dispersion_kept", sub_y), subgroup, sigma_limit, chart
                    )
                else:
                    center, upper, lower, sigma = self._limits_from_individuals(
                        sub_y, sigma_limit
                    )
                meta["excluded"] = int(plot_y.size - keep.sum())
                violations = self._find_violations(
                    plot_y, plot_x, center, sigma, upper, lower, use_nelson
                )
            else:
                applogger.warning(
                    f"{name}: too few points would remain after excluding the "
                    f"flagged ones; the limits use every point.",
                    show_dialog=False,
                    raise_error=False,
                )

        return ControlChartResult(
            source_name=name,
            result_name=f"{name} - {chart}",
            chart=chart,
            x=plot_x,
            y=plot_y,
            center=center,
            upper=upper,
            lower=lower,
            sigma=sigma,
            subgroup_size=subgroup,
            violations=violations,
            metadata=meta,
        )

    @staticmethod
    def _limits_from_individuals(
        values: np.ndarray,
        sigma_limit: float,
    ) -> tuple[float, float, float, float]:
        """Centre and limits for an individuals chart.

        Sigma comes from the average moving range over d2(2), NOT from the
        standard deviation of the values. A process that has shifted has a
        large overall standard deviation because it shifted, so limits built
        from it would be wide enough to swallow the shift.
        """
        moving_range = np.abs(np.diff(values))
        mean_range = float(np.mean(moving_range)) if moving_range.size else 0.0
        d2 = SPC_CONSTANTS[2][0]
        sigma = mean_range / d2 if d2 else 0.0
        center = float(np.mean(values))
        return center, center + sigma_limit * sigma, center - sigma_limit * sigma, sigma

    def _individuals_chart(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        sigma_limit: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float, int, dict[str, Any]]:
        center, upper, lower, sigma = self._limits_from_individuals(y_values, sigma_limit)
        return (
            x_values,
            y_values,
            center,
            upper,
            lower,
            sigma,
            1,
            {"estimator": "average moving range / d2(2)"},
        )

    def _moving_range_chart(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        sigma_limit: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float, int, dict[str, Any]]:
        moving_range = np.abs(np.diff(y_values))
        mean_range = float(np.mean(moving_range)) if moving_range.size else 0.0
        _d2, _d3, _c4, _a2, d3_limit, d4_limit, _b3, _b4 = SPC_CONSTANTS[2]

        # The MR chart uses D3/D4 rather than centre +/- k sigma: a range is
        # non-negative and its distribution is skewed, so symmetric limits
        # would put the lower one below zero and never signal.
        upper = d4_limit * mean_range
        lower = d3_limit * mean_range
        sigma = (upper - mean_range) / sigma_limit if sigma_limit else 0.0

        return (
            # One shorter than the source: the first point has no predecessor.
            x_values[1:],
            moving_range,
            mean_range,
            upper,
            lower,
            sigma,
            2,
            {"estimator": "D3/D4 on the moving range"},
        )

    def _limits_from_subgroup_stats(
        self,
        means: np.ndarray,
        dispersions: np.ndarray,
        size: int,
        sigma_limit: float,
        chart: str,
    ) -> tuple[float, float, float, float]:
        """Centre and limits for X-bar, from within-subgroup dispersion."""
        constants, _exact = self._constants(size)
        d2, _d3, c4, _a2, _d3l, _d4l, _b3, _b4 = constants

        center = float(np.mean(means))
        mean_dispersion = float(np.mean(dispersions)) if dispersions.size else 0.0

        if chart == CHART_XBAR_S:
            sigma = mean_dispersion / c4 if c4 else 0.0
        else:
            sigma = mean_dispersion / d2 if d2 else 0.0

        # The standard error of a subgroup mean, which is what the X-bar chart
        # plots - not sigma itself. Using sigma would give limits far too wide
        # and a chart that never signals.
        #
        # Derived from d2 rather than applied as the tabulated A2 shortcut,
        # because A2 has the 3 of "three sigma" baked into it and this dialog
        # lets the limit be set to something else. The two agree to about
        # 1e-3 of the limit - the difference is the rounding in the published
        # d2 and A2, not a disagreement about the method.
        standard_error = sigma / np.sqrt(size) if size else 0.0
        return (
            center,
            center + sigma_limit * standard_error,
            center - sigma_limit * standard_error,
            sigma,
        )

    def _subgrouped_chart(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        chart: str,
        params: Mapping[str, Any],
        sigma_limit: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float, int, dict[str, Any]]:
        size = max(2, int(params.get("subgroup", 5)))
        count = y_values.size // size
        if count < 2:
            raise ValueError(
                f"a subgroup size of {size} gives {count} subgroup(s); "
                f"at least 2 are needed"
            )

        used = count * size
        remainder = y_values.size - used
        grouped = y_values[:used].reshape(count, size)

        means = grouped.mean(axis=1)
        if chart == CHART_XBAR_S:
            # ddof=1: the within-subgroup standard deviation is an estimate
            # from a sample, and c4 is tabulated for the ddof=1 form.
            dispersions = grouped.std(axis=1, ddof=1)
        else:
            dispersions = grouped.max(axis=1) - grouped.min(axis=1)

        center, upper, lower, sigma = self._limits_from_subgroup_stats(
            means, dispersions, size, sigma_limit, chart
        )

        # One x per subgroup: the midpoint of the points it covers, so the
        # chart still lines up with the source's axis.
        subgroup_x = x_values[:used].reshape(count, size).mean(axis=1)

        _constants, exact = self._constants(size)
        meta: dict[str, Any] = {
            "estimator": (
                "average within-subgroup s / c4"
                if chart == CHART_XBAR_S
                else "average within-subgroup range / d2"
            ),
            "subgroups": count,
        }
        if remainder:
            meta["dropped"] = remainder
        if not exact:
            meta["constants"] = f"approximated for n={size}"

        return subgroup_x, means, center, upper, lower, sigma, size, meta

    # ------------------------------------------------------------------
    # Nelson rules
    # ------------------------------------------------------------------

    def _find_violations(
        self,
        values: np.ndarray,
        positions: np.ndarray,
        center: float,
        sigma: float,
        upper: float,
        lower: float,
        use_nelson: bool,
    ) -> list[Violation]:
        """Return every rule broken, most fundamental first.

        Rule 1 is always applied; the rest are the Nelson run rules, which
        catch what stays inside the limits. They are what makes a control
        chart more than an outlier test, and also why a chart of a stable
        process still shows the occasional flag: eight rules each with a
        false-alarm rate compound.
        """
        found: list[Violation] = []
        size = values.size

        def flag(index: int, rule: int, description: str) -> None:
            found.append(
                Violation(
                    index=int(index),
                    x=float(positions[index]),
                    y=float(values[index]),
                    rules=(rule,),
                    descriptions=(description,),
                )
            )

        # Rule 1: outside the control limits.
        for index in np.flatnonzero((values > upper) | (values < lower)):
            flag(index, 1, "beyond the control limits")

        if not use_nelson or sigma <= 0.0:
            return self._deduplicate(found)

        above = values > center
        below = values < center

        # Rule 2: nine in a row on one side of the centre - a shift.
        for start in range(size - 8):
            window = slice(start, start + 9)
            if above[window].all() or below[window].all():
                flag(start + 8, 2, "nine in a row on one side of the centre")

        # Rule 3: six in a row steadily increasing or decreasing - a trend.
        differences = np.diff(values)
        for start in range(size - 5):
            window = differences[start : start + 5]
            if window.size == 5 and ((window > 0).all() or (window < 0).all()):
                flag(start + 5, 3, "six in a row trending in one direction")

        # Rule 4: fourteen alternating up and down - overcontrol.
        if size >= 14:
            signs = np.sign(differences)
            for start in range(size - 13):
                window = signs[start : start + 13]
                if window.size == 13 and np.all(window[:-1] * window[1:] < 0):
                    flag(start + 13, 4, "fourteen alternating up and down")

        # Rule 5: two of three beyond two sigma, same side.
        two_sigma_up = center + 2.0 * sigma
        two_sigma_down = center - 2.0 * sigma
        for start in range(size - 2):
            window = values[start : start + 3]
            if (window > two_sigma_up).sum() >= 2 or (window < two_sigma_down).sum() >= 2:
                flag(start + 2, 5, "two of three beyond two sigma on one side")

        # Rule 6: four of five beyond one sigma, same side.
        one_sigma_up = center + sigma
        one_sigma_down = center - sigma
        for start in range(size - 4):
            window = values[start : start + 5]
            if (window > one_sigma_up).sum() >= 4 or (window < one_sigma_down).sum() >= 4:
                flag(start + 4, 6, "four of five beyond one sigma on one side")

        # Rule 7: fifteen in a row within one sigma - too good, which usually
        # means the limits are wrong or the data has been smoothed.
        within = (values < one_sigma_up) & (values > one_sigma_down)
        for start in range(size - 14):
            if within[start : start + 15].all():
                flag(start + 14, 7, "fifteen in a row hugging the centre line")

        # Rule 8: eight in a row all beyond one sigma, either side.
        outside = ~within
        for start in range(size - 7):
            if outside[start : start + 8].all():
                flag(start + 7, 8, "eight in a row beyond one sigma, both sides")

        return self._deduplicate(found)

    @staticmethod
    def _deduplicate(found: Sequence[Violation]) -> list[Violation]:
        """One entry per point, carrying every rule that point broke.

        Merged rather than filtered: a point is one thing to investigate, so
        it should be one row, but which rules it broke is exactly what tells
        you what to look for.
        """
        merged: dict[int, Violation] = {}
        for violation in found:
            current = merged.get(violation.index)
            if current is None:
                merged[violation.index] = violation
                continue

            rules = dict(zip(current.rules, current.descriptions))
            rules.update(zip(violation.rules, violation.descriptions))
            ordered = sorted(rules)
            merged[violation.index] = Violation(
                index=current.index,
                x=current.x,
                y=current.y,
                rules=tuple(ordered),
                descriptions=tuple(rules[rule] for rule in ordered),
            )

        return [merged[index] for index in sorted(merged)]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_to_frame(self, result: ControlChartResult) -> pd.DataFrame:
        return result.to_frame()

    #: Shared by every series this operation draws, so that
    #: remove_previous_generated_series takes the whole chart away rather than
    #: leaving orphaned limit lines behind when it is re-applied.
    def _series_style(self, result: ControlChartResult, **overrides: Any) -> dict[str, Any]:
        style: dict[str, Any] = {
            "generated_control_chart": True,
            "control_chart_dialog": "series_control_chart",
            "source_name": result.source_name,
            "chart": result.chart,
        }
        style.update(overrides)
        return style

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: ControlChartResult,
    ) -> ResultSeriesSpec:
        """The plotted values themselves - the series the chart is about."""
        del axis_id
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style=self._series_style(
                result,
                linestyle="-",
                linewidth=1.2,
                marker="o",
                markersize=4.0,
            ),
        )

    def result_series_specs(
        self,
        axis_id: int,
        table_name: str,
        result: ControlChartResult,
    ) -> Sequence[ResultSeriesSpec]:
        """The points, plus whichever reference lines are switched on.

        A control chart is the points *and* the lines: without limits it is a
        run chart, which shows the same numbers and answers a different
        question. Hence the lines default to on and this is not a one-series
        operation.

        Order matters to the drawing: the reference lines go first so the data
        is drawn over them, and the flagged points go last so they sit on top
        of everything.
        """
        params = self.parameter_values()
        specs: list[ResultSeriesSpec] = []

        def line(column: str, name: str, **style: Any) -> ResultSeriesSpec:
            return ResultSeriesSpec(
                name=f"{result.result_name} - {name}",
                # Aliased to y so the renderer's x/y roles need no special
                # case: every one of these is an ordinary two-column series.
                sql_query=f'SELECT x, {column} AS y FROM "{table_name}" ORDER BY x',
                roles={"x": "x", "y": "y"},
                style=self._series_style(result, marker="", **style),
            )

        if bool(params.get("draw_zones", False)):
            for column, label in (
                ("zone_2_upper", "+2s"),
                ("zone_2_lower", "-2s"),
                ("zone_1_upper", "+1s"),
                ("zone_1_lower", "-1s"),
            ):
                specs.append(
                    line(column, label, linestyle=":", linewidth=0.7, color="#9e9e9e")
                )

        if bool(params.get("draw_limits", True)):
            specs.append(
                line("ucl", "UCL", linestyle="--", linewidth=1.1, color="#c62828")
            )
            specs.append(
                line("lcl", "LCL", linestyle="--", linewidth=1.1, color="#c62828")
            )

        if bool(params.get("draw_center", True)):
            specs.append(
                line("center", "CL", linestyle="-", linewidth=1.0, color="#2e7d32")
            )

        specs.append(self.result_series_spec(axis_id, table_name, result))

        if bool(params.get("draw_violations", True)) and result.violations:
            specs.append(
                ResultSeriesSpec(
                    name=f"{result.result_name} - signals",
                    # violation_y is NULL for every point that did not signal,
                    # so this draws only the flagged ones without a second
                    # table or a WHERE clause the preview would have to repeat.
                    sql_query=(
                        f'SELECT x, violation_y AS y FROM "{table_name}" '
                        f"WHERE violation_y IS NOT NULL ORDER BY x"
                    ),
                    roles={"x": "x", "y": "y"},
                    style=self._series_style(
                        result,
                        linestyle="",
                        marker="o",
                        markersize=9.0,
                        color="#c62828",
                    ),
                )
            )

        return specs

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {
            "generated_control_chart": True,
            "control_chart_dialog": "series_control_chart",
        }

    def result_table_name(self, axis_id: int, result: ControlChartResult) -> str:
        return generated_table_name(
            f"ControlChart_axis{axis_id}_{result.source_name}_{result.chart}",
            fallback="ControlChart_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Control Chart"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[ControlChartResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        sections: list[str] = []
        for result in results:
            summary = report_html.summary_table(
                (
                    (_("Chart"), result.chart),
                    (_("Points plotted"), result.y.size),
                    (_("Subgroup size"), result.subgroup_size),
                    (_("Centre line"), report_html.format_number(result.center)),
                    (_("Upper control limit"), report_html.format_number(result.upper)),
                    (_("Lower control limit"), report_html.format_number(result.lower)),
                    (_("Sigma estimate"), report_html.format_number(result.sigma)),
                    (_("Estimator"), result.metadata.get("estimator", "")),
                )
                + tuple(
                    (key, value)
                    for key, value in result.metadata.items()
                    if key != "estimator"
                )
            )

            if result.violations:
                table = report_html.table(
                    (_("Point"), "x", "y", _("Rule"), _("Signal")),
                    [
                        (
                            str(violation.index + 1),
                            report_html.format_number(violation.x),
                            report_html.format_number(violation.y),
                            ", ".join(str(rule) for rule in violation.rules),
                            "; ".join(_(text) for text in violation.descriptions),
                        )
                        for violation in result.violations
                    ],
                    align=("right", "right", "right", "right", "left"),
                )
                verdict = report_html.note(
                    _(
                        "{count} of {total} points signal. The process is not "
                        "in statistical control."
                    ).format(count=len(result.violations), total=result.y.size)
                )
            else:
                table = ""
                verdict = report_html.note(
                    _(
                        "No points signal. The process is in statistical "
                        "control - which says it is stable, not that it meets "
                        "any specification."
                    )
                )

            sections.append(
                report_html.section(result.source_name, summary, verdict, table)
            )

        return report_html.document(_("Control Chart"), self._chart(), *sections)
