"""Shewhart control charts for a chart series - variables and attributes.

A control chart asks one question: is this process varying the way a stable
process varies, or has something changed?  It answers it by drawing limits
around the process's *own* variation and flagging the points that fall
outside.

Eight charts, in two families, because that is one decision a user should
not have to make before they can find the tool:

**Variables** - the measurement charts.  What makes these control charts
rather than a scatter plot with error bars is where sigma comes from.  It is
**never** the standard deviation of all the data: a process that has drifted
has a large overall standard deviation precisely *because* it drifted, so
limits built from it are wide enough to contain the drift and the chart
declares the process fine.  Sigma is estimated instead from variation
*within* subgroups - the average moving range for individual measurements,
the average range or standard deviation within subgroups otherwise - which
is unaffected by shifts between them.  That is the whole idea, and it is the
one thing easy to get wrong.

**Attributes** - the count charts, the half the variables charts do not
cover.  These plot a proportion or a count rather than a measurement, so the
limits come from the distribution the count follows rather than from a
within-subgroup spread:

* **p** - fraction defective, ``d / n``.  Binomial.  Centre ``p̄ = Σd / Σn``,
  limits ``p̄ ± L·√(p̄(1-p̄)/nᵢ)``: when the sample size varies, so does the
  limit, and the chart draws a different band at every point.
* **np** - number defective, ``d``, at a fixed ``n``.  Centre ``n·p̄``,
  limits ``n·p̄ ± L·√(n·p̄(1-p̄))``.
* **c** - defects in a unit of constant size.  Poisson.  ``c̄ ± L·√c̄``.
* **u** - defects per unit, ``c / n``.  Poisson.  ``ū ± L·√(ū/nᵢ)`` - again
  per point when ``n`` varies.

The lower limit is clipped at zero on the count charts: a count cannot be
negative, and an unclipped LCL below zero never signals.

The variables estimators need the unbiasing constants d2, d3, c4, A2, D3,
D4, B3 and B4.  They are tabulated below rather than computed: c4 has a
closed form in gamma functions, but d2 and d3 are integrals over the range
distribution with no elementary form, and every SPC text ships the same
table.  Using anything else would put this chart's limits at odds with every
other tool's.

Violations are reported by the Nelson rules, which catch the patterns that
stay inside the limits - a run on one side, a trend, a hug of the centre
line - and are what a chart is for beyond spotting the obvious outlier.
Which rules are *legal* depends on the chart, and that is the one thing the
two families have to disagree about: see :meth:`_find_violations`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QComboBox, QFormLayout, QVBoxLayout, QWidget

from app.data.data_source import parse_roles, row_value
from app.data.repo.tables import QueryColumns, coerce_numeric_array
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.parameter_spec import BoolParam, FloatParam, IntParam
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.styles.style import create_doc_link, set_doc_link
from app.utils.i18n import _
from app.utils import report_html

# Variables charts - a measurement per point.
CHART_INDIVIDUALS = "Individuals (I-MR)"
CHART_MOVING_RANGE = "Moving range (MR)"
CHART_XBAR_R = "X-bar and R"
CHART_XBAR_S = "X-bar and S"

# Attribute charts - a count per point.
CHART_P = "p (fraction defective)"
CHART_NP = "np (count defective)"
CHART_C = "c (defects per unit)"
CHART_U = "u (defects per unit, variable size)"

VARIABLES_CHARTS = (
    CHART_INDIVIDUALS,
    CHART_MOVING_RANGE,
    CHART_XBAR_R,
    CHART_XBAR_S,
)
ATTRIBUTE_CHARTS = (CHART_P, CHART_NP, CHART_C, CHART_U)
CONTROL_CHARTS = VARIABLES_CHARTS + ATTRIBUTE_CHARTS

#: Charts averaging several readings into each plotted point.
SUBGROUPED = (CHART_XBAR_R, CHART_XBAR_S)
#: Charts that read a sample-size column.  ``c`` is the exception among the
#: attribute charts - it assumes a constant area of opportunity, so there is
#: nothing to divide by.
NEEDS_SIZE_COLUMN = frozenset({CHART_P, CHART_NP, CHART_U})
#: Charts on the binomial (a proportion, bounded above by its sample size).
BINOMIAL = frozenset({CHART_P, CHART_NP})

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
    CHART_P: ("p-chart", "https://en.wikipedia.org/wiki/P-chart"),
    CHART_NP: ("np-chart", "https://en.wikipedia.org/wiki/Np-chart"),
    CHART_C: ("c-chart", "https://en.wikipedia.org/wiki/C-chart"),
    CHART_U: ("u-chart", "https://en.wikipedia.org/wiki/U-chart"),
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

#: Below this an attribute chart's limits are too soft to trust; the report
#: says so rather than refusing to draw them.
RECOMMENDED_SUBGROUPS = 20


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
    """One control chart - of either family - for one source series.

    ``upper``/``lower``/``sigma`` are scalars: the constant value on every
    variables chart and on np/c, and the *mean* band on a p or u chart whose
    limits genuinely move with the sample size - kept scalar rather than
    dropped so the many existing checks against a variables chart's own
    limits (``result.upper == ...``) still read the single number they
    always meant. ``*_band`` are the same thing per point, always populated
    (a repeated value where the chart has no reason to vary): ``to_frame``
    and the violation pass need the real band, not its average, and only a
    p/u chart's is not just that scalar broadcast.
    """

    source_name: str
    result_name: str
    chart: str
    x: np.ndarray
    y: np.ndarray
    center: float
    upper: float
    lower: float
    sigma: float
    upper_band: np.ndarray
    lower_band: np.ndarray
    sigma_band: np.ndarray
    subgroup_size: int
    violations: list[Violation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def limits_vary(self) -> bool:
        """True when the band moves point to point (a p or u chart on an
        uneven sample size), which is what makes the zone rules illegal."""
        return bool(self.sigma_band.size and np.ptp(self.sigma_band) > 0.0)

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
                "ucl": self.upper_band,
                "lcl": self.lower_band,
                "zone_2_upper": self.center + 2.0 * self.sigma_band,
                "zone_2_lower": self.center - 2.0 * self.sigma_band,
                "zone_1_upper": self.center + self.sigma_band,
                "zone_1_lower": self.center - self.sigma_band,
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


# ----------------------------------------------------------------------
# The attribute numerics - pure, no Qt, so a test can call them directly
# ----------------------------------------------------------------------
def attribute_limits(
    chart: str,
    counts: np.ndarray,
    sizes: np.ndarray,
    sigma_limit: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(plotted_statistic, centre, upper, lower, metadata)``.

    ``counts`` is the series' y (defectives for p/np, defects for c/u).
    ``sizes`` is the sample size per point; the c chart ignores it.
    """
    counts = np.asarray(counts, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    point_count = counts.size
    meta: dict[str, Any] = {}

    if chart == CHART_P:
        total_n = float(sizes.sum())
        pbar = float(counts.sum() / total_n) if total_n else 0.0
        statistic = np.divide(counts, sizes, out=np.zeros_like(counts), where=sizes > 0)
        spread = sigma_limit * np.sqrt(
            np.divide(pbar * (1.0 - pbar), sizes, out=np.zeros_like(sizes), where=sizes > 0)
        )
        center, upper, lower = pbar, pbar + spread, pbar - spread
        meta["p-bar"] = pbar

    elif chart == CHART_NP:
        n = float(np.mean(sizes)) if sizes.size else 0.0
        pbar = float(counts.sum() / (n * point_count)) if n and point_count else 0.0
        statistic = counts
        spread = sigma_limit * np.sqrt(max(n * pbar * (1.0 - pbar), 0.0))
        center = n * pbar
        upper = np.full(point_count, center + spread)
        lower = np.full(point_count, center - spread)
        meta["p-bar"] = pbar
        meta["sample size"] = n

    elif chart == CHART_C:
        cbar = float(np.mean(counts)) if counts.size else 0.0
        statistic = counts
        spread = sigma_limit * np.sqrt(max(cbar, 0.0))
        center = cbar
        upper = np.full(point_count, cbar + spread)
        lower = np.full(point_count, cbar - spread)
        meta["c-bar"] = cbar

    elif chart == CHART_U:
        total_n = float(sizes.sum())
        ubar = float(counts.sum() / total_n) if total_n else 0.0
        statistic = np.divide(counts, sizes, out=np.zeros_like(counts), where=sizes > 0)
        spread = sigma_limit * np.sqrt(
            np.divide(ubar, sizes, out=np.zeros_like(sizes), where=sizes > 0)
        )
        center, upper, lower = ubar, ubar + spread, ubar - spread
        meta["u-bar"] = ubar

    else:  # pragma: no cover - the combo cannot hold anything else
        raise ValueError(f"unknown attribute chart {chart!r}")

    upper = np.broadcast_to(np.asarray(upper, dtype=float), (point_count,)).copy()
    lower = np.broadcast_to(np.asarray(lower, dtype=float), (point_count,)).copy()
    # A count is non-negative; an LCL below zero would never signal.
    np.clip(lower, 0.0, None, out=lower)
    # A proportion is bounded above by 1.
    if chart == CHART_P:
        np.clip(upper, None, 1.0, out=upper)
    return np.asarray(statistic, dtype=float), float(center), upper, lower, meta


class SeriesControlChartDialog(SeriesOperationDialogBase):
    """Draw a Shewhart control chart - variables or attributes - for a series."""

    Name: str = "Control Chart"
    Description = "Monitor process stability (I-MR, X-bar, p, np, c, u)"

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
                "limits - the signals a limits-only chart misses. The zone "
                "rules switch themselves off on a chart whose limits move."
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
                "Recomputes the limits without the points they flagged - the "
                "trial-limits-then-revised-limits pass every SPC text runs. "
                "Use only when the flagged points have an assigned cause you "
                "have removed; otherwise it hides the problem."
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
            height=680,
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

        self.model_combo.addItems(VARIABLES_CHARTS)
        # A separator, not two combos or a family radio above them: picking a
        # chart is one decision, and a user who knows they have counts rather
        # than measurements should not have to say so twice.
        self.model_combo.insertSeparator(self.model_combo.count())
        self.model_combo.addItems(ATTRIBUTE_CHARTS)
        self.model_combo.setToolTip(
            _("Measurements above the line, counts below it.")
        )
        form.addRow(_("Chart:"), self.model_combo)
        form.addRow(_("Docs:"), self._doc_link)

        layout.addWidget(container)
        return panel

    def build_parameter_selector(self) -> QWidget:
        """The declared parameters, plus the one that cannot be declared.

        Everything in ``PARAMS`` is built by the base class and shown or
        hidden per chart by ``visible_for``.  The sample-size column is the
        exception: its choices are the selected series' own columns, which
        are not known until a series is picked, so it is an ordinary combo
        appended to the same form and gated by hand in ``_refresh_visibility``.
        """
        widget = super().build_parameter_selector()

        self._size_column_combo.setToolTip(
            _("The column holding the sample size (units inspected) at each point.")
        )
        if self._parameter_form is not None:
            self._parameter_form.addRow(
                _("Sample size column:"), self._size_column_combo
            )
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

    def _refresh_visibility(self) -> None:
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()
        self.set_row_visible(
            self._size_column_combo, self._chart() in NEEDS_SIZE_COLUMN
        )
        title, url = CONTROL_DOCS.get(self._chart(), ("", ""))
        set_doc_link(self._doc_link, title, url)

    def _refresh_size_columns(self) -> None:
        """Fill the sample-size combo with the selected series' columns."""
        wanted = self._size_column_combo.currentText()
        columns: list[str] = []
        for row in self.selected_series():
            result = self._safe_columns(row)
            if result is None:
                continue
            for column in result.columns:
                if column not in columns:
                    columns.append(column)

        self._size_column_combo.blockSignals(True)
        self._size_column_combo.clear()
        self._size_column_combo.addItems(columns)
        if wanted in columns:
            self._size_column_combo.setCurrentText(wanted)
        elif "n" in columns:
            self._size_column_combo.setCurrentText("n")
        self._size_column_combo.blockSignals(False)

    def _chart(self) -> str:
        return self.model_combo.currentText() or CHART_INDIVIDUALS

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
    def _safe_columns(self, row: Any) -> QueryColumns | None:
        query = str(row_value(row, "sql_query", "query", default="")).strip()
        if not query:
            return None
        try:
            return self._repo.query_arrays(query)
        except Exception:  # noqa: BLE001 - a bad query is reported at compute time
            return None

    @staticmethod
    def _looks_numeric(values: np.ndarray) -> bool:
        """True when coercing *values* produces at least one real number.

        The fallback-column heuristic below needs to tell "this column is
        usable as counts" from "this column is text" without pandas' own
        dtype introspection (there is no DataFrame here to introspect) - a
        column that coerces to all-NaN is exactly the text case.
        """
        coerced = coerce_numeric_array(values)
        return bool(coerced.size) and not bool(np.all(np.isnan(coerced)))

    def _series_counts(
        self, row: Any, name: str, chart: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(x, counts, sizes)`` for one series, aligned and finite.

        Only the attribute charts come through here; the variables charts
        read x/y through the shared ``series_xy``, which they can because
        they need no third column. Read through query_arrays rather than
        query_df: no DataFrame is built for a dialog that only ever wanted
        two or three plain numeric columns out of it. The one column that
        might be a timestamp - x - is still handed to numeric_x exactly as
        it always was, wrapped in a single-column Series rather than a
        whole frame, because that is real date-parsing logic this is not
        trying to reimplement.
        """
        columns_result = self._safe_columns(row)
        if columns_result is None or columns_result.empty:
            raise ValueError("the series query returned no rows")

        roles = parse_roles(row_value(row, "roles", default={}))
        columns = list(columns_result.columns)

        y_col = str(roles.get("y") or "y")
        if y_col not in columns:
            numeric = [c for c in columns if self._looks_numeric(columns_result[c])]
            y_col = numeric[-1] if numeric else columns[-1]
        counts = coerce_numeric_array(columns_result[y_col])

        x_col = str(roles.get("x") or "x")
        x_values = (
            self.numeric_x(pd.Series(columns_result[x_col]), name)
            if x_col in columns
            else np.arange(counts.size, dtype=float)
        )

        if chart in NEEDS_SIZE_COLUMN:
            size_col = self._size_column_combo.currentText().strip()
            size_col = size_col or str(roles.get("n") or roles.get("sample_size") or "")
            if size_col not in columns:
                raise ValueError(
                    "pick the column holding the sample size in the parameters pane"
                )
            sizes = coerce_numeric_array(columns_result[size_col])
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
        if chart in BINOMIAL and np.any(counts > sizes):
            raise ValueError("a subgroup has more defectives than its sample size")

        order = np.argsort(x_values, kind="stable")
        return x_values[order], counts[order], sizes[order]

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
                if chart in ATTRIBUTE_CHARTS:
                    x_values, counts, sizes = self._series_counts(row, name, chart)
                    results.append(
                        self._build_attribute_chart(
                            name, x_values, counts, sizes, chart, params
                        )
                    )
                else:
                    x_values, y_values = self.series_xy(row, name)
                    results.append(
                        self._build_chart(name, x_values, y_values, chart, params)
                    )
            except Exception as exc:  # noqa: BLE001 - collected, then reported
                errors.append(f"{name}: {exc}")

        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message, show_dialog=False, raise_error=False)

        return results

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

    # -- Attribute family ----------------------------------------------

    def _build_attribute_chart(
        self,
        name: str,
        x_values: np.ndarray,
        counts: np.ndarray,
        sizes: np.ndarray,
        chart: str,
        params: Mapping[str, Any],
    ) -> ControlChartResult:
        sigma_limit = float(params.get("sigma_limit", 3.0))
        use_nelson = bool(params.get("nelson", True))

        statistic, center, upper_band, lower_band, meta = attribute_limits(
            chart, counts, sizes, sigma_limit
        )
        sigma_band = self._sigma_from_limits(upper_band, center, sigma_limit)
        violations = self._find_violations(
            statistic, x_values, center, sigma_band, upper_band, lower_band, use_nelson
        )

        if bool(params.get("exclude_violations", False)) and violations:
            keep = np.ones(statistic.size, dtype=bool)
            keep[[violation.index for violation in violations]] = False
            if int(keep.sum()) >= 3:
                # Trial limits, then revised limits - the second pass every
                # SPC text runs once an assignable cause has been removed.
                _kept, center, upper_kept, lower_kept, meta = attribute_limits(
                    chart, counts[keep], sizes[keep], sigma_limit
                )
                # The revised centre is recomputed from the kept points, but
                # the band is still drawn against every point's own sample
                # size, so it has to be rebuilt at full length rather than
                # carried over from the shorter pass.
                upper_band, lower_band = self._rebuild_attribute_band(
                    chart, center, sizes, sigma_limit
                )
                del upper_kept, lower_kept
                sigma_band = self._sigma_from_limits(upper_band, center, sigma_limit)
                meta["excluded"] = int(statistic.size - keep.sum())
                violations = self._find_violations(
                    statistic, x_values, center, sigma_band, upper_band, lower_band,
                    use_nelson,
                )
            else:
                applogger.warning(
                    f"{name}: too few points would remain after excluding the "
                    f"flagged ones; the limits use every point.",
                    show_dialog=False,
                    raise_error=False,
                )

        if x_values.size < RECOMMENDED_SUBGROUPS:
            meta["note"] = (
                f"only {x_values.size} subgroups; "
                f"{RECOMMENDED_SUBGROUPS}+ give trustworthy limits"
            )
        if chart in NEEDS_SIZE_COLUMN:
            meta["mean sample size"] = float(np.mean(sizes))
            if float(sizes.min()) != float(sizes.max()):
                meta["limits"] = "vary with the sample size"

        return ControlChartResult(
            source_name=name,
            result_name=f"{name} - {chart.split(' ')[0]}",
            chart=chart,
            x=x_values,
            y=statistic,
            center=center,
            upper=float(np.mean(upper_band)),
            lower=float(np.mean(lower_band)),
            sigma=float(np.mean(sigma_band)),
            upper_band=upper_band,
            lower_band=lower_band,
            sigma_band=sigma_band,
            subgroup_size=1,
            violations=violations,
            metadata=meta,
        )

    @staticmethod
    def _rebuild_attribute_band(
        chart: str, center: float, sizes: np.ndarray, sigma_limit: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The band around a revised centre, at every point's own sample size."""
        if chart in (CHART_P,):
            spread = sigma_limit * np.sqrt(
                np.divide(
                    center * (1.0 - center), sizes,
                    out=np.zeros_like(sizes), where=sizes > 0,
                )
            )
        elif chart == CHART_U:
            spread = sigma_limit * np.sqrt(
                np.divide(center, sizes, out=np.zeros_like(sizes), where=sizes > 0)
            )
        elif chart == CHART_NP:
            n = float(np.mean(sizes)) if sizes.size else 0.0
            p = center / n if n else 0.0
            spread = np.full(
                sizes.size, sigma_limit * np.sqrt(max(n * p * (1.0 - p), 0.0))
            )
        else:  # CHART_C
            spread = np.full(sizes.size, sigma_limit * np.sqrt(max(center, 0.0)))

        upper = center + spread
        lower = np.clip(center - spread, 0.0, None)
        if chart == CHART_P:
            upper = np.clip(upper, None, 1.0)
        return upper, lower

    @staticmethod
    def _sigma_from_limits(
        upper: np.ndarray, center: float, sigma_limit: float,
    ) -> np.ndarray:
        """Back out the per-point sigma the band was drawn at.

        The attribute formulas produce a band directly rather than a sigma,
        but the zone lines and the zone rules are both phrased in sigma, so
        it is recovered here rather than special-cased in four places.
        """
        if sigma_limit <= 0:
            return np.zeros_like(upper)
        return (np.asarray(upper, dtype=float) - center) / sigma_limit

    # -- Variables family ----------------------------------------------

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
        upper_band = np.full(plot_y.size, upper)
        lower_band = np.full(plot_y.size, lower)
        sigma_band = np.full(plot_y.size, sigma)

        violations = self._find_violations(
            plot_y, plot_x, center, sigma_band, upper_band, lower_band, use_nelson
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
                upper_band = np.full(plot_y.size, upper)
                lower_band = np.full(plot_y.size, lower)
                sigma_band = np.full(plot_y.size, sigma)
                meta["excluded"] = int(plot_y.size - keep.sum())
                violations = self._find_violations(
                    plot_y, plot_x, center, sigma_band, upper_band, lower_band, use_nelson
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
            upper_band=upper_band,
            lower_band=lower_band,
            sigma_band=sigma_band,
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
        sigma: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray,
        use_nelson: bool,
    ) -> list[Violation]:
        """Return every rule broken, most fundamental first.

        Rule 1 is always applied; the rest are the Nelson run rules, which
        catch what stays inside the limits. They are what makes a control
        chart more than an outlier test, and also why a chart of a stable
        process still shows the occasional flag: eight rules each with a
        false-alarm rate compound.

        Which of them are *legal* depends on the chart. Rules 2-4 read only
        the values and the centre line, so they hold on any chart. Rules 5-8
        are phrased in equal-width one- and two-sigma zones, which a chart
        whose sigma moves point to point - a p or u chart on an uneven sample
        size - does not have; applying them there would invent a zone
        boundary per point and flag patterns that mean nothing. So they are
        skipped exactly when sigma is not constant, which is also why sigma
        is carried as an array rather than a scalar.
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

        if not use_nelson or not sigma.size or float(np.max(sigma)) <= 0.0:
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

        # Rules 5-8 need equal-width zones; see the docstring.
        if float(np.ptp(sigma)) > 0.0:
            return self._deduplicate(found)
        constant_sigma = float(sigma[0])

        # Rule 5: two of three beyond two sigma, same side.
        two_sigma_up = center + 2.0 * constant_sigma
        two_sigma_down = center - 2.0 * constant_sigma
        for start in range(size - 2):
            window = values[start : start + 3]
            if (window > two_sigma_up).sum() >= 2 or (window < two_sigma_down).sum() >= 2:
                flag(start + 2, 5, "two of three beyond two sigma on one side")

        # Rule 6: four of five beyond one sigma, same side.
        one_sigma_up = center + constant_sigma
        one_sigma_down = center - constant_sigma
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
        of everything. On a p or u chart the limit columns step from point to
        point rather than being flat, which is the honest picture of a band
        that really does move with the sample size.
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
            varies = result.limits_vary
            summary_rows: list[tuple[str, Any]] = [
                (_("Chart"), result.chart),
                (_("Points plotted"), result.y.size),
            ]
            if result.chart not in ATTRIBUTE_CHARTS:
                summary_rows.append((_("Subgroup size"), result.subgroup_size))
            summary_rows.append(
                (_("Centre line"), report_html.format_number(result.center))
            )
            summary_rows += [
                (
                    _("Mean upper limit") if varies else _("Upper control limit"),
                    report_html.format_number(result.upper),
                ),
                (
                    _("Mean lower limit") if varies else _("Lower control limit"),
                    report_html.format_number(result.lower),
                ),
                (_("Sigma estimate"), report_html.format_number(result.sigma)),
                (_("Estimator"), result.metadata.get("estimator", "")),
            ]
            summary_rows += [
                (str(key), report_html.format_number(value)
                 if isinstance(value, float) else value)
                for key, value in result.metadata.items()
                if key != "estimator"
            ]
            summary = report_html.summary_table(summary_rows)

            if varies:
                zones = report_html.note(
                    _(
                        "The limits move with the sample size, so the zone "
                        "rules (Nelson 5-8) do not apply and were not run."
                    )
                )
            else:
                zones = ""

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
                report_html.section(result.source_name, summary, zones, verdict, table)
            )

        return report_html.document(_("Control Chart"), self._chart(), *sections)
