"""Read-only statistics dialog for selected chart series.

The dialog inherits the shared SeriesOperationDialogBase shell. It does not
create generated chart series and does not write result tables. It reports:

- descriptive statistics for every checked series;
- one-sample tests for every checked series;
- paired-sample tests for every pair when multiple series are checked;
- optional normality and correlation sections, controlled by the model combo.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import html
import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.data.data_source import quote_identifier
from app.data.data_source import row_value,parse_roles
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.series_operation_dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
)
from app.styles.style import create_card_widget, stdSizeAndlayout
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.widgets import report_html
from app.utils.distribution_fit import (
    CURATED_DISTRIBUTIONS,
    DEFAULT_RANK,
    DistributionFit,
    fit_distributions,
)
from app.utils.i18n import _

_SERIES_ROLE = Qt.ItemDataRole.UserRole


@dataclass(slots=True)
class SeriesStatsSample:
    """One selected series converted to finite numeric arrays."""

    name: str
    x: np.ndarray
    y: np.ndarray
    rowids: np.ndarray


@dataclass(slots=True)
class SeriesStatsResult:
    """Complete statistics result for one dialog run."""

    samples: list[SeriesStatsSample]
    popmean: float
    alternative: str
    model: str
    trim_percent: float
    distribution: str = "best"
    exhaustive: bool = False
    rank_by: str = DEFAULT_RANK


class SeriesStatisticsDialog(SeriesOperationDialogBase):
    """Read-only statistical summary and hypothesis-test dialog.

    The model selector controls which sections are shown:
    - all: descriptive, one-sample, normality, paired, and correlation results;
    - descriptive: summary statistics only;
    - one_sample: one-sample tests only;
    - normality: normality/shape tests only;
    - paired: paired-sample tests only;
    - correlation: paired correlation/association tests only.
    """
    Name: str = "Statistics"
    Description = "Compute metrics"

    Icon = """
    <path d="M5 19h14"/>
    <path d="M7 16v-5"/>
    <path d="M12 16V7"/>
    <path d="M17 16v-8"/>
    <path d="M6 5h12"/>
    """
    def __init__(
        self,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        # Set before super(): the base constructor calls the builder hooks,
        # which reach these.
        #
        # The axis this dialog adds when asked to chart its results: created on
        # the first Preview, reused afterwards, removed on Close unless Apply
        # confirmed it.
        self._result_axis_id: int | None = None
        self._result_chart_type: str = ""
        self._result_axis_options: dict[str, Any] = {}
        self._applied = False
        self._fit_cache: dict[
            tuple[bytes, int, bool, str], list[DistributionFit]
        ] = {}

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Statistics",
            parent=parent,
            width=1060,
            height=720,
        )
        self.set_doc_link(
            "SciPy statistics",
            "https://docs.scipy.org/doc/scipy/reference/stats.html#summary-statistics",
        )

        # Statistics is most useful as a multi-series operation.  Start with all
        # series on the current axis checked so the All/Paired models immediately
        # show paired-sample results when more than one series exists.
        self.series_selector.select_all_series()
        self._sync_model_controls()
        self.refresh_results()

    # ------------------------------------------------------------------
    # UI hooks
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        self.model_combo.clear()
        self.model_combo.addItem(_("All statistics"), "all")
        self.model_combo.addItem(_("Descriptive statistics"), "descriptive")
        self.model_combo.addItem(_("One-sample tests"), "one_sample")
        self.model_combo.addItem(_("Normality / shape tests"), "normality")
        self.model_combo.addItem(_("Paired-sample tests"), "paired")
        self.model_combo.addItem(_("Correlation / association"), "correlation")
        self.model_combo.addItem(_("Distribution fit"), "distribution")
        self.model_combo.setToolTip(_("Choose which statistics section to display."))

        # Whether applying also draws a chart, and which one.  Off by default:
        # the dialog is a report, and a chart appearing unasked in the figure
        # is a side effect the user did not request.
        self.create_chart_check = QCheckBox(_("Add a chart"), self)
        self.create_chart_check.setChecked(False)
        self.create_chart_check.setToolTip(
            _("Add an axis to the current figure showing the analysed series. "
            "Preview draws it; closing without applying removes it again.")
        )

        # Filled by _sync_model_controls, because what a chart can usefully
        # show depends on which statistics were asked for.
        self.chart_type_combo = QComboBox(self)
        self.chart_type_combo.setEnabled(False)
        self.chart_type_combo.setToolTip(
            _("Which chart to add. 'Match the model' picks the one that shows "
            "what the selected statistics measure.")
        )

        # Which distribution the report ranks first and the histogram draws.
        # "Best fit" rather than a fixed default: the point of the sweep is
        # that the answer is not known in advance.
        # Also filled per model: the normality model offers the two families
        # its tests are about, the distribution model the whole sweep.
        self.distribution_combo = QComboBox(self)
        self.distribution_combo.setToolTip(
            _("Which fitted distribution the histogram draws. Best fit follows "
            "whichever candidate ranks first for the series.")
        )

        self.rank_combo = QComboBox(self)
        self.rank_combo.addItem(_("AIC (penalises free parameters)"), "aic")
        self.rank_combo.addItem(_("BIC (penalises them harder)"), "bic")
        self.rank_combo.addItem(_("KS statistic (closest curve)"), "ks")
        self.rank_combo.setToolTip(
            _("How the candidates are ordered. AIC is the default because the "
            "KS statistic alone always favours the distributions with the most "
            "free parameters, whatever the data really is.")
        )

        self.exhaustive_check = QCheckBox(_("Try every SciPy distribution"), self)
        self.exhaustive_check.setChecked(False)
        self.exhaustive_check.setToolTip(
            _("Fit all ~100 continuous distributions instead of the 15 common "
            "ones. Considerably slower - seconds per refresh - and many of the "
            "extra candidates fit no real measurement.")
        )

        self.popmean_spin = QDoubleSpinBox(self)
        self.popmean_spin.setDecimals(8)
        self.popmean_spin.setRange(-1.0e12, 1.0e12)
        self.popmean_spin.setValue(0.0)
        self.popmean_spin.setToolTip(_("Reference value for one-sample location tests."))

        self.alternative_combo = QComboBox(self)
        self.alternative_combo.addItem(_("Two-sided"), "two-sided")
        self.alternative_combo.addItem(_("Greater"), "greater")
        self.alternative_combo.addItem(_("Less"), "less")
        self.alternative_combo.setToolTip(_("Alternative hypothesis for supported tests."))

        self.trim_percent_spin = QSpinBox(self)
        self.trim_percent_spin.setRange(0, 40)
        self.trim_percent_spin.setValue(10)
        self.trim_percent_spin.setSuffix(" %")
        self.trim_percent_spin.setToolTip(_("Trim percentage used for trimmed mean."))

        self.anderson_method_combo = QComboBox(self)
        self.anderson_method_combo.addItem(_("Anderson p-value: interpolated"), "interpolate")
        self.anderson_method_combo.addItem(_("Anderson p-value: Monte Carlo"), "monte_carlo")
        self.anderson_method_combo.setToolTip(
            _("Choose how Anderson-Darling p-values are calculated when supported by SciPy.")
        )

        self.monte_carlo_resamples_spin = QSpinBox(self)
        self.monte_carlo_resamples_spin.setRange(100, 1_000_000)
        self.monte_carlo_resamples_spin.setSingleStep(1000)
        self.monte_carlo_resamples_spin.setValue(9999)
        self.monte_carlo_resamples_spin.setToolTip(
            _("Number of Monte Carlo resamples for Anderson-Darling p-values.")
        )

        self.monte_carlo_batch_spin = QSpinBox(self)
        self.monte_carlo_batch_spin.setRange(0, 1_000_000)
        self.monte_carlo_batch_spin.setSingleStep(1000)
        self.monte_carlo_batch_spin.setValue(0)
        self.monte_carlo_batch_spin.setSpecialValueText(_("Auto"))
        self.monte_carlo_batch_spin.setToolTip(
            _("Batch size for Monte Carlo resampling. Auto lets SciPy choose.")
        )

    def build_model_selector(self) -> QWidget:
        panel = create_card_widget(self, "statisticsModelCard")
        layout = QVBoxLayout(panel)
        stdSizeAndlayout(layout)
        layout.addWidget(self.model_combo)
        note = QLabel(
            _("Check one or more series. One-sample tests are calculated for "
            "each checked series. Paired tests are calculated for every pair "
            "when two or more series are checked."),
            panel,
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        layout.addWidget(note)

        # Beside the model rather than among the parameters: what to draw is
        # part of choosing what to compute, and the parameters below are the
        # knobs of whichever model is chosen here.
        chart_form = QFormLayout()
        stdSizeAndlayout(chart_form)
        chart_form.addRow(self.create_chart_check)
        self._chart_label = QLabel(_("Chart:"), panel)
        chart_form.addRow(self._chart_label, self.chart_type_combo)
        layout.addLayout(chart_form)
        return panel

    def build_parameter_selector(self) -> QWidget:
        panel = create_card_widget(self, "statisticsParametersCard")
        form = QFormLayout(panel)
        stdSizeAndlayout(form)
        form.addRow(_("Reference value:"), self.popmean_spin)
        form.addRow(_("Alternative:"), self.alternative_combo)
        form.addRow(_("Trimmed mean:"), self.trim_percent_spin)
        form.addRow(_("Anderson method:"), self.anderson_method_combo)
        form.addRow(_("MC resamples:"), self.monte_carlo_resamples_spin)
        form.addRow(_("MC batch:"), self.monte_carlo_batch_spin)
        self._rank_label = QLabel(_("Rank by:"), panel)
        form.addRow(self._rank_label, self.rank_combo)
        self._distribution_label = QLabel(_("Distribution:"), panel)
        form.addRow(self._distribution_label, self.distribution_combo)
        form.addRow(self.exhaustive_check)
        self._parameter_form = form
        return panel

    def _sync_model_controls(self) -> None:
        """Show and fill the chart controls the selected model can use.

        Hidden rather than disabled when a model has no chart: "all" prints
        every section at once and no single picture illustrates that, and a
        greyed-out control invites the user to hunt for what would enable it.

        The current choice is preserved across the refill where the new model
        still offers it, so moving between two models that both draw a
        histogram does not silently reset the chart to something else.
        """
        model = str(self.model_combo.currentData() or "all")

        charts = self.CHARTS_BY_MODEL.get(model, ())
        has_chart = bool(charts)
        for widget in (self.create_chart_check, self._chart_label, self.chart_type_combo):
            widget.setVisible(has_chart)

        previous = str(self.chart_type_combo.currentData() or "")
        self.chart_type_combo.clear()
        for chart_type, _title in charts:
            self.chart_type_combo.addItem(_(chart_type), chart_type)
        restored = self.chart_type_combo.findData(previous)
        self.chart_type_combo.setCurrentIndex(max(0, restored))

        # The distribution choice only means something where a curve is drawn.
        distributions = self.DISTRIBUTIONS_BY_MODEL.get(model, ())
        if not distributions and has_chart:
            distributions = self.DEFAULT_DISTRIBUTIONS
        for widget in (self._distribution_label, self.distribution_combo):
            widget.setVisible(bool(distributions))

        previous = str(self.distribution_combo.currentData() or "")
        self.distribution_combo.clear()
        for value, label in distributions:
            self.distribution_combo.addItem(_(label), value)
        restored = self.distribution_combo.findData(previous)
        self.distribution_combo.setCurrentIndex(max(0, restored))

        # Ranking and the exhaustive sweep only exist to order the candidates,
        # so they belong to the one model that ranks them.
        ranks = model == "distribution"
        for widget in (self._rank_label, self.rank_combo, self.exhaustive_check):
            widget.setVisible(ranks)

    def connect_operation_signals(self) -> None:
        self.create_chart_check.toggled.connect(self.chart_type_combo.setEnabled)
        self.model_combo.currentIndexChanged.connect(lambda *_args: self._sync_model_controls())
        self.model_combo.currentIndexChanged.connect(lambda *_args: self.refresh_results())
        self.popmean_spin.valueChanged.connect(lambda *_args: self.refresh_results())
        self.alternative_combo.currentIndexChanged.connect(lambda *_args: self.refresh_results())
        self.trim_percent_spin.valueChanged.connect(lambda *_args: self.refresh_results())
        self.anderson_method_combo.currentIndexChanged.connect(lambda *_args: self.refresh_results())
        self.monte_carlo_resamples_spin.valueChanged.connect(lambda *_args: self.refresh_results())
        self.monte_carlo_batch_spin.valueChanged.connect(lambda *_args: self.refresh_results())
        self.distribution_combo.currentIndexChanged.connect(lambda *_args: self.refresh_results())
        self.exhaustive_check.toggled.connect(lambda *_args: self.refresh_results())
        self.rank_combo.currentIndexChanged.connect(lambda *_args: self.refresh_results())

    # ------------------------------------------------------------------
    # Base pipeline overrides: read-only operation
    # ------------------------------------------------------------------
    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"statistics_dialog": "series_statistics"}

    def result_to_frame(self, result: Any) -> Any:
        raise NotImplementedError("Statistics results are read-only and are not saved as tables.")

    def result_series_spec(self, axis_id: int, table_name: str, result: Any) -> ResultSeriesSpec:
        raise NotImplementedError("Statistics results are read-only and do not create chart series.")

    def preview(self) -> bool:
        return self._run_statistics(verb="Preview")

    def apply(self) -> bool:
        self._remember_state()
        return self._run_statistics(verb="Applied")

    def ok(self) -> None:
        if self.apply():
            self.accept()

    def refresh_results(self) -> None:
        try:
            results = list(self.compute_results())
        except Exception:
            self.set_results_html(
                "<p style='color:#666;'>Check one or more numeric series to compute statistics.</p>"
            )
            return
        self.publish_results(self.format_results(results))

    def compute_results(self) -> Sequence[SeriesStatsResult]:
        samples = self._selected_samples()
        if not samples:
            raise ValueError("Select at least one source series.")
        return [
            SeriesStatsResult(
                samples=samples,
                popmean=float(self.popmean_spin.value()),
                alternative=str(self.alternative_combo.currentData() or "two-sided"),
                model=str(self.model_combo.currentData() or "all"),
                trim_percent=float(self.trim_percent_spin.value()) / 100.0,
                distribution=str(self.distribution_combo.currentData() or "best"),
                exhaustive=bool(self.exhaustive_check.isChecked()),
                rank_by=str(self.rank_combo.currentData() or DEFAULT_RANK),
            )
        ]

    def format_results(self, results: Sequence[Any]) -> str:
        if not results:
            return ""
        result = results[0]
        if not isinstance(result, SeriesStatsResult):
            return ""
        return self._format_statistics_html(result)

    def results_report_html(self, formatted: str, results: Sequence[Any]) -> str:
        return formatted

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------
    def _checked_series_rows(self) -> list[Any]:
        """Return all checked series rows directly from the selector widget.

        This bypasses any single-current-item semantics and guarantees that the
        statistics dialog uses every checked series, not just the last one the
        user clicked.
        """
        series_list = getattr(self.series_selector, "series_list", None)
        if series_list is None:
            return list(self.selected_series())

        rows: list[Any] = []
        for index in range(series_list.count()):
            item = series_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                rows.append(item.data(_SERIES_ROLE))
        return rows

    def _rows_for_statistics(self) -> list[Any]:
        """Return rows that should participate in this statistics run.

        The dialog still respects checked rows.  For multi-series models, if the
        selector reports only one checked row while the axis contains several
        visible series, use the visible axis series instead.  This covers the
        common workflow where the dialog is opened from a single active chart
        series but the requested statistics model needs pairs.
        """
        checked = self._checked_series_rows()
        model = str(self.model_combo.currentData() or "all")
        if len(checked) >= 2 or model not in {"all", "paired", "correlation"}:
            return checked

        current_axis_series = getattr(self.series_selector, "current_axis_series", None)
        if current_axis_series is None:
            return checked

        visible = list(current_axis_series() or [])
        return visible if len(visible) > len(checked) else checked

    def _selected_samples(self) -> list[SeriesStatsSample]:
        """Return numeric samples for all selected rows.

        Statistics should not depend on the X axis being numeric. The shared
        ``get_data`` helper is designed for chart operations that need finite X
        and Y values; for this dialog we only require a numeric measurement
        column. This loader therefore reads the series SQL directly, tries the
        configured Y/Z role first, then falls back to the first numeric column.
        """
        samples: list[SeriesStatsSample] = []
        skipped: list[str] = []

        for row in self._rows_for_statistics():
            sample = self._sample_from_series_row(row)
            if sample is None:
                skipped.append(str(row["name"] or f"Series {row['id']}"))
                continue
            samples.append(sample)

        if skipped:
            applogger.warning(
                "Statistics skipped %d series with no numeric values: %s",
                len(skipped),
                ", ".join(skipped),
            )
        return samples

    def _sample_from_series_row(self, row: Any) -> SeriesStatsSample | None:
        """Build one numeric sample from a series descriptor row."""
        name = str(row["name"] or f"Series {row['id']}")
        roles = parse_roles(row["roles"])
        frame = self._series_frame(str(row["sql_query"]))
        if frame.empty:
            return None

        y_column = self._best_numeric_column(frame, roles)
        if y_column is None:
            return None

        y = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            return None

        x = self._x_values_for_statistics(frame, roles, finite)
        rowids = np.arange(1, int(np.sum(finite)) + 1, dtype=int)
        sample_name = name if str(y_column) in {name, str(roles.get("y", "")), str(roles.get("z", ""))} else f"{name} [{y_column}]"
        return SeriesStatsSample(
            name=sample_name,
            x=x,
            y=y[finite],
            rowids=rowids,
        )

    def _series_frame(self, sql_query: str) -> pd.DataFrame:
        """Execute a series SQL query as a read-only DataFrame.

        Avoid ``query_source_table`` here: some chart descriptors contain SQL
        fragments or generated-series queries that do not resolve to a physical
        user table, and passing those through the table-name resolver logs
        ``Invalid table name: ''``. Statistics only needs the query result rows.
        """
        sql = str(sql_query or "").strip().rstrip(";").strip()
        if not sql:
            return pd.DataFrame()

        try:
            with self._repo.connect() as con:
                if sql.lower().startswith(("select", "with")):
                    return pd.read_sql_query(sql, con)

                # Treat a bare descriptor value as a physical table name.
                table_name = sql.strip().strip('"')
                if table_name:
                    return pd.read_sql_query(
                        f"SELECT * FROM {quote_identifier(table_name)}", con
                    )
        except Exception:
            applogger.debug("Failed to read series data for statistics.", exc_info=True)
        return pd.DataFrame()

    def _best_numeric_column(self, frame: pd.DataFrame, roles: Mapping[str, Any]) -> str | None:
        """Return the most appropriate numeric measurement column."""
        preferred = [
            str(roles.get("y", "")).strip(),
            str(roles.get("z", "")).strip(),
            str(roles.get("value", "")).strip(),
            str(roles.get("values", "")).strip(),
        ]
        for column in preferred:
            if column and column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                if values.notna().any():
                    return column

        excluded = {
            "hide",
            "clusterid",
            "__rowid__",
            str(roles.get("x", "")).strip().lower(),
        }
        best_column: str | None = None
        best_count = 0
        for column in frame.columns:
            if str(column).strip().lower() in excluded:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            count = int(values.notna().sum())
            if count > best_count:
                best_column = str(column)
                best_count = count
        return best_column

    def _x_values_for_statistics(
        self,
        frame: pd.DataFrame,
        roles: Mapping[str, Any],
        finite_y: np.ndarray,
    ) -> np.ndarray:
        """Return optional numeric X values, or row order when X is categorical."""
        x_column = str(roles.get("x", "")).strip()
        if x_column and x_column in frame.columns:
            x_values = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(x_values[finite_y]).any():
                return x_values[finite_y]
        return np.arange(1, int(np.sum(finite_y)) + 1, dtype=float)

    # ------------------------------------------------------------------
    # SciPy result helpers, Pylance-safe
    # ------------------------------------------------------------------
    @staticmethod
    def _result_statistic(result: Any) -> float:
        return float(result[0])

    @staticmethod
    def _result_pvalue(result: Any) -> float:
        return float(result[1])

    @staticmethod
    def _named_pvalue(result: Any) -> float:
        return float(getattr(result, "pvalue", np.nan))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _describe(self, sample: np.ndarray, *, trim_percent: float) -> dict[str, Any]:
        values = np.asarray(sample, dtype=float)
        values = values[np.isfinite(values)]
        n = int(values.size)
        if n == 0:
            return {"n": 0}

        q1, median, q3 = np.percentile(values, [25, 50, 75])
        sem = float(stats.sem(values, nan_policy="omit")) if n > 1 else np.nan
        ci_low = ci_high = np.nan
        if n > 1 and np.isfinite(sem):
            ci = stats.t.interval(0.95, df=n - 1, loc=float(np.mean(values)), scale=sem)
            ci_low, ci_high = float(ci[0]), float(ci[1])

        mode_res = stats.mode(values, keepdims=False)
        try:
            mode_value = float(mode_res.mode)
            mode_count = int(mode_res.count)
        except Exception:
            mode_value = np.nan
            mode_count = 0

        positive = values[values > 0]
        entropy = np.nan
        if positive.size > 0 and float(np.sum(positive)) > 0:
            probs = positive / float(np.sum(positive))
            entropy = float(stats.entropy(probs))

        return {
            "n": n,
            "mean": float(np.mean(values)),
            "trimmed_mean": float(stats.trim_mean(values, proportiontocut=trim_percent)) if n else np.nan,
            "gmean": float(stats.gmean(positive)) if positive.size == n else np.nan,
            "hmean": float(stats.hmean(positive)) if positive.size == n else np.nan,
            "median": float(median),
            "mode": mode_value,
            "mode_count": mode_count,
            "std": float(np.std(values, ddof=1)) if n > 1 else np.nan,
            "var": float(np.var(values, ddof=1)) if n > 1 else np.nan,
            "sem": sem,
            "min": float(np.min(values)),
            "q1": float(q1),
            "q3": float(q3),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "iqr": float(stats.iqr(values)),
            "mad": float(stats.median_abs_deviation(values, scale=1.4826)),
            "skewness": float(stats.skew(values, bias=False)) if n > 2 else np.nan,
            "kurtosis": float(stats.kurtosis(values, bias=False)) if n > 3 else np.nan,
            "entropy": entropy,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }

    # ------------------------------------------------------------------
    # Distribution fit
    # ------------------------------------------------------------------
    def _distribution_fits(
        self, values: np.ndarray, *, exhaustive: bool, rank_by: str
    ) -> list[DistributionFit]:
        """Return the ranked fits for one sample, memoised.

        Memoised because the dialog refreshes on every control it owns, and a
        sweep costs half a second curated or a dozen seconds exhaustive.
        Without this, nudging the reference value - which has nothing to do
        with the fit - would refit every candidate and freeze the window.

        Keyed on the sample's bytes: two different series with identical values
        genuinely have identical fits, and the key stays correct when a series
        is edited underneath us.
        """
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        key = (finite.tobytes(), finite.size, exhaustive, rank_by)

        cached = self._fit_cache.get(key)
        if cached is None:
            cached = fit_distributions(
                finite, exhaustive=exhaustive, rank_by=rank_by
            )
            # Bounded so a long session cannot grow without limit; the useful
            # entries are the samples currently checked, which is a handful.
            if len(self._fit_cache) > 32:
                self._fit_cache.clear()
            self._fit_cache[key] = cached
        return cached

    def _distribution_section(self, result: SeriesStatsResult) -> str:
        """Rank the candidate distributions for every checked series."""
        blocks: list[str] = []
        for sample in result.samples:
            fits = self._distribution_fits(
                sample.y, exhaustive=result.exhaustive, rank_by=result.rank_by
            )
            if not fits:
                blocks.append(
                    report_html.note(
                        f"{sample.name}: no candidate distribution could be "
                        "fitted to this sample."
                    )
                )
                continue

            rows = [
                [
                    fit.name,
                    self._format_number(fit.ks_statistic),
                    self._format_pvalue(fit.pvalue),
                    self._format_number(fit.aic),
                    self._format_number(fit.bic),
                    # scipy's own parameter order, which is what its
                    # documentation for that distribution describes.
                    ", ".join(f"{value:.6g}" for value in fit.params),
                ]
                for fit in fits[: self.DISTRIBUTION_ROWS]
            ]
            blocks.append(
                report_html.section(
                    f"{sample.name} (n = {fits[0].n})",
                    self._table(
                        ["Distribution", "KS D", "p", "AIC", "BIC", "Parameters"],
                        rows,
                    ),
                )
            )

            if len(fits) > self.DISTRIBUTION_ROWS:
                # No silent truncation: a ranking that quietly drops candidates
                # reads as "these are all of them".
                blocks.append(
                    report_html.note(
                        f"{len(fits) - self.DISTRIBUTION_ROWS} further candidates "
                        "fitted worse and are not shown."
                    )
                )

        blocks.append(
            report_html.note(
                f"Ranked by {result.rank_by.upper()}. D is the largest gap "
                "between the sample's empirical CDF and the fitted one, and is "
                "shown as a diagnostic rather than as the ordering: on samples "
                "drawn from known families, ranking by D alone puts "
                "four-parameter beta first every time, because it measures how "
                "close the curve got and not how many free parameters it took. "
                "The p-value is optimistic for the same reason it always is "
                "here - the parameters were estimated from the sample being "
                "tested, so it asks whether the data could have come from this "
                "fitted curve, not from that family."
            )
        )
        return "".join(blocks)

    def _one_sample_tests(
        self,
        values: np.ndarray,
        *,
        popmean: float,
        alternative: str,
    ) -> list[dict[str, Any]]:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        tests: list[dict[str, Any]] = []

        if values.size > 1:
            t_res = stats.ttest_1samp(values, popmean=popmean, alternative=alternative)
            tests.append(_test_row(_("One-sample t-test"), values.size, self._result_statistic(t_res), self._result_pvalue(t_res), f"mean = {popmean:g}"))

        diff = values - float(popmean)
        nonzero = diff[np.isfinite(diff) & (np.abs(diff) > 0)]
        if nonzero.size > 0:
            try:
                w_res = stats.wilcoxon(nonzero, alternative=alternative, zero_method="wilcox")
                tests.append(_test_row(_("Wilcoxon signed-rank"), nonzero.size, self._result_statistic(w_res), self._result_pvalue(w_res), f"median = {popmean:g}"))
            except ValueError as exc:
                tests.append(_note_row(_("Wilcoxon signed-rank"), nonzero.size, str(exc)))

            positives = int(np.sum(diff > 0))
            negatives = int(np.sum(diff < 0))
            trials = positives + negatives
            if trials > 0:
                b_res = stats.binomtest(positives, trials, p=0.5, alternative=alternative)
                tests.append(_test_row(_("Sign test"), trials, positives, self._named_pvalue(b_res), "positive signs"))

        return tests

    def _normality_tests(self, values: np.ndarray) -> list[dict[str, Any]]:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        n = int(values.size)
        tests: list[dict[str, Any]] = []

        if n >= 3:
            try:
                res = stats.shapiro(values)
                tests.append(_test_row(_("Shapiro-Wilk normality"), n, self._result_statistic(res), self._result_pvalue(res)))
            except Exception as exc:
                tests.append(_note_row(_("Shapiro-Wilk normality"), n, str(exc)))

        if n >= 8:
            for name, func in (
                ("D'Agostino-Pearson normality", stats.normaltest),
                ("Skewness test", stats.skewtest),
            ):
                try:
                    res = func(values)
                    tests.append(_test_row(name, n, self._result_statistic(res), self._result_pvalue(res)))
                except Exception as exc:
                    tests.append(_note_row(name, n, str(exc)))

        if n >= 5:
            try:
                res = stats.kurtosistest(values)
                tests.append(_test_row(_("Kurtosis test"), n, self._result_statistic(res), self._result_pvalue(res)))
            except Exception as exc:
                tests.append(_note_row(_("Kurtosis test"), n, str(exc)))

        if n >= 2:
            try:
                res = stats.jarque_bera(values)
                tests.append(_test_row(_("Jarque-Bera normality"), n, self._result_statistic(res), self._result_pvalue(res)))
            except Exception as exc:
                tests.append(_note_row(_("Jarque-Bera normality"), n, str(exc)))

            std = float(np.std(values, ddof=1))
            if std > 0:
                z = (values - float(np.mean(values))) / std
                try:
                    res = stats.kstest(z, "norm")
                    tests.append(_test_row(_("Kolmogorov-Smirnov vs normal"), n, self._result_statistic(res), self._result_pvalue(res), "standardized sample"))
                except Exception as exc:
                    tests.append(_note_row(_("Kolmogorov-Smirnov vs normal"), n, str(exc)))

            try:
                res, note = self._anderson_result(values)
                tests.append(
                    {
                        "test": "Anderson-Darling normality",
                        "n": n,
                        "statistic": float(getattr(res, "statistic", np.nan)),
                        "pvalue": float(getattr(res, "pvalue", np.nan)),
                        "note": note,
                    }
                )
            except Exception as exc:
                tests.append(_note_row(_("Anderson-Darling normality"), n, str(exc)))

        return tests

    def _anderson_result(self, values: np.ndarray) -> tuple[Any, str]:
        """Return Anderson-Darling result using selected p-value method."""
        method_name = str(self.anderson_method_combo.currentData() or "interpolate")

        if method_name == "monte_carlo":
            n_resamples = int(self.monte_carlo_resamples_spin.value())
            batch_value = int(self.monte_carlo_batch_spin.value())
            batch = None if batch_value <= 0 else batch_value
            monte_carlo_method = stats.MonteCarloMethod(
                n_resamples=n_resamples,
                batch=batch,
            )
            return (
                stats.anderson(values, dist="norm", method=monte_carlo_method),
                f"Monte Carlo p-value; resamples={n_resamples:,}; batch={batch or 'Auto'}",
            )

        try:
            return (
                stats.anderson(values, dist="norm", method="interpolate"),
                "Interpolated p-value",
            )
        except ValueError:
            return (
                stats.anderson(values, dist="norm", method="interpolated"),
                "Interpolated p-value",
            )
        except TypeError:
            # Older SciPy versions do not have the method parameter.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                return (
                    stats.anderson(values, dist="norm"),
                    "Critical values only; installed SciPy does not support p-value method",
                )

    def _paired_values(self, left: SeriesStatsSample, right: SeriesStatsSample) -> tuple[np.ndarray, np.ndarray, str]:
        left_by_x: dict[float, float] = {}
        for x_value, y_value in zip(left.x, left.y, strict=False):
            if np.isfinite(x_value) and np.isfinite(y_value):
                left_by_x.setdefault(float(x_value), float(y_value))

        paired_left: list[float] = []
        paired_right: list[float] = []
        for x_value, y_value in zip(right.x, right.y, strict=False):
            key = float(x_value)
            if key in left_by_x and np.isfinite(y_value):
                paired_left.append(left_by_x[key])
                paired_right.append(float(y_value))

        if paired_left:
            return np.asarray(paired_left), np.asarray(paired_right), "common X values"

        n = min(left.y.size, right.y.size)
        return left.y[:n], right.y[:n], "row order, truncated to common length"

    def _paired_tests(self, left: SeriesStatsSample, right: SeriesStatsSample, *, alternative: str) -> list[dict[str, Any]]:
        a, b, alignment = self._paired_values(left, right)
        finite = np.isfinite(a) & np.isfinite(b)
        a = a[finite]
        b = b[finite]
        n = int(a.size)
        if n == 0:
            return [_note_row(_("Paired tests"), 0, "No paired observations.")]

        diff = a - b
        tests: list[dict[str, Any]] = [
            _test_row(_("Paired difference summary"), n, float(np.mean(diff)), np.nan, f"mean difference; {alignment}")
        ]

        if n > 1:
            res = stats.ttest_rel(a, b, alternative=alternative)
            tests.append(_test_row(_("Paired t-test"), n, self._result_statistic(res), self._result_pvalue(res), alignment))

        nonzero = diff[np.isfinite(diff) & (np.abs(diff) > 0)]
        if nonzero.size > 0:
            try:
                res = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
                tests.append(_test_row(_("Wilcoxon signed-rank paired test"), nonzero.size, self._result_statistic(res), self._result_pvalue(res), alignment))
            except ValueError as exc:
                tests.append(_note_row(_("Wilcoxon signed-rank paired test"), nonzero.size, str(exc)))

            positives = int(np.sum(diff > 0))
            negatives = int(np.sum(diff < 0))
            trials = positives + negatives
            if trials > 0:
                res = stats.binomtest(positives, trials, p=0.5, alternative=alternative)
                tests.append(_test_row(_("Sign test"), trials, positives, self._named_pvalue(res), f"positive signs; {alignment}"))

        if n >= 3:
            try:
                res = stats.shapiro(diff)
                tests.append(_test_row(_("Normality of paired differences"), n, self._result_statistic(res), self._result_pvalue(res), alignment))
            except Exception as exc:
                tests.append(_note_row(_("Normality of paired differences"), n, str(exc)))

        return tests

    def _correlation_tests(self, left: SeriesStatsSample, right: SeriesStatsSample) -> list[dict[str, Any]]:
        a, b, alignment = self._paired_values(left, right)
        finite = np.isfinite(a) & np.isfinite(b)
        a = a[finite]
        b = b[finite]
        n = int(a.size)
        if n < 2:
            return [_note_row(_("Correlation tests"), n, "Need at least two paired observations.")]

        tests: list[dict[str, Any]] = []
        for name, func in (
            ("Pearson correlation", stats.pearsonr),
            ("Spearman rank correlation", stats.spearmanr),
            ("Kendall tau", stats.kendalltau),
        ):
            try:
                res = func(a, b)
                tests.append(_test_row(name, n, self._result_statistic(res), self._result_pvalue(res), alignment))
            except Exception as exc:
                tests.append(_note_row(name, n, str(exc)))

        try:
            res = stats.linregress(a, b)
            tests.append(_test_row(_("Linear regression slope"), n, float(getattr(res, "slope", np.nan)), float(getattr(res, "pvalue", np.nan)), alignment))
        except Exception as exc:
            tests.append(_note_row(_("Linear regression slope"), n, str(exc)))

        return tests

    # ------------------------------------------------------------------
    # HTML formatting
    # ------------------------------------------------------------------
    def _format_number(self, value: Any) -> str:
        """Format numeric results compactly but readably for HTML tables."""
        try:
            number = float(value)
        except Exception:
            return html.escape(str(value))
        if not np.isfinite(number):
            return ""
        abs_number = abs(number)
        if abs_number == 0:
            return "0"
        if abs_number < 1.0e-4 or abs_number >= 1.0e6:
            return f"{number:.3e}"
        if abs_number >= 1000:
            return f"{number:,.2f}"
        if abs_number >= 100:
            return f"{number:,.3f}"
        if abs_number >= 10:
            return f"{number:,.4f}".rstrip("0").rstrip(".")
        return f"{number:,.5f}".rstrip("0").rstrip(".")

    def _format_pvalue(self, value: Any) -> str:
        """Return a p-value in the house notation (``<.0001`` when small)."""
        return report_html.format_p_value(value)

    def _table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
        """Return a headed table in the shared report style.

        Text columns stay left-aligned; everything else is right-aligned, so a
        column of numbers can be read down the page.
        """
        text_columns = {"series", "pair", "test", "note", "alignment"}
        align = [
            "left" if str(header).strip().lower() in text_columns else "right"
            for header in headers
        ]
        return report_html.table(
            headers,
            rows,
            align=align,
            empty_message="No applicable results for this selection.",
        )

    def _show_section(self, model: str, section: str) -> bool:
        # "distribution" is not part of "all": a sweep costs half a second per
        # series curated and a dozen seconds exhaustive, and All is the default
        # model.  Making it opt-in keeps the default fast.
        if section == "distribution":
            return model == "distribution"
        return model == "all" or model == section

    def _format_statistics_html(self, result: SeriesStatsResult) -> str:
        """Return the report: what was run, then one section per model part."""
        sections: list[str] = [
            report_html.summary_table(
                [
                    ("Series", len(result.samples)),
                    ("Model", result.model),
                    ("Reference value", f"{result.popmean:g}"),
                    ("Alternative", result.alternative),
                    (
                        "Anderson method",
                        str(self.anderson_method_combo.currentData() or "interpolate"),
                    ),
                ]
            )
        ]
        parts = sections

        if self._show_section(result.model, "descriptive"):
            desc_rows = []
            for sample in result.samples:
                desc = self._describe(sample.y, trim_percent=result.trim_percent)
                desc_rows.append([
                    html.escape(sample.name), str(desc.get("n", "")),
                    self._format_number(desc.get("mean")), self._format_number(desc.get("trimmed_mean")),
                    self._format_number(desc.get("gmean")), self._format_number(desc.get("hmean")),
                    self._format_number(desc.get("median")), self._format_number(desc.get("mode")), str(desc.get("mode_count", "")),
                    self._format_number(desc.get("std")), self._format_number(desc.get("var")), self._format_number(desc.get("sem")),
                    self._format_number(desc.get("min")), self._format_number(desc.get("q1")), self._format_number(desc.get("q3")),
                    self._format_number(desc.get("max")), self._format_number(desc.get("range")), self._format_number(desc.get("iqr")),
                    self._format_number(desc.get("mad")), self._format_number(desc.get("skewness")), self._format_number(desc.get("kurtosis")),
                    self._format_number(desc.get("entropy")),
                    f"{self._format_number(desc.get('ci95_low'))} .. {self._format_number(desc.get('ci95_high'))}",
                ])
            parts.append(report_html.section(
                _("Descriptive statistics"),
                self._table([
                "Series", "n", "Mean", "Trim mean", "GMean", "HMean", "Median", "Mode", "Mode n",
                "Std", "Var", "SEM", "Min", "Q1", "Q3", "Max", "Range", "IQR", "MAD", "Skew", "Kurtosis", "Entropy", "95% CI mean",
            ], desc_rows),
            ))

        if self._show_section(result.model, "one_sample"):
            rows = []
            for sample in result.samples:
                for test in self._one_sample_tests(sample.y, popmean=result.popmean, alternative=result.alternative):
                    rows.append(_html_test_row(sample.name, test, self._format_number, self._format_pvalue))
            parts.append(report_html.section(
                _("One-sample tests"),
                self._table(["Series", "Test", "n", "Statistic", "p-value", "Note"], rows),
            ))

        if self._show_section(result.model, "normality"):
            rows = []
            for sample in result.samples:
                for test in self._normality_tests(sample.y):
                    rows.append(_html_test_row(sample.name, test, self._format_number, self._format_pvalue))
            parts.append(report_html.section(
                _("Normality / shape tests"),
                self._table(["Series", "Test", "n", "Statistic", "p-value", "Note"], rows),
            ))

        if len(result.samples) > 1 and self._show_section(result.model, "paired"):
            rows = []
            for left, right in itertools.combinations(result.samples, 2):
                pair_name = f"{left.name} vs {right.name}"
                for test in self._paired_tests(left, right, alternative=result.alternative):
                    rows.append(_html_test_row(pair_name, test, self._format_number, self._format_pvalue))
            parts.append(report_html.section(
                _("Paired-sample tests"),
                self._table(["Pair", "Test", "n", "Statistic", "p-value", "Note"], rows),
            ))

        if len(result.samples) > 1 and self._show_section(result.model, "correlation"):
            rows = []
            for left, right in itertools.combinations(result.samples, 2):
                pair_name = f"{left.name} vs {right.name}"
                for test in self._correlation_tests(left, right):
                    rows.append(_html_test_row(pair_name, test, self._format_number, self._format_pvalue))
            parts.append(report_html.section(
                _("Correlation / association"),
                self._table(["Pair", "Test", "n", "Statistic", "p-value", "Note"], rows),
            ))

        if self._show_section(result.model, "distribution"):
            parts.append(
                report_html.section(
                    _("Distribution fit"), self._distribution_section(result)
                )
            )

        if len(result.samples) < 2 and result.model in {"all", "paired", "correlation"}:
            parts.append(
                report_html.note(
                    "Paired-sample and correlation sections need at least two "
                    "series on the selected axis."
                )
            )

        return report_html.document(
            "Statistics",
            f"{len(result.samples)} series",
            *parts,
        )

    def _run_statistics(self, *, verb: str) -> bool:
        try:
            results = list(self.compute_results())
            if not results:
                show_message(self, "series.no_series_selected", title=self.operation_label)
                return False
            self.store_cached_results(results)
            formatted = self.format_results(results)
            self.publish_results(formatted)

            if self.create_chart_check.isChecked():
                # On Preview as well, like every other operation dialog: a
                # preview you cannot see is not a preview.  Closing without
                # applying removes it again - see discard_operation_artifacts.
                self._create_model_axis(results[0])
                self.applied.emit()

            if verb == "Applied":
                self._applied = True
                self.results_published.emit(self.results_report_html(formatted, results))
                self.applied.emit()
            return True
        except Exception as exc:
            applogger.error("Statistics failed: %s", exc, show_dialog=True)
            return False

    # ------------------------------------------------------------------
    # The chart the numbers describe
    # ------------------------------------------------------------------
    #: Model -> the renderer that shows what that model measures.  A test
    #: reports a number; the chart beside it is what makes the number
    #: believable, and which chart depends on what was tested.
    #: Rows of the distribution ranking to print per series.  Ten is enough
    #: to see whether the winner is clear or the top few are indistinguishable,
    #: which is the question a ranking answers.
    DISTRIBUTION_ROWS: int = 10

    #: Which charts each model offers, and what its axis is titled.  Not one
    #: chart per model but a list, because more than one picture answers the
    #: same question and which is clearer depends on the data: a histogram
    #: shows where the mass is, an ECDF shows the fit without a bin width to
    #: argue about.
    #:
    #: "all" is deliberately empty.  It prints every section at once, so no
    #: single chart illustrates it - and the controls are hidden rather than
    #: disabled, since a disabled control invites the user to look for what
    #: would enable it.
    CHARTS_BY_MODEL: Mapping[str, tuple[tuple[str, str], ...]] = {
        "all": (),
        "descriptive": (
            ("Histogram", "Descriptive statistics"),
            ("ECDF", "Descriptive statistics"),
        ),
        "one_sample": (
            ("Histogram", "One-sample tests"),
            ("ECDF", "One-sample tests"),
        ),
        # Normality is a claim about shape, so the picture is the sample
        # against the curve it is being tested for.
        "normality": (
            ("Histogram", "Normality"),
            ("ECDF", "Normality"),
        ),
        # Both of these are about two samples together, so the picture is one
        # plotted against the other - every pair of them.
        "paired": (("Scatter Plot", "Paired samples"),),
        "correlation": (("Scatter Plot", "Correlation"),),
        "distribution": (("Histogram", "Distribution fit"),),
    }

    #: Which distributions the chart may be asked to draw, per model.  The
    #: normality model offers exactly the two families its tests are about; the
    #: distribution model offers the ranked sweep and every curated name.
    DISTRIBUTIONS_BY_MODEL: Mapping[str, tuple[tuple[str, str], ...]] = {
        "normality": (("norm", "Normal"), ("lognorm", "Lognormal")),
        "distribution": (
            ("best", "Best fit"),
            ("top3", "Top 3 fits"),
            ("top5", "Top 5 fits"),
            *((name, name) for name in CURATED_DISTRIBUTIONS),
        ),
    }
    #: Everything else that can draw a fitted curve gets the ranked choices.
    DEFAULT_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
        ("", "None"),
        ("best", "Best fit"),
        ("top3", "Top 3 fits"),
    )

    def _axis_options(
        self, chart_type: str, result: SeriesStatsResult
    ) -> dict[str, Any]:
        """Return the axis options for the chart this run draws.

        The fitted curve is the point of charting most of these models, so the
        distribution the dialog resolved is passed to whichever renderer can
        draw it - the histogram over its bars, the ECDF over its steps.  Both
        read it through the same module the report ranks with, so the curve
        cannot contradict the table beside it.
        """
        options: dict[str, Any] = {"grid": True}
        if chart_type not in ("Histogram", "ECDF") or not result.distribution:
            return options

        options["distribution_fit"] = result.distribution
        if chart_type == "Histogram":
            # Semi-transparent and outlined, so a curve crossing the bars stays
            # readable against them.
            options["histtype"] = "stepfilled"
            options["alpha"] = 0.55
        return options

    def _create_model_axis(self, result: SeriesStatsResult) -> None:
        """Add an axis showing what the selected model measured.

        Only when the user asked for a chart - the dialog's job is to report
        numbers, and an axis appearing unrequested is a side effect.

        Created on the first Preview and reused afterwards, so changing the
        model or a parameter redraws the same axis instead of stacking a new
        one each time.  Closing without Apply deletes it again, which is what
        keeps a preview a preview: see discard_operation_artifacts.

        The axis carries the same series the statistics were computed from, so
        the numbers and the picture cannot disagree.
        """
        charts = self.CHARTS_BY_MODEL.get(result.model, ())
        if not charts:
            # A model with nothing to draw: "all" prints every section, and no
            # one picture illustrates that.
            return

        # The chosen entry, or the model's first, which is the one the list was
        # ordered to put there.
        chosen = str(self.chart_type_combo.currentData() or "")
        chart_type, title = next(
            ((name, label) for name, label in charts if name == chosen), charts[0]
        )

        options = self._axis_options(chart_type, result)

        if (
            self._result_axis_id is not None
            and self._result_chart_type == chart_type
            and self._result_axis_options == options
        ):
            return

        # A different renderer needs different role columns, so the axis is
        # rebuilt rather than re-labelled.
        self._remove_result_axis()

        axis_id = self.create_result_axis(
            chart_type=chart_type,
            title=title,
            x_label="" if chart_type != "ECDF" else "value",
            y_label="",
            options=options,
        )
        self._result_axis_id = axis_id
        self._result_chart_type = chart_type
        self._result_axis_options = options
        rows = self._rows_for_statistics()
        if result.model in ("paired", "correlation") and chart_type == "Scatter Plot":
            # Every pair, not every series.  These two models test one sample
            # against another, and the report already prints a row per pair -
            # so charting each series against its own x would show something
            # the statistics never looked at.
            for left, right in itertools.combinations(rows, 2):
                self._attach_series_pair(axis_id, left, right)
        else:
            for row in rows:
                self._attach_source_series(axis_id, row, chart_type)

    def _remove_result_axis(self) -> None:
        """Delete the axis this dialog added, if it added one."""
        axis_id = self._result_axis_id
        self._result_axis_id = None
        self._result_chart_type = ""
        self._result_axis_options = {}
        if axis_id is None:
            return
        try:
            self._repo.delete_axis(axis_id)
        except Exception:
            applogger.exception("Failed to remove the statistics axis %s", axis_id)

    def discard_operation_artifacts(self) -> None:
        """Remove the previewed chart when Apply never happened.

        Creating an axis commits, so it is not covered by the preview
        savepoint and has to be undone by hand.
        """
        if self._applied:
            return
        self._remove_result_axis()

    def _attach_series_pair(self, axis_id: int, left: Any, right: Any) -> None:
        """Scatter one series' values against another's, as the tests read them.

        Joined on row order, which is what pairing means here and what
        ``_paired_values`` does when it computes the numbers: the nth
        observation of one series against the nth of the other.  A join on a
        shared key would be a different, defensible pairing - and a different
        statistic from the one printed above the chart, which is why it is not
        used.
        """
        left_sql = str(left["sql_query"] or "").strip()
        right_sql = str(right["sql_query"] or "").strip()
        if not left_sql or not right_sql:
            return

        left_y = str(parse_roles(row_value(left, "roles")).get("y", "y") or "y")
        right_y = str(parse_roles(row_value(right, "roles")).get("y", "y") or "y")
        left_name, right_name = str(left["name"]), str(right["name"])

        # ROW_NUMBER over the source order: SQLite has no positional join, and
        # rowid belongs to a table rather than to an arbitrary SELECT.
        sql = (
            "SELECT a.value AS x, b.value AS y FROM ("
            f"SELECT {quote_identifier(left_y)} AS value, "
            f"ROW_NUMBER() OVER () AS position FROM ({left_sql})) AS a "
            "JOIN ("
            f"SELECT {quote_identifier(right_y)} AS value, "
            f"ROW_NUMBER() OVER () AS position FROM ({right_sql})) AS b "
            "ON a.position = b.position"
        )

        self._repo.create_series_descriptor(
            axis_id=axis_id,
            series_index=self._repo.next_series_index(axis_id),
            name=f"{left_name} vs {right_name} [statistics]",
            sql_query=sql,
            roles={"x": "x", "y": "y"},
            style={
                **dict(self.generated_style_filter),
                "label": f"{left_name} vs {right_name}",
                "marker": ".",
            },
        )

    def _attach_source_series(self, axis_id: int, row: Any, chart_type: str) -> None:
        """Put one analysed series onto the new axis, in that axis's roles.

        Each renderer names its columns differently - a box plot wants
        ``value``, a scatter wants x and y - so the same source SQL is aliased
        per chart type rather than copied per chart type.
        """
        roles = parse_roles(row_value(row, "roles"))
        sql = str(row["sql_query"] or "").strip()
        if not sql:
            return

        name = str(row["name"])
        y_column = str(roles.get("y", "y") or "y")
        x_column = str(roles.get("x", "x") or "x")

        if chart_type == "Scatter Plot":
            select = f"SELECT {quote_identifier(x_column)} AS x, {quote_identifier(y_column)} AS y"
            axis_roles: dict[str, Any] = {"x": "x", "y": "y"}
        elif chart_type == "ECDF":
            select = f"SELECT {quote_identifier(y_column)} AS value"
            axis_roles = {"value": "value"}
        elif chart_type == "Histogram":
            # The histogram splits on "dataset", not on "group".  Sent the
            # group column it would ignore it, pool every checked series into
            # one sample and fit a single distribution across all of them -
            # which is the one thing the ranking must not describe.
            select = (
                f"SELECT {quote_identifier(y_column)} AS value, "
                f"'{name}' AS dataset"
            )
            axis_roles = {"value": "value", "dataset": "dataset"}
        else:  # Box Plot and anything else grouped by series name
            select = (
                f"SELECT {quote_identifier(y_column)} AS value, "
                f"'{name}' AS \"group\""
            )
            axis_roles = {"value": "value", "group": "group"}

        self._repo.create_series_descriptor(
            axis_id=axis_id,
            series_index=self._repo.next_series_index(axis_id),
            name=f"{name} [statistics]",
            sql_query=f"{select} FROM ({sql})",
            roles=axis_roles,
            style={**dict(self.generated_style_filter), "label": name, "marker": "."},
        )


def _test_row(test: str, n: int, statistic: Any, pvalue: Any, note: str = "") -> dict[str, Any]:
    return {"test": test, "n": int(n), "statistic": statistic, "pvalue": pvalue, "note": note}


def _note_row(test: str, n: int, note: str) -> dict[str, Any]:
    return {"test": test, "n": int(n), "statistic": np.nan, "pvalue": np.nan, "note": note}


def _html_test_row(
    label: str,
    test: Mapping[str, Any],
    format_number,
    format_pvalue,
) -> list[str]:
    return [
        html.escape(str(label)),
        html.escape(str(test.get("test", ""))),
        str(test.get("n", "")),
        format_number(test.get("statistic")),
        format_pvalue(test.get("pvalue")),
        html.escape(str(test.get("note", ""))),
    ]
