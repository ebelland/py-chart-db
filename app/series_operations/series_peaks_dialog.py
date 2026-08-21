"""Find peaks in a chart series, and measure them.

``find_peaks`` on its own answers a question nobody asks.  A raw local-maximum
search on real data returns hundreds of hits, almost all of them noise, and the
list is useless until it is filtered by something meaningful.  So this dialog
is built around the filters rather than around the search: prominence, height,
distance and width are the parameters, and the peak list is what falls out.

**Prominence is the default filter, not height.**  Height compares a peak to
zero, which is only meaningful when the baseline is at zero - on a sloping or
raised background it selects whichever part of the signal happens to sit
highest, not the peaks.  Prominence measures how far a peak stands above the
higher of the two saddles flanking it, so it asks "how much of a peak is this,
locally", which is the question that survives a moving baseline.

Each peak is reported with its width at half prominence, and with the x bounds
of that width - which are what the Calculus dialog's integral wants, so a peak
found here can be integrated there.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget
from scipy.signal import find_peaks, peak_prominences, peak_widths

from app.data.data_source import parse_roles, row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.parameter_spec import BoolParam, ChoiceParam, FloatParam, IntParam
from app.series_operations.series_operation_dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.styles.style import create_doc_link, set_doc_link
from app.widgets import report_html
from app.utils.i18n import _

PEAKS_MAXIMA = "Maxima"
PEAKS_MINIMA = "Minima"
PEAKS_BOTH = "Maxima and minima"

PEAK_MODELS = (PEAKS_MAXIMA, PEAKS_MINIMA, PEAKS_BOTH)

PEAK_DOCS = {
    PEAKS_MAXIMA: (
        "Topographic prominence",
        "https://en.wikipedia.org/wiki/Topographic_prominence",
    ),
    PEAKS_MINIMA: (
        "Topographic prominence",
        "https://en.wikipedia.org/wiki/Topographic_prominence",
    ),
    PEAKS_BOTH: (
        "Topographic prominence",
        "https://en.wikipedia.org/wiki/Topographic_prominence",
    ),
}


@dataclass(slots=True)
class Peak:
    """One located peak, with the measurements that describe it."""

    x: float
    y: float
    prominence: float
    width: float
    left_x: float
    right_x: float
    is_minimum: bool = False


@dataclass(slots=True)
class PeakResult:
    """Every peak found in one source series."""

    source_name: str
    result_name: str
    model: str
    peaks: list[Peak] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x": [peak.x for peak in self.peaks],
                "y": [peak.y for peak in self.peaks],
                "prominence": [peak.prominence for peak in self.peaks],
                "width": [peak.width for peak in self.peaks],
                "left_x": [peak.left_x for peak in self.peaks],
                "right_x": [peak.right_x for peak in self.peaks],
                "is_minimum": [int(peak.is_minimum) for peak in self.peaks],
            }
        )


class SeriesPeaksDialog(SeriesOperationDialogBase):
    """Locate and measure peaks in a chart series."""

    Name: str = "Peaks"
    Description = "Find and measure peaks"

    # A peak is defined by its neighbours, so the points must be in x order.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    # Three is the smallest series that can contain an interior maximum.
    INPUT_MINIMUM_POINTS = 3

    PARAMS = (
        ChoiceParam(
            "filter_by",
            "Filter by:",
            tooltip=(
                "Prominence measures a peak against its own surroundings and "
                "survives a sloping baseline. Height compares it to zero, "
                "which only means something when the baseline is at zero."
            ),
            choices=(
                ("Prominence (relative)", "prominence"),
                ("Height (absolute)", "height"),
            ),
        ),
        FloatParam(
            "threshold",
            "Minimum:",
            tooltip=(
                "As a fraction of the signal's full range. 0.1 keeps peaks "
                "standing at least a tenth of the range above their "
                "surroundings."
            ),
            default_value=0.05,
            minimum=0.0,
            maximum=1.0,
            decimals=4,
            step=0.01,
        ),
        IntParam(
            "distance",
            "Minimum separation:",
            tooltip=(
                "Points. Of any two peaks closer than this, the more "
                "prominent is kept - which is how a single noisy peak stops "
                "being reported as several."
            ),
            default_value=1,
            minimum=1,
            maximum=100_000,
        ),
        FloatParam(
            "min_width",
            "Minimum width:",
            tooltip=(
                "Points at half prominence. Raise it to reject single-sample "
                "spikes, which are usually instrument artefacts rather than "
                "features."
            ),
            default_value=0.0,
            minimum=0.0,
            maximum=100_000.0,
            decimals=2,
            step=1.0,
        ),
        IntParam(
            "limit",
            "Report at most:",
            tooltip="Keeps the most prominent peaks when many are found.",
            default_value=50,
            minimum=1,
            maximum=10_000,
        ),
        BoolParam(
            "mark_only",
            "Mark peaks only:",
            tooltip=(
                "Draw the located peaks as markers rather than writing a "
                "table of measurements."
            ),
            default_value=True,
        ),
    )

    Icon = """
    <path d="M3 18l4-8 3 5 4-11 3 8 4-4"/>
    <circle cx="14" cy="4" r="1.6"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesPeaksDialog requires a repository instance.")

        self._last_results: list[PeakResult] = []
        self._parameter_form: QFormLayout | None = None

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Peaks",
            parent=parent,
            width=780,
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

        self.model_combo.addItems(PEAK_MODELS)
        self.model_combo.setToolTip(_("Which turning points to look for."))
        form.addRow(_("Find:"), self.model_combo)
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
        title, url = PEAK_DOCS[self._model()]
        set_doc_link(self._doc_link, title, url)

    def _model(self) -> str:
        return self.model_combo.currentText() or PEAKS_MAXIMA

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

    def compute_results(self) -> list[PeakResult]:
        model = self._model()
        params = self.parameter_values()

        results: list[PeakResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self._series_xy(row, name)
                results.append(self._find_one(name, x_values, y_values, model, params))
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

    def _find_one(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> PeakResult:
        peaks: list[Peak] = []

        if model in (PEAKS_MAXIMA, PEAKS_BOTH):
            peaks.extend(self._search(x_values, y_values, params, minimum=False))
        if model in (PEAKS_MINIMA, PEAKS_BOTH):
            # A minimum is a maximum of the inverted signal. Inverting rather
            # than writing a second search keeps one implementation of the
            # prominence and width logic, which is where the subtlety is.
            peaks.extend(self._search(x_values, y_values, params, minimum=True))

        peaks.sort(key=lambda peak: peak.x)

        return PeakResult(
            source_name=name,
            result_name=f"{name} - peaks",
            model=model,
            peaks=peaks,
            metadata={
                "found": len(peaks),
                "filter": str(params.get("filter_by", "prominence")),
            },
        )

    def _search(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        params: Mapping[str, Any],
        *,
        minimum: bool,
    ) -> list[Peak]:
        """Run find_peaks once, on the signal or on its inverse."""
        signal = -y_values if minimum else y_values

        # Thresholds are given as a fraction of the signal's range so that one
        # setting means the same thing on a millivolt trace and on a count
        # rate. An absolute default would be meaningless on arrival.
        span = float(np.ptp(y_values))
        if span <= 0.0:
            return []

        threshold = float(params.get("threshold", 0.05)) * span
        distance = max(1, int(params.get("distance", 1)))
        min_width = float(params.get("min_width", 0.0))

        kwargs: dict[str, Any] = {"distance": distance}
        if str(params.get("filter_by", "prominence")) == "height":
            kwargs["height"] = float(np.min(signal)) + threshold
        else:
            kwargs["prominence"] = max(threshold, 1e-12)
        if min_width > 0.0:
            kwargs["width"] = min_width

        indices, _properties = find_peaks(signal, **kwargs)
        if indices.size == 0:
            return []

        prominences = peak_prominences(signal, indices)[0]
        # rel_height=0.5 is the width at half prominence, not at half height:
        # on a raised baseline those differ, and the half-prominence width is
        # the one that describes the peak rather than the background.
        widths, _heights, left_ips, right_ips = peak_widths(
            signal, indices, rel_height=0.5
        )

        # peak_widths returns fractional sample positions, so the x bounds have
        # to be interpolated back onto the real axis rather than indexed.
        sample_positions = np.arange(x_values.size, dtype=float)
        left_x = np.interp(left_ips, sample_positions, x_values)
        right_x = np.interp(right_ips, sample_positions, x_values)
        width_in_x = right_x - left_x

        found = [
            Peak(
                x=float(x_values[index]),
                y=float(y_values[index]),
                prominence=float(prominence),
                width=float(width),
                left_x=float(left),
                right_x=float(right),
                is_minimum=minimum,
            )
            for index, prominence, width, left, right in zip(
                indices, prominences, width_in_x, left_x, right_x
            )
        ]

        limit = int(params.get("limit", 50))
        if len(found) > limit:
            # Keep the most prominent, not the first: truncating in x order
            # would discard the strongest peaks whenever they are late in the
            # series.
            found.sort(key=lambda peak: peak.prominence, reverse=True)
            found = found[:limit]

        return found

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_to_frame(self, result: PeakResult) -> pd.DataFrame:
        return result.to_frame()

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: PeakResult,
    ) -> ResultSeriesSpec:
        del axis_id
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                "generated_peaks": True,
                "peaks_dialog": "series_peaks",
                "source_name": result.source_name,
                "model": result.model,
                # Markers with no connecting line: the peaks are separate
                # findings, and joining them would draw a curve through
                # nothing.
                "linestyle": "",
                "marker": "v",
                "markersize": 8.0,
            },
        )

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_peaks": True, "peaks_dialog": "series_peaks"}

    def result_table_name(self, axis_id: int, result: PeakResult) -> str:
        return generated_table_name(
            f"Peaks_axis{axis_id}_{result.source_name}",
            fallback="Peaks_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Peaks"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[PeakResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        sections: list[str] = []
        for result in results:
            if not result.peaks:
                sections.append(
                    report_html.section(
                        result.source_name,
                        report_html.note(
                            _(
                                "No peaks passed the filter. Lower the minimum, "
                                "or switch from height to prominence if the "
                                "baseline is not at zero."
                            )
                        ),
                    )
                )
                continue

            rows = [
                (
                    str(index + 1),
                    report_html.format_number(peak.x),
                    report_html.format_number(peak.y),
                    report_html.format_number(peak.prominence, digits=4),
                    report_html.format_number(peak.width, digits=4),
                    f"{report_html.format_number(peak.left_x)} .. "
                    f"{report_html.format_number(peak.right_x)}",
                    _("minimum") if peak.is_minimum else _("maximum"),
                )
                for index, peak in enumerate(result.peaks)
            ]
            sections.append(
                report_html.section(
                    f"{result.source_name} \u2014 {len(result.peaks)}",
                    report_html.table(
                        (
                            "#",
                            "x",
                            "y",
                            _("Prominence"),
                            _("Width"),
                            _("Half-prominence bounds"),
                            _("Kind"),
                        ),
                        rows,
                        align=(
                            "right", "right", "right", "right", "right",
                            "right", "left",
                        ),
                    ),
                )
            )

        return report_html.document(_("Peaks"), self._model(), *sections)
