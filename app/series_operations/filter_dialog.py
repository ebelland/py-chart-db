"""Frequency-selective filtering and analytic-signal work (scipy.signal).

The gap between the two scipy.signal-based operations that already exist:
**Spectral Analysis** estimates a spectrum but never changes the series
itself, and **Smoothing** only ever lowpasses (one Butterworth model among
several denoising ones). This dialog is the frequency-domain complement:

* **IIR filter** - ``iirfilter`` -> ``sosfiltfilt``, zero-phase. Butterworth,
  Chebyshev I/II, Bessel or Elliptic; lowpass, highpass, bandpass or
  bandstop.
* **FIR filter** - ``firwin`` -> ``filtfilt``, zero-phase, linear-phase
  design (no ringing on a step the way an IIR filter can have).
* **Analytic signal** - ``hilbert``: amplitude envelope, instantaneous
  phase, or instantaneous frequency. AM demodulation, phase-based work.
* **Detrend** - ``scipy.signal.detrend``: remove a linear or constant trend
  before any of the above, or on its own.

fs (the sampling frequency) is derived from the median spacing of the
selected series' x role, exactly as the Spectral Analysis dialog does it -
including the same warning when the spacing is not uniform enough to trust.
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import detrend as scipy_detrend
from scipy.signal import filtfilt, firwin, hilbert, iirfilter, sosfiltfilt

from app.data.data_source import row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.styles.style import create_doc_link, set_doc_link
from app.utils.i18n import _

FILTER_IIR = "IIR filter"
FILTER_FIR = "FIR filter"
FILTER_ANALYTIC = "Analytic signal (Hilbert)"
FILTER_DETREND = "Detrend"
FILTER_MODELS = (FILTER_IIR, FILTER_FIR, FILTER_ANALYTIC, FILTER_DETREND)

#: Models that read fs (a sampling rate) rather than a bare sample count.
NEEDS_FS = frozenset({FILTER_IIR, FILTER_FIR, FILTER_ANALYTIC})
#: Models with a response shape (lowpass/highpass/bandpass/bandstop).
NEEDS_RESPONSE = frozenset({FILTER_IIR, FILTER_FIR})

RESP_LOWPASS, RESP_HIGHPASS, RESP_BANDPASS, RESP_BANDSTOP = (
    "lowpass", "highpass", "bandpass", "bandstop",
)
#: (label, value) - value is passed straight through to iirfilter's ``btype``
#: and firwin's ``pass_zero``, both of which accept these same four strings.
RESPONSES = (
    (_("Lowpass"), RESP_LOWPASS),
    (_("Highpass"), RESP_HIGHPASS),
    (_("Bandpass"), RESP_BANDPASS),
    (_("Bandstop"), RESP_BANDSTOP),
)
TWO_CUTOFF_RESPONSES = frozenset({RESP_BANDPASS, RESP_BANDSTOP})

#: value is iirfilter's ``ftype``.
IIR_FAMILIES = (
    (_("Butterworth"), "butter"),
    (_("Chebyshev I"), "cheby1"),
    (_("Chebyshev II"), "cheby2"),
    (_("Bessel"), "bessel"),
    (_("Elliptic"), "ellip"),
)
NEEDS_RIPPLE = frozenset({"cheby1", "ellip"})  # passband ripple, rp
NEEDS_ATTEN = frozenset({"cheby2", "ellip"})  # stopband attenuation, rs

FIR_WINDOWS = ("hamming", "hann", "blackman", "bartlett", "boxcar")

ANALYTIC_ENVELOPE, ANALYTIC_PHASE, ANALYTIC_FREQUENCY = "envelope", "phase", "frequency"
ANALYTIC_OUTPUTS = (
    (_("Amplitude envelope"), ANALYTIC_ENVELOPE),
    (_("Instantaneous phase"), ANALYTIC_PHASE),
    (_("Instantaneous frequency"), ANALYTIC_FREQUENCY),
)

DETREND_LINEAR, DETREND_CONSTANT = "linear", "constant"
DETREND_TYPES = ((_("Linear"), DETREND_LINEAR), (_("Constant"), DETREND_CONSTANT))

FILTER_DOCS = {
    FILTER_IIR: ("scipy.signal.iirfilter",
                 "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirfilter.html"),
    FILTER_FIR: ("scipy.signal.firwin",
                 "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html"),
    FILTER_ANALYTIC: ("scipy.signal.hilbert",
                       "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html"),
    FILTER_DETREND: ("scipy.signal.detrend",
                      "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html"),
}


@dataclass(slots=True)
class FilterResult:
    """One filtered/derived series for one source series."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"x": self.x, "y": self.y})


# ----------------------------------------------------------------------
# The numerics - pure, no Qt, so a test can call them directly
# ----------------------------------------------------------------------
def _check_cutoff(cutoff: float, fs: float) -> None:
    nyquist = fs / 2.0
    if not (0.0 < cutoff < nyquist):
        raise ValueError(
            f"a cutoff of {cutoff:g} must be between 0 and the Nyquist "
            f"frequency ({nyquist:g}, half of fs={fs:g})"
        )


def apply_iir_filter(
    y: np.ndarray, fs: float, *, family: str, response: str, order: int,
    cutoff: float, cutoff2: float | None = None,
    ripple: float | None = None, atten: float | None = None,
) -> np.ndarray:
    """Zero-phase IIR filter: ``iirfilter`` designs it, ``sosfiltfilt`` runs it."""
    _check_cutoff(cutoff, fs)
    wn: float | list[float] = cutoff
    if response in TWO_CUTOFF_RESPONSES:
        if cutoff2 is None:
            raise ValueError(f"{response} needs a second cutoff")
        _check_cutoff(cutoff2, fs)
        if cutoff2 <= cutoff:
            raise ValueError("the second cutoff must be higher than the first")
        wn = [cutoff, cutoff2]

    kwargs: dict[str, float] = {}
    if family in NEEDS_RIPPLE:
        kwargs["rp"] = float(ripple) if ripple is not None else 1.0
    if family in NEEDS_ATTEN:
        kwargs["rs"] = float(atten) if atten is not None else 40.0

    sos = iirfilter(order, wn, btype=response, ftype=family, output="sos", fs=fs, **kwargs)
    return np.asarray(sosfiltfilt(sos, y), dtype=float)


def apply_fir_filter(
    y: np.ndarray, fs: float, *, numtaps: int, window: str, response: str,
    cutoff: float, cutoff2: float | None = None,
) -> np.ndarray:
    """Zero-phase FIR filter: ``firwin`` designs it, ``filtfilt`` runs it."""
    _check_cutoff(cutoff, fs)
    cutoff_arg: float | list[float] = cutoff
    if response in TWO_CUTOFF_RESPONSES:
        if cutoff2 is None:
            raise ValueError(f"{response} needs a second cutoff")
        _check_cutoff(cutoff2, fs)
        if cutoff2 <= cutoff:
            raise ValueError("the second cutoff must be higher than the first")
        cutoff_arg = [cutoff, cutoff2]

    # firwin requires an odd tap count for a filter that must pass Nyquist
    # (highpass, bandstop) - an even one has a zero response there instead.
    taps = int(numtaps)
    if response in (RESP_HIGHPASS, RESP_BANDSTOP) and taps % 2 == 0:
        taps += 1

    coefficients = firwin(taps, cutoff_arg, window=window, pass_zero=response, fs=fs)
    return np.asarray(filtfilt(coefficients, [1.0], y), dtype=float)


def analytic_signal(y: np.ndarray, fs: float, output: str) -> np.ndarray:
    """Envelope, unwrapped phase, or instantaneous frequency of ``y``.

    Frequency is one sample shorter than the input (it comes from a
    difference of the phase) and the last value is repeated so the result
    still lines up with the source's x column.
    """
    analytic = hilbert(y)
    if output == ANALYTIC_ENVELOPE:
        return np.abs(analytic)

    phase = np.unwrap(np.angle(analytic))
    if output == ANALYTIC_PHASE:
        return phase

    frequency = np.diff(phase) / (2.0 * np.pi) * fs
    return np.concatenate([frequency, frequency[-1:]]) if frequency.size else frequency


def apply_detrend(y: np.ndarray, kind: str) -> np.ndarray:
    """Remove a linear or constant trend (against sample index, not x)."""
    return np.asarray(scipy_detrend(y, type=kind), dtype=float)


# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------
class SeriesFilterDialog(SeriesOperationDialogBase):
    """Filter, detrend or demodulate a series (scipy.signal)."""

    Name: str = "Filtering"
    Description = "Filter, detrend or demodulate a series (scipy.signal)"

    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    INPUT_MINIMUM_POINTS = 8

    RESULTS_ARE_HTML = False

    Icon = """
    <path d="M3 12h4l2-7 4 14 2-7h6"/>
    """

    def __init__(
        self, *, repo: SqliteRepo, figure_id: int, parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesFilterDialog requires a repository instance.")

        self._last_results: list[FilterResult] = []
        super().__init__(
            repo=repo, figure_id=figure_id, title="Filtering", parent=parent,
            width=820, height=660,
        )
        self.series_selector.reload(select_all_series=True)
        self._refresh_visibility()
        self.refresh_results()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        self._doc_link = create_doc_link(self)
        self._fs_auto_check = QCheckBox("", self)
        self._fs_spin = QDoubleSpinBox(self)
        self._family_combo = QComboBox(self)
        self._response_combo = QComboBox(self)
        self._order_spin = QSpinBox(self)
        self._numtaps_spin = QSpinBox(self)
        self._window_combo = QComboBox(self)
        self._cutoff1_spin = QDoubleSpinBox(self)
        self._cutoff2_spin = QDoubleSpinBox(self)
        self._ripple_spin = QDoubleSpinBox(self)
        self._atten_spin = QDoubleSpinBox(self)
        self._analytic_combo = QComboBox(self)
        self._detrend_combo = QComboBox(self)

    def build_model_selector(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo.addItems(FILTER_MODELS)
        self.model_combo.setToolTip(_("Choose the operation."))
        form.addRow(_("Operation:"), self.model_combo)
        form.addRow(_("Docs:"), self._doc_link)

        layout.addLayout(form)
        return panel

    def build_parameter_selector(self) -> QWidget:
        widget = QWidget(self)
        self._parameter_form = QFormLayout(widget)
        self._parameter_form.setContentsMargins(0, 0, 0, 0)
        self._parameter_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._fs_auto_check.setChecked(True)
        self._fs_auto_check.setToolTip(
            _("Take the sampling frequency from the spacing of the x role.")
        )
        self._parameter_form.addRow(_("Sampling rate:"), self._fs_auto_check)

        self._fs_spin.setRange(1e-9, 1e12)
        self._fs_spin.setDecimals(6)
        self._fs_spin.setValue(1.0)
        self._fs_spin.setToolTip(_("Samples per unit of x, used when not derived."))
        self._parameter_form.addRow(_("fs:"), self._fs_spin)

        for label, value in IIR_FAMILIES:
            self._family_combo.addItem(label, value)
        self._family_combo.setToolTip(_("The filter family. Butterworth is maximally flat; "
                                         "Chebyshev/Elliptic trade ripple for a sharper roll-off."))
        self._parameter_form.addRow(_("Family:"), self._family_combo)

        for label, value in RESPONSES:
            self._response_combo.addItem(label, value)
        self._response_combo.setToolTip(_("Which band the filter keeps."))
        self._parameter_form.addRow(_("Response:"), self._response_combo)

        self._order_spin.setRange(1, 20)
        self._order_spin.setValue(4)
        self._order_spin.setToolTip(_("Filter order. Higher is a sharper roll-off "
                                       "and more phase distortion before sosfiltfilt cancels it."))
        self._parameter_form.addRow(_("Order:"), self._order_spin)

        self._numtaps_spin.setRange(3, 20001)
        self._numtaps_spin.setValue(101)
        self._numtaps_spin.setToolTip(_("Number of FIR coefficients. More taps sharpen the "
                                         "transition band at the cost of needing more samples."))
        self._parameter_form.addRow(_("Taps:"), self._numtaps_spin)

        self._window_combo.addItems(FIR_WINDOWS)
        self._window_combo.setToolTip(_("Taper applied to the FIR coefficients."))
        self._parameter_form.addRow(_("Window:"), self._window_combo)

        self._cutoff1_spin.setRange(1e-9, 1e12)
        self._cutoff1_spin.setDecimals(6)
        self._cutoff1_spin.setValue(1.0)
        self._cutoff1_spin.setToolTip(_("Cutoff frequency (same units as fs)."))
        self._parameter_form.addRow(_("Cutoff:"), self._cutoff1_spin)

        self._cutoff2_spin.setRange(1e-9, 1e12)
        self._cutoff2_spin.setDecimals(6)
        self._cutoff2_spin.setValue(5.0)
        self._cutoff2_spin.setToolTip(_("Second cutoff, for a bandpass/bandstop response."))
        self._parameter_form.addRow(_("Second cutoff:"), self._cutoff2_spin)

        self._ripple_spin.setRange(0.001, 20.0)
        self._ripple_spin.setDecimals(3)
        self._ripple_spin.setValue(1.0)
        self._ripple_spin.setToolTip(_("Passband ripple, in dB (Chebyshev I / Elliptic)."))
        self._parameter_form.addRow(_("Ripple (dB):"), self._ripple_spin)

        self._atten_spin.setRange(1.0, 200.0)
        self._atten_spin.setDecimals(1)
        self._atten_spin.setValue(40.0)
        self._atten_spin.setToolTip(_("Minimum stopband attenuation, in dB (Chebyshev II / Elliptic)."))
        self._parameter_form.addRow(_("Attenuation (dB):"), self._atten_spin)

        for label, value in ANALYTIC_OUTPUTS:
            self._analytic_combo.addItem(label, value)
        self._analytic_combo.setToolTip(_("What to derive from the analytic signal."))
        self._parameter_form.addRow(_("Output:"), self._analytic_combo)

        for label, value in DETREND_TYPES:
            self._detrend_combo.addItem(label, value)
        self._detrend_combo.setToolTip(_("Linear removes a best-fit line; "
                                          "constant removes only the mean."))
        self._parameter_form.addRow(_("Trend:"), self._detrend_combo)

        return widget

    def connect_operation_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self._family_combo.currentIndexChanged.connect(self._refresh_visibility)
        self._response_combo.currentIndexChanged.connect(self._refresh_visibility)

        for combo in (
            self.model_combo, self._family_combo, self._response_combo,
            self._window_combo, self._analytic_combo, self._detrend_combo,
        ):
            combo.currentIndexChanged.connect(self.refresh_results)
        for check in (self._fs_auto_check,):
            check.toggled.connect(self.refresh_results)
        for spin in (
            self._fs_spin, self._order_spin, self._numtaps_spin,
            self._cutoff1_spin, self._cutoff2_spin, self._ripple_spin, self._atten_spin,
        ):
            spin.valueChanged.connect(self.refresh_results)

    def _model(self) -> str:
        return self.model_combo.currentText() or FILTER_IIR

    def _refresh_visibility(self) -> None:
        model = self._model()
        is_iir = model == FILTER_IIR
        is_fir = model == FILTER_FIR
        needs_response = model in NEEDS_RESPONSE
        response = self._response_combo.currentData()
        needs_second_cutoff = needs_response and response in TWO_CUTOFF_RESPONSES
        family = self._family_combo.currentData()

        self.set_row_visible(self._fs_auto_check, model in NEEDS_FS)
        self.set_row_visible(self._fs_spin, model in NEEDS_FS and not self._fs_auto_check.isChecked())
        self.set_row_visible(self._family_combo, is_iir)
        self.set_row_visible(self._response_combo, needs_response)
        self.set_row_visible(self._order_spin, is_iir)
        self.set_row_visible(self._numtaps_spin, is_fir)
        self.set_row_visible(self._window_combo, is_fir)
        self.set_row_visible(self._cutoff1_spin, needs_response)
        self.set_row_visible(self._cutoff2_spin, needs_second_cutoff)
        self.set_row_visible(self._ripple_spin, is_iir and family in NEEDS_RIPPLE)
        self.set_row_visible(self._atten_spin, is_iir and family in NEEDS_ATTEN)
        self.set_row_visible(self._analytic_combo, model == FILTER_ANALYTIC)
        self.set_row_visible(self._detrend_combo, model == FILTER_DETREND)

        title, url = FILTER_DOCS.get(model, ("", ""))
        set_doc_link(self._doc_link, title, url)

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
    def _sampling_frequency(self, x_values: np.ndarray, name: str) -> float:
        """Return fs in samples per unit of x - same estimator and the same
        non-uniform-sampling warning as the Spectral Analysis dialog."""
        if not self._fs_auto_check.isChecked():
            return float(self._fs_spin.value())
        if x_values.size < 2:
            return 1.0

        spacing = np.diff(x_values)
        median_spacing = float(np.median(spacing))
        if median_spacing <= 0.0:
            applogger.warning(
                "Series '%s' has a non-increasing x role; assuming fs = 1.",
                name, show_dialog=False, raise_error=False,
            )
            return 1.0

        deviation = float(np.max(np.abs(spacing - median_spacing)) / median_spacing)
        if deviation > 0.01:
            applogger.warning(
                "Series '%s' is not uniformly sampled (spacing varies by %.1f %%); "
                "fs is approximate.",
                name, deviation * 100.0, show_dialog=False, raise_error=False,
            )
        return 1.0 / median_spacing

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def _apply_model(self, model: str, y: np.ndarray, fs: float) -> tuple[np.ndarray, dict[str, Any]]:
        if model == FILTER_IIR:
            family = str(self._family_combo.currentData())
            response = str(self._response_combo.currentData())
            cutoff2 = float(self._cutoff2_spin.value()) if response in TWO_CUTOFF_RESPONSES else None
            y_out = apply_iir_filter(
                y, fs, family=family, response=response,
                order=int(self._order_spin.value()),
                cutoff=float(self._cutoff1_spin.value()), cutoff2=cutoff2,
                ripple=float(self._ripple_spin.value()) if family in NEEDS_RIPPLE else None,
                atten=float(self._atten_spin.value()) if family in NEEDS_ATTEN else None,
            )
            meta = {
                "family": self._family_combo.currentText(),
                "response": self._response_combo.currentText(),
                "order": int(self._order_spin.value()),
                "fs": fs,
            }
            return y_out, meta

        if model == FILTER_FIR:
            response = str(self._response_combo.currentData())
            cutoff2 = float(self._cutoff2_spin.value()) if response in TWO_CUTOFF_RESPONSES else None
            y_out = apply_fir_filter(
                y, fs, numtaps=int(self._numtaps_spin.value()),
                window=self._window_combo.currentText(), response=response,
                cutoff=float(self._cutoff1_spin.value()), cutoff2=cutoff2,
            )
            meta = {
                "response": self._response_combo.currentText(),
                "taps": int(self._numtaps_spin.value()),
                "window": self._window_combo.currentText(),
                "fs": fs,
            }
            return y_out, meta

        if model == FILTER_ANALYTIC:
            output = str(self._analytic_combo.currentData())
            y_out = analytic_signal(y, fs, output)
            return y_out, {"output": self._analytic_combo.currentText(), "fs": fs}

        # FILTER_DETREND
        kind = str(self._detrend_combo.currentData())
        y_out = apply_detrend(y, kind)
        return y_out, {"trend": self._detrend_combo.currentText()}

    def compute_results(self) -> list[FilterResult]:
        model = self._model()
        results: list[FilterResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self.series_xy(row, name)
                fs = self._sampling_frequency(x_values, name) if model in NEEDS_FS else 1.0
                y_out, meta = self._apply_model(model, y_values, fs)
                if y_out.size != x_values.size:
                    raise ValueError(
                        f"the result has {y_out.size} points but the series has "
                        f"{x_values.size}"
                    )
                results.append(
                    FilterResult(
                        source_name=name,
                        result_name=f"{name} - {model.split(' ')[0]}",
                        model=model,
                        x=x_values,
                        y=y_out,
                        metadata=meta,
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
    def result_to_frame(self, result: FilterResult) -> pd.DataFrame:
        return result.to_frame()

    def result_series_spec(
        self, axis_id: int, table_name: str, result: FilterResult,
    ) -> ResultSeriesSpec:
        del axis_id
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                "generated_filter": True,
                "filter_dialog": "series_filter",
                "source_name": result.source_name,
                "model": result.model,
                "linestyle": "-",
                "linewidth": 1.6,
                "marker": "",
            },
        )

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_filter": True, "filter_dialog": "series_filter"}

    def result_table_name(self, axis_id: int, result: FilterResult) -> str:
        return generated_table_name(
            f"Filter_axis{axis_id}_{result.source_name}_{result.model.split(' ')[0]}",
            fallback="Filter_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Filtering"

    def format_results(self, results: Sequence[FilterResult]) -> str:
        if not results:
            return _("No results.")
        lines = []
        for result in results:
            details = ", ".join(f"{key}={value}" for key, value in result.metadata.items())
            lines.append(f"{result.source_name}: {result.model} ({details})")
        return "\n".join(lines)
