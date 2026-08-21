"""Dialog for spectral and correlation analysis of chart series.

Estimators, and why each is here:

* **PSD (Welch)** - how power is distributed over frequency for one signal.
* **CSD (Welch)** - the same for a *pair*, showing where two signals share
  power and with what phase.
* **Coherence** - the normalised version of the CSD: how linearly related two
  signals are per frequency, on a 0-1 scale that is comparable across data.
* **Magnitude / phase / angle spectrum** - the raw one-sided FFT, for when the
  signal is deterministic rather than a noise process and Welch's averaging
  would smear the very peaks being measured.
* **Autocorrelation / cross-correlation** - the time-domain view: at what lag
  does a signal repeat, or does one signal lead another.

Two-input estimators (CSD, coherence, cross-correlation) pair the **first
selected series** with each of the others. That rule is arbitrary but it has to
be *some* rule, and "first is the reference" is the one a user can predict.

Sampling frequency is derived from the x role when it is uniformly spaced,
because a frequency axis in samples-per-x is meaningless otherwise; the value
can be overridden. Results are written to normal tables and attached to the
axis as ordinary series, so they persist in the .dhub and re-render like any
other data.
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
from scipy import signal as scipy_signal

from app.data.data_source import parse_roles, row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
import html

from app.utils.messages import show_message
from app.widgets import report_html
from app.series_operations.series_operation_dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
)
from app.styles.style import create_doc_link, set_doc_link
from app.utils.i18n import _

# ----------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------
METHOD_PSD = "Power spectral density (Welch)"
METHOD_CSD = "Cross spectral density (Welch)"
METHOD_COHERENCE = "Coherence"
METHOD_MAGNITUDE = "Magnitude spectrum"
METHOD_PHASE = "Phase spectrum (unwrapped)"
METHOD_ANGLE = "Angle spectrum (wrapped)"
METHOD_ACORR = "Autocorrelation"
METHOD_XCORR = "Cross-correlation"
METHOD_LAPLACE = "Laplace transform (damped FFT)"
METHOD_WAVELET = "Wavelet power spectrum (Morlet)"

SPECTRAL_METHODS: tuple[str, ...] = (
    METHOD_PSD,
    METHOD_CSD,
    METHOD_COHERENCE,
    METHOD_MAGNITUDE,
    METHOD_PHASE,
    METHOD_ANGLE,
    METHOD_LAPLACE,
    METHOD_WAVELET,
    METHOD_ACORR,
    METHOD_XCORR,
)

# Methods that consume a pair of signals rather than one.
PAIRED_METHODS: frozenset[str] = frozenset(
    {METHOD_CSD, METHOD_COHERENCE, METHOD_XCORR}
)

# Methods whose output lives in the frequency domain.
FREQUENCY_METHODS: frozenset[str] = frozenset(
    {
        METHOD_PSD,
        METHOD_CSD,
        METHOD_COHERENCE,
        METHOD_MAGNITUDE,
        METHOD_PHASE,
        METHOD_ANGLE,
        METHOD_LAPLACE,
        METHOD_WAVELET,
    }
)

# Methods that use Welch segmentation (nperseg / overlap / window).
WELCH_METHODS: frozenset[str] = frozenset({METHOD_PSD, METHOD_CSD, METHOD_COHERENCE})

WINDOWS: tuple[str, ...] = (
    "hann",
    "hamming",
    "blackman",
    "bartlett",
    "flattop",
    "boxcar",
)

DETREND_MODES: tuple[str, ...] = ("constant", "linear", "none")

CORRELATION_NORMALISATIONS: tuple[str, ...] = ("unbiased", "biased", "none")

METHOD_DOCS: dict[str, tuple[str, str]] = {
    METHOD_PSD: (
        "scipy.signal.welch",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html",
    ),
    METHOD_CSD: (
        "scipy.signal.csd",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html",
    ),
    METHOD_COHERENCE: (
        "scipy.signal.coherence",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html",
    ),
    METHOD_MAGNITUDE: (
        "Magnitude spectrum",
        "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.magnitude_spectrum.html",
    ),
    METHOD_PHASE: (
        "Phase spectrum",
        "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.phase_spectrum.html",
    ),
    METHOD_ANGLE: (
        "Angle spectrum",
        "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.angle_spectrum.html",
    ),
    METHOD_ACORR: (
        "Autocorrelation",
        "https://numpy.org/doc/stable/reference/generated/numpy.correlate.html",
    ),
    METHOD_XCORR: (
        "Cross-correlation",
        "https://numpy.org/doc/stable/reference/generated/numpy.correlate.html",
    ),
    METHOD_LAPLACE: (
        "Laplace transform",
        "https://en.wikipedia.org/wiki/Laplace_transform",
    ),
    METHOD_WAVELET: (
        "Continuous wavelet transform",
        "https://en.wikipedia.org/wiki/Continuous_wavelet_transform",
    ),
}


@dataclass(slots=True)
class SpectralResult:
    """One spectral or correlation estimate, ready to save and plot."""

    source_name: str
    result_name: str
    model: str
    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return the two-column frame written to the result table."""
        return pd.DataFrame({self.x_label: self.x, self.y_label: self.y})


class SeriesSpectralDialog(SeriesOperationDialogBase):
    """Compute spectra and correlations and attach them to the chart."""

    # format_results builds a table; without this the pane would show the
    # markup as literal text.
    Name: str = "Spectral Analysis"
    Description = "Analyse frequencies"

    # An FFT maps sample index to frequency, so it assumes a constant step.
    # Uneven x does not fail - it returns a spectrum whose frequency axis is
    # meaningless, which is the worst of both worlds, hence the warning.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIFORM_X = True
    INPUT_MINIMUM_POINTS = 4

    Icon = """
    <path d="M4 18.5h16"/>
    <path d="M4.5 18V5"/>
    <path d="M7 16v-3"/>
    <path d="M10 16V8"/>
    <path d="M13 16v-6"/>
    <path d="M16 16V6"/>
    <path d="M19 16v-4"/>
    """
    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error(
                "SeriesSpectralDialog requires a repository instance.",
                show_dialog=True,
                raise_error=True,
            )

        self._last_results: list[SpectralResult] = []
        self._parameter_form: QFormLayout | None = None

        # The axis this dialog adds to the current figure for its results.
        # Created on the first Preview, kept across further previews, deleted
        # on Close unless Apply has confirmed it.
        self._result_axis_id: int | None = None
        self._applied = False

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Spectral Analysis",
            parent=parent,
            width=900,
            height=680,
        )
        self.setModal(True)
        self.series_selector.set_series_filter(self._has_query)
        self.series_selector.reload(select_all_series=False)
        self._refresh_visibility()
        self.refresh_results()

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------
    def init_operation_widgets(self) -> None:
        """Create the controls before the base class builds the panels."""
        self._doc_link = create_doc_link(self)
        self._fs_auto_check = QCheckBox(_("Derive from the x role"), self)
        self._fs_spin = QDoubleSpinBox(self)
        self._nperseg_spin = QSpinBox(self)
        self._overlap_spin = QDoubleSpinBox(self)
        self._window_combo = QComboBox(self)
        self._detrend_combo = QComboBox(self)
        self._scaling_combo = QComboBox(self)
        self._onesided_check = QCheckBox(_("One-sided spectrum"), self)
        self._db_check = QCheckBox(_("Convert to decibels"), self)
        self._maxlags_spin = QSpinBox(self)
        self._sigma_spin = QDoubleSpinBox(self)
        self._wavelet_w0_spin = QDoubleSpinBox(self)
        self._wavelet_scales_spin = QSpinBox(self)
        self._corr_norm_combo = QComboBox(self)
        self._parameter_form = None

    def build_model_selector(self) -> QWidget:
        """Method combo plus a documentation link for the selected method."""
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo.addItems(SPECTRAL_METHODS)
        self.model_combo.setToolTip(_("Choose the spectral or correlation estimator."))
        form.addRow(_("Method:"), self.model_combo)

        form.addRow(_("Docs:"), self._doc_link)

        layout.addLayout(form)
        return panel

    def build_parameter_selector(self) -> QWidget:
        """Parameters, shown and hidden per method by _refresh_visibility."""
        widget = QWidget(self)
        self._parameter_form = QFormLayout(widget)
        self._parameter_form.setContentsMargins(0, 0, 0, 0)
        self._parameter_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

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

        self._nperseg_spin.setRange(8, 1_048_576)
        self._nperseg_spin.setValue(256)
        self._nperseg_spin.setToolTip(
            _("Samples per Welch segment. Longer segments resolve finer frequency "
            "detail; shorter ones average away more noise.")
        )
        self._parameter_form.addRow(_("Segment length:"), self._nperseg_spin)

        self._overlap_spin.setRange(0.0, 0.95)
        self._overlap_spin.setDecimals(2)
        self._overlap_spin.setSingleStep(0.05)
        self._overlap_spin.setValue(0.5)
        self._overlap_spin.setToolTip(_("Fraction of each segment shared with the next."))
        self._parameter_form.addRow(_("Segment overlap:"), self._overlap_spin)

        self._window_combo.addItems(WINDOWS)
        self._window_combo.setToolTip(_("Taper applied to each segment before the FFT."))
        self._parameter_form.addRow(_("Window:"), self._window_combo)

        self._sigma_spin.setRange(0.0, 1e6)
        self._sigma_spin.setDecimals(6)
        self._sigma_spin.setSingleStep(0.1)
        self._sigma_spin.setValue(0.0)
        self._sigma_spin.setToolTip(
            _("Damping sigma of the Laplace variable s = sigma + i*omega. "
            "Zero reduces the transform to the Fourier transform; increasing it "
            "weights early samples more and suppresses late ones, which is what "
            "makes a decaying or unstable signal integrable.")
        )
        self._parameter_form.addRow(_("Damping (sigma):"), self._sigma_spin)

        self._wavelet_w0_spin.setRange(3.0, 30.0)
        self._wavelet_w0_spin.setDecimals(1)
        self._wavelet_w0_spin.setSingleStep(0.5)
        self._wavelet_w0_spin.setValue(6.0)
        self._wavelet_w0_spin.setToolTip(
            _("Morlet central frequency w0. Higher values resolve frequency more "
            "finely at the cost of time resolution; 6 is the usual compromise "
            "and the value at which the wavelet is near-admissible.")
        )
        self._parameter_form.addRow(_("Morlet w0:"), self._wavelet_w0_spin)

        self._wavelet_scales_spin.setRange(8, 512)
        self._wavelet_scales_spin.setValue(64)
        self._wavelet_scales_spin.setToolTip(
            _("How many scales to evaluate between the longest period the record "
            "supports and the Nyquist frequency, spaced logarithmically.")
        )
        self._parameter_form.addRow(_("Scales:"), self._wavelet_scales_spin)

        self._detrend_combo.addItems(DETREND_MODES)
        self._detrend_combo.setToolTip(
            _("Remove a constant or linear trend from each segment first. A trend "
            "leaks into the lowest frequency bins and hides everything near it.")
        )
        self._parameter_form.addRow(_("Detrend:"), self._detrend_combo)

        self._scaling_combo.addItems(("density", "spectrum"))
        self._scaling_combo.setToolTip(
            _("density: power per unit frequency. spectrum: power per segment.")
        )
        self._parameter_form.addRow(_("Scaling:"), self._scaling_combo)

        self._onesided_check.setChecked(True)
        self._onesided_check.setToolTip(
            _("Fold the negative frequencies onto the positive ones. Correct for "
            "real-valued signals.")
        )
        self._parameter_form.addRow("", self._onesided_check)

        self._db_check.setToolTip(_("Express the result as 10*log10 of the value."))
        self._parameter_form.addRow("", self._db_check)

        self._maxlags_spin.setRange(0, 1_000_000)
        self._maxlags_spin.setValue(0)
        self._maxlags_spin.setSpecialValueText(_("all lags"))
        self._maxlags_spin.setToolTip(_("Largest lag to keep. 0 keeps every lag."))
        self._parameter_form.addRow(_("Max lags:"), self._maxlags_spin)

        self._corr_norm_combo.addItems(CORRELATION_NORMALISATIONS)
        self._corr_norm_combo.setToolTip(
            _("unbiased divides by the overlap at each lag; biased divides by N; "
            "none returns the raw sum of products.")
        )
        self._parameter_form.addRow(_("Normalisation:"), self._corr_norm_combo)

        return widget

    def connect_operation_signals(self) -> None:
        """Recompute the preview whenever a parameter changes."""
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.model_combo.currentIndexChanged.connect(self.refresh_results)
        self._fs_auto_check.toggled.connect(self._refresh_visibility)
        self._fs_auto_check.toggled.connect(self.refresh_results)

        for widget in (
            self._fs_spin,
            self._overlap_spin,
        ):
            widget.valueChanged.connect(self.refresh_results)
        for widget in (self._nperseg_spin, self._maxlags_spin):
            widget.valueChanged.connect(self.refresh_results)
        for widget in (
            self._window_combo,
            self._detrend_combo,
            self._scaling_combo,
            self._corr_norm_combo,
        ):
            widget.currentIndexChanged.connect(self.refresh_results)
        for widget in (self._onesided_check, self._db_check):
            widget.toggled.connect(self.refresh_results)
        for widget in (
            self._sigma_spin,
            self._wavelet_w0_spin,
            self._wavelet_scales_spin,
        ):
            widget.valueChanged.connect(self.refresh_results)

    def _refresh_visibility(self) -> None:
        """Show only the parameters the selected method actually uses."""
        if self._parameter_form is None:
            return

        method = self.model_combo.currentText()
        is_welch = method in WELCH_METHODS
        is_frequency = method in FREQUENCY_METHODS
        is_correlation = not is_frequency

        name, url = METHOD_DOCS.get(method, ("", ""))
        set_doc_link(self._doc_link, name, url)

        self.set_row_visible(self._fs_auto_check, is_frequency)
        self.set_row_visible(self._fs_spin, is_frequency and not self._fs_auto_check.isChecked())
        self.set_row_visible(self._nperseg_spin, is_welch)
        self.set_row_visible(self._overlap_spin, is_welch)
        self.set_row_visible(self._window_combo, is_welch)
        self.set_row_visible(self._detrend_combo, is_welch)
        self.set_row_visible(self._scaling_combo, method in {METHOD_PSD, METHOD_CSD})
        self.set_row_visible(self._onesided_check, is_welch)
        self.set_row_visible(
            self._db_check,
            method in {
                METHOD_PSD,
                METHOD_CSD,
                METHOD_MAGNITUDE,
                METHOD_LAPLACE,
                METHOD_WAVELET,
            },
        )
        self.set_row_visible(self._maxlags_spin, is_correlation)
        self.set_row_visible(self._corr_norm_combo, is_correlation)
        self.set_row_visible(self._sigma_spin, method == METHOD_LAPLACE)
        self.set_row_visible(self._wavelet_w0_spin, method == METHOD_WAVELET)
        self.set_row_visible(self._wavelet_scales_spin, method == METHOD_WAVELET)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    @staticmethod
    def _has_query(row: Any) -> bool:
        """Only expose series backed by an SQL query."""
        return bool(row["sql_query"] != "")

    def _series_signal(self, row: Any) -> tuple[str, np.ndarray, np.ndarray]:
        """Return (name, x, y) for one selected series, finite values only."""
        roles = parse_roles(row_value(row, "roles"))
        frame = self._repo.query_df(str(row["sql_query"]))
        name = str(row["name"])

        y_column = str(roles.get("y", "y") or "y")
        if y_column not in frame.columns:
            y_column = "y"
        if y_column not in frame.columns:
            raise ValueError(f"series '{name}' has no y role")

        y_values = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=float)

        x_column = str(roles.get("x", "x") or "x")
        if x_column in frame.columns:
            x_values = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
        else:
            x_values = np.arange(y_values.size, dtype=float)

        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if int(np.count_nonzero(finite)) < 8:
            raise ValueError(f"series '{name}' has fewer than 8 usable points")

        x_finite = x_values[finite]
        y_finite = y_values[finite]

        order = np.argsort(x_finite, kind="stable")
        return name, x_finite[order], y_finite[order]

    def _sampling_frequency(self, x_values: np.ndarray, name: str) -> float:
        """Return fs in samples per unit of x.

        Derived from the median spacing when the user asked for it.  A spectrum
        of unevenly sampled data is not defined, so a non-uniform x is reported
        rather than silently averaged: the numbers would look fine and mean
        nothing.
        """
        if not self._fs_auto_check.isChecked():
            return float(self._fs_spin.value())

        if x_values.size < 2:
            return 1.0

        spacing = np.diff(x_values)
        median_spacing = float(np.median(spacing))
        if median_spacing <= 0.0:
            applogger.warning(
                "Series '%s' has a non-increasing x role; assuming fs = 1.",
                name,
                show_dialog=False,
                raise_error=False,
            )
            return 1.0

        deviation = float(np.max(np.abs(spacing - median_spacing)) / median_spacing)
        if deviation > 0.01:
            applogger.warning(
                "Series '%s' is not uniformly sampled (spacing varies by %.1f %%); "
                "the frequency axis is approximate.",
                name,
                deviation * 100.0,
                show_dialog=False,
                raise_error=False,
            )

        return 1.0 / median_spacing

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def _welch_kwargs(self, fs: float, sample_count: int) -> dict[str, Any]:
        """Shared Welch parameters, with nperseg clamped to the data length."""
        nperseg = min(int(self._nperseg_spin.value()), sample_count)
        nperseg = max(8, nperseg)
        overlap = float(self._overlap_spin.value())
        detrend = self._detrend_combo.currentText()

        return {
            "fs": fs,
            "window": self._window_combo.currentText(),
            "nperseg": nperseg,
            "noverlap": int(nperseg * overlap),
            "detrend": False if detrend == "none" else detrend,
            "return_onesided": bool(self._onesided_check.isChecked()),
        }

    def _to_decibels(self, values: np.ndarray) -> np.ndarray:
        """Return 10*log10(values), with zeros floored to the smallest positive.

        Why floor rather than drop: a zero bin is a real measurement and
        removing it would shift every later point on the frequency axis.
        """
        positive = values[values > 0.0]
        floor = float(positive.min()) if positive.size else 1e-20
        return 10.0 * np.log10(np.maximum(values, floor))

    def _one_sided_fft(self, y_values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, complex spectrum) for the positive half."""
        spectrum = np.fft.rfft(y_values - float(np.mean(y_values)))
        frequencies = np.fft.rfftfreq(y_values.size, d=1.0 / fs)
        return frequencies, spectrum

    def _laplace_spectrum(
        self, y_values: np.ndarray, fs: float, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, |F(sigma + i*omega)|) along a vertical line in s.

        The one-sided Laplace transform of a sampled signal is

            F(s) = dt * sum_k y_k * exp(-s * t_k),   s = sigma + i*omega

        and splitting the exponential turns that into the Fourier transform of
        a damped signal::

            F(sigma + i*omega) = dt * FFT[ y_k * exp(-sigma * t_k) ]

        so no new machinery is needed - one multiply before the FFT already in
        use.  What it buys is the part of the s-plane the Fourier transform
        cannot reach: a growing or non-decaying signal has no Fourier transform,
        but with sigma large enough the damped signal does, which is the whole
        reason the Laplace transform exists.

        sigma = 0 is exactly the Fourier magnitude spectrum, which makes the
        control easy to understand: turn it up and watch the transform become
        defined.
        """
        raw = np.asarray(y_values, dtype=float)
        times = np.arange(raw.size, dtype=float) / float(fs)

        # exp(-sigma * t) underflows to zero over a long record; that is
        # arithmetically right - those samples contribute nothing - but it is
        # worth not letting it produce a NaN through 0 * inf.
        damping = np.exp(-float(sigma) * times)
        damped = np.nan_to_num(raw * damping, nan=0.0, posinf=0.0, neginf=0.0)

        # The mean is removed *after* damping, not before.  Removing it first
        # subtracts the mean of the undamped signal, which for a growing signal
        # is enormous and leaves a ramp that swamps every real peak - a sine
        # multiplied by exp(1.5t), damped by exp(-1.5t), came back peaked at
        # DC instead of at its own frequency.  Damping first and centring the
        # result leaves exactly the sine, which is the point of choosing that
        # sigma.
        damped = damped - float(np.mean(damped))

        spectrum = np.fft.rfft(damped) / float(fs)
        frequencies = np.fft.rfftfreq(damped.size, d=1.0 / fs)
        return frequencies, np.abs(spectrum)

    def _wavelet_power(
        self, y_values: np.ndarray, fs: float, w0: float, n_scales: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, time-averaged power) of a Morlet CWT.

        A scalogram is two-dimensional and this dialog produces (x, y) series,
        so what is returned is the *global* wavelet spectrum: the power at each
        scale averaged over time.  That is the standard summary of a scalogram
        and is directly comparable with a Fourier power spectrum - with the
        difference that it is computed from a basis localised in time, so a
        frequency present in only part of the record still shows up without the
        leakage a single long FFT would spread around it.

        Computed by convolution in the Fourier domain, which is both the fast
        way and the only way that stays exact for a wavelet defined
        analytically in frequency::

            psi_hat(s*omega) = pi^-1/4 * H(omega) * exp(-(s*omega - w0)^2 / 2)

        ``H`` being the Heaviside step: the Morlet wavelet is analytic, so it
        has no negative-frequency content.  No PyWavelets dependency for one
        wavelet whose transform is four lines of numpy.
        """
        centred = np.asarray(y_values, dtype=float) - float(np.mean(y_values))
        n = centred.size

        # From the longest period the record can support to the Nyquist limit.
        # Anything outside that is not measurable from this data.
        lowest = max(float(fs) / float(n), 1e-12)
        highest = float(fs) / 2.0
        frequencies = np.logspace(np.log10(lowest), np.log10(highest), int(n_scales))

        # Morlet: the scale that responds to frequency f is w0 / (2*pi*f).
        scales = float(w0) / (2.0 * np.pi * frequencies)

        transformed = np.fft.fft(centred)
        omega = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / float(fs))

        power = np.empty(frequencies.size, dtype=float)
        for index, scale in enumerate(scales):
            scaled = scale * omega
            wavelet = (np.pi ** -0.25) * np.exp(-0.5 * (scaled - float(w0)) ** 2)
            wavelet[omega <= 0.0] = 0.0  # analytic: no negative frequencies
            # sqrt(scale) keeps power comparable across scales; without it the
            # spectrum slopes purely because wide wavelets integrate more.
            coefficients = np.fft.ifft(transformed * wavelet) * np.sqrt(scale)
            power[index] = float(np.mean(np.abs(coefficients) ** 2))

        return frequencies, power

    def _correlate(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lags, correlation) for two mean-removed signals."""
        length = min(first.size, second.size)
        a = first[:length] - float(np.mean(first[:length]))
        b = second[:length] - float(np.mean(second[:length]))

        raw = np.correlate(a, b, mode="full")
        lags = np.arange(-length + 1, length)

        mode = self._corr_norm_combo.currentText()
        if mode == "biased":
            raw = raw / float(length)
        elif mode == "unbiased":
            # Each lag averages a different number of overlapping samples;
            # dividing by that count removes the artificial taper towards the
            # extreme lags.
            raw = raw / (length - np.abs(lags))

        max_lags = int(self._maxlags_spin.value())
        if max_lags > 0:
            keep = np.abs(lags) <= max_lags
            lags = lags[keep]
            raw = raw[keep]

        return lags.astype(float), raw

    def compute_results(self) -> list[SpectralResult]:
        """Compute one result per selected series, or per pair when paired."""
        selected = self.selected_series()
        if not selected:
            return []

        method = self.model_combo.currentText()
        paired = method in PAIRED_METHODS

        if paired and len(selected) < 2:
            show_message(
                self,
                "series.needs_two_series",
                title=self.operation_label,
                method=method,
            )
            return []

        signals: list[tuple[str, np.ndarray, np.ndarray]] = []
        errors: list[str] = []
        for row in selected:
            try:
                signals.append(self._series_signal(row))
            except Exception as exc:
                errors.append(str(exc))

        if errors:
            show_message(
                self,
                "series.some_failed",
                title=self.operation_label,
                errors="\n".join(errors),
            )
        if not signals:
            return []

        results: list[SpectralResult] = []
        if paired:
            reference = signals[0]
            for other in signals[1:]:
                result = self._compute_pair(method, reference, other)
                if result is not None:
                    results.append(result)
        else:
            for entry in signals:
                result = self._compute_single(method, entry)
                if result is not None:
                    results.append(result)

        return results

    def _compute_single(
        self,
        method: str,
        entry: tuple[str, np.ndarray, np.ndarray],
    ) -> SpectralResult | None:
        """Compute a one-input estimate."""
        name, x_values, y_values = entry
        fs = self._sampling_frequency(x_values, name)

        try:
            if method == METHOD_PSD:
                frequencies, power = scipy_signal.welch(
                    y_values,
                    scaling=self._scaling_combo.currentText(),
                    **self._welch_kwargs(fs, y_values.size),
                )
                values = self._to_decibels(power) if self._db_check.isChecked() else power
                unit = "dB" if self._db_check.isChecked() else "power"
                return self._frequency_result(method, name, frequencies, values, unit)

            if method in {METHOD_MAGNITUDE, METHOD_PHASE, METHOD_ANGLE}:
                frequencies, spectrum = self._one_sided_fft(y_values, fs)
                if method == METHOD_MAGNITUDE:
                    values = np.abs(spectrum)
                    if self._db_check.isChecked():
                        values = self._to_decibels(values)
                        unit = "dB"
                    else:
                        unit = "magnitude"
                elif method == METHOD_PHASE:
                    values = np.unwrap(np.angle(spectrum))
                    unit = "radians"
                else:
                    values = np.angle(spectrum)
                    unit = "radians"
                return self._frequency_result(method, name, frequencies, values, unit)

            if method == METHOD_LAPLACE:
                frequencies, magnitude = self._laplace_spectrum(
                    y_values, fs, self._sigma_spin.value()
                )
                if self._db_check.isChecked():
                    magnitude = self._to_decibels(magnitude)
                    unit = "dB"
                else:
                    unit = "magnitude"
                return self._frequency_result(method, name, frequencies, magnitude, unit)

            if method == METHOD_WAVELET:
                frequencies, power = self._wavelet_power(
                    y_values,
                    fs,
                    self._wavelet_w0_spin.value(),
                    int(self._wavelet_scales_spin.value()),
                )
                if self._db_check.isChecked():
                    power = self._to_decibels(power)
                    unit = "dB"
                else:
                    unit = "power"
                return self._frequency_result(method, name, frequencies, power, unit)

            if method == METHOD_ACORR:
                lags, correlation = self._correlate(y_values, y_values)
                return SpectralResult(
                    source_name=name,
                    result_name=f"{name} - Autocorrelation",
                    model=method,
                    x=lags,
                    y=correlation,
                    x_label="lag",
                    y_label="correlation",
                    metadata={"fs": fs, "points": int(y_values.size)},
                )
        except Exception as exc:
            applogger.exception("Spectral estimate failed for '%s'", name)
            show_message(
                self,
                "series.estimate_failed",
                title=self.operation_label,
                series=name,
                error=exc,
            )
            return None

        applogger.error("Unhandled spectral method: %r", method, show_dialog=False, raise_error=False)
        return None

    def _compute_pair(
        self,
        method: str,
        reference: tuple[str, np.ndarray, np.ndarray],
        other: tuple[str, np.ndarray, np.ndarray],
    ) -> SpectralResult | None:
        """Compute a two-input estimate against the reference series."""
        reference_name, reference_x, reference_y = reference
        other_name, _other_x, other_y = other

        length = min(reference_y.size, other_y.size)
        if length < 8:
            applogger.warning(
                "Pair '%s' / '%s' skipped: fewer than 8 shared samples.",
                reference_name,
                other_name,
            )
            return None

        first = reference_y[:length]
        second = other_y[:length]
        fs = self._sampling_frequency(reference_x[:length], reference_name)
        pair_label = f"{reference_name} x {other_name}"

        try:
            if method == METHOD_CSD:
                frequencies, cross = scipy_signal.csd(
                    first,
                    second,
                    scaling=self._scaling_combo.currentText(),
                    **self._welch_kwargs(fs, length),
                )
                values = np.abs(cross)
                if self._db_check.isChecked():
                    values = self._to_decibels(values)
                    unit = "dB"
                else:
                    unit = "power"
                return self._frequency_result(method, pair_label, frequencies, values, unit)

            if method == METHOD_COHERENCE:
                welch_kwargs = self._welch_kwargs(fs, length)
                # coherence has no return_onesided or scaling parameter: it is
                # a ratio, so both would cancel.
                welch_kwargs.pop("return_onesided", None)
                frequencies, coherence = scipy_signal.coherence(
                    first, second, **welch_kwargs
                )
                return self._frequency_result(
                    method, pair_label, frequencies, coherence, "coherence"
                )

            if method == METHOD_XCORR:
                lags, correlation = self._correlate(first, second)
                return SpectralResult(
                    source_name=pair_label,
                    result_name=f"{pair_label} - Cross-correlation",
                    model=method,
                    x=lags,
                    y=correlation,
                    x_label="lag",
                    y_label="correlation",
                    metadata={"fs": fs, "points": int(length)},
                )
        except Exception as exc:
            applogger.exception("Spectral estimate failed for '%s'", pair_label)
            show_message(
                self,
                "series.estimate_failed",
                title=self.operation_label,
                series=pair_label,
                error=exc,
            )
            return None

        applogger.error("Unhandled paired method: %r", method, show_dialog=False, raise_error=False)
        return None

    @staticmethod
    def _frequency_result(
        method: str,
        source_name: str,
        frequencies: np.ndarray,
        values: np.ndarray,
        unit: str,
    ) -> SpectralResult:
        """Wrap a frequency-domain estimate in a result."""
        return SpectralResult(
            source_name=source_name,
            result_name=f"{source_name} - {method}",
            model=method,
            x=np.asarray(frequencies, dtype=float),
            y=np.asarray(values, dtype=float),
            x_label="frequency",
            y_label=unit,
            metadata={"points": int(np.size(frequencies))},
        )

    # ------------------------------------------------------------------
    # Base-class hooks
    # ------------------------------------------------------------------
    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        """Identify the series this dialog generates, for cleanup on re-apply."""
        return {"generated_spectral": True, "spectral_dialog": "series_spectral"}

    def apply(self) -> bool:
        """Apply, and keep the figure the results were written to."""
        applied = super().apply()
        # Only now is the new chart the user's rather than this dialog's.
        self._applied = self._applied or applied
        return applied

    # ------------------------------------------------------------------
    # The results get a chart of their own
    # ------------------------------------------------------------------
    def resolve_target_axis_id(self, selected_axis_id: int, results: Sequence[Any]) -> int:
        """Return an axis of this dialog's own, creating it once.

        A spectrum does not belong on the axis its input came from: the x axis
        stops being time and becomes frequency (or lag), so the two cannot
        share a scale, a label or a grid.

        A new axis on the *same figure*, not a new figure: the spectrum is a
        second view of the data in this chart, and a separate tab would put it
        where nobody compares it with the original.

        The axis is created on the first Preview and reused afterwards, so
        adjusting nperseg repeatedly does not leave a trail of empty axes.  If
        the dialog is closed without Apply, ``discard_operation_artifacts``
        removes it again.
        """
        del selected_axis_id

        if self._result_axis_id is None:
            self._result_axis_id = self.create_result_axis(
                chart_type="Scatter Plot",
                title=self.model_combo.currentText(),
                options={"grid": True, "linestyle": "-", "marker": ""},
            )

        self._label_result_axis(results)
        return self._result_axis_id

    def _label_result_axis(self, results: Sequence[Any]) -> None:
        """Name the axes after what the current estimate actually produced.

        The labels come from the results rather than from the method, because
        the same method yields different units depending on the options: a PSD
        is power or dB, a correlation is a lag axis rather than a frequency
        one.
        """
        if self._result_axis_id is None or not results:
            return

        first = results[0]
        x_label = str(getattr(first, "x_label", "") or "")
        y_label = str(getattr(first, "y_label", "") or "")
        if x_label == "frequency":
            x_label = "frequency [1/x]"

        try:
            self._repo.update_axis_descriptor(
                axis_id=self._result_axis_id,
                title=str(getattr(first, "model", "") or ""),
                x_label=x_label,
                y_label=y_label,
            )
        except Exception:
            applogger.exception("Failed to label the spectral result axis")

    def discard_operation_artifacts(self) -> None:
        """Delete the axis this dialog created, when Apply never happened.

        Closing without applying must leave the chart exactly as it was, and
        the axis is not covered by the preview savepoint: creating it commits.
        """
        if self._applied or self._result_axis_id is None:
            return

        axis_id = self._result_axis_id
        self._result_axis_id = None
        try:
            self._repo.delete_axis(axis_id)
            applogger.info("Discarded the unapplied spectral axis %s.", axis_id)
        except Exception:
            applogger.exception("Failed to discard spectral axis %s", axis_id)

    def result_to_frame(self, result: SpectralResult) -> pd.DataFrame:
        return result.to_frame()

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: SpectralResult,
    ) -> ResultSeriesSpec:
        """Attach the result as an ordinary x/y line series."""
        del axis_id
        quoted = f'"{table_name}"'
        style = dict(self.generated_style_filter)
        style.update(
            {
                "linestyle": "-",
                "marker": "",
                "label": result.result_name,
                "spectral_model": result.model,
            }
        )
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=(
                f'SELECT "{result.x_label}" AS x, "{result.y_label}" AS y FROM {quoted}'
            ),
            roles={"x": "x", "y": "y"},
            style=style,
        )

    def format_results(self, results: Sequence[SpectralResult]) -> str:
        """Return an HTML summary of the computed estimates."""
        if not results:
            return report_html.note("Select one or more source series.")

        rows = []
        for result in results:
            finite = result.y[np.isfinite(result.y)]
            peak = ""
            if finite.size and result.x.size == result.y.size:
                peak_index = int(np.nanargmax(np.abs(result.y)))
                peak = (
                    f"{report_html.format_number(result.x[peak_index])}"
                    f" &rarr; {report_html.format_number(result.y[peak_index])}"
                )
            rows.append(
                (
                    html.escape(result.source_name),
                    html.escape(result.model),
                    str(result.x.size),
                    peak,
                )
            )

        first = results[0]
        return report_html.document(
            "Spectral analysis",
            first.model,
            report_html.section(
                _("Estimates"),
                report_html.table(
                    [
                        "Source",
                        "Method",
                        "Points",
                        f"Peak ({first.x_label} &rarr; {first.y_label})",
                    ],
                    rows,
                    align=["left", "left", "right", "right"],
                ),
            ),
        )

    def refresh_results(self) -> None:
        """Recompute and show the preview text without touching the database."""
        try:
            results = list(self.compute_results())
        except Exception as exc:
            self._last_results = []
            # Always plain: the exception text is not markup.
            self.set_results_text(f"Error:\n{exc}")
            return

        self._last_results = results
        self.publish_results(self.format_results(results))
