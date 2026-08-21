"""Numerics for the four operations added after the review.

These were each verified against analytic results while being written; this is
that verification made permanent. The properties asserted are the ones that
would be silently wrong rather than loud - a derivative off by the sample
spacing, a control limit built from the wrong variance - because those are the
ones a smoke test would not catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.series_operations.series_calculus_dialog import (
    BASELINE_ENDPOINTS,
    BASELINE_MINIMUM,
    BASELINE_NONE,
    DERIV_GRADIENT,
    DERIV_SAVGOL,
    DERIV_SPLINE,
    INTEGRAL_CUMULATIVE,
    INTEGRAL_DEFINITE,
    SeriesCalculusDialog,
)
from app.series_operations.series_control_chart_dialog import (
    CHART_INDIVIDUALS,
    CHART_MOVING_RANGE,
    CHART_XBAR_R,
    CHART_XBAR_S,
    SPC_CONSTANTS,
    SeriesControlChartDialog,
)
from app.series_operations.series_function_dialog import (
    SPACING_LINEAR,
    SPACING_LOG,
    SeriesFunctionDialog,
)
from app.series_operations.series_peaks_dialog import (
    PEAKS_MAXIMA,
    PEAKS_MINIMA,
    SeriesPeaksDialog,
)


def _bare(cls):
    """Build an instance without its Qt dialog.

    The numerics are plain methods on the class and need no window; going
    through __init__ would require a repository and a figure and would test
    the shell rather than the arithmetic.
    """
    instance = cls.__new__(cls)
    return instance


# ======================================================================
# Calculus: derivatives
# ======================================================================

X = np.linspace(0.0, 2.0 * np.pi, 201)
Y = np.sin(X)


@pytest.mark.parametrize(
    "model, params, tolerance",
    [
        (DERIV_GRADIENT, {"order": 1}, 1e-3),
        (DERIV_SAVGOL, {"order": 1, "window": 11, "polyorder": 3}, 1e-3),
        (DERIV_SPLINE, {"order": 1, "smoothing": 0}, 1e-5),
    ],
)
def test_the_derivative_of_sine_is_cosine(model, params, tolerance) -> None:
    result = _bare(SeriesCalculusDialog)._differentiate("s", X, Y, model, params)
    assert np.max(np.abs(result.y - np.cos(X))) < tolerance


def test_the_second_derivative_of_sine_is_minus_sine() -> None:
    result = _bare(SeriesCalculusDialog)._differentiate(
        "s", X, Y, DERIV_SAVGOL, {"order": 2, "window": 21, "polyorder": 4}
    )
    assert np.max(np.abs(result.y + np.sin(X))) < 1e-2


def test_savgol_scales_by_the_sample_spacing() -> None:
    """The bug this guards: savgol without `delta` returns a derivative per
    sample index, which is correct only when the step happens to be 1."""
    dialog = _bare(SeriesCalculusDialog)
    params = {"order": 1, "window": 11, "polyorder": 3}

    # Same curve, sampled on a different x scale. d/dx must not change.
    fine = np.linspace(0.0, 1.0, 201)
    coarse = fine * 100.0
    on_fine = dialog._differentiate("s", fine, fine**2, DERIV_SAVGOL, params)
    on_coarse = dialog._differentiate("s", coarse, coarse**2, DERIV_SAVGOL, params)

    # d/dx of x^2 is 2x in both cases, read at the same fractional position.
    assert on_fine.y[100] == pytest.approx(2.0 * fine[100], rel=1e-3)
    assert on_coarse.y[100] == pytest.approx(2.0 * coarse[100], rel=1e-3)


def test_a_derivative_returns_one_value_per_input_point() -> None:
    """np.diff would return n-1 and no longer line up with the source axis."""
    result = _bare(SeriesCalculusDialog)._differentiate(
        "s", X, Y, DERIV_GRADIENT, {"order": 1}
    )
    assert result.y.size == X.size


def test_smoothing_beats_a_raw_difference_on_noisy_data() -> None:
    """The reason smoothing is inside the derivative rather than a step
    before it: differentiation amplifies noise."""
    dialog = _bare(SeriesCalculusDialog)
    rng = np.random.default_rng(0)
    noisy = Y + rng.normal(0.0, 0.01, Y.size)

    raw = dialog._differentiate("s", X, noisy, DERIV_GRADIENT, {"order": 1})
    smoothed = dialog._differentiate(
        "s", X, noisy, DERIV_SAVGOL, {"order": 1, "window": 21, "polyorder": 3}
    )

    truth = np.cos(X)
    raw_error = np.sqrt(np.mean((raw.y - truth) ** 2))
    smoothed_error = np.sqrt(np.mean((smoothed.y - truth) ** 2))
    assert smoothed_error < raw_error / 3.0


def test_a_savgol_window_larger_than_the_series_is_reduced() -> None:
    """SciPy would raise about array shapes rather than about the control."""
    window, polyorder = _bare(SeriesCalculusDialog)._savgol_window(
        9, {"window": 101, "polyorder": 3}
    )
    assert window <= 9
    assert window % 2 == 1
    assert polyorder < window


# ======================================================================
# Calculus: integrals
# ======================================================================

def test_the_definite_integral_of_sine_over_half_a_period_is_two() -> None:
    x = np.linspace(0.0, np.pi, 201)
    result = _bare(SeriesCalculusDialog)._integrate(
        "s", x, np.sin(x), INTEGRAL_DEFINITE,
        {"baseline": BASELINE_NONE, "simpson": True},
    )
    assert result.total == pytest.approx(2.0, abs=1e-6)


def test_simpson_is_more_accurate_than_trapezoid_on_a_smooth_curve() -> None:
    x = np.linspace(0.0, np.pi, 201)
    dialog = _bare(SeriesCalculusDialog)
    simpson = dialog._integrate(
        "s", x, np.sin(x), INTEGRAL_DEFINITE,
        {"baseline": BASELINE_NONE, "simpson": True},
    )
    trapezoid = dialog._integrate(
        "s", x, np.sin(x), INTEGRAL_DEFINITE,
        {"baseline": BASELINE_NONE, "simpson": False},
    )
    assert abs(simpson.total - 2.0) < abs(trapezoid.total - 2.0)


def test_the_cumulative_integral_of_cosine_is_sine() -> None:
    x = np.linspace(0.0, np.pi, 201)
    result = _bare(SeriesCalculusDialog)._integrate(
        "s", x, np.cos(x), INTEGRAL_CUMULATIVE, {"baseline": BASELINE_NONE}
    )
    assert result.y.size == x.size, "must line up with the source axis"
    assert result.y[0] == pytest.approx(0.0)
    assert np.max(np.abs(result.y - np.sin(x))) < 1e-4


@pytest.mark.parametrize("baseline", [BASELINE_MINIMUM, BASELINE_ENDPOINTS])
def test_baseline_subtraction_recovers_a_peak_area_from_an_offset(baseline) -> None:
    """Without it the offset contributes offset x width, which for this peak
    is 97% of the reported area."""
    x = np.linspace(-5.0, 5.0, 401)
    peak = np.exp(-(x**2))          # true area sqrt(pi)
    dialog = _bare(SeriesCalculusDialog)

    corrected = dialog._integrate(
        "s", x, peak + 5.0, INTEGRAL_DEFINITE,
        {"baseline": baseline, "simpson": True},
    )
    assert corrected.total == pytest.approx(np.sqrt(np.pi), abs=1e-3)


def test_without_baseline_subtraction_the_offset_dominates() -> None:
    x = np.linspace(-5.0, 5.0, 401)
    result = _bare(SeriesCalculusDialog)._integrate(
        "s", x, np.exp(-(x**2)) + 5.0, INTEGRAL_DEFINITE,
        {"baseline": BASELINE_NONE, "simpson": True},
    )
    assert result.total > 50.0, "the documented failure mode still holds"


# ======================================================================
# Peaks
# ======================================================================

PEAK_X = np.linspace(0.0, 100.0, 1001)


def _gaussian(centre: float, amplitude: float, width: float) -> np.ndarray:
    return amplitude * np.exp(-((PEAK_X - centre) ** 2) / (2.0 * width**2))


THREE_PEAKS = _gaussian(20.0, 1.0, 2.0) + _gaussian(50.0, 0.5, 4.0) + _gaussian(80.0, 2.0, 1.5)

DEFAULT_PEAK_PARAMS = {
    "filter_by": "prominence",
    "threshold": 0.05,
    "distance": 1,
    "min_width": 0.0,
    "limit": 50,
}


def test_three_known_peaks_are_found_at_their_known_positions() -> None:
    result = _bare(SeriesPeaksDialog)._find_one(
        "s", PEAK_X, THREE_PEAKS, PEAKS_MAXIMA, DEFAULT_PEAK_PARAMS
    )
    positions = sorted(peak.x for peak in result.peaks)
    assert len(positions) == 3
    for found, expected in zip(positions, (20.0, 50.0, 80.0)):
        assert found == pytest.approx(expected, abs=0.2)


def test_prominence_survives_a_sloping_baseline_and_height_does_not() -> None:
    """The reason prominence is the default. On a rising baseline, height
    selects whatever sits highest rather than what is a peak.

    Noise matters to this demonstration and is not incidental: a perfectly
    smooth slope has no local maxima at all, so height only misbehaves once
    there is something for it to catch on - which is every real measurement.
    """
    dialog = _bare(SeriesPeaksDialog)
    rng = np.random.default_rng(1)
    sloped = THREE_PEAKS + 0.02 * PEAK_X + 3.0 + rng.normal(0.0, 0.01, PEAK_X.size)

    by_prominence = dialog._find_one(
        "s", PEAK_X, sloped, PEAKS_MAXIMA,
        {**DEFAULT_PEAK_PARAMS, "filter_by": "prominence", "threshold": 0.15},
    )
    by_height = dialog._find_one(
        "s", PEAK_X, sloped, PEAKS_MAXIMA,
        {**DEFAULT_PEAK_PARAMS, "filter_by": "height", "threshold": 0.3},
    )

    assert len(by_prominence.peaks) <= 3
    assert len(by_height.peaks) > len(by_prominence.peaks) * 3


def test_a_peak_reports_bounds_the_integral_can_use() -> None:
    result = _bare(SeriesPeaksDialog)._find_one(
        "s", PEAK_X, THREE_PEAKS, PEAKS_MAXIMA, DEFAULT_PEAK_PARAMS
    )
    for peak in result.peaks:
        assert peak.left_x < peak.x < peak.right_x
        assert peak.width == pytest.approx(peak.right_x - peak.left_x, rel=1e-6)


def test_minima_are_found_by_inverting_the_signal() -> None:
    result = _bare(SeriesPeaksDialog)._find_one(
        "s", PEAK_X, -THREE_PEAKS, PEAKS_MINIMA, DEFAULT_PEAK_PARAMS
    )
    positions = sorted(peak.x for peak in result.peaks)
    assert len(positions) == 3
    assert all(peak.is_minimum for peak in result.peaks)


def test_a_flat_series_has_no_peaks() -> None:
    result = _bare(SeriesPeaksDialog)._find_one(
        "s", PEAK_X, np.ones_like(PEAK_X), PEAKS_MAXIMA, DEFAULT_PEAK_PARAMS
    )
    assert result.peaks == []


def test_the_limit_keeps_the_most_prominent_not_the_first() -> None:
    """Truncating in x order would discard the strongest peaks whenever they
    are late in the series - here the tallest is last."""
    result = _bare(SeriesPeaksDialog)._find_one(
        "s", PEAK_X, THREE_PEAKS, PEAKS_MAXIMA,
        {**DEFAULT_PEAK_PARAMS, "limit": 1},
    )
    assert len(result.peaks) == 1
    assert result.peaks[0].x == pytest.approx(80.0, abs=0.2)


# ======================================================================
# Control charts
# ======================================================================

CONTROL_PARAMS = {
    "subgroup": 5,
    "sigma_limit": 3.0,
    "nelson": False,
    "exclude_violations": False,
}


def test_sigma_comes_from_within_group_variation_not_the_overall_spread() -> None:
    """The property that makes it a control chart. A process that shifted has
    a large overall standard deviation *because* it shifted, so limits built
    from it are wide enough to contain the shift and never signal."""
    rng = np.random.default_rng(7)
    x = np.arange(60, dtype=float)
    shifted = np.concatenate([rng.normal(100.0, 1.0, 30), rng.normal(106.0, 1.0, 30)])

    result = _bare(SeriesControlChartDialog)._build_chart(
        "s", x, shifted, CHART_INDIVIDUALS, CONTROL_PARAMS
    )

    assert result.sigma < np.std(shifted, ddof=1) / 2.0
    assert len(result.violations) > 10, "the shift must be caught"


def test_a_stable_process_signals_rarely() -> None:
    rng = np.random.default_rng(3)
    x = np.arange(100, dtype=float)
    result = _bare(SeriesControlChartDialog)._build_chart(
        "s", x, rng.normal(0.0, 1.0, 100), CHART_INDIVIDUALS, CONTROL_PARAMS
    )
    assert len(result.violations) <= 2


def test_xbar_limits_match_the_textbook_a2_formula() -> None:
    """Agreement is to about 1e-3, which is the rounding in the published d2
    and A2 rather than a disagreement about the method."""
    rng = np.random.default_rng(5)
    values = rng.normal(50.0, 2.0, 100)
    x = np.arange(100, dtype=float)

    result = _bare(SeriesControlChartDialog)._build_chart(
        "s", x, values, CHART_XBAR_R, CONTROL_PARAMS
    )

    grouped = values.reshape(20, 5)
    centre = grouped.mean(axis=1).mean()
    mean_range = (grouped.max(axis=1) - grouped.min(axis=1)).mean()
    a2 = SPC_CONSTANTS[5][3]

    assert result.upper == pytest.approx(centre + a2 * mean_range, abs=1e-2)
    assert result.lower == pytest.approx(centre - a2 * mean_range, abs=1e-2)


def test_the_xbar_chart_plots_subgroup_means_not_the_raw_points() -> None:
    x = np.arange(100, dtype=float)
    result = _bare(SeriesControlChartDialog)._build_chart(
        "s", x, np.arange(100, dtype=float), CHART_XBAR_R, CONTROL_PARAMS
    )
    assert result.y.size == 20
    assert result.subgroup_size == 5


def test_xbar_limits_are_narrower_than_individuals_limits() -> None:
    """A subgroup mean varies by sigma/sqrt(n). Using sigma itself would give
    limits far too wide and a chart that never signals."""
    rng = np.random.default_rng(9)
    values = rng.normal(0.0, 1.0, 100)
    x = np.arange(100, dtype=float)
    dialog = _bare(SeriesControlChartDialog)

    individuals = dialog._build_chart("s", x, values, CHART_INDIVIDUALS, CONTROL_PARAMS)
    subgrouped = dialog._build_chart("s", x, values, CHART_XBAR_R, CONTROL_PARAMS)

    assert (subgrouped.upper - subgrouped.lower) < (individuals.upper - individuals.lower)


def test_the_moving_range_chart_never_has_a_negative_lower_limit() -> None:
    """A range is non-negative and its distribution is skewed, so it uses
    D3/D4 rather than symmetric limits."""
    rng = np.random.default_rng(2)
    x = np.arange(60, dtype=float)
    result = _bare(SeriesControlChartDialog)._build_chart(
        "s", x, rng.normal(10.0, 1.0, 60), CHART_MOVING_RANGE, CONTROL_PARAMS
    )
    assert result.lower >= 0.0
    assert result.y.size == 59, "one fewer than the source: no predecessor"


def test_the_run_rules_catch_a_shift_that_stays_inside_the_limits() -> None:
    """What a limits-only chart misses, and the reason the run rules exist."""
    rng = np.random.default_rng(11)
    x = np.arange(60, dtype=float)
    values = rng.normal(0.0, 1.0, 60)
    values[40:] += 1.2

    result = _bare(SeriesControlChartDialog)._build_chart(
        "s", x, values, CHART_INDIVIDUALS, {**CONTROL_PARAMS, "nelson": True}
    )

    assert ((values <= result.upper) & (values >= result.lower)).all()
    assert result.violations, "no point breaches the limits, so only a run rule can see it"
    assert 1 not in {rule for v in result.violations for rule in v.rules}


def test_a_point_reports_every_rule_it_broke() -> None:
    """The rule numbers are historical, not a severity ranking, so reporting
    only the lowest hides the more interesting half."""
    rng = np.random.default_rng(11)
    x = np.arange(60, dtype=float)
    tight = np.concatenate(
        [rng.normal(0.0, 1.0, 20), rng.normal(0.0, 0.05, 25), rng.normal(0.0, 1.0, 15)]
    )
    result = _bare(SeriesControlChartDialog)._build_chart(
        "s", x, tight, CHART_INDIVIDUALS, {**CONTROL_PARAMS, "nelson": True}
    )
    assert any(len(violation.rules) > 1 for violation in result.violations)


def test_excluding_flagged_points_tightens_the_limits() -> None:
    rng = np.random.default_rng(4)
    x = np.arange(60, dtype=float)
    values = rng.normal(100.0, 1.0, 60)
    values[30] = 130.0
    dialog = _bare(SeriesControlChartDialog)

    kept = dialog._build_chart("s", x, values, CHART_INDIVIDUALS, CONTROL_PARAMS)
    excluded = dialog._build_chart(
        "s", x, values, CHART_INDIVIDUALS,
        {**CONTROL_PARAMS, "exclude_violations": True},
    )

    assert excluded.sigma < kept.sigma
    assert excluded.metadata.get("excluded") == 1


def test_a_subgroup_size_that_yields_one_group_is_refused() -> None:
    with pytest.raises(ValueError, match="subgroup"):
        _bare(SeriesControlChartDialog)._build_chart(
            "s", np.arange(6, dtype=float), np.arange(6, dtype=float),
            CHART_XBAR_R, {**CONTROL_PARAMS, "subgroup": 5},
        )


def test_xbar_s_uses_c4_and_agrees_with_xbar_r_on_stable_data() -> None:
    """Two estimators of the same sigma; on well-behaved data they should not
    disagree much, which is the check that neither constant is misapplied."""
    rng = np.random.default_rng(13)
    values = rng.normal(0.0, 1.0, 250)
    x = np.arange(250, dtype=float)
    dialog = _bare(SeriesControlChartDialog)

    by_range = dialog._build_chart("s", x, values, CHART_XBAR_R, CONTROL_PARAMS)
    by_sigma = dialog._build_chart("s", x, values, CHART_XBAR_S, CONTROL_PARAMS)

    assert by_sigma.sigma == pytest.approx(by_range.sigma, rel=0.15)


def test_constants_outside_the_table_fall_back_and_say_so() -> None:
    constants, exact = SeriesControlChartDialog._constants(17)
    assert exact is False
    assert constants == SPC_CONSTANTS[15], "nearest smaller, not interpolated"


# ======================================================================
# Function plotting
# ======================================================================

def test_a_linear_range_spans_the_requested_endpoints() -> None:
    values = SeriesFunctionDialog._build_range(
        {"start": 0.0, "stop": 10.0, "points": 5, "spacing": SPACING_LINEAR}
    )
    assert values.tolist() == [0.0, 2.5, 5.0, 7.5, 10.0]


def test_a_log_range_is_evenly_spaced_in_decades() -> None:
    values = SeriesFunctionDialog._build_range(
        {"start": 1.0, "stop": 1000.0, "points": 4, "spacing": SPACING_LOG}
    )
    assert values.tolist() == pytest.approx([1.0, 10.0, 100.0, 1000.0])


def test_a_log_range_through_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="above zero"):
        SeriesFunctionDialog._build_range(
            {"start": -1.0, "stop": 10.0, "points": 10, "spacing": SPACING_LOG}
        )


def test_an_empty_range_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        SeriesFunctionDialog._build_range(
            {"start": 5.0, "stop": 5.0, "points": 10, "spacing": SPACING_LINEAR}
        )


def test_the_scanner_finds_both_builtin_and_user_functions() -> None:
    from app.scanners.functions_scanner import FunctionScanner

    catalog = FunctionScanner().catalog()
    assert catalog, "no functions discovered at all"
    assert "User functions" in catalog, "user_functions.py must be picked up"


def test_a_discovered_function_evaluates_over_a_range() -> None:
    """End to end: the scanner's callable applied to a built range."""
    from app.scanners.functions_scanner import FunctionScanner

    scanner = FunctionScanner()
    payload = next(
        payload
        for functions in scanner.catalog().values()
        for payload in functions
        if payload.get("discovery_entry", {}).get("name") == "linear"
    )
    model = scanner.make_model(dict(payload))
    x = SeriesFunctionDialog._build_range(
        {"start": 1.0, "stop": 10.0, "points": 7, "spacing": SPACING_LINEAR}
    )
    y = np.asarray(model(x, np.asarray([0.0, 1.0])), dtype=float)

    assert y == pytest.approx(x), "intercept 0, slope 1 is the identity"
