"""Tests for fitting continuous distributions and ranking them.

The ranking is the product here, so the tests are about the ranking: that a
sample drawn from a known family puts that family at the top, that a candidate
which cannot describe the sample drops out instead of raising, and that the
histogram draws the same fit the table reports.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.utils.distribution_fit import (
    CURATED_DISTRIBUTIONS,
    available_distributions,
    best_fit,
    curve_points,
    fits_for_spec,
    fit_distributions,
    fit_one,
)

RNG = np.random.default_rng(11)


# ----------------------------------------------------------------------
# Fitting one
# ----------------------------------------------------------------------
def test_a_known_sample_recovers_its_own_parameters() -> None:
    fit = fit_one(RNG.normal(10.0, 2.0, 4000), "norm")

    assert fit is not None
    location, scale = fit.params
    assert location == pytest.approx(10.0, abs=0.2)
    assert scale == pytest.approx(2.0, abs=0.2)


def test_a_degenerate_sample_is_dropped_not_raised() -> None:
    """A sample with no spread has no scale, so the fit is undefined."""
    assert fit_one(np.full(200, 4.0), "norm") is None


def test_a_negative_sample_does_not_disqualify_a_positive_family() -> None:
    """The obvious assumption, pinned because it is wrong.

    ``lognorm`` is documented on positive support, so a sweep over negative
    data looks like it should drop it.  It does not: every scipy family has a
    ``loc`` that slides its support, and ``fit`` uses it.  Anything filtering
    candidates by "is this data positive" would be filtering on a rule scipy
    does not follow.
    """
    assert fit_one(RNG.normal(-50.0, 1.0, 500), "lognorm") is not None


def test_an_unknown_name_is_reported_not_raised() -> None:
    assert fit_one(RNG.normal(0.0, 1.0, 200), "not_a_distribution") is None


def test_too_few_points_is_not_a_fit() -> None:
    assert fit_one(np.array([1.0, 2.0]), "norm") is None


def test_non_finite_values_are_ignored_not_fatal() -> None:
    sample = np.concatenate([RNG.normal(5.0, 1.0, 500), [np.nan, np.inf, -np.inf]])
    fit = fit_one(sample, "norm")

    assert fit is not None
    assert fit.n == 500


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("family", "sample"),
    [
        ("norm", RNG.normal(10.0, 2.0, 3000)),
        ("lognorm", RNG.lognormal(1.0, 0.5, 3000)),
        ("expon", RNG.exponential(3.0, 3000)),
        ("uniform", RNG.uniform(0.0, 1.0, 3000)),
    ],
)
def test_a_sample_ranks_its_own_family_at_the_top(family: str, sample) -> None:
    """The one claim the ranking has to earn, and only AIC earns it."""
    fits = fit_distributions(sample)

    assert fits, "nothing fitted at all"
    assert fits[0].name == family


def test_ranking_by_ks_alone_favours_the_most_flexible_candidate() -> None:
    """Why AIC is the default, pinned so the reason cannot be forgotten.

    D says how close the fitted curve got and nothing about how many free
    parameters it took to get there, so a four-parameter shape wins against the
    family the data actually came from.  This is not a quirk of one sample:
    order an exponential sample by D and ``expon`` is not even first.
    """
    sample = RNG.exponential(3.0, 3000)

    by_ks = fit_distributions(sample, rank_by="ks")
    by_aic = fit_distributions(sample, rank_by="aic")

    assert by_aic[0].name == "expon"
    assert by_ks[0].name != "expon"
    assert len(by_ks[0].params) > len(by_aic[0].params)


@pytest.mark.parametrize("criterion", ["aic", "bic", "ks"])
def test_each_criterion_actually_orders_by_itself(criterion: str) -> None:
    fits = fit_distributions(RNG.gamma(2.0, 2.0, 2000), rank_by=criterion)
    key = {"aic": "aic", "bic": "bic", "ks": "ks_statistic"}[criterion]
    values = [getattr(fit, key) for fit in fits]

    assert values == sorted(values)


def test_an_unknown_criterion_falls_back_rather_than_raising() -> None:
    fits = fit_distributions(RNG.normal(0.0, 1.0, 500), rank_by="nonsense")

    assert [fit.aic for fit in fits] == sorted(fit.aic for fit in fits)


def test_information_criteria_penalise_free_parameters() -> None:
    """Why AIC and BIC are reported beside D.

    D alone rewards whichever candidate has the most parameters to bend, so a
    four-parameter shape can edge out the family the data actually came from.
    BIC penalises harder than AIC, which is the whole reason both are shown.
    """
    fit = fit_one(RNG.normal(0.0, 1.0, 1000), "norm")

    assert fit is not None
    assert fit.bic > fit.aic  # ln(1000) > 2, so the BIC penalty is larger
    assert fit.aic == pytest.approx(2 * len(fit.params) - 2 * fit.log_likelihood)


def test_an_all_nan_sample_ranks_nothing_rather_than_raising() -> None:
    assert fit_distributions(np.full(100, np.nan)) == []
    assert best_fit(np.full(100, np.nan)) is None


# ----------------------------------------------------------------------
# The candidate list
# ----------------------------------------------------------------------
def test_the_curated_list_is_the_default() -> None:
    assert available_distributions() == CURATED_DISTRIBUTIONS


def test_the_exhaustive_list_is_larger_and_excludes_the_slow_ones() -> None:
    """levy_stable alone can take minutes, and this dialog refits on every edit."""
    exhaustive = available_distributions(exhaustive=True)

    assert len(exhaustive) > len(CURATED_DISTRIBUTIONS)
    assert "levy_stable" not in exhaustive
    assert "studentized_range" not in exhaustive
    assert "norm" in exhaustive


# ----------------------------------------------------------------------
# The curve the histogram draws
# ----------------------------------------------------------------------
def test_the_density_curve_integrates_to_about_one() -> None:
    """It is a probability density, which is why the histogram must be one too."""
    fit = best_fit(RNG.normal(10.0, 2.0, 3000))
    assert fit is not None

    x, pdf = curve_points(fit, 0.0, 20.0, points=2000)

    assert np.isfinite(pdf).all()
    assert float(np.trapezoid(pdf, x)) == pytest.approx(1.0, abs=0.02)


def test_the_curve_is_drawn_at_its_own_resolution() -> None:
    """Sampling once per bin would just redraw the bars as a staircase."""
    fit = best_fit(RNG.normal(0.0, 1.0, 500))
    assert fit is not None

    x, pdf = curve_points(fit, -3.0, 3.0, points=321)

    assert x.size == 321 and pdf.size == 321


# ----------------------------------------------------------------------
# Resolving what a chart asked for
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("", 0),
        ("best", 1),
        ("top3", 3),
        ("top5", 5),
        ("norm", 1),
        ("norm, lognorm", 2),
        ("  NORM ,  LOGNORM  ", 2),
    ],
)
def test_a_chart_gets_the_curves_it_asked_for(spec: str, expected: int) -> None:
    """The histogram and the ECDF read the same spec, so it resolves once."""
    fits = fits_for_spec(RNG.gamma(2.0, 2.0, 1200), spec)

    assert len(fits) == expected


def test_named_distributions_keep_the_order_they_were_written_in() -> None:
    """A reader comparing two curves expects them in the order they asked."""
    fits = fits_for_spec(RNG.normal(0.0, 1.0, 800), "lognorm, norm")

    assert [fit.name for fit in fits] == ["lognorm", "norm"]


def test_an_unknown_name_yields_no_curve_rather_than_raising() -> None:
    assert fits_for_spec(RNG.normal(0.0, 1.0, 500), "not_a_distribution") == []


def test_the_cumulative_curve_runs_from_zero_to_one() -> None:
    """What the ECDF overlay draws, on the ECDF's own vertical scale."""
    fit = best_fit(RNG.normal(10.0, 2.0, 2000))
    assert fit is not None

    _x, cdf = curve_points(fit, 0.0, 20.0, points=1000, cumulative=True)

    assert cdf[0] == pytest.approx(0.0, abs=1e-3)
    assert cdf[-1] == pytest.approx(1.0, abs=1e-3)
    assert np.all(np.diff(cdf) >= -1e-12), "a CDF cannot decrease"
