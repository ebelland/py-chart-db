"""Fit continuous distributions to a sample and rank them by goodness of fit.

One implementation, because two callers need the same answer: the statistics
dialog reports a ranked table, and the histogram renderer draws the winner's
density over the bars.  If each fitted independently they could disagree about
which distribution won, and the picture would contradict the table beside it.

Hand-rolled over ``scipy.stats`` rather than through ``fitter``: the loop is
short, the candidate list belongs to this application, and a failed fit has to
be reported rather than printed.  scipy is already a dependency.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from app.logs.logger import applogger

#: What measurement data actually looks like.  Fifteen, in a deliberate order:
#: the first few are what most samples turn out to be, so a reader scanning the
#: table meets the likely answers first even before it is sorted.
CURATED_DISTRIBUTIONS: tuple[str, ...] = (
    "norm",
    "lognorm",
    "expon",
    "gamma",
    "beta",
    "weibull_min",
    "weibull_max",
    "gumbel_r",
    "logistic",
    "laplace",
    "chi2",
    "t",
    "pareto",
    "rayleigh",
    "uniform",
)

#: Excluded from the exhaustive sweep.  Every one of these takes seconds to
#: minutes per fit - ``levy_stable`` alone can outlast the user's patience by a
#: wide margin - and this dialog refits whenever any parameter changes.  Named
#: rather than time-limited because a timeout would make the result depend on
#: how busy the machine is.
_TOO_SLOW: frozenset[str] = frozenset(
    {
        "levy_stable",
        "studentized_range",
        "ncf",
        "nct",
        "ncx2",
        "vonmises",
        "vonmises_line",
        "kstwo",
        "genhyperbolic",
        "norminvgauss",
        "skewcauchy",
        "irwinhall",
    }
)


@dataclass(frozen=True, slots=True)
class DistributionFit:
    """One candidate's fitted parameters and how well they describe the sample."""

    name: str
    params: tuple[float, ...]
    ks_statistic: float
    pvalue: float
    log_likelihood: float
    aic: float
    bic: float
    n: int

    def curve(self, x: np.ndarray, *, cumulative: bool = False) -> np.ndarray:
        """Return the fitted density, or its CDF, evaluated at *x*.

        One method for both because the two callers differ only in which they
        want: the histogram overlays a density on its bars, the ECDF overlays
        the matching cumulative curve on its steps.  Splitting them would mean
        two nearly identical bodies and two places to get the error handling
        wrong.

        A curve that cannot be evaluated comes back as zeros rather than
        raising: one unusable candidate must not take the chart with it.
        """
        try:
            distribution = getattr(stats, self.name)
            function = distribution.cdf if cumulative else distribution.pdf
            values = np.asarray(function(x, *self.params), dtype=float)
        except Exception:
            applogger.exception("Could not evaluate the %s curve.", self.name)
            return np.zeros_like(np.asarray(x, dtype=float))
        return np.where(np.isfinite(values), values, 0.0)


def available_distributions(*, exhaustive: bool = False) -> tuple[str, ...]:
    """Return the candidate names, curated or as much of scipy as is practical."""
    if not exhaustive:
        return CURATED_DISTRIBUTIONS

    names = sorted(
        name
        for name in dir(stats)
        if isinstance(getattr(stats, name, None), stats.rv_continuous)
        and name not in _TOO_SLOW
    )
    return tuple(names)


def fit_one(values: np.ndarray, name: str) -> DistributionFit | None:
    """Fit one named distribution, or return None with a line in the log.

    Returns None rather than raising: a candidate that cannot describe this
    sample is an ordinary outcome of a sweep, not an error in the sweep.

    Which candidates those are is less obvious than it looks.  Every scipy
    family carries a ``loc`` that shifts its support, so the ones documented as
    needing positive data - ``lognorm``, ``pareto``, ``expon`` - fit a negative
    sample perfectly happily by sliding the whole distribution left.  What
    actually fails is degeneracy: a sample with no spread gives a zero scale
    and an infinite log-likelihood, which the finiteness check below rejects.
    """
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 3:
        return None

    distribution = getattr(stats, name, None)
    if distribution is None:
        applogger.warning(
            "No distribution named %r in scipy.stats; skipping it.",
            name,
            show_dialog=False,
            raise_error=False,
        )
        return None

    try:
        # scipy warns freely while fitting - overflow in exp, invalid value in
        # subtract - and a warning per candidate would bury the log for what is
        # a normal part of trying fifteen shapes against one sample.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = tuple(float(value) for value in distribution.fit(sample))
            # The distribution's own cdf, not its name. Passing the name made
            # SciPy look the function up itself, and in 1.18 that lookup
            # started returning the bare special function (ndtr for "norm"),
            # which takes one argument - so loc and scale arrived as extra
            # positional arguments and every fit raised TypeError. Caught
            # below, so the sweep reported "could not be fitted" for all
            # fifteen candidates and the feature simply stopped working.
            result = stats.kstest(sample, distribution.cdf, args=params)
            log_likelihood = float(
                np.sum(distribution.logpdf(sample, *params))
            )
    except Exception:
        applogger.info("The %s distribution could not be fitted to this sample.", name)
        return None

    if not np.isfinite(log_likelihood):
        # An observation with zero density under the fitted parameters: the
        # fit technically converged but assigns the data impossible values.
        return None

    k = len(params)
    n = int(sample.size)
    return DistributionFit(
        name=name,
        params=params,
        ks_statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        log_likelihood=log_likelihood,
        aic=float(2 * k - 2 * log_likelihood),
        bic=float(k * np.log(n) - 2 * log_likelihood),
        n=n,
    )


#: How the ranking is ordered.  AIC is the default deliberately - see
#: ``fit_distributions``.
RANK_CRITERIA: tuple[str, ...] = ("aic", "bic", "ks")
DEFAULT_RANK: str = "aic"

_RANK_KEYS = {
    "aic": lambda fit: fit.aic,
    "bic": lambda fit: fit.bic,
    "ks": lambda fit: fit.ks_statistic,
}


def fit_distributions(
    values: np.ndarray,
    names: tuple[str, ...] | None = None,
    *,
    exhaustive: bool = False,
    rank_by: str = DEFAULT_RANK,
) -> list[DistributionFit]:
    """Fit every candidate and return them best first.

    Ranked by AIC by default, not by the KS statistic, and the difference is
    not academic.  Measured on samples drawn from known families, ordering by D
    puts four-parameter ``beta`` first for *every* one of them - normal,
    exponential, gamma alike - because D measures how close the fitted curve
    got and says nothing about how many free parameters it took to get there.
    A distribution with four of them can contort to fit anything.  Ordering the
    same fits by AIC recovers the true family in each case.

    D and its p-value are still reported, because they answer a different and
    useful question: how far off is this curve, in the units of the data's own
    CDF.  But they are a diagnostic, not an ordering.

    The p-value carries its own caveat: the parameters were estimated from the
    sample being tested, which makes ``kstest`` optimistic.  It asks whether
    the data could come from *this fitted curve*, not from that family.
    """
    candidates = names if names is not None else available_distributions(
        exhaustive=exhaustive
    )

    fits = [fit for fit in (fit_one(values, name) for name in candidates) if fit]
    skipped = len(candidates) - len(fits)
    if skipped:
        applogger.info(
            "%d of %d distributions could not be fitted to this sample.",
            skipped,
            len(candidates),
        )

    key = _RANK_KEYS.get(str(rank_by).strip().lower())
    if key is None:
        applogger.warning(
            "Unknown ranking criterion %r; ranking by %s instead.",
            rank_by,
            DEFAULT_RANK,
            show_dialog=False,
            raise_error=False,
        )
        key = _RANK_KEYS[DEFAULT_RANK]

    fits.sort(key=key)
    return fits


def best_fit(
    values: np.ndarray, *, exhaustive: bool = False, rank_by: str = DEFAULT_RANK
) -> DistributionFit | None:
    """Return the best-ranked candidate, or None when none of them fit."""
    fits = fit_distributions(values, exhaustive=exhaustive, rank_by=rank_by)
    return fits[0] if fits else None


#: What a ``distribution_fit`` option may say.  "best" and "top N" need the
#: whole sweep and then take the head of it; anything else is read as one or
#: more distribution names, comma separated.
_TOP_N_PREFIX: str = "top"


def fits_for_spec(
    values: np.ndarray,
    spec: str,
    *,
    exhaustive: bool = False,
    rank_by: str = DEFAULT_RANK,
) -> list[DistributionFit]:
    """Resolve a chart's ``distribution_fit`` option to the fits it names.

    One resolver, shared by the histogram and the ECDF, so a chart showing the
    same spec always shows the same curves.

    Accepted: "" for none, "best" for the top-ranked candidate, "top3"/"top5"
    for that many, or a comma-separated list of scipy names.  Names are fitted
    individually and left in the order written, because a reader comparing
    "norm, lognorm" expects them in that order and not re-sorted underneath
    them.
    """
    wanted = str(spec or "").strip().lower()
    if not wanted:
        return []

    if wanted == "best" or wanted.startswith(_TOP_N_PREFIX):
        count = 1
        if wanted.startswith(_TOP_N_PREFIX):
            digits = wanted[len(_TOP_N_PREFIX) :].strip()
            count = int(digits) if digits.isdigit() else 1
        ranked = fit_distributions(values, exhaustive=exhaustive, rank_by=rank_by)
        return ranked[: max(1, count)]

    names = tuple(part.strip() for part in wanted.split(",") if part.strip())
    return [fit for fit in (fit_one(values, name) for name in names) if fit]


def curve_points(
    fit: DistributionFit,
    low: float,
    high: float,
    points: int = 512,
    *,
    cumulative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) across [low, high] for drawing over a chart.

    Evaluated on its own grid rather than at the histogram's bin centres: a
    density is a curve, and sampling it once per bin turns a smooth shape into
    the same staircase the bars already show.
    """
    x = np.linspace(float(low), float(high), max(2, int(points)))
    return x, fit.curve(x, cumulative=cumulative)
