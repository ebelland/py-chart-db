"""The algorithms a fit can be run with, behind one call.

``least_squares`` with its default settings is the right answer for most fits
and the wrong one for two situations that come up constantly:

* **a bad starting point.** Every local method walks downhill from where it
  starts, so on a multi-modal surface - two exponentials, a peak on a sloping
  baseline, a five-parameter logistic - it converges to whichever valley it
  was dropped into and reports success. The global methods here sample the
  whole box first, so the answer stops depending on the guess;
* **outliers.** A squared residual weights a point ten times off the curve a
  hundred times more than a point one off, so a handful of bad points drag the
  whole fit. The robust losses re-weight those residuals instead of trusting
  them.

Every method is reached through :func:`run_optimizer`, which takes a residual
function over the *free* parameters and returns the same outcome whatever ran
underneath. Two consequences worth knowing:

* the global methods are followed by a local fit from where they finished.
  Random search finds the right valley and gradient descent finds its floor,
  and neither does the other's job well. It is also what produces a Jacobian,
  which is where the parameter uncertainties come from - a global method alone
  reports parameters with no error bars;
* a sampler needs a finite box. Where a parameter has no bound, the box is a
  window around its current value (``monte_carlo._sampling_bounds``), which is
  why setting real bounds is worth the trouble: a bounded parameter is
  searched where the answer is, an unbounded one only near where it started.

Pure numpy/scipy - no Qt - so every method here is testable without a window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import (
    basinhopping,
    differential_evolution,
    dual_annealing,
    least_squares,
    minimize,
    shgo,
)

from app.functions.monte_carlo import _sampling_bounds, monte_carlo_search

Residual = Callable[[np.ndarray], np.ndarray]

# ----------------------------------------------------------------------
# What can be chosen
# ----------------------------------------------------------------------

#: Robust losses ``least_squares`` accepts, as (key, label).  ``linear`` is
#: ordinary least squares; the others cap how much a large residual can
#: contribute, which is what makes a fit survive a few bad points.
LOSSES: tuple[tuple[str, str], ...] = (
    ("linear", "Least squares"),
    ("soft_l1", "Soft L1 (robust)"),
    ("huber", "Huber (robust)"),
    ("cauchy", "Cauchy (very robust)"),
    ("arctan", "Arctan (very robust)"),
)


@dataclass(frozen=True, slots=True)
class Optimizer:
    """One algorithm, and what it needs to be usable."""

    key: str
    label: str
    description: str
    #: "local" walks downhill from the starting point; "global" samples the
    #: whole box before it does.
    kind: str = "local"
    #: Whether the robust-loss choice applies.  Only the least_squares family
    #: takes a loss; everything else minimises a plain sum of squares.
    supports_loss: bool = False
    #: Whether bounds are honoured at all.  Levenberg-Marquardt is the
    #: exception: SciPy refuses bounds with method="lm" outright.
    supports_bounds: bool = True


TRUST_REGION = "trf"
DOGBOX = "dogbox"
LEVENBERG = "lm"
NELDER_MEAD = "nelder-mead"
POWELL = "powell"
DIFFERENTIAL_EVOLUTION = "differential-evolution"
DUAL_ANNEALING = "dual-annealing"
BASIN_HOPPING = "basin-hopping"
SHGO = "shgo"
MONTE_CARLO = "monte-carlo"

OPTIMIZERS: tuple[Optimizer, ...] = (
    Optimizer(
        TRUST_REGION,
        "Trust region (default)",
        "Bounded least squares. The right answer for most fits.",
        supports_loss=True,
    ),
    Optimizer(
        DOGBOX,
        "Dogleg box",
        "Bounded least squares, often better when a parameter sits on a bound.",
        supports_loss=True,
    ),
    Optimizer(
        LEVENBERG,
        "Levenberg-Marquardt",
        "The classic unbounded least squares. Bounds are ignored.",
        supports_bounds=False,
    ),
    Optimizer(
        NELDER_MEAD,
        "Nelder-Mead simplex",
        "Derivative-free. Slow, but unbothered by a rough or noisy surface.",
    ),
    Optimizer(
        POWELL,
        "Powell",
        "Derivative-free line searches; often reaches a valley a simplex crawls to.",
    ),
    Optimizer(
        DIFFERENTIAL_EVOLUTION,
        "Differential evolution",
        "Searches the whole box with a population, then fits locally. "
        "Use when the starting point is a guess.",
        kind="global",
    ),
    Optimizer(
        DUAL_ANNEALING,
        "Dual annealing",
        "Simulated annealing over the box, then a local fit. Escapes deep local minima.",
        kind="global",
    ),
    Optimizer(
        BASIN_HOPPING,
        "Basin hopping",
        "Repeated local fits from perturbed starts. Good when minima are many but shallow.",
        kind="global",
    ),
    Optimizer(
        SHGO,
        "SHGO",
        "Systematic sampling of the box; finds every basin rather than sampling at random.",
        kind="global",
    ),
    Optimizer(
        MONTE_CARLO,
        "Monte Carlo",
        "Uniform random sampling, then a local fit. Assumes nothing about the model.",
        kind="global",
    ),
)

BY_KEY: dict[str, Optimizer] = {optimizer.key: optimizer for optimizer in OPTIMIZERS}

DEFAULT_OPTIMIZER: str = TRUST_REGION


@dataclass(slots=True)
class FitOutcome:
    """What an optimiser produced, in the shape the caller needs."""

    params: np.ndarray
    #: Jacobian of the residuals at the optimum, over the free parameters, or
    #: None when the method could not produce one.  Uncertainties come from
    #: this, so None means a fit with no error bars.
    jac: np.ndarray | None
    success: bool
    message: str
    cost: float
    optimizer: str = DEFAULT_OPTIMIZER


# ----------------------------------------------------------------------
# Running one
# ----------------------------------------------------------------------
def run_optimizer(
    key: str,
    residual: Residual,
    p0: Any,
    lower: Any,
    upper: Any,
    *,
    max_nfev: int = 800,
    loss: str = "linear",
    seed: int | None = None,
    iterations: int = 20_000,
) -> FitOutcome:
    """Minimise ``sum(residual(p)**2)`` from *p0*, with the named method."""
    optimizer = BY_KEY.get(str(key), BY_KEY[DEFAULT_OPTIMIZER])
    start = np.asarray(p0, dtype=float).ravel()
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()

    if start.size == 0:
        return FitOutcome(start, None, True, "Nothing to fit.", float("nan"), optimizer.key)

    # least_squares refuses an infeasible start outright ("x0 is infeasible")
    # rather than clipping it, so a starting value a hair outside a bound
    # would abort the fit with a message about x0 that says nothing about the
    # bound the user typed.
    start = np.clip(start, np.where(np.isfinite(low), low, -np.inf),
                    np.where(np.isfinite(high), high, np.inf))

    def cost(params: np.ndarray) -> float:
        try:
            values = np.asarray(residual(np.asarray(params, dtype=float)), dtype=float)
        except Exception:
            # A sampler will land on parameters the model cannot evaluate.
            # That is "bad, move on", not a reason to stop the search.
            return float(np.inf)
        if not np.all(np.isfinite(values)):
            return float(np.inf)
        return float(np.sum(values * values))

    if optimizer.kind == "global":
        found, message = _search_globally(
            optimizer, cost, start, low, high,
            max_nfev=max_nfev, seed=seed, iterations=iterations,
        )
        # Polished locally: the search found the valley, this finds its floor
        # - and produces the Jacobian the uncertainties are read from.
        polished = _least_squares(
            residual, found, low, high, method=TRUST_REGION, loss="linear", max_nfev=max_nfev
        )
        polished.message = f"{message} {polished.message}".strip()
        polished.optimizer = optimizer.key
        return polished

    if optimizer.key in (TRUST_REGION, DOGBOX, LEVENBERG):
        outcome = _least_squares(
            residual,
            start,
            low,
            high,
            method=optimizer.key,
            loss=loss if optimizer.supports_loss else "linear",
            max_nfev=max_nfev,
            bounded=optimizer.supports_bounds,
        )
        outcome.optimizer = optimizer.key
        return outcome

    return _minimize_scalar(optimizer, residual, cost, start, low, high, max_nfev=max_nfev)


def _least_squares(
    residual: Residual,
    start: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    method: str,
    loss: str,
    max_nfev: int,
    bounded: bool = True,
) -> FitOutcome:
    """Run ``least_squares`` and report it, whatever it does."""
    kwargs: dict[str, Any] = {"max_nfev": max(1, int(max_nfev))}
    if bounded:
        kwargs["bounds"] = (low, high)
        kwargs["method"] = method
    else:
        # SciPy refuses bounds with "lm" rather than ignoring them, so the
        # bounds are dropped here and the choice is documented on the option.
        kwargs["method"] = LEVENBERG

    if loss and loss != "linear":
        # f_scale stays at its default: it is the residual magnitude beyond
        # which a point counts as an outlier, and 1.0 is right whenever the
        # residuals are of order one - which is what the weighting option is
        # for. Exposing both would be two controls that only make sense
        # together.
        kwargs["loss"] = loss

    try:
        result = least_squares(residual, start, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return FitOutcome(start, None, False, f"{type(exc).__name__}: {exc}", float("nan"), method)

    return FitOutcome(
        params=np.asarray(result.x, dtype=float),
        jac=np.asarray(result.jac, dtype=float),
        success=bool(result.success),
        message=str(result.message),
        cost=float(result.cost),
        optimizer=method,
    )


def _minimize_scalar(
    optimizer: Optimizer,
    residual: Residual,
    cost: Callable[[np.ndarray], float],
    start: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    max_nfev: int,
) -> FitOutcome:
    """Run a derivative-free scalar minimiser, then read off a Jacobian."""
    method = "Nelder-Mead" if optimizer.key == NELDER_MEAD else "Powell"
    bounds = _finite_bounds_or_none(low, high)

    try:
        result = minimize(
            cost,
            start,
            method=method,
            bounds=bounds,
            options={"maxfev": max(1, int(max_nfev))},
        )
        found = np.asarray(result.x, dtype=float)
        message = str(getattr(result, "message", "")) or method
        success = bool(getattr(result, "success", True))
    except Exception as exc:  # noqa: BLE001
        return FitOutcome(start, None, False, f"{type(exc).__name__}: {exc}", float("nan"), optimizer.key)

    # One least_squares step from the optimum, purely for the Jacobian: these
    # methods use no derivatives, so without it the fit has no uncertainties.
    polished = _least_squares(
        residual, found, low, high, method=TRUST_REGION, loss="linear", max_nfev=max(2, max_nfev // 4)
    )
    polished.success = success and polished.success
    polished.message = f"{message}. {polished.message}".strip()
    polished.optimizer = optimizer.key
    return polished


def _search_globally(
    optimizer: Optimizer,
    cost: Callable[[np.ndarray], float],
    start: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    max_nfev: int,
    seed: int | None,
    iterations: int,
) -> tuple[np.ndarray, str]:
    """Return the best parameters a global method found, and what it said."""
    # Every sampler needs somewhere to sample. An unbounded parameter gets a
    # window around its current value - see the module docstring.
    box_low, box_high = _sampling_bounds(start, low, high)
    box = list(zip(box_low.tolist(), box_high.tolist()))

    try:
        if optimizer.key == DIFFERENTIAL_EVOLUTION:
            result = differential_evolution(
                cost, box, seed=seed, maxiter=max(5, int(max_nfev) // 10), polish=False,
            )
            return np.asarray(result.x, dtype=float), "Differential evolution:"

        if optimizer.key == DUAL_ANNEALING:
            result = dual_annealing(
                cost, box, seed=seed, maxiter=max(5, int(max_nfev) // 10),
            )
            return np.asarray(result.x, dtype=float), "Dual annealing:"

        if optimizer.key == BASIN_HOPPING:
            result = basinhopping(
                cost,
                start,
                niter=max(5, int(max_nfev) // 100),
                seed=seed,
                minimizer_kwargs={"method": "L-BFGS-B", "bounds": box},
            )
            return np.asarray(result.x, dtype=float), "Basin hopping:"

        if optimizer.key == SHGO:
            result = shgo(cost, box, options={"maxfev": max(10, int(max_nfev))})
            return np.asarray(result.x, dtype=float), "SHGO:"

        found, score, done = monte_carlo_search(
            cost, start, low, high, iterations=iterations, seed=seed,
        )
        return found, f"Monte Carlo: {done} samples, best cost {score:.6g}."
    except Exception as exc:  # noqa: BLE001
        # A global method that fails leaves the starting point untouched; the
        # local polish below still runs, so the user gets an ordinary fit and
        # a message saying the search did not happen.
        return start, f"{optimizer.label} failed ({exc});"


def _finite_bounds_or_none(
    low: np.ndarray,
    high: np.ndarray,
) -> list[tuple[float, float]] | None:
    """Return bounds for ``minimize``, or None when there are none to give.

    All or nothing: ``minimize`` takes one sequence for every parameter, and a
    half-filled one with infinities in it is refused by some methods and
    silently ignored by others.
    """
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        return None
    return list(zip(low.tolist(), high.tolist()))
