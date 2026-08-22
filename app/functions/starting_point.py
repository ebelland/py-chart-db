"""Where a fit starts from, and who decided.

Optimisers do not find the best parameters; they find the bottom of the valley
they were dropped into.  For a straight line every valley is the same one and
the starting point is irrelevant.  For a sum of two exponentials, a peak on a
baseline or a five-parameter logistic, the wrong valley is a converged fit
with a plausible RMSE and completely wrong parameters - which is worse than a
failure, because it is quiet.

So the starting point is chosen in this order:

1. the function's own ``initial_guess(x, y)``, which reads the parameters off
   the data - a Gaussian's centre is where the mass is, a line's slope is a
   polyfit.  Nothing here can beat a function that knows its own parameters;
2. a random search of the parameter space, which assumes nothing about the
   model and therefore works on the forty-five shipped functions that have no
   estimator, and on user functions this library has never seen;
3. the declared ``p0``, when the search cannot improve on it.

The result says which of the three it was, because "the fit is wrong" and "the
guess was invented" are different problems and the report should not make the
user guess which one they have.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from app.functions.monte_carlo import monte_carlo_p0, sum_of_squares

#: Samples drawn when no estimator is available.  Enough to find the right
#: valley for the shapes this library holds, few enough to stay under a
#: second - it runs while the user watches a dialog.
DEFAULT_ITERATIONS: int = 20_000

#: How the values were arrived at.  Stored in English; the report translates.
FROM_FUNCTION: str = "initial_guess"
FROM_SEARCH: str = "monte carlo"
FROM_DECLARED: str = "declared"


@dataclass(slots=True)
class StartingPoint:
    """Parameters to start a fit from, and where they came from."""

    values: np.ndarray
    source: str
    #: Residual sum of squares at ``values``; NaN when it was not evaluated.
    cost: float = float("nan")

    @property
    def estimated(self) -> bool:
        """True when the data decided, rather than a declaration."""
        return self.source in (FROM_FUNCTION, FROM_SEARCH)


def ask_the_function(
    function_class: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    expected: int,
) -> np.ndarray | None:
    """Return the class's own estimate, or None when it has none.

    A guess of the wrong length is refused rather than padded: it would put
    values in the wrong parameters, and a fit started from the wrong
    parameters in the right *shape* is the failure that looks like success.
    """
    estimator = getattr(function_class, "initial_guess", None)
    if not callable(estimator):
        return None

    try:
        guess = estimator(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    except Exception:
        # An estimator that raises is a bug in one function, not a reason to
        # refuse the fit: the search below still produces a starting point.
        return None

    if guess is None:
        return None

    values = np.asarray(list(guess), dtype=float).ravel()
    if values.size != expected or not np.all(np.isfinite(values)):
        return None
    return values


def choose_starting_point(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: Any,
    y: Any,
    *,
    declared: Any,
    lower: Any,
    upper: Any,
    function_class: Any = None,
    fixed: Any = None,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int | None = None,
) -> StartingPoint:
    """Return the parameters a fit should start from, and their provenance."""
    x_array = np.asarray(x, dtype=float).ravel()
    y_array = np.asarray(y, dtype=float).ravel()
    start = np.asarray(declared, dtype=float).ravel()
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()

    held = (
        np.zeros(start.size, dtype=bool)
        if fixed is None
        else np.asarray(fixed, dtype=bool).ravel()
    )

    estimate = ask_the_function(function_class, x_array, y_array, expected=start.size)
    if estimate is not None:
        # A fixed parameter is the user's answer, not the estimator's.
        estimate[held] = start[held]
        estimate = clip_into_bounds(estimate, low, high)
        return StartingPoint(
            values=estimate,
            source=FROM_FUNCTION,
            cost=sum_of_squares(model, x_array, y_array, estimate),
        )

    searched, score, _done = monte_carlo_p0(
        model,
        x_array,
        y_array,
        start,
        low,
        high,
        fixed=held,
        iterations=iterations,
        seed=seed,
    )
    if np.allclose(searched, start):
        # The search kept its incumbent: nothing it drew was better than the
        # declared values, so say so rather than claim an estimate.
        return StartingPoint(values=start, source=FROM_DECLARED, cost=score)

    return StartingPoint(values=searched, source=FROM_SEARCH, cost=score)


def clip_into_bounds(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return *values* moved inside the bounds.

    An estimate read off the data knows nothing about the limits the user
    typed, and ``least_squares`` refuses an infeasible start outright ("x0 is
    infeasible") rather than clipping it - so a good guess one micron outside
    a bound would abort the fit.
    """
    clipped = np.array(values, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)

    with np.errstate(invalid="ignore"):
        clipped = np.where(np.isfinite(low), np.maximum(clipped, low), clipped)
        clipped = np.where(np.isfinite(high), np.minimum(clipped, high), clipped)
    return clipped
