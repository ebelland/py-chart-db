"""Searching for parameters at random, when nothing better is available.

Two jobs, and the second is only worth doing because of the first.

A fit needs a starting point.  A function that implements ``initial_guess``
supplies one read off the data, which is always the better answer - it knows
what its own parameters mean.  For the rest, and for a fit the optimiser walks
away from, the alternative is to look: sample the parameter space, keep the
best sum of squares, and hand that to the optimiser.  It is not clever, and
that is the point - it makes no assumption about the model at all, so it works
on the forty-five functions that have no estimator and on user functions this
library has never seen.

``monte_carlo_p0`` is also offered as an optimiser in its own right (see
``app/functions/optimizers.py``), where it is followed by a local fit: random
search finds the right valley, gradient descent finds its floor, and neither
does the other's job well.

Pure numpy: no Qt, no repository, nothing that needs a window to test.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


# ----------------------------------------------------------------------
# Monte Carlo search
# ----------------------------------------------------------------------

#: Where a parameter has no bound, sampling has to happen somewhere. This is
#: the multiple of the current value used as a range, and the absolute range
#: used when that value is zero. Wide enough to escape a bad guess, narrow
#: enough that the samples are not all useless.
UNBOUNDED_SPAN: float = 10.0


def _sampling_bounds(
    p0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite bounds to sample between.

    An infinite bound cannot be sampled from uniformly, so it is replaced by a
    window around the current value. This is why bounds are worth setting: a
    parameter with real limits is searched where the answer is, and one
    without is searched near wherever it happens to start.
    """
    low = np.array(lower, dtype=float)
    high = np.array(upper, dtype=float)

    for index in range(p0.size):
        centre = float(p0[index])
        span = abs(centre) * UNBOUNDED_SPAN if centre else UNBOUNDED_SPAN
        if not np.isfinite(low[index]):
            low[index] = centre - span
        if not np.isfinite(high[index]):
            high[index] = centre + span
        if low[index] > high[index]:
            low[index], high[index] = high[index], low[index]

    return low, high


def sum_of_squares(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
) -> float:
    """Return the residual sum of squares, or infinity when it cannot be had.

    Infinity rather than an exception: a random sample will land on parameters
    a model cannot evaluate - a log of a negative, an overflow - and the search
    has to treat that as "bad, move on" rather than stop.
    """
    try:
        predicted = np.asarray(model(x, params), dtype=float)
    except Exception:
        return np.inf
    if predicted.shape != y.shape:
        return np.inf

    residual = y - predicted
    if not np.all(np.isfinite(residual)):
        return np.inf
    return float(np.sum(residual * residual))


def monte_carlo_p0(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: Any,
    y: Any,
    p0: Any,
    lower: Any,
    upper: Any,
    *,
    fixed: Any | None = None,
    iterations: int = 20_000,
    batch: int = 500,
    seed: int | None = None,
    should_continue: Callable[[int, float], bool] | None = None,
) -> tuple[np.ndarray, float, int]:
    """Search a model's parameters at random against ``y``; see the search below."""
    x_array = np.asarray(x, dtype=float).ravel()
    y_array = np.asarray(y, dtype=float).ravel()

    return monte_carlo_search(
        lambda params: sum_of_squares(model, x_array, y_array, params),
        p0,
        lower,
        upper,
        fixed=fixed,
        iterations=iterations,
        batch=batch,
        seed=seed,
        should_continue=should_continue,
    )


def monte_carlo_search(
    cost: Callable[[np.ndarray], float],
    p0: Any,
    lower: Any,
    upper: Any,
    *,
    fixed: Any | None = None,
    iterations: int = 20_000,
    batch: int = 500,
    seed: int | None = None,
    should_continue: Callable[[int, float], bool] | None = None,
) -> tuple[np.ndarray, float, int]:
    """Search the parameter space at random; return the best found.

    Takes a cost rather than a model so that the same search serves both jobs:
    finding a starting point (cost = residual sum of squares against the data)
    and running as an optimiser over whatever residual the caller has already
    built - weights, robust loss and fixed parameters included.

    Returns ``(parameters, sum_of_squares, iterations_done)``. The current
    values are the incumbent, so the result is never worse than what was
    started with - a search that finds nothing better returns the guess it was
    given rather than a random sample.

    ``should_continue(done, best)`` is called between batches and returning
    False stops the search. Between batches rather than per sample because the
    callback is how the interface stays responsive, and asking it a hundred
    thousand times would cost more than the search.

    Fixed parameters are held at their given values and never sampled: the
    user fixed them, and a search that moved them anyway would be answering a
    different question.
    """
    start = np.asarray(p0, dtype=float).ravel()

    held = (
        np.zeros(start.size, dtype=bool)
        if fixed is None
        else np.asarray(fixed, dtype=bool).ravel()
    )
    free = ~held
    if not np.any(free) or start.size == 0:
        return start, cost(start), 0

    low, high = _sampling_bounds(start, np.asarray(lower, dtype=float),
                                 np.asarray(upper, dtype=float))

    best = start.copy()
    best_score = cost(best)

    rng = np.random.default_rng(seed)
    done = 0
    free_count = int(np.count_nonzero(free))

    while done < iterations:
        this_batch = min(batch, iterations - done)
        samples = rng.uniform(
            low[free], high[free], size=(this_batch, free_count)
        )

        for row in samples:
            candidate = best.copy()
            # From the incumbent, not from start: holding the best found so far
            # means a later batch refines it rather than re-exploring from the
            # original guess every time.
            candidate[free] = row
            score = cost(candidate)
            if score < best_score:
                best_score = score
                best = candidate

        done += this_batch
        if should_continue is not None and not should_continue(done, best_score):
            break

    return best, best_score, done
