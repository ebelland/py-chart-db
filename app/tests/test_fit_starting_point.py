"""Where a fit starts, and which algorithm finishes it.

Both were previously one hard-wired answer: the declared ``p0`` and
``least_squares`` with its defaults. Both failures they cause are quiet ones -
a converged fit in the wrong valley reports a plausible RMSE and completely
wrong parameters, and a fit dragged by three outliers reports success too.
So these tests assert the properties that distinguish a right answer from a
confident wrong one, not that the code runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.functions import functions as library
from app.functions.base import base_function
from app.functions.monte_carlo import monte_carlo_p0, monte_carlo_search, sum_of_squares
from app.functions.optimizers import (
    BY_KEY,
    DEFAULT_OPTIMIZER,
    DIFFERENTIAL_EVOLUTION,
    LEVENBERG,
    MONTE_CARLO,
    OPTIMIZERS,
    TRUST_REGION,
    run_optimizer,
)
from app.functions.starting_point import (
    FROM_DECLARED,
    FROM_FUNCTION,
    FROM_SEARCH,
    ask_the_function,
    choose_starting_point,
    clip_into_bounds,
)

X = np.linspace(0.5, 10.0, 240)


# ======================================================================
# The functions' own estimators
# ======================================================================
@pytest.mark.parametrize(
    "function, truth, tolerance",
    [
        (library.linear, [3.0, 2.0], 1e-6),
        (library.quadratic, [1.0, -2.0, 0.5], 1e-6),
        (library.cubic, [0.0, 1.0, -0.3, 0.02], 1e-6),
        (library.gaussian_peak, [5.0, 4.0, 0.8, 1.0], 0.05),
        (library.michaelis_menten, [10.0, 2.0, 0.0], 0.05),
        (library.logistic4, [1.0, 8.0, 2.0, 5.0], 1.5),
        (library.power_law, [2.0, 1.5, 0.0], 0.2),
    ],
)
def test_an_estimator_lands_near_the_parameters_that_made_the_data(
    function, truth, tolerance
) -> None:
    """The point of an estimator: the optimiser starts in the right valley."""
    y = function.execute(X, np.asarray(truth, dtype=float))

    guess = function.initial_guess(X, y)

    assert guess is not None, "this function declares an estimator"
    assert len(guess) == len(truth)
    assert np.allclose(guess, truth, atol=tolerance, rtol=0.2)


def test_an_estimated_start_converges_where_the_declared_one_does_not() -> None:
    """A peak far from the origin: the declared p0 is a flat line at zero, and
    a local method started there has no gradient to follow."""
    truth = np.array([6.0, 8.0, 0.30, 2.0])
    y = library.gaussian_peak.execute(X, truth)
    residual = lambda p: y - library.gaussian_peak.execute(X, p)  # noqa: E731

    from_declared = run_optimizer(
        TRUST_REGION, residual, library.gaussian_peak.p0,
        [-np.inf] * 4, [np.inf] * 4, max_nfev=400,
    )
    from_estimate = run_optimizer(
        TRUST_REGION, residual, library.gaussian_peak.initial_guess(X, y),
        [-np.inf] * 4, [np.inf] * 4, max_nfev=400,
    )

    assert from_estimate.cost < 1e-12
    assert from_estimate.cost < from_declared.cost


def test_a_function_without_an_estimator_says_so_rather_than_guessing() -> None:
    """None is a different answer from a bad guess: the caller searches."""
    assert base_function.initial_guess(X, X) is None
    assert library.hill.initial_guess(X, X) is None


def test_a_guess_of_the_wrong_length_is_refused() -> None:
    """It would put values in the wrong parameters - the failure that looks
    like success, because the shape is right and only the meaning is wrong."""

    class Wrong:
        @staticmethod
        def initial_guess(x, y):
            return [1.0, 2.0]

    assert ask_the_function(Wrong, X, X, expected=4) is None


def test_an_estimator_that_raises_does_not_stop_the_fit() -> None:
    """One broken function is not a reason to refuse every fit."""

    class Explodes:
        @staticmethod
        def initial_guess(x, y):
            raise ValueError("no")

    assert ask_the_function(Explodes, X, X, expected=2) is None


def test_a_guess_with_a_nan_in_it_is_refused() -> None:
    class Nan:
        @staticmethod
        def initial_guess(x, y):
            return [1.0, float("nan")]

    assert ask_the_function(Nan, X, X, expected=2) is None


# ======================================================================
# Choosing a starting point
# ======================================================================
def _model(function):
    return lambda x, p: function.execute(np.asarray(x, dtype=float), np.asarray(p, dtype=float))


def test_the_function_is_asked_first() -> None:
    truth = np.array([5.0, 4.0, 0.8, 1.0])
    y = library.gaussian_peak.execute(X, truth)

    start = choose_starting_point(
        _model(library.gaussian_peak), X, y,
        declared=library.gaussian_peak.p0,
        lower=[-np.inf] * 4, upper=[np.inf] * 4,
        function_class=library.gaussian_peak,
    )

    assert start.source == FROM_FUNCTION
    assert start.estimated
    assert np.allclose(start.values, truth, atol=0.05)


def test_a_function_with_no_estimator_gets_a_search() -> None:
    truth = np.array([2.0, 3.0, 2.0, 0.0])
    y = library.hill.execute(X, truth)

    start = choose_starting_point(
        _model(library.hill), X, y,
        declared=library.hill.p0,
        lower=[-np.inf] * 4, upper=[np.inf] * 4,
        function_class=library.hill,
        iterations=3000, seed=5,
    )

    assert start.source == FROM_SEARCH
    assert start.cost < sum_of_squares(
        _model(library.hill), X, y, np.asarray(library.hill.p0, dtype=float)
    )


def test_a_search_that_finds_nothing_better_says_so() -> None:
    """Reported rather than dressed up: "the guess was invented" and "the fit
    is wrong" are different problems."""
    y = library.linear.execute(X, np.array([3.0, 2.0]))
    exact = [3.0, 2.0]

    start = choose_starting_point(
        _model(library.linear), X, y,
        declared=exact, lower=exact, upper=exact,
        function_class=None, iterations=200, seed=1,
    )

    assert start.source == FROM_DECLARED
    assert np.allclose(start.values, exact)


def test_a_fixed_parameter_keeps_the_value_the_user_fixed() -> None:
    """The estimator does not get to overrule an answer the user gave."""
    truth = np.array([5.0, 4.0, 0.8, 1.0])
    y = library.gaussian_peak.execute(X, truth)

    start = choose_starting_point(
        _model(library.gaussian_peak), X, y,
        declared=[1.0, 1.0, 1.0, 99.0],
        lower=[-np.inf] * 4, upper=[np.inf] * 4,
        function_class=library.gaussian_peak,
        fixed=[False, False, False, True],
    )

    assert start.values[3] == 99.0
    assert start.values[1] == pytest.approx(4.0, abs=0.05)


def test_an_estimate_outside_the_bounds_is_clipped_into_them() -> None:
    """least_squares refuses an infeasible start outright ("x0 is
    infeasible"), so a good guess a micron outside a bound would abort a fit
    the user had every reason to expect."""
    truth = np.array([5.0, 4.0, 0.8, 1.0])
    y = library.gaussian_peak.execute(X, truth)

    start = choose_starting_point(
        _model(library.gaussian_peak), X, y,
        declared=[1.0, 1.0, 1.0, 0.0],
        lower=[-np.inf, -np.inf, -np.inf, -np.inf],
        upper=[2.0, np.inf, np.inf, np.inf],
        function_class=library.gaussian_peak,
    )

    assert start.values[0] <= 2.0


def test_clipping_leaves_an_unbounded_parameter_alone() -> None:
    values = clip_into_bounds(
        np.array([5.0, -3.0]), np.array([-np.inf, 0.0]), np.array([np.inf, np.inf])
    )
    assert values.tolist() == [5.0, 0.0]


# ======================================================================
# The random search
# ======================================================================
def test_the_search_is_never_worse_than_what_it_started_with() -> None:
    """The incumbent is the starting point, so a search that finds nothing
    returns the guess rather than a random sample."""
    y = library.linear.execute(X, np.array([3.0, 2.0]))
    model = _model(library.linear)
    exact = np.array([3.0, 2.0])

    found, score, _done = monte_carlo_p0(
        model, X, y, exact, [-np.inf, -np.inf], [np.inf, np.inf],
        iterations=500, seed=2,
    )

    assert np.allclose(found, exact)
    assert score == pytest.approx(0.0, abs=1e-18)


def test_the_search_leaves_fixed_parameters_alone() -> None:
    y = library.linear.execute(X, np.array([3.0, 2.0]))

    found, _score, _done = monte_carlo_p0(
        _model(library.linear), X, y, [0.0, 0.0],
        [-np.inf, -np.inf], [np.inf, np.inf],
        fixed=[True, False], iterations=800, seed=4,
    )

    assert found[0] == 0.0, "a fixed parameter is the user's answer"


def test_the_same_seed_gives_the_same_search() -> None:
    """A fit that cannot be reproduced cannot be reported."""
    y = library.hill.execute(X, np.array([2.0, 3.0, 2.0, 0.0]))
    kwargs = dict(iterations=600, seed=11)

    first = monte_carlo_p0(_model(library.hill), X, y, library.hill.p0,
                           [-np.inf] * 4, [np.inf] * 4, **kwargs)
    second = monte_carlo_p0(_model(library.hill), X, y, library.hill.p0,
                            [-np.inf] * 4, [np.inf] * 4, **kwargs)

    assert np.array_equal(first[0], second[0])


def test_the_search_can_be_stopped_between_batches() -> None:
    """How the interface stays responsive during a long search."""
    seen: list[int] = []

    def stop_after_one(done: int, _best: float) -> bool:
        seen.append(done)
        return False

    _found, _score, done = monte_carlo_search(
        lambda p: float(np.sum(p * p)), [1.0], [-1.0], [1.0],
        iterations=10_000, batch=100, should_continue=stop_after_one,
    )

    assert done == 100
    assert seen == [100]


def test_parameters_the_model_cannot_evaluate_are_scored_as_bad_not_raised() -> None:
    def explodes(_x, p):
        if p[0] > 0:
            raise ValueError("nope")
        return np.zeros_like(X)

    assert sum_of_squares(explodes, X, np.zeros_like(X), np.array([1.0])) == np.inf


# ======================================================================
# The algorithms
# ======================================================================
DECAY_TRUTH = np.array([4.0, 0.7, 0.5])


def _decay_residual() -> tuple:
    y = library.exponential_decay.execute(X, DECAY_TRUTH)
    return y, (lambda p: y - library.exponential_decay.execute(X, p))


@pytest.mark.parametrize("optimizer", [o.key for o in OPTIMIZERS])
def test_every_algorithm_recovers_a_known_curve(optimizer: str) -> None:
    """Ten ways to the same answer; each one has to actually arrive."""
    _y, residual = _decay_residual()

    outcome = run_optimizer(
        optimizer, residual, [1.0, 1.0, 0.0], [-np.inf] * 3, [np.inf] * 3,
        max_nfev=600, seed=2, iterations=3000,
    )

    assert np.allclose(outcome.params, DECAY_TRUTH, atol=1e-4)


@pytest.mark.parametrize("optimizer", [o.key for o in OPTIMIZERS])
def test_every_algorithm_reports_a_jacobian(optimizer: str) -> None:
    """Uncertainties are read from it. A method that returns none leaves the
    user with parameters and no error bars, which is half an answer."""
    _y, residual = _decay_residual()

    outcome = run_optimizer(
        optimizer, residual, [1.0, 1.0, 0.0], [-np.inf] * 3, [np.inf] * 3,
        max_nfev=600, seed=2, iterations=2000,
    )

    assert outcome.jac is not None
    assert outcome.jac.shape == (X.size, 3)


def test_a_global_method_finds_the_valley_a_local_one_misses() -> None:
    """The reason the global methods are offered at all: from a bad start on a
    multi-modal surface, a local method converges to the wrong minimum and
    reports success."""
    truth = np.array([1.0, 3.0, 0.0, 0.0])
    y = library.sine.execute(X, truth)
    residual = lambda p: y - library.sine.execute(X, p)  # noqa: E731
    # A frequency far from the right one: the nearest minimum is not it.
    bad_start = [1.0, 0.4, 0.0, 0.0]

    local = run_optimizer(TRUST_REGION, residual, bad_start,
                          [-np.inf] * 4, [np.inf] * 4, max_nfev=800)
    globally = run_optimizer(DIFFERENTIAL_EVOLUTION, residual, bad_start,
                             [0.0, 0.0, -np.pi, -1.0], [5.0, 6.0, np.pi, 1.0],
                             max_nfev=2000, seed=3)

    assert local.cost > 1.0, "the local method is stuck, and says success"
    assert globally.cost < local.cost / 10.0


def test_a_robust_loss_survives_outliers_that_drag_least_squares() -> None:
    """A squared residual weights a point ten times off a hundred times more
    than a point one off, so a handful of bad points move the whole line."""
    truth = np.array([2.0, 1.5])
    y = library.linear.execute(X, truth)
    spoiled = y.copy()
    spoiled[[10, 60, 130]] += 60.0
    residual = lambda p: spoiled - library.linear.execute(X, p)  # noqa: E731

    ordinary = run_optimizer(TRUST_REGION, residual, [0.0, 0.0],
                             [-np.inf] * 2, [np.inf] * 2, max_nfev=400)
    robust = run_optimizer(TRUST_REGION, residual, [0.0, 0.0],
                           [-np.inf] * 2, [np.inf] * 2, max_nfev=400,
                           loss="cauchy")

    ordinary_error = float(np.max(np.abs(ordinary.params - truth)))
    robust_error = float(np.max(np.abs(robust.params - truth)))
    assert robust_error < ordinary_error / 5.0


def test_bounds_are_honoured_where_they_are_supported() -> None:
    _y, residual = _decay_residual()

    outcome = run_optimizer(TRUST_REGION, residual, [1.0, 1.0, 0.0],
                            [-np.inf, -np.inf, 1.0], [np.inf, np.inf, np.inf],
                            max_nfev=400)

    assert outcome.params[2] >= 1.0 - 1e-9


def test_levenberg_marquardt_declares_that_it_ignores_bounds() -> None:
    """SciPy refuses bounds with method="lm" outright rather than ignoring
    them, so the option says so and the runner drops them."""
    assert BY_KEY[LEVENBERG].supports_bounds is False

    _y, residual = _decay_residual()
    outcome = run_optimizer(LEVENBERG, residual, [1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0], [10.0, 10.0, 10.0], max_nfev=600)

    assert np.allclose(outcome.params, DECAY_TRUTH, atol=1e-4)


def test_an_unknown_algorithm_falls_back_to_the_default() -> None:
    """A saved fit naming a method that no longer exists still runs."""
    _y, residual = _decay_residual()

    outcome = run_optimizer("no-such-method", residual, [1.0, 1.0, 0.0],
                            [-np.inf] * 3, [np.inf] * 3, max_nfev=400)

    assert outcome.optimizer == DEFAULT_OPTIMIZER
    assert np.allclose(outcome.params, DECAY_TRUTH, atol=1e-4)


def test_a_failing_global_search_still_produces_a_fit() -> None:
    """The local polish runs either way, so a search that breaks costs the
    search, not the fit."""
    _y, residual = _decay_residual()

    outcome = run_optimizer(MONTE_CARLO, residual, [1.0, 1.0, 0.0],
                            [-np.inf] * 3, [np.inf] * 3,
                            max_nfev=400, seed=1, iterations=0)

    assert np.allclose(outcome.params, DECAY_TRUTH, atol=1e-4)


def test_only_the_least_squares_family_takes_a_loss() -> None:
    """A control that silently does nothing is worse than no control; the
    dialog disables it from this flag."""
    with_loss = {o.key for o in OPTIMIZERS if o.supports_loss}

    assert with_loss == {"trf", "dogbox"}
