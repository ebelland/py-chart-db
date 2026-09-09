"""The root-finding operation, checked against roots that are known exactly.

Every assertion here compares against an analytic answer - the zeros of a
sine, the crossing of a straight line - because a root finder is the kind of
code that is convincingly wrong: it returns a number of the right magnitude
whatever it does, and only arithmetic that knows the answer can tell.

The properties worth pinning are the two halves of the method. The bracket
scan decides *what can be found at all*: a crossing no two samples straddle
is invisible, a sample sitting on the level is a root already, and a series
that rests on the level is not crossing it. The SciPy refinement decides
*how well*: which is why the accuracy tests compare a solver's answer to the
straight-line crossing it started from rather than only to a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.series_operations.roots_dialog import (
    BRACKETED_MODELS,
    INTERP_CUBIC,
    INTERP_LINEAR,
    INTERP_PCHIP,
    ROOT_BISECT,
    ROOT_BRENT,
    ROOT_MODELS,
    ROOT_NEWTON,
    ROOT_TOMS748,
    SeriesRootsDialog,
)


def _bare() -> SeriesRootsDialog:
    """An instance without its Qt dialog - the numerics are plain methods."""
    return SeriesRootsDialog.__new__(SeriesRootsDialog)


def _params(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "level": 0.0,
        "interpolation": INTERP_LINEAR,
        "tolerance_digits": 12,
        "max_iter": 100,
        "limit": 100,
    }
    values.update(overrides)
    return values


def _solve(x, y, model=ROOT_BRENT, **overrides):
    return _bare()._solve_one("s", np.asarray(x, dtype=float),
                              np.asarray(y, dtype=float), model, _params(**overrides))


# Two full periods, sampled finely enough that linear interpolation is close
# and coarsely enough that cubic is visibly closer.
SINE_X = np.linspace(0.0, 4.0 * np.pi, 200)
SINE_Y = np.sin(SINE_X)
SINE_ZEROS = np.array([0.0, np.pi, 2.0 * np.pi, 3.0 * np.pi])


# ======================================================================
# What is found
# ======================================================================
@pytest.mark.parametrize("model", ROOT_MODELS)
def test_every_solver_finds_the_zeros_of_a_sine(model: str) -> None:
    result = _solve(SINE_X, SINE_Y, model)

    assert len(result.roots) == len(SINE_ZEROS)
    assert np.allclose([root.x for root in result.roots], SINE_ZEROS, atol=1e-5)


def test_the_roots_come_back_in_x_order() -> None:
    result = _solve(SINE_X, SINE_Y)

    positions = [root.x for root in result.roots]
    assert positions == sorted(positions)


def test_a_level_other_than_zero_is_the_same_problem_shifted() -> None:
    """sin(x) = 0.5 at pi/6 and at 5pi/6."""
    result = _solve(SINE_X, SINE_Y, level=0.5, interpolation=INTERP_CUBIC)

    first_two = [root.x for root in result.roots][:2]
    assert np.allclose(first_two, [np.pi / 6, 5 * np.pi / 6], atol=1e-6)


def test_a_series_that_never_reaches_the_level_has_no_roots() -> None:
    result = _solve(SINE_X, SINE_Y + 5.0)

    assert result.roots == []
    assert result.metadata["found"] == 0


def test_a_crossing_no_two_samples_straddle_cannot_be_found() -> None:
    """The honest limit of the method, stated as a test: a dip below zero
    that falls entirely between two samples leaves no sign change to see."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 1.0, 1.0])  # the dip at 1.5 was never sampled

    assert _solve(x, y).roots == []


def test_a_sample_sitting_on_the_level_is_a_root_already() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([-1.0, 0.0, 1.0])

    roots = _solve(x, y).roots
    assert len(roots) == 1
    assert roots[0].x == 1.0
    assert roots[0].method == "sample", "nothing was left to solve"


def test_a_series_resting_on_the_level_is_not_crossing_it() -> None:
    """A run of samples exactly on the level would otherwise be reported as
    one root each - the series does not cross there, it sits there."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([-1.0, 0.0, 0.0, 0.0, 1.0])

    roots = _solve(x, y).roots
    assert [root.x for root in roots] == [1.0]


def test_touching_the_level_and_turning_back_is_reported_once() -> None:
    """y = x^2 at its minimum: it reaches zero without changing sign."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = x**2

    roots = _solve(x, y).roots
    assert [root.x for root in roots] == [0.0]


def test_the_direction_of_each_crossing_is_reported() -> None:
    result = _solve(SINE_X, SINE_Y)

    assert [root.rising for root in result.roots] == [True, False, True, False]


def test_the_limit_truncates_and_says_so() -> None:
    result = _solve(SINE_X, SINE_Y, limit=2)

    assert len(result.roots) == 2
    assert result.metadata["truncated"] is True


# ======================================================================
# How well it is found
# ======================================================================
def test_a_straight_line_crossing_is_exact() -> None:
    """y = 2x - 3 crosses zero at 1.5, and every solver should say so to
    the tolerance rather than to the sample spacing."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x - 3.0

    for model in ROOT_MODELS:
        root = _solve(x, y, model).roots[0].x
        assert abs(root - 1.5) < 1e-9, model


def test_a_cubic_interpolant_beats_a_straight_line_on_a_smooth_curve() -> None:
    """The whole reason the choice is offered. On a coarsely sampled curve
    the straight-line answer is limited by the sampling - it cuts the corner
    of every curved segment - while the spline is limited by the solver.

    e^x - 2, whose root is ln 2, and deliberately not a sine: a sine is
    antisymmetric about its zero, so a symmetric pair of samples straddling
    it gives the straight line the exact answer for the wrong reason.
    """
    coarse_x = np.linspace(0.0, 3.0, 13)
    coarse_y = np.exp(coarse_x) - 2.0
    true_root = float(np.log(2.0))

    linear = _solve(coarse_x, coarse_y).roots[0].x
    cubic = _solve(coarse_x, coarse_y, interpolation=INTERP_CUBIC).roots[0].x

    assert abs(linear - true_root) > 1e-3, "the sampling is too fine to show it"
    assert abs(cubic - true_root) < abs(linear - true_root) / 100.0


@pytest.mark.parametrize("interpolation", [INTERP_LINEAR, INTERP_CUBIC, INTERP_PCHIP])
def test_every_interpolant_finds_the_same_crossings(interpolation: str) -> None:
    result = _solve(SINE_X, SINE_Y, interpolation=interpolation)

    assert np.allclose([root.x for root in result.roots], SINE_ZEROS, atol=1e-4)


def test_a_series_too_short_for_a_spline_falls_back_to_the_line() -> None:
    """The assumption degrades; the answer still exists."""
    x = np.array([0.0, 1.0])
    y = np.array([-1.0, 1.0])

    root = _solve(x, y, interpolation=INTERP_CUBIC).roots[0]
    assert root.x == pytest.approx(0.5, abs=1e-9)


def test_the_residual_says_how_well_the_solver_converged() -> None:
    result = _solve(SINE_X, SINE_Y, level=0.5, interpolation=INTERP_CUBIC)

    for root in result.roots:
        assert abs(root.y - result.level) < 1e-9


def test_the_solver_and_its_iteration_count_are_recorded() -> None:
    """Per root, not per result: the first zero of this sine is a sample
    sitting on the level, which no solver was run for."""
    result = _solve(SINE_X, SINE_Y, ROOT_BISECT)

    refined = [root for root in result.roots if root.method != "sample"]
    assert len(refined) == 3
    assert {root.method for root in refined} == {ROOT_BISECT}
    assert all(root.iterations > 0 for root in refined)
    assert all(root.iterations == 0 for root in result.roots if root not in refined)


# ======================================================================
# The solvers themselves
# ======================================================================
@pytest.mark.parametrize("model", BRACKETED_MODELS)
def test_a_bracketing_solver_stays_inside_its_bracket(model: str) -> None:
    """Which is what makes the two-step method safe: the scan decides which
    crossing, and the solver can only sharpen it."""
    x = np.linspace(0.0, 10.0, 41)
    y = np.sin(x)
    brackets = _bare()._brackets(x, y)

    for left, right, exact in brackets:
        if exact is not None:
            continue
        found = _bare()._refine(
            lambda value: float(np.interp(value, x, y)),
            left,
            right,
            model,
            xtol=1e-12,
            max_iter=100,
        )
        assert found is not None
        assert left <= found[0] <= right


def test_newton_is_refused_when_it_leaves_the_bracket() -> None:
    """It is not a bracketing method: a result outside the interval is a
    different crossing, or none, so it is not this interval's root."""
    dialog = _bare()

    with pytest.raises(RuntimeError, match="left the bracket"):
        # Constant slope pointing far to the right of the bracket: the
        # secant step lands nowhere near [0, 1].
        dialog._refine_newton(lambda value: 0.001 * (value - 900.0), 0.0, 1.0, 1e-9, 50)


def test_a_bracket_the_interpolant_does_not_cross_costs_only_that_root() -> None:
    """A curved interpolant can disagree with the samples that bracketed it;
    the other crossings are still worth reporting."""
    dialog = _bare()

    found = dialog._refine(
        lambda value: 1.0, 0.0, 1.0, ROOT_BRENT, xtol=1e-12, max_iter=100
    )

    assert found is None


def test_the_tolerance_is_asked_for_as_digits() -> None:
    """A spin box showing 0.000000001000 is a control nobody can read."""
    dialog = _bare()

    assert dialog._xtol({"tolerance_digits": 9}) == pytest.approx(1e-9)
    assert dialog._xtol({"tolerance_digits": 3}) == pytest.approx(1e-3)
    assert dialog._xtol({}) == pytest.approx(1e-9), "the default stands alone"
    assert dialog._xtol({"tolerance_digits": "nonsense"}) == pytest.approx(1e-9)


def test_bisection_converges_to_within_the_tolerance() -> None:
    dialog = _bare()

    root, iterations = dialog._refine_bisect(lambda v: v - 0.3, 0.0, 1.0, 1e-9, 200)

    assert abs(root - 0.3) <= 1e-9
    assert iterations < 200


def test_the_solvers_offered_are_the_ones_scipy_provides() -> None:
    assert ROOT_BRENT in ROOT_MODELS
    assert ROOT_TOMS748 in ROOT_MODELS
    assert ROOT_NEWTON in ROOT_MODELS
    assert set(BRACKETED_MODELS) < set(ROOT_MODELS)
    assert ROOT_NEWTON not in BRACKETED_MODELS


# ======================================================================
# The result, as the rest of the app sees it
# ======================================================================
def test_the_result_frame_carries_every_root_and_its_measurements() -> None:
    result = _solve(SINE_X, SINE_Y)
    frame = result.to_frame()

    assert list(frame.columns) == [
        "x", "y", "level", "rising", "method", "iterations"
    ]
    assert len(frame) == len(result.roots)


def test_the_generated_series_is_drawn_as_marks_not_a_line() -> None:
    """Joining the crossings would draw a horizontal line at the level that
    looks like a series of its own."""
    result = _solve(SINE_X, SINE_Y)
    spec = _bare().result_series_spec(1, "_Roots_axis1_s", result)

    assert spec.style["linestyle"] == ""
    assert spec.style["marker"] == "o"
    assert spec.style["generated_roots"] is True
