"""Find where a series crosses a level: the roots of y(x) - level.

The question this answers is asked constantly and answered by eye: where does
the titration curve cross the endpoint, where does the response fall to half
its value, at what temperature does the difference change sign. Reading it off
the chart is accurate to about a pixel; this is accurate to the tolerance,
which is usually eleven digits better.

**Brackets first, then SciPy.** ``scipy.optimize`` solves ``f(x) = 0`` for a
function it can evaluate anywhere, and a series is not that - it is samples.
So the work is done in two halves, and the first is the one that decides
whether the answer is right:

1.  Scan consecutive samples for a sign change in ``y - level``. Every such
    pair *brackets* a crossing, by the intermediate value theorem, and a
    crossing that no pair brackets cannot be found from these samples at all.
    A sample sitting exactly on the level is a root already and is taken as
    one.
2.  Refine each bracket with a SciPy solver over an interpolant of the
    series. Bracketing solvers - Brent, bisection, TOMS 748 - cannot leave
    the interval they were given, so the refinement can only improve the
    answer, never wander off to a different crossing.

**Which interpolant is the real assumption.** Linear is the honest default:
it claims nothing between the samples that the samples do not already say,
and on a linear interpolant the root is the straight-line crossing - the
answer everyone computes by hand. Cubic is better for a smooth signal that
was sampled coarsely, and worse for a noisy one, where it overshoots between
points and can invent crossings that are not there. Both are offered; only
the first is the default.

**Newton is offered and is not the default.** ``scipy.optimize.newton`` -
the secant method here, since a sampled series has no analytic derivative -
converges faster and is not bracketed, so it can step outside the interval
and return a crossing somewhere else entirely, or none. It is kept because
it is the right tool on a smooth, well-separated curve, and its result is
checked against the bracket it came from before being accepted.

Sitting next to the peaks dialog on purpose: peaks are where the derivative
crosses zero, so a derivative from the calculus dialog run through this one
locates them a second way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QFormLayout, QVBoxLayout, QWidget
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq, newton, toms748

from app.data.data_source import row_value
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.series_operations.dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.series_operations.parameter_spec import (
    ChoiceParam,
    FloatParam,
    IntParam,
)
from app.styles.style import create_doc_link, set_doc_link
from app.utils import report_html
from app.utils.i18n import _

# --- Solvers ----------------------------------------------------------

ROOT_BRENT = "Brent"
ROOT_BISECT = "Bisection"
ROOT_TOMS748 = "TOMS 748"
ROOT_NEWTON = "Newton (secant)"

ROOT_MODELS = (ROOT_BRENT, ROOT_BISECT, ROOT_TOMS748, ROOT_NEWTON)

#: The bracketing solvers, which cannot return an x outside the interval they
#: were handed. Newton is the exception, and is checked afterwards.
BRACKETED_MODELS = (ROOT_BRENT, ROOT_BISECT, ROOT_TOMS748)

ROOT_DOCS = {
    ROOT_BRENT: (
        "Brent's method",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html",
    ),
    ROOT_BISECT: (
        "Bisection method",
        "https://en.wikipedia.org/wiki/Bisection_method",
    ),
    ROOT_TOMS748: (
        "TOMS 748",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.toms748.html",
    ),
    ROOT_NEWTON: (
        "Secant method",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html",
    ),
}

INTERP_LINEAR = "linear"
INTERP_CUBIC = "cubic"
INTERP_PCHIP = "pchip"


@dataclass(slots=True)
class Root:
    """One located crossing."""

    x: float
    #: The interpolant's value there. Not exactly the level - it is the
    #: residual that says how well the solver converged, and a large one is
    #: the sign of a bracket the interpolant does not really cross.
    y: float
    #: True when the series is going up through the level at this x. A
    #: rising and a falling crossing are different events - a threshold
    #: being exceeded and a recovery - and the report says which.
    rising: bool
    #: How the value was arrived at: the solver's name, or "sample" for a
    #: point that sat on the level to begin with.
    method: str
    iterations: int = 0


@dataclass(slots=True)
class RootResult:
    """Every crossing found in one source series."""

    source_name: str
    result_name: str
    model: str
    level: float
    roots: list[Root] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x": [root.x for root in self.roots],
                "y": [root.y for root in self.roots],
                "level": [self.level for _root in self.roots],
                "rising": [int(root.rising) for root in self.roots],
                "method": [root.method for root in self.roots],
                "iterations": [root.iterations for root in self.roots],
            }
        )


class SeriesRootsDialog(SeriesOperationDialogBase):
    """Solve y(x) = level for x, on every crossing the samples bracket."""

    Name: str = "Roots"
    Description = "Find where a series crosses a level"

    # A crossing is a property of consecutive samples, so x has to be in
    # order and each x can only have one y.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_REQUIRES_UNIQUE_X = True
    # Two points are one interval, which is the smallest thing that can
    # bracket a crossing.
    INPUT_MINIMUM_POINTS = 2

    PARAMS = (
        FloatParam(
            "level",
            "Level:",
            tooltip=(
                "The y value to solve for. 0 finds the zeros; any other "
                "value finds where the series crosses that level, which is "
                "the same problem shifted."
            ),
            default_value=0.0,
            minimum=-1.0e12,
            maximum=1.0e12,
            decimals=6,
            step=0.1,
        ),
        ChoiceParam(
            "interpolation",
            "Between samples:",
            tooltip=(
                "What the series is assumed to do between two points. "
                "Linear claims nothing the samples do not; the two curved "
                "options fit a smooth signal better and can overshoot a "
                "noisy one into crossings that are not there."
            ),
            choices=(
                ("Straight line", INTERP_LINEAR),
                ("Cubic spline", INTERP_CUBIC),
                ("Monotone cubic (PCHIP)", INTERP_PCHIP),
            ),
        ),
        IntParam(
            "tolerance_digits",
            "Tolerance (digits):",
            tooltip=(
                "How many decimal places of x the solver has to settle "
                "before it stops. Nine is far below the precision of any "
                "measured x; the cost of raising it is a few iterations."
            ),
            default_value=9,
            minimum=3,
            maximum=14,
        ),
        IntParam(
            "max_iter",
            "Maximum iterations:",
            tooltip=(
                "Per crossing. Brent and TOMS 748 rarely need ten; the "
                "limit is what stops a pathological bracket from hanging "
                "the dialog."
            ),
            default_value=100,
            minimum=5,
            maximum=10_000,
        ),
        IntParam(
            "limit",
            "Report at most:",
            tooltip=(
                "Keeps the first crossings in x order when a noisy series "
                "crosses the level hundreds of times."
            ),
            default_value=100,
            minimum=1,
            maximum=10_000,
        ),
    )

    #: A curve crossing a horizontal line, with the crossing marked.
    Icon = """
    <path d="M3 12h18"/>
    <path d="M3 19c4 0 5-14 9-14s5 10 9 10"/>
    <circle cx="8.4" cy="12" r="1.8"/>
    """

    def __init__(
        self,
        *,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        if repo is None:
            applogger.error("SeriesRootsDialog requires a repository instance.")

        self._last_results: list[RootResult] = []

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Roots",
            parent=parent,
            width=780,
            height=640,
        )
        self.series_selector.reload(select_all_series=True)
        self._refresh_visibility()
        self.refresh_results()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_operation_widgets(self) -> None:
        self._doc_link = create_doc_link(self)

    def build_model_selector(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        container = QWidget(panel)
        form = QFormLayout(container)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo.addItems(ROOT_MODELS)
        self.model_combo.setToolTip(
            _(
                "How each bracketed crossing is refined. The three bracketing "
                "solvers cannot leave the interval they were given; Newton "
                "can, and is checked afterwards."
            )
        )
        form.addRow(_("Solver:"), self.model_combo)
        form.addRow(_("Docs:"), self._doc_link)

        layout.addWidget(container)
        return panel

    def connect_operation_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.model_combo.currentIndexChanged.connect(self.refresh_results)

    def _refresh_visibility(self) -> None:
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()
        title, url = ROOT_DOCS[self._model()]
        set_doc_link(self._doc_link, title, url)

    def _model(self) -> str:
        return self.model_combo.currentText() or ROOT_BRENT

    def refresh_results(self) -> None:
        try:
            results = self.compute_results()
        except Exception as exc:
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
    # Computation
    # ------------------------------------------------------------------

    def compute_results(self) -> list[RootResult]:
        model = self._model()
        params = self.parameter_values()

        results: list[RootResult] = []
        errors: list[str] = []

        for row in self.selected_series():
            name = str(row_value(row, "name", "series_name", default="Series"))
            try:
                x_values, y_values = self.series_xy(row, name)
                results.append(self._solve_one(name, x_values, y_values, model, params))
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if errors and not results:
            raise ValueError("; ".join(errors))
        for message in errors:
            applogger.warning(message, show_dialog=False, raise_error=False)

        return results

    def _solve_one(
        self,
        name: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        model: str,
        params: Mapping[str, Any],
    ) -> RootResult:
        level = float(params.get("level", 0.0))
        limit = int(params.get("limit", 100))
        xtol = self._xtol(params)
        max_iter = int(params.get("max_iter", 100))

        offset = y_values - level
        interpolant = self._interpolant(
            x_values, offset, str(params.get("interpolation", INTERP_LINEAR))
        )

        roots: list[Root] = []
        for left, right, exact in self._brackets(x_values, offset):
            if exact is not None:
                roots.append(
                    Root(
                        x=float(exact),
                        y=level,
                        rising=self._is_rising(interpolant, float(exact), x_values),
                        method="sample",
                    )
                )
                continue

            found = self._refine(
                interpolant, left, right, model, xtol=xtol, max_iter=max_iter
            )
            if found is None:
                continue
            x_root, iterations = found
            roots.append(
                Root(
                    x=float(x_root),
                    y=float(interpolant(x_root)) + level,
                    rising=self._is_rising(interpolant, float(x_root), x_values),
                    method=model,
                    iterations=int(iterations),
                )
            )

        roots.sort(key=lambda root: root.x)
        truncated = len(roots) > limit
        if truncated:
            # In x order, not by any measure of quality: a crossing is a
            # crossing, and the first hundred is the only defensible
            # "first" when they are all equally real.
            roots = roots[:limit]

        return RootResult(
            source_name=name,
            result_name=f"{name} - roots",
            model=model,
            level=level,
            roots=roots,
            metadata={
                "found": len(roots),
                "truncated": truncated,
                "interpolation": str(params.get("interpolation", INTERP_LINEAR)),
                "samples": int(x_values.size),
            },
        )

    # --- The bracketing half, which decides what can be found ----------

    def _brackets(
        self, x_values: np.ndarray, offset: np.ndarray
    ) -> list[tuple[float, float, float | None]]:
        """Return the intervals that must contain a crossing.

        ``(left, right, exact)``: *exact* is set when a sample sits on the
        level, in which case there is nothing to solve. A run of consecutive
        samples exactly on the level would otherwise be reported as one root
        each, so only the first of such a run is taken - the series does not
        cross there, it rests there.
        """
        brackets: list[tuple[float, float, float | None]] = []
        previous_was_zero = False

        for index in range(offset.size):
            value = float(offset[index])
            if value == 0.0:
                if not previous_was_zero:
                    brackets.append((0.0, 0.0, float(x_values[index])))
                previous_was_zero = True
                continue
            previous_was_zero = False

            if index + 1 >= offset.size:
                break
            following = float(offset[index + 1])
            if following != 0.0 and (value > 0.0) != (following > 0.0):
                brackets.append(
                    (float(x_values[index]), float(x_values[index + 1]), None)
                )

        return brackets

    def _interpolant(
        self, x_values: np.ndarray, offset: np.ndarray, kind: str
    ) -> Any:
        """Return a callable for ``y(x) - level`` between the samples.

        Cubic and PCHIP need four and two points respectively and both need
        strictly increasing x, which ``prepare_input_xy`` has already
        guaranteed. A series too short for the chosen interpolant falls back
        to the straight line rather than failing: the assumption degrades,
        the answer still exists.
        """
        if kind == INTERP_CUBIC and x_values.size >= 4:
            return CubicSpline(x_values, offset)
        if kind == INTERP_PCHIP and x_values.size >= 2:
            return PchipInterpolator(x_values, offset)
        return lambda value: np.interp(value, x_values, offset)

    # --- The SciPy half ------------------------------------------------

    def _xtol(self, params: Mapping[str, Any]) -> float:
        """Return the x tolerance, from the number of digits asked for.

        Asked for as digits rather than as a number because that is how
        anyone thinks about it, and because a spin box showing
        "0.000000001000" is a control nobody can read or set.
        """
        try:
            digits = int(params.get("tolerance_digits", 9))
        except (TypeError, ValueError):
            digits = 9
        return 10.0 ** -min(max(digits, 1), 15)

    def _refine(
        self,
        interpolant: Any,
        left: float,
        right: float,
        model: str,
        *,
        xtol: float,
        max_iter: int,
    ) -> tuple[float, int] | None:
        """Solve inside one bracket, or return None with a log line.

        Every solver here is ``scipy.optimize``'s; what differs is what each
        is allowed to do with the bracket. A failure costs the one crossing
        rather than the whole series: an interpolant that wanders can leave a
        bracket its samples did straddle, and the other twenty crossings are
        still worth reporting.
        """
        def evaluate(value: float) -> float:
            return float(interpolant(value))

        try:
            if model == ROOT_NEWTON:
                return self._refine_newton(evaluate, left, right, xtol, max_iter)

            solver = {
                ROOT_BRENT: brentq,
                ROOT_TOMS748: toms748,
            }.get(model)
            if solver is None:
                return self._refine_bisect(evaluate, left, right, xtol, max_iter)

            root, result = solver(
                evaluate,
                left,
                right,
                xtol=xtol,
                maxiter=max_iter,
                full_output=True,
            )
            if not result.converged:
                raise RuntimeError("the solver did not converge")
            return float(root), int(result.iterations)
        except Exception as exc:
            applogger.info(
                "No root in [%g, %g]: %s. The samples change sign across it, "
                "so the interpolant does not - which is the interpolant "
                "disagreeing with the data rather than an error.",
                left,
                right,
                exc,
            )
            return None

    def _refine_bisect(
        self,
        evaluate: Any,
        left: float,
        right: float,
        xtol: float,
        max_iter: int,
    ) -> tuple[float, int] | None:
        """Halve the bracket until it is narrower than the tolerance.

        SciPy's own ``bisect`` would do this; it is written out because it is
        four lines and because the iteration count it reports is then the
        real one rather than the solver's internal bookkeeping.
        """
        low, high = float(left), float(right)
        f_low = evaluate(low)
        iterations = 0
        while high - low > xtol and iterations < max_iter:
            middle = 0.5 * (low + high)
            f_middle = evaluate(middle)
            if f_middle == 0.0:
                return middle, iterations + 1
            if (f_low > 0.0) != (f_middle > 0.0):
                high = middle
            else:
                low, f_low = middle, f_middle
            iterations += 1
        return 0.5 * (low + high), iterations

    def _refine_newton(
        self,
        evaluate: Any,
        left: float,
        right: float,
        xtol: float,
        max_iter: int,
    ) -> tuple[float, int] | None:
        """Secant from the middle of the bracket, then check it stayed in it.

        ``newton`` is not a bracketing method: with no derivative it runs the
        secant method, which converges faster than Brent on a smooth curve
        and is free to step anywhere. A result outside the bracket is a
        different crossing, or none - it is not the root of *this* interval,
        so it is refused rather than reported at the wrong x.
        """
        root, result = newton(
            evaluate,
            0.5 * (left + right),
            tol=xtol,
            maxiter=max_iter,
            full_output=True,
            disp=False,
        )
        if not result.converged:
            raise RuntimeError("the secant iteration did not converge")
        if not (min(left, right) <= float(root) <= max(left, right)):
            raise RuntimeError(
                "the secant iteration left the bracket it started in"
            )
        return float(root), int(result.iterations)

    def _is_rising(
        self, interpolant: Any, x_root: float, x_values: np.ndarray
    ) -> bool:
        """Say whether the series goes up through the level at *x_root*.

        Measured over a small step either side rather than from a derivative,
        so it means the same thing for every interpolant - ``np.interp`` has
        no derivative to ask for.
        """
        span = float(x_values[-1] - x_values[0])
        step = (span / max(x_values.size - 1, 1)) * 1.0e-3 if span > 0.0 else 1.0e-9
        try:
            before = float(interpolant(x_root - step))
            after = float(interpolant(x_root + step))
        except Exception:  # noqa: BLE001 - outside the interpolant's domain
            return True
        return after >= before

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def result_to_frame(self, result: RootResult) -> pd.DataFrame:
        return result.to_frame()

    def result_series_spec(
        self,
        axis_id: int,
        table_name: str,
        result: RootResult,
    ) -> ResultSeriesSpec:
        del axis_id
        return ResultSeriesSpec(
            name=result.result_name,
            sql_query=f'SELECT x, y FROM "{table_name}" ORDER BY x',
            roles={"x": "x", "y": "y"},
            style={
                "generated_roots": True,
                "roots_dialog": "series_roots",
                "source_name": result.source_name,
                "model": result.model,
                "level": result.level,
                # Markers, no line: the crossings are separate findings,
                # and joining them would draw a horizontal line at the
                # level that looks like a series.
                "linestyle": "",
                "marker": "o",
                "markersize": 7.0,
            },
        )

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_roots": True, "roots_dialog": "series_roots"}

    def result_table_name(self, axis_id: int, result: RootResult) -> str:
        return generated_table_name(
            f"Roots_axis{axis_id}_{result.source_name}",
            fallback="Roots_Result",
        )

    @property
    def operation_label(self) -> str:
        return "Roots"

    RESULTS_ARE_HTML = True

    def format_results(self, results: Sequence[RootResult]) -> str:
        if not results:
            return report_html.note(_("No results."))

        sections: list[str] = []
        for result in results:
            if not result.roots:
                sections.append(
                    report_html.section(
                        result.source_name,
                        report_html.note(
                            _(
                                "The series never crosses this level. Check "
                                "the level against the data's range - a "
                                "crossing no two samples straddle cannot be "
                                "found from these samples."
                            )
                        ),
                    )
                )
                continue

            rows = [
                (
                    str(index + 1),
                    report_html.format_number(root.x),
                    _("rising") if root.rising else _("falling"),
                    root.method,
                    str(root.iterations) if root.iterations else "-",
                    report_html.format_number(root.y - result.level, digits=3),
                )
                for index, root in enumerate(result.roots)
            ]
            heading = f"{result.source_name} — {len(result.roots)}"
            if result.metadata.get("truncated"):
                heading = f"{heading} ({_('truncated')})"

            sections.append(
                report_html.section(
                    heading,
                    report_html.table(
                        (
                            "#",
                            "x",
                            _("Direction"),
                            _("Solver"),
                            _("Iterations"),
                            _("Residual"),
                        ),
                        rows,
                        align=("right", "right", "left", "left", "right", "right"),
                    ),
                )
            )

        subtitle = _("level {level}").format(
            level=report_html.format_number(results[0].level)
        )
        return report_html.document(
            _("Roots"), f"{self._model()} — {subtitle}", *sections
        )
