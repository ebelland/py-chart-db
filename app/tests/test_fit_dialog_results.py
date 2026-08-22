"""Tests for the fit dialog's two verbs and its report.

Fitting and drawing used to be the same button: Preview called ``on_fit``, so
the optimiser replaced the parameters before anything reached the chart and a
hand-typed starting guess could never be seen.  They are now separate, with the
parameter table as the single source of truth for what gets drawn.

The dialog needs a QDialog to exist, so the behaviour that can be checked
without one - the report markup, and the shape of the split - is checked here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
FIT_SOURCE = (APP_DIR / "series_operations" / "fit_dialog.py").read_text(
    encoding="utf-8"
)


def _body(source: str, name: str) -> str:
    """Return one method's source, up to the next method at class level."""
    start = source.index(f"def {name}")
    end = source.index("\n    def ", start + 10)
    return source[start:end]


# ----------------------------------------------------------------------
# Fit and Preview are different acts
# ----------------------------------------------------------------------
def test_preview_no_longer_refits() -> None:
    """The bug: a starting guess was optimised away before it was drawn."""
    body = _body(FIT_SOURCE, "compute_results")

    assert "optimise=False" in body
    assert "self.on_fit()" not in body


def test_fit_optimises() -> None:
    body = _body(FIT_SOURCE, "on_fit")
    assert "optimise=True" in body


def test_fit_writes_the_optimum_into_the_parameter_table() -> None:
    """Otherwise Preview would immediately draw the old guess again."""
    body = _body(FIT_SOURCE, "on_fit")
    assert "_set_initial_params" in body


def test_fit_previews_what_it_found() -> None:
    body = _body(FIT_SOURCE, "on_fit")
    assert "self.preview()" in body


def test_both_paths_share_one_evaluation() -> None:
    """Metrics computed two ways would disagree sooner or later."""
    assert "def _evaluate(self, *, optimise: bool)" in FIT_SOURCE
    assert FIT_SOURCE.count("least_squares(") == 1


def test_the_fit_button_sits_in_the_shared_action_row() -> None:
    body = _body(FIT_SOURCE, "build_extra_action_buttons")

    # Its label and tooltip come from the catalogue, not from the call site.
    assert 'action_id="run_fit"' in body
    assert "action=self.on_fit" in body


def test_the_hook_exists_on_the_base() -> None:
    base = (APP_DIR / "series_operations" / "dialog_base.py").read_text(
        encoding="utf-8"
    )
    assert "def build_extra_action_buttons" in base
    assert "self.build_extra_action_buttons(action_row)" in base


# ----------------------------------------------------------------------
# The report
# ----------------------------------------------------------------------
def test_the_results_are_html() -> None:
    assert "RESULTS_ARE_HTML: bool = True" in FIT_SOURCE


def test_format_results_returns_the_table() -> None:
    body = _body(FIT_SOURCE, "format_results")
    assert "_results_html" in body


def test_the_report_has_a_row_per_parameter_with_its_error() -> None:
    body = _body(FIT_SOURCE, "_results_html")

    assert "Std. error" in body
    # The row label carries the parameter's meaning, not just its index.
    assert "_param_label(row)" in body
    assert "report_html.table(" in body


def test_the_report_covers_the_metrics() -> None:
    body = _body(FIT_SOURCE, "_results_html")

    for key in ("r2", "rmse", "ss_res", "aic", "bic"):
        assert key in body


def test_the_summary_values_are_escaped() -> None:
    """A table name with an angle bracket would truncate the report.

    The escaping moved into report_html.summary_table, which is the only
    place that turns a value into a cell.
    """
    from app.utils import report_html

    markup = report_html.summary_table([("Source", "a <b> table")])
    assert "&lt;b&gt;" in markup and "<b>" not in markup


def test_an_unknown_standard_error_is_blank_not_nan() -> None:
    """nan next to a fixed parameter reads as a failure, not as an absence."""
    body = _body(FIT_SOURCE, "_std_text")
    assert 'return ""' in body
    assert "isfinite" in body


def test_no_stray_browser_is_written_to() -> None:
    """The report goes to the shared results pane like every other dialog."""
    assert "_results_browser" not in FIT_SOURCE
    assert not re.search(r"def _set_fit_results_html", FIT_SOURCE)


# ----------------------------------------------------------------------
# What the parameters mean
# ----------------------------------------------------------------------
def _catalog() -> dict:
    """Return the model catalogue by calling the real method.

    Not ast.literal_eval: the defaults contain expressions (``2*np.pi``,
    ``[0.0]*6``) that only mean something when evaluated.  The method touches
    no widget, so an unbound call is enough.
    """
    from app.series_operations.fit_dialog import SeriesFitDialog

    return SeriesFitDialog.__dict__["_catalog_data"](object())


def _models(kind: str | None = None) -> list[dict]:
    return [
        model
        for models in _catalog().values()
        for model in models
        if kind is None or model.get("kind") == kind
    ]


def test_the_expression_box_shows_the_formula_not_a_helper_name() -> None:
    """``helpers.gaussian(x, p)`` said nothing about what p[2] was."""
    expressions = {model["name"]: model["expr"] for model in _models("expr")}

    assert expressions["Gaussian"] == "p[0]*exp(-(x - p[1])**2/(2*p[2]**2)) + p[3]"

    # Only the models needing a function the evaluator does not expose: the
    # Faddeeva function, and a per-point branch.
    helper_backed = sorted(
        name for name, expr in expressions.items() if "helpers." in expr
    )
    assert helper_backed == ["Asymmetric Gaussian", "Pseudo-Voigt", "Voigt"]


def test_every_expression_model_names_its_parameters() -> None:
    """A five-parameter peak is unusable if p[3] could be anything."""
    missing = [model["name"] for model in _models("expr") if not model.get("params")]
    assert missing == []


def test_the_names_match_the_parameter_count() -> None:
    """A legend that stops at p[2] of a five-parameter model is worse than none."""
    mismatched = {
        model["name"]: (len(model["params"]), len(model["p0"]))
        for model in _models()
        if model.get("params") and len(model["params"]) != len(model["p0"])
    }
    assert mismatched == {}


def test_the_inlined_formulas_still_compute_what_the_helpers_did() -> None:
    """The formulas were transcribed by hand; this is what proves them."""
    import numpy as np

    from app.series_operations.fit_dialog import (
        ModelHelpers,
        make_model_from_expression,
    )

    cases = {
        "Linear": (ModelHelpers.linear, [0.3, 1.7]),
        "Exp decay": (ModelHelpers.exp_decay, [2.0, 0.5, 0.1]),
        "Double exp decay": (ModelHelpers.double_exp_decay, [2.0, 0.5, 1.0, 0.1, 0.3]),
        "Saturating exp": (ModelHelpers.saturating_exp, [2.0, 0.4, 0.1]),
        "Logistic": (ModelHelpers.logistic, [3.0, 1.2, 0.5, 0.2]),
        "Gompertz": (ModelHelpers.gompertz, [3.0, 1.2, 0.5, 0.2]),
        "Sinusoid": (ModelHelpers.sine, [1.5, 2.0, 0.3, 0.4]),
        "Cosine": (ModelHelpers.cosine, [1.5, 2.0, 0.3, 0.4]),
        "Damped sine": (ModelHelpers.damped_sine, [1.5, 2.0, 0.3, 0.2, 0.4]),
        "Michaelis-Menten": (ModelHelpers.michaelis_menten, [2.0, 1.0, 0.3]),
        "Hill": (ModelHelpers.hill, [2.0, 1.0, 2.0, 0.3]),
        "Power law": (ModelHelpers.power_law, [2.0, -1.0, 0.3]),
        "Gaussian": (ModelHelpers.gaussian, [2.0, 0.5, 1.2, 0.3]),
        "Lorentzian": (ModelHelpers.lorentzian, [2.0, 0.5, 1.2, 0.3]),
        "Double Gaussian": (
            ModelHelpers.double_gaussian,
            [2.0, -1.0, 1.0, 0.8, 1.5, 1.2, 0.3],
        ),
    }
    expressions = {model["name"]: model["expr"] for model in _models("expr")}
    x = np.linspace(0.5, 4.0, 25)

    for name, (helper, values) in cases.items():
        params = np.asarray(values, dtype=float)
        model = make_model_from_expression(expressions[name], params)
        assert model(x, params) == pytest.approx(helper(x, params)), name


def test_the_legend_is_shown_under_the_expression() -> None:
    assert "self._param_legend" in FIT_SOURCE
    assert "layout.addWidget(self._param_legend)" in FIT_SOURCE


def test_the_parameter_table_is_relabelled_when_the_model_changes() -> None:
    """Written every time, not only when the cell is missing."""
    body = _body(FIT_SOURCE, "_ensure_params_rows")

    assert "self._param_label(row)" in body
    assert "if self._params_table.item(row, 0) is None" not in body


def test_the_report_carries_the_expression() -> None:
    body = _body(FIT_SOURCE, "_results_html")
    assert '"Expression"' in body
