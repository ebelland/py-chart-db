"""The declarative parameter spec: coercion, visibility, and round-tripping.

The spec half is pure data, so it tests without a window server. The form half
needs Qt but not a display, and is exercised through the outlier dialog's real
declaration rather than a toy one - a spec that works only on examples written
for it is not evidence of anything.
"""

from __future__ import annotations

import pytest

from app.series_operations.parameter_spec import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    IntParam,
    TextParam,
    coerce_all,
    defaults,
    visible_names,
)


# ----------------------------------------------------------------------
# Coercion
# ----------------------------------------------------------------------

def test_a_float_is_clamped_to_its_range() -> None:
    param = FloatParam("t", "T", default_value=3.0, minimum=0.1, maximum=30.0)
    assert param.coerce(99.0) == 30.0
    assert param.coerce(-5.0) == 0.1
    assert param.coerce(7.5) == 7.5


def test_unparseable_input_falls_back_to_the_default() -> None:
    param = FloatParam("t", "T", default_value=3.0, minimum=0.1, maximum=30.0)
    assert param.coerce("not a number") == 3.0
    assert param.coerce(None) == 3.0


@pytest.mark.parametrize("given, expected", [(4, 5), (10, 11), (11, 11), (3, 3)])
def test_an_odd_only_int_rounds_up_to_odd(given, expected) -> None:
    """Up, not down: these are window lengths, and rounding down narrows a
    window the user just widened."""
    param = IntParam("w", "W", default_value=11, minimum=3, maximum=501, odd_only=True)
    assert param.coerce(given) == expected


def test_an_odd_only_int_stays_inside_its_range_at_the_top() -> None:
    """Rounding up must not step past the maximum."""
    param = IntParam("w", "W", default_value=5, minimum=3, maximum=10, odd_only=True)
    assert param.coerce(10) <= 10
    assert param.coerce(10) % 2 == 1


def test_a_choice_outside_the_list_falls_back_to_the_default() -> None:
    param = ChoiceParam("m", "M", choices=("a", "b"))
    assert param.coerce("c") == "a"
    assert param.coerce("b") == "b"


def test_a_choice_can_separate_its_label_from_its_value() -> None:
    param = ChoiceParam("m", "M", choices=(("Rolling median", "rolling"),))
    assert param.labelled_choices() == (("Rolling median", "rolling"),)
    assert param.default == "rolling"


def test_bool_and_text_coerce_predictably() -> None:
    assert BoolParam("b", "B").coerce("anything") is True
    assert BoolParam("b", "B").coerce(0) is False
    assert TextParam("t", "T").coerce(None) == ""
    assert TextParam("t", "T").coerce(12) == "12"


# ----------------------------------------------------------------------
# Visibility
# ----------------------------------------------------------------------

PARAMS = (
    FloatParam("threshold", "Threshold", default_value=3.0,
               visible_for={"model": ("zscore", "rolling")}),
    IntParam("window", "Window", default_value=11,
             visible_for={"model": ("rolling",)}),
    FloatParam("always", "Always", default_value=1.0),
)


@pytest.mark.parametrize("model, expected", [
    ("zscore", ("threshold", "always")),
    ("rolling", ("threshold", "window", "always")),
    ("iqr", ("always",)),
])
def test_visibility_follows_the_declared_rules(model, expected) -> None:
    assert visible_names(PARAMS, {"model": model}) == expected


def test_a_rule_naming_an_absent_value_hides_the_row() -> None:
    """Better to hide than to show a control that cannot apply."""
    assert "threshold" not in visible_names(PARAMS, {})


def test_a_parameter_with_no_rule_is_always_visible() -> None:
    assert "always" in visible_names(PARAMS, {})


# ----------------------------------------------------------------------
# Defaults and forward compatibility
# ----------------------------------------------------------------------

def test_defaults_covers_every_declared_name() -> None:
    assert set(defaults(PARAMS)) == {"threshold", "window", "always"}


def test_state_saved_before_a_parameter_existed_still_loads() -> None:
    """This is what makes it safe to add a parameter to a shipped operation."""
    restored = coerce_all(PARAMS, {"threshold": 5.0})
    assert restored["threshold"] == 5.0
    assert restored["window"] == 11, "missing name falls back to its default"


def test_coerce_all_repairs_a_corrupted_saved_value() -> None:
    assert coerce_all(PARAMS, {"window": "twelve"})["window"] == 11


# ----------------------------------------------------------------------
# The form, against a real declaration
# ----------------------------------------------------------------------

def test_the_outlier_declaration_reproduces_its_old_visibility(qapp) -> None:
    from PySide6.QtWidgets import QComboBox, QWidget

    from app.series_operations.parameter_form import ParameterForm
    from app.series_operations.outlier_dialog import (
        OUTLIER_IQR,
        OUTLIER_MAD,
        OUTLIER_ROLLING,
        OUTLIER_ZSCORE,
        SeriesOutlierDialog,
    )

    host = QWidget()
    combo = QComboBox(host)
    combo.addItems([OUTLIER_ZSCORE, OUTLIER_IQR, OUTLIER_MAD, OUTLIER_ROLLING])
    form = ParameterForm(
        SeriesOutlierDialog.PARAMS,
        host,
        context=lambda: {"model": combo.currentText()},
    )
    host.show()

    def visible() -> set[str]:
        return {
            name
            for name in ("threshold", "iqr_factor", "window")
            if form.widget_for(name).isVisible()
        }

    # The four cases the hand-written _refresh_visibility used to encode.
    expected = {
        OUTLIER_ZSCORE: {"threshold"},
        OUTLIER_IQR: {"iqr_factor"},
        OUTLIER_MAD: {"threshold"},
        OUTLIER_ROLLING: {"threshold", "window"},
    }
    for model, rows in expected.items():
        combo.setCurrentIndex(combo.findText(model))
        form.refresh_visibility()
        assert visible() == rows, model


def test_hidden_parameters_are_still_readable(qapp) -> None:
    """An operation reads params by name whatever the model is, so a hidden
    control must not vanish from the values dict."""
    from PySide6.QtWidgets import QComboBox, QWidget

    from app.series_operations.parameter_form import ParameterForm
    from app.series_operations.outlier_dialog import (
        OUTLIER_IQR,
        SeriesOutlierDialog,
    )

    host = QWidget()
    combo = QComboBox(host)
    combo.addItem(OUTLIER_IQR)
    form = ParameterForm(
        SeriesOutlierDialog.PARAMS, host, context=lambda: {"model": OUTLIER_IQR}
    )
    assert set(form.values()) == {"threshold", "iqr_factor", "window"}


def test_the_form_round_trips_its_values(qapp) -> None:
    from PySide6.QtWidgets import QWidget

    from app.series_operations.parameter_form import ParameterForm
    from app.series_operations.outlier_dialog import SeriesOutlierDialog

    form = ParameterForm(SeriesOutlierDialog.PARAMS, QWidget())
    form.set_values({"threshold": 7.5, "window": 20})
    values = form.values()
    assert values["threshold"] == 7.5
    assert values["window"] == 21, "odd_only applies on restore too"


def test_setting_an_unknown_name_is_ignored(qapp) -> None:
    """A saved state from a different operation must not raise."""
    from PySide6.QtWidgets import QWidget

    from app.series_operations.parameter_form import ParameterForm
    from app.series_operations.outlier_dialog import SeriesOutlierDialog

    form = ParameterForm(SeriesOutlierDialog.PARAMS, QWidget())
    form.set_values({"not_a_parameter": 1})
    assert form.values()["threshold"] == 3.0
