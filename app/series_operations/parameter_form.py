"""Build a live form from a parameter declaration, and read it back.

The Qt half of ``parameter_spec``.  Kept separate so the declarations stay
importable and testable without a window server, which is most of what makes
them worth having.

One class, three jobs: build the widgets in declaration order, hand back the
current values by name, and show or hide each row as the declared
``visible_for`` rules dictate.  Every widget's change signal is routed to a
single callback, because the thing every existing dialog does with it is call
``refresh_results`` - and connecting eight signals by hand is exactly the step
that gets forgotten when a ninth control is added.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from app.series_operations.parameter_spec import (
    BoolParam,
    ChoiceParam,
    FloatParam,
    IntParam,
    Param,
    TextParam,
)
from app.utils.i18n import _


class ParameterForm:
    """A QFormLayout built from ``Param`` declarations."""

    def __init__(
        self,
        params: Sequence[Param],
        parent: QWidget,
        *,
        on_change: Callable[[], None] | None = None,
        context: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        """``context`` supplies values this form does not own.

        Visibility routinely depends on the model combo, which belongs to the
        base dialog rather than to the parameter set - so without this, a rule
        like ``visible_for={"model": ...}`` would look up a name that is never
        present and hide the row forever.
        """
        self._params = tuple(params)
        self._parent = parent
        self._on_change = on_change
        self._context = context
        self._widgets: dict[str, QWidget] = {}

        self.widget = QWidget(parent)
        self.layout = QFormLayout(self.widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        for param in self._params:
            widget = self._build_widget(param)
            self._widgets[param.name] = widget
            self.layout.addRow(_(param.label), widget)

        self.refresh_visibility()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def _build_widget(self, param: Param) -> QWidget:
        """Return the control for one parameter, already wired."""
        if isinstance(param, ChoiceParam):
            widget: QWidget = QComboBox(self.widget)
            for label, value in param.labelled_choices():
                widget.addItem(_(str(label)), value)
            index = widget.findData(param.default)
            widget.setCurrentIndex(max(0, index))
            widget.currentIndexChanged.connect(self._changed)

        elif isinstance(param, IntParam):
            widget = QSpinBox(self.widget)
            widget.setRange(param.minimum, param.maximum)
            # An odd-only control stepping by one would let the user land on
            # an even value that coerce() then silently moves. Step by two so
            # the control cannot express what the parameter refuses.
            widget.setSingleStep(param.step * 2 if param.odd_only else param.step)
            widget.setValue(param.default)
            if param.suffix:
                widget.setSuffix(_(param.suffix))
            widget.valueChanged.connect(self._changed)

        elif isinstance(param, FloatParam):
            widget = QDoubleSpinBox(self.widget)
            widget.setRange(param.minimum, param.maximum)
            widget.setDecimals(param.decimals)
            widget.setSingleStep(param.step)
            widget.setValue(param.default)
            if param.suffix:
                widget.setSuffix(_(param.suffix))
            widget.valueChanged.connect(self._changed)

        elif isinstance(param, BoolParam):
            widget = QCheckBox("", self.widget)
            widget.setChecked(param.default)
            widget.toggled.connect(self._changed)

        elif isinstance(param, TextParam):
            widget = QLineEdit(self.widget)
            widget.setText(param.default)
            if param.placeholder:
                widget.setPlaceholderText(_(param.placeholder))
            widget.editingFinished.connect(self._changed)

        else:
            raise TypeError(f"Unsupported parameter type: {type(param).__name__}")

        if param.tooltip:
            widget.setToolTip(_(param.tooltip))
        return widget

    def _changed(self, *_args: Any) -> None:
        """Re-evaluate visibility, then tell the dialog something moved."""
        self.refresh_visibility()
        if self._on_change is not None:
            self._on_change()

    # ------------------------------------------------------------------
    # Reading and writing
    # ------------------------------------------------------------------

    def values(self) -> dict[str, Any]:
        """Return every parameter's current value, coerced, keyed by name.

        Every parameter, including the hidden ones: a hidden control still
        holds a value, and an operation that reads ``params["window"]``
        regardless of which model is selected must not get a KeyError for it.
        """
        result: dict[str, Any] = {}
        for param in self._params:
            widget = self._widgets[param.name]
            result[param.name] = param.coerce(self._read_widget(widget))
        return result

    @staticmethod
    def _read_widget(widget: QWidget) -> Any:
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def set_values(self, values: Mapping[str, Any]) -> None:
        """Restore saved values, ignoring names this form does not have."""
        for param in self._params:
            if param.name not in values:
                continue
            widget = self._widgets[param.name]
            value = param.coerce(values[param.name])

            widget.blockSignals(True)
            try:
                if isinstance(widget, QComboBox):
                    widget.setCurrentIndex(max(0, widget.findData(value)))
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
            finally:
                widget.blockSignals(False)

        self.refresh_visibility()

    def widget_for(self, name: str) -> QWidget | None:
        """Return one parameter's control, for the rare hand-tuned case."""
        return self._widgets.get(name)

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def refresh_visibility(self) -> None:
        """Show or hide each row according to the declared rules."""
        current = dict(self.values())
        if self._context is not None:
            # Context does not override a real parameter of the same name: the
            # form's own control is the authority on its own value.
            for key, value in self._context().items():
                current.setdefault(key, value)

        for param in self._params:
            widget = self._widgets[param.name]
            visible = param.is_visible(current)
            widget.setVisible(visible)
            # The label is a separate widget, and hiding only the field leaves
            # a labelled empty row - which is what makes a hand-written
            # visibility method easy to get subtly wrong.
            label = self.layout.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
