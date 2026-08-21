"""Declare an operation's parameters as data, and let the base build the UI.

The eight operation dialogs run 775 to 1705 lines each against a 964-line base,
and most of that is the same four jobs written eight times: create a widget,
put it in a form with a label, connect its change signal to a refresh, and read
its value back into a dict.  Repeating that has a cost beyond the line count -
every copy is a place to forget the tooltip, to connect the wrong signal, or to
let the parameter dict drift from the widget that fills it.

So a parameter says what it is:

    PARAMS = (
        ChoiceParam("model", "Model", choices=("Z-score", "IQR")),
        FloatParam("threshold", "Threshold", default=3.0, minimum=0.1,
                   maximum=30.0, visible_for={"model": ("Z-score",)}),
        IntParam("window", "Window size", default=11, minimum=3, odd_only=True),
    )

and the base does the rest: builds the form in declaration order, wires every
change to one refresh, reads them all back by name, and shows or hides each row
as ``visible_for`` dictates.

Deliberately not a general form framework.  It covers the shapes these dialogs
actually use, and an operation with a genuinely unusual control still overrides
``build_parameter_selector`` and builds it by hand - the spec is there to
remove repetition, not to forbid the exceptional case.

No Qt imports: the declarations are plain data and have to stay testable
without a window server.  The widget building lives in ``parameter_form``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Param:
    """One declared parameter.

    ``visible_for`` maps another parameter's name to the values of that
    parameter for which this row should be shown.  It is how the existing
    dialogs behave - the outlier dialog shows a window size only for the
    rolling method - expressed as data rather than as a hand-written
    ``_refresh_visibility``.  An empty mapping means always visible.
    """

    name: str
    label: str
    tooltip: str = ""
    visible_for: Mapping[str, Sequence[Any]] = field(default_factory=dict)

    def is_visible(self, values: Mapping[str, Any]) -> bool:
        """Say whether this row should be shown for the current values."""
        for other, allowed in self.visible_for.items():
            if values.get(other) not in tuple(allowed):
                return False
        return True

    @property
    def default(self) -> Any:
        raise NotImplementedError

    def coerce(self, value: Any) -> Any:
        """Return ``value`` in this parameter's own type, or the default."""
        return value


@dataclass(frozen=True, slots=True)
class FloatParam(Param):
    """A real-valued parameter, shown as a spin box."""

    default_value: float = 0.0
    minimum: float = -1.0e9
    maximum: float = 1.0e9
    decimals: int = 3
    step: float = 0.1
    suffix: str = ""

    @property
    def default(self) -> float:
        return float(self.default_value)

    def coerce(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return self.default
        return min(max(number, self.minimum), self.maximum)


@dataclass(frozen=True, slots=True)
class IntParam(Param):
    """An integer parameter, shown as a spin box.

    ``odd_only`` exists because several of these feed window lengths, and
    ``savgol_filter`` and ``medfilt`` both reject an even window with an
    exception from inside SciPy rather than a message about the control the
    user just moved.
    """

    default_value: int = 0
    minimum: int = -1_000_000
    maximum: int = 1_000_000
    step: int = 1
    odd_only: bool = False
    suffix: str = ""

    @property
    def default(self) -> int:
        return self.coerce(self.default_value)

    def coerce(self, value: Any) -> int:
        try:
            number = int(round(float(value)))
        except (TypeError, ValueError):
            number = int(self.default_value)
        number = min(max(number, self.minimum), self.maximum)
        if self.odd_only and number % 2 == 0:
            # Up, not down: these are window lengths, and rounding 4 down to 3
            # silently narrows a window the user widened.
            number = min(number + 1, self.maximum)
            if number % 2 == 0:
                # Stepping up hit an even maximum, so the only odd value in
                # range is one below it.
                number -= 1
        return number


@dataclass(frozen=True, slots=True)
class ChoiceParam(Param):
    """A fixed set of options, shown as a combo box.

    ``choices`` may be plain strings, or (label, value) pairs when what the
    user reads should differ from what the operation receives.
    """

    choices: Sequence[Any] = ()
    default_value: Any = None

    def labelled_choices(self) -> tuple[tuple[str, Any], ...]:
        """Return every choice as a (label, value) pair."""
        pairs: list[tuple[str, Any]] = []
        for choice in self.choices:
            if isinstance(choice, tuple) and len(choice) == 2:
                pairs.append((str(choice[0]), choice[1]))
            else:
                pairs.append((str(choice), choice))
        return tuple(pairs)

    @property
    def default(self) -> Any:
        if self.default_value is not None:
            return self.default_value
        pairs = self.labelled_choices()
        return pairs[0][1] if pairs else None

    def coerce(self, value: Any) -> Any:
        values = [pair[1] for pair in self.labelled_choices()]
        return value if value in values else self.default


@dataclass(frozen=True, slots=True)
class BoolParam(Param):
    """An on/off parameter, shown as a check box."""

    default_value: bool = False

    @property
    def default(self) -> bool:
        return bool(self.default_value)

    def coerce(self, value: Any) -> bool:
        return bool(value)


@dataclass(frozen=True, slots=True)
class TextParam(Param):
    """A free-text parameter, shown as a line edit."""

    default_value: str = ""
    placeholder: str = ""

    @property
    def default(self) -> str:
        return str(self.default_value)

    def coerce(self, value: Any) -> str:
        return "" if value is None else str(value)


def defaults(params: Sequence[Param]) -> dict[str, Any]:
    """Return every parameter's default, keyed by name."""
    return {param.name: param.default for param in params}


def coerce_all(
    params: Sequence[Param],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Return ``values`` with every parameter coerced to its own type.

    Missing names fall back to the default, so a saved state written before a
    parameter existed still loads instead of raising - which is what makes it
    safe to add a parameter to a shipped operation.
    """
    return {
        param.name: param.coerce(values.get(param.name, param.default))
        for param in params
    }


def visible_names(
    params: Sequence[Param],
    values: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the names of the parameters visible for the current values."""
    return tuple(param.name for param in params if param.is_visible(values))
