"""Fit-function scanner for SeriesFitDialog.

The fit dialog is intentionally decoupled from concrete model definitions.
Built-in functions live in ``app/functions/functions.py`` and user functions
live in ``app/functions/user_functions.py``.  This scanner uses the generic
``class_discovery`` helpers to find classes that directly inherit from
``base_function`` and then loads their ``execute(x, p)`` method on demand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

import numpy as np

from app.logs.logger import applogger
from app.scanners.class_discovery import discover_classes, import_class_from_discovery_entry


@dataclass(frozen=True, slots=True)
class FitFunctionSpec:
    """Metadata and loader information for one discovered fit function."""

    name: str
    category: str
    description: str
    expression: str
    p0: tuple[float, ...]
    params: tuple[str, ...]
    discovery_entry: dict[str, Any]
    input_dimensions: int = 1

    @property
    def class_name(self) -> str:
        return str(self.discovery_entry.get("name", ""))

    @property
    def path(self) -> str:
        return str(self.discovery_entry.get("path", ""))

    def as_catalog_payload(self) -> dict[str, Any]:
        """Return the payload stored on a QTreeWidgetItem."""
        return {
            "name": self.name,
            "kind": "function",  # internal dialog dispatch only, not a base_function field
            "category": self.category,
            "description": self.description,
            "expression": self.expression,
            "p0": list(self.p0),
            "params": list(self.params),
            "discovery_entry": dict(self.discovery_entry),
            "function_class": self.class_name,
            "path": self.path,
            "input_dimensions": self.input_dimensions,
            "dimensions": self.input_dimensions,
        }


class FunctionScanner:
    """Discover and instantiate fit functions using class_discovery helpers.

    Function classes must directly inherit from ``base_function`` and expose:

    - ``name``: label shown in the model tree
    - ``category``: top-level tree category
    - ``description``: short human-readable purpose
    - ``expression``: HTML or plain formula displayed below Parameters/Expression
    - ``params``: list of parameter names
    - ``p0``: list of numeric initial values
    - ``execute(x, p)``: static/class method used by least-squares fitting
    """

    BASE_CLASS_NAME: Final[str] = "base_function"
    DEFAULT_CATEGORY: Final[str] = "Functions"

    def __init__(self, *, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent.joinpath("..", "functions")
        self._specs: list[FitFunctionSpec] | None = None

    def refresh(self) -> list[FitFunctionSpec]:
        """Force a rescan and return discovered specs."""
        self._specs = self._discover()
        return list(self._specs)

    def specs(self) -> list[FitFunctionSpec]:
        """Return cached specs, scanning if needed."""
        if self._specs is None:
            self._specs = self._discover()
        return list(self._specs)

    def catalog(self) -> dict[str, list[dict[str, Any]]]:
        """Return discovered functions grouped for the SeriesFitDialog tree."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for spec in self.specs():
            grouped.setdefault(spec.category, []).append(spec.as_catalog_payload())

        for models in grouped.values():
            models.sort(key=lambda item: str(item.get("name", "")).lower())

        return dict(sorted(grouped.items(), key=lambda item: item[0].lower()))

    def load_class(self, payload: dict[str, Any]) -> Any:
        """Return the function class a tree payload describes.

        The callable from :meth:`make_model` is enough to evaluate a function
        and not enough to ask it anything: the starting-point estimator is a
        second staticmethod on the class (``initial_guess``), so the caller
        that wants one needs the class rather than the closure.
        """
        entry = payload.get("discovery_entry")
        if not isinstance(entry, dict):
            raise ValueError("Function payload has no discovery_entry.")

        cls = import_class_from_discovery_entry(entry, module_prefix="_fit_function")
        if cls is None:
            raise ValueError(f"Could not load function class from entry: {entry!r}")
        return cls

    def make_model(self, payload: dict[str, Any]) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Build a fit callable from a tree payload."""
        cls = self.load_class(payload)

        execute = getattr(cls, "execute", None)
        if not callable(execute):
            raise TypeError(f"Discovered function {cls!r} has no callable execute(x, p).")

        input_dimensions = self._payload_input_dimensions(payload, cls)

        def model(x_or_xy: np.ndarray, p: np.ndarray) -> np.ndarray:
            inputs = np.asarray(x_or_xy, dtype=float)
            if input_dimensions < 2:
                inputs = self._primary_x(inputs)
            elif inputs.ndim != 2 or inputs.shape[1] < 2:
                raise ValueError(
                    "3D surface functions require an (N, 2) array of X/Y coordinates."
                )
            params = np.asarray(p, dtype=float).reshape(-1)
            return np.asarray(execute(inputs, params), dtype=float)

        return model

    def _discover(self) -> list[FitFunctionSpec]:
        entries = discover_classes(
            root=self.root,
            base_class_name=self.BASE_CLASS_NAME,
            value_attr="name",
            string_attrs=("category", "description", "expression"),
            string_list_attrs=("params",),
            require_value_attr=False,
            skip_init=True,
        )

        specs: list[FitFunctionSpec] = []
        seen: set[tuple[str, str]] = set()

        for entry in entries:
            cls_name = str(entry.get("name", ""))
            path = str(entry.get("path", ""))
            key = (path, cls_name)
            if key in seen:
                continue
            seen.add(key)

            cls = import_class_from_discovery_entry(entry, module_prefix="_fit_function_meta")
            if cls is None:
                continue

            execute = getattr(cls, "execute", None)
            if not callable(execute):
                applogger.warning("Fit function %s in %s has no execute(x, p); skipped.", cls_name, path)
                continue

            display_name = str(entry.get("value") or getattr(cls, "name", cls_name) or cls_name)
            category = str(entry.get("category") or getattr(cls, "category", self.DEFAULT_CATEGORY) or self.DEFAULT_CATEGORY)
            description = str(entry.get("description") or getattr(cls, "description", "") or "")
            expression = str(entry.get("expression") or getattr(cls, "expression", "") or "")
            params = self._string_tuple(entry.get("params") or getattr(cls, "params", []))
            p0 = self._float_tuple(getattr(cls, "p0", []), fallback_length=len(params))
            input_dimensions = self._class_input_dimensions(cls, category, path)

            if params and len(params) != len(p0):
                applogger.warning(
                    "Fit function %s has %d params but %d initial values.",
                    display_name,
                    len(params),
                    len(p0),
                )

            specs.append(
                FitFunctionSpec(
                    name=display_name,
                    category=category,
                    description=description,
                    expression=expression,
                    p0=p0,
                    params=params,
                    discovery_entry=dict(entry),
                    input_dimensions=input_dimensions,
                )
            )

        specs.sort(key=lambda item: (item.category.lower(), item.name.lower()))
        return specs

    @staticmethod
    def _class_input_dimensions(cls: Any, category: str, path: str) -> int:
        """Return 2 for Z=f(X,Y) surfaces and 1 for ordinary Y=f(X) curves."""
        declared = getattr(cls, "input_dimensions", getattr(cls, "dimensions", None))
        try:
            if declared is not None:
                return max(1, int(declared))
        except (TypeError, ValueError):
            pass
        if category.strip().lower().startswith("3d ") or "functions_3d" in path.lower():
            return 2
        return 1

    @classmethod
    def _payload_input_dimensions(cls, payload: dict[str, Any], function_class: Any) -> int:
        declared = payload.get("input_dimensions", payload.get("dimensions"))
        try:
            if declared is not None:
                return max(1, int(declared))
        except (TypeError, ValueError):
            pass
        return cls._class_input_dimensions(
            function_class,
            str(payload.get("category", "")),
            str(payload.get("path", "")),
        )

    @staticmethod
    def _primary_x(value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 2:
            return arr[:, 0]
        return arr

    @staticmethod
    def _string_tuple(value: Any) -> tuple[str, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
        return ()

    @staticmethod
    def _float_tuple(value: Any, *, fallback_length: int = 0) -> tuple[float, ...]:
        try:
            values = tuple(float(item) for item in value)
        except Exception:
            values = ()
        if values:
            return values
        if fallback_length > 0:
            return tuple(1.0 for _ in range(fallback_length))
        return (1.0, 1.0)
