"""Your own fit functions.

A class here appears in the fit dialog and in the function plotter with no
registration of any kind: both scan this folder.  The example below shows the
whole contract - there is nothing else to implement.

``expression`` is not decoration.  It is the only thing in the dialog that says
what the parameters mean before a fit has run, so a function without one is a
column of spin boxes labelled p[0], p[1], p[2].

``initial_guess`` is optional and worth writing.  Every optimiser finds the
bottom of the valley it was dropped into, so on anything with more than one
valley the starting point decides the answer.  Return None when the data does
not let you read one off; the application then searches the parameter space at
random instead, which works but is slower and less certain.
"""
from __future__ import annotations

import numpy as np

from app.functions.base import base_function


class sample_user_function(base_function):
    name = "Sample user linear"
    category = "User functions"
    description = "Example user-defined linear function. Edit or replace this class."
    expression = "<b>Sample user linear</b><br>y = b + m x"
    p0 = [0.0, 1.0]
    params = ["intercept", "slope"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x

    @staticmethod
    def initial_guess(x: np.ndarray, y: np.ndarray) -> list[float] | None:
        """Read the intercept and slope straight off the data."""
        if x.size < 2:
            return None
        slope, intercept = np.polyfit(x, y, 1)
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            return None
        return [float(intercept), float(slope)]
