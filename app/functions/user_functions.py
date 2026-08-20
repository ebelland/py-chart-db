from __future__ import annotations

import numpy as np

from app.functions.functions import base_function


class sample_user_function(base_function):
    name = "Sample user linear"
    category = "User functions"
    description = "Example user-defined linear function. Edit or replace this class."
    p0 = [0.0, 1.0]
    params = ["intercept", "slope"]

    @staticmethod
    def execute(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x
