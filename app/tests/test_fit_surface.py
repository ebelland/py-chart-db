"""Fitting a surface: Z = f(X, Y) through the Fit dialog.

The dialog fitted only Y = f(X) until the function library grew a 3D half
(``app/functions/functions_3d.py``). Two stubs held the old assumption -
``_is_2d_fit`` returned False outright and ``_selected_column_names`` returned
no second input - so picking a 3D function fed the surface a 1-D X and it
raised before the optimiser ever ran.

What matters here is the shape of the input, and it is why these tests build
real data rather than asserting on the source: a surface model is called with
an ``(N, 2)`` array of X/Y pairs, and every 1-D habit on the way in - sorting
by x, averaging repeated x - is wrong for it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.sqlite_repo import SqliteRepo
from app.scanners.functions_scanner import FunctionScanner
from app.series_operations.fit_dialog import SeriesFitDialog

#: The plane the fixtures plant, as z = C + a*x + b*y.
OFFSET, X_SLOPE, Y_SLOPE = 3.0, 0.5, -1.25


def _payload(name: str) -> dict:
    """Return one discovered function's payload by its English name."""
    catalog = FunctionScanner().catalog()
    return next(
        dict(payload)
        for payloads in catalog.values()
        for payload in payloads
        if payload.get("name") == name
    )


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    for path in (
        tmp_db_path,
        tmp_db_path.with_suffix(".dhub-wal"),
        tmp_db_path.with_suffix(".dhub-shm"),
    ):
        path.unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    axis = np.linspace(-4.0, 4.0, 25)
    x_grid, y_grid = np.meshgrid(axis, axis)
    built.import_dataframe(
        pd.DataFrame(
            {
                "x": x_grid.ravel(),
                "y": y_grid.ravel(),
                "z": OFFSET + X_SLOPE * x_grid.ravel() + Y_SLOPE * y_grid.ravel(),
            }
        ),
        table_name="surf",
        normalize_columns=False,
    )
    yield built
    built.close()


@pytest.fixture
def figure_id(repo: SqliteRepo) -> int:
    figure_id = repo.create_figure_descriptor(name="fig")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Surface Plot",
        title="surface",
        x_label="",
        y_label="",
        options={"projection": "3d"},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="SURF",
        sql_query="SELECT x, y, z FROM surf",
        roles={"x": "x", "y": "y", "z": "z"},
        style={},
    )
    return figure_id


@pytest.fixture
def dialog(qapp, repo: SqliteRepo, figure_id: int) -> SeriesFitDialog:
    built = SeriesFitDialog(repo=repo, figure_id=figure_id)
    built.series_selector.select_all_series()
    built._residual_chart_check.setChecked(False)
    built._fit_vs_measured_chart_check.setChecked(False)
    built._apply_model_choice_from_payload(_payload("Plane"))
    return built


# ----------------------------------------------------------------------
# Which kind of fit this is
# ----------------------------------------------------------------------
def test_the_function_decides_the_dimension_not_the_data(
    dialog: SeriesFitDialog,
) -> None:
    """The same series can source either kind, so the model is what is asked."""
    assert dialog._is_2d_fit() is True

    dialog._apply_model_choice_from_payload(_payload("Linear"))
    assert dialog._is_2d_fit() is False


def test_the_third_column_is_named_only_for_a_surface(
    dialog: SeriesFitDialog,
) -> None:
    dialog._load_fit_data()
    x_col, x2_col, target_col = dialog._selected_column_names()

    assert (x_col, x2_col, target_col) == ("x", "y", "z")


# ----------------------------------------------------------------------
# The shape of the input
# ----------------------------------------------------------------------
def test_the_loader_hands_the_model_xy_pairs(dialog: SeriesFitDialog) -> None:
    """The bug this feature fixes: a surface was handed a 1-D X and raised."""
    inputs, target, clean = dialog._load_fit_data()

    assert inputs.ndim == 2 and inputs.shape[1] == 2
    assert inputs.shape[0] == target.shape[0] == 625
    assert list(clean.columns) == ["x", "y", "target"]


def test_points_sharing_an_x_all_survive(dialog: SeriesFitDialog) -> None:
    """Why prepare_input_xy is skipped: its repairs are 1-D ideas.

    A grid is 25 distinct x values and 625 points. Averaging repeated x - what
    the 1-D path does - would collapse it to 25, and sorting by x would
    scramble which y each z belonged to.
    """
    inputs, _target, _clean = dialog._load_fit_data()

    assert np.unique(inputs[:, 0]).size == 25
    assert inputs.shape[0] == 625


def test_a_non_finite_row_is_dropped_rather_than_fitted(
    qapp, repo: SqliteRepo
) -> None:
    """A NULL z is a hole in the measurement, not a zero-height point."""
    figure_id = repo.create_figure_descriptor(name="holes")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Surface Plot",
        title="holes",
        x_label="",
        y_label="",
        options={"projection": "3d"},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="GAPPY",
        sql_query=(
            "SELECT x, y, CASE WHEN rowid <= 4 THEN NULL ELSE z END AS z "
            "FROM surf"
        ),
        roles={"x": "x", "y": "y", "z": "z"},
        style={},
    )

    built = SeriesFitDialog(repo=repo, figure_id=figure_id)
    built.series_selector.select_all_series()
    built._apply_model_choice_from_payload(_payload("Plane"))

    inputs, target, _clean = built._load_fit_data()

    assert inputs.shape[0] == target.shape[0] == 621
    assert np.isfinite(target).all()


# ----------------------------------------------------------------------
# The fit itself
# ----------------------------------------------------------------------
def test_a_plane_fit_recovers_the_planted_coefficients(
    dialog: SeriesFitDialog,
) -> None:
    dialog._evaluate(optimise=True)
    result = dialog._last_result

    assert result is not None
    assert result.fit_mode == "2D"
    assert result.params == pytest.approx([OFFSET, X_SLOPE, Y_SLOPE], abs=1e-6)
    assert result.metrics["r2"] == pytest.approx(1.0, abs=1e-9)


def test_the_output_frame_carries_the_fitted_z(dialog: SeriesFitDialog) -> None:
    """z_fit is what the result series reads; without it the chart is empty."""
    dialog._evaluate(optimise=True)
    frame = dialog._last_result.frame

    assert {"x", "y", "z", "z_fit"} <= set(frame.columns)
    assert frame["z_fit"].to_numpy() == pytest.approx(
        frame["z"].to_numpy(), abs=1e-6
    )


# ----------------------------------------------------------------------
# What gets drawn
# ----------------------------------------------------------------------
def test_a_fitted_surface_is_drawn_as_a_surface(dialog: SeriesFitDialog) -> None:
    """Three roles and the surface flag - a dashed line would mean nothing."""
    dialog._evaluate(optimise=True)
    spec = dialog.result_series_spec(0, "fit_out", dialog._last_result)

    assert spec.roles == {"x": "x", "y": "y", "z": "z"}
    assert "z_fit AS z" in spec.sql_query
    assert spec.style["surface"] is True
    assert spec.style["fit_mode"] == "2D"
    assert "linestyle" not in spec.style


def test_a_curve_fit_still_draws_a_dashed_line(
    dialog: SeriesFitDialog,
) -> None:
    """The 1-D path is unchanged, which is the other half of the contract."""
    dialog._apply_model_choice_from_payload(_payload("Linear"))
    dialog._evaluate(optimise=True)
    spec = dialog.result_series_spec(0, "fit_out", dialog._last_result)

    assert spec.roles == {"x": "x", "y": "y"}
    assert spec.style["fit_mode"] == "1D"
    assert spec.style["linestyle"] == "--"
    assert "surface" not in spec.style


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------
def test_a_series_with_no_z_says_so(
    qapp, repo: SqliteRepo
) -> None:
    """It used to reach frame[""] and report a KeyError instead.

    applogger.error does not raise by default, so the loader has to ask for
    the raise - otherwise the message naming Z is shown and the code carries
    on to fail somewhere that names nothing.
    """
    figure_id = repo.create_figure_descriptor(name="flat")
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="flat",
        x_label="",
        y_label="",
        options={},
    )
    repo.create_series_descriptor(
        axis_id=axis_id,
        series_index=0,
        name="XY",
        sql_query="SELECT x, y FROM surf",
        roles={"x": "x", "y": "y"},
        style={},
    )

    built = SeriesFitDialog(repo=repo, figure_id=figure_id)
    built.series_selector.select_all_series()
    built._apply_model_choice_from_payload(_payload("Plane"))

    with pytest.raises(Exception, match="X, Y and Z"):
        built._load_fit_data()
