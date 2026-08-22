"""Build a demo ``.dhub`` project showing what the application can do.

Shipped rather than kept with the tests, because the application offers it:
the first run of a fresh install has no database, and an empty one shows
nothing at all - so startup offers to build this instead (see
``app/utils/startup.py``).

Run it directly::

    python -m app.data.demo_project --output "Demo Project.dhub"

Everything is written through :class:`SqliteRepo`, not by hand-crafted SQL, so
the demo exercises the same code path the application uses and cannot drift
from the real schema.  The data is synthetic but shaped like real measurements
- a sensor network with drift and outages, a batch process with per-batch
spread, a calibration run with measurement uncertainty - because a demo made of
random noise shows the chart types without showing what they are *for*.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger

SEED = 20260101

# A calm, print-friendly style applied to every figure in the demo.
DEMO_STYLE = """
figure.facecolor: FBFBFD
axes.facecolor: FFFFFF
axes.edgecolor: C8CCD4
axes.linewidth: 1.0
axes.grid: True
axes.axisbelow: True
axes.titlesize: 13
axes.titleweight: 600
axes.labelcolor: 3C4250
axes.labelsize: 10
grid.color: E4E7EC
grid.linewidth: 0.8
xtick.color: 6B7280
ytick.color: 6B7280
xtick.labelsize: 9
ytick.labelsize: 9
legend.frameon: True
legend.framealpha: 0.9
legend.edgecolor: E4E7EC
legend.fontsize: 9
lines.linewidth: 1.8
font.size: 10
axes.prop_cycle: cycler('color', ['4C78A8', 'F58518', '54A24B', 'E45756', '72B7B2', 'B279A2'])
"""


@dataclass(slots=True)
class SeriesSpec:
    """One series to attach to an axis."""

    name: str
    sql: str
    roles: dict[str, str]
    style: dict[str, Any]


@dataclass(slots=True)
class FigureSpec:
    """One demo figure: a title, a chart type, and its series."""

    name: str
    chart_type: str
    title: str
    x_label: str
    y_label: str
    series: list[SeriesSpec]
    axis_options: dict[str, Any]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def _sensor_network(rng: np.random.Generator, hours: int = 24 * 6) -> pd.DataFrame:
    """Three sensors sampled hourly: daily cycle, drift, and one outage."""
    time = np.arange(hours, dtype=float)
    daily = np.sin(2.0 * np.pi * time / 24.0)

    rows: list[pd.DataFrame] = []
    for index, (sensor, offset, drift, noise) in enumerate(
        [("north", 18.0, 0.004, 0.35), ("south", 21.5, -0.002, 0.30), ("roof", 16.0, 0.010, 0.55)]
    ):
        values = offset + 4.0 * daily + drift * time + rng.normal(0.0, noise, hours)

        # One sensor loses power for a day and a half; the time-series renderer
        # draws that as a gap rather than a straight line through it.
        if sensor == "roof":
            values[24 * 3 : 24 * 3 + 20] = np.nan

        rows.append(
            pd.DataFrame(
                {
                    "hour": time,
                    "sensor": sensor,
                    "temperature": values,
                    "humidity": 55.0 + 8.0 * np.cos(2 * np.pi * time / 24.0 + index)
                    + rng.normal(0.0, 1.2, hours),
                }
            )
        )

    frame = pd.concat(rows, ignore_index=True)
    return frame.dropna(subset=["temperature"]).reset_index(drop=True)


def _calibration(rng: np.random.Generator, points: int = 12) -> pd.DataFrame:
    """A calibration curve with asymmetric, level-dependent uncertainty."""
    applied = np.linspace(0.0, 100.0, points)
    measured = 0.98 * applied + 1.2 + rng.normal(0.0, 0.6, points)

    # Uncertainty grows with the reading and is not symmetric: the instrument
    # under-reads more than it over-reads at the top of its range.
    scale = 0.4 + 0.02 * applied
    return pd.DataFrame(
        {
            "applied": applied,
            "measured": measured,
            "err_low": scale * rng.uniform(0.8, 1.4, points),
            "err_high": scale * rng.uniform(0.4, 0.9, points),
            "applied_err": np.full(points, 0.5),
        }
    )


def _batches(rng: np.random.Generator, per_batch: int = 220) -> pd.DataFrame:
    """Four production batches with different spread and one skewed batch."""
    rows: list[pd.DataFrame] = []
    for index, batch in enumerate(("A", "B", "C", "D")):
        if batch == "D":
            # A skewed batch is what makes a violin say something a box cannot.
            values = 48.0 + rng.gamma(shape=2.0, scale=2.4, size=per_batch)
        else:
            values = rng.normal(50.0 + 1.5 * index, 1.2 + 0.5 * index, per_batch)
        rows.append(pd.DataFrame({"batch": batch, "yield_pct": values}))
    return pd.concat(rows, ignore_index=True)


def _throughput(rng: np.random.Generator) -> pd.DataFrame:
    """Monthly throughput per line, with a measurement error per bar."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    base = np.array([120.0, 138.0, 131.0, 155.0, 168.0, 162.0])
    return pd.DataFrame(
        {
            "month": months,
            "throughput": base + rng.normal(0.0, 3.0, len(months)),
            "throughput_err": rng.uniform(3.0, 9.0, len(months)),
        }
    )


def _particles(rng: np.random.Generator, count: int = 4000) -> pd.DataFrame:
    """Two overlapping particle populations, for a multi-dataset histogram."""
    fine = rng.lognormal(mean=1.1, sigma=0.35, size=count)
    coarse = rng.lognormal(mean=2.0, sigma=0.30, size=count // 2)
    return pd.DataFrame(
        {
            "population": ["fine"] * count + ["coarse"] * (count // 2),
            "diameter_um": np.concatenate([fine, coarse]),
        }
    )


def _scatter_cloud(rng: np.random.Generator, count: int = 2500) -> pd.DataFrame:
    """A correlated cloud with a continuous colour and a size channel."""
    cloud = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.72], [0.72, 1.0]], size=count)
    energy = np.hypot(cloud[:, 0], cloud[:, 1])
    return pd.DataFrame(
        {
            "pressure": 40.0 + 6.0 * cloud[:, 0],
            "flow": 12.0 + 2.4 * cloud[:, 1],
            "energy": energy,
            "weight": 12.0 + 40.0 * (energy / energy.max()),
        }
    )


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _figure_specs() -> list[FigureSpec]:
    """Return every demo figure, in the order they appear as tabs."""
    line = {"linestyle": "-", "marker": "", "show_in_legend": True}

    return [
        FigureSpec(
            name="1 · Sensor network",
            chart_type="Time Series",
            title="Hourly temperature by sensor",
            x_label="hours since start",
            y_label="temperature (°C)",
            axis_options={
                "grid": True,
                # The rolling average would double the number of lines on a
                # chart that is already three series deep.
                "show_rolling": False,
                # Numeric x, so the gap threshold is a number of x units: the
                # roof sensor's outage reads as a gap instead of a straight
                # line drawn through it.
                "gap_threshold": 2.0,
            },
            series=[
                SeriesSpec(
                    name=sensor.capitalize(),
                    sql=(
                        "SELECT hour AS x, temperature AS y FROM sensor_readings "
                        f"WHERE sensor = '{sensor}' ORDER BY hour"
                    ),
                    roles={"x": "x", "y": "y"},
                    style=dict(line),
                )
                for sensor in ("north", "south", "roof")
            ],
        ),
        FigureSpec(
            name="2 · Calibration",
            chart_type="Scatter Plot",
            title="Calibration curve with measurement uncertainty",
            x_label="applied (units)",
            y_label="measured (units)",
            axis_options={"grid": True, "capsize": 3.0, "elinewidth": 1.1},
            series=[
                SeriesSpec(
                    name="Run 1",
                    sql=(
                        "SELECT applied AS x, measured AS y, "
                        "applied_err AS xerr, err_low AS yerr_low, err_high AS yerr_high "
                        "FROM calibration ORDER BY applied"
                    ),
                    roles={"x": "x", "y": "y"},
                    style={"marker": "o", "linestyle": "-", "alpha": 0.95},
                ),
            ],
        ),
        FigureSpec(
            name="3 · Process spread",
            chart_type="Violin Plot",
            title="Yield distribution per batch",
            x_label="batch",
            y_label="yield (%)",
            axis_options={
                "grid": True,
                "grid_axis": "y",
                "showmedians": True,
                "quantiles": "0.25, 0.75",
                "alpha": 0.75,
            },
            series=[
                SeriesSpec(
                    name=f"Batch {batch}",
                    sql=(
                        'SELECT batch AS "group", yield_pct AS value '
                        f"FROM batch_yields WHERE batch = '{batch}'"
                    ),
                    roles={"value": "value", "group": "group"},
                    style={},
                )
                for batch in ("A", "B", "C", "D")
            ],
        ),
        FigureSpec(
            name="4 · Process summary",
            chart_type="Box Plot",
            title="Yield summary per batch",
            x_label="batch",
            y_label="yield (%)",
            axis_options={"grid": True, "grid_axis": "y", "showmeans": True},
            series=[
                SeriesSpec(
                    name=f"Batch {batch}",
                    sql=(
                        'SELECT batch AS "group", yield_pct AS value '
                        f"FROM batch_yields WHERE batch = '{batch}'"
                    ),
                    roles={"value": "value", "group": "group"},
                    style={},
                )
                for batch in ("A", "B", "C", "D")
            ],
        ),
        FigureSpec(
            name="5 · Particle sizes",
            chart_type="Histogram",
            title="Particle diameter by population",
            x_label="diameter (µm)",
            y_label="",
            axis_options={"bins": 45, "alpha": 0.8, "grid": True, "grid_axis": "y"},
            series=[
                SeriesSpec(
                    name="Populations",
                    sql="SELECT population AS dataset, diameter_um AS value FROM particles",
                    roles={"value": "value", "dataset": "dataset"},
                    style={},
                ),
            ],
        ),
        FigureSpec(
            name="6 · Throughput",
            chart_type="Bar Chart",
            title="Monthly throughput",
            x_label="month",
            y_label="units / day",
            axis_options={"grid": True, "grid_axis": "y", "capsize": 4.0},
            series=[
                SeriesSpec(
                    name="Line 1",
                    sql=(
                        "SELECT month AS X, throughput AS Y, throughput_err AS YError "
                        "FROM throughput"
                    ),
                    roles={"X": "X", "Y": "Y"},
                    style={"alpha": 0.9},
                ),
            ],
        ),
        FigureSpec(
            name="7 · Operating envelope",
            chart_type="Scatter Plot",
            title="Flow against pressure, coloured by energy",
            x_label="pressure (bar)",
            y_label="flow (l/s)",
            axis_options={"grid": True},
            series=[
                SeriesSpec(
                    name="Samples",
                    sql=(
                        "SELECT pressure AS x, flow AS y, energy AS color, weight AS size "
                        "FROM operating_points"
                    ),
                    roles={"x": "x", "y": "y", "color": "color", "size": "size"},
                    style={"marker": "o", "alpha": 0.55},
                ),
            ],
        ),
        FigureSpec(
            name="8 · Saved query",
            chart_type="Time Series",
            title="Daily mean temperature (from a saved query)",
            x_label="day",
            y_label="mean temperature (°C)",
            axis_options={"grid": True},
            series=[],  # filled in from the saved query, see build_demo_project
        ),
    ]


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
def build_demo_project(db_path: Path) -> Path:
    """Create the demo project and return the path actually written."""
    rng = np.random.default_rng(SEED)

    db_path = SqliteRepo.ensure_dhub_extension(Path(db_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    repo = SqliteRepo(db_path=db_path)

    tables = {
        "sensor_readings": _sensor_network(rng),
        "calibration": _calibration(rng),
        "batch_yields": _batches(rng),
        "throughput": _throughput(rng),
        "particles": _particles(rng),
        "operating_points": _scatter_cloud(rng),
    }
    for name, frame in tables.items():
        repo.import_dataframe(frame, table_name=name, normalize_columns=False)
        applogger.info("Demo: wrote table %s (%d rows)", name, len(frame))

    # Saved queries: one aggregate that would be tedious to rebuild by hand,
    # and one filter, to show both uses.
    repo.save_query(
        "daily_mean_temperature",
        "SELECT CAST(hour / 24 AS INTEGER) AS day, AVG(temperature) AS mean_c "
        "FROM sensor_readings GROUP BY day ORDER BY day",
    )
    repo.save_query(
        "roof_sensor",
        "SELECT hour, temperature FROM sensor_readings WHERE sensor = 'roof'",
    )

    specs = _figure_specs()

    # The saved-query figure is built from the source itself, which is exactly
    # what the chart dialog does: the subquery is inlined so the series stays
    # self-contained.
    query_source = repo.get_data_source("daily_mean_temperature")
    if query_source is not None:
        specs[-1].series = [
            SeriesSpec(
                name="Daily mean",
                sql=f'SELECT "day" AS x, "mean_c" AS y FROM {query_source.from_clause()}',
                roles={"x": "x", "y": "y"},
                style={"linestyle": "-", "marker": "o", "show_rolling": False},
            )
        ]

    for spec in specs:
        _create_figure(repo, spec)

    report = repo.optimize_db()
    applogger.info("Demo project check: %s", report.summary())
    repo.close()
    return db_path


def _create_figure(repo: SqliteRepo, spec: FigureSpec) -> int:
    """Create one figure, its single axis, and its series."""
    figure_id = repo.create_figure_descriptor(
        name=spec.name,
        nrows=1,
        ncols=1,
        options={"mpl_style": DEMO_STYLE, "layout_mode": "constrained"},
    )
    axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type=spec.chart_type,
        title=spec.title,
        x_label=spec.x_label,
        y_label=spec.y_label,
        options={"title": spec.title, **spec.axis_options},
    )

    for index, series in enumerate(spec.series):
        repo.create_series_descriptor(
            axis_id=axis_id,
            series_index=index,
            name=series.name,
            sql_query=series.sql,
            roles=series.roles,
            style=series.style,
        )

    applogger.info("Demo: created figure '%s' (%s)", spec.name, spec.chart_type)
    return int(figure_id)


def main() -> None:
    """Write the demo project to the requested path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="Demo Project.dhub",
        help="Path of the .dhub file to create (overwritten if present).",
    )
    args = parser.parse_args()

    written = build_demo_project(Path(args.output))
    print(f"Demo project written to {written}")


if __name__ == "__main__":
    main()
