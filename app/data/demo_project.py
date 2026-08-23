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
from collections.abc import Sequence
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
    """One demo figure: a title, a chart type, and its series.

    ``key`` names it for a demo project's figure list, and ``tables`` says
    what it reads - which is what lets a single-subject demo file carry only
    the tables its own charts need instead of all six.
    """

    name: str
    chart_type: str
    title: str
    x_label: str
    y_label: str
    series: list[SeriesSpec]
    axis_options: dict[str, Any]
    key: str = ""
    tables: tuple[str, ...] = ()
    queries: tuple[str, ...] = ()


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
def _process_run(rng: np.random.Generator, count: int = 120) -> pd.DataFrame:
    """A measured process that shifts part-way through.

    Shaped for the Control Chart operation, and shifted on purpose: limits
    built from the *overall* spread would be wide enough to contain the shift
    and would declare the process fine, which is the mistake the operation
    exists to avoid. A demo that only shows a stable process cannot show that.
    """
    values = rng.normal(50.0, 1.0, count)
    values[80:] += 2.5

    return pd.DataFrame(
        {
            "sample": np.arange(1, count + 1, dtype=int),
            "measurement": values,
        }
    )


def _peak_scan(rng: np.random.Generator, points: int = 240) -> pd.DataFrame:
    """One peak on a sloping baseline, with noise.

    Shaped for the Fit operation: a Gaussian sitting at x=8.2 rather than at
    the origin, because that is where a starting point read off the data earns
    its keep - the declared default puts the curve where there is no signal at
    all and a local optimiser has no gradient to follow.
    """
    x = np.linspace(0.0, 20.0, points)
    peak = 5.2 * np.exp(-((x - 8.2) ** 2) / (2.0 * 0.9**2))
    baseline = 1.4 + 0.06 * x

    return pd.DataFrame(
        {
            "wavelength_nm": x,
            "intensity": peak + baseline + rng.normal(0.0, 0.05, points),
        }
    )


def _wafer_map(rng: np.random.Generator, side: int = 41) -> pd.DataFrame:
    """Film thickness measured on a regular grid across a 300mm wafer.

    A complete x/y grid, which is what the gridded Surface Plot needs, and
    the case its circular_mask option exists for: the measurements cover the
    square grid, but a wafer is round, so the corners are outside the wafer
    rather than zero-thickness parts of it.

    The shape is a shallow dome (thicker at the centre, as a spin-coated
    film tends to be) plus a radial ripple, so the surface has something to
    show at both scales.
    """
    axis = np.linspace(-150.0, 150.0, side)
    x_grid, y_grid = np.meshgrid(axis, axis)
    radius = np.hypot(x_grid, y_grid)

    dome = 850.0 - 0.0035 * radius**2
    ripple = 6.0 * np.cos(radius / 18.0)
    thickness = dome + ripple + rng.normal(0.0, 1.2, x_grid.shape)

    return pd.DataFrame(
        {
            "x_mm": x_grid.ravel(),
            "y_mm": y_grid.ravel(),
            "thickness_nm": thickness.ravel(),
        }
    )


def _terrain_survey(rng: np.random.Generator, points: int = 600) -> pd.DataFrame:
    """Elevation at scattered survey points - deliberately not on a grid.

    The counterpart to _wafer_map: the same kind of quantity sampled where
    the surveyor could stand rather than at every grid intersection, which
    is what the triangulated Surface Plot (Scattered) is for. Pivoting this
    into a grid would invent values for the gaps; the triangulation does
    not.
    """
    angle = rng.uniform(0.0, 2.0 * np.pi, points)
    # sqrt keeps the points evenly spread over the area rather than piling
    # up at the centre, which is what uniform radius would do.
    distance = 1200.0 * np.sqrt(rng.uniform(0.0, 1.0, points))
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)

    ridge = 180.0 * np.exp(-((x - 250.0) ** 2 + (y + 150.0) ** 2) / 2.6e5)
    valley = -120.0 * np.exp(-((x + 400.0) ** 2 + (y - 300.0) ** 2) / 3.4e5)
    slope = 0.05 * x + 0.02 * y

    return pd.DataFrame(
        {
            "easting_m": x,
            "northing_m": y,
            "elevation_m": 420.0 + ridge + valley + slope + rng.normal(0.0, 4.0, points),
        }
    )


def _figure_specs() -> list[FigureSpec]:
    """Return every demo figure, in the order they appear as tabs."""
    line = {"linestyle": "-", "marker": "", "show_in_legend": True}

    return [
        FigureSpec(
            name="1 · Sensor network",
            key="sensors",
            tables=('sensor_readings',),
            queries=(),
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
            key="calibration",
            tables=('calibration',),
            queries=(),
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
            key="spread",
            tables=('batch_yields',),
            queries=(),
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
            key="summary",
            tables=('batch_yields',),
            queries=(),
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
            key="particles",
            tables=('particles',),
            queries=(),
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
            key="throughput",
            tables=('throughput',),
            queries=(),
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
            key="envelope",
            tables=('operating_points',),
            queries=(),
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
            key="saved_query",
            tables=('sensor_readings',),
            queries=('daily_mean_temperature',),
            chart_type="Time Series",
            title="Daily mean temperature (from a saved query)",
            x_label="day",
            y_label="mean temperature (°C)",
            axis_options={"grid": True},
            series=[],  # filled in from the saved query, see build_demo_project
        ),
        FigureSpec(
            name="9 · Process run",
            key="process_run",
            tables=("process_run",),
            queries=(),
            chart_type="Scatter Plot",
            title="Measurements ready for a control chart",
            x_label="sample",
            y_label="measurement",
            axis_options={"grid": True},
            series=[
                SeriesSpec(
                    name="Run",
                    sql=(
                        "SELECT sample AS x, measurement AS y FROM process_run "
                        "ORDER BY sample"
                    ),
                    roles={"x": "x", "y": "y"},
                    style={"marker": "o", "linestyle": "-", "markersize": 4.0},
                ),
            ],
        ),
        FigureSpec(
            name="10 · Peak scan",
            key="peak_scan",
            tables=("peak_scan",),
            queries=(),
            chart_type="Scatter Plot",
            title="A peak ready to fit",
            x_label="wavelength (nm)",
            y_label="intensity",
            axis_options={"grid": True},
            series=[
                SeriesSpec(
                    name="Scan",
                    sql=(
                        "SELECT wavelength_nm AS x, intensity AS y FROM peak_scan "
                        "ORDER BY wavelength_nm"
                    ),
                    roles={"x": "x", "y": "y"},
                    style={"marker": ".", "linestyle": "", "markersize": 4.0},
                ),
            ],
        ),
        FigureSpec(
            name="11 · Wafer map",
            key="wafer_map",
            tables=("wafer_map",),
            queries=(),
            chart_type="Surface Plot",
            title="Film thickness across a 300mm wafer",
            x_label="x (mm)",
            y_label="y (mm)",
            axis_options={
                # 3D axes are requested through this option, which
                # render_figure._subplot_kwargs_for_axis reads for any axis.
                "projection": "3d",
                # The measurements are on a square grid but the wafer is
                # round: without this the corners are drawn as if they were
                # part of it.
                "circular_mask": True,
                "cmap": "viridis",
            },
            series=[
                SeriesSpec(
                    name="Thickness",
                    sql=(
                        "SELECT x_mm AS x, y_mm AS y, thickness_nm AS z "
                        "FROM wafer_map"
                    ),
                    roles={"x": "x", "y": "y", "z": "z"},
                    style={},
                ),
            ],
        ),
        FigureSpec(
            name="12 · Terrain survey",
            key="terrain_survey",
            tables=("terrain_survey",),
            queries=(),
            chart_type="Surface Plot (Scattered)",
            title="Elevation from scattered survey points",
            x_label="easting (m)",
            y_label="northing (m)",
            axis_options={"projection": "3d", "cmap": "terrain"},
            series=[
                SeriesSpec(
                    name="Elevation",
                    sql=(
                        "SELECT easting_m AS x, northing_m AS y, elevation_m AS z "
                        "FROM terrain_survey"
                    ),
                    roles={"x": "x", "y": "y", "z": "z"},
                    style={},
                ),
            ],
        ),
    ]


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
#: Table name -> the function that makes it.  A demo file writes only the
#: tables its own figures read, which is what keeps a single-subject demo
#: small enough to open and understand.
TABLE_BUILDERS: dict[str, Any] = {
    "sensor_readings": _sensor_network,
    "calibration": _calibration,
    "batch_yields": _batches,
    "throughput": _throughput,
    "particles": _particles,
    "operating_points": _scatter_cloud,
    "process_run": _process_run,
    "peak_scan": _peak_scan,
    "wafer_map": _wafer_map,
    "terrain_survey": _terrain_survey,
}

#: Saved query name -> its SQL, and the table it reads.
QUERY_SOURCES: dict[str, tuple[str, str]] = {
    "daily_mean_temperature": (
        "SELECT CAST(hour / 24 AS INTEGER) AS day, AVG(temperature) AS mean_c "
        "FROM sensor_readings GROUP BY day ORDER BY day",
        "sensor_readings",
    ),
    "roof_sensor": (
        "SELECT hour, temperature FROM sensor_readings WHERE sensor = 'roof'",
        "sensor_readings",
    ),
}


@dataclass(frozen=True, slots=True)
class DemoProject:
    """One demo file: what it is called, and what it contains.

    The file name is the documentation.  Someone with eight .dhub files in a
    folder should be able to open the one that answers their question without
    opening the other seven, which means the name has to say both the subject
    and what it demonstrates.
    """

    file_name: str
    summary: str
    figures: tuple[str, ...]

    @property
    def path_name(self) -> str:
        """Return the file name with its extension."""
        return f"{self.file_name}.dhub"


#: The demo set.  The first is the complete project - every chart type over
#: every table - and the rest are one subject each.
DEMO_PROJECTS: tuple[DemoProject, ...] = (
    DemoProject(
        "Getting started - a bit of everything",
        "Every chart type in the set, over six tables and two saved queries.",
        (),
    ),
    DemoProject(
        "Sensor network - time series with a gap",
        "Three sensors sampled hourly. One loses power, and the chart draws "
        "the outage as a gap rather than a line through it.",
        ("sensors",),
    ),
    DemoProject(
        "Calibration - error bars on both axes",
        "A calibration run with uncertainty in what was applied and in what "
        "was measured.",
        ("calibration",),
    ),
    DemoProject(
        "Batch yields - distributions compared",
        "The same four batches as a violin plot and as a box plot: one shows "
        "the shape, the other the summary, and batch D is why that matters.",
        ("spread", "summary"),
    ),
    DemoProject(
        "Particle sizes - histogram of two populations",
        "Two overlapping populations in one histogram.",
        ("particles",),
    ),
    DemoProject(
        "Throughput - bar chart with error bars",
        "Monthly throughput per line, with the spread on each bar.",
        ("throughput",),
    ),
    DemoProject(
        "Operating points - scatter coloured by a third variable",
        "Flow against pressure, with energy as colour and weight as marker "
        "size: four variables on two axes.",
        ("envelope",),
    ),
    DemoProject(
        "Saved query - a chart built on a query",
        "A daily average that is computed on every read rather than stored, "
        "so editing the query updates the chart.",
        ("saved_query",),
    ),
    DemoProject(
        "Process run - ready for a control chart",
        "A process that shifts part-way through. Run Series operations, "
        "Control Chart on the series to see the shift caught.",
        ("process_run",),
    ),
    DemoProject(
        "Peak scan - ready for the Fit operation",
        "One Gaussian peak on a sloping baseline. Run Series operations, "
        "Fit and pick Gaussian peak: the starting values come from the data.",
        ("peak_scan",),
    ),
    DemoProject(
        "3D surfaces - gridded and scattered",
        "The same kind of measurement in the two layouts the surface "
        "renderers each need: film thickness on a regular wafer grid, "
        "masked to the round wafer, and elevation at scattered survey "
        "points, triangulated rather than interpolated onto a grid.",
        ("wafer_map", "terrain_survey"),
    ),
)


def build_demo_project(db_path: Path, figures: Sequence[str] = ()) -> Path:
    """Create a demo project and return the path actually written.

    ``figures`` selects by key; empty means every figure, which is the
    complete project. Only the tables and saved queries those figures read are
    written, so a single-subject file carries one table rather than six.
    """
    rng = np.random.default_rng(SEED)

    db_path = SqliteRepo.ensure_dhub_extension(Path(db_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    specs = _figure_specs()
    if figures:
        wanted = tuple(figures)
        specs = [spec for spec in specs if spec.key in wanted]

    wanted_queries = {name for spec in specs for name in spec.queries}
    wanted_tables = {name for spec in specs for name in spec.tables}
    # A saved query needs its own source table even when no figure reads that
    # table directly.
    wanted_tables.update(
        QUERY_SOURCES[name][1] for name in wanted_queries if name in QUERY_SOURCES
    )

    repo = SqliteRepo(db_path=db_path)

    tables = {
        name: builder(rng)
        for name, builder in TABLE_BUILDERS.items()
        if name in wanted_tables
    }
    for name, frame in tables.items():
        repo.import_dataframe(frame, table_name=name, normalize_columns=False)
        applogger.info("Demo: wrote table %s (%d rows)", name, len(frame))

    # Saved queries: one aggregate that would be tedious to rebuild by hand,
    # and one filter, to show both uses. The complete project gets both; a
    # single-subject file gets only what its own figures read, because a query
    # nothing charts is a loose end in a file meant to be read.
    for name, (sql, source_table) in QUERY_SOURCES.items():
        if source_table not in wanted_tables:
            continue
        if figures and name not in wanted_queries:
            continue
        repo.save_query(name, sql)

    # The saved-query figure is built from the source itself, which is exactly
    # what the chart dialog does: the subquery is inlined so the series stays
    # self-contained.
    for spec in specs:
        if spec.key != "saved_query":
            continue
        query_source = repo.get_data_source("daily_mean_temperature")
        if query_source is not None:
            spec.series = [
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


def build_demo_projects(directory: Path) -> list[Path]:
    """Write the whole demo set into *directory*, complete project first.

    Several files rather than one, each named for what it shows: a folder of
    self-describing projects is browsable, and the one that answers today's
    question can be opened without reading the other nine.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for demo in DEMO_PROJECTS:
        path = build_demo_project(target / demo.path_name, demo.figures)
        applogger.info("Demo: wrote %s - %s", path.name, demo.summary)
        written.append(path)
    return written


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
    """Write the demo project, or the whole set, to the requested path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="Demo Project.dhub",
        help="Path of the .dhub file to create (overwritten if present).",
    )
    parser.add_argument(
        "--all",
        metavar="DIRECTORY",
        help="Write every demo project into DIRECTORY, one file per subject.",
    )
    args = parser.parse_args()

    if args.all:
        for path in build_demo_projects(Path(args.all)):
            print(f"Demo project written to {path}")
        return

    written = build_demo_project(Path(args.output))
    print(f"Demo project written to {written}")


if __name__ == "__main__":
    main()
