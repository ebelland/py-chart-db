"""Build the shipped demo set from the sample data in ``sample data/``.

The demo used to be synthetic data generated fresh on every click of "Create
demo" - four batches of `rng.normal`, a peak placed exactly where the Fit
dialog needed one. That showed the chart types, but nothing in it was a
dataset anyone would recognise, and generating it on demand meant "Create
demo" could only ever be as fast as the slowest table it built.

The demo set is built ahead of time instead, from the CSV and Excel files
under ``sample data/`` at the repository root - mostly real, ordinary
datasets (Palmer penguins, a stock's daily close, a DLVO force curve, the
classic driver-clustering set) each shaped for a chart type or a Series
Operations tool, plus a handful of classic teaching examples from
mathematics and physics that are exact by construction rather than measured
(Anscombe's quartet, Lissajous curves, a signal built from known
frequencies) - see :func:`_multi_axis_figure_specs` for those. Run this
module directly, or the thin wrapper at the repository root::

    python _make_demo_project.py

which writes one ``.dhub`` file per subject into ``demo/`` - see
:data:`DEMO_DIR`. The running application never calls :func:`build_demo_project`
itself; "Load demo" copies the pre-built file instead, through
:func:`copy_demo_project`. Only this module and the tests that check its
output need pandas' read_csv/read_excel machinery.
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.charts import layout_presets
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger

#: The repository root: three levels above this file (app/data/demo_project.py).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Where the source CSV/Excel files live.
SAMPLE_DATA_DIR: Path = REPO_ROOT / "sample data"

#: Where the built demo set is written, and where "Load demo" reads it from.
#: Not version-controlled (see .gitignore's ``*.dhub``): every environment
#: that wants the demo set runs this module once to build it.
DEMO_DIR: Path = REPO_ROOT / "demo"

#: Where "Load demo" drops its working copy of a demo before opening it. A
#: folder of its own beside the application rather than the home directory:
#: the copies are throwaway (Load demo overwrites its own last copy in
#: place), and one predictable place to find - or delete - all of them at
#: once beats them scattered loose in ``~``. Not version-controlled, same as
#: ``demo/``.
PROJECTS_DIR: Path = REPO_ROOT / "projects"

# A calm, print-friendly style applied to every figure in the demo.
DEMO_STYLE = """
figure.facecolor: FBFBFD
axes.facecolor: FFFFFF
axes.edgecolor: C8CCD4
axes.linewidth: 1.0
axes.grid: True
axes.axisbelow: True
axes.titlesize: 13
# bold, not 600: DejaVu Sans (Matplotlib's bundled fallback) ships only
# normal/bold weights, so a numeric 600 request always misses and silently
# substitutes bold anyway - "findfont: Failed to find font weight 600, now
# using 700" on every render. Asking for what is actually there renders
# identically without the warning.
axes.titleweight: bold
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
    the tables its own charts need instead of all twelve.
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


@dataclass(slots=True)
class AxisSpec:
    """One axis within a MultiAxisFigureSpec - the same shape a FigureSpec's
    own fields describe, just not flattened into it: a multi-axis figure has
    several of these instead of one chart_type/title/x_label/y_label/series."""

    chart_type: str
    title: str
    x_label: str
    y_label: str
    series: list[SeriesSpec]
    axis_options: dict[str, Any]


@dataclass(slots=True)
class MultiAxisFigureSpec:
    """A figure with more than one axis, arranged by a layout preset.

    ``layout`` is one of app.charts.layout_presets.PRESETS - the same
    one-click arrangements Figure Properties offers, applied here through
    SqliteRepo.apply_axis_layout once every axis exists, so a demo figure is
    built by exactly the code path a user picking a preset would run.
    """

    name: str
    key: str
    tables: tuple[str, ...]
    layout: str
    axes: list[AxisSpec]
    queries: tuple[str, ...] = ()


# ----------------------------------------------------------------------
# Data: one loader per table, each reading a file under sample data/
# ----------------------------------------------------------------------
def _antibiotics() -> pd.DataFrame:
    """Sixteen bacteria against three antibiotics - Neomycin, Penicillin and
    Streptomycin - and the Gram stain that turns out to explain the pattern."""
    frame = pd.read_csv(SAMPLE_DATA_DIR / "Antibiotics.csv")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    return frame


def _penguins() -> pd.DataFrame:
    """The Palmer penguins, minus the handful of rows missing a measurement."""
    frame = pd.read_csv(SAMPLE_DATA_DIR / "penguins.csv")
    return frame.dropna(
        subset=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    ).reset_index(drop=True)


def _yeast() -> pd.DataFrame:
    """Yeast protein measurements, with their localisation class."""
    frame = pd.read_csv(SAMPLE_DATA_DIR / "Yeast1141.csv")
    frame.columns = [
        str(c).strip().lower().replace(" ", "_") for c in frame.columns
    ]
    return frame


def _dlvo_curve() -> pd.DataFrame:
    """A DLVO force-distance curve: repulsive at short range, attractive
    beyond it. Already clean - column names are the physical quantities."""
    return pd.read_csv(SAMPLE_DATA_DIR / "dlvo_standalone_output.csv")


def _stock_prices() -> pd.DataFrame:
    """Daily OHLC for three tickers, kept down to date/close/stock."""
    frame = pd.read_csv(SAMPLE_DATA_DIR / "hello-world-stock.csv")
    frame = frame.rename(columns={"Date": "date", "Close": "close", "Stock": "stock"})
    return frame[["date", "close", "stock"]]


def _employee_compensation() -> pd.DataFrame:
    """Salaries across five departments, with one genuine outlier."""
    return pd.read_csv(SAMPLE_DATA_DIR / "outlier_sample_dataset.csv")


def _driver_behaviour() -> pd.DataFrame:
    """The classic two-feature driver-clustering dataset."""
    frame = pd.read_csv(SAMPLE_DATA_DIR / "Clustering.csv")
    return frame.rename(
        columns={
            "Driver_ID": "driver_id",
            "Distance_Feature": "distance_feature",
            "Speeding_Feature": "speeding_feature",
        }
    )[["driver_id", "distance_feature", "speeding_feature"]]


def _transactions() -> pd.DataFrame:
    """Card, online, mobile and branch transactions, a small fraction fraud."""
    return pd.read_csv(SAMPLE_DATA_DIR / "synthetic_balancing_resampling_dataset.csv")


def _pairwise_sample() -> pd.DataFrame:
    """Two coordinates plus a colour and a marker size - built for a scatter
    plot with every optional role in use."""
    return pd.read_excel(SAMPLE_DATA_DIR / "pairwise_sample.xlsx")


def _surface_grid() -> pd.DataFrame:
    """A smooth surface on a regular 40x40 x/y grid."""
    return pd.read_excel(SAMPLE_DATA_DIR / "sample_3d_surface.xlsx")


def _scattered_surface() -> pd.DataFrame:
    """The same kind of surface, sampled at 200 scattered (non-gridded)
    points - what the triangulated Surface Plot (Scattered) is for."""
    return pd.read_excel(SAMPLE_DATA_DIR / "sample_3d_scatter.xlsx")


def _parametric_curve() -> pd.DataFrame:
    """A helix: x=cos(t), y=sin(t), z=t. Read as x against y, it is a circle
    coloured by how far along the curve each point is."""
    return pd.read_excel(SAMPLE_DATA_DIR / "sample_3d_parametric.xlsx")


def _anscombe() -> pd.DataFrame:
    """Anscombe's quartet: four x/y datasets, one ``dataset`` column (I-IV)
    telling them apart. Textbook values - same mean, variance and
    correlation to three decimal places on every one of the four - not
    measured, and not needing to be: the numbers themselves are the whole
    demonstration, and everyone who plots them gets the same four shapes.
    """
    return pd.read_csv(SAMPLE_DATA_DIR / "anscombe.csv")


def _lissajous() -> pd.DataFrame:
    """Four Lissajous figures - x=sin(a*t+delta), y=sin(b*t) - at frequency
    ratios 1:1, 1:2, 3:2 and 5:4, one ``curve`` column telling them apart.
    Generated, not measured: a Lissajous figure is defined by its ratio, so
    there is no real-world reading to substitute for computing it.
    """
    return pd.read_csv(SAMPLE_DATA_DIR / "lissajous.csv")


def _signal_time_domain() -> pd.DataFrame:
    """A synthetic signal - 5, 20 and 50 Hz tones plus noise, 2 s at 500 Hz -
    built (not measured) so its true frequency content is known exactly,
    which is what the paired spectrum in :func:`_signal_spectrum` is there
    to recover. Fixed random seed: the noise looks the same on every
    machine that builds this demo.
    """
    return pd.read_csv(SAMPLE_DATA_DIR / "signal_time_domain.csv")


def _signal_spectrum() -> pd.DataFrame:
    """The single-sided amplitude spectrum of :func:`_signal_time_domain`'s
    signal (``numpy.fft.rfft``, normalised so each tone's peak reads off as
    its actual amplitude) - the three injected tones are its three tallest
    peaks, at 5, 20 and 50 Hz."""
    return pd.read_csv(SAMPLE_DATA_DIR / "signal_spectrum.csv")


def _attribute_chart_counts() -> pd.DataFrame:
    """Montgomery's *Introduction to Statistical Quality Control*, Example
    6.1: 20 samples of 50 printed circuit boards, each with its count of
    defective units. p-bar works out to 0.214, the UCL to 0.388 and the LCL
    to 0.040 - textbook numbers a p-chart run over this table should
    reproduce exactly, with sample 15 (22 defectives) the one point outside
    them. ``inspected`` is included as its own column, not folded into a
    fixed constant, so the Control Chart operation's own "sample size
    column" picker (shown for its attribute charts - p, np, u) has
    something to pick.
    """
    defectives = [
        12, 15, 8, 10, 4, 7, 16, 9, 14, 10,
        5, 6, 17, 12, 22, 8, 10, 5, 13, 11,
    ]
    return pd.DataFrame(
        {
            "sample": range(1, len(defectives) + 1),
            "defectives": defectives,
            "inspected": [50] * len(defectives),
        }
    )


def _baseline_spectrum() -> pd.DataFrame:
    """A synthetic spectrum: two Gaussian peaks on a background that both
    drifts (a linear ramp) and wanders (a slow sine) - the two components a
    convex rubber-band baseline cannot follow but AsLS can. Fixed random
    seed, same convention as :func:`_signal_time_domain`: the noise looks
    the same on every machine that builds this demo.
    """
    rng = np.random.default_rng(20260911)
    x = np.linspace(0.0, 100.0, 400)
    background = 5.0 + 0.02 * x + 3.0 * np.sin(x / 30.0)
    peaks = (
        10.0 * np.exp(-((x - 30.0) ** 2) / 4.0)
        + 15.0 * np.exp(-((x - 70.0) ** 2) / 8.0)
    )
    intensity = background + peaks + 0.15 * rng.standard_normal(x.size)
    return pd.DataFrame({"x": x, "intensity": intensity})


#: Table name -> the function that loads it. A demo file writes only the
#: tables its own figures read, which is what keeps a single-subject demo
#: small enough to open and understand.
TABLE_SOURCES: dict[str, Callable[[], pd.DataFrame]] = {
    "antibiotics": _antibiotics,
    "penguins": _penguins,
    "yeast": _yeast,
    "dlvo_curve": _dlvo_curve,
    "stock_prices": _stock_prices,
    "employee_compensation": _employee_compensation,
    "driver_behaviour": _driver_behaviour,
    "transactions": _transactions,
    "pairwise_sample": _pairwise_sample,
    "surface_grid": _surface_grid,
    "scattered_surface": _scattered_surface,
    "parametric_curve": _parametric_curve,
    "anscombe": _anscombe,
    "lissajous": _lissajous,
    "signal_time_domain": _signal_time_domain,
    "signal_spectrum": _signal_spectrum,
    "attribute_chart_counts": _attribute_chart_counts,
    "baseline_spectrum": _baseline_spectrum,
}

#: Saved query name -> its SQL, and the table it reads.
QUERY_SOURCES: dict[str, tuple[str, str]] = {
    "avg_amount_by_channel": (
        "SELECT channel AS X, AVG(transaction_amount) AS Y, COUNT(*) AS n "
        "FROM transactions GROUP BY channel ORDER BY Y DESC",
        "transactions",
    ),
}


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _figure_specs() -> list[FigureSpec]:
    """Return every demo figure, in the order they appear as tabs."""
    return [
        FigureSpec(
            name="1 · Antibiotic potency",
            key="antibiotics",
            tables=("antibiotics",),
            queries=(),
            chart_type="Horizontal Bar Chart",
            title="Antibiotic potency by bacteria (lower is more effective)",
            x_label="minimum inhibitory concentration",
            y_label="bacteria",
            axis_options={"grid": True, "grid_axis": "x"},
            series=[
                SeriesSpec(
                    name=column.capitalize(),
                    sql=(
                        f"SELECT bacteria AS X, {column} AS Y FROM antibiotics "
                        f"ORDER BY {column}"
                    ),
                    roles={"X": "X", "Y": "Y"},
                    style={"alpha": 0.85},
                )
                for column in ("penicillin", "streptomycin", "neomycin")
            ],
        ),
        FigureSpec(
            name="2 · Penguin bill dimensions",
            key="penguin_scatter",
            tables=("penguins",),
            queries=(),
            chart_type="Scatter Plot",
            title="Bill length against bill depth, by species",
            x_label="bill length (mm)",
            y_label="bill depth (mm)",
            axis_options={"grid": True},
            series=[
                SeriesSpec(
                    name=species,
                    sql=(
                        "SELECT bill_length_mm AS x, bill_depth_mm AS y "
                        f"FROM penguins WHERE species = '{species}'"
                    ),
                    roles={"x": "x", "y": "y"},
                    style={"marker": "o", "alpha": 0.75},
                )
                for species in ("Adelie", "Chinstrap", "Gentoo")
            ],
        ),
        FigureSpec(
            name="3 · Penguin body mass",
            key="penguin_hist",
            tables=("penguins",),
            queries=(),
            chart_type="Histogram",
            title="Body mass by species",
            x_label="body mass (g)",
            y_label="",
            axis_options={"bins": 30, "alpha": 0.8, "grid": True, "grid_axis": "y"},
            series=[
                SeriesSpec(
                    name="Species",
                    sql="SELECT species AS dataset, body_mass_g AS value FROM penguins",
                    roles={"value": "value", "dataset": "dataset"},
                    style={},
                ),
            ],
        ),
        FigureSpec(
            name="4 · Penguin flipper length - spread",
            key="penguin_violin",
            tables=("penguins",),
            queries=(),
            chart_type="Violin Plot",
            title="Flipper length distribution by species",
            x_label="species",
            y_label="flipper length (mm)",
            axis_options={
                "grid": True,
                "grid_axis": "y",
                "showmedians": True,
                "alpha": 0.75,
            },
            series=[
                SeriesSpec(
                    name=species,
                    sql=(
                        'SELECT species AS "group", flipper_length_mm AS value '
                        f"FROM penguins WHERE species = '{species}'"
                    ),
                    roles={"value": "value", "group": "group"},
                    style={},
                )
                for species in ("Adelie", "Chinstrap", "Gentoo")
            ],
        ),
        FigureSpec(
            name="5 · Penguin flipper length - summary",
            key="penguin_box",
            tables=("penguins",),
            queries=(),
            chart_type="Box Plot",
            title="Flipper length summary by species",
            x_label="species",
            y_label="flipper length (mm)",
            axis_options={"grid": True, "grid_axis": "y", "showmeans": True},
            series=[
                SeriesSpec(
                    name=species,
                    sql=(
                        'SELECT species AS "group", flipper_length_mm AS value '
                        f"FROM penguins WHERE species = '{species}'"
                    ),
                    roles={"value": "value", "group": "group"},
                    style={},
                )
                for species in ("Adelie", "Chinstrap", "Gentoo")
            ],
        ),
        FigureSpec(
            name="6 · Yeast protein localisation",
            key="yeast_hist",
            tables=("yeast",),
            queries=(),
            chart_type="Histogram",
            title="mcg signal by localisation class",
            x_label="mcg",
            y_label="",
            axis_options={"bins": 40, "alpha": 0.8, "grid": True, "grid_axis": "y"},
            series=[
                SeriesSpec(
                    name="Class",
                    sql=(
                        "SELECT class AS dataset, mcg AS value FROM yeast "
                        "WHERE class IN ('CYT', 'NUC', 'MIT')"
                    ),
                    roles={"value": "value", "dataset": "dataset"},
                    style={},
                ),
            ],
        ),
        FigureSpec(
            name="7 · DLVO force curve",
            key="dlvo_force",
            tables=("dlvo_curve",),
            queries=(),
            chart_type="Scatter Plot",
            title="Total force against separation",
            x_label="separation (nm)",
            y_label="force (nN)",
            axis_options={"grid": True},
            series=[
                SeriesSpec(
                    name="Force",
                    sql=(
                        "SELECT separation_nm AS x, F_total_nN AS y FROM dlvo_curve "
                        "ORDER BY separation_nm"
                    ),
                    roles={"x": "x", "y": "y"},
                    style={"marker": "", "linestyle": "-"},
                ),
            ],
        ),
        FigureSpec(
            name="8 · Stock prices",
            key="stock_timeseries",
            tables=("stock_prices",),
            queries=(),
            chart_type="Time Series",
            title="Daily closing price",
            x_label="date",
            y_label="close (USD)",
            axis_options={"grid": True, "show_rolling": False, "gap_threshold": "5D"},
            series=[
                SeriesSpec(
                    name=ticker,
                    sql=(
                        "SELECT date AS x, close AS y FROM stock_prices "
                        f"WHERE stock = '{ticker}' ORDER BY date"
                    ),
                    roles={"x": "x", "y": "y"},
                    style={"linestyle": "-", "marker": ""},
                )
                for ticker in ("AAPL", "TSLA", "COKE")
            ],
        ),
        FigureSpec(
            name="9 · Employee compensation",
            key="employee_box",
            tables=("employee_compensation",),
            queries=(),
            chart_type="Box Plot",
            title="Salary by department",
            x_label="department",
            y_label="salary (EUR)",
            axis_options={"grid": True, "grid_axis": "y", "showmeans": True},
            series=[
                SeriesSpec(
                    name=department,
                    sql=(
                        'SELECT department AS "group", salary_eur AS value '
                        f"FROM employee_compensation WHERE department = '{department}'"
                    ),
                    roles={"value": "value", "group": "group"},
                    style={},
                )
                for department in ("Operations", "Sales", "Finance", "IT", "HR")
            ],
        ),
        FigureSpec(
            name="10 · Driver behaviour",
            key="driver_scatter",
            tables=("driver_behaviour",),
            queries=(),
            chart_type="Scatter Plot",
            title="Distance against speeding, four thousand drivers",
            x_label="distance feature",
            y_label="speeding feature",
            axis_options={"grid": True},
            series=[
                SeriesSpec(
                    name="Drivers",
                    sql=(
                        "SELECT distance_feature AS x, speeding_feature AS y "
                        "FROM driver_behaviour"
                    ),
                    roles={"x": "x", "y": "y"},
                    style={"marker": ".", "linestyle": "", "alpha": 0.35, "markersize": 4.0},
                ),
            ],
        ),
        FigureSpec(
            name="11 · Transaction amounts",
            key="transactions_hist",
            tables=("transactions",),
            queries=(),
            chart_type="Histogram",
            title="Transaction amount, fraud against legitimate",
            x_label="transaction amount",
            y_label="",
            axis_options={"bins": 40, "alpha": 0.8, "grid": True, "grid_axis": "y"},
            series=[
                SeriesSpec(
                    name="Outcome",
                    sql=(
                        "SELECT CASE fraud WHEN 1 THEN 'fraud' ELSE 'legitimate' END "
                        "AS dataset, transaction_amount AS value FROM transactions"
                    ),
                    roles={"value": "value", "dataset": "dataset"},
                    style={},
                ),
            ],
        ),
        FigureSpec(
            name="12 · Transactions by channel",
            key="transactions_query",
            tables=("transactions",),
            queries=("avg_amount_by_channel",),
            chart_type="Bar Chart",
            title="Average transaction amount by channel (from a saved query)",
            x_label="channel",
            y_label="average amount",
            axis_options={"grid": True, "grid_axis": "y"},
            series=[],  # filled in from the saved query, see build_demo_project
        ),
        FigureSpec(
            name="13 · Pairwise sample",
            key="pairwise_scatter",
            tables=("pairwise_sample",),
            queries=(),
            chart_type="Scatter Plot",
            title="Two coordinates, coloured and sized by two more",
            x_label="x1",
            y_label="x2",
            axis_options={"grid": True},
            series=[
                SeriesSpec(
                    name="Samples",
                    sql=(
                        "SELECT x1 AS x, x2 AS y, color AS color, size AS size "
                        "FROM pairwise_sample"
                    ),
                    roles={"x": "x", "y": "y", "color": "color", "size": "size"},
                    style={"marker": "o", "alpha": 0.7},
                ),
            ],
        ),
        FigureSpec(
            name="14 · 3D surface",
            key="surface_3d",
            tables=("surface_grid",),
            queries=(),
            chart_type="Surface Plot",
            title="A surface on a regular grid",
            x_label="x",
            y_label="y",
            axis_options={"projection": "3d", "cmap": "viridis"},
            series=[
                SeriesSpec(
                    name="Surface",
                    sql="SELECT x AS x, y AS y, z AS z FROM surface_grid",
                    roles={"x": "x", "y": "y", "z": "z"},
                    style={},
                ),
            ],
        ),
        FigureSpec(
            name="15 · 3D scattered surface",
            key="scattered_3d",
            tables=("scattered_surface",),
            queries=(),
            chart_type="Surface Plot (Scattered)",
            title="The same kind of surface, from scattered points",
            x_label="x",
            y_label="y",
            axis_options={"projection": "3d", "cmap": "terrain"},
            series=[
                SeriesSpec(
                    name="Surface",
                    sql="SELECT x AS x, y AS y, z AS z FROM scattered_surface",
                    roles={"x": "x", "y": "y", "z": "z"},
                    style={},
                ),
            ],
        ),
        FigureSpec(
            name="16 · Parametric curve",
            key="parametric_scatter",
            tables=("parametric_curve",),
            queries=(),
            chart_type="Scatter Plot",
            title="A helix, seen end-on and coloured by progress",
            x_label="x",
            y_label="y",
            axis_options={"grid": True},
            series=[
                SeriesSpec(
                    name="Helix",
                    sql="SELECT x AS x, y AS y, t AS color FROM parametric_curve ORDER BY t",
                    roles={"x": "x", "y": "y", "color": "color"},
                    style={"marker": ".", "linestyle": "", "markersize": 5.0},
                ),
            ],
        ),
        FigureSpec(
            name="23 · Defectives per sample - ready for the Control Chart operation",
            key="attribute_counts",
            tables=("attribute_chart_counts",),
            queries=(),
            chart_type="Scatter Plot",
            title="Defective circuit boards, 20 samples of 50 (Montgomery 6.1)",
            x_label="sample",
            y_label="defectives",
            # The annotation and the reference line double as a demo of the
            # Overlay panel's colour/font/size editors: both carry an
            # explicit colour and the annotation an explicit font and size,
            # so opening Overlay properties on this axis shows every one of
            # them already populated rather than at "(none)"/"Default".
            axis_options={
                "grid": True,
                "annotations": [
                    {
                        "x": 15.0, "y": 22.0, "type": "text",
                        "text": "sample 15: 22/50 - beyond the p-chart's UCL",
                        "kwargs": {
                            "color": "#c62828",
                            "fontfamily": "sans-serif",
                            "fontsize": 9,
                            "xytext": [8, 10],
                            "textcoords": "offset points",
                        },
                    },
                ],
                "lines": [
                    {
                        "orientation": "horizontal", "value": 10.7,
                        "kwargs": {
                            "color": "#2e7d32",
                            "linestyle": "--",
                            "label": "n x p-bar",
                        },
                    },
                ],
            },
            series=[
                SeriesSpec(
                    name="Defectives",
                    sql=(
                        "SELECT sample AS x, defectives AS y, inspected AS n "
                        "FROM attribute_chart_counts ORDER BY sample"
                    ),
                    roles={"x": "x", "y": "y"},
                    style={"marker": "o", "linestyle": "-"},
                ),
            ],
        ),
        FigureSpec(
            name="24 · Spectrum with background - ready for Baseline Correction",
            key="baseline_spectrum",
            tables=("baseline_spectrum",),
            queries=(),
            chart_type="Scatter Plot",
            title="Two peaks on a drifting, wandering background",
            x_label="x",
            y_label="intensity",
            axis_options={
                "grid": True,
                "annotations": [
                    {
                        "x": 70.0, "y": 23.0, "type": "text",
                        "text": "background rises and wanders under both peaks",
                        "kwargs": {
                            "color": "#1565c0",
                            "fontfamily": "serif",
                            "fontsize": 9,
                            "xytext": [-90, 12],
                            "textcoords": "offset points",
                        },
                    },
                ],
            },
            series=[
                SeriesSpec(
                    name="Spectrum",
                    sql="SELECT x, intensity AS y FROM baseline_spectrum ORDER BY x",
                    roles={"x": "x", "y": "y"},
                    style={"marker": "", "linestyle": "-", "linewidth": 1.0},
                ),
            ],
        ),
    ]


def _multi_axis_figure_specs() -> list[MultiAxisFigureSpec]:
    """Return every demo figure with more than one axis.

    The first three are one per layout preset - main+secondary, shared
    grid, overlapping - each reusing a table a single-axis figure above
    already needs, so showing the layout costs no new data. The rest are
    classic multi-panel teaching examples from mathematics, physics and
    signal processing (Anscombe's quartet, Lissajous curves, a signal and
    its spectrum), each with its own small table: the point there is the
    subject, and the layout preset is whichever one actually suits it.
    """
    return [
        MultiAxisFigureSpec(
            name="17 · Stock prices - main and secondary",
            key="stock_main_secondary",
            tables=("stock_prices",),
            queries=(),
            layout=layout_presets.MAIN_AND_SECONDARY,
            axes=[
                AxisSpec(
                    chart_type="Time Series",
                    title="AAPL - daily close",
                    x_label="date",
                    y_label="close (USD)",
                    axis_options={"grid": True, "show_rolling": False, "gap_threshold": "5D"},
                    series=[
                        SeriesSpec(
                            name="AAPL",
                            sql="SELECT date AS x, close AS y FROM stock_prices "
                            "WHERE stock = 'AAPL' ORDER BY date",
                            roles={"x": "x", "y": "y"},
                            style={"linestyle": "-", "marker": ""},
                        ),
                    ],
                ),
                *[
                    AxisSpec(
                        chart_type="Time Series",
                        title=ticker,
                        x_label="date",
                        y_label="close (USD)",
                        axis_options={"grid": True, "show_rolling": False, "gap_threshold": "5D"},
                        series=[
                            SeriesSpec(
                                name=ticker,
                                sql="SELECT date AS x, close AS y FROM stock_prices "
                                f"WHERE stock = '{ticker}' ORDER BY date",
                                roles={"x": "x", "y": "y"},
                                style={"linestyle": "-", "marker": ""},
                            ),
                        ],
                    )
                    for ticker in ("TSLA", "COKE")
                ],
            ],
        ),
        MultiAxisFigureSpec(
            name="18 · Penguins - shared scale grid",
            key="penguin_shared_grid",
            tables=("penguins",),
            queries=(),
            layout=layout_presets.SHARED_GRID,
            axes=[
                AxisSpec(
                    chart_type="Histogram",
                    title=species,
                    x_label="body mass (g)",
                    y_label="",
                    axis_options={"bins": 20, "alpha": 0.8, "grid": True, "grid_axis": "y"},
                    series=[
                        SeriesSpec(
                            name=species,
                            sql="SELECT body_mass_g AS value FROM penguins "
                            f"WHERE species = '{species}'",
                            roles={"value": "value"},
                            style={},
                        ),
                    ],
                )
                for species in ("Adelie", "Chinstrap", "Gentoo")
            ],
        ),
        MultiAxisFigureSpec(
            name="19 · DLVO force and potential - overlapping axes",
            key="dlvo_overlapping",
            tables=("dlvo_curve",),
            queries=(),
            layout=layout_presets.OVERLAPPING,
            axes=[
                AxisSpec(
                    chart_type="Scatter Plot",
                    title="Force and potential energy against separation",
                    x_label="separation (nm)",
                    y_label="force (nN)",
                    axis_options={"grid": True},
                    series=[
                        SeriesSpec(
                            name="Force",
                            sql="SELECT separation_nm AS x, F_total_nN AS y "
                            "FROM dlvo_curve ORDER BY separation_nm",
                            roles={"x": "x", "y": "y"},
                            style={"marker": "", "linestyle": "-"},
                        ),
                    ],
                ),
                AxisSpec(
                    chart_type="Scatter Plot",
                    # Left blank on purpose: a twin shares its target's grid
                    # cell, so a second title here would draw on top of the
                    # primary axis's - see render_figure's twin handling.
                    title="",
                    x_label="separation (nm)",
                    y_label="potential energy (aJ)",
                    axis_options={"grid": False},
                    series=[
                        SeriesSpec(
                            name="Potential energy",
                            sql="SELECT separation_nm AS x, U_total_aJ AS y "
                            "FROM dlvo_curve ORDER BY separation_nm",
                            roles={"x": "x", "y": "y"},
                            style={"marker": "", "linestyle": "--", "color": "#E45756"},
                        ),
                    ],
                ),
            ],
        ),
        MultiAxisFigureSpec(
            name="20 · Anscombe's quartet - the matplotlib classic",
            key="anscombe_quartet",
            tables=("anscombe",),
            queries=(),
            layout=layout_presets.SHARED_GRID,
            axes=[
                AxisSpec(
                    chart_type="Scatter Plot",
                    title=f"Dataset {dataset}",
                    x_label="x",
                    y_label="y",
                    axis_options={"grid": True},
                    series=[
                        SeriesSpec(
                            name=f"Dataset {dataset}",
                            sql="SELECT x, y FROM anscombe "
                            f"WHERE dataset = '{dataset}' ORDER BY x",
                            roles={"x": "x", "y": "y"},
                            style={"marker": "o", "linestyle": ""},
                        ),
                    ],
                )
                for dataset in ("I", "II", "III", "IV")
            ],
        ),
        MultiAxisFigureSpec(
            name="21 · Lissajous curves - equal-aspect grid",
            key="lissajous_grid",
            tables=("lissajous",),
            queries=(),
            # Plain GRID, not SHARED_GRID: Matplotlib refuses equal aspect on
            # axes that share *both* x and y (see the axis_options note
            # below) - and every curve here already lives in the same [-1,
            # 1] box by construction, so sharing buys no extra comparability
            # that equal aspect was not already giving it.
            layout=layout_presets.GRID,
            axes=[
                AxisSpec(
                    chart_type="Scatter Plot",
                    title=curve,
                    x_label="x",
                    y_label="y",
                    # Equal aspect, or a circle (the 1:1 ratio) would draw as
                    # an ellipse and misrepresent the curve. adjustable=
                    # "datalim" rather than the default "box" only matters
                    # once an axis shares a scale with another - harmless
                    # here, and worth setting anyway since it is the one
                    # value of the two that is never wrong.
                    axis_options={"grid": True, "aspect": "equal", "adjustable": "datalim"},
                    series=[
                        SeriesSpec(
                            name=curve,
                            sql="SELECT x, y FROM lissajous "
                            f"WHERE curve = '{curve}' ORDER BY rowid",
                            roles={"x": "x", "y": "y"},
                            style={"marker": "", "linestyle": "-"},
                        ),
                    ],
                )
                for curve in (
                    "1:1 - circle/ellipse (delta = pi/4)",
                    "1:2 - a figure eight",
                    "3:2",
                    "5:4",
                )
            ],
        ),
        MultiAxisFigureSpec(
            name="22 · Signal analysis - time and frequency domains",
            key="signal_time_and_frequency",
            tables=("signal_time_domain", "signal_spectrum"),
            queries=(),
            layout=layout_presets.GRID,
            axes=[
                AxisSpec(
                    chart_type="Time Series",
                    title="Signal - 5, 20, 50 Hz + noise",
                    x_label="time (s)",
                    y_label="amplitude",
                    # show_rolling defaults on; a rolling mean drawn over a
                    # signal whose own raw shape is the point would only
                    # hide it.
                    axis_options={"grid": True, "show_rolling": False},
                    series=[
                        SeriesSpec(
                            name="signal",
                            sql="SELECT t_s AS x, amplitude AS y "
                            "FROM signal_time_domain ORDER BY t_s",
                            roles={"x": "x", "y": "y"},
                            style={"marker": "", "linestyle": "-", "linewidth": 0.8},
                        ),
                    ],
                ),
                AxisSpec(
                    chart_type="Time Series",
                    title="Spectrum - recovers the tones",
                    x_label="frequency (Hz)",
                    y_label="amplitude",
                    axis_options={"grid": True, "show_rolling": False},
                    series=[
                        SeriesSpec(
                            name="spectrum",
                            sql="SELECT freq_hz AS x, magnitude AS y "
                            "FROM signal_spectrum WHERE freq_hz <= 80 ORDER BY freq_hz",
                            roles={"x": "x", "y": "y"},
                            style={"marker": "", "linestyle": "-"},
                        ),
                    ],
                ),
            ],
        ),
    ]


# ----------------------------------------------------------------------
# The demo set
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DemoProject:
    """One demo file: what it is called, and what it contains.

    The file name is the documentation.  Someone with a dozen .dhub files in a
    folder should be able to open the one that answers their question without
    opening the other eleven, which means the name has to say both the
    subject and what it demonstrates.
    """

    file_name: str
    summary: str
    figures: tuple[str, ...]

    @property
    def path_name(self) -> str:
        """Return the file name with its extension."""
        return f"{self.file_name}.dhub"

    @property
    def source_path(self) -> Path:
        """Return where the pre-built copy of this project lives.

        Built by :func:`build_demo_projects` into :data:`DEMO_DIR`, not at
        the point this is read - see :func:`copy_demo_project`.
        """
        return DEMO_DIR / self.path_name


#: The demo set.  The first is the complete project - every chart type over
#: every table - and the rest are one real dataset each.
DEMO_PROJECTS: tuple[DemoProject, ...] = (
    DemoProject(
        "Getting started - real data across every chart type",
        "Fifteen real datasets, twenty-two figures across every chart type "
        "and six multi-axis layouts, and one saved query.",
        (),
    ),
    DemoProject(
        "Antibiotics - grouped bars from a classic dataset",
        "1930s antibiotic potency data for sixteen bacteria, as a horizontal "
        "bar chart sorted by effectiveness.",
        ("antibiotics",),
    ),
    DemoProject(
        "Penguins - species compared across four chart types",
        "The Palmer penguins: the same three species read as a scatter plot, "
        "a histogram, a violin plot and a box plot.",
        ("penguin_scatter", "penguin_hist", "penguin_violin", "penguin_box"),
    ),
    DemoProject(
        "Yeast proteins - histogram of three localisation classes",
        "Cytoplasm, nucleus and mitochondria: three overlapping distributions "
        "of one measured feature.",
        ("yeast_hist",),
    ),
    DemoProject(
        "DLVO force curve - ready for Fit and Calculus",
        "A real colloidal force-distance curve, repulsive at short range and "
        "attractive beyond it, with a measurable crossover.",
        ("dlvo_force",),
    ),
    DemoProject(
        "Stock prices - three tickers, ready for Smoothing",
        "Three years of daily closing prices for three stocks, noisy enough "
        "that a moving average earns its keep.",
        ("stock_timeseries",),
    ),
    DemoProject(
        "Employee compensation - ready for the Outlier operation",
        "Salaries across five departments, with one real outlier the box "
        "plot already shows and the operation can confirm.",
        ("employee_box",),
    ),
    DemoProject(
        "Driver behaviour - ready for the Cluster operation",
        "Four thousand drivers by distance and speeding: the classic "
        "two-feature clustering dataset.",
        ("driver_scatter",),
    ),
    DemoProject(
        "Transactions - histogram and a saved query",
        "Fraud against legitimate transactions by amount, and a saved query "
        "averaging amount by channel.",
        ("transactions_hist", "transactions_query"),
    ),
    DemoProject(
        "Pairwise sample - four variables on two axes",
        "Two coordinates, a colour and a marker size: four variables read "
        "off one scatter plot.",
        ("pairwise_scatter",),
    ),
    DemoProject(
        "3D surfaces - gridded and scattered",
        "The same kind of surface in the two layouts the surface renderers "
        "each need: a regular grid, and scattered points triangulated into "
        "one.",
        ("surface_3d", "scattered_3d"),
    ),
    DemoProject(
        "Parametric curve - a helix in projection",
        "A helix seen end-on: a circle, coloured by how far along the curve "
        "each point is.",
        ("parametric_scatter",),
    ),
    DemoProject(
        "Figure layouts - shared scale, main+secondary, overlapping axes",
        "Three real datasets, each arranged with a different multi-axis "
        "layout preset: three price series sharing one main chart, three "
        "histograms panning and zooming together, and force paired with "
        "potential energy on a second y-axis.",
        ("stock_main_secondary", "penguin_shared_grid", "dlvo_overlapping"),
    ),
    DemoProject(
        "Classic demos - Anscombe, Lissajous, and a signal spectrum",
        "Three teaching examples from mathematics, physics and signal "
        "processing: Anscombe's quartet on a shared-scale grid (the "
        "matplotlib gallery's own multi-axis example), four Lissajous "
        "figures at different frequency ratios, and a noisy signal beside "
        "the frequency spectrum that recovers its three true tones - the "
        "same signal the Filtering operation's IIR/FIR/Hilbert models can "
        "run on directly (its 5, 20 and 50 Hz tones are exact targets for "
        "a lowpass, a bandpass, or an envelope).",
        ("anscombe_quartet", "lissajous_grid", "signal_time_and_frequency"),
    ),
    DemoProject(
        "Quality and spectroscopy - two new series operations",
        "Textbook defective-unit counts for the Control Chart operation's "
        "attribute charts (p/np/c/u), and a synthetic spectrum with a "
        "drifting, wandering "
        "background for Baseline Correction (AsLS or rubber band) - both "
        "figures already carry an annotation and a reference line with an "
        "explicit colour, so opening Overlay properties shows its colour/"
        "line/font/size editors populated rather than empty.",
        ("attribute_counts", "baseline_spectrum"),
    ),
)


def build_demo_project(db_path: Path, figures: tuple[str, ...] = ()) -> Path:
    """Create a demo project and return the path actually written.

    ``figures`` selects by key; empty means every figure, which is the
    complete project. Only the tables and saved queries those figures read are
    written, so a single-subject file carries one or two tables rather than
    twelve.
    """
    db_path = SqliteRepo.ensure_dhub_extension(Path(db_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    specs = _figure_specs()
    multi_specs = _multi_axis_figure_specs()
    if figures:
        wanted = tuple(figures)
        specs = [spec for spec in specs if spec.key in wanted]
        multi_specs = [spec for spec in multi_specs if spec.key in wanted]

    all_specs: list[FigureSpec | MultiAxisFigureSpec] = [*specs, *multi_specs]
    wanted_queries = {name for spec in all_specs for name in spec.queries}
    wanted_tables = {name for spec in all_specs for name in spec.tables}
    # A saved query needs its own source table even when no figure reads that
    # table directly.
    wanted_tables.update(
        QUERY_SOURCES[name][1] for name in wanted_queries if name in QUERY_SOURCES
    )

    repo = SqliteRepo(db_path=db_path)

    tables = {name: loader() for name, loader in TABLE_SOURCES.items() if name in wanted_tables}
    for name, frame in tables.items():
        repo.import_dataframe(frame, table_name=name, normalize_columns=False)
        applogger.info("Demo: wrote table %s (%d rows)", name, len(frame))

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
        if spec.key != "transactions_query":
            continue
        query_source = repo.get_data_source("avg_amount_by_channel")
        if query_source is not None:
            spec.series = [
                SeriesSpec(
                    name="Channel",
                    sql=f'SELECT "X" AS X, "Y" AS Y FROM {query_source.from_clause()}',
                    roles={"X": "X", "Y": "Y"},
                    style={"alpha": 0.9},
                )
            ]

    for spec in specs:
        _create_figure(repo, spec)
    for multi_spec in multi_specs:
        _create_multi_axis_figure(repo, multi_spec)

    report = repo.optimize_db()
    applogger.info("Demo project check: %s", report.summary())
    repo.close()
    return db_path


def build_demo_projects(directory: Path) -> list[Path]:
    """Write the whole demo set into *directory*, complete project first.

    Several files rather than one, each named for what it shows: a folder of
    self-describing projects is browsable, and the one that answers today's
    question can be opened without reading the other eleven.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for demo in DEMO_PROJECTS:
        path = build_demo_project(target / demo.path_name, demo.figures)
        applogger.info("Demo: wrote %s - %s", path.name, demo.summary)
        written.append(path)
    return written


def copy_demo_project(demo: DemoProject, target: Path) -> Path:
    """Copy *demo*'s pre-built file to *target*, and return the path written.

    The application does not build a demo on the spot any more: several of
    the source datasets (four thousand drivers, three years of daily prices)
    are large enough that rebuilding one on every click would make "Create
    demo" feel like it had hung. Run ``_make_demo_project.py`` once to
    populate :data:`DEMO_DIR`; after that, "Load demo" is just a file copy.
    """
    source = demo.source_path
    if not source.is_file():
        raise FileNotFoundError(
            f"Demo project not built: {source}. "
            "Run _make_demo_project.py to build the demo set into "
            f"{DEMO_DIR}."
        )

    target = SqliteRepo.ensure_dhub_extension(Path(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.copy2(source, target)
    return target


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


def _create_multi_axis_figure(repo: SqliteRepo, spec: MultiAxisFigureSpec) -> int:
    """Create one figure, all of its axes, and lay them out with a preset.

    Every axis is created first, as an ordinary 1x1-grid-worth of a figure -
    what row/column it ends up in is the layout preset's job, applied
    afterwards through apply_axis_layout exactly as Figure Properties would
    when a person picks that same preset.
    """
    figure_id = repo.create_figure_descriptor(
        name=spec.name,
        nrows=1,
        ncols=1,
        options={"mpl_style": DEMO_STYLE, "layout_mode": "constrained"},
    )

    axis_ids: list[int] = []
    for index, axis_spec in enumerate(spec.axes):
        axis_id = repo.create_axis_descriptor(
            figure_id=figure_id,
            axis_index=index,
            chart_type=axis_spec.chart_type,
            title=axis_spec.title,
            x_label=axis_spec.x_label,
            y_label=axis_spec.y_label,
            options={"title": axis_spec.title, **axis_spec.axis_options},
        )
        axis_ids.append(int(axis_id))
        for series_index, series in enumerate(axis_spec.series):
            repo.create_series_descriptor(
                axis_id=axis_id,
                series_index=series_index,
                name=series.name,
                sql_query=series.sql,
                roles=series.roles,
                style=series.style,
            )

    plan = layout_presets.plan_layout(spec.layout, axis_ids)
    repo.apply_axis_layout(
        figure_id=figure_id,
        nrows=plan.nrows,
        ncols=plan.ncols,
        placements=[(p.axis_id, p.axis_index, p.options) for p in plan.axes],
    )

    applogger.info(
        "Demo: created multi-axis figure '%s' (%s, %d axes)",
        spec.name, spec.layout, len(spec.axes),
    )
    return int(figure_id)


def main() -> None:
    """Write the whole demo set, or one project, to the requested path.

    With no arguments, writes the whole set into :data:`DEMO_DIR` - the usual
    case, run once after cloning and again whenever the demo set changes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write a single .dhub file (the complete project) to PATH "
        "instead of building the whole set.",
    )
    parser.add_argument(
        "--all",
        metavar="DIRECTORY",
        default=str(DEMO_DIR),
        help=f"Write every demo project into DIRECTORY (default: {DEMO_DIR}).",
    )
    args = parser.parse_args()

    if args.output:
        written = build_demo_project(Path(args.output))
        print(f"Demo project written to {written}")
        return

    for path in build_demo_projects(Path(args.all)):
        print(f"Demo project written to {path}")


if __name__ == "__main__":
    main()
