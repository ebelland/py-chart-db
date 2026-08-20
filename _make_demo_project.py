"""Build a demo ``.dhub`` project showing what the application can do.

Run it directly::

    python -m app.tests.make_demo_project --output "Demo Project.dhub"

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
def _samples(rng: np.random.Generator, per_group: int = 120) -> pd.DataFrame:
    """Four groups with deliberately different shapes.

    Small on purpose.  A demo project exists to be opened and understood, and
    a hundred points per group show the difference between a normal and a
    skewed distribution as clearly as ten thousand would - while keeping the
    file small enough to open instantly and to attach to a bug report.

    The shapes are chosen so that each of the four figures built on this table
    has something to say: a box plot alone would not distinguish the last two.
    """
    groups = {
        # Symmetric, tight: the reference the others are read against.
        "A": rng.normal(50.0, 1.2, per_group),
        # Same centre, twice the spread.
        "B": rng.normal(50.0, 2.6, per_group),
        # Skewed.  A box plot shows this as a shifted median and little else;
        # a violin and an ECDF show the tail.
        "C": 47.0 + rng.gamma(shape=2.0, scale=1.6, size=per_group),
        # Bimodal.  This is the one a box plot actively misleads about: two
        # populations, one box, a median in the empty space between them.
        "D": np.concatenate(
            [
                rng.normal(46.5, 0.9, per_group // 2),
                rng.normal(53.5, 0.9, per_group - per_group // 2),
            ]
        ),
    }
    return pd.DataFrame(
        {
            "batch": np.repeat(list(groups), [len(v) for v in groups.values()]),
            "yield_pct": np.concatenate(list(groups.values())),
        }
    )


def _operating_points(rng: np.random.Generator, count: int = 300) -> pd.DataFrame:
    """A correlated cloud, for the confidence ellipse and the trend band.

    Correlated on purpose: an ellipse over an uncorrelated cloud is a circle
    and demonstrates nothing.
    """
    cloud = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.72], [0.72, 1.0]], size=count)
    return pd.DataFrame(
        {
            "pressure": 40.0 + 6.0 * cloud[:, 0],
            "flow": 12.0 + 2.4 * cloud[:, 1],
        }
    )


def _calibration(rng: np.random.Generator, points: int = 12) -> pd.DataFrame:
    """A calibration curve with asymmetric, level-dependent uncertainty.

    Asymmetric because symmetric error bars are the easy case and hide whether
    the renderer reads the two limits separately - which is the point of the
    gallery's own errorbar-limits example.
    """
    applied = np.linspace(0.0, 100.0, points)
    measured = 0.98 * applied + 1.2 + rng.normal(0.0, 0.6, points)

    # Uncertainty grows with the reading: the instrument under-reads more than
    # it over-reads at the top of its range.
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


def _particles(rng: np.random.Generator, count: int = 400) -> pd.DataFrame:
    """One lognormal population, for the histogram's fitted-density overlay.

    ``sigma`` is 0.7 rather than something milder, and that was measured, not
    guessed.  At sigma 0.35 the ranking puts ``gumbel_r`` a tenth of an AIC
    point ahead of ``lognorm`` - not a failure of the ranking but a fact about
    the shapes: a barely-skewed lognormal and a Gumbel are the same curve to
    within the noise of 400 samples.  A demo whose answer is a coin toss
    teaches the wrong lesson, so the sample is skewed enough for the fit to be
    decisive: lognorm wins by about 12 AIC points.
    """
    return pd.DataFrame({"diameter_um": rng.lognormal(mean=1.1, sigma=0.7, size=count)})


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _figure_specs() -> list[FigureSpec]:
    """Return the demo figures, in the order they appear as tabs.

    Six of them, each modelled on an example from Matplotlib's statistics
    gallery, and each set up differently - the point of a demo project is to
    show what the options *do*, which a row of charts sharing one configuration
    cannot.

    https://matplotlib.org/stable/gallery/statistics/index.html

    Four of the six read the same table.  That is deliberate: seeing one set of
    numbers as a box, a violin, an ECDF and a histogram is the fastest way to
    learn what each of them hides.
    """
    return [
        # gallery: "Histogram with a fitted distribution"
        FigureSpec(
            name="1 · Histogram and fit",
            chart_type="Histogram",
            title="Particle diameter, with the best-fitting distribution",
            x_label="diameter (µm)",
            y_label="",
            axis_options={
                "grid": True,
                "bins": 24,
                "histtype": "stepfilled",
                "alpha": 0.55,
                # Ranks the candidates and draws the winner's density over
                # the bars.  The sample is lognormal and the curve is labelled
                # with whichever distribution actually won, so this figure is
                # also a check: if it stops saying "lognorm fit", the ranking
                # has changed underneath.
                "distribution_fit": "best",
            },
            series=[
                SeriesSpec(
                    name="Particles",
                    sql="SELECT diameter_um AS value FROM particles",
                    roles={"value": "value"},
                    style={"label": "particles"},
                ),
            ],
        ),
        # gallery: "Box plots with custom fill colors"
        FigureSpec(
            name="2 · Box plot",
            chart_type="Box Plot",
            title="Yield per batch — notched, with means shown",
            x_label="batch",
            y_label="yield (%)",
            axis_options={
                "grid": True,
                # Notches give a rough visual test of whether two medians
                # differ; means as well as medians, because batch D's are far
                # apart and that is the tell that it is bimodal.
                "notch": True,
                "showmeans": True,
                "meanline": True,
                "widths": 0.5,
            },
            series=[
                SeriesSpec(
                    name="Batches",
                    sql='SELECT yield_pct AS value, batch AS "group" FROM batch_yields',
                    roles={"value": "value", "group": "group"},
                    style={},
                ),
            ],
        ),
        # gallery: "Violin plot customization"
        FigureSpec(
            name="3 · Violin plot",
            chart_type="Violin Plot",
            title="The same batches, as densities",
            x_label="batch",
            y_label="yield (%)",
            axis_options={
                "grid": True,
                # Quartiles drawn on the violin, so this figure can be read
                # against the box plot beside it directly.  Batch D shows two
                # bulges here and one box there.
                "quantiles": "0.25, 0.5, 0.75",
                "showmedians": True,
                "showextrema": False,
                "widths": 0.7,
            },
            series=[
                SeriesSpec(
                    name="Batches",
                    sql='SELECT yield_pct AS value, batch AS "group" FROM batch_yields',
                    roles={"value": "value", "group": "group"},
                    style={},
                ),
            ],
        ),
        # gallery: "Empirical cumulative distribution functions"
        FigureSpec(
            name="4 · ECDF",
            chart_type="ECDF",
            title="Batches A and C, with fitted curves",
            x_label="yield (%)",
            y_label="",
            axis_options={
                "grid": True,
                # The honest picture of a fit: no bin width to choose, so the
                # gap between the steps and the curve is the KS statistic
                # itself, visible.
                "distribution_fit": "norm",
                "marker": ".",
                "linestyle": "-",
            },
            series=[
                SeriesSpec(
                    name=f"Batch {batch}",
                    sql=(
                        "SELECT yield_pct AS value FROM batch_yields "
                        f"WHERE batch = '{batch}'"
                    ),
                    roles={"value": "value"},
                    style={"label": f"batch {batch}"},
                )
                for batch in ("A", "C")
            ],
        ),
        # gallery: "Plot a confidence ellipse of a two-dimensional dataset"
        FigureSpec(
            name="5 · Confidence ellipse",
            chart_type="Scatter Plot",
            title="Operating points, with a 2σ ellipse and a trend band",
            x_label="pressure (bar)",
            y_label="flow (l/s)",
            axis_options={
                "grid": True,
                # The ellipse tilts with the correlation; the band is the
                # uncertainty of the fitted line, not of a future point.
                "confidence_ellipse": 2.0,
                "trend_degree": 1,
                "trend_band": "confidence",
                "alpha": 0.5,
            },
            series=[
                SeriesSpec(
                    name="Operating points",
                    sql="SELECT pressure AS x, flow AS y FROM operating_points",
                    roles={"x": "x", "y": "y"},
                    style={"marker": ".", "linestyle": ""},
                ),
            ],
        ),
        # gallery: "Errorbar limit selection"
        FigureSpec(
            name="6 · Error bars",
            chart_type="Scatter Plot",
            title="Calibration curve with asymmetric uncertainty",
            x_label="applied (units)",
            y_label="measured (units)",
            axis_options={"grid": True, "capsize": 3.0, "elinewidth": 1.1},
            series=[
                SeriesSpec(
                    name="Run 1",
                    sql=(
                        "SELECT applied AS x, measured AS y, "
                        "applied_err AS xerr, err_low AS yerr_low, "
                        "err_high AS yerr_high FROM calibration ORDER BY applied"
                    ),
                    roles={"x": "x", "y": "y"},
                    style={"marker": "o", "linestyle": "-", "alpha": 0.95},
                ),
            ],
        ),
    ]


def build_demo_project(db_path: Path) -> Path:
    """Create the demo project and return the path actually written."""
    rng = np.random.default_rng(SEED)

    db_path = SqliteRepo.ensure_dhub_extension(Path(db_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    repo = SqliteRepo(db_path=db_path)

    tables = {
        "batch_yields": _samples(rng),
        "operating_points": _operating_points(rng),
        "calibration": _calibration(rng),
        "particles": _particles(rng),
    }
    for name, frame in tables.items():
        repo.import_dataframe(frame, table_name=name, normalize_columns=False)
        applogger.info("Demo: wrote table %s (%d rows)", name, len(frame))

    # Two saved queries, so the query builder has something in it: one
    # aggregate that would be tedious to rebuild by hand, and one filter.
    repo.save_query(
        "batch_summary",
        "SELECT batch, COUNT(*) AS n, AVG(yield_pct) AS mean_yield "
        "FROM batch_yields GROUP BY batch ORDER BY batch",
    )
    repo.save_query(
        "batch_d_only",
        "SELECT yield_pct FROM batch_yields WHERE batch = 'D'",
    )

    for spec in _figure_specs():
        _create_figure(repo, spec)

    _create_layout_showcase_figure(repo)

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


def _create_layout_showcase_figure(repo: SqliteRepo) -> int:
    """Create the "non-uniform layout" demo figure.

    Every other figure in this project is one chart type on a 1x1 grid. This
    one exists to show the two figure-level features that a single-axis demo
    cannot: a grid layout other than uniform cells (one axis spanning both
    columns of the top row, via ``col_span``), and figure-level
    customization (``frameon: False``, explicit subplot margins instead of a
    layout engine, and a suptitle) applied across mixed chart types sharing
    one figure.
    """
    figure_id = repo.create_figure_descriptor(
        name="7 · Subplots and layout",
        nrows=2,
        ncols=2,
        options={
            "mpl_style": DEMO_STYLE,
            # "none" hands layout entirely to the margins below, rather than
            # to constrained/tight/compressed - the demo for that combination
            # is every other figure in this project, which all use
            # "constrained" instead.
            "layout_mode": "none",
            "margins": {
                "left": 0.06,
                "right": 0.97,
                "bottom": 0.09,
                "top": 0.84,
                "wspace": 0.28,
                "hspace": 0.4,
            },
            # The figure frame is the rectangle Matplotlib draws around the
            # whole canvas, separate from each axis's own spines - this is
            # the "Frame on" checkbox in the Figure panel. Off here so the
            # figure blends into the surrounding chart tab instead of
            # nesting a second border inside it.
            "frameon": False,
            "suptitle": "One figure, three renderers, a non-uniform grid",
        },
    )

    # Top axis: spans both columns of a 2x2 grid (axis_index=0, col_span=2),
    # so it reads as a full-width header chart over the two narrower axes
    # below it rather than one cell among four equal ones.
    top_axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=0,
        chart_type="Scatter Plot",
        title="Operating points (top axis, spans both columns)",
        x_label="pressure (bar)",
        y_label="flow (l/s)",
        options={"grid": True, "col_span": 2, "alpha": 0.6},
    )
    repo.create_series_descriptor(
        axis_id=top_axis_id,
        series_index=0,
        name="Operating points",
        sql_query="SELECT pressure AS x, flow AS y FROM operating_points",
        roles={"x": "x", "y": "y"},
        style={"marker": ".", "linestyle": ""},
    )

    bottom_left_axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=2,
        chart_type="Histogram",
        title="Particle diameter",
        x_label="diameter (µm)",
        y_label="",
        options={"grid": True, "bins": 20, "alpha": 0.7},
    )
    repo.create_series_descriptor(
        axis_id=bottom_left_axis_id,
        series_index=0,
        name="Particles",
        sql_query="SELECT diameter_um AS value FROM particles",
        roles={"value": "value"},
        style={"label": "particles"},
    )

    bottom_right_axis_id = repo.create_axis_descriptor(
        figure_id=figure_id,
        axis_index=3,
        chart_type="Box Plot",
        title="Yield per batch",
        x_label="batch",
        y_label="yield (%)",
        options={"grid": True, "notch": True, "widths": 0.5},
    )
    repo.create_series_descriptor(
        axis_id=bottom_right_axis_id,
        series_index=0,
        name="Batches",
        sql_query='SELECT yield_pct AS value, batch AS "group" FROM batch_yields',
        roles={"value": "value", "group": "group"},
        style={},
    )

    applogger.info("Demo: created layout showcase figure (id=%s)", figure_id)
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