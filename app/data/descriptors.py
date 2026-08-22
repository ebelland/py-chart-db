"""Dataclasses describing a figure, its axes, and their series.

These mirror the three descriptor tables in the database
(``__figure_descriptors__``, ``__axis_descriptors__``, ``__series_descriptors__``)
and are what the render pipeline consumes.  ``SqliteRepo.load_figure_descriptor``
is the only place that builds the tree.

The nesting is the same in both directions: a figure owns its axes, an axis
owns its series, and each level keeps the id of its parent so a single node can
be updated without walking down from the root.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SeriesDescriptor:
    """One plotted series: where its data comes from and how it looks.

    ``roles`` maps a renderer's role names ("x", "y", "value", ...) to columns
    of ``sql_query``; ``style`` carries whatever the renderer accepts, plus the
    markers the series-operation dialogs use to find series they generated.
    """

    id: int
    axis_id: int
    series_index: int
    name: str
    sql_query: str
    roles: dict[str, Any] | None
    style: dict[str, Any] | None


@dataclass(slots=True)
class AxisDescriptor:
    """One axis of a figure, its renderer, and the series drawn on it.

    ``chart_type`` is a renderer's ``Name``, resolved through the scanner at
    render time rather than stored as a class reference, so a project file
    stays readable and survives a renderer being renamed in code.
    """

    id: int
    figure_id: int
    axis_index: int
    chart_type: str
    title: str
    x_label: str
    y_label: str
    z_label: str = ""
    options: dict[str, Any] | None = None
    series: list[SeriesDescriptor] = field(default_factory=list)


@dataclass(slots=True)
class FigureDescriptor:
    """A figure: its grid, its options, and its axes.

    ``nrows``/``ncols`` are the subplot grid.  An axis whose ``axis_index``
    falls outside it is never drawn, which is why the operation dialogs grow
    the grid when they add one.
    """

    id: int
    name: str
    nrows: int
    ncols: int
    options: dict[str, Any] | None = None
    axes: list[AxisDescriptor] = field(default_factory=list)
