"""Table renderer: a data table drawn on an axis instead of a plot.

Matplotlib's own ``Axes.table`` - rows and columns of text, not marks on a
scale, so there is no x/y role system here: the series' own columns become
the table's columns, and its rows become the table's rows.  Only the first
selected series is drawn; a table has nowhere to put a second one that would
not either overlap it or replace it, so later series are logged and skipped
rather than silently merged into something the query never asked for.
"""
from __future__ import annotations

from typing import Any

from app.charts.base import BaseAxisRenderer, SeriesData
from app.logs.logger import applogger


class TableAxisRenderer(BaseAxisRenderer):
    """Renders one series' rows and columns as a table.

    Role columns: none - every column the series query returns becomes a
    table column, in the order the query names them.
    """

    Name: str = "Table"
    Category: str = "Pairwise data"
    Description: str = "A data table, rows and columns of text rather than a plot."
    Link: str = "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.table.html"

    RequiredRoles: list[str] = []
    OptionalRoles: list[str] = []

    Kwargs: dict[str, object] = {
        "loc": {
            "default": "center",
            "type": [
                "center", "top", "bottom", "left", "right",
                "upper left", "upper right", "lower left", "lower right",
            ],
            "group": "Layout",
            "description": "Where the table sits within the axes.",
        },
        "cellLoc": {
            "default": "center",
            "type": ["left", "center", "right"],
            "group": "Layout",
            "description": "Text alignment inside each cell.",
        },
        "colLoc": {
            "default": "center",
            "type": ["left", "center", "right"],
            "group": "Layout",
            "description": "Text alignment inside the column header row.",
        },
        "fontsize": {
            "default": None,
            "type": float,
            "min": 4.0,
            "max": 48.0,
            "group": "Appearance",
            "description": "Cell text size. Left at Matplotlib's automatic size when empty.",
        },
        "row_labels_from_index": {
            "default": False,
            "type": bool,
            "group": "Data",
            "description": "Label rows with the query's own row order/index instead of leaving them unlabelled.",
        },
    }

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        valid_series = [
            sd
            for sd in series
            if (sd.style or {}).get("visible", True) and not sd.df.empty
        ]
        if not valid_series:
            return

        if len(valid_series) > 1:
            applogger.info(
                "Table renders one series; %d more selected on this axis were "
                "not drawn.",
                len(valid_series) - 1,
            )

        sd = valid_series[0]
        merged = self._merge_options(axis_options, sd.style or {})
        self._draw_table(ax, sd.df, merged)
        ax.axis("off")
        self.apply_annotations(ax, axis_options)

    def _draw_table(self, ax: Any, df: Any, options: dict[str, Any]) -> None:
        col_labels = [str(column) for column in df.columns]
        cell_text = df.astype(str).to_numpy().tolist()
        row_labels = (
            [str(value) for value in df.index]
            if bool(self.opt("row_labels_from_index", options))
            else None
        )

        kwargs = self.get_kwargs(options)
        kwargs.pop("row_labels_from_index", None)
        fontsize = kwargs.pop("fontsize", None)
        kwargs = {key: value for key, value in kwargs.items() if value is not None and value != ""}

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            rowLabels=row_labels,
            **kwargs,
        )
        if fontsize:
            table.auto_set_font_size(False)
            table.set_fontsize(float(str(fontsize)))
        table.auto_set_column_width(col=list(range(len(col_labels))))

    def _merge_options(self, axis_options: dict[str, Any], style: dict[str, Any]) -> dict[str, Any]:
        merged = dict(axis_options or {})
        axis_kwargs = dict(merged.get("axis_kwargs", {}) or {})
        axis_kwargs.update(style.get("axis_kwargs", {}) or {})
        for key, value in style.items():
            if key != "axis_kwargs":
                merged[key] = value
        merged["axis_kwargs"] = axis_kwargs
        return merged
