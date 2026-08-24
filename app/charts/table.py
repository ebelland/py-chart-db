"""Table renderer: a data table drawn on an axis instead of a plot.

Matplotlib's own ``Axes.table`` - rows and columns of text, not marks on a
scale, so there is nothing for an x/y role to mean here.  What the roles do
mean is *which* columns, and in what order: ``column_1`` .. ``column_8`` are
optional slots, so the chart picker offers the same column choosers every
other renderer does instead of an empty panel saying no roles are declared.
Map none of them and every column the query returns is drawn, which is what
a table did before the slots existed and what ``SELECT *`` still produces.

The headers stay the source columns' own names.  A role aliases its column in
the SQL, so the DataFrame arrives carrying ``column_1`` where it once carried
``region`` - and a table headed "column_1" would be a table nobody can read.
The original names come back from the series' role map, which is why
``SeriesData`` carries one.

Only the first selected series is drawn; a table has nowhere to put a second
one that would not either overlap it or replace it, so later series are
logged and skipped rather than silently merged into something the query never
asked for.
"""
from __future__ import annotations

from typing import Any

from app.charts.base import BaseAxisRenderer, SeriesData
from app.logs.logger import applogger


#: The column slots the picker offers.  Eight because a table wider than
#: that is unreadable at figure size long before it runs out of slots, and a
#: fixed list is what the role panel needs - it builds one chooser per name.
COLUMN_ROLES: list[str] = [f"column_{index}" for index in range(1, 9)]


class TableAxisRenderer(BaseAxisRenderer):
    """Renders one series' rows and columns as a table.

    Role columns: ``column_1`` .. ``column_8``, all optional.  Mapping some
    of them draws those columns, in slot order; mapping none draws every
    column the query returns, in the order the query names them.
    """

    Name: str = "Table"
    Category: str = "Pairwise data"
    Description: str = "A data table, rows and columns of text rather than a plot."
    Link: str = "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.table.html"

    RequiredRoles: list[str] = []
    OptionalRoles: list[str] = [
        "column_1", "column_2", "column_3", "column_4",
        "column_5", "column_6", "column_7", "column_8",
    ]

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
        df, headers = self._columns_to_draw(sd)
        if df.empty or not headers:
            return
        self._draw_table(ax, df, headers, merged)
        ax.axis("off")
        self.apply_annotations(ax, axis_options)

    def _columns_to_draw(self, sd: SeriesData) -> tuple[Any, list[str]]:
        """Return the frame to draw and the header for each of its columns.

        Two shapes arrive here.  A series whose roles were mapped in the chart
        picker carries ``column_1``.. columns, in which case those are drawn in
        slot order and headed with the source columns the roles name.  A series
        with no roles mapped carries the query's own columns - ``SELECT *``, or
        SQL written by hand - and every one of them is drawn under its own
        name.

        A slot naming a column the query no longer returns is skipped rather
        than drawn empty: the SQL is the authority on what came back, and a
        blank column headed with a name is a column that looks like missing
        data instead of a mapping that has gone stale.
        """
        roles = sd.roles if isinstance(sd.roles, dict) else {}
        selected = [
            (role, str(roles.get(role) or role))
            for role in COLUMN_ROLES
            if role in sd.df.columns
        ]
        if not selected:
            return sd.df, [str(column) for column in sd.df.columns]

        return (
            sd.df[[role for role, _header in selected]],
            [header for _role, header in selected],
        )

    def _draw_table(
        self,
        ax: Any,
        df: Any,
        col_labels: list[str],
        options: dict[str, Any],
    ) -> None:
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
