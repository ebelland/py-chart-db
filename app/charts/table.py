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

from app.charts import kwarg_spec
from app.charts.base import BaseAxisRenderer, SeriesData
from app.logs.logger import applogger
from app.utils.config import get_constant


#: How many rows a table draws before it stops and says so.
#:
#: Matplotlib's table is one Text artist per cell, laid out in Python: 1 000
#: rows of three columns is 3 000 artists and a quarter of a second, 5 000
#: rows is 15 000 artists and well over a second, and every redraw pays it
#: again. A table of a whole imported file would not be slow, it would be a
#: hung window - and it would be unreadable long before that, since 200 rows
#: at figure size is already a grey smear.
MAX_TABLE_ROWS: int = get_constant("max_table_rows", 200)

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

    #: A table fills the axes; a second one would be drawn over it.
    MaxSeries: int | None = 1

    #: Forwarded verbatim to ``ax.table``.  Three keywords, where the schema
    #: used to hold six and remove three again on the way out.
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
    }

    #: Read here and never forwarded.  ``fontsize`` is the clearest case in
    #: the application: ``ax.table`` has no such keyword, and the size is set
    #: on the returned Table afterwards - together with switching off the
    #: automatic sizing, which is the part that actually makes it take effect.
    Options: dict[str, object] = {
        "fontsize": {
            "default": None,
            kwarg_spec.RCPARAM: "font.size",
            kwarg_spec.STYLE_DEFAULT: True,
            "type": float,
            "min": 4.0,
            "max": 48.0,
            "group": "Appearance",
            "description": "Cell text size. Left at Matplotlib's automatic size when empty.",
        },
        "max_rows": {
            "default": MAX_TABLE_ROWS,
            "type": int,
            "min": 1,
            "max": 100_000,
            "group": "Data",
            "description": (
                "How many rows to draw. Matplotlib builds one text artist per "
                "cell, so a whole large table stalls the window - and is "
                "unreadable at figure size long before it does."
            ),
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
        # require_roles=False: this renderer draws whatever columns the
        # query returned rather than named ones, so there are no roles to
        # be missing.
        sd = self.single_series(
            series,
            reason="a table fills the axes",
            require_roles=False,
        )
        if sd is None:
            return
        merged = self.merge_style(axis_options, sd.style or {})
        df, headers = self._columns_to_draw(sd)
        if df.empty or not headers:
            return

        df = self._rows_to_draw(df, merged)
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
        # A table draws rows and columns, not named series - genuinely
        # table-shaped work, so this is one of the few places SeriesFrame
        # steps aside for a real DataFrame (.iloc, .astype, .index below).
        frame = sd.df.to_pandas()
        roles = sd.roles if isinstance(sd.roles, dict) else {}
        selected = [
            (role, str(roles.get(role) or role))
            for role in COLUMN_ROLES
            if role in frame.columns
        ]
        if not selected:
            return frame, [str(column) for column in frame.columns]

        return (
            frame[[role for role, _header in selected]],
            [header for _role, header in selected],
        )

    def _rows_to_draw(self, df: Any, options: dict[str, Any]) -> Any:
        """Return the leading rows of *df*, capped, and say when it was cut.

        A cap rather than a refusal: the first rows of a big table are still
        worth looking at, and the alternative - drawing all of them - is a
        window that stops responding. The message names the row count so the
        chart cannot quietly look like the whole thing.

        LIMIT in the series SQL is the real answer for a big source, and the
        log line says so, because a cap here is about the chart while a LIMIT
        is about what gets read out of the database at all.
        """
        try:
            limit = int(str(self.opt("max_rows", options) or MAX_TABLE_ROWS))
        except (TypeError, ValueError):
            limit = MAX_TABLE_ROWS
        limit = max(1, limit)

        if len(df) <= limit:
            return df

        applogger.info(
            "Table drew the first %d of %d rows. Matplotlib lays out one text "
            "artist per cell, so the whole table would stall the window; raise "
            "'Maximum rows' to draw more, or add a LIMIT to the series SQL.",
            limit,
            len(df),
        )
        return df.iloc[:limit]

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
        fontsize = self.opt("fontsize", options)

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

