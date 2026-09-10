"""Everything drawn on an axis that is not data: annotations and lines.

Both are stored in the axis descriptor's options, both are drawn by
``BaseAxisRenderer.apply_annotations`` after the series, and neither has
anything to do with the scale, the ticks or the drawing kwargs the Axis
properties panel is about. They lived there anyway, as a third tab behind
"Axis options" and "Kwargs", which is where anything axis-shaped ends up
when there is nowhere else to put it.

Here they have somewhere else. The panel follows whichever axis the Axis
properties panel has selected - one axis selector in the application, not
two that can disagree - and shows the axis it is editing at the top so
that is never in doubt.

**Annotations** are text, boxed text or an arrow at a data point: moved
here unchanged, down to the JSON kwargs column, so a figure edited before
this panel existed loads into it exactly as it was.

**Lines** are the axis-wide horizontal and vertical ones - a threshold, a
specification limit, a target. Drawn with ``axhline``/``axvline``, so the
line spans the axes and stays where it was put when the data changes
underneath it, which is the difference from adding a two-point series that
happens to look like a line. See todo.txt N-8, which asked for these from
the chart's own context menu; the storage and the drawing are here, and
the context menu can write the same key when it arrives.
"""
from __future__ import annotations

import json
from typing import Any, Final, cast

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.logs.logger import applogger
from app.styles.style import (
    MARGIN_PANEL,
    apply_card_layout,
    create_action_button,
    create_card_widget,
    create_section_title,
    stdSizeAndlayout,
)
from app.utils.i18n import _
from app.widgets.base_properties import BaseProperties

#: The three shapes ``BaseAxisRenderer.apply_annotation`` knows how to draw.
ANNOTATION_TYPES: Final[tuple[str, ...]] = ("arrow", "text", "boxed text")

#: Which way a reference line runs. The stored value is what
#: ``apply_reference_line`` reads; the label is what the user picks.
LINE_ORIENTATIONS: Final[tuple[tuple[str, str], ...]] = (
    ("Vertical", "vertical"),
    ("Horizontal", "horizontal"),
)

#: Height the two tables ask for. Small enough that the panel fits a narrow
#: properties column, large enough to show three or four rows without
#: scrolling - which is how many annotations a figure usually has.
TABLE_MIN_HEIGHT: Final[int] = 120


class OverlayPropertiesWidget(BaseProperties):
    """Edit the annotations and reference lines of the selected axis."""

    #: Emitted with ``{"axis_id": int, "annotations": [...], "lines": [...]}``.
    #: Deliberately *not* the axis panel's own signal: that payload carries
    #: the whole axis - renderer, projection, hide_axis - and the window
    #: writes all of it, so sending a partial one from here would clear
    #: whatever it left out.
    overlay_options_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_axis_id: int | None = None
        self._build_ui()
        self._install_auto_apply(self._emit_overlay_options_requested)
        self._annotations_table.itemChanged.connect(self._queue_auto_apply)
        self._lines_table.itemChanged.connect(self._queue_auto_apply)
        self.clear_connected_figure()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(*MARGIN_PANEL)
        root.setSpacing(14)

        root.addWidget(self._build_axis_card(), 0)

        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("overlayPropertiesTabs")
        self._tabs.setDocumentMode(True)
        self._tabs.setMinimumHeight(0)
        self._tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._tabs.addTab(self._build_annotations_tab(), _("Annotations"))
        self._tabs.addTab(self._build_lines_tab(), _("Lines"))
        root.addWidget(self._tabs, 1)

    def _build_axis_card(self) -> QWidget:
        """Name the axis being edited, and carry the Apply button.

        A label rather than a second axis combo: two selectors for one
        choice is two things to keep in step, and the one in the Axis
        properties panel is already where a user goes to pick an axis.
        """
        card = create_card_widget(self, "overlayAxisCard")
        layout = QVBoxLayout(card)
        apply_card_layout(layout)

        layout.addWidget(create_section_title(_("Axis"), card))
        self._axis_label = QLabel(_("No axis selected"), card)
        self._axis_label.setWordWrap(True)
        layout.addWidget(self._axis_label)

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)
        self._btn_apply = create_action_button(
            parent=card,
            action_id="apply",
            action=self._emit_overlay_options_requested,
            layout=action_row,
        )
        action_row.addStretch(1)
        layout.addLayout(action_row)
        return card

    def _build_annotations_tab(self) -> QWidget:
        """The annotations table, moved from the Axis properties panel.

        Stored axis option format, unchanged::

            {"annotations": [{"x": 1.0, "y": 2.0, "type": "arrow",
                              "text": "Label",
                              "kwargs": {"xytext": [10, 10]}}]}

        ``kwargs`` is edited as JSON so that Matplotlib options such as
        ``arrowprops``, ``bbox``, ``xycoords``, ``textcoords``, ``ha`` and
        ``va`` can be stored without a widget for every possible key.
        """
        card = create_card_widget(self._tabs, "overlayAnnotationsCard")
        layout = QVBoxLayout(card)
        apply_card_layout(layout)

        self._annotations_table = self._build_table(
            card,
            "axisAnnotationsTable",
            (_("X"), _("Y"), _("Type"), _("Text"), _("Kwargs JSON")),
            _(
                "Annotations are stored in axis options. Kwargs must be JSON, "
                "for example: {\"xytext\": [10, 10], \"textcoords\": \"offset points\"}."
            ),
        )
        layout.addWidget(self._annotations_table, 1)

        self._btn_add_annotation, self._btn_delete_annotation = self._build_row_buttons(
            card,
            layout,
            add_text=_("Add annotation"),
            on_add=self._add_annotation_row,
            on_delete=lambda: self._delete_selected_rows(self._annotations_table),
        )
        return card

    def _build_lines_tab(self) -> QWidget:
        """The reference-line table.

        Stored axis option format::

            {"lines": [{"orientation": "horizontal", "value": 2.5,
                        "kwargs": {"color": "red", "linestyle": "--"}}]}

        The same JSON kwargs column as the annotations, and for the same
        reason: ``axhline`` takes every Line2D property there is, and a
        widget per property would be a worse editor than a text field.
        """
        card = create_card_widget(self._tabs, "overlayLinesCard")
        layout = QVBoxLayout(card)
        apply_card_layout(layout)

        self._lines_table = self._build_table(
            card,
            "axisLinesTable",
            (_("Orientation"), _("Value"), _("Kwargs JSON")),
            _(
                "A vertical or horizontal line across the whole axes, at a "
                "value in data coordinates. Kwargs must be JSON, for "
                "example: {\"color\": \"red\", \"linestyle\": \"--\", "
                "\"label\": \"limit\"}."
            ),
        )
        layout.addWidget(self._lines_table, 1)

        self._btn_add_line, self._btn_delete_line = self._build_row_buttons(
            card,
            layout,
            add_text=_("Add line"),
            on_add=self._add_line_row,
            on_delete=lambda: self._delete_selected_rows(self._lines_table),
        )
        return card

    def _build_table(
        self,
        parent: QWidget,
        object_name: str,
        headers: tuple[str, ...],
        tooltip: str,
    ) -> QTableWidget:
        """One table, built the same way for both tabs."""
        table = QTableWidget(0, len(headers), parent)
        table.setObjectName(object_name)
        table.setHorizontalHeaderLabels(list(headers))
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table.setMinimumHeight(TABLE_MIN_HEIGHT)
        table.setToolTip(tooltip)

        # Every column to its content, and the last - the JSON kwargs in
        # both tables - takes the rest. Without this the columns are equal
        # sixths of a narrow panel: "Horizontal" reads as "Horiz", and the
        # kwargs, which are the long ones, get no more room than "X".
        header = table.horizontalHeader()
        for column in range(len(headers) - 1):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        header.setStretchLastSection(True)
        return table

    def _build_row_buttons(
        self,
        parent: QWidget,
        layout: QVBoxLayout,
        *,
        add_text: str,
        on_add: Any,
        on_delete: Any,
    ) -> tuple[QPushButton, QPushButton]:
        """The Add/Delete pair under a table."""
        row = QWidget(parent)
        row_layout = QHBoxLayout(row)
        stdSizeAndlayout(row_layout)

        add = QPushButton(add_text, row)
        delete = QPushButton(_("Delete selected"), row)
        add.clicked.connect(lambda _checked=False: on_add())
        delete.clicked.connect(lambda _checked=False: on_delete())
        row_layout.addWidget(add)
        row_layout.addWidget(delete)
        row_layout.addStretch(1)
        layout.addWidget(row, 0)
        return add, delete

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_axis(self, axis_id: int | None) -> None:
        """Follow the axis the Axis properties panel has selected."""
        self._current_axis_id = None if axis_id is None else int(axis_id)
        self.reload_controls()

    def current_axis_id(self) -> int | None:
        return self._current_axis_id

    def clear_connected_figure(self) -> None:
        self._current_axis_id = None
        self._annotations_table.setRowCount(0)
        self._lines_table.setRowCount(0)
        self._axis_label.setText(_("No axis selected"))
        super().clear_connected_figure()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _reload_from_descriptor(self) -> None:
        """Fill both tables from the selected axis's stored options."""
        self._annotations_table.setRowCount(0)
        self._lines_table.setRowCount(0)

        options = self._axis_options()
        if options is None:
            self._axis_label.setText(_("No axis selected"))
            self._set_enabled_state(False)
            return

        self._axis_label.setText(self._axis_title(options))
        self._load_annotations(options)
        self._load_lines(options)
        self._set_enabled_state(True)

    def _axis_options(self) -> dict[str, Any] | None:
        """Return the selected axis's options, or None when there is none."""
        if self._repo is None or self._current_axis_id is None:
            return None
        try:
            options = self._repo.get_axis_options(int(self._current_axis_id))
        except Exception:  # noqa: BLE001 - a deleted axis is not an error here
            applogger.exception("Could not read the axis options for the overlays.")
            return None
        return dict(options or {})

    def _axis_title(self, options: dict[str, Any]) -> str:
        """Name the axis being edited the way the axis selector names it.

        The descriptor's own title first - the options copy of it is a
        fallback for figures written before the column existed - because
        this label's only job is to match what the user picked in the Axis
        properties panel.
        """
        title = ""
        if (
            self._repo is not None
            and self._figure_id is not None
            and self._current_axis_id is not None
        ):
            try:
                axes = self._repo.list_axes_for_figure(int(self._figure_id))
            except Exception:  # noqa: BLE001 - the label is not worth raising for
                axes = []
            for axis_id, _index, axis_title in axes:
                if int(axis_id) == int(self._current_axis_id):
                    title = str(axis_title or "").strip()
                    break

        title = title or str(options.get("title") or options.get("label") or "").strip()
        return title or _("Axis {id}").format(id=self._current_axis_id)

    def _load_annotations(self, options: dict[str, Any]) -> None:
        annotations = options.get("annotations", [])
        if not isinstance(annotations, list):
            applogger.warning(
                "Invalid axis annotations=%r", annotations,
                show_dialog=False, raise_error=False,
            )
            return
        for item in annotations:
            if isinstance(item, dict):
                self._add_annotation_row(cast(dict[str, Any], item))

    def _load_lines(self, options: dict[str, Any]) -> None:
        lines = options.get("lines", [])
        if not isinstance(lines, list):
            applogger.warning(
                "Invalid axis lines=%r", lines,
                show_dialog=False, raise_error=False,
            )
            return
        for item in lines:
            if isinstance(item, dict):
                self._add_line_row(cast(dict[str, Any], item))

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------
    def _add_annotation_row(self, annotation: dict[str, Any] | None = None) -> None:
        annotation = dict(annotation or {})
        # Inserting the row and filling its default cells fires itemChanged;
        # the guard keeps that from queuing an apply for a row the user has
        # not touched yet. Their first cell edit is what commits it.
        with self._reloading_controls():
            row = self._annotations_table.rowCount()
            self._annotations_table.insertRow(row)

            combo = QComboBox(self._annotations_table)
            for annotation_type in ANNOTATION_TYPES:
                combo.addItem(annotation_type, annotation_type)
            index = combo.findData(str(annotation.get("type", "text")))
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.currentIndexChanged.connect(self._queue_auto_apply)

            self._annotations_table.setItem(row, 0, self._item(annotation.get("x", 0.0)))
            self._annotations_table.setItem(row, 1, self._item(annotation.get("y", 0.0)))
            self._annotations_table.setCellWidget(row, 2, combo)
            self._annotations_table.setItem(row, 3, self._item(annotation.get("text", "")))
            self._annotations_table.setItem(
                row, 4, self._item(self._kwargs_text(annotation.get("kwargs")))
            )

    def _add_line_row(self, line: dict[str, Any] | None = None) -> None:
        line = dict(line or {})
        with self._reloading_controls():
            row = self._lines_table.rowCount()
            self._lines_table.insertRow(row)

            combo = QComboBox(self._lines_table)
            for label, value in LINE_ORIENTATIONS:
                combo.addItem(_(label), value)
            stored = str(line.get("orientation", "vertical") or "vertical").lower()
            index = combo.findData("horizontal" if stored.startswith("h") else "vertical")
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.currentIndexChanged.connect(self._queue_auto_apply)

            self._lines_table.setCellWidget(row, 0, combo)
            self._lines_table.setItem(row, 1, self._item(line.get("value", 0.0)))
            self._lines_table.setItem(
                row, 2, self._item(self._kwargs_text(line.get("kwargs")))
            )

    def _kwargs_text(self, kwargs: Any) -> str:
        """Render a stored kwargs mapping back into the JSON column."""
        if isinstance(kwargs, dict) and kwargs:
            return json.dumps(kwargs, ensure_ascii=False)
        if isinstance(kwargs, str):
            return kwargs
        return ""

    def _item(self, value: object = "") -> QTableWidgetItem:
        item = QTableWidgetItem(str(value if value is not None else ""))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        return item

    def _delete_selected_rows(self, table: QTableWidget) -> None:
        """Delete the selected rows, or the current one if nothing is selected."""
        rows = {index.row() for index in table.selectedIndexes()}
        if not rows and table.currentRow() >= 0:
            rows = {table.currentRow()}
        if not rows:
            return
        for row in sorted(rows, reverse=True):
            table.removeRow(row)
        self._queue_auto_apply()

    def _cell_text(self, table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _parse_kwargs(self, text: str, *, what: str, row: int) -> dict[str, Any]:
        """Parse one JSON kwargs cell. A bad one costs its kwargs, not the row."""
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            applogger.warning(
                "Skipping invalid %s kwargs JSON on row %s", what, row + 1,
                show_dialog=False, raise_error=False,
            )
            return {}
        if not isinstance(parsed, dict):
            applogger.warning(
                "Skipping non-object %s kwargs on row %s", what, row + 1,
                show_dialog=False, raise_error=False,
            )
            return {}
        return cast(dict[str, Any], parsed)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _annotations_payload(self) -> list[dict[str, Any]]:
        annotations: list[dict[str, Any]] = []
        for row in range(self._annotations_table.rowCount()):
            try:
                x = float(self._cell_text(self._annotations_table, row, 0))
                y = float(self._cell_text(self._annotations_table, row, 1))
            except ValueError:
                applogger.warning(
                    "Skipping annotation row %s with invalid x/y", row + 1,
                    show_dialog=False, raise_error=False,
                )
                continue

            annotation_type = "text"
            editor = self._annotations_table.cellWidget(row, 2)
            if isinstance(editor, QComboBox):
                annotation_type = str(
                    editor.currentData() or editor.currentText()
                ).lower()
            if annotation_type not in ANNOTATION_TYPES:
                applogger.warning(
                    "Unknown annotation type %r on row %s; using text.",
                    annotation_type, row + 1,
                    show_dialog=False, raise_error=False,
                )
                annotation_type = "text"

            annotations.append(
                {
                    "x": x,
                    "y": y,
                    "type": annotation_type,
                    "text": self._cell_text(self._annotations_table, row, 3),
                    "kwargs": self._parse_kwargs(
                        self._cell_text(self._annotations_table, row, 4),
                        what="annotation",
                        row=row,
                    ),
                }
            )
        return annotations

    def _lines_payload(self) -> list[dict[str, Any]]:
        lines: list[dict[str, Any]] = []
        for row in range(self._lines_table.rowCount()):
            try:
                value = float(self._cell_text(self._lines_table, row, 1))
            except ValueError:
                applogger.warning(
                    "Skipping line row %s with no usable value", row + 1,
                    show_dialog=False, raise_error=False,
                )
                continue

            orientation = "vertical"
            editor = self._lines_table.cellWidget(row, 0)
            if isinstance(editor, QComboBox):
                orientation = str(editor.currentData() or "vertical").lower()

            lines.append(
                {
                    "orientation": orientation,
                    "value": value,
                    "kwargs": self._parse_kwargs(
                        self._cell_text(self._lines_table, row, 2),
                        what="line",
                        row=row,
                    ),
                }
            )
        return lines

    def _emit_overlay_options_requested(self) -> None:
        """Send both lists for the selected axis.

        Both every time, even when only one tab was touched: they are two
        keys of one axis's options, and a payload that carried only the
        edited one would still have to say so somehow.
        """
        if self._current_axis_id is None:
            return
        self.overlay_options_requested.emit(
            {
                "axis_id": int(self._current_axis_id),
                "annotations": self._annotations_payload(),
                "lines": self._lines_payload(),
            }
        )

    # ------------------------------------------------------------------
    # Enabled state
    # ------------------------------------------------------------------
    def _set_enabled_state(self, enabled: bool) -> None:
        for widget in (
            self._annotations_table,
            self._btn_add_annotation,
            self._btn_delete_annotation,
            self._lines_table,
            self._btn_add_line,
            self._btn_delete_line,
            self._btn_apply,
        ):
            widget.setEnabled(enabled)
