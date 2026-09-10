"""The Overlay properties panel: an axis's annotations and reference lines.

Both are stored in the axis descriptor's options (``annotations`` and
``lines``) and drawn by ``BaseAxisRenderer.apply_annotations`` after the
series. This panel follows whichever axis the Axis properties panel has
selected and edits the two lists in a table each.

No Apply button: every edit queues an auto-apply through
``BaseProperties``, like the Figure/Axis/Series panels. ``color``,
``linestyle`` and the annotation ``fontfamily``/``fontsize`` each get a
real widget in their own column; whatever those do not cover stays in the
``Kwargs JSON`` column, and the widgets win over the JSON for their keys.
A value a widget cannot represent (an arbitrary hex, ``fontsize="small"``)
is left in the JSON cell rather than dropped. Named colours normalise to
hex, as the Series colour combo already does.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final, cast

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.charts.kwarg_spec import DEFAULT
from app.logs.logger import applogger
from app.styles.style import (
    MARGIN_PANEL,
    apply_card_layout,
    create_card_widget,
    create_section_title,
    stdSizeAndlayout,
)
from app.utils.i18n import _
from app.widgets.base_properties import BaseProperties
from app.widgets.color_combo import MatplotlibColorCombo
from app.widgets.line_combo import LineStyleCombo

#: The shapes ``BaseAxisRenderer.apply_annotation`` can draw.
ANNOTATION_TYPES: Final[tuple[str, ...]] = ("arrow", "text", "boxed text")

#: (label, stored value) for the reference-line orientation combo.
LINE_ORIENTATIONS: Final[tuple[tuple[str, str], ...]] = (
    ("Vertical", "vertical"),
    ("Horizontal", "horizontal"),
)

TABLE_MIN_HEIGHT: Final[int] = 120
#: Fixed width for the combo/spin columns - a colour combo sized to its
#: content is either far too wide or, capped, too narrow to read.
WIDGET_COLUMN_WIDTH: Final[int] = 122
#: Above this an annotation is a title, not a label.
_MAX_FONT_SIZE: Final[int] = 200

#: The CSS generic families Matplotlib resolves itself, offered above the
#: installed faces: a portable descriptor wants "monospace", not "Menlo".
_GENERIC_FONT_FAMILIES: Final[tuple[str, ...]] = (
    "sans-serif", "serif", "monospace", "cursive", "fantasy",
)

# Cell kinds. Plain cells (FLOAT/TEXT/ENUM) read a top-level descriptor
# key; widget cells (COLOR/LINESTYLE/FONT/SIZE) read/write one kwargs key.
_FLOAT, _TEXT, _ENUM = "float", "text", "enum"
_COLOR, _LINESTYLE, _FONT, _SIZE, _KWARGS = "color", "linestyle", "font", "size", "kwargs"

#: kwargs keys a widget column also consumes on load (Matplotlib aliases).
_KWARG_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "fontfamily": ("family", "fontname"),
    "fontsize": ("size",),
    "linestyle": ("ls",),
}


@dataclass(frozen=True)
class _Col:
    """One table column: how to build its cell and where its value lives."""

    header: str
    kind: str
    key: str = ""
    #: (label, value) pairs for an _ENUM column.
    choices: tuple[tuple[str, str], ...] = ()
    default: str = ""


_ANNOTATION_COLS: Final[tuple[_Col, ...]] = (
    _Col("X", _FLOAT, "x"),
    _Col("Y", _FLOAT, "y"),
    _Col("Type", _ENUM, "type",
         tuple((t, t) for t in ANNOTATION_TYPES), default="text"),
    _Col("Text", _TEXT, "text"),
    _Col("Color", _COLOR, "color"),
    _Col("Font", _FONT, "fontfamily"),
    _Col("Size", _SIZE, "fontsize"),
    _Col("Kwargs JSON", _KWARGS),
)
_LINE_COLS: Final[tuple[_Col, ...]] = (
    _Col("Orientation", _ENUM, "orientation", LINE_ORIENTATIONS, default="vertical"),
    _Col("Value", _FLOAT, "value"),
    _Col("Color", _COLOR, "color"),
    _Col("Style", _LINESTYLE, "linestyle"),
    _Col("Kwargs JSON", _KWARGS),
)
_WIDGET_KINDS: Final[frozenset[str]] = frozenset({_COLOR, _LINESTYLE, _FONT, _SIZE})


class OverlayPropertiesWidget(BaseProperties):
    """Edit the annotations and reference lines of the selected axis."""

    #: Emitted with ``{"axis_id": int, "annotations": [...], "lines": [...]}``.
    #: Not the axis panel's own signal: that payload carries the whole axis,
    #: so a partial one from here would clear whatever it left out.
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
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._annotations_table = self._add_tab(
            _("Annotations"), "overlayAnnotationsCard", "axisAnnotationsTable",
            _ANNOTATION_COLS, add_text=_("Add annotation"),
            tooltip=_(
                "Annotations are stored in axis options. Color, font and size "
                "have their own columns; anything else goes in Kwargs JSON, "
                "for example: {\"xytext\": [10, 10], \"textcoords\": \"offset "
                "points\"}."
            ),
        )
        self._lines_table = self._add_tab(
            _("Lines"), "overlayLinesCard", "axisLinesTable",
            _LINE_COLS, add_text=_("Add line"),
            tooltip=_(
                "A vertical or horizontal line across the whole axes, at a "
                "value in data coordinates. Color and style have their own "
                "columns; anything else goes in Kwargs JSON, for example: "
                "{\"linewidth\": 2, \"label\": \"limit\"}."
            ),
        )
        root.addWidget(self._tabs, 1)

    def _build_axis_card(self) -> QWidget:
        """Name the axis being edited - the Axis panel owns the selector."""
        card = create_card_widget(self, "overlayAxisCard")
        layout = QVBoxLayout(card)
        apply_card_layout(layout)
        layout.addWidget(create_section_title(_("Axis"), card))
        self._axis_label = QLabel(_("No axis selected"), card)
        self._axis_label.setWordWrap(True)
        layout.addWidget(self._axis_label)
        return card

    def _add_tab(
        self,
        title: str,
        card_name: str,
        table_name: str,
        cols: tuple[_Col, ...],
        *,
        add_text: str,
        tooltip: str,
    ) -> QTableWidget:
        """Build one tab (table + Add/Delete row) and return its table."""
        card = create_card_widget(self._tabs, card_name)
        layout = QVBoxLayout(card)
        apply_card_layout(layout)

        table = QTableWidget(0, len(cols), card)
        table.setObjectName(table_name)
        table.setHorizontalHeaderLabels([_(c.header) for c in cols])
        table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        table.setMinimumHeight(TABLE_MIN_HEIGHT)
        table.setToolTip(tooltip)

        # Every column to its content, the JSON kwargs (last) takes the rest,
        # the fixed-width widget columns are sized once and left alone.
        header = table.horizontalHeader()
        for index, col in enumerate(cols[:-1]):
            mode = (
                QHeaderView.ResizeMode.Interactive
                if col.kind in _WIDGET_KINDS
                else QHeaderView.ResizeMode.ResizeToContents
            )
            header.setSectionResizeMode(index, mode)
            if col.kind in _WIDGET_KINDS:
                table.setColumnWidth(index, WIDGET_COLUMN_WIDTH)
        header.setStretchLastSection(True)
        layout.addWidget(table, 1)

        row = QWidget(card)
        row_layout = QHBoxLayout(row)
        stdSizeAndlayout(row_layout)
        add = QPushButton(add_text, row)
        delete = QPushButton(_("Delete selected"), row)
        add.clicked.connect(lambda: self._add_row(table, cols))
        delete.clicked.connect(lambda: self._delete_selected_rows(table))
        row_layout.addWidget(add)
        row_layout.addWidget(delete)
        row_layout.addStretch(1)
        layout.addWidget(row, 0)

        buttons = (add, delete)
        if cols is _ANNOTATION_COLS:
            self._btn_add_annotation, self._btn_delete_annotation = buttons
        else:
            self._btn_add_line, self._btn_delete_line = buttons

        self._tabs.addTab(card, title)
        return table

    # ------------------------------------------------------------------
    # Cell widgets
    # ------------------------------------------------------------------
    def _table_combo(self, combo: QComboBox) -> QComboBox:
        stdSizeAndlayout(combo, minimum_contents_length=6)
        combo.setMaximumWidth(WIDGET_COLUMN_WIDTH)
        combo.currentIndexChanged.connect(self._queue_auto_apply)
        return combo

    def _font_combo(self, parent: QWidget) -> QComboBox:
        """Font families, "Default" first, then the CSS generics, then the
        installed faces. Not ``QFontComboBox``: that has no "no family" state."""
        combo = QComboBox(parent)
        combo.addItem(_("Default"), "")
        for generic in _GENERIC_FONT_FAMILIES:
            combo.addItem(generic, generic)
        combo.insertSeparator(combo.count())
        for family in QFontDatabase.families():
            combo.addItem(family, family)
        return combo

    def _size_spin(self, parent: QWidget) -> QSpinBox:
        spin = QSpinBox(parent)
        spin.setRange(0, _MAX_FONT_SIZE)
        spin.setSpecialValueText(_("Default"))
        spin.setMaximumWidth(WIDGET_COLUMN_WIDTH)
        spin.valueChanged.connect(self._queue_auto_apply)
        return spin

    def _make_cell(
        self, table: QTableWidget, col: _Col, data: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> QWidget | QTableWidgetItem:
        """Build one cell. Widget cells pop their kwargs key into themselves;
        a value they cannot show is put back so the JSON column keeps it."""
        if col.kind == _FLOAT:
            return self._item(data.get(col.key, 0.0))
        if col.kind == _TEXT:
            return self._item(data.get(col.key, ""))
        if col.kind == _ENUM:
            combo = QComboBox(table)
            for label, value in col.choices:
                combo.addItem(_(label) if label != value else label, value)
            stored = str(data.get(col.key, col.default) or col.default).lower()
            if col.key == "orientation":
                stored = "horizontal" if stored.startswith("h") else "vertical"
            index = combo.findData(stored)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.currentIndexChanged.connect(self._queue_auto_apply)
            return combo

        stored = self._pop_kwarg(kwargs, col.key)
        text = "" if stored is None else str(stored).strip()

        if col.kind == _COLOR:
            combo = self._table_combo(
                MatplotlibColorCombo(table, include_none=True, none_label="(none)")
            )
            if text and not (
                combo.set_current_hex(text) or combo.set_current_name(text)
            ):
                kwargs[col.key] = stored
            return combo
        if col.kind == _LINESTYLE:
            combo = self._table_combo(LineStyleCombo(table))
            if text and not combo.set_current_linestyle(text):
                kwargs[col.key] = stored
            return combo
        if col.kind == _FONT:
            combo = self._table_combo(self._font_combo(table))
            if text and not self._select_by_text(combo, text):
                kwargs[col.key] = stored
            return combo
        # _SIZE
        spin = self._size_spin(table)
        number = _coerce_number(stored)
        if number is not None and number > 0:
            spin.setValue(min(int(round(number)), _MAX_FONT_SIZE))
        elif stored is not None:
            kwargs[col.key] = stored  # non-numeric ("small") -> keep as JSON
        return spin

    def _read_cell(
        self, table: QTableWidget, col: _Col, row: int, index: int,
        out: dict[str, Any], kwargs: dict[str, Any],
    ) -> bool:
        """Read one cell into *out*/*kwargs*. Return False to drop the row."""
        if col.kind == _FLOAT:
            try:
                out[col.key] = float(self._cell_text(table, row, index))
            except ValueError:
                return False
            return True
        if col.kind == _TEXT:
            out[col.key] = self._cell_text(table, row, index)
            return True
        if col.kind == _ENUM:
            widget = table.cellWidget(row, index)
            value = (
                str(widget.currentData() or col.default).lower()
                if isinstance(widget, QComboBox)
                else col.default
            )
            if col.key == "type" and value not in ANNOTATION_TYPES:
                value = col.default
            out[col.key] = value
            return True
        if col.kind == _KWARGS:
            parsed = self._parse_kwargs(self._cell_text(table, row, index), row=row)
            parsed.update(kwargs)  # the widgets win over the JSON
            out["kwargs"] = parsed
            return True

        widget = table.cellWidget(row, index)
        if isinstance(widget, MatplotlibColorCombo):
            if widget.current_hex().strip():
                kwargs[col.key] = widget.current_hex().strip()
        elif isinstance(widget, LineStyleCombo):
            style = widget.current_linestyle().strip()
            if style and style != DEFAULT:
                kwargs[col.key] = style
        elif isinstance(widget, QSpinBox):
            if widget.value() > 0:
                kwargs[col.key] = widget.value()
        elif isinstance(widget, QComboBox):  # font
            family = str(widget.currentData() or "").strip()
            if family:
                kwargs[col.key] = family
        return True

    @staticmethod
    def _pop_kwarg(kwargs: dict[str, Any], key: str) -> Any:
        """Pop *key* and its aliases; return the first value that was set."""
        value = kwargs.pop(key, None)
        for alias in _KWARG_ALIASES.get(key, ()):
            alt = kwargs.pop(alias, None)
            if value is None:
                value = alt
        return value

    @staticmethod
    def _select_by_text(combo: QComboBox, text: str) -> bool:
        index = combo.findData(text)
        if index < 0:
            index = combo.findText(text, Qt.MatchFlag.MatchFixedString)
        if index < 0:
            return False
        combo.setCurrentIndex(index)
        return True

    def _item(self, value: object = "") -> QTableWidgetItem:
        item = QTableWidgetItem(str(value if value is not None else ""))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        return item

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
        self._annotations_table.setRowCount(0)
        self._lines_table.setRowCount(0)

        options = self._axis_options()
        if options is None:
            self._axis_label.setText(_("No axis selected"))
            self._set_enabled_state(False)
            return

        self._axis_label.setText(self._axis_title(options))
        self._load_rows(self._annotations_table, _ANNOTATION_COLS, options, "annotations")
        self._load_rows(self._lines_table, _LINE_COLS, options, "lines")
        self._set_enabled_state(True)

    def _load_rows(
        self, table: QTableWidget, cols: tuple[_Col, ...],
        options: dict[str, Any], key: str,
    ) -> None:
        items = options.get(key, [])
        if not isinstance(items, list):
            applogger.warning(
                "Invalid axis %s=%r", key, items,
                show_dialog=False, raise_error=False,
            )
            return
        for item in items:
            if isinstance(item, dict):
                self._add_row(table, cols, cast("dict[str, Any]", item))

    def _axis_options(self) -> dict[str, Any] | None:
        if self._repo is None or self._current_axis_id is None:
            return None
        try:
            options = self._repo.get_axis_options(int(self._current_axis_id))
        except Exception:  # noqa: BLE001 - a deleted axis is not an error here
            applogger.exception("Could not read the axis options for the overlays.")
            return None
        return dict(options or {})

    def _axis_title(self, options: dict[str, Any]) -> str:
        """Name the axis the way the Axis properties panel's selector does."""
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

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------
    def _add_row(
        self, table: QTableWidget, cols: tuple[_Col, ...],
        data: dict[str, Any] | None = None,
    ) -> None:
        data = dict(data or {})
        raw = data.get("kwargs")
        kwargs = dict(raw) if isinstance(raw, dict) else {}
        # Populating the cells fires itemChanged; the guard keeps that from
        # queuing an apply for a row the user has not touched yet.
        with self._reloading_controls():
            row = table.rowCount()
            table.insertRow(row)
            for index, col in enumerate(cols):
                if col.kind == _KWARGS:
                    continue  # after the widget columns have claimed their keys
                cell = self._make_cell(table, col, data, kwargs)
                if isinstance(cell, QTableWidgetItem):
                    table.setItem(row, index, cell)
                else:
                    table.setCellWidget(row, index, cell)
            table.setItem(
                row, len(cols) - 1, self._item(self._kwargs_text(kwargs))
            )

    def _delete_selected_rows(self, table: QTableWidget) -> None:
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

    def _kwargs_text(self, kwargs: Any) -> str:
        if isinstance(kwargs, dict) and kwargs:
            return json.dumps(kwargs, ensure_ascii=False)
        return kwargs if isinstance(kwargs, str) else ""

    def _parse_kwargs(self, text: str, *, row: int) -> dict[str, Any]:
        """Parse one JSON kwargs cell. A bad one costs its kwargs, not the row."""
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            applogger.warning(
                "Skipping invalid kwargs JSON on row %s", row + 1,
                show_dialog=False, raise_error=False,
            )
            return {}
        if not isinstance(parsed, dict):
            applogger.warning(
                "Skipping non-object kwargs on row %s", row + 1,
                show_dialog=False, raise_error=False,
            )
            return {}
        return cast("dict[str, Any]", parsed)

    # Kept for callers/tests that add one row at a time.
    def _add_annotation_row(self, annotation: dict[str, Any] | None = None) -> None:
        self._add_row(self._annotations_table, _ANNOTATION_COLS, annotation)

    def _add_line_row(self, line: dict[str, Any] | None = None) -> None:
        self._add_row(self._lines_table, _LINE_COLS, line)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _payload(self, table: QTableWidget, cols: tuple[_Col, ...]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(table.rowCount()):
            out: dict[str, Any] = {}
            kwargs: dict[str, Any] = {}
            if all(
                self._read_cell(table, col, row, index, out, kwargs)
                for index, col in enumerate(cols)
            ):
                rows.append(out)
            else:
                applogger.warning(
                    "Skipping overlay row %s with an unusable value", row + 1,
                    show_dialog=False, raise_error=False,
                )
        return rows

    def _emit_overlay_options_requested(self) -> None:
        """Send both lists for the selected axis - they are two keys of one
        axis's options, so a payload with only one would be ambiguous."""
        if self._current_axis_id is None:
            return
        self.overlay_options_requested.emit(
            {
                "axis_id": int(self._current_axis_id),
                "annotations": self._payload(self._annotations_table, _ANNOTATION_COLS),
                "lines": self._payload(self._lines_table, _LINE_COLS),
            }
        )

    def _set_enabled_state(self, enabled: bool) -> None:
        for widget in (
            self._annotations_table, self._btn_add_annotation,
            self._btn_delete_annotation, self._lines_table,
            self._btn_add_line, self._btn_delete_line,
        ):
            widget.setEnabled(enabled)


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
