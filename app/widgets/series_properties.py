"""Property editor for the series of the selected axis.

Handles per-series style (colour, marker, line style, alpha, visibility, legend
participation), ordering, deletion, and the series SQL query.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextOption
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.styles.style import (
    MARGIN_PANEL,
    create_action_button,
    create_card_widget,
    create_section_title,
    stdSizeAndlayout,
)
from app.widgets.color_combo import MatplotlibColorCombo
from app.widgets.line_combo import LineStyleCombo
from app.widgets.marker_combo import MarkerStyleCombo
from app.logs.logger import applogger
from app.utils.i18n import _
from app.utils.messages import ask


AxisDescriptorLike = Any
SeriesDescriptorLike = Any

SQL_QUERY_VISIBLE_LINES = 6


class SeriesPropertiesWidget(QWidget):
    """Compact editor for series descriptor style/options."""

    series_selected = Signal(int)
    series_options_requested = Signal(dict)
    series_order_requested = Signal(list)
    series_delete_requested = Signal(int)

    # Fixed height, in visible lines, for the SQL query editor. A fixed
    # size rather than Expanding keeps the form layout stable regardless
    # of query length and avoids the editor growing to swallow the panel.

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._repo: Any | None = None
        self._figure_id: int | None = None
        self._figure: Any | None = None
        self._redraw_callback: Any | None = None
        self._current_axis_id: int | None = None
        self._current_series_id: int | None = None
        self._series_map: dict[int, SeriesDescriptorLike] = {}

        self._build_ui()
        self.clear_connected_figure()

    def _build_ui(self) -> None:
        """Build all UI sections inside a resizable scrollable container.

        The selector card stays at natural height. The options card receives
        the remaining vertical space while still allowing the whole panel to
        scroll when the host becomes too small.
        """
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        content = QWidget(self)
        content.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        root = QVBoxLayout(content)
        root.setContentsMargins(*MARGIN_PANEL)
        root.setSpacing(12)

        self._selector_section = self._build_selector_section()
        root.addWidget(self._selector_section, 0)

        self._options_section = self._build_options_section()
        self._options_section.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root.addWidget(self._options_section, 1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        scroll.setWidget(content)

        main = QVBoxLayout(self)
        stdSizeAndlayout(main)
        main.addWidget(scroll, 1)

    def _build_selector_section(self) -> QWidget:
        """Create the series selector and ordering controls."""
        section = create_card_widget(self, "seriesSelectorCard")
        section.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        layout = QVBoxLayout(section)
        stdSizeAndlayout(layout)

        self._series_title = create_section_title(_("Series"), section)
        layout.addWidget(self._series_title)

        self._series_combo = QComboBox(section)
        stdSizeAndlayout(self._series_combo)
        self._series_combo.currentIndexChanged.connect(
            self._on_series_combo_changed
        )
        layout.addWidget(self._series_combo)

        row = QWidget(section)
        row_layout = QHBoxLayout(row)
        stdSizeAndlayout(row_layout)

        self._btn_move_up = create_action_button(
            parent=row,
            action_id="up",
            action=self._move_current_series_up,
            layout=row_layout,
        )
        self._btn_move_down = create_action_button(
            parent=row,
            action_id="down",
            action=self._move_current_series_down,
            layout=row_layout,
        )
        self._btn_delete = create_action_button(
            parent=row,
            action_id="delete",
            action=self._on_delete_clicked,
            layout=row_layout,
        )
        self._btn_apply = create_action_button(
            parent=row,
            action_id="apply",
            action=self._emit_series_options_requested,
            layout=row_layout,
        )

        row_layout.addStretch(1)
        layout.addWidget(row)

        return section

    def _build_options_section(self) -> QWidget:
        """Create editable series style controls."""
        section = create_card_widget(self, "seriesOptionsCard")
        stdSizeAndlayout(section)
        section.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(section)
        stdSizeAndlayout(layout)

        form = QFormLayout()
        stdSizeAndlayout(form)

        self._legend_label_edit = QLineEdit(section)
        stdSizeAndlayout(self._legend_label_edit)

        self._sql_query_edit = QPlainTextEdit(
            section,
            readOnly=False,
            placeholderText="Enter SQL query for this series.",
            undoRedoEnabled=True,
        )
        self._sql_query_edit.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth
        )
        self._sql_query_edit.setWordWrapMode(
            QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
        )
        self._sql_query_edit.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        stdSizeAndlayout(self._sql_query_edit)
        self._sql_query_edit.setFixedHeight(
            self._sql_query_edit.fontMetrics().lineSpacing()
            * SQL_QUERY_VISIBLE_LINES
            + 12
        )

        self._visible_check = QCheckBox(_("Visible"), section)
        self._show_in_legend_check = QCheckBox(_("Show in legend"), section)
        self._sort_x_check = QCheckBox(_("Sort by X ascending"), section)

        stdSizeAndlayout(self._visible_check)
        stdSizeAndlayout(self._show_in_legend_check)
        stdSizeAndlayout(self._sort_x_check)

        flags_row = QWidget(section)
        stdSizeAndlayout(flags_row)

        flags_layout = QHBoxLayout(flags_row)
        stdSizeAndlayout(flags_layout)
        flags_layout.addWidget(self._visible_check, 0)
        flags_layout.addWidget(self._show_in_legend_check, 0)
        flags_layout.addStretch(1)

        self._linestyle_combo = LineStyleCombo(section)
        self._marker_combo = MarkerStyleCombo(section)
        self._color_combo = MatplotlibColorCombo(
            section,
            include_none=True,
        )

        stdSizeAndlayout(self._linestyle_combo, minimum_contents_length=0)
        stdSizeAndlayout(self._marker_combo, minimum_contents_length=0)
        stdSizeAndlayout(self._color_combo, minimum_contents_length=0)

        form.addRow(_("Legend label"), self._legend_label_edit)
        form.addRow(_("SQL query"), self._sql_query_edit)
        form.addRow(_("Visibility"), flags_row)
        form.addRow(_("Sort"), self._sort_x_check)
        form.addRow(_("Line style"), self._linestyle_combo)
        form.addRow(_("Marker"), self._marker_combo)
        form.addRow(_("Color"), self._color_combo)

        layout.addLayout(form, 0)

        # Important:
        # If you have self.series_list in your real code, add it here,
        # outside the QFormLayout, with stretch 1.
        #
        # self.series_list.setMinimumHeight(0)
        # self.series_list.setMaximumHeight(16777215)
        # self.series_list.setSizePolicy(
        #     QSizePolicy.Policy.Expanding,
        #     QSizePolicy.Policy.Expanding,
        # )
        # layout.addWidget(self.series_list, 1)

        return section

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_connected_figure(
        self,
        repo: Any,
        figure_id: int,
        figure: Any,
        redraw_callback: Any | None = None,
    ) -> None:
        """Attach a figure and load series from its descriptor."""
        self._repo = repo
        self._figure_id = int(figure_id)
        self._figure = figure
        self._redraw_callback = redraw_callback
        self._reload_from_descriptor()
        self._set_enabled_state(True)

    def clear_connected_figure(self) -> None:
        """Reset UI and disable editing."""
        self._repo = None
        self._figure_id = None
        self._figure = None
        self._redraw_callback = None
        self._current_axis_id = None
        self._current_series_id = None
        self._series_map.clear()
        self._series_title.setText(_("Series"))
        self._clear_series_combo()
        self._clear_series_fields()
        self._set_enabled_state(False)

    def set_current_axis_id(self, axis_id: int | None) -> None:
        """Limit the series selector to one axis."""
        self._current_axis_id = int(axis_id) if axis_id is not None else None
        if self._repo is not None and self._figure_id is not None:
            self._reload_from_descriptor()

    def current_axis_id(self) -> int | None:
        """Return the current axis id filter."""
        return self._current_axis_id

    # ------------------------------------------------------------------
    # Descriptor loading
    # ------------------------------------------------------------------

    def _reload_from_descriptor(self) -> None:
        """Reload the series selector from the connected figure descriptor.

        This is the single descriptor reload path for the series panel. It is
        safe during shutdown, chart deletion, stale selection changes, and
        descriptor schema errors.
        """
        if self._repo is None or self._figure_id is None:
            self.clear_connected_figure()
            return

        descriptor = self._repo.load_figure_descriptor(self._figure_id)
        previous_series_id = self._current_series_id

        self._series_map.clear()
        self._series_combo.blockSignals(True)
        try:
            self._series_combo.clear()

            if descriptor is None:
                applogger.warning(
                    "No figure descriptor found for figure_id=%s. "
                    "Series editing disabled for this figure.",
                    self._figure_id,
                )
                return

            for axis_desc in self._axes_from_descriptor(descriptor):
                self._add_axis_series(axis_desc)
        finally:
            self._series_combo.blockSignals(False)

        self._update_title()

        if self._series_combo.count() == 0:
            self._load_series_descriptor(None)
            self._set_enabled_state(False)
            return

        self._set_enabled_state(True)
        self._select_series(previous_series_id)

    def _axes_from_descriptor(self, descriptor: Any) -> list:
        """Return axes from a validated descriptor."""
        axes = getattr(descriptor, "axes", None)
        if axes is None:
            applogger.error(
                "Figure descriptor id=%r has no axes list. "
                "Series editing stopped; please check the descriptor schema.",
                getattr(descriptor, "id", self._figure_id),
            )
            return []
        return list(axes)

    def _add_axis_series(self, axis_desc: AxisDescriptorLike) -> None:
        """Add all series for the axis when it passes the current filter."""
        axis_id = int(axis_desc.id)

        if self._current_axis_id is not None and axis_id != self._current_axis_id:
            return

        for series_desc in list(axis_desc.series or []):
            series_id = int(series_desc.id)
            self._series_map[series_id] = series_desc
            self._series_combo.addItem(
                self._series_display_label(series_desc),
                series_id,
            )

    def _select_series(self, preferred_series_id: int | None) -> None:
        """Select the previous series when possible, otherwise first."""
        index_to_select = 0

        if preferred_series_id is not None:
            for index in range(self._series_combo.count()):
                data = self._series_combo.itemData(index)
                if data is not None and int(data) == preferred_series_id:
                    index_to_select = index
                    break

        self._series_combo.setCurrentIndex(index_to_select)
        self._on_series_combo_changed(index_to_select)

    def _update_title(self) -> None:
        """Refresh section title with the current axis filter and count."""
        count = self._series_combo.count()

        if self._current_axis_id is None:
            self._series_title.setText(f"Series ({count})")
            return

        self._series_title.setText(
            f"Series for axis {self._current_axis_id} ({count})"
        )

    # ------------------------------------------------------------------
    # Series binding
    # ------------------------------------------------------------------

    def _series_style(self, series_desc: SeriesDescriptorLike) -> dict[str, Any]:
        """Return a mutable series style dictionary."""
        if series_desc.style is None:
            return {}

        if isinstance(series_desc.style, dict):
            return dict(series_desc.style)

        applogger.error(
            f"Series id={series_desc.id!r} has invalid style. Editing stopped; "
            "please check the descriptor schema."
        )
        return {}

    def _series_display_label(self, series_desc: SeriesDescriptorLike) -> str:
        """Build selector text for one series."""
        style = self._series_style(series_desc)
        name = str(
            series_desc.name
            or style.get("label")
            or f"Series {series_desc.id}"
        ).strip()

        return name if bool(style.get("visible", True)) else f"{name} (hidden)"

    def _load_series_descriptor(
        self,
        series_desc: SeriesDescriptorLike | None,
    ) -> None:
        """Load one series descriptor into the form."""
        if series_desc is None:
            self._clear_series_fields()
            return

        self._current_series_id = int(series_desc.id)
        style = self._series_style(series_desc)

        self._legend_label_edit.setText(str(style.get("label", "") or ""))
        self._sql_query_edit.setPlainText(
            str(getattr(series_desc, "sql_query", "") or "")
        )
        self._visible_check.setChecked(bool(style.get("visible", True)))
        self._show_in_legend_check.setChecked(
            bool(style.get("show_in_legend", True))
        )
        self._sort_x_check.setChecked(bool(style.get("sort_x", False)))

        self._set_combo_value(
            self._linestyle_combo,
            str(style.get("linestyle", "-") or "-"),
        )
        self._set_combo_value(
            self._marker_combo,
            str(style.get("marker", "") or ""),
        )
        self._set_color_value(str(style.get("color", "") or ""))

    def _clear_series_fields(self) -> None:
        """Clear form fields for the no-selection state."""
        self._current_series_id = None
        self._legend_label_edit.clear()
        self._sql_query_edit.clear()
        self._visible_check.setChecked(True)
        self._show_in_legend_check.setChecked(True)
        self._sort_x_check.setChecked(False)
        self._linestyle_combo.setCurrentIndex(0)
        self._marker_combo.setCurrentIndex(0)
        self._color_combo.setCurrentIndex(0)

    def _set_color_value(self, value: str) -> None:
        """Select a color by hex value or Matplotlib color name."""
        matched = False

        if value:
            matched = self._color_combo.set_current_hex(value)
            if not matched:
                matched = self._color_combo.set_current_name(value)

        if not matched:
            self._color_combo.setCurrentIndex(0)

    def _set_combo_value(self, combo: QComboBox, value: str) -> None:
        """Set combo data when present, otherwise select the first item."""
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    # ------------------------------------------------------------------
    # Events and actions
    # ------------------------------------------------------------------

    def _clear_series_combo(self) -> None:
        """Clear the selector without emitting selection-change signals."""
        self._series_combo.blockSignals(True)
        self._series_combo.clear()
        self._series_combo.blockSignals(False)

    def _set_enabled_state(self, enabled: bool) -> None:
        """Enable or disable interactive controls."""
        for widget in (
            self._series_combo,
            self._btn_move_up,
            self._btn_move_down,
            self._btn_delete,
            self._btn_apply,
            self._legend_label_edit,
            self._sql_query_edit,
            self._visible_check,
            self._show_in_legend_check,
            self._sort_x_check,
            self._linestyle_combo,
            self._marker_combo,
            self._color_combo,
        ):
            widget.setEnabled(enabled)

    def _on_series_combo_changed(self, index: int) -> None:
        """Handle series selector changes."""
        if index < 0:
            self._load_series_descriptor(None)
            return

        data = self._series_combo.itemData(index)
        if data is None:
            self._load_series_descriptor(None)
            return

        series_id = int(data)
        series_desc = self._series_map.get(series_id)

        if series_desc is None:
            applogger.error(
                "Series descriptor id=%r not found in the map. "
                "Editing stopped; please check the descriptor.",
                series_id,
            )
            self._load_series_descriptor(None)
            return

        self._load_series_descriptor(series_desc)
        self.series_selected.emit(series_id)

    def _on_delete_clicked(self) -> None:
        """Delete the selected series."""
        index = self._series_combo.currentIndex()
        if index < 0:
            return

        data = self._series_combo.itemData(index)
        if data is None:
            return

        series_id = int(data)

        if not ask(
            self,
            "series.confirm_delete",
            series=self._series_combo.itemText(index),
        ):
            return

        if series_id in self._series_map:
            del self._series_map[series_id]

        self._series_combo.removeItem(index)
        self.series_delete_requested.emit(int(series_id))

    def _move_current_series_up(self) -> None:
        """Move the selected series up in combo order."""
        self._move_current_series(-1)

    def _move_current_series_down(self) -> None:
        """Move the selected series down in combo order."""
        self._move_current_series(1)

    def _move_current_series(self, offset: int) -> None:
        """Move the selected series by one row and emit the new order."""
        index = self._series_combo.currentIndex()
        target = index + offset

        if index < 0 or target < 0 or target >= self._series_combo.count():
            return

        text = self._series_combo.itemText(index)
        data = self._series_combo.itemData(index)

        self._series_combo.removeItem(index)
        self._series_combo.insertItem(target, text, data)
        self._series_combo.setCurrentIndex(target)

        self._emit_series_order_requested()

    def _emit_series_order_requested(self) -> None:
        """Emit the current combo order."""
        ordered_ids: list[int] = []

        for index in range(self._series_combo.count()):
            data = self._series_combo.itemData(index)
            if data is not None:
                ordered_ids.append(int(data))

        self.series_order_requested.emit(ordered_ids)

    def _emit_series_delete_requested(self) -> None:
        """Emit the current series delete request."""
        series_id = self._series_combo.itemData(self._series_combo.currentIndex())
        if series_id is None:
            return

        self.series_delete_requested.emit(int(series_id))

    def _emit_series_options_requested(self) -> None:
        """Emit the current form payload."""
        self.series_options_requested.emit(
            {
                "series_id": self._current_series_id,
                "label": self._legend_label_edit.text().strip(),
                "sql_query": self._sql_query_edit.toPlainText().strip(),
                "visible": bool(self._visible_check.isChecked()),
                "show_in_legend": bool(self._show_in_legend_check.isChecked()),
                "sort_x": bool(self._sort_x_check.isChecked()),
                "linestyle": str(self._linestyle_combo.currentData() or "-"),
                "marker": str(self._marker_combo.currentData() or ""),
                "color": self._color_combo.current_hex().strip(),
            }
        )