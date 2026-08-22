"""PySide6 widget for selecting a figure, axis, and series.

The widget loads figure, axis, and series descriptor rows from ``SqliteRepo``.
It exposes the currently selected figure, axis, and checked series rows to
operation dialogs.

Layout contract
---------------
This widget deliberately does not use fixed heights. The hosting dialog or
QToolBox page must give this widget a vertical ``QSizePolicy.Expanding`` policy.
Inside this widget, the figure and axis controls keep their natural height, while
``series_list`` receives the remaining vertical space and the button row remains
at the bottom of the selector section.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.styles.style import (
    create_action_button,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _


SeriesPredicate = Callable[[Any], bool]
AxisMap = Mapping[str, Sequence[Any]]

_FIGURE_ROLE: Final[Qt.ItemDataRole] = Qt.ItemDataRole.UserRole
_AXIS_ROLE: Final[Qt.ItemDataRole] = Qt.ItemDataRole.UserRole
_SERIES_ROLE: Final[Qt.ItemDataRole] = Qt.ItemDataRole.UserRole
_QT_MAX_WIDGET_HEIGHT: Final[int] = 16777215


class AxisSeriesSelector(QWidget):
    """Select one figure, one axis, and zero or more series."""

    selection_changed = Signal(str, list)
    axis_changed = Signal(str)
    figure_changed = Signal(int)

    def __init__(
        self,
        repo: SqliteRepo,
        figure_id: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._repo = repo
        self._figure_id = int(figure_id)
        self._show_buttons = True
        self._select_all_on_load = False

        # Axis display name -> list of series descriptor rows.
        self._axes: dict[str, list[Any]] = {}

        # Axis display name -> database axis id.
        self._axis_ids_by_name: dict[str, int] = {}

        self._series_filter: SeriesPredicate | None = None

        # Used while rebuilding widgets to avoid duplicate signal cascades.
        self._signals_blocked = False

        self._configure_self_size_policy()
        self._create_widgets()
        self._build_ui()
        self._connect_signals()

        # Load available figures first, then load axes/series for current figure.
        self._load_figures()
        self.reload(select_all_series=self._select_all_on_load)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _configure_self_size_policy(self) -> None:
        """Allow the selector to fill a QToolBox page or any expanding parent."""
        self.setMinimumHeight(0)
        self.setMaximumHeight(_QT_MAX_WIDGET_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    def _create_widgets(self) -> None:
        """Create child widgets and configure compact sizing."""
        self.figure_label = QLabel(_("Figure:"), self)
        self.figure_combo = QComboBox(self)

        self.axis_label = QLabel(_("Axis:"), self)
        self.axis_combo = QComboBox(self)

        self.series_label = QLabel(_("Series:"), self)
        self.series_list = QListWidget(self, sortingEnabled=True)

        # Series are checkable, not row-selected. The list is the only widget in
        # this selector that should consume spare vertical space.
        self.series_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.series_list.setMinimumHeight(0)
        self.series_list.setMaximumHeight(_QT_MAX_WIDGET_HEIGHT)
        self.series_list.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        mark_editor_panel(self.series_list)
        stdSizeAndlayout(self.figure_combo)
        stdSizeAndlayout(self.axis_combo)

    def _build_ui(self) -> None:
        """Build the vertical layout.

        The outer layout gives stretch only to ``series_section``. The section
        layout gives stretch only to ``series_list``. Therefore the list expands
        and the button row stays at the bottom with its natural height.
        """
        button_row = QHBoxLayout()
        stdSizeAndlayout(button_row)

        self.select_all_button = create_action_button(
            parent=self,
            action_id="select_all",
            action=self.select_all_series,
            layout=button_row,
        )
        self.clear_button = create_action_button(
            parent=self,
            action_id="clear",
            action=self.clear_series_selection,
            layout=button_row,
        )

        self.select_all_button.setVisible(self._show_buttons)
        self.clear_button.setVisible(self._show_buttons)
        button_row.addStretch(1)

        series_section = QWidget(self)
        series_section.setObjectName("axisSeriesListSection")
        series_section.setMinimumHeight(0)
        series_section.setMaximumHeight(_QT_MAX_WIDGET_HEIGHT)
        series_section.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        series_section_layout = QVBoxLayout(series_section)
        stdSizeAndlayout(series_section_layout)
        series_section_layout.setSpacing(4)
        series_section_layout.addWidget(self.series_list, 1)
        series_section_layout.addLayout(button_row, 0)

        layout = QVBoxLayout(self)
        stdSizeAndlayout(layout)
        layout.addWidget(self.figure_label, 0)
        layout.addWidget(self.figure_combo, 0)
        layout.addWidget(self.axis_label, 0)
        layout.addWidget(self.axis_combo, 0)
        layout.addWidget(self.series_label, 0)
        layout.addWidget(series_section, 1)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self.figure_combo.currentIndexChanged.connect(self._on_figure_changed)
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.series_list.itemChanged.connect(self._on_series_selection_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reload(
        self,
        *,
        preferred_axis: str | None = None,
        select_all_series: bool | None = None,
    ) -> None:
        """Reload axes and series for the current figure."""
        select_all = (
            self._select_all_on_load
            if select_all_series is None
            else select_all_series
        )

        self.set_axes(
            self._load_axes_from_repo(),
            preferred_axis=preferred_axis,
            select_all_series=select_all,
        )

    def set_figure_id(
        self,
        figure_id: int,
        *,
        preferred_axis: str | None = None,
        select_all_series: bool | None = None,
    ) -> None:
        """Change the figure context and reload axes/series."""
        self._figure_id = int(figure_id)

        # Keep the combo box in sync if set_figure_id is called externally.
        figure_index = self.figure_combo.findData(self._figure_id, _FIGURE_ROLE)
        if figure_index >= 0 and figure_index != self.figure_combo.currentIndex():
            self._signals_blocked = True
            self.figure_combo.setCurrentIndex(figure_index)
            self._signals_blocked = False

        self.reload(
            preferred_axis=preferred_axis,
            select_all_series=select_all_series,
        )

    def set_figure_locked(self, locked: bool) -> None:
        """Show which figure this is, but stop it being changed.

        A series operation is constructed with one figure id and keeps it:
        ``create_result_axis`` adds to that figure, and the results are written
        to the axis selected within it. Letting the combo change figures
        desynchronises the two - the axis list would come from the figure on
        screen while the results went to the one the dialog was opened on.

        Disabled rather than hidden, because which figure is being operated on
        is worth showing even when it cannot be changed.
        """
        self.figure_combo.setEnabled(not locked)
        self.figure_combo.setToolTip(
            _("The operation runs on the figure it was opened from.")
            if locked
            else ""
        )

    def set_series_visible(self, visible: bool) -> None:
        """Show or hide the series list, for operations that read no series.

        An operation that generates a series rather than transforming one - a
        plotted function, say - still needs an axis to draw on, so the figure
        and axis rows stay. A series picker it never reads is a control that
        does nothing, which is worse than no control at all.
        """
        self.series_label.setVisible(visible)
        for widget in (self.series_list, self.select_all_button, self.clear_button):
            widget.setVisible(visible and (widget is self.series_list or self._show_buttons))

        section = self.series_list.parentWidget()
        if section is not None and section.objectName() == "axisSeriesListSection":
            section.setVisible(visible)

    def set_axes(
        self,
        axes: AxisMap,
        *,
        preferred_axis: str | None = None,
        select_all_series: bool = True,
    ) -> None:
        """Replace axis/series contents."""
        self._signals_blocked = True

        self._axes = {str(axis): list(series) for axis, series in axes.items()}
        self.axis_combo.clear()
        self.series_list.clear()

        if not self._axes:
            self.axis_combo.addItem(_("No axes found"), "")
            self._signals_blocked = False
            self._emit_axis_changed()
            self._emit_selection_changed()
            return

        selected_index = 0
        for row_index, axis_name in enumerate(self._axes):
            self.axis_combo.addItem(axis_name, axis_name)
            if preferred_axis is not None and axis_name == preferred_axis:
                selected_index = row_index

        self.axis_combo.setCurrentIndex(selected_index)
        self._signals_blocked = False

        self._rebuild_series(select_all_series=select_all_series)
        self._emit_axis_changed()
        self._emit_selection_changed()

    def set_series_filter(
        self,
        series_filter: SeriesPredicate | None,
        *,
        select_all_series: bool = False,
    ) -> None:
        """Show only series accepted by ``series_filter``."""
        self._series_filter = series_filter
        self._rebuild_series(select_all_series=select_all_series)
        self._emit_selection_changed()

    def selected_figure_id(self) -> int | None:
        """Return the selected figure database id."""
        figure_id = self.figure_combo.currentData(_FIGURE_ROLE)
        return None if figure_id is None else int(figure_id)

    def selected_axis_name(self) -> str:
        """Return the selected axis label."""
        axis_name = self.axis_combo.currentData(_AXIS_ROLE)
        return "" if axis_name is None else str(axis_name)

    def selected_axis_id(self) -> int | None:
        """Return the selected axis database id."""
        axis_id = self._axis_ids_by_name.get(self.selected_axis_name())
        return None if axis_id is None else int(axis_id)

    def selected_series(self) -> list[Any]:
        """Return checked series descriptor rows."""
        return [
            self.series_list.item(row).data(_SERIES_ROLE)
            for row in range(self.series_list.count())
            if self.series_list.item(row).checkState() == Qt.CheckState.Checked
        ]

    def current_axis_series(self) -> list[Any]:
        """Return visible series descriptor rows for the selected axis."""
        return [
            self.series_list.item(row).data(_SERIES_ROLE)
            for row in range(self.series_list.count())
        ]

    def has_axes(self) -> bool:
        """Return True if at least one axis was loaded."""
        return bool(self._axes)

    def select_all_series(self) -> None:
        """Select every visible series."""
        self._set_all_series_selected(True)

    def clear_series_selection(self) -> None:
        """Clear series selection."""
        self._set_all_series_selected(False)

    # ------------------------------------------------------------------
    # Repository loading
    # ------------------------------------------------------------------

    def _load_figures(self) -> None:
        """Load figure names into the figure selector."""
        self._signals_blocked = True
        self.figure_combo.clear()

        for figure_row in self._repo.get_figures():
            figure_id = int(figure_row[0])
            raw_name = str(figure_row[1] or "").strip()
            figure_name = raw_name if raw_name else f"Figure {figure_id}"
            self.figure_combo.addItem(figure_name, figure_id)

        current_index = self.figure_combo.findData(self._figure_id, _FIGURE_ROLE)
        if current_index >= 0:
            self.figure_combo.setCurrentIndex(current_index)

        self._signals_blocked = False

    def _load_axes_from_repo(self) -> dict[str, list[Any]]:
        """Load axes and series for the selected figure."""
        axes: dict[str, list[Any]] = {}
        self._axis_ids_by_name.clear()

        axis_rows = list(self._repo.get_axes(self._figure_id) or [])
        valid_axis_rows = [
            row
            for row in axis_rows
            if row["axis_index"] is not None
        ]

        for axis_row in valid_axis_rows:
            axis_id = int(axis_row["id"])
            axis_index = axis_row["axis_index"]
            default_title = f"Axis {axis_index}"
            axis_title = str(axis_row["title"] or "").strip()
            axis_name = f"{axis_index}: {axis_title or default_title}"

            self._axis_ids_by_name[axis_name] = axis_id
            axes[axis_name] = list(self._repo.get_series(axis_id) or [])

        return axes

    # ------------------------------------------------------------------
    # Events and mutations
    # ------------------------------------------------------------------

    def _on_figure_changed(self, index: int) -> None:
        """Reload axes/series when the selected figure changes."""
        if self._signals_blocked or index < 0:
            return

        figure_id = self.figure_combo.itemData(index, _FIGURE_ROLE)
        if figure_id is None:
            return

        self._figure_id = int(figure_id)
        self.figure_changed.emit(self._figure_id)
        self.reload(select_all_series=self._select_all_on_load)

    def _on_axis_changed(self, index: int) -> None:
        """Rebuild series when the selected axis changes."""
        if self._signals_blocked or index < 0:
            return

        self._rebuild_series(select_all_series=True)
        self._emit_axis_changed()
        self._emit_selection_changed()

    def _on_series_selection_changed(self) -> None:
        """Emit selection changes when a series checkbox changes."""
        if not self._signals_blocked:
            self._emit_selection_changed()

    def _rebuild_series(self, *, select_all_series: bool) -> None:
        """Rebuild the series checklist for the selected axis."""
        self._signals_blocked = True
        self.series_list.clear()

        selected_axis = self.selected_axis_name()
        for series in self._axes.get(selected_axis, []):
            if self._series_filter is not None and not self._series_filter(series):
                continue

            item = QListWidgetItem(str(series["name"]))
            item.setData(_SERIES_ROLE, series)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if select_all_series
                else Qt.CheckState.Unchecked
            )
            self.series_list.addItem(item)

        self._signals_blocked = False

    def _set_all_series_selected(self, selected: bool) -> None:
        """Set all visible series checkboxes to checked/unchecked."""
        self._signals_blocked = True

        state = Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked
        for row in range(self.series_list.count()):
            self.series_list.item(row).setCheckState(state)

        self._signals_blocked = False
        self._emit_selection_changed()

    def _emit_axis_changed(self) -> None:
        """Emit the selected axis name."""
        self.axis_changed.emit(self.selected_axis_name())

    def _emit_selection_changed(self) -> None:
        """Emit selected axis name and checked series rows."""
        self.selection_changed.emit(
            self.selected_axis_name(),
            self.selected_series(),
        )