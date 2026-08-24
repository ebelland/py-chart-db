"""Strict-typed, responsive axis properties editor for Data Hub.

Drop-in replacement for ``app/widgets/axis_properties.py``.

The implementation keeps the existing public signals and host-facing API while
reducing signal cascades during reloads and replacing the Pylance-problematic
renderer scanner imports with typed wrappers.
"""
from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any, Final, TypeAlias, cast

from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.logs.logger import applogger
from app.charts import axis_options
from app.charts.render_figure import (
    GRID_AXES,
    GRID_WHICH,
    SUPPORTED_AXIS_SCALES,
)
from app.styles.style import (
    MARGIN_PANEL,
    apply_card_layout,
    create_card_widget,
    create_action_button,
    create_section_title,
    stdSizeAndlayout,
    configure_combo_width,
)
from app.utils.messages import ask
from app.widgets.dictionary_editor import DictEditorPanel

AxisDescriptorLike: TypeAlias = Any
RendererConfig: TypeAlias = dict[str, Any]
AxisPayload: TypeAlias = dict[str, Any]
ButtonSlot: TypeAlias = Callable[..., Any]

from app.scanners.axis_renderer_scanner import get_renderer,import_class_from_file
from app.utils.i18n import _
MAX_QT_HEIGHT: Final[int] = 16_777_215

# The editor offers exactly what the renderer knows how to apply, so the two
# cannot drift: these come straight from render_figure.
AXIS_SCALES: Final[tuple[str, ...]] = SUPPORTED_AXIS_SCALES
#: Kept for the figure options elsewhere in this module; the axis grid and
#: tick controls now come from app.charts.axis_options, which is the same
#: vocabulary the renderer applies.
GRID_WHICH_CHOICES: Final[tuple[str, ...]] = GRID_WHICH
GRID_AXIS_CHOICES: Final[tuple[str, ...]] = GRID_AXES
# The empty entry means "leave the Matplotlib default alone".

def _plain_options(value: object) -> dict[str, Any]:
    """Return a copy of a descriptor/options mapping with strict typing."""
    if isinstance(value, dict):
        return dict(cast(dict[str, Any], value))
    return {}

class AxisPropertiesWidget(QWidget):
    """Compact editor for axis descriptor properties.

    The widget intentionally works with the current descriptor schema. Missing
    descriptor fields are logged instead of guessed through legacy fallbacks.
    """

    axis_selected = Signal(int)
    axis_options_requested = Signal(dict)
    axis_action_requested = Signal(dict)
    renderer_changed = Signal(str)

    KWARGS_PANEL_MIN_HEIGHT: Final[int] = 120

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo: Any | None = None
        self._figure_id: int | None = None
        self._figure: Any | None = None
        self._redraw_callback: Any | None = None
        self._current_axis_id: int | None = None
        self._axis_map: dict[int, AxisDescriptorLike] = {}
        self._kwargs_editor: DictEditorPanel | None = None
        self.setMinimumHeight(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._build_ui()
        self.clear_connected_figure()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
 
    def _configure_combo_width(
        self,
        combo: QComboBox,
        minimum_contents_length: int = 0,
    ) -> None:
        """Make combo boxes use the available panel width.

        Thin wrapper kept so the call sites read the same as before; the rule
        itself is shared with the figure panel in ``style``.
        """
        configure_combo_width(combo, minimum_contents_length)

    def _build_ui(self) -> None:
        """Build the editor widget tree."""
        root = QVBoxLayout(self)
        root.setContentsMargins(*MARGIN_PANEL)
        root.setSpacing(8)
        root.addWidget(self._build_axis_selector_section(), 0)

        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("axisPropertiesTabs")
        self._tabs.setDocumentMode(True)
        self._tabs.setMinimumHeight(0)
        self._tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._tabs.addTab(self._build_axis_options_section(), _("Axis options"))
        self._tabs.addTab(self._build_kwargs_section(), _("Axis drawing (kwargs)"))
        self._tabs.addTab(self._build_annotations_section(), _("Annotations"))
        root.addWidget(self._tabs, 1)

    def _build_axis_selector_section(self) -> QWidget:
        """Create axis selector, axis actions and renderer display."""
        section = create_card_widget(self, "axisSelectorCard")
        layout = QVBoxLayout(section)
        apply_card_layout(layout)

        self._axis_combo = QComboBox(section)
        self._configure_combo_width(self._axis_combo, minimum_contents_length=24)
        # The tooltip that keeps the elided name readable is wired by
        # configure_combo_width, so it is not repeated here.
        self._axis_combo.currentIndexChanged.connect(self._on_axis_combo_changed)
        layout.addWidget(self._axis_combo)

        action_row = QWidget(section)
        action_layout = QHBoxLayout(action_row)
        stdSizeAndlayout(action_layout)
        
        self._btn_move_up = create_action_button(
                                parent=action_row,
                                action_id="up",
                                action=self._on_move_up_clicked,
                                layout=action_layout,
                            )
        self._btn_move_down = create_action_button(
                                  parent=action_row,
                                  action_id="down",
                                  action=self._on_move_down_clicked,
                                  layout=action_layout,
                              )
        self._btn_delete = create_action_button(
                               parent=action_row,
                               action_id="delete",
                               action=self._on_delete_clicked,
                               layout=action_layout,
                           )
        self._btn_apply = create_action_button(
                              parent=action_row,
                              action_id="apply",
                              action=self._emit_axis_options_requested,
                              layout=action_layout,
                          )
        action_layout.addStretch(1)
        layout.addWidget(action_row)

        self._renderer_value = QLabel("", section)
        self._renderer_value.setObjectName("axisRendererValue")
        self._renderer_value.setProperty("rendererLabel", True)
        self._renderer_value.setContentsMargins(0, 0, 0, 0)
        self._renderer_value.setWordWrap(True)
        self._renderer_value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._renderer_value)
        return section

    def _build_axis_options_section(self) -> QWidget:
        """Create editable axis option controls inside a local scroll area.

        The axis properties panel itself must not be hosted by an outer
        QScrollArea.  Only this long Axis options page needs scrolling because
        it contains many controls.  Keeping the scroll area inside the tab lets
        the QTabWidget and the top-level AxisPropertiesWidget expand to their
        parent instead of advertising a very tall sizeHint.
        """
        scroll = QScrollArea(self._tabs)
        scroll.setObjectName("axisOptionsScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumHeight(0)
        scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # This is the real tab content.  It may be taller than the viewport;
        # the scroll area above owns that overflow, not the whole properties
        # panel and not the QTabWidget.
        section = create_card_widget(scroll, "axisOptionsCard")
        section.setMinimumHeight(0)
        section.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        layout = QVBoxLayout(section)
        apply_card_layout(layout)

        form = QFormLayout()
        stdSizeAndlayout(form)

        self._axis_label_edit = QLineEdit(section)
        self._x_label_edit = QLineEdit(section)
        self._y_label_edit = QLineEdit(section)
        self._z_label_edit = QLineEdit(section)
        for edit in (
            self._axis_label_edit,
            self._x_label_edit,
            self._y_label_edit,
            self._z_label_edit,
        ):
            stdSizeAndlayout(edit)

        self._projection_combo = QComboBox(section)
        self._configure_combo_width(self._projection_combo, minimum_contents_length=14)
        for label, value in (
            ("rectilinear", "rectilinear"),
            ("polar", "polar"),
            ("3d", "3d"),
        ):
            self._projection_combo.addItem(label, value)

        self._sharex_check = QCheckBox(_("Share X"), section)
        self._sharey_check = QCheckBox(_("Share Y"), section)
        self._hide_axis_check = QCheckBox(_("Hide axis"), section)

        share_row = QWidget(section)
        share_layout = QHBoxLayout(share_row)
        share_layout.setContentsMargins(0, 0, 0, 0)
        share_layout.setSpacing(8)
        share_layout.addWidget(self._sharex_check)
        share_layout.addWidget(self._sharey_check)
        share_layout.addStretch(1)

        self._pickradius_spin = QDoubleSpinBox(section)
        self._pickradius_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._pickradius_spin.setRange(0.0, 1000.0)
        self._pickradius_spin.setSingleStep(0.5)
        self._pickradius_spin.setDecimals(2)

        form.addRow(_("Title"), self._axis_label_edit)
        form.addRow(_("X label"), self._x_label_edit)
        form.addRow(_("Y label"), self._y_label_edit)
        form.addRow(_("Z label"), self._z_label_edit)
        form.addRow(_("Projection"), self._projection_combo)
        form.addRow(_("Sharing"), share_row)
        form.addRow(_("Pick radius"), self._pickradius_spin)
        form.addRow(_("Visible"), self._hide_axis_check)
        layout.addLayout(form)

        layout.addWidget(create_section_title(_("Grid position"), section))
        layout.addLayout(self._build_span_form(section))

        layout.addWidget(create_section_title(_("Scale and direction"), section))
        layout.addLayout(self._build_scale_form(section))

        layout.addWidget(create_section_title(_("Ticks, grid and spines"), section))
        layout.addLayout(self._build_decoration_form(section))

        layout.addStretch(1)
        scroll.setWidget(section)
        return scroll

    # Matches the 1..6 range the Figure panel offers for rows/cols, so a span
    # can never claim more of the grid than the grid itself can have.
    MAX_GRID_SPAN: Final[int] = 6

    def _build_span_form(self, section: QWidget) -> QFormLayout:
        """Create the row/column span controls for non-uniform layouts.

        An axis normally fills one grid cell at its position. Raising either
        spin box lets it fill a rectangle of cells instead - e.g. a wide axis
        across the top row of a 2x2 grid with two narrower axes below it.
        Overlapping another axis's cells falls back to a plain compact grid
        on render rather than drawing on top of it.
        """
        form = QFormLayout()
        stdSizeAndlayout(form)

        self._row_span_spin = QSpinBox(section)
        self._col_span_spin = QSpinBox(section)
        for spin in (self._row_span_spin, self._col_span_spin):
            stdSizeAndlayout(spin)
            spin.setRange(1, self.MAX_GRID_SPAN)
            spin.setValue(1)
            spin.setToolTip(
                _(
                    "Grid cells this axis spans from its position, so it can "
                    "take up more room than the other axes in the figure."
                )
            )

        span_row = QWidget(section)
        span_layout = QHBoxLayout(span_row)
        span_layout.setContentsMargins(0, 0, 0, 0)
        span_layout.setSpacing(8)
        span_layout.addWidget(QLabel(_("Rows"), span_row))
        span_layout.addWidget(self._row_span_spin, 1)
        span_layout.addSpacing(8)
        span_layout.addWidget(QLabel(_("Cols"), span_row))
        span_layout.addWidget(self._col_span_spin, 1)

        form.addRow(_("Span"), span_row)
        return form

    def _build_scale_form(self, section: QWidget) -> QFormLayout:
        """Create the scale and direction controls.

        Scale base and symlog threshold are only meaningful for some scales, so
        they are enabled and disabled from the scale combos rather than being
        shown as always-editable fields that silently do nothing.
        """
        form = QFormLayout()
        stdSizeAndlayout(form)

        self._x_scale_combo = QComboBox(section)
        self._y_scale_combo = QComboBox(section)
        self._z_scale_combo = QComboBox(section)
        for combo in (self._x_scale_combo, self._y_scale_combo, self._z_scale_combo):
            self._configure_combo_width(combo, minimum_contents_length=12)
            for scale in AXIS_SCALES:
                combo.addItem(scale, scale)
            combo.currentIndexChanged.connect(self._update_scale_control_state)

        self._x_scale_base_spin = QDoubleSpinBox(section)
        self._y_scale_base_spin = QDoubleSpinBox(section)
        self._z_scale_base_spin = QDoubleSpinBox(section)
        for spin in (
            self._x_scale_base_spin,
            self._y_scale_base_spin,
            self._z_scale_base_spin,
        ):
            stdSizeAndlayout(spin)
            spin.setRange(1.1, 1000.0)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setValue(10.0)

        self._linthresh_spin = QDoubleSpinBox(section)
        stdSizeAndlayout(self._linthresh_spin)
        self._linthresh_spin.setRange(1e-9, 1e9)
        self._linthresh_spin.setDecimals(6)
        self._linthresh_spin.setValue(1.0)
        self._linthresh_spin.setToolTip(
            _("Width of the linear region around zero, for the symlog scale.")
        )

        self._invert_x_check = QCheckBox(_("Invert X"), section)
        self._invert_y_check = QCheckBox(_("Invert Y"), section)
        self._invert_z_check = QCheckBox(_("Invert Z"), section)
        invert_row = QWidget(section)
        invert_layout = QHBoxLayout(invert_row)
        stdSizeAndlayout(invert_layout)
        invert_layout.addWidget(self._invert_x_check)
        invert_layout.addWidget(self._invert_y_check)
        invert_layout.addWidget(self._invert_z_check)
        invert_layout.addStretch(1)

        form.addRow(_("X scale"), self._x_scale_combo)
        form.addRow(_("X log base"), self._x_scale_base_spin)
        form.addRow(_("Y scale"), self._y_scale_combo)
        form.addRow(_("Y log base"), self._y_scale_base_spin)
        # z alongside x and y rather than in a section of its own: it is the
        # same three settings, and a 3D axis is not a different kind of axis.
        # The renderer skips what a 2D axes has no setter for, so these are
        # live on every chart type and matter on the ones with a z.
        form.addRow(_("Z scale"), self._z_scale_combo)
        form.addRow(_("Z log base"), self._z_scale_base_spin)
        form.addRow(_("Symlog threshold"), self._linthresh_spin)
        form.addRow(_("Direction"), invert_row)

        self._update_scale_control_state()
        return form

    def _build_decoration_form(self, section: QWidget) -> QFormLayout:
        """Create the tick, grid and spine controls."""
        form = QFormLayout()
        stdSizeAndlayout(form)

        # Four grid settings and four tick settings: one per axis, per tick
        # class. One switch for the whole axes could not say "x major and y
        # minor", and - being a boolean - could not say "off" at all against a
        # style sheet that turns the grid on.
        self._grid_combos = self._build_setting_combos(
            section, axis_options.GRID_CHOICES
        )
        self._tick_combos = self._build_setting_combos(
            section, axis_options.TICK_CHOICES
        )

        self._tick_length_spin = QDoubleSpinBox(section)
        stdSizeAndlayout(self._tick_length_spin)
        self._tick_length_spin.setRange(0.0, 50.0)
        self._tick_length_spin.setDecimals(1)
        self._tick_length_spin.setSingleStep(0.5)

        self._x_tick_rotation_spin = QDoubleSpinBox(section)
        stdSizeAndlayout(self._x_tick_rotation_spin)
        self._x_tick_rotation_spin.setRange(-180.0, 180.0)
        self._x_tick_rotation_spin.setDecimals(0)
        self._x_tick_rotation_spin.setSingleStep(15.0)

        self._hide_spine_top_check = QCheckBox(_("Top"), section)
        self._hide_spine_right_check = QCheckBox(_("Right"), section)
        self._hide_spine_bottom_check = QCheckBox(_("Bottom"), section)
        self._hide_spine_left_check = QCheckBox(_("Left"), section)
        spine_row = QWidget(section)
        spine_layout = QHBoxLayout(spine_row)
        stdSizeAndlayout(spine_layout)
        for check in (
            self._hide_spine_top_check,
            self._hide_spine_right_check,
            self._hide_spine_bottom_check,
            self._hide_spine_left_check,
        ):
            spine_layout.addWidget(check)
        spine_layout.addStretch(1)

        form.addRow(_("Grid"), self._build_setting_grid(section, self._grid_combos))
        form.addRow(_("Ticks"), self._build_setting_grid(section, self._tick_combos))
        form.addRow(_("Tick length"), self._tick_length_spin)
        form.addRow(_("X tick rotation"), self._x_tick_rotation_spin)
        form.addRow(_("Hide spines"), spine_row)

        self._limits_mode_combo = QComboBox(section)
        self._configure_combo_width(self._limits_mode_combo, minimum_contents_length=18)
        for value, label in axis_options.LIMIT_CHOICES:
            self._limits_mode_combo.addItem(_(label), value)
        self._limits_mode_combo.currentIndexChanged.connect(
            self._update_limit_control_state
        )

        self._limit_spins: dict[tuple[str, str], QDoubleSpinBox] = {}
        limit_rows: dict[str, QWidget] = {}
        for axis in axis_options.LIMIT_AXES:
            row = QWidget(section)
            row_layout = QHBoxLayout(row)
            stdSizeAndlayout(row_layout)
            for edge in ("min", "max"):
                spin = QDoubleSpinBox(section)
                stdSizeAndlayout(spin)
                # Wide enough for real data - counts, wavelengths, timestamps
                # as seconds - and six decimals for the small end of it.
                spin.setRange(-1.0e15, 1.0e15)
                spin.setDecimals(6)
                spin.setSpecialValueText(_("(auto)"))
                # The bottom of the range doubles as "not set", so an axis can
                # have one end fixed and the other left to the data.
                spin.setValue(spin.minimum())
                self._limit_spins[(axis, edge)] = spin
                row_layout.addWidget(spin)
            row_layout.addStretch(1)
            limit_rows[axis] = row

        form.addRow(_("Limits"), self._limits_mode_combo)
        form.addRow(_("X range"), limit_rows["x"])
        form.addRow(_("Y range"), limit_rows["y"])
        form.addRow(_("Z range"), limit_rows["z"])
        self._update_limit_control_state()
        return form

    def _build_setting_combos(
        self, section: QWidget, choices: tuple[tuple[str, str], ...]
    ) -> dict[tuple[str, str], QComboBox]:
        """Return one combo per (axis, tick class), all offering *choices*."""
        combos: dict[tuple[str, str], QComboBox] = {}
        for axis in axis_options.AXES:
            for which in axis_options.WHICH:
                combo = QComboBox(section)
                self._configure_combo_width(combo, minimum_contents_length=12)
                for value, label in choices:
                    combo.addItem(_(label), value)
                combos[(axis, which)] = combo
        return combos

    def _build_setting_grid(
        self, section: QWidget, combos: dict[tuple[str, str], QComboBox]
    ) -> QWidget:
        """Lay four settings out as the 2x2 they are: axis across, class down.

        Four labelled rows would take four times the height and would still
        leave the reader to work out that they are the same question asked
        four times.
        """
        holder = QWidget(section)
        layout = QGridLayout(holder)
        stdSizeAndlayout(layout)

        for column, axis in enumerate(axis_options.AXES, start=1):
            label = QLabel(axis.upper(), holder)
            label.setProperty("muted", True)
            layout.addWidget(label, 0, column)

        for row, which in enumerate(axis_options.WHICH, start=1):
            label = QLabel(_("Major") if which == "major" else _("Minor"), holder)
            label.setProperty("muted", True)
            layout.addWidget(label, row, 0)
            for column, axis in enumerate(axis_options.AXES, start=1):
                layout.addWidget(combos[(axis, which)], row, column)

        layout.setColumnStretch(len(axis_options.AXES) + 1, 1)
        return holder

    def _update_limit_control_state(self) -> None:
        """Enable only the limit boxes the chosen mode actually reads."""
        mode = str(self._limits_mode_combo.currentData() or axis_options.LIMITS_AUTO)
        options = {"limits_mode": mode}
        for (axis, _edge), spin in self._limit_spins.items():
            spin.setEnabled(not axis_options.is_automatic(options, axis))

    def _update_scale_control_state(self) -> None:
        """Enable only the scale parameters the selected scales actually use."""
        x_scale, y_scale, z_scale = self._selected_scales()

        self._x_scale_base_spin.setEnabled(x_scale in {"log", "symlog"})
        self._y_scale_base_spin.setEnabled(y_scale in {"log", "symlog"})
        self._z_scale_base_spin.setEnabled(z_scale in {"log", "symlog"})
        self._linthresh_spin.setEnabled(
            "symlog" in {x_scale, y_scale, z_scale}
        )

    def _selected_scales(self) -> tuple[str, str, str]:
        """Return the chosen x, y and z scales, defaulting to linear."""
        return tuple(  # type: ignore[return-value]
            str(combo.currentData() or "linear")
            for combo in (
                self._x_scale_combo,
                self._y_scale_combo,
                self._z_scale_combo,
            )
        )

    # ------------------------------------------------------------------
    # Scale / tick / grid / spine options
    # ------------------------------------------------------------------
    def _extended_option_widgets(self) -> tuple[QWidget, ...]:
        """Return the controls added by the scale and decoration sections."""
        return (
            self._row_span_spin,
            self._col_span_spin,
            self._x_scale_combo,
            self._y_scale_combo,
            self._z_scale_combo,
            self._x_scale_base_spin,
            self._y_scale_base_spin,
            self._z_scale_base_spin,
            self._linthresh_spin,
            self._invert_x_check,
            self._invert_y_check,
            self._invert_z_check,
            self._tick_length_spin,
            self._x_tick_rotation_spin,
            self._hide_spine_top_check,
            self._hide_spine_right_check,
            self._hide_spine_bottom_check,
            self._hide_spine_left_check,
            self._limits_mode_combo,
            *self._grid_combos.values(),
            *self._tick_combos.values(),
            *self._limit_spins.values(),
        )

    @staticmethod
    def _select_combo_value(combo: QComboBox, value: object, fallback: str) -> None:
        """Select a combo entry by data, falling back when it is unknown."""
        index = combo.findData(str(value if value is not None else fallback))
        combo.setCurrentIndex(index if index >= 0 else max(0, combo.findData(fallback)))

    @staticmethod
    def _float_option(options: dict[str, Any], key: str, default: float) -> float:
        """Read a float option, keeping the default when it is unusable."""
        try:
            return float(options.get(key, default))
        except (TypeError, ValueError):
            applogger.warning("Invalid axis option %s=%r", key, options.get(key))
            return default

    def _int_option(self, options: dict[str, Any], key: str, default: int) -> int:
        """Read a clamped int option, keeping the default when it is unusable."""
        try:
            value = int(options.get(key, default))
        except (TypeError, ValueError):
            applogger.warning("Invalid axis option %s=%r", key, options.get(key))
            return default
        return max(1, min(value, self.MAX_GRID_SPAN))

    def _load_extended_axis_options(self, options: dict[str, Any]) -> None:
        """Populate the scale, tick, grid and spine controls from options."""
        self._row_span_spin.setValue(self._int_option(options, "row_span", 1))
        self._col_span_spin.setValue(self._int_option(options, "col_span", 1))
        self._select_combo_value(self._x_scale_combo, options.get("x_scale"), "linear")
        self._select_combo_value(self._y_scale_combo, options.get("y_scale"), "linear")
        self._select_combo_value(self._z_scale_combo, options.get("z_scale"), "linear")
        self._x_scale_base_spin.setValue(self._float_option(options, "x_scale_base", 10.0))
        self._y_scale_base_spin.setValue(self._float_option(options, "y_scale_base", 10.0))
        self._z_scale_base_spin.setValue(self._float_option(options, "z_scale_base", 10.0))
        self._linthresh_spin.setValue(self._float_option(options, "x_linthresh", 1.0))

        self._invert_x_check.setChecked(bool(options.get("invert_x", False)))
        self._invert_y_check.setChecked(bool(options.get("invert_y", False)))
        self._invert_z_check.setChecked(bool(options.get("invert_z", False)))

        # Through axis_options so a figure saved before these existed opens
        # with the settings it actually had, rather than with four Autos.
        for key, combo in self._grid_combos.items():
            self._select_combo_value(
                combo, axis_options.grid_setting(options, *key), axis_options.AUTO
            )
        for key, combo in self._tick_combos.items():
            self._select_combo_value(
                combo, axis_options.tick_setting(options, *key), axis_options.AUTO
            )

        self._select_combo_value(
            self._limits_mode_combo,
            axis_options.limits_mode(options),
            axis_options.LIMITS_AUTO,
        )
        for (axis, edge), spin in self._limit_spins.items():
            value = axis_options.manual_limit(options, axis, edge)
            spin.setValue(spin.minimum() if value is None else value)
        self._update_limit_control_state()

        self._tick_length_spin.setValue(self._float_option(options, "tick_length", 0.0))
        self._x_tick_rotation_spin.setValue(
            self._float_option(options, "x_tick_rotation", 0.0)
        )

        for name, check in self._spine_checks().items():
            check.setChecked(bool(options.get(f"hide_spine_{name}", False)))

    def _clear_extended_axis_options(self) -> None:
        """Reset the scale, tick, grid and spine controls to their defaults."""
        self._load_extended_axis_options({})
        self._update_scale_control_state()

    def _spine_checks(self) -> dict[str, QCheckBox]:
        """Return the spine visibility checkboxes keyed by spine name."""
        return {
            "top": self._hide_spine_top_check,
            "right": self._hide_spine_right_check,
            "bottom": self._hide_spine_bottom_check,
            "left": self._hide_spine_left_check,
        }

    def _extended_axis_options_payload(self) -> dict[str, Any]:
        """Return the scale, tick, grid and spine part of the axis payload.

        Optional numeric settings are emitted as None when they are zero or the
        control is disabled, so that "not configured" stays distinguishable from
        "configured to zero" once the payload reaches the renderer.
        """
        x_scale, y_scale, z_scale = self._selected_scales()
        tick_length = float(self._tick_length_spin.value())

        payload: dict[str, Any] = {
            "row_span": int(self._row_span_spin.value()),
            "col_span": int(self._col_span_spin.value()),
            "x_scale": x_scale,
            "y_scale": y_scale,
            "z_scale": z_scale,
            "x_scale_base": (
                float(self._x_scale_base_spin.value())
                if x_scale in {"log", "symlog"}
                else None
            ),
            "y_scale_base": (
                float(self._y_scale_base_spin.value())
                if y_scale in {"log", "symlog"}
                else None
            ),
            "z_scale_base": (
                float(self._z_scale_base_spin.value())
                if z_scale in {"log", "symlog"}
                else None
            ),
            "x_linthresh": (
                float(self._linthresh_spin.value())
                if "symlog" in {x_scale, y_scale, z_scale}
                else None
            ),
            "invert_x": bool(self._invert_x_check.isChecked()),
            "invert_y": bool(self._invert_y_check.isChecked()),
            "invert_z": bool(self._invert_z_check.isChecked()),
            "tick_length": tick_length if tick_length > 0.0 else None,
            "x_tick_rotation": float(self._x_tick_rotation_spin.value()),
        }
        # One threshold control for all three: symlog's linear region is a
        # property of the data's units, and an axis whose scale is not symlog
        # ignores the key entirely.
        payload["y_linthresh"] = payload["x_linthresh"]
        payload["z_linthresh"] = payload["x_linthresh"]

        for (axis, which), combo in self._grid_combos.items():
            payload[axis_options.grid_key(axis, which)] = str(
                combo.currentData() or axis_options.AUTO
            )
        for (axis, which), combo in self._tick_combos.items():
            payload[axis_options.tick_key(axis, which)] = str(
                combo.currentData() or axis_options.AUTO
            )

        # The legacy keys are written as the four settings imply, so a figure
        # edited here still renders in a build that predates them - and so
        # nothing downstream reads a stale "grid: True" that the user has
        # since turned off.
        payload["grid"] = False
        payload["grid_which"] = "major"
        payload["grid_axis"] = "both"
        payload["minor_ticks"] = any(
            str(self._tick_combos[(axis, "minor")].currentData() or axis_options.AUTO)
            not in (axis_options.AUTO, axis_options.OFF)
            for axis in axis_options.AXES
        )

        mode = str(self._limits_mode_combo.currentData() or axis_options.LIMITS_AUTO)
        payload["limits_mode"] = mode
        for (axis, edge), spin in self._limit_spins.items():
            automatic = axis_options.is_automatic({"limits_mode": mode}, axis)
            unset = spin.value() <= spin.minimum()
            payload[axis_options.limit_key(axis, edge)] = (
                None if automatic or unset else float(spin.value())
            )

        for name, check in self._spine_checks().items():
            payload[f"hide_spine_{name}"] = bool(check.isChecked())

        return payload

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------
    ANNOTATION_TYPES: Final[tuple[str, ...]] = ("arrow", "text", "boxed text")

    def _build_annotations_section(self) -> QWidget:
        """Create an editable list of axis annotations.

        Stored axis option format::

            {
                "annotations": [
                    {
                        "x": 1.0,
                        "y": 2.0,
                        "type": "arrow",
                        "text": "Label",
                        "kwargs": {"xytext": [10, 10], "textcoords": "offset points"},
                    }
                ]
            }

        ``kwargs`` is edited as JSON so Matplotlib options such as
        ``arrowprops``, ``bbox``, ``xycoords``, ``textcoords``, ``ha`` and
        ``va`` can be stored without adding a widget for every possible key.
        """
        section = create_card_widget(self, "axisAnnotationsCard")
        layout = QVBoxLayout(section)
        apply_card_layout(layout)

        self._annotations_table = QTableWidget(0, 5, section)
        self._annotations_table.setObjectName("axisAnnotationsTable")
        self._annotations_table.setHorizontalHeaderLabels(
            [_("X"), _("Y"), _("Type"), _("Text"), _("Kwargs JSON")]
        )
        self._annotations_table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._annotations_table.setMinimumHeight(self.KWARGS_PANEL_MIN_HEIGHT)
        self._annotations_table.setToolTip(
            _(
                "Annotations are stored in axis options. Kwargs must be JSON, "
                "for example: {\"xytext\": [10, 10], \"textcoords\": \"offset points\"}."
            )
        )
        layout.addWidget(self._annotations_table, 1)

        button_row = QWidget(section)
        button_layout = QHBoxLayout(button_row)
        stdSizeAndlayout(button_layout)
        self._btn_add_annotation = QPushButton(_("Add annotation"), button_row)
        self._btn_delete_annotation = QPushButton(_("Delete selected"), button_row)
        self._btn_add_annotation.clicked.connect(self._add_annotation_row)
        self._btn_delete_annotation.clicked.connect(self._delete_selected_annotation_rows)
        button_layout.addWidget(self._btn_add_annotation)
        button_layout.addWidget(self._btn_delete_annotation)
        button_layout.addStretch(1)
        layout.addWidget(button_row, 0)
        return section

    def _annotation_widgets(self) -> tuple[QWidget, ...]:
        """Return annotation controls for signal blocking/enabling."""
        return (
            self._annotations_table,
            self._btn_add_annotation,
            self._btn_delete_annotation,
        )

    def _annotation_type_combo(self, value: str = "text") -> QComboBox:
        """Create the annotation type editor."""
        combo = QComboBox(self._annotations_table)
        for annotation_type in self.ANNOTATION_TYPES:
            combo.addItem(annotation_type, annotation_type)
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)
        return combo

    def _new_table_item(self, value: object = "") -> QTableWidgetItem:
        """Create an editable table item with a string value."""
        item = QTableWidgetItem(str(value if value is not None else ""))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        return item

    def _add_annotation_row(
        self,
        annotation: dict[str, Any] | None = None,
    ) -> None:
        """Append one annotation to the table."""
        annotation = dict(annotation or {})
        row = self._annotations_table.rowCount()
        self._annotations_table.insertRow(row)
        kwargs = annotation.get("kwargs", {})
        kwargs_text = ""
        if isinstance(kwargs, dict) and kwargs:
            kwargs_text = json.dumps(kwargs, ensure_ascii=False)
        elif isinstance(kwargs, str):
            kwargs_text = kwargs

        self._annotations_table.setItem(row,0,self._new_table_item(annotation.get("x",0.0)))
        self._annotations_table.setItem(row,1,self._new_table_item(annotation.get("y",0.0)))
        self._annotations_table.setCellWidget(row,2,self._annotation_type_combo(str(annotation.get("type","text"))))
        self._annotations_table.setItem(row,3,self._new_table_item(annotation.get("text","")))
        self._annotations_table.setItem(row,4,self._new_table_item(kwargs_text))

    def _delete_selected_annotation_rows(self) -> None:
        """Delete selected annotation rows, or the current row if none selected."""
        rows = {index.row() for index in self._annotations_table.selectedIndexes()}
        if not rows and self._annotations_table.currentRow() >= 0:
            rows = {self._annotations_table.currentRow()}
        for row in sorted(rows, reverse=True):
            self._annotations_table.removeRow(row)

    def _load_annotations(self, options: dict[str, Any]) -> None:
        """Populate the annotation table from axis options."""
        self._annotations_table.setRowCount(0)
        annotations = options.get("annotations", [])
        if not isinstance(annotations, list):
            applogger.warning("Invalid axis annotations=%r", annotations)
            return
        for item in annotations:
            if isinstance(item, dict):
                self._add_annotation_row(cast(dict[str, Any], item))

    def _clear_annotations(self) -> None:
        """Remove all annotation rows."""
        self._annotations_table.setRowCount(0)

    def _annotation_item_text(self, row: int, col: int) -> str:
        """Return stripped text for one annotation table cell."""
        item = self._annotations_table.item(row, col)
        return item.text().strip() if item is not None else ""

    def _annotations_payload(self) -> list[dict[str, Any]]:
        """Return valid annotations from the table as axis-option payload."""
        annotations: list[dict[str, Any]] = []
        for row in range(self._annotations_table.rowCount()):
            try:
                x = float(self._annotation_item_text(row, 0))
                y = float(self._annotation_item_text(row, 1))
            except ValueError:
                applogger.warning("Skipping annotation row %s with invalid x/y", row + 1)
                continue

            annotation_type = "text"
            type_editor = self._annotations_table.cellWidget(row, 2)
            if isinstance(type_editor, QComboBox):
                annotation_type = str(type_editor.currentData() or type_editor.currentText()).lower()
            if annotation_type not in self.ANNOTATION_TYPES:
                applogger.warning(
                    "Unknown annotation type %r on row %s; using text.",
                    annotation_type,
                    row + 1,
                )
                annotation_type = "text"

            kwargs_text = self._annotation_item_text(row, 4)
            kwargs: dict[str, Any] = {}
            if kwargs_text:
                try:
                    parsed = json.loads(kwargs_text)
                    if isinstance(parsed, dict):
                        kwargs = cast(dict[str, Any], parsed)
                    else:
                        applogger.warning(
                            "Skipping non-object annotation kwargs on row %s", row + 1
                        )
                except json.JSONDecodeError:
                    applogger.warning(
                        "Skipping invalid annotation kwargs JSON on row %s", row + 1
                    )

            annotations.append(
                {
                    "x": x,
                    "y": y,
                    "type": annotation_type,
                    "text": self._annotation_item_text(row, 3),
                    "kwargs": kwargs,
                }
            )
        return annotations

    def _build_kwargs_section(self) -> QWidget:
        """Create the host section for DictEditorPanel."""
        section = create_card_widget(self, "axisKwargsCard")
        section.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        layout = QVBoxLayout(section)
        stdSizeAndlayout(layout)

        self._kwargs_host = QFrame(section)
        self._kwargs_host.setObjectName("axisKwargsPanel")
        self._kwargs_host.setFrameShape(QFrame.Shape.NoFrame)
        self._kwargs_host.setMinimumHeight(self.KWARGS_PANEL_MIN_HEIGHT)
        self._kwargs_host.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._kwargs_layout = QVBoxLayout(self._kwargs_host)
        self._kwargs_layout.setContentsMargins(0, 0, 0, 0)
        self._kwargs_layout.setSpacing(0)

        placeholder = QLabel(
            _("Additional kwargs editor can be inserted here by the host."),
            self._kwargs_host,
        )
        placeholder.setWordWrap(True)
        placeholder.setContentsMargins(0, 0, 0, 0)
        self._kwargs_layout.addWidget(placeholder, 0)
        layout.addWidget(self._kwargs_host, 1)
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
        """Attach a figure and load axes from its descriptor."""
        self._repo = repo
        self._figure_id = int(figure_id)
        self._figure = figure
        self._redraw_callback = redraw_callback
        self._reload_from_descriptor()

    def clear_connected_figure(self) -> None:
        """Reset UI and disable editing."""
        self.setUpdatesEnabled(False)
        try:
            self._repo = None
            self._figure_id = None
            self._figure = None
            self._redraw_callback = None
            self._current_axis_id = None
            self._axis_map.clear()
            with QSignalBlocker(self._axis_combo):
                self._axis_combo.clear()
            self._clear_axis_fields()
            self._set_enabled_state(False)
        finally:
            self.setUpdatesEnabled(True)

    def current_axis_id(self) -> int | None:
        """Return the currently selected axis id."""
        return self._current_axis_id

    def set_renderer_options_widget(self, widget: QWidget) -> None:
        """Compatibility no-op for removed renderer-options UI."""
        if widget.parent() is None:
            widget.setParent(self)
        widget.hide()

    def set_kwargs_widget(self, widget: QWidget) -> None:
        """Replace kwargs host content."""
        self._replace_layout_widget(self._kwargs_layout, widget)

    def rebuild_kwargs_editor(self, axis_id: int | None) -> None:
        """Rebuild the renderer kwargs editor for ``axis_id``."""
        if self._kwargs_editor is not None:
            self._kwargs_editor.commit_pending_edits()
            self._kwargs_editor = None

        axis_desc = self._axis_map.get(int(axis_id)) if axis_id is not None else None
        if axis_desc is None:
            self.set_kwargs_widget(self._build_note_widget(_("Select an axis to edit kwargs.")))
            return

        renderer_key = self._axis_renderer(axis_desc, self._axis_options(axis_desc))
        if not renderer_key:
            self.set_kwargs_widget(self._build_note_widget(_("Renderer not found.")))
            return

        renderer: RendererConfig | None = get_renderer(renderer_key)
        if renderer is None:
            self.set_kwargs_widget(self._build_note_widget(_("Renderer not found.")))
            return

        try:
            renderer_class = import_class_from_file(renderer)
            if renderer_class is None:
                raise RuntimeError("Renderer class not found")
            renderer_instance = renderer_class()
        except Exception:
            applogger.exception("Failed to load renderer kwargs schema")
            self.set_kwargs_widget(
                self._build_note_widget(_("Failed to load renderer kwargs schema."))
            )
            return

        schema = getattr(renderer_instance, "Kwargs", None)
        if not schema:
            self.set_kwargs_widget(
                self._build_note_widget(_("No kwargs available for this renderer."))
            )
            return
        if self._repo is None:
            return

        axis_id_int = int(axis_id) if axis_id is not None else 0
        current_options: dict[str, Any] = _plain_options(
            self._repo.get_axis_options(axis_id_int) or {}
        )

        editor = DictEditorPanel(schema, self)
        self._configure_kwargs_editor(editor)
        values = renderer_instance.get_kwargs(current_options)
        if values:
            editor.set_values(values)
            self._configure_kwargs_editor(editor)
        self._kwargs_editor = editor
        self.set_kwargs_widget(editor)

    def clean_kwargs(self) -> dict[str, Any]:
        """Commit and return non-empty kwargs values."""
        if self._kwargs_editor is None:
            return {}
        self._kwargs_editor.commit_pending_edits()
        values = cast(dict[str, Any], self._kwargs_editor.get_values())
        return {key: value for key, value in values.items() if value != ""}

    def _build_note_widget(self, text: str) -> QWidget:
        note = QLabel(text, self)
        note.setWordWrap(True)
        note.setContentsMargins(0, 0, 0, 0)
        return note

    def _configure_kwargs_editor(self, editor: QWidget) -> QWidget:
        """Make renderer kwargs content expand without forcing panel growth."""
        if isinstance(editor, QFrame):
            editor.setFrameShape(QFrame.Shape.NoFrame)
            editor.setLineWidth(0)
            editor.setMidLineWidth(0)

        set_resizable = getattr(editor, "setWidgetResizable", None)
        if callable(set_resizable):
            set_resizable(True)
        horizontal_policy = getattr(editor, "setHorizontalScrollBarPolicy", None)
        if callable(horizontal_policy):
            horizontal_policy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        vertical_policy = getattr(editor, "setVerticalScrollBarPolicy", None)
        if callable(vertical_policy):
            vertical_policy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        editor.setMinimumHeight(min(editor.sizeHint().height(), self.KWARGS_PANEL_MIN_HEIGHT))
        editor.setMaximumHeight(MAX_QT_HEIGHT)
        editor.setContentsMargins(0, 0, 0, 0)
        editor.updateGeometry()
        return editor

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _set_enabled_state(self, enabled: bool) -> None:
        """Enable or disable interactive controls."""
        widgets: tuple[QWidget, ...] = (
            self._axis_combo,
            self._btn_move_up,
            self._btn_move_down,
            self._btn_delete,
            self._axis_label_edit,
            self._x_label_edit,
            self._y_label_edit,
            self._z_label_edit,
            self._projection_combo,
            self._sharex_check,
            self._sharey_check,
            self._hide_axis_check,
            self._pickradius_spin,
            self._btn_apply,
            self._renderer_value,
            self._tabs,
        )
        for widget in widgets:
            widget.setEnabled(enabled)
        self._update_axis_action_buttons()

    def _reload_from_descriptor(self) -> None:
        """Rebuild the axis selector from the current descriptor."""
        if self._repo is None or self._figure_id is None:
            self.clear_connected_figure()
            return

        self.setUpdatesEnabled(False)
        try:
            desc = self._repo.load_figure_descriptor(self._figure_id)
            previous_axis_id = self._current_axis_id
            self._axis_map.clear()
            self._kwargs_editor = None

            with QSignalBlocker(self._axis_combo):
                self._axis_combo.clear()
                if desc is None:
                    applogger.warning(
                        "No figure descriptor found for figure_id=%s. Axis editing disabled.",
                        self._figure_id,
                    )
                    return
                for axis_desc in self._axes_from_descriptor(desc):
                    axis_id = int(axis_desc.id)
                    self._axis_map[axis_id] = axis_desc
                    self._axis_combo.addItem(
                        self._axis_display_label(axis_desc, axis_id),
                        axis_id,
                    )
        finally:
            self.setUpdatesEnabled(True)

        if self._axis_combo.count() == 0:
            self._load_axis_descriptor(None)
            self._update_axis_action_buttons()
            self._set_enabled_state(False)
            return
        self._set_enabled_state(True)
        self._select_axis(previous_axis_id)

    def _axes_from_descriptor(self, desc: Any) -> list[AxisDescriptorLike]:
        """Return axes from a validated descriptor."""
        axes = getattr(desc, "axes", None)
        if axes is None:
            applogger.error(
                "Figure descriptor id=%r has no axes list. Axis editing stopped.",
                getattr(desc, "id", self._figure_id),
            )
            return []
        return list(axes)

    def _select_axis(self, preferred_axis_id: int | None) -> None:
        """Select the previous axis when possible, otherwise select the first."""
        index_to_select = 0
        if preferred_axis_id is not None:
            for index in range(self._axis_combo.count()):
                axis_id_data = self._axis_combo.itemData(index)
                if axis_id_data is not None and int(axis_id_data) == preferred_axis_id:
                    index_to_select = index
                    break
        self._axis_combo.setCurrentIndex(index_to_select)
        self._on_axis_combo_changed(index_to_select)

    def _update_axis_action_buttons(self) -> None:
        """Enable axis structure buttons for the current combo row."""
        has_axes = self._axis_combo.count() > 0
        current_index = self._axis_combo.currentIndex()
        is_enabled = self._axis_combo.isEnabled() and has_axes
        self._btn_move_up.setEnabled(is_enabled and current_index > 0)
        self._btn_move_down.setEnabled(
            is_enabled and current_index < self._axis_combo.count() - 1
        )
        self._btn_delete.setEnabled(is_enabled)

    # ------------------------------------------------------------------
    # Descriptor binding
    # ------------------------------------------------------------------
    def _axis_display_label(
        self,
        axis_desc: AxisDescriptorLike,
        axis_id: int,
    ) -> str:
        """Build combo text for one axis."""
        options = self._axis_options(axis_desc)
        title = str(
            getattr(axis_desc, "title", None)
            or options.get("label")
            or f"Axis {axis_id}"
        ).strip()
        renderer = self._axis_renderer(axis_desc, options)
        return f"{title} [{renderer}]" if renderer else title

    def _load_axis_descriptor(self, axis_desc: AxisDescriptorLike | None) -> None:
        """Load one axis descriptor into the form."""
        if axis_desc is None:
            self._clear_axis_fields()
            return

        options = self._axis_options(axis_desc)
        projection = str(options.get("projection") or "rectilinear").strip()
        renderer = self._axis_renderer(axis_desc, options)
        self._current_axis_id = int(axis_desc.id)

        with self._form_signal_blocker():
            self._axis_label_edit.setText(
                str(
                    options.get("label")
                    or options.get("title")
                    or getattr(axis_desc, "title", "")
                    or ""
                ).strip()
            )
            self._x_label_edit.setText(str(options.get("x_label", "")).strip())
            self._y_label_edit.setText(str(options.get("y_label", "")).strip())
            self._z_label_edit.setText(str(options.get("z_label", "")).strip())
            self._projection_combo.setCurrentIndex(
                max(0, self._projection_combo.findData(projection))
            )
            self._sharex_check.setChecked(bool(options.get("sharex", False)))
            self._sharey_check.setChecked(bool(options.get("sharey", False)))
            self._hide_axis_check.setChecked(
                bool(options.get("hide_axis", options.get("hidden", False)))
            )
            self._pickradius_spin.setValue(float(options.get("pickradius", 0.0) or 0.0))
            self._load_extended_axis_options(options)
            self._load_annotations(options)

        self._renderer_value.setText(renderer)
        self.renderer_changed.emit(renderer)

    class _FormSignalBlocker:
        """Small context manager that owns QSignalBlocker instances."""

        def __init__(self, widgets: tuple[QWidget, ...]) -> None:
            self._widgets = widgets
            self._blockers: list[QSignalBlocker] = []

        def __enter__(self) -> None:
            self._blockers = [QSignalBlocker(widget) for widget in self._widgets]

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            self._blockers.clear()

    def _form_signal_blocker(self) -> _FormSignalBlocker:
        return self._FormSignalBlocker(
            (
                self._axis_label_edit,
                self._x_label_edit,
                self._y_label_edit,
                self._z_label_edit,
                self._projection_combo,
                self._sharex_check,
                self._sharey_check,
                self._hide_axis_check,
                self._pickradius_spin,
            )
            + self._extended_option_widgets()
            + self._annotation_widgets()
        )

    def _clear_axis_fields(self) -> None:
        """Clear axis-specific form fields."""
        self._current_axis_id = None
        with self._form_signal_blocker():
            self._axis_label_edit.clear()
            self._x_label_edit.clear()
            self._y_label_edit.clear()
            self._z_label_edit.clear()
            self._projection_combo.setCurrentIndex(0)
            self._sharex_check.setChecked(False)
            self._sharey_check.setChecked(False)
            self._hide_axis_check.setChecked(False)
            self._pickradius_spin.setValue(0.0)
            self._clear_extended_axis_options()
            self._clear_annotations()
        self._renderer_value.clear()
        self.renderer_changed.emit("")

    def _axis_options(self, axis_desc: AxisDescriptorLike) -> dict[str, Any]:
        """Return axis options as a mutable plain dictionary."""
        options = getattr(axis_desc, "options", None)
        if options is None:
            return {}
        if isinstance(options, dict):
            return dict(cast(dict[str, Any], options))
        applogger.error(
            "Axis id=%r has invalid options. Editing stopped.",
            getattr(axis_desc, "id", None),
        )
        return {}

    def _axis_renderer(
        self,
        axis_desc: AxisDescriptorLike,
        options: dict[str, Any],
    ) -> str:
        """Resolve the renderer name from options or descriptor chart_type."""
        renderer = (
            options.get("renderer")
            or options.get("renderer_name")
            or getattr(axis_desc, "chart_type", "")
            or ""
        )
        return str(renderer).strip()

    # ------------------------------------------------------------------
    # UI events
    # ------------------------------------------------------------------
    def _on_axis_combo_changed(self, index: int) -> None:
        """Handle axis selector changes."""
        if index < 0:
            self._load_axis_descriptor(None)
            self._update_axis_action_buttons()
            return
        axis_id_data = self._axis_combo.itemData(index)
        if axis_id_data is None:
            self._load_axis_descriptor(None)
            self._update_axis_action_buttons()
            return

        axis_id = int(axis_id_data)
        axis_desc = self._axis_map.get(axis_id)
        if axis_desc is None:
            applogger.error(
                "Axis descriptor id=%r not found in axis map. Editing stopped.",
                axis_id,
            )
            self._load_axis_descriptor(None)
            self._update_axis_action_buttons()
            return

        self._load_axis_descriptor(axis_desc)
        self._update_axis_action_buttons()
        self.axis_selected.emit(axis_id)

    def _emit_axis_action(
        self,
        action: str,
        axis_id: int | None = None,
        index: int | None = None,
    ) -> None:
        """Emit a structural axis action for host-side persistence."""
        selected_axis_id = self._axis_combo.currentData() if axis_id is None else axis_id
        selected_index = self._axis_combo.currentIndex() if index is None else index
        payload: AxisPayload = {
            "action": action,
            "axis_id": selected_axis_id,
            "figure_id": self._figure_id,
            "index": selected_index,
        }
        self.axis_action_requested.emit(payload)

    def _swap_axis_rows(self, first_row: int, second_row: int) -> None:
        """Swap two combo rows while preserving user-data axis ids."""
        first_text = self._axis_combo.itemText(first_row)
        first_data = self._axis_combo.itemData(first_row)
        second_text = self._axis_combo.itemText(second_row)
        second_data = self._axis_combo.itemData(second_row)
        with QSignalBlocker(self._axis_combo):
            self._axis_combo.setItemText(first_row, second_text)
            self._axis_combo.setItemData(first_row, second_data)
            self._axis_combo.setItemText(second_row, first_text)
            self._axis_combo.setItemData(second_row, first_data)
            self._axis_combo.setCurrentIndex(second_row)
        self._on_axis_combo_changed(second_row)

    def _move_selected_axis(self, delta: int) -> None:
        """Move the selected axis row locally and notify the host."""
        current_row = self._axis_combo.currentIndex()
        target_row = current_row + delta
        if current_row < 0 or target_row < 0 or target_row >= self._axis_combo.count():
            return
        axis_id = self._axis_combo.currentData()
        if axis_id is None:
            return
        action = "move_up" if delta < 0 else "move_down"
        self._swap_axis_rows(current_row, target_row)
        self._emit_axis_action(action, axis_id=int(axis_id), index=target_row)

    def _delete_selected_axis(self) -> None:
        """Delete selected axis row locally and notify the host."""
        current_row = self._axis_combo.currentIndex()
        axis_id = self._axis_combo.currentData()
        if current_row < 0 or axis_id is None:
            return
        axis_id_int = int(axis_id)
        if not ask(self, "axis.confirm_delete", axis=self._axis_combo.itemText(current_row)):
            return
        self._emit_axis_action("delete", axis_id=axis_id_int, index=current_row)
        self._axis_map.pop(axis_id_int, None)
        with QSignalBlocker(self._axis_combo):
            self._axis_combo.removeItem(current_row)
            next_row = min(current_row, self._axis_combo.count() - 1)
            if next_row >= 0:
                self._axis_combo.setCurrentIndex(next_row)
        if self._axis_combo.count() == 0:
            self._load_axis_descriptor(None)
            self._update_axis_action_buttons()
        else:
            self._on_axis_combo_changed(self._axis_combo.currentIndex())

    def _on_move_up_clicked(self) -> None:
        self._move_selected_axis(-1)

    def _on_move_down_clicked(self) -> None:
        self._move_selected_axis(1)

    def _on_delete_clicked(self) -> None:
        self._delete_selected_axis()

    def _emit_axis_options_requested(self) -> None:
        """Emit the current axis options payload."""
        label_text = self._axis_label_edit.text().strip()
        payload: AxisPayload = {
            "axis_id": self._current_axis_id,
            "label": label_text,
            "title": label_text,
            "x_label": self._x_label_edit.text().strip(),
            "y_label": self._y_label_edit.text().strip(),
            "z_label": self._z_label_edit.text().strip(),
            "projection": str(self._projection_combo.currentData() or "rectilinear"),
            "sharex": bool(self._sharex_check.isChecked()),
            "sharey": bool(self._sharey_check.isChecked()),
            "hide_axis": bool(self._hide_axis_check.isChecked()),
            "pickradius": float(self._pickradius_spin.value()),
            "renderer": self._renderer_value.text().strip(),
        }
        payload.update(self._extended_axis_options_payload())
        payload["annotations"] = self._annotations_payload()
        self.axis_options_requested.emit(payload)

    # ------------------------------------------------------------------
    # Kwargs editor host helpers
    # ------------------------------------------------------------------
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Default event filter retained for compatibility."""
        return super().eventFilter(watched, event)

    def _replace_layout_widget(self, layout: QVBoxLayout, widget: QWidget) -> None:
        """Replace kwargs content with a plain panel widget.

        The outgoing widget is hidden and scheduled for deletion, and
        deliberately *not* unparented on the way out.  ``setParent(None)``
        does not simply detach a widget - it promotes it to a *top-level
        window*, and the kwargs editor rebuilt here on every panel switch
        was left as exactly that: a stray, fully-populated window Qt could
        show on its own afterwards, floating over the application with its
        own title bar.

        ``takeAt`` has already removed it from the layout, so keeping the
        parent costs nothing and means the widget can never become a window
        in the first place - not even for the one event-loop pass between
        ``deleteLater`` and the deletion actually happening, which is a real
        window on screen if Qt shows it in the meantime.
        """
        while layout.count():
            item = layout.takeAt(0)
            child = item.widget() if item is not None else None
            if child is not None:
                child.hide()
                child.deleteLater()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        widget.setContentsMargins(0, 0, 0, 0)
        widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        widget.setMinimumHeight(self.KWARGS_PANEL_MIN_HEIGHT)
        widget.setMaximumHeight(MAX_QT_HEIGHT)
        layout.addWidget(widget, 0)
