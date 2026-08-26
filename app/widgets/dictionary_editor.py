"""Generic key/value editor driven by a schema.

Given a mapping of names to metadata (type, default, allowed values, group,
description), the panel builds the right inline editor per entry: colour picker,
checkbox, spin box, combo, or free text.  It is what makes the renderer
``Kwargs`` schemas and the rcParams override table editable without either of
them writing UI code.
"""
from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QSizePolicy, QSpinBox, QStyledItemDelegate,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from app.widgets.color_combo import MatplotlibColorCombo
import cycler as _cycler
from app.widgets.line_combo import LineStyleCombo
from app.widgets.marker_combo import MarkerStyleCombo
from app.styles.style import (
    MARGIN_NESTED, SPACING_TIGHT, action_presentation, apply_dialog_shell,
    configure_combo_width, create_action_button, create_section_title,
    mark_editor_panel, stdSizeAndlayout,
)
from app.utils.i18n import _


_KIND_BY_EXACT_KEY: dict[str, str] = {
    "color": "color",
    "c": "color",
    "facecolor": "color",
    "facecolors": "color",
    "edgecolor": "color",
    "edgecolors": "color",
    "markerfacecolor": "color",
    "markeredgecolor": "color",
    "mfc": "color",
    "mec": "color",
    "ecolor": "color",
    "labelcolor": "color",
    "linestyle": "linestyle",
    "ls": "linestyle",
    "marker": "marker",
    "alpha": "number",
    "linewidth": "number",
    "linewidths": "number",
    "lw": "number",
    "markersize": "number",
    "ms": "number",
    "capsize": "number",
    "pickradius": "number",
    "zorder": "number",
    "dpi": "number",
    "visible": "bool",
    "animated": "bool",
    "clip_on": "bool",
    "in_layout": "bool",
    "rasterized": "bool",
    "fill": "bool",
    "frameon": "bool",
    "show_legend": "bool",
    "show_in_legend": "bool",
    "sort_x": "bool",
    "log": "bool",
    "joinstyle": "joinstyle",
    "capstyle": "capstyle",
    "axes.prop_cycle": "cycler",
    "prop_cycle": "cycler",
}

_SUFFIX_KIND_RULES: tuple[tuple[str, str], ...] = (
    ("color", "color"),
    ("edgecolor", "color"),
    ("facecolor", "color"),
    ("labelcolor", "color"),
    ("linestyle", "linestyle"),
    ("marker", "marker"),
    ("linewidth", "number"),
    ("markersize", "number"),
    ("dpi", "number"),
    ("alpha", "number"),
    ("joinstyle", "joinstyle"),
    ("capstyle", "capstyle"),
)

_JOINSTYLE_CHOICES = ["miter", "round", "bevel"]
_CAPSTYLE_CHOICES = ["butt", "round", "projecting"]
_GROUP_ORDER = [
    "Appearance",
    "Line",
    "Marker",
    "Patch",
    "Error bars",
    "Geometry",
    "Text",
    "Behavior",
    "Other",
]

_MIN_ROW_HEIGHT = 30
_EDITOR_MIN_HEIGHT = 26
_DEFAULT_PROPERTY_COLUMN_WIDTH = 230
_DEFAULT_VALUE_COLUMN_WIDTH = 260
_DEFAULT_MIN_PANEL_HEIGHT = 360


def _normalize_meta(meta: object) -> dict[str, Any]:
    if isinstance(meta, dict):
        return dict(meta)
    return {"default": meta}


def _value_to_text(value: object) -> str:
    return "" if value is None else str(value)


def _parse_text_value(text: str) -> object:
    stripped = text.strip()
    if stripped == "" or stripped.lower() in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(stripped)
    except Exception:
        return text


def _make_check_item(item: QTreeWidgetItem, value: Any) -> None:
    """Turn one row's value column into an always-visible checkbox.

    A boolean used to be text that became a checkbox only while the row was
    being edited: two clicks to change, and on macOS the editor's box sat
    where the delegate put it rather than where the row was, so a click often
    missed it entirely and the setting appeared not to work.

    A check state is drawn by the view itself, so the box is there whether or
    not anything is being edited, and one click toggles it on every platform.
    The text is cleared because the box already says True or False, and a
    label beside it would only be a second answer to the same question.
    """
    item.setFlags(
        (item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable
    )
    item.setText(1, "")
    item.setCheckState(
        1, Qt.CheckState.Checked if bool(value) else Qt.CheckState.Unchecked
    )


def _kind_for_key(key: str, meta: Mapping[str, Any]) -> str:
    explicit = meta.get("kind", meta.get("editor"))
    if explicit:
        return str(explicit).strip().lower()

    value_type = meta.get("type")
    if isinstance(value_type, list):
        return "enum"
    if value_type is bool:
        return "bool"
    if value_type in (int, float):
        return "number"

    lower = key.strip().lower()
    if lower in _KIND_BY_EXACT_KEY:
        return _KIND_BY_EXACT_KEY[lower]
    for suffix, kind in _SUFFIX_KIND_RULES:
        if lower.endswith(suffix):
            return kind

    default = meta.get("default")
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, (int, float)):
        return "number"
    return "string"


def _group_for_key(key: str, meta: Mapping[str, Any], kind: str) -> str:
    explicit = meta.get("group")
    if explicit:
        return str(explicit)

    lower = key.strip().lower()
    if kind == "linestyle" or lower in {"linewidth", "linewidths", "lw", "alpha"}:
        return "Line"
    if kind == "marker" or lower.startswith("marker") or lower in {"mfc", "mec", "ms"}:
        return "Marker"
    if lower in {"facecolor", "facecolors", "edgecolor", "edgecolors", "hatch", "fill"}:
        return "Patch"
    if lower in {"xerr", "yerr", "ecolor", "capsize", "error_kw"}:
        return "Error bars"
    if lower in {"width", "height", "bottom", "left", "align"}:
        return "Geometry"
    if lower in {"label", "tick_label", "url", "gid"}:
        return "Text"
    if kind == "bool" or lower in {"zorder", "picker", "snap", "clip_on", "animated", "rasterized"}:
        return "Behavior"
    if kind == "color":
        return "Appearance"
    return "Other"


_CYCLER_PRESETS: dict[str, dict[str, list[str]]] = {
    "color": {
        "Matplotlib default": ["1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b", "e377c2", "7f7f7f", "bcbd22", "17becf"],
        "Colorblind friendly": ["0072b2", "e69f00", "009e73", "cc79a7", "56b4e9", "d55e00", "f0e442", "000000"],
        "Classic": ["0000ff", "008000", "ff0000", "00bfbf", "bf00bf", "bfbf00", "000000"],
    },
    "linestyle": {
        "Basic": ["-", "--", "-.", ":"],
        "Solid and dashed": ["-", "--"],
        "Dashed family": ["--", "-.", ":"],
    },
    "marker": {
        "Common": ["o", "s", "^", "D", "v", "P", "X", "*"],
        "Geometric": ["o", "s", "^", "v", "<", ">", "D", "d"],
        "Points and lines": [".", ",", "o", "+", "x", "|", "_"],
    },
}
_CYCLER_CHOICES: dict[str, list[str]] = {
    "color": ["1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b", "e377c2", "7f7f7f", "bcbd22", "17becf"],
    "linestyle": ["-", "--", "-.", ":"],
    "marker": ["o", "s", "^", "v", "<", ">", "D", "d", "p", "h", "8", "P", "X", "*", "+", "x", ".", ",", "|", "_"],
}
_CYCLER_LABELS = {"color": "Colors", "linestyle": "Line styles", "marker": "Marker styles"}


def _strip_color_hash(value: object) -> str:
    text = str(value).strip()
    return text[1:] if text.startswith("#") else text


def _cycler_property(value: object, meta: Mapping[str, Any]) -> str:
    """Resolve the one property edited by this cycler row."""
    explicit = str(meta.get("cycler_property", meta.get("property", ""))).strip().lower()
    aliases = {"colors": "color", "line": "linestyle", "line_style": "linestyle", "markers": "marker"}
    explicit = aliases.get(explicit, explicit)
    if explicit in _CYCLER_PRESETS:
        return explicit
    if isinstance(value, _cycler.Cycler):
        keys = [str(key) for key in value.keys if str(key) in _CYCLER_PRESETS]
        if keys:
            return keys[0]
    text = str(value or "")
    for prop in ("color", "linestyle", "marker"):
        if f"cycler('{prop}'" in text or f'cycler("{prop}"' in text:
            return prop
    return "color"


def _cycler_values(value: object, prop: str) -> list[str]:
    if isinstance(value, _cycler.Cycler):
        values = [row[prop] for row in value if prop in row]
    else:
        import re
        pattern = re.compile(rf"cycler\(\s*['\"]?{re.escape(prop)}['\"]?\s*,\s*(\[[^\]]*\])\s*\)")
        match = pattern.search(str(value or ""))
        if not match:
            return []
        try:
            values = list(ast.literal_eval(match.group(1)))
        except (SyntaxError, ValueError, TypeError):
            return []
    return [_strip_color_hash(item) if prop == "color" else str(item) for item in values]


def _button(parent: QWidget, layout: QHBoxLayout, action_id: str, callback, label: str, tooltip: str):
    """Create a catalogue-backed action button with context-specific wording."""
    icon, _catalog_label, _catalog_tip = action_presentation(action_id)
    return create_action_button(
        parent=parent, action_id=action_id, action=callback, layout=layout,
        presentation=(icon, _(label), _(tooltip)),
    )


class CyclerEditorDialog(QDialog):
    """Edit exactly one Matplotlib cycler property."""

    def __init__(self, prop: str, value: object = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if prop not in _CYCLER_PRESETS:
            raise ValueError(f"Unsupported cycler property: {prop}")
        self.prop = prop
        self.setWindowTitle(_("{0} Cycler Editor").format(_CYCLER_LABELS[prop]))

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="small")

        root.addWidget(create_section_title(_(_CYCLER_LABELS[prop]), self))

        add_row = QHBoxLayout()
        stdSizeAndlayout(add_row)
        add_row.setSpacing(SPACING_TIGHT)
        self.value_combo = QComboBox(self)
        self.value_combo.setEditable(prop == "color")
        self.value_combo.addItems(_CYCLER_CHOICES[prop])
        configure_combo_width(self.value_combo)
        add_row.addWidget(self.value_combo, 1)
        _button(self, add_row, "add", self._add_current, "Add", "Add the selected value to the cycler")
        root.addLayout(add_row)

        self.values_list = QListWidget(self)
        self.values_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.values_list.setMinimumHeight(150)
        mark_editor_panel(self.values_list)
        root.addWidget(self.values_list, 1)

        list_actions = QHBoxLayout()
        stdSizeAndlayout(list_actions)
        list_actions.setSpacing(SPACING_TIGHT)
        _button(self, list_actions, "up", lambda: self._move(-1), "Up", "Move the selected value up")
        _button(self, list_actions, "down", lambda: self._move(1), "Down", "Move the selected value down")
        _button(self, list_actions, "delete", self._remove, "Remove", "Remove the selected values")
        list_actions.addStretch(1)
        root.addLayout(list_actions)

        preset_row = QHBoxLayout()
        stdSizeAndlayout(preset_row)
        preset_row.setSpacing(SPACING_TIGHT)
        preset_row.addWidget(QLabel(_("Defaults:"), self))
        self.preset_combo = QComboBox(self)
        self.preset_combo.addItems(_CYCLER_PRESETS[prop])  # type: ignore
        configure_combo_width(self.preset_combo)
        preset_row.addWidget(self.preset_combo, 1)
        _button(self, preset_row, "add", lambda: self._apply_preset(False), "Add", "Append the selected default cycler")
        _button(self, preset_row, "commit", lambda: self._apply_preset(True), "Use", "Replace the current list with the selected default cycler")
        root.addLayout(preset_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self._accept_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.set_values(_cycler_values(value, prop))

    def values(self) -> list[str]:
        return [str(self.values_list.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(self.values_list.count())]

    def set_values(self, values: Iterable[object]) -> None:
        self.values_list.clear()
        for value in values:
            self._append(str(value))

    def _append(self, value: str) -> None:
        value = value.strip()
        if not value:
            return
        if self.prop == "color":
            value = _strip_color_hash(value)
        item = QListWidgetItem(value)
        item.setData(Qt.ItemDataRole.UserRole, value)
        self.values_list.addItem(item)

    def _add_current(self) -> None:
        self._append(self.value_combo.currentText())

    def _remove(self) -> None:
        for item in reversed(self.values_list.selectedItems()):
            self.values_list.takeItem(self.values_list.row(item))

    def _move(self, delta: int) -> None:
        row = self.values_list.currentRow()
        target = row + delta
        if row < 0 or target < 0 or target >= self.values_list.count():
            return
        item = self.values_list.takeItem(row)
        self.values_list.insertItem(target, item)
        self.values_list.setCurrentRow(target)

    def _apply_preset(self, replace: bool) -> None:
        if replace:
            self.values_list.clear()
        for value in _CYCLER_PRESETS[self.prop][self.preset_combo.currentText()]:
            self._append(value)

    def _accept_valid(self) -> None:
        if not self.values():
            QMessageBox.warning(self, _("Empty cycler"), _("Add at least one value."))
            return
        self.accept()

    def cycler_value(self) -> _cycler.Cycler:
        values = self.values()
        if self.prop == "color":
            values = [value if value.startswith("#") else f"#{value}" for value in values]
        return _cycler.cycler(self.prop, values)


class CyclerValueEditor(QWidget):
    """Inline delegate control for one cycler property."""

    editingFinished = Signal()

    def __init__(self, prop: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.prop = prop
        self._value: object = ""
        layout = QHBoxLayout(self)
        stdSizeAndlayout(layout)
        layout.setContentsMargins(*MARGIN_NESTED)
        layout.setSpacing(SPACING_TIGHT)
        self.preview = QLineEdit(self)
        self.preview.setReadOnly(True)
        layout.addWidget(self.preview, 1)
        _button(self, layout, "edit", self._open, "Edit", "Edit this cycler")

    def set_value(self, value: object) -> None:
        self._value = value
        self.preview.setText(_value_to_text(value))

    def value(self) -> object:
        return self._value

    def _open(self) -> None:
        dialog = CyclerEditorDialog(self.prop, self._value, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._value = dialog.cycler_value()
            self.preview.setText(_value_to_text(self._value))
            self.editingFinished.emit()


class DictValueDelegate(QStyledItemDelegate):
    """Value delegate with matplotlib-style custom editors."""

    def __init__(self, panel: "DictEditorPanel", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel = panel

    def sizeHint(self, option, index):  # noqa: N802, ANN001, ANN201
        size = super().sizeHint(option, index)
        size.setHeight(max(size.height(), _MIN_ROW_HEIGHT))
        return size

    def createEditor(self, parent: QWidget, option, index):  # noqa: N802, ANN001, ANN201
        if index.column() != 1:
            return None
        key = self._panel.key_from_index(index)
        if key is None:
            return None

        meta = self._panel.meta_for_key(key)
        kind = self._panel.kind_for_key(key)

        if kind == "bool":
            # None: the row carries a real checkbox of its own, drawn and
            # clickable whether or not anything is being edited. An editor
            # here would put a second one on top of it - and on macOS that
            # one had to be found and hit before it would take a click at
            # all, which is what made the boolean rows feel broken there.
            return None

        if kind == "color":
            editor = MatplotlibColorCombo(parent, include_none=True)
        elif kind == "linestyle":
            editor = LineStyleCombo(parent)
        elif kind == "marker":
            editor = MarkerStyleCombo(parent)
        elif kind == "cycler":
            prop = _cycler_property(self._panel.value_for_key(key), meta)
            editor = CyclerValueEditor(prop, parent)
            editor.editingFinished.connect(lambda: self.commitData.emit(editor))
        elif kind == "bool":
            editor = QCheckBox(parent)
        elif kind == "number":
            if meta.get("type") is int:
                editor = QSpinBox(parent)
                editor.setRange(
                    int(str(meta.get("min", -2147483648))),
                    int(str(meta.get("max", 2147483647))),
                )
                editor.setSingleStep(int(str(meta.get("step", 1))))
            else:
                editor = QDoubleSpinBox(parent)
                editor.setRange(
                    float(str(meta.get("min", -1.0e12))),
                    float(str(meta.get("max", 1.0e12))),
                )
                editor.setDecimals(int(str(meta.get("decimals", 6))))
                editor.setSingleStep(float(str(meta.get("step", 0.1))))
        elif kind in {"enum", "joinstyle", "capstyle", "loc"}:
            editor = QComboBox(parent)
            if kind == "joinstyle":
                choices = _JOINSTYLE_CHOICES
            elif kind == "capstyle":
                choices = _CAPSTYLE_CHOICES
            else:
                choices = list(meta.get("choices", meta.get("type", [])) or [])
            for choice in choices:
                editor.addItem("" if choice is None else str(choice), choice)
        else:
            editor = QLineEdit(parent)

        editor.setMinimumHeight(_EDITOR_MIN_HEIGHT)
        return editor

    def setEditorData(self, editor, index) -> None:  # noqa: N802, ANN001
        key = self._panel.key_from_index(index)
        if key is None:
            return
        value = self._panel.value_for_key(key)
        text = _value_to_text(value)

        if isinstance(editor, CyclerValueEditor):
            editor.set_value(value)
            return
        if isinstance(editor, MatplotlibColorCombo):
            if text == "":
                editor.setCurrentIndex(0)
            elif not editor.set_current_hex(text):
                editor.set_current_name(text)
            return
        if isinstance(editor, LineStyleCombo):
            editor.set_current_linestyle(text)
            return
        if isinstance(editor, MarkerStyleCombo):
            editor.set_current_marker(text)
            return
        if isinstance(editor, QCheckBox):
            editor.setChecked(bool(value))
            return
        if isinstance(editor, QDoubleSpinBox):
            if text:
                editor.setValue(float(text))
            return
        if isinstance(editor, QSpinBox):
            if text:
                editor.setValue(int(text))
            return
        if isinstance(editor, QComboBox):
            idx = editor.findData(value)
            if idx < 0:
                idx = editor.findText(text)
            if idx >= 0:
                editor.setCurrentIndex(idx)
            elif text:
                editor.insertItem(0, text, value)
                editor.setCurrentIndex(0)
            return
        if isinstance(editor, QLineEdit):
            editor.setText(text)
            editor.selectAll()
            return

    def setModelData(self, editor, model, index) -> None:  # noqa: N802, ANN001
        key = self._panel.key_from_index(index)
        if key is None:
            return

        if isinstance(editor, CyclerValueEditor):
            value: object = editor.value()
        elif isinstance(editor, MatplotlibColorCombo):
            value = editor.current_hex() or None
        elif isinstance(editor, LineStyleCombo):
            value = editor.current_linestyle()
        elif isinstance(editor, MarkerStyleCombo):
            value = editor.current_marker()
        elif isinstance(editor, QCheckBox):
            value = editor.isChecked()
        elif isinstance(editor, QDoubleSpinBox):
            value = editor.value()
        elif isinstance(editor, QSpinBox):
            value = editor.value()
        elif isinstance(editor, QComboBox):
            value = editor.currentData()
        elif isinstance(editor, QLineEdit):
            value = _parse_text_value(editor.text())
        else:
            value = None

        self._panel.set_value_for_key(key, value)
        model.setData(index, _value_to_text(value), Qt.ItemDataRole.DisplayRole)
        model.setData(index, value, Qt.ItemDataRole.UserRole)


class DictEditorPanel(QWidget):
    """Qt-Designer-like property editor for dictionary values.

    Groups are collapsed by default. Searching temporarily expands groups that
    contain matches. The widget expands vertically and horizontally to fill the
    host layout.
    """

    valuesChanged = Signal(dict)

    def __init__(self, config: Mapping[str, object] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.config: dict[str, dict[str, Any]] = {}
        self._values: dict[str, object] = {}
        self._key_items: dict[str, QTreeWidgetItem] = {}
        self._group_items: dict[str, QTreeWidgetItem] = {}
        self._updating = False

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(_DEFAULT_MIN_PANEL_HEIGHT)

        layout = QVBoxLayout(self)
        stdSizeAndlayout(layout)

        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText(_("Search properties..."))
        stdSizeAndlayout(self.search_edit)
        self.search_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search_edit, 0)

        self.tree = QTreeWidget(self)
        self.tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        mark_editor_panel(self.tree)
        self.tree.setMinimumHeight(_DEFAULT_MIN_PANEL_HEIGHT - 32)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Property", "Value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(False)
        self.tree.setIconSize(QSize(20, 20))
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(
            QTreeWidget.EditTrigger.DoubleClicked
            | QTreeWidget.EditTrigger.EditKeyPressed
        )
        self.tree.setItemDelegateForColumn(1, DictValueDelegate(self, self.tree))
        self._configure_header()
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree, 1)

        self.set_config(config or {})

    def _configure_header(self) -> None:
        header = self.tree.header()
        header.setSectionsMovable(False)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setMinimumSectionSize(60)
        header.resizeSection(0, _DEFAULT_PROPERTY_COLUMN_WIDTH)
        header.resizeSection(1, _DEFAULT_VALUE_COLUMN_WIDTH)
        self.tree.setColumnWidth(0, _DEFAULT_PROPERTY_COLUMN_WIDTH)
        self.tree.setColumnWidth(1, _DEFAULT_VALUE_COLUMN_WIDTH)

    def set_config(self, config: Mapping[str, object]) -> None:
        self.commit_pending_edits()
        selected_key = self.current_key()
        filter_text = self.filter_text()
        self._updating = True
        self.config = {key: _normalize_meta(meta) for key, meta in config.items()}
        self._values = {key: meta.get("default") for key, meta in self.config.items()}
        self._rebuild_tree()
        self._updating = False
        self.set_filter_text(filter_text)
        self.set_current_key(selected_key)
        self._emit_values_changed()

    def set_values(self, values: Mapping[str, object] | None) -> None:
        if not values:
            return
        self.commit_pending_edits()
        selected_key = self.current_key()
        filter_text = self.filter_text()
        self._updating = True
        for key, value in values.items():
            if key in self.config:
                self._values[key] = value
                self._update_item_value(key)
        self._updating = False
        self.set_filter_text(filter_text)
        self.set_current_key(selected_key)
        self._emit_values_changed()

    def get_values(self) -> dict[str, object]:
        return dict(self._values)

    def reset_to_defaults(self) -> None:
        self.set_values({key: meta.get("default") for key, meta in self.config.items()})

    def commit_pending_edits(self) -> None:
        focus_widget = QApplication.focusWidget()
        widget = focus_widget
        while widget is not None:
            if isinstance(widget, QComboBox):
                widget.hidePopup()
            widget = widget.parentWidget()
        if focus_widget is not None:
            focus_widget.clearFocus()
        self.tree.clearFocus()
        for item in self._key_items.values():
            self.tree.closePersistentEditor(item, 1)

    def filter_text(self) -> str:
        return self.search_edit.text()

    def set_filter_text(self, text: str | None) -> None:
        self.search_edit.blockSignals(True)
        self.search_edit.setText(text or "")
        self.search_edit.blockSignals(False)
        self._apply_filter()

    def current_key(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        key = item.data(0, Qt.ItemDataRole.UserRole)
        return str(key) if key else None

    def set_current_key(self, key: str | None) -> None:
        if key is None:
            return
        item = self._key_items.get(key)
        if item is None:
            return
        parent = item.parent()
        if parent is not None:
            parent.setExpanded(True)
        self.tree.setCurrentItem(item, 1)
        self.tree.scrollToItem(item)

    def key_from_index(self, index) -> str | None:  # noqa: ANN001
        key = index.siblingAtColumn(0).data(Qt.ItemDataRole.UserRole)
        return str(key) if key else None

    def meta_for_key(self, key: str) -> dict[str, Any]:
        return self.config[key]

    def kind_for_key(self, key: str) -> str:
        return _kind_for_key(key, self.config[key])

    def value_for_key(self, key: str) -> object:
        return self._values.get(key)

    def set_value_for_key(self, key: str, value: object) -> None:
        selected_key = self.current_key()
        filter_text = self.filter_text()
        self._values[key] = value
        self._update_item_value(key)
        self.set_filter_text(filter_text)
        self.set_current_key(selected_key or key)
        self._emit_values_changed()

    def _rebuild_tree(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        self._key_items.clear()
        self._group_items.clear()

        grouped: dict[str, list[str]] = {}
        for key, meta in self.config.items():
            group = _group_for_key(key, meta, self.kind_for_key(key))
            grouped.setdefault(group, []).append(key)

        ordered_groups = [group for group in _GROUP_ORDER if group in grouped]
        ordered_groups.extend(sorted(group for group in grouped if group not in ordered_groups))

        for group in ordered_groups:
            group_item = QTreeWidgetItem(self.tree, [group, ""])
            group_item.setData(0, Qt.ItemDataRole.UserRole, None)
            group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            group_item.setExpanded(False)
            group_item.setFirstColumnSpanned(True)
            group_item.setSizeHint(0, QSize(0, _MIN_ROW_HEIGHT))
            group_item.setSizeHint(1, QSize(0, _MIN_ROW_HEIGHT))
            self._group_items[group] = group_item

            for key in grouped[group]:
                meta = self.config[key]
                value = self._values.get(key)
                label = str(meta.get("label", key))
                item = QTreeWidgetItem(group_item, [label, _value_to_text(value)])
                item.setData(0, Qt.ItemDataRole.UserRole, key)
                item.setData(1, Qt.ItemDataRole.UserRole, value)
                item.setToolTip(0, str(meta.get("description", "")))
                item.setToolTip(1, str(meta.get("description", f"{key} ({self.kind_for_key(key)})")))
                if self.kind_for_key(key) == "bool":
                    _make_check_item(item, value)
                else:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                item.setSizeHint(0, QSize(0, _MIN_ROW_HEIGHT))
                item.setSizeHint(1, QSize(0, _MIN_ROW_HEIGHT))
                self._key_items[key] = item

        self.tree.blockSignals(False)
        self._configure_header()
        self._apply_filter()

    def _update_item_value(self, key: str) -> None:
        item = self._key_items.get(key)
        if item is None:
            return
        value = self._values.get(key)
        self.tree.blockSignals(True)
        if self.kind_for_key(key) == "bool":
            _make_check_item(item, value)
        else:
            item.setText(1, _value_to_text(value))
        item.setData(1, Qt.ItemDataRole.UserRole, value)
        self.tree.blockSignals(False)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 1:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key:
            return
        key_str = str(key)

        if self.kind_for_key(key_str) == "bool":
            # The box is the value. Reading item.text(1) here would parse the
            # empty label beside it and turn every toggle into False.
            value: object = item.checkState(1) == Qt.CheckState.Checked
        else:
            value = self._coerce_value_from_text(key_str, item.text(1))

        self._values[key_str] = value
        item.setData(1, Qt.ItemDataRole.UserRole, value)
        self._emit_values_changed()

    def _coerce_value_from_text(self, key: str, text: str) -> object:
        kind = self.kind_for_key(key)
        meta = self.config[key]
        if text.strip() == "":
            return None
        if kind == "bool":
            return text.strip().lower() in {"true", "1", "yes", "on"}
        if kind == "number":
            return int(text) if meta.get("type") is int else float(text)
        return _parse_text_value(text)

    def _apply_filter(self) -> None:
        needle = self.search_edit.text().strip().lower()
        for group, group_item in self._group_items.items():
            group_match = needle in group.lower() if needle else True
            visible_children = 0
            for index in range(group_item.childCount()):
                child = group_item.child(index)
                key = str(child.data(0, Qt.ItemDataRole.UserRole) or "")
                meta = self.config[key]
                haystack = " ".join(
                    [
                        key,
                        str(meta.get("label", "")),
                        str(meta.get("description", "")),
                        self.kind_for_key(key),
                        _value_to_text(self._values.get(key)),
                        group,
                    ]
                ).lower()
                visible = not needle or needle in haystack or group_match
                child.setHidden(not visible)
                if visible:
                    visible_children += 1
            group_item.setHidden(bool(needle and visible_children == 0 and not group_match))
            if needle and visible_children:
                group_item.setExpanded(True)
            elif not needle:
                group_item.setExpanded(False)

    def _emit_values_changed(self) -> None:
        if not self._updating:
            self.valuesChanged.emit(self.get_values())