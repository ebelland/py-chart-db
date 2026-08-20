"""Dialog for creating a chart and its first axes and series.

Chart types are discovered dynamically from the renderer scanner, so a renderer
dropped into ``app/charts`` appears here with its name, documentation link and
role requirements without any registration step.

Renderers deliberately have no icon.  Every chart type had one, each needed a
file in up to three platform folders, and none of them told the user anything
the name did not - the picker is a list of names, and a 32-pixel glyph of a
scatter plot next to the words "Scatter Plot" is decoration, not information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QListWidget, QListWidgetItem, QSizePolicy, QWidget, QHBoxLayout, QLabel, QVBoxLayout, QRadioButton, QFormLayout, QLineEdit, QComboBox, QPlainTextEdit, QScrollArea, QSplitter, QToolBox

from app.scanners.axis_renderer_scanner import (
    get_renderer,
    import_class_from_file,
    renderers,
)
from app.charts.base_axis import BaseAxisRenderer
from app.data.sqlite_repo import SqliteRepo
from app.data.data_source import quote_identifier
from app.utils.messages import show_message
from app.styles.style import (
    apply_dialog_shell,
    apply_toolbox_header_metrics,
    create_doc_link,
    set_doc_link,
    create_card_widget,
    create_action_button,
    create_section_title,
    load_icon,
    mark_editor_panel,
    stdPlainTextEdit,
    stdSizeAndlayout,
)
from app.utils.i18n import _


@dataclass(slots=True)
class SeriesDraft:
    """In-memory series configuration edited in the dialog."""
    name: str
    sql: str
    roles: dict[str, str | None] = field(default_factory=dict)
    user_named: bool = False


@dataclass(slots=True)
class NewPlotTabResult:
    """Result returned by the dialog."""
    figure_id: int
    axis_id: int
    series_count: int


class NewPlotTabDialog(QDialog):
    """Dialog used to create a figure/axis and one or more plot series.

    Role selection is renderer-driven: the scrollable roles panel is rebuilt
    from ``renderer.required`` and ``renderer.optional`` whenever the selected
    renderer changes.  Each role can be mapped to a table/query column and is
    persisted through ``SqliteRepo.create_series_descriptor(..., roles=...)``.
    """
    Icon = """
        <path d="M4 19.5h16"/>
        <path d="M4.5 19V5"/>
        <path d="M7 15l3.2-3.2 2.5 2.5L18 8"/>
        <path d="M15.5 8H18v2.5"/>
        """
    def __init__(
        self,
        repo: SqliteRepo,
        *,
        current_figure_id: int | None = None,
        current_table: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._repo = repo
        self._current_figure_id = current_figure_id
        self._current_table = current_table
        self._result: NewPlotTabResult | None = None
        self._series_by_item: dict[int, SeriesDraft] = {}
        self.renderer: dict[str, object] | None = None
        self._renderer_class: type[BaseAxisRenderer] | None = None
        self._figure_name_user_modified = False
        self._axis_name_user_modified = False

        self.setWindowTitle(_("New plot"))
        self.setWindowIcon(load_icon("plot"))
        self.setModal(False)
        # Size and root padding come from the shared dialog shell.

        # ------------------------------------------------------------------
        # Left: renderer list.
        # ------------------------------------------------------------------
        self._search_types = QLineEdit()
        self._search_types.setPlaceholderText(_("Search chart types..."))
        self._search_types.setToolTip(
            _("Filter by name or category. Sections with no match are hidden.")
        )
        # The clear button is the standard search affordance on both platforms,
        # and it is one fewer widget than a button beside the field.
        self._search_types.setClearButtonEnabled(True)

        # One accordion section per Matplotlib category, each holding the
        # renderers of that family.  Same widget as the properties panels, so
        # the two sides of the window behave the same way.
        self._types_toolbox = QToolBox()
        self._lists_by_category: dict[str, QListWidget] = {}
        self._fill_renderer_toolbox()

        self._select_first_renderer()
        selected = self._current_item()
        if selected is not None:
            self.renderer = get_renderer(str(selected.data(Qt.ItemDataRole.UserRole)))
            self._renderer_class = self._load_renderer_class()

        # ------------------------------------------------------------------
        # Left: selected renderer details.
        # ------------------------------------------------------------------
        self._renderer_name = QLabel()
        self._renderer_name.setProperty("role", "subtitle")
        self._renderer_name.setStyleSheet("font-weight: 600;")
        self._renderer_name.setWordWrap(True)

        self._renderer_description = QLabel()
        self._renderer_description.setWordWrap(True)
        self._renderer_description.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._renderer_link = create_doc_link()
        self._renderer_link.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )

        renderer_details = create_card_widget(self, "rendererDetailsCard")
        renderer_details_layout = QVBoxLayout(renderer_details)
        stdSizeAndlayout(renderer_details_layout)
        renderer_details_layout.addWidget(self._renderer_name)

        renderer_details_layout.addWidget(self._renderer_link)
        self._renderer_details = renderer_details

        # ------------------------------------------------------------------
        # Left: target figure/axis.
        # ------------------------------------------------------------------
        self._rb_new_figure = QRadioButton(_("Create new figure (new tab)"))
        self._rb_current_figure = QRadioButton(_("Use current figure"))
        self._rb_new_figure.setChecked(True)

        if self._current_figure_id is None:
            self._rb_current_figure.setEnabled(False)
            self._rb_current_figure.setToolTip(_("No current figure selected."))

        self._edit_figure_name = QLineEdit()
        self._edit_figure_name.setPlaceholderText(_("Figure name"))

        fig_group = create_card_widget(self, "plotTargetCard")
        fig_main = QVBoxLayout(fig_group)
        stdSizeAndlayout(fig_main)

        target_title = create_section_title(_("Target"), fig_group)
        target_title.setStyleSheet("font-weight: 700;")
        fig_main.addWidget(target_title)

        fig_form = QWidget(fig_group)
        fig_layout = QFormLayout(fig_form)
        stdSizeAndlayout(fig_layout)
        fig_layout.addRow(self._rb_new_figure)
        fig_layout.addRow(self._rb_current_figure)
        fig_layout.addRow(_("Name:"), self._edit_figure_name)
        fig_main.addWidget(fig_form)

        self._rb_new_axis = QRadioButton(_("Add new axis"))
        self._rb_existing_axis = QRadioButton(_("Use existing axis"))
        self._rb_new_axis.setChecked(True)

        self._combo_axes = QComboBox()
        self._combo_axes.setEnabled(False)

        self._edit_axis_name = QLineEdit()
        self._edit_axis_name.setPlaceholderText(_("Axis name"))

        axis_form = QWidget(fig_group)
        axis_layout = QFormLayout(axis_form)
        stdSizeAndlayout(axis_layout)
        axis_layout.addRow(self._rb_new_axis)
        axis_layout.addRow(self._rb_existing_axis)
        axis_layout.addRow(_("Axis name:"), self._edit_axis_name)

        axis_row = QWidget(fig_group)
        axis_row_layout = QHBoxLayout(axis_row)
        stdSizeAndlayout(axis_row_layout)
        axis_row_layout.addWidget(QLabel(_("Axis:")))
        axis_row_layout.addWidget(self._combo_axes, 1)
        axis_layout.addRow(axis_row)
        fig_main.addWidget(axis_form)

        # ------------------------------------------------------------------
        # Series list.
        # ------------------------------------------------------------------
        self._series_list = QListWidget()
        self._series_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._series_list.setAlternatingRowColors(True)
        self._series_list.setUniformItemSizes(True)
        self._series_list.setSpacing(1)
        mark_editor_panel(self._series_list)
        self._series_list.setMinimumWidth(180)
        self._series_list.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        series_list_group = create_card_widget(self, "plotSeriesListCard")
        series_list_group.setMinimumWidth(220)
        series_list_group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        series_list_layout = QVBoxLayout(series_list_group)
        stdSizeAndlayout(series_list_layout)

        series_header = QWidget(series_list_group)
        series_header_layout = QHBoxLayout(series_header)
        stdSizeAndlayout(series_header_layout)

        series_title = create_section_title(_("Series"), series_header)
        series_title.setStyleSheet("font-weight: 700;")
        series_header_layout.addWidget(series_title)
        series_header_layout.addStretch(1)
        self._btn_add_series = create_action_button(
            parent=series_header,
            action_id="add_series",
            action=self._on_add_series,
            layout=series_header_layout,
        )
        self._btn_remove_series = create_action_button(
                                      parent=series_header,
                                      action_id="delete",
                                      action=self._on_remove_series,
                                      layout=series_header_layout,
                                  )

        series_list_layout.addWidget(series_header)
        series_list_layout.addWidget(self._series_list, 1)


        # Series editor.
        self._combo_table = QComboBox()
        self._combo_table.setEditable(False)

        self._edit_series_name = QLineEdit()
        self._edit_series_name.setPlaceholderText(_("Series name"))

        self._edit_sql = QPlainTextEdit()
        stdPlainTextEdit(self._edit_sql)     
        self._edit_sql.setPlaceholderText(_("SQL query (SELECT / WITH ...)"))


        self._role_combos: dict[str, QComboBox] = {}
        self._role_labels: dict[str, QLabel] = {}
        self._roles_panel = QWidget()
        self._roles_layout = QFormLayout(self._roles_panel)
        stdSizeAndlayout(self._roles_layout)

        self._roles_scroll = QScrollArea()
        self._roles_scroll.setWidgetResizable(True)
        self._roles_scroll.setMinimumHeight(110)
        stdSizeAndlayout(self._roles_scroll)
        mark_editor_panel(self._roles_scroll)
        self._roles_scroll.setWidget(self._roles_panel)
        self._rebuild_role_combos()

        editor_group = create_card_widget(self, "plotSeriesEditorCard")
        editor_layout = QVBoxLayout(editor_group)
        stdSizeAndlayout(editor_layout)

        series_editor_title = create_section_title(_("Series"), editor_group)
        series_editor_title.setStyleSheet("font-weight: 700;")
        editor_layout.addWidget(series_editor_title)

        editor_form = QWidget(editor_group)
        editor_form_layout = QFormLayout(editor_form)
        stdSizeAndlayout(editor_form_layout)
        editor_form_layout.addRow(_("Data source:"), self._combo_table)
        editor_form_layout.addRow(_("Series name:"), self._edit_series_name)
        editor_form_layout.addRow(_("Query:"), self._edit_sql)
        editor_form_layout.addRow(self._roles_scroll)
        editor_layout.addWidget(editor_form, 1)

        # Dialog buttons and layout.
        renderer_panel = QWidget()
        renderer_panel.setMinimumWidth(220)
        renderer_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        renderer_layout = QVBoxLayout(renderer_panel)
        stdSizeAndlayout(renderer_layout)

        renderer_title = create_section_title(_("Renderers"), renderer_panel)
        renderer_title.setStyleSheet("font-weight: 700;")
        renderer_layout.addWidget(renderer_title)
        renderer_layout.addWidget(self._search_types, 0)
        renderer_layout.addWidget(self._types_toolbox, 1)
        renderer_layout.addWidget(self._renderer_details, 0)

        target_series_panel = QWidget()
        target_series_panel.setMinimumWidth(260)
        target_series_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        target_series_layout = QVBoxLayout(target_series_panel)
        stdSizeAndlayout(target_series_layout)
        target_series_layout.addWidget(fig_group, 0)
        target_series_layout.addWidget(series_list_group, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        editor_group.setMinimumWidth(380)
        editor_group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        splitter.addWidget(renderer_panel)
        splitter.addWidget(editor_group)
        splitter.addWidget(target_series_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([230, 430, 330])

        action_row = QHBoxLayout()
        stdSizeAndlayout(action_row)

        action_row.addStretch(1)
        create_action_button(
            parent=self,
            action_id="apply",
            action=self._on_accept,
            layout=action_row,
        )
        create_action_button(
            parent=self,
            action_id="close",
            action=self.reject,
            layout=action_row,
        )
        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="large")
        root.addWidget(splitter)
        root.addLayout(action_row)
        self.setLayout(root)
    
        # ------------------------------------------------------------------
        # Wiring.
        # ------------------------------------------------------------------

        self._rb_current_figure.toggled.connect(self._sync_controls)
        self._rb_new_axis.toggled.connect(self._sync_controls)
        self._rb_existing_axis.toggled.connect(self._sync_controls)
        self._search_types.textChanged.connect(self._filter_renderer_toolbox)
        for renderer_list in self._lists_by_category.values():
            renderer_list.currentItemChanged.connect(self._on_chart_type_changed)
        self._combo_table.currentIndexChanged.connect(self._on_table_changed)
        self._series_list.currentItemChanged.connect(self._on_series_selected)

        self._edit_figure_name.textEdited.connect(self._on_figure_name_edited)
        self._edit_axis_name.textEdited.connect(self._on_axis_name_edited)
        self._edit_series_name.textEdited.connect(self._on_series_name_changed)
        self._edit_sql.textChanged.connect(self._on_sql_changed)

        self._reload_tables()
        self._reload_axes()
        self._sync_controls()
        self._set_default_figure_name(force=True)
        self._set_default_axis_name(force=True)
        self._update_renderer_details()

        # Start with one ready-to-edit series so the dialog is usable by
        # default and the generated SQL is visible immediately.
        self._on_add_series()

    @property
    def result(self) -> NewPlotTabResult | None:
        """Return the dialog result after accept."""
        return self._result

    # Matplotlib's own order, from https://matplotlib.org/stable/plot_types/.
    # Kept rather than sorted alphabetically so the list reads the same way as
    # the documentation the renderers wrap, and so "Pairwise data" - the plots
    # people reach for most - stays at the top.
    CATEGORY_ORDER: tuple[str, ...] = (
        "Pairwise data",
        "Statistical distributions",
        "Gridded data",
        "Irregularly gridded data",
        "3D and volumetric data",
    )

    def _fill_renderer_toolbox(self) -> None:
        """Build one accordion section per category, each a list of renderers.

        A list per section rather than one list with heading rows: the sections
        collapse, which is the point of an accordion on a panel this narrow, and
        it means the search can hide a whole family in one move.
        """
        for category in self._categories_in_order():
            renderer_list = QListWidget()
            renderer_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            renderer_list.setUniformItemSizes(True)
            mark_editor_panel(renderer_list)

            for renderer in self._renderers_in(category):
                item = QListWidgetItem(str(renderer["value"]))
                item.setData(Qt.ItemDataRole.UserRole, renderer["value"])
                item.setToolTip(str(renderer.get("description") or ""))
                renderer_list.addItem(item)

            self._lists_by_category[category] = renderer_list
            self._types_toolbox.addItem(renderer_list, category)

        apply_toolbox_header_metrics(self._types_toolbox)

    def _categories_in_order(self) -> list[str]:
        """Known categories first, in Matplotlib's order, then any others.

        A renderer with a category nobody has heard of still appears - in its
        own section, at the end - rather than vanishing from the dialog.
        """
        present = {str(renderer.get("category") or "") for renderer in renderers}
        known = [name for name in self.CATEGORY_ORDER if name in present]
        return known + sorted(present - set(known))

    def _renderers_in(self, category: str) -> list[dict]:
        return [r for r in renderers if str(r.get("category") or "") == category]

    # ------------------------------------------------------------------
    # Selection across several lists
    # ------------------------------------------------------------------
    def _current_item(self) -> QListWidgetItem | None:
        """Return the selected renderer item, whichever section holds it.

        The selection lives in one of several QListWidgets, so "the current
        item" is not a property of any single one of them; every reader goes
        through here.
        """
        for renderer_list in self._lists_by_category.values():
            item = renderer_list.currentItem()
            if item is not None and item.isSelected():
                return item
        return None

    def _clear_other_selections(self, keep: QListWidget) -> None:
        """Keep the selection exclusive across sections.

        Qt makes each list exclusive on its own, so without this two sections
        could each show a highlighted row and the dialog would look as if two
        chart types were chosen.
        """
        for renderer_list in self._lists_by_category.values():
            if renderer_list is not keep:
                renderer_list.clearSelection()
                # setCurrentRow(-1) rather than setCurrentItem(None): both
                # clear the current item, but only this one is typed for it.
                renderer_list.setCurrentRow(-1)

    def _select_first_renderer(self) -> None:
        """Select the first renderer of the first section that has one."""
        for index, (category, renderer_list) in enumerate(self._lists_by_category.items()):
            if renderer_list.count() == 0 or renderer_list.isHidden():
                continue
            for row in range(renderer_list.count()):
                if not renderer_list.item(row).isHidden():
                    renderer_list.setCurrentRow(row)
                    self._types_toolbox.setCurrentIndex(index)
                    self._clear_other_selections(renderer_list)
                    return

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _filter_renderer_toolbox(self, text: str) -> None:
        """Show only renderers matching *text*, and only sections that have any.

        Matching on the category as well as the name, so typing "stat" brings
        up the whole statistical family - the categories are half of why the
        list is worth searching.
        """
        needle = str(text or "").strip().lower()

        # Indexed by position, not by header text: the header gains a match
        # count while a search is active, so it is not a stable key.
        for index, category in enumerate(self._lists_by_category):
            renderer_list = self._lists_by_category[category]
            category_matches = needle in category.lower()

            visible = 0
            for row in range(renderer_list.count()):
                item = renderer_list.item(row)
                matches = (
                    not needle
                    or category_matches
                    or needle in str(item.data(Qt.ItemDataRole.UserRole)).lower()
                )
                item.setHidden(not matches)
                visible += int(matches)

            # An empty section is disabled rather than removed: removing pages
            # renumbers every index after it, and the numbering is what maps a
            # section back to its list.
            self._types_toolbox.setItemEnabled(index, visible > 0)
            self._types_toolbox.setItemText(
                index, category if not needle else f"{category}  ({visible})"
            )

        # Bound once: two calls could disagree, and the second one is what the
        # type checker cannot see is non-None.
        current = self._current_item()
        if current is None or current.isHidden():
            self._select_first_renderer()

    def _load_renderer_class(self) -> type[BaseAxisRenderer] | None:
        """Import and return the selected renderer class."""
        if self.renderer is None:
            return None
        renderer_class = import_class_from_file(self.renderer)
        return cast(type[BaseAxisRenderer] | None, renderer_class)

    def _current_renderer_class(self) -> type[BaseAxisRenderer] | None:
        """Return the cached renderer class for the selected renderer dict."""
        if self._renderer_class is None:
            self._renderer_class = self._load_renderer_class()
        return self._renderer_class

    # ------------------------------------------------------------------
    # Renderer metadata.
    # ------------------------------------------------------------------
    def _update_renderer_details(self) -> None:
        """Refresh the name and the documentation link from the renderer class."""
        renderer_class = self._current_renderer_class()
        if renderer_class is None:
            self._renderer_name.clear()
            self._renderer_description.clear()
            self._renderer_link.clear()
            self._renderer_link.hide()
            return

        renderer_name = self._current_renderer_name()
        link = str(renderer_class.Link or "").strip()

        self._renderer_name.setText(renderer_name or renderer_class.__name__)
        self._renderer_description.clear()

        if link:
            set_doc_link(self._renderer_link, _("Documentation"), link)
            self._renderer_link.show()
        else:
            self._renderer_link.clear()
            self._renderer_link.hide()

    # ------------------------------------------------------------------
    # Load UI data.
    # ------------------------------------------------------------------
    def _current_table_name(self) -> str:
        """Return the selected data source table name."""
        table = self._combo_table.currentData()
        if isinstance(table, str) and table.strip():
            return table.strip()
        return ""

    def _current_renderer_name(self) -> str:
        """Return the selected renderer's name.

        From the item's data, not its text: the text is indented so that the
        renderer sits under its category heading, and stripping the indent back
        off would be one more place that has to agree about the layout.
        """
        item = self._current_item()
        if item is None:
            return "Axis"

        return str(item.data(Qt.ItemDataRole.UserRole) or "Axis")

    def _default_figure_name(self) -> str:
        """Default figure name is the selected data source."""
        return self._current_table_name() or "Figure"

    def _default_axis_name(self) -> str:
        """Default axis name is the selected renderer."""
        return self._current_renderer_name()

    def _role_value_from_mapping(
        self,
        roles: dict[str, str | None],
        role_name: str,
    ) -> str | None:
        """Return one role value from a mapping, matching keys case-insensitively."""
        if role_name in roles:
            return roles.get(role_name)
        role_name_lower = role_name.lower()
        for key, value in roles.items():
            if key.lower() == role_name_lower:
                return value
        return None

    def _default_series_name_from_roles(self, roles: dict[str, str | None]) -> str:
        """Default series name is '<Data source> - <Y/Value role>'."""
        table_name = self._current_table_name()
        value_name = (
            self._role_value_from_mapping(roles, "y")
            or self._role_value_from_mapping(roles, "value")
            or self._role_value_from_mapping(roles, "values")
            or self._role_value_from_mapping(roles, "z")
            or "Series"
        )
        if table_name:
            return f"{table_name} - {value_name}"
        return str(value_name)

    def _set_default_figure_name(self, *, force: bool = False) -> None:
        """Set the figure name while preserving manual edits."""
        if self._figure_name_user_modified and not force:
            return
        self._edit_figure_name.blockSignals(True)
        try:
            self._edit_figure_name.setText(self._default_figure_name())
        finally:
            self._edit_figure_name.blockSignals(False)
        self._figure_name_user_modified = False

    def _set_default_axis_name(self, *, force: bool = False) -> None:
        """Set the axis name while preserving manual edits."""
        if self._axis_name_user_modified and not force:
            return
        self._edit_axis_name.blockSignals(True)
        try:
            self._edit_axis_name.setText(self._default_axis_name())
        finally:
            self._edit_axis_name.blockSignals(False)
        self._axis_name_user_modified = False

    def _set_default_series_name(
        self,
        item: QListWidgetItem,
        draft: SeriesDraft,
        *,
        force: bool = False,
    ) -> None:
        """Set the series name while preserving manual edits."""
        if draft.user_named and not force:
            return
        draft.name = self._default_series_name_from_roles(draft.roles)
        item.setText(draft.name)
        if item is self._current_series_item():
            self._edit_series_name.blockSignals(True)
            try:
                self._edit_series_name.setText(draft.name)
            finally:
                self._edit_series_name.blockSignals(False)
        draft.user_named = False

    def _on_figure_name_edited(self, _text: str) -> None:
        self._figure_name_user_modified = True

    def _on_axis_name_edited(self, _text: str) -> None:
        self._axis_name_user_modified = True

    def _reload_tables(self) -> None:
        """Load every data source - tables and saved queries - into the combo.

        Saved queries are marked in the label but stored under the same plain
        name in the item data, so everything downstream keeps working with a
        single string and resolves the kind through the repository.
        """
        self._combo_table.clear()
        df = self._repo.list_data_sources()
        if df is None or df.empty:
            return

        for row in df.itertuples(index=False):
            name = str(row.Table)
            is_query = str(getattr(row, "kind", "table")) == "query"
            self._combo_table.addItem(f"{name}  (query)" if is_query else name, name)

        if self._combo_table.count() <= 0:
            return

        index = 0
        if self._current_table:
            found = self._combo_table.findData(self._current_table)
            if found < 0:
                found = self._combo_table.findText(self._current_table)
            if found >= 0:
                index = found

        self._combo_table.setCurrentIndex(index)
        self._on_table_changed(index)

    def _reload_axes(self) -> None:
        """Load existing axes for the current figure."""
        self._combo_axes.clear()
        if self._current_figure_id is None:
            return

        axes = self._repo.list_axes_for_figure(self._current_figure_id)
        for axis_id, axis_index, title in axes:
            label = f"{axis_index}: {title}" if title else f"{axis_index}: (no title)"
            self._combo_axes.addItem(label, int(axis_id))

        if self._combo_axes.count() == 0:
            self._rb_existing_axis.setEnabled(False)

    def _sync_controls(self) -> None:
        """Enable/disable target controls based on figure/axis choices."""
        using_current = self._rb_current_figure.isChecked()
        self._edit_figure_name.setEnabled(not using_current)

        can_use_existing_axis = using_current and self._combo_axes.count() > 0
        self._rb_existing_axis.setEnabled(can_use_existing_axis)
        self._combo_axes.setEnabled(
            can_use_existing_axis and self._rb_existing_axis.isChecked()
        )
        self._edit_axis_name.setEnabled(self._rb_new_axis.isChecked())

        if not can_use_existing_axis:
            self._rb_new_axis.setChecked(True)

    # ------------------------------------------------------------------
    # Series list management.
    # ------------------------------------------------------------------
    def _set_series_editor_enabled(self, enabled: bool) -> None:
        """Enable/disable all series editor widgets."""
        widgets = (
            self._edit_series_name,
            self._edit_sql,
            self._roles_scroll,
            *self._role_combos.values(),
        )
        for widget in widgets:
            widget.setEnabled(enabled)

    def _current_series_item(self) -> QListWidgetItem | None:
        """Return the selected series item."""
        return self._series_list.currentItem()

    def _draft_for_item(self, item: QListWidgetItem) -> SeriesDraft:
        """Return the draft associated with a QListWidgetItem."""
        key = id(item)
        draft = self._series_by_item.get(key)
        if draft is None:
            draft = SeriesDraft(name=item.text(), sql="", user_named=True)
            self._series_by_item[key] = draft
        return draft

    def _on_add_series(self, _checked: bool = False) -> None:
        """Add a new series draft and select it."""
        roles = self._current_role_mapping()
        name = self._default_series_name_from_roles(roles)
        item = QListWidgetItem(name)
        self._series_list.addItem(item)
        self._series_by_item[id(item)] = SeriesDraft(
            name=name,
            sql=self._default_sql_for_current_table(),
            roles=roles,
            user_named=False,
        )
        self._series_list.setCurrentItem(item)

    def _on_remove_series(self, _checked: bool = False) -> None:
        """Remove the selected series draft."""
        item = self._current_series_item()
        if item is None:
            return

        row = self._series_list.row(item)
        self._series_list.takeItem(row)
        self._series_by_item.pop(id(item), None)

        if self._series_list.count() > 0:
            self._series_list.setCurrentRow(max(0, row - 1))
        else:
            self._set_series_editor_enabled(False)

    # ------------------------------------------------------------------
    # Series editor bindings.
    # ------------------------------------------------------------------
    def _on_series_selected(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Load the selected series draft into the editor."""
        if current is None:
            self._edit_series_name.clear()
            self._edit_sql.clear()
            self._set_series_editor_enabled(False)
            return

        self._set_series_editor_enabled(True)
        self._load_draft_into_editor(self._draft_for_item(current))

    def _load_draft_into_editor(self, draft: SeriesDraft) -> None:
        """Load a draft into the editor without recursive updates."""
        self._edit_series_name.blockSignals(True)
        self._edit_sql.blockSignals(True)
        try:
            self._edit_series_name.setText(draft.name)
            self._edit_sql.setPlainText(draft.sql)
        finally:
            self._edit_series_name.blockSignals(False)
            self._edit_sql.blockSignals(False)

        self._populate_role_columns(self._columns_from_table())
        self._set_role_selection(draft)

    def _set_role_selection(self, draft: SeriesDraft) -> None:
        """Select role combo values from the given draft."""
        for key, combo in self._role_combos.items():
            value = draft.roles.get(key)
            combo.blockSignals(True)
            try:
                if value is None:
                    combo.setCurrentIndex(0)
                    continue
                index = combo.findData(value)
                combo.setCurrentIndex(index if index >= 0 else 0)
            finally:
                combo.blockSignals(False)

    def _current_role_mapping(self) -> dict[str, str | None]:
        """Return the current role-to-column selection."""
        roles: dict[str, str | None] = {}
        for key, combo in self._role_combos.items():
            data = combo.currentData()
            roles[key] = str(data) if isinstance(data, str) and data else None
        return roles

    def _persist_editor_to_draft(self) -> None:
        """Copy editor state into the selected draft."""
        item = self._current_series_item()
        if item is None:
            return

        draft = self._draft_for_item(item)
        fallback_name = item.text()
        draft.name = (
            (self._edit_series_name.text() or fallback_name).strip()
            or fallback_name
        )
        draft.sql = (self._edit_sql.toPlainText() or "").strip()

        draft.roles = self._current_role_mapping()

    def _on_series_name_changed(self, text: str) -> None:
        """Keep the list item and draft name synchronized."""
        item = self._current_series_item()
        if item is None:
            return
        item.setText(text)
        draft = self._draft_for_item(item)
        draft.user_named = True
        self._persist_editor_to_draft()

    def _on_sql_changed(self) -> None:
        """Persist edited SQL into the current draft."""
        self._persist_editor_to_draft()

    def _on_roles_changed(self, _index: int = 0) -> None:
        """Persist role changes and refresh SQL automatically."""
        self._persist_editor_to_draft()
        self._update_sql()

    # ------------------------------------------------------------------
    # Columns / roles.
    # ------------------------------------------------------------------
    def _columns_from_table(self) -> list[str]:
        """Return the columns of the selected source, table or saved query."""
        name = self._combo_table.currentData()
        if not isinstance(name, str) or not name:
            return []

        source = self._repo.get_data_source(name)
        if source is None:
            return []
        return [str(column) for column in self._repo.data_source_columns(source)]

    def _default_sql_for_current_table(self) -> str:
        """Generate a default SELECT using the selected renderer role aliases.

        The FROM fragment comes from the data source, so a saved query becomes
        an inline subquery and a table becomes a quoted name.  The series SQL
        that gets stored is therefore self-contained: it keeps working whatever
        the source was, and a saved query is re-executed on every render rather
        than frozen into a table.
        """
        name = self._combo_table.currentData()
        if not isinstance(name, str) or not name:
            return ""

        source = self._repo.get_data_source(name)
        if source is None:
            return ""

        select_parts: list[str] = []
        for alias, combo in self._role_combos.items():
            column = combo.currentData()
            if isinstance(column, str) and column:
                select_parts.append(
                    f"{quote_identifier(column)} AS {quote_identifier(alias)}"
                )

        projection = ", ".join(select_parts) if select_parts else "*"
        return f"SELECT {projection} FROM {source.from_clause()}"

    def _renderer_role_names(self, kind: str) -> list[str]:
        """Return renderer role names from the renderer class."""
        renderer_class = self._current_renderer_class()
        if renderer_class is None:
            return []

        if kind == "required":
            return list(renderer_class.RequiredRoles)
        if kind == "optional":
            return list(renderer_class.OptionalRoles)
        return []

    def _all_renderer_role_names(self) -> list[str]:
        """Return all renderer roles, required first and then optional."""
        roles: list[str] = []
        for kind in ("required", "optional"):
            for role in self._renderer_role_names(kind):
                if role not in roles:
                    roles.append(role)
        return roles

    def _role_value(self, draft: SeriesDraft, role: str) -> str | None:
        """Return a draft role value, matching role names case-insensitively."""
        if role in draft.roles:
            return draft.roles.get(role)

        role_lower = role.lower()
        for key, value in draft.roles.items():
            if key.lower() == role_lower:
                return value
        return None

    def _role_is_required(self, role: str) -> bool:
        """Return True when the role belongs to renderer.required."""
        required = {
            name.lower()
            for name in self._renderer_role_names("required")
        }
        return role.lower() in required

    def _role_label_widget(self, role: str) -> QLabel:
        """Build a styled label for one role row."""
        label = QLabel(f"{role}:", self._roles_panel)
        if self._role_is_required(role):
            label.setProperty("requiredRole", True)
        return label

    def _rebuild_role_combos(self) -> None:
        """Rebuild the scrollable role panel for the selected renderer."""
        previous = {
            key: combo.currentData()
            for key, combo in self._role_combos.items()
        }

        while self._roles_layout.rowCount():
            self._roles_layout.removeRow(0)
        self._role_combos.clear()
        self._role_labels.clear()

        for role in self._all_renderer_role_names():
            combo = QComboBox(self._roles_panel)
            combo.addItem(_("(none)"), None)

            previous_value = previous.get(role)
            if isinstance(previous_value, str):
                combo.addItem(previous_value, previous_value)
                combo.setCurrentIndex(1)

            combo.currentIndexChanged.connect(self._on_roles_changed)
            label = self._role_label_widget(role)
            self._role_combos[role] = combo
            self._role_labels[role] = label
            self._roles_layout.addRow(label, combo)

        if not self._role_combos:
            self._roles_layout.addRow(
                QLabel(_("No renderer roles declared."), self._roles_panel)
            )

    def _populate_role_columns(self, cols: list[str]) -> None:
        """Fill every role combo with available table or query columns."""
        for _role, combo in self._role_combos.items():
            current = combo.currentData()
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItem(_("(none)"), None)
                for column in cols:
                    combo.addItem(str(column), str(column))
                if isinstance(current, str):
                    index = combo.findData(current)
                    if index >= 0:
                        combo.setCurrentIndex(index)
            finally:
                combo.blockSignals(False)

        # Keep convenient defaults for common X/Y role names.
        lower_to_role = {role.lower(): role for role in self._role_combos}
        if cols:
            x_role = lower_to_role.get("x")
            if x_role and self._role_combos[x_role].currentData() is None:
                self._role_combos[x_role].setCurrentIndex(1)

            y_role = lower_to_role.get("y")
            if (
                y_role
                and self._role_combos[y_role].currentData() is None
                and len(cols) > 1
            ):
                self._role_combos[y_role].setCurrentIndex(2)

    def _on_chart_type_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Refresh role rows after the renderer selection changes."""
        self._persist_editor_to_draft()
        item = current or self._current_item()
        # A category heading carries no data; it is unselectable, but the
        # signal can still arrive with one during a rebuild.
        if item is not None and item.data(Qt.ItemDataRole.UserRole):
            chart_type = str(item.data(Qt.ItemDataRole.UserRole))
            self.renderer = get_renderer(chart_type)
            self._renderer_class = self._load_renderer_class()

        self._update_renderer_details()
        self._set_default_axis_name()
        self._rebuild_role_combos()
        self._populate_role_columns(self._columns_from_table())

        series_item = self._current_series_item()
        if series_item is not None:
            draft = self._draft_for_item(series_item)
            self._set_role_selection(draft)
            draft.roles = self._current_role_mapping()
            self._set_default_series_name(series_item, draft)
        self._update_sql()

    def _on_table_changed(self, _index: int = 0) -> None:
        """Refresh role choices, SQL and automatic names when the data source changes."""
        self._populate_role_columns(self._columns_from_table())
        self._set_default_figure_name()

        item = self._current_series_item()
        if item is None:
            return

        draft = self._draft_for_item(item)
        draft.roles = self._current_role_mapping()
        self._update_sql()
        self._set_default_series_name(item, draft)

    def _update_sql(self) -> None:
        """Regenerate SQL from the current data source and role choices."""
        item = self._current_series_item()
        if item is None:
            return

        draft = self._draft_for_item(item)
        draft.roles = self._current_role_mapping()
        draft.sql = self._default_sql_for_current_table()
        self._set_default_series_name(item, draft)
        self._load_draft_into_editor(draft)

    # ------------------------------------------------------------------
    # Persist.
    # ------------------------------------------------------------------
    def _selected_chart_type(self) -> str:
        """Return selected renderer/chart type and cache the renderer object."""
        item = self._current_item()
        if item is None:
            return "Scatter Plot"

        key = str(item.data(Qt.ItemDataRole.UserRole))
        renderer = get_renderer(key)
        if renderer != self.renderer:
            self.renderer = renderer
            self._renderer_class = self._load_renderer_class()
        return key

    def _validate_series(self, chart_type: str, draft: SeriesDraft) -> bool:
        """Validate SQL and required renderer roles for one draft."""
        if not draft.sql.strip():
            show_message(self, "chart.series_empty_sql", series=draft.name)
            return False

        renderer_class = self._current_renderer_class()
        if renderer_class is None:
            show_message(self, "chart.renderer_not_found", series=draft.name)
            return False

        for role in self._renderer_role_names("required"):
            if not self._role_value(draft, role):
                show_message(self, "chart.role_missing", series=draft.name, role=role)
                return False
        return True

    def _series_roles(self, draft: SeriesDraft) -> dict[str, str]:
        """Return only assigned renderer roles for descriptor persistence."""
        return {
            role: value
            for role, value in draft.roles.items()
            if isinstance(value, str) and value
        }

    def _axis_title_from_series(self, drafts: list[SeriesDraft]) -> str:
        """Infer a readable axis title from selected series names."""
        names = [draft.name.strip() for draft in drafts if draft.name.strip()]
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:3]) + ("..." if len(names) > 3 else "")

    def _axis_label_for_role(self, drafts: list[SeriesDraft], role_name: str) -> str:
        """Infer an axis label from the first mapped role value."""
        for draft in drafts:
            value = self._role_value(draft, role_name)
            if value:
                return value
        return ""

    def _on_accept(self) -> None:
        """Validate and persist the figure/axis/series descriptors."""
        self._persist_editor_to_draft()
        chart_type = self._selected_chart_type()

        drafts = list(self._series_by_item.values())
        if not drafts:
            show_message(self, "chart.no_series")
            return

        for draft in drafts:
            if not self._validate_series(chart_type, draft):
                return

        if self._rb_current_figure.isChecked():
            if self._current_figure_id is None:
                return
            figure_id = int(self._current_figure_id)
        else:
            name = (self._edit_figure_name.text() or "").strip()
            if not name:
                self._edit_figure_name.setFocus(Qt.FocusReason.OtherFocusReason)
                return
            figure_id = self._repo.create_figure_descriptor(
                name=name,
                nrows=1,
                ncols=1,
            )

        if self._rb_existing_axis.isChecked() and self._rb_current_figure.isChecked():
            axis_id_data = self._combo_axes.currentData()
            if axis_id_data is None:
                return
            axis_id = int(axis_id_data)
            self._repo.update_axis_chart_type(axis_id=axis_id, chart_type=chart_type)
        else:
            axis_index = self._repo.next_axis_index(figure_id)
            self._repo.ensure_figure_grid_capacity(
                figure_id=figure_id,
                needed_axes=axis_index + 1,
            )
            axis_id = self._repo.create_axis_descriptor(
                figure_id=figure_id,
                axis_index=axis_index,
                chart_type=chart_type,
                title=(
                    (self._edit_axis_name.text() or "").strip()
                    or self._axis_title_from_series(drafts)
                ),
                x_label=self._axis_label_for_role(drafts, "x"),
                y_label=self._axis_label_for_role(drafts, "y"),
                z_label=self._axis_label_for_role(drafts, "z"),
                options=None,
            )

        created = 0
        for draft in drafts:
            series_index = self._repo.next_series_index(axis_id)
            self._repo.create_series_descriptor(
                axis_id=axis_id,
                series_index=series_index,
                name=draft.name,
                sql_query=draft.sql,
                roles=self._series_roles(draft),
                style={"label": draft.name},
            )
            created += 1

        self._result = NewPlotTabResult(
            figure_id=int(figure_id),
            axis_id=int(axis_id),
            series_count=int(created),
        )
        self.accept()
