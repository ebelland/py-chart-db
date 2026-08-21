"""Property editor for figure-level settings.

Covers the figure name, grid shape, layout mode and margins, suptitle, and the
attached Matplotlib style.  Every change goes through the connected redraw
callback, which is debounced by the main window.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib import rcParams
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.dialogs.edit_mpl_styles_dialog import (
    MplStyleEditorDialog,
    _sanitize_mplstyle_text,
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
from app.utils.config import MPLSTYLES_DIR
from app.utils.figure_metrics import (
    CM_PER_INCH,
    DEFAULT_FIGURE_DPI,
    DEFAULT_FIGURE_SIZE_IN,
    OPT_FIGURE_DPI,
    OPT_FIGURE_HEIGHT_CM,
    OPT_FIGURE_WIDTH_CM,
    figure_metrics_from_options,
)

from app.logs.logger import applogger
from app.utils.hidpi import apply_configured_dpi
from app.utils.i18n import _

def _read_text_any_encoding(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc).replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


class FigurePropertiesWidget(QWidget):
    """Reusable Figure properties editor extracted from the old dialog Figure tab.

    Width, height, and DPI belong to the figure, not to the application: they
    are stored in the connected figure's descriptor options and pushed into
    rcParams only for the duration of a render, so that switching figures picks
    up that figure's own metrics instead of whatever was edited last.
    """

    style_changed = Signal(str)
    grid_layout_requested = Signal(int, int)
    figure_options_requested = Signal(dict)

    # Sentinel combo entry that opens a native file picker instead of naming a
    # style. Keeps "browse anywhere" available even though the dropdown itself
    # can only ever show what MPLSTYLES_DIR contains.
    _BROWSE_SENTINEL = "__browse_mplstyle__"

    # macOS Aqua draws wider spin-box stepper buttons than Fusion/Windows,
    # so a plain Expanding size policy can still leave DPI/width/height
    # visually cramped and mismatched. A shared minimum width keeps all
    # three the same size and readable on every platform.
    FIGURE_SPIN_MIN_WIDTH = 110

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repo = None
        self._figure_id: int | None = None
        self._figure = None
        self._redraw_callback = None
        self._last_valid_style_index = 0

        self._style_combo: QComboBox
       
        self._nrows_combo: QComboBox
        self._ncols_combo: QComboBox
        self._fig_dpi: QSpinBox
        self._fig_width_cm: QDoubleSpinBox
        self._fig_height_cm: QDoubleSpinBox
        self._fig_frameon: QCheckBox
        self._fig_layout_mode: QComboBox

        self._apply_persisted_metrics_to_rcparams()
        self._build_ui()
        self.clear_connected_figure()

    def _configure_combo_width(
        self,
        combo: QComboBox,
        minimum_contents_length: int = 0,
    ) -> None:
        """Make combo boxes use the available widget width.

        Thin wrapper kept so the call sites read the same as before; the rule
        itself is shared with the axis panel in ``style``.
        """
        configure_combo_width(combo, minimum_contents_length)

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(*MARGIN_PANEL)
        lay.setSpacing(12)

        # ----- Name -----
        name_section = create_card_widget(self, "figureNameCard")
        name_section_lay = QVBoxLayout(name_section)
        apply_card_layout(name_section_lay)
        name_section_lay.addWidget(create_section_title(_("Name"), name_section))

        self._name_edit = QLineEdit(name_section)
        self._name_edit.setPlaceholderText(_("Chart name"))
        self._name_edit.setToolTip(
            _("Name of this figure. Also the title of its chart tab.")
        )
        # Enter applies, matching every other rename field in the app.
        self._name_edit.returnPressed.connect(self._apply_all)
        stdSizeAndlayout(self._name_edit)
        name_section_lay.addWidget(self._name_edit)
        lay.addWidget(name_section)

        # ----- Style -----
        style_section = create_card_widget(self, "figureStyleCard")
        style_section_lay = QVBoxLayout(style_section)
        apply_card_layout(style_section_lay)

        style_title = create_section_title(_("Style"), style_section)
        style_section_lay.addWidget(style_title)

        self._style_combo = QComboBox(style_section)
        self._configure_combo_width(self._style_combo)
        self._style_combo.currentIndexChanged.connect(self._on_style_selected)
        style_section_lay.addWidget(self._style_combo)

        style_buttons_row = QWidget(style_section)
        style_buttons_lay = QHBoxLayout(style_buttons_row)
        style_buttons_lay.setContentsMargins(0, 0, 0, 0)
        style_buttons_lay.setSpacing(8)

        self._btn_default = create_action_button(
                                parent=style_buttons_row,
                                action_id="reload",
                                action=self._set_default_style,
                                layout=style_buttons_lay,
                            )
        self._btn_edit = create_action_button(
                             parent=style_buttons_row,
                             action_id="edit",
                             action=self._edit_current_style,
                             layout=style_buttons_lay,
                         )
        self._btn_apply = create_action_button(
                              parent=style_buttons_row,
                              action_id="apply",
                              action=self._apply_all,
                              layout=style_buttons_lay,
                          )

        style_buttons_lay.addStretch(1)
        style_section_lay.addWidget(style_buttons_row)
        lay.addWidget(style_section)

        # ----- Grid -----
        grid_section = create_card_widget(self, "figureGridCard")
        grid_section_lay = QVBoxLayout(grid_section)
        apply_card_layout(grid_section_lay)

        grid_title = create_section_title(_("Grid"), grid_section)
        grid_section_lay.addWidget(grid_title)

        grid_row = QWidget(grid_section)
        grid_lay = QHBoxLayout(grid_row)
        grid_lay.setContentsMargins(0, 0, 0, 0)
        grid_lay.setSpacing(8)

        self._nrows_combo = QComboBox(grid_row)
        self._ncols_combo = QComboBox(grid_row)
        self._configure_combo_width(self._nrows_combo)
        self._configure_combo_width(self._ncols_combo)

        for i in range(1, 7):
            self._nrows_combo.addItem(str(i), i)
            self._ncols_combo.addItem(str(i), i)

        grid_lay.addWidget(QLabel(_("Rows"), grid_row))
        grid_lay.addWidget(self._nrows_combo, 1)
        grid_lay.addSpacing(8)
        grid_lay.addWidget(QLabel(_("Cols"), grid_row))
        grid_lay.addWidget(self._ncols_combo, 1)
        grid_section_lay.addWidget(grid_row)
        lay.addWidget(grid_section)

        # ----- Figure options -----
        opts_section = create_card_widget(self, "figureOptionsCard")
        opts_section_lay = QVBoxLayout(opts_section)
        apply_card_layout(opts_section_lay)

        opts_title = create_section_title(_("Figure options"), opts_section)
        opts_section_lay.addWidget(opts_title)

        form = QFormLayout()
        stdSizeAndlayout(form)

        self._fig_dpi = QSpinBox(opts_section)
        self._fig_dpi.setRange(20, 2400)
        self._fig_dpi.setSingleStep(10)
        self._fig_dpi.setMinimumWidth(self.FIGURE_SPIN_MIN_WIDTH)
        self._fig_dpi.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._fig_width_cm = QDoubleSpinBox(opts_section)
        self._fig_width_cm.setRange(0.1, 500.0)
        self._fig_width_cm.setDecimals(2)
        self._fig_width_cm.setSingleStep(0.5)
        self._fig_width_cm.setSuffix(_(" cm"))
        self._fig_width_cm.setMinimumWidth(self.FIGURE_SPIN_MIN_WIDTH)
        self._fig_width_cm.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._fig_height_cm = QDoubleSpinBox(opts_section)
        self._fig_height_cm.setRange(0.1, 500.0)
        self._fig_height_cm.setDecimals(2)
        self._fig_height_cm.setSingleStep(0.5)
        self._fig_height_cm.setSuffix(_(" cm"))
        self._fig_height_cm.setMinimumWidth(self.FIGURE_SPIN_MIN_WIDTH)
        self._fig_height_cm.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._fig_frameon = QCheckBox(_("Draw figure frame"), opts_section)

        self._fig_layout_mode = QComboBox(opts_section)
        self._configure_combo_width(self._fig_layout_mode, minimum_contents_length=14)
        for label, value in [
            ("constrained", "constrained"),
            ("compressed", "compressed"),
            ("tight", "tight"),
            ("none", "none"),
        ]:
            self._fig_layout_mode.addItem(label, value)

        form.addRow(_("DPI"), self._fig_dpi)
        form.addRow(_("Width"), self._fig_width_cm)
        form.addRow(_("Height"), self._fig_height_cm)
        form.addRow(_("Frame on"), self._fig_frameon)
        form.addRow(_("Figure layout"), self._fig_layout_mode)
        opts_section_lay.addLayout(form)
        lay.addWidget(opts_section)
        lay.addStretch(1)

    def set_connected_figure(
        self,
        repo,
        figure_id: int,
        figure,
        redraw_callback=None,
    ) -> None:
        self._repo = repo
        self._figure_id = int(figure_id)
        self._figure = figure
        self._redraw_callback = redraw_callback
        self._apply_persisted_metrics_to_rcparams()
        self._apply_rcparams_to_connected_figure()
        self._reload_from_descriptor()

    def clear_connected_figure(self) -> None:
        self._repo = None
        self._figure_id = None
        self._figure = None
        self._redraw_callback = None

        self._style_combo.blockSignals(True)
        try:
            self._style_combo.clear()
            self._style_combo.addItem(_("(Default)"), "")
        finally:
            self._style_combo.blockSignals(False)
        self._update_style_buttons()

        width_cm, height_cm = self._rcparams_figsize_cm()
        self._nrows_combo.setCurrentIndex(0)
        self._ncols_combo.setCurrentIndex(0)
        self._fig_dpi.setValue(self._rcparams_dpi())
        self._fig_width_cm.setValue(width_cm)
        self._fig_height_cm.setValue(height_cm)
        self._fig_frameon.setChecked(True)
        self._fig_layout_mode.setCurrentIndex(0)
        self._name_edit.clear()
        self._set_enabled_state(False)

    def _set_enabled_state(self, enabled: bool) -> None:
        for widget in (
            self._name_edit,
            self._style_combo,
            self._btn_default,
            self._btn_edit,
            self._nrows_combo,
            self._ncols_combo,
            self._fig_dpi,
            self._fig_width_cm,
            self._fig_height_cm,
            self._fig_frameon,
            self._fig_layout_mode,
            self._btn_apply,
        ):
            widget.setEnabled(enabled)

    def _figure_options(self) -> dict[str, Any]:
        """Return figure options safely from the connected descriptor."""
        if self._repo is None or self._figure_id is None:
            return {}

        desc = self._repo.load_figure_descriptor(self._figure_id)

        if desc is None:
            applogger.warning(
                "No figure descriptor found for figure_id=%s. "
                "Figure options unavailable.",
                self._figure_id,
            )
            return {}

        options = getattr(desc, "options", None)

        if options is None:
            return {}

        if isinstance(options, dict):
            return dict(options)

        applogger.error(
            "Figure id=%r has invalid options. Editing stopped; "
            "please check the descriptor schema.",
            getattr(desc, "id", self._figure_id),
        )
        return {}

    def _reload_from_descriptor(self) -> None:
        """Reload figure controls from the current descriptor.

        This is the single descriptor reload path for the figure panel. It
        handles missing descriptors safely and keeps the UI in a valid state.
        """
        if self._repo is None or self._figure_id is None:
            self.clear_connected_figure()
            return

        desc = self._repo.load_figure_descriptor(self._figure_id)

        if desc is None:
            applogger.warning(
                "No figure descriptor found for figure_id=%s. "
                "Figure editing disabled for this figure.",
                self._figure_id,
            )
            self._reload_style_combo(current_style="")
            self._nrows_combo.setCurrentIndex(0)
            self._ncols_combo.setCurrentIndex(0)

            width_cm, height_cm = self._rcparams_figsize_cm()
            self._fig_dpi.setValue(self._rcparams_dpi())
            self._fig_width_cm.setValue(width_cm)
            self._fig_height_cm.setValue(height_cm)
            self._fig_frameon.setChecked(True)
            self._fig_layout_mode.setCurrentIndex(0)
            self._set_enabled_state(False)
            return

        options = getattr(desc, "options", None)

        if options is None:
            fig_opts: dict[str, Any] = {}
        elif isinstance(options, dict):
            fig_opts = dict(options)
        else:
            applogger.error(
                "Figure id=%r has invalid options. Editing stopped; "
                "please check the descriptor schema.",
                getattr(desc, "id", self._figure_id),
            )
            fig_opts = {}

        self._name_edit.setText(str(getattr(desc, "name", "") or ""))

        self._reload_style_combo(
            current_style=str(fig_opts.get("mpl_style", "") or "")
        )

        nrows = int(getattr(desc, "nrows", None) or 1)
        ncols = int(getattr(desc, "ncols", None) or 1)

        self._nrows_combo.setCurrentIndex(
            max(0, self._nrows_combo.findData(nrows))
        )
        self._ncols_combo.setCurrentIndex(
            max(0, self._ncols_combo.findData(ncols))
        )

        self._load_metrics_into_spins(fig_opts)
        self._fig_frameon.setChecked(bool(fig_opts.get("frameon", True)))

        current_layout = str(
            fig_opts.get("layout_mode", fig_opts.get("layout", "constrained"))
            or "constrained"
        )

        self._fig_layout_mode.setCurrentIndex(
            max(0, self._fig_layout_mode.findData(current_layout))
        )

        self._set_enabled_state(True)

    def _list_mplstyle_files(self) -> list[Path]:
        """Return every ``.mplstyle`` file under MPLSTYLES_DIR, subfolders included.

        The style library ships styles grouped into subfolders (``color/``,
        ``journals/``, ``color/discrete-rainbow/``, ...). A plain top-level
        ``glob("*.mplstyle")`` never saw any of those, so most of the library
        was invisible in this dropdown even though the files were right there
        on disk. ``rglob`` walks the whole tree instead.
        """
        if not MPLSTYLES_DIR.exists():
            return []
        return sorted(
            MPLSTYLES_DIR.rglob("*.mplstyle"),
            key=lambda path: str(path.relative_to(MPLSTYLES_DIR)).lower(),
        )

    @staticmethod
    def _style_label_for_path(path: Path) -> str:
        """Return a "folder.style" label for a style file under MPLSTYLES_DIR.

        Example: ``mplstyles/color/discrete-rainbow/discrete-rainbow-10.mplstyle``
        becomes ``color.discrete-rainbow.discrete-rainbow-10`` - the folder it
        lives in stays visible in the dropdown instead of collapsing every
        subfolder's styles into indistinguishable bare filenames.
        """
        try:
            rel = path.relative_to(MPLSTYLES_DIR)
        except ValueError:
            return path.name
        return ".".join(rel.with_suffix("").parts)

    def _reload_style_combo(self, current_style: str) -> None:
        self._style_combo.blockSignals(True)
        try:
            self._style_combo.clear()
            self._style_combo.addItem(_("(Default)"), "")
            self._style_combo.addItem(_("Browse…"), self._BROWSE_SENTINEL)
            for path in self._list_mplstyle_files():
                self._style_combo.addItem(self._style_label_for_path(path), str(path))
            if current_style:
                matched = False
                for i in range(2, self._style_combo.count()):
                    path = Path(str(self._style_combo.itemData(i)))
                    if not path.exists():
                        continue
                    text = _sanitize_mplstyle_text(_read_text_any_encoding(path))
                    if text == current_style:
                        self._style_combo.setCurrentIndex(i)
                        matched = True
                        break
                if not matched:
                    self._style_combo.addItem(_("(Embedded style)"), current_style)
                    self._style_combo.setCurrentIndex(self._style_combo.count() - 1)
            else:
                self._style_combo.setCurrentIndex(0)
        finally:
            self._style_combo.blockSignals(False)
        self._last_valid_style_index = self._style_combo.currentIndex()
        self._update_style_buttons()

    def _browse_for_style_file(self) -> None:
        """Open a native file dialog to pick any ``.mplstyle`` file.

        Covers the case the dropdown cannot: a style kept somewhere other
        than MPLSTYLES_DIR entirely. Cancelling restores the previous
        selection rather than leaving "Browse…" selected.
        """
        start_dir = str(MPLSTYLES_DIR) if MPLSTYLES_DIR.exists() else ""
        path_str, _unused = QFileDialog.getOpenFileName(
            self,
            _("Select Matplotlib Style"),
            start_dir,
            _("Matplotlib Style (*.mplstyle);;All Files (*)"),
        )
        if not path_str:
            blocked = self._style_combo.blockSignals(True)
            try:
                self._style_combo.setCurrentIndex(self._last_valid_style_index)
            finally:
                self._style_combo.blockSignals(blocked)
            self._update_style_buttons()
            return

        style_text = _sanitize_mplstyle_text(_read_text_any_encoding(Path(path_str)))
        self.style_changed.emit(style_text)
        self._reload_style_combo(current_style=style_text)

    def _update_style_buttons(self) -> None:
        """Enable Edit only when a concrete or embedded style is selected."""
        data = self._style_combo.currentData()
        self._btn_edit.setEnabled(bool(str(data or "").strip()))

    def _current_style_text(self) -> str:
        """Return the selected style text, resolving files when needed."""
        data = self._style_combo.currentData()
        if not isinstance(data, str) or not data.strip():
            return ""

        if data.endswith(".mplstyle"):
            path = Path(data)
            if path.exists():
                return _sanitize_mplstyle_text(_read_text_any_encoding(path))
            return ""

        return _sanitize_mplstyle_text(data)

    def _on_style_selected(self) -> None:
        if self._repo is None or self._figure_id is None:
            return
        if self._style_combo.currentData() == self._BROWSE_SENTINEL:
            self._browse_for_style_file()
            return
        self._last_valid_style_index = self._style_combo.currentIndex()
        self._update_style_buttons()
        self.style_changed.emit(self._current_style_text())

    def _set_default_style(self) -> None:
        if self._repo is None or self._figure_id is None:
            return
        self._style_combo.setCurrentIndex(0)
        self._last_valid_style_index = 0
        self.style_changed.emit("")

    def _edit_current_style(self) -> None:
        """Open the style editor preloaded with the current selected style."""
        if self._repo is None or self._figure_id is None:
            return

        style_text = self._current_style_text()
        if not style_text.strip():
            return

        style_holder: dict[str, str] = {"text": ""}

        def capture_style(style_path: str) -> None:
            path = Path(style_path)
            if not path.exists():
                applogger.error(
                    f"Style file not found: {path}. Editing stopped; please check it."
                )
            style_holder["text"] = _sanitize_mplstyle_text(
                _read_text_any_encoding(path)
            )
            dialog.accept()

        dialog = MplStyleEditorDialog(
            parent=self,
            apply_callback=capture_style,
            initial_style_text=style_text,
        )
        if not dialog.exec():
            return

        edited_style_text = style_holder["text"]
        if not edited_style_text.strip():
            return

        self.style_changed.emit(edited_style_text)
        self._reload_style_combo(current_style=edited_style_text)

    def _apply_all(self) -> None:
        """Apply both grid layout and figure options from one button."""
        self._apply_grid_layout()
        self._save_figure_options()

    def _apply_grid_layout(self) -> None:
        nrows = int(self._nrows_combo.currentData() or 1)
        ncols = int(self._ncols_combo.currentData() or 1)
        self.grid_layout_requested.emit(nrows, ncols)

    def _rcparams_dpi(self) -> int:
        try:
            dpi = int(round(float(rcParams.get("figure.dpi", DEFAULT_FIGURE_DPI))))
        except Exception:
            return int(DEFAULT_FIGURE_DPI)
        return max(1, dpi)

    def _load_metrics_into_spins(self, fig_opts: dict[str, Any]) -> None:
        """Show this figure's own metrics, falling back to the rcParams ones.

        A figure saved before metrics were per-figure carries no keys, so the
        fallback is the live rcParams value - which is what that figure has
        been rendering with all along.
        """
        metrics = figure_metrics_from_options(fig_opts)
        if metrics is None:
            width_cm, height_cm = self._rcparams_figsize_cm()
            dpi = self._rcparams_dpi()
        else:
            width_cm, height_cm, dpi_value = metrics
            dpi = int(round(dpi_value))

        self._fig_dpi.setValue(max(1, dpi))
        self._fig_width_cm.setValue(width_cm)
        self._fig_height_cm.setValue(height_cm)

    def _rcparams_figsize_cm(self) -> tuple[float, float]:
        try:
            width_in, height_in = rcParams.get("figure.figsize", DEFAULT_FIGURE_SIZE_IN)
            width_cm = float(width_in) * CM_PER_INCH
            height_cm = float(height_in) * CM_PER_INCH
        except Exception:
            width_cm = DEFAULT_FIGURE_SIZE_IN[0] * CM_PER_INCH
            height_cm = DEFAULT_FIGURE_SIZE_IN[1] * CM_PER_INCH

        if width_cm <= 0.0 or height_cm <= 0.0:
            width_cm = DEFAULT_FIGURE_SIZE_IN[0] * CM_PER_INCH
            height_cm = DEFAULT_FIGURE_SIZE_IN[1] * CM_PER_INCH
        return width_cm, height_cm

    def _apply_size_to_rcparams(self) -> None:
        width_cm = float(self._fig_width_cm.value())
        height_cm = float(self._fig_height_cm.value())
        width_in = width_cm / CM_PER_INCH
        height_in = height_cm / CM_PER_INCH
        dpi = float(self._fig_dpi.value())

        rcParams["figure.figsize"] = [width_in, height_in]
        rcParams["figure.dpi"] = dpi
        # No persistence here: the values ride out in the options payload that
        # _save_figure_options emits, and are stored against this figure alone.
        self._apply_rcparams_to_connected_figure()

    def _apply_rcparams_to_connected_figure(self) -> None:
        """Push the edited size and dpi onto the figure this panel is showing.

        ``set_dpi`` is not called directly.  rcParams holds the dpi the user
        typed into the spin box, with no display scaling in it, while the Qt
        backend keeps ``figure.dpi`` in device pixels.  Writing the configured
        value straight onto the figure broke that invariant for whatever
        ChartPanel was showing, so editing any figure property re-introduced
        the sizing bug that ChartPanel had just fixed - which is what made
        FIXED zoom wrong again.
        """
        if self._figure is None:
            return
        try:
            width_in, height_in = rcParams.get("figure.figsize", DEFAULT_FIGURE_SIZE_IN)
            dpi = float(rcParams.get("figure.dpi", DEFAULT_FIGURE_DPI))
            apply_configured_dpi(self._figure, self._figure.canvas, dpi)
            self._figure.set_size_inches(float(width_in), float(height_in), forward=False)
        except Exception:
            return

    def _apply_persisted_metrics_to_rcparams(self) -> None:
        """Push the connected figure's own metrics into rcParams.

        Called before a render rather than once at startup: rcParams is global
        and every figure wants a different answer, so the value has to be set
        from the descriptor each time a figure becomes the current one.
        """
        metrics = figure_metrics_from_options(self._figure_options())
        if metrics is None:
            return

        width_cm, height_cm, dpi = metrics
        width_in = float(width_cm) / CM_PER_INCH
        height_in = float(height_cm) / CM_PER_INCH
        if width_in > 0.0 and height_in > 0.0:
            rcParams["figure.figsize"] = [width_in, height_in]
        if dpi > 0.0:
            rcParams["figure.dpi"] = float(dpi)

    def _save_figure_options(self) -> None:
        self._apply_size_to_rcparams()

        payload = {
            "name": self._name_edit.text().strip(),
            "frameon": bool(self._fig_frameon.isChecked()),
            "layout_mode": str(self._fig_layout_mode.currentData() or "constrained"),
            OPT_FIGURE_WIDTH_CM: float(self._fig_width_cm.value()),
            OPT_FIGURE_HEIGHT_CM: float(self._fig_height_cm.value()),
            OPT_FIGURE_DPI: float(self._fig_dpi.value()),
        }
        self.figure_options_requested.emit(payload)

        if callable(self._redraw_callback):
            self._redraw_callback()