"""Shared base for the Figure/Axis/Series properties editors.

The three editors show and edit different things - a figure's size and
layout, one axis's scale and drawing kwargs, one series' style and query -
and each needs its own arrangement of controls, so this is deliberately not
a base *layout*: every subclass still builds its own UI, in its own
``_build_ui()``, exactly as before. What they did share, byte for byte in
three places, is the small bit of state around "which figure this editor is
currently pointed at" and the connect/disconnect lifecycle around it.

A subclass must NOT call ``self.setLayout(...)`` before its own
``_build_ui()`` runs (and must not let Qt's default "first QVBoxLayout(self)
wins" behaviour surprise it either) - ``BaseProperties.__init__`` sets no
layout of its own for exactly that reason: a widget can only ever have one,
and the previous version of this class claimed it before any subclass got
the chance to build its real one, which is why every properties panel
briefly rendered at its empty-widget size (22x22) instead of its content's.
"""
from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy, QWidget


class BaseProperties(QWidget):
    """Common connected-figure state for a properties editor.

    Holds the four attributes every editor already read and wrote
    identically (``_repo``, ``_figure_id``, ``_figure``,
    ``_redraw_callback``) and the pair of lifecycle methods that used to be
    three near-identical copies. A subclass still owns its own UI and its
    own ``_reload_from_descriptor()``/``_set_enabled_state()`` - this only
    owns calling them at the right time, not what they do.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # A stable name so macos_native.qss / fluent_win11.qss can each say
        # what the ground behind a properties panel's cards should be,
        # without matching on plain QWidget. They answer differently on
        # purpose: white on Windows, where the WinUI3 Settings app puts
        # outlined cards on one continuous white page, and the window
        # ground on macOS, where System Settings floats white grouped
        # boxes on grey. See the #basePropertiesPanel rule in each.
        self.setObjectName("basePropertiesPanel")
        # Without this a stylesheet background on a plain QWidget subclass
        # is parsed and then never painted - the long-standing Qt gotcha
        # that makes QSS look like it "does not work" on custom widgets.
        # QFrame-based widgets (create_card_widget's cards) do not need it,
        # which is why theirs worked and this one would not have.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._repo: Any | None = None
        self._figure_id: int | None = None
        self._figure: Any | None = None
        self._redraw_callback: Callable[[], None] | None = None

    def set_connected_figure(
        self,
        repo: Any,
        figure_id: int,
        figure: Any,
        redraw_callback: Callable[[], None] | None = None,
    ) -> None:
        """Attach a figure and reload this editor's fields from it.

        Does not itself call ``_set_enabled_state`` - every subclass's own
        ``_reload_from_descriptor`` already ends by enabling or disabling
        its controls itself (disabled when the descriptor turns out to be
        missing or empty, enabled otherwise), and calling it again
        unconditionally here would override that: an axis panel with no
        axes left, in particular, needs to end up disabled even though a
        figure *is* connected.
        """
        self._repo = repo
        self._figure_id = int(figure_id)
        self._figure = figure
        self._redraw_callback = redraw_callback
        self._before_reload()
        self._reload_from_descriptor()

    def _before_reload(self) -> None:
        """Run right after connecting, right before the first reload.

        A no-op here. FigurePropertiesWidget is the one subclass that needs
        something between "the four attributes are set" and "reload the
        fields from them" - pushing this figure's own DPI/width/height into
        rcParams and onto the connected Matplotlib figure, both of which
        read from ``self._figure_id`` and so cannot run before it is set -
        so it overrides this rather than the whole of
        ``set_connected_figure``.
        """

    def clear_connected_figure(self) -> None:
        """Detach the current figure and disable editing.

        A subclass with more of its own state to reset (a selection map, a
        current axis/series id, a combo to clear) overrides this, calls
        ``super().clear_connected_figure()`` for the shared part, and does
        the rest itself - exactly what each already did before this existed,
        just without repeating these four lines to get there.
        """
        self._repo = None
        self._figure_id = None
        self._figure = None
        self._redraw_callback = None
        self._set_enabled_state(False)

    def _reload_from_descriptor(self) -> None:
        """Reload this editor's fields from the connected descriptor.

        Every subclass overrides this. Nothing to share about *what* gets
        reloaded - only that ``set_connected_figure`` always reloads right
        after connecting, which is the part worth not repeating.
        """
        raise NotImplementedError

    def _set_enabled_state(self, enabled: bool) -> None:
        """Enable or disable this editor's interactive controls.

        Every subclass overrides this with its own control list. Nothing to
        share about *which* controls - only that connecting and
        disconnecting always toggle them the same way.
        """
        raise NotImplementedError
