"""Collapsible panel used by the responsive properties accordion."""
from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

MAX_HEIGHT = 16777215


class CollapsiblePanel(QFrame):
    """Reusable property-style panel with a clickable header and body."""

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        content: QWidget | None = None,
        parent: QWidget | None = None,
        *,
        expanded: bool = True,
        compact: bool = False,
        animation_duration: int = 2,
    ) -> None:
        super().__init__(parent)

        self._content = content or QWidget(self)
        self._expanded = expanded
        self._compact = compact
        self._animation_duration = max(0, int(animation_duration))
        self._animation: QPropertyAnimation | None = None

        self._header: QWidget | None = None
        self._body: QWidget | None = None
        self._toggle_button: QToolButton | None = None

        self._build_ui(title)
        self.set_expanded(self._expanded, animated=False)

    def sizeHint(self) -> QSize:
        """Return a compact size while collapsed."""
        base = super().sizeHint()
        assert self._header is not None
        assert self._body is not None

        header_height = self._header.sizeHint().height()
        if not self._expanded:
            return QSize(base.width(), header_height)

        spacing = self._panel_layout().spacing()
        body_height = self._body.sizeHint().height()
        return QSize(base.width(), header_height + spacing + body_height)

    def minimumSizeHint(self) -> QSize:
        """Return only the header as the vertical minimum.

        The accordion must always be able to reserve space for every panel
        header. Expanded body content is expected to shrink or scroll inside
        the active panel rather than pushing later panel headers out of view.
        """
        assert self._header is not None
        return QSize(0, self._header.sizeHint().height())

    def content_widget(self) -> QWidget:
        """Return the current body widget."""
        return self._content

    def set_content_widget(self, widget: QWidget) -> None:
        """Replace the body content widget.

        The outgoing widget is hidden and scheduled for deletion rather than
        unparented: ``setParent(None)`` promotes a widget to a *top-level
        window*, which Qt is then free to show on its own - see the matching
        note in AxisPropertiesWidget._replace_layout_widget, where exactly
        that put a stray floating panel on screen.  The incoming widget is
        never deleted, only ones being replaced.
        """
        old_widget = self._content
        self._content = widget

        body_layout = self._body_layout()
        while body_layout.count():
            item = body_layout.takeAt(0)
            child = item.widget() if item is not None else None
            if child is not None and child is not widget:
                child.hide()
                child.deleteLater()

        if old_widget is not widget and old_widget is not None:
            old_widget.hide()
            old_widget.deleteLater()

        body_layout.addWidget(widget)
        self._apply_geometry(animated=False)

    def is_expanded(self) -> bool:
        """Return whether the panel body is visible."""
        return self._expanded

    def set_expanded(
        self,
        expanded: bool,
        *,
        animated: bool = True,
    ) -> None:
        """Expand or collapse the body."""
        expanded = bool(expanded)

        if self._expanded == expanded and self._body is not None:
            self._sync_header_state()
            self._apply_geometry(animated=False)
            return

        self._expanded = expanded
        self._sync_header_state()
        self._apply_geometry(animated=animated)
        self.toggled.emit(self._expanded)

    def set_title(self, title: str) -> None:
        """Set the header text and tooltip."""
        assert self._toggle_button is not None
        text = str(title)
        self._toggle_button.setText(text)
        self._toggle_button.setToolTip(text)

    def set_icon_position(self, position: Qt.ToolButtonStyle) -> None:
        """Set the header icon and text placement."""
        assert self._toggle_button is not None
        self._toggle_button.setToolButtonStyle(position)

    def set_animation_duration(self, duration_ms: int) -> None:
        """Set expand/collapse animation duration in milliseconds."""
        self._animation_duration = max(0, int(duration_ms))

    def _panel_layout(self) -> QVBoxLayout:
        layout = self.layout()
        assert isinstance(layout, QVBoxLayout)
        return layout

    def _body_layout(self) -> QVBoxLayout:
        assert self._body is not None
        layout = self._body.layout()
        assert isinstance(layout, QVBoxLayout)
        return layout

    def _build_ui(self, title: str) -> None:
        self.setObjectName("collapsiblePanel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumHeight(0)
        self.setMaximumHeight(MAX_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._toggle_button = QToolButton(self)
        self._toggle_button.setObjectName("panelToggleButton")
        self._toggle_button.setCheckable(True)
        self._toggle_button.setAutoRaise(True)
        self._toggle_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._toggle_button.setText(title)
        self._toggle_button.setToolTip(title)
        self._toggle_button.clicked.connect(self.set_expanded)
        self._toggle_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self._header = QWidget(self)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(*self._header_margins())
        header_layout.setSpacing(0)
        header_layout.addWidget(self._toggle_button)

        self._body = QWidget(self)
        self._body.setObjectName("panelBody")
        self._body.setMinimumHeight(0)
        self._body.setMaximumHeight(MAX_HEIGHT)
        self._body.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3 if self._compact else 6)
        layout.addWidget(self._header, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._body, 1)

    def _header_margins(self) -> tuple[int, int, int, int]:
        if self._compact:
            return 6, 4, 6, 4
        return 8, 6, 8, 6

    def _sync_header_state(self) -> None:
        assert self._toggle_button is not None
        self._toggle_button.setChecked(self._expanded)
        self._toggle_button.setArrowType(
            Qt.ArrowType.DownArrow
            if self._expanded
            else Qt.ArrowType.RightArrow
        )

    def _collapsed_height(self) -> int:
        assert self._header is not None
        return self._header.sizeHint().height()

    def _expanded_height_hint(self) -> int:
        assert self._header is not None
        assert self._body is not None

        spacing = self._panel_layout().spacing()
        header_height = self._header.sizeHint().height()
        body_height = max(
            self._body.sizeHint().height(),
            self._body.minimumSizeHint().height(),
        )
        return header_height + spacing + body_height

    def _apply_geometry(self, *, animated: bool) -> None:
        assert self._body is not None

        self._stop_animation()

        collapsed_height = self._collapsed_height()
        start_height = max(self.height(), collapsed_height)
        self._body.setVisible(True)

        if self._expanded:
            self.setMinimumHeight(0)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            target_height = self._expanded_height_hint()
        else:
            self.setMinimumHeight(collapsed_height)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            target_height = collapsed_height

        should_animate = (
            animated
            and self._animation_duration > 0
            and self.isVisible()
        )
        if should_animate:
            self._animate_height(
                start_height=start_height,
                target_height=target_height,
            )
            return

        self._finish_geometry_update()

    def _stop_animation(self) -> None:
        """Stop and dispose the current animation if it is still valid."""
        animation = self._animation
        self._animation = None
        if animation is None:
            return

        try:
            animation.stop()
            animation.deleteLater()
        except RuntimeError:
            return

    def _animate_height(
        self,
        *,
        start_height: int,
        target_height: int,
    ) -> None:
        animation = QPropertyAnimation(self, b"maximumHeight", self)
        animation.setDuration(self._animation_duration)
        animation.setStartValue(start_height)
        animation.setEndValue(target_height)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(self._on_animation_finished)

        self._animation = animation
        animation.start()

    def _on_animation_finished(self) -> None:
        """Restore final geometry after the height animation completes."""
        animation = self._animation
        self._animation = None
        if animation is not None:
            try:
                animation.deleteLater()
            except RuntimeError:
                pass

        self._finish_geometry_update()

    def _finish_geometry_update(self) -> None:
        """Apply final collapsed or expanded constraints."""
        assert self._body is not None

        if self._expanded:
            self.setMinimumHeight(0)
            self.setMaximumHeight(MAX_HEIGHT)
            self._body.setVisible(True)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
        else:
            collapsed_height = self._collapsed_height()
            self.setMinimumHeight(collapsed_height)
            self.setMaximumHeight(collapsed_height)
            self._body.setVisible(False)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )

        self.updateGeometry()
