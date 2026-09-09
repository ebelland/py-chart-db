"""BaseProperties must never claim the layout slot its subclasses need.

Regression guard for a real bug: BaseProperties.__init__ used to call
self.setLayout(QVBoxLayout()) before any subclass got a chance to build its
own. QWidget accepts exactly one layout, so every subclass's own
QVBoxLayout(self) call in _build_ui() silently failed (Qt logs a warning
and keeps the first layout, not the second), leaving the widget's real
controls built but never attached to anything the widget would actually
lay out - every properties panel rendered at an empty widget's size (22x22)
instead of its content's.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.widgets.axis_properties import AxisPropertiesWidget
from app.widgets.base_properties import BaseProperties
from app.widgets.figure_properties import FigurePropertiesWidget
from app.widgets.series_properties import SeriesPropertiesWidget

WIDGET_CLASSES = (FigurePropertiesWidget, AxisPropertiesWidget, SeriesPropertiesWidget)


def test_base_properties_sets_no_layout_of_its_own(qapp) -> None:
    """The base class must leave the layout slot free for a subclass."""
    bare = BaseProperties()
    assert bare.layout() is None


@pytest.mark.parametrize("widget_class", WIDGET_CLASSES)
def test_each_editor_s_own_layout_actually_attaches(qapp, widget_class) -> None:
    widget = widget_class()

    layout = widget.layout()
    assert layout is not None
    assert layout.count() > 0, (
        f"{widget_class.__name__}'s own layout has no content - its "
        "_build_ui() lost the fight for the layout slot"
    )


def test_before_reload_runs_between_attaching_and_reloading(qapp) -> None:
    """The one extension point BaseProperties offers a subclass that needs
    something between "attributes set" and "fields reloaded" -
    FigurePropertiesWidget's rcParams push is the real user of this."""
    seen: list[tuple[Any, Any]] = []

    class _Probe(BaseProperties):
        def _before_reload(self) -> None:
            seen.append((self._repo, self._figure_id))

        def _reload_from_descriptor(self) -> None:
            # _before_reload must already have run by the time this does.
            assert seen == [("a-repo", 7)]

        def _set_enabled_state(self, enabled: bool) -> None:
            pass

    probe = _Probe()
    probe.set_connected_figure("a-repo", 7, figure=None)

    assert seen == [("a-repo", 7)]


@pytest.mark.parametrize("widget_class", WIDGET_CLASSES)
def test_each_editor_reports_a_real_content_sized_hint(qapp, widget_class) -> None:
    """22x22 - QWidget's own bare default - is exactly the empty-layout
    symptom; every one of these has far more controls than that."""
    widget = widget_class()

    hint = widget.sizeHint()
    assert hint.width() > 100
    assert hint.height() > 100


# ----------------------------------------------------------------------
# The ground the cards sit on
# ----------------------------------------------------------------------
def test_a_stylesheet_background_can_actually_paint(qapp) -> None:
    """WA_StyledBackground, the Qt gotcha: without it a stylesheet
    background on a plain QWidget subclass is parsed and never painted,
    which is what makes QSS look like it "does not work" on custom
    widgets. QFrame-based cards do not need it; this does."""
    from PySide6.QtCore import Qt

    assert BaseProperties().testAttribute(Qt.WidgetAttribute.WA_StyledBackground)


@pytest.mark.parametrize("qss_name", ["fluent_win11.qss", "macos_native.qss"])
def test_both_stylesheets_say_what_the_panel_ground_is(qss_name: str) -> None:
    """base_properties.py's own comment promises a #basePropertiesPanel
    rule in each sheet - it went one round without one, so it was
    describing something that did not exist."""
    from app.styles.style import _load_qss

    qss, _path = _load_qss(qss_name)
    assert qss is not None
    assert "#basePropertiesPanel" in qss


def test_the_two_platforms_answer_differently_on_purpose() -> None:
    """White on Windows (WinUI3 Settings: outlined cards on one white
    page), the window ground on macOS (System Settings: white grouped
    boxes floating on grey). Matching them up would be the bug."""
    from app.styles.style import _load_qss

    def panel_rule(name: str) -> str:
        qss, _path = _load_qss(name)
        block = qss.split("#basePropertiesPanel", 1)[1]
        return block.split("}", 1)[0]

    assert "palette(base)" in panel_rule("fluent_win11.qss")
    assert "palette(window)" in panel_rule("macos_native.qss")
