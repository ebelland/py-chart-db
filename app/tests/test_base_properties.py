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
