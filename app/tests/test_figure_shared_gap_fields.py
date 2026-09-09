"""The two Shared gap fields in Figure properties.

Axes that share a scale are drawn flush - that was the fix for "charts are
not contiguous", and it is the right default. It is not always what is
wanted, though: a shared-x column of three signals often reads better with
a hairline between the panels. These two fields are how to ask for one,
and the point of them is that they are *not* Manual spacing: they open the
gap only where axes actually share the scale that gap runs across, and
they work under the automatic layout engines too.

The renderer half is covered in test_shared_axes_contiguous.py; this is
the panel round-trip - what the spins show, and what Apply writes.
"""
from __future__ import annotations

import pytest
from matplotlib.figure import Figure

from app.charts.render_figure import (
    OPT_SHARED_HORIZONTAL_SPACE,
    OPT_SHARED_VERTICAL_SPACE,
)
from app.data.sqlite_repo import SqliteRepo
from app.widgets.figure_properties import FigurePropertiesWidget


def _connected(repo: SqliteRepo, options: dict | None = None) -> tuple[
    FigurePropertiesWidget, int
]:
    figure_id = int(
        repo.create_figure_descriptor(name="F", nrows=2, ncols=1, options=options or {})
    )
    widget = FigurePropertiesWidget()
    widget.set_connected_figure(repo, figure_id, Figure())
    return widget, figure_id


def test_a_figure_that_never_set_them_shows_flush(qapp, repo: SqliteRepo) -> None:
    widget, _figure_id = _connected(repo)

    assert widget._shared_vspace.value() == pytest.approx(0.0)
    assert widget._shared_hspace.value() == pytest.approx(0.0)


def test_saved_gaps_come_back_into_the_spins(qapp, repo: SqliteRepo) -> None:
    widget, _figure_id = _connected(
        repo,
        {OPT_SHARED_VERTICAL_SPACE: 0.3, OPT_SHARED_HORIZONTAL_SPACE: 0.15},
    )

    assert widget._shared_vspace.value() == pytest.approx(0.3)
    assert widget._shared_hspace.value() == pytest.approx(0.15)


def test_applying_sends_both_gaps_with_the_other_options(
    qapp, repo: SqliteRepo
) -> None:
    """The panel emits; the window writes. What matters here is that the two
    keys are in the payload under the names the renderer reads."""
    widget, _figure_id = _connected(repo)
    sent: list[dict] = []
    widget.figure_options_requested.connect(sent.append)

    widget._shared_vspace.setValue(0.2)
    widget._shared_hspace.setValue(0.05)
    widget._save_figure_options()

    assert len(sent) == 1
    assert sent[0][OPT_SHARED_VERTICAL_SPACE] == pytest.approx(0.2)
    assert sent[0][OPT_SHARED_HORIZONTAL_SPACE] == pytest.approx(0.05)


def test_the_gaps_are_independent_of_manual_spacing(qapp, repo: SqliteRepo) -> None:
    """They travel beside the grid, not in ``margins``: Manual spacing is one
    layout mode's margins for the whole figure, these apply whatever the
    layout engine."""
    widget, _figure_id = _connected(repo)
    sent: list[dict] = []
    widget.figure_options_requested.connect(sent.append)

    widget._shared_vspace.setValue(0.2)
    widget._save_figure_options()

    assert OPT_SHARED_VERTICAL_SPACE not in sent[0]["margins"]


def test_the_spins_are_disabled_with_no_figure_connected(
    qapp, repo: SqliteRepo
) -> None:
    widget, _figure_id = _connected(repo)
    widget.clear_connected_figure()

    assert not widget._shared_vspace.isEnabled()
    assert not widget._shared_hspace.isEnabled()
