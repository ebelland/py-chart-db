"""The Style combo's "Browse…" entry: pick any .mplstyle file, not only one
already sitting under MPLSTYLES_DIR.

Written to lock the feature in - it already exists in figure_properties.py
(_BROWSE_SENTINEL / _browse_for_style_file) but had no test of its own.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.data.sqlite_repo import SqliteRepo

# `repo` (a fresh SqliteRepo at this test's tmp_db_path) comes from conftest.py.


@pytest.fixture
def widget(qapp, repo: SqliteRepo):
    from matplotlib.figure import Figure

    from app.widgets.figure_properties import FigurePropertiesWidget

    figure_id = int(repo.create_figure_descriptor(name="F", nrows=1, ncols=1))
    built = FigurePropertiesWidget()
    built.set_connected_figure(repo, figure_id, Figure())
    return built


def test_browse_is_offered_right_after_default(widget) -> None:
    assert widget._style_combo.itemText(0) == "(Default)"
    assert widget._style_combo.itemText(1) == "Browse…"
    assert widget._style_combo.itemData(1) == widget._BROWSE_SENTINEL


def test_picking_a_file_applies_its_sanitized_text_and_reselects_it(
    widget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    style_file = tmp_path / "outside_mplstyles_dir.mplstyle"
    style_file.write_text("lines.linewidth: 3.0\nbackend: Agg\n", encoding="utf-8")

    monkeypatch.setattr(
        "app.widgets.figure_properties.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(style_file), ""),
    )

    emitted: list[str] = []
    widget.style_changed.connect(emitted.append)

    index = widget._style_combo.findData(widget._BROWSE_SENTINEL)
    widget._style_combo.setCurrentIndex(index)

    # backend is a runtime-only key _sanitize_mplstyle_text strips.
    assert emitted, "expected style_changed to fire"
    assert "linewidth" in emitted[-1]
    assert "backend" not in emitted[-1]

    # The picked file lives outside MPLSTYLES_DIR, so it cannot be one of
    # the catalogue entries the combo lists by scanning that folder - its
    # sanitized text is carried as an "embedded" style instead, the same way
    # a hand-edited one already is. Either way, "Browse…" itself is not left
    # selected.
    assert widget._style_combo.currentData() != widget._BROWSE_SENTINEL
    assert widget._style_combo.currentText() == "(Embedded style)"
    assert widget._style_combo.currentData() == emitted[-1]


def test_cancelling_the_file_dialog_reverts_the_selection(
    widget, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.widgets.figure_properties.QFileDialog.getOpenFileName",
        lambda *a, **k: ("", ""),
    )

    before = widget._style_combo.currentIndex()
    index = widget._style_combo.findData(widget._BROWSE_SENTINEL)
    widget._style_combo.setCurrentIndex(index)

    assert widget._style_combo.currentIndex() == before
    assert widget._style_combo.currentData() != widget._BROWSE_SENTINEL
