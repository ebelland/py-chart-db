"""What "Default" looks like in the Series properties combos.

The three combos on that row - Line style, Marker, Color - each have a
first entry meaning "let the style sheet decide". Two of them were drawing
that entry as something it is not:

* Line style drew it *dashed*, so the row read as a washed-out second
  "Dashed" rather than as a default. Faintness is the signal the marker
  combo already uses; it only works if it is the only difference.
* Color had no such entry at all - it was "(none)", drawn as a blank
  square, which reads as "this series has no colour" when what actually
  happens is that the style sheet's ``axes.prop_cycle`` picks one.
"""
from __future__ import annotations

from matplotlib import rcParams

from app.charts.kwarg_spec import DEFAULT
from app.widgets.color_combo import (
    DEFAULT_COLOR_LABEL,
    MatplotlibColorCombo,
    _cycle_colors,
    _entry_icon,
)
from app.widgets.line_combo import LineStyleCombo, _line_icon


def _midline_ink(linestyle: str) -> list[bool]:
    """True/False per pixel along the middle of the preview icon."""
    image = _line_icon(linestyle).pixmap(28, 14).toImage()
    row = image.height() // 2
    return [image.pixelColor(x, row).alpha() > 0 for x in range(image.width())]


# ----------------------------------------------------------------------
# Line style
# ----------------------------------------------------------------------
def test_the_default_line_preview_is_solid_not_dashed(qapp) -> None:
    """A gap in the middle of the line is what makes "Dashed" recognisable,
    and Default must not have one."""
    default_ink = _midline_ink(DEFAULT)
    dashed_ink = _midline_ink("--")

    assert any(default_ink), "the Default entry draws no line at all"
    assert all(default_ink[2:-2]), "the Default entry is drawn with gaps"
    assert not all(dashed_ink[2:-2]), "the Dashed entry lost its gaps"


def test_the_default_line_preview_is_fainter_than_solid(qapp) -> None:
    """Faintness is the whole signal now that the shape is the same."""
    def darkest(linestyle: str) -> int:
        image = _line_icon(linestyle).pixmap(28, 14).toImage()
        row = image.height() // 2
        return max(
            image.pixelColor(x, row).alpha() for x in range(image.width())
        )

    assert darkest(DEFAULT) < darkest("-")


def test_default_is_the_first_line_style_offered(qapp) -> None:
    combo = LineStyleCombo()

    assert combo.itemData(0) == DEFAULT


# ----------------------------------------------------------------------
# Colour
# ----------------------------------------------------------------------
def test_the_colour_combo_offers_default_first(qapp) -> None:
    combo = MatplotlibColorCombo(none_label=DEFAULT_COLOR_LABEL)

    assert combo.itemText(0) == "Default"
    # The stored value is unchanged: the renderer already reads "" as
    # "take the next colour from the cycle".
    assert combo.itemData(0, MatplotlibColorCombo.ROLE_VALUE) == ""
    assert combo.current_hex() == ""


def test_the_none_label_is_still_available_for_the_kwargs_editor(qapp) -> None:
    """There, an empty value clears a key rather than deferring to a cycle."""
    assert MatplotlibColorCombo().itemText(0) == "(none)"


def test_the_default_swatch_shows_the_style_cycle(qapp) -> None:
    """Blank read as "no colour"; the cycle is what actually happens."""
    icon = _entry_icon("")
    image = icon.pixmap(14, 14).toImage()
    row = image.height() // 2
    painted = {image.pixelColor(x, row).name() for x in range(2, image.width() - 2)}

    assert len(painted) > 1, f"the swatch is one flat colour: {painted}"
    assert "#000000" not in painted, "the cycle colours were not parsed"


def test_the_cycle_is_read_as_hex_whatever_the_style_stores(qapp) -> None:
    """Matplotlib's own default style holds RGB tuples, and QColor renders
    the string form of a tuple as black."""
    from cycler import cycler

    saved = rcParams["axes.prop_cycle"]
    try:
        rcParams["axes.prop_cycle"] = cycler(color=[(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)])
        assert _cycle_colors() == ("#1a334c", "#668099")
    finally:
        rcParams["axes.prop_cycle"] = saved


def test_a_style_with_no_cycle_still_gives_a_swatch(qapp, monkeypatch) -> None:
    """A hand-written .mplstyle can leave the cycle empty; an icon builder
    is not the place to raise about it."""
    import app.widgets.color_combo as module

    monkeypatch.setattr(module, "_CYCLE_SWATCH_COLORS", 0)

    assert module._cycle_colors() == ("#1f77b4",)
