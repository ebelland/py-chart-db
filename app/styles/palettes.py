"""Colour palettes for the application styles.

The two hand-written stylesheets turned out to be almost entirely *relative*
colour: alpha-composited greys over whatever the Qt palette provides, plus one
accent.  Only three opaque hex colours appear in a thousand lines of Fluent
QSS, and two of them are commented out.  That is what makes theming cheap here
- the sheets already defer to the palette for their surfaces, so a theme is a
QPalette plus a handful of substituted colour tokens, not a second sheet.

Two things therefore define a theme:

* the **QPalette**, which decides what window, base, text and highlight
  actually are, and which every unstyled widget reads;
* the **tokens**, which decide the accent and - crucially - which direction the
  overlays go.  ``rgba(0, 0, 0, 0.08)`` is a shadow on a light background and
  invisible on a dark one, so on a dark theme the same overlay has to become
  white.  That inversion is what a naive dark mode gets wrong and why the
  sheets carry ``@SHADOW_RGB@`` rather than a literal black.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtGui import QColor, QPalette


@dataclass(frozen=True, slots=True)
class Palette:
    """One theme: what the QPalette says and what the QSS tokens resolve to."""

    key: str
    label: str
    dark: bool

    window: str
    base: str
    alternate_base: str
    text: str
    bright_text: str
    button: str
    button_text: str
    highlight: str
    highlight_text: str
    disabled_text: str
    tooltip_base: str
    tooltip_text: str

    #: "r, g, b" triples, substituted into the sheets' rgba() calls so one
    #: token serves every alpha the sheet uses.
    accent_rgb: str
    shadow_rgb: str
    glow_rgb: str
    neutral_rgb: str = "127, 127, 127"

    #: Extra token values, for anything a single theme needs and others do not.
    extra_tokens: dict[str, str] = field(default_factory=dict)

    def tokens(self) -> dict[str, str]:
        """Return every ``@TOKEN@`` this palette resolves, without the braces."""
        return {
            "ACCENT_RGB": self.accent_rgb,
            "SHADOW_RGB": self.shadow_rgb,
            "GLOW_RGB": self.glow_rgb,
            "NEUTRAL_RGB": self.neutral_rgb,
            "WINDOW": self.window,
            "BASE": self.base,
            "TEXT": self.text,
            "HIGHLIGHT": self.highlight,
            **self.extra_tokens,
        }

    def qpalette(self) -> QPalette:
        """Build the QPalette this theme installs on the application.

        Disabled text is set explicitly on all three groups.  Qt derives a
        disabled colour from the enabled one when it is not given, and its
        derivation assumes a light background - on a dark theme that produces
        disabled text brighter than the enabled text beside it.
        """
        palette = QPalette()
        pairs = (
            (QPalette.ColorRole.Window, self.window),
            (QPalette.ColorRole.WindowText, self.text),
            (QPalette.ColorRole.Base, self.base),
            (QPalette.ColorRole.AlternateBase, self.alternate_base),
            (QPalette.ColorRole.Text, self.text),
            (QPalette.ColorRole.BrightText, self.bright_text),
            (QPalette.ColorRole.Button, self.button),
            (QPalette.ColorRole.ButtonText, self.button_text),
            (QPalette.ColorRole.Highlight, self.highlight),
            (QPalette.ColorRole.HighlightedText, self.highlight_text),
            (QPalette.ColorRole.ToolTipBase, self.tooltip_base),
            (QPalette.ColorRole.ToolTipText, self.tooltip_text),
            (QPalette.ColorRole.PlaceholderText, self.disabled_text),
            (QPalette.ColorRole.Link, self.highlight),
        )
        for role, value in pairs:
            palette.setColor(role, QColor(value))

        disabled = QColor(self.disabled_text)
        for role in (
            QPalette.ColorRole.Text,
            QPalette.ColorRole.WindowText,
            QPalette.ColorRole.ButtonText,
        ):
            palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)

        return palette


#: The light palette, matching what the sheets were written against: Windows'
#: Fluent accent over near-white surfaces.  Named "light" rather than "fluent"
#: because the same palette serves the macOS sheet, which differs in metrics
#: and radii rather than in colour.
LIGHT = Palette(
    key="light",
    label="Light",
    dark=False,
    window="#f3f3f3",
    base="#ffffff",
    alternate_base="#f7f7f7",
    text="#1a1a1a",
    bright_text="#ffffff",
    button="#fdfdfd",
    button_text="#1a1a1a",
    highlight="#0078d4",
    highlight_text="#ffffff",
    disabled_text="#9a9a9a",
    tooltip_base="#ffffff",
    tooltip_text="#1a1a1a",
    accent_rgb="0, 120, 212",
    shadow_rgb="0, 0, 0",
    glow_rgb="255, 255, 255",
)

#: The dark palette.  Surfaces are near-black but never black: pure #000000
#: against a bright chart panel is harsh, and it leaves no room for a *darker*
#: surface to sit behind a card.  The accent is lifted from Fluent's #0078d4 to
#: #4cc2ff because the original does not carry enough contrast against a dark
#: window to read as a selection.
DARK = Palette(
    key="dark",
    label="Dark",
    dark=True,
    window="#202020",
    base="#1b1b1b",
    alternate_base="#262626",
    text="#eaeaea",
    bright_text="#ffffff",
    button="#2b2b2b",
    button_text="#eaeaea",
    highlight="#0a84ff",
    highlight_text="#ffffff",
    disabled_text="#6f6f6f",
    tooltip_base="#2b2b2b",
    tooltip_text="#eaeaea",
    accent_rgb="76, 194, 255",
    # Inverted: an overlay that darkens a light surface has to lighten a dark
    # one, or every border and shadow in the sheet disappears.
    shadow_rgb="255, 255, 255",
    glow_rgb="0, 0, 0",
)

PALETTES: dict[str, Palette] = {palette.key: palette for palette in (LIGHT, DARK)}


def themed_qss(qss: str, key: str | None) -> tuple[str, Palette]:
    """Resolve a palette by key and apply its tokens to *qss*.

    The two halves are one call because they are never wanted apart: a sheet
    substituted with one palette's tokens while the application wears another's
    QPalette is a window belonging to neither theme.  Returning the palette
    too lets the caller install it and read ``dark`` without looking it up
    again.

    An unknown key resolves to light rather than raising - a typo in
    config.json should leave the app looking ordinary, not unstyled.

    Only the palette's own tokens are replaced.  The icon URLs are substituted
    by the caller afterwards; eating every ``@...@`` here would leave the sheet
    with empty ``url()`` rules and no combo arrows.
    """
    palette = PALETTES.get(str(key or "").strip().lower(), LIGHT)
    for name, value in palette.tokens().items():
        qss = qss.replace(f"@{name}@", value)
    return qss, palette
