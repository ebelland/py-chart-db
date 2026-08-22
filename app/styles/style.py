"""Shared UI vocabulary: icons, stylesheets, widget factories, layout scale.

Everything visual that more than one dialog needs lives here so it is defined
once:

* platform stylesheet loading (Fluent on Windows, native-leaning on macOS);
* icon lookup by convention from ``app/icons/{win11,macOs,common}``;
* factories for buttons, menus, cards, and section titles that set the dynamic
  properties the stylesheets key on;
* one layout scale (``SPACING_*``, ``MARGIN_*``) and one ``DIALOG_SIZES`` table.

If a dialog hand-rolls a margin or a button it will drift from the rest of the
app; route it through here instead.
"""
# app/styles/style.py
from __future__ import annotations

import html
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PySide6 import QtWidgets
from PySide6.QtCore import QByteArray, QRect, QRectF, QSize, Qt
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QFontDatabase,
    QGuiApplication,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QTextOption,
)
from PySide6.QtWidgets import QApplication, QBoxLayout, QCheckBox, QComboBox, QFormLayout, QFrame, QLineEdit, QMenu, QPlainTextEdit, QScrollArea, QSizePolicy, QToolButton, QWidget, QPushButton
from app.logs.logger import applogger
from app.styles.palettes import themed_qss
from app.utils.config import get_section, get_value
from app.utils.i18n import tr

_UI_DIR = Path(__file__).resolve().parent
_APP_DIR = _UI_DIR.parent
_ICONS_DIR = _APP_DIR / "icons"
_STYLES_DIR = _APP_DIR / "styles"
_IS_MACOS = platform.system().lower() == "darwin"
_IS_WINDOWS = platform.system().lower().startswith("win")

MenuCallback = Callable[..., Any]
ShortcutLike = QKeySequence.StandardKey | QKeySequence | str | int | None


# ----------------------------------------------------------------------
# Layout scale
# ----------------------------------------------------------------------
# One scale for the whole app.  Before this existed there were seven different
# margin conventions across 58 hand-written call sites, so two sibling dialogs
# could not be made to line up without editing both.
SPACING_NONE: int = 0
SPACING_TIGHT: int = 4
SPACING_DEFAULT: int = 8
SPACING_LOOSE: int = 12

MARGIN_NESTED: tuple[int, int, int, int] = (0, 0, 0, 0)
MARGIN_CARD: tuple[int, int, int, int] = (10, 10, 10, 10)
MARGIN_DIALOG: tuple[int, int, int, int] = (12, 12, 12, 12)
MARGIN_PANEL: tuple[int, int, int, int] = (6, 6, 6, 6)

# Canonical dialog sizes, keyed by the role a dialog plays.  Anything that needs
# a size picks one of these instead of inventing another number.
DIALOG_SIZE_SMALL: QSize = QSize(560, 360)
DIALOG_SIZE_MEDIUM: QSize = QSize(900, 640)
DIALOG_SIZE_LARGE: QSize = QSize(1020, 700)

DIALOG_SIZES: dict[str, QSize] = {
    "small": DIALOG_SIZE_SMALL,
    "medium": DIALOG_SIZE_MEDIUM,
    "large": DIALOG_SIZE_LARGE,
}

# How many characters a combo box is guaranteed to show.  This is the single
# biggest driver of how narrow a side panel can be made: a combo asking for 24
# characters puts a ~200 px floor under every panel that contains one, which
# is what used to pin the main window's left panel at about 300 px however the
# splitter was dragged.  Six characters plus elision keeps the panel free to
# shrink; the full value stays readable through the combo's tooltip.
COMBO_MIN_CONTENTS_LENGTH: int = 6

# Room for the drop-down arrow and the frame, on top of the text itself.
_COMBO_CHROME_WIDTH: int = 34

# Splitter handle: wide enough to grab without aiming, narrow enough not to
# read as a divider.
SPLITTER_HANDLE_WIDTH: int = 6

# Floor for the resizable side panels.  Not zero: a panel dragged to nothing
# looks like a bug and cannot be grabbed again.
PANEL_MIN_WIDTH: int = 140


@dataclass(frozen=True, slots=True)
class PlatformStyle:
    """Resolved platform styling information."""

    system: str
    qss_file: Path | None


@dataclass(slots=True)
class MenuItem:
    """Typed descriptor for an item created by ``create_menu``."""

    text: str
    tooltip: str = ""
    shortcut: ShortcutLike = None
    callback: MenuCallback | None = None
    checkable: bool = False
    # A str is looked up through load_icon (action id, SVG name, or Fluent
    # glyph token), same as before. A QIcon is used as-is - for a menu item
    # that has to render pixel-identical to an icon built elsewhere from raw
    # SVG source (icon_from_svg_source), such as a plugin's own artwork,
    # rather than to whatever a same-named catalogue/file token resolves to.
    icon: str | QIcon | None = None
    action_id: str | None = None
    checked: bool = False
    enabled: bool = True


# ----------------------------------------------------------------------
# Icons
# ----------------------------------------------------------------------
ICON_PLATFORM_DIRS: dict[str, str] = {"windows": "win11", "darwin": "macOs"}
ICON_COMMON_DIR: str = "common"

_FLUENT_GLYPH_MIN = 0xE700
_FLUENT_GLYPH_MAX = 0xF8FF
# Fallback ink only.  The real colour comes from _symbol_tint(), the same
# source the SF Symbol path uses - the two glyph backends were tinted
# differently, so the Windows glyphs stayed a fixed dark grey and vanished on
# a dark theme while the macOS ones followed the palette.
_FLUENT_DEFAULT_COLOR = "#404040"
# Keyed by what changes the bitmap: the glyph or symbol, the logical size,
# the tint, and the pixel ratio it was rasterised for.  The ratio belongs in
# the key - without it a 1x icon built during startup is handed back on a
# Retina screen, which is the blur this was meant to fix.
_FLUENT_ICON_CACHE: dict[tuple[str, int, str, float], QIcon] = {}
_SF_SYMBOL_ICON_CACHE: dict[tuple[str, int, str, float], QIcon] = {}

# The AppKit module, looked up once.  None means "not available"; the flag
# distinguishes that from "not looked up yet", so the warning is logged once
# rather than for every button.
_SF_SYMBOL_BRIDGE: Any | None = None
_SF_SYMBOL_BRIDGE_CHECKED: bool = False

# The oldest pyobjc-core that is safe to import, per Python version.
#
# pyobjc-core is a C extension built against CPython's internals, and it is
# unusually sensitive to them.  A wheel built during a Python beta keeps its
# cp3XX tag after those internals move underneath it, so pip installs it
# happily and the interpreter then dies on import - a native crash, below the
# level any try/except can reach.  Refusing to import is the only defence.
#
# Each entry reads: on this Python or newer, require this pyobjc-core.
_MIN_PYOBJC_CORE: tuple[tuple[tuple[int, int], tuple[int, ...]], ...] = (
    ((3, 14), (12, 0)),  # 11.1's cp314 wheel predates the 3.14 release
    ((3, 10), (11, 0)),
)


def _icon_search_dirs() -> list[Path]:
    """Return platform-specific and common SVG icon folders."""
    platform_dir = ICON_PLATFORM_DIRS.get(platform.system().lower())
    dirs = [_ICONS_DIR / platform_dir] if platform_dir else []
    dirs.append(_ICONS_DIR / ICON_COMMON_DIR)
    return dirs


def _icon_search_key() -> tuple[str, ...]:
    """Return the search directories as a hashable cache key.

    Recomputed per call rather than cached: it reads ``platform.system()``,
    which the tests substitute to check that a platform folder wins over
    ``common``, and a cached key would make that substitution invisible.
    """
    return tuple(str(directory) for directory in _icon_search_dirs())


def _normalize_icon_name(icon: str | None) -> str:
    """Return a normalized action id, SVG icon id, SF Symbol id or glyph id."""
    return str(icon or "").strip()



def action_presentation(action_id: str) -> tuple[QIcon, str, str]:
    """Return the icon, label and tooltip configured for an action.

    The action catalogue is the only source for button text, tooltips and icon
    tokens. Missing action entries intentionally produce an empty presentation,
    which makes catalogue mistakes visible during development instead of hiding
    them behind hard-coded fallbacks.
    """
    spec = action(action_id)
    text = spec.translated_text()
    tooltip = spec.translated_description() or text
    return icon_from_action_spec(spec), text, tooltip


def _is_fluent_glyph(value: str | None) -> bool:
    """Return True for a Segoe Fluent private-use glyph or hex code."""
    text = _normalize_icon_name(value)
    if not text:
        return False

    if len(text) == 1:
        return _FLUENT_GLYPH_MIN <= ord(text) <= _FLUENT_GLYPH_MAX

    candidate = text.upper()
    if candidate.startswith("U+"):
        candidate = candidate[2:]
    elif candidate.startswith("0X"):
        candidate = candidate[2:]

    if not (3 <= len(candidate) <= 6):
        return False

    try:
        codepoint = int(candidate, 16)
    except ValueError:
        return False

    return _FLUENT_GLYPH_MIN <= codepoint <= _FLUENT_GLYPH_MAX


def _fluent_glyph_char(value: str) -> str:
    """Return the actual glyph character for a Fluent code or glyph."""
    text = _normalize_icon_name(value)
    if len(text) == 1:
        return text

    candidate = text.upper()
    if candidate.startswith("U+"):
        candidate = candidate[2:]
    elif candidate.startswith("0X"):
        candidate = candidate[2:]

    return chr(int(candidate, 16))


def _best_fluent_font_family() -> str | None:
    """Return an installed Fluent icon font, or None.

    Do not fall back to a generic/system font here. On Windows, asking Qt to
    render a private-use Fluent glyph with a missing font can trigger DirectWrite
    fallback through legacy bitmap fonts such as 8514oem or Fixedsys, which
    produces noisy CreateFontFaceFromHDC warnings.
    """
    families = set(QFontDatabase.families())
    for family in ("Segoe Fluent Icons", "Segoe MDL2 Assets"):
        if family in families:
            return family
    return None


def _create_fluent_icon(
    glyph: str,
    *,
    size: int = 20,
    color: str = _FLUENT_DEFAULT_COLOR,
) -> QIcon:
    """Create a QIcon by painting a Segoe Fluent glyph into a pixmap.

    Returns an empty QIcon when no Fluent icon font is installed. That lets the
    caller fall back to the configured SVG icon without causing DirectWrite font
    fallback warnings.
    """
    font_family = _best_fluent_font_family()
    if font_family is None:
        return QIcon()

    glyph_char = _fluent_glyph_char(glyph)
    color_name = str(color or _FLUENT_DEFAULT_COLOR)
    ratio = _icon_device_pixel_ratio()
    cache_key = (glyph_char, int(size), color_name, round(ratio, 2))

    cached = _FLUENT_ICON_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # The pixmap carries the display's pixel ratio, so the coordinates below
    # stay logical while the glyph is rasterised at full resolution.
    pixmap = create_hidpi_pixmap(size, size)

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        font = QFont(font_family)
        font.setPixelSize(max(1, int(size * 0.85)))
        painter.setFont(font)
        qcolor = QColor(color_name)
        painter.setPen(qcolor if qcolor.isValid() else QColor(_FLUENT_DEFAULT_COLOR))
        painter.drawText(
            QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, glyph_char
        )
    finally:
        painter.end()

    icon = QIcon(pixmap)
    _FLUENT_ICON_CACHE[cache_key] = icon
    return icon


@lru_cache(maxsize=None)
def _svg_icon_path_cached(name: str, search_dirs: tuple[str, ...]) -> Path | None:
    """Look the file up on disk.  Cached: the icon set does not change at run time.

    Every button and every menu item resolves an icon, repeatedly - rebuilding
    a toolbar asks for the same forty names again - and each miss costs one
    stat per search directory.  Measured at 0.34 ms per ``load_icon`` before
    this, which was most of the cost of opening a populated menu.

    ``search_dirs`` is part of the key rather than read inside, because it
    depends on the platform: caching on the name alone would answer for macOS
    with whatever Windows was asked first.
    """
    for directory in search_dirs:
        path = Path(directory) / f"{name}.svg"
        if path.exists():
            return path

    return None


def _svg_icon_path(icon_id: str | None) -> Path | None:
    """Return an SVG icon path for a raw icon id. No aliases, no PNG/ICO."""
    name = _normalize_icon_name(icon_id).lower()
    if not name:
        return None

    return _svg_icon_path_cached(name, _icon_search_key())


def resolve_icon_path(icon: str | None) -> Path | None:
    """Return only SVG-based icon paths.

    If ``icon`` is an action id, the action's regular ``icon`` field is used.
    Platform-rendered symbol icons intentionally do not have paths.

    An action whose id happens to match a file name falls through to that file
    rather than resolving to nothing: ``new`` is both an action and new.svg,
    and an action that names no SVG would otherwise make the file with the same
    name unreachable - which is a trap, because the two are unrelated.
    """
    token = _normalize_icon_name(icon)
    if not token or _is_fluent_glyph(token):
        return None

    path = _svg_icon_path(action(token).icon)
    return path if path is not None else _svg_icon_path(token)


def get_icon_file_name(icon: str) -> str | None:
    """Return the SVG icon file name relative to the icons directory."""
    path = resolve_icon_path(icon)
    return None if path is None else str(path.relative_to(_ICONS_DIR))


def _pyobjc_core_is_safe_to_import() -> bool:
    """Say whether the installed pyobjc-core can be imported without crashing.

    Read from the distribution metadata rather than by importing the package:
    the failure being guarded against is a segmentation fault inside the import
    itself, so asking the question has to cost nothing native.

    An unknown or unparseable version is treated as safe.  The guard exists to
    catch one known-bad combination, not to police the dependency: refusing to
    run on a version this table has not heard of would age badly.
    """
    try:
        import importlib.metadata

        installed = importlib.metadata.version("pyobjc-core")
    except Exception:
        return True

    try:
        parsed = tuple(int(part) for part in installed.split(".")[:3])
    except ValueError:
        return True

    for python_version, minimum in _MIN_PYOBJC_CORE:
        if sys.version_info[:2] >= python_version:
            if parsed >= minimum:
                return True
            applogger.warning(
                "pyobjc-core %s is not built for Python %d.%d and importing it "
                "would crash the application, so the SVG icons are used "
                "instead. Upgrade it with: pip install -U "
                "'pyobjc-framework-Cocoa>=%s'",
                installed,
                sys.version_info[0],
                sys.version_info[1],
                ".".join(str(part) for part in minimum),
                show_dialog=False,
                raise_error=False,
            )
            return False

    return True


def _sf_symbol_bridge() -> Any | None:
    """Return the AppKit module, or None with one explanation in the log.

    SF Symbols are drawn by AppKit, which reaches Python through pyobjc.  It is
    a real dependency and it is easy not to have: without it every symbol
    lookup fails, and the previous version of this code swallowed that in a
    bare ``except``, so the whole icon set silently fell back to SVG with
    nothing anywhere to say why.
    """
    global _SF_SYMBOL_BRIDGE, _SF_SYMBOL_BRIDGE_CHECKED

    if _SF_SYMBOL_BRIDGE_CHECKED:
        return _SF_SYMBOL_BRIDGE

    # Cleared up front so that both failure paths below leave the memo saying
    # "looked, found nothing" rather than whatever it happened to hold.
    _SF_SYMBOL_BRIDGE_CHECKED = True
    _SF_SYMBOL_BRIDGE = None
    if not _pyobjc_core_is_safe_to_import():
        return None

    try:
        import AppKit  # type: ignore[import-not-found]
    except ImportError:
        applogger.warning(
            "SF Symbols need pyobjc, which is not installed; using the SVG "
            "icons instead. Install it with: pip install pyobjc-framework-Cocoa",
            show_dialog=False,
            raise_error=False,
        )
        return None

    _SF_SYMBOL_BRIDGE = AppKit
    return AppKit


def _create_sf_symbol_icon(symbol: str, *, size: int = 20) -> QIcon:
    """Create a QIcon from a macOS SF Symbol id.

    Returns an empty QIcon when the symbol cannot be drawn, so the caller falls
    back to the SVG.  Every failure is logged once per symbol: an icon that is
    quietly missing is the kind of thing nobody reports and nobody fixes.

    The symbol is rasterised at the display's pixel density rather than at the
    logical size: a 20 px bitmap stretched across a Retina screen's 40 device
    pixels is the whole of why these icons looked soft next to the SVGs, which
    are drawn as vectors and so never had the problem.
    """
    name = _normalize_icon_name(symbol)
    if not name or not _IS_MACOS:
        return QIcon()

    ratio = _icon_device_pixel_ratio()
    # The tint belongs in the key, exactly as it does for the Fluent glyphs.
    # Without it, symbols built once under the light palette were handed back
    # unchanged after a switch to the dark theme - dark ink on a dark button,
    # cached and therefore permanent until restart.
    tint = _symbol_tint()
    cache_key = (name, int(size), tint, round(ratio, 2))
    cached = _SF_SYMBOL_ICON_CACHE.get(cache_key)
    if cached is not None:
        return cached

    appkit = _sf_symbol_bridge()
    if appkit is None:
        return QIcon()

    icon = QIcon()
    try:
        image = appkit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
            name, None
        )
        if image is None:
            # A name this version of macOS does not know: worth saying, since
            # the catalogue is hand-written and a typo looks like nothing.
            applogger.warning(
                "Unknown SF Symbol %r; using the SVG icon instead.",
                name,
                show_dialog=False,
                raise_error=False,
            )
            _SF_SYMBOL_ICON_CACHE[cache_key] = icon
            return icon

        # Ask for the symbol at the *device* size.  A point size of 20 gives a
        # 20 px bitmap, which Qt then has to stretch over 40 physical pixels on
        # a Retina display; asking for 40 up front means no resampling at all.
        configuration = appkit.NSImageSymbolConfiguration.configurationWithPointSize_weight_scale_(
            float(size) * ratio, appkit.NSFontWeightRegular, 2  # 2 = .medium scale
        )
        configured = image.imageWithSymbolConfiguration_(configuration) or image
        configured.setTemplate_(True)

        tiff = configured.TIFFRepresentation()
        if tiff is None:
            raise ValueError("the symbol produced no TIFF representation")

        rep = appkit.NSBitmapImageRep.imageRepWithData_(tiff)
        if rep is None:
            raise ValueError("the TIFF representation could not be read")

        # An empty properties dictionary, not None: the parameter is typed as
        # an NSDictionary and pyobjc will not convert None for it.
        png_data = rep.representationUsingType_properties_(
            appkit.NSBitmapImageFileTypePNG, {}
        )
        if png_data is None:
            raise ValueError("the symbol could not be encoded as PNG")

        # No format argument: Pylance's PySide6 stubs type that overload
        # incorrectly, and Qt detects PNG from the bytes anyway.
        qimage = QImage.fromData(bytes(png_data))
        if qimage.isNull():
            raise ValueError("Qt could not read the PNG data")

        icon = QIcon(
            _tinted_pixmap(QPixmap.fromImage(qimage), size, tint, ratio=ratio)
        )
    except Exception as error:
        applogger.warning(
            "SF Symbol %r could not be rendered (%s); using the SVG icon instead.",
            name,
            error,
            show_dialog=False,
            raise_error=False,
        )
        icon = QIcon()

    # Cached either way, including the empty icon: a symbol that failed once
    # will fail every time, and this runs for every button on every menu.
    _SF_SYMBOL_ICON_CACHE[cache_key] = icon
    return icon


def _icon_device_pixel_ratio() -> float:
    """Return how many physical pixels the display gives one logical pixel.

    Read from the primary screen rather than from the application: the ratio
    is a property of a display, and the application-level accessor is not part
    of QCoreApplication's interface even where it happens to resolve.  Icons
    are cached, so they cannot follow a window between screens of different
    densities anyway; the primary screen is the honest approximation.

    Falls back to 2.0 on macOS when no screen exists yet, since icons are
    sometimes built while the app is still starting and every Mac that can run
    this is Retina.  Clamped to 3: beyond that the bitmaps grow faster than
    anyone can see the difference.
    """
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return 2.0 if _IS_MACOS else 1.0

    return min(max(float(screen.devicePixelRatio()), 1.0), 3.0)


def create_hidpi_pixmap(width: int, height: int) -> QPixmap:
    """Return a transparent pixmap that paints sharply on a Retina display.

    The bitmap is allocated at the display's physical pixel count and labelled
    with its ratio, which is what tells Qt to draw it one device pixel per
    pixel rather than interpolating.  Because the ratio is set on the pixmap,
    QPainter scales for it: callers keep drawing in the logical coordinates
    they already use, and only the result is sharper.

    Use this anywhere an icon is painted by hand.  Every such place in the app
    allocated the bitmap at logical size, so all of them were soft on a Mac
    while the SVG icons beside them - vectors, rasterised on demand - were not.
    """
    ratio = _icon_device_pixel_ratio()
    pixmap = QPixmap(
        max(1, int(round(width * ratio))), max(1, int(round(height * ratio)))
    )
    pixmap.fill(Qt.GlobalColor.transparent)
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


#: Whether the theme currently applied is a dark one.  Set by
#: apply_platform_style, which is the only thing that knows.
_ACTIVE_THEME_IS_DARK: bool = False

#: Glyph ink, light and dark.  Near-black rather than pure black because the
#: system draws its own symbols that way and it is what makes them read as
#: crisp; near-white for the same reason in reverse.
SYMBOL_INK_LIGHT: str = "#eaeaea"
SYMBOL_INK_DARK: str = "#1a1a1a"


def _symbol_tint() -> str:
    """Return the colour to paint SF Symbols and Fluent glyphs in.

    Decided by the theme this application applied, not by reading ButtonText
    off the palette.  The palette is not a reliable answer here: a Qt style
    plugin installs its own, and on a Mac whose system appearance is dark that
    palette comes back light-on-dark even though the app is wearing a light
    theme - which painted white glyphs onto white buttons.

    The rule is therefore the simple one: light ink only under a dark theme,
    dark ink under everything else.
    """
    return SYMBOL_INK_LIGHT if _ACTIVE_THEME_IS_DARK else SYMBOL_INK_DARK


def _tinted_pixmap(
    pixmap: QPixmap,
    size: int,
    color: str = _FLUENT_DEFAULT_COLOR,
    *,
    ratio: float = 1.0,
) -> QPixmap:
    """Repaint a pixmap's opaque parts in one colour at the display's density.

    ``size`` is in logical pixels and ``ratio`` says how many real ones each of
    those is worth.  The bitmap is built at the physical size and then labelled
    with its ratio, so Qt draws it one-for-one instead of interpolating - the
    difference between a sharp icon and a smeared one.
    """
    physical = max(1, int(round(size * ratio)))
    scaled = (
        pixmap
        if pixmap.width() == physical and pixmap.height() == physical
        else pixmap.scaled(
            physical,
            physical,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    )

    qcolor = QColor(color)
    if qcolor.isValid():
        painter = QPainter(scaled)
        try:
            # SourceIn keeps the alpha and replaces the colour.
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceIn
            )
            painter.fillRect(scaled.rect(), qcolor)
        finally:
            painter.end()

    # Without this Qt treats the bitmap as logical pixels and draws it at twice
    # the intended size.
    scaled.setDevicePixelRatio(ratio)
    return scaled

_SVG_ICON_CACHE: dict[tuple[str, tuple[str, ...]], QIcon] = {}


def _svg_icon(icon_id: str | None) -> QIcon:
    """Return the shipped SVG for an id, or an empty icon when there is none.

    A QIcon built from an SVG rasterises on demand at whatever size it is drawn,
    so unlike the two bitmap paths this one needs neither a size nor a pixel
    ratio in its key - the same object is correct on every display.  The search
    directories are in the key for the same reason as above.
    """
    name = _normalize_icon_name(icon_id).lower()
    if not name:
        return QIcon()

    key = (name, _icon_search_key())
    cached = _SVG_ICON_CACHE.get(key)
    if cached is not None:
        return cached

    path = _svg_icon_path(name)
    icon = QIcon(str(path)) if path is not None else QIcon()
    _SVG_ICON_CACHE[key] = icon
    return icon


_SVG_SOURCE_ICON_CACHE: dict[tuple[str, int, float], QIcon] = {}


#: The document a bare icon body is wrapped in.  An operation declares its
#: artwork as path data alone - see ``SeriesOperationDialogBase.Icon`` - which
#: is the readable half; the frame around it is always the same and repeating
#: it in eleven plugins would only be eleven places to get the viewBox wrong.
SVG_ICON_DOCUMENT: str = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
    'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
)


def svg_icon_document(body: str, color: str | None = None) -> str:
    """Return icon path data wrapped in the standard SVG document."""
    return SVG_ICON_DOCUMENT.format(color=color or _symbol_tint(), body=body)


def icon_from_svg_source(
    svg: str | None,
    *,
    size: int = 20,
    color: str | None = None,
) -> QIcon:
    """Return a QIcon drawn from an SVG document held in memory.

    For icons that travel inside the thing they belong to rather than in
    ``app/icons``: a series operation is a plugin, and one .py file dropped
    into ``app/series_operations`` has to be the whole of it, artwork
    included.  See ``SeriesOperationDialogBase.Icon``.

    Unlike ``_svg_icon`` this rasterises once at a chosen size, because the
    QIcon is built from a pixmap rather than from a file Qt can re-read.  So
    the pixel ratio belongs in the key, exactly as it does for the SF Symbol
    and Fluent glyph paths.

    Malformed markup returns an empty icon and says so once: an operation that
    silently lost its icon is the kind of thing nobody reports.
    """
    source = (svg or "").strip()
    if not source:
        return QIcon()

    if "<svg" not in source.lower():
        # Path data on its own: wrap it. This used to be done by the widget
        # that lists the operations, so every *other* caller got an empty icon
        # from perfectly good markup - which is why the operation dialogs had
        # no window icon and nothing said so.
        source = svg_icon_document(source, color)

    ratio = _icon_device_pixel_ratio()
    # Keyed on the document itself.  These come from class attributes, so the
    # same str object arrives every time and Python has already cached its
    # hash; there is nothing to gain from digesting it first.
    cache_key = (source, int(size), round(ratio, 2))  # colour is inside source
    cached = _SVG_SOURCE_ICON_CACHE.get(cache_key)
    if cached is not None:
        return cached

    renderer = QSvgRenderer(QByteArray(source.encode("utf-8")))
    if not renderer.isValid():
        applogger.warning(
            "An embedded icon could not be parsed as SVG (starts %r); the "
            "widget will have no icon.",
            source[:60],
            show_dialog=False,
            raise_error=False,
        )
        icon = QIcon()
    else:
        # KeepAspectRatio rather than the default stretch: these are hand-drawn
        # and a non-square viewBox should letterbox, not distort.
        renderer.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        pixmap = create_hidpi_pixmap(size, size)
        painter = QPainter(pixmap)
        try:
            renderer.render(painter, QRectF(0, 0, size, size))
        finally:
            painter.end()
        icon = QIcon(pixmap)

    # Cached either way: markup that failed to parse will fail every time, and
    # this runs for every operation button the window builds.
    _SVG_SOURCE_ICON_CACHE[cache_key] = icon
    return icon


def _svg_icon_or_empty(icon_id: str | None) -> QIcon:
    """Return an SVG icon by name, or an empty icon when the file is missing.

    SVG lookup already searches the platform-specific icon folder first and the
    common folder second. There is no implicit fallback to the action id here:
    only the configured ``icon`` value is considered.
    """
    return _svg_icon(icon_id) if _normalize_icon_name(icon_id) else QIcon()


def icon_from_action_spec(spec: "ActionSpec") -> QIcon:
    """Return the icon for an action using the catalogue priority order.

    Priority is intentionally simple:
    1. On macOS, use ``SFSymbol`` when configured.
    2. On Windows, use ``SegoeFluent`` when configured.
    3. Otherwise - or when the symbol or glyph could not be drawn - load the
       configured SVG ``icon`` from the platform folder, then from ``common``.
    4. If no configured source exists or the configured source cannot be loaded,
       return an empty ``QIcon``.
    """
    # Each step falls through when it produces nothing, which is the whole
    # reason every action also names an SVG. Returning the empty icon straight
    # from the symbol path is why a Mac without pyobjc had a toolbar of blank
    # buttons while the SVGs sat unused beside it.
    if _IS_MACOS and spec.sf_symbol:
        icon = _create_sf_symbol_icon(spec.sf_symbol)
        if not icon.isNull():
            return icon

    if _IS_WINDOWS and spec.segoe_fluent and _is_fluent_glyph(spec.segoe_fluent):
        icon = _create_fluent_icon(spec.segoe_fluent, color=_symbol_tint())
        if not icon.isNull():
            return icon

    return _svg_icon_or_empty(spec.icon)


def load_icon(icon: str | None) -> QIcon:
    """Return an icon for an action id, SVG name, or Fluent glyph token.

    Action ids use ``icon_from_action_spec`` and therefore follow the catalogue
    priority order. Non-action tokens are treated literally: a Fluent glyph token
    is rendered as Fluent, and any other token is looked up as an SVG name. When
    a token cannot be rendered or found, the result is an empty ``QIcon``.
    """
    token = _normalize_icon_name(icon)
    if not token:
        return QIcon()

    spec = _catalog().get(token)
    if spec is not None:
        return icon_from_action_spec(spec)

    if _is_fluent_glyph(token):
        return _create_fluent_icon(token, color=_symbol_tint())

    return _svg_icon_or_empty(token)


# ----------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------
def _load_qss(name: str) -> tuple[str | None, Path | None]:
    """Load a QSS file by name from the app styles directory."""
    path = _STYLES_DIR / name
    if not path.exists():
        applogger.warning("QSS not found: %s", path)
        return None, None

    try:
        return path.read_text(encoding="utf-8"), path
    except Exception:
        applogger.exception("Failed to read QSS: %s", path)
        return None, None


def _icon_url(icon_name: str, *, dark: bool = False) -> str:
    """Return a repository-relative icon URL for QSS token replacement.

    ``dark`` picks the light-ink variant.  The chevrons the sheets draw into
    combo boxes and spin boxes are a single fixed-colour path, so on a dark
    theme the original is a dark glyph on a dark background - present, correct,
    and invisible.  A second file rather than a runtime tint because a Qt
    stylesheet takes a ``url()`` and nothing else: there is no way to hand it a
    QIcon we have already recoloured.
    """
    if dark:
        # Resolved as a file directly, not through get_icon_file_name: that
        # goes via the action catalogue first, and "down_on_dark" is a file
        # name rather than an action - so every dark theme applied logged two
        # warnings about actions nobody had claimed existed.
        variant = _svg_icon_path(f"{icon_name}_on_dark")
        if variant is not None:
            return f"app/icons/{variant.relative_to(_ICONS_DIR).as_posix()}"
    filename = get_icon_file_name(icon_name)
    return f"app/icons/{filename}" if filename else ""


#: What the user can choose between, and what each choice does.
#:
#: Two different mechanisms sit behind one setting, deliberately.  A *theme* is
#: one of this application's stylesheets plus a palette; a *Qt style* is a
#: plugin Qt itself provides - Fusion, Windows, and whatever the desktop has
#: installed - drawn entirely by Qt with no sheet of ours on top.  They are
#: offered together because from where the user sits they answer one question,
#: and kept apart in code because they have nothing else in common.
APP_STYLE_AUTOMATIC: str = "automatic"

#: theme key -> (stylesheet file, palette key)
APP_STYLE_QSS: dict[str, tuple[str, str]] = {
    "fluent_win11": ("fluent_win11.qss", "light"),
    "macos_native": ("macos_native.qss", "light"),
    # The dark theme is the Fluent *structure* with the dark palette: the
    # sheets carry no opaque surface colours worth speaking of, so a second
    # 1,000-line file would differ from this one only in tokens.
    "dark": ("fluent_win11.qss", "dark"),
}

#: Prefix marking a Qt style plugin rather than one of our themes.
QT_STYLE_PREFIX: str = "qt:"

CONFIG_APP_STYLE: str = "app_style"

#: Labels for the themes.  Qt styles are labelled by their own key, since that
#: is the name their documentation and their users know them by.
APP_STYLE_LABELS: dict[str, str] = {
    APP_STYLE_AUTOMATIC: "Automatic",
    "fluent_win11": "Fluent (Windows 11)",
    "macos_native": "macOS native",
    "dark": "Dark",
}


def available_qt_styles() -> list[str]:
    """Return the Qt style plugins this installation actually has.

    Enumerated rather than listed.  Fusion and Windows ship with Qt; Breeze,
    Oxygen and QtCurve are KDE plugins that have to be installed separately and
    are absent from a stock macOS or Windows machine.  Hard-coding those names
    would put three entries in the menu that silently do nothing - so they
    appear here exactly when they can be applied, and not otherwise.
    """
    return list(QtWidgets.QStyleFactory.keys())


def available_app_styles() -> list[tuple[str, str]]:
    """Return every selectable style as (stored key, label to show)."""
    styles = [(key, APP_STYLE_LABELS[key]) for key in APP_STYLE_LABELS]
    styles.extend(
        (f"{QT_STYLE_PREFIX}{name}", f"Qt: {name}") for name in available_qt_styles()
    )
    return styles


def _automatic_qss_name() -> str | None:
    """Return the stylesheet this desktop would pick for itself."""
    system = platform.system().lower()
    if system.startswith("win"):
        return "fluent_win11.qss"
    if system == "darwin":
        return "macos_native.qss"
    return None


def resolve_app_style(preference: str | None = None) -> str:
    """Return a valid app-style key, falling back to automatic.

    An unknown value reads as automatic rather than as "no stylesheet": a typo
    in config.json, or a Qt plugin that was uninstalled since it was chosen,
    should leave the app looking normal rather than unstyled.
    """
    clean = str(preference or "").strip()
    if not clean:
        clean = str(get_value(CONFIG_APP_STYLE, "") or "").strip()

    if clean.startswith(QT_STYLE_PREFIX):
        name = clean[len(QT_STYLE_PREFIX) :]
        available = {key.lower(): key for key in available_qt_styles()}
        if name.lower() in available:
            return f"{QT_STYLE_PREFIX}{available[name.lower()]}"
        applogger.warning(
            "The Qt style %r is not installed; using the automatic style.",
            name,
            show_dialog=False,
            raise_error=False,
        )
        return APP_STYLE_AUTOMATIC

    lowered = clean.lower()
    return lowered if lowered in APP_STYLE_QSS else APP_STYLE_AUTOMATIC


def apply_platform_style(
    app: QApplication, preference: str | None = None
) -> PlatformStyle:
    """Apply the configured theme or Qt style.

    ``preference`` overrides what config.json says, so the settings dialog can
    show a choice before committing it.

    Every path here sets *both* the stylesheet and the palette, including to
    nothing.  A style change has to undo the previous one completely: leaving a
    dark palette under a light sheet is how a theme switcher produces a window
    that belongs to neither theme.
    """
    system = platform.system().lower()
    style_key = resolve_app_style(preference)

    global _ACTIVE_THEME_IS_DARK

    if style_key.startswith(QT_STYLE_PREFIX):
        name = style_key[len(QT_STYLE_PREFIX) :]
        # A Qt style draws itself; none of ours is a dark theme, so the glyphs
        # stay dark whatever palette the plugin brings with it.
        _ACTIVE_THEME_IS_DARK = False
        app.setStyleSheet("")
        app.setPalette(QtWidgets.QStyleFactory.create(name).standardPalette())
        app.setStyle(name)
        return PlatformStyle(system, None)

    if style_key == APP_STYLE_AUTOMATIC:
        qss_name, palette_key = _automatic_qss_name(), "light"
    else:
        qss_name, palette_key = APP_STYLE_QSS[style_key]

    qss, path = _load_qss(qss_name) if qss_name else ("", None)

    # Read the sheet first, then theme it: themed_qss hands back the palette it
    # resolved, and the palette has to be installed even when there is no sheet
    # to go with it - a desktop with no QSS of its own still gets the theme's
    # colours through Qt.
    themed, palette = themed_qss(qss or "", palette_key)
    _ACTIVE_THEME_IS_DARK = palette.dark
    app.setPalette(palette.qpalette())

    if not themed:
        # Nothing to install, so clear whatever the previous choice left rather
        # than leaving half of it behind.
        app.setStyleSheet("")
        return PlatformStyle(system, None)

    # The chevrons are files, and there is a light-ink copy of each for the
    # dark themes; see _icon_url.

    themed = themed.replace("@ICON_CHEVRON_DOWN@", _icon_url("down", dark=palette.dark))
    themed = themed.replace("@ICON_CHEVRON_UP@", _icon_url("up", dark=palette.dark))
    app.setStyleSheet(themed)
    return PlatformStyle(system, path)

# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------

_PRIMARY_BUTTON_TEXTS = {"Apply", "OK"}


def _mark_primary_button(widget: QWidget, text: str) -> None:
    """Mark primary buttons for QSS styling."""
    if text in _PRIMARY_BUTTON_TEXTS:
        widget.setProperty("primary", True)



def create_section_title(text: str, parent: QWidget | None = None) -> QtWidgets.QLabel:
    """Create a standard section title label styled through QSS."""
    label = QtWidgets.QLabel(text, parent)
    label.setProperty("sectionTitle", True)
    label.setContentsMargins(0, 0, 0, 0)
    return label


def create_card_widget(parent: QWidget | None = None, object_name: str | None = None) -> QFrame:
    """Create a lightweight card/container frame styled through QSS."""
    card = QFrame(parent)
    if object_name:
        card.setObjectName(object_name)
    card.setProperty("card", True)
    card.setFrameShape(QFrame.Shape.NoFrame)
    card.setMinimumWidth(0)
    card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    return card


def mark_editor_panel(widget: QWidget) -> QWidget:
    """Mark list/tree/editor widgets for shared panel styling."""
    widget.setProperty("editorPanel", True)
    repolish_widget(widget)
    return widget


def create_action_button(
    parent: QWidget,
    action_id: str,
    action: MenuCallback | None,
    layout: QBoxLayout | None = None,
    *,
    presentation: tuple[QIcon, str, str] | None = None,
) -> QPushButton:
    """Create a push button from a catalogue action, and nothing else.

    Label, tooltip and icon all come from ``config.json`` and go through the
    translator, so a button is one line at the call site and its wording is
    editable and localisable without touching Python.  A button that wants
    different words wants its own action id.

    ``presentation`` supplies ``(icon, text, tooltip)`` directly, for the
    callers that are deliberately not in the catalogue: a series operation
    carries its own name, description and icon so that the plugin is one
    self-contained file.  Passing it skips the lookup entirely rather than
    overriding it afterwards - an id that is not in config.json is worth a
    warning, and a plugin would trip it on every button.
    """
    icon, text, tooltip = (
        presentation if presentation is not None else action_presentation(action_id)
    )

    button = QPushButton(text, parent)
    button.setToolTip(tooltip)
    button.setStatusTip(tooltip)
    button.setIcon(icon)

    if action is not None:
        button.clicked.connect(action)
    else:
        # Nothing to connect: a button that silently does nothing when pressed
        # is worse than one that looks unavailable.
        button.setEnabled(False)

    _mark_primary_button(button, text)

    if layout is not None:
        layout.addWidget(button)

    return button


def create_toolbar_button(
    parent: QWidget,
    action_id: str,
    action: MenuCallback | None,
    layout: QBoxLayout | None = None,
) -> QToolButton:
    """Create a flat toolbar button from a catalogue action.

    Same contract as ``create_action_button``; only the widget class differs.
    """
    icon, text, tooltip = action_presentation(action_id)

    button = QToolButton(parent)
    # Named after the action, not after its label: the object name is what QSS
    # selects on, and a stylesheet keyed to translated text would stop matching
    # the moment the interface was translated.
    button.setObjectName(f"{action_id}Button")
    button.setToolTip(tooltip)
    button.setStatusTip(tooltip)
    button.setAutoRaise(True)
    button.setText(text)
    button.setIcon(icon)

    if action is not None:
        button.clicked.connect(action)
    else:
        button.setEnabled(False)

    _mark_primary_button(button, text)

    if layout is not None:
        layout.addWidget(button)

    return button

def create_menu_item(
    parent: QWidget,
    menu: QMenu | None,
    icon: str | QIcon | None,
    checkable: bool,
    text: str,
    tooltip: str,
    key: ShortcutLike,
    action: MenuCallback | None,
    *,
    action_id: str | None = None,
    checked: bool = False,
    enabled: bool = True,
) -> QAction:
    """Create and optionally add a QAction to a menu."""
    qaction = QAction(parent)
    qaction.setText(text)
    qaction.setToolTip(tooltip)
    qaction.setStatusTip(tooltip)
    qaction.setIconVisibleInMenu(True)
    qaction.setCheckable(checkable)
    qaction.setChecked(checked)
    qaction.setEnabled(enabled)

    if isinstance(icon, QIcon):
        qaction.setIcon(icon)
    elif icon:
        qaction.setIcon(load_icon(icon))

    if action_id is not None:
        qaction.setData(action_id)

    # Guarded rather than passed through: PySide6 raises if setShortcut is
    # given None, and "this action has no shortcut" is the common case.
    if key is not None:
        qaction.setShortcut(key)

    if action is not None:
        qaction.triggered.connect(action)

    if menu is not None:
        menu.addAction(qaction)

    return qaction


def create_menu(
    parent: QWidget,
    items: list[MenuItem | None],
) -> QMenu:
    """Create a QMenu from typed menu item descriptors."""
    menu = QMenu(parent)

    for item in items:
        if item is None:
            menu.addSeparator()
            continue

        create_menu_item(
            parent=parent,
            menu=menu,
            icon=item.icon,
            checkable=item.checkable,
            text=item.text,
            tooltip=item.tooltip,
            key=item.shortcut,
            action=item.callback,
            action_id=item.action_id,
            checked=item.checked,
            enabled=item.enabled,
        )

    return menu


def repolish_widget(widget: QtWidgets.QWidget) -> None:
    """Force Qt to re-apply style rules to a widget."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    # QWidget.update explicitly, not widget.update().  On an item view the name
    # resolves to QAbstractItemView.update(index), which requires an argument
    # and raises TypeError - so this raised on every list, tree and table it was
    # asked to repolish, which is every panel that goes through
    # mark_editor_panel.
    QWidget.update(widget)


def create_doc_link(parent: QWidget | None = None) -> QtWidgets.QLabel:
    """Create the label used for "read the documentation" links.

    Five dialogs each configured their own; four of them forgot to escape the
    title and URL, so a ``&`` in either produced broken markup.  Building the
    label here means the escaping is not something a caller can forget.
    """
    label = QtWidgets.QLabel("", parent)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    label.setOpenExternalLinks(True)
    label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    label.setWordWrap(True)
    return label


def set_doc_link(label: QtWidgets.QLabel, title: str, url: str) -> None:
    """Point a documentation label at a URL, or clear it when there is none."""
    clean_title = str(title or "").strip()
    clean_url = str(url or "").strip()

    if not clean_url:
        label.clear()
        label.setToolTip("")
        return

    label.setText(
        f'<a href="{html.escape(clean_url, quote=True)}">'
        f"{html.escape(clean_title or clean_url)}</a>"
    )
    label.setToolTip(clean_url)


def apply_card_layout(
    layout: QBoxLayout,
    *,
    margins: tuple[int, int, int, int] = MARGIN_CARD,
    spacing: int = SPACING_DEFAULT,
) -> None:
    """Give a card's inner layout breathing room.

    ``stdSizeAndlayout`` zeroes the margins, which is right for a *nested*
    layout but wrong for the outermost layout inside a card: the card draws a
    border and a background, and content flush against that border reads as
    cramped.  Use this for the layout directly inside a create_card_widget().
    """
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)


def apply_toolbox_header_metrics(
    toolbox: QtWidgets.QToolBox,
    *,
    extra_height: int = 22,
    minimum_height: int = 40,
) -> None:
    """Give a QToolBox's section headers enough room for their text.

    Why this cannot be done in QSS alone: a QToolBox header is a private
    ``QToolBoxButton`` whose height comes from its own ``sizeHint()``, computed
    from the font and icon *before* the stylesheet's padding is added.  Setting
    ``padding`` in QSS therefore grows the box the text is drawn in without
    growing the button, so the label ends up vertically clipped - which is
    exactly what the properties accordion was doing.

    Setting a minimum height from the real font metrics fixes it for whatever
    font the platform actually resolved, instead of guessing a pixel value.
    Call again after adding items.
    """
    for button in toolbox.findChildren(QtWidgets.QAbstractButton):
        # Only the section headers are direct children of the toolbox itself;
        # anything deeper belongs to a page.
        if button.parent() is not toolbox:
            continue

        line_height = button.fontMetrics().height()
        height = max(minimum_height, line_height + extra_height)
        # Both, not just the minimum: on macOS the style draws the header from
        # its own sizeHint, and a minimum alone still let it collapse back to
        # the default height once the toolbox was laid out.
        button.setMinimumHeight(height)
        button.setFixedHeight(height)
        button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        # Headers are section titles, not truncatable captions.
        button.setToolTip(button.text())


def apply_dialog_shell(
    dialog: QtWidgets.QDialog,
    root_layout: QBoxLayout,
    *,
    size: str | QSize | None = "medium",
    margins: tuple[int, int, int, int] = MARGIN_DIALOG,
    spacing: int = SPACING_DEFAULT,
) -> None:
    """Give a dialog the shared root margin, spacing, and size.

    Every dialog root goes through here so padding is decided in one place.
    ``size`` accepts a key from DIALOG_SIZES, an explicit QSize, or None to
    leave the current size alone.
    """
    root_layout.setContentsMargins(*margins)
    root_layout.setSpacing(spacing)

    if size is None:
        return

    resolved = DIALOG_SIZES.get(size) if isinstance(size, str) else size
    if resolved is None:
        applogger.warning(
            "Unknown dialog size %r; leaving the dialog size untouched.",
            size,
            show_dialog=False,
            raise_error=False,
        )
        return

    dialog.resize(resolved)

def configure_combo_width(
    combo: QComboBox,
    minimum_contents_length: int = 0,
) -> QComboBox:
    """Make a combo fill the available width without dictating a minimum.

    ``minimum_contents_length`` is the width the caller would *like*, in
    characters; it is capped at :data:`COMBO_MIN_CONTENTS_LENGTH` because the
    combo's minimum propagates all the way up to the window's minimum width.
    Text that no longer fits is elided by Qt, so the full value is put in the
    tooltip and kept in step with the selection.

    Two identical copies of this lived in the axis and figure property panels,
    with different caps - which is why the two panels could not be narrowed to
    the same width.
    """
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    characters = min(minimum_contents_length or COMBO_MIN_CONTENTS_LENGTH,
                     COMBO_MIN_CONTENTS_LENGTH)
    combo.setMinimumContentsLength(characters)

    # And an explicit pixel minimum, which is what actually takes effect.
    # QComboBox caches its minimum size hint and only recomputes it when the
    # item list changes, so setting the policy on a combo that is already
    # populated - which is exactly what relax_minimum_width does - leaves the
    # old, item-derived floor in place.  An explicit minimum size is consulted
    # by the layout ahead of the hint, so it does not depend on that timing.
    combo.setMinimumWidth(
        combo.fontMetrics().horizontalAdvance("x") * characters + _COMBO_CHROME_WIDTH
    )

    combo.setToolTip(combo.currentText())
    combo.currentTextChanged.connect(combo.setToolTip)
    return combo


def relax_minimum_width(root: QWidget, *, minimum: int = 0) -> QWidget:
    """Let *root* and everything inside it shrink to *minimum* pixels wide.

    A container is only as narrow as its widest child's minimum, and that
    minimum is rarely set on purpose: a QComboBox reserves room for N
    characters, a QAbstractScrollArea reserves a couple of rows, and the sum
    quietly becomes the floor for the whole panel - and, through the layout,
    for the window.  This walks the tree once and removes those implicit
    floors, so a splitter can be dragged to whatever the caller decided the
    real minimum is.

    Widgets whose width was pinned on purpose - ``setFixedWidth`` leaves the
    minimum and the maximum equal - are left alone: the activity rail and its
    icon buttons are that width because someone decided so, not by accident.

    Returns *root* so it can be used inline where a widget is expected.
    """
    for widget in [root, *root.findChildren(QWidget)]:
        if widget.minimumWidth() == widget.maximumWidth():
            continue
        widget.setMinimumWidth(minimum)
        if isinstance(widget, QComboBox):
            # Through configure_combo_width, not setMinimumContentsLength
            # alone: the contents length is only consulted under an
            # AdjustToMinimumContentsLength* policy, so on a combo built
            # elsewhere setting it by itself changed nothing at all.
            configure_combo_width(widget)
        elif isinstance(widget, QtWidgets.QAbstractScrollArea):
            # Views size themselves from their content; without this a table
            # with one wide column dictates the panel width.
            widget.setSizeAdjustPolicy(
                QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
            )
    return root


def stdSizeAndlayout(
    obj: QWidget | QBoxLayout | QFormLayout | QScrollArea | QPlainTextEdit,
    minimum_contents_length: int = 0,
    visible_lines:int = 5,
    word_wrap:bool=True
) -> None:
    """
    Apply standard compact sizing/layout rules.

    The intent is:
    - widgets expand when there is room;
    - widgets remain shrinkable in narrow side panels;
    - combo boxes do not force the parent wider than necessary;
    - forms grow their field column properly;
    - scroll areas do not introduce horizontal scrolling.
    """
    if isinstance(obj, QPlainTextEdit):
        stdPlainTextEdit(obj,visible_lines=visible_lines,word_wrap=word_wrap)
        return
    if isinstance(obj, QScrollArea):
        obj.setMinimumWidth(0)
        obj.setWidgetResizable(True)
        obj.setFrameShape(QFrame.Shape.NoFrame)
        obj.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        obj.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        obj.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        return

    if isinstance(obj, QFormLayout):
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setHorizontalSpacing(8)
        obj.setVerticalSpacing(10)
        obj.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        obj.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        obj.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        return

    if isinstance(obj, QBoxLayout):
        obj.setContentsMargins(0, 0, 0, 0)
        obj.setSpacing(8)
        return

    if not isinstance(obj, QWidget):
        return

    obj.setMinimumWidth(0)

    if isinstance(obj, QCheckBox):
        obj.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )
        return

    if isinstance(obj, QComboBox):
        obj.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        obj.setMinimumContentsLength(minimum_contents_length)
        obj.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        return

    if isinstance(obj, QLineEdit):
        obj.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        return

    if isinstance(obj, QPlainTextEdit):
        obj.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        return

    obj.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Preferred,
    )


def stdPlainTextEdit(
    editor: QPlainTextEdit,
    *,
    visible_lines: int = 5,
    word_wrap: bool = True,
) -> None:
    """Apply standard sizing and wrapping for compact multiline editors."""
    editor.setMinimumWidth(0)

    if word_wrap:
        editor.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth
        )
        editor.setWordWrapMode(
            QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
        )
        editor.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
    else:
        editor.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.NoWrap
        )
        editor.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

    editor.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Fixed,
    )

    editor.setFixedHeight(
        editor.fontMetrics().lineSpacing() * visible_lines + 12
    )


# ----------------------------------------------------------------------
# The action catalogue
# ----------------------------------------------------------------------
# config.json is the only source of an action's icon, label, tooltip and
# shortcut.  Read once into ActionSpec and cached, because every menu and
# every button asks for one while the window is being built.
@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Everything the UI needs to present one action."""

    action_id: str
    icon: str | None
    text: str
    description: str
    shortcut: ShortcutLike = None
    checkable: bool = False
    sf_symbol: str | None = None
    segoe_fluent: str | None = None

    @classmethod
    def from_config(cls, action_id: str, entry: dict[str, Any]) -> "ActionSpec":
        return cls(
            action_id=action_id,
            icon=(entry.get("icon") or None),
            text=str(entry.get("text") or action_id),
            description=str(entry.get("description") or ""),
            shortcut=(entry.get("shortcut") or None),
            sf_symbol=(entry.get("SFSymbol") or entry.get("sf_symbol") or None),
            checkable=bool(entry.get("checkable", False)),
            segoe_fluent=(entry.get("SegoeFluent") or entry.get("segoe_fluent") or None),
        )

    def translated_text(self) -> str:
        return tr(self.text)

    def translated_description(self) -> str:
        return tr(self.description) if self.description else ""

_cache: dict[str, ActionSpec] | None = None
# The section the cache was built from.  load_config returns the same object
# until config.json changes on disk, so comparing identity is enough to notice
# an edit without re-parsing on every button.
_cache_source: object | None = None


def _actions_section() -> dict[str, Any]:
    """Return the ``actions`` object from config.json, or an empty one.

    Not ``load_config()[...]``: a config.json without the section would raise
    while the first menu is being built, taking the window with it.
    """
    return get_section("actions")


def _catalog() -> dict[str, ActionSpec]:
    global _cache, _cache_source
    section = _actions_section()
    if _cache is not None and _cache_source is section:
        return _cache

    _cache_source = section
    _cache = {
        action_id: ActionSpec.from_config(action_id, entry)
        for action_id, entry in section.items()
        if isinstance(entry, dict)
    }
    return _cache


def reload_actions() -> None:
    """Forget the cached catalogue; the next lookup re-reads config.json."""
    global _cache, _cache_source
    _cache = None
    _cache_source = None


def action(action_id: str) -> ActionSpec:
    key = str(action_id or "").strip()
    spec = _catalog().get(key)
    if spec is not None:
        return spec

    applogger.warning(
        "No action %r in config.json; the widget will have no label or icon.",
        action_id,
        show_dialog=False,
        raise_error=False,
    )
    # Empty, not a placeholder labelled with the id: a button reading
    # "series_outliers" looks like a working button with an odd name, and the
    # gap goes unnoticed.  A blank one does not.
    return ActionSpec(key, None, "", "")


def action_menu_item(
    action_id: str,
    callback,
    *,
    checkable: bool | None = None,
    checked: bool = False,
    enabled: bool = True,
    shortcut: ShortcutLike | Ellipsis = ...,  # type: ignore[valid-type]
) -> MenuItem:
    spec = action(action_id)
    return MenuItem(
        text=spec.translated_text(),
        tooltip=spec.translated_description(),
        shortcut=spec.shortcut if shortcut is ... else shortcut,
        callback=callback,
        checkable=spec.checkable if checkable is None else checkable,
        # Pass the action id to style.load_icon(); icon selection remains fully
        # catalogue-driven.
        icon=spec.action_id,
        action_id=spec.action_id,
        checked=checked,
        enabled=enabled,
    )