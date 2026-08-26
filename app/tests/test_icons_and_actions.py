"""Tests for icon resolution, the action catalogue, and translation.

The catalogue lives in ``config.json`` and is the only copy: this module holds
no defaults.  An action names up to four presentations - an SF Symbol for
macOS, a Segoe Fluent glyph for Windows, a freedesktop theme name for
everywhere else, and an SVG as the last resort - and ``load_icon`` picks per
platform, falling back down the list.

The theme name is the one that covers Linux, where neither of the platform
glyph sets exists and where the desktop already has an icon theme the user
chose.  Before it existed, every Linux install fell straight to the SVGs, and
three actions that name no SVG at all showed as blank buttons.

The SVGs themselves are down to one folder.  There were three - a "macOs" and
a "win11" searched ahead of "common" - from when a hand-drawn icon was the
only way to look like the platform it was drawn for.  Three real icon sets
answer ahead of any SVG now, so those folders were reached only when a Mac had
no pyobjc or a PC had no Fluent font, and what they gave there was a second
drawing of the same thing.

The tests below pin what cannot be seen by looking at the running app: that a
named icon has a file, that no icon file is shipped for nobody, and that the
catalogue survives a config.json that is missing or half-edited.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPixmap

from app.styles import style
from app.styles.style import ActionSpec, action, action_menu_item
from app.utils import i18n
from app.utils.config import get_section

ICONS_DIR = Path(style.__file__).resolve().parent.parent / "icons"
APP_DIR = Path(style.__file__).resolve().parent.parent


@pytest.fixture
def actions() -> dict[str, ActionSpec]:
    """The catalogue as config.json defines it."""
    return {
        action_id: action(action_id)
        for action_id, entry in get_section("actions").items()
        if isinstance(entry, dict)
    }


# ----------------------------------------------------------------------
# No shipped icon files at all
# ----------------------------------------------------------------------
def test_nothing_ships_an_icon_file() -> None:
    """There were three folders and 67 SVGs; there are none.

    The chain reads SF Symbols on macOS, Segoe Fluent on Windows, the
    desktop's icon theme everywhere else - three real icon sets, each the
    platform's own. What sat underneath them was a drawing made once, by
    hand, matching no system in particular, and reached only by a Mac with no
    pyobjc, a PC with no Fluent font or a Linux box with no theme installed.

    Those three cases now show no icon rather than a hand-drawn one. That is
    the decision this test pins: a fallback nobody sees is a set of files
    everybody maintains.
    """
    assert not APP_DIR.joinpath("icons").exists()
    assert list(APP_DIR.rglob("*.svg")) == []


def test_no_stylesheet_asks_for_an_icon_file() -> None:
    """A Qt stylesheet takes url(file) and cannot be handed a themed QIcon.

    That is what kept four chevrons - up, down, and a light-ink copy of each -
    in the repository after every other icon had been replaced. The rules that
    named them are gone, and Qt draws the platform's own arrow for a combo box
    and a spin box when no image is given.
    """
    asking = [
        str(path.relative_to(APP_DIR))
        for path in APP_DIR.rglob("*.qss")
        if ".svg" in path.read_text(encoding="utf-8")
    ]

    assert asking == []


def test_no_code_still_reaches_for_the_file_lookup() -> None:
    """resolve_icon_path, get_icon_file_name and _icon_url are gone with it."""
    for name in ("resolve_icon_path", "get_icon_file_name", "_icon_url", "_svg_icon"):
        assert not hasattr(style, name), f"{name} outlived the files it read"


# ----------------------------------------------------------------------
# The catalogue still names an icon for everything
# ----------------------------------------------------------------------
def test_every_action_names_all_three_icon_sets() -> None:
    """One name per backend, because each covers a platform the others do not.

    An action missing SFSymbol is blank on macOS, missing SegoeFluent is blank
    on Windows, and missing ThemeIcon is blank everywhere else. There is no
    shipped SVG underneath any more to hide the gap, so the catalogue has to
    be complete rather than nearly complete.
    """
    incomplete = {
        action_id: [
            field
            for field in ("SFSymbol", "SegoeFluent", "ThemeIcon")
            if not entry.get(field)
        ]
        for action_id, entry in get_section("actions").items()
        if isinstance(entry, dict)
        and not all(entry.get(field) for field in ("SFSymbol", "SegoeFluent", "ThemeIcon"))
    }

    assert incomplete == {}


# ----------------------------------------------------------------------
# The action catalogue
# ----------------------------------------------------------------------
def test_every_action_has_text_and_a_description(actions) -> None:
    for action_id, spec in actions.items():
        assert spec.text.strip(), f"{action_id} has no label"
        assert spec.description.strip(), f"{action_id} has no description"


def test_the_id_matches_the_key(actions) -> None:
    for key, spec in actions.items():
        assert spec.action_id == key


def test_an_unknown_action_degrades_to_a_placeholder() -> None:
    """A typo must not take the menu down with it."""
    spec = action("no_such_action")
    assert spec.action_id == "no_such_action"
    assert spec.text == ""
    assert spec.theme_icon is None


def test_a_missing_catalogue_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """config.json is the only copy, so an absent section must degrade."""
    from app.styles import style as actions_module

    monkeypatch.setattr(actions_module, "get_section", lambda *_a, **_k: {})
    actions_module.reload_actions()
    try:
        spec = actions_module.action("new")
        assert spec.action_id == "new"
    finally:
        actions_module.reload_actions()


@pytest.fixture(autouse=True)
def _fresh_action_catalogue():
    """Drop the cached catalogue around every test in this module.

    ``_catalog`` holds the parsed actions for the life of the process, which is
    right for the app and wrong for tests: one test that swaps config.json for
    an empty section used to leave every later test reading that empty one.
    """
    style.reload_actions()
    yield
    style.reload_actions()


def test_style_also_survives_a_missing_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same lookup happens again in style.load_icon, on every button."""
    import app.utils.config as config_module

    monkeypatch.setattr(config_module, "load_config", lambda: {})
    assert style._actions_section() == {}
    assert style.action("new").text == ""


def test_a_menu_item_carries_the_catalogue_values() -> None:
    # A catalogue action, deliberately: the series operations used to be in
    # here and are not any more, since a plugin carries its own presentation.
    item = action_menu_item("copy", callback=lambda: None)

    assert item.text
    assert item.action_id == "copy"


def test_a_shortcut_can_be_overridden_or_suppressed() -> None:
    assert action_menu_item("new", lambda: None, shortcut=None).shortcut is None
    assert action_menu_item("new", lambda: None, shortcut="F2").shortcut == "F2"


# ----------------------------------------------------------------------
# Translation
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restore_language():
    """Every test starts and ends in English."""
    previous = i18n.language()
    i18n.set_language(i18n.DEFAULT_LANGUAGE)
    yield
    i18n.set_language(previous)


def test_the_source_string_is_returned_when_untranslated() -> None:
    assert i18n.tr("Nothing will match this") == "Nothing will match this"


def test_menu_items_follow_the_active_language() -> None:
    """Switching language is a matter of rebuilding the menus."""
    english = action_menu_item("open", lambda: None)
    i18n.set_language("it")
    italian = action_menu_item("open", lambda: None)

    assert english.text == "Open"
    assert italian.text == "Apri"


def test_an_unknown_language_is_refused_and_the_current_one_kept() -> None:
    assert i18n.set_language("kl") == i18n.DEFAULT_LANGUAGE
    assert i18n.language() == i18n.DEFAULT_LANGUAGE


def test_english_is_always_available_and_listed_first() -> None:
    languages = i18n.available_languages()
    assert languages[0] == i18n.DEFAULT_LANGUAGE
    assert "it" in languages


def test_the_italian_locale_covers_every_action(actions) -> None:
    """A partial locale would show a mix of two languages in one menu."""
    catalog = i18n._parse_po(
        i18n.LOCALES_DIR / "it" / "LC_MESSAGES" / f"{i18n.DOMAIN}.po"
    )
    missing = [
        text
        for spec in actions.values()
        for text in (spec.text, spec.description)
        if text and text not in catalog
    ]
    assert missing == []


def test_the_italian_locale_covers_every_operation() -> None:
    """The operations' wording is not in config.json and needs its own sweep.

    ``test_the_italian_locale_covers_every_action`` walks the action catalogue,
    which the series operations left when they became self-contained; without
    this their Name and Description would silently stop being translated.
    """
    from app.scanners.series_operation_scanner import series_operations

    catalog = i18n._parse_po(
        i18n.LOCALES_DIR / "it" / "LC_MESSAGES" / f"{i18n.DOMAIN}.po"
    )
    missing = [
        text
        for operation in series_operations
        for text in (operation["value"], operation.get("description"))
        if text and text not in catalog
    ]
    assert missing == []


def test_every_operation_icon_actually_draws(qapp) -> None:
    """A plugin's icon is markup, so it can be wrong in ways a path cannot.

    An unparseable document, or one whose paths all fall outside the viewBox,
    yields a perfectly valid QIcon that paints nothing - and a blank button is
    exactly the failure this whole move was meant to make impossible.  So the
    check is on pixels, not on the QIcon being non-null.
    """
    from app.scanners.series_operation_scanner import series_operations

    blank = []
    for operation in series_operations:
        image = style.icon_from_svg_source(operation.get("icon")).pixmap(
            QSize(20, 20)
        ).toImage()
        drawn = any(
            image.pixelColor(x, y).alpha() > 40
            for y in range(image.height())
            for x in range(image.width())
        )
        if not drawn:
            blank.append(operation["value"])

    assert blank == []


def test_a_broken_embedded_icon_is_reported_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, qapp
) -> None:
    """Markup that does not parse must leave a trace, like a missing symbol."""
    warnings: list[str] = []
    monkeypatch.setattr(style, "_SVG_SOURCE_ICON_CACHE", {})
    monkeypatch.setattr(
        style.applogger,
        "warning",
        lambda message, *args, **kwargs: warnings.append(
            str(message) % args if args else str(message)
        ),
    )

    assert style.icon_from_svg_source("<svg><this is not markup").isNull()
    assert len(warnings) == 1

    # Nothing to report for an operation that simply has no icon.
    assert style.icon_from_svg_source("").isNull()
    assert style.icon_from_svg_source(None).isNull()
    assert len(warnings) == 1


def test_a_partial_translation_falls_back_per_string() -> None:
    """A missing message renders as its English source, never as an id."""
    i18n.set_language("it")
    assert i18n.tr("This string does not exist") == "This string does not exist"


# ----------------------------------------------------------------------
# SF Symbols
# ----------------------------------------------------------------------
def test_a_missing_bridge_is_reported_once_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole icon set fell back to SVG with nothing to say why.

    ``_create_sf_symbol_icon`` caught every exception and returned an empty
    QIcon, so a missing pyobjc, a typo in a symbol name and a genuine AppKit
    failure all looked identical: no icon, no log line, nothing to search for.
    """
    warnings: list[str] = []
    # Make the import itself fail, which is the thing being simulated.  Only
    # clearing the memo does not: it just forces the lookup to run again, and
    # on a machine that has pyobjc the second attempt succeeds.  That is why
    # this passed for as long as the dependency was missing and started
    # failing the day it was installed.  A None in sys.modules makes ``import
    # AppKit`` raise ImportError without touching the real module.
    monkeypatch.setitem(sys.modules, "AppKit", None)
    monkeypatch.setattr(style, "_SF_SYMBOL_BRIDGE", None)
    monkeypatch.setattr(style, "_SF_SYMBOL_BRIDGE_CHECKED", False)
    monkeypatch.setattr(
        style.applogger,
        "warning",
        lambda message, *args, **kwargs: warnings.append(str(message) % args if args else str(message)),
    )

    assert style._sf_symbol_bridge() is None
    assert len(warnings) == 1
    assert "pyobjc" in warnings[0]

    # Looked up once, not once per button.
    assert style._sf_symbol_bridge() is None
    assert len(warnings) == 1


def test_the_symbol_cache_remembers_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A symbol that failed once fails every time; retrying costs a menu."""
    monkeypatch.setattr(style, "_IS_MACOS", True)
    monkeypatch.setattr(style, "_SF_SYMBOL_ICON_CACHE", {})
    monkeypatch.setattr(style, "_SF_SYMBOL_BRIDGE_CHECKED", True)
    monkeypatch.setattr(style, "_SF_SYMBOL_BRIDGE", None)

    icon = style._create_sf_symbol_icon("chevron.up")

    assert icon.isNull()


def test_a_symbol_that_cannot_be_drawn_falls_through_to_the_theme(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Mac with no pyobjc used to land on a shipped SVG. It lands on the
    icon theme now, and on nothing at all where there is no theme - which is
    the trade this made deliberately."""
    from PySide6.QtGui import QIcon

    monkeypatch.setattr(style.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(style, "_IS_MACOS", True)
    monkeypatch.setattr(style, "_create_sf_symbol_icon", lambda *_a, **_k: style.QIcon())

    if not style._installed_icon_themes():
        pytest.skip("no icon theme installed on this machine")

    style.ensure_icon_theme()
    style._THEME_ICON_CACHE.clear()
    assert not style.load_icon("new").isNull()


def test_pyobjc_is_declared_for_macos() -> None:
    """An undeclared dependency is why the symbols were never drawn."""
    requirements = (APP_DIR.parent / "requirements.txt").read_text(encoding="utf-8")

    assert "pyobjc" in requirements
    assert 'sys_platform == "darwin"' in requirements


def test_an_unsafe_pyobjc_is_refused_before_it_is_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The crash this guards against happens inside the import, in C.

    pyobjc-core 11.1 published a wheel tagged cp314 while Python 3.14 was in
    beta.  pip installs it on a released 3.14 without complaint and the
    interpreter then dies on ``import AppKit`` - a native crash, so there is no
    exception to catch and no log line.  The version has to be read from the
    metadata and rejected before any of it is loaded.
    """
    warnings: list[str] = []
    monkeypatch.setattr(style.sys, "version_info", (3, 14, 5, "final", 0))
    monkeypatch.setattr(
        style.applogger,
        "warning",
        lambda message, *args, **kwargs: warnings.append(str(message) % args if args else str(message)),
    )
    monkeypatch.setattr(style, "_SF_SYMBOL_BRIDGE", None)
    monkeypatch.setattr(style, "_SF_SYMBOL_BRIDGE_CHECKED", False)

    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "11.1")

    assert style._pyobjc_core_is_safe_to_import() is False
    assert style._sf_symbol_bridge() is None
    assert "11.1" in warnings[0] and "12.0" in warnings[0]


def test_a_new_enough_pyobjc_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(style.sys, "version_info", (3, 14, 5, "final", 0))
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "12.2.2")

    assert style._pyobjc_core_is_safe_to_import() is True


def test_an_unknown_version_is_not_second_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The table catches one known-bad build; it is not a dependency policy."""
    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError()),
    )

    assert style._pyobjc_core_is_safe_to_import() is True


def test_the_appkit_import_was_not_left_commented_out() -> None:
    """It was commented out by hand to stop the crash; the guard replaces that."""
    source = (APP_DIR / "styles" / "style.py").read_text(encoding="utf-8")

    assert "#import AppKit" not in source
    assert "import AppKit" in source
    assert "if not _pyobjc_core_is_safe_to_import():" in source


def test_requirements_exclude_the_crashing_build() -> None:
    requirements = (APP_DIR.parent / "requirements.txt").read_text(encoding="utf-8")

    assert "pyobjc-framework-Cocoa>=12.2" in requirements


# ----------------------------------------------------------------------
# Sharpness and colour
# ----------------------------------------------------------------------
def _method_source(name: str) -> str:
    source = (APP_DIR / "styles" / "style.py").read_text(encoding="utf-8")
    start = source.index(f"def {name}")
    return source[start : source.index("\ndef ", start + 10)]


class _FakeScreen:
    """Stands in for QScreen where only the pixel ratio matters."""

    def __init__(self, ratio: float) -> None:
        self._ratio = ratio

    def devicePixelRatio(self) -> float:
        return self._ratio


def test_a_non_square_source_is_centred_on_a_square_canvas(qapp) -> None:
    """The bug: a trash can is narrower than it is tall. scaled(...,
    KeepAspectRatio) alone returns a pixmap sized to *that* glyph, not to the
    square slot every icon in a menu is supposed to share - so a document icon
    sat flush against its label while a narrower one left a visible gap, in
    the same menu. Every icon must report the same footprint regardless of
    what shape its own glyph happens to be.
    """
    narrow = QPixmap(8, 20)
    narrow.fill(Qt.GlobalColor.transparent)

    pixmap = style._tinted_pixmap(narrow, 20, "#000000", ratio=1.0)

    assert (pixmap.width(), pixmap.height()) == (20, 20)


def test_the_non_square_glyph_lands_centred_not_flush(qapp) -> None:
    """Painted flush at (0, 0), a narrower glyph would still be off-centre
    even inside a square pixmap - centring is a placement, not just a size."""
    narrow = QPixmap(8, 20)
    narrow.fill(Qt.GlobalColor.transparent)
    painter = QPainter(narrow)
    painter.fillRect(0, 0, 8, 20, QColor("black"))
    painter.end()

    pixmap = style._tinted_pixmap(narrow, 20, "#000000", ratio=1.0)
    image = pixmap.toImage()

    opaque_columns = [
        x for x in range(image.width())
        if any(image.pixelColor(x, y).alpha() > 20 for y in range(image.height()))
    ]
    assert opaque_columns, "the glyph must still be drawn somewhere"
    left_margin = opaque_columns[0]
    right_margin = image.width() - 1 - opaque_columns[-1]
    assert abs(left_margin - right_margin) <= 1


def test_icons_are_rasterised_at_the_display_s_pixel_density(qapp) -> None:
    """Soft icons next to sharp SVGs: the bitmaps were built at logical size.

    A 20 px bitmap has to be stretched across 40 physical pixels on a Retina
    screen, and that interpolation is what "blurry" was.  The SVGs never had
    the problem because they are drawn as vectors at whatever size is asked
    for, which is why only these two icon paths looked wrong.
    """
    source = QPixmap(8, 8)

    pixmap = style._tinted_pixmap(source, 20, "#000000", ratio=2.0)

    assert (pixmap.width(), pixmap.height()) == (40, 40)
    assert pixmap.devicePixelRatio() == 2.0
    # What the widget lays out is still 20 logical pixels.
    assert pixmap.width() / pixmap.devicePixelRatio() == 20


def test_a_non_retina_display_is_not_upscaled(qapp) -> None:
    pixmap = style._tinted_pixmap(QPixmap(8, 8), 20, "#000000", ratio=1.0)

    assert (pixmap.width(), pixmap.height()) == (20, 20)
    assert pixmap.devicePixelRatio() == 1.0


def test_a_pixmap_already_at_the_right_size_is_not_resampled(qapp) -> None:
    """Scaling 40 px to 40 px still runs a filter over every pixel."""
    body = _method_source("_tinted_pixmap")

    assert "pixmap.width() == physical" in body


def test_the_ratio_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past 3x the bitmap grows faster than anyone can see the difference."""
    monkeypatch.setattr(
        style.QGuiApplication, "primaryScreen", lambda: _FakeScreen(8.0)
    )
    assert style._icon_device_pixel_ratio() == 3.0

    monkeypatch.setattr(
        style.QGuiApplication, "primaryScreen", lambda: _FakeScreen(0.5)
    )
    assert style._icon_device_pixel_ratio() == 1.0


def test_icons_built_before_the_app_exists_still_assume_retina(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some icons are built while the application is still starting."""
    monkeypatch.setattr(style.QGuiApplication, "primaryScreen", lambda: None)
    monkeypatch.setattr(style, "_IS_MACOS", True)

    assert style._icon_device_pixel_ratio() == 2.0


def test_symbols_are_not_painted_in_the_windows_grey(qapp) -> None:
    """#404040 read as faded, as if every button on the toolbar were disabled."""
    tint = style._symbol_tint()

    assert tint != style._FLUENT_DEFAULT_COLOR
    assert QColor(tint).isValid()


def test_the_tint_follows_the_applied_theme_not_the_palette(qapp) -> None:
    """A hard-coded black icon is invisible on a dark toolbar - and a white one
    is invisible on a light toolbar, which is the failure this replaced.

    The tint used to be read from the palette's ButtonText.  That was wrong in
    the other direction: a Qt style plugin installs its own palette, and on a
    Mac whose system appearance is dark it comes back light-on-dark even while
    the application is wearing a light theme, so the glyphs went white on white
    buttons.  The applied theme is the only thing that actually knows.
    """
    original = qapp.palette()
    try:
        for key in ("automatic", "fluent_win11", "macos_native", "qt:Fusion"):
            style.apply_platform_style(qapp, key)
            assert QColor(style._symbol_tint()) == QColor(style.SYMBOL_INK_DARK), key

        style.apply_platform_style(qapp, "dark")
        assert QColor(style._symbol_tint()) == QColor(style.SYMBOL_INK_LIGHT)

        # A palette pushed in from outside must not move the ink.
        style.apply_platform_style(qapp, "automatic")
        pushed = QPalette(qapp.palette())
        pushed.setColor(
            QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText, QColor("#ffffff")
        )
        qapp.setPalette(pushed)
        assert QColor(style._symbol_tint()) == QColor(style.SYMBOL_INK_DARK)
    finally:
        qapp.setPalette(original)


def test_the_ratio_comes_from_the_screen_not_the_application(qapp) -> None:
    """QCoreApplication has no such attribute; the ratio belongs to a display.

    It resolved at runtime, so the tests passed and the type checker was the
    only thing that noticed.
    """
    source = (APP_DIR / "styles" / "style.py").read_text(encoding="utf-8")

    assert "QGuiApplication.primaryScreen()" in source
    assert "QApplication.instance().devicePixelRatio" not in source


def test_the_cache_keys_are_typed_for_what_they_hold() -> None:
    """The ratio was added to both keys without the annotations following."""
    source = (APP_DIR / "styles" / "style.py").read_text(encoding="utf-8")

    assert "_FLUENT_ICON_CACHE: dict[tuple[str, int, str, float], QIcon]" in source
    assert "_SF_SYMBOL_ICON_CACHE: dict[tuple[str, int, str, float], QIcon]" in source


def test_the_symbol_cache_keeps_the_ratio() -> None:
    """Otherwise a 1x icon built at startup is reused on a 2x screen."""
    assert "round(ratio, 2)" in _method_source("_create_sf_symbol_icon")


def test_both_glyph_caches_key_on_the_tint() -> None:
    """A cached icon must not survive a change of theme.

    The Fluent key carried the colour and the SF Symbol key did not, so on
    macOS the symbols built under the light palette were handed straight back
    after switching to the dark theme: dark ink on a dark button, and cached,
    so it stayed wrong until the app was restarted.

    Asserted on the source rather than by building icons, because neither
    backend exists everywhere - the Segoe Fluent font is Windows-only and the
    SF Symbol bridge is macOS-only, so on any one machine at least one of the
    two caches stays empty and a behavioural test would pass by vacancy.
    """
    assert "tint, round(ratio, 2)" in _method_source("_create_sf_symbol_icon")
    assert "color_name, round(ratio, 2)" in _method_source("_create_fluent_icon")


def test_the_glyph_path_is_tinted_from_the_same_source() -> None:
    """The two backends were tinted from different places.

    The Segoe glyphs took a hard-coded grey while the SF Symbols followed the
    palette, so a dark theme lightened the icons on macOS and left them
    invisible on Windows.
    """
    source = (APP_DIR / "styles" / "style.py").read_text(encoding="utf-8")

    assert "_create_fluent_icon(spec.segoe_fluent, color=_symbol_tint())" in source
    assert "_create_fluent_icon(token, color=_symbol_tint())" in source


def test_the_glyph_cache_keeps_the_ratio() -> None:
    assert "round(ratio, 2)" in _method_source("_create_fluent_icon")


def test_the_symbol_is_asked_for_at_the_device_size() -> None:
    """Rasterising at 20 pt and enlarging afterwards is the blur itself."""
    body = _method_source("_create_sf_symbol_icon")

    assert "float(size) * ratio" in body


def test_every_hand_painted_icon_goes_through_the_shared_helper(qapp) -> None:
    """The combo swatches were soft for exactly the same reason the symbols were.

    Colour swatches, line-style previews and marker previews are all painted by
    hand, and all three allocated their bitmap at the logical size.  Nothing
    stops a fourth from doing the same except this test.
    """
    hand_painted = [
        APP_DIR / "widgets" / "color_combo.py",
        APP_DIR / "widgets" / "line_combo.py",
        APP_DIR / "widgets" / "marker_combo.py",
    ]

    for path in hand_painted:
        source = path.read_text(encoding="utf-8")
        assert "create_hidpi_pixmap(" in source, path.name
        assert "QPixmap(" not in source, f"{path.name} allocates a bitmap at logical size"


def test_the_helper_allocates_physical_pixels_and_labels_them(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(style, "_icon_device_pixel_ratio", lambda: 2.0)

    pixmap = style.create_hidpi_pixmap(16, 12)

    assert (pixmap.width(), pixmap.height()) == (32, 24)
    assert pixmap.devicePixelRatio() == 2.0
    # Painting code keeps its logical coordinates.
    assert pixmap.deviceIndependentSize().toSize() == QSize(16, 12)


def test_the_helper_starts_transparent(qapp) -> None:
    """A swatch drawn over an opaque bitmap has square black corners."""
    pixmap = style.create_hidpi_pixmap(8, 8)

    assert pixmap.toImage().pixelColor(0, 0).alpha() == 0


# ----------------------------------------------------------------------
# Theme icons: the backend that covers Linux
# ----------------------------------------------------------------------
def test_every_action_names_a_theme_icon(actions) -> None:
    """It is the only backend left on Linux, and there is nothing beneath it."""
    missing = sorted(
        action_id
        for action_id, spec in actions.items()
        if not style._theme_icon_names(spec.theme_icon)
    )

    assert missing == []


def test_an_alternative_theme_name_is_allowed_and_ordered(actions) -> None:
    """No single freedesktop name is in every theme: "office-chart-line" is
    Breeze's and Papirus's, and GNOME ships no chart icon at all. A list is
    tried best-first, so Breeze gets the chart and Adwaita the fallback."""
    assert style._theme_icon_names("one") == ("one",)
    assert style._theme_icon_names(["one", "two"]) == ("one", "two")
    assert style._theme_icon_names(None) == ()

    assert style._theme_icon_names(actions["new_plot"].theme_icon)[0] == "office-chart-line"


def test_no_theme_name_is_invented(actions) -> None:
    """A typo here is silent: fromTheme returns a null icon and the SVG takes
    over, so the only symptom is one icon that never modernises.

    Every name must be either one Qt standardises or one listed in
    EXTRA_THEME_ICON_NAMES, which is the deliberate-decision list.
    """
    known = style.known_theme_icon_names()
    unknown = sorted(
        {
            name
            for spec in actions.values()
            for name in style._theme_icon_names(spec.theme_icon)
            if name not in known
        }
    )

    assert unknown == []


def test_the_standard_names_come_from_qt_rather_than_a_copy() -> None:
    """The list grows with Qt; a hand-written one would not."""
    names = style._standard_theme_icon_names()

    assert "document-open" in names
    assert "edit-copy" in names
    assert "zoom-fit-best" in names
    assert "n-theme-icons" not in names, "that member is a count, not an icon"


def test_camel_case_becomes_the_freedesktop_spelling() -> None:
    assert style._camel_to_kebab("DocumentOpen") == "document-open"
    assert style._camel_to_kebab("ZoomFitBest") == "zoom-fit-best"
    assert style._camel_to_kebab("Computer") == "computer"


def test_every_extra_name_is_actually_used(actions) -> None:
    """The list is deliberate decisions, not a graveyard."""
    configured = {
        name
        for spec in actions.values()
        for name in style._theme_icon_names(spec.theme_icon)
    }
    unused = sorted(style.EXTRA_THEME_ICON_NAMES - configured)

    assert unused == []


def test_a_symbolic_variant_is_tried_when_the_plain_name_is_missing() -> None:
    """GNOME's Adwaita dropped the full-colour action icons at version 45, so
    on a current GNOME desktop ``document-open`` does not exist and
    ``document-open-symbolic`` is the icon.  Breeze still ships both."""
    assert style._theme_icon_candidates("document-open") == (
        "document-open",
        "document-open-symbolic",
    )


def test_a_name_that_is_already_symbolic_is_not_doubled() -> None:
    assert style._theme_icon_candidates("go-up-symbolic") == ("go-up-symbolic",)


def test_the_plain_name_is_preferred_over_the_symbolic_one() -> None:
    """Where a theme ships both, the plain one is its full-colour artwork."""
    plain, symbolic = style._theme_icon_candidates("edit-copy")

    assert plain == "edit-copy"
    assert symbolic.endswith("-symbolic")


def test_an_empty_name_asks_the_theme_for_nothing(qapp) -> None:
    assert style._create_theme_icon("").isNull()
    assert style._create_theme_icon(None).isNull()


def test_a_missing_theme_icon_now_draws_nothing(qapp) -> None:
    """It used to fall through to a shipped SVG. There is no longer one."""
    spec = ActionSpec(
        action_id="probe",
        text="Probe",
        description="",
        theme_icon="no-such-icon-anywhere",
    )

    assert style.icon_from_action_spec(spec).isNull()
    assert style.icon_source_for_action(spec) == "none"


def test_an_action_naming_no_icon_at_all_reports_none(qapp) -> None:
    spec = ActionSpec(
        action_id="probe",
        text="Probe",
        description="",
        theme_icon="",
    )

    assert style.icon_source_for_action(spec) == "none"


def test_the_report_accounts_for_every_action(qapp, actions) -> None:
    """It is the tool for retiring the SVGs, so it has to be complete."""
    report = style.report_icon_sources()
    counted = sum(len(ids) for ids in report.values())

    assert set(report) == set(style.ICON_SOURCES)
    assert counted == len(actions)


def test_the_report_agrees_with_what_is_actually_drawn(qapp, actions) -> None:
    """Two implementations of one priority order would drift."""
    for action_id, spec in actions.items():
        source = style.icon_source_for_action(spec)
        drawn = style.icon_from_action_spec(spec)

        assert drawn.isNull() == (source == "none"), action_id

