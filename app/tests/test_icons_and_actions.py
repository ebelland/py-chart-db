"""Tests for icon resolution, the action catalogue, and translation.

The catalogue lives in ``config.json`` and is the only copy: this module holds
no defaults.  An action names up to three presentations - an SF Symbol for
macOS, a Segoe Fluent glyph for Windows, and an SVG that works anywhere - and
``load_icon`` picks per platform, falling back down the list.

The tests below pin what cannot be seen by looking at the running app: that a
named icon has a file, that no icon file is shipped for nobody, and that the
catalogue survives a config.json that is missing or half-edited.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QPalette, QPixmap

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
# Layout of the icon folders
# ----------------------------------------------------------------------
def test_the_three_icon_folders_exist() -> None:
    for name in ("common", "win11", "macOs"):
        assert (ICONS_DIR / name).is_dir(), f"missing icon folder {name}"


def test_no_icons_are_left_at_the_top_level() -> None:
    assert list(ICONS_DIR.glob("*.svg")) == []


def test_platform_prefixes_are_gone() -> None:
    """win_x / mac_x became x inside their own folder."""
    stray = [
        path.name
        for path in ICONS_DIR.rglob("*.svg")
        if path.stem.startswith(("win_", "mac_"))
    ]
    assert stray == []


def test_every_platform_icon_has_a_common_fallback() -> None:
    """A platform with no folder of its own must still get every icon."""
    common = {path.name for path in (ICONS_DIR / "common").glob("*.svg")}
    for folder in ("win11", "macOs"):
        for path in (ICONS_DIR / folder).glob("*.svg"):
            assert path.name in common, f"{folder}/{path.name} has no common fallback"


# ----------------------------------------------------------------------
# Nothing unused, nothing dangling
# ----------------------------------------------------------------------
# How an SVG id reaches the resolver.  An action reaches it through its "icon"
# field in config.json; anything else names the file directly in code.
_ICON_REFERENCE_PATTERNS = [
    re.compile(r"""icon\s*=\s*["']([\w ]+)["']"""),
    re.compile(r"""icon_name\s*=\s*["']([\w ]+)["']"""),
    re.compile(r"""load_icon\(\s*["']([\w ]+)["']"""),
    re.compile(r"""create_toolbar_button\(\s*[^,]+,\s*["']([\w ]+)["']"""),
]


def _referenced_icon_names() -> set[str]:
    """Return every SVG id the application can ask for.

    Two kinds of reference, kept apart on purpose.  ``from_config`` are SVG ids
    an action names in config.json.  ``from_code`` are tokens written at a call
    site, most of which are *action ids* rather than file names - and an action
    id that happens to match a file name (``copy``, ``delete``) must not count
    as a reference to that file, or the check would pass for icons nothing can
    reach.  Only the second set is filtered.
    """
    from_config = {
        str(entry["icon"]).strip().lower()
        for entry in get_section("actions").values()
        if isinstance(entry, dict) and entry.get("icon")
    }
    names: set[str] = set()

    for path in APP_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in _ICON_REFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                names.add(Path(match.group(1)).stem.strip().lower())

    for path in APP_DIR.rglob("*.qss"):
        for match in re.finditer(r"([\w.\-]+)\.svg", path.read_text(encoding="utf-8")):
            names.add(match.group(1).rsplit("/", 1)[-1].lower())

    names -= set(get_section("actions"))
    referenced = {name for name in names | from_config if name}

    # A "<name>_on_dark" file is the dark-theme variant of "<name>", reached by
    # _icon_url building the name at runtime rather than naming it in source.
    # Without this the variants read as unused and the check would push someone
    # to delete the chevrons the dark theme needs.
    referenced |= {f"{name}_on_dark" for name in referenced}
    return referenced


def _icon_stems() -> set[str]:
    return {path.stem.lower() for path in ICONS_DIR.rglob("*.svg")}


def test_no_icon_file_is_unused() -> None:
    """An icon nobody asks for is up to three files to ship and none to see."""
    unused = sorted(_icon_stems() - _referenced_icon_names())
    assert unused == [], f"these icons are never requested: {unused}"


def test_every_configured_svg_icon_has_a_file() -> None:
    """A name with no file is a blank button and a line in the log."""
    missing = [
        f"{action_id} -> {entry['icon']}"
        for action_id, entry in get_section("actions").items()
        if isinstance(entry, dict)
        and entry.get("icon")
        and style.resolve_icon_path(str(entry["icon"])) is None
    ]
    assert missing == []


def test_every_action_has_some_presentation() -> None:
    """An action with no icon, no glyph and no symbol is an empty button."""
    naked = [
        action_id
        for action_id, entry in get_section("actions").items()
        if isinstance(entry, dict)
        and not entry.get("icon")
        and not entry.get("SFSymbol")
        and not entry.get("SegoeFluent")
    ]
    assert naked == []


# ----------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------
def test_a_name_resolves_to_a_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(style.platform, "system", lambda: "Linux")
    assert style.get_icon_file_name("clustering") == "common/clustering.svg"


def test_the_platform_folder_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(style.platform, "system", lambda: "Windows")
    assert style.get_icon_file_name("new") == "win11/new.svg"

    monkeypatch.setattr(style.platform, "system", lambda: "Darwin")
    assert style.get_icon_file_name("new") == "macOs/new.svg"


def test_common_is_used_when_the_platform_has_no_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(style.platform, "system", lambda: "Darwin")
    # There is no macOs/clustering.svg; the shared one is correct.
    assert style.get_icon_file_name("clustering") == "common/clustering.svg"


def test_an_unknown_name_resolves_to_nothing() -> None:
    assert style.resolve_icon_path("definitely_not_an_icon") is None
    assert style.resolve_icon_path("") is None
    assert style.resolve_icon_path(None) is None


def test_resolution_is_case_insensitive() -> None:
    assert style.resolve_icon_path("Clustering") == style.resolve_icon_path("clustering")


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
    assert spec.icon is None


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


def test_a_symbol_that_cannot_be_drawn_falls_back_to_the_svg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback is the whole reason every action also names an SVG."""
    monkeypatch.setattr(style.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(style, "_IS_MACOS", True)
    monkeypatch.setattr(style, "_create_sf_symbol_icon", lambda *_a, **_k: style.QIcon())

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
