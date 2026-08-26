"""Localization on top of the standard library's :mod:`gettext`.

The catalogue layout is the GNU one that every translation tool expects::

    app/locales/<lang>/LC_MESSAGES/datahub.po     <- edited by translators
    app/locales/<lang>/LC_MESSAGES/datahub.mo     <- compiled, read at runtime

Call sites use the English string itself as the message id::

    from app.utils.i18n import tr
    tr("Delete chart")

which is what makes the source readable, lets ``xgettext``/``pybabel extract``
find every string without a key list, and guarantees that a missing
translation renders as English rather than as an identifier.

``.po`` files are compiled to ``.mo`` on load whenever the ``.po`` is newer,
so editing a translation is enough to see it - there is no build step to
forget.  The compiler is the standard ``.mo`` format writer (the same layout
CPython's ``Tools/i18n/msgfmt.py`` produces), kept here so the application has
no runtime dependency on gettext tooling being installed.
"""
from __future__ import annotations

import array
import gettext
import os
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.logs.logger import applogger

LOCALES_DIR: Path = Path(__file__).resolve().parent.parent / "locales"
DOMAIN: str = "datahub"
DEFAULT_LANGUAGE: str = "en"

#: Stored in config.json to mean "whatever this machine is set to".  A
#: sentinel rather than an empty string, because empty already means "the key
#: was never written" everywhere else in the config, and the two want
#: different answers: an unset key is a fresh install, which should follow the
#: platform, while "auto" is a choice the user made and can be shown back to
#: them in the combo as the choice they made.
AUTO_LANGUAGE: str = "auto"

# Set through set_language(); read on every lookup.  Always a real language
# code - AUTO_LANGUAGE is resolved on the way in, never stored here, so that
# every lookup is a dictionary hit rather than a locale query.
_language: str = DEFAULT_LANGUAGE


# ----------------------------------------------------------------------
# .po -> .mo
# ----------------------------------------------------------------------
def _parse_po(path: Path) -> dict[str, str]:
    """Return the ``msgid -> msgstr`` map of a .po file.

    Deliberately small: it understands comments, ``msgid``/``msgstr`` and the
    continuation strings that follow them, which is all this project's
    catalogues use.  Plural forms and contexts are not supported; if they are
    ever needed, compile with ``msgfmt`` instead and this reader is bypassed.
    """
    entries: dict[str, str] = {}
    key: str | None = None
    value: str = ""
    target: str | None = None

    def flush() -> None:
        if key:
            entries[key] = value

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("msgid "):
            flush()
            key, value, target = _po_string(line[6:]), "", "msgid"
        elif line.startswith("msgstr "):
            value, target = _po_string(line[7:]), "msgstr"
        elif line.startswith('"') and target is not None:
            # A continuation line appends to whichever field is open.
            if target == "msgid":
                key = (key or "") + _po_string(line)
            else:
                value += _po_string(line)

    flush()

    # The entry with the empty msgid is the header: it is metadata rather than
    # a translation, but it must survive into the .mo, because that is where
    # gettext reads the charset from.  Without it every lookup is decoded as
    # ASCII and the first accented character raises UnicodeDecodeError.
    entries.setdefault("", "Content-Type: text/plain; charset=UTF-8\n")
    return entries


def _po_string(token: str) -> str:
    """Unquote one ``"..."`` token from a .po file."""
    token = token.strip()
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        token = token[1:-1]
    return token.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def _write_mo(entries: dict[str, str], path: Path) -> None:
    """Write a GNU .mo file: two sorted string tables plus an offset index."""
    items = sorted((k.encode("utf-8"), v.encode("utf-8")) for k, v in entries.items())
    # Each string is stored NUL-terminated while its recorded length excludes
    # the NUL.  The terminator is not optional: gettext rejects a catalogue
    # whose last string ends exactly at end-of-file as corrupt.
    ids = b"".join(k + b"\x00" for k, _ in items)
    strs = b"".join(v + b"\x00" for _, v in items)

    # 7 * 4 bytes of header, then two (length, offset) tables of 8 bytes each.
    key_table_start = 7 * 4
    value_table_start = key_table_start + len(items) * 8
    ids_start = value_table_start + len(items) * 8
    strs_start = ids_start + len(ids)

    key_offsets: list[int] = []
    value_offsets: list[int] = []
    offset = 0
    for key, _ in items:
        key_offsets += [len(key), ids_start + offset]
        offset += len(key) + 1
    offset = 0
    for _, value in items:
        value_offsets += [len(value), strs_start + offset]
        offset += len(value) + 1

    output = struct.pack(
        "Iiiiiii",
        0x950412DE,          # magic
        0,                   # revision
        len(items),
        key_table_start,
        value_table_start,
        0,                   # hash table size: unused
        0,                   # hash table offset: unused
    )
    output += array.array("i", key_offsets + value_offsets).tobytes()
    output += ids + strs
    path.write_bytes(output)


def _mo_path(language: str) -> Path:
    """Return the compiled catalogue path for a language."""
    return LOCALES_DIR / language / "LC_MESSAGES" / f"{DOMAIN}.mo"


def compile_catalog(language: str, *, force: bool = False) -> Path | None:
    """Compile ``<lang>/LC_MESSAGES/datahub.po`` when it is newer than the .mo.

    Returns the .mo path when one exists afterwards, else None.  Never raises:
    a broken catalogue must degrade to untranslated text, not stop the app.
    """
    po_path = _mo_path(language).with_suffix(".po")
    mo_path = _mo_path(language)
    if not po_path.exists():
        return mo_path if mo_path.exists() else None

    fresh = mo_path.exists() and mo_path.stat().st_mtime >= po_path.stat().st_mtime
    if fresh and not force:
        return mo_path

    try:
        mo_path.parent.mkdir(parents=True, exist_ok=True)
        _write_mo(_parse_po(po_path), mo_path)
    except Exception:
        applogger.exception("Failed to compile locale %s", po_path)
        return mo_path if mo_path.exists() else None
    return mo_path


# ----------------------------------------------------------------------
# Lookup
# ----------------------------------------------------------------------
@lru_cache(maxsize=8)
def _translation(language: str) -> gettext.NullTranslations:
    """Return the gettext catalogue for a language, cached.

    ``fallback=True`` means a language with no catalogue yields a
    NullTranslations, whose ``gettext`` returns the message unchanged - the
    English source string.
    """
    compile_catalog(language)
    return gettext.translation(
        DOMAIN,
        localedir=str(LOCALES_DIR),
        languages=[language],
        fallback=True,
    )


def available_languages() -> list[str]:
    """Return language codes that have a catalogue, English always first."""
    codes = {DEFAULT_LANGUAGE}
    if LOCALES_DIR.is_dir():
        codes.update(
            path.name
            for path in LOCALES_DIR.iterdir()
            if (path / "LC_MESSAGES").is_dir()
        )
    return [DEFAULT_LANGUAGE, *sorted(codes - {DEFAULT_LANGUAGE})]


def language() -> str:
    """Return the active language code.

    Always a real code: ``set_language(AUTO_LANGUAGE)`` resolves before it
    stores, so a caller that wants to know what is *shown* asks here and one
    that wants to know what was *chosen* reads config.json.
    """
    return _language


def platform_language() -> str:
    """Return the language this machine is set to, as a bare code.

    ``QLocale.system().uiLanguages()`` is the question to ask, and asking the
    wrong one is what made "Auto" read English on an Italian Mac.  macOS keeps
    two settings that Qt reports separately: the *format* locale, which
    ``QLocale.system().name()`` gives and which follows the region - an
    Italian in Ireland has ``en_IE`` there - and the *preferred UI languages*,
    which is the ordered list the user actually chose to read the interface
    in.  Only the second is the one this setting means.  Windows draws the
    same distinction; Linux mostly does not, but answers the same way.

    The list arrives most specific first (``it-Latn-IT``, ``it-IT``,
    ``it-Latn``, ``it``) and in preference order when several are configured,
    so the first entry whose language has a catalogue wins - a Mac set to
    Italian then English gets Italian, and one set to a language nothing is
    translated into falls through to English rather than to the first entry.

    The environment variables come last and only as a rescue: Qt reports the C
    locale when a Linux session sets none, and ``LANG`` is then the only thing
    that knows better.  They are a Unix convention that Windows and macOS do
    not follow, which is why they cannot be asked first.
    """
    for candidate in _platform_language_candidates():
        code = _language_subtag(candidate)
        if code and code in available_languages():
            return code

    applogger.info(
        "No catalogue for anything this machine is set to; using %s.",
        DEFAULT_LANGUAGE,
    )
    return DEFAULT_LANGUAGE


def _platform_language_candidates() -> list[str]:
    """Return every locale name this machine offers, best answer first."""
    candidates: list[str] = []

    try:
        from PySide6.QtCore import QLocale

        system = QLocale.system()
        candidates.extend(str(name) for name in system.uiLanguages())
        candidates.append(str(system.name()))
    except Exception:  # pragma: no cover - depends on the Qt build
        applogger.warning(
            "Could not read the platform locale from Qt.",
            show_dialog=False,
            raise_error=False,
        )

    # Only useful where Qt came back with nothing usable, which in practice
    # means a Linux session that sets LANG and no desktop locale.
    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(variable, "")
        # LANGUAGE is a colon-separated preference list; the rest are single
        # locales, and splitting one of those on ":" simply yields itself.
        candidates.extend(part for part in value.split(":") if part)

    return candidates


def _language_subtag(name: str) -> str:
    """Return the language half of a locale name, lowercased.

    ``it-Latn-IT``, ``it_IT.UTF-8`` and ``it`` all answer ``it``: the
    catalogues are named for the language, so a script, a region and an
    encoding are all noise here.  ``C`` and ``POSIX`` answer nothing, because
    they are Qt's and libc's way of saying no locale is set rather than the
    names of languages.
    """
    text = str(name or "").strip()
    if not text:
        return ""

    for separator in ("-", "_", "."):
        text = text.split(separator, 1)[0]

    code = text.lower()
    return "" if code in ("c", "posix") else code


def set_language(code: str) -> str:
    """Set the active language and return the code actually selected.

    ``AUTO_LANGUAGE`` - and an empty value, which is a config.json that has
    never been written - resolve through :func:`platform_language`.  An
    unknown code is refused rather than silently accepted, so a typo in
    config.json shows up in the log instead of as an untranslated interface.
    """
    global _language
    clean = str(code or "").strip().lower()
    if clean in ("", AUTO_LANGUAGE):
        _language = platform_language()
        applogger.info("Language set to follow the platform: %s", _language)
        return _language

    if clean != DEFAULT_LANGUAGE and clean not in available_languages():
        applogger.warning(
            "No catalogue for '%s'; keeping %s.",
            clean,
            _language,
            show_dialog=False,
            raise_error=False,
        )
        return _language

    _language = clean
    return _language


#: Qt's own catalogues, in the order they are loaded.  ``qtbase`` carries the
#: standard dialog buttons - Yes, No, Cancel, Open, Save - and ``qt`` is the
#: umbrella catalogue older builds used for the same strings.  Both are tried
#: because which one a given PySide6 wheel ships has changed over releases.
_QT_CATALOGUES: tuple[str, ...] = ("qtbase", "qt")

#: Kept alive for the life of the process.  QCoreApplication.installTranslator
#: does not take ownership, so a QTranslator that goes out of scope is
#: collected and its strings silently revert to English - which is a bug that
#: only shows up in a release build, once the collector runs.
_qt_translators: list[Any] = []


def install_qt_translations(app: Any) -> list[str]:
    """Translate Qt's own strings - the dialog buttons above all.

    ``QMessageBox`` builds its Yes and No from Qt's catalogue, not from ours:
    the text never passes through :func:`tr`, so no amount of translating this
    application reaches it.  Every confirmation in the app therefore asked in
    Italian and answered in English until this was installed.

    Returns the catalogue names actually loaded, which is what the tests
    assert on - "it did not raise" is not the same as "it worked", and the
    failure here is silent by nature.
    """
    from PySide6.QtCore import QLibraryInfo, QTranslator

    if _language == DEFAULT_LANGUAGE:
        # Qt's source strings are English, so there is nothing to load and a
        # missing English catalogue is not a warning worth printing.
        return []

    directory = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    loaded: list[str] = []
    for catalogue in _QT_CATALOGUES:
        translator = QTranslator()
        if not translator.load(f"{catalogue}_{_language}", directory):
            continue
        if not app.installTranslator(translator):
            continue
        _qt_translators.append(translator)
        loaded.append(catalogue)

    if not loaded:
        applogger.info(
            "Qt ships no %r translation in %s; its standard dialog buttons "
            "stay in English.",
            _language,
            directory,
        )
    return loaded


def tr(message: str) -> str:
    """Translate one English source string into the active language.

    This is the standard gettext contract.  ``tr`` is the implementation and
    :data:`_` is the name call sites use; they are the same function, so both
    spellings are found by ``xgettext`` and either may be called.
    """
    if _language == DEFAULT_LANGUAGE:
        return message
    return _translation(_language).gettext(message)


#: The conventional gettext alias, and what the application calls.
#:
#: One hazard comes with the name, and it is silent: ``_`` is also Python's
#: throwaway variable, so a single ``for _ in range(3)`` anywhere in a module
#: rebinds it, and every ``_("...")`` after that point raises *'int' object is
#: not callable* - at runtime, in whichever dialog happens to open.  Bind
#: throwaways to ``_unused`` in any module that imports this.
#: ``test_localization`` fails if one slips back in.
_ = tr

