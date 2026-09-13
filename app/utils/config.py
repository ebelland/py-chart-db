"""The application's own catalogue, and the settings a person accumulates.

Two files, and the difference between them is who writes them:

``config.json``
    Ships with the application.  The action catalogue and the message
    catalogue - every button's label, icon and tooltip, every message box's
    wording - plus the constants catalogue, the numeric knobs (a timeout, a
    row limit, a minimum size) read one at a time through
    :func:`get_constant`.  Versioned, reviewed, translated where applicable,
    and never written at runtime.

``user.json``
    Written by the application as it runs, and by nothing else.  The last
    database opened, where the windows were, what each dialog was set to last
    time, the chosen style, language and save format.  Not versioned - it
    describes one person's machine.

They used to be one file, and the cost of that showed up in the repository
rather than in a bug report: running the test suite rewrote ``last_database``
to a pytest temporary directory and left it in ``git status`` every time, so
a real diff had to be picked out from around it.  ``app_style``, the window
geometry and twelve dialogs' remembered entries were versioned the same way -
committed by whoever happened to run the application before committing.

Which file a key belongs to is deliberately *not* decided by a list of user
keys.  Such a list needs extending every time a feature remembers something
new, and forgetting to extend it loses the setting silently.  It is the other
way round: :data:`APPLICATION_SECTIONS` is a closed set of three, and anything
written while the application runs is a user setting by definition.

Everything goes through :func:`get_section` / :func:`set_section` rather than
through one accessor pair per key.  Why: every feature that wanted to remember
something was otherwise adding two near-identical functions, and the
read-modify-write dance around ``load_config``/``save_config`` was copied with
it - which is how a section gets silently dropped by a writer that saved a
stale copy of the whole document.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.logs.logger import applogger
from app.utils.i18n import AUTO_LANGUAGE


def _repo_root() -> Path:
    """Repository root: parent of the 'app' package directory."""
    return Path(__file__).resolve().parent.parent


CONFIG_PATH = _repo_root().parent / "config.json"
USER_CONFIG_PATH = _repo_root().parent / "user.json"
MPLSTYLES_DIR = _repo_root().parent / "mplstyles"

#: The only top-level keys ``config.json`` owns.  Closed on purpose: a key
#: that is not in here and is found in config.json is a setting left over from
#: before the split, and is moved out on the next load.
APPLICATION_SECTIONS: frozenset[str] = frozenset({"actions", "messages", "constants"})


# Each file parsed once, with the signature it was parsed from, plus the
# merged view the readers actually get.  See _read for why this is worth it.
_cache: dict[Path, dict[str, Any]] = {}
_cache_stamp: dict[Path, tuple[int, int] | None] = {}
_merged: dict[str, Any] | None = None
_merged_stamp: tuple[Any, Any] | None = None
_migrated_from: Path | None = None


def _file_stamp(path: Path) -> tuple[int, int] | None:
    """Return (mtime_ns, size) of *path*, or None when it is missing."""
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def _read(path: Path) -> dict[str, Any]:
    """Return one file's contents, or an empty mapping if unreadable.

    Cached against the file's mtime and size.  This matters more than it
    looks: config.json holds the action catalogue, and resolving one button's
    icon, label and tooltip asks for it several times - so every menu and
    every toolbar was re-reading and re-parsing the whole file dozens of times
    while building.  Keying on the stamp rather than caching forever keeps a
    hand-edit picked up on the next call.
    """
    stamp = _file_stamp(path)
    if stamp is None:
        _cache[path], _cache_stamp[path] = {}, None
        return {}
    if _cache_stamp.get(path) == stamp and path in _cache:
        return _cache[path]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        applogger.exception("Failed to load config: %s", path)
        _cache[path], _cache_stamp[path] = {}, stamp
        return {}

    _cache[path] = data if isinstance(data, dict) else {}
    _cache_stamp[path] = stamp
    return _cache[path]


def _write(path: Path, data: dict[str, Any]) -> bool:
    """Write one file, pretty-printed.  True when it reached the disk.

    ``indent=2`` and ``sort_keys=False`` are deliberate: both files are meant
    to be opened and read by a person, and reordering them on every save would
    make every diff unreadable.
    """
    global _merged
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        applogger.exception("Failed to save config: %s", path)
        return False

    # Adopt what was just written instead of invalidating: a save is usually
    # followed by a read, and the writer already holds the whole document.
    _cache[path], _cache_stamp[path] = data, _file_stamp(path)
    _merged = None
    return True


def load_config() -> dict[str, Any]:
    """Return both files as one mapping, the user's settings winning.

    A read-only view: mutating the result changes nothing on disk, and which
    of the two files a key came from is not something a reader should have to
    know.
    """
    global _merged, _merged_stamp

    _migrate_user_keys()
    application = _read(CONFIG_PATH)
    user = _read(USER_CONFIG_PATH)

    stamp = (_cache_stamp.get(CONFIG_PATH), _cache_stamp.get(USER_CONFIG_PATH))
    if _merged is not None and stamp == _merged_stamp:
        return _merged

    _merged = {**application, **user}
    _merged_stamp = stamp
    return _merged


def load_user_config() -> dict[str, Any]:
    """Return the user's settings alone, without the shipped catalogue."""
    _migrate_user_keys()
    return _read(USER_CONFIG_PATH)


def save_user_config(cfg: dict[str, Any]) -> None:
    """Replace the user's settings wholesale.

    A caller wanting to change one thing wants :func:`set_section` or
    :func:`update_section`: this overwrites everything, including the sections
    the caller never looked at and may have read minutes ago.
    """
    _write(USER_CONFIG_PATH, cfg)


# ----------------------------------------------------------------------
# Migration
# ----------------------------------------------------------------------
def _migrate_user_keys() -> None:
    """Move settings left in config.json into user.json, once per file.

    A settings file cannot be split by editing the repository's copy: the file
    on a person's machine is the one holding their windows, their language and
    their last database, and shipping a stripped config.json would simply lose
    them.  So the split happens where the real data is - on the first load
    after the update - and config.json is left holding the two catalogues.

    Rewriting config.json is attempted but not required: an application
    installed into a read-only directory keeps working, because until the keys
    can be moved they are still read from where they are.  Anything already in
    user.json wins, being by definition newer than what is still in
    config.json.
    """
    global _migrated_from
    if _migrated_from == CONFIG_PATH:
        return

    application = _read(CONFIG_PATH)
    settings = {
        name: value
        for name, value in application.items()
        if name not in APPLICATION_SECTIONS
    }
    if not settings:
        _migrated_from = CONFIG_PATH
        return

    user = dict(_read(USER_CONFIG_PATH))
    moved = sorted(name for name in settings if name not in user)
    if not _write(USER_CONFIG_PATH, {**settings, **user}):
        return

    _migrated_from = CONFIG_PATH
    _write(
        CONFIG_PATH,
        {
            name: value
            for name, value in application.items()
            if name in APPLICATION_SECTIONS
        },
    )
    applogger.info(
        "Moved %d setting(s) out of %s into %s: %s",
        len(moved),
        CONFIG_PATH.name,
        USER_CONFIG_PATH.name,
        ", ".join(moved) or "none that were not already there",
    )


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------
def get_section(name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one top-level object from either file, the user's winning.

    A missing or non-object section reads as *default* (empty by default), so
    callers never have to guard against a hand-edited file.
    """
    value = load_config().get(name)
    if isinstance(value, dict):
        return value
    return dict(default or {})


def get_constant(name: str, default: Any = None) -> Any:
    """Return one tunable from the ``constants`` catalogue in config.json.

    A thin wrapper over :func:`get_section`, for modules that want a single
    named value rather than the whole section - a numeric knob (a timeout, a
    row limit, a minimum size) that a person can retune by editing
    config.json without touching the code that uses it.
    """
    return get_section("constants").get(name, default)


#: How many databases the Open recent menu keeps. Ten is what a file menu
#: can show without becoming a list to search rather than a shortcut.
MAX_RECENT_DATABASES: int = get_constant("max_recent_databases", 10)

CONFIG_RECENT_DATABASES: str = "recent_databases"


def set_section(name: str, value: dict[str, Any]) -> None:
    """Replace one top-level object in user.json, leaving the rest untouched.

    Always user.json, whatever the section is called: a section written while
    the application runs is a setting, and a catalogue is not edited by the
    program that reads it.  A caller naming a catalogue section is doing
    something wrong and is told so, rather than quietly overwriting a shipped,
    translated, reviewed file.
    """
    if name in APPLICATION_SECTIONS:
        applogger.warning(
            "Refusing to write section %r: it belongs to the catalogue in %s.",
            name,
            CONFIG_PATH.name,
            show_dialog=False,
            raise_error=False,
        )
        return

    cfg = dict(load_user_config())
    cfg[name] = value
    save_user_config(cfg)


def update_section(name: str, **values: Any) -> None:
    """Merge *values* into one top-level object."""
    section = get_section(name)
    section.update(values)
    set_section(name, section)


def get_value(name: str, default: Any = None) -> Any:
    """Return one top-level scalar from either file, the user's winning."""
    return load_config().get(name, default)


def set_value(name: str, value: Any) -> None:
    """Write one top-level scalar to user.json, leaving the rest untouched."""
    cfg = dict(load_user_config())
    cfg[name] = value
    save_user_config(cfg)


# ----------------------------------------------------------------------
# Named preferences
# ----------------------------------------------------------------------
def get_last_database() -> Path | None:
    """Return the last opened database, or None if it is gone."""
    raw = get_value("last_database")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return path if path.exists() else None


def set_last_database(db_path: Path) -> None:
    """Remember the database to reopen at the next start, and in the menu.

    Both, from one call, deliberately: every place that makes a database
    the current one is by definition a place it belongs at the top of the
    recent list, and two functions to call would eventually be one.
    """
    set_value("last_database", str(db_path))
    remember_recent_database(db_path)


def get_recent_databases() -> list[Path]:
    """Return the recently opened databases, most recent first.

    Paths that no longer exist are dropped rather than listed: a menu
    entry that can only fail is worse than a shorter menu, and a project
    file gets renamed and moved like any other file.
    """
    raw = get_value(CONFIG_RECENT_DATABASES, [])
    if not isinstance(raw, list):
        return []

    seen: set[Path] = set()
    found: list[Path] = []
    for entry in raw:
        try:
            path = Path(str(entry)).expanduser()
        except (TypeError, ValueError):
            continue
        if path in seen or not path.exists():
            continue
        seen.add(path)
        found.append(path)
    return found[:MAX_RECENT_DATABASES]


def remember_recent_database(db_path: Path) -> None:
    """Put one database at the top of the recent list."""
    path = Path(db_path).expanduser()
    remaining = [entry for entry in get_recent_databases() if entry != path]
    set_value(
        CONFIG_RECENT_DATABASES,
        [str(path), *(str(entry) for entry in remaining)][:MAX_RECENT_DATABASES],
    )


def clear_recent_databases() -> None:
    """Forget every recent database. The current one is not reopened by it."""
    set_value(CONFIG_RECENT_DATABASES, [])


def get_language() -> str:
    """Return the configured UI language *setting*, which may be "auto".

    Returned as stored, not resolved: ``i18n.set_language`` turns "auto" into
    the platform's language, and the Settings dialog needs the raw value to
    show Auto back as the choice that was made rather than as the language it
    happened to resolve to.

    No stored language means a fresh installation, and a fresh installation
    should speak the language the machine is set to rather than English - so
    the fallback is "auto" and not a code.
    """
    return str(get_value("language", AUTO_LANGUAGE) or AUTO_LANGUAGE)


def get_import_data_dialog_config() -> dict[str, Any]:
    """Return the remembered entries of the import dialog."""
    return get_section("import_data_dialog")


def set_import_data_dialog_config(dialog_cfg: dict[str, Any]) -> None:
    """Remember the entries of the import dialog."""
    set_section("import_data_dialog", dialog_cfg)


def get_connect_database_config() -> dict[str, Any]:
    """Return the last connection the connect-to-database dialog accepted.

    Never a password: see ``DatabaseConnection``, which keeps them out of the
    project file for the same reason they are kept out of this one - a
    settings file is copied, synced and looked at, and none of those are
    things to do with a credential.
    """
    return get_section("connect_database_dialog")


def set_connect_database_config(dialog_cfg: dict[str, Any]) -> None:
    """Remember the connection and table the user just picked."""
    set_section("connect_database_dialog", dialog_cfg)


def get_user_web_sources() -> list[dict[str, Any]]:
    """Return the web sources this user added themselves.

    A list of {name, url, description}, layered onto the bundled catalogue
    in web_sources.json by ``app.utils.data_sources.load_web_data_sources`` -
    that file ships with the application and is never written here.
    """
    entries = get_section("web_sources").get("entries")
    return [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []


def set_user_web_sources(entries: list[dict[str, Any]]) -> None:
    """Remember the web sources the user has added themselves."""
    set_section("web_sources", {"entries": entries})
