"""Read and write the application-level ``config.json``.

This file holds machine-level preferences: last opened database, icon sets,
the action catalogue, remembered dialog entries, window geometry.  Anything
that belongs to a specific figure is stored in the figure descriptor instead,
so that it travels with the .dhub file.

Everything here goes through :func:`get_section` / :func:`set_section` rather
than through one accessor pair per key.  Why: every feature that wants to
remember something was otherwise adding two near-identical functions, and the
read-modify-write dance around ``load_config``/``save_config`` was copied with
it - which is how a section can be silently dropped by a writer that saved a
stale copy of the whole file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.logs.logger import applogger


def _repo_root() -> Path:
    """Repository root: parent of the 'app' package directory."""
    return Path(__file__).resolve().parent.parent


CONFIG_PATH = _repo_root().parent / "config.json"
MPLSTYLES_DIR = _repo_root().parent / "mplstyles"


# Parsed config.json, with the file signature it was parsed from.
# See load_config for why this is worth caching.
_cache: dict[str, Any] | None = None
_cache_stamp: tuple[int, int] | None = None


def _file_stamp() -> tuple[int, int] | None:
    """Return (mtime_ns, size) of config.json, or None when it is missing."""
    try:
        info = CONFIG_PATH.stat()
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def load_config() -> dict[str, Any]:
    """Return the whole configuration, or an empty mapping if unreadable.

    The result is cached against the file's mtime and size.  This matters more
    than it looks: config.json now holds the action catalogue, and resolving
    one button's icon, label and tooltip asks for it several times - so every
    menu and every toolbar was re-reading and re-parsing the whole file dozens
    of times while building.  Keying on the stamp rather than caching forever
    keeps a hand-edit picked up on the next call.
    """
    global _cache, _cache_stamp

    stamp = _file_stamp()
    if stamp is None:
        _cache, _cache_stamp = {}, None
        return {}
    if _cache is not None and stamp == _cache_stamp:
        return _cache

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        applogger.exception("Failed to load config: %s", CONFIG_PATH)
        return {}

    _cache = data if isinstance(data, dict) else {}
    _cache_stamp = stamp
    return _cache


def save_config(cfg: dict[str, Any]) -> None:
    """Write the whole configuration back, pretty-printed.

    ``indent=2`` and ``sort_keys=False`` are deliberate: config.json is meant
    to be opened and edited by hand, and reordering it on every save would make
    every diff unreadable.
    """
    global _cache, _cache_stamp
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        applogger.exception("Failed to save config: %s", CONFIG_PATH)
        return

    # Adopt what was just written instead of invalidating: a save is usually
    # followed by a read, and the writer already holds the whole document.
    _cache, _cache_stamp = cfg, _file_stamp()


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------
def get_section(name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one top-level object from the configuration.

    A missing or non-object section reads as *default* (empty by default), so
    callers never have to guard against a hand-edited file.
    """
    value = load_config().get(name)
    if isinstance(value, dict):
        return value
    return dict(default or {})


def set_section(name: str, value: dict[str, Any]) -> None:
    """Replace one top-level object, leaving every other section untouched."""
    cfg = load_config()
    cfg[name] = value
    save_config(cfg)


def update_section(name: str, **values: Any) -> None:
    """Merge *values* into one top-level object."""
    section = get_section(name)
    section.update(values)
    set_section(name, section)


def get_value(name: str, default: Any = None) -> Any:
    """Return one top-level scalar from the configuration."""
    return load_config().get(name, default)


def set_value(name: str, value: Any) -> None:
    """Write one top-level scalar, leaving every other key untouched."""
    cfg = load_config()
    cfg[name] = value
    save_config(cfg)


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
    """Remember the database to reopen at the next start."""
    set_value("last_database", str(db_path))


def get_language() -> str:
    """Return the configured UI language code."""
    return str(get_value("language", "en") or "en")


def get_import_data_dialog_config() -> dict[str, Any]:
    """Return the remembered entries of the import dialog."""
    return get_section("import_data_dialog")


def set_import_data_dialog_config(dialog_cfg: dict[str, Any]) -> None:
    """Remember the entries of the import dialog."""
    set_section("import_data_dialog", dialog_cfg)
