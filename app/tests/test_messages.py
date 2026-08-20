"""Tests for the message catalogue.

Fifty-eight message boxes had their wording written inline, which made them
untranslatable and inconsistent with each other.  These tests pin the contract
of the catalogue that replaced them.

They matter more than usual because config.json is now the only copy: there is
no table in Python to fall back on, so an id the code uses and the file does
not define would reach the user as a box titled ``import.no_table_name``.  The
test below that walks every show_message/ask call site is what makes that
impossible to ship.
"""
from __future__ import annotations

import pytest

from app.utils import i18n
from app.utils.messages import (
    CONFIG_SECTION,
    MessageSpec,
    catalog,
    message,
)


@pytest.fixture(autouse=True)
def _english():
    """Every test starts and ends in English."""
    previous = i18n.language()
    i18n.set_language(i18n.DEFAULT_LANGUAGE)
    yield
    i18n.set_language(previous)


@pytest.fixture
def messages() -> dict[str, MessageSpec]:
    """The catalogue as config.json defines it."""
    return catalog()


# ----------------------------------------------------------------------
# The catalogue
# ----------------------------------------------------------------------
def test_every_message_has_a_title_and_a_text(messages) -> None:
    for message_id, spec in messages.items():
        assert spec.title.strip(), f"{message_id} has no title"
        assert spec.text.strip(), f"{message_id} has no text"


def test_every_level_is_one_the_ui_can_render(messages) -> None:
    for message_id, spec in messages.items():
        assert spec.level in {"info", "warning", "error", "question"}, message_id


def test_questions_are_phrased_as_questions(messages) -> None:
    """A yes/no box whose text is not a question is a UI bug."""
    for message_id, spec in messages.items():
        if spec.level == "question":
            assert "?" in spec.text, f"{message_id} is a question with no question"


def test_the_catalogue_round_trips_through_json(messages) -> None:
    for message_id, spec in messages.items():
        assert MessageSpec.from_config(message_id, spec.to_config()) == spec


def test_an_unknown_id_still_produces_something_visible() -> None:
    spec = message("no.such.message")
    assert spec.message_id == "no.such.message"
    assert spec.title == "no.such.message"


def test_an_unknown_level_falls_back_to_info() -> None:
    spec = MessageSpec.from_config("x", {"level": "catastrophe", "title": "t", "text": "x"})
    assert spec.level == "info"


# ----------------------------------------------------------------------
# Placeholders
# ----------------------------------------------------------------------
def test_placeholders_are_filled_from_the_keywords() -> None:
    _title, text = message("query.confirm_delete").translated(name="daily")
    assert "daily" in text
    assert "{name}" not in text


def test_a_missing_placeholder_leaves_the_text_readable() -> None:
    """A box reporting a failure must not fail itself."""
    _title, text = message("query.confirm_delete").translated()
    assert text  # not an exception, and not empty


def test_placeholders_are_named_so_a_translation_can_reorder_them(messages) -> None:
    """Positional {} would force every language into English word order."""
    offenders = [
        message_id for message_id, spec in messages.items() if "{}" in spec.text
    ]
    assert offenders == []


def test_an_exception_can_be_passed_straight_in() -> None:
    """Call sites pass the exception, not str(exception)."""
    _title, text = message("preview.delete_column_failed").translated(
        error=ValueError("no such column")
    )
    assert "no such column" in text


# ----------------------------------------------------------------------
# Translation
# ----------------------------------------------------------------------
def test_the_italian_locale_covers_every_message(messages) -> None:
    translations = i18n._parse_po(
        i18n.LOCALES_DIR / "it" / "LC_MESSAGES" / f"{i18n.DOMAIN}.po"
    )
    missing = [
        message_id
        for message_id, spec in messages.items()
        for text in (spec.title, spec.text)
        if text and text not in translations
    ]
    assert missing == []


def test_the_placeholders_survive_translation(messages) -> None:
    """A translation that drops {name} would show the wrong thing silently."""
    import re

    translations = i18n._parse_po(
        i18n.LOCALES_DIR / "it" / "LC_MESSAGES" / f"{i18n.DOMAIN}.po"
    )
    pattern = re.compile(r"\{(\w+)\}")

    mismatched = [
        message_id
        for message_id, spec in messages.items()
        if spec.text in translations
        and set(pattern.findall(spec.text))
        != set(pattern.findall(translations[spec.text]))
    ]
    assert mismatched == []


def test_a_translated_message_still_fills_its_placeholders() -> None:
    i18n.set_language("it")
    _title, text = message("query.confirm_delete").translated(name="giornaliera")
    assert "giornaliera" in text
    assert "{name}" not in text


# ----------------------------------------------------------------------
# config.json is the only copy
# ----------------------------------------------------------------------
def test_config_json_actually_carries_the_catalogue() -> None:
    """There is no Python fallback: an empty section means an empty UI."""
    from app.utils.config import CONFIG_PATH, get_section

    assert CONFIG_PATH.is_file(), "config.json ships with the application"
    assert get_section(CONFIG_SECTION), "config.json has no messages section"


def test_every_id_the_code_uses_is_defined(messages) -> None:
    """The one that matters: an undefined id reaches the user as its id."""
    import re
    from pathlib import Path

    app_dir = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"""(?:show_message|ask)\(\s*[^,()]+,\s*["']([\w.]+)["']""")

    used: dict[str, str] = {}
    for path in app_dir.rglob("*.py"):
        if "__pycache__" in path.parts or path.parent.name == "tests":
            continue
        for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
            used[match.group(1)] = str(path.relative_to(app_dir))

    undefined = {mid: where for mid, where in used.items() if mid not in messages}
    assert undefined == {}, f"message ids used but not defined in config.json: {undefined}"


def test_the_catalogue_has_no_entries_nobody_uses(messages) -> None:
    """Kept as a report rather than a failure: see the assertion message."""
    import re
    from pathlib import Path

    app_dir = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"""(?:show_message|ask)\(\s*[^,()]+,\s*["']([\w.]+)["']""")

    used = set()
    for path in app_dir.rglob("*.py"):
        if "__pycache__" in path.parts or path.parent.name == "tests":
            continue
        used.update(
            pattern.findall(path.read_text(encoding="utf-8", errors="replace"))
        )

    unused = sorted(set(messages) - used)
    assert unused == [], (
        "these messages are defined but never shown; either wire them up or "
        f"remove them from config.json: {unused}"
    )
