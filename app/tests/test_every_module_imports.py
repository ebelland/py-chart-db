"""Import every module in the app, once.

A stale import is invisible to a suite that reads modules as text.  When six
button factories were collapsed into two, ``main_window`` kept importing one of
the deleted names; every test still passed, because the tests that care about
main_window open it with ``read_text`` and match on strings.  The application
would have raised ImportError on the first launch.

Importing costs almost nothing and catches the whole class: deleted helpers,
renamed modules, circular imports, and anything that raises at module scope.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent

# Not importable by design, and nothing else imports them either:
#   tests      - collected by pytest directly, and importing them here would
#                run every module-level fixture twice;
#   manual_*   - runnable scripts that need a real window server.
SKIP_PARTS = {"tests", "__pycache__"}


def _module_names() -> list[str]:
    names = []
    for path in sorted(APP_DIR.rglob("*.py")):
        parts = path.relative_to(APP_DIR.parent).with_suffix("").parts
        if SKIP_PARTS & set(parts) or path.name.startswith("manual_"):
            continue
        if path.name == "__init__.py":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


MODULES = _module_names()


def test_there_are_modules_to_check() -> None:
    """A glob that matches nothing would make every test below vacuous."""
    assert len(MODULES) > 20


@pytest.mark.parametrize("module_name", MODULES)
def test_the_module_imports(module_name: str, qapp) -> None:
    """Import it.

    ``qapp`` because a few modules touch QtWidgets classes at module scope, and
    constructing those without a QApplication aborts the process rather than
    raising.
    """
    importlib.import_module(module_name)


def test_no_module_imports_a_deleted_button_factory() -> None:
    """The specific mistake, named, so the diagnosis is in the failure.

    These four were merged into ``create_action_button`` and
    ``create_toolbar_button``.  Each took a label and a tooltip "as fallbacks",
    which is how the wording stayed hard-coded outside config.json.
    """
    deleted = (
        "create_push_button",
        "create_icon_button",
        "create_action_push_button",
        "action_label_tooltip_icon",
        "_resolve_action_presentation",
    )

    offenders = {}
    for path in APP_DIR.rglob("*.py"):
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        found = [name for name in deleted if name in source]
        if found:
            offenders[path.name] = found

    assert offenders == {}
