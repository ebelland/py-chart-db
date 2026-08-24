"""Guards for the ``_()`` sweep.

The application's user-facing strings go through ``app.utils.i18n._``, the
conventional gettext alias.  Three things can go wrong quietly, and each has a
test here, because none of them shows up as a crash in the language the strings
are already written in.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.utils import i18n

APP_DIR = Path(__file__).resolve().parent.parent
PO_PATH = i18n.LOCALES_DIR / "it" / "LC_MESSAGES" / f"{i18n.DOMAIN}.po"


def _modules() -> list[Path]:
    return [
        path
        for path in sorted(APP_DIR.rglob("*.py"))
        if "__pycache__" not in path.parts and "tests" not in path.parts
    ]


def _translator_calls(tree: ast.AST) -> list[str]:
    return [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("_", "tr")
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]


# ----------------------------------------------------------------------
# The alias
# ----------------------------------------------------------------------
def test_the_alias_is_the_same_function() -> None:
    """Two names, one implementation, so xgettext finds either spelling."""
    assert i18n._ is i18n.tr


def test_no_module_shadows_the_translator() -> None:
    """A throwaway ``_`` anywhere in a function breaks every ``_()`` in it.

    Python binds names per function, not per statement, so one
    ``path, _ = QFileDialog.getSaveFileName(self, _("Export CSV"), ...)`` makes
    ``_`` local to that whole method - and the ``_("Export CSV")`` on the same
    line then raises UnboundLocalError the first time it runs.  Nothing about
    the source looks wrong, which is why this is a test and not a convention.
    """
    offenders: list[str] = []
    for path in _modules():
        source = path.read_text(encoding="utf-8")
        if "from app.utils.i18n import" not in source or "import _" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Name)
                and node.id == "_"
                and isinstance(node.ctx, ast.Store)
            ):
                offenders.append(
                    f"{path.relative_to(APP_DIR)}:{node.lineno} - use _unused"
                )

    assert offenders == []


# ----------------------------------------------------------------------
# The catalogue
# ----------------------------------------------------------------------
def test_every_translated_string_is_in_the_italian_catalogue() -> None:
    """A wrapped string with no entry renders as English, mid-sentence."""
    catalog = i18n._parse_po(PO_PATH)
    missing = sorted(
        {
            message
            for path in _modules()
            for message in _translator_calls(
                ast.parse(path.read_text(encoding="utf-8"))
            )
            if message not in catalog
        }
    )

    assert missing == [], f"{len(missing)} strings have no Italian: {missing[:5]}"


def test_labels_held_in_tables_are_translated_too() -> None:
    """The gap the ``_()`` sweep cannot see.

    A label defined as module-level data and passed as ``tr(label)`` carries no
    literal at the call site, so neither xgettext nor
    ``test_every_translated_string_is_in_the_italian_catalogue`` can find it.
    "Dark" reached the settings dialog untranslated exactly this way. Each of
    these tables has to be named here by hand - which is the cost of holding UI
    text as data, and the reason to keep such tables few.
    """
    from app.styles.style import APP_STYLE_LABELS

    # LANGUAGE_NAMES is deliberately absent: a language picker names each
    # language in that language - Italiano stays Italiano in the English UI -
    # so those are endonyms rather than strings to translate, and the dialog
    # passes them without tr() for the same reason.
    #
    # RESIZE_MODES lived here too, until the Settings dialog stopped offering
    # a global default fit mode at all - see app/dialogs/settings_dialog.py.
    # Its replacement, app.widgets.chart_panel.RESIZE_MODE_CHOICES, is covered
    # by test_the_fit_mode_choices_are_translated below instead: it carries a
    # tooltip alongside each label, which this flat list of labels has no
    # room for.
    catalog = i18n._parse_po(PO_PATH)
    labels = [
        *APP_STYLE_LABELS.values(),
    ]
    missing = sorted({label for label in labels if label not in catalog})

    assert missing == []


def test_declared_operation_parameters_are_translated() -> None:
    """PARAMS is the same trap, and a growing one.

    A declared parameter's label and tooltip are class data, wrapped in _() by
    ParameterForm when it builds the widget rather than at the point they are
    written. So they are invisible to the sweep in
    ``test_every_translated_string_is_in_the_italian_catalogue`` and would ship
    in English with nothing failing.

    Discovered rather than hand-listed, unlike the tables above: every
    operation that declares PARAMS is covered the day it is written, without
    anyone remembering to add it here.
    """
    from app.scanners.series_operation_scanner import (
        _discover_series_operations,
        import_class_from_file,
    )

    catalog = i18n._parse_po(PO_PATH)
    missing: set[str] = set()

    for operation in _discover_series_operations():
        cls = import_class_from_file(operation)
        if cls is None:
            continue
        for param in getattr(cls, "PARAMS", ()) or ():
            for text in (param.label, param.tooltip):
                if text and text not in catalog:
                    missing.add(text)
            for label, _value in getattr(param, "labelled_choices", lambda: ())():
                if label and label not in catalog:
                    missing.add(label)

    assert sorted(missing) == [], (
        f"{len(missing)} declared parameter strings have no Italian"
    )


def test_the_function_library_is_translated() -> None:
    """A function's name, category and description are class data too.

    The fit dialog and the function dialog both build their tree from
    ``FunctionScanner``, so these strings reach the user through ``tr(value)``
    with no literal at the call site - invisible to the sweep above, exactly
    like the operations' PARAMS. 57 functions in 10 categories would otherwise
    be the largest block of English left in an Italian window.

    Only the display is translated: the payload keeps the English name, which
    is what a saved fit refers to. A catalogue whose identities changed with
    the interface language would lose every fit saved in another one.
    """
    from app.scanners.functions_scanner import FunctionScanner

    catalog = i18n._parse_po(PO_PATH)
    missing: set[str] = set()

    for category, payloads in FunctionScanner().catalog().items():
        if category and str(category) not in catalog:
            missing.add(str(category))
        for payload in payloads:
            for key in ("name", "description"):
                text = str(payload.get(key) or "")
                if text and text not in catalog:
                    missing.add(text)

    assert sorted(missing) == [], (
        f"{len(missing)} function-library strings have no Italian"
    )


def test_the_fit_algorithms_are_translated() -> None:
    """The optimiser names and their explanations are data too.

    They are declared in ``app/functions/optimizers.py`` and handed to the
    combo as ``_(optimizer.label)``, so there is no literal at the call site.
    The description is the more important half: it is the only thing in the
    interface that says when to reach for a global method rather than the
    default.
    """
    from app.functions.optimizers import LOSSES, OPTIMIZERS

    catalog = i18n._parse_po(PO_PATH)
    missing = {
        text
        for optimizer in OPTIMIZERS
        for text in (optimizer.label, optimizer.description)
        if text and text not in catalog
    }
    missing |= {label for _key, label in LOSSES if label not in catalog}

    assert sorted(missing) == []


def test_the_axis_grid_and_tick_choices_are_translated() -> None:
    """Eight combos in the axis properties read their labels from data.

    app/charts/axis_options.py holds the vocabulary the widget shows and the
    renderer applies, so the labels reach the user as ``_(label)`` with no
    literal at the call site - the same blind spot as PARAMS and the function
    library.
    """
    from app.charts import axis_options

    catalog = i18n._parse_po(PO_PATH)
    missing = {
        label
        for choices in (
            axis_options.GRID_CHOICES,
            axis_options.TICK_CHOICES,
            axis_options.LIMIT_CHOICES,
        )
        for _value, label in choices
        if label and label not in catalog
    }

    assert sorted(missing) == []


def test_the_demo_project_names_and_summaries_are_translated() -> None:
    """The Create demo picker reads both from DEMO_PROJECTS as data.

    Same blind spot as the fit-mode choices above: no literal at the call
    site, so xgettext and the main sweep cannot see them.
    """
    from app.data.demo_project import DEMO_PROJECTS

    catalog = i18n._parse_po(PO_PATH)
    missing = {
        text
        for demo in DEMO_PROJECTS
        for text in (demo.file_name, demo.summary)
        if text and text not in catalog
    }

    assert sorted(missing) == []


def test_the_fit_mode_choices_are_translated() -> None:
    """The combo in Figure Properties reads label and tooltip from data.

    RESIZE_MODE_CHOICES replaced Settings' RESIZE_MODES (labels only, no
    tooltip) when the fit-mode picker moved to being per-figure; this is that
    table's equivalent of test_labels_held_in_tables_are_translated_too.
    """
    from app.widgets.chart_panel import RESIZE_MODE_CHOICES

    catalog = i18n._parse_po(PO_PATH)
    missing = {
        text
        for _value, label, tooltip in RESIZE_MODE_CHOICES
        for text in (label, tooltip)
        if text and text not in catalog
    }

    assert sorted(missing) == []


def test_the_catalogue_has_no_empty_translations() -> None:
    """An empty msgstr silently falls back, so it reads as untranslated."""
    catalog = i18n._parse_po(PO_PATH)
    blank = sorted(key for key, value in catalog.items() if key and not value.strip())

    assert blank == []


def test_switching_language_actually_changes_the_strings() -> None:
    """The whole point, asserted once end to end."""
    previous = i18n.language()
    try:
        i18n.set_language("it")
        assert i18n._("Settings") == "Impostazioni"
        assert i18n._("Axis properties") == "Proprietà dell'asse"
        i18n.set_language("en")
        assert i18n._("Settings") == "Settings"
    finally:
        i18n.set_language(previous)


# ----------------------------------------------------------------------
# What must NOT be translated
# ----------------------------------------------------------------------
def test_combo_data_values_are_not_translated() -> None:
    """``addItem(label, data)`` - the data is compared against, not read.

    Wrapping it makes the combo return "tutti" where the code tests for
    "all", which fails only in Italian and only at the point of use.
    """
    wrapped: list[str] = []
    for path in _modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "attr", None) not in ("addItem", "insertItem"):
                continue
            # Two wrapped arguments means the second is data: no Qt widget
            # takes two labels here.
            if len(node.args) >= 2 and all(
                isinstance(argument, ast.Call)
                and getattr(argument.func, "id", None) in ("_", "tr")
                for argument in node.args[:2]
            ):
                wrapped.append(f"{path.relative_to(APP_DIR)}:{node.lineno}")

    assert wrapped == []


@pytest.mark.parametrize("method", ["info", "warning", "error", "exception", "debug"])
def test_log_messages_are_not_translated(method: str) -> None:
    """The log is read by whoever is debugging, against English source.

    A translated log line cannot be searched for in the code that emitted it.
    """
    translated: list[str] = []
    for path in _modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == method
                and node.args
                and isinstance(node.args[0], ast.Call)
                and getattr(node.args[0].func, "id", None) in ("_", "tr")
            ):
                translated.append(f"{path.relative_to(APP_DIR)}:{node.lineno}")

    assert translated == []


# ----------------------------------------------------------------------
# Qt's own strings, which ours cannot reach
# ----------------------------------------------------------------------
def test_qt_standard_buttons_are_translated(qapp) -> None:
    """QMessageBox builds Yes and No from Qt's catalogue, not from ours.

    The text never passes through tr(), so no amount of translating this
    application reached it: every confirmation asked in Italian and answered
    in English until a QTranslator was installed for Qt itself.
    """
    from PySide6.QtWidgets import QMessageBox

    before = i18n.language()
    try:
        i18n.set_language("it")
        loaded = i18n.install_qt_translations(qapp)
        if not loaded:
            pytest.skip("this PySide6 wheel ships no Italian Qt catalogue")

        box = QMessageBox()
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        labels = [button.text().replace("&", "") for button in box.buttons()]

        assert "Yes" not in labels, labels
        assert "Sì" in labels, labels
    finally:
        i18n.set_language(before)


def test_english_loads_no_qt_catalogue(qapp) -> None:
    """Qt's source strings are English; a missing en catalogue is not news."""
    before = i18n.language()
    try:
        i18n.set_language("en")
        assert i18n.install_qt_translations(qapp) == []
    finally:
        i18n.set_language(before)


def test_the_translators_are_kept_alive(qapp) -> None:
    """installTranslator does not take ownership, so a QTranslator that goes
    out of scope is collected and its strings silently revert - a bug that
    only appears once the collector runs."""
    before = i18n.language()
    try:
        i18n.set_language("it")
        i18n.install_qt_translations(qapp)
        assert i18n._qt_translators
    finally:
        i18n.set_language(before)

