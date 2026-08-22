"""The credits page, which is a licence statement as much as a thank-you.

Hand-kept credits go wrong quietly: they name a version nobody has installed,
or miss a dependency added last week. So the page is read from
requirements.txt and from the installed packages' own metadata, and these
tests are about that reading - plus the one thing no amount of machinery can
recover if it is dropped, which is who wrote the thing.
"""
from __future__ import annotations

from email.message import Message
from importlib import metadata
from pathlib import Path

import pytest

from app import APP_NAME, APP_VERSION
from app.utils import credits


# ----------------------------------------------------------------------
# Reading requirements.txt
# ----------------------------------------------------------------------
def test_a_pinned_requirement_is_taken_apart() -> None:
    parsed = credits.parse_requirements("numpy==2.5.1\n")

    assert parsed == [credits.Requirement(name="numpy", specifier="==2.5.1")]


def test_comments_and_blank_lines_are_skipped() -> None:
    parsed = credits.parse_requirements(
        "# a comment\n\nnumpy==2.5.1  # why this version\n\n"
    )

    assert [requirement.name for requirement in parsed] == ["numpy"]
    assert parsed[0].specifier == "==2.5.1"


def test_an_environment_marker_is_kept_not_dropped() -> None:
    """"macOS only" is part of what the dependency *is*; the page says so."""
    parsed = credits.parse_requirements(
        'pyobjc-framework-Cocoa>=12.2; sys_platform == "darwin"\n'
    )

    assert parsed[0].name == "pyobjc-framework-Cocoa"
    assert parsed[0].specifier == ">=12.2"
    assert parsed[0].marker == 'sys_platform == "darwin"'


def test_extras_do_not_confuse_the_name() -> None:
    parsed = credits.parse_requirements("uvicorn[standard]>=0.30\n")

    assert parsed[0].name == "uvicorn"


def test_pip_directives_are_not_packages() -> None:
    parsed = credits.parse_requirements("-r other.txt\n-e .\nnumpy==2.5.1\n")

    assert [requirement.name for requirement in parsed] == ["numpy"]


def test_the_real_requirements_file_is_readable() -> None:
    assert credits.REQUIREMENTS_PATH.is_file()
    assert credits.parse_requirements(
        credits.REQUIREMENTS_PATH.read_text(encoding="utf-8")
    )


# ----------------------------------------------------------------------
# Describing what is installed
# ----------------------------------------------------------------------
def test_an_installed_package_reports_its_own_version() -> None:
    """Installed, not pinned: the page describes this machine."""
    package = credits.describe("numpy", required="==1.0.0")

    assert package.present
    assert package.installed == metadata.version("numpy")
    assert package.required == "==1.0.0"
    assert package.summary


def test_a_missing_package_still_gets_a_row() -> None:
    """An optional dependency that is absent is a fact worth showing, not a
    reason to make the list look shorter than the project's dependencies."""
    package = credits.describe("definitely-not-installed-xyz")

    assert package.present is False
    assert package.installed == ""


def test_every_declared_dependency_is_described() -> None:
    described = credits.packages()

    assert len(described) >= 8
    assert {"numpy", "pandas", "PySide6"} <= {package.name for package in described}


def test_every_installed_dependency_names_a_licence() -> None:
    """The column that makes this a licence statement rather than a list."""
    unlicensed = [
        package.name
        for package in credits.packages()
        if package.present and not package.license
    ]

    assert unlicensed == []


def test_a_licence_never_arrives_as_the_whole_licence_text() -> None:
    """matplotlib, SciPy, pandas and scikit-image all put their entire licence
    in the metadata's free-text field. A table cell is not where it is read."""
    for package in credits.packages():
        assert "\n" not in package.license
        assert len(package.license) <= 60, package.name


def test_the_spdx_expression_wins_over_the_classifier() -> None:
    message = Message()
    message["License-Expression"] = "MIT AND BSD-3-Clause"
    message["Classifier"] = "License :: OSI Approved :: GNU General Public License"

    assert credits.license_of(message) == "MIT AND BSD-3-Clause"


def test_the_classifier_is_used_when_there_is_no_expression() -> None:
    message = Message()
    message["Classifier"] = "License :: OSI Approved :: BSD License"
    message["License"] = "Copyright (c) 2001\nAll rights reserved.\n" * 40

    assert credits.license_of(message) == "BSD License"


def test_a_long_free_text_licence_is_refused_rather_than_truncated() -> None:
    message = Message()
    message["License"] = "This is a very long licence text " * 10

    assert credits.license_of(message) == ""


# ----------------------------------------------------------------------
# The page itself
# ----------------------------------------------------------------------
@pytest.fixture
def page(qapp) -> str:
    from app.dialogs.credits_dialog import credits_html

    return credits_html()


def test_the_author_is_named(page: str) -> None:
    """The one thing no metadata can recover if it is dropped."""
    assert credits.AUTHOR == "Artafasio Pippoz"
    assert credits.AUTHOR in page


def test_both_assistants_are_named(page: str) -> None:
    """They wrote code here; someone reading this in a year is entitled to
    know how it was written."""
    assert "Claude" in page
    assert "GitHub Copilot" in page


def test_every_assistant_says_what_it_did(page: str) -> None:
    for _name, _maker, contribution in credits.ASSISTANTS:
        assert contribution.strip()
        assert contribution.split(",")[0] in page


def test_the_page_carries_the_application_and_its_version(page: str) -> None:
    assert APP_NAME in page
    assert APP_VERSION in page


def test_every_library_appears_with_its_licence(page: str) -> None:
    for package in credits.packages():
        assert package.name in page
        if package.present:
            assert package.installed in page


def test_the_dialog_opens_and_shows_the_page(qapp) -> None:
    from app.dialogs.credits_dialog import CreditsDialog

    dialog = CreditsDialog()

    assert dialog.windowTitle()
    assert credits.AUTHOR in dialog._view._html


def test_a_build_without_requirements_still_credits_the_people(
    tmp_path: Path, qapp
) -> None:
    """Refusing to open would be the worst possible failure for this dialog."""
    assert credits.packages(tmp_path / "nothing-here.txt") == []
