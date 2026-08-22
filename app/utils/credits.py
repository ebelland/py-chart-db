"""Who and what this application is made of.

Everything here is read rather than written down: the libraries come from
``requirements.txt`` and their versions and licences from the installed
packages' own metadata.  A hand-kept list would be wrong within a release -
it would name a version nobody has installed, or miss a dependency added
last week - and being wrong is worse than being absent, because a credits
page is a licence statement as much as a thank-you.

No Qt here.  The dialog that shows it is one screen of formatting; what is
worth testing is this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

#: The project's dependency list, one directory above ``app``.
REQUIREMENTS_PATH: Path = Path(__file__).resolve().parent.parent.parent / "requirements.txt"

#: Whose project this is.
AUTHOR: str = "Artafasio Pippoz"

#: The assistants that wrote code alongside the author, and what each did.
#: Named because they did the work, and because someone reading the source a
#: year from now is entitled to know how it was written.
ASSISTANTS: tuple[tuple[str, str, str], ...] = (
    (
        "Claude",
        "Anthropic",
        "Series operations, the fit engine, the renderers and most of the "
        "test suite, written in conversation.",
    ),
    (
        "GitHub Copilot",
        "GitHub",
        "Completions throughout, and the first draft of much of the dialog "
        "plumbing.",
    ),
)

#: A requirement line: name, optional extras, optional specifier, optional
#: environment marker after a semicolon.
_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9._-]+)"
    r"(?:\[(?P<extras>[^\]]*)\])?"
    r"(?P<specifier>[^;]*)"
    r"(?:;\s*(?P<marker>.*))?$"
)


class DistributionMetadata(Protocol):
    """The two lookups this module needs from a package's metadata.

    Structural rather than nominal on purpose: what ``importlib.metadata``
    hands back is a private adapter class in some versions and an
    ``email.message.Message`` in others, and a test wants to pass a plain
    Message of its own. Naming the two methods says what is actually required
    and satisfies all three.
    """

    def get(self, name: str, failobj: Any = None) -> Any: ...

    def get_all(self, name: str, failobj: Any = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class Requirement:
    """One line of requirements.txt, taken apart."""

    name: str
    specifier: str = ""
    marker: str = ""


@dataclass(frozen=True, slots=True)
class Package:
    """A dependency as the running installation actually has it."""

    name: str
    required: str
    installed: str
    summary: str
    license: str
    marker: str = ""

    @property
    def present(self) -> bool:
        """True when the package is installed in this interpreter."""
        return bool(self.installed)


def parse_requirements(text: str) -> list[Requirement]:
    """Return the requirements in *text*, in the order they are written.

    Comments, blank lines and the ``-r``/``-e`` directives are skipped; extras
    and environment markers are kept, because "macOS only" is part of the
    answer to what this application depends on.
    """
    requirements: list[Requirement] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue

        match = _REQUIREMENT_RE.match(line)
        if match is None:
            continue
        requirements.append(
            Requirement(
                name=match.group("name"),
                specifier=(match.group("specifier") or "").strip(),
                marker=(match.group("marker") or "").strip(),
            )
        )
    return requirements


def describe(name: str, *, required: str = "", marker: str = "") -> Package:
    """Return what the installed distribution says about itself.

    A package that is not installed still gets a row: an optional dependency
    that is absent is a fact about this installation, and hiding it would make
    the list look shorter than the project's actual dependencies.
    """
    try:
        distribution = metadata.metadata(name)
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        return Package(name, required, "", "", "", marker)

    return Package(
        name=str(distribution.get("Name") or name),
        required=required,
        installed=str(installed),
        summary=str(distribution.get("Summary") or "").strip(),
        license=license_of(distribution),
        marker=marker,
    )


def license_of(distribution: DistributionMetadata) -> str:
    """Return a licence short enough to put in a table cell.

    Three sources, in this order, because packaging has changed its mind
    twice: the SPDX ``License-Expression`` where a package has adopted it, the
    OSI classifier where it has not, and the free-text ``License`` field only
    when it is short. That last guard matters - matplotlib, SciPy, pandas and
    scikit-image all put their *entire licence text* in that field, and a
    table cell is not where anyone reads it.
    """
    expression = str(distribution.get("License-Expression") or "").strip()
    if expression:
        return expression

    for classifier in distribution.get_all("Classifier") or []:
        text = str(classifier)
        if text.startswith("License ::"):
            # "License :: OSI Approved :: BSD License" -> "BSD License"
            return text.rsplit("::", 1)[-1].strip()

    free_text = str(distribution.get("License") or "").strip()
    if free_text and "\n" not in free_text and len(free_text) <= 60:
        return free_text
    return ""


def packages(path: Path | None = None) -> list[Package]:
    """Return every dependency in requirements.txt, described."""
    source = path or REQUIREMENTS_PATH
    try:
        text = source.read_text(encoding="utf-8")
    except OSError:
        # A packaged build may not ship requirements.txt. The credits then
        # name the author and the assistants and say nothing about libraries,
        # which is better than refusing to open.
        return []

    return [
        describe(requirement.name, required=requirement.specifier, marker=requirement.marker)
        for requirement in parse_requirements(text)
    ]
