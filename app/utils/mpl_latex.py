"""Guard against LaTeX-dependent Matplotlib rcParams on machines without LaTeX.

Matplotlib does not degrade gracefully here: with ``text.usetex: True`` and no
TeX distribution on PATH, every single text draw raises, so a style file written
on a machine that has LaTeX makes a chart unrenderable everywhere else.  The
failure surfaces deep inside the renderer as an unrelated-looking exception.

This module detects what is actually installed once per process and strips the
parameters that cannot work, leaving everything else untouched.

Public API:
    latex_available()                 - True when usetex can work
    filter_latex_rcparams(mapping)    - drop unusable keys from a dict
    filter_latex_style_text(text)     - same, for raw .mplstyle text
    latex_unavailable_reason()        - human-readable explanation, or ""
"""
from __future__ import annotations

import shutil
from functools import lru_cache
from typing import Any, Mapping

from app.logs.logger import applogger

# rcParams that require a working TeX installation.  Anything under these
# prefixes is dropped together with the exact keys.
LATEX_RCPARAM_KEYS: frozenset[str] = frozenset(
    {
        "text.usetex",
        "text.latex.preamble",
        "pgf.texsystem",
        "pgf.preamble",
        "pgf.rcfonts",
        "ps.usedistiller",
    }
)
LATEX_RCPARAM_PREFIXES: tuple[str, ...] = ("pgf.", "text.latex.")

# Executables Matplotlib shells out to for the usetex path with the Agg backend.
_REQUIRED_EXECUTABLES: tuple[str, ...] = ("latex", "dvipng")


@lru_cache(maxsize=1)
def _missing_executables() -> tuple[str, ...]:
    """Return the TeX executables that are not on PATH (cached per process)."""
    return tuple(name for name in _REQUIRED_EXECUTABLES if shutil.which(name) is None)


def latex_available() -> bool:
    """True when Matplotlib's usetex path can actually run."""
    return not _missing_executables()


def latex_unavailable_reason() -> str:
    """Return why LaTeX cannot be used, or an empty string when it can."""
    missing = _missing_executables()
    if not missing:
        return ""
    return ": missing " + ", ".join(missing)


def is_latex_rcparam(key: str) -> bool:
    """True when an rcParam key depends on a TeX installation."""
    name = str(key).strip()
    if name in LATEX_RCPARAM_KEYS:
        return True
    return any(name.startswith(prefix) for prefix in LATEX_RCPARAM_PREFIXES)


def filter_latex_rcparams(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return *params* without LaTeX-dependent keys when LaTeX is missing.

    When LaTeX is present the mapping is returned unchanged (as a plain dict).
    """
    result = dict(params)
    if latex_available():
        return result

    dropped = [key for key in result if is_latex_rcparam(key)]
    for key in dropped:
        del result[key]

    if dropped:
        applogger.warning(
            "%s; ignoring rcParams: %s",
            latex_unavailable_reason(),
            ", ".join(sorted(dropped)),
            show_dialog=False,
            raise_error=False,
        )
    return result


def filter_latex_style_text(text: str) -> str:
    """Return .mplstyle text with LaTeX-dependent lines commented out.

    Lines are commented rather than deleted so that a user who opens the style
    in the editor can still see what the original file asked for.
    """
    if latex_available() or not text:
        return text

    output: list[str] = []
    dropped: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            output.append(line)
            continue

        key = stripped.split(":", 1)[0].strip()
        if is_latex_rcparam(key):
            dropped.append(key)
            output.append(f"# [disabled: no LaTeX] {line}")
        else:
            output.append(line)

    if dropped:
        applogger.warning(
            "%s; disabled style entries: %s",
            latex_unavailable_reason(),
            ", ".join(sorted(set(dropped))),
            show_dialog=False,
            raise_error=False,
        )

    return "\n".join(output)
