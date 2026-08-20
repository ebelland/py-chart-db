"""Tests for the LaTeX rcParams guard."""
from __future__ import annotations

import pytest

from app.utils import mpl_latex

STYLE_TEXT = """
# a comment
figure.facecolor: white
text.usetex: True
text.latex.preamble: \\usepackage{amsmath}
pgf.texsystem: pdflatex
axes.grid: True
"""


@pytest.fixture
def without_latex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the "no TeX installed" branch regardless of the host."""
    monkeypatch.setattr(mpl_latex, "_missing_executables", lambda: ("latex", "dvipng"))


@pytest.fixture
def with_latex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the "TeX installed" branch regardless of the host."""
    monkeypatch.setattr(mpl_latex, "_missing_executables", lambda: ())


def test_key_classification() -> None:
    assert mpl_latex.is_latex_rcparam("text.usetex")
    assert mpl_latex.is_latex_rcparam("text.latex.preamble")
    assert mpl_latex.is_latex_rcparam("pgf.rcfonts")
    assert not mpl_latex.is_latex_rcparam("figure.facecolor")
    assert not mpl_latex.is_latex_rcparam("mathtext.fontset")


def test_rcparams_are_filtered_without_latex(without_latex: None) -> None:
    filtered = mpl_latex.filter_latex_rcparams(
        {"text.usetex": True, "pgf.texsystem": "pdflatex", "axes.grid": True}
    )
    assert filtered == {"axes.grid": True}
    assert mpl_latex.latex_unavailable_reason()


def test_rcparams_pass_through_with_latex(with_latex: None) -> None:
    params = {"text.usetex": True, "axes.grid": True}
    assert mpl_latex.filter_latex_rcparams(params) == params
    assert mpl_latex.latex_unavailable_reason() == ""


def test_style_text_entries_are_commented_out(without_latex: None) -> None:
    result = mpl_latex.filter_latex_style_text(STYLE_TEXT)

    assert "figure.facecolor: white" in result
    assert "axes.grid: True" in result
    for line in result.splitlines():
        if "usetex" in line or "pgf." in line or "latex.preamble" in line:
            assert line.lstrip().startswith("#")


def test_style_text_untouched_with_latex(with_latex: None) -> None:
    assert mpl_latex.filter_latex_style_text(STYLE_TEXT) == STYLE_TEXT
