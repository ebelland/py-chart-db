"""Choosing the application stylesheet from a file on disk.

The style setting used to name one of three shipped themes or a Qt plugin.
A ``file:`` key is the third form: the rest of it is an absolute path, and
what it points at is installed as the application stylesheet.

The interesting part is the *palette*. Our own themes name theirs in
``APP_STYLE_QSS`` — the "dark" theme is the Fluent sheet paired with the dark
palette — and a file the user chose cannot. Pairing a dark sheet with the
light palette is the "window belonging to neither theme" that ``themed_qss``
warns about: every widget the sheet does not cover stays light, and the icon
tinting reads the wrong way round. So the sheet is asked what it paints the
window, and the answer is checked here against the sheets that actually ship.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.styles import style as st

STYLES_DIR = st.STYLES_DIR


def _key(path: Path) -> str:
    return f"{st.FILE_STYLE_PREFIX}{path.resolve()}"


# ----------------------------------------------------------------------
# Reading the key
# ----------------------------------------------------------------------
def test_a_file_key_carries_its_path() -> None:
    assert st.style_file_path("file:/tmp/theme.qss") == Path("/tmp/theme.qss")


def test_a_user_directory_is_expanded() -> None:
    """Stored keys are absolute, but a hand-edited config.json may not be."""
    resolved = st.style_file_path("file:~/theme.qss")

    assert resolved is not None
    assert "~" not in str(resolved)


@pytest.mark.parametrize("key", ["dark", "automatic", "qt:Fusion", ""])
def test_every_other_kind_of_key_has_no_path(key: str) -> None:
    assert st.style_file_path(key) is None


# ----------------------------------------------------------------------
# Which palette a sheet asks for
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("sheet", "expected"),
    [
        ("AMOLED.qss", "dark"),
        ("ConsoleStyle.qss", "dark"),
        ("ElegantDark.qss", "dark"),
        ("ManjaroMix.qss", "dark"),
        ("MaterialDark.qss", "dark"),
        ("Aqua.qss", "light"),
        ("MacOS.qss", "light"),
        ("Ubuntu.qss", "light"),
    ],
)
def test_each_shipped_sheet_asks_for_the_palette_it_looks_like(
    sheet: str, expected: str
) -> None:
    path = STYLES_DIR / sheet
    if not path.is_file():
        pytest.skip(f"{sheet} is not installed")

    assert st.sheet_palette_key(path.read_text(encoding="utf-8")) == expected


def test_a_sheet_that_sets_no_window_background_reads_as_light() -> None:
    """NeonButtons styles buttons and nothing else, which is a real answer.

    Light is what an unstyled application already looks like, so a sheet that
    never says otherwise gets that rather than a guess.
    """
    assert st.sheet_palette_key("QPushButton { background-color: #101010; }") == "light"


def test_a_variant_rule_does_not_decide_the_palette() -> None:
    """The bug this heuristic had first.

    Taking the first background declaration anywhere read our own Fluent sheet
    as dark, off a ``QFrame[acrylicDark="true"]`` rule several hundred lines
    in — a real rule, but not one that says what the sheet looks like.
    """
    qss = """
        QMainWindow { background-color: #f0f0f0; }
        QFrame[acrylicDark="true"] { background: rgba(32, 32, 32, 0.68); }
    """

    assert st.sheet_palette_key(qss) == "light"


def test_the_broad_rule_wins_even_when_it_comes_second() -> None:
    qss = """
        QToolTip { background-color: #ffffff; }
        QWidget { background-color: rgb(20, 20, 20); }
    """

    assert st.sheet_palette_key(qss) == "dark"


@pytest.mark.parametrize(
    ("colour", "expected"),
    [
        ("#000", "dark"),
        ("#ffffff", "light"),
        ("rgb(0,0,0)", "dark"),
        ("rgba(240, 240, 240, 0.5)", "light"),
    ],
)
def test_the_spellings_a_sheet_may_use(colour: str, expected: str) -> None:
    assert st.sheet_palette_key("QWidget { background-color: %s; }" % colour) == expected


def test_a_comment_is_not_read_as_a_rule() -> None:
    qss = """
        /* QWidget { background-color: #000000; } */
        QWidget { background-color: #fafafa; }
    """

    assert st.sheet_palette_key(qss) == "light"


# ----------------------------------------------------------------------
# Resolving the preference
# ----------------------------------------------------------------------
def test_a_sheet_that_is_there_resolves_to_itself(tmp_path: Path) -> None:
    sheet = tmp_path / "theme.qss"
    sheet.write_text("QWidget { background-color: #202020; }", encoding="utf-8")

    assert st.resolve_app_style(_key(sheet)) == _key(sheet)


def test_a_sheet_that_is_gone_falls_back_to_automatic(tmp_path: Path) -> None:
    """Same class of thing as a Qt plugin uninstalled since it was chosen.

    The application has to start looking ordinary, not unstyled — this is read
    at startup, before there is any window to report an error in.
    """
    assert (
        st.resolve_app_style(_key(tmp_path / "never-existed.qss"))
        == st.APP_STYLE_AUTOMATIC
    )


def test_a_directory_is_not_a_stylesheet(tmp_path: Path) -> None:
    assert st.resolve_app_style(_key(tmp_path)) == st.APP_STYLE_AUTOMATIC


# ----------------------------------------------------------------------
# Applying it
# ----------------------------------------------------------------------
def test_a_chosen_sheet_is_installed_with_its_own_palette(qapp, tmp_path: Path) -> None:
    sheet = tmp_path / "night.qss"
    sheet.write_text(
        "QMainWindow { background-color: #050505; }\n"
        "QPushButton { color: #eeeeee; }",
        encoding="utf-8",
    )

    result = st.apply_platform_style(qapp, _key(sheet))

    assert result.qss_file == sheet.resolve()
    assert "QPushButton" in qapp.styleSheet()
    assert st._ACTIVE_THEME_IS_DARK is True


def test_a_light_sheet_leaves_the_light_palette(qapp, tmp_path: Path) -> None:
    sheet = tmp_path / "day.qss"
    sheet.write_text("QMainWindow { background-color: #fbfbfd; }", encoding="utf-8")

    st.apply_platform_style(qapp, _key(sheet))

    assert st._ACTIVE_THEME_IS_DARK is False


def test_a_sheet_that_disappears_does_not_leave_the_app_unstyled(
    qapp, tmp_path: Path
) -> None:
    """The race the apply path guards: resolve found it, then it went away."""
    sheet = tmp_path / "vanishing.qss"
    sheet.write_text("QWidget { background-color: #ffffff; }", encoding="utf-8")
    key = _key(sheet)
    sheet.unlink()

    result = st.apply_platform_style(qapp, key)

    # Whatever this desktop's automatic style is - the point is that it is not
    # the missing file, and that nothing raised.
    assert result.qss_file != sheet
