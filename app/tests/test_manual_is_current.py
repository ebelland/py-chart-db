"""The user manual's catalogues, checked against the code they describe.

A manual goes stale silently: nothing fails when a chart type is added and
@renderers still lists 23 of them, and the reader who counts is the one who
was trusting it. These tests pin only the parts that are *enumerations of
the code* - every renderer's name, every operation's name, and the count the
prose quotes - so prose stays free to be prose while the lists cannot drift.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

MANUAL = (
    Path(__file__).resolve().parents[2] / "docs" / "manual" / "user_manual.typ"
)


@pytest.fixture(scope="module")
def manual_text() -> str:
    if not MANUAL.exists():  # pragma: no cover - the manual ships with the repo
        pytest.skip(f"user manual not found at {MANUAL}")
    return MANUAL.read_text(encoding="utf-8")


def test_every_chart_type_is_listed(manual_text: str) -> None:
    from app.scanners.axis_renderer_scanner import renderers

    missing = sorted(
        str(renderer["value"])
        for renderer in renderers
        if f'[*{renderer["value"]}*]' not in manual_text
    )
    assert missing == [], f"chart types missing from the manual: {missing}"


def test_the_quoted_chart_type_count_is_right(manual_text: str) -> None:
    """The prose says "ships N chart types"; N is checked, not trusted."""
    from app.scanners.axis_renderer_scanner import renderers

    match = re.search(r"ships (\d+) chart types", manual_text)
    assert match, "the manual no longer states how many chart types ship"
    assert int(match.group(1)) == len(renderers)


def test_every_series_operation_is_listed(manual_text: str) -> None:
    from app.scanners.series_operation_scanner import series_operations

    missing = sorted(
        str(operation["value"])
        for operation in series_operations
        if f'[*{operation["value"]}*]' not in manual_text
    )
    assert missing == [], f"operations missing from the manual: {missing}"


def test_the_manual_calls_the_application_by_its_name(manual_text: str) -> None:
    """The rename left nine mentions of the old name behind once already."""
    from app import APP_NAME

    assert "Data Hub" not in manual_text
    assert APP_NAME in manual_text


def test_the_manual_states_the_shipped_version(manual_text: str) -> None:
    from app import APP_VERSION

    match = re.search(r'#let version = "([^"]+)"', manual_text)
    assert match, "the manual no longer declares a version"
    assert match.group(1) == APP_VERSION
