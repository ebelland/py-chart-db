"""Tests for the static AST-based axis-renderer scanner."""
from __future__ import annotations

from app.scanners.axis_renderer_scanner import (
    CHART_TYPE_ALIASES,
    _discover_axis_renderers,
    get_renderer,
    import_class_from_file,
    renderers,
)


def test_discovery_finds_renderers() -> None:
    """The scanner must find the renderers shipped in app/charts."""
    found = _discover_axis_renderers()
    assert found, "no axis renderers discovered"

    values = {entry["value"] for entry in found}
    assert {"Scatter Plot", "Time Series"} <= values


def test_discovery_matches_module_level_cache() -> None:
    """The module-level `renderers` list must match a fresh scan."""
    fresh = sorted(entry["value"] for entry in _discover_axis_renderers())
    cached = sorted(entry["value"] for entry in renderers)
    assert fresh == cached


def test_get_renderer_returns_none_for_unknown() -> None:
    """Unknown chart types resolve to None rather than raising."""
    assert get_renderer("No Such Chart Type") is None


def test_renamed_chart_types_still_resolve() -> None:
    """chart_type is stored in the database, so a rename must not orphan figures."""
    for old_name, new_name in CHART_TYPE_ALIASES.items():
        entry = get_renderer(old_name)
        assert entry is not None, f"{old_name!r} no longer resolves to a renderer"
        assert entry["value"] == new_name


def test_aliases_point_at_renderers_that_exist() -> None:
    """An alias to a name nobody implements would be a silent dead end."""
    known = {entry["value"] for entry in renderers}
    for old_name, new_name in CHART_TYPE_ALIASES.items():
        assert new_name in known, f"alias {old_name!r} -> {new_name!r} has no renderer"
        assert old_name not in known, f"alias {old_name!r} shadows a real renderer"


def test_import_class_from_file_is_cached() -> None:
    """Repeated imports must return the very same class object.

    Why: identity stability is what makes `isinstance` checks against a
    previously obtained renderer class work.
    """
    entry = get_renderer("Scatter Plot")
    assert entry is not None

    first = import_class_from_file(entry)
    second = import_class_from_file(entry)

    assert first is not None
    assert first is second
    assert first.Name == "Scatter Plot"
