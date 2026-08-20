"""
Generic static class discovery and dynamic class loading.

This module is intentionally independent from axis renderers, chart renderers,
series operation dialogs, or any other plugin-like class family.

It provides:
- AST-based discovery of classes that directly inherit from a named base class.
- Static extraction of literal class attributes.
- Cached dynamic class loading from disk.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.logs.logger import applogger


@lru_cache(maxsize=None)
def _load_class(file_path: str, class_name: str, mtime_ns: int, module_prefix: str):
    """
    Import ``class_name`` from ``file_path`` exactly once per file revision.

    ``mtime_ns`` is part of the cache key only. It makes the cache miss when the
    file changes on disk.
    """
    del mtime_ns  # cache key only

    path = Path(file_path)
    module_name = f"{module_prefix}_{path.stem}_{abs(hash(file_path))}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        applogger.error(f"Could not create import spec for: {path}.")
        return None

    module = importlib.util.module_from_spec(spec)

    # Register so that relative imports and module-level singletons behave
    # consistently during exec_module.
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        applogger.error(
            "Class %s failed to load from %s (%s: %s).",
            class_name,
            path,
            type(exc).__name__,
            exc,
            show_dialog=False,
            raise_error=False,
        )
        sys.modules.pop(module_name, None)
        return None

    cls = getattr(module, class_name, None)

    if cls is None:
        applogger.error(
            f"File '{path}' loaded as module '{module_name}', "
            f"but class '{class_name}' was not found."
        )
        return None

    if not isinstance(cls, type):
        applogger.error(
            f"'{class_name}' found in '{path}', but it is not a class "
            f"(got {type(cls)!r})."
        )
        return None

    return cls


def import_class_from_discovery_entry(
    entry: dict[str, Any],
    *,
    module_prefix: str = "_dynamic_discovered",
):
    """
    Return the class described by a discovery entry.

    Required entry keys:
        - path
        - name
    """
    file_path = Path(entry["path"]).resolve()

    try:
        mtime_ns = file_path.stat().st_mtime_ns
    except OSError:
        applogger.error(f"{file_path} not found.")
        return None

    return _load_class(
        str(file_path),
        str(entry["name"]),
        mtime_ns,
        module_prefix,
    )


def extract_class_string_attr(
    class_node: ast.ClassDef,
    attr_name: str,
) -> str | None:
    """
    Extract a class-level string attribute from a class body.

    Accepted:
        Name = "Scatter"
        Name: str = "Scatter"

    Rejected:
        Name = SOME_CONST
        Name = 123
    """
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    try:
                        value = ast.literal_eval(stmt.value)
                    except Exception:
                        return None
                    return value if isinstance(value, str) else None

        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == attr_name:
                if stmt.value is None:
                    return None
                try:
                    value = ast.literal_eval(stmt.value)
                except Exception:
                    return None
                return value if isinstance(value, str) else None

    return None


def extract_class_string_list_attr(
    class_node: ast.ClassDef,
    attr_name: str,
) -> list[str] | None:
    """
    Extract a class-level list[str] attribute from a class body.

    Accepted:
        RequiredRoles = ["x", "y"]
        RequiredRoles: list[str] = ["x", "y"]

    Rejected:
        RequiredRoles = ("x", "y")
        RequiredRoles = ["x", 1]
        RequiredRoles = SOME_CONST
    """
    for stmt in class_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    return literal_string_list(stmt.value)

        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == attr_name:
                if stmt.value is None:
                    return None
                return literal_string_list(stmt.value)

    return None


def literal_string_list(node: ast.AST) -> list[str] | None:
    """
    Return a literal list[str] from an AST node, else None.
    """
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None

    if not isinstance(value, list):
        return None

    if not all(isinstance(item, str) for item in value):
        return None

    return value


def directly_inherits_from(
    class_node: ast.ClassDef,
    base_class_name: str,
) -> bool:
    """
    Return True if class_node directly inherits from base_class_name.

    Strictly matches:
        class X(BaseClass):

    Also supports:
        class X(module.BaseClass):
    """
    for base in class_node.bases:
        if isinstance(base, ast.Name) and base.id == base_class_name:
            return True

        if isinstance(base, ast.Attribute) and base.attr == base_class_name:
            return True

    return False


def discover_classes(
    *,
    root: Path,
    base_class_name: str,
    value_attr: str | None = "Name",
    string_attrs: tuple[str, ...] = (),
    string_list_attrs: tuple[str, ...] = (),
    require_value_attr: bool = True,
    skip_init: bool = True,
) -> list[dict[str, Any]]:
    """
    Discover classes under root that directly inherit from base_class_name.

    Returns entries shaped like:
        {
            "name": class name,
            "path": file path as posix string,
            "value": value_attr content or class name,
            ...
        }
    """
    discovered: list[dict[str, Any]] = []

    if not root.is_dir():
        return discovered

    for path in root.rglob("*.py"):
        if skip_init and path.name == "__init__.py":
            continue

        src = path.read_text(encoding="utf-8", errors="replace")

        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            if not directly_inherits_from(node, base_class_name):
                continue

            value = None

            if value_attr is not None:
                value = extract_class_string_attr(node, value_attr)

                if require_value_attr and value is None:
                    continue

            entry: dict[str, Any] = {
                "name": node.name,
                "path": path.as_posix(),
                "value": value if value is not None else node.name,
            }

            for attr_name in string_attrs:
                entry[attr_name_to_key(attr_name)] = extract_class_string_attr(
                    node,
                    attr_name,
                )

            for attr_name in string_list_attrs:
                entry[attr_name_to_key(attr_name)] = extract_class_string_list_attr(
                    node,
                    attr_name,
                )

            discovered.append(entry)

    return discovered


def attr_name_to_key(attr_name: str) -> str:
    """
    Convert class attribute names to discovery dictionary keys.

    Examples:
        Description -> description
        RequiredRoles -> required
        OptionalRoles -> optional
    """
    explicit = {
        "Name": "value",
        "Description": "description",
        "RequiredRoles": "required",
        "OptionalRoles": "optional",
    }

    if attr_name in explicit:
        return explicit[attr_name]

    return attr_name[:1].lower() + attr_name[1:]