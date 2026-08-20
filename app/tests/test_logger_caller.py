"""The log must name the class and method that produced each record.

``logging`` records the function name but not the class, so two ``reload``
methods on different widgets look identical in the log - which is exactly the
situation where the log is being read.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.logs.logger import AppLogger, applogger, logging


@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    """Point the application logger at a temporary file for one test."""
    AppLogger.reset()
    AppLogger.configure(
        log_dir=tmp_path,
        console_level=logging.CRITICAL,
        file_level=logging.DEBUG,
        max_bytes=1_000_000,
        backup_count=1,
    )
    yield tmp_path / "datahub.log"
    AppLogger.reset()


def _read(log_file: Path) -> str:
    for handler in applogger.handlers:
        if hasattr(handler, "flush"):
            handler.flush()
    return log_file.read_text(encoding="utf-8")


class _Widget:
    """Stand-in for an application class that logs."""

    def reload(self, message: str) -> None:
        applogger.info(message)

    @classmethod
    def build(cls, message: str) -> None:
        applogger.info(message)


def _module_level_function(message: str) -> None:
    applogger.info(message)


def test_instance_method_is_reported_as_class_dot_method(log_file: Path) -> None:
    _Widget().reload("from an instance method")
    assert "_Widget.reload | from an instance method" in _read(log_file)


def test_classmethod_is_reported_through_cls(log_file: Path) -> None:
    _Widget.build("from a classmethod")
    assert "_Widget.build | from a classmethod" in _read(log_file)


def test_plain_function_is_reported_by_name(log_file: Path) -> None:
    _module_level_function("from a function")
    contents = _read(log_file)
    assert "_module_level_function | from a function" in contents
    # No spurious class prefix when there is no class.
    assert "._module_level_function |" not in contents


@pytest.mark.parametrize(
    "level_name", ["debug", "info", "warning", "error", "critical"]
)
def test_every_level_carries_the_caller(log_file: Path, level_name: str) -> None:
    """The attribute is added in one place, but every level must reach it."""
    getattr(applogger, level_name)(
        f"level {level_name}", show_dialog=False, raise_error=False
    )
    assert f"_caller_for_level | level {level_name}" not in _read(log_file)
    assert f"level {level_name}" in _read(log_file)


def test_exception_records_carry_the_caller(log_file: Path) -> None:
    class _Failing:
        def run(self) -> None:
            try:
                raise ValueError("boom")
            except ValueError:
                applogger.exception("handled", show_dialog=False, raise_error=False)

    _Failing().run()
    contents = _read(log_file)
    assert "_Failing.run | handled" in contents
    assert "ValueError: boom" in contents


def test_records_from_plain_logging_do_not_break_the_formatter(log_file: Path) -> None:
    """A library logging through the same logger must not raise a KeyError."""
    logging.getLogger("datahub").info("straight through logging")
    assert "straight through logging" in _read(log_file)
