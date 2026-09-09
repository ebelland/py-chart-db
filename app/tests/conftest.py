"""Shared pytest fixtures and options for the Data Hub test suite.

Artifacts (databases, saved figures) go to the directory chosen by
``_artifacts_root``; set ``DHUB_TEST_ARTIFACTS`` to redirect them.
"""
# app/tests/conftest.py
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import matplotlib
import pytest

from app.data.sqlite_repo import SqliteRepo

matplotlib.use("Agg")


def _artifacts_root() -> Path:
    """Return the directory that receives test artifacts.

    Resolution order:
      1. ``DHUB_TEST_ARTIFACTS`` environment variable, when set;
      2. ``C:/bin/dhub`` on Windows, the fixed location used on the dev machine;
      3. an in-repo ``app/test_results`` folder;
      4. the system temp directory.

    Why: the previous implementation referenced ``root`` before assignment and
    only worked because the resulting NameError fell through to the fallback.
    """
    candidates: list[Path] = []

    override = os.environ.get("DHUB_TEST_ARTIFACTS", "").strip()
    if override:
        candidates.append(Path(override))
    if sys.platform.startswith("win"):
        candidates.append(Path("C:/bin/dhub"))
    candidates.append(Path(__file__).resolve().parents[1] / "test_results")
    candidates.append(Path(tempfile.gettempdir()) / "dhub_test_results")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            # Why: a mounted read-only folder passes mkdir(exist_ok=True) but
            # fails on the first write, so probe an actual create/unlink cycle.
            probe = candidate / ".write_probe"
            probe.touch()
            probe.unlink()
        except Exception:
            continue
        return candidate
    raise RuntimeError("No writable directory available for test artifacts")


@pytest.fixture(scope="session")
def test_results_dir() -> Path:
    base = _artifacts_root() / "dhub_tests"
    base.mkdir(parents=True, exist_ok=True)
    return base


@pytest.fixture(scope="session")
def plots_dir(test_results_dir: Path) -> Path:
    path = test_results_dir / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def tmp_db_path(test_results_dir: Path, request: pytest.FixtureRequest) -> Path:
    name = (request.node.name or "test").replace("/", "_").replace("\\", "_")
    return test_results_dir / f"{name}.dhub"


@pytest.fixture
def repo(tmp_db_path: Path) -> SqliteRepo:
    """A fresh, empty SqliteRepo at this test's own tmp_db_path.

    The base shape most test modules were redefining by hand: open, yield,
    close. ``tmp_db_path`` names its file after the test rather than a new
    temp path every run, so a database - and its WAL/SHM siblings - left
    over from an earlier local run has to be cleared before this test
    builds its own, or a stale table or link from that run leaks into this
    one's assertions. A module that needs more than this - seed data, a
    non-default constructor argument - defines its own ``repo`` fixture,
    which shadows this one for that module only; nothing here changes for
    every module that already does.
    """
    for suffix in (".dhub", ".dhub-wal", ".dhub-shm"):
        tmp_db_path.with_suffix(suffix).unlink(missing_ok=True)

    built = SqliteRepo(db_path=tmp_db_path)
    yield built
    built.close()


def pytest_addoption(parser: pytest.Parser) -> None:
    # Keep show-plots available for other tests.
    parser.addoption(
        "--show-plots",
        action="store_true",
        default=True,
        help="Save rendered matplotlib figures to the plots dir",
    )
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help="Run performance tests (may be slow)",
    )


@pytest.fixture(scope="session")
def show_plots(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--show-plots"))


@pytest.fixture(scope="session")
def run_perf(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-perf"))


class _TestLogger:
    def debug(self, *_args, **_kwargs) -> None: ...
    def info(self, *_args, **_kwargs) -> None: ...
    def warning(self, *_args, **_kwargs) -> None: ...
    def error(self, *_args, **_kwargs) -> None: ...
    def exception(self, *_args, **_kwargs) -> None: ...


def _install_logger_stub() -> None:
    module = ModuleType("app.utils.logger")
    module.__dict__.update(
        {
            "AppLogger": object,
        }
    )
    sys.modules["app.utils.logger"] = module


_install_logger_stub()


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole session.

    Qt allows exactly one, and creating a second aborts the process, so every
    test that needs widgets - or only a palette and a device pixel ratio -
    shares this rather than making its own.  Two modules used to define their
    own copy of this fixture, which shadowed it and duplicated the check below.

    Some environments ship a PySide6 whose widget constructors exist but reject
    their arguments; there the whole group is skipped rather than failing one
    test at a time with an error that says nothing about the cause.
    """
    qt = pytest.importorskip("PySide6.QtWidgets")

    try:
        app = qt.QApplication.instance() or qt.QApplication([])
        qt.QLineEdit(qt.QWidget())
    except Exception:  # pragma: no cover - depends on the installed PySide6
        pytest.skip("PySide6 widgets are not usable in this environment")

    yield app
