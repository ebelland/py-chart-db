"""Tests for the application logger configuration."""
from __future__ import annotations

from pathlib import Path

from app.logs.logger import AppLogger, applogger, logging


def test_logger_writes_to_file(tmp_path: Path) -> None:
    """
    Verifies that after configuration, a log record is persisted to disk.

    Notes:
    - Use file_level=INFO so an INFO message is guaranteed to be written.
    - Flush handlers to avoid buffering issues on some platforms.
    """
    # The application logger is already configured at import time, and
    # configure() is a no-op on an already-configured singleton, so tear the
    # current handlers down before pointing it at the temporary directory.
    AppLogger.reset()

    # Configure logger to write under a temporary directory
    logger = AppLogger.configure(
        log_dir=tmp_path,
        console_level=logging.CRITICAL,  # keep test output clean
        file_level=logging.INFO,
        max_bytes=1_000_000,
        backup_count=1,
    )

    # Ensure we use the same logger instance used by the app
    log = applogger
    assert log.name == logger.name

    # Emit a unique message
    msg = "pytest: file logging works"
    log.info(msg)

    # Flush file handlers to ensure data is physically written
    for h in log.handlers:
        if hasattr(h, "flush"):
            h.flush()

    log_file = tmp_path / "datahub.log"
    assert log_file.exists(), "Log file was not created"

    content = log_file.read_text(encoding="utf-8")
    assert msg in content, "Log message not found in log file"
