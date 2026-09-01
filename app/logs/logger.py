"""Application logger with optional GUI dialogs and raise-on-log behaviour.

``applogger`` is a singleton :class:`DataHubLogger` that writes to a rotating
file, mirrors records to a Qt signal for the in-app log viewer, and can raise or
show a message box per call:

    applogger.warning("Recoverable")                    # log only
    applogger.error("Failed", show_dialog=False)        # log, no dialog
    applogger.critical("Unrecoverable")                    # dialog and raise

Defaults are per level: DEBUG/INFO/WARNING are silent, ERROR and above show a
dialog.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import traceback
from types import TracebackType
from typing import Any, Final, Self, cast

from PySide6.QtCore import QObject, QCoreApplication, QtMsgType, Signal, qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QMessageBox, QStatusBar, QWidget

_LOGGER_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "logs"
_LOGGER_NAME: Final[str] = "datahub"

# Where the rotating file handler writes by default.  Public so the log viewer
# can tell the user where the full log lives without rebuilding the path.
LOG_FILE: Final[Path] = _LOGGER_DIR / "datahub.log"

ExceptionInfo = tuple[type[BaseException], BaseException, TracebackType | None]

_DIALOG_DEFAULT_BY_LEVEL: Final[dict[int, bool]] = {
    logging.DEBUG: False,
    logging.INFO: False,
    logging.WARNING: False,
    logging.ERROR: True,
    logging.CRITICAL: True
}
_RAISE_DEFAULT_BY_LEVEL: Final[dict[int, bool]] = {
    logging.DEBUG: False,
    logging.INFO: False,
    logging.WARNING: False,
    logging.ERROR: False,
    logging.CRITICAL: False
}


class LoggedError(RuntimeError):
    """Raised when a log record is configured to raise after being emitted."""


class LogRecordEmitter(QObject):
    """Qt signal bridge emitted for every new application log record."""

    record_emitted = Signal(str, int, str)


log_events = LogRecordEmitter()


_UNKNOWN_CALLER: Final[str] = "<unknown>"


def _caller_qualname(*, stacklevel: int) -> str:
    """Return ``Class.method`` (or ``function``) of the code that logged.

    ``logging`` already records the function name, but not the class, so two
    ``reload`` methods on different widgets are indistinguishable in the log -
    which is exactly when the log matters.

    The class is recovered from the calling frame's ``self`` or ``cls`` local
    rather than from the qualified name, because ``__qualname__`` belongs to the
    function object and is not reachable from a frame.  ``sys._getframe`` is
    used deliberately: it is what ``logging`` itself uses, and it costs a few
    microseconds against ``inspect.stack``, which builds and reads source
    context for every frame.
    """
    try:
        frame = sys._getframe(stacklevel)
    except (ValueError, AttributeError):
        return _UNKNOWN_CALLER

    function_name = frame.f_code.co_name
    local_names = frame.f_locals

    owner = local_names.get("self", None)
    if owner is not None:
        return f"{type(owner).__name__}.{function_name}"

    owner_class = local_names.get("cls", None)
    if isinstance(owner_class, type):
        return f"{owner_class.__name__}.{function_name}"

    return function_name


class _CallerFilter(logging.Filter):
    """Guarantee every record has a ``caller`` attribute.

    Records emitted through the standard ``logging`` API - by libraries, or by
    ``logging.getLogger(...)`` elsewhere - never pass through
    ``_log_with_policy``, so the formatter would raise a KeyError on them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "caller"):
            record.caller = f"{record.module}.{record.funcName}"
        return True


class DataHubLogger(logging.Logger):
    """Logger with optional GUI dialog and raise behavior per log call.

    Usage examples:
        applogger.warning("Recoverable issue")
        applogger.warning("Show this", show_dialog=True)
        applogger.error("Fail this")  # default: dialog=True, raise=False
        applogger.error("Log only", show_dialog=False)
        applogger.error("Stop here", raise_error=True)  # raises LoggedError
    """

    # The six level methods below only exist to widen the standard signature
    # with show_dialog/raise_error; all the behaviour lives in
    # _log_with_policy.  They cannot be generated in a loop because the
    # stacklevel used to find the caller's class depends on a fixed call depth.

    _status_bar: QStatusBar | None = None

    def debug(
        self,
        msg: object,
        *args: object,
        show_dialog: bool | None = None,
        raise_error: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Log at DEBUG. Silent by default: no dialog, no raise."""
        self._log_with_policy(logging.DEBUG, msg, args, show_dialog, raise_error, **kwargs)

    def info(
        self,
        msg: object,
        *args: object,
        show_dialog: bool | None = None,
        raise_error: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Log at INFO. Silent by default: no dialog, no raise."""
        self._log_with_policy(logging.INFO, msg, args, show_dialog, raise_error, **kwargs)

    def warning(
        self,
        msg: object,
        *args: object,
        show_dialog: bool | None = None,
        raise_error: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Log at WARNING. Silent by default; pass show_dialog=True to surface it."""
        self._log_with_policy(logging.WARNING, msg, args, show_dialog, raise_error, **kwargs)

    def error(
        self,
        msg: object,
        *args: object,
        show_dialog: bool | None = None,
        raise_error: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Log at ERROR. Shows a dialog by default but does not raise."""
        self._log_with_policy(logging.ERROR, msg, args, show_dialog, raise_error, **kwargs)

    def critical(
        self,
        msg: object,
        *args: object,
        show_dialog: bool | None = None,
        raise_error: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Log at CRITICAL. Shows a dialog *and* raises LoggedError."""
        self._log_with_policy(logging.CRITICAL, msg, args, show_dialog, raise_error, **kwargs)

    def exception(
        self,
        msg: object,
        *args: object,
        exc_info: bool | BaseException | tuple[type[BaseException], BaseException, TracebackType | None] = True,
        show_dialog: bool | None = None,
        raise_error: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Log at ERROR with the active traceback attached.

        Unlike :meth:`critical` this does not re-raise, so it is the right call
        from inside an ``except`` block that intends to recover.
        """
        kwargs["exc_info"] = exc_info
        self._log_with_policy(logging.ERROR, msg, args, show_dialog, raise_error, **kwargs)

    def _log_with_policy(
        self,
        level: int,
        msg: object,
        args: tuple[object, ...],
        show_dialog: bool | None,
        raise_error: bool | None,
        **kwargs: Any,
    ) -> None:
        """Emit one record, then apply the dialog/raise policy for its level.

        ``show_dialog`` and ``raise_error`` travel to the handlers through
        ``extra`` rather than through a custom record class, so records created
        by plain ``logging`` calls stay valid (see :class:`_CallerFilter`).

        The raise happens *after* ``super()._log`` on purpose: the record must
        reach the file and the log viewer even when the caller is about to be
        unwound by the exception.
        """
        if not self.isEnabledFor(level):
            return

        extra = dict(cast(dict[str, Any], kwargs.pop("extra", {}) or {}))
        extra["show_dialog"] = (
            _DIALOG_DEFAULT_BY_LEVEL.get(level, False) if show_dialog is None else bool(show_dialog)
        )
        extra["raise_error"] = (
            _RAISE_DEFAULT_BY_LEVEL.get(level, False) if raise_error is None else bool(raise_error)
        )
        extra.setdefault("caller", _caller_qualname(stacklevel=3))
        kwargs["extra"] = extra

        super()._log(level, msg, args, **kwargs)

        if bool(extra["raise_error"]):
            raise LoggedError(str(msg) % args if args else str(msg))

        if self._status_bar is not None and level > logging.DEBUG:
            self._status_bar.showMessage(str(msg) % args if args else str(msg), 5000)

    def set_status_bar(self, status_bar: QStatusBar | None) -> None:
        """Set the status bar that receives log messages, or None to disable."""
        self._status_bar = status_bar


class QtLogEventHandler(logging.Handler):
    """Handler that emits a Qt signal for each formatted log record."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_events.record_emitted.emit(
                self.format(record),
                int(record.levelno),
                record.getMessage(),
            )
        except Exception:
            self.handleError(record)


class GuiLogHandler(logging.Handler):
    """Handler that can show QMessageBox dialogs for selected log records."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not bool(getattr(record, "show_dialog", False)):
                return

            # The message box this handler used to raise is disabled.  The
            # call was commented out but the four values it needed were still
            # computed on every record carrying show_dialog, so the work was
            # done and thrown away; only the computation is removed here, not
            # the decision.
            #
            # While it stays disabled, ``show_dialog=True`` on a log call has
            # no visible effect - dialogs come from app.utils.messages instead.
            return
        except Exception:
            self.handleError(record)


class AppLogger:
    """Singleton application logger factory and GUI error reporter."""

    _instance: Self | None = None
    _logger: DataHubLogger | None = None
    _configured: bool = False
    _hooks_installed: bool = False
    

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def configure(
        cls,
        log_dir: Path = _LOGGER_DIR,
        *,
        console_level: int = logging.INFO,
        file_level: int = logging.WARNING,
        gui_level: int = logging.DEBUG,
        max_bytes: int = 1_000_000,
        backup_count: int = 3,
    ) -> DataHubLogger:
        """Configure and return the singleton application logger.

        GUI dialog and raise behavior are controlled per call with optional
        logger method parameters:
            show_dialog: bool | None = None
            raise_error: bool | None = None

        Defaults:
            DEBUG, INFO, WARNING: show_dialog=False, raise_error=False
            ERROR, CRITICAL: show_dialog=True, raise_error=True
        """
        if cls._configured and cls._logger is not None:
            return cls._logger

        logging.setLoggerClass(DataHubLogger)

        log_dir.mkdir(parents=True, exist_ok=True)
        # LOG_FILE when the default directory is used; a relocated directory
        # (tests, a packaged build) keeps its own file.
        log_file: Path = log_dir / LOG_FILE.name
        logger = cast(DataHubLogger, logging.getLogger(_LOGGER_NAME))
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        formatter = logging.Formatter(
            fmt = "%(asctime)s | %(levelname)-8s | %(caller)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Attached to the logger, not to each handler: a filter on the logger
        # runs once per record and before any handler sees it.
        if not any(isinstance(existing, _CallerFilter) for existing in logger.filters):
            logger.addFilter(_CallerFilter())

        if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        has_file_handler = any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers)
        if not has_file_handler:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if not any(isinstance(handler, QtLogEventHandler) for handler in logger.handlers):
            event_handler = QtLogEventHandler()
            event_handler.setLevel(logging.DEBUG)
            event_handler.setFormatter(formatter)
            logger.addHandler(event_handler)
        if not any(isinstance(handler, GuiLogHandler) for handler in logger.handlers):
            gui_handler = GuiLogHandler()
            gui_handler.setLevel(gui_level)
            gui_handler.setFormatter(formatter)
            logger.addHandler(gui_handler)

        cls._logger = logger
        cls._configured = True
        logger.debug("Log file: %s", log_file, show_dialog=False, raise_error=False)
        return logger

    @classmethod
    def reset(cls) -> None:
        """Detach every handler and forget the singleton configuration.

        Why: ``configure`` is a no-op once the singleton exists, so anything that
        needs to point the logger at a different directory (tests, a relocated
        log folder) has to tear the current configuration down first.
        """
        logger = cls._logger or cast(DataHubLogger, logging.getLogger(_LOGGER_NAME))
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                # A handler that fails to close must not block reconfiguration.
                pass
        cls._logger = None
        cls._configured = False

    @classmethod
    def get_logger(cls) -> DataHubLogger:
        """Return the configured singleton logger, initialising it if needed."""
        if cls._logger is None:
            return cls.configure()
        return cls._logger



    @classmethod
    def install_exception_hooks(cls) -> None:
        """Install global Python and Qt hooks for fatal errors."""
        if cls._hooks_installed:
            return
        qInstallMessageHandler(cls._handle_qt_message)
        cls._hooks_installed = True


    @classmethod
    def _handle_qt_message(
        cls,
        mode: QtMsgType,
        context: object,
        message: str,
    ) -> None:
        """Route a Qt-internal message into the application log.

        Installed with ``qInstallMessageHandler`` so warnings raised by Qt
        itself - unknown QSS properties, layout complaints, painting errors -
        land in the same file as the application's own records instead of on
        stderr, where a packaged build would lose them.
        """
        del context
        logger = cls.get_logger()
        match mode:
            case QtMsgType.QtCriticalMsg| QtMsgType.QtFatalMsg:
                logger.critical("Qt critical: %s", message)
            case QtMsgType.QtWarningMsg:
                logger.warning("Qt warning: %s", message)
            case QtMsgType.QtInfoMsg:
                logger.info("Qt info: %s", message)
            case _:
                logger.debug("Qt debug: %s", message)

    @staticmethod
    def _application_instance() -> QApplication | None:
        """Return QApplication.instance() narrowed from QCoreApplication | None."""
        app = QCoreApplication.instance()
        if isinstance(app, QApplication):
            return app
        return None

    @staticmethod
    def _show_message_box(
        icon: QMessageBox.Icon,
        title: str,
        text: str,
        details: str,
        parent: QWidget | None = None,
    ) -> None:
        """Show a modal message box, or do nothing when there is no GUI.

        The no-GUI early return is what lets the same logging calls run under
        pytest and in headless scripts: without a QApplication, constructing a
        QMessageBox would abort the process.
        """
        app = AppLogger._application_instance()
        if app is None:
            return

        dialog = QMessageBox(parent)
        dialog.setIcon(icon)
        dialog.setWindowTitle(title)
        dialog.setText(text)
        dialog.setDetailedText(details)
        dialog.exec()

    @staticmethod
    def _close_application(exit_code: int) -> None:
        """Shut the application down, falling back to SystemExit when headless."""
        app = AppLogger._application_instance()
        if app is not None:
            app.closeAllWindows()
            app.exit(exit_code)
        else:
            raise SystemExit(exit_code)


def format_exception(exc_info: ExceptionInfo) -> str:
    """Return a formatted traceback string for an exception tuple."""
    return "".join(traceback.format_exception(*exc_info))


# Initialise ASAP at import time so module-level imports are always safe.
applogger: DataHubLogger = AppLogger.configure()
AppLogger.install_exception_hooks()


def get_logger() -> DataHubLogger:
    """Return the singleton application logger."""
    return AppLogger.get_logger()

