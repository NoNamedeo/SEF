"""In-memory log capture for the Streamlit diagnostics terminal."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

_MAX_LOG_RECORDS = 500
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}
_records_lock = threading.Lock()
_records: deque["UILogRecord"] = deque(maxlen=_MAX_LOG_RECORDS)
_handler: "_UILogHandler | None" = None


@dataclass(frozen=True, slots=True)
class UILogRecord:
    """Small immutable view of a Python log record."""

    created_at: float
    level: str
    logger: str
    message: str

    def as_terminal_line(self) -> str:
        """Return a terminal-like log line for compact display."""
        timestamp = time.strftime("%H:%M:%S", time.localtime(self.created_at))
        return f"{timestamp} {self.level:<7} {self.logger}: {self.message}"


class _UILogHandler(logging.Handler):
    """Thread-safe handler that stores bounded log records for Streamlit."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = UILogRecord(
                created_at=record.created,
                level=record.levelname,
                logger=record.name,
                message=self.format(record),
            )
            with _records_lock:
                _records.append(entry)
        except Exception:
            self.handleError(record)


def install_ui_log_capture(level: str = "INFO") -> None:
    """
    Attach the bounded UI log handler to SEF loggers.

    The handler is installed once per Streamlit process. Logger levels remain
    user-configurable through ``set_capture_level`` so the terminal can switch
    between DEBUG, INFO, WARNING and ERROR.
    """
    global _handler
    if _handler is None:
        _handler = _UILogHandler()
        _handler.setFormatter(logging.Formatter("%(message)s"))
        for logger_name in ("sef", "ui"):
            logging.getLogger(logger_name).addHandler(_handler)
    set_capture_level(level)


def set_capture_level(level: str) -> None:
    """Set the minimum level captured and emitted by UI-owned loggers."""
    normalized = level.upper()
    if normalized not in _LOG_LEVELS:
        raise ValueError(f"Unsupported log level: {level}")
    numeric_level = _LOG_LEVELS[normalized]
    if _handler is not None:
        _handler.setLevel(numeric_level)
    for logger_name in ("sef", "ui"):
        logging.getLogger(logger_name).setLevel(numeric_level)


def log_records(min_level: str = "INFO") -> list[UILogRecord]:
    """Return captured records filtered by the requested minimum level."""
    normalized = min_level.upper()
    numeric_level = _LOG_LEVELS.get(normalized, logging.INFO)
    with _records_lock:
        return [
            record
            for record in list(_records)
            if _LOG_LEVELS.get(record.level, logging.INFO) >= numeric_level
        ]


def clear_log_records() -> None:
    """Clear the UI diagnostics terminal."""
    with _records_lock:
        _records.clear()


def available_log_levels() -> tuple[str, ...]:
    """Return supported log levels in verbosity order."""
    return tuple(_LOG_LEVELS)
