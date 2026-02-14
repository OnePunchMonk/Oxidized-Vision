"""
OxidizedVision — Structured Logging.

Provides a centralized, structured logging configuration for the OxidizedVision
Python client. Uses Python's built-in logging with Rich handler for pretty
terminal output, plus optional JSON formatting for production environments.

Usage:
    from oxidizedvision.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Converting model", extra={"model": "unet", "format": "onnx"})
"""

import logging
import json
import sys
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


_CONFIGURED = False

# Module-level console for Rich output
_console = Console(stderr=True)


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for production/CI environments."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include any extra fields passed via `extra={}`
        for key in record.__dict__:
            if key not in (
                "name", "msg", "args", "created", "relativeCreated",
                "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "levelno", "msecs", "pathname", "filename", "module",
                "thread", "threadName", "processName", "process",
                "message", "levelname", "taskName",
            ):
                log_entry[key] = record.__dict__[key]

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])

        return json.dumps(log_entry)


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    verbose: bool = False,
) -> None:
    """
    Configure the root OxidizedVision logger.

    Call this once at application startup (e.g., in CLI entry point).
    Subsequent calls are no-ops.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, emit logs as JSON lines (for CI/production).
        verbose: If True, set level to DEBUG regardless of `level` param.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    effective_level = "DEBUG" if verbose else level.upper()

    root_logger = logging.getLogger("oxidizedvision")
    root_logger.setLevel(effective_level)

    # Remove any existing handlers
    root_logger.handlers.clear()

    if json_output:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(StructuredFormatter())
    else:
        handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )

    root_logger.addHandler(handler)

    # Prevent propagation to the root logger
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a structured logger for the given module name.

    If logging has not been configured yet, it will be auto-configured
    with default settings (INFO level, Rich pretty output).

    Args:
        name: Module name, typically ``__name__``.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    if not _CONFIGURED:
        configure_logging()

    return logging.getLogger(name)
