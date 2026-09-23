"""Logging configuration for InferenceOS.

Text logs go to stderr for humans; an optional file handler writes JSONL so that
later phases (audit/reconstruction) can parse structured events.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER_NAME = "inferenceos"


class TextFormatter(logging.Formatter):
    """Compact, human-readable single-line format."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
            timespec="milliseconds"
        )
        return f"{timestamp} {record.levelname:<7} {record.name}: {record.getMessage()}"


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    *,
    log_file: Path | None = None,
    json_lines: bool = False,
) -> logging.Logger:
    """Configure and return the package logger.

    Idempotent: repeated calls replace the previously installed handlers.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.getLevelName(level.upper()))
    logger.propagate = False
    logger.handlers.clear()

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(JsonLinesFormatter() if json_lines else TextFormatter())
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonLinesFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``inferenceos`` namespace."""
    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
