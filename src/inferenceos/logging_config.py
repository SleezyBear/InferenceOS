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

# Bounded per-file size / backup count for operational logs. These are the fallback
# values when ``Settings`` cannot be read; the configured values take precedence.
_DEFAULT_LOG_MAX_BYTES = 8 * 1024 * 1024
_DEFAULT_LOG_BACKUP_COUNT = 3


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


def _log_rotation_limits(
    max_bytes: int | None, backup_count: int | None
) -> tuple[int, int]:
    """Resolve rotation limits, preferring explicit args then ``Settings``."""
    resolved_max = max_bytes
    resolved_backups = backup_count
    if resolved_max is None or resolved_backups is None:
        try:
            from inferenceos.config import Settings

            settings = Settings()
            if resolved_max is None:
                resolved_max = int(getattr(settings, "log_max_bytes", _DEFAULT_LOG_MAX_BYTES))
            if resolved_backups is None:
                resolved_backups = int(
                    getattr(settings, "log_backup_count", _DEFAULT_LOG_BACKUP_COUNT)
                )
        except Exception:  # noqa: BLE001 - configuration must never break logging
            resolved_max = _DEFAULT_LOG_MAX_BYTES if resolved_max is None else resolved_max
            resolved_backups = (
                _DEFAULT_LOG_BACKUP_COUNT if resolved_backups is None else resolved_backups
            )
    return max(int(resolved_max), 0), max(int(resolved_backups), 0)


def setup_logging(
    level: str = "INFO",
    *,
    log_file: Path | None = None,
    json_lines: bool = False,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> logging.Logger:
    """Configure and return the package logger.

    Idempotent: repeated calls replace the previously installed handlers. File logging
    is size-bounded and rotated via the standard ``RotatingFileHandler`` so unattended
    operation cannot grow a log without bound. A logging setup failure (for example a
    full disk) degrades to stderr-only logging and never prevents the command running.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.getLevelName(level.upper()))
    logger.propagate = False
    for existing in list(logger.handlers):
        try:
            existing.close()
        except Exception:  # noqa: BLE001 - closing a handler must not break setup
            pass
    logger.handlers.clear()

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(JsonLinesFormatter() if json_lines else TextFormatter())
    logger.addHandler(stream)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            resolved_max, resolved_backups = _log_rotation_limits(max_bytes, backup_count)
            if resolved_max > 0 and resolved_backups > 0:
                from logging.handlers import RotatingFileHandler

                file_handler: logging.Handler = RotatingFileHandler(
                    log_file,
                    maxBytes=resolved_max,
                    backupCount=resolved_backups,
                    encoding="utf-8",
                )
            else:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(JsonLinesFormatter())
            logger.addHandler(file_handler)
        except OSError as exc:
            # Disk/logging failures must not crash the command; fall back to stderr.
            print(
                f"warning: file logging disabled ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``inferenceos`` namespace."""
    if name == LOGGER_NAME or name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
