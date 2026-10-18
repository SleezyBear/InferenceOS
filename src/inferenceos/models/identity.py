"""Cheap file-generation identity shared by metadata and measurement owners."""
from __future__ import annotations

from pathlib import Path


def model_file_signature(path: Path) -> tuple[str, int, int, int]:
    """Detect ordinary writes/replacements without hashing or warming model data.

    This is not attestation against hostile preservation of file metadata.
    """
    resolved = path.resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ino)
