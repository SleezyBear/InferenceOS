"""OS identity characterization."""

from __future__ import annotations

from typing import Any, Mapping

from inferenceos.contracts.hardware import OSProfile


def build_os_profile(os_data: Mapping[str, Any], *, fallback_architecture: str = "64-bit") -> OSProfile:
    caption = _str_or_none(os_data.get("Caption")) or "Windows"
    raw_version = _str_or_none(os_data.get("Version")) or ""
    version = _major_minor(raw_version) or raw_version
    build = _str_or_none(os_data.get("BuildNumber")) or _build_from_version(raw_version)
    architecture = _str_or_none(os_data.get("OSArchitecture")) or fallback_architecture
    return OSProfile(
        caption=caption,
        version=version,
        build=build,
        architecture=architecture,
    )


def _major_minor(version: str) -> str | None:
    parts = version.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return None


def _build_from_version(version: str) -> str:
    parts = version.split(".")
    return parts[2] if len(parts) >= 3 else "0"


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
