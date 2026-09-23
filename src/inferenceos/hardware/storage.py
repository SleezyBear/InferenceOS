"""Storage characterization.

Maps volumes back to their physical device and classifies media as SSD vs HDD so
later phases can distinguish model loading from internal SSD versus external
storage. Uses safe, read-only system inventory (WMI/CIM) — no destructive test and
no large benchmark writes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from inferenceos.contracts.enums import StorageMediaType
from inferenceos.contracts.hardware import StorageDevice
from inferenceos.native import windows

_MEDIA_BY_CODE: dict[int, StorageMediaType] = {
    3: StorageMediaType.HDD,
    4: StorageMediaType.SSD,
    5: StorageMediaType.SSD,  # SCM class memory; nearest SSD semantics
}

_BUS_BY_CODE: dict[int, str] = {
    0: "unknown",
    1: "SCSI",
    2: "ATAPI",
    3: "ATA",
    4: "1394",
    5: "SSA",
    6: "FibreChannel",
    7: "USB",
    8: "RAID",
    9: "iSCSI",
    10: "SAS",
    11: "SATA",
    12: "SD",
    13: "MMC",
    14: "Virtual",
    15: "FileBackedVirtual",
    16: "StorageSpaces",
    17: "NVMe",
    18: "SCM",
    19: "UFS",
}

_REMOVABLE_DRIVE_TYPES = {2}


@dataclass(frozen=True)
class PathStorageInfo:
    path: str
    volume: str | None
    device_name: str | None
    media_type: StorageMediaType
    is_removable: bool
    size_bytes: int | None
    free_bytes: int | None
    bus_type: str | None

    @property
    def is_external(self) -> bool:
        return self.is_removable or (self.bus_type or "").lower() == "usb"


def media_type_from_code(code: Any) -> StorageMediaType:
    parsed = _as_int(code)
    if parsed is None:
        return StorageMediaType.UNKNOWN
    return _MEDIA_BY_CODE.get(parsed, StorageMediaType.UNKNOWN)


def bus_type_from_code(code: Any) -> str | None:
    parsed = _as_int(code)
    if parsed is None:
        return None
    return _BUS_BY_CODE.get(parsed, f"code:{parsed}")


def build_storage_devices(
    disk_drives: Sequence[Mapping[str, Any]],
    physical_disks: Sequence[Mapping[str, Any]],
    logical_disks: Sequence[Mapping[str, Any]],
    associations: Sequence[Mapping[str, Any]],
) -> list[StorageDevice]:
    """Build physical-device-centric :class:`StorageDevice` entries."""
    physical_by_id = {str(disk.get("DeviceId")): disk for disk in physical_disks}
    logical_by_letter = {
        str(disk.get("DeviceID", "")).upper(): disk for disk in logical_disks
    }
    letters_by_disk: dict[int, list[str]] = {}
    for association in associations:
        disk_index = _as_int(association.get("DiskIndex"))
        letter = association.get("Letter")
        if disk_index is None or not letter:
            continue
        letters_by_disk.setdefault(disk_index, []).append(str(letter).upper())

    devices: list[StorageDevice] = []
    for drive in disk_drives:
        index = _as_int(drive.get("Index"))
        if index is None:
            continue
        physical = physical_by_id.get(str(index), {})
        volumes = sorted(set(letters_by_disk.get(index, [])))
        media_type = media_type_from_code(physical.get("MediaType"))
        bus = bus_type_from_code(physical.get("BusType")) or _str_or_none(
            drive.get("InterfaceType")
        )
        free_bytes: int | None = None
        removable = (bus or "").lower() == "usb"
        letter_free: list[int] = []
        for letter in volumes:
            logical = logical_by_letter.get(letter)
            if logical is None:
                continue
            if _as_int(logical.get("DriveType")) in _REMOVABLE_DRIVE_TYPES:
                removable = True
            free = _as_int(logical.get("FreeSpace"))
            if free is not None:
                letter_free.append(free)
        if letter_free:
            free_bytes = sum(letter_free)
        size = _as_int(drive.get("Size")) or _as_int(physical.get("Size")) or 0
        name = (
            _str_or_none(physical.get("FriendlyName"))
            or _clean_model(drive.get("Model"))
            or f"PhysicalDrive{index}"
        )
        devices.append(
            StorageDevice(
                name=name,
                media_type=media_type,
                bus_type=bus,
                size_bytes=size,
                free_bytes=free_bytes,
                is_removable=removable,
                volumes=volumes,
            )
        )
    return devices


def classify_path(path: Path | str, devices: Sequence[StorageDevice]) -> PathStorageInfo | None:
    """Identify which physical device backs a filesystem path."""
    resolved = os.path.abspath(str(path))
    drive = os.path.splitdrive(resolved)[0]
    volume = (drive + "\\").upper() if drive else None
    volume_key = drive.upper()
    for device in devices:
        if volume_key and volume_key in device.volumes:
            return PathStorageInfo(
                path=resolved,
                volume=volume,
                device_name=device.name,
                media_type=device.media_type,
                is_removable=device.is_removable,
                size_bytes=device.size_bytes,
                free_bytes=device.free_bytes,
                bus_type=device.bus_type,
            )
    return None


def detect_storage() -> list[StorageDevice]:
    snapshot = windows.collect_cim_snapshot()
    return build_storage_devices(
        snapshot.disk_drives,
        snapshot.physical_disks,
        snapshot.logical_disks,
        snapshot.associations,
    )


def _clean_model(model: Any) -> str | None:
    text = _str_or_none(model)
    if not text:
        return None
    for suffix in (" SCSI Disk Device", " ATA Device", " Disk Device"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.strip() or None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
