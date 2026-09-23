"""Storage characterization: media classification and path-backing resolution."""

from __future__ import annotations

from typing import Any

from inferenceos.contracts.enums import StorageMediaType
from inferenceos.hardware.storage import (
    build_storage_devices,
    classify_path,
    media_type_from_code,
)

DISK_DRIVES: list[dict[str, Any]] = [
    {
        "Index": 0,
        "Model": "Apple APPLE SSD AP0512 SCSI Disk Device",
        "MediaType": "Fixed hard disk media",
        "InterfaceType": "SCSI",
        "Size": 500_000_000_000,
    },
    {
        "Index": 1,
        "Model": "WD My Passport 259F USB Device",
        "MediaType": "Fixed hard disk media",
        "InterfaceType": "USB",
        "Size": 2_000_000_000_000,
    },
]

PHYSICAL_DISKS: list[dict[str, Any]] = [
    {"DeviceId": "0", "FriendlyName": "Apple APPLE SSD AP0512", "MediaType": 4, "BusType": 1},
    {"DeviceId": "1", "FriendlyName": "WD My Passport 259F", "MediaType": 3, "BusType": 7},
]

LOGICAL_DISKS: list[dict[str, Any]] = [
    {"DeviceID": "C:", "DriveType": 3, "Size": 110_000_000_000, "FreeSpace": 50_000_000_000},
    {"DeviceID": "D:", "DriveType": 2, "Size": 2_000_000_000_000, "FreeSpace": 1_500_000_000_000},
]

ASSOCIATIONS: list[dict[str, Any]] = [
    {"DiskIndex": 0, "Letter": "C:"},
    {"DiskIndex": 1, "Letter": "D:"},
]


def test_media_type_from_code() -> None:
    assert media_type_from_code(4) is StorageMediaType.SSD
    assert media_type_from_code(3) is StorageMediaType.HDD
    assert media_type_from_code(None) is StorageMediaType.UNKNOWN
    assert media_type_from_code(99) is StorageMediaType.UNKNOWN


def test_build_storage_devices_ssd_and_external_hdd() -> None:
    devices = build_storage_devices(DISK_DRIVES, PHYSICAL_DISKS, LOGICAL_DISKS, ASSOCIATIONS)
    assert len(devices) == 2
    internal, external = devices

    assert internal.name == "Apple APPLE SSD AP0512"
    assert internal.media_type is StorageMediaType.SSD
    assert internal.bus_type == "SCSI"
    assert internal.volumes == ["C:"]
    assert internal.free_bytes == 50_000_000_000
    assert internal.is_removable is False

    assert external.media_type is StorageMediaType.HDD
    assert external.bus_type == "USB"
    assert external.volumes == ["D:"]
    assert external.free_bytes == 1_500_000_000_000
    assert external.is_removable is True


def test_classify_path_resolves_backing_device() -> None:
    devices = build_storage_devices(DISK_DRIVES, PHYSICAL_DISKS, LOGICAL_DISKS, ASSOCIATIONS)
    internal = classify_path(r"C:\models\qwen.gguf", devices)
    assert internal is not None
    assert internal.media_type is StorageMediaType.SSD
    assert internal.is_external is False

    external = classify_path(r"D:\models\qwen.gguf", devices)
    assert external is not None
    assert external.media_type is StorageMediaType.HDD
    assert external.is_external is True


def test_classify_path_unknown_drive_returns_none() -> None:
    devices = build_storage_devices(DISK_DRIVES, PHYSICAL_DISKS, LOGICAL_DISKS, ASSOCIATIONS)
    assert classify_path(r"Z:\nowhere\x.gguf", devices) is None


def test_malformed_entries_degrade_gracefully() -> None:
    devices = build_storage_devices(
        [{"Model": "No Index Disk"}],
        [],
        [],
        [],
    )
    assert devices == []

    devices = build_storage_devices(
        [{"Index": 0, "Model": "Mystery Disk", "Size": 1000}],
        [],
        [],
        [],
    )
    assert len(devices) == 1
    assert devices[0].media_type is StorageMediaType.UNKNOWN
    assert devices[0].volumes == []
    assert devices[0].free_bytes is None
