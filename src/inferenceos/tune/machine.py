"""The machine check with no model loaded (step 5).

Only the parts something uses today are read: free graphics card memory, the power
source and plan, the graphics driver and the program version. The last check is saved
so the next one can say what changed (for example a driver update).
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from inferenceos.config import Settings
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.tune import MachineCheck, PowerPlan
from inferenceos.hardware.gpu import planning_vram_bytes
from inferenceos.logging_config import get_logger
from inferenceos.signature import hardware_signature

logger = get_logger("tune.machine")

#: Reads (used graphics memory in bytes or None, power source) right now.
LiveReader = Callable[[], tuple[int | None, PowerState]]
#: Reads the active power plan.
PowerPlanReader = Callable[[], PowerPlan]
#: Reads the program version.
VersionReader = Callable[[], str | None]

_GUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"\s*(?:\((.*)\))?"
)
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def parse_power_plan(text: str) -> PowerPlan:
    """Parse ``powercfg /getactivescheme`` output (any display language)."""
    match = _GUID_RE.search(text or "")
    if match is None:
        return PowerPlan()
    name = match.group(2).strip() if match.group(2) else None
    return PowerPlan(plan_id=match.group(1).lower(), name=name or None)


def read_power_plan() -> PowerPlan:
    """The active Windows power plan, or an empty plan when it can't be read."""
    if sys.platform != "win32":
        return PowerPlan()
    try:  # pragma: no cover - Windows only
        proc = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15.0,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        logger.warning("could not read the power plan: %s", exc)
        return PowerPlan()
    return parse_power_plan(proc.stdout or "")  # pragma: no cover


def read_live_state() -> tuple[int | None, PowerState]:
    """Used graphics memory for one unambiguous adapter, and the power source."""
    from inferenceos.telemetry.system import sample_system_state

    state = sample_system_state(include_gpu=True, top_processes=1)
    used: int | None = None
    if state.gpu_telemetry_available and len(state.gpu_memory) == 1:
        values = [
            int(memory.memory_used_bytes)
            for memory in state.gpu_memory
            if type(memory.memory_used_bytes) is int and memory.memory_used_bytes >= 0
        ]
        used = max(values) if values else None
    return used, state.power_state


def read_program_version(settings: Settings) -> str | None:
    """The installed llama.cpp version (from ``llama-server --version``)."""
    from inferenceos.backends.llama_cpp_toolchain import cli_version
    from inferenceos.backends.llama_server_toolchain import resolve_server_executable

    server = resolve_server_executable(settings.llama_cpp_dir)
    if not Path(server).is_file():
        return None
    return cli_version(server, timeout_s=30.0)


def check_machine(
    settings: Settings,
    *,
    hardware: HardwareProfile,
    live_reader: LiveReader | None = None,
    power_plan_reader: PowerPlanReader | None = None,
    version_reader: VersionReader | None = None,
) -> MachineCheck:
    """Check the machine now. Unreadable values stay empty; nothing is guessed."""
    notes: list[str] = []
    try:
        used, power = (live_reader or read_live_state)()
    except Exception as exc:  # noqa: BLE001 - an unreadable value stays unknown
        notes.append(f"live readings unavailable: {type(exc).__name__}: {exc}")
        used, power = None, PowerState.UNKNOWN
    try:
        plan = (power_plan_reader or read_power_plan)()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"power plan unavailable: {type(exc).__name__}: {exc}")
        plan = PowerPlan()
    try:
        version = (
            version_reader() if version_reader is not None else read_program_version(settings)
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"program version unavailable: {type(exc).__name__}: {exc}")
        version = None

    total = planning_vram_bytes(hardware)
    if total is None and (hardware.gpus or hardware.backend_availability):
        notes.append("graphics memory capacity unavailable or not bound to one selected device")
    if used is not None and (type(used) is not int or used < 0):
        notes.append("invalid graphics memory usage; free memory remains unknown")
        used = None
    free = max(total - used, 0) if total is not None and used is not None else None
    drivers = sorted(
        f"{gpu.name}: {gpu.driver_version}"
        for gpu in hardware.gpus
        if gpu.driver_version
    )
    return MachineCheck(
        hostname=hardware.hostname or socket.gethostname(),
        hardware_signature=hardware_signature(hardware),
        graphics_drivers=drivers,
        total_graphics_memory_bytes=total,
        used_graphics_memory_bytes=used,
        free_graphics_memory_bytes=free,
        power_source=power,
        power_plan=plan,
        program_version=version,
        notes=notes,
    )


def machine_changes(before: MachineCheck, now: MachineCheck) -> list[str]:
    """Plain-language list of what changed that could make a saved setup stale."""
    changes: list[str] = []
    if before.hardware_signature and now.hardware_signature != before.hardware_signature:
        changes.append("the hardware or Windows version changed")
    if before.graphics_drivers and now.graphics_drivers != before.graphics_drivers:
        changes.append(
            "the graphics driver changed ("
            + ", ".join(before.graphics_drivers)
            + " -> "
            + (", ".join(now.graphics_drivers) or "unknown")
            + ")"
        )
    if before.program_version and now.program_version != before.program_version:
        changes.append(
            f"the llama.cpp version changed ({before.program_version} -> "
            f"{now.program_version or 'unknown'})"
        )
    if (
        before.power_plan.plan_id
        and now.power_plan.plan_id
        and now.power_plan.plan_id != before.power_plan.plan_id
    ):
        changes.append(
            f"the power plan changed ({before.power_plan.name or before.power_plan.plan_id}"
            f" -> {now.power_plan.name or now.power_plan.plan_id})"
        )
    return changes


def last_check_path(settings: Settings) -> Path:
    return settings.data_dir / "machine_check.json"


def load_last_check(settings: Settings) -> MachineCheck | None:
    path = last_check_path(settings)
    if not path.is_file():
        return None
    try:
        return MachineCheck.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("ignoring unreadable saved machine check %s: %s", path, exc)
        return None


def save_check(settings: Settings, check: MachineCheck) -> Path:
    path = last_check_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(check.model_dump_json(indent=2), encoding="utf-8")
    return path


def gib(value: int | None) -> str:
    """Bytes as a short GB figure for messages."""
    return "unknown" if value is None else f"{value / 1024**3:.1f} GB"


__all__ = [
    "LiveReader",
    "PowerPlanReader",
    "VersionReader",
    "check_machine",
    "gib",
    "last_check_path",
    "load_last_check",
    "machine_changes",
    "parse_power_plan",
    "read_live_state",
    "read_power_plan",
    "read_program_version",
    "save_check",
]
