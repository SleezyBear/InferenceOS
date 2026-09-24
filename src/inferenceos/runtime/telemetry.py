"""Lightweight runtime telemetry sampling (always-on) vs. the full harness.

The runtime sampler is deliberately cheap: it reads psutil-backed system/process
statistics plus the workload's most recent generation metrics. It does **not** invoke
the expensive Phase 3 counter-sampling harness (``Get-Counter``/WMI), which is
reserved for pre-deployment profiling. Unavailable signals are absent rather than
fabricated.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from inferenceos.config import Settings
from inferenceos.contracts.runtime import RuntimeObservation
from inferenceos.contracts.system_state import ProcessState, SystemState
from inferenceos.runtime.workload import ActiveWorkload, config_key
from inferenceos.telemetry.system import compute_memory_pressure

try:  # pragma: no cover - psutil is a hard dependency in practice
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


class RuntimeTelemetrySampler(ABC):
    """Abstract runtime sampler (tests inject a deterministic implementation)."""

    @abstractmethod
    def sample(
        self, workload: ActiveWorkload, *, sequence: int, condition: str
    ) -> RuntimeObservation:
        ...


class SystemTelemetrySampler(RuntimeTelemetrySampler):
    """Cheap, always-on runtime telemetry over the live system (psutil-backed)."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        interval_s: float | None = None,
        include_gpu: bool = False,
    ) -> None:
        self._settings = settings or Settings()
        self._interval_s = interval_s or self._settings.telemetry_interval_s
        self._include_gpu = include_gpu
        if psutil is not None:
            # Prime the CPU percent delta so the first sample is meaningful.
            psutil.cpu_percent(interval=None)

    def sample(
        self, workload: ActiveWorkload, *, sequence: int, condition: str
    ) -> RuntimeObservation:
        snapshot = workload.snapshot()
        configuration = workload.configuration()
        descriptor = workload.descriptor()
        state = self._system_state(workload)
        metrics: dict[str, float] = {
            "memory_pressure": state.memory_pressure,
            "available_ram_bytes": float(state.available_ram_bytes),
            "total_ram_bytes": float(state.total_ram_bytes),
            "system_cpu_percent": state.cpu_percent_total,
        }
        if state.hard_page_faults_per_sec:
            metrics["hard_page_faults_per_sec"] = state.hard_page_faults_per_sec
        metrics.update(workload.last_metrics())
        return RuntimeObservation(
            observation_id=f"rtobs-{uuid.uuid4().hex[:10]}",
            sequence=sequence,
            captured_at=datetime.now(timezone.utc),
            workload_state=snapshot.state,
            generating=snapshot.generating,
            context_length=configuration.context_length,
            config_key=config_key(configuration),
            workload_ref=descriptor.workload_id,
            model_ref=descriptor.model_ref,
            metrics=metrics,
            system_state=state,
            run_id=snapshot.in_flight_request_id,
            condition=condition,
        )

    def _system_state(self, workload: ActiveWorkload) -> SystemState:
        if psutil is None:  # pragma: no cover - hard dependency
            total = available = 0
            cpu = 0.0
            processes: list[ProcessState] = []
            target = None
        else:
            virtual = psutil.virtual_memory()
            total = int(virtual.total)
            available = int(virtual.available)
            cpu = float(psutil.cpu_percent(interval=None))
            processes = self._top_processes()
            target_pid = workload.target_pid()
            target = next((item for item in processes if item.pid == target_pid), None)
        return SystemState(
            captured_at=datetime.now(timezone.utc),
            total_ram_bytes=total,
            available_ram_bytes=available,
            memory_pressure=compute_memory_pressure(total, available, None, None),
            cpu_percent_total=cpu,
            hard_page_faults_per_sec=0.0,
            processes=processes,
            gpu_engines=[],
            gpu_memory=[],
            target_process=target,
        )

    def _top_processes(self) -> list[ProcessState]:
        if psutil is None:  # pragma: no cover - hard dependency
            return []
        limit = max(self._settings.telemetry_top_processes, 1)
        entries: list[ProcessState] = []
        for process in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
            try:
                info = process.info
                memory = info.get("memory_info")
                if memory is None:
                    continue
                entries.append(
                    ProcessState(
                        pid=int(info["pid"]),
                        name=str(info.get("name") or f"pid-{info['pid']}"),
                        cpu_percent=float(info.get("cpu_percent") or 0.0),
                        rss_bytes=int(memory.rss),
                        vms_bytes=int(getattr(memory, "vms", 0) or 0),
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, TypeError, ValueError):
                continue
        entries.sort(key=lambda item: item.rss_bytes, reverse=True)
        return entries[:limit]
