"""Lightweight runtime telemetry sampling (always-on) vs. the full harness.

The runtime sampler is deliberately cheap: it reads psutil-backed system/process
statistics plus the workload's most recent generation metrics. It does **not** invoke
the expensive Phase 3 counter-sampling harness (``Get-Counter``/WMI), which is
reserved for pre-deployment profiling. Unavailable signals are absent rather than
fabricated.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from inferenceos.config import Settings
from inferenceos.contracts.runtime import RuntimeObservation
from inferenceos.contracts.system_state import (
    GPUEngineUtilization,
    GPUMemoryState,
    ProcessState,
    SystemState,
)
from inferenceos.native import get_platform
from inferenceos.native.base import VulkanDevice
from inferenceos.runtime.workload import ActiveWorkload, config_key
from inferenceos.telemetry.counters import CounterSampler
from inferenceos.telemetry.system import (
    GPU_ADAPTER_MEMORY_COUNTER,
    GPU_ENGINE_COUNTER,
    HARD_FAULT_COUNTER,
    aggregate_gpu_samples,
    compute_commit_pressure,
    compute_memory_pressure,
    compute_physical_pressure,
    latest_hard_faults_from_samples,
    observe_power_performance,
)

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
    """Cheap, always-on runtime telemetry over the live system (psutil-backed).

    The dominant cost on Windows is enumerating every process and reading its memory
    info (``psutil`` ``memory_info`` for ~200 processes costs ~0.2-0.5 s). That full
    process snapshot changes slowly, so it is refreshed at
    ``telemetry_process_scan_interval_s`` and reused between refreshes; the monitored
    target process is always refreshed with a single cheap query. Memory totals and CPU
    percentage are inexpensive and remain per-sample. This keeps the process information
    diagnosis relies on while making the common sample cheap.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        interval_s: float | None = None,
        include_gpu: bool = False,
        process_scan_interval_s: float | None = None,
        gpu_scan_interval_s: float | None = None,
        include_paging: bool = False,
        paging_scan_interval_s: float | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._interval_s = interval_s or self._settings.telemetry_interval_s
        self._include_gpu = include_gpu
        configured_scan = getattr(self._settings, "telemetry_process_scan_interval_s", 15.0)
        self._process_scan_interval_s = float(
            process_scan_interval_s if process_scan_interval_s is not None else configured_scan
        )
        # GPU cadence is low by default and further reduced by the monitor's modes; it is
        # never sampled per tick (that would recreate the earlier overhead problem).
        configured_gpu = getattr(self._settings, "telemetry_gpu_scan_interval_s", 5.0)
        self._gpu_scan_interval_s = float(
            gpu_scan_interval_s if gpu_scan_interval_s is not None else configured_gpu
        )
        # Paging cadence reuses the same Windows ``\Memory\Pages Input/sec`` counter
        # sampler as Phase 3, kept low and independent of GPU so paging is observable
        # even when GPU telemetry is disabled. Failure is explicit unavailable.
        self._include_paging = include_paging
        configured_paging = getattr(self._settings, "telemetry_paging_scan_interval_s", 5.0)
        self._paging_scan_interval_s = float(
            paging_scan_interval_s
            if paging_scan_interval_s is not None
            else configured_paging
        )
        self._process_cache: list[ProcessState] = []
        self._process_cache_at: float = 0.0
        # Target-process CPU rate (delta cpu-time / delta wall across samples). ``None``
        # until a second observation exists; never fabricated.
        self._target_cpu_prev: tuple[int, float, float] | None = None
        self._target_cpu_percent: float | None = None
        # GPU telemetry (reuses the existing Windows GPU Engine counter sampler).
        self._gpu_sampler: CounterSampler | None = None
        self._gpu_consumed = 0
        self._gpu_last_poll = 0.0
        self._gpu_vulkan_devices: list[VulkanDevice] = []
        self._gpu_engines: list[GPUEngineUtilization] = []
        self._gpu_memory: list[GPUMemoryState] = []
        self._gpu_available = False
        self._gpu_observed_at: datetime | None = None
        self._gpu_source: str | None = None
        # Paging telemetry (reuses the existing Windows ``\Memory\Pages Input/sec``
        # counter sampler). ``_paging_value`` stays ``None`` until a real reading is
        # seen, so an unavailable counter is never presented as a measured zero.
        self._paging_sampler: CounterSampler | None = None
        self._paging_consumed = 0
        self._paging_last_poll = 0.0
        self._paging_value: float | None = None
        self._paging_available = False
        self._paging_observed_at: datetime | None = None
        self._paging_source: str | None = None
        # Instrumentation: cost of each stage of the most recent sample (seconds).
        self.last_breakdown: dict[str, float] = {}
        # Set once ``close()`` runs so a lazily-started sampler can never survive it.
        self._closed = False
        if psutil is not None:
            # Prime the CPU percent delta so the first sample is meaningful.
            psutil.cpu_percent(interval=None)

    def set_gpu_scan_interval(self, seconds: float) -> None:
        """Adjust GPU sampling cadence (monitor mode control). ``<=0`` disables it.

        A disabled cadence stops new GPU polling; the last observed values are retained
        with their ``gpu_observed_at`` timestamp so a stale value is never presented as
        fresh.
        """
        self._gpu_scan_interval_s = float(seconds)

    @property
    def gpu_scan_interval_s(self) -> float:
        return self._gpu_scan_interval_s

    def close(self) -> None:
        """Stop any background counter samplers. Never raises."""
        self._closed = True
        for attr in ("_gpu_sampler", "_paging_sampler"):
            sampler = getattr(self, attr)
            if sampler is not None:
                try:
                    sampler.stop()
                except Exception:  # noqa: BLE001 - cleanup must not raise
                    pass
                setattr(self, attr, None)

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
        if state.physical_memory_pressure is not None:
            metrics["physical_memory_pressure"] = state.physical_memory_pressure
        if state.committed_bytes is not None:
            metrics["committed_bytes"] = float(state.committed_bytes)
        if state.commit_limit_bytes:
            metrics["commit_limit_bytes"] = float(state.commit_limit_bytes)
        if state.commit_pressure is not None:
            metrics["commit_pressure"] = state.commit_pressure
        # A measured zero is propagated (``paging_telemetry_available``), not dropped;
        # an unavailable counter leaves the metric absent rather than a fabricated 0.0.
        if state.paging_telemetry_available:
            metrics["hard_page_faults_per_sec"] = state.hard_page_faults_per_sec
        if self._target_cpu_percent is not None:
            # Measured from the monitored target process (not host CPU, not the monitor).
            metrics["process_cpu_percent"] = self._target_cpu_percent
        if state.gpu_telemetry_available:
            if state.gpu_engines:
                metrics["gpu_utilization_percent"] = max(
                    engine.utilization_percent for engine in state.gpu_engines
                )
            used = [m.memory_used_bytes for m in state.gpu_memory if m.memory_used_bytes]
            if used:
                metrics["gpu_memory_used_bytes"] = float(max(used))
        metrics.update(workload.last_metrics())
        try:
            effective_threads = workload.effective_threads()
        except Exception:  # noqa: BLE001 - an unavailable observation is unknown
            effective_threads = None
        return RuntimeObservation(
            observation_id=f"rtobs-{uuid.uuid4().hex[:10]}",
            sequence=sequence,
            captured_at=datetime.now(timezone.utc),
            workload_state=snapshot.state,
            generating=snapshot.generating,
            context_length=configuration.context_length,
            config_key=config_key(configuration),
            requested_threads=configuration.threads,
            effective_threads=effective_threads,
            workload_ref=descriptor.workload_id,
            model_ref=descriptor.model_ref,
            metrics=metrics,
            system_state=state,
            run_id=snapshot.in_flight_request_id,
            requests_served=int(snapshot.requests_served),
            condition=condition,
        )

    def set_expensive_scan_interval(self, seconds: float) -> None:
        """Adjust how often the expensive full process-table scan runs.

        The monitor uses this to reduce the expensive cadence in lower-intensity modes
        without touching the cheap per-sample observations (target RSS/CPU, memory, CPU,
        power, workload metrics).
        """
        if seconds > 0:
            self._process_scan_interval_s = float(seconds)

    @property
    def expensive_scan_interval_s(self) -> float:
        return self._process_scan_interval_s

    def _system_state(self, workload: ActiveWorkload) -> SystemState:
        breakdown: dict[str, float] = {}
        committed: int | None = None
        commit_limit: int | None = None
        if psutil is None:  # pragma: no cover - hard dependency
            total = available = 0
            cpu = 0.0
            processes: list[ProcessState] = []
            target = None
        else:
            started = time.perf_counter()
            virtual = psutil.virtual_memory()
            total = int(virtual.total)
            available = int(virtual.available)
            # Windows commit usage/limit reuse the same ``GlobalMemoryStatusEx`` source
            # as the Phase 3 system telemetry. If it is unavailable it stays ``None``
            # rather than being derived from physical RAM.
            memory = None
            try:
                memory = get_platform().global_memory_status()
            except Exception:  # noqa: BLE001 - unavailable memory info is not fatal
                memory = None
            if memory is not None:
                committed = max(
                    memory.total_pagefile_bytes - memory.available_pagefile_bytes, 0
                )
                commit_limit = memory.total_pagefile_bytes
            breakdown["memory"] = time.perf_counter() - started

            started = time.perf_counter()
            cpu = float(psutil.cpu_percent(interval=None))
            breakdown["cpu"] = time.perf_counter() - started

            target_pid = workload.target_pid()
            started = time.perf_counter()
            processes, target = self._processes_cached(target_pid)
            breakdown["processes"] = time.perf_counter() - started
        started = time.perf_counter()
        power = observe_power_performance()
        breakdown["power"] = time.perf_counter() - started

        started = time.perf_counter()
        gpu_engines, gpu_memory, gpu_available, gpu_observed_at, gpu_source = self._gpu_state(
            time.perf_counter()
        )
        breakdown["gpu"] = time.perf_counter() - started

        started = time.perf_counter()
        hard_faults, paging_available, paging_source = self._paging_state(time.perf_counter())
        breakdown["paging"] = time.perf_counter() - started
        self.last_breakdown = breakdown
        return SystemState(
            captured_at=datetime.now(timezone.utc),
            total_ram_bytes=total,
            available_ram_bytes=available,
            committed_bytes=committed,
            commit_limit_bytes=commit_limit,
            memory_pressure=compute_memory_pressure(total, available, committed, commit_limit),
            physical_memory_pressure=compute_physical_pressure(total, available),
            commit_pressure=compute_commit_pressure(committed, commit_limit),
            cpu_percent_total=cpu,
            hard_page_faults_per_sec=hard_faults if hard_faults is not None else 0.0,
            paging_telemetry_available=paging_available,
            paging_observed_at=self._paging_observed_at if paging_available else None,
            paging_source=paging_source,
            processes=processes,
            gpu_engines=gpu_engines,
            gpu_memory=gpu_memory,
            gpu_telemetry_available=gpu_available,
            gpu_observed_at=gpu_observed_at,
            gpu_source=gpu_source,
            target_process=target,
            power_state=power.power_state,
            cpu_current_mhz=power.current_mhz,
            cpu_max_mhz=power.max_mhz,
            cpu_limit_mhz=power.limit_mhz,
            cpu_performance_ratio=power.ratio,
        )

    def _paging_state(self, now: float) -> tuple[float | None, bool, str | None]:
        """Return the most recent hard-page-fault rate with explicit availability.

        The heavy counter read only happens at the configured cadence; between
        refreshes the cached value is returned. ``None`` means unavailable telemetry
        (there is no fabricated zero), while a returned ``0.0`` is a genuine measured
        zero.
        """
        if not self._include_paging or self._paging_scan_interval_s <= 0:
            return None, False, "unavailable: runtime paging telemetry disabled"
        if self._paging_sampler is None:
            self._start_paging_sampler()
        if self._paging_sampler is None:
            return (
                None,
                False,
                self._paging_source or "unavailable: paging sampler unavailable",
            )
        if (
            self._paging_observed_at is None
            or (now - self._paging_last_poll) >= self._paging_scan_interval_s
        ):
            try:
                samples = self._paging_sampler.snapshot()
            except Exception as exc:  # noqa: BLE001 - paging telemetry must never crash sampling
                self._paging_source = f"unavailable: {type(exc).__name__}: {exc}"
                samples = []
            new_samples = samples[self._paging_consumed :]
            self._paging_consumed = len(samples)
            self._paging_last_poll = now
            value = latest_hard_faults_from_samples(new_samples)
            if value is not None:
                self._paging_value = value
                self._paging_available = True
                self._paging_observed_at = datetime.now(timezone.utc)
                self._paging_source = "windows_memory_pages_input_counter"
        if not self._paging_available and self._paging_source is None:
            self._paging_source = "unavailable: no paging counter samples yet"
        return self._paging_value, self._paging_available, self._paging_source

    def _start_paging_sampler(self) -> None:
        try:
            sampler = CounterSampler([HARD_FAULT_COUNTER], interval_s=0.5)
            sampler.start()
            sampler.wait_until_ready(timeout_s=2.0)
            # Rate counters need at least one real reading; wait for it (bounded) so the
            # first runtime sample reports a measured value instead of a transient
            # "no samples yet". The wait is bounded and skipped for samplers that do not
            # expose it.
            wait_first = getattr(sampler, "wait_for_first_sample", None)
            if callable(wait_first):
                wait_first(timeout_s=3.0)
            if self._closed:
                # ``close()`` raced the lazy start; never leave the sampler alive.
                try:
                    sampler.stop()
                except Exception:  # noqa: BLE001 - cleanup must not raise
                    pass
                self._paging_sampler = None
                return
            self._paging_sampler = sampler
            self._paging_last_poll = 0.0
        except Exception as exc:  # noqa: BLE001 - unavailable telemetry is not fatal
            self._paging_source = f"unavailable: {type(exc).__name__}: {exc}"
            self._paging_sampler = None

    def _gpu_state(
        self, now: float
    ) -> tuple[
        list[GPUEngineUtilization], list[GPUMemoryState], bool, datetime | None, str | None
    ]:
        """Return the most recent GPU snapshot with explicit availability/freshness.

        The heavy counter read only happens at the mode-controlled cadence; between
        refreshes the cached values are returned together with ``gpu_observed_at`` so a
        stale value is never mistaken for a fresh observation.
        """
        if not self._include_gpu or self._gpu_scan_interval_s <= 0:
            return [], [], False, None, "unavailable: runtime gpu telemetry disabled"
        if self._gpu_sampler is None:
            self._start_gpu_sampler()
        if self._gpu_sampler is None:
            return [], [], False, None, self._gpu_source or "unavailable: gpu sampler unavailable"
        if self._gpu_observed_at is None or (now - self._gpu_last_poll) >= self._gpu_scan_interval_s:
            try:
                samples = self._gpu_sampler.snapshot()
            except Exception as exc:  # noqa: BLE001 - gpu telemetry must never crash sampling
                self._gpu_source = f"unavailable: {type(exc).__name__}: {exc}"
                samples = []
            new_samples = samples[self._gpu_consumed :]
            self._gpu_consumed = len(samples)
            self._gpu_last_poll = now
            if new_samples:
                aggregate = aggregate_gpu_samples(
                    new_samples, vulkan_devices=self._gpu_vulkan_devices
                )
                if aggregate.engines or aggregate.memory:
                    self._gpu_engines = list(aggregate.engines)
                    self._gpu_memory = list(aggregate.memory)
                    self._gpu_available = True
                    self._gpu_observed_at = datetime.now(timezone.utc)
                    self._gpu_source = "windows_gpu_engine_counters"
        if not self._gpu_available and self._gpu_source is None:
            self._gpu_source = "unavailable: no gpu counter samples yet"
        return (
            list(self._gpu_engines),
            list(self._gpu_memory),
            self._gpu_available,
            self._gpu_observed_at,
            self._gpu_source,
        )

    def _start_gpu_sampler(self) -> None:
        if self._closed:
            return
        try:
            self._gpu_vulkan_devices = list(get_platform().vulkan_enumerate())
        except Exception:  # noqa: BLE001 - identity is best-effort
            self._gpu_vulkan_devices = []
        try:
            sampler = CounterSampler(
                [GPU_ENGINE_COUNTER, GPU_ADAPTER_MEMORY_COUNTER], interval_s=0.5
            )
            sampler.start()
            sampler.wait_until_ready(timeout_s=2.0)
            if self._closed:
                try:
                    sampler.stop()
                except Exception:  # noqa: BLE001 - cleanup must not raise
                    pass
                self._gpu_sampler = None
                return
            self._gpu_sampler = sampler
            self._gpu_last_poll = 0.0
        except Exception as exc:  # noqa: BLE001 - unavailable telemetry is not fatal
            self._gpu_source = f"unavailable: {type(exc).__name__}: {exc}"
            self._gpu_sampler = None

    def _processes_cached(
        self, target_pid: int | None
    ) -> tuple[list[ProcessState], ProcessState | None]:
        """Return (top processes, fresh target process).

        The full top-process snapshot is refreshed at most once per
        ``process_scan_interval_s``; the target process is always measured fresh with a
        single cheap query and merged into the returned list.
        """
        now = time.perf_counter()
        if (
            not self._process_cache
            or (now - self._process_cache_at) >= self._process_scan_interval_s
        ):
            self._process_cache = self._top_processes()
            self._process_cache_at = now
        processes = self._process_cache
        target = self._process_state(target_pid) if target_pid is not None else None
        if target is not None:
            processes = [target] + [item for item in processes if item.pid != target.pid]
        return processes, target

    def _process_state(self, pid: int) -> ProcessState | None:
        if psutil is None:  # pragma: no cover - hard dependency
            return None
        try:
            process = psutil.Process(pid)
            with process.oneshot():
                memory = process.memory_info()
                name = process.name()
                cpu = process.cpu_times()
            # Compute the target process CPU% from the cpu-time delta between samples.
            # This measures the monitored target (never the monitor's own process) and
            # reuses the single cheap per-sample query, so it adds no full-process scan.
            cpu_seconds = float(getattr(cpu, "user", 0.0) or 0.0) + float(
                getattr(cpu, "system", 0.0) or 0.0
            )
            wall = time.perf_counter()
            previous = self._target_cpu_prev
            if previous is not None and previous[0] == pid and wall > previous[1]:
                delta_cpu = cpu_seconds - previous[2]
                self._target_cpu_percent = max(delta_cpu / (wall - previous[1]) * 100.0, 0.0)
            else:
                # First observation of this target: the rate is not yet known.
                self._target_cpu_percent = None
            self._target_cpu_prev = (pid, wall, cpu_seconds)
            return ProcessState(
                pid=pid,
                name=str(name or f"pid-{pid}"),
                cpu_percent=self._target_cpu_percent or 0.0,
                rss_bytes=int(memory.rss),
                vms_bytes=int(getattr(memory, "vms", 0) or 0),
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, TypeError, ValueError):
            return None

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

