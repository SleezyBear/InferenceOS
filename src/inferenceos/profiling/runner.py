"""Baseline profiler: real repeated measurements with full run context.

Measurement only. For each (model, workload, mode) it runs the real backend through
the :class:`BackendAdapter`, samples real system telemetry over the inference
interval, and persists a self-contained :class:`ProfilingRun`. It does not detect
bottlenecks or choose configurations.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from inferenceos import __version__
from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.backends.llama_cpp_toolchain import backend_type_from_name
from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    BackendType,
    MeasurementMode,
    MetricQuality,
    ObservationSource,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.profiling import (
    BaselineReport,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunGroupStats,
    RunIdentity,
)
from inferenceos.contracts.system_state import (
    GPUEngineUtilization,
    GPUMemoryState,
    SystemState,
)
from inferenceos.hardware.profiler import detect_hardware_profile
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    BackendCapabilities,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.native import get_platform
from inferenceos.native.base import CounterSample, GlobalMemory
from inferenceos.profiling.statistics import summarize
from inferenceos.profiling.workloads import (
    BENCHMARK_SUITE_VERSION,
    BenchmarkWorkload,
    choose_context_length,
    workload_profiles,
)
from inferenceos.telemetry.counters import (
    AnswerTelemetry,
    CounterSampler,
    MeasurementMonitor,
    MonitorCleanupUnresolved,
    gpu_adapter_memory_peaks,
    gpu_engine_peaks,
    max_value,
)
from inferenceos.telemetry.system import (
    GPU_ADAPTER_MEMORY_COUNTER,
    CPU_PERFORMANCE_COUNTER,
    GPU_ENGINE_COUNTER,
    HARD_FAULT_COUNTER,
    compute_memory_pressure,
    observe_power_performance,
)

try:
    import psutil
except ImportError:  # pragma: no cover - hard dependency
    psutil = None  # type: ignore[assignment]

PROFILER_VERSION = f"{__version__}+phase3"

#: Reading cadence of the experimental measurement-scoped monitor (NextSample() is
#: instantaneous, so a short cadence costs no waiting; finer intervals leave less
#: ambiguity at answer edges).
SCOPED_INTERVAL_S = 0.1
#: Marks a scoped run's notes until finalize_runs has attributed its readings.
SCOPED_PENDING = "scoped monitor: paging pending finalization"
_SCOPED_KEEP = 2000

COUNTER_PATHS: tuple[str, ...] = (
    HARD_FAULT_COUNTER,
    GPU_ENGINE_COUNTER,
    GPU_ADAPTER_MEMORY_COUNTER,
    CPU_PERFORMANCE_COUNTER,
)

COLD_WARM_SEMANTICS = (
    "Under the current process-per-generation llama.cpp adapter, every measured run "
    "is a fresh llama-cli process that loads weights into memory. COLD labels a run "
    "with no preceding warm-up for that model/workload in this session (the model "
    "file is typically not yet in the OS page cache). WARM labels measured runs "
    "after an explicit warm-up run, so the model file is resident in the OS page "
    "cache; WARM does NOT mean a resident model process or reused KV/weights. "
    "Resident-process warm inference is not supported by this backend adapter."
)

LIMITATIONS: tuple[str, ...] = (
    "COLD runs are performed once per model/workload: the OS page cache cannot be "
    "evicted without elevation, so repetition of COLD runs would not be genuinely "
    "cold.",
    "GPU engine/adapter-memory and hard-page-fault telemetry are sampled over the "
    "inference interval; GPU total VRAM is not exposed by Windows counters and is "
    "reported as unavailable.",
    "system_cpu_percent is measured over the whole interval and therefore includes "
    "the profiler's own counter-sampling process overhead.",
    "TTFT is derived as prefill + one decode step (llama.cpp exposes no first-token "
    "timestamp); it is labelled DERIVED.",
)

_AGGREGATED_METRICS: tuple[str, ...] = (
    "model_load_time_s",
    "ttft_s",
    "prefill_latency_s",
    "decode_latency_s",
    "prompt_tokens_per_sec",
    "decode_tokens_per_sec",
    "total_latency_s",
    "process_wall_s",
    "prompt_tokens",
    "generated_tokens",
    "total_tokens",
    "peak_rss_bytes",
    "process_cpu_percent",
    "system_cpu_percent",
    "gpu_utilization_percent",
    "gpu_memory_used_bytes",
    "hard_page_faults_per_sec",
)


@dataclass
class ProfilerOptions:
    gpu_offload_layers: int = 99
    telemetry_sampling: bool = True
    log_verbosity: int = 4
    threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    use_mmap: bool | None = None
    flash_attention: bool | None = None
    kv_cache_type: str | None = None
    device: str | None = None
    threads_batch: int | None = None
    engine_switches: str | None = None
    kv_cache_type_k: str | None = None
    kv_cache_type_v: str | None = None
    placement_overrides: str | None = None


@dataclass(frozen=True)
class ProfileTask:
    model: ModelProfile
    workload: BenchmarkWorkload
    context_length: int


class BaselineProfiler:
    """Runs real repeated measurements and assembles a baseline report."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        adapter: BackendAdapter | None = None,
        hardware: HardwareProfile | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._adapter: BackendAdapter = adapter or LlamaCppAdapter(settings=self._settings)
        self._hardware = hardware
        self._capabilities: BackendCapabilities | None = None
        #: Experimental measurement-scoped monitor (review 10; off by default).
        self.measurement_monitor = bool(
            getattr(self._settings, "tune_experimental_measurement_monitor", False)
        )
        self._monitor: MeasurementMonitor | None = None
        self._monitor_load: object = None
        #: Monitor costs and per-answer sample counts (bounded): one "open" and one
        #: "close" per measurement, one "answer" per answer -- nothing counted twice.
        self.monitor_events: deque[dict[str, object]] = deque(maxlen=500)
        self.monitor_factory = MeasurementMonitor
        #: Provenance of every scoped answer (run id -> its monitor and state: "open",
        #: "closed" -- its monitor closed or restarted before finalization -- or
        #: "finalized"). Kept independently of the live monitor so a restart can never
        #: erase an unresolved answer; bounded to the most recent ``_SCOPED_KEEP``.
        self._scoped: dict[str, dict[str, object]] = {}
        #: Notes from the last :meth:`finalize_runs` (e.g. why paging is unknown).
        self.last_finalize_notes: list[str] = []
        #: Set when a scoped sampler's cleanup could not be confirmed: from then on no
        #: run starts (``run_once`` raises :class:`MonitorCleanupUnresolved`) and tune
        #: publishes nothing. Never cleared by this profiler.
        self.monitor_blocked: str | None = None

    @property
    def hardware(self) -> HardwareProfile:
        if self._hardware is None:
            self._hardware = detect_hardware_profile(self._settings)
        return self._hardware

    def capabilities(self) -> BackendCapabilities:
        if self._capabilities is None:
            self._capabilities = self._adapter.capabilities()
        return self._capabilities

    def make_tasks(
        self,
        models: Sequence[ModelProfile],
        workload_ids: Sequence[str] | None,
        context_override: int | None = None,
    ) -> list[ProfileTask]:
        from inferenceos.profiling.workloads import select_workloads

        tasks: list[ProfileTask] = []
        for model in models:
            workloads = select_workloads(workload_ids, model)
            context = choose_context_length(model, context_override)
            for workload in workloads:
                if workload.min_context_tokens <= context:
                    tasks.append(ProfileTask(model=model, workload=workload, context_length=context))
        return tasks

    def request_fresh_start(self) -> None:
        """Ask the adapter for a fresh load before the next run (if it supports it).
        A fresh load (a new measurement, or a repeat) also ends the current monitor."""
        self.end_measurement()
        fresh = getattr(self._adapter, "request_fresh_start", None)
        if callable(fresh):
            fresh()

    def end_measurement(self) -> None:
        """Close the measurement-scoped monitor, if one is open (idempotent)."""
        monitor, self._monitor = self._monitor, None
        if monitor is not None and not monitor.closed:
            cost = monitor.close()
            stop = getattr(monitor, "stop_record", {}) or {}
            self.monitor_events.append({"event": "close", "s": round(cost, 4), "answers": monitor.windows,
                                        "resolved": stop.get("resolved"), "errors": stop.get("errors"),
                                        "sampler_errors": getattr(monitor, "errors", [])})
            if stop.get("resolved") is False and self.monitor_blocked is None:
                self.monitor_blocked = ("a counter sampler's cleanup was not confirmed ("
                                        + "; ".join(str(e) for e in (stop.get("errors") or ["no detail"])) + ")")
                self.monitor_events.append({"event": "blocked", "reason": self.monitor_blocked})
        if monitor is not None:
            # Answers not finalized yet (warm-ups, an interrupted measurement, a restart
            # in the middle of one) are not forgotten: they are marked, and finalize_runs
            # treats them as unknown.
            for record in self._scoped.values():
                if record["monitor"] is monitor and record["state"] == "open":
                    record["state"] = "closed"

    def _stop_if_blocked(self) -> None:
        """The one check (reviewer, monitor round 3), made at entry and right after every
        step that can discover an unresolved cleanup (closing a scope): nothing further
        starts -- no load, sampler or answer -- while a sampler may still be running."""
        if self.monitor_blocked is not None:
            raise MonitorCleanupUnresolved(f"no further measurement: {self.monitor_blocked}")

    def _load_token(self) -> object:
        adapter = self._adapter
        return (id(getattr(adapter, "last_load", None)), getattr(adapter, "pid", None))

    def run_once(
        self,
        task: ProfileTask,
        *,
        mode: MeasurementMode,
        sequence: int,
        repetition: int,
        warmup: bool,
        options: ProfilerOptions,
    ) -> ProfilingRun:
        if psutil is None:  # pragma: no cover - hard dependency
            raise RuntimeError("psutil is required for profiling")
        self._stop_if_blocked()
        scoped = options.telemetry_sampling and self.measurement_monitor
        if self._monitor is not None and (warmup or not scoped):
            # A scope transition known before loading (a warm-up opens a new scope; a run
            # without the scoped monitor ends it): close the old scope first and check
            # its cleanup before anything is loaded.
            self.end_measurement()
            self._stop_if_blocked()
        adapter = self._adapter
        model = task.model
        workload = task.workload
        requested = _requested_config(task, options)
        timings: dict[str, float] = {}
        mark = time.perf_counter()
        adapter.load(
            model,
            RuntimeParams(
                gpu_offload_layers=options.gpu_offload_layers,
                context_length=task.context_length,
                threads=options.threads or 0,
                threads_batch=options.threads_batch,
                engine_switches=options.engine_switches,
                batch_size=options.batch_size,
                ubatch_size=options.ubatch_size,
                use_mmap=options.use_mmap,
                flash_attention=options.flash_attention,
                kv_cache_type=options.kv_cache_type,
                kv_cache_type_k=options.kv_cache_type_k,
                kv_cache_type_v=options.kv_cache_type_v,
                placement_overrides=options.placement_overrides,
                device=options.device,
                log_verbosity=options.log_verbosity,
                ignore_eos=task.workload.ignore_eos,
            ),
        )

        timings["load_s"] = time.perf_counter() - mark
        mark = time.perf_counter()
        sampler: CounterSampler | None = None
        monitor: MeasurementMonitor | None = None
        if scoped:
            token = self._load_token()
            if warmup or token != self._monitor_load or self._monitor is None:
                # A new measurement scope: after a warm-up's fresh load, after any
                # reload, or when none is open. Closing the old scope can discover an
                # unresolved cleanup (e.g. after an unexpected reload): stop here, before
                # any new sampler or answer.
                self.end_measurement()
                self._stop_if_blocked()
                self._monitor = self.monitor_factory(COUNTER_PATHS, interval_s=SCOPED_INTERVAL_S)
                self._monitor_load = token
                cost = self._monitor.open(first_sample_timeout_s=6.0)
                self.monitor_events.append({"event": "open", "s": round(cost, 4)})
            monitor = self._monitor
        elif options.telemetry_sampling:
            sampler = CounterSampler(COUNTER_PATHS, interval_s=0.25)
            sampler.start()
            sampler.wait_for_first_sample(timeout_s=6.0)
        timings["monitor_start_s"] = time.perf_counter() - mark

        psutil.cpu_percent(interval=None)
        memory_before = get_platform().global_memory_status()
        run_id = f"run-{sequence:04d}-{uuid.uuid4().hex[:8]}"
        started = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        started_wall = time.time()  # the scoped monitor places readings on the wall clock
        samples: list[CounterSample] = []
        try:
            result = adapter.generate(
                GenerationRequest(
                    prompt=workload.prompt,
                    n_predict=workload.n_predict,
                    seed=workload.seed,
                    temperature=workload.temperature,
                    ignore_eos=bool(workload.ignore_eos),
                )
            )
        except BaseException:
            # An exception or a cancellation ends the measurement's monitor too.
            if monitor is not None:
                self.end_measurement()
            raise
        finally:
            ended_perf = time.perf_counter()
            ended_wall = time.time()
            wall_s = ended_perf - started_perf
            timings["request_s"] = wall_s
            mark = time.perf_counter()
            if monitor is not None:
                if not monitor.closed:
                    # Nothing is decided here: the answer's interval is noted and its
                    # readings are attributed in finalize_runs (late readings kept).
                    monitor.note_answer(run_id, wall_start=started_wall, wall_end=ended_wall,
                                        perf_start=started_perf, perf_end=ended_perf)
                self._scoped[run_id] = {"monitor": monitor, "state": "closed" if monitor.closed else "open"}
                while len(self._scoped) > _SCOPED_KEEP:
                    del self._scoped[next(iter(self._scoped))]
                timings["monitor_collect_s"] = time.perf_counter() - mark
                timings["monitor_stop_s"] = 0.0  # paid once, at the measurement's close
                self.monitor_events.append({"event": "answer", "run_id": run_id,
                                            "answer_s": round(wall_s, 4),
                                            "collect_s": round(timings["monitor_collect_s"], 4)})
            else:
                samples = sampler.stop() if sampler is not None else []
                timings["monitor_stop_s"] = time.perf_counter() - mark
                if sampler is not None and getattr(sampler, "cleanup_confirmed", True) is False \
                        and self.monitor_blocked is None:
                    # The per-answer sampler's process or reader may still be running.
                    self.monitor_blocked = ("the per-answer counter sampler's cleanup was not confirmed "
                                            "(reader or process still running after the bounded wait)")
                    self.monitor_events.append({"event": "blocked", "reason": self.monitor_blocked})
        finished = datetime.now(timezone.utc)
        mark = time.perf_counter()
        telemetry = adapter.telemetry()
        device_info = adapter.last_device
        system_cpu = float(psutil.cpu_percent(interval=None))
        memory_after = get_platform().global_memory_status()
        timings["after_reads_s"] = time.perf_counter() - mark
        mark = time.perf_counter()
        adapter.unload()
        timings["unload_s"] = time.perf_counter() - mark
        first_word = getattr(result, "ttft_s", None)
        if isinstance(first_word, (int, float)) and first_word > 0:
            # Prompt processing plus the first word, as the engine timed it.
            timings["first_word_s"] = float(first_word)

        metrics = _build_metrics(
            result=result,
            samples=samples,
            telemetry_sampling=options.telemetry_sampling,
            system_cpu=system_cpu,
            memory_after=memory_after,
            wall_s=wall_s,
        )
        system_state = _build_system_state(
            samples=samples,
            captured_at=finished,
            system_cpu=system_cpu,
            memory_after=memory_after,
            hard_faults=metrics.hard_page_faults_per_sec,
        )
        quality = _metric_quality(metrics)
        backend_type = (
            backend_type_from_name(device_info.backend)
            if device_info is not None
            else requested.backend_type
        )
        config = requested.model_copy(
            update={
                "backend_type": backend_type,
                "backend_name": self.capabilities().name,
                "backend_version": self.capabilities().version,
                "device": telemetry.device,
                "gpu_layers_offloaded": telemetry.gpu_layers_offloaded,
                "gpu_layers_total": telemetry.gpu_layers_total,
                "observed": dict(telemetry.observed),
            }
        )

        workload_profile = workload.to_workload_profile()
        observation = Observation(
            observation_id=run_id,
            captured_at=finished,
            source=ObservationSource.HARNESS,
            mode=mode,
            backend=backend_type,
            device=telemetry.device,
            model_ref=model.name,
            workload_ref=workload_profile.workload_id,
            metrics=metrics,
            system_state=system_state,
            generated_text=result.text or None,
        )
        notes = (
            "warm-up run (excluded from aggregation)"
            if warmup
            else f"repetition {repetition} of measured {mode.value} runs"
        )
        if not options.telemetry_sampling:
            notes += "; external telemetry sampling disabled"
        if monitor is not None:
            notes += "; " + SCOPED_PENDING  # finalize_runs recognises scoped runs by this too
        _ = memory_before  # retained for symmetry/future interval-pressure diagnostics
        return ProfilingRun(
            identity=RunIdentity(
                run_id=run_id,
                profiler_version=PROFILER_VERSION,
                sequence=sequence,
                repetition=repetition,
                mode=mode,
                warmup=warmup,
                started_at=started,
                finished_at=finished,
            ),
            context=RunContext(
                hardware=self.hardware,
                model=model,
                workload=workload_profile,
                config=config,
            ),
            observation=observation,
            metric_quality=quality,
            notes=notes,
            phase_timings_s={key: round(value, 4) for key, value in timings.items()},
        )

    def finalize_runs(self, runs: Sequence[ProfilingRun]) -> list[ProfilingRun]:
        """Attribute the scoped monitor's readings to ``runs`` (a measurement's scored
        runs) and return them with their counter measures filled in.

        Runs that were not measured with the scoped monitor come back unchanged (the
        default per-answer monitor is not affected). Paging is per answer: measured, or
        unknown (missing, incomplete, ambiguous or unusable evidence -- never zero).
        Because a measurement's paging is the median of its answers, one unknown answer
        makes the whole measurement's paging unknown; otherwise the median of the known
        answers would hide it. The bounded wait for the reading that closes the last
        answer is reported separately (``monitor_finalize_s`` and a ``finalize`` event).
        """
        self.last_finalize_notes = []
        # Every scoped run supplied is inspected -- also those whose monitor closed or
        # restarted, and those this profiler has no record of any more.
        keys = [run.identity.run_id for run in runs
                if run.identity.run_id in self._scoped and self._scoped[run.identity.run_id]["state"] != "finalized"
                or (run.identity.run_id not in self._scoped and SCOPED_PENDING in (run.notes or ""))]
        if not keys:
            return list(runs)
        telemetry: dict[str, AnswerTelemetry] = {}
        lost: dict[str, str] = {}
        waited = 0.0
        monitors: dict[int, tuple[MeasurementMonitor, list[str]]] = {}
        for key in keys:
            record = self._scoped.get(key)
            if record is None:
                lost[key] = "no record of its monitor (not measured by this profiler, or too old)"
            elif record["state"] == "closed" or getattr(record["monitor"], "closed", True):
                lost[key] = "its monitor closed or restarted before the answer was finalized"
            else:
                monitor = record["monitor"]
                assert isinstance(monitor, MeasurementMonitor) or hasattr(monitor, "finalize")
                monitors.setdefault(id(monitor), (monitor, []))[1].append(key)  # type: ignore[arg-type]
        for monitor, mine in monitors.values():
            found, wait_s = monitor.finalize(mine)
            telemetry.update(found)
            waited += wait_s
        for key in keys:
            if key in self._scoped:
                self._scoped[key]["state"] = "finalized"
        unknown: list[str] = []
        for key in keys:
            item = telemetry.get(key)
            if item is None:
                unknown.append(f"{key}: {lost.get(key, 'no evidence was attributed')}")
            elif item.paging.status != "measured":
                unknown.append(f"{key}: {item.paging.reason}")
        measurement_reason = None
        if unknown:
            measurement_reason = (f"paging unknown for this measurement ({len(unknown)} of {len(keys)} "
                                  f"answer(s) without usable evidence; first: {unknown[0]})")
            self.last_finalize_notes.append(measurement_reason)
        out: list[ProfilingRun] = []
        last_key = keys[-1]
        for run in runs:
            key = run.identity.run_id
            if key not in keys:
                out.append(run)
                continue
            item = telemetry.get(key)
            known = measurement_reason is None and item is not None
            paging = item.paging.value if known and item is not None else None
            # Bounds always travel with a known answer; the single value only when exact
            # (reviewer, monitor round 3: an upper bound is never passed off as a reading).
            low = item.paging.lower_bound_rate if known and item is not None else None
            high = item.paging.upper_bound_rate if known and item is not None else None
            cpu = item.cpu_performance_percent if item is not None else None
            metrics = run.observation.metrics.model_copy(
                update={"hard_page_faults_per_sec": paging, "hard_page_faults_per_sec_lower": low,
                        "hard_page_faults_per_sec_upper": high, "cpu_performance_percent": cpu})
            quality = dict(run.metric_quality)
            quality["hard_page_faults_per_sec"] = MetricQuality.MEASURED if paging is not None else MetricQuality.UNAVAILABLE
            quality["cpu_performance_percent"] = MetricQuality.MEASURED if cpu is not None else MetricQuality.UNAVAILABLE
            notes = (run.notes or "").replace(SCOPED_PENDING, "scoped monitor: finalized")
            if paging is None and low is not None and high is not None:
                notes += f"; paging bounded: peak between {low:.1f}/s and {high:.1f}/s (bounds, not exact)"
            elif paging is None:
                notes += "; paging unknown (" + (item.paging.reason if item is not None and
                                                 item.paging.status != "measured" else
                                                 lost.get(key) or str(measurement_reason)) + ")"
            timings = dict(run.phase_timings_s)
            if key == last_key:
                timings["monitor_finalize_s"] = round(waited, 4)
            out.append(run.model_copy(update={
                "observation": run.observation.model_copy(update={"metrics": metrics}),
                "metric_quality": quality, "notes": notes, "phase_timings_s": timings,
            }))
        self.monitor_events.append({
            "event": "finalize", "wait_s": round(waited, 4),
            "measurement_paging": "unknown" if measurement_reason else "measured",
            "answers": [{"run_id": key, **(_trim(telemetry[key].record()) if key in telemetry else
                                           {"paging": {"status": "unknown", "reason": lost.get(key, "no evidence")}})}
                        for key in keys],
        })
        return out

    def profile(
        self,
        tasks: Sequence[ProfileTask],
        *,
        modes: Sequence[MeasurementMode],
        repetitions: int = 3,
        options: ProfilerOptions | None = None,
        suite_version: str = BENCHMARK_SUITE_VERSION,
    ) -> BaselineReport:
        options = options or ProfilerOptions()
        runs: list[ProfilingRun] = []
        sequence = 0
        for task in tasks:
            for mode in modes:
                if mode is MeasurementMode.COLD:
                    sequence += 1
                    runs.append(
                        self.run_once(
                            task,
                            mode=mode,
                            sequence=sequence,
                            repetition=0,
                            warmup=False,
                            options=options,
                        )
                    )
                else:
                    sequence += 1
                    runs.append(
                        self.run_once(
                            task,
                            mode=mode,
                            sequence=sequence,
                            repetition=0,
                            warmup=True,
                            options=options,
                        )
                    )
                    for repetition in range(max(repetitions, 1)):
                        sequence += 1
                        runs.append(
                            self.run_once(
                                task,
                                mode=mode,
                                sequence=sequence,
                                repetition=repetition,
                                warmup=False,
                                options=options,
                            )
                        )
        groups = _aggregate(runs)
        backend = runs[0].context.config if runs else ProfilingConfig()
        return BaselineReport(
            profiler_version=PROFILER_VERSION,
            suite_version=suite_version,
            generated_at=datetime.now(timezone.utc),
            hardware=self.hardware,
            backend=backend,
            workloads=workload_profiles([task.workload for task in tasks]),
            runs=runs,
            groups=groups,
            cold_warm_semantics=COLD_WARM_SEMANTICS,
            resident_warm_supported=False,
            limitations=list(LIMITATIONS),
        )


def _requested_config(task: ProfileTask, options: ProfilerOptions) -> ProfilingConfig:
    return ProfilingConfig(
        backend_name="llama.cpp",
        backend_type=(
            BackendType.VULKAN if options.gpu_offload_layers > 0 else BackendType.CPU
        ),
        device_selector=options.device,
        gpu_layers_offloaded=options.gpu_offload_layers,
        threads=options.threads,
        batch_size=options.batch_size,
        ubatch_size=options.ubatch_size,
        use_mmap=options.use_mmap,
        flash_attention=options.flash_attention,
        kv_cache_type=options.kv_cache_type,
        kv_cache_type_k=options.kv_cache_type_k,
        kv_cache_type_v=options.kv_cache_type_v,
        placement_overrides=options.placement_overrides,
        log_verbosity=options.log_verbosity,
        ignore_eos=task.workload.ignore_eos,
        context_length=task.context_length,
        n_predict=task.workload.n_predict,
        seed=task.workload.seed,
        temperature=task.workload.temperature,
        telemetry_sampling=options.telemetry_sampling,
    )


def _build_metrics(
    *,
    result: GenerationResult,
    samples: Sequence[CounterSample],
    telemetry_sampling: bool,
    system_cpu: float,
    memory_after: GlobalMemory | None,
    wall_s: float,
) -> RunMetrics:
    engine_peaks = gpu_engine_peaks(samples)
    gpu_util = _peak_utilization(engine_peaks)
    gpu_mem = _peak_adapter_memory(samples)
    hard_faults = max_value(samples, "pages input") if telemetry_sampling else None
    cpu_performance = mean_value(samples, "% processor performance") if telemetry_sampling else None
    total_tokens = (
        result.prompt_tokens + result.generated_tokens
        if result.prompt_tokens is not None and result.generated_tokens is not None
        else None
    )
    memory = memory_after
    return RunMetrics(
        model_load_time_s=result.model_load_s,
        ttft_s=result.ttft_s,
        prefill_latency_s=result.prefill_s,
        decode_latency_s=result.decode_s,
        prompt_tokens_per_sec=result.prompt_tokens_per_sec,
        decode_tokens_per_sec=result.decode_tokens_per_sec,
        total_latency_s=result.total_s,
        process_wall_s=result.process_wall_s if result.process_wall_s is not None else wall_s,
        prompt_tokens=result.prompt_tokens,
        generated_tokens=result.generated_tokens,
        total_tokens=total_tokens,
        peak_rss_bytes=result.peak_rss_bytes,
        final_rss_bytes=result.final_rss_bytes,
        process_cpu_percent=result.peak_process_cpu_percent,
        system_cpu_percent=system_cpu,
        cpu_seconds=result.cpu_seconds,
        cpu_seconds_per_token=_per_token(result.cpu_seconds, result.generated_tokens),
        context_switches_voluntary=result.context_switches_voluntary,
        context_switches_involuntary=result.context_switches_involuntary,
        context_switches_total=result.context_switches_total,
        context_switches_per_token=_per_token(
            result.context_switches_total, result.generated_tokens
        ),
        gpu_utilization_percent=gpu_util,
        gpu_memory_used_bytes=gpu_mem,
        gpu_memory_total_bytes=None,
        hard_page_faults=None,
        hard_page_faults_per_sec=hard_faults,
        total_ram_bytes=memory.total_physical_bytes if memory is not None else None,
        available_ram_bytes=memory.available_physical_bytes if memory is not None else None,
        committed_bytes=(
            max(memory.total_pagefile_bytes - memory.available_pagefile_bytes, 0)
            if memory is not None
            else None
        ),
        commit_limit_bytes=memory.total_pagefile_bytes if memory is not None else None,
        cpu_performance_percent=cpu_performance,
    )


def _trim(record: dict[str, object]) -> dict[str, object]:
    """Keep the evidence that matters in a bounded event: every interval with pages,
    every edge interval and every unusable one (all others read zero)."""
    paging = dict(record.get("paging") or {})  # type: ignore[call-overload]
    intervals = paging.get("intervals") or []
    paging["intervals"] = [iv for iv in intervals
                           if iv.get("place") == "boundary" or iv.get("count") or not iv.get("usable", True)]
    paging["intervals_total"] = len(intervals)
    return {**record, "paging": paging}


def mean_value(samples: Sequence[CounterSample], path_contains: str) -> float | None:
    """Average of the samples whose counter path contains ``path_contains``."""
    needle = path_contains.lower()
    values = [float(sample.value) for sample in samples if needle in sample.path.lower()]
    return sum(values) / len(values) if values else None


def _per_token(value: float | int | None, tokens: int | None) -> float | None:
    """Normalize a per-generation quantity by generated tokens.

    Returns ``None`` (unknown) when the value is unavailable or the token count is
    missing/zero, so a ratio is never fabricated.
    """
    if value is None or tokens is None or tokens <= 0:
        return None
    return float(value) / float(tokens)


def _peak_utilization(engine_peaks: Sequence[tuple[str, str, float]]) -> float | None:
    if not engine_peaks:
        return None
    compute = [value for _luid, engine, value in engine_peaks if engine == "compute"]
    if compute:
        return min(max(compute), 100.0)
    return min(max(value for _luid, _engine, value in engine_peaks), 100.0)


def _peak_adapter_memory(samples: Sequence[CounterSample]) -> int | None:
    peaks = gpu_adapter_memory_peaks(samples)
    return max((value for _luid, value in peaks), default=None)


def _build_system_state(
    *,
    samples: Sequence[CounterSample],
    captured_at: datetime,
    system_cpu: float,
    memory_after: GlobalMemory | None,
    hard_faults: float | None,
) -> SystemState:
    engine_peaks = gpu_engine_peaks(samples)
    mem_peaks = gpu_adapter_memory_peaks(samples)
    luids = sorted({luid for luid, _e, _v in engine_peaks} | {luid for luid, _v in mem_peaks})
    index = {luid: position for position, luid in enumerate(luids)}
    engines = [
        GPUEngineUtilization(
            device_index=index[luid],
            engine=engine,
            utilization_percent=min(value, 100.0),
        )
        for luid, engine, value in engine_peaks
    ]
    memory = [
        GPUMemoryState(device_index=index[luid], memory_used_bytes=value, memory_total_bytes=None)
        for luid, value in mem_peaks
    ]
    if memory_after is not None:
        total = memory_after.total_physical_bytes
        available = memory_after.available_physical_bytes
        committed = max(memory_after.total_pagefile_bytes - memory_after.available_pagefile_bytes, 0)
        commit_limit = memory_after.total_pagefile_bytes
    else:  # pragma: no cover - Windows API available in practice
        total = 0
        available = 0
        committed = None
        commit_limit = None
    power = observe_power_performance()
    return SystemState(
        captured_at=captured_at,
        total_ram_bytes=total,
        available_ram_bytes=available,
        committed_bytes=committed,
        commit_limit_bytes=commit_limit,
        memory_pressure=compute_memory_pressure(total, available, committed, commit_limit),
        cpu_percent_total=system_cpu,
        hard_page_faults_per_sec=hard_faults if hard_faults is not None else 0.0,
        processes=[],
        gpu_engines=engines,
        gpu_memory=memory,
        target_process=None,
        power_state=power.power_state,
        cpu_current_mhz=power.current_mhz,
        cpu_max_mhz=power.max_mhz,
        cpu_limit_mhz=power.limit_mhz,
        cpu_performance_ratio=power.ratio,
    )


def _metric_quality(metrics: RunMetrics) -> dict[str, MetricQuality]:
    quality: dict[str, MetricQuality] = {}
    for name in _AGGREGATED_METRICS:
        value = getattr(metrics, name)
        if value is None:
            quality[name] = MetricQuality.UNAVAILABLE
        elif name in {"ttft_s", "total_tokens"}:
            quality[name] = MetricQuality.DERIVED
        else:
            quality[name] = MetricQuality.MEASURED
    quality["gpu_memory_total_bytes"] = MetricQuality.UNAVAILABLE
    quality["hard_page_faults"] = MetricQuality.UNAVAILABLE
    return quality


def _metric_value(metrics: RunMetrics, name: str) -> float | None:
    value = getattr(metrics, name)
    if value is None:
        return None
    return float(value)


def _aggregate(runs: Sequence[ProfilingRun]) -> list[RunGroupStats]:
    grouped: dict[tuple[str, str, MeasurementMode], list[ProfilingRun]] = {}
    for run in runs:
        if run.identity.warmup:
            continue
        key = (
            run.context.model.name,
            run.context.workload.workload_id,
            run.identity.mode,
        )
        grouped.setdefault(key, []).append(run)
    groups: list[RunGroupStats] = []
    for (model_ref, workload_id, mode), members in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value)
    ):
        metrics = {
            name: summarize(
                name, [_metric_value(member.observation.metrics, name) for member in members]
            )
            for name in _AGGREGATED_METRICS
        }
        groups.append(
            RunGroupStats(
                model_ref=model_ref,
                workload_id=workload_id,
                mode=mode,
                repetitions=len(members),
                metrics=metrics,
            )
        )
    return groups
