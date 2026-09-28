"""Effective-thread observation tests.

Covers the smallest seam that captures the authoritative llama-server effective thread
count (the startup ``llama threadpool init, n_threads = N`` line) and its representation
as *observed runtime state* distinct from the *requested* thread configuration.

No thread optimization, no diagnosis change, and no candidate-eligibility change is
tested or required here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.backends.llama_cpp_toolchain import parse_effective_threads
from inferenceos.config import Settings
from inferenceos.contracts.enums import WorkloadState
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import ProfilingConfig, ProfilingRun
from inferenceos.contracts.runtime import RuntimeMonitorRun, RuntimeObservation
from inferenceos.interfaces.backend_adapter import (
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.runtime.approval import AutoRejectApprovalProvider
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.notification import CollectingNotificationSink
from inferenceos.runtime.resident_workload import ResidentActiveWorkload
from inferenceos.runtime.telemetry import SystemTelemetrySampler
from inferenceos.runtime.verification import ScriptedRuntimeSampler, ScriptedWorkload
from inferenceos.runtime.workload import ActiveWorkload
from inferenceos.selftest import sample_hardware, sample_model
from tests.resident_helpers import synthetic_model
from tests.runtime_helpers import make_runtime_policy

_THREAD_LINE = "0.01.381.490 I cmn          init: llama threadpool init, n_threads = {value}"


# --------------------------------------------------------------------------- #
# Parser: authoritative server evidence                                        #
# --------------------------------------------------------------------------- #
def test_parse_effective_threads_reads_explicit_request() -> None:
    # A. requested -t 12 -> the server reports the same effective value.
    assert parse_effective_threads(_THREAD_LINE.format(value=12)) == 12


def test_parse_effective_threads_reads_backend_selected_default() -> None:
    # B. no -t -> the backend selected 6 (physical cores) and said so.
    assert parse_effective_threads(_THREAD_LINE.format(value=6)) == 6


def test_parse_effective_threads_missing_evidence_is_unknown() -> None:
    # C. no evidence -> UNKNOWN, never inferred from CPU topology.
    assert parse_effective_threads("load_model: loading model") is None
    assert parse_effective_threads("") is None


def test_parse_effective_threads_generic_line_fallback() -> None:
    assert parse_effective_threads("system_info: n_threads = 8 / 8 |") == 8


def test_parse_effective_threads_zero_is_unknown() -> None:
    assert parse_effective_threads(_THREAD_LINE.format(value=0)) is None


def test_backend_telemetry_default_effective_threads_is_unknown() -> None:
    assert BackendTelemetry().effective_threads is None
    assert ProfilingConfig().effective_threads is None


# --------------------------------------------------------------------------- #
# Resident backend / workload seam                                             #
# --------------------------------------------------------------------------- #
class _TelemetryAdapter:
    """Minimal resident adapter double exposing observed backend telemetry."""

    def __init__(self, effective: int | None) -> None:
        self._effective = effective
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def pid(self) -> int | None:
        return 1234

    @property
    def identity(self) -> object | None:
        return None

    @property
    def last_generation(self) -> object | None:
        return None

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(self, request: GenerationRequest) -> GenerationResult:  # pragma: no cover
        raise RuntimeError("not used")

    def set_runtime_params(self, params: RuntimeParams) -> None:
        return None

    def telemetry(self) -> BackendTelemetry:
        return BackendTelemetry(
            device="vulkan-fake",
            gpu_layers_offloaded=8,
            gpu_layers_total=28,
            effective_threads=self._effective,
        )


class _NoTelemetryAdapter(_TelemetryAdapter):
    def __init__(self) -> None:
        super().__init__(None)

    def telemetry(self) -> BackendTelemetry:  # pragma: no cover - deliberately absent
        raise RuntimeError("telemetry unavailable")


def _resident_workload(adapter: object, *, threads: int | None) -> ResidentActiveWorkload:
    return ResidentActiveWorkload(
        settings=Settings(),
        model=synthetic_model(),
        adapter=adapter,  # type: ignore[arg-type]
        workload_id="short-short",
        configuration=RuntimeConfiguration(context_length=4096, gpu_offload_layers=8, threads=threads),
    )


def test_resident_workload_observes_effective_threads() -> None:
    # A/B. requested 12, effective 12; and requested unset, effective observed 6.
    explicit = _resident_workload(_TelemetryAdapter(12), threads=12)
    assert explicit.configuration().threads == 12
    assert explicit.effective_threads() == 12

    default = _resident_workload(_TelemetryAdapter(6), threads=None)
    assert default.configuration().threads is None
    assert default.effective_threads() == 6


def test_resident_workload_effective_threads_unknown_when_absent() -> None:
    # C. unsupported observable -> UNKNOWN (None), never inferred.
    workload = _resident_workload(_TelemetryAdapter(None), threads=12)
    assert workload.effective_threads() is None

    unavailable = _resident_workload(_NoTelemetryAdapter(), threads=12)
    assert unavailable.effective_threads() is None


# --------------------------------------------------------------------------- #
# Runtime observation representation                                           #
# --------------------------------------------------------------------------- #
class _EffectiveWorkload(ScriptedWorkload):
    def __init__(self, effective: int | None) -> None:
        super().__init__(configuration=RuntimeConfiguration(context_length=2048, gpu_offload_layers=8, threads=12))
        self._effective = effective

    def effective_threads(self) -> int | None:
        return self._effective


def test_sampler_records_effective_threads() -> None:
    sampler = SystemTelemetrySampler(settings=Settings(), include_gpu=False)
    observation = sampler.sample(_EffectiveWorkload(7), sequence=1, condition="observed")
    assert observation.effective_threads == 7


def test_sampler_records_unknown_when_unavailable() -> None:
    sampler = SystemTelemetrySampler(settings=Settings(), include_gpu=False)
    observation = sampler.sample(_EffectiveWorkload(None), sequence=1, condition="observed")
    assert observation.effective_threads is None


# --------------------------------------------------------------------------- #
# Runtime report: requested and effective remain distinguishable               #
# --------------------------------------------------------------------------- #
class _ConfigWorkload(ScriptedWorkload):
    def __init__(self, model: ModelProfile, configuration: RuntimeConfiguration) -> None:
        super().__init__(configuration=configuration)
        self._model = model

    def model_profile(self) -> ModelProfile:
        return self._model


def _to_run(*, threads: int | None, effective: int | None, offload: int = 8) -> ProfilingRun:
    model = sample_model().model_copy(update={"n_layers": 28})
    configuration = RuntimeConfiguration(
        context_length=4096, gpu_offload_layers=offload, threads=threads
    )
    workload = _ConfigWorkload(model, configuration)
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=ScriptedRuntimeSampler(lambda sequence, condition: {}),
        policy=make_runtime_policy(),
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
    )
    observation = RuntimeObservation(
        observation_id="obs-threads",
        captured_at=datetime.now(timezone.utc),
        workload_state=WorkloadState.RUNNING,
        context_length=4096,
        effective_threads=effective,
        metrics={"decode_tokens_per_sec": 20.0, "process_cpu_percent": 500.0},
    )
    return monitor._to_run(
        observation,
        condition="observed",
        model=model,
        workload=workload.workload_profile(),
        hardware=sample_hardware(),
    )


def test_to_run_preserves_requested_and_effective_threads() -> None:
    # A/D. requested 12 -> report carries requested=12 and effective=12, distinctly.
    run = _to_run(threads=12, effective=12)
    assert run.context.config.requested_threads == 12
    assert run.context.config.effective_threads == 12
    # The existing strategy input is deliberately unchanged (eligibility semantics).
    assert run.context.config.threads is None


def test_to_run_keeps_requested_unset_but_records_effective() -> None:
    # B/D. requested unset -> requested stays None, effective is the observed value.
    run = _to_run(threads=None, effective=6)
    assert run.context.config.requested_threads is None
    assert run.context.config.effective_threads == 6


def test_to_run_unknown_effective_stays_none() -> None:
    # C. unavailable observation stays UNKNOWN in the report.
    run = _to_run(threads=12, effective=None)
    assert run.context.config.requested_threads == 12
    assert run.context.config.effective_threads is None


def test_to_run_unrelated_settings_unchanged() -> None:
    # E. Preserving threads must not disturb offload/context representation.
    run = _to_run(threads=12, effective=12, offload=14)
    config = run.context.config
    assert config.gpu_layers_offloaded == 14
    assert config.gpu_layers_total == 28
    assert config.context_length == 4096
    assert config.threads is None  # unchanged strategy input
    assert run.observation.metrics.decode_tokens_per_sec == 20.0


# --------------------------------------------------------------------------- #
# Persisted runtime-monitor run evidence                                       #
# --------------------------------------------------------------------------- #
class _EffectiveSampler(ScriptedRuntimeSampler):
    def sample(
        self, workload: ActiveWorkload, *, sequence: int, condition: str
    ) -> RuntimeObservation:
        observation = super().sample(workload, sequence=sequence, condition=condition)
        return observation.model_copy(
            update={
                "requested_threads": workload.configuration().threads,
                "effective_threads": workload.effective_threads(),
            }
        )


def _run_monitor(effective: int | None) -> RuntimeMonitorRun:
    configuration = RuntimeConfiguration(context_length=2048, gpu_offload_layers=8, threads=12)
    workload = _EffectiveWorkload(effective)
    workload._configuration = configuration
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=_EffectiveSampler(lambda sequence, condition: {"decode_tokens_per_sec": 24.0}),
        policy=make_runtime_policy(),
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
    )
    return monitor.run(duration_s=2.0).run


def test_monitor_run_records_requested_and_effective_threads() -> None:
    run = _run_monitor(12)
    assert run.requested_threads == 12
    assert run.effective_threads == 12


def test_monitor_run_records_unknown_effective_when_unavailable() -> None:
    run = _run_monitor(None)
    assert run.requested_threads == 12
    assert run.effective_threads is None
