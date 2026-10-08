"""Active workload abstraction for runtime monitoring.

The current llama.cpp backend is process-per-generation: there is no long-lived
resident server and no OS-level suspend. The controlled workload therefore implements
the strongest *honest* semantics:

* ``pause`` stops accepting new work and **drains the in-flight generation to
  completion** (it never kills it);
* ``resume`` restarts accepting work and can be verified by observing a new real
  generation complete;
* the checkpoint covers runtime configuration + identity + resume information, not an
  OS/process memory snapshot.

Future resident/agent backends can implement the same interface with real suspend
semantics without changing the monitor.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from abc import ABC, abstractmethod

from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.config import Settings
from inferenceos.contracts.enums import RestorationStatus, WorkloadState
from inferenceos.contracts.experiment import (
    ApplicationResult,
    ExperimentChange,
    RestorationResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.optimization import CandidateAction
from inferenceos.contracts.runtime import WorkloadDescriptor, WorkloadSnapshot
from inferenceos.contracts.workload import ContextRequirement, WorkloadProfile
from inferenceos.cards.safety import ContextSafetyError
from inferenceos.experiment.executor import (
    ActionExecutor,
    action_fields,
    apply_to_configuration,
    supported_actions,
    validate_parameters,
)
from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
from inferenceos.profiling.workloads import choose_context_length, get_workload


class ContextRequirementViolation(RuntimeError):
    """A request would exceed the configured context under a known hard requirement.

    Refusing is the safe alternative to sending a request the backend would reject or
    silently truncate. It is only raised when the requirement is *known*; unknown
    requirements never block serving.
    """


def context_requirement_violation(
    configuration: RuntimeConfiguration, requirement: ContextRequirement | None
) -> str | None:
    """Return a refusal reason if the configured context is below a known hard requirement.

    ``None`` means serving is permitted: no requirement is declared, the requirement is
    unknown, or the configured context already satisfies it. The requirement is never
    inferred from the prompt, averages, or the model maximum.
    """
    if requirement is None:
        return None
    required = requirement.required_context_length()
    configured = configuration.context_length
    if required is None or configured is None:
        return None
    if configured < required:
        return (
            f"configured context {configured} is below the workload's hard context "
            f"requirement {required}; refusing to serve a request that could be rejected "
            "or silently truncated"
        )
    return None


def runtime_params_from_config(
    configuration: RuntimeConfiguration, *, n_predict: int, seed: int, ignore_eos: bool
) -> RuntimeParams:
    return RuntimeParams(
        threads=configuration.threads,
        threads_batch=configuration.threads_batch,
        engine_switches=configuration.engine_switches,
        batch_size=configuration.batch_size,
        ubatch_size=configuration.ubatch_size,
        gpu_offload_layers=(
            configuration.gpu_offload_layers
            if configuration.gpu_offload_layers is not None
            else 99
        ),
        use_mmap=configuration.use_mmap,
        context_length=configuration.context_length,
        flash_attention=configuration.flash_attention,
        device=configuration.device,
        kv_cache_type=configuration.kv_cache_type,
        kv_cache_type_k=configuration.kv_cache_type_k,
        kv_cache_type_v=configuration.kv_cache_type_v,
        placement_overrides=configuration.placement_overrides,
        ignore_eos=ignore_eos,
    )


def config_key(configuration: RuntimeConfiguration) -> str:
    parts = [f"{key}={value}" for key, value in sorted(configuration.as_dict().items()) if value is not None]
    return "|".join(parts) if parts else "default"


class ActiveWorkload(ABC):
    """Abstract active inference workload InferenceOS can observe and control."""

    @abstractmethod
    def descriptor(self) -> WorkloadDescriptor:
        """Identity + capabilities of the workload."""

    @abstractmethod
    def snapshot(self) -> WorkloadSnapshot:
        """A cheap point-in-time continuity/safety view."""

    @abstractmethod
    def configuration(self) -> RuntimeConfiguration:
        """The workload's current runtime configuration."""

    @abstractmethod
    def set_configuration(self, configuration: RuntimeConfiguration) -> None:
        """Update the runtime configuration used for subsequent requests."""

    @abstractmethod
    def pause(self) -> bool:
        """Stop accepting new work and drain the in-flight request; True if drained."""

    @abstractmethod
    def resume(self) -> bool:
        """Resume accepting work; True if the workload is running again."""

    @abstractmethod
    def start(self) -> None:
        """Start serving work (if owned)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the workload."""

    def target_pid(self) -> int | None:
        return None

    @abstractmethod
    def model_profile(self) -> ModelProfile:
        """The real model profile backing this workload."""

    @abstractmethod
    def workload_profile(self) -> WorkloadProfile:
        """The workload profile (for diagnosis/report context)."""

    def last_metrics(self) -> dict[str, float]:
        return {}

    def effective_threads(self) -> int | None:
        """Observed *effective* generation thread count, or ``None`` when unavailable.

        This is authoritative backend runtime state (never inferred from CPU topology).
        The default is unknown; backends that can observe it override this.
        """
        return None

    def effective_mmap(self) -> bool | None:
        """Observed *effective* weight-load mode, or ``None`` when unavailable.

        Authoritative backend evidence (never inferred from the configured value). The
        default is unknown; backends that can observe it override this.
        """
        return None

    def last_text(self) -> str | None:
        return None

    def verify_resume(self, *, previous_requests: int, timeout_s: float) -> bool:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self.snapshot().requests_served > previous_requests:
                return True
            if self.snapshot().terminated_externally:
                return False
            time.sleep(0.2)
        return self.snapshot().requests_served > previous_requests


class ControlledLlamaWorkload(ActiveWorkload):
    """A real, InferenceOS-owned workload serving real llama.cpp generations.

    Generations are sequential (one in flight at a time). A shift to a longer
    context/workload can be scheduled to exercise runtime workload-change detection.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        model: ModelProfile,
        workload_id: str = "short-short",
        shift_workload_id: str | None = None,
        shift_after_requests: int = 0,
        configuration: RuntimeConfiguration | None = None,
        shift_context_length: int | None = None,
        adapter: LlamaCppAdapter | None = None,
    ) -> None:
        self._settings = settings
        self._model = model
        self._workload = get_workload(workload_id)
        self._shift_workload = (
            get_workload(shift_workload_id) if shift_workload_id is not None else None
        )
        self._shift_after = max(shift_after_requests, 0)
        self._shift_context = shift_context_length
        self._adapter = adapter or LlamaCppAdapter(settings=settings)
        self._configuration = configuration or RuntimeConfiguration(
            context_length=choose_context_length(model),
            gpu_offload_layers=settings.default_gpu_layers,
        )
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._accepting = False
        self._in_flight = False
        self._in_flight_id: str | None = None
        self._requests = 0
        self._phase_long = False
        self._terminated = False
        self._last_error: str | None = None
        self._last_metrics: dict[str, float] = {}
        self._last_text: str | None = None
        self.pause_calls = 0
        self.resume_calls = 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #
    @property
    def adapter(self) -> LlamaCppAdapter:
        return self._adapter

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._accepting = True
        self._thread = threading.Thread(target=self._serve, name="inferenceos-workload", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._accepting = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=30.0)
        self._thread = None

    def _serve(self) -> None:
        while self._running:
            if not self._accepting:
                time.sleep(0.05)
                continue
            self._serve_one()
            time.sleep(0.05)

    def _serve_one(self) -> None:
        with self._lock:
            workload = self._workload
            if self._phase_long and self._shift_workload is not None:
                workload = self._shift_workload
            configuration = self._configuration
            request_id = f"req-{uuid.uuid4().hex[:10]}"
            self._in_flight = True
            self._in_flight_id = request_id
        try:
            violation = context_requirement_violation(
                configuration, workload.context_requirement
            )
            if violation is not None:
                raise ContextRequirementViolation(violation)
            params = runtime_params_from_config(
                configuration,
                n_predict=workload.n_predict,
                seed=workload.seed,
                ignore_eos=workload.ignore_eos,
            )
            self._adapter.load(self._model, params)
            result = self._adapter.generate(
                GenerationRequest(
                    prompt=workload.prompt,
                    n_predict=workload.n_predict,
                    seed=workload.seed,
                    temperature=workload.temperature,
                )
            )
            self._adapter.unload()
            metrics: dict[str, float] = {}
            for name, value in (
                ("decode_tokens_per_sec", result.decode_tokens_per_sec),
                ("total_latency_s", result.total_s),
                ("peak_rss_bytes", result.peak_rss_bytes),
                ("final_rss_bytes", result.final_rss_bytes),
                ("prompt_tokens_per_sec", result.prompt_tokens_per_sec),
            ):
                if value is not None:
                    metrics[name] = float(value)
            with self._lock:
                self._requests += 1
                self._last_metrics = metrics
                self._last_text = result.text
                if (
                    self._shift_after > 0
                    and self._requests >= self._shift_after
                    and not self._phase_long
                    and (self._shift_workload is not None or self._shift_context is not None)
                ):
                    self._phase_long = True
                    if self._shift_context is not None:
                        self._configuration = self._configuration.model_copy(
                            update={"context_length": self._shift_context}
                        )
                self._last_error = None
        except Exception as exc:  # noqa: BLE001 - a failed request must not crash the monitor
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._in_flight = False
                self._in_flight_id = None

    # ------------------------------------------------------------------ #
    # ActiveWorkload                                                       #
    # ------------------------------------------------------------------ #
    @property
    def model(self) -> ModelProfile:
        return self._model

    def descriptor(self) -> WorkloadDescriptor:
        with self._lock:
            configuration = self._configuration
            workload = self._workload
        return WorkloadDescriptor(
            workload_id=workload.workload_id,
            workload_kind=workload.kind,
            model_ref=self._model.name,
            model_path=self._model.path,
            backend_name="llama.cpp",
            configuration=configuration,
            context_length=configuration.context_length,
            prompts_ref=workload.to_workload_profile().prompt_set_ref,
            owned=True,
            controllable=True,
            pause_supported=True,
            checkpoint_supported=True,
            resume_supported=True,
            limitations=[
                "process-per-generation backend: pause drains the in-flight generation",
                "checkpoint covers runtime configuration + identity, not process memory",
            ],
        )

    def snapshot(self) -> WorkloadSnapshot:
        with self._lock:
            if self._terminated:
                state = WorkloadState.TERMINATED_EXTERNALLY
            elif not self._running:
                state = WorkloadState.STOPPED
            elif self._in_flight and not self._accepting:
                state = WorkloadState.DRAINING
            elif not self._accepting:
                state = WorkloadState.PAUSED
            else:
                state = WorkloadState.RUNNING
            return WorkloadSnapshot(
                state=state,
                running=self._running and not self._terminated,
                generating=self._in_flight,
                terminated_externally=self._terminated,
                safe_to_pause=not self._in_flight,
                can_resume=self._running and not self._terminated,
                requests_served=self._requests,
                in_flight_request_id=self._in_flight_id,
                last_error=self._last_error,
            )

    def configuration(self) -> RuntimeConfiguration:
        with self._lock:
            return self._configuration

    def set_configuration(self, configuration: RuntimeConfiguration) -> None:
        with self._lock:
            self._configuration = configuration

    def pause(self) -> bool:
        self.pause_calls += 1
        with self._lock:
            self._accepting = False
        deadline = time.perf_counter() + 180.0
        while time.perf_counter() < deadline:
            with self._lock:
                if not self._in_flight:
                    return True
            time.sleep(0.1)
        return False

    def resume(self) -> bool:
        self.resume_calls += 1
        with self._lock:
            if not self._running or self._terminated:
                return False
            self._accepting = True
        return True

    def target_pid(self) -> int | None:
        return os.getpid()

    def model_profile(self) -> ModelProfile:
        return self._model

    def workload_profile(self) -> WorkloadProfile:
        with self._lock:
            workload = self._workload
        return workload.to_workload_profile()

    def last_metrics(self) -> dict[str, float]:
        with self._lock:
            return dict(self._last_metrics)

    def last_text(self) -> str | None:
        with self._lock:
            return self._last_text


class RuntimeWorkloadExecutor(ActionExecutor):
    """Phase 7 :class:`ActionExecutor` bound to the active workload's configuration.

    Reuses Phase 7 parameter validation/application so the runtime layer cannot
    invent a new mutation path.
    """

    def __init__(self, workload: ActiveWorkload, *, backend_name: str = "llama.cpp") -> None:
        self._workload = workload
        self._backend_name = backend_name

    @property
    def name(self) -> str:
        return f"runtime-workload[{self._backend_name}]"

    def supports(self, action_id: str) -> bool:
        return action_id in supported_actions()

    def set_configuration(self, configuration: RuntimeConfiguration) -> None:
        self._workload.set_configuration(configuration)

    def current(self) -> RuntimeConfiguration:
        return self._workload.configuration()

    def apply(
        self, action: CandidateAction, parameters: dict[str, object]
    ) -> ApplicationResult:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        if not self.supports(action.action_id):
            return ApplicationResult(
                applied=False,
                action_id=action.action_id,
                supported=False,
                reason=f"action {action.action_id!r} has no safe runtime surface",
                applied_at=now,
            )
        ok, reason = validate_parameters(action, parameters)
        if not ok:
            return ApplicationResult(
                applied=False,
                action_id=action.action_id,
                supported=True,
                reason=reason,
                applied_at=now,
            )
        prior = self._workload.configuration()
        try:
            updated = apply_to_configuration(action.action_id, parameters, prior)
        except ContextSafetyError as exc:
            return ApplicationResult(
                applied=False,
                action_id=action.action_id,
                supported=True,
                reason=str(exc),
                applied_at=now,
            )
        self._workload.set_configuration(updated)
        field_names = _action_fields(action.action_id)
        changes = [
            ExperimentChange(
                parameter=field,
                prior_value=getattr(prior, field, None),
                applied_value=getattr(updated, field, None),
            )
            for field in field_names
        ]
        return ApplicationResult(
            applied=True,
            action_id=action.action_id,
            supported=True,
            changes=changes,
            reason="applied via registered runtime surface",
            applied_at=now,
        )

    def restore(self, expected: RuntimeConfiguration) -> RestorationResult:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        self._workload.set_configuration(expected)
        verified = self._workload.configuration() == expected
        return RestorationResult(
            attempted=True,
            status=RestorationStatus.RESTORED if verified else RestorationStatus.FAILED,
            expected_configuration=expected,
            restored_configuration=self._workload.configuration(),
            verified=verified,
            reason=(
                "baseline runtime configuration restored and verified"
                if verified
                else "restored configuration does not match the protected baseline"
            ),
            restored_at=now,
        )


def _action_fields(action_id: str) -> tuple[str, ...]:
    # Single source of truth: the card's program setting (see ``inferenceos.cards``).
    return action_fields(action_id)
