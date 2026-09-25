"""Phase 10R resident active workload.

Unlike the older process-per-generation workload, this keeps a **real resident
``llama-server``** running (Phase 5R infrastructure) and drives it with a real client
thread issuing sequential inference requests. The monitor therefore observes a target
process that is genuinely distinct from the monitor process, and each request exposes
real submission / first-token / completion timestamps.

The workload never kills an in-flight request: ``pause`` stops accepting new work and
drains the current request to completion before returning.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Protocol

from inferenceos.backends.process_control import process_identity
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.config import Settings
from inferenceos.contracts.enums import WorkloadState
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.lifecycle import ProcessIdentity, ResidentGenerationMeasurement
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.runtime import (
    RuntimeRequestRecord,
    WorkloadDescriptor,
    WorkloadSnapshot,
)
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.interfaces.backend_adapter import (
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.logging_config import get_logger
from inferenceos.profiling.workloads import BenchmarkWorkload, get_workload
from inferenceos.runtime.workload import ActiveWorkload, runtime_params_from_config

logger = get_logger("runtime.resident_workload")


class ResidentInferenceAdapter(Protocol):
    """The subset of the resident adapter the workload depends on (injectable)."""

    @property
    def is_loaded(self) -> bool: ...

    @property
    def pid(self) -> int | None: ...

    @property
    def identity(self) -> ProcessIdentity | None: ...

    @property
    def last_generation(self) -> ResidentGenerationMeasurement | None: ...

    def load(self, model: ModelProfile, params: RuntimeParams) -> None: ...

    def unload(self) -> None: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def set_runtime_params(self, params: RuntimeParams) -> None: ...



class ResidentActiveWorkload(ActiveWorkload):
    """A genuinely resident, actively-serving inference workload."""

    def __init__(
        self,
        *,
        settings: Settings,
        model: ModelProfile,
        adapter: ResidentInferenceAdapter | None = None,
        workload: BenchmarkWorkload | None = None,
        workload_id: str = "short-short",
        configuration: RuntimeConfiguration | None = None,
        request_gap_s: float = 0.05,
    ) -> None:
        self._settings = settings
        self._model = model
        self._workload = workload or get_workload(workload_id)
        self._adapter: ResidentInferenceAdapter = adapter or LlamaCppResidentAdapter(
            settings=settings
        )
        self._configuration = configuration or RuntimeConfiguration(
            context_length=2048, gpu_offload_layers=0
        )
        self._request_gap_s = max(request_gap_s, 0.0)

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._accepting = False
        self._terminated = False
        self._last_error: str | None = None
        self._last_metrics: dict[str, float] = {}
        self._last_text: str | None = None
        self._requests: list[RuntimeRequestRecord] = []
        self._completed = 0
        self._in_flight = False
        self._in_flight_id: str | None = None
        self._in_flight_started_at: datetime | None = None
        self._target_pid: int | None = None
        self._target_identity: ProcessIdentity | None = None
        self.pause_calls = 0
        self.resume_calls = 0

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #
    @property
    def adapter(self) -> ResidentInferenceAdapter:
        return self._adapter

    def target_identity(self) -> ProcessIdentity | None:
        """The resident target's PID-reuse-safe identity (None if not running)."""
        if self._adapter.is_loaded:
            return self._adapter.identity
        return None

    def target_pid(self) -> int | None:
        return self._adapter.pid

    def target_alive(self) -> bool:
        identity = self.target_identity()
        if identity is None:
            return False
        current = process_identity(identity.pid)
        return current is not None and abs(current.create_time - identity.create_time) < 0.01

    def request_records(self) -> list[RuntimeRequestRecord]:
        with self._lock:
            return list(self._requests)

    def last_request(self) -> RuntimeRequestRecord | None:
        with self._lock:
            return self._requests[-1] if self._requests else None

    def in_flight_request_id(self) -> str | None:
        with self._lock:
            return self._in_flight_id

    def completed_requests(self) -> int:
        with self._lock:
            return self._completed

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ensure_loaded()
        self._target_pid = self._adapter.pid
        self._target_identity = self._adapter.identity
        self._running = True
        self._accepting = True
        self._thread = threading.Thread(
            target=self._serve, name="inferenceos-resident-workload", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._accepting = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=320.0)
        self._thread = None
        try:
            if self._adapter.is_loaded:
                self._adapter.unload()
        except Exception as exc:  # noqa: BLE001 - cleanup must not crash
            logger.warning("resident workload unload failed: %s", exc)

    def _ensure_loaded(self) -> None:
        if self._adapter.is_loaded:
            return
        params = runtime_params_from_config(
            self._configuration,
            n_predict=self._workload.n_predict,
            seed=self._workload.seed,
            ignore_eos=self._workload.ignore_eos,
        )
        self._adapter.load(self._model, params)
        self._target_pid = self._adapter.pid
        self._target_identity = self._adapter.identity

    def _serve(self) -> None:
        while self._running:
            with self._lock:
                accepting = self._accepting
            if not accepting:
                time.sleep(0.02)
                continue
            self._serve_one()
            time.sleep(self._request_gap_s)

    def _serve_one(self) -> None:
        request_id = f"rreq-{uuid.uuid4().hex[:10]}"
        submitted = datetime.now(timezone.utc)
        with self._lock:
            self._in_flight = True
            self._in_flight_id = request_id
            self._in_flight_started_at = submitted
        try:
            self._ensure_loaded()
            result = self._adapter.generate(
                GenerationRequest(
                    prompt=self._workload.prompt,
                    n_predict=self._workload.n_predict,
                    seed=self._workload.seed,
                    temperature=self._workload.temperature,
                )
            )
            generation = self._adapter.last_generation
            completed = datetime.now(timezone.utc)
            record = RuntimeRequestRecord(
                request_id=request_id,
                submitted_at=submitted,
                first_token_at=generation.first_token_at if generation is not None else None,
                completed_at=completed,
                latency_s=result.total_s,
                generated_tokens=result.generated_tokens,
                decode_tokens_per_sec=result.decode_tokens_per_sec,
                success=bool(result.text),
            )
            metrics: dict[str, float] = {}
            for name, value in (
                ("decode_tokens_per_sec", result.decode_tokens_per_sec),
                ("total_latency_s", result.total_s),
                ("ttft_s", result.ttft_s),
                ("peak_rss_bytes", result.peak_rss_bytes),
            ):
                if value is not None:
                    metrics[name] = float(value)
            with self._lock:
                self._requests.append(record)
                self._completed += 1
                self._last_metrics = metrics
                self._last_text = result.text
                self._last_error = None
        except Exception as exc:  # noqa: BLE001 - a failed request must not crash
            with self._lock:
                self._requests.append(
                    RuntimeRequestRecord(
                        request_id=request_id,
                        submitted_at=submitted,
                        completed_at=datetime.now(timezone.utc),
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._in_flight = False
                self._in_flight_id = None
                self._in_flight_started_at = None

    def pause(self) -> bool:
        """Stop accepting new work and drain the in-flight request to completion."""
        self.pause_calls += 1
        with self._lock:
            self._accepting = False
        deadline = time.perf_counter() + 300.0
        while time.perf_counter() < deadline:
            with self._lock:
                if not self._in_flight:
                    return True
            time.sleep(0.05)
        return False

    def resume(self) -> bool:
        self.resume_calls += 1
        with self._lock:
            if not self._running or self._terminated:
                return False
            self._accepting = True
        return True

    # ------------------------------------------------------------------ #
    # ActiveWorkload                                                       #
    # ------------------------------------------------------------------ #
    def descriptor(self) -> WorkloadDescriptor:
        with self._lock:
            configuration = self._configuration
        return WorkloadDescriptor(
            workload_id=self._workload.workload_id,
            workload_kind=self._workload.kind,
            model_ref=self._model.name,
            model_path=self._model.path,
            backend_name="llama.cpp (resident)",
            configuration=configuration,
            context_length=configuration.context_length,
            prompts_ref=self._workload.to_workload_profile().prompt_set_ref,
            owned=True,
            controllable=True,
            pause_supported=True,
            checkpoint_supported=True,
            resume_supported=True,
            limitations=[
                "resident llama-server target; pause drains the in-flight request",
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
                requests_served=self._completed,
                in_flight_request_id=self._in_flight_id,
                last_error=self._last_error,
            )

    def configuration(self) -> RuntimeConfiguration:
        with self._lock:
            return self._configuration

    def set_configuration(self, configuration: RuntimeConfiguration) -> None:
        with self._lock:
            self._configuration = configuration
        # Apply to the live resident target via the controlled reload semantics so the
        # intervention genuinely changes the running server (not just an in-memory field).
        if self._adapter.is_loaded:
            self._adapter.set_runtime_params(
                runtime_params_from_config(
                    configuration,
                    n_predict=self._workload.n_predict,
                    seed=self._workload.seed,
                    ignore_eos=self._workload.ignore_eos,
                )
            )
            self._target_pid = self._adapter.pid
            self._target_identity = self._adapter.identity

    def model_profile(self) -> ModelProfile:
        return self._model

    def workload_profile(self) -> WorkloadProfile:
        return self._workload.to_workload_profile()

    def last_metrics(self) -> dict[str, float]:
        with self._lock:
            return dict(self._last_metrics)

    def last_text(self) -> str | None:
        with self._lock:
            return self._last_text

    def monitor_pid(self) -> int:
        return os.getpid()
