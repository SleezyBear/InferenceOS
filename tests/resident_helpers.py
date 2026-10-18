"""Shared fakes for Phase 5R unit tests (no real server is started)."""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.contracts.lifecycle import (
    ProcessIdentity,
    ResidentGenerationMeasurement,
    ResidentLoadMeasurement,
    ResidentUnloadMeasurement,
    ResourceSnapshot,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.interfaces.backend_adapter import (
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.interfaces.reasoning_engine import ReasoningEngine


def synthetic_model(path: str = "synthetic.gguf", name: str = "synthetic") -> ModelProfile:
    from inferenceos.contracts.enums import QuantizationType

    return ModelProfile(
        path=path,
        name=name,
        format="gguf",
        architecture="synthetic",
        quantization=QuantizationType.Q4_K_M,
        file_size_bytes=1024,
        n_layers=4,
        n_attention_heads=4,
        n_kv_heads=2,
        head_dim=64,
        key_length=64,
        value_length=64,
        weight_bytes=1024,
        kv_cache_bytes_per_token=4096,
    )


def _snapshot(pid: int | None, rss: int = 1024) -> ResourceSnapshot:
    return ResourceSnapshot(
        pid=pid,
        exists=pid is not None,
        rss_bytes=rss if pid is not None else None,
        system_available_bytes=8 * 1024**3,
        system_total_bytes=16 * 1024**3,
    )


class FakeResidentBackend:
    """Deterministic stand-in for a real resident server backend."""

    def __init__(self, *, rss: int = 512 * 1024 * 1024) -> None:
        self._pid = 1000
        self._running = False
        self._rss = rss
        self._load: ResidentLoadMeasurement | None = None
        self._unload: ResidentUnloadMeasurement | None = None
        self._generation: ResidentGenerationMeasurement | None = None
        self._starts = 0
        # failure injection
        self.fail_start_times: int = 0
        self.fail_stop = False
        self.fail_generate = False

    @property
    def pid(self) -> int | None:
        return self._pid if self._running else None

    @property
    def identity(self) -> ProcessIdentity | None:
        if not self._running:
            return None
        return ProcessIdentity(pid=self._pid, create_time=float(self._pid))

    @property
    def base_url(self) -> str | None:
        return "http://127.0.0.1:0" if self._running else None

    @property
    def last_load(self) -> ResidentLoadMeasurement | None:
        return self._load

    @property
    def last_unload(self) -> ResidentUnloadMeasurement | None:
        return self._unload

    @property
    def last_generation(self) -> ResidentGenerationMeasurement | None:
        return self._generation

    def telemetry(self) -> BackendTelemetry:
        return BackendTelemetry(device="vulkan-fake", gpu_layers_offloaded=4, gpu_layers_total=4,
                                observed={"context_length": self._context})

    def is_running(self) -> bool:
        return self._running

    def start(self, model: ModelProfile, params: RuntimeParams) -> ResidentLoadMeasurement:
        if self._running:
            raise RuntimeError("already running")
        if self.fail_start_times > 0:
            self.fail_start_times -= 1
            raise RuntimeError("injected start failure")
        self._starts += 1
        self._pid = 1000 + self._starts
        self._running = True
        self._context = params.context_length
        measurement = ResidentLoadMeasurement(
            model_ref=model.name,
            model_path=model.path,
            pid=self._pid,
            identity=self.identity,
            host="127.0.0.1",
            port=18000 + self._starts,
            ready=True,
            startup_s=0.25,
            ready_s=0.25,
            before=_snapshot(None),
            after=_snapshot(self._pid, self._rss + 100),
            resident_verified=True,
        )
        self._load = measurement
        return measurement

    def stop(self) -> ResidentUnloadMeasurement:
        if not self._running:
            raise RuntimeError("not running")
        pid = self._pid
        self._running = False
        if self.fail_stop:
            measurement = ResidentUnloadMeasurement(
                pid=pid,
                exited=False,
                stop_method="none",
                before=_snapshot(pid, self._rss),
                after=_snapshot(pid, self._rss),
                unload_s=0.1,
                cleanup_complete=False,
                notes=["injected stop failure"],
            )
        else:
            measurement = ResidentUnloadMeasurement(
                pid=pid,
                exited=True,
                exit_code=0,
                stop_method="job_object",
                before=_snapshot(pid, self._rss),
                after=_snapshot(None),
                unload_s=0.1,
                cleanup_complete=True,
            )
        self._unload = measurement
        return measurement

    def cleanup(self) -> None:
        self._running = False
        # A successful current cleanup supersedes the prior failed stop attempt.
        self._unload = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._running:
            raise RuntimeError("not running")
        if self.fail_generate:
            raise RuntimeError("injected generation failure")
        now = datetime.now(timezone.utc)
        self._generation = ResidentGenerationMeasurement(
            submitted_at=now,
            first_token_at=now,
            completed_at=now,
            ttft_s=0.02,
            total_s=0.2,
            generated_tokens=request.n_predict,
            prompt_tokens=4,
            decode_tokens_per_sec=50.0,
            streamed=True,
            text="ok",
        )
        return GenerationResult(
            text="ok",
            generated_tokens=request.n_predict,
            prompt_tokens=4,
            decode_tokens_per_sec=50.0,
            ttft_s=0.02,
            total_s=0.2,
        )


class FakeReasoner:
    """Deterministic stand-in for a resident reasoner."""

    def __init__(self, *, engine: ReasoningEngine, expected_bytes: int = 256 * 1024 * 1024) -> None:
        self._engine = engine
        self._expected = expected_bytes
        self._running = False
        self._load: ResidentLoadMeasurement | None = None
        self._unload: ResidentUnloadMeasurement | None = None
        self.fail_start = False
        self.fail_stop = False
        self.engine_error: str | None = None

    @property
    def model_ref(self) -> str:
        return "fake-reasoner"

    @property
    def expected_bytes(self) -> int | None:
        return self._expected

    def start(self) -> ResidentLoadMeasurement:
        if self.fail_start:
            raise RuntimeError("injected reasoner load failure")
        self._running = True
        self._load = ResidentLoadMeasurement(
            model_ref="fake-reasoner",
            model_path="reasoner.gguf",
            pid=9001,
            identity=ProcessIdentity(pid=9001, create_time=9001.0),
            host="127.0.0.1",
            port=19000,
            ready=True,
            startup_s=0.3,
            ready_s=0.3,
            before=_snapshot(None),
            after=_snapshot(9001, 300 * 1024 * 1024),
            resident_verified=True,
        )
        return self._load

    def stop(self) -> ResidentUnloadMeasurement:
        if self.fail_stop:
            raise RuntimeError("injected reasoner unload failure")
        self._running = False
        self._unload = ResidentUnloadMeasurement(
            pid=9001,
            exited=True,
            stop_method="job_object",
            before=_snapshot(9001, 300 * 1024 * 1024),
            after=_snapshot(None),
            unload_s=0.05,
            cleanup_complete=True,
        )
        return self._unload

    def engine(self) -> ReasoningEngine:
        if self.engine_error is not None:
            raise RuntimeError(self.engine_error)
        return self._engine

    def is_running(self) -> bool:
        return self._running
