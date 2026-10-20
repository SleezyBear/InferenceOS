"""Resident llama.cpp :class:`BackendAdapter`.

Unlike the process-per-generation CLI adapter, this adapter keeps a real resident
``llama-server`` alive across requests. It exposes measured load/unload evidence and
a real streamed TTFT, and it applies runtime parameters with correct ``None``
semantics (an explicitly-provided ``None`` clears the value; an omitted field is left
unchanged). Parameters that llama-server cannot change live trigger a controlled
reload that preserves the ability to restore the prior configuration.

The adapter remains behind :class:`BackendAdapter`; the CLI adapter is unchanged and
stays available for isolated/cold benchmarking.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import cast

from inferenceos.backends.llama_cpp_toolchain import DeviceInfo, cli_version, list_devices
from inferenceos.backends.llama_server_toolchain import resolve_server_executable
from inferenceos.backends.process_control import identity_matches
from inferenceos.backends.release import ReleaseEvidence, release_evidence
from inferenceos.backends.resident_server import (
    LlamaServerResidentBackend,
    ResidentBackendError,
    ResidentStartCancelled,
    ResidentServerBackend,
)
from inferenceos.config import Settings
from inferenceos.contracts.enums import BackendType
from inferenceos.contracts.lifecycle import (
    ProcessIdentity,
    ResidentGenerationMeasurement,
    ResidentLoadMeasurement,
    ResidentUnloadMeasurement,
    RuntimeParamApplication,
    RuntimeParamChange,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    BackendCapabilities,
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.logging_config import get_logger

logger = get_logger("backends.llama_cpp_resident_adapter")

# Every server runtime knob is load-time: changing it requires a controlled reload.
_RELOAD_FIELDS: frozenset[str] = frozenset(RuntimeParams.model_fields)
_RESIDENT_FIELDS: frozenset[str] = frozenset(
    {"context_length", "gpu_offload_layers", "threads", "batch_size", "ubatch_size",
     "flash_attention", "device", "use_mmap", "kv_cache_type", "threads_batch",
     "engine_switches", "kv_cache_type_k", "kv_cache_type_v", "placement_overrides"}
)


@dataclass
class _StopIntent:
    """One stop group; resuming it never authorizes an older operation."""
    by_user: bool
    halted: bool = True
    revision: int = 0
    resume_problem: str | None = None


@dataclass
class _RequestOperation:
    cancelled: threading.Event = field(default_factory=threading.Event)
    stopped_by: _StopIntent | None = None


class LlamaCppResidentAdapter(BackendAdapter):
    """Resident llama.cpp adapter with real load/unload and streamed TTFT."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        server: Path | None = None,
        backend: ResidentServerBackend | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._server = server or resolve_server_executable(self._settings.llama_cpp_dir)
        self._backend: ResidentServerBackend = backend or LlamaServerResidentBackend(
            server=self._server,
            host=self._settings.resident_host,
            port=self._settings.resident_port,
            readiness_timeout_s=self._settings.resident_readiness_timeout_s,
            request_timeout_s=self._settings.resident_request_timeout_s,
        )
        self._model: ModelProfile | None = None
        self._params = RuntimeParams()
        # Long backend operations remain cancellable; only request publication and
        # clearing are locked. A late release may clear only its own generation.
        self._request_lock = threading.RLock()
        self._request_generation = object()
        self._pending_request: _RequestOperation | None = None
        # Start permission outlives publication/release generations. Unload and
        # cleanup release resources; only a current successful resume lifts a halt.
        self._stop_intent: _StopIntent | None = None
        #: A log level for the next start only (e.g. the engine's debug level, to read
        #: which pieces a placement moved); the setup's own level is kept for the rest.
        self._next_log_verbosity: int | None = None
        self._param_application: RuntimeParamApplication | None = None
        #: Set by a user stop (``halt(by_user=True)``), cleared when starts are allowed again.
        self._halt_by_user = False

    # ------------------------------------------------------------------ #
    # Introspection (beyond the abstract interface, for evidence)         #
    # ------------------------------------------------------------------ #
    @property
    def server_path(self) -> Path:
        return self._server

    @property
    def params(self) -> RuntimeParams:
        return self._params

    @property
    def last_load(self) -> ResidentLoadMeasurement | None:
        return self._backend.last_load

    @property
    def last_unload(self) -> ResidentUnloadMeasurement | None:
        return self._backend.last_unload

    @property
    def last_generation(self) -> ResidentGenerationMeasurement | None:
        return self._backend.last_generation

    @property
    def last_device(self) -> DeviceInfo | None:
        return getattr(self._backend, "last_device", None)

    @property
    def last_param_application(self) -> RuntimeParamApplication | None:
        return self._param_application

    @property
    def pid(self) -> int | None:
        return self._backend.pid

    @property
    def identity(self) -> ProcessIdentity | None:
        return self._backend.identity

    @property
    def base_url(self) -> str | None:
        return self._backend.base_url

    # ------------------------------------------------------------------ #
    # BackendAdapter                                                      #
    # ------------------------------------------------------------------ #
    def capabilities(self) -> BackendCapabilities:
        devices: list[str] = []
        try:
            devices = list_devices(self._server)
        except (OSError, FileNotFoundError):
            devices = []
        backends: list[BackendType] = [BackendType.CPU]
        if any("vulkan" in device.lower() for device in devices):
            backends.insert(0, BackendType.VULKAN)
        try:
            version = cli_version(self._server)
        except OSError:
            version = None
        return BackendCapabilities(
            name="llama.cpp (resident)",
            version=version,
            backends=backends,
            devices=devices or ["CPU"],
            supports_runtime_params=True,
        )

    def use_log_verbosity_once(self, level: int) -> None:
        """Start the server at ``level`` the next time it starts (load, reload or
        restart), then go back to the setup's own level."""
        with self._request_lock:
            self._next_log_verbosity = int(level)

    def _start_params(self, params: RuntimeParams) -> RuntimeParams:
        level, self._next_log_verbosity = self._next_log_verbosity, None
        return params if level is None else params.model_copy(update={"log_verbosity": level})

    def _require_starts_allowed(self) -> None:
        intent = self._stop_intent
        if intent is not None and intent.halted:
            if intent.resume_problem is not None:
                raise ResidentBackendError(intent.resume_problem)
            raise ResidentStartCancelled("resident server start refused: stopped until resumed")

    def _claim_request(self) -> _RequestOperation:
        # Called with the metadata lock held, before any backend call. A pending
        # request has not published its params, so even a would-be no-op is refused.
        if self._pending_request is not None:
            raise ResidentBackendError("resident configuration operation already in progress")
        self._require_starts_allowed()
        generation = _RequestOperation()
        self._request_generation = self._pending_request = generation
        return generation

    def _require_current(self, generation: _RequestOperation) -> None:
        if self._request_generation is not generation or generation.cancelled.is_set():
            raise ResidentStartCancelled("resident configuration operation superseded by a later request")

    def _require_load(self, generation: _RequestOperation, load: ResidentLoadMeasurement) -> None:
        self._require_current(generation)
        # Liveness alone may describe a replacement. Check this start's evidence,
        # including its process identity, before publishing a verified application.
        if (self._backend.last_load != load or not load.ready
                or not self._backend.is_running() or self._backend.pid != load.pid
                or self._backend.identity != load.identity):
            raise ResidentStartCancelled("resident configuration load is no longer current")

    def _start_request(
        self, generation: _RequestOperation, model: ModelProfile, params: RuntimeParams,
    ) -> ResidentLoadMeasurement:
        with self._request_lock:
            self._require_current(generation)
        guarded = getattr(self._backend, "start_if_current", None)
        if callable(guarded):
            # Only this Event is read under the backend lock: no metadata/lifecycle
            # lock inversion, and resume can never clear an older operation's Event.
            return cast(ResidentLoadMeasurement, guarded(
                model, params, current=lambda: not generation.cancelled.is_set()))
        return self._backend.start(model, params)

    def _stop_request(self, generation: _RequestOperation) -> None:
        with self._request_lock:
            self._require_current(generation)
        guarded = getattr(self._backend, "stop_if_current", None)
        if callable(guarded):
            guarded(current=lambda: not generation.cancelled.is_set())
        else:
            self._backend.stop()

    def _finish_request(self, generation: _RequestOperation) -> None:
        with self._request_lock:
            if self._pending_request is generation:
                self._pending_request = None

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        path = Path(model.path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        with self._request_lock:
            generation = self._claim_request()
            start_params = self._start_params(params)
        try:
            load = self._start_request(generation, model, start_params)
            with self._request_lock:
                self._require_load(generation, load)
                self._model = model
                self._params = params
                self._param_application = None
        finally:
            self._finish_request(generation)
        logger.info("Resident model loaded: %s", model.name)

    def _begin_release(self) -> object:
        with self._request_lock:
            # Invalidate pending publication before the backend can release its
            # process. Keep start permission independent: release never resumes
            # a halt, even when an older native resume later clears its own flag.
            generation = object()
            if self._pending_request is not None:
                self._pending_request.cancelled.set()
            self._request_generation = generation
            self._pending_request = None
            return generation

    def unload(self) -> None:
        generation = self._begin_release()
        if self.is_active:
            try:
                self._backend.stop()
            except ResidentBackendError:
                if self.is_active:
                    raise
        self._clear_released_request(generation)
        logger.info("Resident model unloaded")

    def cleanup(self) -> None:
        """Release the backend's owned target, including an already-dead process."""
        generation = self._begin_release()
        self._backend.cleanup()
        self._clear_released_request(generation)

    def _clear_released_request(self, generation: object) -> None:
        with self._request_lock:
            if self._request_generation is generation:
                self._model = None
                self._params = RuntimeParams()
                self._param_application = None

    def release_evidence(self) -> ReleaseEvidence:
        """Read the existing cleanup owner's evidence, including retained credentials."""
        return release_evidence(self._backend)

    def verify_process_identity(self, identity: ProcessIdentity) -> bool:
        """Return True only when the live OS process matches ``identity``.

        A PID alone is insufficient: an unrelated process that later occupies the
        same PID has a different creation time and must not be accepted.
        """
        return identity_matches(identity)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._backend.is_running():
            raise RuntimeError("no resident model loaded; call load() first")
        return self._backend.generate(request)

    def set_runtime_params(self, params: RuntimeParams) -> None:
        """Apply runtime params, honouring explicit ``None`` and reload-required fields.

        Fields that are *not* present in ``params.model_fields_set`` are left
        unchanged. A field explicitly set to ``None`` clears the value (this is the
        behaviour the previous adapter could not express). Any change to a load-time
        server parameter triggers a controlled reload.
        """
        with self._request_lock:
            self._require_starts_allowed()
            if self._pending_request is not None:
                raise ResidentBackendError("resident configuration operation already in progress")
            if not params.model_fields_set:
                self._param_application = RuntimeParamApplication(
                    changed=False, reason="no fields provided; configuration unchanged"
                )
                return
            changes = [RuntimeParamChange(parameter=name, prior_value=getattr(self._params, name),
                                          applied_value=getattr(params, name))
                       for name in sorted(params.model_fields_set)
                       if getattr(self._params, name) != getattr(params, name)]
            if not changes:
                self._param_application = RuntimeParamApplication(
                    changed=False, reason="provided values match the current configuration"
                )
                return
            prior_params, model = self._params, self._model
            merged = prior_params.model_copy(update={c.parameter: c.applied_value for c in changes})
            requires_reload = any(c.parameter in _RELOAD_FIELDS for c in changes)
            if not requires_reload or not self._backend.is_running():
                self._request_generation = object()
                self._params = merged
                self._param_application = RuntimeParamApplication(
                    changed=True, requires_reload=False, reloaded=False, verified=True,
                    changes=changes, reason="parameters updated in place",
                )
                return
            if model is None:
                raise ResidentBackendError("cannot reload without a loaded model")
            generation = self._claim_request()
        self._apply_with_reload(generation, model, prior_params, merged, changes)

    def _apply_with_reload(
        self, generation: _RequestOperation, model: ModelProfile, prior_params: RuntimeParams,
        merged: RuntimeParams, changes: list[RuntimeParamChange],
    ) -> None:
        try:
            with self._request_lock:
                self._require_current(generation)
            self._stop_request(generation)
            with self._request_lock:
                self._require_current(generation)
                # A failed stop never attempted a start; keep its one-shot log
                # level for the next actual start attempt, as before extraction.
                start_params = self._start_params(merged)
            try:
                load = self._start_request(generation, model, start_params)
            except Exception as exc:  # noqa: BLE001 - surface the real failure
                with self._request_lock:
                    stop = generation.stopped_by
                    stopped = stop is not None and self._request_generation is stop
                    if (self._request_generation is not generation and not stopped
                            or self._backend.is_running()):
                        raise  # never restore or publish into a replacement
                    cancelled = stopped or isinstance(exc, ResidentStartCancelled) or getattr(self._backend, "halted", False)
                    if cancelled:
                        by_user = stop.by_user if stopped and stop is not None else self._halt_by_user
                        words = ("stopped by the user; engine stopped" if by_user
                                 else "stopped by a timed stop (stuck operation or cleanup); engine stopped")
                        self._params = prior_params
                        self._param_application = RuntimeParamApplication(
                            changed=True, requires_reload=True, reloaded=False, verified=False,
                            changes=changes, reason=f"reload {words}",
                            error=f"{type(exc).__name__}: {exc}; the prior configuration was not restarted",
                        )
                        raise ResidentStartCancelled(f"reload {words}") from exc
                restored = False
                restore_error = ""
                try:
                    with self._request_lock:
                        self._require_current(generation)
                    restored_load = self._start_request(generation, model, prior_params)
                    with self._request_lock:
                        self._require_load(generation, restored_load)
                        restored = True
                except Exception as restore_exc:  # noqa: BLE001 - reported below
                    with self._request_lock:
                        if self._request_generation is not generation or self._backend.is_running():
                            raise exc  # preserve the original failure; no stale report
                    restore_error = f"; restore error: {type(restore_exc).__name__}: {restore_exc}"
                with self._request_lock:
                    self._require_current(generation)
                    self._params = prior_params
                    outcome = ("prior configuration restored" if restored
                               else "prior configuration NOT restored; target is not running")
                    self._param_application = RuntimeParamApplication(
                        changed=True, requires_reload=True, reloaded=False, verified=False,
                        changes=changes, reason=f"reload failed; {outcome}",
                        error=f"{type(exc).__name__}: {exc}{restore_error}",
                    )
                raise ResidentBackendError(f"runtime parameter reload failed ({outcome}): {exc}") from exc
            with self._request_lock:
                self._require_load(generation, load)
                self._params = merged
                self._param_application = RuntimeParamApplication(
                    changed=True, requires_reload=True, reloaded=True, verified=True,
                    changes=changes,
                    reason="configuration requires reload; controlled reload performed and verified",
                )
        finally:
            self._finish_request(generation)

    def restart(self) -> None:
        """Reload the current setup on a fresh process, preserving cancellation."""
        with self._request_lock:
            if self._pending_request is not None:
                raise ResidentBackendError("resident configuration operation already in progress")
            if not self._backend.is_running():
                return
            model, params = self._model, self._params
            if model is None:
                raise ResidentBackendError("cannot reload without a loaded model")
            generation = self._claim_request()
        self._apply_with_reload(generation, model, params, params, [])

    def telemetry(self) -> BackendTelemetry:
        return self._backend.telemetry()

    @property
    def is_loaded(self) -> bool:
        return self._backend.is_running()

    @property
    def is_active(self) -> bool:
        """True while the backend owns a process: loading, loaded or stopping."""
        active = getattr(self._backend, "is_active", None)
        return bool(active()) if callable(active) else self._backend.is_running()

    def halt(self, *, by_user: bool = False) -> None:
        """Stop now in any state and refuse new starts until :meth:`resume`.
        ``by_user`` marks a user stop, so a reload it cancels is reported as such."""
        with self._request_lock:
            self._halt_by_user = self._halt_by_user or by_user
            prior = self._stop_intent
            intent = prior if prior is not None and prior.halted else _StopIntent(self._halt_by_user)
            intent.by_user = intent.by_user or by_user
            intent.revision += 1
            intent.resume_problem = None
            self._stop_intent = intent
            self._request_generation = intent
            if self._pending_request is not None:
                self._pending_request.stopped_by = intent
                self._pending_request.cancelled.set()
                self._pending_request = None
        halt = getattr(self._backend, "halt", None)
        if callable(halt):
            halt()
        elif self._backend.is_running():
            self._backend.stop()

    def resume(self, *, unless: Callable[[], bool] | None = None) -> bool:
        """Allow starts again (the backend first confirms the old process is gone).
        ``unless`` is re-checked under the backend's lock just before the halt is
        lifted (see ``LlamaServerResidentBackend.resume``)."""
        with self._request_lock:
            intent = self._stop_intent
            revision = intent.revision if intent is not None else None
        resume = getattr(self._backend, "resume", None)
        try:
            resumed = bool(resume(unless=unless)) if callable(resume) else True
        except ResidentBackendError as exc:
            with self._request_lock:
                if (self._stop_intent is intent and intent is not None
                        and intent.revision == revision):
                    # Keep the resource owner's unresolved-exit diagnostic visible;
                    # this cached message is not a new process-exit verification.
                    intent.resume_problem = str(exc)
            raise
        if not resumed:
            return False
        with self._request_lock:
            if (unless is not None and unless()
                    or self._stop_intent is not intent
                    or intent is not None and intent.revision != revision):
                return False
            if intent is not None:
                intent.halted = False
                intent.resume_problem = None
            self._halt_by_user = False
        return True
