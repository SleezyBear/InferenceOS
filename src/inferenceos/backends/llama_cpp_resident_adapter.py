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

from pathlib import Path

from inferenceos.backends.llama_cpp_toolchain import DeviceInfo, cli_version, list_devices
from inferenceos.backends.llama_server_toolchain import resolve_server_executable
from inferenceos.backends.process_control import identity_matches
from inferenceos.backends.resident_server import (
    LlamaServerResidentBackend,
    ResidentBackendError,
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
     "flash_attention", "device", "use_mmap", "kv_cache_type", "threads_batch"}
)


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
        self._param_application: RuntimeParamApplication | None = None

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

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        path = Path(model.path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        self._backend.start(model, params)
        self._model = model
        self._params = params
        self._param_application = None
        logger.info("Resident model loaded: %s", model.name)

    def unload(self) -> None:
        if self._backend.is_running():
            self._backend.stop()
        self._model = None
        self._params = RuntimeParams()
        logger.info("Resident model unloaded")

    def cleanup(self) -> None:
        """Release a (possibly already-dead) target and its containment handle.

        Unlike :meth:`unload`, this is safe to call after the target has died and
        never raises, so a failed/lost target cannot leak its Job Object handle or
        temporary key file.
        """
        self._backend.cleanup()
        self._model = None
        self._params = RuntimeParams()
        self._param_application = None

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
        if not params.model_fields_set:
            self._param_application = RuntimeParamApplication(
                changed=False, reason="no fields provided; configuration unchanged"
            )
            return

        changes: list[RuntimeParamChange] = []
        for name in sorted(params.model_fields_set):
            value = getattr(params, name)
            prior = getattr(self._params, name)
            if prior != value:
                changes.append(
                    RuntimeParamChange(parameter=name, prior_value=prior, applied_value=value)
                )
        if not changes:
            self._param_application = RuntimeParamApplication(
                changed=False, reason="provided values match the current configuration"
            )
            return

        merged = self._params.model_copy(update={c.parameter: c.applied_value for c in changes})
        requires_reload = any(c.parameter in _RELOAD_FIELDS for c in changes)
        if not requires_reload or not self._backend.is_running():
            self._params = merged
            self._param_application = RuntimeParamApplication(
                changed=True,
                requires_reload=False,
                reloaded=False,
                verified=True,
                changes=changes,
                reason="parameters updated in place",
            )
            return
        self._apply_with_reload(merged, changes)

    def _apply_with_reload(
        self, merged: RuntimeParams, changes: list[RuntimeParamChange]
    ) -> None:
        if self._model is None:
            raise ResidentBackendError("cannot reload without a loaded model")
        prior_params = self._params
        self._backend.stop()
        try:
            self._backend.start(self._model, merged)
        except Exception as exc:  # noqa: BLE001 - surface the real failure
            # Attempt to restore the prior configuration so the target is not lost.
            restored = False
            restore_error = ""
            try:
                self._backend.start(self._model, prior_params)
                restored = self._backend.is_running()
            except Exception as restore_exc:  # noqa: BLE001 - reported below
                restored = False
                restore_error = f"; restore error: {type(restore_exc).__name__}: {restore_exc}"
            # Never report the failed new settings as current. When the restore also
            # failed nothing is running, and the reason says so plainly.
            self._params = prior_params
            outcome = (
                "prior configuration restored"
                if restored
                else "prior configuration NOT restored; target is not running"
            )
            self._param_application = RuntimeParamApplication(
                changed=True,
                requires_reload=True,
                reloaded=False,
                verified=False,
                changes=changes,
                reason=f"reload failed; {outcome}",
                error=f"{type(exc).__name__}: {exc}{restore_error}",
            )
            raise ResidentBackendError(
                f"runtime parameter reload failed ({outcome}): {exc}"
            ) from exc
        self._params = merged
        self._param_application = RuntimeParamApplication(
            changed=True,
            requires_reload=True,
            reloaded=True,
            verified=self._backend.is_running(),
            changes=changes,
            reason="configuration requires reload; controlled reload performed and verified",
        )

    def telemetry(self) -> BackendTelemetry:
        return self._backend.telemetry()

    @property
    def is_loaded(self) -> bool:
        return self._backend.is_running()
