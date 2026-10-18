"""Persisted :class:`DeploymentState` validation and a minimal resident serving entry.

This is deliberately narrow: it consumes an already-persisted, already-optimized
deployment state, validates it, starts the **existing** resident ``llama-server``
backend with the retained runtime configuration, verifies process identity and the
effective configuration, serves local inference, and shuts down cleanly.

It is *not* an orchestration platform: no remote/cloud/container deployment, no
multi-user management, no second serving architecture, and no external-process
control. It reuses the Phase 5R resident server and its secure API-key mechanism.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field

from inferenceos.backends.observed import observed_configuration
from inferenceos.backends import process_control
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.backends.llama_cpp_toolchain import cli_version
from inferenceos.config import Settings
from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.fresh import DeploymentState, DeploymentStatus
from inferenceos.contracts.lifecycle import ProcessIdentity
from inferenceos.contracts.model import ModelProfile
from inferenceos.experiment.restoration import runtime_params_from
from inferenceos.interfaces.backend_adapter import BackendTelemetry, GenerationRequest
from inferenceos.storage import atomic_write_text

DEPLOYMENT_SCHEMA_VERSION = "deployment.1"

# A deployment is servable only when it represents a real retained (or unchanged)
# state. Rejected/safety/no-candidate states are not deployments to serve.
SERVABLE_STATUSES = {DeploymentStatus.RETAINED, DeploymentStatus.NO_CHANGE}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def deployment_integrity(state: DeploymentState) -> str:
    """Deterministic content hash over the deployment payload (excluding its hash)."""
    payload = state.model_dump(mode="json")
    payload.pop("integrity_hash", None)
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def write_deployment_state(path: Path | str, state: DeploymentState) -> DeploymentState:
    # Set the schema version before hashing so the hash covers it; only the hash field
    # itself is excluded from its own computation.
    versioned = state.model_copy(update={"schema_version": DEPLOYMENT_SCHEMA_VERSION})
    sealed = versioned.model_copy(update={"integrity_hash": deployment_integrity(versioned)})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, sealed.model_dump_json(indent=2))
    return sealed


class DeploymentIntegrityError(ValueError):
    """The persisted deployment state was modified after it was written."""


def load_deployment_state(path: Path | str) -> DeploymentState:
    text = Path(path).read_text(encoding="utf-8")
    state = DeploymentState.model_validate_json(text)
    if state.integrity_hash is None:
        raise DeploymentIntegrityError(
            "deployment state has no integrity hash; it was not written by "
            "InferenceOS and cannot be trusted"
        )
    if deployment_integrity(state) != state.integrity_hash:
        raise DeploymentIntegrityError("deployment state integrity check failed (tampered)")
    if state.schema_version != DEPLOYMENT_SCHEMA_VERSION:
        raise DeploymentIntegrityError(
            f"deployment schema {state.schema_version!r} != {DEPLOYMENT_SCHEMA_VERSION!r}"
        )
    return state


# --------------------------------------------------------------------------- #
# Contracts                                                                   #
# --------------------------------------------------------------------------- #
class DeploymentValidation(ContractBase):
    valid: bool
    reasons: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    status: str | None = None
    model_ref: str | None = None
    model_path: str | None = None
    model_hash_verified: bool = False
    backend_server: str | None = None
    backend_version: str | None = None


class DeploymentServeEvidence(ContractBase):
    served: bool
    deployment_run_id: str
    model_ref: str
    model_path: str
    backend_name: str | None = None
    backend_server: str | None = None
    pid: int | None = None
    identity: ProcessIdentity | None = None
    configuration_requested: RuntimeConfiguration
    #: Only what the engine itself showed (review R3); a field it did not show is None.
    configuration_observed: RuntimeConfiguration | None = None
    #: True: every requested setting was shown as requested. None: nothing contradicts
    #: the request, but some requested settings could not be confirmed (listed in
    #: ``configuration_unverified``); serving still proceeds. False: the engine showed
    #: a different value (listed in ``configuration_contradictions``); not served.
    configuration_matches: bool | None = False
    configuration_unverified: list[str] = Field(default_factory=list)
    configuration_contradictions: list[str] = Field(default_factory=list)
    residency_verified: bool = False
    probe_prompt: str = ""
    probe_text: str = ""
    probe_generated_tokens: int | None = None
    probe_decode_tokens_per_sec: float | None = None
    peak_rss_bytes: int | None = None
    final_rss_bytes: int | None = None
    clean_shutdown: bool = False
    orphan_process: bool = False
    validation: DeploymentValidation
    reasons: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def validate_deployment(
    state: DeploymentState,
    *,
    settings: Settings,
    model_profile: ModelProfile | None = None,
) -> DeploymentValidation:
    """Fail-closed validation of a persisted deployment state before serving."""
    reasons: list[str] = []
    checks: list[str] = []

    if state.status not in SERVABLE_STATUSES:
        reasons.append(
            f"deployment status {state.status.value!r} is not servable "
            f"(expected one of {sorted(s.value for s in SERVABLE_STATUSES)})"
        )
    else:
        checks.append(f"status={state.status.value}")

    model_path = Path(state.model_path)
    if not model_path.is_file():
        reasons.append(f"referenced model file is missing: {model_path}")
    else:
        checks.append("model file exists")

    model_hash_verified = False
    if state.model_content_hash:
        actual = file_sha256(model_path) if model_path.is_file() else None
        if actual is None or actual != state.model_content_hash:
            reasons.append(
                "model content hash does not match the persisted deployment "
                "(stale or tampered model)"
            )
        else:
            model_hash_verified = True
            checks.append("model content hash verified")
    else:
        checks.append("model content hash not recorded (cannot verify identity)")

    server_path = settings.llama_cpp_dir / "llama-server.exe"
    if not server_path.is_file():
        fallback = settings.llama_cpp_dir / "llama-server"
        server_path = fallback if fallback.is_file() else server_path
    backend_version: str | None = None
    if not server_path.is_file():
        reasons.append(f"resident llama-server backend not found in {settings.llama_cpp_dir}")
    else:
        checks.append("resident backend present")
        try:
            backend_version = cli_version(server_path)
        except OSError:  # pragma: no cover - environment dependent
            backend_version = None
        if state.backend_version and backend_version and state.backend_version != backend_version:
            reasons.append(
                f"backend version {backend_version!r} != deployment {state.backend_version!r}"
            )
        if state.backend_name and "llama" not in state.backend_name.lower():
            reasons.append(f"backend name {state.backend_name!r} is not the llama.cpp backend")

    config_reasons = _validate_configuration(state.final_configuration, model_profile)
    reasons.extend(config_reasons)
    if not config_reasons:
        checks.append("runtime configuration compatible with the model")

    if state.run_id.strip() == "":
        reasons.append("deployment has no originating run id (provenance missing)")
    else:
        checks.append("provenance run id present")
    if state.provenance_hash is None:
        reasons.append("deployment provenance hash is missing")
    else:
        checks.append("provenance hash present")

    return DeploymentValidation(
        valid=not reasons,
        reasons=reasons,
        checks=checks,
        status=state.status.value,
        model_ref=state.model_ref,
        model_path=state.model_path,
        model_hash_verified=model_hash_verified,
        backend_server=str(server_path),
        backend_version=backend_version,
    )


def _validate_configuration(
    configuration: RuntimeConfiguration, model: ModelProfile | None
) -> list[str]:
    reasons: list[str] = []
    context = configuration.context_length
    if context is not None and context < 1:
        reasons.append("context_length must be >= 1")
    if (
        context is not None
        and model is not None
        and model.trained_context_length is not None
        and context > model.trained_context_length
    ):
        reasons.append(
            f"context_length {context} exceeds the model's trained context "
            f"{model.trained_context_length}"
        )
    offload = configuration.gpu_offload_layers
    if offload is not None and offload < 0:
        reasons.append("gpu_offload_layers must be >= 0")
    if offload is not None and model is not None and offload > model.n_layers + 1:
        reasons.append(
            f"gpu_offload_layers {offload} exceeds the model's {model.n_layers} blocks plus output layer"
        )
    if configuration.threads is not None and configuration.threads < 1:
        reasons.append("threads must be >= 1 when set")
    return reasons


# --------------------------------------------------------------------------- #
# Serve                                                                       #
# --------------------------------------------------------------------------- #
def serve_deployment(
    state: DeploymentState,
    *,
    settings: Settings,
    prompt: str,
    n_predict: int = 32,
    serve_seconds: float = 0.0,
    adapter: LlamaCppResidentAdapter | None = None,
) -> DeploymentServeEvidence:
    """Validate, start the real resident backend, verify, probe and shut down cleanly.

    ``serve_seconds > 0`` keeps the verified resident server alive (serving) for that
    bounded window before the clean shutdown; ``0`` performs the verified serve/probe
    cycle and shuts down immediately.
    """
    started = _utcnow()
    model_profile = _safe_model_profile(state.model_path)
    validation = validate_deployment(state, settings=settings, model_profile=model_profile)
    observed: RuntimeConfiguration | None = None
    pid: int | None = None
    identity: ProcessIdentity | None = None
    residency = False
    matches: bool | None = False
    unverified: list[str] = []
    contradictions: list[str] = []
    probe_text = ""
    probe_tokens: int | None = None
    probe_tps: float | None = None
    peak: int | None = None
    final: int | None = None
    clean_shutdown = False
    orphan = False
    reasons = list(validation.reasons)

    if not validation.valid:
        return DeploymentServeEvidence(
            served=False,
            deployment_run_id=state.run_id,
            model_ref=state.model_ref,
            model_path=state.model_path,
            backend_name=state.backend_name,
            backend_server=validation.backend_server,
            configuration_requested=state.final_configuration,
            validation=validation,
            reasons=reasons,
            started_at=started,
            finished_at=_utcnow(),
        )

    assert model_profile is not None  # validated above
    resident = adapter or LlamaCppResidentAdapter(settings=settings)
    params = runtime_params_from(state.final_configuration)
    try:
        resident.load(model_profile, params)
        pid = resident.pid
        identity = resident.identity
        observed, contradictions, unverified = observed_configuration(
            state.final_configuration, _read_telemetry(resident)
        )
        if contradictions:
            matches = False
        else:
            matches = None if unverified else True
        residency = bool(resident.is_loaded and pid is not None and identity is not None)
        if contradictions:
            reasons.append(
                "served configuration does not match the deployment state: "
                + "; ".join(contradictions)
            )
        if not residency:
            reasons.append("resident process identity/readiness could not be verified")
        if not reasons:
            result = resident.generate(
                GenerationRequest(prompt=prompt, n_predict=n_predict, temperature=0.0)
            )
            probe_text = result.text
            probe_tokens = result.generated_tokens
            probe_tps = result.decode_tokens_per_sec
            peak = result.peak_rss_bytes
            final = result.final_rss_bytes
            if serve_seconds > 0:
                time.sleep(serve_seconds)
    finally:
        try:
            resident.cleanup()
        except Exception:  # noqa: BLE001 - shutdown must not mask the outcome
            reasons.append("clean shutdown reported an error")
        orphan = bool(pid is not None and process_control.process_exists(pid))
        release = resident.release_evidence()
        clean_shutdown = not orphan and release.clean
        if not release.clean:
            reasons.append(release.reason or "resident resources were not confirmed released")

    served = bool(
        residency and matches is not False and probe_text and clean_shutdown and not reasons
    )
    return DeploymentServeEvidence(
        served=served,
        deployment_run_id=state.run_id,
        model_ref=state.model_ref,
        model_path=state.model_path,
        backend_name=state.backend_name,
        backend_server=validation.backend_server,
        pid=pid,
        identity=identity,
        configuration_requested=state.final_configuration,
        configuration_observed=observed,
        configuration_matches=matches,
        configuration_unverified=unverified,
        configuration_contradictions=contradictions,
        residency_verified=residency,
        probe_prompt=prompt,
        probe_text=probe_text,
        probe_generated_tokens=probe_tokens,
        probe_decode_tokens_per_sec=probe_tps,
        peak_rss_bytes=peak,
        final_rss_bytes=final,
        clean_shutdown=clean_shutdown,
        orphan_process=orphan,
        validation=validation,
        reasons=reasons,
        started_at=started,
        finished_at=_utcnow(),
    )


def _read_telemetry(resident: object) -> BackendTelemetry | None:
    reader = getattr(resident, "telemetry", None)
    if not callable(reader):
        return None
    try:
        telemetry = reader()
    except Exception:  # noqa: BLE001 - unavailable evidence stays unknown
        return None
    return telemetry if isinstance(telemetry, BackendTelemetry) else None


def file_sha256(path: Path | str) -> str | None:
    path = Path(path)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:  # pragma: no cover - filesystem dependent
        return None


def _safe_model_profile(path: str) -> ModelProfile | None:
    from inferenceos.models.loader import ModelMetadataError, load_model_profile

    try:
        return load_model_profile(path)
    except (ModelMetadataError, OSError):
        return None


def deployment_from_loop_result(
    run_id: str,
    loop_result: object,
    model: ModelProfile,
    *,
    backend_name: str | None,
    model_content_hash: str | None = None,
) -> DeploymentState:
    """Build a :class:`DeploymentState` from a completed Phase 9 loop result."""
    from inferenceos.contracts.optimization_loop import LoopResult

    assert isinstance(loop_result, LoopResult)
    changed = bool(loop_result.changed)
    termination = loop_result.termination.reason.value
    if termination in {"restoration_failure", "quality_safety_stop"}:
        status = DeploymentStatus.SAFETY_STOP
    elif changed:
        status = DeploymentStatus.RETAINED
    elif termination in {
        "no_meaningful_opportunity",
        "insufficient_evidence",
        "no_eligible_candidates",
        "all_candidates_rejected",
    }:
        status = DeploymentStatus.NO_CANDIDATE
    else:
        status = DeploymentStatus.NO_CHANGE
    source_experiment = next(
        (
            iteration.experiment_id
            for iteration in reversed(loop_result.run.iterations)
            if iteration.retained and iteration.experiment_id
        ),
        None,
    )
    provenance_material = json.dumps(
        {
            "run_id": run_id,
            "model_path": model.path,
            "backend_name": backend_name,
            "final_configuration": loop_result.final_configuration.model_dump(mode="json"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return DeploymentState(
        run_id=run_id,
        status=status,
        model_ref=model.name,
        model_path=model.path,
        model_content_hash=model_content_hash,
        backend_name=backend_name,
        backend_version=None,
        baseline_configuration=loop_result.initial_configuration,
        final_configuration=loop_result.final_configuration,
        retained=changed,
        retained_candidate_ids=list(loop_result.run.retained_candidate_ids),
        source_experiment_id=source_experiment,
        provenance_hash=hashlib.sha256(provenance_material.encode("utf-8")).hexdigest(),
        notes=[
            "structured deployment state consumed by `inferenceos serve --deployment`",
            f"termination={termination}",
        ],
    )
