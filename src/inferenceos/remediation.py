"""Cross-phase remediation acceptance (real, reproducible evidence).

Covers:
  R1 candidate-surface truth (real executable GPU-offload increase; no forward refs)
  R2 true peak-vs-final RSS measurement
  R4 independent architecture-derived KV validation
  R5 minimal ``serve --deployment`` over a persisted DeploymentState

Perplexity-proxy is deliberately **not** implemented (the plan makes it optional);
this is documented in the README rather than faked here.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.backends.process_control import PeakRssAccumulator
from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.diagnosis import BottleneckDiagnosis, DiagnosisContext
from inferenceos.contracts.enums import (
    BackendType,
    BottleneckKind,
    EvidenceStrength,
    MeasurementMode,
    WorkloadKind,
)
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.fresh import DeploymentState, DeploymentStatus
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.profiling import (
    BaselineReport,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunIdentity,
)
from inferenceos.contracts.system_state import GPUMemoryState, SystemState
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.profiling.runner import COLD_WARM_SEMANTICS
from inferenceos.deployment import (
    DeploymentIntegrityError,
    file_sha256,
    load_deployment_state,
    serve_deployment,
    write_deployment_state,
)
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.experiment.quality_policy import _SEMANTIC_ACTIONS
from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
from inferenceos.models.kv_validation import (
    KV_CACHE_RELATIVE_TOLERANCE,
    derive_expected_kv_cache_bytes,
    validate_reported_kv_cache,
)
from inferenceos.models.loader import load_model_profile
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.selftest import sample_hardware, sample_model


def _largest_model(settings: object) -> object:
    models_dir = Path(getattr(settings, "models_dir"))
    profiles = [load_model_profile(path) for path in sorted(models_dir.glob("*.gguf"))]
    if not profiles:
        raise FileNotFoundError(f"no .gguf models found in {models_dir}")
    return max(profiles, key=lambda profile: profile.file_size_bytes)


def _server_path(settings: object) -> Path:
    base = Path(getattr(settings, "llama_cpp_dir"))
    candidate = base / "llama-server.exe"
    return candidate if candidate.is_file() else base / "llama-server"


def run_remediation_acceptance(
    settings: object, *, probe_prompt: str
) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    scenarios: dict[str, object] = {}
    checks: list[tuple[str, bool]] = []
    registry = build_default_registry()

    # ------------------------------------------------------------------ #
    # R1 — candidate-surface truth                                        #
    # ------------------------------------------------------------------ #
    increase = registry.get_optional("runtime.increase_gpu_offload_layers")
    executor = RuntimeConfigurationExecutor()
    r1_registry = bool(
        increase is not None
        and BottleneckKind.GPU_UNDERUTILIZATION in increase.supported_bottlenecks
        and executor.supports("runtime.increase_gpu_offload_layers")
    )
    forward_refs = [action_id for action_id in _SEMANTIC_ACTIONS if action_id not in registry]

    diagnosis = _gpu_underutilization_diagnosis(settings)
    report = _gpu_underutilization_report(settings)
    plan = generate_candidates(
        diagnosis, report=report, registry=registry, policy=OptimizationPolicy()
    )
    candidate = next(
        (c for c in plan.candidates if c.action_id == "runtime.increase_gpu_offload_layers"),
        None,
    )
    raw_target = candidate.parameters.get("gpu_offload_layers") if candidate else None
    target = raw_target if isinstance(raw_target, int) else None
    r1_candidate = bool(
        candidate is not None
        and candidate.executable
        and target is not None
        and target > 14
        and target <= 28
        # The resource-aware planner never proposes the full layer total blindly.
        and target < 28
    )
    apply_ok = False
    if candidate is not None and target is not None:
        executor.set_configuration(
            RuntimeConfiguration(context_length=2048, gpu_offload_layers=14)
        )
        action = registry.get(candidate.action_id)
        applied = executor.apply(action, candidate.parameters)
        apply_ok = bool(
            applied.applied and executor.current().gpu_offload_layers == target
        )
        restored = executor.restore(
            RuntimeConfiguration(context_length=2048, gpu_offload_layers=14)
        )
        apply_ok = apply_ok and restored.verified
    scenarios["R1_candidate_surface"] = {
        "registry_actions": registry.ids(),
        "increase_action_executable": r1_registry,
        "generated_candidate": (
            {
                "action_id": candidate.action_id,
                "parameters": candidate.parameters,
                "executable": candidate.executable,
            }
            if candidate is not None
            else None
        ),
        "executor_apply_restore_ok": apply_ok,
        "unregistered_semantic_references": forward_refs,
    }
    checks.append(
        (
            "R1a: GPU-underutilization generates a real executable increase-offload candidate",
            r1_registry and r1_candidate and apply_ok,
        )
    )
    checks.append(
        (
            "R1b: no quality-policy action reference is unregistered (no forward refs)",
            not forward_refs,
        )
    )

    # ------------------------------------------------------------------ #
    # R2 — true peak vs final RSS                                         #
    # ------------------------------------------------------------------ #
    accumulator = PeakRssAccumulator()
    for sample in (100, 200, 150):
        accumulator.add(rss_bytes=sample)
    acc_ok = accumulator.peak_rss_bytes == 200 and accumulator.final_rss_bytes == 150
    real_peak = real_final = None
    rss_samples = 0
    peak_differs = False
    r2_real_ok = False
    try:
        model = _largest_model(settings)
        adapter = LlamaCppAdapter(settings=settings)  # type: ignore[arg-type]
        adapter.load(
            model,  # type: ignore[arg-type]
            RuntimeParams(gpu_offload_layers=0, device="none", use_mmap=False, context_length=1024),
        )
        try:
            first = adapter.generate(
                GenerationRequest(prompt=probe_prompt, n_predict=64, seed=7)
            )
            second = adapter.generate(
                GenerationRequest(prompt="Reply with one short word.", n_predict=4, seed=7)
            )
        finally:
            adapter.unload()
        real_peak = second.peak_rss_bytes
        real_final = second.final_rss_bytes
        # For the CLI adapter use the first run's peak as the high-water reference.
        high_water = max(
            value for value in (first.peak_rss_bytes, real_peak) if value is not None
        ) if (first.peak_rss_bytes or real_peak) else None
        peak_differs = bool(
            real_peak is not None and real_final is not None and real_peak > real_final
        )
        r2_real_ok = bool(
            real_peak is not None
            and real_final is not None
            and real_peak >= real_final
            and (peak_differs or (high_water is not None and high_water > real_final))
        )
        scenarios["R2_peak_memory"] = {
            "accumulator_peak_final": [accumulator.peak_rss_bytes, accumulator.final_rss_bytes],
            "first_generation_peak_rss_bytes": first.peak_rss_bytes,
            "second_generation_peak_rss_bytes": real_peak,
            "second_generation_final_rss_bytes": real_final,
            "real_peak_strictly_exceeds_final": peak_differs,
            "measurement": (
                "process-per-generation CLI adapter samples true peak RSS over the whole "
                "process lifetime (load + generation) and records the final sample"
            ),
        }
    except (OSError, RuntimeError, TimeoutError) as exc:  # noqa: BLE001
        scenarios["R2_peak_memory"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(
        (
            "R2a: peak and final RSS are tracked as distinct measurements (deterministic)",
            acc_ok,
        )
    )
    checks.append(
        (
            "R2b: real generation reports peak >= final RSS from the real target process",
            r2_real_ok,
        )
    )

    # ------------------------------------------------------------------ #
    # R4 — independent architecture-derived KV validation                 #
    # ------------------------------------------------------------------ #
    r4_positive = r4_negative = r4_seen = False
    kv_scenario: dict[str, object] = {}
    try:
        model = _largest_model(settings)
        adapter = LlamaCppAdapter(settings=settings)  # type: ignore[arg-type]
        adapter.load(
            model,  # type: ignore[arg-type]
            RuntimeParams(
                gpu_offload_layers=0, device="none", use_mmap=False, context_length=1024
            ),
        )
        try:
            adapter.generate(GenerationRequest(prompt="Hello", n_predict=4, seed=1))
            reported = adapter.telemetry().kv_cache_bytes
        finally:
            adapter.unload()
        key_length = model.key_length or model.head_dim  # type: ignore[attr-defined]
        value_length = model.value_length or model.head_dim  # type: ignore[attr-defined]
        expected = derive_expected_kv_cache_bytes(
            n_layers=model.n_layers,  # type: ignore[attr-defined]
            n_kv_heads=model.n_kv_heads,  # type: ignore[attr-defined]
            key_length=key_length,
            value_length=value_length,
            context_length=1024,
        )
        wrong = derive_expected_kv_cache_bytes(
            n_layers=model.n_layers,  # type: ignore[attr-defined]
            n_kv_heads=model.n_attention_heads,  # type: ignore[attr-defined]
            key_length=key_length,
            value_length=value_length,
            context_length=1024,
        )
        positive = validate_reported_kv_cache(expected_bytes=expected, reported_bytes=reported)
        negative = validate_reported_kv_cache(expected_bytes=wrong, reported_bytes=reported)
        r4_seen = positive.within_tolerance is not None
        r4_positive = positive.within_tolerance is True
        is_gqa = model.n_kv_heads != model.n_attention_heads  # type: ignore[attr-defined]
        r4_negative = (negative.within_tolerance is False) if is_gqa else True
        kv_scenario = {
            "model": model.name,  # type: ignore[attr-defined]
            "context_length": 1024,
            "independent_expected_kv_bytes": expected,
            "attention_head_derivation_kv_bytes": wrong,
            "backend_reported_kv_bytes": reported,
            "relative_error": positive.relative_error,
            "tolerance": KV_CACHE_RELATIVE_TOLERANCE,
            "positive_within_tolerance": positive.within_tolerance,
            "wrong_source_within_tolerance": negative.within_tolerance,
            "is_gqa": is_gqa,
            "basis": positive.basis,
        }
    except (OSError, RuntimeError, TimeoutError) as exc:  # noqa: BLE001
        kv_scenario = {"error": f"{type(exc).__name__}: {exc}"}
    scenarios["R4_kv_validation"] = kv_scenario
    checks.append(
        (
            "R4a: independently derived KV matches the real backend-reported allocation",
            r4_positive and r4_seen,
        )
    )
    checks.append(
        (
            "R4b: the wrong (attention-head) derivation does not falsely agree",
            r4_negative,
        )
    )

    # ------------------------------------------------------------------ #
    # R5 — serve --deployment over a persisted DeploymentState            #
    # ------------------------------------------------------------------ #
    r5_valid = r5_tamper = r5_reject = r5_clean = False
    resident_rss_ok = False
    serve_scenario: dict[str, object] = {}
    tmp = Path(tempfile.mkdtemp(prefix="io_remediation_"))
    try:
        model = _largest_model(settings)
        state = DeploymentState(
            run_id="remediation-serve",
            status=DeploymentStatus.NO_CHANGE,
            model_ref=model.name,  # type: ignore[attr-defined]
            model_path=model.path,  # type: ignore[attr-defined]
            model_content_hash=file_sha256(model.path),  # type: ignore[attr-defined]
            backend_name="llama.cpp (resident)",
            baseline_configuration=RuntimeConfiguration(context_length=1024),
            final_configuration=RuntimeConfiguration(
                context_length=1024, gpu_offload_layers=0, use_mmap=False
            ),
            provenance_hash="remediation-provenance",
        )
        path = tmp / "deployment.json"
        write_deployment_state(path, state)
        loaded = load_deployment_state(path)
        evidence = serve_deployment(
            loaded,
            settings=settings,  # type: ignore[arg-type]
            prompt=probe_prompt,
            n_predict=16,
        )
        r5_valid = bool(
            evidence.served
            and evidence.configuration_matches
            and evidence.residency_verified
            and evidence.identity is not None
            and evidence.probe_text
        )
        r5_clean = bool(evidence.clean_shutdown and not evidence.orphan_process)
        resident_rss_ok = bool(
            evidence.peak_rss_bytes is not None
            and evidence.final_rss_bytes is not None
            and evidence.peak_rss_bytes >= evidence.final_rss_bytes
        )
        serve_scenario = {
            "deployment_path": str(path),
            "served": evidence.served,
            "pid": evidence.pid,
            "identity_create_time": (
                evidence.identity.create_time if evidence.identity else None
            ),
            "configuration_requested": evidence.configuration_requested.as_dict(),
            "configuration_observed": (
                evidence.configuration_observed.as_dict()
                if evidence.configuration_observed
                else None
            ),
            "configuration_matches": evidence.configuration_matches,
            "residency_verified": evidence.residency_verified,
            "probe_text": evidence.probe_text[:200],
            "peak_rss_bytes": evidence.peak_rss_bytes,
            "final_rss_bytes": evidence.final_rss_bytes,
            "clean_shutdown": evidence.clean_shutdown,
            "orphan_process": evidence.orphan_process,
            "validation_checks": evidence.validation.checks,
        }

        # Tampered state must be rejected by the integrity check.
        tampered_path = tmp / "tampered.json"
        payload = loaded.model_dump(mode="json")
        payload["final_configuration"]["context_length"] = 4096
        tampered_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_deployment_state(tampered_path)
            r5_tamper = False
        except DeploymentIntegrityError:
            r5_tamper = True

        # Stale/incompatible state must be rejected before any server starts.
        stale = state.model_copy(
            update={"status": DeploymentStatus.SAFETY_STOP, "integrity_hash": None}
        )
        stale_path = tmp / "stale.json"
        write_deployment_state(stale_path, stale)
        stale_evidence = serve_deployment(
            load_deployment_state(stale_path),
            settings=settings,  # type: ignore[arg-type]
            prompt=probe_prompt,
            n_predict=4,
        )
        r5_reject = bool(not stale_evidence.served and stale_evidence.pid is None)
        serve_scenario["tamper_rejected"] = r5_tamper
        serve_scenario["stale_rejected_before_start"] = r5_reject
        serve_scenario["stale_reasons"] = stale_evidence.validation.reasons
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:  # noqa: BLE001
        serve_scenario = {"error": f"{type(exc).__name__}: {exc}"}
    scenarios["R5_serve"] = serve_scenario
    checks.append(
        (
            "R5a: a valid persisted DeploymentState starts a real resident server and serves",
            r5_valid,
        )
    )
    checks.append(
        (
            "R5b: served configuration/identity verified and shutdown is clean (no orphan)",
            r5_clean,
        )
    )
    checks.append(
        (
            "R5c: tampered and stale/incompatible state are rejected before serving",
            r5_tamper and r5_reject,
        )
    )
    checks.append(
        (
            "R2c: the real resident target reports true peak >= final RSS",
            resident_rss_ok,
        )
    )

    scenarios["R3_perplexity_proxy"] = {
        "implemented": False,
        "decision": (
            "The plan states a cheap perplexity proxy MAY be used; it is not part of the "
            "required v1 acceptance contract, so no speculative perplexity subsystem was "
            "built. No code claims one exists."
        ),
    }
    return scenarios, checks


def _gpu_underutilization_report(settings: object) -> BaselineReport:
    """A controlled profiling report supplying the model + fresh VRAM telemetry.

    The resource-aware planner requires real model geometry and fresh used-VRAM
    telemetry to construct a safe feasible range; this controlled report provides
    both, consistently with ``_gpu_underutilization_diagnosis``.
    """
    hardware = sample_hardware()
    model = sample_model().model_copy(
        update={"name": "remediation-model", "n_layers": 28, "weight_bytes": 2 * 1024**3}
    )
    workload = WorkloadProfile(
        workload_id="short-short",
        kind=WorkloadKind.SHORT,
        description="controlled gpu-underutilization workload",
        prompt_set_ref="short-short@remediation",
        prompts=["controlled prompt"],
        seed=1,
        expected_context_length=2048,
    )
    config = ProfilingConfig(
        backend_name="llama.cpp",
        backend_type=BackendType.VULKAN,
        gpu_layers_offloaded=14,
        gpu_layers_total=28,
        context_length=2048,
        n_predict=32,
        seed=1,
        temperature=0.0,
    )
    now = datetime.now(timezone.utc)
    state = SystemState(
        captured_at=now,
        total_ram_bytes=16 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        memory_pressure=0.35,
        cpu_percent_total=90.0,
        hard_page_faults_per_sec=0.0,
        gpu_memory=[GPUMemoryState(device_index=0, memory_used_bytes=700 * 1024**2)],
        gpu_telemetry_available=True,
        gpu_observed_at=now,
        gpu_source="windows_gpu_engine_counters",
    )
    run = ProfilingRun(
        identity=RunIdentity(
            run_id="remediation-gpu-run",
            profiler_version="remediation",
            mode=MeasurementMode.WARM,
            started_at=now,
            finished_at=now,
        ),
        context=RunContext(
            hardware=hardware, model=model, workload=workload, config=config
        ),
        observation=Observation(
            observation_id="remediation-gpu-obs",
            captured_at=now,
            mode=MeasurementMode.WARM,
            backend=BackendType.VULKAN,
            device="Radeon Pro 560X",
            model_ref=model.name,
            workload_ref=workload.workload_id,
            metrics=RunMetrics(
                decode_tokens_per_sec=6.0,
                total_latency_s=3.0,
                process_cpu_percent=520.0,
                gpu_utilization_percent=3.0,
                gpu_memory_used_bytes=700 * 1024**2,
                available_ram_bytes=8 * 1024**3,
                total_ram_bytes=16 * 1024**3,
            ),
            system_state=state,
        ),
        metric_quality={},
    )
    return BaselineReport(
        profiler_version="remediation",
        suite_version="remediation",
        generated_at=now,
        hardware=hardware,
        backend=config,
        workloads=[workload],
        runs=[run],
        groups=[],
        cold_warm_semantics=COLD_WARM_SEMANTICS,
        resident_warm_supported=False,
    )


def _gpu_underutilization_diagnosis(settings: object) -> BottleneckDiagnosis:
    """A controlled GPU-underutilization diagnosis over real hardware characteristics."""
    hardware = sample_hardware()
    bottleneck = Bottleneck(
        bottleneck_id="bn-remediation-gpu",
        kind=BottleneckKind.GPU_UNDERUTILIZATION,
        confidence=0.75,
        evidence_strength=EvidenceStrength.MEDIUM,
        is_meaningful=True,
        summary="Offload configured but GPU idle while process CPU is saturated (controlled)",
        evidence=[
            EvidenceItem(
                metric="gpu_utilization_percent",
                observed_value=3.0,
                reference_value=20.0,
                ratio=0.15,
                description="GPU configured to hold half the layers but idle",
            ),
            EvidenceItem(
                metric="process_cpu_saturation",
                observed_value=0.95,
                reference_value=0.8,
                ratio=1.19,
                description="process CPU saturated while GPU idle",
            ),
        ],
        contributing_factors=["offload configured", "CPU-bound execution"],
        recommended_surface=[],
        source="rule.gpu_underutilization.v1",
    )
    context = DiagnosisContext(
        source_report_path="<remediation-controlled-gpu-underutilization>",
        detector_version="remediation.controlled",
        hardware=hardware,
        backend=ProfilingConfig(
            backend_name="llama.cpp",
            backend_type=BackendType.VULKAN,
            gpu_layers_offloaded=14,
            gpu_layers_total=28,
            context_length=2048,
        ),
        model_refs=["remediation-model"],
        workload_ids=["short-short"],
    )
    return BottleneckDiagnosis(
        diagnosis_id="diag-remediation-gpu",
        created_at=datetime.now(timezone.utc),
        context=context,
        meaningful_opportunity=True,
        insufficient_evidence=False,
        escalation_required=False,
        confidence=0.75,
        evidence_strength=EvidenceStrength.MEDIUM,
        primary=bottleneck,
        summary="Controlled GPU-underutilization diagnosis for candidate-surface validation",
    )
