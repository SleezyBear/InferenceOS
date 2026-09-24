"""Phase 0 self-tests: contract round-trips and the ReasoningEngine interface.

These are used by the ``inferenceos contracts-check`` and
``inferenceos reasoning-selftest`` commands so a human can independently confirm the
foundation works, and are also exercised directly by the test suite.
"""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.audit.schema import AppliedChange, AuditRecord
from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.candidate import Candidate
from inferenceos.contracts.diagnosis import BottleneckDiagnosis, DiagnosisContext
from inferenceos.contracts.enums import (
    ApprovalRequirement,
    AttentionType,
    BackendType,
    BottleneckKind,
    CandidateEligibility,
    CandidateKind,
    CandidatePlanOutcome,
    Decision,
    EvidenceStrength,
    ExperimentDecisionKind,
    ExperimentStage,
    GPUDeviceType,
    GPUVendor,
    MeasurementMode,
    ObservationSource,
    PreconditionKind,
    QualityStatus,
    QuantizationType,
    ReasoningEngineKind,
    ReasoningStatus,
    RestorationStatus,
    StorageMediaType,
    WorkloadKind,
)
from inferenceos.contracts.experiment import (
    ApplicationResult,
    BaselineSnapshot,
    ExperimentChange,
    ExperimentComparison,
    ExperimentDecision,
    ExperimentEvent,
    ExperimentMeasurement,
    ExperimentOverhead,
    ExperimentPolicy,
    ExperimentRequest,
    ExperimentResult,
    MeasurementDelta,
    QualityCheck,
    QualityScore,
    QualityValidationResult,
    RestorationResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.hardware import (
    BackendAvailability,
    BackendDevice,
    CacheProfile,
    CPUProfile,
    GPUProfile,
    HardwareProfile,
    MemoryModule,
    MemoryProfile,
    OSProfile,
    StorageDevice,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.optimization import (
    BenefitEstimate,
    CandidateAction,
    CandidatePlan,
    CandidatePrecondition,
    CostEstimate,
    OptimizationCandidate,
    OptimizerOverhead,
    RiskAssessment,
)
from inferenceos.contracts.profiling import MetricStats, ProfilingConfig
from inferenceos.contracts.reasoning import ReasoningRequest, ReasoningResponse
from inferenceos.contracts.system_state import (
    GPUEngineUtilization,
    ProcessState,
    SystemState,
)
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.reasoning.stub import StubReasoningEngine
from inferenceos.signature import Signature


def _now() -> datetime:
    return datetime.now(timezone.utc)


def sample_hardware() -> HardwareProfile:
    return HardwareProfile(
        captured_at=_now(),
        hostname="sample-host",
        os=OSProfile(caption="Windows 11", version="10.0", build="26100", architecture="64-bit"),
        cpu=CPUProfile(
            name="Sample CPU",
            vendor="GenuineIntel",
            architecture="x86_64",
            physical_cores=6,
            logical_cores=12,
            max_clock_mhz=2600,
            caches=CacheProfile(l1_bytes=393216, l2_bytes=1572864, l3_bytes=9437184),
            instruction_sets=["sse4_2", "avx", "avx2"],
            detected_via="cpuid-probe",
        ),
        gpus=[
            GPUProfile(
                name="Radeon Pro 560X",
                vendor=GPUVendor.AMD,
                device_type=GPUDeviceType.DISCRETE,
                vram_bytes=4 * 1024**3,
                driver_version="31.0.21921.1000",
                supported_backends=[BackendType.VULKAN],
                vulkan_device_index=0,
                vulkan_api_version="1.3.260",
                pci_id="1002:67ef",
            )
        ],
        memory=MemoryProfile(
            total_bytes=16 * 1024**3,
            available_bytes=8 * 1024**3,
            modules=[MemoryModule(capacity_bytes=8 * 1024**3, speed_mhz=2400, manufacturer="SK Hynix")],
        ),
        storage=[
            StorageDevice(
                name="APPLE SSD AP0512",
                media_type=StorageMediaType.SSD,
                bus_type="SCSI",
                size_bytes=500 * 1000**3,
                free_bytes=60 * 1000**3,
                volumes=["C:"],
            )
        ],
        backends_available=[BackendType.VULKAN, BackendType.CPU],
        backend_availability=[
            BackendAvailability(
                backend=BackendType.CPU,
                detected=True,
                usable=True,
                devices=[
                    BackendDevice(
                        name="Sample CPU",
                        device_type=GPUDeviceType.CPU,
                        usable=True,
                        detail="6 physical / 12 logical cores",
                    )
                ],
                detail="CPU execution always available",
            ),
            BackendAvailability(
                backend=BackendType.VULKAN,
                detected=True,
                usable=True,
                devices=[
                    BackendDevice(
                        name="Radeon Pro 560X",
                        device_index=0,
                        device_type=GPUDeviceType.DISCRETE,
                        vram_bytes=4 * 1024**3,
                        usable=True,
                        detail="vulkan-api=1.3.260",
                    )
                ],
                detail="1 Vulkan physical device(s) enumerated",
            ),
            BackendAvailability(
                backend=BackendType.CUDA,
                detected=False,
                usable=False,
                detail="CUDA runtime (nvcuda.dll) not present",
            ),
        ],
        vulkan_available=True,
        cuda_available=False,
        rocm_available=False,
    )


def sample_model() -> ModelProfile:
    n_heads = 16
    n_kv_heads = 2
    head_dim = 64
    n_layers = 24
    kv_per_token = 2 * n_kv_heads * head_dim * 2 * n_layers
    return ModelProfile(
        path="models/sample.gguf",
        name="sample",
        format="gguf",
        architecture="qwen2",
        quantization=QuantizationType.Q4_K_M,
        file_size_bytes=1_000_000_000,
        parameter_count=1_500_000_000,
        tensor_count=291,
        trained_context_length=32768,
        n_layers=n_layers,
        n_attention_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        key_length=head_dim,
        value_length=head_dim,
        embedding_dim=1536,
        vocabulary_size=151936,
        attention_type=AttentionType.UNKNOWN,
        weight_bytes=1_000_000_000,
        kv_cache_bytes_per_token=kv_per_token,
        kv_cache_estimates=[],
        metadata={"general.architecture": "qwen2"},
    )


def sample_workload() -> WorkloadProfile:
    return WorkloadProfile(
        workload_id="phase0-smoke",
        kind=WorkloadKind.SHORT,
        description="Single short prompt",
        prompts=["Say hello."],
        seed=1234,
        expected_context_length=512,
        concurrency=1,
        modes=[MeasurementMode.COLD, MeasurementMode.WARM],
    )


def sample_system_state() -> SystemState:
    return SystemState(
        captured_at=_now(),
        total_ram_bytes=16 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        committed_bytes=6 * 1024**3,
        commit_limit_bytes=18 * 1024**3,
        memory_pressure=0.5,
        cpu_percent_total=12.5,
        hard_page_faults_per_sec=3.0,
        processes=[ProcessState(pid=1234, name="llama-cli.exe", cpu_percent=55.0, rss_bytes=900 * 1024**2)],
        gpu_engines=[GPUEngineUtilization(device_index=0, engine="Compute", utilization_percent=42.0)],
        target_process=ProcessState(pid=1234, name="llama-cli.exe", cpu_percent=55.0, rss_bytes=900 * 1024**2),
    )


def sample_observation() -> Observation:
    return Observation(
        observation_id="obs-1",
        captured_at=_now(),
        source=ObservationSource.HARNESS,
        mode=MeasurementMode.COLD,
        backend=BackendType.VULKAN,
        device="Radeon Pro 560X",
        model_ref="sample",
        workload_ref="phase0-smoke",
        metrics=RunMetrics(
            model_load_time_s=0.7,
            ttft_s=0.3,
            decode_tokens_per_sec=22.0,
            total_latency_s=2.0,
            generated_tokens=64,
            gpu_utilization_percent=42.0,
        ),
        generated_text="Hello!",
    )


def sample_bottleneck() -> Bottleneck:
    return Bottleneck(
        bottleneck_id="bn-1",
        kind=BottleneckKind.MEMORY_PRESSURE,
        confidence=0.82,
        is_meaningful=True,
        summary="Model footprint dominates available RAM",
        evidence=[
            EvidenceItem(
                metric="model_rss_over_available_ram",
                observed_value=0.75,
                reference_value=0.25,
                ratio=3.0,
                description="Model RSS is 3x the significance threshold vs available RAM",
            )
        ],
        contributing_factors=["large context", "no mmap"],
        recommended_surface=[CandidateKind.CONTEXT_LENGTH, CandidateKind.MMAP],
        source="rule.memory_pressure.v1",
    )


def sample_diagnosis() -> BottleneckDiagnosis:
    return BottleneckDiagnosis(
        diagnosis_id="diag-1",
        created_at=_now(),
        context=DiagnosisContext(
            source_report_path="<sample>",
            detector_version="sample-detector",
            hardware=sample_hardware(),
            backend=ProfilingConfig(backend_name="sample-backend"),
            model_refs=["sample"],
            workload_ids=["phase0-smoke"],
        ),
        meaningful_opportunity=True,
        insufficient_evidence=False,
        escalation_required=False,
        confidence=0.6,
        evidence_strength=EvidenceStrength.MEDIUM,
        primary=sample_bottleneck(),
        summary="Sample memory-pressure diagnosis",
    )


def sample_candidate_action() -> CandidateAction:
    from inferenceos.optimization.actions import build_default_registry

    return build_default_registry().get("runtime.right_size_context")


def sample_optimization_candidate() -> OptimizationCandidate:
    action = sample_candidate_action()
    return OptimizationCandidate(
        candidate_id="cand-phase6-1",
        action_id=action.action_id,
        action_name=action.name,
        surface=action.surface,
        subsystem=action.subsystem,
        parameters={"context_length": 8192},
        source_diagnosis_id="diag-1",
        source_bottleneck_id="bn-1",
        diagnosis_kind=BottleneckKind.MEMORY_PRESSURE,
        diagnosis_confidence=0.6,
        diagnosis_strength=EvidenceStrength.MEDIUM,
        causal_rationale=action.causal_rationale,
        evidence_refs=["E1"],
        hardware=sample_hardware(),
        model_ref="sample",
        model_architecture="qwen2",
        workload_id="phase0-smoke",
        workload_kind=WorkloadKind.SHORT,
        observed_metrics={"peak_rss_bytes": 1_300_000_000.0},
        preconditions=[
            CandidatePrecondition(
                check_id="evidence_supported",
                kind=PreconditionKind.EVIDENCE,
                description="diagnosis evidence supports this action",
                satisfied=True,
            )
        ],
        benefit=BenefitEstimate(
            metric="peak_rss_bytes",
            direction="decrease",
            unit="bytes",
            known=True,
            magnitude_low=100_000_000.0,
            magnitude_high=100_000_000.0,
            confidence=0.6,
            evidence_basis=["E1"],
        ),
        cost=CostEstimate(execution_cost_s=1.0, total_cost_s=3.0),
        risk=RiskAssessment(
            risk_level=action.risk_level,
            reversibility=action.reversibility,
            scope=action.scope,
            privilege=action.privilege,
        ),
        approval=ApprovalRequirement.NOT_REQUIRED,
        eligibility=CandidateEligibility.ELIGIBLE,
        generated_by="deterministic",
        executable=True,
    )


def sample_candidate_plan() -> CandidatePlan:
    return CandidatePlan(
        plan_id="plan-1",
        source_diagnosis_id="diag-1",
        source_diagnosis=sample_diagnosis(),
        effective_bottleneck=BottleneckKind.MEMORY_PRESSURE,
        effective_confidence=0.6,
        outcome=CandidatePlanOutcome.CANDIDATES_GENERATED,
        terminal_reason="1 eligible candidate generated; none executed",
        candidates=[sample_optimization_candidate()],
        eligible_ids=["cand-phase6-1"],
        ranking=["cand-phase6-1"],
        selected_ids=["cand-phase6-1"],
        overhead=OptimizerOverhead(
            generation_latency_s=0.01, candidates_generated=1, eligible=1
        ),
    )


def sample_candidate() -> Candidate:
    return Candidate(
        candidate_id="cand-1",
        kind=CandidateKind.CONTEXT_LENGTH,
        description="Reduce context length from 32768 to 8192",
        parameters={"context_length": 8192},
        expected_benefit=0.6,
        confidence=0.7,
        cost=12.0,
        expected_value=(0.6 * 0.7) / 12.0,
        rationale="KV-cache footprint scales linearly with context length",
        source_bottleneck_id="bn-1",
    )


def sample_experiment() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp-1",
        candidate_id="cand-1",
        baseline_observation_id="obs-1",
        result_observation_id="obs-2",
        performance_delta={"decode_tokens_per_sec": 3.2, "peak_rss_bytes": -400 * 1024**2},
        quality_before=[QualityScore(metric="task_accuracy", value=0.9)],
        quality_after=[QualityScore(metric="task_accuracy", value=0.89)],
        tolerance_band=0.05,
        decision=Decision.KEEP,
        notes="Within tolerance; performance improved.",
    )


def sample_runtime_configuration() -> RuntimeConfiguration:
    return RuntimeConfiguration(
        context_length=32768, gpu_offload_layers=0, threads=None, use_mmap=None
    )


def sample_experiment_policy() -> ExperimentPolicy:
    return ExperimentPolicy()


def sample_experiment_request() -> ExperimentRequest:
    baseline = sample_runtime_configuration()
    return ExperimentRequest(
        experiment_id="exp-1",
        candidate_id="cand-phase6-1",
        action_id="runtime.right_size_context",
        parameters={"context_length": 8192},
        source_diagnosis_id="diag-1",
        source_plan_id="plan-1",
        diagnosed_bottleneck="memory_pressure",
        bottleneck_metric="peak_rss_bytes",
        bottleneck_direction="decrease",
        model=sample_model(),
        hardware=sample_hardware(),
        workload=sample_workload(),
        backend=ProfilingConfig(
            backend_name="llama.cpp", backend_type=BackendType.CPU, context_length=32768
        ),
        baseline_configuration=baseline,
        experiment_configuration=baseline.model_copy(update={"context_length": 8192}),
        policy=sample_experiment_policy(),
        seed=1234,
    )


def _sample_metric_stats() -> dict[str, MetricStats]:
    return {
        "decode_tokens_per_sec": MetricStats(
            metric="decode_tokens_per_sec", count=2, mean=22.0, median=22.0,
            stdev=0.4, min=21.6, max=22.4, cv=0.018,
        ),
        "peak_rss_bytes": MetricStats(
            metric="peak_rss_bytes", count=2, mean=1_300_000_000.0, median=1_300_000_000.0,
            stdev=1_000_000.0, min=1_299_000_000.0, max=1_301_000_000.0, cv=0.0008,
        ),
    }


def sample_experiment_measurement() -> ExperimentMeasurement:
    return ExperimentMeasurement(
        measurement_id="meas-1",
        phase="baseline",
        configuration=sample_runtime_configuration(),
        repetitions=2,
        valid_runs=2,
        run_ids=["run-1", "run-2"],
        metrics=_sample_metric_stats(),
        run_metrics=[
            RunMetrics(decode_tokens_per_sec=22.0, peak_rss_bytes=1_300_000_000)
        ],
        generated_texts=["A key value cache stores attention state."],
    )


def sample_baseline_snapshot() -> BaselineSnapshot:
    return BaselineSnapshot(
        snapshot_id="snap-1",
        configuration=sample_runtime_configuration(),
        model_ref="sample",
        model_path="models/sample.gguf",
        workload_id="phase0-smoke",
        metrics=_sample_metric_stats(),
        quality_reference_text="A key value cache stores attention state.",
    )


def sample_quality_result() -> QualityValidationResult:
    return QualityValidationResult(
        status=QualityStatus.OK,
        validated=True,
        execution_ok=True,
        structural_ok=True,
        reference_available=True,
        deterministic_match=True,
        similarity=1.0,
        checks=[
            QualityCheck(
                check_id="execution_completed",
                description="output produced",
                passed=True,
            )
        ],
        rationale=["output matches the deterministic reference"],
    )


def sample_application() -> ApplicationResult:
    return ApplicationResult(
        applied=True,
        action_id="runtime.right_size_context",
        supported=True,
        changes=[
            ExperimentChange(
                parameter="context_length", prior_value=32768, applied_value=8192
            )
        ],
    )


def sample_restoration() -> RestorationResult:
    config = sample_runtime_configuration()
    return RestorationResult(
        attempted=True,
        status=RestorationStatus.RESTORED,
        expected_configuration=config,
        restored_configuration=config,
        verified=True,
    )


def sample_experiment_overhead() -> ExperimentOverhead:
    return ExperimentOverhead(
        baseline_setup_s=2.0,
        application_s=0.001,
        measurement_s=5.0,
        validation_s=0.01,
        restoration_s=0.001,
        total_s=7.0,
    )


def sample_comparison() -> ExperimentComparison:
    delta = MeasurementDelta(
        metric="decode_tokens_per_sec",
        role="objective",
        direction="higher_is_better",
        baseline_mean=22.0,
        candidate_mean=22.5,
        absolute_delta=0.5,
        relative_delta=0.023,
        sample_count=2,
        noise_band=0.1,
        improved=True,
        significant=True,
    )
    return ExperimentComparison(
        deltas={"decode_tokens_per_sec": delta},
        bottleneck_metric="peak_rss_bytes",
        bottleneck_direction="lower_is_better",
        bottleneck_improved=True,
        bottleneck_persistent=False,
        sample_count=2,
    )


def sample_decision() -> ExperimentDecision:
    return ExperimentDecision(
        kind=ExperimentDecisionKind.IMPROVED,
        keep=True,
        reason="bottleneck improved beyond noise",
        confidence=0.8,
        bottleneck_improved=True,
        restoration_status=RestorationStatus.RESTORED,
    )


def sample_experiment_result() -> ExperimentResult:
    measurement = sample_experiment_measurement()
    return ExperimentResult(
        experiment_id="exp-1",
        candidate_id="cand-phase6-1",
        action_id="runtime.right_size_context",
        parameters={"context_length": 8192},
        request=sample_experiment_request(),
        status=ExperimentDecisionKind.IMPROVED,
        baseline_snapshot=sample_baseline_snapshot(),
        baseline_measurement=measurement,
        candidate_measurement=measurement.model_copy(update={"phase": "candidate"}),
        application=sample_application(),
        quality=sample_quality_result(),
        comparison=sample_comparison(),
        decision_detail=sample_decision(),
        restoration=sample_restoration(),
        events=[ExperimentEvent(stage=ExperimentStage.REQUESTED, detail="sample")],
        overhead=sample_experiment_overhead(),
        executed=True,
        environment_restored=True,
        decision=Decision.KEEP,
        notes="sample",
    )


def sample_reasoning_request() -> ReasoningRequest:
    return ReasoningRequest(
        request_id="req-1",
        reason="deterministic confidence below threshold",
        objective="Decide whether memory pressure is causal",
        hardware=sample_hardware(),
        model=sample_model(),
        workload=sample_workload(),
        observations=[sample_observation()],
        deterministic_findings=[sample_bottleneck()],
        constraints=["do not modify other processes"],
    )


def sample_reasoning_response() -> ReasoningResponse:
    return ReasoningResponse(
        request_id="req-1",
        status=ReasoningStatus.OK,
        engine=ReasoningEngineKind.STUB,
        model_used="stub-reasoner",
        conclusion="Stub echo",
        bottleneck=sample_bottleneck(),
        recommended_candidates=[sample_candidate()],
        confidence=0.5,
        rationale="Phase 0 stub",
        evidence_refs=["bn-1"],
        duration_s=0.001,
    )


def sample_audit_record() -> AuditRecord:
    return AuditRecord(
        record_id="audit-1",
        signature=Signature(hardware="hw", model="m", workload="w"),
        baseline=sample_observation(),
        diagnosis=sample_bottleneck(),
        reasoning_request=sample_reasoning_request(),
        reasoning_response=sample_reasoning_response(),
        candidates=[sample_candidate()],
        experiment=sample_experiment(),
        decision=Decision.KEEP,
        applied_changes=[AppliedChange(parameter="context_length", prior_value=32768, applied_value=8192)],
        notes="Phase 0 schema sample",
    )


def sample_contracts() -> dict[str, ContractBase]:
    """Return one representative instance of every core contract."""
    return {
        "HardwareProfile": sample_hardware(),
        "ModelProfile": sample_model(),
        "WorkloadProfile": sample_workload(),
        "SystemState": sample_system_state(),
        "Observation": sample_observation(),
        "Bottleneck": sample_bottleneck(),
        "BottleneckDiagnosis": sample_diagnosis(),
        "Candidate": sample_candidate(),
        "CandidateAction": sample_candidate_action(),
        "OptimizationCandidate": sample_optimization_candidate(),
        "CandidatePlan": sample_candidate_plan(),
        "ExperimentResult": sample_experiment_result(),
        "RuntimeConfiguration": sample_runtime_configuration(),
        "ExperimentPolicy": sample_experiment_policy(),
        "ExperimentRequest": sample_experiment_request(),
        "BaselineSnapshot": sample_baseline_snapshot(),
        "ExperimentMeasurement": sample_experiment_measurement(),
        "QualityValidationResult": sample_quality_result(),
        "ExperimentComparison": sample_comparison(),
        "ExperimentDecision": sample_decision(),
        "ApplicationResult": sample_application(),
        "RestorationResult": sample_restoration(),
        "ExperimentOverhead": sample_experiment_overhead(),
        "ReasoningRequest": sample_reasoning_request(),
        "ReasoningResponse": sample_reasoning_response(),
        "AuditRecord": sample_audit_record(),
    }


def verify_contract_roundtrip() -> list[str]:
    """Serialize and re-validate every sample contract; return the passed names.

    Raises ``AssertionError`` if any contract does not survive a JSON round-trip.
    """
    passed: list[str] = []
    for name, instance in sample_contracts().items():
        payload = instance.model_dump_json()
        restored = type(instance).model_validate_json(payload)
        if restored != instance:
            raise AssertionError(f"{name} failed JSON round-trip")
        passed.append(name)
    return passed


def run_reasoning_selftest() -> ReasoningResponse:
    """Round-trip a real request through the stub ReasoningEngine interface."""
    engine = StubReasoningEngine()
    if not engine.is_available():
        raise AssertionError("stub engine unexpectedly unavailable")
    request = sample_reasoning_request()
    response = engine.reason(request)
    if response.request_id != request.request_id:
        raise AssertionError("reasoning response request_id mismatch")
    restored = ReasoningResponse.model_validate_json(response.model_dump_json())
    if restored != response:
        raise AssertionError("reasoning response failed JSON round-trip")
    return response
