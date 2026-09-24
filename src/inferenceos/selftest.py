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
    AnomalyClass,
    ApplicabilityClass,
    ApprovalOutcome,
    ApprovalRequirement,
    AttentionType,
    BackendType,
    BottleneckKind,
    CandidateEligibility,
    CandidateKind,
    CandidatePlanOutcome,
    CheckpointStatus,
    Decision,
    DimensionStatus,
    EvidenceStrength,
    ExperimentDecisionKind,
    ExperimentStage,
    GPUDeviceType,
    GPUVendor,
    InterventionClass,
    KnowledgeAvailability,
    KnowledgeEventType,
    KnowledgeHealthStatus,
    KnowledgeSelectionEffect,
    KnowledgeWriteStatus,
    LoopDecisionKind,
    LoopIterationOutcome,
    LoopState,
    LoopTerminationReason,
    MeasurementMode,
    ObservationSource,
    OptimizationOutcome,
    PreconditionKind,
    QualityStatus,
    QuantizationType,
    ReasoningEngineKind,
    ReasoningStatus,
    RestorationStatus,
    RevalidationTrigger,
    RuntimeState,
    RuntimeTerminationReason,
    StorageMediaType,
    WorkloadKind,
    WorkloadState,
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
from inferenceos.contracts.knowledge import (
    ApplicabilityContext,
    ApplicabilityResult,
    ContextFingerprint,
    DimensionMatch,
    KnowledgeEvent,
    KnowledgeHealth,
    KnowledgeQuery,
    KnowledgeWriteResult,
    OptimizationRecord,
    RollbackRecord,
    RollbackVerification,
    StorageStats,
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
from inferenceos.contracts.optimization_loop import (
    KnowledgeEvidence,
    LoopContext,
    LoopDecision,
    LoopIteration,
    LoopOverhead,
    LoopPolicy,
    LoopResult,
    LoopStageEvent,
    LoopTermination,
    OptimizationBudget,
    OptimizationRun,
    SafetyCheckpoint,
)
from inferenceos.contracts.runtime import (
    ApprovalRecord,
    InterventionCostModel,
    InterventionEconomics,
    MetricDeviation,
    ObservationWindow,
    ResourceImpact,
    RuntimeBaseline,
    RuntimeBudget,
    RuntimeCheckpoint,
    RuntimeCooldown,
    RuntimeIntervention,
    RuntimeKnowledgeQuery,
    RuntimeMonitorResult,
    RuntimeMonitorRun,
    RuntimeNotification,
    RuntimeObservation,
    RuntimeOpportunity,
    RuntimeOverhead,
    RuntimePolicy,
    RuntimeRecommendation,
    RuntimeStateEvent,
    RuntimeTermination,
    WorkloadDescriptor,
    WorkloadSnapshot,
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


def sample_context_fingerprint() -> ContextFingerprint:
    return ContextFingerprint(
        identity_hash="identity-abc123",
        conditions_hash="conditions-def456",
        hardware_key="cpu=Sample CPU;cores=6/12;ram=17179869184;gpu=amd:Radeon Pro 560X",
        cpu_key="GenuineIntel:Sample CPU;cores=6/12",
        gpu_key="amd:Radeon Pro 560X:discrete:4294967296",
        os_key="Windows 11;10.0.26100;64-bit",
        backend_key="llama.cpp@b11139;vulkan",
        model_key="sample;arch=qwen2;quant=q4_k_m;layers=24;size=1000000000",
        quantization="q4_k_m",
        model_size_bytes=1_000_000_000,
        workload_key="phase0-smoke;kind=short",
        runtime_key='{"context_length":32768,"gpu_offload_layers":0}',
        diagnosis="memory_pressure",
        conditions={"available_ram_bytes": 8 * 1024**3, "memory_pressure": 0.7},
    )


def sample_applicability_context() -> ApplicabilityContext:
    return ApplicabilityContext(
        fingerprint=sample_context_fingerprint(),
        hardware=sample_hardware(),
        model=sample_model(),
        workload=sample_workload(),
        backend=ProfilingConfig(backend_name="llama.cpp", backend_version="b11139"),
        runtime_configuration=sample_runtime_configuration(),
        diagnosed_bottleneck="memory_pressure",
    )


def sample_optimization_record() -> OptimizationRecord:
    return OptimizationRecord(
        record_id="opt-exp-1",
        experiment_id="exp-1",
        candidate_id="cand-phase6-1",
        action_id="runtime.right_size_context",
        action_registered=True,
        parameters={"context_length": 8192},
        diagnosis_id="diag-1",
        diagnosed_bottleneck="memory_pressure",
        outcome=OptimizationOutcome.IMPROVED,
        decision=ExperimentDecisionKind.IMPROVED,
        keep=True,
        bottleneck_improved=True,
        restoration_status=RestorationStatus.RESTORED,
        environment_restored=True,
        quality_status=QualityStatus.OK,
        quality_validated=True,
        quality_similarity=1.0,
        context=sample_applicability_context(),
        baseline_metrics=_sample_metric_stats(),
        candidate_metrics=_sample_metric_stats(),
        measured_benefit={"peak_rss_bytes": -0.05},
        measured_cost={"experiment_overhead_s": 7.0},
        experiment_overhead_s=7.0,
        evidence=["bottleneck=peak_rss_bytes improved=True"],
        uncertainty="2 samples; low variance",
        rollback_id="rb-exp-1",
    )


def sample_rollback_verification() -> RollbackVerification:
    return RollbackVerification(
        verified=True,
        action_registered=True,
        parameters_match=True,
        expected_matches_previous=True,
        context_compatible=True,
        reasons=["rollback record consistent and integrity metadata present"],
    )


def sample_rollback_record() -> RollbackRecord:
    baseline = sample_runtime_configuration()
    return RollbackRecord(
        rollback_id="rb-exp-1",
        experiment_id="exp-1",
        candidate_id="cand-phase6-1",
        action_id="runtime.right_size_context",
        target_scope="target_model_runtime",
        previous_state=baseline,
        resulting_state=baseline.model_copy(update={"context_length": 8192}),
        expected_configuration=baseline,
        changes=[
            ExperimentChange(
                parameter="context_length", prior_value=32768, applied_value=8192
            )
        ],
        restore_parameters={"context_length": 32768},
        experiment_restored=True,
        restoration_status=RestorationStatus.RESTORED,
        context_identity="identity-abc123",
        verification=sample_rollback_verification(),
        integrity_hash="integrity-xyz789",
    )


def sample_dimension_match() -> DimensionMatch:
    return DimensionMatch(
        dimension="hardware",
        status=DimensionStatus.MATCH,
        historical="cpu=Sample CPU;cores=6/12",
        current="cpu=Sample CPU;cores=6/12",
    )


def sample_applicability_result() -> ApplicabilityResult:
    return ApplicabilityResult(
        classification=ApplicabilityClass.DIRECTLY_APPLICABLE,
        record_id="opt-exp-1",
        experiment_id="exp-1",
        action_id="runtime.right_size_context",
        record_outcome=OptimizationOutcome.IMPROVED,
        score=1.0,
        components={"hardware": 1.0, "model": 1.0, "backend": 1.0},
        matches=[sample_dimension_match()],
        revalidation_triggers=[RevalidationTrigger.NONE],
        reasons=["classification=directly_applicable"],
        rationale="identity dimensions match; historical evidence is directly relevant",
    )


def sample_knowledge_query() -> KnowledgeQuery:
    return KnowledgeQuery(action_id="runtime.right_size_context", limit=10)


def sample_knowledge_write_result() -> KnowledgeWriteResult:
    return KnowledgeWriteResult(
        status=KnowledgeWriteStatus.PERSISTED,
        experiment_id="exp-1",
        record_id="opt-exp-1",
        duplicate=False,
        detail="record persisted",
        latency_s=0.001,
    )


def sample_knowledge_event() -> KnowledgeEvent:
    return KnowledgeEvent(
        event_id="evt-1",
        record_id="opt-exp-1",
        event_type=KnowledgeEventType.RECORDED,
        detail="outcome=improved",
    )


def sample_knowledge_health() -> KnowledgeHealth:
    return KnowledgeHealth(
        status=KnowledgeHealthStatus.OK,
        schema_version="1.0",
        expected_schema_version="1.0",
        record_count=1,
        rollback_count=1,
        event_count=1,
        db_size_bytes=16384,
        detail="knowledge store healthy",
    )


def sample_storage_stats() -> StorageStats:
    return StorageStats(
        db_path="data/knowledge.db",
        db_size_bytes=16384,
        optimization_records=1,
        rollback_records=1,
        knowledge_events=1,
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


def sample_optimization_budget() -> OptimizationBudget:
    return OptimizationBudget(
        max_iterations=4,
        max_experiments=4,
        max_wall_clock_s=1800.0,
        max_optimizer_overhead_s=300.0,
        max_deep_diagnoses=2,
        iterations_used=1,
        experiments_used=1,
    )


def sample_loop_policy() -> LoopPolicy:
    return LoopPolicy(budget=sample_optimization_budget())


def sample_safety_checkpoint() -> SafetyCheckpoint:
    return SafetyCheckpoint(
        checkpoint_id="ckpt-sample",
        status=CheckpointStatus.CAPTURED,
        runtime_configuration=sample_runtime_configuration(),
        model_ref="sample",
        model_path="models/sample.gguf",
        hardware_hostname="sample-host",
        workload_id="short-short",
        diagnosis_id="diag-1",
        context_identity="identity-abc123",
        baseline_metrics={"peak_rss_bytes": 1_300_000_000.0},
    )


def sample_loop_stage_event() -> LoopStageEvent:
    return LoopStageEvent(
        stage=LoopState.INITIALIZED,
        detail="baseline captured",
        data={"configuration": {"context_length": 32768}},
    )


def sample_knowledge_evidence() -> KnowledgeEvidence:
    return KnowledgeEvidence(
        candidate_id="cand-phase6-1",
        action_id="runtime.right_size_context",
        effect=KnowledgeSelectionEffect.NEUTRAL,
        availability=KnowledgeAvailability.OK,
        records_considered=1,
        classifications={"directly_applicable": 1},
        outcomes={"no_meaningful_improvement": 1},
        applicable_negative=1,
        detail="records=1; effect=neutral",
    )


def sample_loop_decision() -> LoopDecision:
    return LoopDecision(
        kind=LoopDecisionKind.PROCEED,
        reason="selected runtime.right_size_context",
        selected_candidate_id="cand-phase6-1",
        action_id="runtime.right_size_context",
        params={"context_length": 8192},
        ranked_candidate_ids=["cand-phase6-1"],
        knowledge=[sample_knowledge_evidence()],
    )


def sample_loop_iteration() -> LoopIteration:
    return LoopIteration(
        iteration=1,
        diagnosis_id="diag-1",
        diagnosis_kind="memory_pressure",
        diagnosis_confidence=0.6,
        plan_id="plan-1",
        plan_outcome=CandidatePlanOutcome.CANDIDATES_GENERATED,
        candidates_considered=1,
        eligible_candidate_ids=["cand-phase6-1"],
        decision=sample_loop_decision(),
        selected_candidate_id="cand-phase6-1",
        selected_action_id="runtime.right_size_context",
        selected_parameters={"context_length": 8192},
        knowledge=[sample_knowledge_evidence()],
        checkpoint=sample_safety_checkpoint(),
        experiment_id="exp-1",
        experiment_status=ExperimentDecisionKind.NO_MEANINGFUL_IMPROVEMENT,
        experiment_decision=Decision.REVERT,
        retained=False,
        restoration_status=RestorationStatus.RESTORED,
        restoration_verified=True,
        persist_status=KnowledgeWriteStatus.PERSISTED,
        baseline_metrics={"peak_rss_bytes": 1_300_000_000.0},
        candidate_metrics={"peak_rss_bytes": 1_290_000_000.0},
        outcome=LoopIterationOutcome.CANDIDATE_REVERTED,
        state_events=[sample_loop_stage_event()],
    )


def sample_loop_overhead() -> LoopOverhead:
    return LoopOverhead(
        observation_s=0.01,
        diagnosis_s=0.02,
        candidate_generation_s=0.03,
        knowledge_query_s=0.004,
        selection_s=0.001,
        checkpoint_s=0.001,
        experiment_s=5.0,
        persistence_s=0.01,
        optimizer_overhead_s=0.077,
        total_s=5.077,
    )


def sample_loop_termination() -> LoopTermination:
    return LoopTermination(
        reason=LoopTerminationReason.NO_MEANINGFUL_OPPORTUNITY,
        detail="no meaningful optimization opportunity exists",
    )


def sample_loop_context() -> LoopContext:
    return LoopContext(
        report_source="reports/baseline.json",
        hardware_hostname="sample-host",
        model_ref="sample",
        model_architecture="qwen2",
        workload_id="short-short",
        backend_name="llama.cpp",
    )


def sample_optimization_run() -> OptimizationRun:
    return OptimizationRun(
        run_id="run-sample",
        context=sample_loop_context(),
        initial_configuration=sample_runtime_configuration(),
        final_configuration=sample_runtime_configuration(),
        policy=sample_loop_policy(),
        budget=sample_optimization_budget(),
        iterations=[sample_loop_iteration()],
        state_events=[sample_loop_stage_event()],
        termination=sample_loop_termination(),
        overhead=sample_loop_overhead(),
        summary="sample run",
    )


def sample_loop_result() -> LoopResult:
    return LoopResult(
        run=sample_optimization_run(),
        termination=sample_loop_termination(),
        initial_configuration=sample_runtime_configuration(),
        final_configuration=sample_runtime_configuration(),
        iterations_performed=1,
        summary="sample run",
    )


def sample_intervention_cost_model() -> InterventionCostModel:
    return InterventionCostModel()


def sample_runtime_budget() -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=900.0,
        telemetry_interval_s=0.5,
        observations_used=12,
        diagnoses_used=1,
    )


def sample_runtime_policy() -> RuntimePolicy:
    return RuntimePolicy(budget=sample_runtime_budget())


def sample_workload_descriptor() -> WorkloadDescriptor:
    return WorkloadDescriptor(
        workload_id="short-short",
        workload_kind=WorkloadKind.SHORT,
        model_ref="sample",
        model_path="models/sample.gguf",
        backend_name="llama.cpp",
        configuration=sample_runtime_configuration(),
        context_length=32768,
        prompts_ref="short-short@1.0",
    )


def sample_workload_snapshot() -> WorkloadSnapshot:
    return WorkloadSnapshot(
        state=WorkloadState.RUNNING,
        running=True,
        generating=False,
        safe_to_pause=True,
        can_resume=True,
        requests_served=5,
    )


def sample_runtime_observation() -> RuntimeObservation:
    return RuntimeObservation(
        observation_id="rtobs-1",
        sequence=1,
        workload_state=WorkloadState.RUNNING,
        context_length=2048,
        config_key="context_length=2048",
        workload_ref="short-short",
        model_ref="sample",
        metrics={"memory_pressure": 0.6, "decode_tokens_per_sec": 22.0},
        condition="observed",
    )


def sample_metric_deviation() -> MetricDeviation:
    return MetricDeviation(
        metric="hard_page_faults_per_sec",
        direction="lower_is_better",
        baseline_mean=0.5,
        recent_mean=18000.0,
        absolute_delta=17999.5,
        relative_delta=35999.0,
        deviation_score=1.0,
        exceeds_threshold=True,
    )


def sample_observation_window() -> ObservationWindow:
    return ObservationWindow(
        capacity=120,
        total_observed=12,
        baseline_count=8,
        recent_count=3,
        anomaly=AnomalyClass.PERSISTENT,
        persistence_samples=3,
        condition_key="telemetry:hard_page_faults_per_sec:short-short",
        deviations=[sample_metric_deviation()],
    )


def sample_runtime_baseline() -> RuntimeBaseline:
    return RuntimeBaseline(
        baseline_id="rtbase-1",
        workload_id="short-short",
        model_ref="sample",
        backend_name="llama.cpp",
        configuration=sample_runtime_configuration(),
        context_length=32768,
        metrics=_sample_metric_stats(),
        observation_count=8,
    )


def sample_runtime_opportunity() -> RuntimeOpportunity:
    return RuntimeOpportunity(
        opportunity_id="rtopp-1",
        anomaly=AnomalyClass.PERSISTENT,
        condition_key="telemetry:hard_page_faults_per_sec:short-short",
        kind="hard_page_faults_per_sec",
        persistence_samples=3,
        deviation_score=0.9,
        summary="hard_page_faults_per_sec sustained for 3 sample(s)",
        evidence=["hard_page_faults_per_sec: 0.5 -> 18000 (+35999.0%)"],
        diagnosis_id="diag-1",
        diagnosis_kind=BottleneckKind.PAGING,
        confidence=0.7,
        strength=EvidenceStrength.MEDIUM,
        meaningful=True,
    )


def sample_intervention_economics() -> InterventionEconomics:
    return InterventionEconomics(
        expected_benefit_score=0.6,
        evidence_score=0.5,
        persistence_score=1.0,
        cost_score=0.52,
        net_value=0.1,
        beneficial=True,
        requires_approval=True,
        intervention_class=InterventionClass.DISRUPTIVE_ADAPTATION,
        costs={"experiment": 30.0},
        rationale=["net_value=0.1"],
    )


def sample_runtime_recommendation() -> RuntimeRecommendation:
    return RuntimeRecommendation(
        proposal_id="rtprop-1",
        opportunity_id="rtopp-1",
        condition="paging sustained",
        why_it_matters="latency may degrade",
        analysis_summary="controlled experiment",
        proposed_action_id="runtime.right_size_context",
        proposed_parameters={"context_length": 1024},
        expected_benefit="peak_rss_bytes decrease",
        expected_cost_s=33.0,
        expected_disruption_s=4.0,
        context_identity="ctx-1",
    )


def sample_runtime_notification() -> RuntimeNotification:
    return RuntimeNotification(
        notification_id="notif-1",
        proposal_id="rtprop-1",
        title="runtime recommendation",
        summary="paging sustained",
        recommendation=sample_runtime_recommendation(),
    )


def sample_approval_record() -> ApprovalRecord:
    return ApprovalRecord(
        approval_id="appr-1",
        proposal_id="rtprop-1",
        outcome=ApprovalOutcome.APPROVED,
        context_identity="ctx-1",
        decided_by="cli",
    )


def sample_runtime_checkpoint() -> RuntimeCheckpoint:
    return RuntimeCheckpoint(
        checkpoint_id="rtckpt-1",
        configuration=sample_runtime_configuration(),
        workload_id="short-short",
        model_ref="sample",
        backend_name="llama.cpp",
        context_identity="ctx-1",
        reason="runtime paging intervention",
        resume_token="resume-1",
        integrity="integrity-1",
        verified=True,
    )


def sample_runtime_state_event() -> RuntimeStateEvent:
    return RuntimeStateEvent(
        state=RuntimeState.MONITORING, observation=1, detail="monitoring active workload"
    )


def sample_runtime_intervention() -> RuntimeIntervention:
    return RuntimeIntervention(
        intervention_id="rtint-1",
        proposal_id="rtprop-1",
        approval_outcome=ApprovalOutcome.APPROVED,
        checkpoint_id="rtckpt-1",
        candidate_id="cand-1",
        action_id="runtime.right_size_context",
        retained=True,
        decision=ExperimentDecisionKind.IMPROVED,
        quality_status=QualityStatus.OK,
        restoration_status=RestorationStatus.RESTORED,
        validation_ok=True,
        resume_verified=True,
    )


def sample_runtime_cooldown() -> RuntimeCooldown:
    return RuntimeCooldown(
        cooldown_id="rtcd-1",
        key="telemetry:hard_page_faults_per_sec:short-short",
        condition_key="telemetry:hard_page_faults_per_sec:short-short",
        reason="intervention completed",
        trigger="intervention",
        expires_at=sample_diagnosis().created_at,
    )


def sample_runtime_knowledge_query() -> RuntimeKnowledgeQuery:
    return RuntimeKnowledgeQuery(
        candidate_id="cand-1",
        action_id="runtime.right_size_context",
        effect=KnowledgeSelectionEffect.NEUTRAL,
        records_considered=1,
    )


def sample_runtime_overhead() -> RuntimeOverhead:
    return RuntimeOverhead(
        detection_s=0.02,
        diagnosis_s=0.05,
        checkpoint_s=0.01,
        optimizer_overhead_s=0.08,
        total_s=0.08,
        sample_count=12,
    )


def sample_resource_impact() -> ResourceImpact:
    return ResourceImpact(
        sampling_overhead_s=0.24,
        decode_tps_mean=24.5,
        decode_tps_cv=0.03,
        throughput_within_tolerance=True,
    )


def sample_runtime_termination() -> RuntimeTermination:
    return RuntimeTermination(
        reason=RuntimeTerminationReason.DURATION_REACHED, detail="bounded run complete"
    )


def sample_runtime_monitor_run() -> RuntimeMonitorRun:
    return RuntimeMonitorRun(
        run_id="rtrun-1",
        workload=sample_workload_descriptor(),
        policy=sample_runtime_policy(),
        budget=sample_runtime_budget(),
        baseline=sample_runtime_baseline(),
        baselines=[sample_runtime_baseline()],
        states=[sample_runtime_state_event()],
        window=sample_observation_window(),
        opportunities=[sample_runtime_opportunity()],
        recommendations=[sample_runtime_recommendation()],
        notifications=[sample_runtime_notification()],
        approvals=[sample_approval_record()],
        checkpoints=[sample_runtime_checkpoint()],
        interventions=[sample_runtime_intervention()],
        cooldowns=[sample_runtime_cooldown()],
        knowledge_queries=[sample_runtime_knowledge_query()],
        observations_collected=12,
        diagnosis_events=1,
        final_state=RuntimeState.TERMINAL_SAFE,
        termination=sample_runtime_termination(),
        overhead=sample_runtime_overhead(),
        resource_impact=sample_resource_impact(),
        summary="sample runtime run",
    )


def sample_runtime_monitor_result() -> RuntimeMonitorResult:
    return RuntimeMonitorResult(
        run=sample_runtime_monitor_run(),
        termination=sample_runtime_termination(),
        final_state=RuntimeState.TERMINAL_SAFE,
        observations_collected=12,
        summary="sample runtime run",
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
        "ContextFingerprint": sample_context_fingerprint(),
        "ApplicabilityContext": sample_applicability_context(),
        "OptimizationRecord": sample_optimization_record(),
        "RollbackVerification": sample_rollback_verification(),
        "RollbackRecord": sample_rollback_record(),
        "DimensionMatch": sample_dimension_match(),
        "ApplicabilityResult": sample_applicability_result(),
        "KnowledgeQuery": sample_knowledge_query(),
        "KnowledgeWriteResult": sample_knowledge_write_result(),
        "KnowledgeEvent": sample_knowledge_event(),
        "KnowledgeHealth": sample_knowledge_health(),
        "StorageStats": sample_storage_stats(),
        "ReasoningRequest": sample_reasoning_request(),
        "ReasoningResponse": sample_reasoning_response(),
        "AuditRecord": sample_audit_record(),
        "OptimizationBudget": sample_optimization_budget(),
        "LoopPolicy": sample_loop_policy(),
        "SafetyCheckpoint": sample_safety_checkpoint(),
        "LoopStageEvent": sample_loop_stage_event(),
        "KnowledgeEvidence": sample_knowledge_evidence(),
        "LoopDecision": sample_loop_decision(),
        "LoopIteration": sample_loop_iteration(),
        "LoopOverhead": sample_loop_overhead(),
        "LoopTermination": sample_loop_termination(),
        "LoopContext": sample_loop_context(),
        "OptimizationRun": sample_optimization_run(),
        "LoopResult": sample_loop_result(),
        "InterventionCostModel": sample_intervention_cost_model(),
        "RuntimeBudget": sample_runtime_budget(),
        "RuntimePolicy": sample_runtime_policy(),
        "WorkloadDescriptor": sample_workload_descriptor(),
        "WorkloadSnapshot": sample_workload_snapshot(),
        "RuntimeObservation": sample_runtime_observation(),
        "MetricDeviation": sample_metric_deviation(),
        "ObservationWindow": sample_observation_window(),
        "RuntimeBaseline": sample_runtime_baseline(),
        "RuntimeOpportunity": sample_runtime_opportunity(),
        "InterventionEconomics": sample_intervention_economics(),
        "RuntimeRecommendation": sample_runtime_recommendation(),
        "RuntimeNotification": sample_runtime_notification(),
        "ApprovalRecord": sample_approval_record(),
        "RuntimeCheckpoint": sample_runtime_checkpoint(),
        "RuntimeStateEvent": sample_runtime_state_event(),
        "RuntimeIntervention": sample_runtime_intervention(),
        "RuntimeCooldown": sample_runtime_cooldown(),
        "RuntimeKnowledgeQuery": sample_runtime_knowledge_query(),
        "RuntimeOverhead": sample_runtime_overhead(),
        "ResourceImpact": sample_resource_impact(),
        "RuntimeTermination": sample_runtime_termination(),
        "RuntimeMonitorRun": sample_runtime_monitor_run(),
        "RuntimeMonitorResult": sample_runtime_monitor_result(),
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
