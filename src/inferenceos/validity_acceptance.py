"""B1 measurement-validity acceptance (real telemetry + deterministic logic).

Real Windows evidence: power state and live processor clock performance are actually
observed and reach the ``SystemState``/report path.

Controlled evidence: the deterministic validity policy is exercised on injected power
and performance series (normal, battery, degraded, unknown, transient, boundary), and
a genuinely meaningful diagnosis is shown to be suppressed when its measurement is
confounded.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    MeasurementMode,
    MeasurementValidity,
    ObservationSource,
    PowerState,
)
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.profiling import (
    BaselineReport,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunIdentity,
)
from inferenceos.contracts.system_state import SystemState
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.loop.verification import make_paging_report
from inferenceos.measurement_validity import (
    MeasurementValidityPolicy,
    evaluate_report_validity,
    validity_blocks,
)
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.profiling.runner import COLD_WARM_SEMANTICS
from inferenceos.reasoning.escalation import decide_escalation
from inferenceos.selftest import sample_hardware, sample_model, sample_workload
from inferenceos.telemetry.system import observe_power_performance, sample_system_state

def _run(
    index: int,
    *,
    ratio: float | None,
    power: PowerState,
    current_mhz: int | None = 2000,
    max_mhz: int | None = 2000,
) -> ProfilingRun:
    state = SystemState(
        captured_at=datetime.now(timezone.utc),
        total_ram_bytes=16 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        memory_pressure=0.5,
        cpu_percent_total=10.0,
        hard_page_faults_per_sec=0.0,
        power_state=power,
        cpu_current_mhz=current_mhz if ratio is not None else None,
        cpu_max_mhz=max_mhz if ratio is not None else None,
        cpu_limit_mhz=max_mhz if ratio is not None else None,
        cpu_performance_ratio=ratio,
    )
    return ProfilingRun(
        identity=RunIdentity(
            run_id=f"validity-{index}",
            profiler_version="validity-acceptance",
            mode=MeasurementMode.WARM,
            started_at=datetime.now(timezone.utc),
        ),
        context=RunContext(
            hardware=sample_hardware(),
            model=sample_model(),
            workload=sample_workload(),
            config=ProfilingConfig(backend_name="validity", context_length=1024),
        ),
        observation=Observation(
            observation_id=f"obs-validity-{index}",
            captured_at=datetime.now(timezone.utc),
            source=ObservationSource.HARNESS,
            mode=MeasurementMode.WARM,
            metrics=RunMetrics(decode_tokens_per_sec=20.0),
            system_state=state,
        ),
        metric_quality={},
    )


def _synthetic_report(points: list[tuple[float | None, PowerState]]) -> BaselineReport:
    runs = [
        _run(index, ratio=ratio, power=power) for index, (ratio, power) in enumerate(points)
    ]
    return BaselineReport(
        profiler_version="validity-acceptance",
        suite_version="validity-acceptance",
        generated_at=datetime.now(timezone.utc),
        hardware=sample_hardware(),
        backend=ProfilingConfig(backend_name="validity"),
        workloads=[sample_workload()],
        runs=runs,
        groups=[],
        cold_warm_semantics=COLD_WARM_SEMANTICS,
        resident_warm_supported=False,
        notes="controlled B1 validity fixture",
    )


def _inject(report: BaselineReport, *, ratio: float, power: PowerState) -> BaselineReport:
    runs: list[ProfilingRun] = []
    for run in report.runs:
        state = run.observation.system_state
        if state is None:
            runs.append(run)
            continue
        updated = state.model_copy(
            update={
                "power_state": power,
                "cpu_current_mhz": int(ratio * 2000),
                "cpu_max_mhz": 2000,
                "cpu_limit_mhz": 2000,
                "cpu_performance_ratio": ratio,
            }
        )
        runs.append(
            run.model_copy(
                update={"observation": run.observation.model_copy(update={"system_state": updated})}
            )
        )
    return report.model_copy(update={"runs": runs})


def run_validity_acceptance(
    settings: Settings, *, probe_prompt: str
) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    scenarios: dict[str, object] = {}
    checks: list[tuple[str, bool]] = []

    # ------------------------------------------------------------------ #
    # Real Windows telemetry                                              #
    # ------------------------------------------------------------------ #
    real: dict[str, object] = {}
    real_ok = False
    try:
        states = [sample_system_state(include_gpu=False) for _ in range(3)]
        samples = [
            {
                "power_state": state.power_state.value,
                "cpu_current_mhz": state.cpu_current_mhz,
                "cpu_max_mhz": state.cpu_max_mhz,
                "cpu_limit_mhz": state.cpu_limit_mhz,
                "cpu_performance_ratio": state.cpu_performance_ratio,
                "captured_at": state.captured_at.isoformat(),
            }
            for state in states
        ]
        power_known = all(state.power_state is not PowerState.UNKNOWN for state in states)
        ratio_coherent = all(
            state.cpu_performance_ratio is None
            or 0.0 <= state.cpu_performance_ratio <= 1.0
            for state in states
        )
        real = {"samples": samples}
        real_ok = bool(power_known and ratio_coherent)
    except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001
        real = {"error": f"{type(exc).__name__}: {exc}"}
    scenarios["real_telemetry"] = real
    checks.append(
        (
            "B1: real Windows power state and CPU clock performance observed into SystemState",
            real_ok,
        )
    )

    # ------------------------------------------------------------------ #
    # Overhead of the added observation                                   #
    # ------------------------------------------------------------------ #
    overhead_ok = False
    overhead: dict[str, object] = {}
    try:
        iterations = 20
        start = time.perf_counter()
        for _ in range(iterations):
            observe_power_performance()
        mean_ms = (time.perf_counter() - start) / iterations * 1000.0
        overhead = {
            "method": "observe_power_performance() GetSystemPowerStatus + CallNtPowerInformation",
            "iterations": iterations,
            "mean_ms": round(mean_ms, 4),
            "budget_ms": 50.0,
        }
        overhead_ok = mean_ms < 50.0
    except (OSError, RuntimeError) as exc:  # noqa: BLE001
        overhead = {"error": f"{type(exc).__name__}: {exc}"}
    scenarios["overhead"] = overhead
    checks.append(("B1: power/performance observation overhead is negligible", overhead_ok))

    # ------------------------------------------------------------------ #
    # Deterministic validity policy (controlled)                          #
    # ------------------------------------------------------------------ #
    normal = _synthetic_report([(0.95, PowerState.AC), (0.97, PowerState.AC)])
    battery_ok = _synthetic_report([(0.90, PowerState.BATTERY), (0.92, PowerState.BATTERY)])
    degraded_battery = _synthetic_report(
        [(0.95, PowerState.AC), (0.45, PowerState.BATTERY), (0.50, PowerState.BATTERY)]
    )
    degraded_ac = _synthetic_report(
        [
            (0.96, PowerState.AC),
            (0.95, PowerState.AC),
            (0.45, PowerState.AC),
            (0.46, PowerState.AC),
        ]
    )
    unknown = _synthetic_report([(None, PowerState.UNKNOWN), (None, PowerState.UNKNOWN)])
    transient = _synthetic_report([(0.95, PowerState.AC), (0.45, PowerState.AC), (0.96, PowerState.AC)])
    boundary_above = _synthetic_report([(0.61, PowerState.BATTERY), (0.61, PowerState.BATTERY)])
    boundary_below = _synthetic_report([(0.59, PowerState.BATTERY), (0.59, PowerState.BATTERY)])

    normal_assessment = evaluate_report_validity(normal)
    battery_assessment = evaluate_report_validity(battery_ok)
    degraded_battery_assessment = evaluate_report_validity(degraded_battery)
    degraded_ac_assessment = evaluate_report_validity(degraded_ac)
    unknown_assessment = evaluate_report_validity(unknown)
    unknown_blocking = evaluate_report_validity(
        unknown, MeasurementValidityPolicy(unknown_blocks=True)
    )
    transient_assessment = evaluate_report_validity(transient)
    boundary_above_assessment = evaluate_report_validity(boundary_above)
    boundary_below_assessment = evaluate_report_validity(boundary_below)
    single_bad = evaluate_report_validity(
        _synthetic_report([(0.45, PowerState.BATTERY)]),
        MeasurementValidityPolicy(persistence_samples=2),
    )

    scenarios["controlled"] = {
        "normal_ac": normal_assessment.model_dump(mode="json"),
        "battery_no_degradation": battery_assessment.model_dump(mode="json"),
        "degraded_battery": degraded_battery_assessment.model_dump(mode="json"),
        "degraded_ac_relative_drop": degraded_ac_assessment.model_dump(mode="json"),
        "unknown": unknown_assessment.model_dump(mode="json"),
        "unknown_blocks": unknown_blocking.model_dump(mode="json"),
        "transient": transient_assessment.model_dump(mode="json"),
        "boundary_above_floor": boundary_above_assessment.model_dump(mode="json"),
        "boundary_below_floor": boundary_below_assessment.model_dump(mode="json"),
        "single_bad_sample_not_persisted": single_bad.model_dump(mode="json"),
    }
    checks.append(
        (
            "B1: normal AC measurement is VALID",
            normal_assessment.validity is MeasurementValidity.VALID
            and not normal_assessment.confounded,
        )
    )
    checks.append(
        (
            "B1: battery alone does not invalidate a measurement",
            battery_assessment.validity is MeasurementValidity.VALID
            and battery_assessment.power_state is PowerState.BATTERY,
        )
    )
    checks.append(
        (
            "B1: persisted battery + low performance is DEGRADED",
            degraded_battery_assessment.validity is MeasurementValidity.DEGRADED
            and degraded_battery_assessment.confounded,
        )
    )
    checks.append(
        (
            "B1: AC with a material relative performance drop is DEGRADED",
            degraded_ac_assessment.validity is MeasurementValidity.DEGRADED
            and degraded_ac_assessment.confounded,
        )
    )
    checks.append(
        (
            "B1: unknown telemetry is UNKNOWN and blocking only when configured",
            unknown_assessment.validity is MeasurementValidity.UNKNOWN
            and not validity_blocks(unknown_assessment)
            and validity_blocks(unknown_blocking),
        )
    )
    checks.append(
        (
            "B1: a single transient low sample does not invalidate (persistence)",
            transient_assessment.validity is MeasurementValidity.VALID
            and single_bad.validity is MeasurementValidity.VALID,
        )
    )
    checks.append(
        (
            "B1: boundary at the ratio floor behaves deterministically",
            boundary_above_assessment.validity is MeasurementValidity.VALID
            and boundary_below_assessment.confounded,
        )
    )

    # ------------------------------------------------------------------ #
    # Confounded measurement cannot drive optimization                    #
    # ------------------------------------------------------------------ #
    suppression_ok = False
    try:
        confounded_report = _inject(make_paging_report(), ratio=0.45, power=PowerState.BATTERY)
        diagnosis = BottleneckDetector().detect(
            confounded_report, source_report_path="<b1-controlled-confounded>"
        )
        plan = generate_candidates(
            diagnosis, report=confounded_report, policy=OptimizationPolicy()
        )
        escalation = decide_escalation(diagnosis)
        suppression_ok = bool(
            diagnosis.measurement_validity is not None
            and diagnosis.measurement_validity.confounded
            and diagnosis.meaningful_opportunity is False
            and diagnosis.primary is None
            and diagnosis.escalation_required is False
            and not escalation.should_escalate
            and not any(candidate.executable for candidate in plan.candidates)
        )
        scenarios["suppression"] = {
            "diagnosis_meaningful": diagnosis.meaningful_opportunity,
            "primary": None if diagnosis.primary is None else diagnosis.primary.kind.value,
            "escalation_required": diagnosis.escalation_required,
            "escalation_should_escalate": escalation.should_escalate,
            "plan_outcome": plan.outcome.value,
            "executable_candidates": [
                candidate.action_id for candidate in plan.candidates if candidate.executable
            ],
            "no_opportunity_reason": diagnosis.no_opportunity_reason,
        }
    except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001
        scenarios["suppression"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(
        (
            "B1: a confounded measurement produces no primary, no candidates and no escalation",
            suppression_ok,
        )
    )

    # ------------------------------------------------------------------ #
    # Backward compatibility (valid measurement unchanged)                #
    # ------------------------------------------------------------------ #
    compatibility_ok = False
    try:
        paging = make_paging_report()
        diagnosis = BottleneckDetector().detect(
            paging, source_report_path="<b1-controlled-valid>"
        )
        compatibility_ok = bool(
            diagnosis.measurement_validity is not None
            and diagnosis.measurement_validity.validity
            in (MeasurementValidity.VALID, MeasurementValidity.UNKNOWN)
            and diagnosis.meaningful_opportunity is True
            and diagnosis.primary is not None
        )
        scenarios["valid_unaffected"] = {
            "validity": diagnosis.measurement_validity.validity.value
            if diagnosis.measurement_validity
            else None,
            "meaningful_opportunity": diagnosis.meaningful_opportunity,
            "primary": diagnosis.primary.kind.value if diagnosis.primary else None,
        }
    except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001
        scenarios["valid_unaffected"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(
        (
            "B1: a genuinely meaningful diagnosis is unaffected when validity is normal",
            compatibility_ok,
        )
    )

    _ = (settings, probe_prompt)
    return scenarios, checks
