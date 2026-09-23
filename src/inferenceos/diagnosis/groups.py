"""Extract comparative observation groups from a Phase 3 :class:`BaselineReport`.

The detector groups runs itself (rather than relying on the report's aggregation)
so it can include memory/pressure fields the profiler does not aggregate, and so it
can compare the same model/workload under different conditions (e.g. GPU- vs
CPU-offloaded) when a ``scenario``/``condition`` tag is present.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferenceos.contracts.enums import BackendType, MeasurementMode
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import (
    BaselineReport,
    MetricStats,
    ProfilingRun,
)
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.profiling.statistics import summarize

DIAGNOSTIC_METRICS: tuple[str, ...] = (
    "model_load_time_s",
    "ttft_s",
    "prefill_latency_s",
    "decode_latency_s",
    "prompt_tokens_per_sec",
    "decode_tokens_per_sec",
    "total_latency_s",
    "process_wall_s",
    "prompt_tokens",
    "generated_tokens",
    "total_tokens",
    "peak_rss_bytes",
    "process_cpu_percent",
    "system_cpu_percent",
    "gpu_utilization_percent",
    "gpu_memory_used_bytes",
    "hard_page_faults_per_sec",
    "available_ram_bytes",
    "total_ram_bytes",
    "committed_bytes",
    "commit_limit_bytes",
)


def condition_of(run: ProfilingRun) -> str:
    extra = run.context.config.extra or {}
    return str(extra.get("scenario") or extra.get("condition") or "")


@dataclass(frozen=True)
class GroupObservation:
    model_ref: str
    workload_id: str
    mode: MeasurementMode
    condition: str
    runs: tuple[ProfilingRun, ...]
    n: int
    stats: dict[str, MetricStats]
    memory_pressure_mean: float | None
    model: ModelProfile
    workload: WorkloadProfile
    backend_type: BackendType | None
    gpu_layers_offloaded: int | None
    gpu_layers_total: int | None

    @property
    def key(self) -> str:
        suffix = f"[{self.condition}]" if self.condition else ""
        return f"{self.model_ref}/{self.workload_id}/{self.mode.value}{suffix}"

    def mean(self, metric: str) -> float | None:
        stats = self.stats.get(metric)
        return stats.mean if stats is not None else None

    def cv(self, metric: str) -> float | None:
        stats = self.stats.get(metric)
        return stats.cv if stats is not None else None

    def peak(self, metric: str) -> float | None:
        stats = self.stats.get(metric)
        return stats.max if stats is not None else None

    @property
    def offload_configured(self) -> bool:
        return bool(
            self.gpu_layers_offloaded
            and self.gpu_layers_total
            and self.gpu_layers_offloaded == self.gpu_layers_total
            and self.backend_type is BackendType.VULKAN
        )

    @property
    def workload_kind(self) -> str:
        return self.workload.kind.value


def extract_groups(report: BaselineReport) -> list[GroupObservation]:
    grouped: dict[tuple[str, str, MeasurementMode, str], list[ProfilingRun]] = {}
    for run in report.runs:
        if run.identity.warmup:
            continue
        key = (
            run.context.model.name,
            run.context.workload.workload_id,
            run.identity.mode,
            condition_of(run),
        )
        grouped.setdefault(key, []).append(run)

    groups: list[GroupObservation] = []
    for (model_ref, workload_id, mode, condition), members in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value, item[0][3])
    ):
        groups.append(_build_group(model_ref, workload_id, mode, condition, members))
    return groups


def _build_group(
    model_ref: str,
    workload_id: str,
    mode: MeasurementMode,
    condition: str,
    members: list[ProfilingRun],
) -> GroupObservation:
    stats = {
        metric: summarize(
            metric,
            [
                _metric_value(member, metric)
                for member in members
            ],
        )
        for metric in DIAGNOSTIC_METRICS
    }
    pressures = [_memory_pressure(member) for member in members]
    pressure_mean = _mean([value for value in pressures if value is not None])
    first = members[0]
    config = first.context.config
    return GroupObservation(
        model_ref=model_ref,
        workload_id=workload_id,
        mode=mode,
        condition=condition,
        runs=tuple(members),
        n=len(members),
        stats=stats,
        memory_pressure_mean=pressure_mean,
        model=first.context.model,
        workload=first.context.workload,
        backend_type=config.backend_type,
        gpu_layers_offloaded=config.gpu_layers_offloaded,
        gpu_layers_total=config.gpu_layers_total,
    )


def _metric_value(run: ProfilingRun, metric: str) -> float | None:
    value = getattr(run.observation.metrics, metric, None)
    if value is None:
        return None
    return float(value)


def _memory_pressure(run: ProfilingRun) -> float | None:
    state = run.observation.system_state
    if state is not None:
        return state.memory_pressure
    metrics = run.observation.metrics
    if metrics.committed_bytes is not None and metrics.commit_limit_bytes:
        return min(max(metrics.committed_bytes / metrics.commit_limit_bytes, 0.0), 1.0)
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
