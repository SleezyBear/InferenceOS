"""Bounded runtime observation window with temporal qualification.

One anomalous telemetry sample is *not* a bottleneck. The window keeps a bounded
history, distinguishes transient / persistent / recovered / repeated anomalies with
deterministic thresholds + hysteresis, and explicitly detects workload/configuration
changes.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field

from inferenceos.contracts.enums import AnomalyClass
from inferenceos.contracts.profiling import MetricStats
from inferenceos.contracts.runtime import (
    MetricDeviation,
    ObservationWindow,
    RuntimeBaseline,
    RuntimeObservation,
)
from inferenceos.profiling.statistics import summarize

# (metric, higher_is_better)
WATCHED_METRICS: tuple[tuple[str, bool], ...] = (
    ("memory_pressure", False),
    ("hard_page_faults_per_sec", False),
    ("peak_rss_bytes", False),
    ("available_ram_bytes", True),
    ("decode_tokens_per_sec", True),
    ("total_latency_s", False),
    ("ttft_s", False),
)

# Relative deviation is only meaningful if the baseline itself is not wildly noisy.
_MAX_BASELINE_CV = 0.35

# Metrics that describe a completed workload generation/request. Their baseline
# statistics must be built from *distinct* completed generations (one sample per
# generation), not from repeated monitor ticks that observe the same completed
# generation, otherwise the baseline spread (CV) is understated.
GENERATION_METRICS: frozenset[str] = frozenset(
    {
        "decode_tokens_per_sec",
        "total_latency_s",
        "ttft_s",
        "peak_rss_bytes",
        "final_rss_bytes",
        "prompt_tokens",
        "generated_tokens",
        "prompt_tokens_per_sec",
    }
)


def generation_aware_values(
    window: list[RuntimeObservation], metric: str
) -> list[float]:
    """Return one value per distinct completed generation for a generation metric.

    Uses the observation's ``requests_served`` (the workload's existing monotonic
    completed-request counter) as the generation identity. Falls back to the raw sample
    values when no generation identity is available (e.g. an injected test sampler), so
    existing behaviour is preserved.
    """
    observations = [o for o in window if metric in o.metrics]
    per_generation: dict[int, float] = {}
    unkeyed: list[float] = []
    for observation in observations:
        if observation.requests_served is None:
            unkeyed.append(observation.metrics[metric])
        else:
            per_generation[observation.requests_served] = observation.metrics[metric]
    if per_generation:
        return list(per_generation.values())
    return unkeyed


def _improvement_relative(metric: str, higher_is_better: bool, base: float, recent: float) -> float:
    """Relative change in the *worse* direction (>=0 means degraded)."""
    if base == 0.0:
        # A lower-is-better metric going from a genuine measured zero to a nonzero
        # value is a material degradation (e.g. paging starts from a paging-free
        # baseline); a relative ratio is undefined, so report a saturated deviation.
        # A higher-is-better metric with a zero baseline cannot demonstrate a
        # degradation, so it remains no-deviation.
        if not higher_is_better and recent > 0.0:
            return 1.0
        return 0.0
    if higher_is_better:
        # A drop is degradation.
        return (base - recent) / abs(base)
    # A rise is degradation.
    return (recent - base) / abs(base)


@dataclass
class ObservationBuffer:
    """Bounded ring buffer of runtime observations + baseline/drift tracking."""

    capacity: int
    baseline_count: int
    _observations: deque[RuntimeObservation] = field(init=False)
    _baseline: RuntimeBaseline | None = field(init=False, default=None)
    _seen_conditions: set[str] = field(init=False, default_factory=set)
    _active_conditions: set[str] = field(init=False, default_factory=set)
    _recovered_pending: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self._observations = deque(maxlen=self.capacity)

    # ------------------------------------------------------------------ #
    # Mutation                                                             #
    # ------------------------------------------------------------------ #
    def append(self, observation: RuntimeObservation) -> None:
        self._observations.append(observation)

    @property
    def total(self) -> int:
        return len(self._observations)

    @property
    def observations(self) -> list[RuntimeObservation]:
        return list(self._observations)

    @property
    def baseline(self) -> RuntimeBaseline | None:
        return self._baseline

    def establish_baseline(
        self,
        *,
        workload_id: str,
        model_ref: str,
        backend_name: str,
        configuration: object,
        context_length: int | None,
        source: str = "initial",
    ) -> RuntimeBaseline | None:
        window = [o for o in self._observations if o.condition == "baseline"]
        if len(window) < self.baseline_count:
            return None
        return self._make_baseline(
            window,
            workload_id=workload_id,
            model_ref=model_ref,
            backend_name=backend_name,
            configuration=configuration,
            context_length=context_length,
            source=source,
            supersedes=self._baseline.baseline_id if self._baseline is not None else None,
        )

    def rebase(
        self,
        *,
        workload_id: str,
        model_ref: str,
        backend_name: str,
        configuration: object,
        context_length: int | None,
        source: str = "post_intervention",
    ) -> RuntimeBaseline | None:
        """Establish a new baseline from the most recent settled observations."""
        window = list(self._observations)[-max(self.baseline_count, 1) :]
        if len(window) < 2:
            return None
        return self._make_baseline(
            window,
            workload_id=workload_id,
            model_ref=model_ref,
            backend_name=backend_name,
            configuration=configuration,
            context_length=context_length,
            source=source,
            supersedes=self._baseline.baseline_id if self._baseline is not None else None,
        )

    def _make_baseline(
        self,
        window: list[RuntimeObservation],
        *,
        workload_id: str,
        model_ref: str,
        backend_name: str,
        configuration: object,
        context_length: int | None,
        source: str,
        supersedes: str | None,
    ) -> RuntimeBaseline:
        metrics: dict[str, MetricStats] = {}
        for metric, _higher in WATCHED_METRICS:
            if metric in GENERATION_METRICS:
                values = generation_aware_values(window, metric)
            else:
                values = [o.metrics[metric] for o in window if metric in o.metrics]
            if values:
                metrics[metric] = summarize(metric, values)
        from inferenceos.contracts.experiment import RuntimeConfiguration

        config = configuration if isinstance(configuration, RuntimeConfiguration) else RuntimeConfiguration()
        baseline = RuntimeBaseline(
            baseline_id=f"rtbase-{uuid.uuid4().hex[:10]}",
            workload_id=workload_id,
            model_ref=model_ref,
            backend_name=backend_name,
            configuration=config,
            context_length=context_length,
            metrics=metrics,
            observation_count=len(window),
            source=source,
            supersedes=supersedes,
        )
        self._baseline = baseline
        return baseline

    # ------------------------------------------------------------------ #
    # Qualification                                                        #
    # ------------------------------------------------------------------ #
    def deviations(self, *, threshold: float) -> list[MetricDeviation]:
        baseline = self._baseline
        if baseline is None or not self._observations:
            return []
        recent = list(self._observations)[-2:]
        deviations: list[MetricDeviation] = []
        for metric, higher in WATCHED_METRICS:
            stats = baseline.metrics.get(metric)
            if stats is None or stats.mean is None:
                continue
            recent_values = [o.metrics[metric] for o in recent if metric in o.metrics]
            if not recent_values:
                continue
            recent_mean = sum(recent_values) / len(recent_values)
            relative = _improvement_relative(metric, higher, stats.mean, recent_mean)
            exceed = relative >= threshold
            if stats.cv is not None and stats.cv > _MAX_BASELINE_CV:
                exceed = False
            deviations.append(
                MetricDeviation(
                    metric=metric,
                    direction="higher_is_better" if higher else "lower_is_better",
                    baseline_mean=stats.mean,
                    recent_mean=recent_mean,
                    absolute_delta=recent_mean - stats.mean,
                    relative_delta=relative,
                    deviation_score=max(0.0, min(abs(relative), 1.0)),
                    baseline_cv=stats.cv,
                    exceeds_threshold=exceed,
                )
            )
        return deviations

    def qualify(self, *, threshold: float, hysteresis: float, persistence: int) -> ObservationWindow:
        baseline = self._baseline
        observations = list(self._observations)
        deviations = self.deviations(threshold=threshold)

        context_changed = False
        if baseline is not None and observations:
            baseline_keys = {o.config_key for o in observations if o.condition == "baseline"}
            recent_key = observations[-1].config_key
            context_changed = bool(baseline_keys and recent_key not in baseline_keys)

        exceed_metrics = [d.metric for d in deviations if d.exceeds_threshold]
        persistence_count = self._persistence_count(exceed_metrics, threshold)
        anomaly = AnomalyClass.NONE
        rationale: list[str] = []
        condition_key: str | None = None

        if context_changed:
            condition_key = f"workload_change:{observations[-1].config_key[:48]}"
            sustained = self._config_sustained(persistence)
            anomaly = AnomalyClass.PERSISTENT if sustained else AnomalyClass.TRANSIENT
            rationale.append(
                "workload/runtime configuration changed"
                + (" and is sustained" if sustained else " (recent)")
            )
        elif exceed_metrics and persistence_count >= persistence:
            condition_key = self._condition_key(exceed_metrics, observations[-1])
            repeated = condition_key in self._seen_conditions
            anomaly = AnomalyClass.REPEATED if repeated else AnomalyClass.PERSISTENT
            self._seen_conditions.add(condition_key)
            self._active_conditions.add(condition_key)
            rationale.append(
                f"sustained deviation in {', '.join(exceed_metrics)} for "
                f"{persistence_count} sample(s)"
            )
        elif exceed_metrics:
            condition_key = self._condition_key(exceed_metrics, observations[-1])
            self._active_conditions.add(condition_key)
            anomaly = AnomalyClass.TRANSIENT
            rationale.append(
                f"deviation in {', '.join(exceed_metrics)} is not yet persistent "
                f"({persistence_count}/{persistence})"
            )
        else:
            recovered = [
                key
                for key in self._active_conditions
                if self._within_hysteresis(key, deviations, hysteresis)
            ]
            if recovered:
                for key in recovered:
                    self._active_conditions.discard(key)
                    self._recovered_pending.add(key)
                anomaly = AnomalyClass.RECOVERED
                condition_key = recovered[0]
                rationale.append(f"condition recovered within hysteresis: {recovered[0]}")

        return ObservationWindow(
            capacity=self.capacity,
            total_observed=len(observations),
            baseline_count=baseline.observation_count if baseline is not None else 0,
            recent_count=min(len(observations), 2),
            window_seconds=self._window_seconds(observations),
            anomaly=anomaly,
            persistence_samples=persistence_count,
            context_changed=context_changed,
            condition_key=condition_key,
            deviations=deviations,
            rationale=rationale,
        )

    def _condition_key(self, metrics: list[str], observation: RuntimeObservation) -> str:
        return f"telemetry:{'+'.join(sorted(metrics))}:{observation.workload_ref}:{observation.config_key[:24]}"

    def _within_hysteresis(
        self, key: str, deviations: list[MetricDeviation], hysteresis: float
    ) -> bool:
        # A recovered condition is one whose tracked metric is now well inside the band.
        if not key.startswith("telemetry:"):
            return False
        tracked = {
            metric
            for metric, _higher in WATCHED_METRICS
            if any(part == metric for part in key.split(":")[1].split("+"))
        }
        for deviation in deviations:
            if deviation.metric in tracked and deviation.relative_delta is not None:
                if abs(deviation.relative_delta) > hysteresis:
                    return False
        return True

    def _persistence_count(self, metrics: list[str], threshold: float) -> int:
        if not metrics or not self._observations:
            return 0
        count = 0
        baseline = self._baseline
        if baseline is None:
            return 0
        for observation in reversed(list(self._observations)):
            degraded = False
            for metric in metrics:
                stats = baseline.metrics.get(metric)
                if stats is None or stats.mean is None or metric not in observation.metrics:
                    continue
                higher = next((h for m, h in WATCHED_METRICS if m == metric), False)
                relative = _improvement_relative(
                    metric, higher, stats.mean, observation.metrics[metric]
                )
                if relative >= threshold:
                    degraded = True
                    break
            if not degraded:
                break
            count += 1
        return count

    def _config_sustained(self, persistence: int) -> bool:
        observations = list(self._observations)
        if len(observations) < persistence:
            return False
        recent = observations[-persistence:]
        keys = {o.config_key for o in recent}
        return len(keys) == 1

    def _window_seconds(self, observations: list[RuntimeObservation]) -> float:
        if len(observations) < 2:
            return 0.0
        return (observations[-1].captured_at - observations[0].captured_at).total_seconds()
