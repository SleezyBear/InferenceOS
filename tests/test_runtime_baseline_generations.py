"""Focused tests for generation-aware workload baseline statistics.

Workload-baseline statistics must be computed from *distinct completed generations*, not
from repeated monitor ticks observing the same completed generation (which understated the
baseline spread and produced a degenerate ``decode_tps_cv``).
"""

from __future__ import annotations

from typing import Any

from inferenceos.contracts.runtime import RuntimeBudget, RuntimeObservation
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.approval import AutoRejectApprovalProvider
from inferenceos.runtime.notification import CollectingNotificationSink
from inferenceos.runtime.telemetry import RuntimeTelemetrySampler
from inferenceos.runtime.verification import ScriptedWorkload
from inferenceos.runtime.window import ObservationBuffer, generation_aware_values
from inferenceos.selftest import sample_hardware
from tests.runtime_helpers import make_runtime_policy


def _obs(decode: float, *, generation: int | None) -> RuntimeObservation:
    return RuntimeObservation(
        observation_id=f"obs-{generation}-{decode}",
        metrics={
            "decode_tokens_per_sec": decode,
            "total_latency_s": 1.0,
            "peak_rss_bytes": 1_000.0,
            "memory_pressure": 0.5,
            "available_ram_bytes": 8.0,
        },
        requests_served=generation,
        condition="baseline",
    )


def test_repeated_observations_of_one_generation_are_one_sample() -> None:
    window = [_obs(3.4, generation=1) for _ in range(5)]
    values = generation_aware_values(window, "decode_tokens_per_sec")
    assert values == [3.4]


def test_two_generations_produce_two_samples() -> None:
    window = [_obs(3.4, generation=1) for _ in range(5)] + [
        _obs(3.6, generation=2) for _ in range(3)
    ]
    values = generation_aware_values(window, "decode_tokens_per_sec")
    assert values == [3.4, 3.6]


def test_baseline_mean_and_cv_use_distinct_generations() -> None:
    buffer = ObservationBuffer(capacity=40, baseline_count=4)
    for _ in range(5):
        buffer.append(_obs(3.4, generation=1))
    for _ in range(3):
        buffer.append(_obs(3.6, generation=2))
    baseline = buffer.establish_baseline(
        workload_id="w",
        model_ref="m",
        backend_name="llama.cpp",
        configuration=None,
        context_length=2048,
    )
    assert baseline is not None
    stats = baseline.metrics["decode_tokens_per_sec"]
    assert stats.count == 2
    assert stats.mean is not None and abs(stats.mean - 3.5) < 1e-9
    assert stats.cv is not None and stats.cv > 0.0


def test_no_generation_identity_falls_back_to_raw_samples() -> None:
    window = [_obs(3.4, generation=None) for _ in range(4)]
    values = generation_aware_values(window, "decode_tokens_per_sec")
    assert values == [3.4, 3.4, 3.4, 3.4]


class _GenerationSampler(RuntimeTelemetrySampler):
    """Emits a new completed generation every 4 samples; metrics repeat between them."""

    def __init__(self) -> None:
        self.count = 0

    def sample(
        self, workload: Any, *, sequence: int, condition: str
    ) -> RuntimeObservation:
        self.count += 1
        generation = self.count // 4
        metrics: dict[str, float] = {
            "memory_pressure": 0.5,
            "hard_page_faults_per_sec": 0.5,
            "available_ram_bytes": float(8 * 1024**3),
            "total_ram_bytes": float(16 * 1024**3),
            "system_cpu_percent": 20.0,
        }
        if generation >= 1:
            metrics.update(
                {
                    "decode_tokens_per_sec": 3.4 + 0.1 * generation,
                    "total_latency_s": 1.0 + 0.05 * generation,
                    "ttft_s": 0.2,
                    "peak_rss_bytes": 1_300_000_000.0,
                }
            )
        return RuntimeObservation(
            observation_id=f"gen-{self.count}",
            metrics=metrics,
            requests_served=generation,
            condition=condition,
        )


def test_monitor_baseline_uses_distinct_generations() -> None:
    class _Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            return self.t

        def sleep(self, seconds: float) -> None:
            self.t += max(seconds, 0.0)

    clock = _Clock()
    policy = make_runtime_policy(
        budget=RuntimeBudget(
            max_duration_s=20.0,
            telemetry_interval_s=0.5,
            observation_window_size=40,
            baseline_samples=4,
            max_optimizer_overhead_s=120.0,
        ),
        baseline_workload_samples=2,
        baseline_max_wait_s=15.0,
    )
    monitor = RuntimeMonitor(
        workload=ScriptedWorkload(),
        sampler=_GenerationSampler(),
        policy=policy,
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
        clock=clock,
        sleep=clock.sleep,
    )
    result = monitor.run()
    run = result.run
    assert run.baseline is not None
    stats = run.baseline.metrics["decode_tokens_per_sec"]
    assert stats.count == 2  # distinct completed generations
    assert stats.mean is not None and abs(stats.mean - 3.55) < 1e-9
    assert stats.cv is not None and stats.cv > 0.0
    # The derived resource-impact CV is no longer degenerate.
    assert run.resource_impact.decode_tps_cv is not None
    assert run.resource_impact.decode_tps_cv > 0.0
