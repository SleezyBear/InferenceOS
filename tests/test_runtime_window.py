"""Phase 10 observation-window / temporal-qualification tests."""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.contracts.enums import AnomalyClass
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.runtime import RuntimeObservation
from inferenceos.runtime.window import ObservationBuffer

_HEALTHY = {
    "memory_pressure": 0.5,
    "hard_page_faults_per_sec": 0.5,
    "available_ram_bytes": float(8 * 1024**3),
    "decode_tokens_per_sec": 24.0,
    "total_latency_s": 1.4,
    "peak_rss_bytes": 1_300_000_000.0,
}
_DEGRADED = {
    "memory_pressure": 0.85,
    "hard_page_faults_per_sec": 18000.0,
    "available_ram_bytes": float(4 * 1024**3),
    "decode_tokens_per_sec": 9.0,
    "total_latency_s": 3.0,
    "peak_rss_bytes": 1_700_000_000.0,
}


def _obs(sequence: int, metrics: dict[str, float], *, condition: str, config_key: str = "c1") -> RuntimeObservation:
    return RuntimeObservation(
        observation_id=f"rtobs-{sequence}",
        sequence=sequence,
        captured_at=datetime.now(timezone.utc),
        metrics=dict(metrics),
        condition=condition,
        config_key=config_key,
        workload_ref="short-short",
        model_ref="sample",
    )


def _buffer_with_baseline(metrics_list: list[dict[str, float]] | None = None) -> ObservationBuffer:
    buffer = ObservationBuffer(capacity=30, baseline_count=4)
    for index, metrics in enumerate(metrics_list or [_HEALTHY] * 4):
        buffer.append(_obs(index, metrics, condition="baseline"))
    buffer.establish_baseline(
        workload_id="short-short",
        model_ref="sample",
        backend_name="llama.cpp",
        configuration=RuntimeConfiguration(context_length=2048),
        context_length=2048,
    )
    return buffer


def test_transient_anomaly_is_not_persistent() -> None:
    buffer = _buffer_with_baseline()
    buffer.append(_obs(10, _DEGRADED, condition="observed"))
    buffer.append(_obs(11, _HEALTHY, condition="observed"))
    window = buffer.qualify(threshold=0.25, hysteresis=0.125, persistence=3)
    assert window.anomaly is AnomalyClass.TRANSIENT


def test_persistent_anomaly_detected_after_persistence() -> None:
    buffer = _buffer_with_baseline()
    for index in range(3):
        buffer.append(_obs(10 + index, _DEGRADED, condition="observed"))
    window = buffer.qualify(threshold=0.25, hysteresis=0.125, persistence=3)
    assert window.anomaly is AnomalyClass.PERSISTENT
    assert window.persistence_samples >= 3
    assert window.condition_key is not None
    assert any(d.exceeds_threshold for d in window.deviations)


def test_recovered_anomaly_after_hysteresis() -> None:
    buffer = _buffer_with_baseline()
    for index in range(3):
        buffer.append(_obs(10 + index, _DEGRADED, condition="observed"))
    assert buffer.qualify(threshold=0.25, hysteresis=0.125, persistence=3).anomaly is AnomalyClass.PERSISTENT
    # Return to well within the hysteresis band.
    buffer.append(_obs(20, _HEALTHY, condition="observed"))
    buffer.append(_obs(21, _HEALTHY, condition="observed"))
    recovered = buffer.qualify(threshold=0.25, hysteresis=0.125, persistence=3)
    assert recovered.anomaly is AnomalyClass.RECOVERED


def test_repeated_anomaly_after_recovery() -> None:
    buffer = _buffer_with_baseline()
    for index in range(3):
        buffer.append(_obs(10 + index, _DEGRADED, condition="observed"))
    buffer.qualify(threshold=0.25, hysteresis=0.125, persistence=3)
    buffer.append(_obs(20, _HEALTHY, condition="observed"))
    buffer.append(_obs(21, _HEALTHY, condition="observed"))
    buffer.qualify(threshold=0.25, hysteresis=0.125, persistence=3)
    for index in range(3):
        buffer.append(_obs(30 + index, _DEGRADED, condition="observed"))
    repeated = buffer.qualify(threshold=0.25, hysteresis=0.125, persistence=3)
    assert repeated.anomaly is AnomalyClass.REPEATED


def test_baseline_drift_records_history() -> None:
    buffer = _buffer_with_baseline()
    for index in range(3):
        buffer.append(_obs(10 + index, _DEGRADED, condition="observed"))
    baseline = buffer.rebase(
        workload_id="short-short",
        model_ref="sample",
        backend_name="llama.cpp",
        configuration=RuntimeConfiguration(context_length=1024),
        context_length=1024,
        source="post_intervention",
    )
    assert baseline is not None
    assert baseline.source == "post_intervention"
    assert baseline.supersedes is not None
    # The new baseline reflects the degraded/post-change window, so nothing deviates.
    window = buffer.qualify(threshold=0.25, hysteresis=0.125, persistence=3)
    assert window.anomaly is AnomalyClass.NONE


def test_config_change_is_detected() -> None:
    buffer = _buffer_with_baseline()
    for index in range(3):
        buffer.append(_obs(10 + index, _HEALTHY, condition="observed", config_key="c2"))
    window = buffer.qualify(threshold=0.25, hysteresis=0.125, persistence=3)
    assert window.context_changed is True
    assert window.anomaly is AnomalyClass.PERSISTENT
