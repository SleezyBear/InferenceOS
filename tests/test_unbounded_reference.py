"""The replay gate must not turn changed evidence into a match."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import unbounded_reference as ref
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_model


def different(a: Any, b: Any) -> bool:
    differences: list[str] = []
    ref._diff(ref._normalise(ref.summary(a)), ref._normalise(ref.summary(b)), "", differences)
    return bool(differences)


@pytest.mark.parametrize("key,a,b", [
    ("score", 0.12345678901, 0.12345678904), ("seconds", 10.0, 90.0),
    ("elapsed_s", 20.0, 200.0), ("deadline", 900.0, 1800.0),
    ("workload", {"id": "short", "prompt": "one"}, {"id": "short", "prompt": "two"}),
    ("candidates", [1, 2], [2, 1]), ("selected", [3, 8, 512], [3, 9, 512]),
    ("blocks", 4, 5),
])
def test_decision_and_numeric_changes_are_retained(key: str, a: Any, b: Any) -> None:
    assert different({key: a}, {key: b})


def test_nested_records_are_not_silently_truncated() -> None:
    a: Any = 1
    b: Any = 2
    for _ in range(30):
        a, b = {"nested": a}, {"nested": b}
    assert different(a, b)
    with pytest.raises(ref.UnsupportedRecord):
        ref.summary(object())
    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(ref.UnsupportedRecord):
        ref.summary(cycle)


def test_real_clock_exclusion_is_local_and_simulated_time_is_retained() -> None:
    def record(clock: str, deadline: float) -> dict[str, Any]:
        return {"clock": clock, "events": [{"kind": "call", "args": {"deadline": deadline}}],
                "extra": {"seconds": 12.0, "elapsed_s": 8.0}}
    assert different(record("simulated", 900.0), record("simulated", 1800.0))
    assert not different(record("real", 900.0), record("real", 1800.0))
    a, b = record("real", 900.0), record("real", 900.0)
    b["extra"]["elapsed_s"] = 9.0
    assert different(a, b)


def test_answer_timestamps_and_id_relationships_are_preserved() -> None:
    a = {"clock": "simulated", "identity": {"started_at": "2026-01-01T01:00:00Z"}}
    b = {"clock": "simulated", "identity": {"started_at": "2026-01-01T01:00:01Z"}}
    assert different(a, b)
    assert not different(["meas-1234567890", "meas-1234567890"], ["meas-abcdef1234", "meas-abcdef1234"])
    assert different(["meas-1234567890", "meas-1234567890"], ["meas-abcdef1234", "meas-abcdef1235"])
    assert different("C:/Temp/simorder-abcd/models/a.gguf", "C:/Temp/simorder-efgh/models/b.gguf")


def test_only_the_real_clock_display_eta_is_normalized() -> None:
    def record(clock: str, seconds: int, count: int = 3, kind: str = "say") -> dict[str, Any]:
        return {"clock": clock, "events": [{"kind": kind,
                "line": f"  {count} tests done; a rough guess is about {seconds}s more"}]}
    assert not different(record("real", 0), record("real", 1))
    assert different(record("simulated", 0), record("simulated", 1))
    assert different(record("real", 0), record("real", 1, count=4))
    assert different(record("real", 0, kind="result"), record("real", 1, kind="result"))


def test_actual_run_recorder_retains_changed_prompt_under_the_same_id() -> None:
    tracer = ref.Tracer()
    tracer.current = []
    harness = SimpleNamespace(run_once=lambda task, **kwargs: None)
    tuner = SimpleNamespace(harness=harness, approver=None, say=lambda line: None)
    undo = tracer._wrap_instance(tuner)
    workload = get_workload("short-short")
    try:
        harness.run_once(SimpleNamespace(workload=workload, model=sample_model(), context_length=4096))
        harness.run_once(SimpleNamespace(workload=replace(workload, prompt="changed prompt"),
                                        model=sample_model(), context_length=4096))
    finally:
        undo()
    assert len(tracer.workloads) == 2
    assert tracer.current[0]["workload"] == tracer.current[1]["workload"]
    assert tracer.current[0]["workload_ref"] != tracer.current[1]["workload_ref"]
    assert {w["prompt"] for w in tracer.workloads.values()} == {workload.prompt, "changed prompt"}


@pytest.mark.parametrize("warmup,fails", [(True, False), (False, False), (True, True)])
def test_run_request_records_context_and_model_before_execution(warmup: bool, fails: bool) -> None:
    tracer = ref.Tracer()
    tracer.current = []
    error = RuntimeError("answer failed")
    def run(task: Any, **kwargs: Any) -> None:
        if fails:
            raise error
    harness = SimpleNamespace(run_once=run)
    undo = tracer._wrap_instance(SimpleNamespace(harness=harness, approver=None, say=lambda line: None))
    model = sample_model()
    other = model.model_copy(update={"path": "models/other.gguf"})
    try:
        for context, m in [(8192, model), (16384, model), (8192, other)]:
            task = SimpleNamespace(workload=get_workload("short-short"), model=m, context_length=context)
            if fails:
                with pytest.raises(RuntimeError) as caught:
                    harness.run_once(task, warmup=warmup)
                assert caught.value is error
            else:
                harness.run_once(task, warmup=warmup)
    finally:
        undo()
    a, b, c = tracer.current
    assert a["context_length"] == 8192 and b["context_length"] == 16384
    assert a["model_ref"] == b["model_ref"] != c["model_ref"]
    assert tracer.models[c["model_ref"]]["path"] == "models/other.gguf"
    assert different(a, b) and different(a, c)


@pytest.mark.parametrize("after", [[{"scenario": "b"}], [{"scenario": "a"}, {"scenario": "b"}]])
def test_comparator_rejects_changed_scenario_identity_or_count(tmp_path: Path, after: list[dict[str, Any]]) -> None:
    before_path, after_path = tmp_path / "before.json", tmp_path / "after.json"
    before_path.write_text(json.dumps({"scenarios": [{"scenario": "a"}]}), encoding="utf-8")
    after_path.write_text(json.dumps({"scenarios": after}), encoding="utf-8")
    assert ref.compare(before_path, after_path, []) == 1
