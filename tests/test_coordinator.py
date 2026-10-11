"""The shared search coordinator and Unbounded's policy (C3-6, 11 Oct 2026).

Three groups:

* the coordinator's contract: it drives a policy, records every outcome, calls the
  policy's transition after every completed action, records nested work, and lets an
  exception leave unchanged after recording the action as interrupted;
* Unbounded's option-path loop through ``UnboundedPolicy``, branch by branch, with a
  scripted tuner (already tried, revisit limits, the layer path off, a memory-shaping
  change with and without a better layer count, the engine-switch check before its
  path, no change, the time limit before dispatch, a user stop, an error), and whole
  tunes through the real ``Tuner`` and the ``tune`` command's own cleanup and report;
* Practical's representation (C3-7 is not wired): a planner step keeps its captured
  reference, and categories map to canonical path ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.tune import ProposedChange
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from inferenceos.tune import coordinator as co
from inferenceos.tune.unbounded_policy import UnboundedPolicy
from tests.experiment_helpers import SampleResult, ScriptedHarness
from tests.test_tune import CONTEXT, _settings, _tuner

START = RuntimeConfiguration(context_length=CONTEXT, gpu_offload_layers=8)


# --------------------------------------------------------------------------- #
# The coordinator's contract                                                   #
# --------------------------------------------------------------------------- #
class ScriptedPolicy:
    mode = co.UNBOUNDED

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.log: list[str] = []

    def next_action(self, trace: co.SearchTrace) -> co.Action | co.Finished:
        self.log.append("next")
        if not self.outcomes:
            return co.Finished("nothing left")
        return trace.issue(kind=co.PATH_ROUND, path="threads", label="threads", reason="test",
                           decided_by="script", carried=START)

    def execute(self, action: co.Action, trace: co.SearchTrace) -> co.Outcome:
        self.log.append(f"execute {action.id}")
        item = self.outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[no-any-return]

    def advance(self, action: co.Action, outcome: co.Outcome, trace: co.SearchTrace) -> None:
        self.log.append(f"advance {action.id} {outcome.status}")


def test_every_completed_outcome_is_recorded_and_advanced() -> None:
    policy = ScriptedPolicy([co.Outcome(co.ALREADY_TRIED, "same setup"), co.Outcome(co.NO_CHANGE, "flat"),
                             co.Outcome(co.CHANGED, "5 threads")])
    trace = co.coordinate(policy, co.SearchTrace(co.UNBOUNDED))
    assert policy.log == ["next", "execute 1", "advance 1 already tried", "next", "execute 2", "advance 2 no change",
                          "next", "execute 3", "advance 3 changed", "next"]
    assert [e["status"] for e in trace.entries] == ["already tried", "no change", "changed"]
    assert trace.finished == "nothing left"
    assert trace.round()["round"] == co.TRACE_ROUND


@pytest.mark.parametrize("error", [RuntimeError("engine failed"), KeyboardInterrupt()])
def test_an_exception_is_recorded_once_and_leaves_unchanged(error: BaseException) -> None:
    policy = ScriptedPolicy([co.Outcome(co.NO_CHANGE), error, co.Outcome(co.CHANGED)])
    trace = co.SearchTrace(co.UNBOUNDED)
    with pytest.raises(type(error)) as caught:
        co.coordinate(policy, trace)
    assert caught.value is error
    assert policy.log == ["next", "execute 1", "advance 1 no change", "next", "execute 2"]  # no retry, no advance
    assert trace.entries[-1]["status"] == co.INTERRUPTED and trace.entries[-1]["why"].startswith(type(error).__name__)
    assert trace.finished is None


def test_nested_work_is_a_child_and_an_unknown_status_is_refused() -> None:
    trace = co.SearchTrace(co.UNBOUNDED)
    parent = trace.issue(kind=co.PATH_ROUND, path="kv_cache", label="cache", reason="r", decided_by="d")
    trace.run_child(parent, kind=co.LAYER_REVISIT, path="layers", label="layers again", reason="memory",
                    decided_by="MEMORY_PATHS", carried=START, work=lambda: co.Outcome(co.NO_CHANGE, "flat"))
    with pytest.raises(ValueError, match="boom"):
        trace.run_child(parent, kind=co.DIAGNOSIS, path="diagnosis", label="d", reason="r", decided_by="d",
                        carried=START, work=lambda: (_ for _ in ()).throw(ValueError("boom")))
    children = [e for e in trace.entries if e["parent"] == parent.id]
    assert [(e["kind"], e["status"]) for e in children] == [("layer revisit", "no change"), ("diagnosis", "interrupted")]
    with pytest.raises(ValueError, match="unknown outcome status"):
        co.Outcome("kept for ever")


def test_configurations_are_detached_snapshots() -> None:
    trace = co.SearchTrace(co.UNBOUNDED)
    action = trace.issue(kind=co.PATH_ROUND, path="threads", label="t", reason="r", decided_by="d", carried=START,
                         reference=START, candidates=[START.model_copy(update={"threads": 5})])
    trace.entries[0]["carried"]["threads"] = 9  # editing the report copy ...
    assert action.carried is not None and "threads" not in action.carried  # ... never reaches the action
    with pytest.raises(TypeError):
        action.carried["threads"] = 3  # type: ignore[index]
    assert action.candidates[0]["threads"] == 5 and trace.entries[0]["candidates"] == [dict(action.candidates[0])]


# --------------------------------------------------------------------------- #
# Unbounded's policy, branch by branch (scripted tuner)                        #
# --------------------------------------------------------------------------- #
def _change(setting: str, old: Any, new: Any, n: int) -> ProposedChange:
    return ProposedChange(change_id=f"c{n}", card_id="card", setting=setting, description=setting,
                          old_value=old, new_value=new)


class ScriptedTuner:
    """The tuner steps UnboundedPolicy calls, scripted; ``calls`` keeps their order."""

    def __init__(self, paths: str, *, revisits: int = 2, layer_path: bool = True,
                 path_round: Callable[[Any, RuntimeConfiguration, int], Any] | None = None,
                 layer_round: Callable[[RuntimeConfiguration], Any] | None = None) -> None:
        self.settings = SimpleNamespace(tune_paths=paths, tune_path_revisits=revisits, tune_layer_path=layer_path,
                                        tune_diagnosis=False)
        self.lines: list[str] = []
        self.calls: list[tuple[Any, ...]] = []
        self._long_question_record: dict[str, object] | None = None
        self._change_workloads: dict[str, list[Any]] = {}
        self.workload = get_workload("short-short")
        self._path_round_script = path_round or (lambda path, current, n: ({"round": path.label, "path": path.name,
                                                                            "applies": True, "reason": "flat"}, None, None))
        self._layer_script = layer_round
        self.n = 0

    def say(self, line: str) -> None:
        self.lines.append(line)

    def _path_names(self) -> list[str]:
        return str(self.settings.tune_paths).split(",")

    def _facts_round(self, current: RuntimeConfiguration) -> dict[str, object]:
        self.calls.append(("facts",))
        return {"round": "facts"}

    def _check_switches(self, current: RuntimeConfiguration, depth: str, deadline: Any) -> dict[str, object]:
        self.calls.append(("switch check", depth))
        return {"round": "engine switch checks", "reason": "none passed"}

    def _path_round(self, path: Any, current: RuntimeConfiguration, watch: Any, deadline: Any, *, depth: str) -> Any:
        self.n += 1
        self.calls.append(("path", path.name, current.model_dump(exclude_none=True)))
        return self._path_round_script(path, current, self.n)

    def _layer_round(self, current: RuntimeConfiguration, start: Any, watch: Any, deadline: Any, *,
                     revisit: bool) -> Any:
        self.calls.append(("layers", revisit, current.model_dump(exclude_none=True), deadline))
        assert self._layer_script is not None
        return self._layer_script(current)


def _policy(tuner: ScriptedTuner, *, deadline: float | None = None) -> tuple[UnboundedPolicy, list[dict[str, object]], list[str]]:
    rounds: list[dict[str, object]] = []
    not_reached: list[str] = []
    policy = UnboundedPolicy(tuner, current=START, changes=[], rounds=rounds, not_reached=not_reached,  # type: ignore[arg-type]
                             starting=START, start_measurement=None, watch=None,  # type: ignore[arg-type]
                             safety_ids=set(), path_deadline=deadline)
    return policy, rounds, not_reached


def _drive(policy: UnboundedPolicy) -> co.SearchTrace:
    trace = co.SearchTrace(policy.mode)
    policy.start(trace)
    return co.coordinate(policy, trace)


def _kept(setting: str, value: Any) -> Callable[[Any, RuntimeConfiguration, int], Any]:
    """A path round that keeps ``setting = value`` the first time its path runs."""

    def script(path: Any, current: RuntimeConfiguration, n: int) -> Any:
        record: dict[str, object] = {"round": path.label, "path": path.name, "applies": True}
        if path.setting == setting and getattr(current, setting) != value:
            record["selected"] = value
            return record, _change(setting, getattr(current, setting), value, n), current.model_copy(update={setting: value})
        record["reason"] = "nothing better"
        return record, None, None

    return script


def test_a_kept_change_brings_back_affected_finished_paths_once_each() -> None:
    tuner = ScriptedTuner("threads,prompt_batch", path_round=_kept("ubatch_size", 1024))
    policy, rounds, _ = _policy(tuner)
    trace = _drive(policy)
    assert [c[1] for c in tuner.calls if c[0] == "path"] == ["threads", "prompt_batch", "threads"]
    tops = [e for e in trace.entries if e["parent"] is None]
    assert [(e["path"], e["status"]) for e in tops] == [("threads", "no change"), ("prompt_batch", "changed"),
                                                      ("threads", "no change")]
    assert tops[2]["reason"] == "revisit after a kept change on prompt_batch (affected_paths)"
    assert tops[2]["carried"]["ubatch_size"] == 1024 and tops[1]["carried_after"]["ubatch_size"] == 1024
    assert "  will look again at processor threads: the change can move what limits it" in tuner.lines
    assert policy.current.ubatch_size == 1024 and [c.setting for c in policy.changes] == ["ubatch_size"]
    assert [r["round"] for r in rounds][0] == "facts"


def test_exhausted_revisits_queue_nothing_more() -> None:
    tuner = ScriptedTuner("threads,prompt_batch", revisits=0, path_round=_kept("ubatch_size", 1024))
    policy, _, _ = _policy(tuner)
    trace = _drive(policy)
    assert [c[1] for c in tuner.calls if c[0] == "path"] == ["threads", "prompt_batch"]
    assert not any("will look again" in line for line in tuner.lines)
    assert trace.finished == "every chosen path and revisit was handled"


def test_a_path_already_tried_on_this_exact_setup_is_skipped() -> None:
    tuner = ScriptedTuner("threads,prompt_batch")
    policy, _, _ = _policy(tuner)
    trace = co.SearchTrace(policy.mode)
    policy.start(trace)
    threads = policy.by_name["threads"]
    policy.remaining.append(threads)  # queued again with nothing changed in between
    co.coordinate(policy, trace)
    assert [c[1] for c in tuner.calls if c[0] == "path"] == ["threads", "prompt_batch"]
    assert [e["status"] for e in trace.entries] == ["no change", "no change", "already tried"]
    assert "Round 4 (processor threads (again, after a kept change)) skipped: already tried on this exact setup." in tuner.lines


def test_already_tried_is_decided_before_the_time_limit() -> None:
    """Both apply: the old loop said "already tried" and did not count the path as not
    reached (the order of the two checks is kept)."""
    tuner = ScriptedTuner("threads")
    policy, _, not_reached = _policy(tuner)
    trace = co.SearchTrace(policy.mode)
    policy.start(trace)
    co.coordinate(policy, trace)
    policy.remaining.append(policy.by_name["threads"])
    policy.path_deadline = -1.0  # the time is up as well
    co.coordinate(policy, trace)
    assert [e["status"] for e in trace.entries] == ["no change", "already tried"]
    assert not_reached == [] and tuner.lines[-1].endswith("skipped: already tried on this exact setup.")


def test_the_time_limit_stops_a_path_before_it_is_dispatched() -> None:
    tuner = ScriptedTuner("threads,prompt_batch")
    policy, _, not_reached = _policy(tuner, deadline=-1.0)  # already past
    trace = _drive(policy)
    assert not [c for c in tuner.calls if c[0] == "path"]
    assert [e["status"] for e in trace.entries] == ["no time", "no time"]
    assert not_reached == ["processor threads (time limit reached)", "prompt batch size (time limit reached)"]
    assert "Round 2 (processor threads) not reached: time limit." in tuner.lines


def test_the_engine_switch_check_runs_before_its_path_as_a_child() -> None:
    tuner = ScriptedTuner("engine_switches")
    policy, rounds, _ = _policy(tuner)
    trace = _drive(policy)
    assert [c[0] for c in tuner.calls] == ["facts", "switch check", "path"]
    assert [r["round"] for r in rounds] == ["facts", "engine switch checks", "engine switches"]
    parent, child = trace.entries
    assert child["parent"] == parent["id"] and child["kind"] == co.SWITCH_CHECK and child["status"] == co.DONE
    assert "Round 2: engine switches" in tuner.lines


def _layer_round(new_layers: int | None) -> Callable[[RuntimeConfiguration], Any]:
    def script(current: RuntimeConfiguration) -> Any:
        record: dict[str, object] = {"round": "graphics card layers", "revisit": True}
        if new_layers is None:
            record["reason"] = "no count beats the current one"
            return record, None, None
        return record, _change("gpu_offload_layers", current.gpu_offload_layers, new_layers, 99), None

    return script


@pytest.mark.parametrize("better", [13, None])
def test_a_memory_shaping_change_brings_the_layer_count_back_first(better: int | None) -> None:
    tuner = ScriptedTuner("threads,kv_cache", path_round=_kept("kv_cache_type_k", "q8_0"), layer_round=_layer_round(better))
    policy, rounds, _ = _policy(tuner)
    trace = _drive(policy)
    order = [c[0] if c[0] != "path" else c[1] for c in tuner.calls]
    # The layer revisit runs inside the kept change, before the affected paths run again
    # (a better layer count also brings the memory format back: layers affect it).
    assert order == ["facts", "threads", "kv_cache", "layers", "threads"] + (["kv_cache"] if better else [])
    layer_call = next(c for c in tuner.calls if c[0] == "layers")
    assert layer_call[1] is True and layer_call[2]["kv_cache_type_k"] == "q8_0" and layer_call[3] == float("inf")
    cache = next(e for e in trace.entries if e["path"] == "kv_cache")
    child = next(e for e in trace.entries if e["kind"] == co.LAYER_REVISIT)
    assert child["parent"] == cache["id"]
    assert trace.entries.index(child) < len(trace.entries) - 1  # recorded before the threads revisit
    again = [r for r in rounds if r.get("round") == "graphics card layers"]
    assert again and again[0]["again"] is True
    threads_again = [e for e in trace.entries if e["path"] == "threads"][-1]
    if better is None:
        assert child["status"] == co.NO_CHANGE and "  no better layer count: no count beats the current one" in tuner.lines
        assert threads_again["reason"] == "revisit after a kept change on kv_cache (affected_paths)"
        assert policy.current.gpu_offload_layers == 8
    else:
        assert child["status"] == co.CHANGED and child["carried_after"]["gpu_offload_layers"] == 13
        assert threads_again["reason"] == "revisit after a kept change on kv_cache/layers (affected_paths)"
        assert threads_again["carried"]["gpu_offload_layers"] == 13 and policy.current.gpu_offload_layers == 13
        assert {c.setting for c in policy.changes} == {"kv_cache_type_k", "gpu_offload_layers"}


def test_with_the_layer_path_off_a_memory_change_does_not_bring_layers_back() -> None:
    tuner = ScriptedTuner("threads,kv_cache", layer_path=False, path_round=_kept("kv_cache_type_k", "q8_0"),
                          layer_round=_layer_round(13))
    policy, _, _ = _policy(tuner)
    trace = _drive(policy)
    assert not [c for c in tuner.calls if c[0] == "layers"]
    assert not [e for e in trace.entries if e["kind"] == co.LAYER_REVISIT]


def test_a_path_that_does_not_apply_is_recorded_as_such() -> None:
    def script(path: Any, current: RuntimeConfiguration, n: int) -> Any:
        return {"round": path.label, "path": path.name, "applies": False, "reason": "skipped: only one core"}, None, None

    tuner = ScriptedTuner("threads", path_round=script)
    policy, _, _ = _policy(tuner)
    trace = _drive(policy)
    assert trace.entries[0]["status"] == co.NOT_APPLICABLE and trace.entries[0]["why"] == "skipped: only one core"
    assert tuner.lines[-1] == "  no change: skipped: only one core"


@pytest.mark.parametrize("error", [RuntimeError("engine crashed"), KeyboardInterrupt()])
def test_an_error_or_a_user_stop_inside_a_path_leaves_through_the_coordinator(error: BaseException) -> None:
    def script(path: Any, current: RuntimeConfiguration, n: int) -> Any:
        if path.name == "prompt_batch":
            raise error
        return {"round": path.label, "path": path.name, "applies": True, "reason": "flat"}, None, None

    tuner = ScriptedTuner("threads,prompt_batch,flash_attention", path_round=script)
    policy, _, _ = _policy(tuner)
    trace = co.SearchTrace(policy.mode)
    policy.start(trace)
    with pytest.raises(type(error)) as caught:
        co.coordinate(policy, trace)
    assert caught.value is error
    assert [c[1] for c in tuner.calls if c[0] == "path"] == ["threads", "prompt_batch"]  # nothing after it
    assert [e["status"] for e in trace.entries] == ["no change", "interrupted"]
    assert ("prompt_batch", "prompt_batch") not in policy.tested and "prompt_batch" not in policy.finished


# --------------------------------------------------------------------------- #
# Whole Unbounded tunes through the real Tuner and the `tune` command           #
# --------------------------------------------------------------------------- #
def _revisit_responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    speed, latency = 20.0, 2.0
    if task.workload.workload_id.startswith("long-") and options.ubatch_size == 1024:
        latency *= 0.80
    if options.ubatch_size == 1024 and options.threads == 5:
        speed *= 1.15
    return SampleResult(decode_tps=speed, total_latency_s=latency)


def test_a_real_unbounded_tune_records_its_search_actions(tmp_path: Path) -> None:
    settings = _settings(tmp_path, optimization_mode="performance_first", tune_paths="threads,prompt_batch",
                         tune_diagnosis=False, tune_layer_path=False)
    tuner, _, _, _ = _tuner(tmp_path, _revisit_responder, answers=[True, True], settings=settings)
    result = tuner.run()
    trace = tuner._search_trace
    tops = [e for e in trace.entries if e["parent"] is None]
    assert [(e["path"], e["status"]) for e in tops] == [("threads", "no change"), ("prompt_batch", "changed"),
                                                      ("threads", "changed"), ("prompt_batch", "no change")]
    assert tops[3]["reason"] == "revisit after a kept change on threads (affected_paths)"
    assert tops[2]["label"] == "Round 4: processor threads (again, after a kept change)"
    record = next(r for r in result.rounds if r.get("round") == co.TRACE_ROUND)
    assert record["mode"] == "unbounded" and record["actions"] == trace.entries
    assert json.loads(result.to_json())["rounds"]  # the report stays serialisable
    assert result.final_configuration.threads == 5 and result.final_configuration.ubatch_size == 1024


class _InterruptingHarness(ScriptedHarness):
    """Raises once a candidate of the prompt-batch path is measured."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(_revisit_responder)
        self.error = error
        self.monitor_blocked: str | None = None

    def run_once(self, task: Any, **kwargs: Any) -> Any:
        options = kwargs["options"]
        if options.ubatch_size is not None:
            if isinstance(self.error, Exception) and type(self.error).__name__ == "MonitorCleanupUnresolved":
                self.monitor_blocked = "a counter sampler's cleanup was not confirmed (scripted)"
            raise self.error
        return super().run_once(task, **kwargs)


def _run_tune(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException) -> tuple[Any, Any, list[str], Path]:
    import inferenceos.backends.llama_cpp_resident_adapter as lra
    import inferenceos.profiling.runner as runner
    import inferenceos.runtime.gpu_search as gpu_search
    import inferenceos.runtime.resident_workload as rw
    import inferenceos.tune.commands as commands

    settings = _settings(tmp_path, optimization_mode="performance_first", tune_paths="threads,prompt_batch",
                         tune_diagnosis=False, tune_layer_path=False, tune_heat_check_runs=0)
    tuner, _, _, store = _tuner(tmp_path, _revisit_responder, answers=[True] * 4, settings=settings)
    tuner.harness = _InterruptingHarness(error)
    engine_calls: list[str] = []

    class Resident:
        def __init__(self, **_: Any) -> None:
            pass

        def stop(self) -> None:
            engine_calls.append("stop")

        def cancel(self) -> None:
            engine_calls.append("cancel")

        def target_identity(self) -> None:
            return None

    monkeypatch.setattr(lra, "LlamaCppResidentAdapter", lambda settings: object())
    monkeypatch.setattr(rw, "ResidentActiveWorkload", Resident)
    monkeypatch.setattr(gpu_search, "ResidentMeasurementAdapter", lambda resident, explicit_defaults=True: object())
    monkeypatch.setattr(runner, "BaselineProfiler", lambda **_: tuner.harness)
    monkeypatch.setattr(commands, "detect_hardware_profile", lambda settings: tuner.hardware)
    monkeypatch.setattr(commands, "check_machine", lambda settings, hardware: tuner.machine)
    monkeypatch.setattr(commands, "load_last_check", lambda settings: None)
    monkeypatch.setattr(commands, "save_check", lambda settings, machine: None)
    monkeypatch.setattr(commands, "load_model_profile", lambda path: tuner.model)
    monkeypatch.setattr(commands, "read_live_state", lambda: (0, None))
    monkeypatch.setattr(commands, "make_try_load", lambda **_: tuner.try_load)
    monkeypatch.setattr(commands, "make_piece_lines_reader", lambda **_: None)
    monkeypatch.setattr(commands, "Tuner", lambda **_: tuner)
    out = tmp_path / "reports" / "tune_report.json"

    def run() -> int:
        return commands.run_tune(tuner.settings, model_path=tuner.model_path, context_length=CONTEXT,
                                 approve_all=True, out=out)

    if isinstance(error, RuntimeError):
        # A measurement error is a failed run, not an exception; an error raised by the
        # path round itself leaves the action.
        tuner.harness = ScriptedHarness(_revisit_responder)
        original = tuner._path_round

        def failing(path: Any, *args: Any, **kwargs: Any) -> Any:
            if path.name == "prompt_batch":
                raise error
            return original(path, *args, **kwargs)

        tuner._path_round = failing  # type: ignore[method-assign]
    tuner._run_tune = run  # type: ignore[attr-defined]
    return tuner, store, engine_calls, out


def _interrupted_path(tuner: Any) -> dict[str, Any]:
    entries = tuner._search_trace.entries
    assert [e["path"] for e in entries] == ["threads", "prompt_batch"]
    assert entries[0]["status"] == co.NO_CHANGE
    return entries[-1]  # type: ignore[no-any-return]


def test_an_unresolved_monitor_cleanup_in_a_path_reaches_the_tune_report(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.telemetry.counters import MonitorCleanupUnresolved

    tuner, store, engine_calls, out = _run_tune(
        tmp_path, monkeypatch, MonitorCleanupUnresolved("no further measurement: cleanup not confirmed"))
    code = tuner._run_tune()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and report["outcome"].startswith("failed: no further measurement")
    assert report["error"]["type"] == "MonitorCleanupUnresolved" and report["saved_setup"] is None
    assert store.find(tuner.model_path, context_length=CONTEXT, power=tuner.machine.power_source) is None
    assert engine_calls == ["stop"]
    entry = _interrupted_path(tuner)
    assert entry["status"] == co.INTERRUPTED and entry["why"].startswith("MonitorCleanupUnresolved")


@pytest.mark.parametrize("error", [RuntimeError("engine crashed"), KeyboardInterrupt()])
def test_an_error_or_user_stop_in_a_path_leaves_tune_with_cleanup_and_nothing_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    tuner, store, engine_calls, out = _run_tune(tmp_path, monkeypatch, error)
    with pytest.raises(type(error)) as caught:
        tuner._run_tune()
    assert caught.value is error
    assert engine_calls == ["stop"] and not out.exists()
    assert store.find(tuner.model_path, context_length=CONTEXT, power=tuner.machine.power_source) is None
    entry = _interrupted_path(tuner)
    assert entry["status"] == co.INTERRUPTED and entry["why"].startswith(type(error).__name__)


# --------------------------------------------------------------------------- #
# Practical's representation (not wired until C3-7)                            #
# --------------------------------------------------------------------------- #
def test_practical_categories_map_to_canonical_path_ids() -> None:
    assert dict(co.PRACTICAL_PATH_IDS) == {"placement": "layers", "threads": "threads", "batch": "prompt_batch"}
    assert "placement" not in co.PRACTICAL_PATH_IDS.values()  # per-piece placement is not the layer count


def test_a_practical_step_keeps_its_captured_reference() -> None:
    from inferenceos.tune.planner import BATCH, PLACEMENT, THREADS, Planner, Structure

    planner = Planner(Structure(count_max=33, top=33, uses_experts=False, mixed_kinds=False, physical_cores=6,
                                logical_cores=12),
                      start_layers=20, start_threads=8, start_batch=512, processor_share=lambda layers: 0.6)
    planner.chosen = {PLACEMENT: 33, THREADS: 6, BATCH: 1024}
    planner.placement = 33
    assert planner.current_setup() == (33, 6, 1024)
    assert planner.base_for(THREADS) == (33, 8, 1024)  # threads measured against their reference
    assert planner.base_for(PLACEMENT) == (20, 8, 512)  # placement against the start

    def config_of(setup: tuple[int, int, int]) -> RuntimeConfiguration:
        return RuntimeConfiguration(context_length=CONTEXT, gpu_offload_layers=setup[0], threads=setup[1],
                                    ubatch_size=setup[2])

    step = planner._step(THREADS, [10, 9], "preferred candidates")
    assert step is not None
    trace = co.SearchTrace(co.PRACTICAL)
    action = co.practical_step_action(trace, step, planner.current_setup(), config_of)
    planner.chosen[THREADS] = 10  # the first candidate changes the planner's choice ...
    assert action.path == "threads" and action.kind == co.SCREEN
    assert action.carried is not None and action.carried["threads"] == 6
    assert action.reference is not None and (action.reference["gpu_offload_layers"], action.reference["threads"],
                                             action.reference["ubatch_size"]) == (33, 8, 1024)
    # ... but every candidate stays measured from the step's captured base.
    assert [(c["threads"], c["ubatch_size"]) for c in action.candidates] == [(10, 1024), (9, 1024)]
    assert [step.setup_for(v) for v in step.values] == [(33, 10, 1024), (33, 9, 1024)]
    layers = planner._step(PLACEMENT, [0, 1], "compare very different placements first")
    assert layers is not None
    placement = co.practical_step_action(trace, layers, planner.current_setup(), config_of)
    assert placement.path == "layers" and placement.reference is not None
    assert placement.reference["gpu_offload_layers"] == 20 and placement.reference["threads"] == 8
