"""The first practical-mode hardware evaluation (8 Oct 2026): the fixed starting
configuration, the search-order record, the explanation of a run, the historical
comparison rule, and the evaluation runner's storage isolation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.tune.budgeted import fixed_start
from tests.test_budgeted_tune import Clock, _practical, _round, _world, clock  # noqa: F401 - fixture
from tests.test_tune import CONTEXT, _settings, _tuner

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> Any:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- the fixed start
def test_fixed_start_is_exactly_what_was_given() -> None:
    config, source = fixed_start('{"gpu_offload_layers": 23}', 4096)
    print("\nfixed start:", config.model_dump(exclude_none=True), "|", source)
    assert config == RuntimeConfiguration(gpu_offload_layers=23, context_length=4096)
    assert all(v is None for k, v in config.model_dump().items() if k not in ("gpu_offload_layers", "context_length"))
    assert "every other setting the engine default" in source


@pytest.mark.parametrize("text, why", [
    ('{"gpu_offload_layers": 23, "context_length": 2048}', "may not set the context size"),
    ("{not json", "not valid JSON"),
    ("{}", "non-empty JSON object"),
    ('{"gpu_layers": 23}', "gpu_layers"),
])
def test_fixed_start_refuses_anything_else(text: str, why: str) -> None:
    with pytest.raises(Exception) as caught:
        fixed_start(text, 4096)
    assert why in str(caught.value)


def test_practical_search_starts_from_the_fixed_configuration(tmp_path: Path) -> None:
    usual, _, driver_usual, _ = _practical(tmp_path / "a", _world(None), 1e6)
    result_usual = driver_usual.run()
    fixed, _, driver, _ = _practical(tmp_path / "b", _world(None), 1e6,
                                     tune_experimental_start_config='{"gpu_offload_layers": 7}')
    result = driver.run()
    print("\nusual start:", result_usual.starting_configuration.gpu_offload_layers, result_usual.starting_point_source,
          "| fixed start:", result.starting_configuration.gpu_offload_layers, result.starting_point_source)
    assert result.starting_configuration == RuntimeConfiguration(gpu_offload_layers=7, context_length=CONTEXT)
    assert result.starting_point_source.startswith("fixed starting configuration for this evaluation")
    assert result_usual.starting_configuration.gpu_offload_layers != 7
    assert not result_usual.starting_point_source.startswith("fixed")


def test_unbounded_never_reads_the_fixed_start(tmp_path: Path) -> None:
    s = _settings(tmp_path, tune_heat_check_runs=0, tune_experimental_start_config='{"gpu_offload_layers": 7}')
    tuner, _, _, _ = _tuner(tmp_path, _world(None), answers=[True] * 8, settings=s)
    result = tuner.run()
    assert result.starting_configuration.gpu_offload_layers != 7
    assert not any(r.get("round") == "time budget" for r in result.rounds)


# --------------------------------------------------------- the search-order record
def test_every_operation_and_planner_step_is_recorded_in_order(tmp_path: Path, clock: Clock) -> None:
    tuner, _, driver, _ = _practical(tmp_path, _world(clock, answer_s=5.0, long_answer_s=12.0), 4000,
                                     clock=clock, load_s=4.0)
    result = driver.run()
    budget = _round(result, "time budget")
    ops, steps = budget["operations"], budget["steps"]
    kinds = [o["kind"] for o in ops]
    print("\noutcome:", result.outcome, "| operations:", [(o["n"], o["kind"], o["label"], o["wall_s"]) for o in ops][:12],
          "| steps:", [(s["category"], s["values"], s["reason"]) for s in steps])
    assert "measure" in kinds and ("load-try" in kinds) == (tuner.try_load is not None)
    assert [o["n"] for o in ops] == list(range(1, len(ops) + 1))
    began = [o["began_s"] for o in ops if o["kind"] != "final"]
    assert began == sorted(began)
    assert all(o["wall_s"] >= 0 for o in ops)
    assert ops[[o["kind"] for o in ops].index("measure")]["label"] == "start"
    assert steps and all(s["reason"] for s in steps) and budget["planner_log"]
    if result.outcome == "saved":
        assert "final" in kinds  # each final-check measurement is in the record too


# -------------------------------------------------------------- the explanation
def test_explanation_of_a_saved_run(tmp_path: Path, clock: Clock) -> None:
    explain = _load("explain_practical_run")
    _, _, driver, _ = _practical(tmp_path, _world(clock, answer_s=5.0, long_answer_s=12.0), 4000, clock=clock, load_s=4.0)
    report = json.loads(driver.run().to_json())
    e = explain.explain(report)
    print("\noutcome:", e["outcome"], "| strongest:", e["strongest"], "| time:", e["time"], "| verified:", e["verified"])
    assert e["order"] and e["steps"] and e["planner_log"]
    assert e["strongest"] is not None and e["strongest"]["first_seen_s"] is not None
    assert "delayed_by_low_value" in e["strongest"] and e["strongest"]["why"]
    assert e["time"]["searching_s"] > 0 and e["time"]["used_s"] is not None
    assert e["outcome"] == "saved" and e["verified"] and all(v["passed"] for v in e["verified"])
    assert e["limiting_step"] is None
    assert "## Search order" in explain.render_markdown(e)
    left_out = any("prompt batch left out" in line for line in e["planner_log"])
    if left_out and not e["long_question_measured"]:  # deferred on a predicted cost, said so
        assert any("PREDICTED cost" in u for u in e["unanswered"])


def test_explanation_names_the_limiting_step_when_nothing_is_verified(tmp_path: Path, clock: Clock) -> None:
    explain = _load("explain_practical_run")
    _, _, driver, _ = _practical(tmp_path, _world(clock, flat=True), 4000, clock=clock)
    report = json.loads(driver.run().to_json())
    e = explain.explain(report)
    print("\noutcome:", e["outcome"], "| limiting step:", e["limiting_step"])
    assert e["outcome"] != "saved" and e["limiting_step"]
    assert "no candidate reading was clear enough" in e["limiting_step"]


def test_explanation_says_when_verification_did_not_fit(tmp_path: Path, clock: Clock) -> None:
    """A target too short for any final check: the limiting step is verification time,
    not a missing gain (the next change must address that, not a longer target)."""
    explain = _load("explain_practical_run")
    _, _, driver, _ = _practical(tmp_path, _world(clock, answer_s=30.0), 700, clock=clock, load_s=10.0)
    report = json.loads(driver.run().to_json())
    e = explain.explain(report)
    print("\noutcome:", e["outcome"], "| limiting step:", e["limiting_step"], "| exploration:", e["why_ended"]["exploration"])
    assert e["outcome"] != "saved" and e["limiting_step"]
    assert any(w in e["limiting_step"] for w in ("verification was predicted not to fit", "exploration stopped: target"))


def _event(n: int, at: float, category: str, value: int, cand: list[int], ref: list[int], gain: float,
           noise: float, cand_op: int, ref_op: int, agg: tuple[float, float, int] | None = None,
           kind: str = "screen") -> dict[str, Any]:
    start = [23, 6, 512]
    names = {0: "gpu_offload_layers", 1: "threads", 2: "ubatch_size"}

    def side(setup: list[int], op: int) -> dict[str, Any]:
        return {"setup": setup, "settings": {names[i]: v for i, v in enumerate(setup) if v != start[i]},
                "measurement": op}
    g, nz, c = agg if agg is not None else (gain, noise, 1)
    return {"n": n, "kind": kind, "at_s": at, "category": category, "value": value, "question": "short-short",
            "reference": side(ref, ref_op), "candidate": side(cand, cand_op), "gain": gain, "noise": noise,
            "aggregate": {"gain": g, "noise": nz, "count": c}, "vs_start": None}


def _op(n: int, label: str, began: float, wall: float, setup: list[int]) -> dict[str, Any]:
    return {"n": n, "kind": "measure", "label": label, "question": "short-short", "began_s": began, "wall_s": wall,
            "settings": {"gpu_offload_layers": setup[0], "threads": setup[1], "ubatch_size": setup[2]}, "phases": {}}


def _synthetic(ops: list[dict[str, Any]], events: list[dict[str, Any]], *, readings: list[dict[str, Any]],
               chosen: list[int] | None, final: dict[str, Any] | None = None,
               outcome: str = "not confirmed") -> dict[str, Any]:
    names = {0: "gpu_offload_layers", 1: "threads", 2: "ubatch_size"}
    chosen_settings = None if chosen is None else {names[i]: v for i, v in enumerate(chosen) if v != [23, 6, 512][i]}
    rounds: list[dict[str, Any]] = [
        {"round": "practical search", "start_setup": [23, 6, 512], "readings": readings, "chosen": chosen_settings,
         "chosen_setup": chosen, "deferred": [], "measured_setups": [], "plan": []},
        {"round": "time budget", "operations": ops, "screenings": events, "steps": [], "costs": {}},
    ]
    if final is not None:
        rounds.append(final)
    return {"starting_configuration": {"gpu_offload_layers": 23, "context_length": 4096}, "context_length": 4096,
            "effort_mode": "performance_first", "goal_mode": "balanced", "outcome": outcome, "rounds": rounds}


def test_explanation_compares_complete_setups_not_one_setting() -> None:
    """Reviewer's reproduction: 10 threads read +42% with batch 512, but the setup sent to
    the final check used batch 2048, where 10 threads read only +5%. The +42% setup was
    NOT pursued; the old explanation (matching ``threads-10``) said it was."""
    explain = _load("explain_practical_run")
    ops = [_op(1, "start", 0, 30, [23, 6, 512]), _op(2, "threads-10", 30, 30, [23, 10, 512]),
           _op(3, "batch-reference", 60, 30, [23, 6, 2048]), _op(4, "threads-10", 90, 30, [23, 10, 2048])]
    events = [_event(1, 60, "threads", 10, [23, 10, 512], [23, 6, 512], 0.42, 0.02, 2, 1),
              _event(2, 120, "threads", 10, [23, 10, 2048], [23, 6, 2048], 0.05, 0.02, 4, 3)]
    readings = [{"category": "threads", "value": 10, "gain": 0.42, "noise": 0.02, "setup": [23, 10, 512], "repeats": 1},
                {"category": "threads", "value": 10, "gain": 0.05, "noise": 0.02, "setup": [23, 10, 2048], "repeats": 1}]
    report = _synthetic(ops, events, readings=readings, chosen=[23, 10, 2048])
    e = explain.explain(report)
    st = e["strongest"]
    print("\nstrongest:", st["settings"], st["gain"], "| pursued:", st["pursued"], "|", st["why"])
    assert st["setup"] == [23, 10, 512] and st["gain"] == pytest.approx(0.42)
    assert st["pursued"] is False and st["confirmed"] is False
    assert "a different complete setup" in st["why"] and "'ubatch_size': 2048" in st["why"] and "+5.0%" in st["why"]
    by_n = {o["n"]: o for o in e["order"]}
    assert by_n[2]["reading"]["gain"] == pytest.approx(0.42) and by_n[4]["reading"]["gain"] == pytest.approx(0.05)
    assert by_n[4]["reading"]["reference"] == {"ubatch_size": 2048}  # gain against its local reference


def test_explanation_never_backdates_a_later_aggregate() -> None:
    """A weak first reading (+1%) improved by a later repeat (+9%, average +5%): the first
    measurement shows +1% as read then; the average appears only at the repeat."""
    explain = _load("explain_practical_run")
    ops = [_op(1, "start", 0, 30, [23, 6, 512]), _op(2, "threads-10", 30, 30, [23, 10, 512]),
           _op(3, "placement-reference", 60, 30, [23, 6, 512]), _op(4, "placement-29", 90, 30, [29, 6, 512]),
           _op(5, "weak-repeat-reference", 300, 30, [23, 6, 512]), _op(6, "weak-repeat-threads-10", 330, 30, [23, 10, 512])]
    events = [_event(1, 60, "threads", 10, [23, 10, 512], [23, 6, 512], 0.01, 0.02, 2, 1),
              _event(2, 120, "placement", 29, [29, 6, 512], [23, 6, 512], 0.03, 0.02, 4, 3),
              _event(3, 360, "threads", 10, [23, 10, 512], [23, 6, 512], 0.09, 0.02, 6, 5,
                     agg=(0.05, 0.0141, 2), kind="weak repeat")]
    readings = [{"category": "threads", "value": 10, "gain": 0.05, "noise": 0.0141, "setup": [23, 10, 512], "repeats": 2},
                {"category": "placement", "value": 29, "gain": 0.03, "noise": 0.02, "setup": [29, 6, 512], "repeats": 1}]
    e = explain.explain(_synthetic(ops, events, readings=readings, chosen=None))
    by_n = {o["n"]: o for o in e["order"]}
    print("\nfirst:", by_n[2]["reading"], "\nrepeat:", by_n[6]["reading"])
    assert by_n[2]["reading"]["gain"] == pytest.approx(0.01) and "aggregate_after" not in by_n[2]["reading"]
    assert by_n[2]["reading"]["class"] == "weak"
    assert by_n[6]["reading"]["gain"] == pytest.approx(0.09)
    assert by_n[6]["reading"]["aggregate_after"]["gain"] == pytest.approx(0.05)
    st = e["strongest"]
    assert st["setup"] == [23, 10, 512] and st["gain"] == pytest.approx(0.05) and st["readings_averaged"] == 2
    assert st["first_seen_s"] == 60 and st["first_reading"]["gain"] == pytest.approx(0.01)
    assert st["evidence_complete_s"] == 360  # the +5% average existed only from the repeat on


def test_screening_events_are_recorded_when_each_comparison_happens(tmp_path: Path, clock: Clock) -> None:
    """The search records each comparison with complete setups, both measurements'
    operation numbers, its own gain and the aggregate right after it."""
    _, _, driver, _ = _practical(tmp_path, _world(clock, answer_s=5.0, long_answer_s=12.0), 4000, clock=clock, load_s=4.0)
    result = driver.run()
    budget = _round(result, "time budget")
    events, ops = budget["screenings"], {o["n"]: o for o in budget["operations"]}
    print("\nscreenings:", [(e["n"], e["kind"], e["category"], e["value"], e["candidate"]["setup"], e["gain"],
                             e["aggregate"]) for e in events])
    assert events and [e["n"] for e in events] == list(range(1, len(events) + 1))
    assert [e["at_s"] for e in events] == sorted(e["at_s"] for e in events)
    for e in events:
        for side in ("reference", "candidate"):
            op = ops[e[side]["measurement"]]
            assert op["question"] == e["question"] and op["kind"] == "measure"
            assert op["began_s"] <= e["at_s"]  # the measurement existed when the comparison was made
            layers, threads, batch = e[side]["setup"]
            assert e[side]["configuration"]["gpu_offload_layers"] == layers
        assert e["candidate"]["setup"][{"placement": 0, "threads": 1, "batch": 2}[e["category"]]] == e["value"]
        if e["aggregate"]["count"] == 1:
            assert e["aggregate"]["gain"] == e["gain"]


# ------------------------------------------------------- the historical comparison
def _hist(start_layers: int = 23, long_id: str = "long-question-21x-steady") -> dict[str, Any]:
    return {"label": "job 101", "starting_configuration": {"gpu_offload_layers": start_layers, "context_length": 4096},
            "context_length": 4096, "effort_mode": "performance_first", "goal_mode": "balanced",
            "final_checks": {"short-short": {"score": 0.141, "lower": 0.103, "upper": 0.178, "blocks": 4,
                                             "decision": "better"},
                             long_id: {"score": 0.232, "lower": 0.176, "upper": 0.288, "blocks": 4,
                                       "decision": "better"}},
            "final_configuration": {"gpu_offload_layers": 29, "ubatch_size": 1024}, "total_minutes": 147.3,
            "outcome": "saved"}


def _report(score: float = 0.13, extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    checks = [{"workload": "short-short", "must_be": "better", "passed": True,
               "fair": {"mean": score, "lower_bound": 0.09, "upper_bound": 0.17, "block_scores": [0.1] * 4,
                        "decision": "better"}}] + (extra or [])
    return {"starting_configuration": {"gpu_offload_layers": 23, "context_length": 4096}, "context_length": 4096,
            "effort_mode": "performance_first", "goal_mode": "balanced", "outcome": "saved",
            "rounds": [{"round": "final check", "checks": checks}, {"round": "time budget", "used_s": 960.0}]}


def test_historical_comparison_only_where_start_question_and_scoring_match() -> None:
    explain = _load("explain_practical_run")
    long_guard = {"workload": "long-question-20x-steady", "must_be": "better", "passed": True,
                  "fair": {"mean": 0.2, "block_scores": [0.2] * 4, "decision": "better"}}
    report = _report(extra=[long_guard])
    checks = explain.explain(report)["final_checks"]
    matched = explain.compare_historical(report, checks, _hist())
    by_q = {q["question"]: q for q in matched["per_question"]}
    print("\nmatched:", matched["per_question"])
    assert matched["matched_start_and_scoring"] and by_q["short-short"]["comparable"]
    assert by_q["short-short"]["ratio"] == pytest.approx(0.13 / 0.141, abs=1e-3)
    assert not by_q["long-question-20x-steady"]["comparable"]  # a different long question
    other = explain.compare_historical(report, checks, _hist(start_layers=24))
    print("different start:", other["differences"])
    assert not other["matched_start_and_scoring"] and not any(q["comparable"] for q in other["per_question"])
    small = explain.compare_historical(_report(score=0.02), explain.explain(_report(score=0.02))["final_checks"], _hist())
    assert "ratio" not in small["per_question"][0]  # under 3%: absolute difference only


def test_a_failed_final_check_gets_no_historical_gain_ratio() -> None:
    """Reviewer's reproduction: an unsure, failed final check with a +10% estimate was
    reported comparable with a 0.714 ratio against job 101's +14.1%. It is now shown as
    an unverified reading, with no ratio."""
    explain = _load("explain_practical_run")
    report = _report(score=0.10)
    check = report["rounds"][0]["checks"][0]
    check["passed"] = False
    check["fair"].update({"decision": "unsure", "lower_bound": -0.01})
    report["outcome"] = "not confirmed: the final check did not show a gain"
    e = explain.explain(report, historical=_hist())
    q = e["historical"]["per_question"][0]
    print("\nfailed check:", q, "| confirmation:", e["confirmation"])
    assert e["verified"] == [] and e["final_checks"][0]["label"].startswith("unverified reading")
    assert q["comparable"] is False and "ratio" not in q and q["this_run_unverified"] == pytest.approx(0.10)
    assert "unverified" in q["why"] and "did not pass on short-short" in q["why"]
    assert "UNVERIFIED" in explain.render_markdown(e) and "ratio" not in explain.render_markdown(e).split("## Historical")[1]
    # Incomplete or interrupted checks likewise.
    for key in ("abandoned", "interrupted", "error"):
        r = _report(score=0.13)
        r["rounds"][0][key] = "scripted"
        q = explain.explain(r, historical=_hist())["historical"]["per_question"][0]
        assert q["comparable"] is False and "ratio" not in q
    # The historical side needs the same eligibility.
    unsure_then = _hist()
    unsure_then["final_checks"]["short-short"]["decision"] = "unsure"
    q = explain.explain(_report(score=0.13), historical=unsure_then)["historical"]["per_question"][0]
    assert q["comparable"] is False and "ratio" not in q and "decided 'unsure'" in q["why"]
    # The earlier-code-version caveat stays.
    assert "earlier code version" in e["historical"]["note"]


def test_historical_report_is_read_from_its_final_check_notes() -> None:
    explain = _load("explain_practical_run")
    old = {"starting_configuration": {"gpu_offload_layers": 23, "context_length": 4096}, "context_length": 4096,
           "effort_mode": "performance_first", "goal_mode": "balanced", "outcome": "saved",
           "started_at": "2026-10-05T05:20:21.920639Z", "finished_at": "2026-10-05T07:47:39.761667Z",
           "notes": ["final check, short-short: 4 block(s): writing_speed +16.8%, first_word +11.3%, score +14.1% "
                     "(95% range +10.3% to +17.8%, 4 usable block(s)), processor speed 61-97% -> better; keep rule: kept",
                     "final check, long-question-21x-steady: 4 block(s): writing_speed +41.6%, first_word +4.8%, "
                     "score +23.2% (95% range +17.6% to +28.8%, 4 usable block(s)) -> better"]}
    h = explain.historical_from_unbounded_report(old, label="job 101")
    print("\nhistorical:", h["final_checks"], h["total_minutes"])
    assert {k: v for k, v in h["final_checks"]["short-short"].items() if k != "decision"} == pytest.approx(
        {"score": 0.141, "lower": 0.103, "upper": 0.178, "blocks": 4})
    assert h["final_checks"]["short-short"]["decision"] == "better"
    assert h["final_checks"]["long-question-21x-steady"]["decision"] == "better"
    assert h["final_checks"]["long-question-21x-steady"]["score"] == pytest.approx(0.232)
    assert h["total_minutes"] == pytest.approx(147.3, abs=0.1)


# ------------------------------------------------------------ runner isolation
@pytest.fixture()
def env_guard() -> Iterator[None]:
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _runner_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Path, Path, str, list[Any]]:
    runner = _load("run_practical_eval")
    real = tmp_path / "real_data"
    real.mkdir()
    for name, text in (("saved_setups.json", '{"setups": []}'), ("load_spread.json", "{}"),
                       ("memory_speed.json", "{}"), ("machine_check.json", "{}")):
        (real / name).write_text(text, encoding="utf-8")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model bytes")
    digest = hashlib.sha256(b"model bytes").hexdigest()
    monkeypatch.setattr(runner, "default_data_dir", lambda: real)
    calls: list[Any] = []
    return runner, real, model, digest, calls


def _args(tmp_path: Path, model: Path, digest: str, **over: str) -> list[str]:
    base = {"--model": str(model), "--context": "4096", "--budget-s": "900", "--start-config": "gpu_offload_layers=23",
            "--effort": "performance_first", "--goal": "balanced", "--scratch": str(tmp_path / "scratch"),
            "--out": str(tmp_path / "out" / "report.json"), "--summary": str(tmp_path / "out" / "summary.json"),
            "--explain": str(tmp_path / "out" / "explain"), "--run-id": "rid-1", "--expect-model-sha256": digest}
    base.update(over)
    return [x for k, v in base.items() for x in (k, v)]


def _valid_report(model_path: Path, context_length: int, *, outcome: str = "not confirmed: nothing better found",
                  start: int = 23) -> str:
    """A valid tune report for this evaluation (what ``run_tune`` writes)."""
    from inferenceos.contracts.tune import MachineCheck, TuneResult

    config = RuntimeConfiguration(gpu_offload_layers=start, context_length=context_length)
    return TuneResult(
        run_id="tune-1", model_path=str(model_path), context_length=context_length, effort_mode="performance_first",
        goal_mode="balanced", time_limit_s=900.0, machine=MachineCheck(), starting_configuration=config,
        starting_point_source="fixed starting configuration for this evaluation", final_configuration=config,
        rounds=[{"round": "practical search", "plan": [], "readings": [], "chosen": None},
                {"round": "time budget", "used_s": 400.0, "target_s": 900.0, "operations": [], "screenings": [],
                 "steps": [], "costs": {}}],
        outcome=outcome,
    ).to_json()


def _fake_tune(calls: list[Any], behaviour: str, real: Path) -> Any:
    def run_tune(settings: Any, *, model_path: Path, context_length: int, approve_all: bool, out: Path) -> int:
        calls.append({"data_dir": settings.data_dir, "budget": settings.tune_experimental_budget_s,
                      "start": settings.tune_experimental_start_config, "approve_all": approve_all,
                      "monitor": settings.tune_experimental_measurement_monitor})
        from inferenceos.tune.profiles import SetupStore

        Path(SetupStore(settings)._path).write_text('{"setups": ["scratch"]}', encoding="utf-8")  # noqa: SLF001
        (Path(settings.data_dir) / "load_spread.json").write_text('{"k": [0.01]}', encoding="utf-8")
        if behaviour == "breach":
            (real / "saved_setups.json").write_text('{"setups": ["oops"]}', encoding="utf-8")
        if behaviour == "error":
            raise RuntimeError("scripted failure inside the tune")
        if behaviour == "interrupt":
            raise KeyboardInterrupt("scripted user stop")
        out.parent.mkdir(parents=True, exist_ok=True)
        if behaviour == "missing report":
            return 0
        if behaviour == "malformed report":
            out.write_text('{"outcome": "saved", "rounds": [', encoding="utf-8")
            return 0
        if behaviour == "not a tune report":
            out.write_text(json.dumps({"outcome": "no change", "starting_configuration": {}, "rounds": []}),
                           encoding="utf-8")
            return 0
        if behaviour == "another start":
            out.write_text(_valid_report(model_path, context_length, start=29), encoding="utf-8")
            return 0
        if behaviour == "failed outcome, exit 0":
            out.write_text(_valid_report(model_path, context_length, outcome="failed: scripted"), encoding="utf-8")
            return 0
        if behaviour == "saved outcome, exit 1":
            out.write_text(_valid_report(model_path, context_length, outcome="saved"), encoding="utf-8")
            return 1
        if behaviour == "tune failed":
            out.write_text(_valid_report(model_path, context_length, outcome="failed: scripted failure"),
                           encoding="utf-8")
            return 1
        out.write_text(_valid_report(model_path, context_length), encoding="utf-8")
        return 0
    return run_tune


@pytest.mark.parametrize("behaviour, code", [("normal", 0), ("error", 3), ("interrupt", 3), ("breach", 6)])
def test_runner_isolates_the_owners_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_guard: None,
                                           behaviour: str, code: int) -> None:
    runner, real, model, digest, calls = _runner_world(tmp_path, monkeypatch)
    import inferenceos.tune.commands as commands

    monkeypatch.setattr(commands, "run_tune", _fake_tune(calls, behaviour, real))
    before = {p.name: p.read_bytes() for p in real.iterdir()}
    exit_code = runner.main(_args(tmp_path, model, digest))
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    print(f"\n{behaviour}: exit {exit_code} | status {summary['status']} | isolation {summary.get('isolation')}")
    assert exit_code == code and len(calls) == 1
    call = calls[0]
    assert Path(call["data_dir"]).resolve().is_relative_to((tmp_path / "scratch").resolve())
    assert call["budget"] == 900 and json.loads(call["start"]) == {"gpu_offload_layers": 23}
    assert call["approve_all"] is True and call["monitor"] is False
    assert "scratch/data/saved_setups.json".replace("scratch/", "") in "\n".join(summary["isolation"]["scratch_files"])
    if behaviour == "breach":
        assert not summary["isolation"]["real_data_unchanged"]
        assert summary["status"].startswith("isolation breach") and "saved_setups.json: changed" in summary["status"]
    else:
        assert summary["isolation"]["real_data_unchanged"]
        assert {p.name: p.read_bytes() for p in real.iterdir()} == before  # byte-identical, nothing restored


@pytest.mark.parametrize("behaviour, expect", [
    ("missing report", "the tune report is missing"),
    ("malformed report", "not readable JSON"),
    ("not a tune report", "not a valid tune report"),
    ("another start", "is not the fixed start"),
    ("failed outcome, exit 0", "exited 0 but its outcome is 'failed: scripted'"),
    ("saved outcome, exit 1", "exited 1 but its outcome is 'saved'"),
    ("explanation fails", "the explanation could not be written"),
])
def test_runner_never_reports_done_without_its_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        env_guard: None, behaviour: str, expect: str) -> None:
    """Reviewer's reproduction: a tune returning 0 without a report gave exit 0, status
    "done", outcome missing. Missing, malformed or contradictory evidence, or a failed
    explanation, is now a reporting error (exit 8); the artifacts are kept."""
    runner, real, model, digest, calls = _runner_world(tmp_path, monkeypatch)
    import inferenceos.tune.commands as commands

    monkeypatch.setattr(commands, "run_tune", _fake_tune(calls, "normal" if behaviour == "explanation fails"
                                                         else behaviour, real))
    if behaviour == "explanation fails":
        explain = _load("explain_practical_run")

        def broken(*_: Any, **__: Any) -> Any:
            raise RuntimeError("scripted explanation failure")
        monkeypatch.setattr(explain, "explain", broken)
    exit_code = runner.main(_args(tmp_path, model, digest))
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    print(f"\n{behaviour}: exit {exit_code} | {summary['status'][:200]}")
    assert exit_code == 8 and summary["status"].startswith("reporting error") and expect in summary["status"]
    assert summary["completion"]["complete"] is False and summary["isolation"]["real_data_unchanged"]
    if behaviour not in ("missing report",):
        assert (tmp_path / "out" / "report.json").is_file()  # the evidence is kept


def test_runner_reports_a_tune_failure_and_a_valid_no_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             env_guard: None) -> None:
    runner, real, model, digest, calls = _runner_world(tmp_path, monkeypatch)
    import inferenceos.tune.commands as commands

    monkeypatch.setattr(commands, "run_tune", _fake_tune(calls, "normal", real))
    assert runner.main(_args(tmp_path / "a", model, digest)) == 0
    ok = json.loads((tmp_path / "a" / "out" / "summary.json").read_text(encoding="utf-8"))
    assert ok["status"] == "done" and ok["completion"]["complete"] and ok["outcome"].startswith("not confirmed")
    assert all(Path(p).is_file() for p in ok["completion"]["explanation"])
    monkeypatch.setattr(commands, "run_tune", _fake_tune(calls, "tune failed", real))
    assert runner.main(_args(tmp_path / "b", model, digest)) == 1
    failed = json.loads((tmp_path / "b" / "out" / "summary.json").read_text(encoding="utf-8"))
    print("\nno change:", ok["status"], ok["outcome"], "| tune failure:", failed["status"], failed["outcome"])
    assert failed["status"] == "failed" and failed["outcome"].startswith("failed") and failed["completion"]["complete"]


@pytest.mark.parametrize("damage, expect", [
    ("scratch exists", "scratch folder already exists"),
    ("threshold changed", "tune_guard_margin"),
    ("scoped monitor on", "tune_experimental_measurement_monitor"),
    ("guard allowance 99%", "goal_guard_limit_percent"),
    ("quality limits weakened", "phase7r_base_similarity"),
    ("wrong model", "the model file is the expected one"),
    ("context in start", "may not set the context size"),
    ("report exists", "an output of this run already exists"),
])
def test_runner_refuses_before_starting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_guard: None,
                                        damage: str, expect: str) -> None:
    runner, real, model, digest, calls = _runner_world(tmp_path, monkeypatch)
    import inferenceos.tune.commands as commands

    monkeypatch.setattr(commands, "run_tune", _fake_tune(calls, "normal", real))
    over: dict[str, str] = {}
    if damage == "scratch exists":
        (tmp_path / "scratch").mkdir()
    elif damage == "threshold changed":
        monkeypatch.setenv("INFERENCEOS_TUNE_GUARD_MARGIN", "0.03")
    elif damage == "scoped monitor on":
        monkeypatch.setenv("INFERENCEOS_TUNE_EXPERIMENTAL_MEASUREMENT_MONITOR", "true")
    elif damage == "guard allowance 99%":  # reviewer's reproduction: a 99% guard allowance
        monkeypatch.setenv("INFERENCEOS_GOAL_GUARD_LIMIT_PERCENT", "99")
    elif damage == "quality limits weakened":  # reviewer's reproduction: quality thresholds at 0.01
        for name in ("PHASE7R_BASE_SIMILARITY", "PHASE7R_MIN_SIMILARITY", "PHASE7R_CONTENT_OVERLAP_MIN"):
            monkeypatch.setenv(f"INFERENCEOS_{name}", "0.01")
    elif damage == "wrong model":
        over["--expect-model-sha256"] = "0" * 64
    elif damage == "report exists":
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "report.json").write_text("{}", encoding="utf-8")
    else:
        over["--start-config"] = "gpu_offload_layers=23,context_length=2048"
    exit_code = runner.main(_args(tmp_path, model, digest, **over))
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    print(f"\n{damage}: exit {exit_code} | {summary['status'][:220]}")
    assert exit_code == 7 and not calls and summary["status"].startswith("refused")
    assert expect in summary["status"]
    if damage == "quality limits weakened":
        assert all(n in summary["status"] for n in ("phase7r_min_similarity", "phase7r_content_overlap_min"))


def test_preflight_freezes_every_setting_but_the_evaluations_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 env_guard: None) -> None:
    """The fixed-rules check covers every setting (not only ``tune_*``), reports the
    difference rather than resetting it, and records the resolved policies."""
    runner, real, model, digest, calls = _runner_world(tmp_path, monkeypatch)
    from inferenceos.config import Settings

    s = Settings(data_dir=tmp_path / "d", reports_dir=tmp_path / "r", logs_dir=tmp_path / "l",
                 effort_mode="performance_first", goal_mode="balanced", tune_experimental_budget_s=900.0,
                 tune_experimental_start_config='{"gpu_offload_layers": 23}')
    assert runner.frozen_rule_differences(s) == {}
    for name, value in (("goal_guard_limit_percent", 99.0), ("phase7r_base_similarity", 0.01),
                        ("payback_answering_hours_while_running", 5.0), ("phase6_gpu_offload_safety_headroom_fraction", 0.0),
                        ("telemetry_interval_s", 9.0), ("log_level", "DEBUG")):
        changed = s.model_copy(update={name: value})
        diff = runner.frozen_rule_differences(changed)
        assert list(diff) == [name] and diff[name]["value"] == value
        assert getattr(changed, name) == value  # reported, never reset
    policies = runner.resolved_policies(s)
    print("\nresolved goal policy:", policies["goal"], "| groups:", sorted(policies["settings_by_policy"]))
    assert policies["goal"]["guard_limit"] == pytest.approx(0.10)
    assert {"goal", "effort", "quality", "measurement", "resource safety", "payback"} <= set(policies["settings_by_policy"])
    assert policies["settings_by_policy"]["quality"]["phase7r_base_similarity"] == 0.60


def test_start_setting_text_form_for_powershell() -> None:
    runner = _load("run_practical_eval")
    assert json.loads(runner.start_config_json("gpu_offload_layers=23")) == {"gpu_offload_layers": 23}
    assert json.loads(runner.start_config_json('{"gpu_offload_layers": 23}')) == {"gpu_offload_layers": 23}
    assert json.loads(runner.start_config_json("threads=8, flash_attention=true")) == {"threads": 8, "flash_attention": True}


def _history_pick(final_passed: bool) -> dict[str, Any]:
    """A weak placement reading (+4%, not repeated: no time) whose exact setup is picked
    from complete-setup history for the final check; an unrelated weak thread reading in
    another setup stays deferred."""
    ops = [_op(1, "start", 0, 30, [23, 6, 512]), _op(2, "placement-27", 30, 30, [27, 6, 512]),
           _op(3, "threads-10", 60, 30, [27, 10, 512])]
    events = [_event(1, 60, "placement", 27, [27, 6, 512], [23, 6, 512], 0.04, 0.028, 2, 1),
              _event(2, 90, "threads", 10, [27, 10, 512], [27, 6, 512], 0.02, 0.028, 3, 2)]
    final = {"round": "final check", "setup": [27, 6, 512], "settings": {"gpu_offload_layers": 27},
             "checks": [{"workload": "short-short", "must_be": "better", "passed": final_passed,
                         "fair": {"mean": 0.034, "lower_bound": 0.006 if final_passed else -0.01,
                                  "block_scores": [0.03] * 11, "decision": "better" if final_passed else "unsure"}}]}
    report = _synthetic(ops, events, readings=[], chosen=[27, 6, 512], final=final,
                        outcome="saved" if final_passed else "not confirmed")
    search = report["rounds"][0]
    search["deferred"] = [{"category": "placement", "value": 27, "gain": 0.04, "noise": 0.028, "setup": [27, 6, 512]},
                          {"category": "threads", "value": 10, "gain": 0.02, "noise": 0.028, "setup": [27, 10, 512]}]
    search["plan"] = ["weak placement 27 (+4.0%) not repeated: repeat 59 s + its check 645 s do not fit the target",
                      "weak threads 10 (+2.0%) not repeated: repeat 59 s + its check 645 s do not fit the target"]
    return report


def test_a_confirmed_setup_is_not_left_unanswered() -> None:
    """Reviewer's finding 4 (job 125): 27 layers was confirmed and saved, yet 'weak reading
    deferred: placement 27' stayed in Unanswered. Now the exact confirmed setup is answered
    by the final check; the search decision stays in the planner log; an unrelated deferred
    setup stays unanswered."""
    explain = _load("explain_practical_run")
    e = explain.explain(_history_pick(final_passed=True))
    print("\nunanswered:", e["unanswered"], "\nanswered:", e["answered_by_final_check"])
    assert not any("placement 27" in u for u in e["unanswered"])
    assert any("placement 27" in u for u in e["answered_by_final_check"])
    assert any("threads 10" in u for u in e["unanswered"])  # unrelated setup: still open
    assert any("weak placement 27" in line for line in e["planner_log"])  # history kept
    assert "answered by the final check" in explain.render_markdown(e)
    # A final check that did not pass answers nothing.
    failed = explain.explain(_history_pick(final_passed=False))
    assert any("placement 27" in u for u in failed["unanswered"]) and failed["answered_by_final_check"] == []
