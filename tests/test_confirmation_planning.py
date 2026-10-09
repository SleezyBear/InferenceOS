"""Confirmation planning for the practical search's one fresh final check (9 Oct 2026).

The block count is chosen before the check starts: enough for the planned power under an
uncertainty-inflated spread, only when the whole check is predicted to finish by the soft
target, never fewer than the selection-time count, and never changed once the check has
begun. The test, its alpha, its single look and the acceptance rule are unchanged."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from inferenceos.tune import budgeted
from inferenceos.tune.budgeted import BudgetedTune
from inferenceos.tune.noise_store import NoiseStore
from tests.test_budgeted_tune import Clock, _practical, _round, _world, clock  # noqa: F401 - fixture

def _run(tmp_path: Path, clock: Clock, gain: float, target: float, *,
         slow_candidate: float = 1.0) -> tuple[Any, dict[str, Any] | None]:
    """A world where 20+ layers read ``gain`` (deterministic answers, so the screening
    noise width is the fallback load-to-load term alone)."""
    hang = None
    if slow_candidate != 1.0:
        def hang(task: Any, options: Any) -> float:
            long = task.workload.workload_id.startswith("long-")
            base = 30.0 if long else 10.0
            return base * (slow_candidate - 1.0) if (options.gpu_offload_layers or 0) >= 20 else 0.0
    world = _world(clock, factor=lambda L, t, b, long: gain if L >= 20 else 1.0, hang=hang)
    tuner, store, driver, _ = _practical(tmp_path, world, target, clock=clock, optimization_mode="performance_first")
    result = driver.run()
    final = next((r for r in result.rounds if r.get("round") == "final check"), None)
    return result, final


def test_planned_power_follows_the_t_test_it_plans_for() -> None:
    power = BudgetedTune.planned_power
    assert power(4, 0.0, 0.02, 0.05) == 0.0 and power(4, 0.05, 0.0, 0.05) == 0.0
    series = [power(n, 0.03, 0.03, 0.05) for n in range(4, 17)]
    assert series == sorted(series) and series[-1] > 0.9 > series[0]
    # The Student t critical value makes four blocks clearly weaker than the normal
    # formula the selection count uses assumed.
    z_power = 0.5 * (1 + math.erf((0.03 * 2 / 0.03 - 1.645) / math.sqrt(2)))
    assert power(4, 0.03, 0.03, 0.05) < z_power - 0.1


def test_a_small_gain_gets_more_blocks_when_the_larger_check_fits_the_target(tmp_path: Path, clock: Clock) -> None:
    result, final = _run(tmp_path, clock, 1.03, 3600.0)
    assert final is not None and result.outcome == "saved"
    plan = final["confirmation_plan"]
    q = plan["questions"]["short"]
    print("\nplan:", {k: plan[k] for k in ("predicted_check_s", "predicted_finish_s", "soft_target_s", "reason")},
          "| selection", q["selection_blocks"], "chosen", q["chosen_blocks"])
    assert q["chosen_blocks"] > q["selection_blocks"]
    assert plan["predicted_finish_s"] <= plan["soft_target_s"]
    assert "finish by the soft target" in plan["reason"]
    # The check ran exactly the planned count, at one look and the unchanged alpha.
    fair = final["checks"][0]["fair"]
    assert fair["looks"] == [q["chosen_blocks"]] and fair["alpha_per_look"] == pytest.approx(0.05)
    assert final["blocks_per_question"] == {"short": q["chosen_blocks"]}
    assert final["calibration"]["short"]["chosen_blocks"] == q["chosen_blocks"]
    assert final["calibration"]["short"]["selection_blocks"] == q["selection_blocks"]
    # Planned before the check began; nothing appended afterwards.
    assert plan["planned_at_s"] <= final["started_at_s"]
    assert fair["attempts"] <= plan["max_attempts_per_question"]["short"]


def test_the_plan_records_measurements_apart_from_predictions(tmp_path: Path, clock: Clock) -> None:
    _, final = _run(tmp_path, clock, 1.03, 3600.0)
    assert final is not None
    plan = final["confirmation_plan"]
    q = plan["questions"]["short"]
    measured, predicted = q["measured"], q["predicted"]
    assert set(measured) >= {"direct_reading_vs_start", "screening_noise_width", "noise_inputs",
                             "start_side_measurement_s", "candidate_side_measurement_s"}
    assert measured["noise_inputs"]["between_source"] == "fallback"
    assert set(predicted) >= {"effect", "effect_basis", "noise_source", "uncertainty_allowance", "allowance_basis",
                              "planning_spread", "blocks_for_power", "power_at_chosen", "power_at_selection_count",
                              "per_block_s", "check_s"}
    assert predicted["uncertainty_allowance"] == budgeted.NOISE_ALLOWANCE_FALLBACK
    assert predicted["planning_spread"] == pytest.approx(measured["screening_noise_width"] * 1.75, rel=1e-4)
    assert predicted["power_at_chosen"] >= budgeted.PLAN_POWER > predicted["power_at_selection_count"]
    assert {"available_to_target_s", "save_and_cleanup_allowance_s", "hard_limit_s", "selection_estimate_s",
            "measured_check_s", "rule", "status"} <= set(plan)
    assert plan["measured_check_s"] == final["took_s"]


def test_a_check_that_would_pass_the_target_keeps_the_selection_count(tmp_path: Path, clock: Clock) -> None:
    """More blocks are only ever bought inside the soft target: when even the selection
    count needs the extension, nothing is added (the existing extension behaviour)."""
    _, final = _run(tmp_path, clock, 1.03, 1500.0)
    assert final is not None
    plan = final["confirmation_plan"]
    q = plan["questions"]["short"]
    print("\nreason:", plan["reason"])
    assert q["chosen_blocks"] == q["selection_blocks"]
    assert q["predicted"]["blocks_for_power"] > q["selection_blocks"]
    assert "none fit the soft target" in plan["reason"] and "extension" in plan["reason"]


@pytest.mark.parametrize("target", [1800.0, 2000.0, 2400.0, 3600.0])
def test_more_blocks_never_push_the_predicted_finish_past_the_target(tmp_path: Path, clock: Clock,
                                                                     target: float) -> None:
    _, final = _run(tmp_path, clock, 1.03, target)
    assert final is not None
    plan = final["confirmation_plan"]
    q = plan["questions"]["short"]
    print(f"\ntarget {target}: selection {q['selection_blocks']} chosen {q['chosen_blocks']} "
          f"finish {plan['predicted_finish_s']}")
    assert q["chosen_blocks"] >= q["selection_blocks"]
    if q["chosen_blocks"] > q["selection_blocks"]:
        assert plan["predicted_finish_s"] <= plan["soft_target_s"]
        assert q["chosen_blocks"] <= q["predicted"]["blocks_for_power"]


def test_a_large_gain_still_finishes_with_the_minimum(tmp_path: Path, clock: Clock) -> None:
    result, final = _run(tmp_path, clock, 1.5, 3600.0)
    assert final is not None and result.outcome == "saved"
    plan = final["confirmation_plan"]
    q = plan["questions"]["short"]
    assert q["chosen_blocks"] == q["selection_blocks"] == 4
    assert plan["reason"].startswith("selection count kept: it already gives the planned power")


def test_a_learned_load_spread_uses_the_smaller_allowance(tmp_path: Path, clock: Clock,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NoiseStore, "estimate", lambda self, key: 0.012)
    _, final = _run(tmp_path, clock, 1.03, 3600.0)
    assert final is not None
    q = final["confirmation_plan"]["questions"]["short"]
    assert q["measured"]["noise_inputs"]["between_source"] == "learned"
    assert q["predicted"]["uncertainty_allowance"] == budgeted.NOISE_ALLOWANCE_LEARNED
    assert "learned for this model" in q["predicted"]["noise_source"]


def test_unequal_side_costs_are_forecast_from_each_side(tmp_path: Path, clock: Clock) -> None:
    _, final = _run(tmp_path, clock, 1.03, 3600.0, slow_candidate=2.0)
    assert final is not None
    plan = final["confirmation_plan"]
    q = plan["questions"]["short"]
    start_s = q["measured"]["start_side_measurement_s"]
    cand_s = q["measured"]["candidate_side_measurement_s"]
    print(f"\nstart side {start_s} s, candidate side {cand_s} s, predicted {plan['predicted_check_s']} s, "
          f"measured {plan['measured_check_s']} s")
    assert cand_s > 1.5 * start_s
    assert q["predicted"]["per_block_s"] == pytest.approx(start_s + cand_s, abs=0.2)
    assert plan["predicted_check_s"] == pytest.approx((start_s + cand_s) * (q["chosen_blocks"] + 1), abs=1.0)
    # The measured check lies within the forecast (which keeps one re-attempt in reserve).
    assert plan["measured_check_s"] <= plan["predicted_check_s"] + 1.0


def test_the_simulation_compares_planners_on_the_same_acceptance_test() -> None:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "simulate_confirmation.py"
    spec = importlib.util.spec_from_file_location("simulate_confirmation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["simulate_confirmation"] = module
    spec.loader.exec_module(module)
    world = next(w for w in module.worlds() if w.label == "weak, noise as assumed")
    current = module.run(world, "seed-1", "current")
    planned = module.run(world, "seed-1", "planned")
    print("\ncurrent:", current, "\nplanned:", planned)
    # Same seed, same search: the two differ only from the final check on.
    assert (current.blocks is None) == (planned.blocks is None)
    if planned.blocks is not None:
        assert current.blocks is not None and planned.blocks >= current.blocks
    # The planner is restored after each run.
    assert budgeted.BudgetedTune._plan_confirmation is not module._current_plan


def test_the_explanation_shows_the_plan_and_labels_predictions(tmp_path: Path, clock: Clock) -> None:
    import importlib.util
    import json
    import sys

    result, final = _run(tmp_path, clock, 1.03, 3600.0)
    assert final is not None
    path = Path(__file__).resolve().parents[1] / "scripts" / "explain_practical_run.py"
    spec = importlib.util.spec_from_file_location("explain_practical_run", path)
    assert spec is not None and spec.loader is not None
    explain = importlib.util.module_from_spec(spec)
    sys.modules["explain_practical_run"] = explain
    spec.loader.exec_module(explain)
    e = explain.explain(json.loads(result.to_json()))
    plan = e["confirmation_plan"]
    q = plan["questions"]["short"]
    assert q["chosen_blocks"] == final["confirmation_plan"]["questions"]["short"]["chosen_blocks"]
    text = explain.render_markdown(e)
    print("\n" + text.split("## Confirmation plan")[1].split("## Final check")[0])
    assert "## Confirmation plan (chosen before the check started)" in text
    assert "Measured:" in text and "Predicted:" in text
    # A report from before this change has no plan and still explains.
    old = json.loads(result.to_json())
    for r in old["rounds"]:
        r.pop("confirmation_plan", None)
    assert explain.explain(old)["confirmation_plan"] is None


# ----------------------------------------------- reviewer findings, 9 Oct 2026 (reporting)
def _planner_only(tmp_path: Path, clock: Clock, *, width: float, effect: float, now: float, target: float,
                  sides: tuple[float, float] = (28.03, 27.62)) -> Any:
    """A practical driver whose planning inputs are set directly, so ``_plan_confirmation``
    can be checked on exact numbers (the reviewer's reproductions)."""
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from tests.test_tune import CONTEXT

    tuner, _, driver, _ = _practical(tmp_path, _world(clock), target, clock=clock, optimization_mode="performance_first")
    workload = tuner.workload
    driver._short, driver._start_setup = workload, (23, 6, 512)
    driver._t0, driver._target, driver._hard, driver._save_reserve = 0.0, target, target + 180.0, 30.0
    clock.seconds = now
    driver._config = lambda setup: RuntimeConfiguration(context_length=CONTEXT, gpu_offload_layers=setup[0])  # type: ignore[method-assign]
    driver._vs_start = lambda setup, question: effect / 0.7  # type: ignore[method-assign]
    driver._effect = lambda setup, question, planning, assumed: effect  # type: ignore[method-assign]
    driver._cost = lambda question: 29.3  # type: ignore[method-assign]
    driver._spread = {"short": width}
    driver._spread_info = {"short": {"between_source": "fallback"}}
    driver.operations = [
        {"kind": "measure", "question": workload.workload_id, "error": None, "wall_s": wall,
         "settings": {"gpu_offload_layers": layers, "threads": None, "ubatch_size": None}}
        for layers, wall in ((23, sides[0]), (27, sides[1]))]
    return driver


def test_a_capped_plan_never_claims_the_planned_power(tmp_path: Path, clock: Clock) -> None:
    """Reviewer's reproduction: width 0.1, effect 0.014, fallback allowance, selection count
    16, ample time. The count stays 16, but no count up to the cap reaches 80%: the record
    says so instead of 'it already gives the planned power'."""
    driver = _planner_only(tmp_path, clock, width=0.1, effect=0.014, now=100.0, target=100000.0)
    plan, record = driver._plan_confirmation((27, 6, 512), {"short": ("better", 16)})
    q = record["questions"]["short"]
    print("\nreason:", record["reason"])
    assert plan == {"short": ("better", 16)}  # the selected count is unchanged
    assert q["predicted"]["blocks_for_power"] is None
    assert q["predicted"]["power_target_attainable"] is False
    assert q["predicted"]["power_at_chosen"] == pytest.approx(0.086, abs=0.002)
    assert q["planned_power_limited_by"] == "block cap"
    assert "already gives the planned power" not in record["reason"]
    assert record["reason"].startswith("selection count kept at the 16-block cap")


def test_a_capped_plan_with_time_to_spare_uses_the_cap_and_says_why(tmp_path: Path, clock: Clock) -> None:
    driver = _planner_only(tmp_path, clock, width=0.1, effect=0.014, now=100.0, target=100000.0)
    plan, record = driver._plan_confirmation((27, 6, 512), {"short": ("better", 4)})
    q = record["questions"]["short"]
    assert plan == {"short": ("better", 16)}  # as before: the cap is the count aimed for
    assert q["predicted"]["blocks_for_power"] is None and q["planned_power_limited_by"] == "block cap"
    assert "not attainable within the 16-block cap" in record["reason"]
    assert "reaches" not in record["reason"]


def test_job_125_plan_is_time_limited_and_its_target_unattainable(tmp_path: Path, clock: Clock) -> None:
    """Job 125's recorded inputs: effect 0.028231, width 0.027771 (planning spread
    0.048599), sides 28.03 + 27.62 s, planned at 200.4 s, 900 s target. It chose 11
    blocks (selection 10); 16 gives only about 71%, so 16 is not 'blocks for the planned
    power' -- the soft target limited the count and the cap would not have reached 80%."""
    driver = _planner_only(tmp_path, clock, width=0.027771, effect=0.028231, now=200.4, target=900.0)
    plan, record = driver._plan_confirmation((27, 6, 512), {"short": ("better", 10)})
    q = record["questions"]["short"]
    print("\nreason:", record["reason"])
    assert plan == {"short": ("better", 11)}
    assert record["predicted_check_s"] == pytest.approx(667.8, abs=0.2)
    assert q["predicted"]["blocks_for_power"] is None and q["predicted"]["power_target_attainable"] is False
    assert q["predicted"]["power_at_cap"] == pytest.approx(0.707, abs=0.002)
    assert q["predicted"]["power_at_chosen"] == pytest.approx(0.542, abs=0.002)
    assert q["planned_power_limited_by"] == "soft target"
    assert "limited by the soft target; not attainable even at the 16-block cap" in record["reason"]


def test_an_attainable_target_limited_by_time_is_reported_as_such(tmp_path: Path, clock: Clock) -> None:
    driver = _planner_only(tmp_path, clock, width=0.0198, effect=0.0387, now=200.0, target=600.0)
    plan, record = driver._plan_confirmation((27, 6, 512), {"short": ("better", 4)})
    q = record["questions"]["short"]
    print("\nreason:", record["reason"], q["predicted"]["blocks_for_power"])
    assert q["predicted"]["blocks_for_power"] == 7 and q["predicted"]["power_target_attainable"] is True
    assert plan["short"][1] < 7 and q["planned_power_limited_by"] == "soft target"
    assert "limited by the soft target" in record["reason"] and "not attainable" not in record["reason"]


def test_planning_assumptions_give_different_four_block_estimates() -> None:
    """Reviewer's finding 2: the 46% figure for job 123's four blocks is the estimate under the
    NEW allowance, not the old check's own estimate. Under the same effect, the screening
    width alone gives about 87%, the old 1.25 margin about 71%, the 1.75 allowance about
    46%. These are planning estimates, not empirical probabilities."""
    power = BudgetedTune.planned_power
    effect, width = 0.038662, 0.019835
    assert power(4, effect, width, 0.05) == pytest.approx(0.867, abs=0.002)
    assert power(4, effect, width * 1.25, 0.05) == pytest.approx(0.709, abs=0.002)
    assert power(4, effect, width * 1.75, 0.05) == pytest.approx(0.464, abs=0.002)


def test_the_explanation_says_when_the_planned_power_is_not_attainable(tmp_path: Path, clock: Clock) -> None:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "explain_practical_run.py"
    spec = importlib.util.spec_from_file_location("explain_practical_run", path)
    assert spec is not None and spec.loader is not None
    explain = importlib.util.module_from_spec(spec)
    sys.modules["explain_practical_run"] = explain
    spec.loader.exec_module(explain)
    driver = _planner_only(tmp_path, clock, width=0.027771, effect=0.028231, now=200.4, target=900.0)
    _, record = driver._plan_confirmation((27, 6, 512), {"short": ("better", 10)})
    summary = explain._plan_summary(record)
    q = summary["questions"]["short"]
    assert q["power_target_attainable"] is False and q["planned_power_limited_by"] == "soft target"
    text = explain.render_markdown({"confirmation_plan": summary})
    print("\n" + text.split("## Confirmation plan")[1].split("## Final check")[0])
    assert "NOT attainable within the block cap" in text and "16 blocks for the planned power" not in text
    # A record written before the fix (job 125 itself) said "blocks_for_power: 16"; the
    # explanation recomputes and reports it as not attainable.
    old = json.loads(json.dumps(record))
    pred = old["questions"]["short"]["predicted"]
    for key in ("power_target_attainable", "power_at_cap", "block_cap"):
        pred.pop(key)
    pred["blocks_for_power"] = 16
    old["questions"]["short"].pop("planned_power_limited_by")
    old.pop("format_version")
    old.pop("selection_limitation")
    old["rule"] = explain.FIRST_PLAN_RULE
    old_summary = explain._plan_summary(old)["questions"]["short"]
    assert old_summary["power_target_attainable"] is False
    assert old_summary["predicted_power_at_cap"] == pytest.approx(0.707, abs=0.002)


def _explain_module() -> Any:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "explain_practical_run.py"
    spec = importlib.util.spec_from_file_location("explain_practical_run", path)
    assert spec is not None and spec.loader is not None
    explain = importlib.util.module_from_spec(spec)
    sys.modules["explain_practical_run"] = explain
    spec.loader.exec_module(explain)
    return explain


def _first_format_plan(alpha: Any, **predicted: Any) -> dict[str, Any]:
    """A plan in the first confirmation-plan format (job 125's): alpha at plan level,
    power target recorded, ``blocks_for_power`` clipped to the cap, no cap or
    attainability fields."""
    plan: dict[str, Any] = {"rule": _explain_module().FIRST_PLAN_RULE, "status": "experimental / provisional",
                            "questions": {"short": {"mode": "better", "selection_blocks": 10, "chosen_blocks": 11,
                                                    "predicted": dict(predicted)}}}
    if alpha is not None:
        plan["alpha_one_sided"] = alpha
    return plan


def test_an_older_plan_is_explained_with_its_own_recorded_alpha() -> None:
    """Reviewer reproduction (9 Oct): at alpha 0.01 the planner's formula gives 0.523 at
    16 blocks, so the target was not attainable. The explanation used to substitute
    0.05 and report attainable with 0.808 at the cap."""
    explain = _explain_module()
    assert BudgetedTune.planned_power(16, 0.02, 0.03, 0.01) == pytest.approx(0.523, abs=0.001)
    assert BudgetedTune.planned_power(16, 0.02, 0.03, 0.05) == pytest.approx(0.808, abs=0.001)
    plan = _first_format_plan(0.01, effect=0.02, planning_spread=0.03, blocks_for_power=16, power_target=0.8)
    q = explain._plan_summary(plan)["questions"]["short"]
    assert q["power_target_attainable"] is False
    assert q["predicted_power_at_cap"] == pytest.approx(0.523, abs=0.001)
    assert q["power_target_attainable_source"].startswith("derived") and "alpha 0.01" in q["power_target_attainable_source"]
    assert "16 from the first confirmation-plan format" in q["predicted_power_at_cap_source"]
    assert explain._plan_summary(plan)["format"] == "first"
    text = explain.render_markdown({"confirmation_plan": explain._plan_summary(plan)})
    assert "NOT attainable" in text and "0.523 at the cap (derived)" in text and "[derived" in text


def test_job_125_older_record_still_reads_unattainable_at_alpha_005() -> None:
    explain = _explain_module()
    plan = _first_format_plan(0.05, effect=0.028231, planning_spread=0.048599, blocks_for_power=16, power_target=0.8)
    q = explain._plan_summary(plan)["questions"]["short"]
    assert q["power_target_attainable"] is False
    assert q["predicted_power_at_cap"] == pytest.approx(0.707, abs=0.002)
    assert "alpha 0.05" in q["power_target_attainable_source"]


def test_an_older_plan_missing_an_input_is_labelled_unknown_not_filled_in() -> None:
    explain = _explain_module()
    no_alpha = _first_format_plan(None, effect=0.02, planning_spread=0.03, blocks_for_power=16, power_target=0.8)
    q = explain._plan_summary(no_alpha)["questions"]["short"]
    assert q["power_target_attainable"] is None and q["predicted_power_at_cap"] is None
    assert q["power_target_attainable_source"] == "unknown: the record lacks alpha_one_sided"
    no_target = _first_format_plan(0.05, effect=0.02, planning_spread=0.03, blocks_for_power=16)
    q = explain._plan_summary(no_target)["questions"]["short"]
    assert q["power_target_attainable"] is None and "power_target" in q["power_target_attainable_source"]
    no_count = _first_format_plan(0.05, effect=0.02, planning_spread=0.03, power_target=0.8)
    q = explain._plan_summary(no_count)["questions"]["short"]
    assert q["power_target_attainable"] is None and q["predicted_power_at_cap"] is None
    assert "contradicts the first format" in q["predicted_power_at_cap_source"]
    text = explain.render_markdown({"confirmation_plan": explain._plan_summary(no_alpha)})
    assert "attainable is unknown [unknown: the record lacks alpha_one_sided]" in text
    assert "blocks for the planned power" not in text


def test_recorded_plan_values_stay_authoritative() -> None:
    """A record that stores attainability, cap and power at the cap is read as recorded,
    never recomputed, even when today's formula or another alpha would say otherwise."""
    explain = _explain_module()
    plan = _first_format_plan(0.01, effect=0.02, planning_spread=0.03, blocks_for_power=12, power_target=0.8,
                              power_target_attainable=True, block_cap=16, power_at_cap=0.9)
    q = explain._plan_summary(plan)["questions"]["short"]
    assert q["power_target_attainable"] is True and q["power_target_attainable_source"] == "recorded"
    assert q["predicted_power_at_cap"] == 0.9 and q["predicted_power_at_cap_source"] == "recorded"
    text = explain.render_markdown({"confirmation_plan": explain._plan_summary(plan)})
    assert "12 blocks for the planned power" in text and "derived" not in text


def _simulator() -> Any:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "simulate_confirmation.py"
    spec = importlib.util.spec_from_file_location("simulate_confirmation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["simulate_confirmation"] = module
    spec.loader.exec_module(module)
    return module


def _recorded_rounds(module: Any, label: str, seed: str) -> tuple[list[dict[str, Any]], Any]:
    caught: dict[str, Any] = {}
    original = budgeted.BudgetedTune.run

    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        caught["r"] = result = original(self, *args, **kwargs)
        return result

    world = next(w for w in module.worlds() if w.label == label)
    try:
        setattr(budgeted.BudgetedTune, "run", run)
        module.run(world, seed, "planned")
    finally:
        setattr(budgeted.BudgetedTune, "run", original)
    return json.loads(json.dumps(caught["r"].rounds)), caught["r"].starting_configuration


def test_equal_final_check_start_times_do_not_count_as_the_same_search() -> None:
    """Reviewer (9 Oct): ``paired()`` compared only the final check's start time. Two
    different candidate sequences, or different selected setups, can start the check at
    the same moment; the pairing now compares every decision before the check."""
    module = _simulator()
    rounds, starting = _recorded_rounds(module, "weak, noise as assumed", "seed-1")
    final = next(r for r in rounds if r.get("round") == "final check")
    budget = next(r for r in rounds if r.get("round") == "time budget")
    reference = module.fingerprint_hash(module.search_fingerprint(rounds, starting))
    assert reference == module.fingerprint_hash(module.search_fingerprint(json.loads(json.dumps(rounds)), starting))

    reordered = json.loads(json.dumps(rounds))
    ops = next(r for r in reordered if r.get("round") == "time budget")["operations"]
    measures = [i for i, op in enumerate(ops) if op.get("kind") == "measure"]
    assert len(measures) >= 3
    i, j = measures[1], measures[2]
    ops[i]["settings"], ops[j]["settings"] = ops[j]["settings"], ops[i]["settings"]
    assert module.fingerprint_hash(module.search_fingerprint(reordered, starting)) != reference

    other_setup = json.loads(json.dumps(rounds))
    other_final = next(r for r in other_setup if r.get("round") == "final check")
    other_final["setup"] = [final["setup"][0] - 1] + list(final["setup"][1:])
    assert other_final["started_at_s"] == final["started_at_s"]
    assert module.fingerprint_hash(module.search_fingerprint(other_setup, starting)) != reference

    later = json.loads(json.dumps(rounds))
    next(r for r in later if r.get("round") == "final check")["blocks_per_question"] = {"short": 99}
    next(r for r in later if r.get("round") == "time budget")["operations"].append(
        {"kind": "final", "began_s": final["started_at_s"] + 1.0, "label": "final check short start"})
    assert module.fingerprint_hash(module.search_fingerprint(later, starting)) == reference
    assert budget["operations"]

    def trial(fp: str | None, start: float | None) -> Any:
        return module.Trial(confirmed=False, accepted=False, true_short=0.0, total=100.0, blocks=None,
                            predicted_check=None, measured_check=None, stop_status="", outcome="done",
                            final_start=start, seed="s", search_fingerprint=fp)

    pair = module.paired([trial("a", 200.0)], [trial("b", 200.0)])
    assert pair["same_confirmation_start_time"] == 1
    assert pair["same_search_until_final_check"] == 0 and pair["search_differs"] == ["s"]


def test_runs_without_a_final_check_are_compared_on_their_whole_search() -> None:
    module = _simulator()
    rounds, starting = _recorded_rounds(module, "weak, noise as assumed", "seed-1")
    no_final = [r for r in rounds if r.get("round") != "final check"]
    fp = module.search_fingerprint(no_final, starting)
    assert fp["final_check"] is None
    assert len(fp["operations"]) == len([op for op in next(r for r in rounds if r.get("round") == "time budget")[
        "operations"] if op.get("kind") != "final"])

    def trial(fp_: str | None) -> Any:
        return module.Trial(confirmed=False, accepted=False, true_short=0.0, total=100.0, blocks=None,
                            predicted_check=None, measured_check=None, stop_status="", outcome="done",
                            final_start=None, seed="s", search_fingerprint=fp_)

    different = module.paired([trial("a")], [trial("b")])
    assert different["both_without_final_check"] == 1 and different["same_confirmation_start_time"] == 0
    assert different["same_search_until_final_check"] == 0 and different["same_search_both_without_final_check"] == 0
    same = module.paired([trial("a")], [trial("a")])
    assert same["same_search_until_final_check"] == 1 and same["same_search_both_without_final_check"] == 1
    unknown = module.paired([trial(None)], [trial("a")])
    assert unknown["search_unknown"] == ["s"] and unknown["same_search_until_final_check"] == 0


def test_a_record_outside_the_first_formats_range_is_not_given_its_cap() -> None:
    """Reviewer (9 Oct): ``blocks_for_power`` 64 with no cap was taken as the first format
    and given cap 16, although that format never recorded more than 16."""
    explain = _explain_module()
    plan = _first_format_plan(0.05, effect=0.02, planning_spread=0.03, blocks_for_power=64, power_target=0.8)
    q = explain._plan_summary(plan)["questions"]["short"]
    assert q["power_target_attainable"] is None and q["predicted_power_at_cap"] is None
    assert "blocks_for_power 64 is not a count from 2 to 16" in q["power_target_attainable_source"]
    for change in ({"chosen_blocks": 9}, {"selection_blocks": 20}, {"chosen_blocks": True}):
        bad = _first_format_plan(0.05, effect=0.02, planning_spread=0.03, blocks_for_power=16, power_target=0.8)
        bad["questions"]["short"].update(change)
        assert explain._plan_summary(bad)["questions"]["short"]["power_target_attainable"] is None, change
    bad_alpha = _first_format_plan(5, effect=0.02, planning_spread=0.03, blocks_for_power=16, power_target=0.8)
    assert explain._plan_summary(bad_alpha)["questions"]["short"]["power_target_attainable"] is None


def test_unrecognised_plan_formats_stay_unknown() -> None:
    explain = _explain_module()
    no_rule = _first_format_plan(0.05, effect=0.02, planning_spread=0.03, blocks_for_power=16, power_target=0.8)
    no_rule.pop("rule")
    other_rule = json.loads(json.dumps(no_rule)) | {"rule": "something else"}
    mixed = json.loads(json.dumps(no_rule)) | {"rule": explain.FIRST_PLAN_RULE, "selection_limitation": "x"}
    future = json.loads(json.dumps(no_rule)) | {"format_version": "confirmation-plan-9"}
    for plan in (no_rule, other_rule, mixed, future):
        summary = explain._plan_summary(plan)
        q = summary["questions"]["short"]
        assert summary["format"] == "unrecognised"
        assert q["power_target_attainable"] is None and q["predicted_power_at_cap"] is None
        assert q["power_target_attainable_source"] == "unknown: the plan's format is not recognised"


def test_new_plans_carry_an_explicit_format_version(tmp_path: Path, clock: Clock) -> None:
    explain = _explain_module()
    driver = _planner_only(tmp_path, clock, width=0.027771, effect=0.028231, now=200.4, target=900.0)
    _, record = driver._plan_confirmation((27, 6, 512), {"short": ("better", 10)})
    assert record["format_version"] == budgeted.PLAN_FORMAT_VERSION == "confirmation-plan-3"
    assert explain.plan_format(record) == "confirmation-plan-3"
    q = explain._plan_summary(record)["questions"]["short"]
    assert q["power_target_attainable_source"] == "recorded" and q["predicted_power_at_cap_source"] == "recorded"
    missing = json.loads(json.dumps(record))
    for key in ("power_target_attainable", "power_at_cap"):
        missing["questions"]["short"]["predicted"].pop(key)
    q = explain._plan_summary(missing)["questions"]["short"]
    assert q["power_target_attainable"] is None and "lacks the stored value" in q["power_target_attainable_source"]


def test_job_125s_actual_record_is_recognised_as_the_first_format() -> None:
    explain = _explain_module()
    path = Path(__file__).resolve().parents[1] / "_jobs" / "out" / "125_practical_llama_20261009_183143_d59cba6e.json"
    if not path.exists():
        pytest.skip("job 125's report is only on the laptop")
    report = json.loads(path.read_text(encoding="utf-8"))
    plan = next(r for r in report["rounds"] if r.get("round") == "final check")["confirmation_plan"]
    assert explain.plan_format(plan) == "first"
    q = explain._plan_summary(plan)["questions"]["short"]
    assert q["power_target_attainable"] is False and q["predicted_power_at_cap"] == pytest.approx(0.707, abs=0.002)
