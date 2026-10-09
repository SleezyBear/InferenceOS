"""Explain a practical-mode tune report: the search order, where the time went, what was
verified, what is left unanswered and why the run ended (practical-mode hardware
evaluation, 8 Oct 2026).

Reads the JSON report ``tune`` writes for a practical (time-target) run and produces an
explanation (JSON) and a plain-language summary (Markdown). Optionally compares with a
historical Unbounded report, **only where the starting configuration, the questions
(workloads) and the scoring match**; anything else is reported as "not comparable" with
the reason, never as a retention figure.

Usage::

    python scripts/explain_practical_run.py REPORT.json [--historical OLD.json]
        [--json OUT.json] [--markdown OUT.md]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

#: Smallest worthwhile gain per effort mode (``optimization/keep.py`` defaults).
SMALLEST = {"balanced": 0.05, "performance_first": 0.015, "minimize_cost": 0.15}
#: A ratio of gains is reported only when both are at least this large (smaller gains
#: are compared as absolute differences only).
RATIO_FLOOR = 0.03
#: Labels of screening measurements, as ``tune/budgeted.py`` names them.
CATEGORY_OF = {"placement": "gpu_offload_layers", "threads": "threads", "batch": "ubatch_size"}


def _round(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next((r for r in report.get("rounds") or [] if isinstance(r, dict) and r.get("round") == name), {})


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.1%}"


def _config_diff(settings: dict[str, Any] | None, start: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (settings or {}).items():
        if value != start.get(key):
            out[key] = value
    return out


SLOT = {"placement": 0, "threads": 1, "batch": 2}


def _events(budget: dict[str, Any]) -> list[dict[str, Any]]:
    """The screening comparisons, as recorded when each happened (never re-derived)."""
    return [e for e in budget.get("screenings") or [] if isinstance(e, dict)]


def _setup_key(value: Any) -> tuple[int, ...] | None:
    return tuple(int(v) for v in value) if isinstance(value, (list, tuple)) else None


def _settings_of(setup: tuple[int, ...], start_setup: tuple[int, ...] | None) -> dict[str, int]:
    """The setup as the report's ``settings`` dicts write it: only what differs from the start."""
    if start_setup is None:
        return {}
    return {CATEGORY_OF[c]: setup[i] for c, i in SLOT.items() if setup[i] != start_setup[i]}


def _event_reading(e: dict[str, Any], smallest: float) -> dict[str, Any]:
    """One comparison's own reading (its gain against its own local reference)."""
    agg = e.get("aggregate") or {}
    out: dict[str, Any] = {
        "screening": e.get("n"), "kind": e.get("kind"), "at_s": e.get("at_s"),
        "category": e.get("category"), "value": e.get("value"),
        "gain": e.get("gain"), "noise": e.get("noise"), "class": _classify(e, smallest),
        "reference": (e.get("reference") or {}).get("settings"),
        "reference_measurement": (e.get("reference") or {}).get("measurement"),
        "candidate": (e.get("candidate") or {}).get("settings"),
        "vs_start_then": e.get("vs_start"),
    }
    if int(agg.get("count") or 1) > 1:
        out["aggregate_after"] = {"gain": agg.get("gain"), "noise": agg.get("noise"), "count": agg.get("count"),
                                  "class": _classify(agg, smallest)}
    return out


def explain(report: dict[str, Any], *, historical: dict[str, Any] | None = None) -> dict[str, Any]:
    search = _round(report, "practical search")
    budget = _round(report, "time budget")
    final = _round(report, "final check")
    effort = str(report.get("effort_mode") or "")
    smallest = SMALLEST.get(effort, 0.015)
    start = dict(report.get("starting_configuration") or {})
    start_setup = _setup_key(search.get("start_setup"))
    events = _events(budget)
    operations = [op for op in budget.get("operations") or [] if isinstance(op, dict)]
    record_notes: list[str] = []
    if operations and not events:
        record_notes.append("this report has no screening events (older format): readings are not attached "
                            "to measurements, and no strongest candidate is named")

    # 1. Every operation in order, with its cost and the comparison(s) it was the
    #    candidate side of, each with its own reading as recorded at that time.
    by_candidate_op: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        n = (e.get("candidate") or {}).get("measurement")
        if isinstance(n, int):
            by_candidate_op.setdefault(n, []).append(e)
    order: list[dict[str, Any]] = []
    for op in operations:
        item: dict[str, Any] = {k: op.get(k) for k in ("n", "kind", "label", "question", "began_s", "wall_s")}
        if op.get("kind") == "measure":
            item["changed"] = _config_diff(op.get("settings"), start)
            mine = [e for e in by_candidate_op.get(int(op.get("n") or 0), [])
                    if e.get("question") == op.get("question")]
            if mine:
                readings = [_event_reading(e, smallest) for e in mine]
                item["readings"] = readings
                item["reading"] = readings[0]
            phases = op.get("phases") or {}
            item["load_s"] = phases.get("load_s")
            item["monitor_s"] = phases.get("monitor_overhead_s")
        elif op.get("kind") == "load-try":
            item["result"] = op.get("result")
        order.append(item)

    # 2. The strongest candidate: a complete setup, judged by the planner's latest
    #    aggregate for it; when it was first seen, when its evidence was complete, and
    #    what happened to that exact setup.
    strongest = None
    by_setup: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for e in events:
        key = _setup_key((e.get("candidate") or {}).get("setup"))
        if key is not None:
            by_setup.setdefault(key, []).append(e)

    def latest_gain(evs: list[dict[str, Any]]) -> float | None:
        agg = evs[-1].get("aggregate") or {}
        g = agg.get("gain", evs[-1].get("gain"))
        return float(g) if isinstance(g, (int, float)) else None

    ranked = [(latest_gain(evs), key) for key, evs in by_setup.items() if latest_gain(evs) is not None]
    if ranked:
        gain, key = max(ranked, key=lambda x: x[0])  # type: ignore[arg-type, return-value]
        evs = by_setup[key]
        first, last = evs[0], evs[-1]
        latest = dict(last.get("aggregate") or {"gain": last.get("gain"), "noise": last.get("noise"), "count": 1})
        chosen_key = _setup_key(search.get("chosen_setup"))
        final_key = _setup_key(final.get("setup"))
        pursued = chosen_key is not None and chosen_key == key
        confirmed = final_key == key and confirmation_status(report)[0]
        deferred = any(_setup_key(d.get("setup")) == key for d in search.get("deferred") or [])
        settings = _settings_of(key, start_setup)
        # What became of this exact setup: matched by its complete setup (older reports
        # without one: by the settings that differ from the start).
        status = next((m for m in search.get("measured_setups") or []
                       if _setup_key(m.get("setup")) == key
                       or (m.get("setup") is None and start_setup is not None
                           and (m.get("settings") or {}) == settings)), None)
        chosen_settings = None if chosen_key is None else _settings_of(chosen_key, start_setup)
        same_change_there = [e for e in by_setup.get(chosen_key or (), [])
                             if e.get("category") == last.get("category") and e.get("value") == last.get("value")]
        strongest = {
            "setup": list(key), "settings": settings,
            "category": last.get("category"), "value": last.get("value"),
            "gain": latest.get("gain"), "noise": latest.get("noise"), "readings_averaged": latest.get("count"),
            "class": _classify(latest, smallest),
            "gain_is": "the setting's gain against its local reference in this exact setup (planner aggregate)",
            "first_seen_s": first.get("at_s"),
            "first_reading": _event_reading(first, smallest),
            "evidence_complete_s": last.get("at_s"),
            "evidence": [_event_reading(e, smallest) for e in evs],
            "vs_start": None if status is None else status.get("direct"),
            "vs_start_is": "this whole setup against the original start (direct readings at the end of exploration)",
            "pursued": pursued, "confirmed": confirmed, "deferred": deferred,
            "status": None if status is None else status.get("status"),
            "chosen_setup": None if chosen_key is None else list(chosen_key), "chosen_settings": chosen_settings,
        }
        strongest["why"] = _strongest_why(strongest, smallest, status, same_change_there)
        # 3. Did low-value exploration come first (everything that finished before the
        #    strongest setup's first reading and read below the smallest worthwhile gain)?
        seen_at = float(first.get("at_s") or 0.0)
        low = [o for o in order if o.get("kind") == "measure" and o.get("readings")
               and float(o["began_s"]) + float(o["wall_s"]) <= seen_at
               and int(o["n"]) != int((first.get("candidate") or {}).get("measurement") or -1)
               and all(r.get("gain") is None or float(r["gain"]) < smallest for r in o["readings"])]
        strongest["time_before_first_seen_s"] = first.get("at_s")
        strongest["low_value_before_s"] = round(sum(float(o["wall_s"]) for o in low), 1)
        strongest["low_value_before"] = [o["label"] for o in low]
        strongest["delayed_by_low_value"] = bool(low)

    # 4. Where the time went.
    costs = budget.get("costs") or {}
    by_kind = costs.get("seconds_by_kind") or {}
    phases = costs.get("phase_seconds") or {}
    used_s = budget.get("used_s")
    time_split = {
        "used_s": used_s, "target_s": budget.get("target_s"),
        "allowed_extension_s": budget.get("allowed_extension_s"), "extension_s": budget.get("extension_s"),
        "searching_s": round(sum(float(by_kind.get(k, 0.0)) for k in ("measure", "size-long", "failed")), 1),
        "load_only_tries_s": by_kind.get("load-try"),
        "confirming_s": by_kind.get("final"),
        "cleanup_s": by_kind.get("cleanup"),
        "of_which_loading_s": phases.get("load_s"),
        "of_which_monitoring_s": phases.get("monitor_overhead_s"),
        "of_which_answering_s": phases.get("request_s"),
        "discovery_ended_at_s": search.get("discovery_s"),
        "final_check_started_at_s": final.get("started_at_s"),
    }

    # 5. What the final check attempted, and what it verified. A gain is verified only
    #    when the complete required confirmation passed for that exact setup.
    confirmed, confirmation_why = confirmation_status(report)
    final_checks: list[dict[str, Any]] = []
    for check in final.get("checks") or []:
        fair = check.get("fair") or {}
        final_checks.append({
            "question": check.get("workload"), "must_be": check.get("must_be"), "passed": check.get("passed"),
            "score": fair.get("mean"), "lower": fair.get("lower_bound"), "upper": fair.get("upper_bound"),
            "blocks": len(fair.get("block_scores") or []), "decision": fair.get("decision"),
            "verified": confirmed,
            "label": "verified" if confirmed else f"unverified reading ({confirmation_why})",
        })
    verified = [c for c in final_checks if c["verified"]]

    # 6. Unanswered questions. A deferred weak reading whose exact setup then went
    #    through the final check and was saved is answered by that check (reviewer,
    #    9 Oct 2026); the search decision itself stays in the planner log and timeline.
    confirmed_setup = _setup_key(final.get("setup")) if confirmed else None
    deferred_readings = [d for d in search.get("deferred") or [] if isinstance(d, dict)]
    answered = [d for d in deferred_readings if confirmed_setup is not None and _setup_key(d.get("setup")) == confirmed_setup]
    answered_keys = {(str(d.get("category")), d.get("value")) for d in answered}
    still_open_keys = {(str(d.get("category")), d.get("value")) for d in deferred_readings if d not in answered}
    settled_words = [f"weak {c} {v} " for c, v in answered_keys - still_open_keys] + [
        f"repeat the weak {c} {v} " for c, v in answered_keys - still_open_keys]
    unanswered: list[str] = []
    answered_by_final: list[str] = []
    for line in search.get("plan") or []:
        if "left out" in line or "skipped" in line or "deferred" in line or "not repeated" in line:
            if any(str(line).startswith(w) or w in str(line) for w in settled_words):
                answered_by_final.append(f"{line} -- the exact setup was then confirmed by the final check")
            else:
                unanswered.append(line)
    for d in deferred_readings:
        text = (f"weak reading deferred: {d.get('category')} {d.get('value')} ({_pct(d.get('gain'))}, "
                f"noise {d.get('noise')})")
        if d in answered:
            answered_by_final.append(f"{text} -- its exact setup {d.get('setup')} passed the final check and was saved")
        else:
            unanswered.append(text)
    for s in search.get("measured_setups") or []:
        if str(s.get("status", "")).startswith(("missing evidence", "cannot reasonably finish")):
            unanswered.append(f"{s.get('settings')}: {s.get('status')}")
    long_measured = any(str(o.get("question", "")).startswith("long-") for o in order)
    if not long_measured:
        unanswered.append("the long question was never measured, so the prompt batch (and every change's "
                          "effect on long prompts) is unanswered")
        if any("prompt batch left out" in str(line) for line in search.get("plan") or []):
            unanswered.append("the prompt batch was deferred on a PREDICTED cost: its screening and final-check "
                              "times are estimates (the long question was never measured in this run), so this "
                              "run does not show that verifying it would not fit")

    outcome = str(report.get("outcome") or "")
    result = {
        "run_id": report.get("run_id"), "outcome": outcome,
        "starting_configuration": start, "starting_point_source": report.get("starting_point_source"),
        "final_configuration": report.get("final_configuration"),
        "order": order, "steps": budget.get("steps") or [], "planner_log": search.get("plan") or [],
        "screenings": events, "record_notes": record_notes,
        "strongest": strongest, "time": time_split, "verified": verified,
        "confirmation_plan": _plan_summary(final.get("confirmation_plan")),
        "final_checks": final_checks, "confirmation": {"verified": confirmed, "why": confirmation_why,
                                                       "setup": final.get("setup"), "settings": final.get("settings")},
        "unanswered": unanswered,
        "answered_by_final_check": answered_by_final,
        "extensions": budget.get("extension_reasons") or [],
        "why_ended": {
            "outcome": outcome,
            "exploration": search.get("stopped_for"),
            "final_check": final.get("abandoned") or final.get("interrupted") or (
                "completed" if final else "not started"),
            "notes": report.get("notes") or [],
        },
        "limiting_step": None if outcome == "saved" else limiting_step(report, search, final, budget),
        "long_question_measured": long_measured,
        "historical": None if historical is None else compare_historical(report, final_checks, historical),
    }
    return result


def _plan_summary(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    """The final check's confirmation plan, as recorded before it started (reports
    from before 9 Oct 2026 have none)."""
    if not isinstance(plan, dict) or not plan:
        return None
    questions = {}
    for name, q in (plan.get("questions") or {}).items():
        measured, predicted = q.get("measured") or {}, q.get("predicted") or {}
        attainable, attainable_source = _attainable(plan, q)
        at_cap, at_cap_source = _power_at_cap(plan, q)
        questions[name] = {
            "mode": q.get("mode"), "selection_blocks": q.get("selection_blocks"), "chosen_blocks": q.get("chosen_blocks"),
            "measured_direct_reading": measured.get("direct_reading_vs_start"),
            "measured_screening_noise_width": measured.get("screening_noise_width"),
            "load_to_load_source": (measured.get("noise_inputs") or {}).get("between_source"),
            "predicted_effect": predicted.get("effect"), "uncertainty_allowance": predicted.get("uncertainty_allowance"),
            "predicted_planning_spread": predicted.get("planning_spread"),
            "predicted_blocks_for_power": predicted.get("blocks_for_power"),
            "power_target_attainable": attainable,
            "power_target_attainable_source": attainable_source,
            "predicted_power_at_cap": at_cap,
            "predicted_power_at_cap_source": at_cap_source,
            "planned_power_limited_by": q.get("planned_power_limited_by"),
            "predicted_power_at_chosen": predicted.get("power_at_chosen"),
            "predicted_power_at_selection_count": predicted.get("power_at_selection_count"),
            "measured_side_costs_s": [measured.get("start_side_measurement_s"), measured.get("candidate_side_measurement_s")],
        }
    return {"questions": questions, "reason": plan.get("reason"), "planned_at_s": plan.get("planned_at_s"),
            "predicted_check_s": plan.get("predicted_check_s"), "measured_check_s": plan.get("measured_check_s"),
            "predicted_finish_s": plan.get("predicted_finish_s"), "available_to_target_s": plan.get("available_to_target_s"),
            "soft_target_s": plan.get("soft_target_s"), "status": plan.get("status"),
            "format": plan_format(plan) or "unrecognised"}


FIRST_PLAN_FORMAT = "first confirmation-plan format (9 Oct 2026, job 125)"
FIRST_PLAN_RULE = ("fixed block count chosen before the check: the count for the planned power under the planning "
                   "spread if the check finishes by the soft target, else the largest count that does, never fewer "
                   "than the selection count; one look, the existing test and alpha")
FIRST_PLAN_FORMAT_BLOCK_CAP = 16
"""The first format's block cap. Its code searched counts from the selection minimum up to
a fixed 16 and recorded ``blocks_for_power`` as the first count reaching the target, or 16
when none did; it recorded the plan's alpha and power target but not the cap."""
SECOND_PLAN_RULE = ("fixed block count chosen before the check: the count for the planned power under the planning "
                    "spread (the 16-block cap when no count up to it reaches that power) if the check finishes by "
                    "the soft target, else the largest count that does, never fewer than the selection count; one "
                    "look, the existing test and alpha")
#: Plan formats this script knows. Records from the reporting fix on store attainability,
#: the cap and the power at the cap themselves; only the first format needs them derived.
KNOWN_PLAN_FORMATS = {
    "first": FIRST_PLAN_FORMAT,
    "second": "reporting-fix format (9 Oct 2026; values recorded, no version field)",
    "confirmation-plan-3": "confirmation-plan-3 format (versioned; values recorded)",
}


def plan_format(plan: dict[str, Any]) -> str | None:
    """Which known format a recorded plan is (a ``KNOWN_PLAN_FORMATS`` key), from the whole
    plan: its version field when present, else its recorded rule text and the fields that
    format always or never wrote. ``None``: not a format this script knows."""
    version = plan.get("format_version")
    if version is not None:
        return version if version in KNOWN_PLAN_FORMATS and version not in ("first", "second") else None
    rule = plan.get("rule")
    if rule == SECOND_PLAN_RULE and "selection_limitation" in plan:
        return "second"
    if rule == FIRST_PLAN_RULE and "selection_limitation" not in plan:
        return "first"
    return None


def _is_count(value: Any, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def _first_format_problems(plan: dict[str, Any], question: dict[str, Any]) -> list[str]:
    """Ways one question of a plan contradicts the first format (empty: consistent)."""
    cap = FIRST_PLAN_FORMAT_BLOCK_CAP
    predicted = question.get("predicted") or {}
    problems = []
    if not _is_count(predicted.get("blocks_for_power"), 2, cap):
        problems.append(f"blocks_for_power {predicted.get('blocks_for_power')!r} is not a count from 2 to {cap}")
    selection, chosen = question.get("selection_blocks"), question.get("chosen_blocks")
    if not _is_count(selection, 2, cap):
        problems.append(f"selection_blocks {selection!r} is not a count from 2 to {cap}")
    if not (isinstance(chosen, int) and _is_count(chosen, 2, cap)) or (
            isinstance(selection, int) and _is_count(selection, 2, cap) and chosen < selection):
        problems.append(f"chosen_blocks {chosen!r} is not a count from the selection count to {cap}")
    for key in ("power_target_attainable", "block_cap", "power_at_cap"):
        if key in predicted:
            problems.append(f"it records {key}, which the first format never wrote")
    if "planned_power_limited_by" in question:
        problems.append("it records planned_power_limited_by, which the first format never wrote")
    alpha, target = plan.get("alpha_one_sided"), predicted.get("power_target")
    if not (isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and 0 < alpha < 0.5):
        problems.append(f"alpha_one_sided {alpha!r} is not a one-sided alpha")
    if not (isinstance(target, (int, float)) and not isinstance(target, bool) and 0 < target < 1):
        problems.append(f"power_target {target!r} is not a probability")
    return problems


def _plan_inputs(plan: dict[str, Any], question: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The inputs needed to rederive attainability and power at the cap for a record that
    did not store them, and where each came from. Only a record recognised as the first
    format, and consistent with it, is rederived (its cap is that format's 16); anything
    else gives ``None`` with the reason the result is unknown. Nothing is filled from
    today's constants."""
    predicted = question.get("predicted") or {}
    known = plan_format(plan)
    if known is None:
        return None, "unknown: the plan's format is not recognised"
    if known != "first":
        return None, f"unknown: this {KNOWN_PLAN_FORMATS[known]} record lacks the stored value"
    missing = [name for name, value in (("alpha_one_sided", plan.get("alpha_one_sided")),
                                        ("effect", predicted.get("effect")),
                                        ("planning_spread", predicted.get("planning_spread")),
                                        ("power_target", predicted.get("power_target")))
               if not isinstance(value, (int, float)) or isinstance(value, bool)]
    if missing:
        return None, "unknown: the record lacks " + ", ".join(missing)
    problems = _first_format_problems(plan, question)
    if problems:
        return None, "unknown: the record contradicts the first format (" + "; ".join(problems) + ")"
    cap = FIRST_PLAN_FORMAT_BLOCK_CAP
    inputs = {"alpha": float(plan["alpha_one_sided"]), "effect": float(predicted["effect"]),
              "spread": float(predicted["planning_spread"]), "target": float(predicted["power_target"]), "cap": cap}
    return inputs, (f"derived with the planner's power formula from the recorded alpha {inputs['alpha']}, effect, "
                    f"planning spread and power target {inputs['target']}; block cap {cap} from the {FIRST_PLAN_FORMAT}")


def _power(blocks: int, inputs: dict[str, Any]) -> float | None:
    try:
        from inferenceos.tune.budgeted import BudgetedTune
    except ImportError:  # pragma: no cover - the explanation still renders without it
        return None
    return float(BudgetedTune.planned_power(blocks, inputs["effect"], inputs["spread"], inputs["alpha"]))


def _attainable(plan: dict[str, Any], question: dict[str, Any]) -> tuple[bool | None, str]:
    """Whether the planned power was reachable within the block cap, and the source of
    that answer. Stored values (every format from the reporting fix on) are authoritative.
    For the first format, which stored the cap as ``blocks_for_power`` even when it did not
    reach the target, it is rederived from the record's own inputs and labelled derived."""
    predicted = question.get("predicted") or {}
    if "power_target_attainable" in predicted:
        return bool(predicted["power_target_attainable"]), "recorded"
    inputs, source = _plan_inputs(plan, question)
    if inputs is None:
        return None, source
    power = _power(inputs["cap"], inputs)
    return (None, "unknown: the planner is not importable") if power is None else (power >= inputs["target"], source)


def _power_at_cap(plan: dict[str, Any], question: dict[str, Any]) -> tuple[float | None, str]:
    """Planned power at the block cap and its source (recorded, derived for first-format
    records from their own inputs, or unknown)."""
    predicted = question.get("predicted") or {}
    if predicted.get("power_at_cap") is not None:
        return float(predicted["power_at_cap"]), "recorded"
    inputs, source = _plan_inputs(plan, question)
    if inputs is None:
        return None, source
    power = _power(inputs["cap"], inputs)
    return (None, "unknown: the planner is not importable") if power is None else (round(power, 3), source)


def _attainability_text(q: dict[str, Any]) -> str:
    """The plan line's attainability clause, saying when a value was derived or is unknown."""
    source = q.get("power_target_attainable_source") or "recorded"
    note = "" if source == "recorded" else f" [{source}]"
    if q.get("power_target_attainable") is True:
        blocks = q.get("predicted_blocks_for_power")
        return (f"{blocks} blocks for the planned power" if source == "recorded" else
                f"the planned power is attainable within the block cap (recorded count {blocks}){note}")
    if q.get("power_target_attainable") is False:
        at_cap = q.get("predicted_power_at_cap")
        cap_note = "" if (q.get("predicted_power_at_cap_source") or "recorded") == "recorded" else " (derived)"
        return ("the planned power is NOT attainable within the block cap"
                + (f" (at most {at_cap} at the cap{cap_note})" if at_cap is not None else "") + note)
    return (f"whether the planned power was attainable is unknown{note}; recorded blocks_for_power "
            f"{q.get('predicted_blocks_for_power')} (in older records this may be the cap)")


def _classify(r: dict[str, Any], smallest: float) -> str:
    gain, noise = r.get("gain"), float(r.get("noise") or 0.0)
    if not isinstance(gain, (int, float)):
        return "missing"
    if gain > 2 * noise and gain >= smallest:
        return "clear"
    if gain > 0:
        return "weak"
    if gain < -noise:
        return "worse"
    return "flat"


def _strongest_why(st: dict[str, Any], smallest: float, status: dict[str, Any] | None,
                   same_change_there: list[dict[str, Any]]) -> str:
    cls, gain = st["class"], _pct(st.get("gain"))
    what = f"{st.get('settings') or st.get('setup')} ({st.get('category')} {st.get('value')} read {gain}, {cls})"
    if st.get("confirmed"):
        return f"pursued and confirmed: this exact setup {what} passed the final check"
    if st.get("pursued"):
        return f"pursued: this exact setup {what} went to the final check"
    if st.get("chosen_setup") is not None:
        there = ""
        if same_change_there:
            e = same_change_there[-1]
            agg = e.get("aggregate") or {}
            there = (f"; in that setup {e.get('category')} {e.get('value')} read "
                     f"{_pct(agg.get('gain', e.get('gain')))}")
        return (f"not pursued: a different complete setup, {st['chosen_settings']}, went to the final check "
                f"(this one: {what}){there}")
    if st.get("deferred"):
        return (f"deferred: {what} was within the noise ({st.get('noise')}) or below the smallest "
                f"worthwhile gain ({smallest:.1%}), so the planner did not choose it")
    if status is not None:
        return f"not pursued: {status.get('status')}"
    return f"not pursued: {what}"


def confirmation_status(report: dict[str, Any]) -> tuple[bool, str]:
    """Whether this run's final check is a complete, passed confirmation of one setup."""
    final = _round(report, "final check")
    outcome = str(report.get("outcome") or "")
    if not final:
        return False, "no final check ran"
    for key, what in (("error", "stopped on an error"), ("interrupted", "interrupted"),
                      ("abandoned", "stopped incomplete")):
        if final.get(key):
            return False, f"the final check {what}: {final[key]}"
    checks = final.get("checks") or []
    if not checks:
        return False, "the final check has no results"
    failed = [str(c.get("workload")) for c in checks if not c.get("passed")]
    if failed:
        return False, "the final check did not pass on " + ", ".join(failed)
    if outcome != "saved":
        return False, f"the run did not save the confirmed setup (outcome: {outcome or 'missing'})"
    return True, "every required question of the final check passed and the setup was saved"


def limiting_step(report: dict[str, Any], search: dict[str, Any], final: dict[str, Any],
                  budget: dict[str, Any]) -> str:
    """The measured step that kept this run from a verified change."""
    outcome = str(report.get("outcome") or "")
    if outcome.startswith("cancelled"):
        return "the run was stopped by the user"
    if final.get("abandoned"):
        return f"the final check was too slow to finish within the extension: {final['abandoned']}"
    if final.get("interrupted"):
        return f"the final check was interrupted: {final['interrupted']}"
    if final:
        failed = [c for c in final.get("checks") or [] if not c.get("passed")]
        if failed:
            c = failed[0]
            fair = c.get("fair") or {}
            return (f"the final check did not show {c.get('must_be')} on {c.get('workload')}: decision "
                    f"{fair.get('decision')}, mean {_pct(fair.get('mean'))} (lower bound {_pct(fair.get('lower_bound'))}, "
                    f"{len(fair.get('block_scores') or [])} blocks)")
    stopped = str(search.get("stopped_for") or "")
    options = search.get("options") or []
    if search and search.get("chosen") is None and stopped.startswith("target"):
        return ("verification was predicted not to fit the target: exploration stopped early because " + stopped
                + "; no candidate whose predicted final check fits was left (a prediction from measured "
                "per-answer costs where they exist, otherwise an estimate)")
    if search and search.get("chosen") is None:
        if not options:
            return ("no candidate reading was clear enough to choose (every reading flat, worse, weak or "
                    "missing)" + (f"; exploration stopped: {stopped}" if stopped else ""))
        reasons = sorted({str(o.get("why") or o.get("status") or "") for o in options if not o.get("eligible")})
        unaffordable = [o for o in options if o.get("eligible")
                        and str(o.get("why") or "").startswith("cannot reasonably finish")]
        if unaffordable:
            return ("verification was predicted not to fit: the best eligible candidate's final check was predicted to take "
                    + ", ".join(f"{o.get('predicted_s')} s" for o in unaffordable[:3])
                    + f", more than the time left ({unaffordable[0].get('why')})")
        return "no candidate was eligible: " + "; ".join(r for r in reasons if r)
    if stopped:
        return f"exploration stopped: {stopped}"
    return outcome or "unknown"


# --------------------------------------------------------------------------- #
# Historical comparison (only where everything that defines the result matches)  #
# --------------------------------------------------------------------------- #
_FINAL_NOTE = re.compile(r"final check, (?P<wid>[\w.-]+): .*?score (?P<score>[+-]\d+(?:\.\d+)?)%"
                         r" \(95% range (?P<lo>[+-]\d+(?:\.\d+)?)% to (?P<hi>[+-]\d+(?:\.\d+)?)%, "
                         r"(?P<blocks>\d+) usable block[^)]*\)(?:.*?-> (?P<decision>[a-z ]+?)(?:;|$))?")


def historical_from_unbounded_report(report: dict[str, Any], *, label: str) -> dict[str, Any]:
    """What an Unbounded ``tune`` report says about its start, scoring and verified gains."""
    from datetime import datetime

    finals: dict[str, dict[str, Any]] = {}
    for note in report.get("notes") or []:
        m = _FINAL_NOTE.search(str(note))
        if m:
            finals[m.group("wid")] = {"score": float(m.group("score")) / 100, "lower": float(m.group("lo")) / 100,
                                      "upper": float(m.group("hi")) / 100, "blocks": int(m.group("blocks")),
                                      "decision": (m.group("decision") or "").strip() or None}
    minutes = None
    try:
        a = datetime.fromisoformat(str(report["started_at"]).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(report["finished_at"]).replace("Z", "+00:00"))
        minutes = round((b - a).total_seconds() / 60, 1)
    except (KeyError, ValueError):
        pass
    return {
        "label": label, "starting_configuration": report.get("starting_configuration") or {},
        "context_length": report.get("context_length"), "effort_mode": report.get("effort_mode"),
        "goal_mode": report.get("goal_mode"), "final_configuration": report.get("final_configuration"),
        "final_checks": finals, "total_minutes": minutes, "outcome": report.get("outcome"),
    }


def _historical_verified(historical: dict[str, Any], wid: str) -> tuple[bool, str]:
    """The same eligibility for the historical evidence: a better decision on that
    question in a run that saved its setup."""
    old = (historical.get("final_checks") or {}).get(wid) or {}
    outcome = str(historical.get("outcome") or "")
    if outcome != "saved":
        return False, f"the historical run did not save a confirmed setup (outcome: {outcome or 'missing'})"
    if old.get("decision") not in (None, "better"):
        return False, f"the historical final check on {wid} decided {old.get('decision')!r}, not better"
    if old.get("decision") is None and "decision" in old:
        return False, f"the historical final check on {wid} has no recorded decision"
    return True, ""


def _same_start(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Differences between two starting configurations (unset == engine default)."""
    keys = sorted(set(a) | set(b))
    return [f"{k}: {a.get(k)!r} vs {b.get(k)!r}" for k in keys if a.get(k) != b.get(k)]


def compare_historical(report: dict[str, Any], final_checks: list[dict[str, Any]],
                       historical: dict[str, Any]) -> dict[str, Any]:
    """Per question, a verified-gain comparison only when this run's complete required
    confirmation passed for its setup AND the historical question was verified the same
    way; start, context, effort and goal must match. Anything else is shown, labelled,
    without a ratio."""
    reasons = _same_start(dict(report.get("starting_configuration") or {}),
                          dict(historical.get("starting_configuration") or {}))
    for key in ("context_length", "effort_mode", "goal_mode"):
        if report.get(key) != historical.get(key):
            reasons.append(f"{key}: {report.get(key)!r} vs {historical.get(key)!r}")
    confirmed, confirmation_why = confirmation_status(report)
    per_question: list[dict[str, Any]] = []
    for v in final_checks:
        wid = str(v.get("question"))
        old = (historical.get("final_checks") or {}).get(wid)
        if reasons:
            per_question.append({"question": wid, "comparable": False, "why": "; ".join(reasons)})
            continue
        if old is None:
            per_question.append({"question": wid, "comparable": False,
                                 "why": f"the historical run has no final check on {wid} (question differs)"})
            continue
        if v.get("must_be") != "better":
            per_question.append({"question": wid, "comparable": False,
                                 "why": f"this run only guarded {wid} (not worse), so its score is not a found gain"})
            continue
        if not confirmed:
            per_question.append({"question": wid, "comparable": False, "verified": False,
                                 "this_run_unverified": v.get("score"),
                                 "why": f"this run's reading is unverified: {confirmation_why}"})
            continue
        ok, why = _historical_verified(historical, wid)
        if not ok:
            per_question.append({"question": wid, "comparable": False, "verified": True,
                                 "historical_unverified": old.get("score"), "why": why})
            continue
        new_score = v.get("score")
        item = {"question": wid, "comparable": True, "verified": True, "this_run": new_score,
                "historical": old["score"],
                "difference": None if new_score is None else round(float(new_score) - old["score"], 4)}
        if isinstance(new_score, (int, float)) and new_score >= RATIO_FLOOR and old["score"] >= RATIO_FLOOR:
            item["ratio"] = round(float(new_score) / old["score"], 3)
        per_question.append(item)
    budget = _round(report, "time budget")
    return {
        "label": historical.get("label"),
        "matched_start_and_scoring": not reasons,
        "differences": reasons,
        "per_question": per_question,
        "historical_final_checks": historical.get("final_checks"),
        "historical_final_configuration": historical.get("final_configuration"),
        "time": {"this_run_minutes": None if budget.get("used_s") is None else round(float(budget["used_s"]) / 60, 1),
                 "historical_minutes": historical.get("total_minutes")},
        "note": ("historical figures come from an earlier code version (Unbounded); they are compared only "
                 "where the start, question and scoring match, and no overall retention figure is claimed"),
    }


# --------------------------------------------------------------------------- #
def render_markdown(e: dict[str, Any]) -> str:
    out: list[str] = [f"# Practical-mode run {e.get('run_id')}", "", f"**Outcome:** {e.get('outcome')}", ""]
    t = e.get("time") or {}
    out += [f"**Time:** {t.get('used_s')} s used against a {t.get('target_s')} s target "
            f"(extension allowed {t.get('allowed_extension_s')} s, used {t.get('extension_s')} s).", ""]
    out += ["## Search order", "", "| # | Began (s) | Took (s) | What | Changed vs start | Reading |",
            "| --- | --- | --- | --- | --- | --- |"]
    for o in e.get("order") or []:
        parts = []
        for r in o.get("readings") or []:
            text = (f"{r.get('category')} {r.get('value')}: {_pct(r.get('gain'))} vs {r.get('reference') or 'start'} "
                    f"(noise {r.get('noise')}, {r.get('class')})")
            agg = r.get("aggregate_after")
            if agg:
                text += f"; average of {agg.get('count')} readings now {_pct(agg.get('gain'))} ({agg.get('class')})"
            parts.append(text)
        reading = "; ".join(parts)
        what = f"{o.get('label')}" + (f" [{o.get('question')}]" if o.get("question") else "")
        if o.get("result"):
            what += f": {o['result']}"
        out.append(f"| {o.get('n')} | {o.get('began_s')} | {o.get('wall_s')} | {what} | "
                   f"{o.get('changed') or ''} | {reading} |")
    out += ["", "## Planner reasons", ""] + [f"- {line}" for line in e.get("planner_log") or []] + [""]
    s = e.get("strongest")
    out += ["## Strongest candidate", ""]
    if s:
        fr = s.get("first_reading") or {}
        out += [f"- Setup {s.get('settings')}: {s['category']} {s['value']} reads {_pct(s.get('gain'))} against its "
                f"local reference ({s.get('class')}, {s.get('readings_averaged')} reading(s)).",
                f"- First seen at {s.get('first_seen_s')} s (that first reading: {_pct(fr.get('gain'))}, "
                f"{fr.get('class')}); evidence complete at {s.get('evidence_complete_s')} s.",
                f"- Whole setup against the original start: {s.get('vs_start')}.",
                f"- {s.get('why')}.",
                f"- Low-value exploration before it: {s.get('low_value_before_s')} s "
                f"({', '.join(s.get('low_value_before') or []) or 'none'})."]
    else:
        out.append("- No candidate reading was recorded.")
    out += [f"- Note: {n}" for n in e.get("record_notes") or []]
    out += ["", "## Where the time went", ""] + [f"- {k}: {v}" for k, v in t.items()] + [""]
    plan = e.get("confirmation_plan")
    if plan:
        out += ["## Confirmation plan (chosen before the check started)", "",
                f"- {plan.get('reason')}.",
                f"- Predicted check {plan.get('predicted_check_s')} s, finishing at {plan.get('predicted_finish_s')} s "
                f"against a {plan.get('soft_target_s')} s soft target; measured check {plan.get('measured_check_s')} s."]
        for name, q in (plan.get("questions") or {}).items():
            out.append(
                f"- {name} ({q.get('mode')}): {q.get('chosen_blocks')} blocks (selection count {q.get('selection_blocks')}). "
                f"Measured: direct reading {_pct(q.get('measured_direct_reading'))}, screening noise width "
                f"{q.get('measured_screening_noise_width')} (load-to-load {q.get('load_to_load_source')}), side costs "
                f"{q.get('measured_side_costs_s')} s. Predicted: effect {_pct(q.get('predicted_effect'))}, planning spread "
                f"{q.get('predicted_planning_spread')} (allowance {q.get('uncertainty_allowance')}), "
                + _attainability_text(q)
                + f", power {q.get('predicted_power_at_chosen')} at the chosen count "
                f"({q.get('predicted_power_at_selection_count')} at the selection count)"
                + (f"; limited by: {q.get('planned_power_limited_by')}" if q.get("planned_power_limited_by") else "")
                + ".")
        out.append("")
    conf = e.get("confirmation") or {}
    out += ["## Final check and what it verified", "",
            f"- {'Verified' if conf.get('verified') else 'Nothing verified'}: {conf.get('why')} "
            f"(setup {conf.get('settings')})."]
    for v in e.get("final_checks") or []:
        out.append(f"- {v.get('question')} ({v.get('must_be')}): {'passed' if v.get('passed') else 'not passed'}, "
                   f"score {_pct(v.get('score'))} (bounds {_pct(v.get('lower'))} to {_pct(v.get('upper'))}, "
                   f"{v.get('blocks')} blocks, {v.get('decision')}) - {v.get('label')}")
    out += ["", "## Unanswered", ""] + [f"- {u}" for u in e.get("unanswered") or ["(none)"]]
    if e.get("answered_by_final_check"):
        out += ["", "## Deferred during the search, answered by the final check", ""] + [
            f"- {u}" for u in e["answered_by_final_check"]]
    out += ["", "## Extensions", ""] + [f"- {x}" for x in e.get("extensions") or ["(none)"]]
    w = e.get("why_ended") or {}
    out += ["", "## Why it ended", "", f"- outcome: {w.get('outcome')}", f"- exploration: {w.get('exploration')}",
            f"- final check: {w.get('final_check')}"]
    if e.get("limiting_step"):
        out += ["", "## Limiting step (no verified change)", "", f"- {e['limiting_step']}"]
    h = e.get("historical")
    if h:
        out += ["", f"## Historical comparison ({h.get('label')})", "", f"- {h.get('note')}"]
        if not h.get("matched_start_and_scoring"):
            out.append("- Not comparable: " + "; ".join(h.get("differences") or []))
        for q in h.get("per_question") or []:
            if q.get("comparable"):
                out.append(f"- {q['question']}: this run {_pct(q.get('this_run'))}, historical "
                           f"{_pct(q.get('historical'))}" + (f", ratio {q['ratio']}" if "ratio" in q else ""))
            elif q.get("this_run_unverified") is not None:
                out.append(f"- {q['question']}: no gain comparison; this run's reading "
                           f"{_pct(q['this_run_unverified'])} is UNVERIFIED ({q.get('why')})")
            else:
                out.append(f"- {q['question']}: not comparable ({q.get('why')})")
        tm = h.get("time") or {}
        out.append(f"- time: this run {tm.get('this_run_minutes')} min, historical {tm.get('historical_minutes')} min")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--historical", default=None, help="an Unbounded tune report to compare with (if it matches)")
    parser.add_argument("--historical-label", default="historical Unbounded run")
    parser.add_argument("--json", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    historical = None
    if args.historical:
        historical = historical_from_unbounded_report(
            json.loads(Path(args.historical).read_text(encoding="utf-8")), label=args.historical_label)
    e = explain(report, historical=historical)
    if args.json:
        Path(args.json).write_text(json.dumps(e, indent=2, default=str), encoding="utf-8")
    text = render_markdown(e)
    if args.markdown:
        Path(args.markdown).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
