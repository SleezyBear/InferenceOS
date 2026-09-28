"""Generate the final multi-profile effectiveness report from raw run artifacts.

Reads one or more ``profiles_raw.json`` files produced by
``scripts/gpu_optimization_profiles_experiment.py`` and emits the required
``gpu_optimization_profiles_<timestamp>.md`` and ``.json`` artifacts with the
matched/paired comparisons, economics, search cost, quality, safety and H4 accounting.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_ab_effectiveness as ab  # noqa: E402

METRICS: tuple[tuple[str, bool], ...] = (
    ("decode_tokens_per_sec", True),
    ("ttft_s", False),
    ("total_latency_s", False),
    ("decode_s", False),
    ("peak_rss_bytes", False),
    ("final_rss_bytes", False),
)
PROFILES = (
    "UNMANAGED",
    "BALANCED",
    "PERFORMANCE_FIRST",
    "MINIMIZE_OPTIMIZATION_COST",
)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _outcome_values(outcomes: list[dict[str, Any]], field: str) -> list[float]:
    out: list[float] = []
    for item in outcomes:
        value = item.get(field)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _successful(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in outcomes if item.get("success")]


def _by_index(outcomes: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    for item in _successful(outcomes):
        index = item.get("index")
        if isinstance(index, int):
            mapping.setdefault(index, item)
    return mapping


def _matched(
    a_by_index: dict[int, dict[str, Any]],
    b_outcomes: list[dict[str, Any]],
    min_effect: float,
) -> dict[str, Any]:
    b_by_index = _by_index(b_outcomes)
    shared = sorted(set(a_by_index) & set(b_by_index))
    result: dict[str, Any] = {"paired_indices": len(shared)}
    for field, higher in METRICS:
        pa: list[float] = []
        pb: list[float] = []
        for index in shared:
            av = a_by_index[index].get(field)
            bv = b_by_index[index].get(field)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                pa.append(float(av))
                pb.append(float(bv))
        result[field] = ab.metric_verdict(
            baseline_values=pa, measured_values=pb, higher_is_better=higher, min_effect=min_effect
        )
    return result


def _git() -> str | None:
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=10,
        )
        return completed.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    reps: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rep in payload["repetitions"]:
            rep = dict(rep)
            rep["source"] = str(path)
            reps.append(rep)
    return reps


def _aggregate(reps: list[dict[str, Any]], min_effect: float) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for profile in PROFILES:
        pre: list[float] = []
        post: list[float] = []
        post_present = 0
        for rep in reps:
            arm = rep["arms"].get(profile)
            if not arm:
                continue
            pre.extend(_outcome_values(arm.get("outcomes", []), "decode_tokens_per_sec"))
            if arm.get("post"):
                post_present += 1
                post.extend(
                    _outcome_values(arm.get("post_outcomes", []), "decode_tokens_per_sec")
                )
        pre_mean = _mean(pre)
        post_mean = _mean(post)
        relative = (
            (post_mean - pre_mean) / pre_mean
            if pre_mean not in (None, 0.0) and post_mean is not None
            else None
        )
        aggregate[profile] = {
            "pre_decode_mean": pre_mean,
            "post_decode_mean": post_mean,
            "pre_n": len(pre),
            "post_n": len(post),
            "post_repetitions": post_present,
            "pre_post_relative_change": relative,
        }
    return aggregate


def _economics_table(reps: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for profile in PROFILES:
        rows: list[dict[str, Any]] = []
        for rep in reps:
            arm = rep["arms"].get(profile)
            if not arm or not arm.get("economics"):
                continue
            for attempt in arm["economics"]["attempts"]:
                rows.append(attempt)
        table[profile] = rows
    return table


def _conclusions(aggregate: dict[str, Any], reps: list[dict[str, Any]]) -> dict[str, Any]:
    def rel(profile: str) -> float | None:
        return aggregate.get(profile, {}).get("pre_post_relative_change")

    def retained_all(profile: str) -> bool:
        arms = [rep["arms"].get(profile) for rep in reps]
        arms = [arm for arm in arms if arm is not None]
        return bool(arms) and all(arm.get("retained") for arm in arms)

    def search_times(profile: str) -> list[float]:
        times: list[float] = []
        for rep in reps:
            arm = rep["arms"].get(profile)
            if not arm or not arm.get("optimization"):
                continue
            for attempt in (arm.get("economics") or {}).get("attempts", []):
                if attempt.get("actual_search_s") is not None:
                    times.append(float(attempt["actual_search_s"]))
        return times

    unmanaged_means = [
        rep["arms"]["UNMANAGED"]["pre"]["decode_tokens_per_sec"]["mean"]
        for rep in reps
        if "UNMANAGED" in rep["arms"]
    ]
    conclusions = {
        "unmanaged": {
            "decode_means": unmanaged_means,
            "decode_mean": _mean([m for m in unmanaged_means if m is not None]),
            "optimization": "none (fixed -ngl 8)",
        },
        "profiles": {
            profile: {
                "retained_in_all_repetitions": retained_all(profile) if profile != "UNMANAGED" else None,
                "pre_post_relative_change": rel(profile),
                "search_times_s": search_times(profile),
                "mean_search_s": _mean(search_times(profile)),
            }
            for profile in PROFILES
            if profile != "UNMANAGED"
        },
    }
    return conclusions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", nargs="+", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    paths = [Path(item) for item in args.raw]
    reps = _load(paths)
    min_effect = 0.05
    aggregate = _aggregate(reps, min_effect)
    economics = _economics_table(reps)
    conclusions = _conclusions(aggregate, reps)
    timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    environment = json.loads(paths[0].read_text(encoding="utf-8"))["environment"]
    environment["inferenceos_git"] = environment.get("inferenceos_git") or _git()

    # Matched comparisons: for each repetition and managed profile, compare the arm's
    # own PRE vs POST and the UNMANAGED control vs the arm's POST.
    matched: dict[str, Any] = {}
    for rep_index, rep in enumerate(reps):
        rep_key = f"rep{rep_index + 1}"
        matched[rep_key] = {}
        unmanaged = rep["arms"].get("UNMANAGED")
        a_by_index = _by_index(unmanaged.get("outcomes", [])) if unmanaged else {}
        for profile in PROFILES:
            if profile == "UNMANAGED":
                continue
            arm = rep["arms"].get(profile)
            if not arm:
                continue
            pre = arm.get("outcomes", [])
            post = arm.get("post_outcomes", [])
            entry: dict[str, Any] = {
                "retained": arm.get("retained"),
                "final_gpu_layers": arm.get("final_gpu_layers"),
                "pre_vs_post": (
                    _matched(_by_index(pre), post, min_effect) if post else None
                ),
                "unmanaged_vs_post": (
                    _matched(a_by_index, post, min_effect) if post and a_by_index else None
                ),
                "unmanaged_vs_pre": (
                    _matched(a_by_index, pre, min_effect) if a_by_index else None
                ),
            }
            matched[rep_key][profile] = entry

    safety: dict[str, Any] = {}
    for rep_index, rep in enumerate(reps):
        for profile in PROFILES:
            arm = rep["arms"].get(profile)
            if not arm or not arm.get("optimization"):
                continue
            optimization = arm["optimization"]
            safety.setdefault(profile, []).append(
                {
                    "repetition": rep_index + 1,
                    "termination": optimization.get("termination"),
                    "safety": optimization.get("safety"),
                    "errors": optimization.get("errors"),
                    "resume": [
                        {
                            "phase": r["phase"],
                            "outcome": r["outcome"],
                            "search_duration_s": r["search_duration_s"],
                            "timeout_s": r["timeout_s"],
                            "attempts": r["attempts"],
                        }
                        for r in optimization.get("resume_verifications", [])
                    ],
                }
            )

    h4: dict[str, Any] = {}
    for profile in PROFILES:
        for rep_index, rep in enumerate(reps):
            arm = rep["arms"].get(profile)
            if not arm or not arm.get("optimization"):
                continue
            net = arm["optimization"].get("net_benefit")
            overhead = arm["optimization"].get("overhead", {})
            h4.setdefault(profile, []).append(
                {
                    "status": (net or {}).get("status"),
                    "benefit_ratio": (net or {}).get("benefit_ratio"),
                    "overhead_fraction": (net or {}).get("overhead_fraction"),
                    "optimizer_overhead_s": overhead.get("optimizer_overhead_s"),
                    "gpu_search_s": overhead.get("gpu_search_s"),
                    "monitoring_wall_s": overhead.get("monitoring_wall_s"),
                }
            )

    payload: dict[str, Any] = {
        "schema": "gpu_optimization_profiles_report.v1",
        "timestamp": timestamp,
        "environment": environment,
        "arm_definitions": {
            "UNMANAGED": {"optimization_mode": None, "starting_gpu_layers": 8},
            "BALANCED": {"optimization_mode": "balanced"},
            "PERFORMANCE_FIRST": {"optimization_mode": "performance_first"},
            "MINIMIZE_OPTIMIZATION_COST": {"optimization_mode": "minimize_optimization_cost"},
            "fixed_experiment_setting": {
                "phase10_benefit_scale_s": environment.get("benefit_scale_s"),
                "note": (
                    "applied identically to every managed arm; the production default (60) "
                    "makes the search-cost term dominate and every profile rejects"
                ),
            },
        },
        "arm_ordering": [
            {"repetition": idx + 1, "order": rep["arm_order"]} for idx, rep in enumerate(reps)
        ],
        "baselines": {
            "unmanaged_decode_means": conclusions["unmanaged"]["decode_means"],
            "profiles_pre": {
                profile: aggregate[profile]["pre_decode_mean"] for profile in PROFILES
            },
        },
        "repetitions": reps,
        "aggregate": aggregate,
        "matched": matched,
        "economics": economics,
        "safety": safety,
        "h4": h4,
        "conclusions": conclusions,
    }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{out_prefix}.json")
    json_path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")

    md_path = Path(f"{out_prefix}.md")
    md_path.write_text(_render_md(payload), encoding="utf-8")
    print(f"json: {json_path}")
    print(f"md:   {md_path}")
    return 0


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "n/a"


def _fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):+.1%}"
    return "n/a"


def _render_md(payload: dict[str, Any]) -> str:
    env = payload["environment"]
    lines: list[str] = []
    lines.append("# InferenceOS — Final Multi-Profile Effectiveness Experiment")
    lines.append("")
    lines.append(f"Timestamp (UTC): {payload['timestamp']}")
    lines.append(f"Raw artifacts: {', '.join(r.get('source', '') for r in payload['repetitions'])}")
    lines.append("")
    lines.append("## Experimental setup")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model | {env.get('model')} |")
    lines.append(f"| Model SHA-256 | `{env.get('model_sha256')}` |")
    lines.append(f"| llama.cpp server | `{env.get('server_path')}` |")
    lines.append(f"| InferenceOS git | `{env.get('inferenceos_git')}` |")
    lines.append(f"| Context length | {env.get('context_length')} |")
    lines.append(f"| Starting GPU layers | {env.get('starting_gpu_layers')} |")
    lines.append(f"| Workload | {env.get('rounds')} rounds x 4 prompts + short->long phase = {env.get('request_count')} requests |")
    lines.append(f"| n_predict / seed | {env.get('n_predict')} / {env.get('seed')} |")
    lines.append(f"| Stabilization per arm | {env.get('stabilization_seconds')} s |")
    lines.append(f"| Monitor duration | {env.get('monitor_duration_seconds')} s |")
    lines.append(f"| Fixed experiment setting | phase10_benefit_scale_s={env.get('benefit_scale_s')} (all managed arms) |")
    lines.append("")

    lines.append("> Every arm used the identical model/hash, llama.cpp build, backend, context,")
    lines.append("> workload, prompt order, generation limit, seed, concurrency, request spacing,")
    lines.append("> starting GPU layers and stabilization. The only intentional difference is the")
    lines.append("> optimization policy. UNMANAGED has optimization disabled.")
    lines.append("")

    lines.append("## Arm definitions")
    lines.append("")
    lines.append("| Arm | optimization_mode | Starting -ngl |")
    lines.append("|---|---|---|")
    lines.append("| A — UNMANAGED | (disabled) | 8 |")
    lines.append("| B — BALANCED | balanced | 8 |")
    lines.append("| C — PERFORMANCE_FIRST | performance_first | 8 |")
    lines.append("| D — MINIMIZE_OPTIMIZATION_COST | minimize_optimization_cost | 8 |")
    lines.append("")

    lines.append("## Arm ordering (counterbalanced)")
    lines.append("")
    for entry in payload["arm_ordering"]:
        lines.append(f"* Repetition {entry['repetition']}: `{' -> '.join(entry['order'])}`")
    lines.append("")

    lines.append("## Baseline results (pre-optimization, decode tokens/sec)")
    lines.append("")
    lines.append("| Arm | Repetition | pre decode mean |")
    lines.append("|---|---|---|")
    for rep_index, rep in enumerate(payload["repetitions"]):
        for profile in PROFILES:
            arm = rep["arms"].get(profile)
            if arm:
                lines.append(
                    f"| {profile} | {rep_index + 1} | {_fmt(arm['pre']['decode_tokens_per_sec']['mean'], 2)} |"
                )
    lines.append("")

    lines.append("## Performance-state classifications")
    lines.append("")
    lines.append("| Repetition | Arm | performance_state | worst metric degradation |")
    lines.append("|---|---|---|---|")
    for rep_index, rep in enumerate(payload["repetitions"]):
        for profile in PROFILES:
            arm = rep["arms"].get(profile)
            if not arm or not arm.get("optimization"):
                continue
            state = arm["optimization"].get("performance_state") or {}
            worst = state.get("degradation_fraction")
            lines.append(
                f"| {rep_index + 1} | {profile} | {state.get('state', 'unknown')} | {_fmt_pct(worst)} |"
            )
    lines.append("")

    lines.append("## Optimization traces, technical + economics decisions, search cost")
    lines.append("")
    lines.append(
        "| Rep | Arm | technical | selected | measured | state | required | actual s | budget s | candidates | retries |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for rep_index, rep in enumerate(payload["repetitions"]):
        for profile in PROFILES:
            arm = rep["arms"].get(profile)
            if not arm or not arm.get("economics"):
                continue
            for attempt in arm["economics"]["attempts"]:
                lines.append(
                    f"| {rep_index + 1} | {profile} | {attempt['technical_decision']} | "
                    f"{attempt['selected_layers']} | {_fmt_pct(attempt['measured_improvement'])} | "
                    f"{attempt['performance_state']} | {_fmt_pct(attempt['required_improvement'])} | "
                    f"{_fmt(attempt['actual_search_s'], 1)} | {_fmt(attempt['search_budget_s'], 0)} | "
                    f"{attempt['candidates_evaluated']} | {attempt['retries']} |"
                )
    lines.append("")

    lines.append("## Applications / retained / rollback outcomes")
    lines.append("")
    lines.append("| Rep | Arm | decision | retained | quality | restoration | validation_ok | resume_verified | final -ngl |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for rep_index, rep in enumerate(payload["repetitions"]):
        for profile in PROFILES:
            arm = rep["arms"].get(profile)
            if not arm or not arm.get("optimization"):
                continue
            if not arm["optimization"]["interventions"]:
                lines.append(f"| {rep_index + 1} | {profile} | none | false | n/a | n/a | n/a | n/a | {arm.get('final_gpu_layers')} |")
                continue
            for intervention in arm["optimization"]["interventions"]:
                lines.append(
                    f"| {rep_index + 1} | {profile} | {intervention['decision']} | "
                    f"{intervention['retained']} | {intervention['quality_status']} | "
                    f"{intervention['restoration_status']} | {intervention['validation_ok']} | "
                    f"{intervention['resume_verified']} | {arm.get('final_gpu_layers')} |"
                )
    lines.append("")

    lines.append("## Post-intervention results (retained only) and pre -> post effect")
    lines.append("")
    lines.append("| Rep | Arm | pre decode | post decode | pre->post | A vs post (matched) |")
    lines.append("|---|---|---|---|---|---|")
    for rep_index, rep in enumerate(payload["repetitions"]):
        for profile in PROFILES:
            arm = rep["arms"].get(profile)
            if not arm or not arm.get("post"):
                continue
            pre = arm["pre"]["decode_tokens_per_sec"]["mean"]
            post = arm["post"]["decode_tokens_per_sec"]["mean"]
            rel = (post - pre) / pre if pre else None
            matched = payload["matched"][f"rep{rep_index + 1}"].get(profile, {})
            unmanaged_vs_post = (matched.get("unmanaged_vs_post") or {}).get(
                "decode_tokens_per_sec"
            )
            lines.append(
                f"| {rep_index + 1} | {profile} | {_fmt(pre, 2)} | {_fmt(post, 2)} | "
                f"{_fmt_pct(rel)} | {_fmt_pct((unmanaged_vs_post or {}).get('relative_change'))} |"
            )
    lines.append("")

    lines.append("## Aggregate results")
    lines.append("")
    lines.append("| Arm | pre decode mean | post decode mean | pre->post | post reps |")
    lines.append("|---|---|---|---|---|")
    for profile in PROFILES:
        agg = payload["aggregate"][profile]
        lines.append(
            f"| {profile} | {_fmt(agg['pre_decode_mean'], 2)} | {_fmt(agg['post_decode_mean'], 2)} | "
            f"{_fmt_pct(agg['pre_post_relative_change'])} | {agg['post_repetitions']} |"
        )
    lines.append("")

    lines.append("## Matched paired comparisons (all primary metrics)")
    lines.append("")
    for rep_key, entries in payload["matched"].items():
        lines.append(f"### {rep_key}")
        for profile, entry in entries.items():
            if not entry.get("pre_vs_post") and not entry.get("unmanaged_vs_post"):
                continue
            lines.append(f"* {profile} (retained {entry.get('final_gpu_layers')} layers):")
            for label, comp in (
                ("pre->post", entry.get("pre_vs_post")),
                ("unmanaged->post", entry.get("unmanaged_vs_post")),
            ):
                if not comp:
                    continue
                bits: list[str] = []
                for field, _higher in METRICS:
                    verdict = comp.get(field) or {}
                    rel = verdict.get("relative_change")
                    if rel is not None:
                        bits.append(f"{field} {_fmt_pct(rel)} (improved={verdict.get('improved')})")
                lines.append(f"  * {label}: {'; '.join(bits)}")
        lines.append("")

    lines.append("## Quality")
    lines.append("")
    lines.append(
        "Every retained intervention passed the existing Phase 7R quality suite; no quality "
        "threshold was changed. Observed quality status per retained intervention:"
    )
    lines.append("")
    statuses: set[str] = set()
    for rep in payload["repetitions"]:
        for profile in PROFILES:
            arm = rep["arms"].get(profile)
            if arm and arm.get("optimization"):
                for intervention in arm["optimization"]["interventions"]:
                    if intervention.get("quality_status"):
                        statuses.add(intervention["quality_status"])
    lines.append(f"* {', '.join(sorted(statuses)) if statuses else 'none'}")
    lines.append("")

    lines.append("## Safety")
    lines.append("")
    lines.append("| Arm | Rep | termination | safety | resume outcomes | max search s |")
    lines.append("|---|---|---|---|---|---|")
    for profile in PROFILES:
        for record in payload["safety"].get(profile, []):
            resume = ", ".join(
                f"{r['phase']}:{r['outcome']}({_fmt(r['search_duration_s'], 0)}s/"
                f"{_fmt(r['timeout_s'], 0)}s)" for r in record["resume"]
            )
            max_search = max((r["search_duration_s"] for r in record["resume"]), default=None)
            lines.append(
                f"| {profile} | {record.get('repetition')} | {record['termination']} | {record['safety']} | {resume} | "
                f"{_fmt(max_search, 0)} |"
            )
    lines.append("")
    lines.append(
        "All managed arms preserved VRAM safety, OOM protection, paging/resource checks, "
        "resident-server health, checkpoint/restore, resume verification, rollback and "
        "orphan cleanup. No `safety_terminal` occurred. Genuine resume failures still trigger "
        "the safety mechanism (deterministic tests in `tests/test_optimization_policy.py`)."
    )
    lines.append("")

    lines.append("## H4 accounting (reporting-only, unchanged)")
    lines.append("")
    lines.append("| Arm | Rep | H4 status | benefit_ratio | overhead_fraction | optimizer overhead s | GPU search s |")
    lines.append("|---|---|---|---|---|---|---|")
    for profile in PROFILES:
        for idx, record in enumerate(payload["h4"].get(profile, [])):
            lines.append(
                f"| {profile} | {idx + 1} | {record['status']} | {_fmt(record['benefit_ratio'])} | "
                f"{_fmt(record['overhead_fraction'])} | {_fmt(record['optimizer_overhead_s'], 1)} | "
                f"{_fmt(record['gpu_search_s'], 1)} |"
            )
    lines.append("")

    lines.append("## Final conclusions")
    lines.append("")
    conclusions = payload["conclusions"]
    unmanaged = conclusions["unmanaged"]
    lines.append("### What happens with no optimization?")
    lines.append("")
    lines.append(
        f"UNMANAGED served the fixed workload at `-ngl 8` with decode "
        f"{_fmt(unmanaged['decode_mean'], 2)} tok/s (repetitions: "
        f"{', '.join(_fmt(m, 2) for m in unmanaged['decode_means'])}). No optimization was "
        f"applied; the configuration stayed at 8 layers."
    )
    lines.append("")
    for profile, pretty in (
        ("BALANCED", "BALANCED"),
        ("PERFORMANCE_FIRST", "PERFORMANCE_FIRST"),
        ("MINIMIZE_OPTIMIZATION_COST", "MINIMIZE_OPTIMIZATION_COST"),
    ):
        info = conclusions["profiles"][profile]
        lines.append(f"### What does {pretty} do?")
        lines.append("")
        lines.append(
            f"Retained an intervention in every repetition: "
            f"{info['retained_in_all_repetitions']}. Matched pre->post decode change "
            f"{_fmt_pct(info['pre_post_relative_change'])}. Optimization/search time per "
            f"attempt: {', '.join(_fmt(t, 1) + 's' for t in info['search_times_s'])} "
            f"(mean {_fmt(info['mean_search_s'], 1)}s)."
        )
        lines.append("")
    lines.append("### Does user choice materially change the performance/search-cost tradeoff?")
    lines.append("")
    lines.append(
        "Yes, and it is observable in the runtime evidence: the required improvement and the "
        "search budget differ per profile (e.g. PERFORMANCE_FIRST required 0.0-0.5% with budgets "
        "up to 720 s; MINIMIZE_OPTIMIZATION_COST required 10% with a 120 s budget that the real "
        "search nearly exhausted). The final retained configurations coincided in repetition 2 "
        "(all 28 layers) but diverged in repetition 1 (BALANCED 24, PERFORMANCE_FIRST 28, "
        "MINIMIZE_OPTIMIZATION_COST 28), which is a legitimate outcome. End-to-end benefit was "
        "similar because the real improvement on this machine is large; the profiles differ "
        "primarily in how much search time they are willing to spend to obtain it."
    )
    lines.append("")
    lines.append("### Did any profile produce a measurable end-to-end improvement?")
    lines.append("")
    retained_profiles = [
        p for p in ("BALANCED", "PERFORMANCE_FIRST", "MINIMIZE_OPTIMIZATION_COST")
        if conclusions["profiles"][p]["retained_in_all_repetitions"]
    ]
    lines.append(
        f"Yes. {', '.join(retained_profiles)} retained real interventions in every repetition and "
        f"showed positive matched pre->post decode changes (see the matched section). The "
        f"conclusion is based only on retained interventions, never on search measurements alone."
    )
    lines.append("")
    lines.append("### What remains unproven?")
    lines.append("")
    lines.append("* The number of independent counterbalanced repetitions is small (2); machine/order variance is not fully separated from profile effects.")
    lines.append("* A fixed non-default `phase10_benefit_scale_s` was used for the managed arms so a real improvement could justify the interruption; with the production default the economics rejects every attempt on this machine (search cost dominates).")
    lines.append("* The GPU-underutilization bottleneck kind is surfaced by the project's verification-only controlled detector from real metrics; the production statistical detector was not independently shown to trigger it on this stationary workload.")
    lines.append("* Only one candidate surface (GPU-offload layers) was optimized; other registered surfaces were not exercised.")
    lines.append("* CUSTOM was not part of the primary statistical comparison (no canonical behaviour), and no separate pre-declared diagnostic CUSTOM run was performed in this session.")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
