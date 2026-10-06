"""Fair comparison in independent blocks (tune.fair, tune.stats; rebuilt 3 Oct 2026).

User decisions: a change that does not really help is called better at most 5% of the
time; a "must not lose" check passes only when any loss is shown to be under 1%;
"unsure" fails; the goal uses fixed weights.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.goal import KeepVerdict
from inferenceos.optimization.keep import ModeRules
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_model
from inferenceos.tune.fair import (
    block_score,
    CLOCK_METRIC,
    TurnReading,
    compare_fairly,
    decide,
    guard_change,
    guard_episodes,
    planned_looks,
)
from inferenceos.tune.measure import DisturbanceWatch, judge_pair, measure_steady
from inferenceos.tune.stats import block_bounds, t_quantile
from tests.experiment_helpers import SampleResult, ScriptedHarness, build_test_policy

START = RuntimeConfiguration(context_length=2048, gpu_offload_layers=8)
NEW = START.model_copy(update={"threads": 5})


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "reports_dir": tmp_path / "reports",
        "logs_dir": tmp_path / "logs",
    }
    base.update(overrides)
    return Settings(**base)


def _watch(limit: float = 25.0, repeats: int = 0) -> DisturbanceWatch:
    return DisturbanceWatch(logical_cores=12, limit_percent=limit, max_repeats=repeats)


def _drifting(gain: float, *, drift_per_call: float = 0.004) -> Any:
    """Speed falls a little with every run (the laptop warms up); NEW is ``gain`` faster
    than START measured at the same moment."""

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        base = 20.0 * (1.0 - drift_per_call * call)
        speed = base * (1.0 + gain) if options.threads == 5 else base
        return SampleResult(decode_tps=speed, total_latency_s=2.0)

    return responder


def _compare(
    tmp_path: Path,
    responder: Any,
    *,
    effort: str = "balanced",
    mode: str = "better",
    harness: Any = None,
    min_blocks: int | None = None,
    max_blocks: int | None = None,
    deadline: float | None = None,
    on_step: Any = None,
) -> Any:
    settings = _settings(tmp_path, optimization_mode=effort)
    return compare_fairly(
        harness=harness or ScriptedHarness(responder),
        model=sample_model(),
        workload=get_workload("short-short"),
        start=START,
        candidate=NEW,
        policy=build_test_policy(),
        settings=settings,
        rules=ModeRules.from_settings(settings),
        watch=_watch(),
        card_id="runtime.reduce_threads",
        phase="test",
        mode=mode,  # type: ignore[arg-type]
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        deadline=deadline,
        on_step=on_step,
    )


# --------------------------------------------------------------------------- #
# The statistics                                                               #
# --------------------------------------------------------------------------- #
def test_t_quantiles_match_the_tables() -> None:
    assert abs(t_quantile(0.95, 2) - 2.920) < 1e-3
    assert abs(t_quantile(0.95, 4) - 2.132) < 1e-3
    assert abs(t_quantile(0.975, 7) - 2.365) < 1e-3


def _simulate(true_mean: float, noise: float, *, mode: str, runs: int, seed: int) -> float:
    """Share of simulated comparisons that pass, with the default look schedule."""
    rng = random.Random(seed)
    settings = Settings()
    looks = [int(settings.tune_fair_min_blocks), int(settings.tune_fair_max_blocks)]
    alpha = float(settings.tune_fair_alpha) / len(looks)
    passes = 0
    for _ in range(runs):
        scores: list[float] = []
        answer = None
        for number, look in enumerate(looks):
            while len(scores) < look:
                scores.append(rng.gauss(true_mean, noise))
            answer = decide(
                block_bounds(scores, alpha=alpha),
                mode=mode,
                margin=float(settings.tune_guard_margin),
                smallest_gain=0.015,
                final_look=number == len(looks) - 1,
            )
            if answer is not None:
                break
        if answer in ("better", "not worse"):
            passes += 1
    return passes / runs


def test_a_change_that_does_not_help_is_called_better_at_most_5_percent_of_the_time() -> None:
    for noise in (0.01, 0.03, 0.08):
        rate = _simulate(0.0, noise, mode="better", runs=4000, seed=11)
        assert rate <= 0.05, (noise, rate)


def test_a_change_that_loses_the_margin_passes_not_worse_at_most_5_percent_of_the_time() -> None:
    margin = float(Settings().tune_guard_margin)
    assert margin == 0.02  # user decision, 4 Oct 2026
    for noise in (0.01, 0.03):
        rate = _simulate(-margin, noise, mode="not_worse", runs=4000, seed=12)
        assert rate <= 0.05, (noise, rate)


def test_a_clear_gain_is_found_with_typical_noise() -> None:
    assert _simulate(0.06, 0.02, mode="better", runs=1000, seed=13) > 0.9


def test_unsure_at_the_last_look_is_reported_as_unsure() -> None:
    wide = block_bounds([0.05, -0.04, 0.06, -0.03], alpha=0.025)
    assert decide(wide, mode="better", margin=0.01, smallest_gain=0.015, final_look=False) is None
    assert decide(wide, mode="better", margin=0.01, smallest_gain=0.015, final_look=True) == "unsure"
    assert decide(wide, mode="not_worse", margin=0.01, smallest_gain=0.015, final_look=True) == "unsure"


def test_a_change_that_cannot_reach_the_smallest_gain_stops_early() -> None:
    tight = block_bounds([0.001, 0.002, 0.0, 0.001], alpha=0.025)
    assert decide(tight, mode="better", margin=0.01, smallest_gain=0.015, final_look=False) == "not better"


# --------------------------------------------------------------------------- #
# The comparison                                                               #
# --------------------------------------------------------------------------- #
def test_a_real_gain_under_drift_is_kept_where_a_single_early_reading_hides_it(
    tmp_path: Path,
) -> None:
    responder = _drifting(0.12)
    settings = _settings(tmp_path, optimization_mode="balanced")
    harness = ScriptedHarness(responder)
    policy = build_test_policy()
    early = measure_steady(
        harness=harness, model=sample_model(), workload=get_workload("short-short"),
        config=START, policy=policy, phase="early", watch=_watch(),
    )
    for _ in range(25):  # the search goes on for a while; the laptop drifts
        measure_steady(
            harness=harness, model=sample_model(), workload=get_workload("short-short"),
            config=START, policy=policy, phase="filler", watch=_watch(),
        )
    late = measure_steady(
        harness=harness, model=sample_model(), workload=get_workload("short-short"),
        config=NEW, policy=policy, phase="late", watch=_watch(),
    )
    old = judge_pair(
        before=early, after=late, workload=get_workload("short-short"), model=sample_model(),
        policy=policy, settings=settings, rules=ModeRules.from_settings(settings),
        card_id="runtime.reduce_threads",
    )
    assert not old.technically_better  # the stale reading hides the real gain

    fair = _compare(tmp_path, responder)
    assert fair.decision == "better" and fair.kept
    # Fixed weights (balanced): +12% writing, first word unchanged -> about +6%.
    assert fair.gain is not None and 0.05 <= fair.gain <= 0.07


def test_blocks_alternate_which_setup_goes_first(tmp_path: Path) -> None:
    fair = _compare(tmp_path, _drifting(0.12), min_blocks=4, max_blocks=4)
    assert [item.order for item in fair.turns] == [
        "start,candidate",
        "candidate,start",
        "start,candidate",
        "candidate,start",
    ]


def test_no_real_difference_is_not_kept(tmp_path: Path) -> None:
    fair = _compare(tmp_path, _drifting(0.0))
    assert not fair.kept
    assert fair.decision != "better"


def test_a_small_gain_follows_the_effort_mode(tmp_path: Path) -> None:
    # +6% writing (+3% on the balanced score) is real but below Balanced's 5% smallest
    # worthwhile gain; Performance first (1.5%) keeps it.
    balanced = _compare(tmp_path, _drifting(0.06), effort="balanced")
    assert balanced.decision == "better" and not balanced.kept
    assert balanced.judgement is not None
    assert balanced.judgement.verdict is KeepVerdict.NOT_WORTH_IT
    fast = _compare(tmp_path, _drifting(0.06), effort="performance_first")
    assert fast.kept


def test_a_guarded_measure_that_gets_much_worse_is_refused(tmp_path: Path) -> None:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if options.threads == 5:
            return SampleResult(decode_tps=24.0, total_latency_s=2.0, rss_bytes=1_800_000_000)
        return SampleResult(decode_tps=20.0, total_latency_s=2.0, rss_bytes=1_300_000_000)

    fair = _compare(tmp_path, responder)
    assert not fair.kept
    assert "peak_rss_bytes" in fair.guard_breaches
    guard = _compare(tmp_path, responder, mode="not_worse")
    assert not guard.not_worse  # a guarded measure counts in a "must not lose" check too


def test_a_quality_failure_is_never_kept(tmp_path: Path) -> None:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if options.threads == 5:
            return SampleResult(decode_tps=30.0, text="zzz qqq www completely different")
        return SampleResult(decode_tps=20.0)

    fair = _compare(tmp_path, responder)
    assert not fair.quality_passed
    assert not fair.kept
    assert not _compare(tmp_path, responder, mode="not_worse").not_worse


def test_not_worse_needs_proof_and_unsure_fails(tmp_path: Path) -> None:
    same = _compare(tmp_path, _drifting(0.0), mode="not_worse")
    assert same.decision == "not worse" and same.not_worse
    rng = random.Random(3)

    def noisy(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(decode_tps=20.0 * (1.0 + rng.uniform(-0.25, 0.25)), total_latency_s=2.0)

    unsure = _compare(tmp_path, noisy, mode="not_worse")
    assert unsure.decision in ("unsure", "worse") and not unsure.not_worse


def test_one_block_never_decides(tmp_path: Path) -> None:
    fair = _compare(tmp_path, _drifting(0.12), min_blocks=1, max_blocks=1)
    assert len(fair.turns) == 2


def test_every_measurement_starts_from_a_fresh_load(tmp_path: Path) -> None:
    class _FreshHarness(ScriptedHarness):
        def __init__(self, responder: Any) -> None:
            super().__init__(responder)
            self.fresh = 0

        def request_fresh_start(self) -> None:
            self.fresh += 1

    harness = _FreshHarness(_drifting(0.12))
    fair = _compare(tmp_path, None, harness=harness, min_blocks=4, max_blocks=4)
    assert harness.fresh == 2 * len(fair.turns)


def test_a_measurement_still_disturbed_after_its_repeats_is_not_used(tmp_path: Path) -> None:
    harness = ScriptedHarness(_drifting(0.0))
    # The scripted runs show 30% machine-wide use against 10% (120% / 12 cores) from
    # the model: 20% from other activity, over a 10% limit, every time.
    measurement = measure_steady(
        harness=harness, model=sample_model(), workload=get_workload("short-short"),
        config=START, policy=build_test_policy(), phase="busy", watch=_watch(limit=10.0, repeats=1),
    )
    assert measurement.valid_runs == 0
    assert any("not used" in note for note in measurement.notes)


def test_every_block_is_recorded_for_the_report(tmp_path: Path) -> None:
    fair = _compare(tmp_path, _drifting(0.12))
    record = fair.record()
    turns = record["turns"]
    assert isinstance(turns, list) and len(turns) >= 4
    first = turns[0]
    assert isinstance(first, dict)
    assert first["start"]["decode_tokens_per_sec"] is not None
    assert record["decision"] == "better" and record["lower_bound"] is not None
    assert isinstance(record["block_scores"], list)


# --------------------------------------------------------------------------- #
# Heat                                                                         #
# --------------------------------------------------------------------------- #
def _clock_compare(tmp_path: Path, throttled_new_block: int | None, gain: float) -> Any:
    """START and NEW in 4 blocks; NEW runs throttled (70% speed reading, 15% slower)
    in block ``throttled_new_block``."""
    state = {"new_measurements": 0, "hot": False}

    class _ClockHarness(ScriptedHarness):
        def run_once(self, task: Any, **kwargs: Any) -> Any:
            options = kwargs["options"]
            if options.threads == 5 and kwargs["warmup"]:
                state["new_measurements"] += 1
                state["hot"] = state["new_measurements"] == throttled_new_block
            run = super().run_once(task, **kwargs)
            hot = options.threads == 5 and state["hot"]
            metrics = run.observation.metrics.model_copy(
                update={"cpu_performance_percent": 70.0 if hot else 99.0}
            )
            return run.model_copy(
                update={"observation": run.observation.model_copy(update={"metrics": metrics})}
            )

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        speed = 20.0 * (1.0 + gain) if options.threads == 5 else 20.0
        if options.threads == 5 and state["hot"]:
            speed *= 0.85
        return SampleResult(decode_tps=speed, total_latency_s=2.0)

    return _compare(
        tmp_path, None, harness=_ClockHarness(responder), effort="performance_first",
        min_blocks=4, max_blocks=4,
    )


def test_a_cool_run_reports_no_heat_effect(tmp_path: Path) -> None:
    fair = _clock_compare(tmp_path, None, gain=0.10)
    assert fair.kept
    assert fair.heat_effect == ""


def test_no_block_is_left_out_for_processor_speed(tmp_path: Path) -> None:
    # User decision (4 Oct 2026): a real 10% gain, but the new setup runs 15% slower at
    # a low processor speed in block 2. That block counts (heat a setting causes itself
    # counts against it), so 4 blocks cannot show the gain; the speed change is
    # reported, never called overheating, and nothing is left out.
    fair = _clock_compare(tmp_path, 2, gain=0.10)
    assert len(fair.block_scores) == 4
    assert not fair.kept and fair.decision == "unsure"
    assert "every block was counted" in fair.heat_effect
    assert "does not show overheating" in fair.heat_effect
    assert "heat_changed_decision" not in fair.record()


def test_a_large_gain_survives_a_slow_block(tmp_path: Path) -> None:
    fair = _clock_compare(tmp_path, 2, gain=0.30)
    assert fair.kept and len(fair.block_scores) == 4
    assert fair.heat_effect


# --------------------------------------------------------------------------- #
# Planned looks (review 2, 4 Oct 2026)                                         #
# --------------------------------------------------------------------------- #
def test_looks_follow_the_effort_mode(tmp_path: Path) -> None:
    for effort, looks in (
        ("minimize_optimization_cost", [4]),
        ("balanced", [4, 8]),
        ("performance_first", [4, 16]),
    ):
        settings = _settings(tmp_path, optimization_mode=effort)
        assert planned_looks(settings, ModeRules.from_settings(settings)) == looks, effort
    capped = _settings(
        tmp_path, optimization_mode="performance_first", tune_fair_max_blocks_performance=40
    )
    assert planned_looks(capped, ModeRules.from_settings(capped)) == [4, 16]


def test_a_deadline_before_a_planned_look_gives_unsure(tmp_path: Path) -> None:
    """Review 2: two blocks and then the time limit used to be judged anyway (and
    kept). Only a planned look may decide."""
    import time as _time

    calls = {"n": 0}

    class _SlowHarness(ScriptedHarness):
        def run_once(self, task: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return super().run_once(task, **kwargs)

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(decode_tps=30.0 if options.threads == 5 else 20.0)

    harness = _SlowHarness(responder)
    deadline_holder: dict[str, float] = {}

    def on_step(step: str) -> None:
        if step.startswith("block 3"):
            deadline_holder["hit"] = 1.0

    real = _time.perf_counter
    try:
        _time.perf_counter = lambda: (1e12 if "hit" in deadline_holder else real())
        fair = _compare(tmp_path, None, harness=harness, deadline=1e11, on_step=on_step)
    finally:
        _time.perf_counter = real
    assert fair.decision == "unsure" and not fair.kept
    assert fair.looks == []
    assert "time limit reached" in fair.stopped_early


def test_the_first_look_needs_four_usable_blocks(tmp_path: Path) -> None:
    """Review 2: 4 attempts with 2 unusable blocks used to decide on 2 blocks. Now the
    look waits for 4 usable blocks, within ``tune_fair_extra_attempts`` extra tries."""
    state = {"start_measurements": 0}

    class _FlakyHarness(ScriptedHarness):
        def run_once(self, task: Any, **kwargs: Any) -> Any:
            options = kwargs["options"]
            if options.threads != 5 and kwargs["warmup"]:
                state["start_measurements"] += 1
            return super().run_once(task, **kwargs)

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        broken = options.threads != 5 and state["start_measurements"] in (2, 3)
        if broken:
            return SampleResult(fail=True)
        return SampleResult(decode_tps=30.0 if options.threads == 5 else 20.0)

    fair = _compare(
        tmp_path, None, harness=_FlakyHarness(responder), min_blocks=4, max_blocks=4
    )
    assert fair.looks == [4]
    assert len(fair.block_scores) == 4
    assert fair.attempts == 6
    assert fair.kept


def test_running_out_of_attempts_gives_unsure(tmp_path: Path) -> None:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if options.threads != 5:
            return SampleResult(fail=True)
        return SampleResult(decode_tps=30.0)

    fair = _compare(tmp_path, responder, min_blocks=4, max_blocks=4)
    assert fair.decision == "unsure" and not fair.kept
    assert fair.attempts == 4 + 2
    assert "planned look at 4 was not reached" in fair.stopped_early


def test_a_missing_goal_measure_makes_a_block_unusable(tmp_path: Path) -> None:
    """Review 2: under balanced, +8% writing with the first-word time missing used to
    score +8% (re-weighted) instead of +4%. A score with a missing measure is never
    used."""
    rules = ModeRules.from_settings(_settings(tmp_path))
    assert block_score({"decode_tokens_per_sec": 0.08}, rules) is None
    full = block_score({"decode_tokens_per_sec": 0.08, "ttft_s": 0.0}, rules)
    assert full is not None and abs(full - 0.04) < 1e-9


def test_paging_noise_near_zero_is_not_a_guard_breach() -> None:
    # Real readings (laptop, 4 Oct 2026): the 9B compared with itself on the long
    # question. Hard page faults per second wobble between 0 and 16; the old average
    # of per-block ratios called this "paging 54% worse".
    pages = [(0.99, 0.0), (0.99, 14.64), (0.0, 0.98), (15.89, 1.96), (13.69, 0.98),
             (0.98, 0.99), (0.98, 0.0), (0.0, 1.97)]
    blocks = [
        TurnReading(
            index,
            "start,candidate",
            {"hard_page_faults_per_sec": start},
            {"hard_page_faults_per_sec": new},
            {},
        )
        for index, (start, new) in enumerate(pages, 1)
    ]
    change = guard_change(blocks, "hard_page_faults_per_sec")
    assert change is not None and abs(change) < 0.10
    # One spike does not decide (9B, batch 256 re-check: 791/s once among 0-130/s).
    spike = [(21.5, 0.0), (130.2, 0.0), (8.4, 791.0), (51.1, 0.5)]
    spiky = [
        TurnReading(index, "start,candidate", {"hard_page_faults_per_sec": start},
                    {"hard_page_faults_per_sec": new}, {})
        for index, (start, new) in enumerate(spike, 1)
    ]
    spike_change = guard_change(spiky, "hard_page_faults_per_sec")
    assert spike_change is not None and spike_change > -0.10
    # Heavy paging still counts: 200 -> 400 per second in most blocks is two
    # tolerances (2 x 100/s) worse, a breach (limit: one tolerance).
    heavy = [TurnReading(1, "start,candidate", {"hard_page_faults_per_sec": 200.0},
                         {"hard_page_faults_per_sec": 400.0}, {})]
    assert guard_change(heavy, "hard_page_faults_per_sec") == -2.0
    # Memory has no floor: +40% memory is 40% worse.
    memory = [TurnReading(1, "start,candidate", {"peak_rss_bytes": 1.0e9},
                          {"peak_rss_bytes": 1.4e9}, {})]
    worse = guard_change(memory, "peak_rss_bytes")
    assert worse is not None and abs(worse + 0.4) < 1e-9


def test_repeated_severe_paging_in_the_new_setup_is_a_breach() -> None:
    """Review 2: 10,000 faults/s in 3 of 8 blocks used to vanish under the median."""
    pages = [(0.0, 10000.0)] * 3 + [(0.0, 0.0)] * 5
    blocks = [
        TurnReading(index, "start,candidate", {"hard_page_faults_per_sec": a},
                    {"hard_page_faults_per_sec": b}, {})
        for index, (a, b) in enumerate(pages, 1)
    ]
    against, words = guard_episodes(blocks, "hard_page_faults_per_sec")
    assert against and len(words) == 3
    # One spike is reported, not counted against the setup.
    one = blocks[:1] + [
        TurnReading(9, "x", {"hard_page_faults_per_sec": 0.0}, {"hard_page_faults_per_sec": 0.0}, {})
    ]
    against_one, words_one = guard_episodes(one, "hard_page_faults_per_sec")
    assert not against_one and len(words_one) == 1


def test_the_real_comparison_code_keeps_rarely_under_hard_conditions() -> None:
    """Review 2: simulate through ``compare_fairly`` itself (not a copy of its maths)
    with drift carried between measurements, heavy tails, failed measurements and time
    limits. A small run as a guard against regressions; the full run (2,000 each) is
    ``scripts/simulate_fair.py`` (4 Oct 2026: 1.3-5.4% in every cell)."""
    import importlib.util
    import sys as _sys
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parents[1] / "scripts" / "simulate_fair.py"
    spec = importlib.util.spec_from_file_location("simulate_fair", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    _sys.modules["simulate_fair"] = module
    spec.loader.exec_module(module)
    hard = [("all of the above", {"drift_rho": 0.7, "heavy_tails": True,
                                  "fail_share": 0.2, "cut_share": 0.5})]
    found = module.rates(150, 5, scenarios=hard, efforts=("performance_first",),
                         say=lambda _line: None)
    assert max(found.values()) <= 0.10, found
