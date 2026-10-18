"""Review 6 (6 Oct 2026): four acceptance defects found by the adversarial review.

1. A memory-safety correction must show it runs and answers correctly.
2. A missing required (guarded) measure makes any acceptance route inconclusive.
3. Failed runs of the new setup stay in the evidence; later good blocks cannot hide them.
4. A failed speed check never restores a starting setup already found unsafe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.optimization.keep import ModeRules
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_model
from inferenceos.tune.fair import compare_fairly
from inferenceos.tune.layer_top import LoadAttempt
from inferenceos.tune.measure import DisturbanceWatch
from inferenceos.tune.pipeline import Tuner
from inferenceos.tune.paths import OptionPath
from tests.experiment_helpers import SampleResult, ScriptedHarness, build_test_policy
from tests.test_tune import CONTEXT, _settings, _tuner

START = RuntimeConfiguration(context_length=2048, gpu_offload_layers=8)
NEW = START.model_copy(update={"threads": 5})
RIGHT = "A key value cache stores attention state for 2468 tokens."
WRONG = "Bananas are yellow and the moon is far away, 9999."


def _fair(tmp_path: Path, responder: Any, *, mode: str = "better") -> Any:
    settings = _settings(tmp_path, optimization_mode="balanced")
    return compare_fairly(
        harness=ScriptedHarness(responder),
        model=sample_model(),
        workload=get_workload("short-short"),
        start=START,
        candidate=NEW,
        policy=build_test_policy(),
        settings=settings,
        rules=ModeRules.from_settings(settings),
        watch=DisturbanceWatch(logical_cores=12, limit_percent=25.0, max_repeats=0),
        card_id="runtime.reduce_threads",
        phase="test",
        mode=mode,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# 2. Missing guarded measures                                                  #
# --------------------------------------------------------------------------- #
def _no_paging(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    faster = options.threads == 5
    return SampleResult(
        decode_tps=24.0 if faster else 20.0,
        total_latency_s=1.7 if faster else 2.0,
        hard_faults_per_sec=None,
        text=RIGHT,
    )


def test_a_missing_guarded_measure_never_passes_better(tmp_path: Path) -> None:
    fair = _fair(tmp_path, _no_paging)
    assert "hard_page_faults_per_sec" in fair.unavailable_guards
    assert fair.decision == "unsure"
    assert not fair.kept
    assert fair.judgement is not None and not fair.judgement.keep
    assert any("not measured in every counted block" in item for item in fair.acceptance_blockers())


def test_a_missing_guarded_measure_never_passes_not_worse(tmp_path: Path) -> None:
    fair = _fair(tmp_path, _no_paging, mode="not_worse")
    assert fair.decision == "unsure"
    assert not fair.not_worse


def test_a_guard_read_in_only_some_blocks_is_not_enough(tmp_path: Path) -> None:
    def patchy(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        sample = _no_paging(task, options, call)
        # Paging is read only in the first few calls (the first block), then never.
        sample.hard_faults_per_sec = 0.5 if call <= 6 else None
        return sample

    fair = _fair(tmp_path, patchy)
    covered = fair.guard_coverage["hard_page_faults_per_sec"]
    assert 0 < covered < len(fair.turns)
    assert "hard_page_faults_per_sec" in fair.unavailable_guards
    assert not fair.kept


def test_a_tune_without_paging_readings_saves_no_speed_change(tmp_path: Path) -> None:
    tuner, _, _, store = _tuner(tmp_path, _no_paging, answers=[True, True, True])
    result = tuner.run()
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None
    assert saved.configuration.threads is None
    assert result.outcome in ("no change", "not confirmed")


# --------------------------------------------------------------------------- #
# 3. Failed runs of the new setup                                              #
# --------------------------------------------------------------------------- #
def test_a_new_setup_that_crashed_early_is_not_kept_after_later_good_blocks(tmp_path: Path) -> None:
    seen = {"candidate": 0}

    def crashes_then_fast(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if options.threads == 5:
            seen["candidate"] += 1
            if seen["candidate"] <= 6:
                return SampleResult(fail=True)
            return SampleResult(decode_tps=26.0, total_latency_s=1.6, text=RIGHT)
        return SampleResult(decode_tps=20.0, total_latency_s=2.0, text=RIGHT)

    fair = _fair(tmp_path, crashes_then_fast)
    assert fair.candidate_failures > 0
    assert fair.start_failures == 0
    assert fair.reliability == "candidate failed to run"
    assert fair.discarded_blocks, "the failed block is kept as evidence"
    assert not fair.kept
    assert not fair.not_worse
    assert fair.decision == "unsure"


def test_failures_of_both_setups_are_inconclusive(tmp_path: Path) -> None:
    def both_fail_once(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if call in (2, 7):
            return SampleResult(fail=True)
        faster = options.threads == 5
        return SampleResult(decode_tps=24.0 if faster else 20.0, total_latency_s=2.0, text=RIGHT)

    fair = _fair(tmp_path, both_fail_once)
    if fair.candidate_failures and fair.start_failures:
        assert fair.reliability == "both failed to run"
        assert not fair.kept


def test_the_answer_check_counts_failed_runs_as_failed_execution() -> None:
    from inferenceos.config import Settings
    from inferenceos.experiment.quality_policy import build_adaptive_quality_policy
    from inferenceos.experiment.quality_suite import evaluate_suite, build_default_suite

    workload = get_workload("short-short")
    result = evaluate_suite(
        build_default_suite(workload, sample_model()),
        policy=build_adaptive_quality_policy(Settings()),
        baseline_texts=[RIGHT, RIGHT],
        candidate_texts=[RIGHT, RIGHT],
        baseline_tokens=[32, 32],
        candidate_tokens=[32, 32],
        baseline_valid_runs=2,
        candidate_valid_runs=2,
        invalid_runs=3,
        ignore_eos=False,
        n_predict=32,
    )
    execution = next(case for case in result.cases if case.case_id == "execution_completed")
    assert not execution.passed


# --------------------------------------------------------------------------- #
# 1 and 4. Memory-safety corrections                                           #
# --------------------------------------------------------------------------- #
def _monotonic(text_at_safe: str) -> Any:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        layers = options.gpu_offload_layers or 0
        return SampleResult(
            decode_tps=10.0 + layers,
            total_latency_s=max(2.0 - 0.01 * layers, 0.5),
            text=text_at_safe if layers <= 10 else RIGHT,
        )

    return responder


def _saved_high(tmp_path: Path) -> int:
    """A first tune saves a layer count above 10 (later found unsafe)."""
    first, _, _, store = _tuner(tmp_path, _monotonic(RIGHT), answers=[True, True, True])
    first.run()
    saved = store.find(first.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None
    layers = int(saved.configuration.gpu_offload_layers or 0)
    assert layers > 10
    return layers


def test_a_safety_correction_that_answers_wrongly_is_not_saved(tmp_path: Path) -> None:
    high = _saved_high(tmp_path)
    second, _, _, store = _tuner(tmp_path, _monotonic(WRONG), answers=[True, True, True])
    second.try_load = lambda layers: LoadAttempt(loaded=True, fits=layers <= 10)
    result = second.run()
    assert result.outcome == "not saved: memory-safety correction unverified"
    assert result.saved_setup is None
    assert result.confirmation is not None and not result.confirmation.quality_passed
    check = next(item for item in result.rounds if item.get("round") == "memory-safety check")
    assert check["passed"] is False
    saved = store.find(second.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.gpu_offload_layers == high


def test_a_safety_correction_that_answers_correctly_is_checked_and_saved(tmp_path: Path) -> None:
    _saved_high(tmp_path)
    second, _, _, store = _tuner(tmp_path, _monotonic(RIGHT), answers=[True, True, True])
    second.try_load = lambda layers: LoadAttempt(loaded=True, fits=layers <= 10)
    result = second.run()
    check = next(item for item in result.rounds if item.get("round") == "memory-safety check")
    assert check["passed"] is True
    assert result.confirmation is not None and result.confirmation.rounds >= 1
    saved = store.find(second.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.gpu_offload_layers == 10


class _FadingThreads(Tuner):
    """The thread gain is real while the threads path runs, gone in the final check."""

    faded = False

    def _path_round(self, path: OptionPath, *args: Any, **kwargs: Any) -> Any:
        outcome = super()._path_round(path, *args, **kwargs)
        if path.name == "threads" and outcome[1] is not None:
            type(self).faded = True
        return outcome


def _fading_responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    layers = options.gpu_offload_layers or 0
    speed = 10.0 + layers
    latency = max(2.0 - 0.01 * layers, 0.5)
    if options.threads == 5 and not _FadingThreads.faded:
        speed *= 1.15
        latency *= 0.9
    return SampleResult(decode_tps=speed, total_latency_s=latency, text=RIGHT)


def test_a_failed_speed_check_keeps_the_safety_correction(tmp_path: Path) -> None:
    _saved_high(tmp_path)
    _FadingThreads.faded = False
    second, _, approver, store = _tuner(
        tmp_path, _fading_responder, answers=[True, True, True], tuner_cls=_FadingThreads
    )
    second.try_load = lambda layers: LoadAttempt(loaded=True, fits=layers <= 10)
    result = second.run()
    assert any(change.change_id.endswith(":safety:10") for change in result.proposed_changes)
    assert any(change.setting == "threads" for change in result.proposed_changes)
    assert result.outcome == "not confirmed (memory-safety correction saved)"
    assert [item.change_id for item in approver.asked] == ["runtime.increase_gpu_offload_layers:safety:10"]
    saved = store.find(second.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None
    assert saved.configuration.gpu_offload_layers == 10
    assert saved.configuration.threads is None


def test_declining_the_safety_correction_publishes_nothing_new(tmp_path: Path) -> None:
    high = _saved_high(tmp_path)
    _FadingThreads.faded = False
    second, _, _, store = _tuner(
        tmp_path, _fading_responder, answers=[False, False, False], tuner_cls=_FadingThreads
    )
    second.try_load = lambda layers: LoadAttempt(loaded=True, fits=layers <= 10)
    result = second.run()
    assert result.outcome == "not saved: memory-safety correction declined"
    assert result.saved_setup is None
    saved = store.find(second.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.gpu_offload_layers == high


# --------------------------------------------------------------------------- #
# Narrower issues: probe deadline                                              #
# --------------------------------------------------------------------------- #
class _ClockBackend:
    """A probe backend whose questions advance a fake clock."""

    def __init__(self, clock: dict[str, float], step: float) -> None:
        self.clock = clock
        self.step = step
        self.asked = 0
        self.running = False

    def start(self, model: Any, params: Any) -> None:
        self.running = True
        return None

    def generate(self, request: Any) -> Any:
        from inferenceos.interfaces.backend_adapter import GenerationResult

        self.asked += 1
        self.clock["now"] += self.step
        return GenerationResult(text=RIGHT)

    def server_log(self) -> str:
        return ""

    def stop(self) -> None:
        self.running = False
        return None

    def is_running(self) -> bool:
        return self.running


def test_the_probe_starts_no_question_after_its_deadline(tmp_path: Path) -> None:
    from inferenceos.tune.engine_probe import CHECK_PROMPTS, run_switch_probe

    clock = {"now": 0.0}
    backend = _ClockBackend(clock, step=10.0)
    result = run_switch_probe(
        settings=_settings(tmp_path),
        model=sample_model(),
        config=START,
        backend_factory=lambda: backend,  # type: ignore[arg-type, return-value]
        deadline=1.0,
        clock=lambda: clock["now"],
        exit_timeout_s=0.1,
    )
    # The timed question ran (it started before the deadline); nothing after it did.
    assert backend.asked == 1
    assert not result.complete
    assert "time limit" in result.error
    assert len(CHECK_PROMPTS) == 3


# --------------------------------------------------------------------------- #
# Narrower issues: inherited engine switches                                   #
# --------------------------------------------------------------------------- #
def test_every_listed_switch_is_managed() -> None:
    from inferenceos.backends.engine_env import MANAGED_SWITCHES
    from inferenceos.tune.engine_switches import CAPABILITIES

    names = {spec.name for specs in CAPABILITIES.values() for spec in specs}
    assert names <= MANAGED_SWITCHES


def test_an_inherited_switch_never_reaches_the_engine_next_to_a_conflicting_one() -> None:
    from inferenceos.backends.engine_env import parse_switches, process_environment
    from inferenceos.tune.engine_switches import VULKAN_B11139, apply_switch

    parent = {"GGML_VK_DISABLE_MMVQ": "1", "GGML_VK_VISIBLE_DEVICES": "0", "PATH": "x"}
    force = next(spec for spec in VULKAN_B11139 if spec.name == "GGML_VK_FORCE_MMVQ")
    env = process_environment(parent, parse_switches(apply_switch(None, force)))
    assert env.get("GGML_VK_FORCE_MMVQ") == "1"
    assert "GGML_VK_DISABLE_MMVQ" not in env
    assert env["GGML_VK_VISIBLE_DEVICES"] == "0"  # not a managed switch: passed on


def test_removing_a_switch_from_a_setup_means_the_engine_default() -> None:
    from inferenceos.backends.engine_env import parse_switches, process_environment

    parent = {"GGML_VK_DISABLE_MMVQ": "1"}
    env = process_environment(parent, parse_switches(None))
    assert "GGML_VK_DISABLE_MMVQ" not in env


# --------------------------------------------------------------------------- #
# Narrower issues: one clock in the simulation                                 #
# --------------------------------------------------------------------------- #
def test_the_simulation_keeps_a_balanced_tune_near_its_simulated_budget() -> None:
    import random

    from scripts.simulate_tune import ANSWER_S, LOAD_S, World, one

    noise = {"within_writing": 0.03, "within_first": 0.06, "between": 0.012, "rho": 0.3}
    for budget in (240.0, 900.0):
        world = World(
            "threads 4 +8%",
            {("threads", 4): {"short": (0.08, 0.0), "long": (0.08, 0.0)}},
            real=(("threads", 4),),
            budget_s=budget,
        )
        outcome = one(random.Random(3), world, check="light", effort="balanced", noise=noise, runs=6)
        # The time is simulated time (loads and answers), not host time; anything already
        # started may finish, so at most about one more block of two loads goes past it.
        one_block = 2 * (LOAD_S + 8 * ANSWER_S["long"])
        assert outcome.minutes * 60.0 <= budget + one_block, (budget, outcome.minutes)
        assert outcome.overrun_s <= one_block
