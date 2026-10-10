"""The ``tune`` sequence: search, confirm, ask, save (step 5).

1. The machine check (done by the caller) says what is free and what is installed.
2. Starting point: the saved setup for this model if there is one (plugged in or on
   battery as now, else the other one), otherwise as many graphics card layers as the
   model file says fit.
3. A notice asks the user to leave the computer alone. The starting setup is measured.
4. Round 1: graphics card layers across the whole safe range, fewer as well as more.
5. Round 2: processor threads.
6. The final setup is confirmed against the starting one by alternating runs.
7. Each change is shown with its measured gain and gets its own yes or no. If one is
   turned down, the rest is quickly checked again.
8. The approved setup is saved for this model, context size and power source.

Every test undoes itself: each measurement starts the model with the setup being
measured, and nothing is kept until the user says yes. The context size given by the
user is fixed for the whole run; every tried setup is checked against it. The model
is not started for use here: that is the ``start`` command.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inferenceos.cards.builtin.gpu_layers_more import CARD_ID as LAYERS_CARD_ID
from inferenceos.backends.llama_server_capabilities import ServerOptions
from inferenceos.cards.safety import check_context_unchanged
from inferenceos.config import Settings
from inferenceos.contracts.enums import OptimizationMode, PowerState, Situation
from inferenceos.contracts.experiment import (
    ExperimentMeasurement,
    ExperimentPolicy,
    RuntimeConfiguration,
)
from inferenceos.contracts.goal import GoalPreference, KeepJudgement, KeepVerdict
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.tune import (
    ChangeDecision,
    ConfirmationResult,
    MachineCheck,
    ProposedChange,
    SavedSetup,
    TuneResult,
)
from inferenceos.experiment.measurement import MeasurementHarness
from inferenceos.experiment.quality import validate_quality
from inferenceos.optimization.goals import assess_known_gains, improvement, plain_name
from inferenceos.optimization.keep import ModeRules
from inferenceos.optimization.search.evaluation import (
    RobustPointEvaluator,
    RuntimeConfigurationPointEvaluator,
)
from inferenceos.optimization.search.policy import SearchPolicy
from inferenceos.optimization.search.resources import build_resource_envelope
from inferenceos.profiling.workloads import LONG_PROMPT_PASSAGES, BenchmarkWorkload
from inferenceos.tune.approval import ChangeApprover, describe_change
from inferenceos.tune.fair import FairResult, compare_fairly, fair_words, measurement_value
from inferenceos.tune.heat import HeatCheck, analyse_heat
from inferenceos.tune.fit import layers_that_fit
from inferenceos.tune.layer_search import (
    LayerSearchSettings,
    full_layer_kinds_from_gguf,
    score as layer_score,
    search_layers,
    trimmed_mean,
)
from inferenceos.tune.layer_top import LoadAttempt, find_layer_top
from inferenceos.tune.measure import (
    DisturbanceAwareEvaluator,
    DisturbanceWatch,
    judge_pair,
    measure_steady,
    quality_ok,
    answer_costs,
    between_load_spread,
    choose_runs,
    run_score_spread,
)
from inferenceos.tune.diagnose import (
    MEMORY_PATHS,
    NORMAL,
    PROBE,
    Diagnosis,
    affected_paths,
    diagnose,
    processor_bytes_per_word,
)
from inferenceos.tune.engine_probe import ProbeNotSafe, ProbeResult
from inferenceos.tune.memory_speed import MemorySpeed
from inferenceos.tune.noise_store import NoiseStore, noise_key
from inferenceos.tune.placement import (
    card_kinds,
    engine_rule,
    max_count,
    rule_verified,
)
from inferenceos.tune.model_facts import (
    MachineFacts,
    ModelFacts,
    machine_facts,
    processor_share,
    read_model_facts,
    share_words,
)
from inferenceos.tune.paths import (
    LONG_QUESTION,
    long_question_workload,
    steady_workload,
    Candidate,
    OptionPath,
    PathContext,
    apply_candidate,
    piece_lines_key,
    verify_every_run,
)
from inferenceos.tune.acceptance_evidence import safety_resource_evidence
from inferenceos.tune.catalog import CATALOG
from inferenceos.tune.known_answers import KnownAnswerCheck, run_known_answer_check
from inferenceos.tune.quality_risk import (
    QualityRequirement,
    requirement_of_change,
    strictest,
)
from inferenceos.tune.profiles import (
    CommitGuard,
    SetupStore,
    model_identity,
    power_key,
    power_words,
)

THREADS_CARD_ID = "runtime.reduce_threads"
#: On the first look, a candidate that is slower on the guard question by more than
#: this is scored by that loss (the fair check in turns decides in the end).
GUARD_SCREEN_TOLERANCE = 0.03
#: Values a quick probe tries (a path the diagnosis gives little chance).
PROBE_CANDIDATES = 2
#: Word probabilities asked for in engine-switch correctness checks (chosen word plus
#: the closest alternatives).
SWITCH_PROBS = 3

#: Reads the graphics memory in use right now, with the model loaded.
GraphicsMemoryReader = Callable[[], int | None]


def thread_values_to_try(physical_cores: int, current: int | None) -> list[int]:
    """A few thread counts at or below the real core count (never above it)."""
    physical = max(int(physical_cores), 1)
    values = {physical, max(physical - 1, 1), max(physical // 2, 1)}
    return sorted(value for value in values if value != current)


def _tradeoff_invalid(fair: FairResult, question: str) -> str | None:
    """Why a trade-off measurement cannot be offered (None when it can): not completed,
    a guarded measure not read, answers changed, or a resource guard failed. Only the
    speed loss itself may be accepted by the user (review 5, 5 Oct 2026)."""
    if fair.stopped_early or not fair.turns:
        return f"the {question} question was not measured to the end ({fair.stopped_early or 'no blocks'})"
    if fair.unavailable_guards:
        return (
            f"on the {question} question these could not be measured in every block: "
            + ", ".join(plain_name(m) for m in fair.unavailable_guards)
        )
    if fair.reliability != "ok":
        return f"on the {question} question: " + "; ".join(fair.acceptance_blockers())
    if not fair.quality_passed:
        return f"the answers changed on the {question} question"
    if fair.guard_breaches:
        return (
            f"{', '.join(plain_name(m) for m in fair.guard_breaches)} guard failed on the "
            f"{question} question"
        )
    return None


def _gain_words(fair: FairResult) -> str:
    """"+24% (range +21.6% to +26.3%)" from a fair comparison."""
    if fair.gain is None:
        return "not measured"
    test = fair.test
    if test is not None and test.lower is not None and test.upper is not None:
        return f"{fair.gain:+.1%} (range {test.lower:+.1%} to {test.upper:+.1%})"
    return f"{fair.gain:+.1%}"


def setup_key(config: RuntimeConfiguration) -> str:
    """One text per exact setup, so a path is never run twice on the same setup."""
    return json.dumps(config.model_dump(exclude_none=True), sort_keys=True, default=str)


def _candidate_requirement(path: OptionPath, candidate: Candidate) -> tuple[str, QualityRequirement]:
    """(card, requirement) for a path's candidate: the path's card and every setting the
    candidate changes (review 7: derived from the settings, not the card alone)."""
    draft = ProposedChange(
        change_id="draft",
        card_id=path.card_id,
        setting=path.setting,
        description=path.description,
        new_value=candidate.value,
        related={name: value for name, value in candidate.also},
    )
    return requirement_of_change(draft)


def _change_keys(change: ProposedChange) -> set[str]:
    return {change.setting, *change.related}


def _same_value(a: object, b: object) -> bool:
    return (None if a == "" else a) == (None if b == "" else b)


def merge_change(
    changes: list[ProposedChange],
    change: ProposedChange,
    workloads: dict[str, list[BenchmarkWorkload]],
    starting: RuntimeConfiguration,
    *,
    keep_apart: set[str] | None = None,
) -> list[ProposedChange]:
    """Add ``change``; when it changes a setting an earlier change already changed (as
    its own setting or one changed with it), the changes become one change from the
    starting values (one approval, one entry in the final check). Settings back at
    their starting value are no longer part of it; nothing left means no change.
    Changes in ``keep_apart`` (memory-safety corrections) are never merged."""
    apart = keep_apart or set()
    overlapping = [
        item for item in changes
        if item.change_id not in apart and _change_keys(item) & _change_keys(change)
    ]
    if not overlapping:
        return [*changes, change]
    rest = [item for item in changes if item not in overlapping]
    questions: list[BenchmarkWorkload] = []
    for item in [*overlapping, change]:
        for workload in workloads.pop(item.change_id, []) if item is not change else workloads.get(change.change_id, []):
            if all(workload.workload_id != seen.workload_id for seen in questions):
                questions.append(workload)
    final = _apply_changes(starting, [*overlapping, change])
    keys = set().union(*(_change_keys(item) for item in [*overlapping, change]))
    main = change.setting
    related = {
        key: getattr(final, key)
        for key in sorted(keys - {main})
        if not _same_value(getattr(final, key), getattr(starting, key))
    }
    related_old = {key: getattr(starting, key) for key in related}
    start_value = getattr(starting, main)
    if _same_value(getattr(final, main), start_value) and not related:
        workloads.pop(change.change_id, None)
        return rest
    # Review 5: the gains measured were each for one step, never for the whole change;
    # the merged change carries none (the final check measures the whole setup) and the
    # steps are kept as evidence.
    steps: list[str] = []
    for number, item in enumerate([*overlapping, change], start=1):
        steps.append(
            (f"step {number}: " if number == 1 else f"looked at again after another change, step {number}: ")
            + f"{item.old_value} -> {item.new_value}"
            + (f" measured {item.measured_gain:+.1%} (that step only)" if item.measured_gain is not None else "")
        )
    earlier = overlapping[0]
    merged = change.model_copy(
        update={
            "old_value": earlier.old_value if earlier.setting == main else start_value,
            "new_value": getattr(final, main),
            "measured_gain": None,
            "related": related,
            "related_old": related_old,
            "evidence": [
                *[line for item in overlapping for line in item.evidence],
                *steps,
                *change.evidence,
            ],
        }
    )
    # Review 7: a merged change keeps the strictest answer requirement of everything in
    # it (e.g. a layer step merged with a cache-format change still needs the known-answer
    # questions), so it takes the card of its strictest part. Derived from the settings
    # the merged change really changes (a setting back at its starting value no longer
    # counts).
    best_card, _ = requirement_of_change(merged)
    if best_card != merged.card_id:
        merged = merged.model_copy(update={"card_id": best_card})
    workloads[merged.change_id] = questions
    return [*rest, merged]


@dataclass
class Tuner:
    """Runs one ``tune``. Everything that touches the real machine is injectable."""

    settings: Settings
    model: ModelProfile
    model_path: Path
    context_length: int
    rules: ModeRules
    hardware: HardwareProfile
    machine: MachineCheck
    harness: MeasurementHarness
    workload: BenchmarkWorkload
    experiment_policy: ExperimentPolicy
    search_policy: SearchPolicy
    approver: ChangeApprover
    store: SetupStore
    graphics_memory_reader: GraphicsMemoryReader
    machine_changes: list[str] = field(default_factory=list)
    say: Callable[[str], None] = print
    run_id: str | None = None
    model_hash: bool = True
    # Load-only try for the layer top (prompt B). None means "use the estimate".
    try_load: Callable[..., LoadAttempt] | None = None
    # llama-server options this build has (from ``--help``); None = not checked.
    supported_options: frozenset[str] | None = None
    # The same help text with allowed values (``backends/llama_server_capabilities``);
    # None = not read. Answers only "does the engine accept it".
    server_options: ServerOptions | None = None
    _heat_decisions: list[dict[str, object]] = field(default_factory=list, init=False)
    _heat_check: HeatCheck | None = field(default=None, init=False)
    #: A memory-safety layer correction found by the layer round (no speed claim).
    _layer_safety_change: ProposedChange | None = field(default=None, init=False)
    #: Runs the setup once with the engine's timer on (``tune/engine_probe.py``);
    #: None = no timed runs (the diagnosis then has no card share).
    probe_runner: Callable[..., ProbeResult] | None = None
    #: The machine's best memory copy rate (``tune/memory_speed.py``), cached by the caller.
    memory_speed_reader: Callable[[], MemorySpeed | None] | None = None
    #: A fresh load of a setup with the engine at its debug log level, returning what
    #: it showed (``observed``): llama.cpp b11139 prints which pieces a placement moved
    #: only at that level (laptop job 110), and measuring at that level costs writing
    #: speed and floods the log (job 112), so placements read their per-piece lines from
    #: this separate, untimed load. None = not available (placements stay unverified
    #: unless the measured runs themselves list the pieces).
    piece_lines_reader: Callable[[RuntimeConfiguration], dict[str, object] | None] | None = None
    #: Per-piece lines read so far: "<layers>|<overrides>" -> piece -> buffer (None when
    #: the read load showed none).
    _piece_lines: dict[str, dict[str, str] | None] = field(default_factory=dict, init=False)
    #: Stops whatever the engine is doing (the experimental budgeted search calls it at
    #: its deadline and as its timed cleanup); None = nothing to stop.
    cancel_hook: Callable[[], None] | None = None
    #: A user stop (review 10): ends the engine in any state and keeps it from starting
    #: again (no reload half, no recovery). None = ``cancel_hook``.
    user_stop_hook: Callable[[], None] | None = None
    #: Fixed looks for the final check by question id ("*" = any other question): the
    #: budgeted search's one look at N blocks per question, N fixed before the check.
    #: None = the effort mode's planned looks.
    _final_looks: dict[str, dict[str, int]] | None = field(default=None, init=False)
    #: Progress events of the final check's comparisons (review 10; see
    #: ``compare_fairly``): the budgeted search paces the check from them. None = off.
    _final_progress: Callable[[dict[str, Any]], None] | None = field(default=None, init=False)
    #: Every comparison the whole-setup check ran (question id, result), so its fresh
    #: measurements can be timed and reported; cleared by whoever reads it.
    _whole_setup_fairs: list[tuple[str, FairResult]] = field(default_factory=list, init=False)
    #: Engine type seen in a timed run ("vulkan"), and switches that passed the checks.
    _engine_name: str | None = field(default=None, init=False)
    _checked_switches: tuple[tuple[str, str], ...] | None = field(default=None, init=False)
    _last_probe: ProbeResult | None = field(default=None, init=False)
    #: An engine process did not exit after a timed run: no more timed runs.
    _probes_unsafe: bool = field(default=False, init=False)
    #: Engine-switch checks: the two unchanged timed runs, and switches not checked yet
    #: (a quick probe checks only the first ones; widened when one looks better).
    _switch_base: tuple[ProbeResult, ProbeResult] | None = field(default=None, init=False)
    _unchecked_switches: list[tuple[str, str, tuple[str, ...]]] = field(
        default_factory=list, init=False
    )
    _switch_checks: list[dict[str, object]] = field(default_factory=list, init=False)
    #: Trade-offs: clearly better on a path's question, held back by a loss on the
    #: question it must not lose on; (setting, value) -> details. Offered at the end.
    _tradeoffs: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict, init=False)
    #: When the time limit for everything ends (None in Performance first).
    _final_deadline: float | None = field(default=None, init=False)
    #: Runs per measurement chosen for each question (``_set_runs_from``).
    _runs_by_question: dict[str, int] = field(default_factory=dict, init=False)
    _facts: ModelFacts | None = field(default=None, init=False)
    _machine_facts: MachineFacts | None = field(default=None, init=False)
    _long_question_workload: BenchmarkWorkload | None = field(default=None, init=False)
    _long_question_tokens: int | None = field(default=None, init=False)
    _long_question_record: dict[str, object] | None = field(default=None, init=False)
    #: The questions each proposed change was tuned and guarded on (by change id).
    _change_workloads: dict[str, list[BenchmarkWorkload]] = field(
        default_factory=dict, init=False
    )

    # ------------------------------------------------------------------ #
    def run(self) -> TuneResult:
        """Run ``tune``; the heat report (what heat did to any decision, and the heat
        check of the chosen setup) is added to the result and printed at the end."""
        self._heat_decisions = []
        self._heat_check = None
        self._layer_safety_change = None
        self._runs_by_question = {}
        self._engine_name = None
        self._checked_switches = None
        self._last_probe = None
        self._probes_unsafe = False
        self._switch_base = None
        self._unchecked_switches = []
        self._switch_checks = []
        self._tradeoffs = {}
        self._final_deadline = None
        self._long_question_workload = None
        self._long_question_tokens = None
        self._long_question_record = None
        self._change_workloads = {}
        self._piece_lines.clear()
        self._final_looks = None
        self._final_progress = None
        budget = float(getattr(self.settings, "tune_experimental_budget_s", 0.0) or 0.0)
        if budget > 0:
            # Internal experiment (handoff, 6 Oct 2026): a finite total budget with the
            # planner's search order and one final confirmation. Never the default.
            from inferenceos.tune.budgeted import BudgetedTune

            result = BudgetedTune(self, budget).run()
        else:
            result = self._run_search()
        if budget <= 0 and self._tradeoffs and bool(self.settings.tune_offer_tradeoffs):
            result = self._offer_tradeoff(result)
        report, lines = self._heat_report()
        for line in lines:
            self.say(line)
        return result.model_copy(update={"heat_report": report})

    def _offer_tradeoff(self, result: TuneResult) -> TuneResult:
        """Measure the best trade-off properly against the setup tune chose, and offer it
        for an explicit yes or no (owner, 5 Oct 2026).

        A trade-off is a change clearly better on one question but losing more than the
        allowed 2% (or not shown to lose less) on the question it must not lose on. Tune
        never keeps one on its own, not even with ``--yes``. It is measured fresh with the
        effort mode's full looks on both questions, and offered with both gains and their
        ranges; only a "yes" keeps it.
        """
        if result.outcome not in ("no change", "saved", "saved (some changes turned down)"):
            return result
        if self._final_deadline is not None and time.perf_counter() >= self._final_deadline:
            self.say("Trade-off not measured: no time left.")
            return result.model_copy(update={"notes": [*result.notes, "trade-off not measured: no time left"]})
        chosen = result.final_configuration
        options = [
            item for item in self._tradeoffs.values()
            if getattr(chosen, item["path"].setting) != item["candidate"].value
        ]
        if not options:
            return result
        if not getattr(self.approver, "can_accept_tradeoffs", True):
            # --yes never keeps a trade-off, so measuring one would only cost time (on
            # Granite 3.1 MoE, job 102, about an hour). Report what was found instead.
            found = ", ".join(entry["candidate"].label for entry in options)
            words = (
                f"trade-offs found but not measured: {found} (--yes never keeps a "
                "trade-off; run tune without --yes to measure and decide)"
            )
            self.say(words[0].upper() + words[1:])
            rounds = [*result.rounds, {"round": "trade-off", "measured": False, "found": found, "why": words}]
            return result.model_copy(update={"rounds": rounds, "notes": [*result.notes, words]})
        item = max(options, key=lambda entry: entry["main_gain"] or 0.0)
        path: OptionPath = item["path"]
        candidate: Candidate = item["candidate"]
        workload: BenchmarkWorkload = item["workload"]
        guard: BenchmarkWorkload = item["guard"]
        config = apply_candidate(chosen, path, candidate)
        check_context_unchanged(chosen, config, card_id=path.card_id)
        watch = DisturbanceWatch(
            logical_cores=self.hardware.cpu.logical_cores,
            limit_percent=float(self.settings.tune_disturbance_cpu_percent),
            max_repeats=int(self.settings.tune_max_disturbed_repeats),
            on_repeat=lambda reason: self.say(f"  measurement {reason}; repeating it"),
        )
        self.say(
            f"Trade-off: {candidate.label} looked better on the {workload.workload_id} "
            f"question but lost on the {guard.workload_id} question. Measuring both properly."
        )
        untested = [entry["candidate"].label for entry in options if entry is not item]
        if untested:
            self.say("  other trade-offs not measured (untested): " + ", ".join(untested))
        record: dict[str, object] = {
            "round": "trade-off",
            "change": candidate.label,
            "setting": path.setting,
            "value": candidate.value,
            "found_by": item["source"],
            "others_untested": [entry["candidate"].label for entry in options if entry is not item],
        }
        gains: dict[str, FairResult] = {}
        # Review 7: the answer checks use the strictest requirement of every setting the
        # trade-off changes, and every fresh load supplying evidence must show the
        # requested settings (a yes waives only the speed loss, never this evidence).
        quality_card, requirement = _candidate_requirement(path, candidate)
        record["quality_card"] = quality_card
        record["quality_requirement"] = requirement.record()
        to_verify = self._settings_to_verify(chosen, config)
        start_verify = self._settings_to_verify(RuntimeConfiguration(), chosen)
        for label, on, mode in ((workload.workload_id, workload, "better"), (guard.workload_id, guard, "not_worse")):
            self.say(f"  on the {label} question")
            fair = compare_fairly(
                harness=self.harness,
                model=self.model,
                workload=on,
                start=chosen,
                candidate=config,
                policy=self._path_policy(on),
                settings=self.settings,
                rules=self.rules,
                watch=watch,
                card_id=quality_card,
                phase=f"tune-tradeoff-{label}",
                mode=mode,  # type: ignore[arg-type]
                deadline=self._final_deadline,
                on_step=lambda step: self.say(f"    {step}"),
            )
            self.say("    " + fair_words(fair))
            self._note_heat(f"trade-off {candidate.label} on the {label} question", fair)
            self._learn_load_spread(on, fair)
            gains[label] = fair
            record[f"{mode}_check"] = fair.record()
            # Review 5: the user's yes may relax only the speed-loss limit. Both questions
            # need complete, valid measurements, unchanged answers and every resource
            # guard (memory, paging, ...) passed before anything is offered.
            invalid = _tradeoff_invalid(fair, on.workload_id)
            if invalid is None and mode == "better" and not fair.kept:
                invalid = f"the gain on the {label} question did not hold up when measured properly"
            effects = self._verify_effective(config, fair.candidate_runs, to_verify)
            start_effects = self._verify_effective(chosen, fair.start_runs, start_verify)
            record[f"{mode}_observed"] = [
                *({"setup": "new", **item} for item in effects),
                *({"setup": "current", **item} for item in start_effects),
            ]
            not_shown = [
                f"{item['setting']} ({item['why']})"
                for item in [*effects, *start_effects]
                if item["status"] != "shown"
            ]
            if invalid is None and not_shown:
                invalid = (
                    f"the requested settings were not shown by the engine in the fresh "
                    f"measurements on the {label} question: " + "; ".join(not_shown)
                )
            if invalid is not None:
                record["reason"] = f"not offered: {invalid}"
                self.say(f"  not offered: {invalid}")
                return result.model_copy(update={"rounds": [*result.rounds, record]})
        if requirement.known_answers:
            # The user's yes relaxes only the speed-loss limit, never answer quality.
            known = self._known_answers(chosen, config, watch, self._final_deadline, "trade-off")
            record["known_answers"] = known.record()
            if not known.passed:
                record["reason"] = f"not offered: known-answer questions {known.status} ({known.reason})"
                self.say(f"  not offered: known-answer questions {known.status} ({known.reason})")
                return result.model_copy(update={"rounds": [*result.rounds, record]})
        main, other = gains[workload.workload_id], gains[guard.workload_id]
        words = (
            f"{path.description}: {getattr(chosen, path.setting) if getattr(chosen, path.setting) is not None else 'program default'}"
            f" -> {candidate.value}. {workload.workload_id} question: {_gain_words(main)}; "
            f"{guard.workload_id} question: {_gain_words(other)}"
            + (" (within the 2% limit)" if other.not_worse else " (beyond or not shown within your 2% limit)")
        )
        record["words"] = words
        change = ProposedChange(
            change_id=f"{path.card_id}:{candidate.value}:trade-off",
            card_id=path.card_id,
            setting=path.setting,
            description=path.description,
            old_value=getattr(chosen, path.setting) if getattr(chosen, path.setting) is not None else "program default",
            new_value=candidate.value,
            measured_gain=main.gain,
            evidence=[f"trade-off: {words}", "measured in turns: " + fair_words(main), fair_words(other)],
            related={
                name: value for name, value in candidate.also
                if not _same_value(getattr(chosen, name), value)
            },
            related_old={
                name: getattr(chosen, name) for name, value in candidate.also
                if not _same_value(getattr(chosen, name), value)
            },
        )
        decision = self.approver.decide_tradeoff(
            change, words=words, timeout_s=float(self.settings.tune_approval_timeout_s)
        )
        record["decision"] = decision.model_dump()
        notes = [*result.notes, f"trade-off offered: {words}; answer: {decision.answer}"]
        if not decision.approved:
            self.say(f"  trade-off not kept ({decision.detail or decision.answer})")
            return result.model_copy(
                update={"rounds": [*result.rounds, record], "notes": notes,
                        "decisions": [*result.decisions, decision]}
            )
        saved_before = result.saved_setup
        approved = [*(saved_before.approved_changes if saved_before else []), change]
        declined = list(saved_before.declined_changes) if saved_before else []
        saved = self._save(
            run_id=result.run_id,
            power=power_key(self.machine.power_source),
            starting=result.starting_configuration,
            final=config,
            approved=approved,
            declined=declined,
            confirmed_gain=saved_before.confirmed_gain if saved_before else None,
            notes=[*(saved_before.notes if saved_before else []), f"trade-off kept by the user: {words}"],
        )
        self.say(f"  trade-off kept by you: {words}")
        return result.model_copy(
            update={
                "rounds": [*result.rounds, record],
                "notes": notes,
                "decisions": [*result.decisions, decision],
                "final_configuration": config,
                "saved_setup": saved,
                "proposed_changes": [*result.proposed_changes, change],
                "outcome": "saved (with a trade-off you approved)",
            }
        )

    def _note_heat(self, what: str, fair: FairResult) -> None:
        """Record a processor-speed change seen during a comparison (reported only)."""
        if not fair.heat_effect:
            return
        entry: dict[str, object] = {
            "what": what,
            "effect": fair.heat_effect,
            "verdict": fair.judgement.verdict.value if fair.judgement else None,
            "decision": fair.decision,
        }
        self._heat_decisions.append(entry)
        self.say("  processor speed: " + fair.heat_effect)

    def _heat_report(self) -> tuple[dict[str, object], list[str]]:
        """Processor-speed changes seen, and the end-of-tune heat check.

        User decision (4 Oct 2026): no block is ever left out because of processor-speed
        readings, so heat never decides a comparison by being left out; a speed change
        alone is not called overheating. The result is flagged only when the end-of-tune
        check saw answers really get slower in a row.
        """
        lines: list[str] = ["Heat report:"]
        if not self._heat_decisions:
            lines.append("  the processor speed stayed steady in every comparison")
        for item in self._heat_decisions:
            lines.append(f"  {item['what']}: {item['effect']}")
        check = self._heat_check
        if check is not None:
            lines.append("  " + check.words)
        hot_end = check is not None and check.throttled
        if hot_end:
            overall = (
                "Overall: answers got slower as the laptop kept working (end-of-tune "
                "check), so this result may differ when the laptop is cool. Every block "
                "of every comparison was counted."
            )
        elif self._heat_decisions:
            overall = (
                f"Overall: the processor speed changed during {len(self._heat_decisions)} "
                "comparison(s); every block was counted, and the end-of-tune check saw no "
                "slowdown in a row."
            )
        else:
            overall = "Overall: no sign of heat affecting this result."
        lines.append(overall)
        report: dict[str, object] = {
            "decisions": list(self._heat_decisions),
            "heat_check": check.record() if check is not None else None,
            "result_may_not_be_best": bool(hot_end),
            "overall": overall,
            "lines": lines,
        }
        return report, lines

    def _run_search(self) -> TuneResult:
        run_id = self.run_id or (
            f"tune-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:6]}"
        )
        budget = float(self.rules.effort.before_loading_search_time_s)
        started = time.perf_counter()
        search_deadline = started + budget * (1.0 - self._confirm_share())
        final_deadline = started + budget
        notes: list[str] = []
        rounds: list[dict[str, object]] = []
        not_reached: list[str] = []
        power = power_key(self.machine.power_source)
        watch = DisturbanceWatch(
            logical_cores=self.hardware.cpu.logical_cores,
            limit_percent=float(self.settings.tune_disturbance_cpu_percent),
            max_repeats=int(self.settings.tune_max_disturbed_repeats),
            on_repeat=lambda reason: self.say(f"  measurement {reason}; repeating it"),
        )

        starting, source = self._starting_point(power)
        if self.rules.effort.mode is OptimizationMode.PERFORMANCE_FIRST:
            # Review 4: the old message gave the time limit, which Performance first does
            # not use; its planned comparisons decide when it ends.
            how_long = (
                "Performance first has no fixed time limit: it ends when its planned "
                "comparisons are done, which often takes hours (Gemma 3 1B on a laptop: "
                "about 4.5 hours)"
            )
        else:
            how_long = f"up to about {budget / 60:.0f} minutes"
        self.say(
            f"Please leave the computer alone while InferenceOS measures ({how_long}). "
            "Measurements disturbed by other activity are repeated."
        )
        self.say(f"Starting setup ({source}): {_describe_config(starting)}")

        base = TuneResult(
            run_id=run_id,
            model_path=str(self.model_path),
            context_length=self.context_length,
            effort_mode=self.rules.effort.mode.value,
            goal_mode=self.rules.goal.mode.value,
            time_limit_s=budget,
            machine=self.machine,
            machine_changes=list(self.machine_changes),
            starting_configuration=starting,
            starting_point_source=source,
            final_configuration=starting,
        )

        # The start is measured with the paths' fixed run count, so its run-to-run
        # spread can set the runs per measurement for this question (4 Oct 2026).
        began = time.perf_counter()
        start_measurement = measure_steady(
            harness=self.harness,
            model=self.model,
            workload=self.workload,
            config=starting,
            policy=self._path_policy(),
            phase="tune-start",
            watch=watch,
        )
        start_wall_s = time.perf_counter() - began
        if start_measurement.valid_runs < self.experiment_policy.min_valid_samples:
            reason = (
                "the starting setup could not be measured "
                f"({start_measurement.valid_runs} good runs; "
                + "; ".join(start_measurement.notes[-2:])
                + ")"
            )
            self.say(f"Stopped: {reason}. Nothing was saved.")
            return _finish(
                base,
                final_configuration=starting,
                outcome="failed: " + reason,
                notes=notes,
            )

        self._set_runs_from(self.workload, start_measurement, start_wall_s)

        # --- Round 1: graphics card layers -------------------------------- #
        current = starting
        changes: list[ProposedChange] = []
        performance_first = self.rules.effort.mode is OptimizationMode.PERFORMANCE_FIRST
        self.say("Round 1: graphics card layers")
        if performance_first:
            self.say(
                "  performance first: the layer search may take longer than the usual "
                "limit; the limit for the rest of tune starts when it ends"
            )
        if bool(self.settings.tune_layer_path):
            layer_round, layer_change, layer_measurement = self._layer_round(
                current, start_measurement, watch, search_deadline
            )
        else:
            layer_round = {
                "round": "graphics card layers",
                "card_id": LAYERS_CARD_ID,
                "reason": "skipped: the layer path is switched off (tune_layer_path)",
            }
            layer_change, layer_measurement = None, None
        rounds.append(layer_round)
        safety_ids: set[str] = set()
        safety_change = self._layer_safety_change
        if safety_change is not None:
            safety_ids.add(safety_change.change_id)
            changes.append(safety_change)
            safe_layers = safety_change.new_value
            assert isinstance(safe_layers, int)
            current = current.model_copy(update={"gpu_offload_layers": safe_layers})
            self.say(f"  memory safety: {describe_change(safety_change)} (no speed claim)")
        if layer_change is not None and layer_measurement is not None:
            self._change_workloads[layer_change.change_id] = [self.workload]
            new_layers = layer_change.new_value
            assert isinstance(new_layers, int)
            current = current.model_copy(update={"gpu_offload_layers": new_layers})
            changes.append(layer_change)
            self.say(f"  found: {describe_change(layer_change)}")
        else:
            self.say(f"  no better layer count: {layer_round.get('reason')}")

        # Performance first: the layer search had no time limit, so the remaining time
        # (threads and confirmation) starts counting only now.
        if performance_first:
            rest_started = time.perf_counter()
            search_deadline = rest_started + budget * (1.0 - self._confirm_share())
            final_deadline = rest_started + budget

        self._final_deadline = None if performance_first else final_deadline

        # --- Further optimization paths, each from the setup the last one left ---- #
        chosen_paths = CATALOG.by_names(self._path_names())
        if chosen_paths:
            rounds.append(self._facts_round(current))
        # Diagnosis first (owner, 4 Oct 2026): it orders the paths and decides how
        # deeply to look at each; it skips nothing. Run again after each kept change.
        path_deadline = None if performance_first else search_deadline
        diagnosis: Diagnosis | None = None
        if chosen_paths and bool(self.settings.tune_diagnosis):
            diagnosis, diagnosis_record = self._diagnose(current, watch, "start", path_deadline)
            rounds.append(diagnosis_record)
        by_name = {item.name: item for item in chosen_paths}
        remaining = list(chosen_paths)
        # Looking again (owner, 4 Oct 2026): after a kept change, the finished paths it
        # can affect run again, at most ``tune_path_revisits`` extra times each, within
        # the time left; a path is never run twice on the exact same setup.
        revisits_left = {name: max(int(self.settings.tune_path_revisits), 0) for name in by_name}
        # The layer count is searched again after a kept change that moved card memory
        # (placement, conversation memory), within the same revisit limit.
        layer_revisits_left = max(int(self.settings.tune_path_revisits), 0)
        finished: set[str] = set()
        tested: set[tuple[str, str]] = set()
        number = 1
        while remaining:
            if diagnosis is not None:
                order = diagnosis.order([item.name for item in remaining])
                remaining.sort(key=lambda item: order.index(item.name))
            path = remaining.pop(0)
            number += 1
            again = path.name in finished
            label = path.label + (" (again, after a kept change)" if again else "")
            key = (path.name, setup_key(current))
            if key in tested:
                self.say(f"Round {number} ({label}) skipped: already tried on this exact setup.")
                continue
            if path_deadline is not None and time.perf_counter() >= path_deadline:
                not_reached.append(f"{label} (time limit reached)")
                self.say(f"Round {number} ({label}) not reached: time limit.")
                continue
            depth = diagnosis.priority(path.name) if diagnosis is not None else NORMAL
            if path.name == "engine_switches":
                rounds.append(self._check_switches(current, depth, path_deadline))
            self.say(f"Round {number}: {label}" + (" (quick probe)" if depth == PROBE else ""))
            path_round, path_change, new_config = self._path_round(
                path, current, watch, path_deadline, depth=depth,
            )
            tested.add(key)
            finished.add(path.name)
            if again:
                path_round["again"] = True
            if self._long_question_record is not None and not any(
                item.get("round") == "long test question" for item in rounds
            ):
                rounds.append(dict(self._long_question_record))
            rounds.append(path_round)
            if path_change is not None and new_config is not None:
                current = new_config
                changes = merge_change(
                    changes, path_change, self._change_workloads, starting, keep_apart=safety_ids
                )
                self.say(f"  found: {describe_change(path_change)}")
                touched = [path.name]
                layer_key = ("layers", setup_key(current))
                if (
                    path.name in MEMORY_PATHS
                    and bool(self.settings.tune_layer_path)
                    and layer_revisits_left > 0
                    and layer_key not in tested
                    and (path_deadline is None or time.perf_counter() < path_deadline)
                ):
                    layer_revisits_left -= 1
                    tested.add(layer_key)
                    self.say("Graphics card layers (again: the change moved what fits on the card)")
                    layer_again, layer_change_again, _ = self._layer_round(
                        current, start_measurement, watch,
                        path_deadline if path_deadline is not None else float("inf"),
                        revisit=True,
                    )
                    layer_again["again"] = True
                    rounds.append(layer_again)
                    if layer_change_again is not None:
                        new_layers = layer_change_again.new_value
                        assert isinstance(new_layers, int)
                        current = current.model_copy(update={"gpu_offload_layers": new_layers})
                        self._change_workloads[layer_change_again.change_id] = [self.workload]
                        changes = merge_change(
                            changes, layer_change_again, self._change_workloads, starting,
                            keep_apart=safety_ids,
                        )
                        self.say(f"  found: {describe_change(layer_change_again)}")
                        touched.append("layers")
                    else:
                        self.say(f"  no better layer count: {layer_again.get('reason')}")
                queued = {item.name for item in remaining}
                for name in [item for changed in touched for item in affected_paths(changed)]:
                    if name not in by_name or name in queued or name not in finished:
                        continue
                    if revisits_left[name] <= 0:
                        continue
                    revisits_left[name] -= 1
                    remaining.append(by_name[name])
                    queued.add(name)
                    self.say(
                        f"  will look again at {by_name[name].label}: the change can move "
                        "what limits it"
                    )
                if diagnosis is not None and remaining:
                    # The bottleneck can move after a change: look again.
                    new_diagnosis, diagnosis_record = self._diagnose(
                        current, watch, f"after {path.label}", path_deadline
                    )
                    rounds.append(diagnosis_record)
                    if new_diagnosis is not None:
                        diagnosis = new_diagnosis
            else:
                self.say(f"  no change: {path_round.get('reason')}")

        final = current
        check_context_unchanged(starting, final, card_id="tune")
        rounds.append({"round": "runs per measurement", "runs": dict(self._runs_by_question)})

        # --- Heat check: the chosen setup answers several questions in a row ----- #
        heat_runs = int(self.settings.tune_heat_check_runs)
        if heat_runs > 0 and not performance_first and time.perf_counter() >= search_deadline:
            # The time limit counts for the heat check too; the confirmation keeps its
            # own reserved share of the time.
            notes.append("heat check skipped: no time left within the time limit")
            self.say("Heat check skipped: no time left within the time limit.")
        elif heat_runs > 0:
            self._heat_check = self._run_heat_check(final, watch, heat_runs)

        # --- Nothing better found ------------------------------------------ #
        if not changes:
            saved = self._save(
                run_id=run_id,
                power=power,
                starting=starting,
                final=starting,
                approved=[],
                declined=[],
                confirmed_gain=None,
                notes=["no better setup found; the starting setup was saved"],
            )
            self.say("No better setup found. The starting setup was saved for `start`.")
            return _finish(
                base,
                final_configuration=starting,
                rounds=rounds,
                saved_setup=saved,
                outcome="no change",
                not_reached=not_reached,
                notes=notes,
            )

        # --- Confirm the whole new setup against the starting one ------------- #
        # One acceptance route (review, 4 Oct 2026). Every performance change was shown
        # better on its own question against the setup just before it; the exact new
        # setup is now checked against the (memory-safe) start on every question any
        # change was tuned or guarded on. The gain reported is the one measured here,
        # never multiplied together from the separate checks. A memory-safety correction
        # makes no speed claim and is part of the start this is compared with.
        performance = [change for change in changes if change.change_id not in safety_ids]
        safety = [change for change in changes if change.change_id in safety_ids]
        safe_start = _apply_changes(starting, safety)
        safety_check: ConfirmationResult | None = None
        if safety:
            # Review 6: a safety correction needs no speed gain, but it must be shown to
            # run and to answer like the starting setup did. Without that evidence the
            # corrected setup is not published, and neither is the unsafe start.
            self.say("Checking the memory-safe setup runs and answers correctly")
            safety_check, safety_record = self._check_safety_correction(
                starting, safe_start, start_measurement, watch,
                None if performance_first else final_deadline,
            )
            rounds.append(safety_record)
            if not safety_check.confirmed:
                why = "; ".join(safety_check.notes) or "no evidence"
                self.say(
                    "The memory-safe setup was not shown to run and answer correctly "
                    f"({why}). Nothing was saved: the starting setup does not leave enough "
                    "graphics memory free, and the safe one is unverified."
                )
                return _finish(
                    base,
                    final_configuration=starting,
                    rounds=rounds,
                    proposed_changes=changes,
                    confirmation=safety_check,
                    outcome="not saved: memory-safety correction unverified",
                    not_reached=not_reached,
                    notes=[*notes, f"memory-safety correction unverified: {why}"],
                )
        if performance:
            self.say("Confirming: the whole new setup against the starting one")
            whole_ok, confirmation, checks = self._check_whole_setup(
                safe_start, final, performance, watch,
                None if performance_first else final_deadline, label="final",
            )
            rounds.append({"round": "final check", "checks": checks})
            notes.extend(f"final check, {item['workload']}: {item['words']}" for item in checks)
            for line in _final_summary(checks):
                self.say("  " + line)
                notes.append(line)
        else:
            assert safety_check is not None
            confirmation = safety_check
            self.say("Only a memory-safety correction was found (no speed claim).")
        if not confirmation.confirmed:
            if safety:
                # Review 6: failing to show a speed gain must not erase the memory-safety
                # finding. The correction gets its own yes or no; declined, nothing is
                # published (the start does not leave enough graphics memory free).
                self.say(
                    "The speed changes did not hold up, so none are proposed. The "
                    "memory-safety correction still applies."
                )
                safety_decisions = [
                    self.approver.decide(
                        change, timeout_s=float(self.settings.tune_approval_timeout_s)
                    )
                    for change in safety
                ]
                if all(item.approved for item in safety_decisions):
                    saved = self._save(
                        run_id=run_id,
                        power=power,
                        starting=starting,
                        final=safe_start,
                        approved=safety,
                        declined=performance,
                        confirmed_gain=None,
                        notes=[
                            "the speed changes did not hold up when confirmed; the "
                            "memory-safety correction was saved (no speed claim)"
                        ],
                    )
                    self.say(
                        f"Saved for {power_words(power)}: {_describe_config(safe_start)}. "
                        "Run `start` to use it."
                    )
                    outcome = "not confirmed (memory-safety correction saved)"
                    final_saved: SavedSetup | None = saved
                    final_config = safe_start
                else:
                    self.say(
                        "Memory-safety correction turned down: nothing was saved, because "
                        "the starting setup does not leave enough graphics memory free."
                    )
                    outcome = "not saved: memory-safety correction declined"
                    final_saved = None
                    final_config = starting
                return _finish(
                    base,
                    final_configuration=final_config,
                    rounds=rounds,
                    proposed_changes=changes,
                    confirmation=confirmation,
                    decisions=safety_decisions,
                    recheck=safety_check,
                    saved_setup=final_saved,
                    outcome=outcome,
                    not_reached=not_reached,
                    notes=notes,
                )
            saved = self._save(
                run_id=run_id,
                power=power,
                starting=starting,
                final=starting,
                approved=[],
                declined=[],
                confirmed_gain=None,
                notes=[
                    "the changes did not hold up when confirmed; the starting setup was saved"
                ],
            )
            self.say("The changes did not hold up, so none are proposed. Starting setup saved.")
            return _finish(
                base,
                final_configuration=starting,
                rounds=rounds,
                proposed_changes=changes,
                confirmation=confirmation,
                saved_setup=saved,
                outcome="not confirmed",
                not_reached=not_reached,
                notes=notes,
            )

        # --- Ask about each change ------------------------------------------ #
        gain = (
            confirmation.judgement.goal.gain
            if confirmation.judgement and performance
            else None
        )
        self.say(
            f"Together these changes measured {gain:+.1%} on your "
            f"{self.rules.goal.mode.value} goal."
            if gain is not None
            else "Together these changes were confirmed."
        )
        decisions: list[ChangeDecision] = []
        for change in changes:
            decisions.append(
                self.approver.decide(
                    change, timeout_s=float(self.settings.tune_approval_timeout_s)
                )
            )
        approved_ids = {item.change_id for item in decisions if item.approved}
        if safety and not all(change.change_id in approved_ids for change in safety):
            # Review 6: every speed change was measured against the memory-safe start, and
            # the unsafe start is never published as a usable result.
            self.say(
                "Memory-safety correction turned down: nothing was saved, because the "
                "starting setup does not leave enough graphics memory free."
            )
            return _finish(
                base,
                final_configuration=starting,
                rounds=rounds,
                proposed_changes=changes,
                confirmation=confirmation,
                decisions=decisions,
                outcome="not saved: memory-safety correction declined",
                not_reached=not_reached,
                notes=notes,
            )
        approved = [change for change in changes if change.change_id in approved_ids]
        declined = [change for change in changes if change.change_id not in approved_ids]
        approved_performance = [c for c in approved if c.change_id not in safety_ids]

        chosen = _apply_changes(starting, approved)
        recheck: ConfirmationResult | None = None
        confirmed_gain = gain
        outcome = "saved"
        save_notes: list[str] = []
        if not approved_performance and performance:
            chosen = safe_start
            confirmed_gain = None
            recheck = safety_check
            outcome = "no changes approved" if not safety else "saved (some changes turned down)"
            save_notes.append(
                "no speed changes approved; "
                + ("the memory-safe starting setup was saved" if safety else "the starting setup was saved")
            )
        elif declined:
            self.say("Checking the changes you kept on their own")
            recheck_holds, recheck, recheck_checks = self._check_whole_setup(
                safe_start, chosen, approved_performance, watch, None, label="recheck",
            )
            rounds.append(
                {"round": "re-check of the changes you kept", "checks": recheck_checks}
            )
            if recheck_holds:
                confirmed_gain = recheck.judgement.goal.gain if recheck.judgement else None
                outcome = "saved (some changes turned down)"
            else:
                chosen = safe_start
                confirmed_gain = None
                outcome = "kept changes did not hold up on their own"
                save_notes.append(
                    "the approved changes did not hold up without the ones turned down; "
                    + ("the memory-safe starting setup was saved" if safety else "the starting setup was saved")
                )
                approved = list(safety)
                declined = list(performance)

        saved = self._save(
            run_id=run_id,
            power=power,
            starting=starting,
            final=chosen,
            approved=approved,
            declined=declined,
            confirmed_gain=confirmed_gain,
            notes=save_notes,
        )
        self.say(
            f"Saved for {power_words(power)}: {_describe_config(chosen)}. "
            "Run `start` to use it."
        )
        return _finish(
            base,
            final_configuration=chosen,
            rounds=rounds,
            proposed_changes=changes,
            confirmation=confirmation,
            decisions=decisions,
            recheck=recheck,
            saved_setup=saved,
            outcome=outcome,
            not_reached=not_reached,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    def _confirm_share(self) -> float:
        share = float(self.settings.tune_confirm_share)
        return min(max(share, 0.05), 0.9)

    def _starting_point(self, power: PowerState) -> tuple[RuntimeConfiguration, str]:
        same = self.store.find(
            self.model_path, context_length=self.context_length, power=power
        )
        other_power = PowerState.BATTERY if power is PowerState.AC else PowerState.AC
        other = same or self.store.find(
            self.model_path, context_length=self.context_length, power=other_power
        )
        if other is not None:
            saved = other.configuration
            # Every saved setting is carried over; the context size is always the
            # one given for this run.
            config = saved.model_copy(update={"context_length": self.context_length})
            where = "your saved setup" if same is not None else (
                f"your saved setup for {power_words(other_power)}"
            )
            return config, where
        fit = layers_that_fit(
            model=self.model,
            context_length=self.context_length,
            total_graphics_memory_bytes=self.machine.total_graphics_memory_bytes,
            used_before_loading_bytes=self.machine.used_graphics_memory_bytes,
            policy=self.search_policy,
            facts=read_model_facts(self.model_path, self.model),
        )
        layers = fit.layers if fit.layers is not None else 0
        source = (
            f"{layers} graphics card layers fit, from the model file"
            if fit.layers is not None
            else "graphics card room unknown, so starting with no graphics card layers"
        )
        return (
            RuntimeConfiguration(context_length=self.context_length, gpu_offload_layers=layers),
            source,
        )

    def _try_load_with(self, base: RuntimeConfiguration) -> Callable[[int], LoadAttempt] | None:
        """The load-only try with every other setting of ``base`` (placement overrides,
        conversation-memory formats), so the top reflects the memory they free."""
        if self.try_load is None:
            return None
        import inspect

        try_load = self.try_load
        try:
            takes_base = len(inspect.signature(try_load).parameters) >= 2
        except (TypeError, ValueError):
            takes_base = False
        if not takes_base:
            return try_load
        return lambda layers: try_load(layers, base)

    def _layer_round(
        self,
        current: RuntimeConfiguration,
        start_measurement: ExperimentMeasurement,
        watch: DisturbanceWatch,
        deadline: float,
        *,
        revisit: bool = False,
    ) -> tuple[dict[str, object], ProposedChange | None, ExperimentMeasurement | None]:
        """Pick the graphics card layer count with the replay-tested layer search.

        The top (largest count that loads) is found by load-only tries, then
        :func:`search_layers` measures a spread of counts -- per layer kind for mixed
        models -- and the chosen count is judged against the starting one with the
        existing rules. The context size is fixed throughout.
        """
        current_layers = int(current.gpu_offload_layers or 0)
        performance_first = self.rules.effort.mode is OptimizationMode.PERFORMANCE_FIRST
        runs = max(int(self.settings.tune_layer_runs), 1)
        run_policy = self.experiment_policy.model_copy(
            update={
                "measured_repetitions": runs,
                "max_repetitions": runs,
                "min_valid_samples": runs,
                "max_total_runs": runs + self.experiment_policy.warmup_repetitions + 2,
            }
        )
        now = datetime.now(timezone.utc)
        used = self.graphics_memory_reader()
        envelope = build_resource_envelope(
            total_vram_bytes=self.machine.total_graphics_memory_bytes,
            used_vram_bytes=used,
            telemetry_available=used is not None,
            telemetry_observed_at=now if used is not None else None,
            telemetry_source="tune: graphics memory read with the starting setup loaded",
            now=now,
            model=self.model,
            context_length=self.context_length,
            current_layers=current_layers,
            total_layers=self.model.n_layers,
            policy=self.search_policy,
        )
        # Counts run from 0 (nothing on the card) to everything on the card; with the
        # engine's rule that is one more than the number of blocks (tune/placement.py).
        rule = engine_rule(self.machine.program_version)
        if not rule_verified(self.machine.program_version):
            self.say(
                "  note: this engine build's layer order has not been checked on a real "
                "load (scripts/check_placement.py); assuming the output step goes on the "
                "card first, as in build 11139"
            )
        count_max = max_count(int(self.model.n_layers), rule)
        estimate: int | None = None
        if envelope.sufficient and envelope.feasible_max is not None:
            estimate = max(current_layers, int(envelope.feasible_max))
            if self.model.n_layers > 0:
                estimate = min(estimate, count_max)
        top_result = find_layer_top(
            n_layers=count_max,
            estimate=estimate,
            try_load=self._try_load_with(current),
            memory_readable=self.machine.total_graphics_memory_bytes is not None,
            say=self.say,
        )
        top = max(0, int(top_result.top))
        if self.model.n_layers > 0:
            top = min(top, count_max)
        top_reasons = list(top_result.reasons)
        n_layers = count_max
        # Safety net: the starting setup itself loaded and was measured with layers, so
        # load-only tries that say "nothing above 0 fits" cannot be trusted. Fall back
        # to the formula estimate, say so, and keep the reason.
        tries_ignored = top == 0 and current_layers > 0 and bool(top_result.tries)
        if tries_ignored:
            fallback = estimate if estimate is not None else current_layers
            if n_layers > 0:
                fallback = min(fallback, n_layers)
            top = max(int(fallback), 0)
            top_reasons.append(
                f"the load-only tries failed; using the estimate of {top} layers "
                f"(the starting {current_layers} layers loaded and was measured)"
            )
            self.say(f"  the load-only tries failed; using the estimate of {top} layers")
        if revisit and current_layers > top:
            # Looking again from a setup that was measured and kept: it loads, so the
            # top is at least its count (a correction is never made on a revisit).
            top_reasons.append(f"top raised to the kept setup's {current_layers} layers")
            top = current_layers
        start_above_top = current_layers > top
        baseline_layers = min(current_layers, top)
        if baseline_layers != current_layers:
            self.say(
                f"  the starting {current_layers} layers is above the {top} that load; "
                f"using {top}"
            )
        try:
            block_kinds = full_layer_kinds_from_gguf(self.model_path)
        except Exception:  # noqa: BLE001 - an unreadable model file just means no kinds
            block_kinds = None
        # The kind of the piece each count adds, output step included, so the search
        # tests "only the output step on the card" (count 1) as its own group.
        kinds = card_kinds(block_kinds, int(self.model.n_layers))

        inner = RuntimeConfigurationPointEvaluator(
            profiler=self.harness,
            model=self.model,
            workload=self.workload,
            experiment_policy=run_policy,
            context_length=self.context_length,
            envelope=envelope,
            base_configuration=current,
            paging_per_sec_ceiling=self.search_policy.paging_per_sec_ceiling,
            deadline=None if performance_first else deadline,
            # No run-to-run spread limit: every layer decision is made by the fair block
            # check, which handles noise itself. The old 10% limit threw away the whole
            # layer round on Granite 4.0 H Tiny (job 104: short-question spread 10.4%).
            max_primary_cv=None,
        )
        evaluator = RobustPointEvaluator(
            DisturbanceAwareEvaluator(inner, watch, self.harness),
            max_attempts=self.search_policy.max_measure_attempts,
        )
        record: dict[str, object] = {
            "round": "graphics card layers",
            "card_id": LAYERS_CARD_ID,
            "top": top,
            "top_tries": top_result.tries,
            "top_reasons": top_reasons,
            "top_used_estimate": top_result.used_estimate,
            "tries_ignored": tries_ignored,
            "estimated_top": estimate,
            "baseline_layers": baseline_layers,
            "layer_kinds": {
                "known": block_kinds is not None,
                "count": None if block_kinds is None else len(set(block_kinds)),
            },
            "placement_rule": rule,
            "placement_rule_checked": rule_verified(self.machine.program_version),
            "largest_count": count_max,
            "effort_mode": self.rules.effort.mode.value,
            "time_limit_applied": not performance_first,
        }

        baseline = evaluator.measure(baseline_layers)
        if not baseline.ok or baseline.measurement is None:
            record["reason"] = (
                f"the starting {baseline_layers} layers could not be measured "
                f"({baseline.status.value})"
            )
            record["tried"] = [
                {"layers": baseline_layers, "status": baseline.status.value}
            ]
            return record, None, None
        baseline_measurement = baseline.measurement

        tried: list[dict[str, object]] = [
            {"layers": baseline_layers, "status": "starting point"}
        ]
        last_measurement: dict[int, ExperimentMeasurement] = {
            baseline_layers: baseline_measurement
        }
        state = {"tests": 0}
        started_layer_round = time.perf_counter()

        def measure_count(count: int) -> float | None:
            config = inner.configuration_for(count)
            check_context_unchanged(current, config, card_id=LAYERS_CARD_ID)
            self.say(f"  trying {count} layers")
            point = evaluator.measure(count)
            state["tests"] += 1
            entry: dict[str, object] = {
                "layers": count,
                "status": point.status.value,
                "safe": point.safe,
            }
            if point.ok and point.measurement is not None:
                judged0 = judge_pair(
                    before=baseline_measurement,
                    after=point.measurement,
                    workload=self.workload,
                    model=self.model,
                    policy=self.experiment_policy,
                    settings=self.settings,
                    rules=self.rules,
                    card_id=LAYERS_CARD_ID,
                )
                entry["gain"] = judged0.gain
                entry["better"] = (
                    judged0.technically_better
                    and not judged0.guard_breaches
                    and judged0.gain is not None
                    and judged0.gain > 0.0
                )
            tried.append(entry)
            if state["tests"] == 3 and performance_first:
                elapsed = max(time.perf_counter() - started_layer_round, 0.0)
                guess = elapsed / state["tests"] * 24.0
                self.say(
                    f"  {state['tests']} tests done; a rough guess is about "
                    f"{guess:.0f}s more"
                )
            if not point.ok or point.measurement is None:
                return None
            last_measurement[count] = point.measurement
            return _ranking_value(point.measurement, baseline_measurement, self.rules.goal)

        if performance_first:
            should_stop = lambda: False
        else:
            should_stop = lambda: time.perf_counter() >= deadline
        search_settings = LayerSearchSettings(
            tie_margin=float(self.settings.tune_layer_tie_margin)
        )
        result = search_layers(
            top=top,
            n_layers=count_max,
            layer_kinds=kinds,
            measure=measure_count,
            should_stop=should_stop,
            settings=search_settings,
        )
        if not result.complete:
            self.say(
                "  the layer search reached the time limit and is not sure; keeping the "
                "best count found so far"
            )
        record["tried"] = tried
        record["tests"] = result.tests
        record["complete"] = result.complete
        record["not_measured"] = result.not_measured
        record["failed"] = result.failed
        record["chosen"] = result.chosen
        record["reason"] = result.reason
        record["readings"] = {str(k): v for k, v in result.readings.items()}
        record["scores"] = {
            str(k): layer_score(v, search_settings) for k, v in result.readings.items()
        }

        start_change_reason = (
            "the starting count does not leave enough graphics memory free"
        )
        # A start above the top is a memory-safety correction: it moves the start down
        # to the largest count that leaves memory free and makes no speed claim. Any
        # speed claim for a layer count is then shown against that safe count, in the
        # same fair check as every other change (review, 4 Oct 2026: one route).
        safety: ProposedChange | None = None
        reference_layers = current_layers
        reference = current
        if start_above_top and baseline_layers != current_layers:
            safety = ProposedChange(
                change_id=f"{LAYERS_CARD_ID}:safety:{baseline_layers}",
                card_id=LAYERS_CARD_ID,
                setting="gpu_offload_layers",
                description="Graphics card layers (memory safety)",
                old_value=current_layers,
                new_value=baseline_layers,
                measured_gain=None,
                evidence=[
                    start_change_reason,
                    f"memory-safety correction to {baseline_layers} layers; no speed claim",
                    f"top={top}",
                ],
            )
            record["safety_correction"] = baseline_layers
            reference_layers = baseline_layers
            reference = inner.configuration_for(baseline_layers)
        if not revisit:
            self._layer_safety_change = safety
        record["revisit"] = revisit
        if result.chosen is None:
            record["reason"] = result.reason + " (no count could be measured)"
            return record, None, None
        chosen = int(result.chosen)
        if chosen == reference_layers:
            record["reason"] = (
                "no other layer count was better than the starting point"
                if safety is None
                else f"no count was better than the memory-safe {reference_layers}"
            )
            return record, None, None
        measurement = last_measurement.get(chosen)
        if measurement is None:
            record["reason"] = "the chosen count was not measured"
            return record, None, None
        # Fair check: the chosen count and the (safe) starting count measured in
        # blocks, never against the one reading taken at the very start.
        self.say(f"  checking {chosen} layers against {reference_layers} in blocks")
        fair = compare_fairly(
            harness=self.harness,
            model=self.model,
            workload=self.workload,
            start=reference,
            candidate=inner.configuration_for(chosen),
            policy=self._path_policy(self.workload),
            settings=self.settings,
            rules=self.rules,
            watch=watch,
            card_id=LAYERS_CARD_ID,
            phase="tune-layers-fair",
            deadline=None if performance_first else deadline,
            on_step=lambda step: self.say(f"    {step}"),
        )
        self.say("  " + fair_words(fair))
        self._note_heat(f"graphics card layers ({chosen} vs {reference_layers})", fair)
        record["fair"] = fair.record()
        record["gain"] = fair.gain
        if not fair.kept:
            record["reason"] = (
                f"the chosen count did not hold up against {reference_layers} layers "
                "measured in blocks"
            )
            return record, None, None
        record["selected"] = chosen
        change = ProposedChange(
            change_id=f"{LAYERS_CARD_ID}:{chosen}",
            card_id=LAYERS_CARD_ID,
            setting="gpu_offload_layers",
            description="Graphics card layers",
            old_value=reference_layers,
            new_value=chosen,
            measured_gain=fair.gain,
            evidence=[
                str(record["reason"]),
                f"top={top}",
                f"tests={result.tests}",
                "confirmed in blocks: " + fair_words(fair),
            ],
        )
        return record, change, measurement

    def _run_heat_check(
        self, config: RuntimeConfiguration, watch: DisturbanceWatch, runs: int
    ) -> HeatCheck:
        workload = steady_workload(
            str(self.settings.tune_heat_check_workload), default=self.workload
        )
        self.say(
            f"Heat check: {runs} answers in a row with the chosen setup "
            f"({workload.workload_id}), to see whether the laptop slows down from heat"
        )
        policy = self.experiment_policy.model_copy(
            update={
                "measured_repetitions": runs,
                "max_repetitions": runs,
                "min_valid_samples": runs,
                "max_total_runs": runs + self.experiment_policy.warmup_repetitions + 2,
            }
        )
        started = time.perf_counter()
        measurement = measure_steady(
            harness=self.harness,
            model=self.model,
            workload=workload,
            config=config,
            policy=policy,
            phase="tune-heat-check",
            watch=watch,
        )
        check = analyse_heat(
            measurement,
            tolerance=float(self.settings.tune_fair_clock_tolerance),
            seconds=time.perf_counter() - started,
        )
        self.say("  " + check.words)
        return check

    def _path_policy(self, workload: BenchmarkWorkload | None = None) -> ExperimentPolicy:
        """A fixed number of measured runs per measurement (no early stop), so the middle
        value is not moved by one slow run (e.g. the first after a load).

        ``tune_path_runs`` unless a run count was set for this question from the current
        setup's own run-to-run noise (``_set_runs_from``) before any comparison on it.
        """
        runs = max(int(self.settings.tune_path_runs), 1)
        if workload is not None:
            runs = self._runs_by_question.get(workload.workload_id, runs)
        return self.experiment_policy.model_copy(
            update={
                "measured_repetitions": runs,
                "max_repetitions": runs,
                "min_valid_samples": runs,
                "max_total_runs": runs + self.experiment_policy.warmup_repetitions + 2,
            }
        )

    def _set_runs_from(
        self,
        workload: BenchmarkWorkload,
        measurement: ExperimentMeasurement,
        wall_s: float | None = None,
    ) -> int:
        """Choose the runs per measurement for a question, once, before comparing on it.

        Laptop, 4 Oct 2026: on short questions most of the noise is between runs of one
        load (first word 5-7%, writing 2-5%), not between loads (1-2%), so more runs
        per measurement cut the block spread (Gemma 3.8% -> 1.6% with 15 runs, 9B
        2.5% -> 1.8% with 10). But runs cost time: on Gemma's long question one run
        takes ~8 s, and 20 runs made one comparison take ~2 hours.

        So the count is the one that gives the most certainty per minute: a block's
        spread is about sqrt(between^2 + within^2 / runs) and it costs about
        2 x (load + runs x run time), which is best at
        runs = (within / between) x sqrt(load / run time). ``within`` is the goal
        score's run-to-run spread from this measurement's own runs; ``between`` is the
        load-to-load spread measured on the laptop (``tune_between_load_spread``); load
        and run times come from this measurement. Kept between ``tune_path_runs`` and
        ``tune_path_max_runs``, and fixed for the rest of the tune.
        """
        key = workload.workload_id
        if key in self._runs_by_question:
            return self._runs_by_question[key]
        low = max(int(self.settings.tune_path_runs), 1)
        high = max(int(self.settings.tune_path_max_runs), low)
        within = run_score_spread(measurement, self.rules.goal.first)
        saved = self._noise_store().estimate(
            noise_key(
                self.model.name,
                measurement.configuration.gpu_offload_layers,
                key,
                noise_variant(measurement.configuration),
            )
        )
        between = saved if saved is not None else float(self.settings.tune_between_load_spread)
        # Review 4: the time of one more answer includes what recurs per answer
        # (telemetry, gaps), and only the rest is the one-off load cost.
        run_s, load_s = answer_costs(measurement, wall_s)
        if run_s is None:
            run_s = measurement_value(measurement, "total_latency_s")
            if wall_s is not None and run_s:
                done = measurement.valid_runs + measurement.warmup_runs + measurement.invalid_runs
                load_s = max(wall_s - done * run_s, 1.0)
        runs = low
        if within is not None and run_s and load_s is not None and load_s > 0:
            runs = choose_runs(
                within=within, between=between, load_s=load_s, run_s=run_s, low=low, high=high
            )
        self._runs_by_question[key] = runs
        detail = []
        if within is not None:
            detail.append(f"run-to-run spread {within:.1%}")
        detail.append(
            f"load-to-load spread {between:.1%}"
            + (" from earlier comparisons" if saved is not None else " (fallback value, not measured here)")
        )
        if run_s:
            detail.append(f"{run_s:.1f} s per answer")
        if load_s is not None:
            detail.append(f"about {load_s:.0f} s per load")
        self.say(
            f"  {key}: {runs} runs per measurement"
            + (f" ({', '.join(detail)})" if detail else "")
        )
        return runs

    def _noise_store(self) -> NoiseStore:
        return NoiseStore(self.settings.data_dir / "load_spread.json")

    def _learn_load_spread(self, workload: BenchmarkWorkload, fair: FairResult) -> None:
        """Save the load-to-load spread the unchanged setup showed in a comparison."""
        if not fair.start_runs:
            return
        spread = between_load_spread(fair.start_runs, self.rules.goal.first)
        if spread is None:
            return
        config = fair.start_runs[0].configuration
        self._noise_store().add(
            noise_key(self.model.name, config.gpu_offload_layers, workload.workload_id, noise_variant(config)),
            spread,
        )

    def _diagnose(
        self,
        current: RuntimeConfiguration,
        watch: DisturbanceWatch,
        label: str,
        deadline: float | None = None,
    ) -> tuple[Diagnosis | None, dict[str, object]]:
        """Where one written word's time goes in ``current`` (``tune/diagnose.py``).

        A normal measurement (timers off) of the short question gives the time per word
        and to the first word; one run with the engine's timer on gives the card's
        share in that timed run (when a probe runner is available); the memory test
        gives a rough ceiling. Nothing new is started once ``deadline`` has passed
        (review 4, 4 Oct 2026).
        """
        if deadline is not None and time.perf_counter() >= deadline:
            self.say(f"Diagnosis ({label}) skipped: no time left.")
            return None, {"round": f"diagnosis ({label})", "skipped": "no time left"}
        self.say(f"Diagnosis ({label}): where does the time of one written word go?")
        reading = measure_steady(
            harness=self.harness,
            model=self.model,
            workload=self.workload,
            config=current,
            policy=self._path_policy(self.workload),
            phase=f"tune-diagnosis-{label}",
            watch=watch,
            deadline=deadline,
        )
        decode = measurement_value(reading, "decode_tokens_per_sec")
        word_s = 1.0 / decode if decode else None
        first_word_s = measurement_value(reading, "ttft_s")
        reading_s = measurement_value(reading, "prefill_latency_s")
        probe: ProbeResult | None = None
        card = bool(self._machine_facts_now().card) and bool(current.gpu_offload_layers)
        time_left = deadline is None or time.perf_counter() < deadline
        if card and self.probe_runner is not None and not self._probes_unsafe and time_left:
            try:
                probe = (
                    self.probe_runner(current, deadline=deadline)
                    if deadline is not None
                    else self.probe_runner(current)
                )
            except ProbeNotSafe as exc:
                self._probes_unsafe = True
                self.say(f"  {exc}; card share unknown")
            except Exception as exc:  # noqa: BLE001 - a failed probe only means fewer facts
                self.say(f"  the timed run failed ({type(exc).__name__}); card share unknown")
        card_word_s: float | None = 0.0 if not card else None
        if probe is not None and probe.ready:
            self._last_probe = probe
            if probe.engine_seen:
                self._engine_name = probe.engine_seen
            share = probe.card_share()
            if share is not None and word_s:
                # The timer slows the card's work, so its share of the timed run's own
                # word time is applied to the normal word time (an estimate).
                card_word_s = share * word_s
        ceiling = None
        if self.memory_speed_reader is not None:
            try:
                speed = self.memory_speed_reader()
                ceiling = speed.best_bytes_per_s if speed is not None else None
            except Exception:  # noqa: BLE001 - unknown ceiling only means fewer facts
                ceiling = None
        facts = self._model_facts()
        rule = engine_rule(self.machine.program_version)
        result = diagnose(
            word_s=word_s,
            first_word_s=first_word_s,
            reading_s=reading_s,
            card_word_s=card_word_s,
            processor_bytes=processor_bytes_per_word(
                facts, current.gpu_offload_layers, card=card, rule=rule,
                overrides=current.placement_overrides,
            ),
            memory_ceiling=ceiling,
            card=card,
            processor_share_of_work=self._share(current),
            uses_experts=facts.uses_experts,
            hybrid=bool(facts.layer_kinds and len(set(facts.layer_kinds)) > 1),
        )
        for line in result.findings:
            self.say(f"  {line}")
        for name, (priority, why) in result.priorities.items():
            self.say(f"  {name}: {priority} ({why})")
        record: dict[str, object] = {
            "round": f"diagnosis ({label})",
            **result.record(),
            "probe": probe.record() if probe is not None else None,
        }
        return result, record

    def _switches_to_check(self, current: RuntimeConfiguration) -> list[tuple[str, str, tuple[str, ...]]]:
        """(full switch text to try, label, evidence channels) for every listed switch
        not set yet (its conflicts removed), and for every listed switch that is set,
        putting it back to the engine's default."""
        from inferenceos.backends.engine_env import parse_switches
        from inferenceos.tune.engine_switches import (
            apply_switch,
            remove_switch,
            spec_named,
            switches_for,
        )

        engine, version = self._engine_name or "", self.machine.program_version
        features = self._last_probe.device_features() if self._last_probe is not None else None
        now = parse_switches(current.engine_switches)
        todo: list[tuple[str, str, tuple[str, ...]]] = []
        for spec in switches_for(engine, version, features):
            if now.get(spec.name) == spec.value:
                continue
            text = apply_switch(current.engine_switches, spec) or ""
            todo.append((text, f"{spec.label} ({spec.text})", spec.evidence))
        for name in sorted(now):
            known = spec_named(engine, version, name)
            if known is None:
                continue
            text = remove_switch(current.engine_switches, name) or ""
            todo.append((text, f"back to the engine's default for {name}", known.evidence))
        return todo

    def _check_switches(
        self, current: RuntimeConfiguration, depth: str, deadline: float | None = None
    ) -> dict[str, object]:
        """Timed checks of the listed engine switches before any speed test.

        Owner and review 4 (4 Oct 2026): a switch goes to the speed test only when the
        engine's own output shows it took effect (``engine_switches.activation``; a
        timing change is not evidence) and its answers agree with the unchanged setup
        on a fixed set of questions (``answers_agree``). A switch whose effect cannot be
        seen is reported "activation unverified; not tested". A quick probe checks the
        first switches only; the rest are checked if the probe looks better.
        """
        from inferenceos.tune.engine_probe import CHECK_PROMPTS
        from inferenceos.tune.engine_switches import switches_for

        record: dict[str, object] = {"round": "engine switch checks"}
        self._checked_switches = ()
        self._unchecked_switches = []
        self._switch_base = None
        self._switch_checks = []
        if self.probe_runner is None or self._probes_unsafe:
            record["reason"] = "no timed runs available"
            return record
        if not switches_for(self._engine_name or "", self.machine.program_version):
            record["reason"] = "no switches listed for this engine build"
            return record
        todo = self._switches_to_check(current)
        first, rest = (todo[:PROBE_CANDIDATES], todo[PROBE_CANDIDATES:]) if depth == PROBE else (todo, [])
        if deadline is not None and time.perf_counter() >= deadline:
            self._unchecked_switches = todo
            record["reason"] = "no time left for the checks"
            record["not_checked"] = [label for _, label, _ in todo]
            return record
        self.say("Engine switch checks (timed runs; never used for speed):")
        try:
            first_base = self.probe_runner(
                current, switch_check=True, prompts=CHECK_PROMPTS, n_probs=SWITCH_PROBS,
                deadline=deadline,
            )
            # Review 5: the deadline is checked before every timed run, not once.
            if deadline is not None and time.perf_counter() >= deadline:
                self._unchecked_switches = todo
                record["reason"] = "no time left for the checks"
                record["not_checked"] = [label for _, label, _ in todo]
                return record
            base = (
                first_base,
                self.probe_runner(
                    current, switch_check=True, prompts=CHECK_PROMPTS, n_probs=SWITCH_PROBS,
                    deadline=deadline,
                ),
            )
        except ProbeNotSafe as exc:
            self._probes_unsafe = True
            record["reason"] = str(exc)
            return record
        except Exception as exc:  # noqa: BLE001 - no checks means no candidates
            record["reason"] = f"the timed run failed ({type(exc).__name__})"
            return record
        if not (base[0].complete and base[1].complete):
            record["reason"] = "the timed runs of the current setup were incomplete"
            record["base_errors"] = [base[0].error, base[1].error]
            return record
        self._switch_base = base
        record["evidence_channels"] = sorted(set(base[0].evidence) & set(base[1].evidence))
        self._unchecked_switches = list(rest)
        self._run_switch_checks(current, first, deadline)
        record["checks"] = self._switch_checks
        checked: tuple[tuple[str, str], ...] = self._checked_switches or ()
        passed_labels = [label for _, label in checked]
        record["passed"] = passed_labels
        record["not_checked"] = [item[1] for item in self._unchecked_switches]
        if self._unchecked_switches:
            self.say(
                f"  not checked yet ({len(self._unchecked_switches)}): "
                + ", ".join(label for _, label, _ in self._unchecked_switches)
            )
        return record

    def _run_switch_checks(
        self,
        current: RuntimeConfiguration,
        items: list[tuple[str, str, tuple[str, ...]]],
        deadline: float | None,
    ) -> None:
        """Check ``items`` against the unchanged timed runs; passed ones become
        candidates for the speed test. Items left when time runs out stay unchecked."""
        from inferenceos.tune.engine_probe import CHECK_PROMPTS
        from inferenceos.tune.engine_switches import SHOWN, activation, answers_agree

        assert self._switch_base is not None and self.probe_runner is not None
        base, base_again = self._switch_base
        passed = list(self._checked_switches or ())
        for index, (text, label, channels) in enumerate(items):
            if self._probes_unsafe or (deadline is not None and time.perf_counter() >= deadline):
                self._unchecked_switches = list(items[index:]) + self._unchecked_switches
                break
            config = current.model_copy(update={"engine_switches": text or None})
            entry: dict[str, object] = {"switch": label, "setup": text or "engine defaults"}
            if not channels:
                # Nothing this engine prints can show the switch took effect, so a run
                # could never let it through: skip the run (LFM2 tune 099, 5 Oct 2026).
                status, why_effect = activation(channels, base, base_again, base)
                entry.update(activation=status, effect=why_effect, passed=False)
                self._switch_checks.append(entry)
                self.say(f"  {label}: left out ({why_effect})")
                continue
            try:
                probe = self.probe_runner(
                    config, switch_check=True, prompts=CHECK_PROMPTS, n_probs=SWITCH_PROBS,
                    deadline=deadline,
                )
            except ProbeNotSafe as exc:
                self._probes_unsafe = True
                entry.update(passed=False, why=str(exc))
                self._switch_checks.append(entry)
                self._unchecked_switches = list(items[index + 1 :]) + self._unchecked_switches
                break
            except Exception as exc:  # noqa: BLE001 - a failed check drops the switch
                entry.update(passed=False, why=f"the timed run failed: {exc}")
                self._switch_checks.append(entry)
                continue
            status, why_effect = activation(channels, base, base_again, probe)
            entry["activation"] = status
            entry["effect"] = why_effect
            card_a, card_b = base.card_us_per_word(), probe.card_us_per_word()
            if card_a and card_b:
                entry["card_ms_per_word"] = [round(card_a / 1000, 3), round(card_b / 1000, 3)]
            if status != SHOWN:
                entry["passed"] = False
                self._switch_checks.append(entry)
                self.say(f"  {label}: left out ({why_effect})")
                continue
            same, why_answers = answers_agree(
                base, probe, prob_tolerance=float(self.settings.tune_switch_prob_tolerance)
            )
            entry["answers"] = why_answers
            entry["passed"] = same
            self._switch_checks.append(entry)
            self.say(
                f"  {label}: " + ("kept for the speed test" if same else "left out")
                + f" ({why_effect}; {why_answers})"
            )
            if same:
                passed.append((text, label))
        self._checked_switches = tuple(passed)

    def _path_names(self) -> list[str]:
        return [name for name in str(self.settings.tune_paths).split(",") if name.strip()]

    # ------------------------------------------------------------------ #
    # Facts the paths look at                                              #
    # ------------------------------------------------------------------ #
    def _model_facts(self) -> ModelFacts:
        if self._facts is None:
            self._facts = read_model_facts(self.model_path, self.model)
        return self._facts

    def _machine_facts_now(self) -> MachineFacts:
        if self._machine_facts is None:
            self._machine_facts = machine_facts(
                self.hardware, self.machine.free_graphics_memory_bytes
            )
        return self._machine_facts

    def _share(self, config: RuntimeConfiguration) -> float | None:
        return processor_share(
            self._model_facts(),
            config.gpu_offload_layers,
            card=self._machine_facts_now().card,
            overrides=config.placement_overrides,
        )

    def _card_budget(self, config: RuntimeConfiguration) -> int | None:
        """Room on the card for model weights with ``config``'s conversation memory:
        total minus what other programs used before loading, the safety headroom, the
        working reserve and the conversation memory (an estimate, as in ``fit``)."""
        from inferenceos.tune.kv_cache import cache_bytes

        total = self.machine.total_graphics_memory_bytes
        used = self.machine.used_graphics_memory_bytes
        if not total or used is None:
            return None
        keys, values = config.effective_kv_types()
        cache = cache_bytes(self._model_facts(), self.model, self.context_length, keys, values).bytes
        if cache is None:
            return None
        headroom = int(round(self.search_policy.safety_headroom_fraction * total))
        reserve = int(round(self.search_policy.runtime_reserve_fraction * total))
        return max(total - used - headroom - reserve - cache, 0)

    def _facts_round(self, current: RuntimeConfiguration) -> dict[str, object]:
        """Read and show what the paths will look at (model design, where it runs,
        the machine)."""
        facts = self._model_facts()
        machine = self._machine_facts_now()
        share = self._share(current)
        layers = current.gpu_offload_layers
        memory = (
            f"memory {machine.memory_mhz} MHz ({machine.memory_modules} module(s))"
            if machine.memory_mhz
            else "memory speed unknown"
        )
        card = (
            "graphics card: "
            + (
                f"{machine.free_card_bytes / 1024**3:.1f} GB free"
                if machine.free_card_bytes is not None
                else "free memory unknown"
            )
            if machine.card
            else "no usable graphics card"
        )
        self.say("Model and machine facts:")
        self.say(f"  model: {facts.design_words()}")
        self.say(
            "  on the processor with "
            + (f"{layers} graphics card layer(s)" if layers is not None else "llama.cpp's layers")
            + f": {share_words(share)}"
        )
        self.say(
            f"  processor: {machine.physical_cores} cores / {machine.logical_cores} threads; "
            f"{memory}; {card}"
        )
        return {
            "round": "facts",
            "model_design": facts.design_words(),
            "facts_from_model_file": facts.from_file,
            "attention_layers": facts.attention_layers,
            "other_layers": dict(facts.other_layers),
            "experts": facts.experts,
            "experts_used": facts.experts_used,
            "graphics_card_layers": layers,
            "processor_share": share,
            "physical_cores": machine.physical_cores,
            "logical_cores": machine.logical_cores,
            "memory_mhz": machine.memory_mhz,
            "graphics_card": machine.card,
            "free_graphics_memory_bytes": machine.free_card_bytes,
        }

    def _path_context(self, current: RuntimeConfiguration) -> PathContext:
        return PathContext(
            hardware=self.hardware,
            model=self.model,
            total_layers=int(self.model.n_layers),
            current=current,
            supported_options=self.supported_options,
            facts=self._model_facts(),
            machine=self._machine_facts_now(),
            processor_share=self._share(current),
            question_tokens=self._long_question_tokens,
            context_length=self.context_length,
            engine_version=self.machine.program_version,
            engine=self._engine_name,
            checked_switches=self._checked_switches,
            card_budget_bytes=self._card_budget(current),
            accepts=self.server_options.accepts if self.server_options is not None else None,
            grouped_placement_probed=bool(self.settings.tune_grouped_placement_probed),
            piece_lines=self._piece_lines,
        )

    def _read_piece_lines(self, config: RuntimeConfiguration) -> None:
        """Read (once per setup) which pieces the engine moves for ``config``'s overrides,
        from a separate load at the engine's debug level (``piece_lines_reader``)."""
        if not config.placement_overrides or self.piece_lines_reader is None:
            return
        key = piece_lines_key(config.gpu_offload_layers, config.placement_overrides)
        if key in self._piece_lines:
            return
        self.say("    reading which pieces the engine moves (one load at its debug log level)")
        try:
            observed = self.piece_lines_reader(config)
        except Exception as exc:  # noqa: BLE001 - no evidence, never a pass
            self.say(f"    could not read them: {type(exc).__name__}: {exc}")
            observed = None
        lines = (observed or {}).get("overridden_tensors")
        self._piece_lines[key] = (
            {str(name): str(buffer) for name, buffer in lines.items()}
            if isinstance(lines, dict) and lines
            else None
        )

    # ------------------------------------------------------------------ #
    # The long test question, sized to the model                          #
    # ------------------------------------------------------------------ #
    def _long_question(
        self, current: RuntimeConfiguration, watch: DisturbanceWatch, deadline: float | None
    ) -> BenchmarkWorkload:
        """The long test question for this model, built once per tune.

        Up to half the context. With ``tune_long_question_tokens`` set, that many pieces
        (a fixed length, for comparing lengths); otherwise cut so one answer takes at
        most ``tune_long_question_seconds`` on this model with the current setup. The
        built-in long prompt is read once to learn this model's reading speed and how
        many pieces one passage is.
        """
        if self._long_question_workload is not None:
            return self._long_question_workload
        base = steady_workload("long-prefill-steady", default=self.workload)
        record: dict[str, object] = {"round": "long test question"}
        half = max(self.context_length // 2 - base.n_predict - 8, 1)
        fixed = int(self.settings.tune_long_question_tokens)
        seconds = float(self.settings.tune_long_question_seconds)
        policy = self._path_policy().model_copy(
            update={
                "measured_repetitions": 2,
                "max_repetitions": 2,
                "min_valid_samples": 2,
                "max_total_runs": 2 + self.experiment_policy.warmup_repetitions + 2,
            }
        )
        self.say("  sizing the long test question for this model")
        reading = measure_steady(
            harness=self.harness,
            model=self.model,
            workload=base,
            config=current,
            policy=policy,
            phase="tune-long-question-size",
            watch=watch,
            deadline=deadline,
        )
        tokens = measurement_value(reading, "prompt_tokens")
        speed = measurement_value(reading, "prompt_tokens_per_sec")
        first_word = measurement_value(reading, "ttft_s")
        if (speed is None or speed <= 0) and tokens and first_word and first_word > 0:
            # llama-server's streamed answers give no reading speed: the time to the
            # first word is reading the question plus one word, so this is a little
            # slow, which keeps the answer within the time limit.
            speed = float(tokens) / float(first_word)
        writing = measurement_value(reading, "decode_tokens_per_sec")
        record.update(
            {
                "built_in_pieces": tokens,
                "reading_speed": speed,
                "writing_speed": writing,
                "half_context": half,
                "rule": (
                    f"fixed {fixed} pieces (at most half the context)"
                    if fixed > 0
                    else f"at most {seconds:.0f} s per answer, at most half the context"
                ),
            }
        )
        workload = base
        target: int | None = int(tokens) if tokens else None
        if reading.valid_runs < policy.min_valid_samples or not tokens or tokens <= 0:
            record["reason"] = "the built-in long question could not be measured; using it"
        else:
            base_tokens = int(tokens)
            if fixed > 0:
                target = min(fixed, half)
            elif speed and speed > 0:
                write_s = (base.n_predict / writing) if writing and writing > 0 else 0.0
                target = min(half, int(speed * max(seconds - write_s, 1.0)))
            else:
                target = base_tokens
            target = max(int(target or base_tokens), 1)
            per_passage = base_tokens / LONG_PROMPT_PASSAGES
            # Whole passages, never more than the target (the rule is "at most").
            passages = max(int(target / per_passage), 1)
            if passages == LONG_PROMPT_PASSAGES:
                target = base_tokens
                record["reason"] = "the built-in long question already fits the rule"
            else:
                workload = long_question_workload(passages, base=base, per_passage=per_passage)
                target = int(passages * per_passage)
                record["reason"] = f"{passages} passages, about {target} pieces"
        record["pieces"] = target
        record["workload"] = workload.workload_id
        if speed and target:
            record["expected_seconds_per_answer"] = round(
                target / speed + ((base.n_predict / writing) if writing else 0.0), 1
            )
        self.say(
            f"  long test question: about {target or '?'} pieces"
            + (
                f" (~{record['expected_seconds_per_answer']} s per answer)"
                if "expected_seconds_per_answer" in record
                else ""
            )
        )
        self._long_question_workload = workload
        self._long_question_tokens = target
        self._long_question_record = record
        return workload

    def _workload_for(
        self,
        workload_id: str | None,
        current: RuntimeConfiguration,
        watch: DisturbanceWatch,
        deadline: float | None,
    ) -> BenchmarkWorkload:
        if workload_id is None or workload_id == self.workload.workload_id:
            return self.workload
        if workload_id == LONG_QUESTION:
            return self._long_question(current, watch, deadline)
        return steady_workload(workload_id, default=self.workload)

    def _path_round(
        self,
        path: OptionPath,
        current: RuntimeConfiguration,
        watch: DisturbanceWatch,
        deadline: float | None,
        *,
        depth: str = NORMAL,
    ) -> tuple[dict[str, object], ProposedChange | None, RuntimeConfiguration | None]:
        """One optimization path: screen changes, then check the best fairly.

        1. Does the path apply to this model on this machine? If not, say why.
        2. Measure the current setup with the path's question (and the question it must
           not lose on, if any), then each candidate; the current setup is measured
           again afterwards, and a candidate must look better than both readings.
        3. Values next to the best (or one step past the edge) are tried too.
        4. The best-looking candidate is measured against the current setup in turns
           (``compare_fairly``) and kept only if the normal keep rule says so and it
           does not lose on the guard question in turns either.
        """
        context = self._path_context(current)
        record: dict[str, object] = {
            "round": path.label,
            "path": path.name,
            "card_id": path.card_id,
            "setting": path.setting,
            "start_value": getattr(current, path.setting),
            "processor_share": context.processor_share,
        }
        applies, why = path.check(context)
        record["applies"] = applies
        record["applies_reason"] = why
        if not applies:
            record["reason"] = f"skipped: {why}"
            self.say(f"  skipped: {why}")
            return record, None, None
        self.say(f"  {why}")

        def time_left() -> bool:
            return deadline is None or time.perf_counter() < deadline

        workload = self._workload_for(path.workload_id, current, watch, deadline)
        # Candidates may depend on the long question's length.
        context = self._path_context(current)
        candidates = path.candidates(context)
        record["depth"] = depth
        if depth == PROBE and len(candidates) > PROBE_CANDIDATES:
            # A cheap probe (the diagnosis gives this path little chance): only the most
            # likely values; it turns into a full search if one of them looks better.
            record["probe_left_out"] = [c.label for c in candidates[PROBE_CANDIDATES:]]
            candidates = candidates[:PROBE_CANDIDATES]
            self.say(
                "  quick probe (the diagnosis gives this path little chance): "
                + ", ".join(c.label for c in candidates)
            )
        notes = path.notes(context) if path.notes is not None else []
        record["candidates"] = [candidate.label for candidate in candidates]
        if notes:
            record["notes"] = notes
            for note in notes:
                self.say(f"  note: {note}")
        if not candidates:
            record["reason"] = "nothing to try" + self._untested_words(path, record)
            return record, None, None
        guard = (
            self._workload_for(path.guard_workload_id, current, watch, deadline)
            if path.guard_workload_id is not None
            else None
        )
        if guard is not None and guard.workload_id == workload.workload_id:
            guard = None
        record["workload"] = workload.workload_id
        record["guard_workload"] = guard.workload_id if guard is not None else None
        path_policy = self._path_policy()
        record["runs_per_measurement"] = path_policy.measured_repetitions

        def measure(config: RuntimeConfiguration, label: str, on: BenchmarkWorkload) -> ExperimentMeasurement:
            return measure_steady(
                harness=self.harness,
                model=self.model,
                workload=on,
                config=config,
                policy=path_policy,
                phase=f"tune-{path.name}-{label}",
                watch=watch,
                deadline=deadline,
            )

        self.say(f"  measuring the current setup ({workload.workload_id})")
        began = time.perf_counter()
        reference = measure(current, "current", workload)
        reference_wall_s = time.perf_counter() - began
        if reference.valid_runs < path_policy.min_valid_samples:
            record["reason"] = "the current setup could not be measured"
            return record, None, None
        record["current_reading"] = _readings(reference)
        record["question_pieces"] = measurement_value(reference, "prompt_tokens")
        record["runs_for_checks"] = self._set_runs_from(workload, reference, reference_wall_s)
        guard_reference: ExperimentMeasurement | None = None
        if guard is not None:
            self.say(f"  measuring the current setup ({guard.workload_id})")
            began = time.perf_counter()
            guard_reference = measure(current, "current-guard", guard)
            guard_wall_s = time.perf_counter() - began
            if guard_reference.valid_runs < path_policy.min_valid_samples:
                guard_reference = None
            else:
                record["current_guard_reading"] = _readings(guard_reference)
                record["runs_for_guard_checks"] = self._set_runs_from(
                    guard, guard_reference, guard_wall_s
                )

        references: list[ExperimentMeasurement] = [reference]
        screened: list[dict[str, object]] = []
        measured: list[
            tuple[Candidate, ExperimentMeasurement, ExperimentMeasurement | None, dict[str, object]]
        ] = []
        tolerance = GUARD_SCREEN_TOLERANCE
        #: Values that could not be measured; never tried again in this round.
        failed: set[int | bool | str] = set()
        #: Everything tried, by candidate identity (value plus settings changed with it).
        tried_keys: set[tuple[object, ...]] = set()

        def score_of(
            main: ExperimentMeasurement, extra: ExperimentMeasurement | None, entry: dict[str, object]
        ) -> float | None:
            gains = [_screen_gain(ref, main, self.rules.goal) for ref in references]
            known = [gain for gain in gains if gain is not None]
            if not known:
                return None
            score = min(known)
            entry["screen_gain"] = score
            entry["main_gain"] = score
            if len(known) > 1:
                entry["screen_gain_vs_each_reading"] = gains
            if extra is not None and guard_reference is not None:
                guard_gain = _screen_gain(guard_reference, extra, self.rules.goal)
                entry["guard_gain"] = guard_gain
                if guard_gain is not None and guard_gain < -tolerance:
                    score = min(score, guard_gain)
                    entry["screen_gain"] = score
                    entry["guard_note"] = (
                        f"slower on the {guard.workload_id if guard else 'other'} question "
                        f"({guard_gain:+.1%})"
                    )
            return score

        def try_candidates(batch: list[Candidate]) -> None:
            for candidate in batch:
                if not time_left():
                    screened.append({"candidate": candidate.label, "result": "time limit reached"})
                    continue
                config = apply_candidate(current, path, candidate)
                check_context_unchanged(current, config, card_id=path.card_id)
                tried_keys.add(candidate.key)
                self.say(f"  trying {candidate.label}")
                main = measure(config, _label_of(candidate), workload)
                entry: dict[str, object] = {
                    "candidate": candidate.label,
                    "value": candidate.value,
                    "readings": _readings(main),
                }
                if candidate.also:
                    entry["also"] = [[name, value] for name, value in candidate.also]
                if main.valid_runs < path_policy.min_valid_samples:
                    entry["result"] = "could not be measured"
                    entry["why"] = list(main.notes[-3:])
                    screened.append(entry)
                    if not candidate.also:
                        failed.add(candidate.value)
                    self.say(
                        f"    {candidate.label}: could not be measured"
                        + (f" ({main.notes[-1]})" if main.notes else "")
                    )
                    continue
                if path.verify is not None:
                    # Only a setting the engine showed it used goes on (owner rules for
                    # every new path: accepted, applies, took effect).
                    self._read_piece_lines(config)
                    status, why_effect = verify_every_run(path.verify, context, config, main)
                    entry["took_effect"] = {"status": status, "why": why_effect}
                    if status != "shown":
                        entry["result"] = "did not take effect" if status == "not seen" else "effect unverified"
                        screened.append(entry)
                        self.say(f"    {candidate.label}: left out ({why_effect})")
                        continue
                extra: ExperimentMeasurement | None = None
                if guard is not None and guard_reference is not None:
                    extra = measure(config, f"{candidate.value}-guard", guard)
                    entry["guard_readings"] = _readings(extra)
                    if extra.valid_runs < path_policy.min_valid_samples:
                        extra = None
                score = score_of(main, extra, entry)
                screened.append(entry)
                if candidate.screen_only:
                    # Measured and reported, never accepted (e.g. a placement kind that
                    # needs a hardware probe first).
                    entry["result"] = "screened only"
                    entry["not_eligible"] = candidate.screen_only
                    self.say(
                        f"    {candidate.label}: "
                        + (f"{score:+.1%} on your goal (first look); " if score is not None else "")
                        + f"not eligible: {candidate.screen_only}"
                    )
                    continue
                measured.append((candidate, main, extra, entry))
                self.say(
                    f"    {candidate.label}: "
                    + (f"{score:+.1%} on your goal (first look)" if score is not None else "?")
                    + (f"; {entry['guard_note']}" if "guard_note" in entry else "")
                )

        def rescore() -> tuple[float, Candidate] | None:
            best: tuple[float, Candidate] | None = None
            for candidate, main, extra, entry in measured:
                score = score_of(main, extra, entry)
                if score is not None and (best is None or score > best[0]):
                    best = (score, candidate)
            return best

        try_candidates(candidates)
        # Measure the current setup again after the candidates; a candidate must look
        # better than both readings, so one slow reading of the current setup (e.g.
        # right after a workload change) cannot make every candidate look better.
        if measured and time_left():
            self.say("  measuring the current setup again")
            again = measure(current, "current-again", workload)
            if again.valid_runs >= path_policy.min_valid_samples:
                record["current_reading_again"] = _readings(again)
                references.append(again)
        best = rescore()
        if depth == PROBE and best is not None and best[0] > 0.0 and record.get("probe_left_out"):
            # The probe found something: look at the rest too (a full search).
            self.say("  the probe looked better; trying the other values too")
            left_out = record["probe_left_out"]
            assert isinstance(left_out, list)
            left = [c for c in path.candidates(context) if c.label in set(left_out)]
            record["probe_became_full"] = True
            candidates = candidates + left
            try_candidates(left)
            best = rescore()
        if (
            path.name == "engine_switches"
            and depth == PROBE
            and best is not None
            and best[0] > 0.0
            and self._unchecked_switches
            and self._switch_base is not None
            and time_left()
        ):
            # The engine probe looked better: check the other listed switches too, then
            # try the ones that pass (review 4: the probe could never reach them).
            self.say("  the probe looked better; checking the other listed switches too")
            items, self._unchecked_switches = self._unchecked_switches, []
            self._run_switch_checks(current, items, deadline)
            record["probe_became_full"] = True
            record["switch_checks_after_probe"] = list(self._switch_checks)
            context = self._path_context(current)
            seen_values = {c.value for c in candidates}
            left = [c for c in path.candidates(context) if c.value not in seen_values]
            candidates = candidates + left
            try_candidates(left)
            best = rescore()
        # Values next to the best one (or one step past the edge of what was tried).
        refine_ok = depth != PROBE or (best is not None and best[0] > 0.0)
        if path.refine_candidates is not None and refine_ok:
            for _ in range(max(int(self.settings.tune_path_refine_steps), 0)):
                if not time_left() or best is None or best[0] <= 0.0:
                    break
                more = path.refine_candidates(context, best[1], tried_keys)
                if not more:
                    break
                record.setdefault("refined", [])
                refined_list = record["refined"]
                assert isinstance(refined_list, list)
                refined_list.extend(candidate.label for candidate in more)
                self.say("  also trying " + ", ".join(candidate.label for candidate in more))
                try_candidates(more)
                best = rescore()
        if path.refine is not None and refine_ok:
            for _ in range(max(int(self.settings.tune_path_refine_steps), 0)):
                if not time_left():
                    break
                current_value = path.current_value(context)
                tried: set[int | bool | str] = {item[0].value for item in measured} | failed
                tried |= {c.value for c in candidates}
                if current_value is not None:
                    tried.add(current_value)
                centre = best[1].value if best is not None and best[0] > 0.0 else current_value
                if centre is None:
                    break
                more = path.refine(context, centre, tried)
                if not more:
                    break
                record.setdefault("refined", [])
                refined = record["refined"]
                assert isinstance(refined, list)
                refined.extend(candidate.label for candidate in more)
                self.say("  also trying " + ", ".join(candidate.label for candidate in more))
                try_candidates(more)
                best = rescore()
        record["screened"] = screened
        if guard is not None:
            self._note_tradeoffs(path, screened, workload, guard, record)
        if best is None:
            record["reason"] = "no candidate could be measured"
            return record, None, None
        best_score, best_candidate = best
        best_config = apply_candidate(current, path, best_candidate)
        record["best_first_look"] = best_candidate.label
        self.say(f"    best first look: {best_candidate.label} {best_score:+.1%}")
        if best_score <= 0.0:
            record["reason"] = (
                f"nothing looked better than the current setup (best: "
                f"{best_candidate.label}, {best_score:+.1%})"
            ) + self._untested_words(path, record)
            return record, None, None
        floor = float(self.settings.tune_path_check_min_first_look)
        if best_score < floor:
            # Too small at first look to be worth a fair check (provisional; chosen by
            # simulation, owner decision pending).
            record["reason"] = (
                f"best first look too small to check ({best_candidate.label}, "
                f"{best_score:+.1%}; needs {floor:+.1%})"
            ) + self._untested_words(path, record)
            return record, None, None
        if not time_left():
            record["reason"] = "time limit reached before the fair check"
            return record, None, None
        check_mode = str(self.settings.tune_path_check)
        record["path_check"] = check_mode
        old = getattr(current, path.setting)
        tried_labels = [entry.get("candidate") for entry in screened]
        if check_mode == "none":
            # Selection only: the fresh final check of the whole setup decides.
            record["selected"] = best_candidate.value
            record["reason"] = f"{best_candidate.label} looked best (checked only in the final check)"
            change = self._path_change(
                path, best_candidate, current, best_config, best_score,
                [
                    f"tried {', '.join(str(label) for label in tried_labels)}",
                    f"measured with the {workload.workload_id} workload",
                    f"first look {best_score:+.1%}; checked in turns only in the final check",
                ],
                record,
            )
            self._change_workloads[change.change_id] = [workload] + ([guard] if guard else [])
            return record, change, best_config
        light: dict[str, Any] = {}
        if check_mode == "light":
            blocks = max(int(self.settings.tune_path_check_blocks), 2)
            light = {
                "min_blocks": blocks,
                "max_blocks": blocks,
                "alpha": float(self.settings.tune_path_check_alpha),
            }
        self.say(f"  checking {best_candidate.label} against the current setup in turns")
        fair = compare_fairly(
            **light,
            harness=self.harness,
            model=self.model,
            workload=workload,
            start=current,
            candidate=best_config,
            policy=self._path_policy(workload),
            settings=self.settings,
            rules=self.rules,
            watch=watch,
            card_id=path.card_id,
            phase=f"tune-{path.name}-fair",
            deadline=deadline,
            on_step=lambda step: self.say(f"    {step}"),
        )
        self.say("  " + fair_words(fair))
        self._note_heat(f"{path.label}: {best_candidate.label}", fair)
        self._learn_load_spread(workload, fair)
        record["fair"] = fair.record()
        record["gain"] = fair.gain
        if not fair.kept:
            reasons = fair.judgement.reasons[-1:] if fair.judgement else []
            record["reason"] = (
                f"{best_candidate.label} did not hold up in turns"
                + (f" ({reasons[0]})" if reasons else "")
            )
            return record, None, None
        evidence_guard: list[str] = []
        if guard is not None:
            if not time_left():
                record["reason"] = (
                    f"time limit reached before checking {best_candidate.label} on the "
                    f"{guard.workload_id} question"
                )
                return record, None, None
            self.say(
                f"  checking {best_candidate.label} does not lose on the {guard.workload_id} "
                "question, in turns"
            )
            guard_fair = compare_fairly(
                **light,
                harness=self.harness,
                model=self.model,
                workload=guard,
                start=current,
                candidate=best_config,
                policy=self._path_policy(guard),
                settings=self.settings,
                rules=self.rules,
                watch=watch,
                card_id=path.card_id,
                phase=f"tune-{path.name}-guard",
                mode="not_worse",
                deadline=deadline,
                on_step=lambda step: self.say(f"    {step}"),
            )
            self.say("  " + fair_words(guard_fair))
            self._note_heat(
                f"{path.label}: {best_candidate.label} on the {guard.workload_id} question",
                guard_fair,
            )
            self._learn_load_spread(guard, guard_fair)
            record["guard_fair"] = guard_fair.record()
            # Must be shown not worse (any loss under the margin), with good answers and
            # no guarded measure (memory, paging, ...) worse; "unsure" fails. A light
            # check only screens (owner, 4 Oct 2026: the short checks only choose; the
            # fresh full final check decides): it drops a candidate shown worse, or whose
            # estimate is already beyond the loss limit, and leaves "unsure" to the final
            # check (laptop job 090: flash attention off, +21% on the long question, was
            # dropped on an unsure -1.8% on the short one after only 4 blocks).
            margin = float(self.settings.tune_guard_margin)
            if check_mode == "light":
                guard_passes = bool(
                    guard_fair.quality_passed
                    and not guard_fair.guard_breaches
                    and not guard_fair.acceptance_blockers()
                    and guard_fair.decision != "worse"
                    and (guard_fair.gain is None or guard_fair.gain > -margin)
                )
                if guard_passes and not guard_fair.not_worse:
                    record["guard_left_to_final_check"] = True
                    self.say(
                        f"  not yet shown not worse on the {guard.workload_id} question "
                        f"({guard_fair.gain:+.1%}); the final check decides"
                        if guard_fair.gain is not None
                        else f"  not yet shown not worse on the {guard.workload_id} question; "
                        "the final check decides"
                    )
            else:
                guard_passes = guard_fair.not_worse
            if (
                not guard_passes
                and guard_fair.quality_passed
                and not guard_fair.guard_breaches
                and not guard_fair.acceptance_blockers()
            ):
                self._add_tradeoff(
                    path, best_candidate, workload, guard,
                    main_gain=fair.gain, guard_gain=guard_fair.gain, source="path check",
                )
            if not guard_passes:
                why = (
                    "; ".join(guard_fair.acceptance_blockers())
                    if guard_fair.acceptance_blockers()
                    else "answers changed"
                    if not guard_fair.quality_passed
                    else "worse on " + ", ".join(guard_fair.guard_breaches)
                    if guard_fair.guard_breaches
                    else guard_fair.decision
                )
                record["reason"] = (
                    f"{best_candidate.label} helped on the {workload.workload_id} question but "
                    f"was not shown to be not worse on the {guard.workload_id} question ({why}"
                    + (
                        f", {guard_fair.gain:+.1%}"
                        if guard_fair.gain is not None
                        else ""
                    )
                    + ")"
                )
                return record, None, None
            evidence_guard.append(
                (
                    f"not shown worse on the {guard.workload_id} question (left to the final "
                    "check): "
                    if record.get("guard_left_to_final_check")
                    else f"shown not worse on the {guard.workload_id} question: "
                )
                + fair_words(guard_fair)
            )
        _, requirement = _candidate_requirement(path, best_candidate)
        record["quality_requirement"] = requirement.record()
        if requirement.known_answers:
            if not time_left():
                record["reason"] = (
                    f"time limit reached before the known-answer questions for {best_candidate.label}"
                )
                return record, None, None
            self.say(f"  known-answer questions for {best_candidate.label} (it can change answers)")
            known = self._known_answers(current, best_config, watch, deadline, path.name)
            record["known_answers"] = known.record()
            self.say(f"    {known.status}: {known.reason}")
            if not known.passed:
                record["reason"] = (
                    f"{best_candidate.label} helped but "
                    + (
                        "got known answers wrong"
                        if known.status == "fail"
                        else "is not eligible for automatic acceptance (not enough answer-quality evidence)"
                    )
                    + f": {known.reason}"
                )
                return record, None, None
            evidence_guard.append(f"known-answer questions ({known.version}): {known.reason}")
        record["selected"] = best_candidate.value
        record["reason"] = f"{best_candidate.label} confirmed in blocks"
        change = self._path_change(
            path, best_candidate, current, best_config, fair.gain,
            [
                f"tried {', '.join(str(label) for label in tried_labels)}",
                f"measured with the {workload.workload_id} workload",
                "confirmed in turns: " + fair_words(fair),
                *evidence_guard,
            ],
            record,
        )
        self._change_workloads[change.change_id] = [workload] + ([guard] if guard else [])
        return record, change, best_config

    def _path_change(
        self,
        path: OptionPath,
        candidate: Candidate,
        current: RuntimeConfiguration,
        config: RuntimeConfiguration,
        gain: float | None,
        evidence: list[str],
        record: dict[str, object],
    ) -> ProposedChange:
        """The proposed change for a path's chosen candidate, with every setting it
        changes together and what the engine showed when it was measured."""
        old = getattr(current, path.setting)
        related = {
            name: value for name, value in candidate.also
            if not _same_value(getattr(current, name), value)
        }
        observed: dict[str, str] = {}
        screened = record.get("screened")
        if isinstance(screened, list):
            for entry in screened:
                if (
                    isinstance(entry, dict)
                    and entry.get("candidate") == candidate.label
                    and isinstance(entry.get("took_effect"), dict)
                ):
                    effect = entry["took_effect"]
                    observed[path.setting] = f"{effect.get('status')}: {effect.get('why')}"
        change = ProposedChange(
            change_id=f"{path.card_id}:{_label_of(candidate)}",
            card_id=path.card_id,
            setting=path.setting,
            description=path.description,
            old_value=old if old is not None else "program default",
            new_value=candidate.value,
            measured_gain=gain,
            evidence=[f"{candidate.label}", *evidence],
            related=related,
            related_old={name: getattr(current, name) for name in related},
            observed=observed,
        )
        _ = config
        return change

    def _known_answers(
        self,
        start: RuntimeConfiguration,
        candidate: RuntimeConfiguration,
        watch: DisturbanceWatch,
        deadline: float | None,
        label: str,
    ) -> KnownAnswerCheck:
        """The known-answer questions (``tune/known_answers.py``) with both setups."""
        policy = self._path_policy().model_copy(
            update={
                "measured_repetitions": 2,
                "max_repetitions": 2,
                "min_valid_samples": 2,
                "max_total_runs": 2 + self.experiment_policy.warmup_repetitions + 2,
            }
        )

        def measure(config: RuntimeConfiguration, workload: BenchmarkWorkload, phase: str) -> ExperimentMeasurement:
            return measure_steady(
                harness=self.harness,
                model=self.model,
                workload=workload,
                config=config,
                policy=policy,
                phase=f"tune-{label}-{phase}",
                watch=watch,
                deadline=deadline,
            )

        # Review 7: each known-answer measurement must show the settings of the setup it
        # stands for (a q8 cache measured as f16 says nothing about q8).
        to_verify = {
            "new": self._settings_to_verify(start, candidate),
            "current": self._settings_to_verify(RuntimeConfiguration(), start),
        }

        def settings_check(config: RuntimeConfiguration, measurement: ExperimentMeasurement) -> tuple[bool, str]:
            side = "new" if config == candidate else "current"
            effects = self._verify_effective(config, [measurement], to_verify[side])
            missing = [item for item in effects if item["status"] != "shown"]
            if missing:
                return False, "; ".join(f"{item['setting']} {item['status']} ({item['why']})" for item in missing)
            return True, "; ".join(f"{item['setting']} shown" for item in effects) or "nothing to verify"

        return run_known_answer_check(
            start=start,
            candidate=candidate,
            context_length=self.context_length,
            measure=measure,
            time_left=lambda: deadline is None or time.perf_counter() < deadline,
            settings_check=settings_check,
        )

    def _note_tradeoffs(
        self,
        path: OptionPath,
        screened: list[dict[str, object]],
        workload: BenchmarkWorkload,
        guard: BenchmarkWorkload,
        record: dict[str, object],
    ) -> None:
        """Candidates that looked clearly better on the path's question but were held
        back by a loss on the question they must not lose on: trade-offs. Tune never
        keeps them on its own (owner, 5 Oct 2026); the best one is measured properly at
        the end and offered for an explicit yes or no."""
        smallest = float(self.rules.effort.minimum_improvement)
        found: list[str] = []
        for entry in screened:
            if entry.get("not_eligible"):
                continue
            main, held_back = entry.get("main_gain"), entry.get("guard_gain")
            if not isinstance(main, float) or not isinstance(held_back, float):
                continue
            if main >= max(smallest, 0.0) and held_back < -GUARD_SCREEN_TOLERANCE:
                value = entry.get("value")
                if not isinstance(value, (int, bool, str)):
                    continue
                also_raw = entry.get("also")
                also = tuple(
                    (str(item[0]), item[1])
                    for item in (also_raw if isinstance(also_raw, list) else [])
                    if isinstance(item, list) and len(item) == 2
                )
                candidate = Candidate(value, str(entry.get("candidate")), also)
                self._add_tradeoff(
                    path, candidate, workload, guard,
                    main_gain=main, guard_gain=held_back, source="first look",
                )
                found.append(str(entry.get("candidate")))
        if found:
            record["tradeoffs"] = found

    def _add_tradeoff(
        self,
        path: OptionPath,
        candidate: Candidate,
        workload: BenchmarkWorkload,
        guard: BenchmarkWorkload,
        *,
        main_gain: float | None,
        guard_gain: float | None,
        source: str,
    ) -> None:
        key = (path.setting, str(candidate.value))
        current = self._tradeoffs.get(key)
        if current is not None and (current["main_gain"] or 0.0) >= (main_gain or 0.0):
            return
        self._tradeoffs[key] = {
            "path": path,
            "candidate": candidate,
            "workload": workload,
            "guard": guard,
            "main_gain": main_gain,
            "guard_gain": guard_gain,
            "source": source,
        }

    def _untested_words(self, path: OptionPath, record: dict[str, object]) -> str:
        """For a quick probe that found nothing: the values never tried, so the report
        does not read as "this path cannot help" (review 4, 4 Oct 2026)."""
        if record.get("depth") != PROBE or record.get("probe_became_full"):
            return ""
        untested: list[str] = []
        left_out = record.get("probe_left_out")
        if isinstance(left_out, list):
            untested.extend(str(label) for label in left_out)
        if path.name == "engine_switches":
            untested.extend(label for _, label, _ in self._unchecked_switches)
        if not untested:
            return ""
        record["untested"] = untested
        return (
            "; quick probe only, not tried: " + ", ".join(untested)
            + " (this does not show they cannot help)"
        )

    def _check_whole_setup(
        self,
        start: RuntimeConfiguration,
        setup: RuntimeConfiguration,
        changes: list[ProposedChange],
        watch: DisturbanceWatch,
        deadline: float | None,
        *,
        label: str,
    ) -> tuple[bool, ConfirmationResult, list[dict[str, object]]]:
        """Check the exact setup against the start on every question its changes used.

        * On each question a change was *found* on, the whole setup must be shown better
          (lower confidence bound above 0), so a gain that faded is not kept.
        * On each question a change was only *guarded* on, it must be shown not worse
          (any loss under ``tune_guard_margin``).
        * Good answers and no guarded measure worse, everywhere.

        Stops at the first question it fails on. The gain reported is the mean measured
        here on the first question checked (tune's own when a change was found on it),
        never multiplied together from the separate checks.
        """
        found: list[BenchmarkWorkload] = []
        guarded: list[BenchmarkWorkload] = []

        def known(items: list[BenchmarkWorkload], workload: BenchmarkWorkload) -> bool:
            return any(item.workload_id == workload.workload_id for item in items)

        for change in changes:
            used = self._change_workloads.get(change.change_id) or [self.workload]
            if not known(found, used[0]):
                found.append(used[0])
            for workload in used[1:]:
                if not known(guarded, workload):
                    guarded.append(workload)
        if not known(found, self.workload) and not known(guarded, self.workload):
            guarded.insert(0, self.workload)
        guarded = [item for item in guarded if not known(found, item)]
        plan = [(item, "better") for item in found] + [(item, "not_worse") for item in guarded]
        checks: list[dict[str, object]] = []
        passed = True
        main: FairResult | None = None
        failed_on = ""
        balanced_guard: dict[str, Any] = {}
        if self.rules.effort.mode is OptimizationMode.BALANCED:
            # Owner, 4 Oct 2026 (simulation: 8 blocks left "not worse" unsure about two
            # times in three, throwing away real gains): Balanced's final "must not lose"
            # checks look at 8 and 16 blocks, fixed in advance; the 5% error allowance
            # is split across both looks. The time limit still applies; unsure or out
            # of time keeps the current setup.
            first, last = (int(v) for v in str(self.settings.tune_final_guard_looks_balanced).split(","))
            balanced_guard = {"min_blocks": first, "max_blocks": last}
        # Review handoff (6 Oct 2026): the answer checks of a combined proposal use the
        # strictest requirement of any change in it, not the first change's card.
        strict_card, requirement = strictest(changes)
        quality_card = strict_card or LAYERS_CARD_ID
        to_verify = self._settings_to_verify(start, setup)
        start_verify = self._settings_to_verify(RuntimeConfiguration(), start)
        for workload, mode in plan:
            need = "better" if mode == "better" else "not worse"
            self.say(f"  on the {workload.workload_id} question (must be {need})")
            looks = (
                self._final_looks.get(workload.workload_id, self._final_looks.get("*", {}))
                if self._final_looks is not None
                else (balanced_guard if mode == "not_worse" else {})
            )
            fair = compare_fairly(
                **looks,
                harness=self.harness,
                model=self.model,
                workload=workload,
                start=start,
                candidate=setup,
                policy=self._path_policy(workload),
                settings=self.settings,
                rules=self.rules,
                watch=watch,
                card_id=quality_card,
                phase=f"tune-{label}-{workload.workload_id}",
                mode="better" if mode == "better" else "not_worse",
                deadline=deadline,
                on_step=lambda step: self.say(f"    {step}"),
                progress=self._final_progress,
            )
            self._whole_setup_fairs.append((workload.workload_id, fair))
            words = fair_words(fair)
            self.say("    " + words)
            self._note_heat(f"{label} check on the {workload.workload_id} question", fair)
            # Final checks run enough fresh loads to learn the load-to-load spread
            # (review 5: the light path checks alone never qualify).
            self._learn_load_spread(workload, fair)
            # "better" uses the full keep rule: shown better, good answers, no guard
            # breach AND the effort mode's smallest worthwhile gain / payback (review,
            # 4 Oct 2026: the final check must not override "not worth it").
            ok = fair.kept if mode == "better" else fair.not_worse
            effects = self._verify_effective(setup, fair.candidate_runs, to_verify)
            # Review 7: the starting setup's fresh loads must show its own settings too
            # (they are half of the evidence).
            effects += [
                {**item, "setting": f"{item['setting']} (starting setup)"}
                for item in self._verify_effective(start, fair.start_runs, start_verify)
            ]
            not_shown = [item for item in effects if item["status"] != "shown"]
            if not_shown:
                ok = False
                words += "; requested setting not shown by the engine: " + "; ".join(
                    f"{item['setting']} ({item['why']})" for item in not_shown
                )
                self.say("    " + words.split("; requested setting")[1].lstrip("; "))
            checks.append(
                {
                    "workload": workload.workload_id,
                    "must_be": need,
                    "passed": ok,
                    "words": words,
                    "baseline": _describe_config(start),
                    "measures": _measure_summary(fair),
                    "quality_card": quality_card,
                    "quality_requirement": requirement.record(),
                    "observed": effects,
                    "fair": fair.record(),
                }
            )
            if main is None:
                main = fair
            if not ok:
                passed = False
                failed_on = f"{workload.workload_id} question ({need}: {fair.decision})"
                break
        if passed and requirement.known_answers:
            self.say(f"  known-answer questions for the whole setup ({requirement.why})")
            known_check = self._known_answers(start, setup, watch, deadline, label)
            checks.append(
                {
                    "workload": "known answers",
                    "must_be": "right",
                    "passed": known_check.passed,
                    "words": f"{known_check.status}: {known_check.reason}",
                    "known_answers": known_check.record(),
                }
            )
            self.say(f"    {known_check.status}: {known_check.reason}")
            if not known_check.passed:
                passed = False
                failed_on = f"known-answer questions ({known_check.status}: {known_check.reason})"
        assert main is not None
        confirmation = main.as_confirmation()
        reason = (
            "the whole setup passed on: "
            + ", ".join(f"{item['workload']} ({item['must_be']})" for item in checks)
            if passed
            else f"the whole setup did not pass on the {failed_on}"
        )
        judgement = confirmation.judgement
        if judgement is not None:
            # The keep flag and the verdict always agree: the main check's own keep
            # decision can only be taken away here, never granted.
            keep = passed and judgement.keep
            verdict = judgement.verdict
            if not keep and verdict is KeepVerdict.KEPT:
                verdict = KeepVerdict.NOT_BETTER
            judgement = judgement.model_copy(
                update={"keep": keep, "verdict": verdict, "reasons": [*judgement.reasons, reason]}
            )
        confirmation = confirmation.model_copy(
            update={"judgement": judgement, "notes": [*confirmation.notes, reason]}
        )
        return passed, confirmation, checks

    def _check_safety_correction(
        self,
        starting: RuntimeConfiguration,
        safe: RuntimeConfiguration,
        reference: ExperimentMeasurement,
        watch: DisturbanceWatch,
        deadline: float | None,
    ) -> tuple[ConfirmationResult, dict[str, object]]:
        """Show a memory-safety correction stays within its resources, runs and answers
        like the starting setup.

        Review 6: a correction makes no speed claim, but it is still a different setup.
        It is measured fresh on the main question; every run must execute and its
        answers must pass the normal answer check against the starting setup's answers
        (which ran and were measured at the start of this tune).

        Review 7: it exists to keep the machine within its resources, so it must also
        show that, freshly (``acceptance_evidence.safety_resource_evidence``): a load
        check that it fits with the safety margins, every resource reading (paging,
        memory, and whatever the goal guards) present on every run, no severe paging,
        and paging not above the starting setup's by more than the usual tolerance. No
        speed comparison with the unsafe start is made. Any setting the engine can show
        must be shown on every run. Anything less is reported as unverified, never as a
        pass.
        """
        record: dict[str, object] = {
            "round": "memory-safety check",
            "from_layers": starting.gpu_offload_layers,
            "to_layers": safe.gpu_offload_layers,
            "workload": self.workload.workload_id,
            "reference": "the starting setup's answers, measured at the start of this tune",
        }
        notes: list[str] = []
        load_check: LoadAttempt | None = None
        try_load = self._try_load_with(safe)
        if try_load is not None and safe.gpu_offload_layers is not None:
            try:
                load_check = try_load(int(safe.gpu_offload_layers))
            except Exception as exc:  # noqa: BLE001 - recorded as evidence, never hidden
                load_check = LoadAttempt(loaded=False, fits=False, error=f"{type(exc).__name__}: {exc}")
        if load_check is not None and not (load_check.loaded and load_check.fits):
            # No need to measure: the correction does not even show it fits.
            measurement = ExperimentMeasurement(
                measurement_id="not-measured", phase="tune-safety-check", configuration=safe
            )
        else:
            measurement = measure_steady(
                harness=self.harness,
                model=self.model,
                workload=self.workload,
                config=safe,
                policy=self._path_policy(self.workload),
                phase="tune-safety-check",
                watch=watch,
                deadline=deadline,
            )
        record["readings"] = _readings(measurement)
        record["failed_runs"] = measurement.invalid_runs
        passed = False
        need = self._path_policy(self.workload).min_valid_samples
        resources = safety_resource_evidence(
            measurement=measurement,
            reference=reference,
            guarded=self.rules.goal.guarded,
            load_check=load_check,
        )
        record["resources"] = {"passed": resources.passed, "why": resources.reasons, **resources.readings}
        effects = self._verify_effective(
            safe, [measurement], self._settings_to_verify(RuntimeConfiguration(), safe)
        ) if measurement.valid_runs else []
        record["observed"] = effects
        not_shown = [item for item in effects if item["status"] != "shown"]
        if load_check is not None and not (load_check.loaded and load_check.fits):
            notes.extend(resources.reasons)
        elif measurement.valid_runs < need:
            notes.append(
                f"the memory-safe setup could not be measured ({measurement.valid_runs} "
                "good runs" + "".join(f"; {note}" for note in measurement.notes[-2:]) + ")"
            )
        elif measurement.invalid_runs:
            notes.append(f"{measurement.invalid_runs} run(s) of the memory-safe setup failed to execute")
        elif not resources.passed:
            notes.extend(f"resources: {reason}" for reason in resources.reasons)
        elif not_shown:
            notes.append(
                "requested setting not shown by the engine: "
                + "; ".join(f"{item['setting']} ({item['why']})" for item in not_shown)
            )
        else:
            quality = validate_quality(
                baseline=reference,
                candidate=measurement,
                workload=self.workload,
                policy=self.experiment_policy,
                settings=self.settings,
                action_id=LAYERS_CARD_ID,
                model=self.model,
            )
            record["answer_check"] = quality.status.value
            passed = quality_ok(quality)
            notes.append(
                "the memory-safe setup fits, stayed within its resources, ran every time and "
                "its answers passed the answer check against the starting setup"
                if passed
                else f"answer check against the starting setup: {quality.status.value}"
            )
        record["passed"] = passed
        record["why"] = notes
        self.say("  " + "; ".join(notes))
        judgement = KeepJudgement(
            verdict=KeepVerdict.KEPT if passed else KeepVerdict.NOT_BETTER,
            keep=passed,
            situation=Situation.BEFORE_LOADING,
            effort_mode=self.rules.effort.mode.value,
            goal_mode=self.rules.goal.mode.value,
            reasons=["memory-safety correction: no speed claim", *notes],
        )
        return (
            ConfirmationResult(
                rounds=1 if measurement.valid_runs else 0,
                order=["memory-safe setup"],
                starting_samples=reference.valid_runs,
                final_samples=measurement.valid_runs,
                quality_passed=passed,
                judgement=judgement,
                notes=notes,
            ),
            record,
        )

    def _settings_to_verify(
        self, start: RuntimeConfiguration, setup: RuntimeConfiguration
    ) -> list[str]:
        """Settings of ``setup`` whose effect the engine can show and that differ from
        ``start`` or are set at all (one entry per verifier)."""
        chosen: list[str] = []
        seen: set[object] = set()
        for setting, verifier in CATALOG.verifiers().items():
            value, before = getattr(setup, setting), getattr(start, setting)
            if value is None and before is None:
                continue
            if verifier in seen:
                continue
            seen.add(verifier)
            chosen.append(setting)
        return chosen

    def _verify_effective(
        self,
        setup: RuntimeConfiguration,
        measurements: list[ExperimentMeasurement],
        settings: list[str],
    ) -> list[dict[str, str]]:
        """What the engine showed for each verifiable setting in every measurement of
        ``setup`` (all must show it)."""
        verifiers = CATALOG.verifiers()
        results: list[dict[str, str]] = []
        if not settings:
            return results
        if "placement_overrides" in settings:
            self._read_piece_lines(setup)
        context = self._path_context(setup)
        for setting in settings:
            verifier = verifiers[setting]
            statuses = [verify_every_run(verifier, context, setup, item) for item in measurements] or [
                ("unverified", "no measurement of the new setup")
            ]
            worst = next((item for item in statuses if item[0] != "shown"), statuses[0])
            results.append({"setting": setting, "status": worst[0], "why": worst[1]})
        return results

    def _save(
        self,
        *,
        run_id: str,
        power: PowerState,
        starting: RuntimeConfiguration,
        final: RuntimeConfiguration,
        approved: list[ProposedChange],
        declined: list[ProposedChange],
        confirmed_gain: float | None,
        notes: list[str],
        guard: CommitGuard | None = None,
    ) -> SavedSetup:
        check_context_unchanged(starting, final, card_id="tune")
        blocked = getattr(self.harness, "monitor_blocked", None)
        if blocked:
            # An experimental scoped sampler's cleanup was not confirmed: publish nothing.
            from inferenceos.telemetry.counters import MonitorCleanupUnresolved

            raise MonitorCleanupUnresolved(f"not saved: {blocked}")
        if guard is not None:
            guard.emit("hashing")  # part of preparation; a user stop abandons it
        identity = model_identity(
            self.model_path, with_hash=self.model_hash, check=guard.check if guard is not None else None
        )
        setup = SavedSetup(
            setup_id=f"setup-{uuid.uuid4().hex[:10]}",
            model=identity,
            model_name=self.model.name,
            context_length=self.context_length,
            power_source=power,
            machine=self.machine,
            configuration=final,
            starting_configuration=starting,
            approved_changes=approved,
            declined_changes=declined,
            confirmed_gain=confirmed_gain,
            effort_mode=self.rules.effort.mode.value,
            goal_mode=self.rules.goal.mode.value,
            tune_report=run_id,
            notes=notes,
        )
        if guard is None:
            self.store.save(setup)
        else:
            self.store.save(setup, guard=guard)
        return setup


def _ranking_value(
    measurement: ExperimentMeasurement,
    baseline: ExperimentMeasurement,
    goal: GoalPreference,
) -> float | None:
    """The value of one layer test on the active goal (prompt B's ranking value).

    A goal that ranks writing speed alone uses each run's writing speed. A goal that
    combines several measures uses ``1 + gain`` of each run against the baseline's
    first run, scored with the goal's own weighting (the same way ``judge_pair``
    scores a comparison).
    """
    runs = list(measurement.run_metrics)
    if not runs:
        return None
    metrics = list(goal.first)
    if len(metrics) == 1 and metrics[0] == "decode_tokens_per_sec":
        values = [
            float(value)
            for value in (getattr(run, "decode_tokens_per_sec", None) for run in runs)
            if value is not None
        ]
        return trimmed_mean(values) if values else None
    baseline_run = baseline.run_metrics[0] if baseline.run_metrics else None
    combined: list[float] = []
    for run in runs:
        gains: dict[str, float] = {}
        for metric in metrics:
            candidate = getattr(run, metric, None)
            base = getattr(baseline_run, metric, None) if baseline_run is not None else None
            if candidate is None or base is None or float(base) == 0.0:
                continue
            relative = (float(candidate) - float(base)) / abs(float(base))
            gains[metric] = improvement(metric, relative)
        assessment = assess_known_gains(gains, goal)
        if assessment.gain is not None:
            combined.append(1.0 + assessment.gain)
    return trimmed_mean(combined) if combined else None


def _final_summary(checks: list[dict[str, object]]) -> list[str]:
    """Plain lines for the final check: per question, the baseline it was compared
    with, then writing speed, first word, the combined goal score (each with its range)
    and the guarded measures, kept apart (reporting correction, 6 Oct 2026)."""
    lines: list[str] = []
    for item in checks:
        measures = item.get("measures")
        if not isinstance(measures, dict):
            continue
        parts: list[str] = []
        for name in ("writing_speed", "first_word", "combined goal score"):
            entry = measures.get(name)
            if isinstance(entry, dict) and isinstance(entry.get("mean"), float):
                low, high = entry.get("lower"), entry.get("upper")
                span = (
                    f" (range {low:+.1%} to {high:+.1%})"
                    if isinstance(low, float) and isinstance(high, float)
                    else ""
                )
                label = {"writing_speed": "writing speed", "first_word": "first word"}.get(name, name)
                parts.append(f"{label} {entry['mean']:+.1%}{span}")
        guarded = measures.get("guarded")
        if isinstance(guarded, dict) and guarded:
            parts.append(
                "guarded: " + ", ".join(
                    f"{name} {'worse beyond its limit' if isinstance(info, dict) and info.get('breach') else 'within its limit'}"
                    for name, info in guarded.items()
                )
            )
        lines.append(
            f"{item.get('workload')} question, compared with {item.get('baseline')}: "
            + "; ".join(parts)
            + " (+ is better; a combined score is not a gain in every measure)"
        )
    return lines


def _measure_summary(fair: FairResult) -> dict[str, object]:
    """Each measure on its own, with its range over the counted blocks, next to the
    combined goal score (reporting correction, 6 Oct 2026: a combined score is not an
    improvement in every measure)."""
    from inferenceos.tune.stats import block_bounds

    out: dict[str, object] = {}
    alpha = fair.test.alpha if fair.test is not None else 0.05
    for metric in ("decode_tokens_per_sec", "ttft_s", "prompt_tokens_per_sec", "peak_rss_bytes"):
        values = [item.improvements[metric] for item in fair.turns if metric in item.improvements]
        if not values:
            continue
        test = block_bounds(values, alpha=alpha)
        out[plain_name(metric)] = {
            "mean": test.mean,
            "lower": test.lower,
            "upper": test.upper,
            "blocks": len(values),
        }
    if fair.test is not None:
        out["combined goal score"] = {
            "mean": fair.test.mean,
            "lower": fair.test.lower,
            "upper": fair.test.upper,
            "blocks": len(fair.block_scores),
        }
    out["guarded"] = {
        plain_name(metric): {"breach": metric in fair.guard_breaches, "readings_in_blocks": count}
        for metric, count in fair.guard_coverage.items()
    }
    return out


def _label_of(candidate: Candidate) -> str:
    """A short, file-name-safe label for a candidate (used in phase names and ids)."""
    import hashlib

    text = str(candidate.value)
    if candidate.also:
        text += "|" + ";".join(f"{name}={value}" for name, value in candidate.also)
    if len(text) <= 24 and all(ch.isalnum() or ch in "_-.=|;" for ch in text):
        return text
    return "c" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _readings(measurement: ExperimentMeasurement) -> dict[str, object]:
    """Every run's main numbers, so a report can be checked later."""
    runs = []
    for run in measurement.run_metrics:
        runs.append(
            {
                "writing_speed": run.decode_tokens_per_sec,
                "first_word_s": run.ttft_s,
                "prompt_speed": run.prompt_tokens_per_sec,
                "load_s": run.model_load_time_s,
                "cpu_speed_percent": run.cpu_performance_percent,
            }
        )
    return {
        "valid_runs": measurement.valid_runs,
        "writing_speed": measurement_value(measurement, "decode_tokens_per_sec"),
        "first_word_s": measurement_value(measurement, "ttft_s"),
        "runs": runs,
        "at": measurement.created_at.isoformat(),
    }


def _screen_gain(
    reference: ExperimentMeasurement,
    candidate: ExperimentMeasurement,
    goal: GoalPreference,
) -> float | None:
    """A first look at a candidate: the goal gain from medians, no noise zeroing."""
    gains: dict[str, float] = {}
    for metric in goal.first:
        base = measurement_value(reference, metric)
        new = measurement_value(candidate, metric)
        if base is None or new is None or base == 0.0:
            continue
        gains[metric] = improvement(metric, (new - base) / abs(base))
    if not gains:
        return None
    return assess_known_gains(gains, goal).gain


def _finish(base: TuneResult, **update: object) -> TuneResult:
    return base.model_copy(update={**update, "finished_at": datetime.now(timezone.utc)})


def _apply_changes(
    starting: RuntimeConfiguration, changes: list[ProposedChange]
) -> RuntimeConfiguration:
    """``starting`` with each change (its own setting and every setting changed with
    it) applied in order; "" means "none" for text settings."""
    update: dict[str, object] = {}
    for change in changes:
        for setting, value in (*change.related.items(), (change.setting, change.new_value)):
            if setting in ("engine_switches", "placement_overrides") and value == "":
                value = None
            update[setting] = value
    return starting.model_copy(update=update)


def _describe_config(config: RuntimeConfiguration) -> str:
    threads = config.threads if config.threads is not None else "program default"
    text = f"{config.gpu_offload_layers} graphics card layers, {threads} threads"
    if config.threads_batch is not None:
        text += f", {config.threads_batch} reading threads"
    if config.engine_switches:
        text += f", engine switches {config.engine_switches}"
    if config.flash_attention is not None:
        text += ", flash attention " + ("on" if config.flash_attention else "off")
    if config.ubatch_size is not None:
        text += f", prompt batch {config.ubatch_size}"
    if config.placement_overrides:
        text += f", pieces placed by hand ({config.placement_overrides})"
    keys, values = config.effective_kv_types()
    if keys is not None or values is not None:
        text += f", conversation memory keys {keys or 'f16'} / values {values or 'f16'}"
    return text + f", context {config.context_length}"


def noise_variant(config: RuntimeConfiguration) -> str:
    """What else, besides the layer family, makes a setup's noise its own."""
    parts: list[str] = []
    if config.placement_overrides:
        import hashlib

        parts.append("ot" + hashlib.sha1(config.placement_overrides.encode("utf-8")).hexdigest()[:8])
    keys, values = config.effective_kv_types()
    if keys not in (None, "f16") or values not in (None, "f16"):
        parts.append(f"kv{keys or 'f16'}-{values or 'f16'}")
    return ",".join(parts)


def _confirmation_words(result: ConfirmationResult) -> str:
    judgement = result.judgement
    if judgement is None:
        return "could not confirm: " + "; ".join(result.notes or ["no complete round"])
    gain = judgement.goal.gain
    gain_text = f"{gain:+.1%}" if gain is not None else "unknown"
    return (
        f"after {result.rounds} alternating round(s): {gain_text} on your goal -> "
        f"{judgement.verdict.value.replace('_', ' ')}"
    )


__all__ = ["GraphicsMemoryReader", "THREADS_CARD_ID", "Tuner", "thread_values_to_try"]
