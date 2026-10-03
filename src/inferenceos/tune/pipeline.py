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

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from inferenceos.cards.builtin.gpu_layers_more import CARD_ID as LAYERS_CARD_ID
from inferenceos.cards.safety import check_context_unchanged
from inferenceos.config import Settings
from inferenceos.contracts.enums import OptimizationMode, PowerState
from inferenceos.contracts.experiment import (
    ExperimentMeasurement,
    ExperimentPolicy,
    RuntimeConfiguration,
)
from inferenceos.contracts.goal import GoalPreference
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
from inferenceos.optimization.goals import assess_known_gains, improvement
from inferenceos.optimization.keep import ModeRules
from inferenceos.optimization.search.evaluation import (
    RobustPointEvaluator,
    RuntimeConfigurationPointEvaluator,
)
from inferenceos.optimization.search.policy import SearchPolicy
from inferenceos.optimization.search.resources import build_resource_envelope
from inferenceos.profiling.workloads import BenchmarkWorkload
from inferenceos.tune.approval import ChangeApprover, describe_change
from inferenceos.tune.fair import FairResult, compare_fairly, fair_words, measurement_value
from inferenceos.tune.heat import HeatCheck, analyse_heat
from inferenceos.tune.fit import layers_that_fit
from inferenceos.tune.layer_search import (
    LayerSearchSettings,
    layer_kinds_from_gguf,
    score as layer_score,
    search_layers,
    trimmed_mean,
)
from inferenceos.tune.layer_top import LoadAttempt, find_layer_top
from inferenceos.tune.measure import (
    DisturbanceAwareEvaluator,
    DisturbanceWatch,
    confirm_by_alternating,
    judge_pair,
    measure_steady,
)
from inferenceos.tune.paths import (
    steady_workload,
    Candidate,
    OptionPath,
    PathContext,
    apply_candidate,
    paths_by_name,
)
from inferenceos.tune.profiles import (
    SetupStore,
    model_identity,
    power_key,
    power_words,
)

THREADS_CARD_ID = "runtime.reduce_threads"

#: Reads the graphics memory in use right now, with the model loaded.
GraphicsMemoryReader = Callable[[], int | None]


def thread_values_to_try(physical_cores: int, current: int | None) -> list[int]:
    """A few thread counts at or below the real core count (never above it)."""
    physical = max(int(physical_cores), 1)
    values = {physical, max(physical - 1, 1), max(physical // 2, 1)}
    return sorted(value for value in values if value != current)


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
    try_load: Callable[[int], LoadAttempt] | None = None
    # llama-server options this build has (from ``--help``); None = not checked.
    supported_options: frozenset[str] | None = None
    _heat_decisions: list[dict[str, object]] = field(default_factory=list, init=False)
    _heat_check: HeatCheck | None = field(default=None, init=False)

    # ------------------------------------------------------------------ #
    def run(self) -> TuneResult:
        """Run ``tune``; the heat report (what heat did to any decision, and the heat
        check of the chosen setup) is added to the result and printed at the end."""
        self._heat_decisions = []
        self._heat_check = None
        result = self._run_search()
        report, lines = self._heat_report()
        for line in lines:
            self.say(line)
        return result.model_copy(update={"heat_report": report})

    def _note_heat(self, what: str, fair: FairResult) -> None:
        if not fair.heat_effect:
            return
        entry = {
            "what": what,
            "effect": fair.heat_effect,
            "changed_decision": fair.heat_changed_decision,
            "turns_left_out": list(fair.uneven_turns),
            "verdict": fair.judgement.verdict.value if fair.judgement else None,
            "verdict_with_all_turns": fair.verdict_with_all_turns,
        }
        self._heat_decisions.append(entry)
        prefix = "  HEAT! " if fair.heat_changed_decision else "  heat: "
        self.say(prefix + fair.heat_effect)

    def _heat_report(self) -> tuple[dict[str, object], list[str]]:
        lines: list[str] = ["Heat report:"]
        changed = [item for item in self._heat_decisions if item["changed_decision"]]
        if not self._heat_decisions:
            lines.append(
                "  no comparison was disturbed by heat; no measurement was left out"
            )
        for item in self._heat_decisions:
            mark = "DECISION CHANGED BY HEAT - " if item["changed_decision"] else ""
            lines.append(f"  {mark}{item['what']}: {item['effect']}")
        if changed:
            lines.append(
                "  Some results were decided by heat, not by the settings. Running tune "
                "again when the laptop is cool (and plugged in) may give a different answer."
            )
        check = self._heat_check
        if check is not None:
            lines.append("  " + check.words)
        report: dict[str, object] = {
            "decisions": list(self._heat_decisions),
            "decisions_changed_by_heat": len(changed),
            "heat_check": check.record() if check is not None else None,
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
        self.say(
            "Please leave the computer alone while InferenceOS measures "
            f"(up to about {budget / 60:.0f} minutes). Measurements disturbed by other "
            "activity are repeated."
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

        start_measurement = measure_steady(
            harness=self.harness,
            model=self.model,
            workload=self.workload,
            config=starting,
            policy=self.experiment_policy,
            phase="tune-start",
            watch=watch,
        )
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

        # --- Round 1: graphics card layers -------------------------------- #
        current = starting
        current_measurement = start_measurement
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
        if layer_change is not None and layer_measurement is not None:
            new_layers = layer_change.new_value
            assert isinstance(new_layers, int)
            current = current.model_copy(update={"gpu_offload_layers": new_layers})
            current_measurement = layer_measurement
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

        # --- Further optimization paths, each from the setup the last one left ---- #
        fair_checked: set[str] = set()
        # Changes found with a different workload (e.g. the long-prompt one) need not
        # show a gain on tune's own workload in the final check, only no loss.
        other_workload: set[str] = set()
        if layer_change is not None and layer_round.get("fair") is not None:
            fair_checked.add(layer_change.change_id)
        for number, path in enumerate(paths_by_name(self._path_names()), start=2):
            if not performance_first and time.perf_counter() >= search_deadline:
                not_reached.append(f"{path.label} (time limit reached)")
                self.say(f"Round {number} ({path.label}) not reached: time limit.")
                continue
            self.say(f"Round {number}: {path.label}")
            path_round, path_change, new_config = self._path_round(
                path, current, watch, None if performance_first else search_deadline
            )
            rounds.append(path_round)
            if path_change is not None and new_config is not None:
                current = new_config
                changes.append(path_change)
                fair_checked.add(path_change.change_id)
                if path_round.get("workload") != self.workload.workload_id:
                    other_workload.add(path_change.change_id)
                self.say(f"  found: {describe_change(path_change)}")
            else:
                self.say(f"  no change: {path_round.get('reason')}")

        final = current
        check_context_unchanged(starting, final, card_id="tune")

        # --- Heat check: the chosen setup answers several questions in a row ----- #
        heat_runs = int(self.settings.tune_heat_check_runs)
        if heat_runs > 0:
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
        card_for_quality = changes[0].card_id
        all_fair = all(change.change_id in fair_checked for change in changes)
        if all_fair:
            # Every change was already confirmed in turns against the setup just
            # before it. The whole setup is checked once more in turns against the
            # start: it must not be worse (a change found on the long-prompt workload
            # need not show on this one).
            self.say(
                "Confirming: the new setup against the starting one, in turns "
                f"({self.settings.tune_final_check_turns} each)"
            )
            final_fair = compare_fairly(
                harness=self.harness,
                model=self.model,
                workload=self.workload,
                start=starting,
                candidate=final,
                policy=self._path_policy(),
                settings=self.settings,
                rules=self.rules,
                watch=watch,
                card_id=card_for_quality,
                turns=int(self.settings.tune_final_check_turns),
                phase="tune-final",
                on_step=lambda step: self.say(f"  {step}"),
            )
            self.say("  " + fair_words(final_fair))
            self._note_heat("final check of the whole new setup", final_fair)
            confirmation = final_fair.as_confirmation()
            holds = bool(final_fair.turns) and final_fair.quality_passed and (
                not final_fair.guard_breaches
            ) and (final_fair.gain is None or final_fair.gain >= 0.0)
            notes.append(f"final check: {fair_words(final_fair)}")
            rounds.append({"round": "final check", "fair": final_fair.record()})
            only_other_workload = all(change.change_id in other_workload for change in changes)
            if holds and not confirmation.confirmed and only_other_workload:
                # Not separately better on this workload, but not worse either: the
                # changes stand on their own fair checks. Report the chained gain.
                chained = 1.0
                for change in changes:
                    chained *= 1.0 + float(change.measured_gain or 0.0)
                notes.append(
                    "the whole setup was not worse than the start; each change was "
                    f"confirmed on its own (chained gain {chained - 1.0:+.1%})"
                )
                if confirmation.judgement is not None:
                    confirmation = confirmation.model_copy(
                        update={
                            "judgement": confirmation.judgement.model_copy(
                                update={
                                    "keep": True,
                                    "reasons": [
                                        *confirmation.judgement.reasons,
                                        "kept: each change confirmed in turns; the whole "
                                        "setup is not worse",
                                    ],
                                    "goal": confirmation.judgement.goal.model_copy(
                                        update={"gain": chained - 1.0}
                                    ),
                                }
                            )
                        }
                    )
        else:
            self.say(
                "Confirming: measuring the new setup against the starting one, "
                f"alternating {self.settings.tune_confirm_rounds} times"
            )
            confirmation = confirm_by_alternating(
                harness=self.harness,
                model=self.model,
                workload=self.workload,
                starting=starting,
                final=final,
                policy=self.experiment_policy,
                settings=self.settings,
                rules=self.rules,
                rounds=int(self.settings.tune_confirm_rounds),
                watch=watch,
                card_id=card_for_quality,
                deadline=final_deadline,
                on_step=lambda step: self.say(f"  {step}"),
            )
            self.say("  " + _confirmation_words(confirmation))
        if not confirmation.confirmed:
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
        gain = confirmation.judgement.goal.gain if confirmation.judgement else None
        self.say(
            f"Together these changes measured {gain:+.1%} on your "
            f"{self.rules.goal.mode.value} goal."
            if gain is not None
            else "Together these changes were confirmed as better."
        )
        decisions: list[ChangeDecision] = []
        for change in changes:
            decisions.append(
                self.approver.decide(
                    change, timeout_s=float(self.settings.tune_approval_timeout_s)
                )
            )
        approved_ids = {item.change_id for item in decisions if item.approved}
        approved = [change for change in changes if change.change_id in approved_ids]
        declined = [change for change in changes if change.change_id not in approved_ids]

        chosen = _apply_changes(starting, approved)
        recheck: ConfirmationResult | None = None
        confirmed_gain = gain
        outcome = "saved"
        save_notes: list[str] = []
        if not approved:
            chosen = starting
            confirmed_gain = None
            outcome = "no changes approved"
            save_notes.append("no changes approved; the starting setup was saved")
        elif declined:
            self.say("Checking the changes you kept on their own")
            if all_fair:
                recheck_fair = compare_fairly(
                    harness=self.harness,
                    model=self.model,
                    workload=self.workload,
                    start=starting,
                    candidate=chosen,
                    policy=self._path_policy(),
                    settings=self.settings,
                    rules=self.rules,
                    watch=watch,
                    card_id=approved[0].card_id,
                    turns=int(self.settings.tune_recheck_rounds),
                    phase="tune-recheck",
                    on_step=lambda step: self.say(f"  {step}"),
                )
                self.say("  " + fair_words(recheck_fair))
                self._note_heat("re-check of the changes you kept", recheck_fair)
                recheck = recheck_fair.as_confirmation()
                recheck_holds = bool(recheck_fair.turns) and recheck_fair.quality_passed and (
                    not recheck_fair.guard_breaches
                ) and (recheck_fair.gain is None or recheck_fair.gain >= 0.0)
            else:
                recheck = confirm_by_alternating(
                    harness=self.harness,
                    model=self.model,
                    workload=self.workload,
                    starting=starting,
                    final=chosen,
                    policy=self.experiment_policy,
                    settings=self.settings,
                    rules=self.rules,
                    rounds=int(self.settings.tune_recheck_rounds),
                    watch=watch,
                    card_id=approved[0].card_id,
                    on_step=lambda step: self.say(f"  {step}"),
                )
                self.say("  " + _confirmation_words(recheck))
                recheck_holds = recheck.confirmed
            if recheck_holds:
                confirmed_gain = recheck.judgement.goal.gain if recheck.judgement else None
                if all_fair:
                    kept_gain = 1.0
                    for change in approved:
                        kept_gain *= 1.0 + float(change.measured_gain or 0.0)
                    confirmed_gain = kept_gain - 1.0
                outcome = "saved (some changes turned down)"
            else:
                chosen = starting
                confirmed_gain = None
                outcome = "kept changes did not hold up on their own"
                save_notes.append(
                    "the approved changes did not hold up without the ones turned down; "
                    "the starting setup was saved"
                )
                approved, declined = [], changes

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

    def _layer_round(
        self,
        current: RuntimeConfiguration,
        start_measurement: ExperimentMeasurement,
        watch: DisturbanceWatch,
        deadline: float,
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
        estimate: int | None = None
        if envelope.sufficient and envelope.feasible_max is not None:
            estimate = max(current_layers, int(envelope.feasible_max))
            if self.model.n_layers > 0:
                estimate = min(estimate, int(self.model.n_layers))
        top_result = find_layer_top(
            n_layers=int(self.model.n_layers),
            estimate=estimate,
            try_load=self.try_load,
            memory_readable=self.machine.total_graphics_memory_bytes is not None,
            say=self.say,
        )
        top = max(0, int(top_result.top))
        if self.model.n_layers > 0:
            top = min(top, int(self.model.n_layers))
        top_reasons = list(top_result.reasons)
        n_layers = int(self.model.n_layers)
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
        start_above_top = current_layers > top
        baseline_layers = min(current_layers, top)
        if baseline_layers != current_layers:
            self.say(
                f"  the starting {current_layers} layers is above the {top} that load; "
                f"using {top}"
            )
        try:
            kinds = layer_kinds_from_gguf(self.model_path)
        except Exception:  # noqa: BLE001 - an unreadable model file just means no kinds
            kinds = None

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
            max_primary_cv=self.search_policy.max_measure_cv,
        )
        evaluator = RobustPointEvaluator(
            DisturbanceAwareEvaluator(inner, watch),
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
                "known": kinds is not None,
                "count": None if kinds is None else len(set(kinds)),
            },
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
            n_layers=int(self.model.n_layers),
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
        if result.chosen is None:
            if start_above_top and baseline_layers != current_layers:
                # The starting count is unsafe, so keep the largest count that was
                # measured to leave the safety margin free rather than reporting none.
                chosen = baseline_layers
                self.say(
                    f"  the search could not choose a count; using the largest that "
                    f"leaves memory free ({chosen})"
                )
            else:
                record["reason"] = result.reason + " (no count could be measured)"
                return record, None, None
        else:
            chosen = int(result.chosen)
        if chosen == current_layers:
            record["reason"] = "no other layer count was better than the starting point"
            return record, None, None
        measurement = last_measurement.get(chosen)
        if measurement is None:
            record["reason"] = "the chosen count was not measured"
            return record, None, None
        # Compare against the setup actually measured as the starting baseline. When the
        # start was moved down because it did not leave memory free, the comparison is
        # against the old start, not against the moved-down count.
        if not start_above_top and bool(self.settings.tune_layer_fair_check):
            # Fair check: the chosen count and the starting count measured in turns,
            # never against the one reading taken at the very start.
            self.say(f"  checking {chosen} layers against {current_layers} in turns")
            fair = compare_fairly(
                harness=self.harness,
                model=self.model,
                workload=self.workload,
                start=current,
                candidate=inner.configuration_for(chosen),
                policy=self._path_policy(),
                settings=self.settings,
                rules=self.rules,
                watch=watch,
                card_id=LAYERS_CARD_ID,
                turns=int(self.settings.tune_fair_turns),
                phase="tune-layers-fair",
                deadline=None if performance_first else deadline,
                on_step=lambda step: self.say(f"    {step}"),
            )
            self.say("  " + fair_words(fair))
            self._note_heat(f"graphics card layers ({chosen} vs {current_layers})", fair)
            record["fair"] = fair.record()
            record["gain"] = fair.gain
            if not fair.kept:
                record["reason"] = (
                    "the chosen count did not hold up against the starting count "
                    "measured in turns"
                )
                return record, None, None
            record["selected"] = chosen
            change = ProposedChange(
                change_id=f"{LAYERS_CARD_ID}:{chosen}",
                card_id=LAYERS_CARD_ID,
                setting="gpu_offload_layers",
                description="Graphics card layers",
                old_value=current_layers,
                new_value=chosen,
                measured_gain=fair.gain,
                evidence=[
                    str(record["reason"]),
                    f"top={top}",
                    f"tests={result.tests}",
                    "confirmed in turns: " + fair_words(fair),
                ],
            )
            return record, change, measurement
        reference = start_measurement if start_above_top else baseline_measurement
        judged = judge_pair(
            before=reference,
            after=measurement,
            workload=self.workload,
            model=self.model,
            policy=self.experiment_policy,
            settings=self.settings,
            rules=self.rules,
            card_id=LAYERS_CARD_ID,
        )
        better = (
            judged.technically_better
            and not judged.guard_breaches
            and judged.gain is not None
            and judged.gain > 0.0
        )
        record["gain"] = judged.gain
        if start_above_top:
            # Moving down frees the memory the starting count did not leave; propose it
            # as a change from the old start even when the measured speed is not better.
            record["reason"] = start_change_reason
            evidence = [start_change_reason, f"top={top}", f"tests={result.tests}"]
        elif better:
            evidence = [str(record["reason"]), f"top={top}", f"tests={result.tests}"]
        else:
            record["reason"] = "the chosen count was not clearly better than the starting point"
            return record, None, None
        record["selected"] = chosen
        change = ProposedChange(
            change_id=f"{LAYERS_CARD_ID}:{chosen}",
            card_id=LAYERS_CARD_ID,
            setting="gpu_offload_layers",
            description="Graphics card layers",
            old_value=current_layers,
            new_value=chosen,
            measured_gain=judged.gain,
            evidence=evidence,
        )
        return record, change, measurement

    def _threads_round(
        self,
        current: RuntimeConfiguration,
        current_measurement: ExperimentMeasurement,
        watch: DisturbanceWatch,
        deadline: float,
    ) -> tuple[dict[str, object], ProposedChange | None, ExperimentMeasurement | None]:
        values = thread_values_to_try(self.hardware.cpu.physical_cores, current.threads)
        best_value: int | None = None
        best_measurement = current_measurement
        best_gain: float | None = None
        tried: list[dict[str, object]] = []
        stopped = ""
        for value in values:
            if time.perf_counter() >= deadline:
                stopped = "time limit reached"
                break
            candidate = current.model_copy(update={"threads": value})
            check_context_unchanged(current, candidate, card_id=THREADS_CARD_ID)
            self.say(f"  trying {value} threads")
            measurement = measure_steady(
                harness=self.harness,
                model=self.model,
                workload=self.workload,
                config=candidate,
                policy=self.experiment_policy,
                phase=f"tune-threads-{value}",
                watch=watch,
                deadline=deadline,
            )
            if measurement.valid_runs < self.experiment_policy.min_valid_samples:
                tried.append({"threads": value, "result": "could not be measured"})
                continue
            judged = judge_pair(
                before=best_measurement,
                after=measurement,
                workload=self.workload,
                model=self.model,
                policy=self.experiment_policy,
                settings=self.settings,
                rules=self.rules,
                card_id=THREADS_CARD_ID,
            )
            better = (
                judged.technically_better
                and not judged.guard_breaches
                and judged.gain is not None
                and judged.gain > 0.0
            )
            tried.append(
                {"threads": value, "gain_vs_best_so_far": judged.gain, "better": better}
            )
            if better:
                # Gains are chained against the best so far; the total against the
                # round's start is recomputed below.
                best_value = value
                best_measurement = measurement
        record: dict[str, object] = {
            "round": "processor threads",
            "card_id": THREADS_CARD_ID,
            "tried": tried,
            "selected": best_value,
            "reason": stopped
            or ("a better thread count was found" if best_value is not None else "none was better"),
        }
        if best_value is None:
            return record, None, None
        total = judge_pair(
            before=current_measurement,
            after=best_measurement,
            workload=self.workload,
            model=self.model,
            policy=self.experiment_policy,
            settings=self.settings,
            rules=self.rules,
            card_id=THREADS_CARD_ID,
        )
        best_gain = total.gain
        record["gain"] = best_gain
        change = ProposedChange(
            change_id=f"{THREADS_CARD_ID}:{best_value}",
            card_id=THREADS_CARD_ID,
            setting="threads",
            description="Processor threads",
            old_value=current.threads if current.threads is not None else "program default",
            new_value=best_value,
            measured_gain=best_gain,
            evidence=[f"tried {', '.join(str(v) for v in values)} threads"],
        )
        return record, change, best_measurement

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

    def _path_policy(self) -> ExperimentPolicy:
        """Exactly ``tune_path_runs`` measured runs per measurement (no early stop), so
        the middle value is not moved by one slow run (e.g. the first after a load)."""
        runs = max(int(self.settings.tune_path_runs), 1)
        return self.experiment_policy.model_copy(
            update={
                "measured_repetitions": runs,
                "max_repetitions": runs,
                "min_valid_samples": runs,
                "max_total_runs": runs + self.experiment_policy.warmup_repetitions + 2,
            }
        )

    def _path_names(self) -> list[str]:
        return [name for name in str(self.settings.tune_paths).split(",") if name.strip()]

    def _workload_for(self, path: OptionPath) -> BenchmarkWorkload:
        if path.workload_id is None or path.workload_id == self.workload.workload_id:
            return self.workload
        return steady_workload(path.workload_id, default=self.workload)

    def _path_round(
        self,
        path: OptionPath,
        current: RuntimeConfiguration,
        watch: DisturbanceWatch,
        deadline: float | None,
    ) -> tuple[dict[str, object], ProposedChange | None, RuntimeConfiguration | None]:
        """One optimization path: screen a few changes, then check the best fairly.

        1. Does the path apply here? If not, say why and skip it.
        2. Measure the current setup once with the path's workload, then each candidate
           once, and rank them by the goal's ranking value against that reading.
        3. The best-looking candidate is measured against the current setup in turns
           (``compare_fairly``) and kept only if the normal keep rule says so.
        """
        context = PathContext(
            hardware=self.hardware,
            model=self.model,
            total_layers=int(self.model.n_layers),
            current=current,
            supported_options=self.supported_options,
        )
        record: dict[str, object] = {
            "round": path.label,
            "path": path.name,
            "card_id": path.card_id,
            "setting": path.setting,
            "start_value": getattr(current, path.setting),
        }
        applies, why = path.check(context)
        record["applies"] = applies
        record["applies_reason"] = why
        if not applies:
            record["reason"] = f"skipped: {why}"
            return record, None, None
        candidates = path.candidates(context)
        record["candidates"] = [candidate.label for candidate in candidates]
        if not candidates:
            record["reason"] = "nothing to try"
            return record, None, None
        workload = self._workload_for(path)
        record["workload"] = workload.workload_id
        path_policy = self._path_policy()
        record["runs_per_measurement"] = path_policy.measured_repetitions

        def time_left() -> bool:
            return deadline is None or time.perf_counter() < deadline

        self.say(f"  measuring the current setup ({workload.workload_id})")
        reference = measure_steady(
            harness=self.harness,
            model=self.model,
            workload=workload,
            config=current,
            policy=path_policy,
            phase=f"tune-{path.name}-current",
            watch=watch,
            deadline=deadline,
        )
        if reference.valid_runs < path_policy.min_valid_samples:
            record["reason"] = "the current setup could not be measured"
            return record, None, None
        record["current_reading"] = _readings(reference)
        screened: list[dict[str, object]] = []
        best: tuple[float, Candidate, RuntimeConfiguration] | None = None
        for candidate in candidates:
            if not time_left():
                screened.append({"candidate": candidate.label, "result": "time limit reached"})
                continue
            config = apply_candidate(current, path, candidate)
            check_context_unchanged(current, config, card_id=path.card_id)
            self.say(f"  trying {candidate.label}")
            measurement = measure_steady(
                harness=self.harness,
                model=self.model,
                workload=workload,
                config=config,
                policy=path_policy,
                phase=f"tune-{path.name}-{candidate.value}",
                watch=watch,
                deadline=deadline,
            )
            entry: dict[str, object] = {
                "candidate": candidate.label,
                "value": candidate.value,
                "readings": _readings(measurement),
            }
            if measurement.valid_runs < path_policy.min_valid_samples:
                entry["result"] = "could not be measured"
                screened.append(entry)
                continue
            score = _screen_gain(reference, measurement, self.rules.goal)
            entry["screen_gain"] = score
            entry["_measurement"] = measurement
            screened.append(entry)
            self.say(
                f"    {candidate.label}: "
                + (f"{score:+.1%} on your goal (first look)" if score is not None else "?")
            )
            if score is not None and (best is None or score > best[0]):
                best = (score, candidate, config)
        # Measure the current setup again after the candidates; a candidate must look
        # better than both readings, so one slow reading of the current setup (e.g.
        # right after a workload change) cannot make every candidate look better.
        if screened and time_left():
            self.say("  measuring the current setup again")
            again = measure_steady(
                harness=self.harness,
                model=self.model,
                workload=workload,
                config=current,
                policy=path_policy,
                phase=f"tune-{path.name}-current-again",
                watch=watch,
                deadline=deadline,
            )
            if again.valid_runs >= path_policy.min_valid_samples:
                record["current_reading_again"] = _readings(again)
                rescored: tuple[float, Candidate, RuntimeConfiguration] | None = None
                for entry, candidate in zip(screened, candidates):
                    measured = entry.get("_measurement")
                    if not isinstance(measured, ExperimentMeasurement):
                        continue
                    first = _screen_gain(reference, measured, self.rules.goal)
                    second = _screen_gain(again, measured, self.rules.goal)
                    if first is None or second is None:
                        continue
                    # The smaller of the two: a candidate must look better than both
                    # readings of the current setup.
                    score = min(first, second)
                    entry["screen_gain"] = score
                    entry["screen_gain_vs_first_reading"] = first
                    entry["screen_gain_vs_second_reading"] = second
                    if rescored is None or score > rescored[0]:
                        rescored = (score, candidate, apply_candidate(current, path, candidate))
                if rescored is not None:
                    best = rescored
                    self.say(
                        f"    best first look against both readings: {best[1].label} "
                        f"{best[0]:+.1%}"
                    )
        for entry in screened:
            entry.pop("_measurement", None)
        record["screened"] = screened
        if best is None:
            record["reason"] = "no candidate could be measured"
            return record, None, None
        best_score, best_candidate, best_config = best
        record["best_first_look"] = best_candidate.label
        if best_score <= 0.0:
            record["reason"] = (
                f"nothing looked better than the current setup (best: "
                f"{best_candidate.label}, {best_score:+.1%})"
            )
            return record, None, None
        if not time_left():
            record["reason"] = "time limit reached before the fair check"
            return record, None, None
        self.say(f"  checking {best_candidate.label} against the current setup in turns")
        fair = compare_fairly(
            harness=self.harness,
            model=self.model,
            workload=workload,
            start=current,
            candidate=best_config,
            policy=path_policy,
            settings=self.settings,
            rules=self.rules,
            watch=watch,
            card_id=path.card_id,
            turns=int(self.settings.tune_fair_turns),
            phase=f"tune-{path.name}-fair",
            deadline=deadline,
            on_step=lambda step: self.say(f"    {step}"),
        )
        self.say("  " + fair_words(fair))
        self._note_heat(f"{path.label}: {best_candidate.label}", fair)
        record["fair"] = fair.record()
        record["gain"] = fair.gain
        if not fair.kept:
            reasons = fair.judgement.reasons[-1:] if fair.judgement else []
            record["reason"] = (
                f"{best_candidate.label} did not hold up in turns"
                + (f" ({reasons[0]})" if reasons else "")
            )
            if fair.heat_changed_decision:
                record["reason"] = (
                    f"{best_candidate.label} was REJECTED BECAUSE OF HEAT (see the heat "
                    "report)"
                )
            return record, None, None
        record["selected"] = best_candidate.value
        record["reason"] = f"{best_candidate.label} confirmed in turns"
        old = getattr(current, path.setting)
        change = ProposedChange(
            change_id=f"{path.card_id}:{best_candidate.value}",
            card_id=path.card_id,
            setting=path.setting,
            description=path.description,
            old_value=old if old is not None else "program default",
            new_value=best_candidate.value,
            measured_gain=fair.gain,
            evidence=[
                f"tried {', '.join(c.label for c in candidates)}",
                f"measured with the {workload.workload_id} workload",
                "confirmed in turns: " + fair_words(fair),
            ],
        )
        return record, change, best_config

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
    ) -> SavedSetup:
        check_context_unchanged(starting, final, card_id="tune")
        setup = SavedSetup(
            setup_id=f"setup-{uuid.uuid4().hex[:10]}",
            model=model_identity(self.model_path, with_hash=self.model_hash),
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
        self.store.save(setup)
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
    update: dict[str, object] = {}
    for change in changes:
        update[change.setting] = change.new_value
    return starting.model_copy(update=update)


def _describe_config(config: RuntimeConfiguration) -> str:
    threads = config.threads if config.threads is not None else "program default"
    text = f"{config.gpu_offload_layers} graphics card layers, {threads} threads"
    if config.threads_batch is not None:
        text += f", {config.threads_batch} reading threads"
    if config.flash_attention is not None:
        text += ", flash attention " + ("on" if config.flash_attention else "off")
    if config.ubatch_size is not None:
        text += f", prompt batch {config.ubatch_size}"
    return text + f", context {config.context_length}"


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
