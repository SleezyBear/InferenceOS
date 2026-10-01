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
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.experiment import (
    ExperimentMeasurement,
    ExperimentPolicy,
    RuntimeConfiguration,
)
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
from inferenceos.optimization.keep import ModeRules
from inferenceos.optimization.search.evaluation import (
    RobustPointEvaluator,
    RuntimeConfigurationPointEvaluator,
)
from inferenceos.optimization.search.policy import SearchPolicy
from inferenceos.optimization.search.resources import build_resource_envelope
from inferenceos.optimization.search.sampling import sample_layers
from inferenceos.profiling.workloads import BenchmarkWorkload
from inferenceos.tune.approval import ChangeApprover, describe_change
from inferenceos.tune.fit import layers_that_fit
from inferenceos.tune.measure import (
    DisturbanceAwareEvaluator,
    DisturbanceWatch,
    confirm_by_alternating,
    judge_pair,
    measure_steady,
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

    # ------------------------------------------------------------------ #
    def run(self) -> TuneResult:
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
        self.say("Round 1: graphics card layers")
        layer_round, layer_change, layer_measurement = self._layer_round(
            current, watch, search_deadline
        )
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

        # --- Round 2: processor threads ------------------------------------ #
        if time.perf_counter() < search_deadline:
            self.say("Round 2: processor threads")
            thread_round, thread_change, _ = self._threads_round(
                current, current_measurement, watch, search_deadline
            )
            rounds.append(thread_round)
            if thread_change is not None:
                current = current.model_copy(update={"threads": thread_change.new_value})
                changes.append(thread_change)
                self.say(f"  found: {describe_change(thread_change)}")
            else:
                self.say(f"  no better thread count: {thread_round.get('reason')}")
        else:
            not_reached.append("processor threads (time limit reached)")
            self.say("Round 2 (processor threads) not reached: time limit.")

        final = current
        check_context_unchanged(starting, final, card_id="tune")

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

        # --- Confirm by alternating runs ------------------------------------ #
        self.say(
            "Confirming: measuring the new setup against the starting one, "
            f"alternating {self.settings.tune_confirm_rounds} times"
        )
        card_for_quality = changes[0].card_id
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
            if recheck.confirmed:
                confirmed_gain = recheck.judgement.goal.gain if recheck.judgement else None
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
            config = RuntimeConfiguration(
                context_length=self.context_length,
                gpu_offload_layers=saved.gpu_offload_layers,
                threads=saved.threads,
                use_mmap=saved.use_mmap,
            )
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
        watch: DisturbanceWatch,
        deadline: float,
    ) -> tuple[dict[str, object], ProposedChange | None, ExperimentMeasurement | None]:
        """Try layer counts across the whole safe range, fewer as well as more.

        More layers is not always better: a nearly full graphics card can spill into
        slower shared memory. About ``max_samples`` counts from 0 up to the most that
        fit (at the fixed context size, with its conversation memory) are measured,
        nearest to the starting point first. The best gain on the goal wins; when two
        are within the diminishing-returns margin, the smaller count wins.
        """
        current_layers = int(current.gpu_offload_layers or 0)
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
        most = current_layers
        if envelope.sufficient and envelope.feasible_max is not None:
            most = max(current_layers, int(envelope.feasible_max))
        most = min(most, int(self.model.n_layers)) if self.model.n_layers > 0 else most
        values = set(sample_layers(0, most, self.search_policy)) | {current_layers}
        order = sorted(values, key=lambda value: (abs(value - current_layers), value))

        inner = RuntimeConfigurationPointEvaluator(
            profiler=self.harness,
            model=self.model,
            workload=self.workload,
            experiment_policy=self.experiment_policy,
            context_length=self.context_length,
            envelope=envelope,
            base_configuration=current,
            paging_per_sec_ceiling=self.search_policy.paging_per_sec_ceiling,
            deadline=deadline,
            max_primary_cv=self.search_policy.max_measure_cv,
        )
        evaluator = RobustPointEvaluator(
            DisturbanceAwareEvaluator(inner, watch),
            max_attempts=self.search_policy.max_measure_attempts,
        )
        record: dict[str, object] = {
            "round": "graphics card layers",
            "card_id": LAYERS_CARD_ID,
            "safe_range": [0, most],
            "planned": order,
            "envelope": list(envelope.reasons) or list(envelope.basis[-1:]),
        }

        baseline = evaluator.measure(current_layers)
        if not baseline.ok or baseline.measurement is None:
            record["reason"] = (
                f"the starting {current_layers} layers could not be measured "
                f"({baseline.status.value})"
            )
            record["tried"] = [{"layers": current_layers, "status": baseline.status.value}]
            return record, None, None

        tried: list[dict[str, object]] = [
            {"layers": current_layers, "status": "starting point"}
        ]
        unsafe_above: int | None = None
        acceptable: list[tuple[int, float, ExperimentMeasurement]] = []
        skipped: list[int] = []
        for value in order[1:]:
            if unsafe_above is not None and value >= unsafe_above:
                skipped.append(value)
                continue
            if time.perf_counter() >= deadline:
                skipped.append(value)
                continue
            config = inner.configuration_for(value)
            check_context_unchanged(current, config, card_id=LAYERS_CARD_ID)
            self.say(f"  trying {value} layers")
            point = evaluator.measure(value)
            if not point.ok or point.measurement is None:
                tried.append({"layers": value, "status": point.status.value})
                if not point.safe and value > current_layers:
                    unsafe_above = value if unsafe_above is None else min(unsafe_above, value)
                continue
            judged = judge_pair(
                before=baseline.measurement,
                after=point.measurement,
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
            tried.append({"layers": value, "gain": judged.gain, "better": better})
            if better and judged.gain is not None:
                acceptable.append((value, judged.gain, point.measurement))
        record["tried"] = tried
        if skipped:
            record["not_measured"] = skipped

        if not acceptable:
            record["reason"] = "no other layer count was better than the starting point"
            return record, None, None
        best_gain = max(gain for _, gain, _ in acceptable)
        margin = self.search_policy.diminishing_returns_margin
        close = [item for item in acceptable if best_gain - item[1] <= margin]
        chosen_layers, chosen_gain, chosen_measurement = min(close, key=lambda item: item[0])
        record["selected"] = chosen_layers
        record["gain"] = chosen_gain
        record["reason"] = (
            f"{chosen_layers} layers measured {chosen_gain:+.1%} on the goal"
            + (
                f" (the best was {best_gain:+.1%}; within {margin:.0%}, the smaller count wins)"
                if chosen_gain < best_gain
                else ""
            )
        )
        change = ProposedChange(
            change_id=f"{LAYERS_CARD_ID}:{chosen_layers}",
            card_id=LAYERS_CARD_ID,
            setting="gpu_offload_layers",
            description="Graphics card layers",
            old_value=current_layers,
            new_value=chosen_layers,
            measured_gain=chosen_gain,
            evidence=[str(record["reason"])],
        )
        return record, change, chosen_measurement

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
    return (
        f"{config.gpu_offload_layers} graphics card layers, {threads} threads, "
        f"context {config.context_length}"
    )


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
