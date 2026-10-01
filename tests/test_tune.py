"""Step 5: the everyday ``tune`` command (search, confirm, approve, save)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, cast

import pytest

from inferenceos.config import Settings
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.profiling import ProfilingRun
from inferenceos.contracts.tune import MachineCheck, PowerPlan
from inferenceos.optimization.keep import ModeRules
from inferenceos.optimization.search.policy import SearchPolicy
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_hardware, sample_model
from inferenceos.tune.approval import ScriptedChangeApprover
from inferenceos.tune.fit import layers_that_fit
from inferenceos.tune.machine import check_machine, machine_changes, parse_power_plan
from inferenceos.tune.measure import (
    DisturbanceWatch,
    disturbance_reason,
    measure_steady,
    pool,
)
from inferenceos.tune.pipeline import Tuner, thread_values_to_try
from inferenceos.tune.profiles import SetupStore, model_identity, same_model_file
from tests.experiment_helpers import (
    SampleResult,
    ScriptedHarness,
    build_test_policy,
    make_run,
)

GIB = 1024**3
CONTEXT = 2048
# With the sample model (24 layers, 1 GB) on the sample 4 GB card, this much already in
# use before loading leaves room for 8 layers.
USED_BEFORE_LOADING = 3_060_000_000
# The real reading once 8 layers are loaded is lower than the estimate, so the layer
# search finds room above the starting point.
USED_WITH_MODEL = 2_000_000_000


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "reports_dir": tmp_path / "reports",
        "logs_dir": tmp_path / "logs",
    }
    base.update(overrides)
    return Settings(**base)


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "sample.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * 64)
    return path


def _machine(settings: Settings, power: PowerState = PowerState.AC) -> MachineCheck:
    return check_machine(
        settings,
        hardware=sample_hardware(),
        live_reader=lambda: (USED_BEFORE_LOADING, power),
        power_plan_reader=lambda: PowerPlan(plan_id="381b4222-f694-41f0-9685-ff5bb260df2e", name="Balanced"),
        version_reader=lambda: "b11139",
    )


def layer_gain_responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    """More graphics card layers write faster; threads make no difference."""
    assert task.context_length == CONTEXT, "context size must never change"
    layers = options.gpu_offload_layers or 0
    if layers <= 8:
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)
    return SampleResult(decode_tps=24.0, total_latency_s=1.7)


def layer_and_thread_responder(
    task: ProfileTask, options: ProfilerOptions, call: int
) -> SampleResult:
    assert task.context_length == CONTEXT, "context size must never change"
    layers = options.gpu_offload_layers or 0
    speed = 20.0 if layers <= 8 else 24.0
    latency = 2.0 if layers <= 8 else 1.7
    if options.threads == 5:
        speed *= 1.10
        latency *= 0.93
    return SampleResult(decode_tps=speed, total_latency_s=latency)


def _tuner(
    tmp_path: Path,
    responder: Any,
    *,
    answers: list[bool],
    settings: Settings | None = None,
    power: PowerState = PowerState.AC,
    lines: list[str] | None = None,
) -> tuple[Tuner, ScriptedHarness, ScriptedChangeApprover, SetupStore]:
    settings = settings or _settings(tmp_path)
    harness = ScriptedHarness(responder)
    approver = ScriptedChangeApprover(answers)
    store = SetupStore(settings)
    out = lines if lines is not None else []
    tuner = Tuner(
        settings=settings,
        model=sample_model(),
        model_path=_model_file(tmp_path),
        context_length=CONTEXT,
        rules=ModeRules.from_settings(settings),
        hardware=sample_hardware(),
        machine=_machine(settings, power),
        harness=harness,
        workload=get_workload("short-short"),
        experiment_policy=build_test_policy(min_effect=0.02, noise_tolerance=0.02),
        search_policy=SearchPolicy(),
        approver=approver,
        store=store,
        graphics_memory_reader=lambda: USED_WITH_MODEL,
        say=out.append,
    )
    return tuner, harness, approver, store


# --------------------------------------------------------------------------- #
# Pieces                                                                       #
# --------------------------------------------------------------------------- #
def test_layers_that_fit_uses_the_search_safety_margins() -> None:
    fit = layers_that_fit(
        model=sample_model(),
        context_length=CONTEXT,
        total_graphics_memory_bytes=4 * GIB,
        used_before_loading_bytes=USED_BEFORE_LOADING,
        policy=SearchPolicy(),
    )
    assert fit.layers == 8
    everything = layers_that_fit(
        model=sample_model(),
        context_length=CONTEXT,
        total_graphics_memory_bytes=4 * GIB,
        used_before_loading_bytes=0,
        policy=SearchPolicy(),
    )
    assert everything.layers == 24  # never more than the model has
    unknown = layers_that_fit(
        model=sample_model(),
        context_length=CONTEXT,
        total_graphics_memory_bytes=4 * GIB,
        used_before_loading_bytes=None,
        policy=SearchPolicy(),
    )
    assert unknown.layers is None


def test_power_plan_is_read_in_any_language() -> None:
    english = parse_power_plan(
        "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)"
    )
    assert english.plan_id == "381b4222-f694-41f0-9685-ff5bb260df2e"
    assert english.name == "Balanced"
    german = parse_power_plan(
        "GUID des Energieschemas: 8C5E7FDA-E8BF-4A96-9A85-A6E23A8C635C  (Höchstleistung)"
    )
    assert german.plan_id == "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
    assert german.name == "Höchstleistung"
    assert parse_power_plan("nothing useful").plan_id is None


def test_machine_check_reports_free_graphics_memory_and_changes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    before = _machine(settings)
    assert before.total_graphics_memory_bytes == 4 * GIB
    assert before.free_graphics_memory_bytes == 4 * GIB - USED_BEFORE_LOADING
    assert before.power_source is PowerState.AC
    assert before.program_version == "b11139"
    assert machine_changes(before, before) == []
    after = before.model_copy(
        update={
            "graphics_drivers": ["Radeon: 31.0.1"],
            "program_version": "b12000",
            "power_plan": PowerPlan(plan_id="8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c", name="High"),
        }
    )
    changes = machine_changes(before.model_copy(update={"graphics_drivers": ["Radeon: 30.0"]}), after)
    assert any("graphics driver" in item for item in changes)
    assert any("llama.cpp version" in item for item in changes)
    assert any("power plan" in item for item in changes)


def test_unreadable_machine_values_stay_unknown(tmp_path: Path) -> None:
    def broken() -> tuple[int | None, PowerState]:
        raise OSError("no counters")

    check = check_machine(
        _settings(tmp_path),
        hardware=sample_hardware(),
        live_reader=broken,
        power_plan_reader=lambda: PowerPlan(),
        version_reader=lambda: None,
    )
    assert check.used_graphics_memory_bytes is None
    assert check.free_graphics_memory_bytes is None
    assert check.power_source is PowerState.UNKNOWN
    assert any("live readings unavailable" in note for note in check.notes)


def test_thread_values_never_go_above_the_real_cores() -> None:
    assert thread_values_to_try(6, None) == [3, 5, 6]
    assert thread_values_to_try(6, 6) == [3, 5]
    assert thread_values_to_try(1, None) == [1]


def test_model_file_check_notices_a_replaced_file(tmp_path: Path) -> None:
    path = _model_file(tmp_path)
    identity = model_identity(path)
    assert same_model_file(identity, path)[0]
    # Same contents, new date: still the same model.
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    same, why = same_model_file(identity, path)
    assert same and "contents are the same" in why
    # Same size, different contents: a different model.
    path.write_bytes(b"GGUF" + b"\1" * 64)
    same, why = same_model_file(identity, path)
    assert not same and "contents changed" in why


def _disturbed_harness(disturbed_calls: set[int]) -> ScriptedHarness:
    class Harness(ScriptedHarness):
        def run_once(self, task: Any, **kwargs: Any) -> ProfilingRun:
            run = super().run_once(task, **kwargs)
            if len(self.calls) in disturbed_calls:
                metrics = run.observation.metrics.model_copy(update={"system_cpu_percent": 90.0})
                observation = run.observation.model_copy(update={"metrics": metrics})
                return run.model_copy(update={"observation": observation})
            return run

    return Harness(lambda task, options, call: SampleResult())


def test_disturbed_measurement_is_repeated(tmp_path: Path) -> None:
    policy = build_test_policy()
    harness = _disturbed_harness({1, 2, 3})  # warm-up + both measured runs disturbed
    repeats: list[str] = []
    watch = DisturbanceWatch(
        logical_cores=12, limit_percent=25.0, max_repeats=2, on_repeat=repeats.append
    )
    measurement = measure_steady(
        harness=harness,
        model=sample_model(),
        workload=get_workload("short-short"),
        config=RuntimeConfiguration(context_length=CONTEXT, gpu_offload_layers=8),
        policy=policy,
        phase="t",
        watch=watch,
    )
    assert watch.repeats == 1 and repeats and "disturbed" in repeats[0]
    assert disturbance_reason(measurement, logical_cores=12, limit_percent=25.0) is None
    assert any("repeated after" in note for note in measurement.notes)


def test_pooled_measurement_keeps_every_run(tmp_path: Path) -> None:
    policy = build_test_policy()
    harness = ScriptedHarness(lambda task, options, call: SampleResult())
    watch = DisturbanceWatch(logical_cores=12, limit_percent=25.0, max_repeats=0)
    parts = [
        measure_steady(
            harness=harness,
            model=sample_model(),
            workload=get_workload("short-short"),
            config=RuntimeConfiguration(context_length=CONTEXT, gpu_offload_layers=8),
            policy=policy,
            phase="t",
            watch=watch,
        )
        for _ in range(3)
    ]
    pooled = pool(parts, phase="p")
    assert pooled.valid_runs == sum(part.valid_runs for part in parts)
    assert pooled.metrics["decode_tokens_per_sec"].count == pooled.valid_runs


# --------------------------------------------------------------------------- #
# The whole tune                                                               #
# --------------------------------------------------------------------------- #
def test_tune_finds_confirms_and_saves_the_approved_setup(tmp_path: Path) -> None:
    lines: list[str] = []
    tuner, harness, approver, store = _tuner(
        tmp_path, layer_gain_responder, answers=[True, True], lines=lines
    )
    result = tuner.run()

    assert result.starting_configuration.gpu_offload_layers == 8
    assert "fit" in result.starting_point_source
    assert result.outcome == "saved"
    assert result.final_configuration.gpu_offload_layers is not None
    assert result.final_configuration.gpu_offload_layers > 8
    assert result.final_configuration.context_length == CONTEXT
    assert [change.setting for change in result.proposed_changes] == ["gpu_offload_layers"]
    change = result.proposed_changes[0]
    assert change.measured_gain is not None and change.measured_gain > 0.05
    # Confirmed by alternating runs, starting setup first.
    confirmation = result.confirmation
    assert confirmation is not None and confirmation.confirmed
    assert confirmation.order[:4] == ["starting", "final", "starting", "final"]
    assert confirmation.rounds == 3
    # Asked about each change, one by one.
    assert [item.change_id for item in approver.asked] == [change.change_id]
    # Saved for this model, context and power source.
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None
    assert saved.configuration == result.final_configuration
    assert saved.approved_changes == [change]
    assert saved.model.sha256 is not None
    assert saved.machine.program_version == "b11139"
    # Every measurement used the user's context size.
    assert {call[0] for call in harness.calls} == {CONTEXT}
    assert any("leave the computer alone" in line for line in lines)


def test_declining_every_change_saves_the_starting_setup(tmp_path: Path) -> None:
    tuner, _, _, store = _tuner(tmp_path, layer_gain_responder, answers=[False])
    result = tuner.run()
    assert result.outcome == "no changes approved"
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None
    assert saved.configuration == result.starting_configuration
    assert saved.approved_changes == []
    assert len(saved.declined_changes) == 1


def test_turning_one_change_down_rechecks_the_rest(tmp_path: Path) -> None:
    tuner, _, approver, store = _tuner(
        tmp_path, layer_and_thread_responder, answers=[True, False]
    )
    result = tuner.run()
    settings_seen = [change.setting for change in result.proposed_changes]
    assert settings_seen == ["gpu_offload_layers", "threads"]
    assert len(approver.asked) == 2
    assert result.recheck is not None and result.recheck.confirmed
    assert result.recheck.rounds == 2
    assert result.outcome == "saved (some changes turned down)"
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None
    assert saved.configuration.threads is None  # the turned-down change is not in it
    assert saved.configuration.gpu_offload_layers == result.final_configuration.gpu_offload_layers
    assert [change.setting for change in saved.declined_changes] == ["threads"]


def test_next_tune_starts_from_the_saved_setup(tmp_path: Path) -> None:
    tuner, _, _, _ = _tuner(tmp_path, layer_gain_responder, answers=[True])
    first = tuner.run()
    again, harness, _, _ = _tuner(tmp_path, layer_gain_responder, answers=[])
    second = again.run()
    assert second.starting_point_source == "your saved setup"
    assert (
        second.starting_configuration.gpu_offload_layers
        == first.final_configuration.gpu_offload_layers
    )
    # Nothing better above it: no change, and nothing to approve.
    assert second.outcome == "no change"


def test_battery_uses_the_plugged_in_setup_as_a_starting_guess(tmp_path: Path) -> None:
    tuner, _, _, store = _tuner(tmp_path, layer_gain_responder, answers=[True])
    first = tuner.run()
    battery, _, _, _ = _tuner(
        tmp_path, layer_gain_responder, answers=[], power=PowerState.BATTERY
    )
    second = battery.run()
    assert second.starting_point_source == "your saved setup for plugged in"
    assert (
        second.starting_configuration.gpu_offload_layers
        == first.final_configuration.gpu_offload_layers
    )
    # Plugged in and battery are kept separately.
    plugged = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    on_battery = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.BATTERY)
    assert plugged is not None and on_battery is not None
    assert plugged.setup_id != on_battery.setup_id


def test_a_gain_that_does_not_hold_up_is_not_proposed(tmp_path: Path) -> None:
    def fading(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        # Real during the search, gone by the time it is confirmed.
        if call > 40:
            return SampleResult(decode_tps=20.0, total_latency_s=2.0)
        return layer_gain_responder(task, options, call)

    tuner, _, approver, store = _tuner(tmp_path, fading, answers=[True])
    result = tuner.run()
    assert result.outcome == "not confirmed"
    assert approver.asked == []
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration == result.starting_configuration


def test_starting_setup_that_cannot_run_saves_nothing(tmp_path: Path) -> None:
    def failing(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(fail=True)

    tuner, _, _, store = _tuner(tmp_path, failing, answers=[True])
    result = tuner.run()
    assert result.outcome.startswith("failed")
    assert store.all() == []


def test_threads_round_is_skipped_when_time_runs_out(tmp_path: Path) -> None:
    settings = _settings(tmp_path, optimization_before_loading_search_time_s=1.0)
    tuner, _, _, _ = _tuner(tmp_path, layer_gain_responder, answers=[True], settings=settings)

    def slow(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        time.sleep(0.05)
        return layer_gain_responder(task, options, call)

    tuner.harness = ScriptedHarness(slow)
    result = tuner.run()
    assert any("threads" in item for item in result.not_reached)


def test_saving_twice_replaces_the_same_slot(tmp_path: Path) -> None:
    tuner, _, _, store = _tuner(tmp_path, layer_gain_responder, answers=[True])
    tuner.run()
    again, _, _, _ = _tuner(tmp_path, layer_gain_responder, answers=[])
    again.run()
    assert len(store.all()) == 1


@pytest.mark.parametrize("power", [PowerState.UNKNOWN, PowerState.AC])
def test_unknown_power_counts_as_plugged_in(tmp_path: Path, power: PowerState) -> None:
    tuner, _, _, store = _tuner(tmp_path, layer_gain_responder, answers=[True], power=power)
    tuner.run()
    assert store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC) is not None


def _tried(record: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], record["tried"])


def test_layer_round_tries_fewer_layers_too_and_keeps_them_when_faster(tmp_path: Path) -> None:
    """More layers is not always better: a nearly full card can be slower."""

    def spill(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        assert task.context_length == CONTEXT
        layers = options.gpu_offload_layers or 0
        if layers >= 20:  # card nearly full: spills into slower shared memory
            return SampleResult(decode_tps=15.0, total_latency_s=2.6)
        if layers >= 9:
            return SampleResult(decode_tps=22.0, total_latency_s=1.8)
        return SampleResult(decode_tps=12.0, total_latency_s=3.0)

    settings = _settings(tmp_path)
    tuner, harness, _, _ = _tuner(tmp_path, spill, answers=[True], settings=settings)
    # Nothing else on the card: every layer "fits", so the start is all 24 layers.
    tuner.machine = tuner.machine.model_copy(update={"used_graphics_memory_bytes": 0})
    result = tuner.run()
    assert result.starting_configuration.gpu_offload_layers == 24
    tried = [int(str(item["layers"])) for item in _tried(result.rounds[0])]
    assert min(tried) < 24, "fewer layers must be tried"
    assert result.proposed_changes and result.proposed_changes[0].setting == "gpu_offload_layers"
    chosen = result.proposed_changes[0].new_value
    assert isinstance(chosen, int) and 9 <= chosen < 20
    assert result.outcome == "saved"
    assert result.final_configuration.gpu_offload_layers == chosen


def test_layer_round_prefers_the_smaller_count_when_gains_are_close(tmp_path: Path) -> None:
    tuner, _, _, _ = _tuner(tmp_path, layer_gain_responder, answers=[True])
    result = tuner.run()
    better = [int(str(item["layers"])) for item in _tried(result.rounds[0]) if item.get("better")]
    # Every count above 8 is equally fast here, so the smallest of them wins.
    assert result.final_configuration.gpu_offload_layers == min(better)
