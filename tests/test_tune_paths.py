"""Optimization paths in ``tune``: threads, flash attention, prompt batch size."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from inferenceos.backends.resident_server import _options_from
from inferenceos.backends.llama_server_toolchain import build_server_command
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.experiment.restoration import runtime_params_from
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.selftest import sample_hardware, sample_model
from inferenceos.tune.paths import (
    FLASH_ATTENTION,
    PROMPT_BATCH,
    THREADS,
    PathContext,
    flash_candidates,
    paths_by_name,
    prompt_batch_candidates,
    thread_candidates,
)
from tests.experiment_helpers import SampleResult, ScriptedHarness
from tests.test_tune import CONTEXT, _settings, _tuner

BASE = RuntimeConfiguration(context_length=CONTEXT, gpu_offload_layers=8)


def _context(current: RuntimeConfiguration = BASE, options: frozenset[str] | None = None) -> PathContext:
    return PathContext(
        hardware=sample_hardware(),
        model=sample_model(),
        total_layers=24,
        current=current,
        supported_options=options,
    )


# --------------------------------------------------------------------------- #
# What each path tries                                                         #
# --------------------------------------------------------------------------- #
def test_thread_candidates_sit_around_the_physical_core_count() -> None:
    # 6 physical / 12 logical: the default (6) is not re-tried.
    assert [c.value for c in thread_candidates(_context())] == [4, 5, 8, 12]
    current = BASE.model_copy(update={"threads": 5})
    assert [c.value for c in thread_candidates(_context(current))] == [4, 6, 8, 12]


def test_flash_candidates_try_both_when_llama_cpp_decides() -> None:
    assert [c.value for c in flash_candidates(_context())] == [True, False]
    on = BASE.model_copy(update={"flash_attention": True})
    assert [c.value for c in flash_candidates(_context(on))] == [False]


def test_prompt_batch_candidates_skip_the_current_size() -> None:
    assert [c.value for c in prompt_batch_candidates(_context())] == [128, 256, 1024]
    small = BASE.model_copy(update={"ubatch_size": 256})
    assert [c.value for c in prompt_batch_candidates(_context(small))] == [128, 512, 1024]


def test_a_path_whose_option_this_llama_cpp_lacks_is_skipped() -> None:
    ok, why = FLASH_ATTENTION.check(_context(options=frozenset({"--threads"})))
    assert not ok and "--flash-attn" in why
    ok, _ = THREADS.check(_context(options=frozenset({"--threads"})))
    assert ok


def test_paths_by_name_keeps_order_and_ignores_unknown_names() -> None:
    assert [p.name for p in paths_by_name(["prompt_batch", "nope", "threads"])] == [
        "prompt_batch",
        "threads",
    ]


def test_every_candidate_keeps_the_context_size() -> None:
    for path in (THREADS, FLASH_ATTENTION, PROMPT_BATCH):
        for candidate in path.candidates(_context()):
            changed = BASE.model_copy(update={path.setting: candidate.value})
            assert changed.context_length == CONTEXT


# --------------------------------------------------------------------------- #
# A whole tune with the paths                                                  #
# --------------------------------------------------------------------------- #
def _speed(options: ProfilerOptions) -> tuple[float, float]:
    """5 threads write 10% faster; flash attention on writes 8% faster; a prompt
    batch of 256 makes the first word 20% sooner on the long prompt only."""
    speed, latency = 20.0, 2.0
    if options.threads == 5:
        speed *= 1.10
    if options.flash_attention is True:
        speed *= 1.08
    return speed, latency


def stacking_responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    assert task.context_length == CONTEXT, "context size must never change"
    speed, latency = _speed(options)
    if task.workload.workload_id.startswith("long-prefill") and options.ubatch_size == 256:
        latency *= 0.80
    return SampleResult(decode_tps=speed, total_latency_s=latency)


def test_paths_stack_and_the_saved_setup_starts_with_every_change(tmp_path: Path) -> None:
    settings = _settings(tmp_path, optimization_mode="performance_first")
    lines: list[str] = []
    tuner, harness, approver, store = _tuner(
        tmp_path, stacking_responder, answers=[True, True, True], settings=settings, lines=lines
    )
    result = tuner.run()

    settings_changed = [change.setting for change in result.proposed_changes]
    assert settings_changed == ["threads", "flash_attention", "ubatch_size"]
    assert result.outcome == "saved"
    final = result.final_configuration
    assert final.threads == 5
    assert final.flash_attention is True
    assert final.ubatch_size == 256
    assert final.context_length == CONTEXT
    # Each path started from the setup the last one left.
    rounds = {str(item.get("path")): item for item in result.rounds if item.get("path")}
    assert rounds["flash_attention"]["start_value"] is None
    assert rounds["prompt_batch"]["workload"] == "long-prefill-steady"
    assert rounds["prompt_batch"]["runs_per_measurement"] == tuner.settings.tune_path_runs
    assert rounds["threads"]["workload"] == "short-short"
    # Every kept change went through the fair check and its gain is the fair one.
    for change in result.proposed_changes:
        assert any("confirmed in turns" in item for item in change.evidence)
        assert change.measured_gain is not None and change.measured_gain > 0.05
    # Saved, and `start` would pass every setting on to llama-server.
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration == final
    command = build_server_command(
        Path("llama-server.exe"),
        _options_from(Path("m.gguf"), runtime_params_from(saved.configuration)),
        host="127.0.0.1",
        port=8080,
        api_key_file=Path("key"),
    )
    joined = " ".join(command)
    assert "-t 5" in joined and "-fa on" in joined and "-ub 256" in joined
    assert "-c 2048" in joined
    # The next tune starts from everything that was saved.
    again, _, _, _ = _tuner(tmp_path, stacking_responder, answers=[], settings=settings)
    start, _ = again._starting_point(PowerState.AC)
    assert start.threads == 5 and start.flash_attention is True and start.ubatch_size == 256


def test_a_path_with_nothing_better_changes_nothing(tmp_path: Path) -> None:
    def flat(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)

    settings = _settings(tmp_path, optimization_mode="performance_first")
    tuner, _, _, store = _tuner(tmp_path, flat, answers=[], settings=settings)
    result = tuner.run()
    assert result.proposed_changes == []
    assert result.outcome == "no change"
    paths = [item for item in result.rounds if item.get("path")]
    assert [item["path"] for item in paths] == [
        "threads",
        "reading_threads",
        "flash_attention",
        "prompt_batch",
    ]
    assert all("selected" not in item for item in paths)


def test_a_first_look_gain_that_does_not_hold_up_in_turns_is_dropped(tmp_path: Path) -> None:
    seen = {"threads5": 0}

    def lucky_once(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if options.threads == 5:
            seen["threads5"] += 1
            # Only the first measurement of 5 threads (the first look: one warm-up
            # run plus tune_path_runs measured runs) is fast.
            if seen["threads5"] <= 6:
                return SampleResult(decode_tps=24.0, total_latency_s=2.0)
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)

    settings = _settings(tmp_path, optimization_mode="performance_first", tune_paths="threads")
    tuner, _, _, _ = _tuner(tmp_path, lucky_once, answers=[], settings=settings)
    result = tuner.run()
    threads = next(item for item in result.rounds if item.get("path") == "threads")
    assert threads["best_first_look"] == "5 threads"
    assert "did not hold up in turns" in str(threads["reason"])
    assert result.proposed_changes == []


def test_balanced_names_the_paths_it_did_not_reach(tmp_path: Path) -> None:
    settings = _settings(tmp_path, optimization_before_loading_search_time_s=1.0)
    tuner, _, _, _ = _tuner(tmp_path, stacking_responder, answers=[True], settings=settings)

    def slow(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        time.sleep(0.05)
        return stacking_responder(task, options, call)

    tuner.harness = ScriptedHarness(slow)
    result = tuner.run()
    assert any("prompt batch size" in item for item in result.not_reached)


def test_a_path_whose_option_is_missing_is_skipped_with_its_reason(tmp_path: Path) -> None:
    settings = _settings(tmp_path, optimization_mode="performance_first", tune_paths="flash_attention")
    tuner, _, _, _ = _tuner(tmp_path, stacking_responder, answers=[True], settings=settings)
    tuner.supported_options = frozenset({"--threads", "--ubatch-size"})
    result = tuner.run()
    flash = next(item for item in result.rounds if item.get("path") == "flash_attention")
    assert flash["applies"] is False
    assert "--flash-attn" in str(flash["reason"])


def test_the_layer_choice_is_checked_in_turns(tmp_path: Path) -> None:
    from tests.test_tune import layer_gain_responder

    tuner, _, _, _ = _tuner(tmp_path, layer_gain_responder, answers=[True])
    result = tuner.run()
    layers = next(item for item in result.rounds if item.get("round") == "graphics card layers")
    fair: Any = layers.get("fair")
    assert fair is not None and fair["kept"] is True
    assert len(fair["turns"]) == tuner.settings.tune_fair_turns


def test_server_options_are_read_from_help(tmp_path: Path) -> None:
    import sys

    from inferenceos.tune.commands import read_server_options

    script = tmp_path / "fake_server.py"
    script.write_text(
        "print('-t,    --threads N   threads')\n"
        "print('-fa,   --flash-attn [on|off|auto]')\n"
        "print('-ub,   --ubatch-size N')\n",
        encoding="utf-8",
    )
    launcher = tmp_path / ("fake_server.cmd" if sys.platform == "win32" else "fake_server")
    if sys.platform == "win32":
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        launcher.write_text(f'#!/bin/sh\n"{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
    options = read_server_options(launcher)
    assert options is not None
    assert {"--threads", "--flash-attn", "--ubatch-size"} <= options
    assert read_server_options(tmp_path / "missing.exe") is None


def test_the_long_prompt_workload_always_writes_the_full_answer() -> None:
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.tune.paths import steady_workload

    default = get_workload("short-short")
    steady = steady_workload("long-prefill-steady", default=default)
    base = get_workload("long-prefill")
    assert steady.prompt == base.prompt and steady.n_predict == base.n_predict
    assert steady.ignore_eos is True and steady.workload_id == "long-prefill-steady"
    assert steady_workload("long-prefill", default=default) == base
    assert steady_workload("nope", default=default) is default


def test_an_unset_option_reloads_with_the_llama_cpp_default() -> None:
    """After a load with flash attention off, measuring a setup that does not set it
    must really go back to llama.cpp's default (found on the laptop, Oct 2026)."""
    from inferenceos.interfaces.backend_adapter import RuntimeParams
    from inferenceos.runtime.gpu_search import ResidentMeasurementAdapter

    class _Adapter:
        is_loaded = True

    class _Workload:
        adapter = _Adapter()

        def __init__(self) -> None:
            self.config = RuntimeConfiguration(context_length=CONTEXT, gpu_offload_layers=16)

        def model_profile(self) -> object:
            return sample_model()

        def configuration(self) -> RuntimeConfiguration:
            return self.config

        def set_configuration(self, config: RuntimeConfiguration) -> None:
            self.config = config

    for explicit, expected in ((True, None), (False, False)):
        workload = _Workload()
        adapter = ResidentMeasurementAdapter(workload, explicit_defaults=explicit)  # type: ignore[arg-type]
        adapter.load(sample_model(), RuntimeParams(gpu_offload_layers=16, flash_attention=False))
        assert workload.config.flash_attention is False
        adapter.load(sample_model(), RuntimeParams(gpu_offload_layers=16))
        assert workload.config.flash_attention is expected
        assert workload.config.context_length == CONTEXT


def test_ignore_eos_is_sent_with_each_request_when_the_workload_asks() -> None:
    """The long-prompt workload must write its full length on every model; LFM2 ended
    its answer at once on that prompt (found on the laptop, Oct 2026)."""
    from inferenceos.backends.llama_server_toolchain import LlamaServerClient

    sent: list[dict[str, object]] = []

    class _Response:
        def __iter__(self) -> Any:
            return iter([])

        def close(self) -> None:
            return None

    client = LlamaServerClient(host="127.0.0.1", port=1, api_key="k")

    def fake_open(path: str, *, body: dict[str, object], method: str, timeout_s: float | None) -> Any:
        sent.append(body)
        return _Response()

    client._open = fake_open  # type: ignore[method-assign, assignment]
    for flag in (True, False):
        list(
            client.stream_completion(
                prompt="p", n_predict=4, seed=1, temperature=0.0, stop=[],
                json_schema=None, ignore_eos=flag,
            )
        )
    assert sent[0].get("ignore_eos") is True
    assert "ignore_eos" not in sent[1]


def test_a_slow_first_reading_of_the_current_setup_does_not_make_candidates_look_better(
    tmp_path: Path,
) -> None:
    """Every prompt batch size does the same work here; only the first reading of the
    current setup in the round is slow (seen on the 9B, Oct 2026). With the second
    reading no candidate looks better, so no fair check is spent."""
    state = {"first_current_calls": 0, "round_started": False}

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if task.workload.workload_id.startswith("long-prefill") and options.ubatch_size is None:
            if state["first_current_calls"] < 6:  # warm-up + 5 runs of the first reading
                state["first_current_calls"] += 1
                return SampleResult(decode_tps=17.0, total_latency_s=2.4)
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)

    settings = _settings(tmp_path, optimization_mode="performance_first", tune_paths="prompt_batch")
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[], settings=settings)
    result = tuner.run()
    batch = next(item for item in result.rounds if item.get("path") == "prompt_batch")
    assert "current_reading_again" in batch
    screened = batch["screened"]
    assert isinstance(screened, list)
    assert all(float(entry["screen_gain_vs_first_reading"]) > 0.05 for entry in screened)
    assert all(float(entry["screen_gain"]) <= 0.0 for entry in screened)
    assert "fair" not in batch
    assert "nothing looked better" in str(batch["reason"])


def test_start_describes_every_saved_setting(tmp_path: Path) -> None:
    from inferenceos.tune.start import describe_setup

    settings = _settings(tmp_path, optimization_mode="performance_first")
    tuner, _, _, store = _tuner(
        tmp_path, stacking_responder, answers=[True, True, True], settings=settings
    )
    tuner.run()
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None
    text = describe_setup(saved)
    assert "5 threads" in text and "flash attention on" in text and "prompt batch 256" in text


def test_reading_thread_candidates_follow_the_writing_threads() -> None:
    from inferenceos.tune.paths import reading_thread_candidates

    # Writing with 8 threads: reading tries 3, 4 and 6 (8 is the current default).
    eight = BASE.model_copy(update={"threads": 8})
    assert [c.value for c in reading_thread_candidates(_context(eight))] == [3, 4, 6]
    # llama.cpp default writing threads (6 physical): 6 is not re-tried.
    assert [c.value for c in reading_thread_candidates(_context())] == [3, 4]


def test_reading_threads_are_found_and_reach_llama_server(tmp_path: Path) -> None:
    """Reading the question with 3 threads gives a 15% sooner first word on the long
    prompt (as the 9B benchmark suggested); writing is unchanged."""

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        latency = 2.0
        if task.workload.workload_id.startswith("long-prefill") and options.threads_batch == 3:
            latency *= 0.85
        return SampleResult(decode_tps=20.0, total_latency_s=latency)

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="reading_threads"
    )
    tuner, _, _, store = _tuner(tmp_path, responder, answers=[True], settings=settings)
    result = tuner.run()
    assert [change.setting for change in result.proposed_changes] == ["threads_batch"]
    assert result.final_configuration.threads_batch == 3
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.threads_batch == 3
    command = " ".join(
        build_server_command(
            Path("llama-server.exe"),
            _options_from(Path("m.gguf"), runtime_params_from(saved.configuration)),
            host="127.0.0.1",
            port=8080,
            api_key_file=Path("key"),
        )
    )
    assert "-tb 3" in command


class _ClockScriptedHarness(ScriptedHarness):
    """Scripted harness that also reports a processor-speed reading per run."""

    def __init__(self, responder: Any, clock: Any) -> None:
        super().__init__(responder)
        self._clock = clock

    def run_once(self, task: ProfileTask, **kwargs: Any) -> Any:
        run = super().run_once(task, **kwargs)
        speed = self._clock(task, kwargs["options"], kwargs["warmup"], len(self.calls))
        metrics = run.observation.metrics.model_copy(update={"cpu_performance_percent": speed})
        return run.model_copy(
            update={"observation": run.observation.model_copy(update={"metrics": metrics})}
        )


def _plain(options: ProfilerOptions) -> bool:
    return (
        options.threads is None
        and options.flash_attention is None
        and options.ubatch_size is None
        and options.threads_batch is None
    )


def test_a_decision_changed_by_heat_is_reported_to_the_user(tmp_path: Path) -> None:
    """5 threads is no better. During the threads fair check the laptop throttles
    while the current setup is measured in turn 2, which makes 5 threads look better.
    tune must leave that turn out, not keep 5 threads, and say clearly that heat
    decided it. The heat check at the end also sees the laptop slow down."""
    state = {"current_short": 0, "hot": False, "heat_runs": 0, "five": 0}

    def clock(task: ProfileTask, options: ProfilerOptions, warmup: bool, call: int) -> float:
        workload = task.workload.workload_id
        if workload == "short-long-decode":
            if not warmup:
                state["heat_runs"] += 1
            return 99.0 if state["heat_runs"] <= 2 else 72.0
        if workload == "short-short" and _plain(options) and warmup:
            state["current_short"] += 1
            # measurements of the current setup: 1 start, 2 first look, 3 first look
            # again, 4 fair turn 1, 5 fair turn 2 (throttled), 6 fair turn 3
            state["hot"] = state["current_short"] == 5
        if workload == "short-short" and _plain(options) and state["hot"]:
            return 70.0
        if options.threads == 5 and warmup:
            state["five"] += 1
        return 99.0

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if options.threads == 5 and state["five"] <= 1:
            # the first look at 5 threads happens to be fast, so it reaches the fair check
            return SampleResult(decode_tps=23.0, total_latency_s=2.0)
        if task.workload.workload_id == "short-long-decode":
            return SampleResult(decode_tps=20.0 if state["heat_runs"] <= 2 else 17.0)
        if task.workload.workload_id == "short-short" and _plain(options) and state["hot"]:
            return SampleResult(decode_tps=17.0, total_latency_s=2.3)
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)

    settings = _settings(
        tmp_path,
        optimization_mode="performance_first",
        tune_paths="threads",
        tune_heat_check_runs=6,
        tune_layer_path=False,
    )
    lines: list[str] = []
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[True], settings=settings, lines=lines)
    tuner.harness = _ClockScriptedHarness(responder, clock)
    result = tuner.run()

    assert result.proposed_changes == []
    threads = next(item for item in result.rounds if item.get("path") == "threads")
    fair: Any = threads["fair"]
    assert fair["uneven_turns"] == [2]
    assert fair["heat_changed_decision"] is True
    assert "REJECTED BECAUSE OF HEAT" in str(threads["reason"])
    report: Any = result.heat_report
    assert report["decisions_changed_by_heat"] == 1
    assert report["heat_check"]["throttled"] is True
    assert any("DECISION CHANGED BY HEAT" in line for line in lines)
    assert any(line.strip().startswith("HEAT:") for line in lines)


def test_a_cool_tune_reports_no_heat_and_decides_exactly_as_without_the_check(
    tmp_path: Path,
) -> None:
    def clock(task: ProfileTask, options: ProfilerOptions, warmup: bool, call: int) -> float:
        return 98.0 + (call % 3)  # steady: within 2 points

    settings = _settings(tmp_path, optimization_mode="performance_first", tune_heat_check_runs=4)
    lines: list[str] = []
    tuner, _, _, _ = _tuner(
        tmp_path, stacking_responder, answers=[True, True, True], settings=settings, lines=lines
    )
    tuner.harness = _ClockScriptedHarness(stacking_responder, clock)
    result = tuner.run()
    report: Any = result.heat_report
    assert report["decisions"] == [] and report["decisions_changed_by_heat"] == 0
    assert report["heat_check"]["throttled"] is False
    assert any("no comparison was disturbed by heat" in line for line in lines)
    # Same decisions as the run without any processor-speed readings.
    assert [c.setting for c in result.proposed_changes] == [
        "threads",
        "flash_attention",
        "ubatch_size",
    ]
    for item in result.rounds:
        fair = item.get("fair")
        if isinstance(fair, dict):
            assert fair["uneven_turns"] == []
