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
    # 6 physical / 12 logical: every second count up to 12, nearest to the default (6)
    # first; the default itself is not re-tried.
    assert [c.value for c in thread_candidates(_context())] == [4, 8, 2, 10, 12]
    current = BASE.model_copy(update={"threads": 5})
    assert [c.value for c in thread_candidates(_context(current))] == [6, 4, 8, 2, 10, 12]


def test_flash_candidates_try_both_when_llama_cpp_decides() -> None:
    assert [c.value for c in flash_candidates(_context())] == [True, False]
    on = BASE.model_copy(update={"flash_attention": True})
    assert [c.value for c in flash_candidates(_context(on))] == [False]


def test_prompt_batch_candidates_skip_the_current_size() -> None:
    assert [c.value for c in prompt_batch_candidates(_context())] == [64, 128, 256, 1024, 2048]
    small = BASE.model_copy(update={"ubatch_size": 256})
    assert [c.value for c in prompt_batch_candidates(_context(small))] == [64, 128, 512, 1024, 2048]


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
    if task.workload.workload_id.startswith("long-") and options.ubatch_size == 256:
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
    rounds: dict[str, dict[str, object]] = {}
    for item in result.rounds:
        if item.get("path") and not item.get("again"):
            rounds.setdefault(str(item.get("path")), item)
    assert rounds["flash_attention"]["start_value"] is None
    # Paths a kept change can affect were looked at again, on the new setup.
    looked_again = [item for item in result.rounds if item.get("again")]
    assert looked_again and all(item.get("path") for item in looked_again)
    assert str(rounds["prompt_batch"]["workload"]).startswith("long-question")
    assert rounds["prompt_batch"]["runs_per_measurement"] == tuner.settings.tune_path_runs
    assert rounds["threads"]["workload"] == "short-short"
    # Every kept change went through the fair check and its gain is the fair one.
    for change in result.proposed_changes:
        assert any("confirmed in turns" in item for item in change.evidence)
        # Fixed weights (balanced): +10% writing alone scores +5%.
        assert change.measured_gain is not None and change.measured_gain > 0.039
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
        "engine_switches",
    ]
    assert all("selected" not in item for item in paths)
    switches = next(item for item in paths if item["path"] == "engine_switches")
    assert switches["applies"] is False  # no timed runs, no engine seen: skipped with the reason


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
    assert len(fair["turns"]) == tuner.settings.tune_fair_min_blocks


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
        if task.workload.workload_id.startswith("long-question") and options.ubatch_size is None:
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
    assert all(float(entry["screen_gain_vs_each_reading"][0]) > 0.05 for entry in screened)
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

    # Writing with 8 threads: reading tries 6, 4, 3 and 12 (8 is the current default).
    eight = BASE.model_copy(update={"threads": 8})
    assert [c.value for c in reading_thread_candidates(_context(eight))] == [6, 4, 3, 12]
    # llama.cpp default writing threads (6 physical): 6 is not re-tried.
    assert [c.value for c in reading_thread_candidates(_context())] == [4, 3, 12]


def test_reading_threads_are_found_and_reach_llama_server(tmp_path: Path) -> None:
    """Reading the question with 3 threads gives a 15% sooner first word on the long
    prompt (as the 9B benchmark suggested); writing is unchanged."""

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        latency = 2.0
        if task.workload.workload_id.startswith("long-") and options.threads_batch == 3:
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


def test_processor_speed_changes_are_reported_and_every_block_counts(tmp_path: Path) -> None:
    """5 threads really writes 10% faster. In block 2 of its fair check it runs slower
    at a low processor speed. User decision (4 Oct 2026): that block counts (no block is
    left out for processor speed); the speed change is reported in plain words; the
    heat check at the end sees answers get slower and flags the result."""
    state = {"five": 0, "hot": False, "heat_runs": 0}

    def clock(task: ProfileTask, options: ProfilerOptions, warmup: bool, call: int) -> float:
        workload = task.workload.workload_id
        if workload == "short-long-decode":
            if not warmup:
                state["heat_runs"] += 1
            return 99.0 if state["heat_runs"] <= 2 else 72.0
        if options.threads == 5 and workload == "short-short" and warmup:
            state["five"] += 1
            # measurements of 5 threads: 1 first look, 2-5 fair blocks 1-4 (block 2 hot)
            state["hot"] = state["five"] == 3
        if options.threads == 5 and state["hot"]:
            return 70.0
        return 99.0

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if task.workload.workload_id == "short-long-decode":
            return SampleResult(decode_tps=20.0 if state["heat_runs"] <= 2 else 17.0)
        if options.threads == 5:
            if state["hot"]:
                return SampleResult(decode_tps=15.0, total_latency_s=2.6)
            return SampleResult(decode_tps=22.0, total_latency_s=2.0)
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

    threads = next(item for item in result.rounds if item.get("path") == "threads")
    fair: Any = threads["fair"]
    assert "uneven_turns" not in fair and "heat_changed_decision" not in fair
    assert "every block was counted" in fair["heat_effect"]
    report: Any = result.heat_report
    assert "decisions_changed_by_heat" not in report
    assert report["heat_check"]["throttled"] is True
    assert report["result_may_not_be_best"] is True
    assert any(line.strip().startswith("processor speed:") for line in lines)
    assert "answers got slower" in lines[-1]


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
    assert report["decisions"] == []
    assert report["heat_check"]["throttled"] is False
    assert any("the processor speed stayed steady in every comparison" in line for line in lines)
    # Same decisions as the run without any processor-speed readings.
    assert [c.setting for c in result.proposed_changes] == [
        "threads",
        "flash_attention",
        "ubatch_size",
    ]
    for item in result.rounds:
        fair = item.get("fair")
        if isinstance(fair, dict):
            assert fair["heat_effect"] == ""


def test_the_final_check_never_overrides_the_smallest_worthwhile_gain(tmp_path: Path) -> None:
    """Review 2 (4 Oct 2026): a whole setup 0.5% better against Balanced's 5% smallest
    gain used to come out keep=True with verdict not_worth_it. Now it is not kept and
    the keep flag and the verdict agree."""
    state = {"final": False}

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if options.threads == 5:
            return SampleResult(decode_tps=20.1 if state["final"] else 24.0, total_latency_s=2.0)
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)

    settings = _settings(tmp_path, tune_paths="threads", tune_layer_path=False)
    lines: list[str] = []

    class _Lines(list):
        def append(self, line: str) -> None:
            if line.startswith("Confirming"):
                state["final"] = True
            super().append(line)

    watched = _Lines()
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[True], settings=settings, lines=watched)
    tuner.harness = ScriptedHarness(responder)
    result = tuner.run()
    assert state["final"]
    assert [change.new_value for change in result.proposed_changes] == [5]
    assert result.outcome == "not confirmed"
    judgement = result.confirmation.judgement if result.confirmation else None
    assert judgement is not None and judgement.keep is False
    assert judgement.verdict.value != "kept"
    lines.extend(watched)


def test_a_saved_count_of_all_blocks_plus_one_reaches_the_server_unchanged() -> None:
    """Review 2: count L+1 (everything, output step included, on the card) must reach
    llama-server as -ngl L+1 from a saved setup."""
    model = sample_model()
    count = int(model.n_layers) + 1
    config = RuntimeConfiguration(context_length=CONTEXT, gpu_offload_layers=count)
    command = build_server_command(
        Path("llama-server.exe"),
        _options_from(Path("m.gguf"), runtime_params_from(config)),
        host="127.0.0.1",
        port=8080,
        api_key_file=Path("key"),
    )
    joined = " ".join(command)
    assert f"-ngl {count}" in joined and f"-c {CONTEXT}" in joined


def test_the_engines_own_prompt_reading_time_is_read_from_the_last_chunk() -> None:
    """4 Oct 2026: llama-server sends prompt_ms / prompt_per_second in the last streamed
    chunk; they were dropped, so prompt reading looked "not measured"."""
    import os

    from inferenceos.backends.resident_server import LlamaServerResidentBackend
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    class _Process:
        pid = os.getpid()

        def is_alive(self) -> bool:
            return True

    class _Client:
        def stream_completion(self, **_: Any) -> Any:
            yield {"content": "Hello"}
            yield {"content": " there"}
            yield {
                "stop": True,
                "timings": {
                    "prompt_n": 120, "prompt_ms": 240.0, "prompt_per_second": 500.0,
                    "predicted_n": 2, "predicted_ms": 100.0, "predicted_per_second": 20.0,
                    "predicted_per_token_ms": 50.0,
                },
            }

    backend = LlamaServerResidentBackend(server=Path("llama-server.exe"))
    backend._process = _Process()  # type: ignore[assignment]
    backend._client = _Client()  # type: ignore[assignment]
    result = backend.generate(GenerationRequest(prompt="p", n_predict=2))
    assert result.prompt_tokens_per_sec == 500.0
    assert result.prefill_s is not None and abs(result.prefill_s - 0.24) < 1e-9
    assert result.decode_tokens_per_sec == 20.0 and result.prompt_tokens == 120


def test_runs_per_measurement_give_the_most_certainty_per_minute() -> None:
    """4 Oct 2026: noisy, cheap runs get many runs; slow runs get fewer, because a
    fresh load (a new block) buys more certainty per minute there."""
    from inferenceos.tune.measure import choose_runs

    gemma_short = choose_runs(within=0.037, between=0.012, load_s=20, run_s=1.2, low=5, high=20)
    gemma_long = choose_runs(within=0.086, between=0.012, load_s=20, run_s=8.0, low=5, high=20)
    qwen_short = choose_runs(within=0.028, between=0.012, load_s=60, run_s=8.6, low=5, high=20)
    calm = choose_runs(within=0.005, between=0.012, load_s=20, run_s=1.0, low=5, high=20)
    assert gemma_short == 13 and gemma_long == 12 and qwen_short == 7 and calm == 5
    assert choose_runs(within=0.5, between=0.012, load_s=60, run_s=0.5, low=5, high=20) == 20


def test_tune_records_the_runs_per_measurement_it_chose(tmp_path: Path) -> None:
    settings = _settings(tmp_path, tune_paths="threads", tune_layer_path=False)
    tuner, _, _, _ = _tuner(tmp_path, stacking_responder, answers=[], settings=settings)
    result = tuner.run()
    runs = next(item for item in result.rounds if item.get("round") == "runs per measurement")
    chosen: Any = runs["runs"]
    assert 5 <= chosen["short-short"] <= int(settings.tune_path_max_runs)


def test_run_score_spread_combines_the_goal_measures() -> None:
    from inferenceos.contracts.experiment import ExperimentMeasurement
    from inferenceos.contracts.observation import RunMetrics
    from inferenceos.tune.measure import run_score_spread

    runs = [RunMetrics(ttft_s=t, decode_tokens_per_sec=d) for t, d in
            ((0.18, 30.0), (0.20, 30.0), (0.16, 30.0), (0.18, 30.0))]
    fields = ExperimentMeasurement.model_fields
    assert "run_metrics" in fields
    spread = run_score_spread(
        ExperimentMeasurement.model_construct(run_metrics=runs),
        {"ttft_s": 0.5, "decode_tokens_per_sec": 0.5},
    )
    assert spread is not None and 0.04 < spread < 0.05  # 0.5 x 9.1% from the first word


def test_a_path_is_looked_at_again_after_a_change_it_depends_on(tmp_path: Path) -> None:
    """Review 4 reproduction: threads gave nothing, then prompt batch 1024 was kept,
    and trying threads again found +15%. The loop now looks again at the paths a kept
    change can affect (owner: at most 2 extra times each, within the time left)."""

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        speed, latency = 20.0, 2.0
        if task.workload.workload_id.startswith("long-") and options.ubatch_size == 1024:
            latency *= 0.80
        if options.ubatch_size == 1024 and options.threads == 5:
            speed *= 1.15
        return SampleResult(decode_tps=speed, total_latency_s=latency)

    settings = _settings(
        tmp_path,
        optimization_mode="performance_first",
        tune_paths="threads,prompt_batch",
        tune_diagnosis=False,
        tune_layer_path=False,
    )
    lines: list[str] = []
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[True, True], settings=settings, lines=lines)
    result = tuner.run()
    threads_rounds = [item for item in result.rounds if item.get("path") == "threads"]
    assert len(threads_rounds) == 2 and threads_rounds[1].get("again") is True
    assert "proposed_change" not in threads_rounds[0] or threads_rounds[0].get("selected") is None
    final = result.final_configuration
    assert final.ubatch_size == 1024 and final.threads == 5
    assert sorted(change.setting for change in result.proposed_changes) == ["threads", "ubatch_size"]
    assert any("will look again at processor threads" in line for line in lines)


def test_a_setting_changed_twice_becomes_one_change_from_the_start() -> None:
    from inferenceos.contracts.tune import ProposedChange
    from inferenceos.tune.pipeline import merge_change

    start = RuntimeConfiguration(context_length=CONTEXT, threads=8)

    def change(old: int, new: int) -> ProposedChange:
        return ProposedChange(
            change_id=f"runtime.threads:{new}", card_id="runtime.threads", setting="threads",
            description="threads", old_value=old, new_value=new, measured_gain=0.03,
            evidence=[f"{old} -> {new}"],
        )

    workloads: dict[str, list[Any]] = {}
    first = merge_change([], change(8, 10), workloads, start)
    merged = merge_change(first, change(10, 12), workloads, start)
    assert len(merged) == 1 and merged[0].old_value == 8 and merged[0].new_value == 12
    assert merged[0].measured_gain is None  # only step gains were measured (review 5)
    assert any("that step only" in item for item in merged[0].evidence)
    assert any("looked at again" in item for item in merged[0].evidence)
    # Back at the starting value: no longer a change.
    assert merge_change(merged, change(12, 8), workloads, start) == []


def test_a_quick_probe_that_finds_nothing_says_which_values_it_never_tried(tmp_path: Path) -> None:
    """Review 4 reproduction: a probe of 6 and 8 threads found nothing while 12 was
    best. A budgeted probe may stop there, but the report must name what it skipped."""
    from inferenceos.tune.diagnose import PROBE, Diagnosis

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(decode_tps=24.0 if options.threads == 12 else 20.0, total_latency_s=2.0)

    settings = _settings(tmp_path, tune_paths="threads", tune_layer_path=False)
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[], settings=settings)
    probe_only = Diagnosis(priorities={"threads": (PROBE, "test")})
    tuner._diagnose = lambda *args, **kwargs: (probe_only, {"round": "diagnosis (test)"})  # type: ignore[method-assign, assignment]
    result = tuner.run()
    path: Any = next(item for item in result.rounds if item.get("path") == "threads")
    assert path["depth"] == "probe" and not path.get("probe_became_full")
    assert "12 threads" not in path["candidates"]
    assert "quick probe only, not tried" in str(path["reason"])
    assert "(this does not show they cannot help)" in str(path["reason"])
    assert "12 threads" in path["untested"]


def test_the_goal_score_spread_uses_each_runs_measures_together() -> None:
    """Review 4: measures were combined as if independent. When a run's first word and
    writing speed move in opposite directions, their goal scores mostly cancel."""
    from inferenceos.contracts.experiment import ExperimentMeasurement
    from inferenceos.contracts.observation import RunMetrics
    from inferenceos.tune.measure import run_score_spread

    # Slower first word comes with faster writing, by the same factor.
    runs = [RunMetrics(ttft_s=0.2 * f, decode_tokens_per_sec=30.0 * f) for f in (1.0, 1.1, 0.9, 1.05)]
    spread = run_score_spread(
        ExperimentMeasurement.model_construct(run_metrics=runs),
        {"ttft_s": 0.5, "decode_tokens_per_sec": 0.5},
    )
    assert spread is not None and spread < 1e-9


def _timed_measurement(starts: list[float], length: float) -> Any:
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from inferenceos.contracts.experiment import ExperimentMeasurement

    zero = datetime(2026, 10, 4, tzinfo=timezone.utc)
    runs = [
        SimpleNamespace(
            identity=SimpleNamespace(
                started_at=zero + timedelta(seconds=start),
                finished_at=zero + timedelta(seconds=start + length),
            )
        )
        for start in starts
    ]
    return ExperimentMeasurement.model_construct(runs=runs, valid_runs=len(runs))


def test_recurring_per_answer_cost_is_not_counted_as_load_time() -> None:
    """Review 4 reproduction: the profiler's per-answer telemetry was counted as load
    time, so more runs looked cheap (12 runs instead of 5 in the reviewer's case)."""
    from inferenceos.tune.measure import answer_costs, choose_runs

    # Load 10 s; each answer: 1 s of engine time plus 2 s of telemetry start and wait.
    measurement = _timed_measurement([0.0, 3.0, 6.0, 9.0, 12.0, 15.0], length=1.0)
    wall_s = 10.0 + 16.0
    per_answer, load = answer_costs(measurement, wall_s)
    assert per_answer is not None and abs(per_answer - 16.0 / 6) < 1e-9
    assert load is not None and abs(load - 10.0) < 1e-9
    old_load = wall_s - 6 * 1.0  # everything outside the engine's answer time
    old = choose_runs(within=0.05, between=0.012, load_s=old_load, run_s=1.0, low=5, high=20)
    new = choose_runs(within=0.05, between=0.012, load_s=load, run_s=per_answer, low=5, high=20)
    assert new < old


def test_load_to_load_spread_is_estimated_from_fresh_loads_and_saved(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from inferenceos.contracts.experiment import ExperimentMeasurement
    from inferenceos.tune.measure import between_load_spread
    from inferenceos.tune.noise_store import NoiseStore, noise_key

    def reading(speed: float) -> Any:
        return ExperimentMeasurement.model_construct(
            metrics={"decode_tokens_per_sec": SimpleNamespace(median=speed)},
            run_metrics=[],
            valid_runs=10,
        )

    loads = [reading(v) for v in (30.0, 30.6, 29.4, 30.3, 29.7, 30.0, 30.4, 29.6)]
    spread = between_load_spread(loads, {"decode_tokens_per_sec": 1.0})
    assert spread is not None and 0.008 < spread < 0.02
    assert between_load_spread(loads[:4], {"decode_tokens_per_sec": 1.0}) is None
    # Laptop job 092: a near-zero estimate from few loads must not go under the floor.
    flat = [reading(30.0)] * 8
    assert between_load_spread(flat, {"decode_tokens_per_sec": 1.0}) == 0.005
    store = NoiseStore(tmp_path / "load_spread.json")
    key = noise_key("gemma", 1, "short-short")
    assert store.estimate(key) is None
    store.add(key, 0.010)
    assert store.estimate(key) is None  # one comparison alone is not enough
    store.add(key, 0.020)
    store.add(key, 0.030)
    assert store.estimate(key) == 0.020
    assert key.split("|")[1] == "card"


def test_paths_check_lightly_and_the_final_check_stays_full(tmp_path: Path) -> None:
    """Owner, 4 Oct 2026 (after scripts/simulate_tune.py): each path's best value gets a
    short check (one look at 4 blocks, looser), only to choose; one fresh full final
    check decides. Balanced's final "must not lose" checks look at 8 and 16 blocks."""
    settings = _settings(tmp_path, tune_paths="prompt_batch", tune_layer_path=False, tune_diagnosis=False)
    assert settings.tune_path_check == "light"
    tuner, _, _, _ = _tuner(tmp_path, stacking_responder, answers=[True], settings=settings)
    result = tuner.run()
    path: Any = next(item for item in result.rounds if item.get("path") == "prompt_batch")
    assert path["path_check"] == "light"
    assert path["fair"]["looks"] == [4]
    final: Any = next(item for item in result.rounds if item.get("round") == "final check")
    guard = next(item for item in final["checks"] if item["must_be"] == "not worse")
    assert guard["fair"]["looks"][0] == 8 and guard["fair"]["looks"][-1] in (8, 16)


def test_a_light_guard_check_leaves_unsure_to_the_final_check(tmp_path: Path) -> None:
    """Laptop job 090: flash attention off (+21% on the long question) was dropped on an
    unsure -1.8% on the short question after 4 blocks. A light check only screens: it
    drops what is shown worse or already beyond the loss limit; the full final check
    decides the rest."""

    import math
    import random

    noise = random.Random(3)

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        speed, latency = 20.0, 2.0
        if task.workload.workload_id.startswith("long-"):
            if options.ubatch_size == 256:
                latency *= 0.80
        else:
            # Noisy short question; 256 truly loses 1% there (inside the 2% limit),
            # too little to show "not worse" in 4 blocks.
            speed *= math.exp(noise.gauss(0.0, 0.06)) * (0.99 if options.ubatch_size == 256 else 1.0)
        return SampleResult(decode_tps=speed, total_latency_s=latency)

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="prompt_batch",
        tune_layer_path=False, tune_diagnosis=False,
    )
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[True], settings=settings)
    result = tuner.run()
    path: Any = next(item for item in result.rounds if item.get("path") == "prompt_batch")
    assert path["guard_fair"]["decision"] == "unsure"
    assert path.get("guard_left_to_final_check") is True and path.get("selected") == 256
    final: Any = next(item for item in result.rounds if item.get("round") == "final check")
    assert any(item["must_be"] == "not worse" for item in final["checks"])


def _tradeoff_responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    """Prompt batch 256: long question 25% sooner first word, short question writes 6%
    slower (a 3% goal loss, beyond the 2% limit)."""
    speed, latency = 20.0, 2.0
    if options.ubatch_size == 256:
        if task.workload.workload_id.startswith("long-"):
            latency *= 0.75
        else:
            speed *= 0.94
    return SampleResult(decode_tps=speed, total_latency_s=latency)


def test_a_tradeoff_is_never_kept_on_its_own_but_offered(tmp_path: Path) -> None:
    """Owner, 5 Oct 2026 (Gemma: flash attention off, long +24%, short -2.4%): keep the
    2% guard for every keep tune makes itself, --yes included; measure the trade-off
    properly and offer it with both numbers; only an explicit yes keeps it."""
    from inferenceos.tune.approval import ApproveAllChanges

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="prompt_batch",
        tune_layer_path=False, tune_diagnosis=False,
    )
    tuner, _, _, store = _tuner(tmp_path, _tradeoff_responder, answers=[], settings=settings)
    tuner.approver = ApproveAllChanges()
    result = tuner.run()
    offer: Any = next(item for item in result.rounds if item.get("round") == "trade-off")
    # --yes can never keep a trade-off, so it is reported but not measured (job 102).
    assert offer["measured"] is False and "prompt batch 256" in offer["found"]
    assert "--yes never keeps a trade-off" in offer["why"]
    assert result.final_configuration.ubatch_size is None


def test_a_tradeoff_is_measured_and_offered_when_the_user_can_answer(tmp_path: Path) -> None:
    from inferenceos.tune.approval import ScriptedChangeApprover

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="prompt_batch",
        tune_layer_path=False, tune_diagnosis=False,
    )
    tuner, _, _, store = _tuner(tmp_path, _tradeoff_responder, answers=[], settings=settings)
    tuner.approver = ScriptedChangeApprover([], tradeoff_answers=[False])
    result = tuner.run()
    offer: Any = next(item for item in result.rounds if item.get("round") == "trade-off")
    assert offer["value"] == 256 and "long-question" in offer["words"] and "short-short" in offer["words"]
    assert offer["decision"]["approved"] is False
    assert result.final_configuration.ubatch_size is None


def test_a_tradeoff_is_kept_only_with_an_explicit_yes(tmp_path: Path) -> None:
    from inferenceos.tune.approval import ScriptedChangeApprover

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="prompt_batch",
        tune_layer_path=False, tune_diagnosis=False,
    )
    tuner, _, _, store = _tuner(tmp_path, _tradeoff_responder, answers=[], settings=settings)
    approver = ScriptedChangeApprover([], tradeoff_answers=[True])
    tuner.approver = approver
    result = tuner.run()
    assert approver.tradeoffs_asked and "beyond or not shown within your 2% limit" in approver.tradeoffs_asked[0]
    assert result.final_configuration.ubatch_size == 256
    assert result.outcome == "saved (with a trade-off you approved)"
    assert result.saved_setup is not None and result.saved_setup.configuration.ubatch_size == 256


def _tradeoff_world(*, wrong_answers: bool = False, paging: bool = False) -> Any:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        sample = _tradeoff_responder(task, options, call)
        if options.ubatch_size == 256 and not task.workload.workload_id.startswith("long-"):
            if wrong_answers:
                sample.text = "Bananas are yellow and 7 + 5 = 13."
            if paging:
                sample.hard_faults_per_sec = 11_000.0
        return sample

    return responder


def _run_with_yes(tmp_path: Path, responder: Any, **patch: Any) -> Any:
    from inferenceos.tune.approval import ScriptedChangeApprover

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="prompt_batch",
        tune_layer_path=False, tune_diagnosis=False,
    )
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[], settings=settings)
    approver = ScriptedChangeApprover([], tradeoff_answers=[True])
    tuner.approver = approver
    return tuner.run(), approver


def test_a_tradeoff_with_changed_answers_is_never_offered(tmp_path: Path) -> None:
    """Review 5: a 'yes' saved a trade-off whose answers were wrong on the guarded
    question. A yes may relax only the speed-loss limit."""
    result, approver = _run_with_yes(tmp_path, _tradeoff_world(wrong_answers=True))
    offer: Any = next(item for item in result.rounds if item.get("round") == "trade-off")
    assert "answers changed" in offer["reason"]
    assert approver.tradeoffs_asked == []
    assert result.final_configuration.ubatch_size is None


def test_a_tradeoff_that_pages_heavily_is_never_offered(tmp_path: Path) -> None:
    result, approver = _run_with_yes(tmp_path, _tradeoff_world(paging=True))
    offer: Any = next((item for item in result.rounds if item.get("round") == "trade-off"), None)
    assert offer is not None and "guard failed" in offer["reason"]
    assert approver.tradeoffs_asked == []
    assert result.final_configuration.ubatch_size is None


def test_a_tradeoff_not_measured_to_the_end_is_never_offered(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Review 5: a 'yes' saved a trade-off whose guarded question was never measured
    (the time limit ran out)."""
    from inferenceos.tune import pipeline

    real = pipeline.compare_fairly

    def out_of_time(**kwargs: Any) -> Any:
        if str(kwargs.get("phase", "")).startswith("tune-tradeoff-short"):
            kwargs["deadline"] = time.perf_counter() - 1.0
        return real(**kwargs)

    monkeypatch.setattr(pipeline, "compare_fairly", out_of_time)
    result, approver = _run_with_yes(tmp_path, _tradeoff_responder)
    offer: Any = next(item for item in result.rounds if item.get("round") == "trade-off")
    assert "not measured to the end" in offer["reason"]
    assert approver.tradeoffs_asked == []
    assert result.final_configuration.ubatch_size is None


def test_final_checks_teach_the_load_to_load_spread(tmp_path: Path) -> None:
    """Review 5: only light path checks (4 loads) fed the spread estimate, which needs
    8; the final checks now feed it too."""
    import json

    settings = _settings(tmp_path, tune_paths="prompt_batch", tune_layer_path=False, tune_diagnosis=False)
    tuner, _, _, _ = _tuner(tmp_path, stacking_responder, answers=[True], settings=settings)
    result = tuner.run()
    final: Any = next(item for item in result.rounds if item.get("round") == "final check")
    blocks = max(len(item["fair"].get("turns", [])) for item in final["checks"])
    assert blocks >= 8
    saved = json.loads((settings.data_dir / "load_spread.json").read_text(encoding="utf-8"))
    assert any(key.endswith("|short-short") for key in saved)
