"""Engine switches and the engine probe (4 Oct 2026)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inferenceos.backends.engine_env import format_switches, parse_switches, with_switch
from inferenceos.backends.resident_server import _options_from
from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.experiment.restoration import runtime_configuration_from, runtime_params_from
from inferenceos.interfaces.backend_adapter import GenerationResult
from inferenceos.selftest import sample_model
from inferenceos.tune.engine_probe import parse_timing_blocks, run_probe


def test_switch_text_is_canonical_and_round_trips() -> None:
    text = format_switches({"GGML_VK_DISABLE_FUSION": "1", "GGML_VK_DISABLE_ASYNC": "1"})
    assert text == "GGML_VK_DISABLE_ASYNC=1;GGML_VK_DISABLE_FUSION=1"
    assert parse_switches(text) == {"GGML_VK_DISABLE_ASYNC": "1", "GGML_VK_DISABLE_FUSION": "1"}
    assert with_switch(text, "GGML_VK_DISABLE_ASYNC", None) == "GGML_VK_DISABLE_FUSION=1"
    assert with_switch(None, "A_B", "2") == "A_B=2"
    assert format_switches({}) is None and parse_switches(None) == {}
    with pytest.raises(ValueError):
        parse_switches("lower=1")


def test_switches_reach_the_engine_process_and_survive_a_saved_setup() -> None:
    config = RuntimeConfiguration(
        context_length=4096, gpu_offload_layers=8, engine_switches="GGML_VK_DISABLE_FUSION=1"
    )
    params = runtime_params_from(config)
    assert params.engine_switches == "GGML_VK_DISABLE_FUSION=1"
    assert runtime_configuration_from(params).engine_switches == "GGML_VK_DISABLE_FUSION=1"
    options = _options_from(Path("m.gguf"), params)
    assert options.environment == {"GGML_VK_DISABLE_FUSION": "1"}


def test_the_process_gets_its_switches_on_top_of_the_inherited_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inferenceos.backends import llama_server_toolchain as srv

    seen: dict[str, Any] = {}

    class _Popen:
        pid = 4242
        stdout = None
        stderr = None

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            seen.update(kwargs)

    class _Guard:
        kind = "test"

        def assign(self, pid: int) -> None:
            return None

    monkeypatch.setattr(srv.subprocess, "Popen", _Popen)
    monkeypatch.setenv("INFERENCEOS_TEST_KEEP", "yes")
    process = srv.LlamaServerProcess(
        ["llama-server"], guard=_Guard(), environment={"GGML_VK_DISABLE_FUSION": "1"}  # type: ignore[arg-type]
    )
    process.start()
    assert seen["env"]["GGML_VK_DISABLE_FUSION"] == "1"
    assert seen["env"]["INFERENCEOS_TEST_KEEP"] == "yes"
    # An inherited timer never reaches a speed-test process (review 4, 4 Oct 2026).
    monkeypatch.setenv("GGML_VK_PERF_LOGGER", "1")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    plain = srv.LlamaServerProcess(["llama-server"], guard=_Guard())  # type: ignore[arg-type]
    seen.clear()
    plain.start()
    assert "GGML_VK_PERF_LOGGER" not in seen["env"]
    assert "CUDA_LAUNCH_BLOCKING" not in seen["env"]
    assert seen["env"]["INFERENCEOS_TEST_KEEP"] == "yes"
    assert "GGML_VK_PERF_LOGGER" not in plain.engine_settings
    # A diagnostic run asks for it explicitly.
    timed = srv.LlamaServerProcess(
        ["llama-server"], guard=_Guard(), environment={"GGML_VK_PERF_LOGGER": "1"}  # type: ignore[arg-type]
    )
    seen.clear()
    timed.start()
    assert seen["env"]["GGML_VK_PERF_LOGGER"] == "1"
    assert timed.engine_settings["GGML_VK_PERF_LOGGER"] == "1"


_LOG = """llama-server starting
ggml_vulkan: 0 = AMD Radeon Pro 560X (AMD proprietary driver) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: none
----------------
Vulkan Timings:
MUL_MAT q4_K m=1152 n=17 k=1152: 26 x 400 us = 10400 us
Total time: 10400 us.
----------------
Vulkan Timings:
MUL_MAT_VEC q4_K m=1152 n=1 k=1152: 26 x 20.5 us = 533 us
RMS_NORM+MUL: 52 x 3.0 us = 156 us
Total time: 689 us.
----------------
Vulkan Timings:
MUL_MAT_VEC q4_K m=1152 n=1 k=1152: 26 x 21.5 us = 559 us
RMS_NORM+MUL: 52 x 3.0 us = 156 us
Total time: 715 us."""


def test_timing_blocks_are_split_into_reading_and_writing() -> None:
    graphs = parse_timing_blocks(_LOG.splitlines())
    assert [graph.kind for graph in graphs] == ["reading", "writing", "writing"]
    assert graphs[1].total_us == 689.0 and graphs[1].ops[1].name == "RMS_NORM+MUL"


def test_a_probe_turns_the_timer_on_and_reads_the_log() -> None:
    started: dict[str, Any] = {}

    class _Backend:
        def start(self, model: Any, params: Any) -> None:
            started["switches"] = params.engine_switches

        def generate(self, request: Any) -> GenerationResult:
            assert request.temperature == 0.0 and request.ignore_eos
            return GenerationResult(
                text="A cache.", decode_tokens_per_sec=50.0, prefill_s=0.2, prompt_tokens=17
            )

        def server_log(self) -> str:
            return _LOG

        def stop(self) -> None:
            started["stopped"] = True

    config = RuntimeConfiguration(context_length=4096, engine_switches="GGML_VK_DISABLE_FUSION=1")
    result = run_probe(
        settings=Settings(), model=sample_model(), config=config, repeats=2,
        backend_factory=lambda: _Backend(),  # type: ignore[arg-type, return-value]
    )
    assert started["switches"] == (
        "GGML_VK_DISABLE_FUSION=1;GGML_VK_PERF_LOGGER=1;GGML_VK_PIPELINE_STATS=_"
    )
    assert started["stopped"] and result.ready and result.answers == ["A cache.", "A cache."]
    assert result.card_us_per_word() == 702.0
    assert result.writing_ms_per_word == [20.0, 20.0]
    share = result.card_share()
    assert share is not None and abs(share - 702.0 / 20000.0) < 1e-9
    assert result.device_features()["int dot"] == "1"
    assert result.device_features()["matrix cores"] == "none"
    assert result.config.engine_switches == "GGML_VK_DISABLE_FUSION=1"  # timer never saved


def _probe(
    card_word_us: float,
    ops: tuple[str, ...] = ("MUL_MAT_VEC q4_K m=8 n=1 k=8",),
    answers: tuple[str, ...] = ("A", "B", "C"),
    kernels: frozenset[str] | None = frozenset({"mul_mat_vec_q4_k_f32_f32"}),
    choices: list[Any] | None = None,
    error: str = "",
) -> Any:
    from inferenceos.tune.engine_probe import Graph, OpTime, ProbeResult

    result = ProbeResult(config=RuntimeConfiguration(context_length=4096), ready=True)
    graph_ops = tuple(OpTime(name, 1, card_word_us) for name in ops)
    result.graphs = [Graph("writing", card_word_us, graph_ops)] * 3
    result.answers = list(answers)
    result.expected_answers = 3
    result.token_choices = choices if choices is not None else [None] * len(answers)
    result.error = error
    result.evidence = {"steps": frozenset(f"{name} x1" for name in ops)}
    if kernels is not None:
        result.evidence["kernels"] = kernels
    return result


def test_a_timing_change_alone_is_not_proof_a_switch_took_effect() -> None:
    """Review 4: identical setups and steps, card time 5.2 -> 5.6 ms, was called
    "took effect". Only what the engine prints about how it ran counts now."""
    from inferenceos.tune.engine_switches import NOT_SEEN, SHOWN, UNVERIFIED, activation

    base, again = _probe(5200.0), _probe(5200.0)
    assert activation(("steps", "kernels"), base, again, _probe(5600.0))[0] == NOT_SEEN
    shown = activation(("steps",), base, again, _probe(5200.0, ops=("MUL_MAT_VEC q4_K m=8 n=1 k=8", "ADD")))
    assert shown[0] == SHOWN and "steps" in shown[1]
    q8 = frozenset({"mul_mat_vec_q4_k_q8_1_f32", "quantize_q8_1_x4"})
    assert activation(("kernels",), base, again, _probe(5200.0, kernels=q8))[0] == SHOWN
    # Nothing the engine prints shows this switch: not tested, and said so.
    status, why = activation((), base, again, _probe(9000.0))
    assert status == UNVERIFIED and "activation unverified; not tested" in why
    # The channel is missing in this run (e.g. the driver cannot list programs).
    no_kernels = _probe(5200.0, kernels=None)
    assert activation(("kernels",), no_kernels, no_kernels, no_kernels)[0] == UNVERIFIED
    # The unchanged runs disagree with each other: not usable as evidence.
    other_kernels = _probe(5200.0, kernels=frozenset({"x"}))
    assert activation(("kernels",), base, other_kernels, _probe(5200.0, kernels=q8))[0] == UNVERIFIED


def _choices(*words: tuple[str, float, float | None]) -> list[Any]:
    from inferenceos.interfaces.backend_adapter import TokenChoice

    return [TokenChoice(token=t, prob=p, runner_up_prob=r) for t, p, r in words]


def test_every_answer_is_checked_and_runs_must_be_complete() -> None:
    """Review 4: only the first answer was compared, and a probe whose generation
    failed still counted as ready."""
    from inferenceos.tune.engine_switches import answers_agree

    base = _probe(5200.0)
    assert answers_agree(base, _probe(5200.0))[0]
    later_wrong = _probe(5200.0, answers=("A", "B", "WRONG"))
    ok, why = answers_agree(base, later_wrong)
    assert not ok and "answer 3" in why
    partial = _probe(5200.0, answers=("A",), error="ResidentBackendError: generation failed")
    ok, why = answers_agree(base, partial)
    assert not ok and "incomplete" in why
    assert not answers_agree(_probe(5200.0, error="boom"), base)[0]


def test_word_probabilities_allow_small_moves_but_no_different_word() -> None:
    """Review 5: a near tie in the unchanged run let any replacement word pass and
    skipped the rest of the answer. Now any different word fails."""
    from inferenceos.tune.engine_switches import answers_agree

    clear = _choices(("The", 0.90, 0.05), (" cache", 0.80, 0.10), (" keeps", 0.60, 0.30))
    base = _probe(1.0, choices=[clear, clear, clear])
    # Small numeric differences: fine.
    nudged = _choices(("The", 0.89, 0.05), (" cache", 0.81, 0.10), (" keeps", 0.61, 0.30))
    ok, why = answers_agree(base, _probe(1.0, choices=[nudged, clear, clear]))
    assert ok and "identical" in why
    # A probability moved more than the tolerance.
    moved = _choices(("The", 0.70, 0.05), (" cache", 0.80, 0.10), (" keeps", 0.60, 0.30))
    ok, why = answers_agree(base, _probe(1.0, choices=[clear, clear, moved]))
    assert not ok and "answer 3" in why and "moved" in why
    # A different word fails, clear first choice or near tie alike.
    swapped = _choices(("A", 0.90, 0.05), (" cache", 0.80, 0.10), (" keeps", 0.60, 0.30))
    assert not answers_agree(base, _probe(1.0, choices=[swapped, clear, clear]))[0]
    tie = _choices(("The", 0.50, 0.48), (" cache", 0.80, 0.10))
    other = _choices(("A", 0.49, 0.50), (" store", 0.70, 0.10))
    ok, why = answers_agree(
        _probe(1.0, choices=[tie, clear, clear]), _probe(1.0, choices=[other, clear, clear])
    )
    assert not ok and "different word" in why
    # A changed answer text fails even without probabilities.
    assert not answers_agree(_probe(1.0), _probe(1.0, answers=("A", "B", "D")))[0]


def test_switches_with_conflicts_replace_each_other_and_go_back_to_default() -> None:
    from inferenceos.tune.engine_switches import (
        VULKAN_B11139,
        apply_switch,
        remove_switch,
        switches_for,
    )

    force = next(spec for spec in VULKAN_B11139 if spec.name == "GGML_VK_FORCE_MMVQ")
    disable = next(spec for spec in VULKAN_B11139 if spec.name == "GGML_VK_DISABLE_MMVQ")
    assert apply_switch("GGML_VK_DISABLE_MMVQ=1", force) == "GGML_VK_FORCE_MMVQ=1"
    assert apply_switch("GGML_VK_FORCE_MMVQ=1;GGML_VK_DISABLE_FUSION=1", disable) == (
        "GGML_VK_DISABLE_FUSION=1;GGML_VK_DISABLE_MMVQ=1"
    )
    assert remove_switch("GGML_VK_FORCE_MMVQ=1", "GGML_VK_FORCE_MMVQ") is None
    # A card without integer dot products is never asked to force that code.
    version = "0.4.1-dev (build 11139, commit 42916d83f)"
    names = {spec.name for spec in switches_for("vulkan", version, {"int dot": "0"})}
    assert "GGML_VK_FORCE_MMVQ" not in names and "GGML_VK_DISABLE_MMVQ" in names
    assert "GGML_VK_FORCE_MMVQ" in {spec.name for spec in switches_for("vulkan", version)}


def test_switches_are_listed_only_for_the_checked_build() -> None:
    from inferenceos.tune.engine_switches import switches_for

    assert switches_for("vulkan", "version: 0.4.1-dev (build 11139, commit 42916d83f)")
    assert switches_for("vulkan", "version: 0.5.0 (build 12000, commit abcdef123)") == ()
    assert switches_for("cuda", "version: 0.4.1-dev (build 11139, commit 42916d83f)") == ()


_FUSED = ("MUL_MAT_VEC q4_K m=8 n=1 k=8",)
_UNFUSED = ("MUL_MAT_VEC q4_K m=8 n=1 k=8", "ADD", "RMS_NORM")
_Q8 = frozenset({"mul_mat_vec_q4_k_q8_1_f32", "quantize_q8_1_x4"})


def _switch_tuner(tmp_path: Path, *, word_ms: float, faster_with: str | None = None) -> Any:
    """A tune with only the engine-switch path. The timed runs show fusion off as other
    steps and forced integer-dot code as other programs; every other switch shows
    nothing. ``faster_with``: the switch that really makes answers 10% faster."""
    from inferenceos.backends.engine_env import parse_switches
    from inferenceos.contracts.experiment import RuntimeConfiguration as Config
    from tests.experiment_helpers import SampleResult
    from tests.test_tune import _settings, _tuner

    def responder(task: Any, options: Any, call: int) -> SampleResult:
        fast = faster_with is not None and faster_with in parse_switches(options.engine_switches)
        return SampleResult(decode_tps=22.0 if fast else 20.0, total_latency_s=2.0)

    def probe_runner(config: Config, **options: Any) -> Any:
        switches = parse_switches(config.engine_switches)
        result = _probe(
            60000.0,
            ops=_UNFUSED if "GGML_VK_DISABLE_FUSION" in switches else _FUSED,
            kernels=_Q8 if "GGML_VK_FORCE_MMVQ" in switches else frozenset({"mul_mat_vec_q4_k_f32_f32"}),
        )
        result.engine_seen = "vulkan"
        result.writing_ms_per_word = [word_ms]
        return result

    settings = _settings(tmp_path, tune_paths="engine_switches", tune_layer_path=False)
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[True, True], settings=settings)
    tuner.probe_runner = probe_runner
    tuner.machine = tuner.machine.model_copy(
        update={"program_version": "version: 0.4.1-dev (build 11139, commit 42916d83f)"}
    )
    return tuner


def test_tune_tries_only_switches_shown_to_take_effect(tmp_path: Path) -> None:
    tuner = _switch_tuner(tmp_path, word_ms=65.0)  # the card does most of the work
    result = tuner.run()
    checks = next(item for item in result.rounds if item.get("round") == "engine switch checks")
    assert sorted(checks["passed"]) == [
        "always use the integer dot-product code for one-word steps (GGML_VK_FORCE_MMVQ=1)",
        "do not merge neighbouring steps (GGML_VK_DISABLE_FUSION=1)",
    ]
    by_switch = {entry["switch"]: entry for entry in checks["checks"]}
    unseen = by_switch["copy data on the separate transfer queue (GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1)"]
    assert unseen["activation"] == "unverified"
    assert "activation unverified; not tested" in unseen["effect"]
    assert result.proposed_changes == []  # flat speed: nothing kept


def test_switches_the_engine_cannot_show_get_no_timed_run(tmp_path: Path) -> None:
    """LFM2 tune 099: switches with no evidence channel were still run, although a run
    can never let them through. They are now reported without a run."""
    from inferenceos.backends.engine_env import parse_switches

    tuner = _switch_tuner(tmp_path, word_ms=65.0)
    inner = tuner.probe_runner
    probed: list[str] = []

    def counting(config: Any, **options: Any) -> Any:
        probed.extend(parse_switches(config.engine_switches))
        return inner(config, **options)

    tuner.probe_runner = counting
    tuner.run()
    for name in (
        "GGML_VK_ASYNC_USE_TRANSFER_QUEUE", "GGML_VK_MAX_NODES_PER_SUBMIT",
        "GGML_VK_DISABLE_ASYNC", "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM",
    ):
        assert name not in probed
    assert "GGML_VK_DISABLE_FUSION" in probed


def test_a_promising_switch_probe_checks_the_rest_of_the_list(tmp_path: Path) -> None:
    """Review 4: the quick probe cut the list to 2 before the checks, so widening could
    never reach the other switches. Here the first probed switch looks better, so the
    rest of the list is checked and fusion off (9th listed) is tried as well."""
    tuner = _switch_tuner(tmp_path, word_ms=600.0, faster_with="GGML_VK_FORCE_MMVQ")
    result = tuner.run()
    path: Any = next(item for item in result.rounds if item.get("path") == "engine_switches")
    assert path["depth"] == "probe"
    assert path.get("probe_became_full") is True
    tried = [entry["candidate"] for entry in path["screened"]]
    assert any("GGML_VK_DISABLE_FUSION" in label for label in tried)
    assert [change.setting for change in result.proposed_changes] == ["engine_switches"]


def test_a_quick_switch_probe_that_finds_nothing_names_what_it_did_not_try(tmp_path: Path) -> None:
    tuner = _switch_tuner(tmp_path, word_ms=600.0)
    result = tuner.run()
    path: Any = next(item for item in result.rounds if item.get("path") == "engine_switches")
    assert path["depth"] == "probe" and not path.get("probe_became_full")
    assert "quick probe only, not tried" in str(path["reason"])
    assert any("GGML_VK_DISABLE_FUSION" in label for label in path["untested"])


def test_device_lines_come_from_the_startup_log_when_the_timer_fills_the_recent_log() -> None:
    """Laptop run 079: the timer's output pushed the device lines out of the recent
    lines, so the engine went unrecognised and the switches were wrongly skipped."""

    timing_only = "\n".join(
        line for line in _LOG.splitlines() if not line.startswith("ggml_vulkan: 0 =")
    )

    class _Backend:
        def start(self, model: Any, params: Any) -> None:
            return None

        def generate(self, request: Any) -> GenerationResult:
            return GenerationResult(text="A.", decode_tokens_per_sec=50.0)

        def server_log(self) -> str:
            return timing_only

        def server_startup_log(self) -> str:
            return "ggml_vulkan: 0 = Radeon Pro 560X (AMD) | fp16: 1 | int dot: 1 | matrix cores: none"

        def stop(self) -> None:
            return None

    result = run_probe(
        settings=Settings(), model=sample_model(),
        config=RuntimeConfiguration(context_length=4096), repeats=1,
        backend_factory=lambda: _Backend(),  # type: ignore[arg-type, return-value]
    )
    assert any("Radeon" in line for line in result.device_lines)
    assert result.device_features()["int dot"] == "1"
    assert result.engine_seen == "vulkan"


def test_the_engine_is_recognised_from_its_timer_without_device_lines() -> None:
    """Laptop run 081: no device line was found, but the Vulkan timer's blocks were."""
    timing_only = "\n".join(line for line in _LOG.splitlines() if "vulkan" not in line.lower() or "Timings" in line)

    class _Backend:
        def start(self, model: Any, params: Any) -> None:
            return None

        def generate(self, request: Any) -> GenerationResult:
            return GenerationResult(text="A.", decode_tokens_per_sec=50.0)

        def server_log(self) -> str:
            return timing_only

        def stop(self) -> None:
            return None

    result = run_probe(
        settings=Settings(), model=sample_model(),
        config=RuntimeConfiguration(context_length=4096), repeats=1,
        backend_factory=lambda: _Backend(),  # type: ignore[arg-type, return-value]
    )
    assert result.device_lines == [] and result.graphs
    assert result.engine_seen == "vulkan"


def test_the_server_process_keeps_its_first_lines() -> None:
    import io

    from inferenceos.backends.llama_server_toolchain import LlamaServerProcess

    class _Client:
        def health(self) -> bool:
            return True

    process = LlamaServerProcess(["llama-server"], log_limit=5)
    stream = io.StringIO("".join(f"line {i}\n" for i in range(50)))
    process._spawn_reader(stream, "stderr").join(timeout=5)
    assert process.log_text().splitlines() == [f"line {i}" for i in range(45, 50)]
    assert process.startup_text().splitlines() == [f"line {i}" for i in range(50)]
    # Not cut at readiness: the reader threads can lag behind the health check.
    process.wait_ready(_Client(), timeout_s=1.0)  # type: ignore[arg-type]
    process._spawn_reader(io.StringIO("later\n"), "stdout").join(timeout=5)
    assert process.startup_text().splitlines()[-1] == "later"


def test_no_new_timed_runs_start_after_the_deadline(tmp_path: Path) -> None:
    """Review 4: the diagnosis and the switch checks had no deadline."""
    import time as clock

    tuner = _switch_tuner(tmp_path, word_ms=65.0)
    calls: list[Any] = []
    real = tuner.probe_runner

    def counting(config: Any, **options: Any) -> Any:
        calls.append(config)
        return real(config, **options)

    tuner.probe_runner = counting
    tuner._engine_name = "vulkan"
    past = clock.perf_counter() - 1.0
    record = tuner._check_switches(RuntimeConfiguration(context_length=4096, gpu_offload_layers=1), "high", past)
    assert calls == [] and record["reason"] == "no time left for the checks"
    assert record["not_checked"]
    diagnosis, diag_record = tuner._diagnose(
        RuntimeConfiguration(context_length=4096, gpu_offload_layers=1), None, "test", past
    )
    assert diagnosis is None and diag_record["skipped"] == "no time left" and calls == []


def test_a_timed_run_that_leaves_a_process_behind_stops_all_timed_runs(tmp_path: Path) -> None:
    """Review 4: a failed unload was only logged, so two copies could share the card."""
    from inferenceos.tune.engine_probe import ProbeNotSafe

    tuner = _switch_tuner(tmp_path, word_ms=65.0)
    real = tuner.probe_runner
    calls: list[Any] = []

    def unsafe_after_three(config: Any, **options: Any) -> Any:
        calls.append(config)
        if len(calls) == 3:
            raise ProbeNotSafe("the timed run's engine process did not exit")
        return real(config, **options)

    tuner.probe_runner = unsafe_after_three
    tuner._engine_name = "vulkan"
    record = tuner._check_switches(RuntimeConfiguration(context_length=4096, gpu_offload_layers=1), "high")
    assert len(calls) == 3 and tuner._probes_unsafe
    assert record["passed"] == [] and record["not_checked"]
    # No later timed run either (e.g. the diagnosis after a kept change).
    assert tuner._check_switches(RuntimeConfiguration(context_length=4096, gpu_offload_layers=1), "high")[
        "reason"
    ] == "no timed runs available"


def test_run_probe_reports_an_engine_process_that_did_not_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.contracts.lifecycle import ProcessIdentity
    from inferenceos.tune import engine_probe
    from inferenceos.tune.engine_probe import ProbeNotSafe

    class _Backend:
        def start(self, model: Any, params: Any) -> None:
            return None

        def generate(self, request: Any) -> GenerationResult:
            return GenerationResult(text="A.", decode_tokens_per_sec=50.0)

        def server_log(self) -> str:
            return ""

        def identity(self) -> ProcessIdentity:
            return ProcessIdentity(pid=4242, create_time=1.0)

        def stop(self) -> None:
            raise RuntimeError("unload failed")

    monkeypatch.setattr(engine_probe, "identity_matches", lambda identity: True)
    with pytest.raises(ProbeNotSafe):
        run_probe(
            settings=Settings(), model=sample_model(),
            config=RuntimeConfiguration(context_length=4096), repeats=1, exit_timeout_s=0.3,
            backend_factory=lambda: _Backend(),  # type: ignore[arg-type, return-value]
        )


def test_switch_checks_ask_the_questions_with_the_timer_off() -> None:
    """Laptop job 086: with the model split between processor and card, the Vulkan
    timer stops the engine on any question longer than the short one. The correctness
    questions therefore run in a second run without the timer."""
    from inferenceos.tune.engine_probe import CHECK_PROMPTS, run_switch_probe

    starts: list[str] = []
    prompts_seen: list[list[str]] = []

    class _Backend:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def start(self, model: Any, params: Any) -> None:
            starts.append(params.engine_switches or "")
            prompts_seen.append(self.prompts)

        def generate(self, request: Any) -> GenerationResult:
            self.prompts.append(request.prompt)
            return GenerationResult(text="A.", decode_tokens_per_sec=50.0)

        def server_log(self) -> str:
            if "GGML_VK_PERF_LOGGER" in starts[-1]:
                return _LOG
            return "ggml_vulkan: pipeline stats for mul_mat_vec_q4_k_f32_f32:\nok"

        def stop(self) -> None:
            return None

    result = run_switch_probe(
        settings=Settings(), model=sample_model(),
        config=RuntimeConfiguration(context_length=4096),
        backend_factory=lambda: _Backend(),  # type: ignore[arg-type, return-value]
    )
    assert "GGML_VK_PERF_LOGGER" in starts[0] and "GGML_VK_PERF_LOGGER" not in starts[1]
    assert "GGML_VK_PIPELINE_STATS" in starts[1]
    assert len(prompts_seen[0]) == 1 and prompts_seen[1] == list(CHECK_PROMPTS)
    assert result.complete and len(result.answers) == len(CHECK_PROMPTS)
    assert "steps" in result.evidence and result.evidence["kernels"] == frozenset({"mul_mat_vec_q4_k_f32_f32"})
    assert result.card_us_per_word() == 702.0


def test_the_deadline_is_checked_before_every_timed_run(tmp_path: Path) -> None:
    """Review 5: two unchanged timed runs started after a single deadline check."""
    import time as clock

    tuner = _switch_tuner(tmp_path, word_ms=65.0)
    real = tuner.probe_runner
    calls: list[Any] = []

    def slow(config: Any, **options: Any) -> Any:
        calls.append(config)
        clock.sleep(0.2)
        return real(config, **options)

    tuner.probe_runner = slow
    tuner._engine_name = "vulkan"
    record = tuner._check_switches(
        RuntimeConfiguration(context_length=4096, gpu_offload_layers=1), "high",
        clock.perf_counter() + 0.1,
    )
    assert len(calls) == 1 and record["reason"] == "no time left for the checks"
