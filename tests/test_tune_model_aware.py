"""The optimization paths look at the model and the machine (Oct 2026).

Facts from the model file (attention layers, experts, sizes, tied output layer), where
the model runs (processor share for a layer count), which paths apply, what each path
tries, the long test question sized to the model, the "must not lose on the short
question" check, and the overall heat line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.models.gguf import read_gguf
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.selftest import sample_hardware, sample_model
from inferenceos.tune.model_facts import (
    MachineFacts,
    ModelFacts,
    facts_from_gguf,
    processor_share,
)
from inferenceos.tune.paths import (
    FLASH_ATTENTION,
    PROMPT_BATCH,
    READING_THREADS,
    THREADS,
    PathContext,
    batch_working_memory_bytes,
    prompt_batch_candidates,
    prompt_batch_notes,
    prompt_batch_refine,
    thread_refine,
)
from tests.experiment_helpers import SampleResult
from tests.test_gguf import _kv_string, _kv_u32, _tensor, _write_gguf
from tests.test_tune import CONTEXT, _settings, _tuner

GGML_F32 = 0


def _write_model(path: Path, *, kinds: list[str], tied: bool, experts: int | None = None) -> Path:
    """A tiny model file: each layer is 'attention', 'ssm' or 'conv' (100 bytes each),
    the token table is 400 bytes, the output layer 300 bytes (absent when tied)."""
    kv = [_kv_string("general.architecture", "toy"), _kv_u32("toy.block_count", len(kinds))]
    if experts is not None:
        kv.append(_kv_u32("toy.expert_count", experts))
        kv.append(_kv_u32("toy.expert_used_count", 2))
    tensors: list[bytes] = []
    offset = 0

    def add(name: str, size: int) -> None:
        nonlocal offset
        tensors.append(_tensor(name, (size // 4,), GGML_F32, offset))
        offset += size

    add("token_embd.weight", 400)
    for index, kind in enumerate(kinds):
        if kind == "attention":
            add(f"blk.{index}.attn_q.weight", 25)
            add(f"blk.{index}.attn_k.weight", 25)
            add(f"blk.{index}.ffn_up.weight", 50)
        elif kind == "ssm":
            add(f"blk.{index}.attn_qkv.weight", 25)  # hybrid linear attention input
            add(f"blk.{index}.ssm_conv1d.weight", 25)
            add(f"blk.{index}.ffn_up.weight", 50)
        else:
            add(f"blk.{index}.shortconv.conv.weight", 50)
            add(f"blk.{index}.ffn_up.weight", 50)
    add("output_norm.weight", 4)
    if not tied:
        add("output.weight", 300)
    add("last.pad", 4)
    _write_gguf(path, 3, kv, tensors)
    return path


def _facts(tmp_path: Path, **kwargs: Any) -> ModelFacts:
    path = _write_model(tmp_path / "toy.gguf", **kwargs)
    return facts_from_gguf(read_gguf(path), sample_model())


# --------------------------------------------------------------------------- #
# Facts from the model file                                                    #
# --------------------------------------------------------------------------- #
def test_attention_layers_are_counted_and_hybrid_layers_are_not(tmp_path: Path) -> None:
    facts = _facts(tmp_path, kinds=["ssm", "ssm", "ssm", "attention"] * 2, tied=False)
    assert facts.total_layers == 8
    assert facts.attention_layers == 2
    assert facts.other_layers == {"recurrent (state-space)": 6}
    assert "2 use attention (6 recurrent (state-space))" in facts.design_words()
    conv = _facts(tmp_path, kinds=["conv", "attention", "conv"], tied=False)
    assert conv.attention_layers == 1 and conv.other_layers == {"short convolution": 2}


def test_experts_are_read_from_the_model_file(tmp_path: Path) -> None:
    facts = _facts(tmp_path, kinds=["attention"] * 2, tied=False, experts=40)
    assert facts.uses_experts and facts.experts == 40 and facts.experts_used == 2
    assert "40 experts, 2 used per word" in facts.design_words()


def test_layer_and_output_sizes_come_from_the_file(tmp_path: Path) -> None:
    facts = _facts(tmp_path, kinds=["attention"] * 4, tied=False)
    assert facts.layer_bytes == (100, 100, 100, 100)
    assert facts.output_bytes == 304  # output layer and its norm


def test_a_tied_output_layer_counts_the_token_table(tmp_path: Path) -> None:
    """With no output layer of its own, llama.cpp uses a copy of the token table as the
    output layer, placed like one."""
    facts = _facts(tmp_path, kinds=["attention"] * 4, tied=True)
    assert facts.output_bytes == 404  # token table + output norm
    # Count 1 puts only the output step on the card (llama.cpp's rule).
    share = processor_share(facts, 1, card=True)
    assert share is not None and abs(share - 400 / (400 + 404)) < 1e-9
    # Count 4 = output step + the last 3 blocks: block 0 still on the processor.
    share = processor_share(facts, 4, card=True)
    assert share is not None and abs(share - 100 / (400 + 404)) < 1e-9
    assert processor_share(facts, 5, card=True) == 0.0


def test_processor_share_follows_llama_cpp_layer_placement(tmp_path: Path) -> None:
    """llama.cpp b11139 (commit 42916d83f): count N puts the output layer and the last
    N-1 blocks on the card; N = blocks + 1 puts everything there."""
    facts = _facts(tmp_path, kinds=["attention"] * 4, tied=False)
    whole = 400 + 304
    assert processor_share(facts, 0, card=True) == 1.0
    share = processor_share(facts, 2, card=True)  # output + block 3
    assert share is not None and abs(share - 300 / whole) < 1e-9
    assert processor_share(facts, 5, card=True) == 0.0
    assert processor_share(facts, None, card=True) is None
    assert processor_share(facts, 5, card=False) == 1.0


def test_only_the_experts_used_per_word_count_as_work(tmp_path: Path) -> None:
    from inferenceos.tune.model_facts import facts_from_gguf

    path = tmp_path / "moe.gguf"
    kv = [
        _kv_string("general.architecture", "toy"),
        _kv_u32("toy.block_count", 2),
        _kv_u32("toy.expert_count", 4),
        _kv_u32("toy.expert_used_count", 1),
    ]
    tensors = []
    offset = 0
    for name, size in (
        ("token_embd.weight", 400),
        ("blk.0.attn_q.weight", 100),
        ("blk.0.ffn_up_exps.weight", 400),
        ("blk.1.attn_q.weight", 100),
        ("blk.1.ffn_up_exps.weight", 400),
        ("output.weight", 200),
        ("pad", 4),
    ):
        tensors.append(_tensor(name, (size // 4,), GGML_F32, offset))
        offset += size
    _write_gguf(path, 3, kv, tensors)
    facts = facts_from_gguf(read_gguf(path), sample_model())
    assert facts.layer_bytes == (500, 500)
    assert facts.active_layer_bytes == (200, 200)  # 100 + 400 x 1/4


def test_the_placement_rule_matches_llama_cpp() -> None:
    from inferenceos.tune.placement import BLOCKS_FIRST, max_count, placement

    assert placement(0, 4).nothing_on_card
    one = placement(1, 4)
    assert one.output_on_card and one.blocks_on_card == ()
    assert placement(4, 4).blocks_on_card == (1, 2, 3)
    assert placement(5, 4).everything_on_card and max_count(4) == 5
    assert placement(99, 4).everything_on_card
    old = placement(4, 4, BLOCKS_FIRST)
    assert old.blocks_on_card == (0, 1, 2, 3) and not old.output_on_card
    assert "the output step" in one.words()


# --------------------------------------------------------------------------- #
# Which paths apply                                                            #
# --------------------------------------------------------------------------- #
def _context(
    *,
    current: RuntimeConfiguration | None = None,
    facts: ModelFacts | None = None,
    share: float | None = 0.5,
    question: int | None = None,
    free: int | None = 3 * 1024**3,
) -> PathContext:
    machine = MachineFacts(
        physical_cores=6, logical_cores=12, memory_mhz=2400, memory_modules=2, card=True,
        free_card_bytes=free,
    )
    return PathContext(
        hardware=sample_hardware(),
        model=sample_model(),
        total_layers=24,
        current=current or RuntimeConfiguration(context_length=4096, gpu_offload_layers=8),
        facts=facts,
        machine=machine,
        processor_share=share,
        question_tokens=question,
        context_length=4096,
    )


def test_thread_paths_are_skipped_when_the_whole_model_is_on_the_card() -> None:
    for path in (THREADS, READING_THREADS):
        ok, why = path.check(_context(share=0.0))
        assert not ok and "everything" in why and "graphics card" in why
        assert path.check(_context(share=0.2))[0]
        assert path.check(_context(share=None))[0]


def test_flash_attention_is_skipped_for_a_model_without_attention_layers() -> None:
    none = ModelFacts(total_layers=8, from_file=True, attention_layers=0, other_layers={"recurrent (state-space)": 8})
    ok, why = FLASH_ATTENTION.check(_context(facts=none))
    assert not ok and "no attention layers" in why
    few = ModelFacts(total_layers=8, from_file=True, attention_layers=2, other_layers={"recurrent (state-space)": 6})
    ok, why = FLASH_ATTENTION.check(_context(facts=few))
    assert ok and "2 of 8 layers" in why


def test_paths_that_read_the_question_use_the_long_question_and_guard_the_short_one() -> None:
    assert READING_THREADS.workload_id == "long-question"
    assert READING_THREADS.guard_workload_id == "short-short"
    assert FLASH_ATTENTION.workload_id == "long-question"
    assert FLASH_ATTENTION.guard_workload_id == "short-short"
    assert PROMPT_BATCH.workload_id == "long-question"
    assert PROMPT_BATCH.guard_workload_id == "short-short"


# --------------------------------------------------------------------------- #
# What the paths try                                                           #
# --------------------------------------------------------------------------- #
def test_thread_refine_tries_the_neighbours_of_the_best() -> None:
    assert [c.value for c in thread_refine(_context(), 8, {4, 6, 8, 2, 10, 12})] == [7, 9]
    assert [c.value for c in thread_refine(_context(), 12, {12, 10})] == [11]


def test_prompt_batch_sizes_stop_at_the_question_length() -> None:
    context = _context(question=600)
    assert [c.value for c in prompt_batch_candidates(context)] == [64, 128, 256, 1024]
    assert any("about 600 pieces" in note for note in prompt_batch_notes(context))


def test_prompt_batch_sizes_that_cannot_fit_on_the_card_are_dropped() -> None:
    """Only with flash attention turned off is the attention table counted:
    1024 x 4096 x 32 heads x 4 bytes = 0.5 GB, more than 0.4 GB free."""
    facts = ModelFacts(total_layers=24, attention_layers=24, attention_heads=32, vocabulary_size=150_000)
    off = RuntimeConfiguration(context_length=4096, gpu_offload_layers=8, flash_attention=False)
    context = _context(facts=facts, free=400 * 1024**2, current=off)
    need = batch_working_memory_bytes(context, 1024)
    assert need is not None and need > 400 * 1024**2
    values = [c.value for c in prompt_batch_candidates(context)]
    assert 2048 not in values and 1024 not in values and 256 in values
    assert any("would not fit" in note for note in prompt_batch_notes(context))
    # llama.cpp's "auto" usually turns flash attention on: nothing is counted or dropped.
    auto = _context(facts=facts, free=400 * 1024**2)
    assert batch_working_memory_bytes(auto, 2048) is None
    assert 2048 in [c.value for c in prompt_batch_candidates(auto)]
    # Nothing on the card: no attention table on the card either.
    none_on_card = _context(
        facts=facts, free=400 * 1024**2,
        current=RuntimeConfiguration(context_length=4096, gpu_offload_layers=0, flash_attention=False),
    )
    assert batch_working_memory_bytes(none_on_card, 2048) is None


def test_prompt_batch_widens_when_the_best_is_at_the_edge() -> None:
    context = _context(question=3000)
    assert [c.value for c in prompt_batch_refine(context, 64, {64, 128, 256, 512})] == [32]
    assert [c.value for c in prompt_batch_refine(context, 512, {64, 128, 256, 512})] == [1024]
    assert prompt_batch_refine(context, 256, {64, 128, 256, 512}) == []
    # Not past the question length, and not past llama.cpp's limit.
    short = _context(question=700)
    assert prompt_batch_refine(short, 1024, {256, 512, 1024}) == []
    assert prompt_batch_refine(context, 2048, {1024, 2048}) == []


# --------------------------------------------------------------------------- #
# Whole tune runs                                                              #
# --------------------------------------------------------------------------- #
def test_the_long_question_is_sized_to_the_time_limit(tmp_path: Path) -> None:
    """Reading at 40 pieces/s with a 10 s limit gives a question of about 400 minus
    the writing time, never more than half the context."""

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(decode_tps=20.0, prompt_tps=40.0, total_latency_s=2.0)

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="prompt_batch",
        tune_long_question_seconds=10.0, tune_layer_path=False,
    )
    lines: list[str] = []
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[], settings=settings, lines=lines)
    result = tuner.run()
    sized = next(item for item in result.rounds if item.get("round") == "long test question")
    pieces = sized["pieces"]
    assert isinstance(pieces, int) and 300 <= pieces <= 420
    assert sized["expected_seconds_per_answer"] is not None
    assert any("long test question: about" in line for line in lines)
    batch = next(item for item in result.rounds if item.get("path") == "prompt_batch")
    assert batch["workload"] == sized["workload"]


def test_a_fixed_long_question_length_is_capped_at_half_the_context(tmp_path: Path) -> None:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(decode_tps=20.0, prompt_tps=40.0, total_latency_s=2.0)

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="prompt_batch",
        tune_long_question_tokens=100_000, tune_layer_path=False,
    )
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[], settings=settings)
    result = tuner.run()
    sized = next(item for item in result.rounds if item.get("round") == "long test question")
    pieces = sized["pieces"]
    assert isinstance(pieces, int) and pieces <= CONTEXT // 2
    assert pieces > CONTEXT // 2 - 150


def test_a_change_that_loses_on_the_short_question_is_not_kept(tmp_path: Path) -> None:
    """3 reading threads make the long question's first word sooner but the short
    question's first word 40% later (seen on the 9B, Oct 2026): not kept."""

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        latency = 2.0
        if options.threads_batch == 3:
            latency = 1.7 if task.workload.workload_id.startswith("long-") else 2.8
        return SampleResult(decode_tps=20.0, total_latency_s=latency)

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="reading_threads",
        tune_layer_path=False,
    )
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[True], settings=settings)
    result = tuner.run()
    assert result.proposed_changes == []
    reading = next(item for item in result.rounds if item.get("path") == "reading_threads")
    screened: Any = reading["screened"]
    three = next(entry for entry in screened if entry.get("value") == 3)
    assert three["guard_gain"] < -0.1 and "guard_note" in three


def test_a_change_that_holds_on_both_questions_is_kept_with_both_checks(tmp_path: Path) -> None:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        latency = 2.0
        if options.threads_batch == 3 and task.workload.workload_id.startswith("long-"):
            latency = 1.7
        return SampleResult(decode_tps=20.0, total_latency_s=latency)

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="reading_threads",
        tune_layer_path=False,
    )
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[True], settings=settings)
    result = tuner.run()
    assert [change.setting for change in result.proposed_changes] == ["threads_batch"]
    change = result.proposed_changes[0]
    assert any("shown not worse on the short-short question" in item for item in change.evidence)
    reading = next(item for item in result.rounds if item.get("path") == "reading_threads")
    assert "guard_fair" in reading


def test_threads_are_skipped_with_the_reason_when_everything_is_on_the_card(tmp_path: Path) -> None:
    """Layers above the layer count put the output layer on the card too."""

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="threads,reading_threads",
        tune_layer_path=False,
    )
    lines: list[str] = []
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[], settings=settings, lines=lines)
    total = int(tuner.model.n_layers)
    tuner._starting_point = lambda power: (  # type: ignore[method-assign]
        RuntimeConfiguration(context_length=CONTEXT, gpu_offload_layers=total + 1),
        "test",
    )
    result = tuner.run()
    facts = next(item for item in result.rounds if item.get("round") == "facts")
    assert facts["processor_share"] == 0.0
    for name in ("threads", "reading_threads"):
        item = next(entry for entry in result.rounds if entry.get("path") == name)
        assert item["applies"] is False and "everything" in str(item["reason"])
    assert any("Model and machine facts:" in line for line in lines)


def test_the_heat_report_ends_with_one_overall_line(tmp_path: Path) -> None:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="threads", tune_layer_path=False,
    )
    lines: list[str] = []
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[], settings=settings, lines=lines)
    result = tuner.run()
    report: Any = result.heat_report
    assert report["result_may_not_be_best"] is False
    assert report["overall"] == "Overall: no sign of heat affecting this result."
    assert lines[-1] == "Overall: no sign of heat affecting this result."


def test_a_long_question_can_be_rebuilt_from_its_name() -> None:
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.tune.paths import steady_workload

    built = steady_workload("long-question-20x-steady", default=get_workload("short-short"))
    assert built.workload_id == "long-question-20x-steady" and built.ignore_eos is True
    assert built.prompt.count("A key value cache stores attention state") == 20


def test_the_question_is_sized_from_the_first_word_time_when_no_reading_speed_is_given(
    tmp_path: Path,
) -> None:
    """llama-server's streamed answers carry no reading speed (found on Gemma, Oct 2026):
    the time to the first word is used instead."""

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        # first word = 0.1 x total latency; 414 pieces read in 2 s -> about 200 pieces/s
        return SampleResult(decode_tps=20.0, prompt_tps=None, total_latency_s=20.0)  # type: ignore[arg-type]

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="prompt_batch",
        tune_long_question_seconds=6.0, tune_layer_path=False,
    )
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[], settings=settings)
    result = tuner.run()
    sized = next(item for item in result.rounds if item.get("round") == "long test question")
    pieces = sized["pieces"]
    # (6 s - 0.8 s writing) x ~207 pieces/s = ~1076, capped at half the context (1000)
    assert isinstance(pieces, int) and 800 <= pieces <= 1000


# --------------------------------------------------------------------------- #
# The layer path covers every placement                                        #
# --------------------------------------------------------------------------- #
def test_the_layer_path_can_put_everything_on_the_card(tmp_path: Path) -> None:
    """With everything fitting, the search reaches blocks + 1 (output step and all
    blocks) and keeps it when it is fastest."""
    from inferenceos.tune.layer_top import LoadAttempt

    total = int(sample_model().n_layers)

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        layers = options.gpu_offload_layers or 0
        return SampleResult(decode_tps=30.0 if layers > total else 20.0, total_latency_s=2.0)

    settings = _settings(tmp_path, optimization_mode="performance_first", tune_paths="")
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[True], settings=settings)
    tuner.try_load = lambda layers: LoadAttempt(loaded=True, fits=True)
    result = tuner.run()
    layers = next(item for item in result.rounds if item.get("round") == "graphics card layers")
    assert layers["largest_count"] == total + 1 and layers["top"] == total + 1
    assert result.final_configuration.gpu_offload_layers == total + 1


def test_the_layer_path_tests_the_output_step_alone(tmp_path: Path) -> None:
    """Count 1 (only the output step on the card) is its own group, so it is always
    tried (Gemma 3 1B was fastest there, Oct 2026)."""
    from inferenceos.tune.layer_top import LoadAttempt

    tried: list[int] = []

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        layers = options.gpu_offload_layers or 0
        tried.append(layers)
        return SampleResult(decode_tps=30.0 if layers == 1 else 20.0, total_latency_s=2.0)

    settings = _settings(tmp_path, optimization_mode="performance_first", tune_paths="")
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[True], settings=settings)
    tuner.try_load = lambda layers: LoadAttempt(loaded=True, fits=True)
    result = tuner.run()
    assert 1 in tried
    assert result.final_configuration.gpu_offload_layers == 1


def test_the_placement_rule_is_marked_checked_only_for_checked_builds() -> None:
    from inferenceos.tune.placement import rule_verified

    assert rule_verified("version: 0.4.1-dev (build 11139, commit 42916d83f)")
    assert not rule_verified("version: 0.5.0 (build 12000, commit abcdef123)")
    assert not rule_verified(None)
