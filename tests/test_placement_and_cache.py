"""Model-aware piece placement and conversation-memory formats (6 Oct 2026).

Fixture-level checks of the new pieces: the override text, the engine command, what the
engine log shows, what the engine accepts, piece-by-piece placement and memory, the
candidates each path generates, the "took effect" checks, the known-answer policy, the
strictest-requirement rule and merging changes that move several settings together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.backends.llama_cpp_toolchain import parse_effective_settings
from inferenceos.backends.llama_server_capabilities import parse_server_help
from inferenceos.backends.llama_server_toolchain import build_server_command
from inferenceos.backends.resident_server import _options_from
from inferenceos.backends.tensor_overrides import (
    OverrideError,
    check_overrides,
    engine_argument,
    format_overrides,
    is_device_buffer,
    parse_overrides,
)
from inferenceos.contracts.capability import Support
from inferenceos.contracts.experiment import ExperimentMeasurement, RuntimeConfiguration
from inferenceos.contracts.tune import ProposedChange
from inferenceos.experiment.restoration import runtime_params_from
from inferenceos.selftest import sample_model
from inferenceos.tune.kv_cache import cache_bytes, cache_choices, verify_cache
from inferenceos.tune.known_answers import (
    FAIL,
    INCOMPLETE,
    INSUFFICIENT,
    PASS,
    QUESTIONS,
    answer_right,
    run_known_answer_check,
)
from inferenceos.tune.model_facts import ModelFacts, processor_share
from inferenceos.tune.pipeline import _apply_changes, merge_change
from inferenceos.tune.quality_risk import strictest
from inferenceos.tune.tensor_placement import (
    EXPERT_PATTERN,
    NOT_SEEN,
    SHOWN,
    UNVERIFIED,
    piece_placement,
    placement_plans,
    verify_placement,
)

MIB = 1024 * 1024
FIXTURES = Path(__file__).parent / "fixtures"


def dense_facts(blocks: int = 4, block_mib: int = 100, output_mib: int = 200, tied: bool = False) -> ModelFacts:
    tensors: dict[str, int] = {"token_embd.weight": 50 * MIB}
    for index in range(blocks):
        tensors[f"blk.{index}.attn_q.weight"] = 40 * MIB * block_mib // 100
        tensors[f"blk.{index}.ffn_up.weight"] = 60 * MIB * block_mib // 100
    if not tied:
        tensors["output_norm.weight"] = 0
        tensors["output.weight"] = output_mib * MIB
    return ModelFacts(
        total_layers=blocks,
        from_file=True,
        attention_layers=blocks,
        layer_bytes=tuple(block_mib * MIB for _ in range(blocks)),
        active_layer_bytes=tuple(block_mib * MIB for _ in range(blocks)),
        output_bytes=(50 if tied else output_mib) * MIB,
        other_bytes=50 * MIB,
        tensor_bytes=tensors,
        layer_kinds=tuple("attention" for _ in range(blocks)),
        tied_output=tied,
        expert_bytes=tuple(0 for _ in range(blocks)),
        key_widths=tuple(1024 for _ in range(blocks)),
        value_widths=tuple(1024 for _ in range(blocks)),
    )


def moe_facts(blocks: int = 4) -> ModelFacts:
    tensors: dict[str, int] = {"token_embd.weight": 50 * MIB, "output.weight": 100 * MIB}
    for index in range(blocks):
        tensors[f"blk.{index}.attn_q.weight"] = 20 * MIB
        for part in ("up", "down", "gate"):
            tensors[f"blk.{index}.ffn_{part}_exps.weight"] = 100 * MIB
    return ModelFacts(
        total_layers=blocks,
        from_file=True,
        attention_layers=blocks,
        experts=32,
        experts_used=8,
        layer_bytes=tuple(320 * MIB for _ in range(blocks)),
        active_layer_bytes=tuple(95 * MIB for _ in range(blocks)),
        output_bytes=100 * MIB,
        other_bytes=50 * MIB,
        tensor_bytes=tensors,
        layer_kinds=tuple("attention" for _ in range(blocks)),
        tied_output=False,
        expert_bytes=tuple(300 * MIB for _ in range(blocks)),
        key_widths=tuple(512 for _ in range(blocks)),
        value_widths=tuple(512 for _ in range(blocks)),
    )


def hybrid_facts() -> ModelFacts:
    kinds = ("recurrent (state-space)", "recurrent (state-space)", "attention", "recurrent (state-space)")
    tensors: dict[str, int] = {"token_embd.weight": 50 * MIB, "output.weight": 100 * MIB}
    for index, kind in enumerate(kinds):
        tensors[f"blk.{index}.{'attn_q' if kind == 'attention' else 'ssm_in'}.weight"] = 100 * MIB
    return ModelFacts(
        total_layers=4,
        from_file=True,
        attention_layers=1,
        other_layers={"recurrent (state-space)": 3},
        layer_bytes=tuple(100 * MIB for _ in kinds),
        active_layer_bytes=tuple(100 * MIB for _ in kinds),
        output_bytes=100 * MIB,
        other_bytes=50 * MIB,
        tensor_bytes=tensors,
        layer_kinds=kinds,
        tied_output=False,
        expert_bytes=(0, 0, 0, 0),
        key_widths=(0, 0, 512, 0),
        value_widths=(0, 0, 512, 0),
    )


# --------------------------------------------------------------------------- #
# Override text and the engine command                                         #
# --------------------------------------------------------------------------- #
def test_override_text_round_trips_and_keeps_order() -> None:
    text = format_overrides([("^output", "CPU"), (EXPERT_PATTERN, "CPU")])
    assert text == f"^output=CPU;{EXPERT_PATTERN}=CPU"
    assert parse_overrides(text) == [("^output", "CPU"), (EXPERT_PATTERN, "CPU")]
    assert engine_argument(text) == f"^output=CPU,{EXPERT_PATTERN}=CPU"
    with pytest.raises(OverrideError):
        parse_overrides("blk\\.{1,2}=CPU")
    with pytest.raises(OverrideError):
        parse_overrides("no buffer here")


def test_override_check_refuses_unmatched_shadowed_clashing_and_protected() -> None:
    names = ["token_embd.weight", "blk.0.ffn_up_exps.weight", "output.weight"]
    assert check_overrides("^output=CPU", names).ok
    assert "matches no piece" in check_overrides("nothing_like_this=CPU", names).problems[0]
    shadowed = check_overrides("exps=CPU;ffn_up_exps=CPU", names)
    assert not shadowed.ok and "only matches pieces an earlier pattern decided" in shadowed.problems[0]
    clash = check_overrides("exps=CPU;ffn_up=Vulkan0", names, known_buffers=["Vulkan0"])
    assert not clash.ok and any("different buffer" in item for item in clash.problems)
    assert not check_overrides("token_embd=CPU", names).ok
    assert any("not one this engine reported" in item for item in check_overrides("^output=Metal9", names).problems)


def test_device_buffers_are_told_apart_from_host_memory() -> None:
    assert is_device_buffer("Vulkan0") and is_device_buffer("CUDA0")
    assert not is_device_buffer("CPU_Mapped") and not is_device_buffer("CPU_REPACK")
    assert not is_device_buffer("Vulkan_Host")


def test_the_engine_gets_overrides_and_both_cache_sides_with_fixed_precedence() -> None:
    config = RuntimeConfiguration(
        context_length=4096,
        gpu_offload_layers=29,
        kv_cache_type="f16",
        kv_cache_type_k="q8_0",
        placement_overrides="^output=CPU",
    )
    command = build_server_command(
        Path("llama-server.exe"),
        _options_from(Path("m.gguf"), runtime_params_from(config)),
        host="127.0.0.1",
        port=8080,
        api_key_file=Path("key"),
    )
    joined = " ".join(command)
    assert "-ctk q8_0" in joined and "-ctv f16" in joined
    assert "-ot ^output=CPU" in joined and "-c 4096" in joined
    legacy = RuntimeConfiguration(context_length=4096, kv_cache_type="q8_0")
    assert legacy.effective_kv_types() == ("q8_0", "q8_0")
    assert RuntimeConfiguration().effective_kv_types() == (None, None)


# --------------------------------------------------------------------------- #
# What the engine shows and accepts                                            #
# --------------------------------------------------------------------------- #
def test_engine_log_shows_formats_and_buffers() -> None:
    log = "\n".join(
        [
            "0.01 I load_tensors:      Vulkan0 model buffer size =   762.49 MiB",
            "0.01 I load_tensors:   CPU_Mapped model buffer size =   306.00 MiB",
            "0.02 I llama_kv_cache:    Vulkan0 KV buffer size =    16.00 MiB",
            "0.02 I llama_kv_cache: size =   16.00 MiB (  4096 cells,  26 layers,  1/1 seqs), "
            "K (q8_0):    8.50 MiB, V (q8_0):    8.50 MiB",
            "0.02 I llama_kv_cache: size =    4.00 MiB (  1024 cells,  4 layers,  1/1 seqs), "
            "K (f16):    2.00 MiB, V (q8_0):    2.00 MiB",
            "llama_context: Flash Attention was auto, set to enabled",
        ]
    )
    seen = parse_effective_settings(log)
    assert seen["cache_type_v"] == "q8_0"
    assert seen["cache_type_k"] == ["f16", "q8_0"]  # two caches, two formats: reported as both
    assert seen["model_buffers"] == {"Vulkan0": int(round(762.49 * MIB)), "CPU_Mapped": 306 * MIB}
    assert seen["flash_attention"] is True
    assert parse_effective_settings("nothing useful") == {}


def test_the_installed_build_help_text_lists_options_and_values() -> None:
    options = parse_server_help((FIXTURES / "llama_server_help_b11139.txt").read_text())
    assert options is not None
    assert options.accepts("placement_overrides")[0] is Support.YES
    assert options.accepts("kv_cache_type_v", "q8_0")[0] is Support.YES
    assert options.accepts("kv_cache_type_k", "q3_K")[0] is Support.NO
    assert options.accepts("threads_per_core_magic", 1)[0] is Support.UNKNOWN


# --------------------------------------------------------------------------- #
# Piece-by-piece placement                                                     #
# --------------------------------------------------------------------------- #
def test_overrides_are_the_source_of_truth_for_placement() -> None:
    facts = dense_facts()
    plain = piece_placement(facts, 3, None)
    assert plain is not None and plain.output_on_card
    assert plain.card_bytes == (200 + 2 * 100) * MIB
    moved = piece_placement(facts, 5, "^output=CPU")
    assert moved is not None and not any(name.startswith("output") for name in moved.card_pieces)
    assert moved.card_bytes == 4 * 100 * MIB
    assert set(moved.moved) == {"output.weight", "output_norm.weight"}
    layer_only = processor_share(facts, 5, card=True)
    with_override = processor_share(facts, 5, card=True, overrides="^output=CPU")
    assert layer_only == 0.0 and with_override is not None and with_override > 0.2
    unknown = ModelFacts(total_layers=4, layer_bytes=(1, 1, 1, 1), output_bytes=1)
    assert processor_share(unknown, 5, card=True, overrides="^output=CPU") is None


def test_expert_work_counts_only_the_experts_used() -> None:
    facts = moe_facts()
    where = piece_placement(facts, 5, f"{EXPERT_PATTERN}=CPU")
    assert where is not None
    assert where.card_bytes == (4 * 20 + 100) * MIB
    assert where.processor_active_bytes == int(4 * 3 * 100 * MIB * 8 / 32)


def test_placement_plans_cover_the_structural_alternatives() -> None:
    dense = placement_plans(dense_facts(), gpu_layers=2, current_overrides=None, card_budget_bytes=450 * MIB, card=True)
    assert [plan.kind for plan in dense.plans] == ["output", "output"]
    assert dense.plans[0].gpu_layers == 5 and dense.plans[0].card_bytes is not None
    assert dense.plans[0].card_bytes <= 450 * MIB
    tied = placement_plans(dense_facts(tied=True), gpu_layers=2, current_overrides=None, card_budget_bytes=None, card=True)
    assert not tied.plans and any("tied" in note for note in tied.notes)
    moe = placement_plans(moe_facts(), gpu_layers=0, current_overrides=None, card_budget_bytes=900 * MIB, card=True)
    kinds = [plan.kind for plan in moe.plans]
    assert "experts" in kinds
    assert all(plan.card_bytes is not None and plan.card_bytes <= 900 * MIB for plan in moe.plans)
    hybrid = placement_plans(hybrid_facts(), gpu_layers=5, current_overrides=None, card_budget_bytes=None, card=True)
    grouped = [plan for plan in hybrid.plans if plan.kind.startswith("grouped")]
    assert len(grouped) == 2
    reset = placement_plans(dense_facts(), gpu_layers=5, current_overrides="^output=CPU", card_budget_bytes=None, card=True)
    assert reset.plans[0].kind == "reset" and reset.plans[0].overrides is None
    no_card = placement_plans(dense_facts(), gpu_layers=2, current_overrides=None, card_budget_bytes=None, card=False)
    assert not no_card.plans


MOVED_OUTPUT = {"output_norm.weight": "CPU", "output.weight": "CPU"}


def test_a_placement_counts_only_when_the_engine_shows_it() -> None:
    facts = dense_facts()
    expected = piece_placement(facts, 5, "^output=CPU")
    assert expected is not None
    buffers = {"Vulkan0": expected.card_bytes + 10 * MIB, "CPU_Mapped": 300 * MIB}
    shown: dict[str, object] = {"model_buffers": buffers, "overridden_tensors": dict(MOVED_OUTPUT)}
    assert verify_placement(facts, 5, "^output=CPU", shown)[0] == SHOWN
    other: dict[str, object] = {
        "model_buffers": {"Vulkan0": expected.card_bytes + 200 * MIB}, "overridden_tensors": dict(MOVED_OUTPUT),
    }
    assert verify_placement(facts, 5, "^output=CPU", other)[0] == NOT_SEEN
    assert verify_placement(facts, 5, "^output=CPU", {})[0] == UNVERIFIED
    assert verify_placement(ModelFacts(total_layers=4), 5, "^output=CPU", shown)[0] == UNVERIFIED


def test_buffer_totals_alone_never_prove_a_placement() -> None:
    """Review 7 (finding 6): an ignored override can leave the card total within the
    tolerance, and equal-sized pieces placed wrongly give the same total. Identity comes
    from the engine's per-piece lines; totals are only a consistency check."""
    facts = dense_facts(output_mib=16)
    requested = piece_placement(facts, 5, "^output=CPU")
    ignored = piece_placement(facts, 5, None)
    assert requested is not None and ignored is not None
    assert abs(ignored.card_bytes - requested.card_bytes) <= 48 * MIB  # totals cannot tell
    ignored_log: dict[str, object] = {"model_buffers": {"Vulkan0": ignored.card_bytes}}
    status, words = verify_placement(facts, 5, "^output=CPU", ignored_log)
    assert status == UNVERIFIED and "cannot be told apart" in words
    # Lines that miss a requested piece, put it on the card, or move another piece.
    partial = {**ignored_log, "overridden_tensors": {"output.weight": "CPU"}}
    assert verify_placement(facts, 5, "^output=CPU", partial)[0] == NOT_SEEN
    wrong_side = {**ignored_log, "overridden_tensors": {"output_norm.weight": "CPU", "output.weight": "Vulkan0"}}
    assert verify_placement(facts, 5, "^output=CPU", wrong_side)[0] == NOT_SEEN
    swapped = {
        "model_buffers": {"Vulkan0": requested.card_bytes},
        "overridden_tensors": {**MOVED_OUTPUT, "blk.3.ffn_up.weight": "CPU"},
    }
    assert verify_placement(facts, 5, "^output=CPU", swapped)[0] == NOT_SEEN
    # No overrides asked for, but the engine moved pieces: not the requested layout.
    assert verify_placement(facts, 5, None, {**ignored_log, "overridden_tensors": MOVED_OUTPUT})[0] == NOT_SEEN
    assert verify_placement(facts, 5, None, ignored_log)[0] == SHOWN


# --------------------------------------------------------------------------- #
# Conversation memory                                                          #
# --------------------------------------------------------------------------- #
def test_cache_memory_counts_block_scales_and_only_attention_blocks() -> None:
    model = sample_model()
    full = cache_bytes(dense_facts(), model, 2048, None, None)
    half = cache_bytes(dense_facts(), model, 2048, "q8_0", "q8_0")
    assert full.bytes is not None and half.bytes is not None and full.exact_layout
    assert half.bytes * 64 == full.bytes * 34
    hybrid = cache_bytes(hybrid_facts(), model, 2048, None, None)
    assert hybrid.bytes == 2048 * 512 * 2 * 2
    windowed = ModelFacts(**{**dense_facts().__dict__, "sliding_window": 512})
    assert not cache_bytes(windowed, model, 2048, None, None).exact_layout
    assert cache_bytes(dense_facts(), model, 2048, "weird", None).bytes is None


def test_cache_choices_respect_flash_attention_rows_and_the_engine() -> None:
    facts = dense_facts()
    auto = RuntimeConfiguration(context_length=2048)
    choices, notes = cache_choices(facts, auto)
    pairs = {(item.keys, item.values) for item in choices}
    assert pairs == {("q8_0", "q8_0"), ("q8_0", "f16"), ("f16", "q8_0")}
    both = next(item for item in choices if (item.keys, item.values) == ("q8_0", "q8_0"))
    assert ("flash_attention", True) in both.also
    assert any("lower precisions" in note for note in notes)
    off = RuntimeConfiguration(context_length=2048, flash_attention=False)
    assert {(item.keys, item.values) for item in cache_choices(facts, off)[0]} == {("q8_0", "f16")}
    odd = ModelFacts(**{**facts.__dict__, "key_widths": (1000,) * 4, "value_widths": (1000,) * 4})
    assert cache_choices(odd, auto)[0] == []
    recurrent = ModelFacts(**{**facts.__dict__, "attention_layers": 0})
    assert cache_choices(recurrent, auto)[0] == []
    refuse = cache_choices(facts, auto, accepts=lambda setting, value: (Support.NO, "not listed"))
    assert refuse[0] == [] and any("not listed" in note for note in refuse[1])
    back = cache_choices(facts, RuntimeConfiguration(kv_cache_type_k="q8_0", kv_cache_type_v="q8_0", flash_attention=True))
    assert ("f16", "f16") in {(item.keys, item.values) for item in back[0]}


def test_a_cache_format_counts_only_when_the_engine_reports_it() -> None:
    config = RuntimeConfiguration(kv_cache_type_k="q8_0", kv_cache_type_v="f16")
    assert verify_cache(config, {"cache_type_k": "q8_0", "cache_type_v": "f16"})[0] == SHOWN
    assert verify_cache(config, {"cache_type_k": "f16", "cache_type_v": "f16"})[0] == NOT_SEEN
    assert verify_cache(config, {})[0] == UNVERIFIED


# --------------------------------------------------------------------------- #
# Known answers and the strictest requirement                                  #
# --------------------------------------------------------------------------- #
GOOD_ANSWERS = {
    "arithmetic": "17 x 23 = 391 and 46 x 9 = 414, so 391 + 414 = 805.\nANSWER: 805",
    "fact far back": "Note 17 says 209 crates came in and crates went south.\nANSWER: 209, south",
    "code early in a list": "ANSWER: F9595",
}


def _answers(
    right: dict[str, bool],
    *,
    texts: dict[str, str] | None = None,
    valid: dict[str, int] | None = None,
) -> object:
    """Scripted known-answer measurements. ``right``: "<side>:<question>" -> right or
    not; ``texts``: replace an answer; ``valid``: good runs of a measurement."""

    def measure(config: RuntimeConfiguration, workload: object, label: str) -> ExperimentMeasurement:
        question = next(q for q in QUESTIONS if f"known-{q.question_id.replace(' ', '-')}" == getattr(workload, "workload_id"))
        side = "new" if label.endswith("-new") else "current"
        key = f"{side}:{question.question_id}"
        good = GOOD_ANSWERS[question.question_id]
        text = (texts or {}).get(key) or (good if right.get(key, True) else "ANSWER: I am not sure.")
        return ExperimentMeasurement(
            measurement_id="m", phase=label, configuration=config,
            valid_runs=(valid or {}).get(key, 2), generated_texts=[text, text],
        )

    return measure


def test_known_answers_are_known() -> None:
    assert answer_right(QUESTIONS[0], "So the final number is 805.\nANSWER: 805")
    assert not answer_right(QUESTIONS[0], "So the final number is 805.")  # no answer line
    assert not answer_right(QUESTIONS[0], "ANSWER: 850")
    assert answer_right(QUESTIONS[1], "ANSWER: 209 crates, sent to the SOUTH depot")
    assert not answer_right(QUESTIONS[1], "ANSWER: 209 crates, sent to the north depot")


def test_mentioning_the_right_value_is_not_a_right_answer() -> None:
    """Review 7 (finding 3): substrings used to pass: 1805 for 805, "1209 crates to the
    southwest depot" for 209 and south, "XF9595Y" for F9595."""
    arithmetic, fact, code = QUESTIONS
    assert not answer_right(arithmetic, "ANSWER: 1805")
    assert not answer_right(arithmetic, "ANSWER: 1,805 rupees")
    assert not answer_right(arithmetic, "ANSWER: 805 or 806")
    assert not answer_right(arithmetic, "ANSWER: 805, no wait, 806")  # contradicts itself
    # Plain-text completion: lines after the first answer are the model's own continuation
    # (laptop job 110), so only the first answer line counts.
    assert answer_right(code, " F9595\n\nQuestion: what is the code for item 39?\nANSWER: L7109")
    assert not answer_right(code, " L7109\n\nANSWER: F9595")
    assert answer_right(arithmetic, "**ANSWER:** 805.00 rupees")
    assert not answer_right(fact, "ANSWER: 1209 crates to the southwest depot")
    assert not answer_right(fact, "ANSWER: 209 crates to the southwest depot")
    assert not answer_right(fact, "ANSWER: 209, south or east")
    assert not answer_right(fact, "ANSWER: 209 or 215, south")
    assert answer_right(fact, "ANSWER: On day 17, 209 crates; south depot")
    assert not answer_right(code, "ANSWER: XF9595Y")
    assert not answer_right(code, "ANSWER: F9595 or F9596")
    assert answer_right(code, "ANSWER: `F9595`")


def test_wrong_known_answers_containing_the_right_value_fail_the_check() -> None:
    start = RuntimeConfiguration(context_length=4096)
    new = start.model_copy(update={"kv_cache_type_k": "q8_0"})
    for key, text in (
        ("new:arithmetic", "ANSWER: 1805"),
        ("new:fact far back", "ANSWER: 1209 crates to the southwest depot"),
        ("new:code early in a list", "ANSWER: XF9595Y"),
    ):
        check = run_known_answer_check(
            start=start, candidate=new, context_length=4096, measure=_answers({}, texts={key: text}),  # type: ignore[arg-type]
        )
        assert check.status == FAIL, (key, check.reason)


def test_unusable_or_unfinished_known_answer_checks_never_pass() -> None:
    """Review 7 (finding 4): measurements with no good runs but kept answer text, and a
    check cut short by the deadline after two questions, used to pass."""
    start = RuntimeConfiguration(context_length=4096)
    new = start.model_copy(update={"kv_cache_type_k": "q8_0"})
    for key in ("new:arithmetic", "current:fact far back", "new:code early in a list"):
        check = run_known_answer_check(
            start=start, candidate=new, context_length=4096, measure=_answers({}, valid={key: 0}),  # type: ignore[arg-type]
        )
        assert check.status == INCOMPLETE, (key, check.status, check.reason)
        one_run = run_known_answer_check(
            start=start, candidate=new, context_length=4096, measure=_answers({}, valid={key: 1}),  # type: ignore[arg-type]
        )
        assert one_run.status == INCOMPLETE
    calls = {"n": 0}

    def two_questions_then_no_time() -> bool:
        calls["n"] += 1
        return calls["n"] <= 2

    cut = run_known_answer_check(
        start=start, candidate=new, context_length=4096, measure=_answers({}),  # type: ignore[arg-type]
        time_left=two_questions_then_no_time,
    )
    assert cut.status == INCOMPLETE and "no time left" in cut.reason
    unshown = run_known_answer_check(
        start=start, candidate=new, context_length=4096, measure=_answers({}),  # type: ignore[arg-type]
        settings_check=lambda config, m: (config != new, "cache format f16, requested q8_0"),
    )
    assert unshown.status == INCOMPLETE and "not shown" in unshown.reason


def test_known_answer_policy_passes_fails_and_says_when_it_cannot_tell() -> None:
    start = RuntimeConfiguration(context_length=4096)
    new = start.model_copy(update={"kv_cache_type_k": "q8_0"})
    ok = run_known_answer_check(start=start, candidate=new, context_length=4096, measure=_answers({}))  # type: ignore[arg-type]
    assert ok.status == PASS
    wrong = run_known_answer_check(
        start=start, candidate=new, context_length=4096,
        measure=_answers({"new:fact far back": False}),  # type: ignore[arg-type]
    )
    assert wrong.status == FAIL
    unsure = run_known_answer_check(
        start=start, candidate=new, context_length=4096,
        measure=_answers({"current:arithmetic": False, "current:fact far back": False}),  # type: ignore[arg-type]
    )
    assert unsure.status == INSUFFICIENT
    short = run_known_answer_check(start=start, candidate=new, context_length=600, measure=_answers({}))  # type: ignore[arg-type]
    assert short.status == INSUFFICIENT  # only one question fits a 600-piece context


def _change(card: str, setting: str, value: object, **related: object) -> ProposedChange:
    return ProposedChange(
        change_id=f"{card}:{value}", card_id=card, setting=setting, description=setting,
        old_value=None, new_value=value, related=related,  # type: ignore[arg-type]
    )


def test_a_combined_proposal_uses_the_strictest_requirement() -> None:
    layers = _change("runtime.increase_gpu_offload_layers", "gpu_offload_layers", 20)
    cache = _change("runtime.kv_cache_format", "kv_cache_type_k", "q8_0", kv_cache_type_v="q8_0")
    card, requirement = strictest([layers, cache])
    assert card == "runtime.kv_cache_format" and requirement.known_answers
    assert strictest([layers])[1].name == "runtime"
    assert strictest([])[0] is None


def test_changes_that_move_several_settings_merge_and_apply_together() -> None:
    start = RuntimeConfiguration(context_length=2048, gpu_offload_layers=8)
    placement = _change("runtime.tensor_placement", "placement_overrides", "^output=CPU", gpu_offload_layers=14)
    applied = _apply_changes(start, [placement])
    assert applied.placement_overrides == "^output=CPU" and applied.gpu_offload_layers == 14
    layers_again = _change("runtime.increase_gpu_offload_layers", "gpu_offload_layers", 16)
    workloads: dict[str, list[object]] = {}
    merged = merge_change([placement], layers_again, workloads, start)  # type: ignore[arg-type]
    assert len(merged) == 1
    final = _apply_changes(start, merged)
    assert final.gpu_offload_layers == 16 and final.placement_overrides == "^output=CPU"
    safety = _change("runtime.increase_gpu_offload_layers", "gpu_offload_layers", 6)
    kept_apart = merge_change([safety], placement, {}, start, keep_apart={safety.change_id})
    assert len(kept_apart) == 2
    back = _change("runtime.tensor_placement", "placement_overrides", "", gpu_offload_layers=8)
    assert merge_change([placement], back, {}, start) == []
