"""Reuse correctness (job 128, prepared 17 Oct 2026): known-answer pairs, the strict
judge, the effective sliding-window evidence, the pre-registered classifications, the chat
exposure check and its stops, and the runner. Simulated engines establish correctness of
the code only; nothing here measures hardware."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from inferenceos.tune.prompt_reuse import (
    DONE,
    FRESH,
    REUSE,
    STOPPED_CLEANUP,
    STOPPED_POWER,
    STOPPED_VALIDITY,
    SWA_FULL_WARNING,
    EngineVariant,
    build_cases,
    evaluate,
    run_trial,
    swa_cache_evidence,
)
from inferenceos.tune.prompt_reuse_sim import SimFaults, SimulatedReuseEngine
from inferenceos.tune.reuse_correctness import (
    DEFAULT,
    EVIDENCE_INCONCLUSIVE,
    EVIDENCE_NONE,
    EVIDENCE_NOT_INFORMATIVE,
    EVIDENCE_RIGHT,
    EVIDENCE_WRONG,
    FULL_REUSE,
    NO_REUSE,
    PARTIAL_REUSE,
    REUSE_UNKNOWN,
    RIGHT,
    SWA_FULL,
    UNSCORABLE,
    WRONG,
    ChatReading,
    answer_line,
    answer_oracle,
    build_pairs,
    chat_evidence,
    chat_reading,
    chat_summary,
    correctness_summary,
    judge,
    pair_outcome,
    partial_reuse_evidence,
    run_chat_exposure,
    swa_full_comparison,
)

PAIRS = build_pairs()
BY_ID = {p.pair_id: p for p in PAIRS}
REGRESSION = BY_ID["regression-pump-b-unit-7"]
LOADED = "srv    operator(): model loaded"


def _gemma_like(**faults: Any) -> SimulatedReuseEngine:
    return SimulatedReuseEngine(oracle=answer_oracle(PAIRS), sliding_window=True,
                                faults=SimFaults(partial_reuse_wrong=True, **faults))


def _run(engine: Any, variants: tuple[EngineVariant, ...] = (DEFAULT,), blocks: int = 2,
         pairs: Any = PAIRS) -> dict[str, Any]:
    return evaluate(engine, cases=tuple(p.case for p in pairs), variants=variants, blocks=blocks,
                    context_length=4096, configuration={}, run_id="t", power_check=False, cancellation=False)


def _summary(record: dict[str, Any], pairs: Any = PAIRS) -> dict[str, Any]:
    from inferenceos.tune.prompt_reuse import TrialResult

    return correctness_summary([TrialResult(**t) for t in record["trials"]], pairs)


# --------------------------------------------------------------------------- #
# The pairs                                                                   #
# --------------------------------------------------------------------------- #
def test_the_regression_pair_is_job_127s_failing_pair_exactly() -> None:
    shared = {c.case_id: c for c in build_cases(12)}["shared_prefix"]
    assert REGRESSION.case.prime == shared.prime and REGRESSION.case.measured == shared.measured
    assert REGRESSION.regression and REGRESSION.expected == "57"
    assert sum(p.regression for p in PAIRS) == 1 and len(PAIRS) == 9


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p.pair_id)
def test_every_known_answer_is_stated_once_and_never_in_the_questions(pair: Any) -> None:
    whole = re.compile(rf"(?<!\d){pair.expected}(?!\d)")
    # The fact sentence for the measured section states the expected number.
    assert whole.search(pair.document)
    assert not whole.search(pair.question_measured) and not whole.search(pair.question_prime)
    assert pair.case.prime != pair.case.measured
    # Both prompts share the document and part only at the question.
    assert pair.case.prime.startswith(pair.document) and pair.case.measured.startswith(pair.document)


def test_expected_numbers_are_unique_values_of_their_document() -> None:
    from inferenceos.tune.reuse_correctness import _facts

    for pair in PAIRS:
        values = [v for n in range(1, 13) for v in _facts(n).values()]
        assert values.count(int(pair.expected)) == 1, pair.pair_id  # no other fact has this value


def test_the_prompts_fit_well_inside_the_context() -> None:
    for pair in PAIRS:
        assert len(pair.case.measured.split()) * 2 < 4096 // 2


# --------------------------------------------------------------------------- #
# The judge                                                                   #
# --------------------------------------------------------------------------- #
def test_the_judge_reads_job_127s_real_gemma_replies() -> None:
    fresh = "\nPump B starts when the inlet passes 57 degrees.\nThe inlet passes 57 degrees.\nThe"
    reuse = (" The material does not contain the answer.\nFinal Answer: The final answer is "
             "$\\boxed{When the material does not")
    assert judge(fresh, REGRESSION) == (RIGHT, "right")
    verdict, why = judge(reuse, REGRESSION)
    assert verdict == WRONG and "refusal" in why


@pytest.mark.parametrize("text,verdict", [
    # The reviewer's wrong answers, once scored right: a sign, a fraction, swapped roles, a denial.
    ("The answer is -57 degrees.", WRONG),
    ("The answer is \u221257 degrees.", WRONG),
    ("It starts at 57/7 degrees.", WRONG),
    ("It starts at 7 degrees; 57 is the unit number.", UNSCORABLE),
    ("It does not start above 57 degrees.", UNSCORABLE),
    # The reviewer's right answers, once scored wrong.
    ("Fifty-seven degrees.", RIGHT),
    ("Here is the answer:\n57 degrees.", RIGHT),
    # More of the grammar.
    (" 57 degrees.", RIGHT),
    ("57.0 degrees", RIGHT),
    ("Answer: **57** degrees", RIGHT),
    (" Pump B of unit 7 starts when the inlet passes 57 degrees.", RIGHT),  # the question's own identifier
    (" Pump B of unit 7 starts at 57 degrees. Pump A of unit 7 starts at 41.", RIGHT),  # first sentence only
    (" 57\nQuestion: when does pump A start?\nAnswer: 41", RIGHT),  # first answer line only
    (" Pump B of unit 3 starts at 57 degrees.", UNSCORABLE),  # another unit's identifier
    (" 1057 degrees.", WRONG),
    (" 57.5 degrees.", WRONG),
    (" It starts at 58 degrees.", WRONG),
    (" Fifty-eight degrees.", WRONG),
    (" The material does not say.", WRONG),
    (" The answer is not stated in the manual.", WRONG),
    (" 57 or 58 degrees.", UNSCORABLE),
    (" Between 57 and 60 degrees.", UNSCORABLE),
    (" 57-60 degrees.", UNSCORABLE),
    (" It never starts below 58 degrees.", UNSCORABLE),  # a negated other value is not a stated fact
    (" Pump B starts when the inlet is warm.", UNSCORABLE),
    ("", UNSCORABLE),
    ("\n\n", UNSCORABLE),
    (None, UNSCORABLE),
    # Re-review finding 1: numbers read whole, signs honoured, unsupported forms unscorable.
    ("57e2 degrees.", UNSCORABLE),
    ("57E-1 degrees.", UNSCORABLE),
    ("negative fifty-seven degrees.", WRONG),
    ("minus 57 degrees.", WRONG),
    ("The answer is \u201357 degrees.", WRONG),
    ("fifty-seven hundred degrees.", UNSCORABLE),
    ("57 thousand degrees.", UNSCORABLE),
    ("57,000 degrees.", UNSCORABLE),
    ("57k degrees.", UNSCORABLE),
    ("+57 degrees.", RIGHT),
    ("plus 57 degrees.", RIGHT),
    ("positive fifty-seven degrees.", RIGHT),
    ("57\u00b0C.", RIGHT),
    ("57 + 3 degrees.", UNSCORABLE),
    ("57 minus 3 degrees.", UNSCORABLE),
    ("57 times 2.", UNSCORABLE),
    # Re-review r3 finding 1: spaced signs and continuations, and exact fractions.
    ("The answer is - 57 degrees.", UNSCORABLE),
    ("The answer is \u2212 57 degrees.", UNSCORABLE),
    ("The answer is (- 57) degrees.", UNSCORABLE),
    ("The answer is 57 M degrees.", UNSCORABLE),
    ("The answer is 57 k degrees.", UNSCORABLE),
    ("The answer is 57 e2 degrees.", UNSCORABLE),
    ("The answer is 57 E degrees.", UNSCORABLE),
    ("57/1 degrees.", RIGHT),
    ("114/2 degrees.", RIGHT),
    ("57/0 degrees.", UNSCORABLE),
    ("-114/2 degrees.", WRONG),
    ("5 7 degrees.", UNSCORABLE),
    ("57 000 degrees.", UNSCORABLE),
    ("057 degrees.", UNSCORABLE),
    ("57-degree inlet.", UNSCORABLE),
    ("Unit 7: 57 degrees.", RIGHT),
    ("\u0665\u0667 degrees.", UNSCORABLE),  # non-ASCII digits
    ("5\u0667 degrees.", UNSCORABLE),  # an ASCII digit then a non-ASCII one
    ("57\u00b2 degrees.", UNSCORABLE),
])
def test_the_judge_follows_the_pre_registered_grammar(text: str | None, verdict: str) -> None:
    assert judge(text, REGRESSION)[0] == verdict, judge(text, REGRESSION)


_OK_PREFIXES = ["", "+", "plus ", "positive ", "("]
_NEGATIVE_PREFIXES = ["-", "\u2212", "\u2013", "minus ", "negative ", "(-"]
_BAD_PREFIXES = ["- ", "\u2212 ", "+ ", "( - ", "~", "$", "#", "0", "\u00b1"]
_OK_SUFFIXES = ["", ".0", ".00", "/1", "\u00b0C", " \u00b0C", "%", "kPa", " degrees", ")"]
_OTHER_SUFFIXES = ["/7", ".5", "0"]
_BAD_SUFFIXES = ["e2", " e2", "E-1", " E", "k", " k", " M", "M", " m", ",000", " 000", " hundred", " thousand",
                 " million", "-60", " to 60", "x2", " x 2", "\u00b2", "/0", "-degree", " minus 3", " + 3", " - 3"]


@pytest.mark.parametrize("prefix", _OK_PREFIXES + _NEGATIVE_PREFIXES + _BAD_PREFIXES)
def test_every_combination_of_sign_and_continuation_is_read_by_the_rules(prefix: str) -> None:
    """Every prefix x suffix around 57 (20 x 37 forms): right only for an accepted form of
    exactly 57, wrong only for an accepted form of another value, unscorable for anything
    else. A new spacing variant of a rejected form cannot become right by prefix extraction."""
    for suffix in _OK_SUFFIXES + _OTHER_SUFFIXES + _BAD_SUFFIXES:
        text = f"The answer is {prefix}57{suffix}."
        if prefix in _BAD_PREFIXES or suffix in _BAD_SUFFIXES:
            expected = UNSCORABLE
        elif prefix in _NEGATIVE_PREFIXES or suffix in _OTHER_SUFFIXES:
            expected = WRONG
        else:
            expected = RIGHT
        assert judge(text, REGRESSION)[0] == expected, (text, judge(text, REGRESSION))


@pytest.mark.parametrize("sign,verdict", [("minus", WRONG), ("negative", WRONG),
                                         ("plus", RIGHT), ("positive", RIGHT)])
@pytest.mark.parametrize("left,right", [("", ""), ("(", ")"), ("[", "]"),
                                      ("**", "**"), ("__", "__"), ("`", "`")])
@pytest.mark.parametrize("number", ["57", "fifty-seven", "fifty seven"])
def test_sign_words_apply_after_number_formatting(sign: str, verdict: str,
                                                 left: str, right: str, number: str) -> None:
    """Compose the supported grammar pieces: formatting must never hide a negative sign."""
    text = f"The answer is {sign} {left}{number}{right} degrees."
    assert judge(text, REGRESSION)[0] == verdict, (text, judge(text, REGRESSION))


@pytest.mark.parametrize("text", [
    "negative +57 degrees.", "negative -57 degrees.", "positive -57 degrees.",
    "minus minus 57 degrees.", "negative positive **57** degrees.",
    "- negative (57) degrees.", "57 negative degrees.", "57 plus degrees.",
    "57 (*) 2 degrees.", "(*) 57 degrees.", "(**) 57 degrees.",
    "57 -; degrees.", "-; 57 degrees.",
])
def test_ambiguous_signs_and_formatted_operators_are_unscorable(text: str) -> None:
    assert judge(text, REGRESSION)[0] == UNSCORABLE, (text, judge(text, REGRESSION))


@pytest.mark.parametrize("magnitude", [
    "hundred", "hundreds", "thousand", "thousands", "million", "millions", "billion", "billions",
    "trillion", "trillions", "lakh", "lakhs", "crore", "crores", "dozen", "dozens", "bn", "mn",
])
def test_all_registered_magnitude_words_remain_unscorable(magnitude: str) -> None:
    assert judge(f"The answer is **57** ({magnitude}) degrees.", REGRESSION)[0] == UNSCORABLE


@pytest.mark.parametrize("text,verdict", [
    ("Unit **7**: **57** degrees.", RIGHT),
    ("Unit (7): (57) degrees.", RIGHT),
    ("Unit **7** **57** degrees.", UNSCORABLE),
    ("negative **57/1** degrees.", WRONG),
    ("negative **114/2** degrees.", WRONG),
    ("positive **114/2** degrees.", RIGHT),
    ("57 (**minus** 3) degrees.", UNSCORABLE),
])
def test_formatting_preserves_numeric_boundaries_and_exact_values(text: str, verdict: str) -> None:
    assert judge(text, REGRESSION)[0] == verdict, (text, judge(text, REGRESSION))


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p.pair_id)
def test_unicode_numeric_words_cannot_be_ignored(pair: Any) -> None:
    """Numeric discovery covers separated, signed, attached and non-digit numeric glyphs."""
    for modifier in ("\u00b2", "\u00b3", "\u2070", "\u207b\u00b2", "\u00bd", "\u216b"):
        for text in (f"The answer is {pair.expected} {modifier} degrees.",
                     f"The answer is {modifier} {pair.expected} degrees.",
                     f"The answer is {pair.expected}{modifier} degrees."):
            assert judge(text, pair)[0] == UNSCORABLE, (text, judge(text, pair))


def test_every_non_ascii_unicode_numeric_character_is_unscorable() -> None:
    """Scan Python's complete numeric character class, not a blacklist of examples.

    The lead-in check must not skip an unsupported numeric first line, either.
    """
    for ordinal in range(0x110000):
        char = chr(ordinal)
        if not char.isnumeric() or char in "0123456789":
            continue
        for text in (f"57 {char} degrees.", f"{char} 57 degrees.", f"57{char} degrees.",
                     f"{char}:\n57 degrees."):
            assert judge(text, REGRESSION)[0] == UNSCORABLE, (ordinal, text, judge(text, REGRESSION))


@pytest.mark.parametrize("number,verdict", [("eight", WRONG), ("fifty-eight", WRONG),
                                          ("fifty seven", RIGHT), ("57", RIGHT)])
def test_numeric_first_lines_are_not_skipped_after_formatting(number: str, verdict: str) -> None:
    for left, right in (("", ""), ("**", "**"), ("__", "__"), ("`", "`"), ("(", ")")):
        text = f"{left}{number}{right}:\n57 degrees."
        assert answer_line(text) != "57 degrees."
        assert judge(text, REGRESSION)[0] == verdict, (text, judge(text, REGRESSION))
    assert judge("Here is the answer:\n57 degrees.", REGRESSION)[0] == RIGHT


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p.pair_id)
@pytest.mark.parametrize("sign,verdict", [("positive", RIGHT), ("plus", RIGHT),
                                         ("negative", WRONG), ("minus", WRONG)])
def test_identifier_colon_preserves_signed_value_role(pair: Any, sign: str, verdict: str) -> None:
    label = "Row" if "row" in pair.question_measured else "Unit"
    identifier = min(pair.identifiers)
    for left, right in (("", ""), ("**", "**"), ("(", ")")):
        text = f"{label} {left}{identifier}{right}: {sign} {left}{pair.expected}{right} degrees."
        assert judge(text, pair)[0] == verdict, (text, judge(text, pair))


@pytest.mark.parametrize("text", [
    "Unit 3: positive 57 degrees.", "Unit -7: negative 57 degrees.",
    "Unit 7.0: negative 57 degrees.", "Unit 7th: negative 57 degrees.",
    "Unit 7 positive 57 degrees.", "Unit 7 57 degrees.",
    "Unit 7: negative +57 degrees.", "Unit 7: minus minus 57 degrees.",
    "Unit 7: + 57 degrees.", "Unit 7: 57 negative degrees.",
    "57: negative 58 degrees.", "Unit 7: 57 \u00b2 degrees.",
])
def test_identifier_colon_does_not_bypass_other_numeric_guards(text: str) -> None:
    assert judge(text, REGRESSION)[0] == UNSCORABLE, (text, judge(text, REGRESSION))


def test_the_judge_reads_identifiers_and_units_for_every_pair() -> None:
    assert judge(" 25kPa.", BY_ID["manual-pressure-unit-5"])[0] == RIGHT
    assert judge(" Every 20 days for unit 10.", BY_ID["manual-filters-unit-10"])[0] == RIGHT
    assert judge(" Week 45.", BY_ID["notes-pruned-row-5"])[0] == RIGHT
    assert judge(" Row 6 got its bees on day 16.", BY_ID["notes-bees-row-6"])[0] == RIGHT
    assert judge(" Row 6 got its bees on day 6.", BY_ID["notes-bees-row-6"])[0] == WRONG
    assert judge(" Row 4 got its bees on day 16.", BY_ID["notes-bees-row-6"])[0] == UNSCORABLE
    assert {p.pair_id: sorted(p.identifiers) for p in PAIRS}["notes-bloom-row-10"] == ["10"]


def test_the_answer_line_is_the_first_sentence_after_a_label_or_lead_in() -> None:
    assert answer_line("Final answer: 57 degrees. More.") == "57 degrees."
    assert answer_line("Here is the answer:\n\n57.") == "57."
    assert answer_line("Section 7:\n57.") == "Section 7:"  # states a number: not a lead-in
    assert answer_line("  \n") is None and answer_line(None) is None


# --------------------------------------------------------------------------- #
# The effective sliding-window cache                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("log,requested,state", [
    (f"x: {SWA_FULL_WARNING} (ref)\nx: creating non-SWA KV cache, size = 4096 cells\n"
     f"x: creating     SWA KV cache, size = 4096 cells\n{LOADED}", True, "full"),
    (f"x: creating non-SWA KV cache, size = 4096 cells\nx: creating     SWA KV cache, size = 3072 cells\n{LOADED}",
     False, "reduced"),
    (f"llama_kv_cache: size = 448 MiB\n{LOADED}", False, "no sliding-window cache"),
    ("x: creating non-SWA KV cache, size = 4096 cells\nx: creating     SWA KV cache, size = 3072 cells", True,
     "reduced"),  # requested full, the log shows reduced: never taken as full
    (f"x: {SWA_FULL_WARNING}\nx: creating non-SWA KV cache, size = 4096 cells\n"
     f"x: creating     SWA KV cache, size = 3072 cells\n{LOADED}", True, "unverified"),  # contradictory
    ("srv load_model: loading", False, "unverified"),  # incomplete log, no sizes
    (None, False, "unverified"),
])
def test_the_effective_sliding_window_cache_comes_from_the_loads_own_log(log: Any, requested: bool,
                                                                         state: str) -> None:
    evidence = swa_cache_evidence(log, requested_full=requested)
    assert evidence["state"].startswith(state) and evidence["requested_full"] is requested


def test_every_trial_records_its_sliding_window_evidence() -> None:
    trial = run_trial(_gemma_like(), REGRESSION.case, SWA_FULL, REUSE, block=0, context_length=4096,
                      configuration_sha256="c")
    assert trial.swa_cache is not None and trial.swa_cache["state"] == "full"
    assert trial.identity["server_swa_full"] is True
    ignored = run_trial(_gemma_like(swa_full_ignored=True), REGRESSION.case, SWA_FULL, REUSE, block=0,
                        context_length=4096, configuration_sha256="c")
    assert ignored.swa_cache is not None and ignored.swa_cache["state"] == "reduced"


# --------------------------------------------------------------------------- #
# The pre-registered classification                                           #
# --------------------------------------------------------------------------- #
def test_a_gemma_like_engine_is_wrong_with_reuse_and_right_with_a_full_window() -> None:
    summary = _summary(_run(_gemma_like(), (DEFAULT, SWA_FULL)))
    assert summary["variants"]["default"]["outcomes"] == {"wrong with partial reuse": 9}
    assert summary["variants"]["swa-full"]["outcomes"] == {"right and identical with reuse": 9}
    comparison = summary["swa_full_comparison"]
    assert comparison["interpretable"] is True and comparison["logs_verified"] is True
    assert set(comparison["comparable"].values()) == {"wrong only at the default"}
    assert len(comparison["comparable"]) == 9 and comparison["not_comparable"] == {}
    item = summary["pairs"]["default/regression-pump-b-unit-7"]
    assert item["reuse_wrong_with_partial"] == 2 and item["reuse_wrong_without_partial"] == 0
    block = item["blocks"][0]
    assert block["usable"] and block["partial_reuse"]
    assert block["fresh_verdict"] == RIGHT and block["reuse_verdict"] == WRONG


def test_a_correct_engine_is_right_and_identical() -> None:
    engine = SimulatedReuseEngine(oracle=answer_oracle(PAIRS))
    summary = _summary(_run(engine))
    assert summary["variants"]["default"]["outcomes"] == {"right and identical with reuse": 9}


def test_the_swa_full_comparison_needs_the_effective_setting() -> None:
    summary = _summary(_run(_gemma_like(swa_full_ignored=True), (DEFAULT, SWA_FULL)))
    comparison = summary["swa_full_comparison"]
    assert comparison["interpretable"] is False and "not verified" in comparison["reason"]
    only_default = _summary(_run(_gemma_like()))
    assert only_default["swa_full_comparison"]["interpretable"] is False


def test_uninformative_full_cache_pairs_are_never_removed_failures() -> None:
    """Review finding 2: every full-cache fresh answer wrong, logs intact. Those pairs are
    not comparable, and the comparison is not interpretable."""
    class FullCacheCannotAnswer(SimulatedReuseEngine):
        def generate(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int) -> Any:
            reading = super().generate(prompt, cache_prompt=cache_prompt, n_predict=n_predict, seed=seed)
            if self._variant is not None and self._variant.swa_full and not cache_prompt:
                reading.text = " It starts at 3 degrees."
            return reading

    engine = FullCacheCannotAnswer(oracle=answer_oracle(PAIRS), sliding_window=True,
                                   faults=SimFaults(partial_reuse_wrong=True))
    summary = _summary(_run(engine, (DEFAULT, SWA_FULL), pairs=PAIRS[:2]), PAIRS[:2])
    assert summary["variants"]["default"]["outcomes"] == {"wrong with partial reuse": 2}
    comparison = summary["swa_full_comparison"]
    assert comparison["logs_verified"] is True and comparison["comparable"] == {}
    assert comparison["interpretable"] is False and comparison["reason"] == "no conclusive pair"
    assert all("swa-full: not informative" in why for why in comparison["not_comparable"].values())


def test_an_unverified_load_in_a_duplicate_block_is_still_counted() -> None:
    """Review finding 2: a duplicate full-cache load with an unverified log is not hidden."""
    from inferenceos.tune.prompt_reuse import TrialResult

    record = _run(_gemma_like(), (DEFAULT, SWA_FULL), pairs=PAIRS[:1])
    trials = [TrialResult(**t) for t in record["trials"]]
    extra = replace(next(t for t in trials if t.variant_id == "swa-full" and t.arm == REUSE),
                    swa_cache={"state": "unverified: no startup log"})
    summary = correctness_summary(trials + [extra], PAIRS[:1])
    states = summary["variants"]["swa-full"]["swa_cache_states"]
    assert states == {"full": 4, "unverified: no startup log": 1}
    comparison = summary["swa_full_comparison"]
    assert comparison["logs_verified"] is False and comparison["interpretable"] is False


@pytest.mark.parametrize("reuse_text,outcome,evidence", [
    (" 114/2 degrees.", "changed with reuse, still right", EVIDENCE_RIGHT),  # once "wrong with partial reuse"
    (" The answer is - 57 degrees.", "unscorable with reuse", EVIDENCE_INCONCLUSIVE),  # once right
    (" The answer is 57 M degrees.", "unscorable with reuse", EVIDENCE_INCONCLUSIVE),  # once right
    (" The answer is -57 degrees.", "wrong with partial reuse", EVIDENCE_WRONG),
    (" The answer is negative **57** degrees.", "wrong with partial reuse", EVIDENCE_WRONG),
    (" The answer is minus (57) degrees.", "wrong with partial reuse", EVIDENCE_WRONG),
    (" The answer is minus __fifty-seven__ degrees.", "wrong with partial reuse", EVIDENCE_WRONG),
    (" The answer is positive **114/2** degrees.", "changed with reuse, still right", EVIDENCE_RIGHT),
    (" The answer is negative +57 degrees.", "unscorable with reuse", EVIDENCE_INCONCLUSIVE),
    (" The answer is 57 trillions degrees.", "unscorable with reuse", EVIDENCE_INCONCLUSIVE),
    (" The answer is 57 \u00b2 degrees.", "unscorable with reuse", EVIDENCE_INCONCLUSIVE),
    (" The answer is 57 \u00bd degrees.", "unscorable with reuse", EVIDENCE_INCONCLUSIVE),
    (" The answer is 57 \u207b\u00b2 degrees.", "unscorable with reuse", EVIDENCE_INCONCLUSIVE),
])
def test_number_forms_reach_the_summary_as_judged(reuse_text: str, outcome: str, evidence: str) -> None:
    """Re-review r3 finding 1 through ``correctness_summary``: the reuse arm's answers in
    a given form, fresh answers right, partial reuse in every block."""
    class Says(SimulatedReuseEngine):
        def generate(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int) -> Any:
            reading = super().generate(prompt, cache_prompt=cache_prompt, n_predict=n_predict, seed=seed)
            if cache_prompt and self._requests == 2:
                reading.text = reuse_text
            return reading

    summary = _summary(_run(Says(oracle=answer_oracle(PAIRS)), pairs=PAIRS[:1]), PAIRS[:1])
    item = summary["pairs"]["default/regression-pump-b-unit-7"]
    assert item["informative"] and item["partial_reuse_blocks"] == 2
    assert (item["outcome"], item["partial_reuse_evidence"]) == (outcome, evidence)


@pytest.mark.parametrize("text", ["negative **57** degrees.", "minus (57) degrees.",
                                "minus __fifty-seven__ degrees."])
def test_formatted_negative_answers_cannot_establish_no_failure(text: str) -> None:
    record = _run(_gemma_like(), (DEFAULT, SWA_FULL), pairs=PAIRS[:1])
    for trial in record["trials"]:
        if trial["variant_id"] == "default" and trial["arm"] == REUSE:
            trial["measured"]["text"] = text
    summary = _summary(record, PAIRS[:1])
    item = summary["pairs"]["default/" + REGRESSION.pair_id]
    assert item["partial_reuse_evidence"] == EVIDENCE_WRONG
    assert item["reuse_wrong_with_partial"] == 2
    comparison = summary["swa_full_comparison"]
    assert comparison["interpretable"]
    assert comparison["comparable"] == {REGRESSION.pair_id: "wrong only at the default"}


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p.pair_id)
@pytest.mark.parametrize("arm", [FRESH, REUSE])
def test_signed_identifier_answers_preserve_fresh_and_reuse_evidence(pair: Any, arm: str) -> None:
    label = "Row" if "row" in pair.question_measured else "Unit"
    for sign in ("positive", "plus", "negative", "minus"):
        record = _run(_gemma_like(), (DEFAULT, SWA_FULL), pairs=(pair,))
        for trial in record["trials"]:
            if trial["variant_id"] == "default" and trial["arm"] == arm:
                trial["measured"]["text"] = f"{label} {min(pair.identifiers)}: {sign} **{pair.expected}** degrees."
        summary = _summary(record, (pair,))
        item = summary["pairs"]["default/" + pair.pair_id]
        negative = sign in ("negative", "minus")
        if arm == FRESH and negative:
            assert not item["informative"]
            assert item["partial_reuse_evidence"] == EVIDENCE_NOT_INFORMATIVE
        else:
            assert item["informative"]
            expected = EVIDENCE_WRONG if arm == FRESH or negative else EVIDENCE_RIGHT
            assert item["partial_reuse_evidence"] == expected
            comparison = summary["swa_full_comparison"]
            wanted = "wrong only at the default" if expected == EVIDENCE_WRONG else "wrong under neither"
            assert comparison["interpretable"]
            assert comparison["comparable"] == {pair.pair_id: wanted}


def test_unscorable_full_cache_answers_never_remove_a_failure() -> None:
    """Re-review finding 2: every full-cache reuse answer unscorable ("57 or 58"). The
    default's failure is still reported, but nothing is said to be removed."""
    class FullCacheHedges(SimulatedReuseEngine):
        def generate(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int) -> Any:
            reading = super().generate(prompt, cache_prompt=cache_prompt, n_predict=n_predict, seed=seed)
            if self._variant is not None and self._variant.swa_full and cache_prompt and self._requests == 2:
                reading.text = " 57 or 58 degrees."
            return reading

    engine = FullCacheHedges(oracle=answer_oracle(PAIRS), sliding_window=True,
                             faults=SimFaults(partial_reuse_wrong=True))
    summary = _summary(_run(engine, (DEFAULT, SWA_FULL), pairs=PAIRS[:1]), PAIRS[:1])
    full = summary["pairs"]["swa-full/regression-pump-b-unit-7"]
    assert full["outcome"] == "unscorable with reuse" and full["reuse_unscorable"] == 2 and full["informative"]
    assert full["partial_reuse_evidence"] == EVIDENCE_INCONCLUSIVE
    assert summary["pairs"]["default/regression-pump-b-unit-7"]["partial_reuse_evidence"] == EVIDENCE_WRONG
    comparison = summary["swa_full_comparison"]
    assert comparison["logs_verified"] is True and comparison["comparable"] == {}
    assert comparison["inconclusive"] == {"regression-pump-b-unit-7": "wrong at the default; swa-full inconclusive"}
    assert comparison["interpretable"] is False and comparison["reason"] == "no conclusive pair"
    assert summary["variants"]["swa-full"]["partial_reuse_evidence"] == {
        EVIDENCE_INCONCLUSIVE: ["regression-pump-b-unit-7"]}


@pytest.mark.parametrize("default,full,where,label", [
    (EVIDENCE_WRONG, EVIDENCE_RIGHT, "comparable", "wrong only at the default"),
    (EVIDENCE_WRONG, EVIDENCE_WRONG, "comparable", "wrong under both"),
    (EVIDENCE_RIGHT, EVIDENCE_WRONG, "comparable", "wrong only with swa-full"),
    (EVIDENCE_RIGHT, EVIDENCE_RIGHT, "comparable", "wrong under neither"),
    (EVIDENCE_WRONG, EVIDENCE_INCONCLUSIVE, "inconclusive", "wrong at the default; swa-full inconclusive"),
    (EVIDENCE_INCONCLUSIVE, EVIDENCE_WRONG, "inconclusive", "wrong with swa-full; default inconclusive"),
    (EVIDENCE_INCONCLUSIVE, EVIDENCE_RIGHT, "inconclusive", "inconclusive"),
    (EVIDENCE_RIGHT, EVIDENCE_INCONCLUSIVE, "inconclusive", "inconclusive"),
    (EVIDENCE_WRONG, EVIDENCE_NONE, "not_comparable", "swa-full: no partial reuse happened"),
    (EVIDENCE_NOT_INFORMATIVE, EVIDENCE_RIGHT, "not_comparable", "default: not informative: x"),
])
def test_only_scored_evidence_on_both_sides_is_conclusive(default: str, full: str, where: str, label: str) -> None:
    def item(variant: str, evidence: str) -> dict[str, Any]:
        outcome = "not informative: x" if evidence == EVIDENCE_NOT_INFORMATIVE else "y"
        return {"variant": variant, "pair": "p", "outcome": outcome, "partial_reuse_evidence": evidence}

    summary = {"variants": {"default": {"swa_cache_states": {"reduced": 4}}, "swa-full": {"swa_cache_states": {"full": 4}}},
               "pairs": {"default/p": item("default", default), "swa-full/p": item("swa-full", full)}}
    comparison = swa_full_comparison(summary)
    assert comparison[where] == {"p": label}
    assert comparison["interpretable"] is (where == "comparable")


def _row(*, partial: bool, reuse: str, identical: bool = True, fresh: str = RIGHT) -> dict[str, Any]:
    return {"usable": True, "partial_reuse": partial, "fresh_verdict": fresh, "reuse_verdict": reuse,
            "answers_identical": identical}


@pytest.mark.parametrize("rows,outcome", [
    # Review finding 2: one partial-reuse block right, one zero-reuse block wrong. Never "still right".
    ([_row(partial=True, reuse=RIGHT), _row(partial=False, reuse=WRONG, identical=False)],
     "wrong without partial reuse"),
    ([_row(partial=True, reuse=WRONG, identical=False), _row(partial=False, reuse=WRONG, identical=False)],
     "wrong with partial reuse"),
    ([_row(partial=True, reuse=UNSCORABLE, identical=False), _row(partial=True, reuse=RIGHT)],
     "unscorable with reuse"),
    ([_row(partial=False, reuse=RIGHT), _row(partial=False, reuse=RIGHT)], "no partial reuse happened"),
    ([_row(partial=True, reuse=RIGHT, identical=False), _row(partial=True, reuse=RIGHT)],
     "changed with reuse, still right"),
    ([_row(partial=True, reuse=RIGHT), _row(partial=False, reuse=RIGHT)], "right and identical with reuse"),
    ([_row(partial=True, reuse=WRONG)], "not informative: fewer than 2 usable blocks"),
    ([_row(partial=True, reuse=WRONG), _row(partial=True, reuse=WRONG, fresh=UNSCORABLE)],
     "not informative: the fresh arm is not right in every usable block"),
    ([_row(partial=True, reuse=WRONG), _row(partial=True, reuse=WRONG), {"usable": False, "reasons": ["x"]}],
     "wrong with partial reuse"),
])
def test_the_pair_outcome_follows_the_pre_registered_order(rows: list[dict[str, Any]], outcome: str) -> None:
    assert pair_outcome(rows)[0] == outcome


@pytest.mark.parametrize("rows,outcome,evidence", [
    # An observed failure stays a failure, whatever else was unscorable.
    ([_row(partial=True, reuse=WRONG), _row(partial=True, reuse=UNSCORABLE)], "wrong with partial reuse",
     EVIDENCE_WRONG),
    # Unscorable partial-reuse answers: the absence of the failure is not established.
    ([_row(partial=True, reuse=UNSCORABLE), _row(partial=True, reuse=RIGHT)], "unscorable with reuse",
     EVIDENCE_INCONCLUSIVE),
    ([_row(partial=True, reuse=RIGHT), _row(partial=False, reuse=UNSCORABLE)], "unscorable with reuse",
     EVIDENCE_RIGHT),
    ([_row(partial=True, reuse=RIGHT), _row(partial=True, reuse=RIGHT)], "right and identical with reuse",
     EVIDENCE_RIGHT),
    # A wrong answer without partial reuse is reported in the outcome, not as the job 127 condition.
    ([_row(partial=True, reuse=RIGHT), _row(partial=False, reuse=WRONG)], "wrong without partial reuse",
     EVIDENCE_RIGHT),
    ([_row(partial=False, reuse=WRONG), _row(partial=False, reuse=RIGHT)], "wrong without partial reuse",
     EVIDENCE_NONE),
    ([_row(partial=True, reuse=WRONG)], "not informative: fewer than 2 usable blocks", EVIDENCE_NOT_INFORMATIVE),
])
def test_partial_reuse_evidence_is_kept_apart_from_the_outcome(rows: list[dict[str, Any]], outcome: str,
                                                                evidence: str) -> None:
    assert pair_outcome(rows)[0] == outcome
    assert partial_reuse_evidence(rows, outcome) == evidence


def test_a_pair_the_model_cannot_answer_fresh_says_nothing() -> None:
    def wrong_oracle(prompt: str) -> str | None:
        return " I do not know."

    engine = SimulatedReuseEngine(oracle=wrong_oracle, sliding_window=True)
    summary = _summary(_run(engine, pairs=PAIRS[:1]), PAIRS[:1])
    assert summary["pairs"]["default/regression-pump-b-unit-7"]["outcome"].startswith("not informative")


def test_no_partial_reuse_is_not_a_test_of_the_job_127_condition() -> None:
    class NoReuse(SimulatedReuseEngine):
        def generate(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int) -> Any:
            return super().generate(prompt, cache_prompt=False, n_predict=n_predict, seed=seed)

    summary = _summary(_run(NoReuse(oracle=answer_oracle(PAIRS)), pairs=PAIRS[:2]), PAIRS[:2])
    assert {v["outcome"] for v in summary["pairs"].values()} == {"no partial reuse happened"}


def test_incomplete_evidence_or_the_wrong_slot_makes_a_block_unusable() -> None:
    record = _run(_gemma_like(no_slot_ids=True), pairs=PAIRS[:1])
    item = _summary(record, PAIRS[:1])["pairs"]["default/regression-pump-b-unit-7"]
    assert item["usable_blocks"] == 0 and item["outcome"].startswith("not informative: fewer than 2")
    assert any("priming slot" in r for r in item["blocks"][0]["reasons"])


def test_a_duplicate_load_is_never_picked() -> None:
    from inferenceos.tune.prompt_reuse import TrialResult

    record = _run(_gemma_like(), pairs=PAIRS[:1])
    trials = [TrialResult(**t) for t in record["trials"]]
    extra = replace(next(t for t in trials if t.block == 0 and t.arm == REUSE))
    summary = correctness_summary(trials + [extra], PAIRS[:1])
    first = summary["pairs"]["default/regression-pump-b-unit-7"]["blocks"][0]
    assert first["usable"] is False and "2 reuse-arm loads (duplicate)" in first["reasons"][0]
    assert summary["variants"]["default"]["loads"] == 5  # every load is counted for the log check


# --------------------------------------------------------------------------- #
# Chat exposure                                                               #
# --------------------------------------------------------------------------- #
def _chat(engine: Any, pairs: Any = PAIRS[:1], blocks: int = 1, power: Any = lambda: "not checked",
          power_check: bool = False) -> tuple[list[dict[str, Any]], str | None, str | None]:
    return run_chat_exposure(engine, pairs, blocks=blocks, power=power, power_check=power_check)


@dataclass
class _Recorder(SimulatedReuseEngine):
    """Records each load's requests (their cache_prompt values)."""

    loads: list[list[Any]] = field(default_factory=list)

    def start(self, variant: Any) -> None:
        super().start(variant)
        self.loads.append([])

    def chat(self, messages: Any, *, cache_prompt: Any, **kwargs: Any) -> ChatReading:
        self.loads[-1].append(cache_prompt)
        return super().chat(messages, cache_prompt=cache_prompt, **kwargs)


def test_an_ordinary_chat_client_is_exposed_on_a_gemma_like_engine() -> None:
    engine = _gemma_like()
    trials, stop, _ = _chat(engine, PAIRS[:2])
    assert stop is None and engine.starts == engine.stops == 6  # client, matched and unprimed loads
    first = trials[0]
    assert [r["cache_prompt"] for r in first["client"]] == [None, None]
    assert [r["cache_prompt"] for r in first["matched"]] == [None, False]
    assert [r["cache_prompt"] for r in first["unprimed"]] == [False]
    assert first["matched"][1]["cached"] == 0 and 0 < first["client"][1]["cached"]
    assert first["client"][1]["cached"] < first["client"][1]["prompt_tokens"] - 1
    summary = chat_summary(trials, PAIRS)
    assert {v["outcome"] for v in summary["pairs"].values()} == {"exposed: wrong for the client after partial reuse"}
    row = summary["pairs"]["regression-pump-b-unit-7"]["trials"][0]
    assert (row["client_reuse"], row["client_verdict"]) == (PARTIAL_REUSE, WRONG)
    assert (row["matched_reuse"], row["matched_verdict"], row["unprimed_verdict"]) == (NO_REUSE, RIGHT, RIGHT)


def test_the_client_and_matched_reference_alternate_and_are_primed_identically() -> None:
    engine = _Recorder(oracle=answer_oracle(PAIRS))
    trials, stop, _ = _chat(engine, PAIRS[:1], blocks=2)
    assert stop is None and [t["order"] for t in trials] == [["client", "matched", "unprimed"],
                                                             ["matched", "client", "unprimed"]]
    assert engine.loads == [[None, None], [None, False], [False], [None, False], [None, None], [False]]


def test_a_correct_engine_leaves_the_client_right_after_partial_reuse() -> None:
    trials, _, _ = _chat(SimulatedReuseEngine(oracle=answer_oracle(PAIRS)), blocks=2)
    assert chat_summary(trials, PAIRS)["pairs"]["regression-pump-b-unit-7"]["outcome"] == (
        "right for the client after partial reuse")


def test_a_prior_request_effect_is_caught_by_the_matched_reference() -> None:
    """Review finding 4: wrong on every second request of a load, whatever cache_prompt says.
    The matched reference is wrong too, so reuse is not blamed."""
    class SecondRequestWrong(SimulatedReuseEngine):
        def chat(self, messages: Any, **kwargs: Any) -> ChatReading:
            reading = super().chat(messages, **kwargs)
            return replace(reading, text=" It starts at 3 degrees.") if self._requests == 2 else reading

    trials, stop, _ = _chat(SecondRequestWrong(oracle=answer_oracle(PAIRS)), blocks=2)
    item = chat_summary(trials, PAIRS)["pairs"]["regression-pump-b-unit-7"]
    assert stop is None and item["outcome"] == "not informative: the matched reference is not right"
    assert all(t["unprimed_verdict"] == RIGHT for t in item["trials"])  # the unprimed control alone would miss it


def _replace_measured(engine_class: Any, **fields: Any) -> Any:
    class Changed(engine_class):
        def chat(self, messages: Any, *, cache_prompt: Any, **kwargs: Any) -> ChatReading:
            reading: ChatReading = super().chat(messages, cache_prompt=cache_prompt, **kwargs)
            client_measured = cache_prompt is None and self._requests == 2
            return replace(reading, **fields) if client_measured else reading

    return Changed


@pytest.mark.parametrize("fields,problem", [
    # Review finding 3: no reply at all, with a positive cache count.
    ({"text": None, "finish_reason": None, "prompt_tokens": None, "completion_tokens": None},
     "no reply text"),
    ({"text": " "}, "no reply text"),
    ({"finish_reason": "tool_calls"}, "finish reason"),
    ({"finish_reason": None}, "finish reason"),
    ({"completion_tokens": 0}, "completion tokens"),
    ({"completion_tokens": 25}, "completion tokens"),
    ({"prompt_tokens": None}, "prompt or reused count"),  # 999 reused, total unknown
    ({"timings": {"cache_n": 999, "prompt_n": 1}}, "disagree"),
])
def test_a_missing_or_incomplete_chat_reply_is_unusable_never_wrong(fields: dict[str, Any], problem: str) -> None:
    trials, stop, _ = _chat(_replace_measured(SimulatedReuseEngine, **fields)(oracle=answer_oracle(PAIRS)))
    item = chat_summary(trials, PAIRS)["pairs"]["regression-pump-b-unit-7"]
    assert stop is None and item["outcome"] == "not usable" and item["usable"] == 0
    assert any(problem in r for r in item["trials"][0]["reasons"]), item["trials"][0]["reasons"]


def test_a_reply_that_ends_naturally_is_usable() -> None:
    engine = _replace_measured(SimulatedReuseEngine, finish_reason="stop", completion_tokens=6)
    trials, _, _ = _chat(engine(oracle=answer_oracle(PAIRS)))
    item = chat_summary(trials, PAIRS)["pairs"]["regression-pump-b-unit-7"]
    assert item["usable"] == 1 and item["outcome"] == "right for the client after partial reuse"


@pytest.mark.parametrize("reply,category", [
    ({"prompt_tokens": 1000, "cached": 0}, NO_REUSE),
    ({"prompt_tokens": 1000, "cached": 500}, PARTIAL_REUSE),
    ({"prompt_tokens": 1000, "cached": 998}, PARTIAL_REUSE),
    ({"prompt_tokens": 1000, "cached": 999}, FULL_REUSE),  # the reviewer's 999/1: full, not partial
    ({"prompt_tokens": 1000, "cached": 1000}, REUSE_UNKNOWN),
    ({"prompt_tokens": None, "cached": 999}, REUSE_UNKNOWN),
    ({"prompt_tokens": 1000, "cached": None}, REUSE_UNKNOWN),
    ({"prompt_tokens": 1000, "cached": 999, "timings": {"cache_n": 999, "prompt_n": 1}}, FULL_REUSE),
    ({"prompt_tokens": 1000, "cached": 500, "timings": {"cache_n": 499, "prompt_n": 501}}, REUSE_UNKNOWN),
])
def test_chat_reuse_is_classified_from_the_whole_prompt_count(reply: dict[str, Any], category: str) -> None:
    base = {"text": " 57", "finish_reason": "length", "completion_tokens": 24, "timings": {}}
    assert chat_evidence({**base, **reply})[0] == category


def test_a_full_reuse_wrong_answer_is_not_exposure() -> None:
    class FullReuseWrong(SimulatedReuseEngine):
        def chat(self, messages: Any, *, cache_prompt: Any, **kwargs: Any) -> ChatReading:
            reading = super().chat(messages, cache_prompt=cache_prompt, **kwargs)
            if cache_prompt is None and self._requests == 2:
                assert reading.prompt_tokens is not None
                total = reading.prompt_tokens
                return replace(reading, text=" 58.", cached=total - 1, timings={"cache_n": total - 1, "prompt_n": 1})
            return reading

    trials, stop, _ = _chat(FullReuseWrong(oracle=answer_oracle(PAIRS)))
    item = chat_summary(trials, PAIRS)["pairs"]["regression-pump-b-unit-7"]
    assert stop is None and item["outcome"] == "wrong for the client without partial reuse"


def test_priming_reuse_on_a_fresh_load_stops_at_once() -> None:
    """Review finding 5: impossible reuse on a fresh load's first request stops before any
    further request or load."""
    engine = _Recorder(oracle=answer_oracle(PAIRS), faults=SimFaults(prime_reuse=5))
    trials, stop, reason = _chat(engine, PAIRS[:2], blocks=2)
    assert stop == STOPPED_VALIDITY and reason and reason.startswith("E4") and "first request" in reason
    assert len(trials) == 1 and engine.loads == [[None]] and engine.starts == engine.stops == 1


def test_reuse_on_a_fresh_request_stops_at_once() -> None:
    class Dirty(_Recorder):
        def chat(self, messages: Any, *, cache_prompt: Any, **kwargs: Any) -> ChatReading:
            reading = super().chat(messages, cache_prompt=cache_prompt, **kwargs)
            return replace(reading, cached=5) if cache_prompt is False else reading

    engine = Dirty(oracle=answer_oracle(PAIRS))
    trials, stop, reason = _chat(engine, PAIRS, blocks=2)
    assert stop == STOPPED_VALIDITY and reason and reason.startswith("E1") and len(trials) == 1
    assert engine.loads == [[None, None], [None, False]]  # the unprimed load never ran


def test_impossible_reuse_stops_at_once() -> None:
    class Over(_Recorder):
        def chat(self, messages: Any, *, cache_prompt: Any, **kwargs: Any) -> ChatReading:
            reading = super().chat(messages, cache_prompt=cache_prompt, **kwargs)
            if cache_prompt is None and self._requests == 2:
                assert reading.prompt_tokens is not None
                return replace(reading, cached=reading.prompt_tokens)
            return reading

    engine = Over(oracle=answer_oracle(PAIRS))
    trials, stop, reason = _chat(engine, PAIRS, blocks=2)
    assert stop == STOPPED_VALIDITY and reason and reason.startswith("E4") and engine.loads == [[None, None]]


@pytest.mark.parametrize("states,loads", [
    (["battery"], 0),                                # before the first load
    (["ac", "battery", "ac", "ac"], 1),              # observed once between loads, mains back later
    (["ac", "ac", "ac", "unknown", "ac", "ac"], 2),  # an unknown reading is not mains
])
def test_every_observed_power_reading_counts(states: list[str], loads: int) -> None:
    """Review finding 5: a non-mains reading stops the run even when mains returns later."""
    readings = iter(states + ["ac"] * 50)
    engine = _Recorder(oracle=answer_oracle(PAIRS))
    trials, stop, _ = _chat(engine, PAIRS, blocks=2, power=lambda: next(readings), power_check=True)
    assert stop == STOPPED_POWER and len(trials) == 1 and len(engine.loads) == loads
    assert [p["state"] for p in trials[0]["power"]] == states[:len(trials[0]["power"])]


def test_chat_exposure_stops_on_unclean_cleanup() -> None:
    engine = _Recorder(oracle=answer_oracle(PAIRS), faults=SimFaults(leave_credential=True))
    trials, stop, _ = _chat(engine, PAIRS, blocks=2)
    assert stop == STOPPED_CLEANUP and len(trials) == 1 and engine.starts == 1  # no further load


def test_chat_exposure_interruption_keeps_completed_trials() -> None:
    saved: list[Any] = []

    class Interrupted(SimulatedReuseEngine):
        def chat(self, messages: Any, **kwargs: Any) -> ChatReading:
            if self.starts == 4:  # the second trial's first load
                raise KeyboardInterrupt
            return super().chat(messages, **kwargs)

    engine = Interrupted(oracle=answer_oracle(PAIRS))
    with pytest.raises(KeyboardInterrupt):
        run_chat_exposure(engine, PAIRS, blocks=1, power=lambda: "not checked", power_check=False,
                          checkpoint=lambda trials, stop, reason: saved.append((len(trials), stop)))
    assert saved[-1] == (2, "interrupted") and not engine.running and engine.starts == engine.stops == 4


def test_an_interrupted_power_reading_keeps_the_completed_load() -> None:
    """Re-review finding 3: Ctrl+C in the power reading after a completed load. The
    replies and cleanup are checkpointed once, then the interrupt is re-raised, and no
    further load runs."""
    saved: list[Any] = []
    readings = iter(["ac"])

    def power() -> str:
        try:
            return next(readings)
        except StopIteration:
            raise KeyboardInterrupt from None

    engine = _Recorder(oracle=answer_oracle(PAIRS))
    with pytest.raises(KeyboardInterrupt):
        run_chat_exposure(engine, PAIRS, blocks=2, power=power, power_check=True,
                          checkpoint=lambda trials, stop, reason: saved.append(
                              (json.loads(json.dumps(trials, default=str)), stop, reason)))
    assert len(saved) == 1
    trials, stop, reason = saved[0]
    assert stop == "interrupted" and len(trials) == 1
    trial = trials[0]
    assert trial["interrupted"] and "interrupted while reading power after client" in trial["problems"]
    assert [r["role"] for r in trial["client"]] == ["prime", "measured"] and trial["client"][1]["text"]
    assert trial["cleanup"] == [{**trial["cleanup"][0], "load": "client", "clean": True}]
    assert trial["power"] == [{"when": "before client", "state": "ac"}] and "matched" not in trial
    assert engine.loads == [[None, None]] and engine.starts == engine.stops == 1 and not engine.running


def test_an_interrupted_power_reading_before_a_load_starts_nothing() -> None:
    saved: list[Any] = []

    def power() -> str:
        raise KeyboardInterrupt

    engine = _Recorder(oracle=answer_oracle(PAIRS))
    with pytest.raises(KeyboardInterrupt):
        run_chat_exposure(engine, PAIRS, blocks=1, power=power, power_check=True,
                          checkpoint=lambda trials, stop, reason: saved.append((len(trials), stop)))
    assert saved == [(1, "interrupted")] and engine.starts == 0


def test_an_interrupted_power_reading_after_a_completion_trial_keeps_it() -> None:
    """The same boundary in the /completion trial (found while fixing finding 3): the
    trial's readings are checkpointed before the interrupt is re-raised."""
    saved: list[dict[str, Any]] = []
    readings = iter(["ac"])

    def power() -> str:
        try:
            return next(readings)
        except StopIteration:
            raise KeyboardInterrupt from None

    engine = SimulatedReuseEngine(oracle=answer_oracle(PAIRS))
    with pytest.raises(KeyboardInterrupt):
        evaluate(engine, cases=(REGRESSION.case,), variants=(DEFAULT,), blocks=1, context_length=4096,
                 configuration={}, run_id="t", power=power, power_check=True, cancellation=False,
                 checkpoint=lambda record: saved.append(json.loads(json.dumps(record, default=str))))
    last = saved[-1]
    assert last["status"] == "interrupted" and len(last["trials"]) == 1
    trial = last["trials"][0]
    assert trial["interrupted"] and trial["measured"]["text"] and trial["cleanup"]["process_gone"]
    assert trial["power"] == ["ac", "not read (interrupted)"]
    assert engine.starts == engine.stops == 1 and not engine.running


def test_chat_replies_are_read_from_the_engines_own_fields() -> None:
    payload = {"choices": [{"message": {"content": " 57 degrees."}, "finish_reason": "length"}],
               "usage": {"prompt_tokens": 1004, "completion_tokens": 24,
                         "prompt_tokens_details": {"cached_tokens": 990}},
               "timings": {"cache_n": 990, "prompt_n": 14}, "id_slot": 3}
    reading = chat_reading(payload)
    assert (reading.text, reading.finish_reason, reading.prompt_tokens, reading.cached,
            reading.completion_tokens) == (" 57 degrees.", "length", 1004, 990, 24)
    assert reading.timings == {"cache_n": 990, "prompt_n": 14} and not hasattr(reading, "slot")
    assert chat_evidence(asdict(reading)) == (PARTIAL_REUSE, [])
    empty = chat_reading({"choices": [{"message": None}], "usage": "x"})
    assert empty.text is None and empty.cached is None and empty.prompt_tokens is None
    assert chat_evidence(asdict(empty))[0] == REUSE_UNKNOWN


# --------------------------------------------------------------------------- #
# Plumbing                                                                    #
# --------------------------------------------------------------------------- #
def test_the_chat_request_leaves_cache_prompt_out_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.backends.llama_server_toolchain import LlamaServerClient

    sent: list[Any] = []
    client = LlamaServerClient(host="127.0.0.1", port=1, api_key="k")

    class Reply:
        def read(self) -> bytes:
            return b'{"choices": []}'

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

    def fake_open(path: str, *, body: Any = None, method: str = "GET", timeout_s: Any = None) -> Any:
        sent.append((path, body))
        return Reply()

    monkeypatch.setattr(client, "_open", fake_open)
    messages = [{"role": "user", "content": "q"}]
    client.chat_request(messages=messages, max_tokens=4, seed=1, temperature=0.0, cache_prompt=None)
    client.chat_request(messages=messages, max_tokens=4, seed=1, temperature=0.0, cache_prompt=False)
    assert sent[0][0] == "/v1/chat/completions" and "cache_prompt" not in sent[0][1]
    assert sent[1][1]["cache_prompt"] is False


def test_swa_full_reaches_the_command_only_when_set(tmp_path: Path) -> None:
    from inferenceos.backends.llama_server_toolchain import build_server_command
    from inferenceos.backends.resident_server import _options_from
    from inferenceos.interfaces.backend_adapter import RuntimeParams

    model = tmp_path / "m.gguf"
    plain = build_server_command(Path("s"), _options_from(model, RuntimeParams(context_length=4096)),
                                 host="h", port=1, api_key_file=tmp_path / "k")
    full = build_server_command(Path("s"), _options_from(model, RuntimeParams(context_length=4096, swa_full=True)),
                                host="h", port=1, api_key_file=tmp_path / "k")
    assert "--swa-full" not in plain and full.count("--swa-full") == 1


def test_only_the_evaluations_ever_set_swa_full() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "inferenceos"
    allowed = {"tune/prompt_reuse.py", "tune/prompt_reuse_sim.py", "tune/reuse_correctness.py",
               "backends/resident_server.py", "backends/llama_server_toolchain.py", "interfaces/backend_adapter.py"}
    offenders = [p.relative_to(root).as_posix() for p in root.rglob("*.py")
                 if p.relative_to(root).as_posix() not in allowed and "swa_full" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_the_real_adapter_passes_swa_full_and_asks_chat_through_the_backend() -> None:
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from tests.test_prompt_reuse import _script

    seen: dict[str, Any] = {}

    class Backend:
        def start(self, model: Any, params: Any) -> None:
            seen["swa_full"] = params.swa_full

        def chat_request(self, messages: Any, *, max_tokens: int, seed: int, cache_prompt: Any) -> Any:
            seen["cache_prompt"] = cache_prompt
            return {"choices": [{"message": {"content": " 57"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 3}},
                    "timings": {"cache_n": 3, "prompt_n": 2}}

    engine = _script().RealReuseEngine(model=object(), configuration=RuntimeConfiguration(context_length=4096),
                                       backend_factory=Backend)
    engine.start(SWA_FULL)
    reading = engine.chat([{"role": "user", "content": "q"}], cache_prompt=None, max_tokens=4, seed=1)
    assert seen == {"swa_full": True, "cache_prompt": None} and reading.text == " 57" and reading.cached == 3
    assert (reading.prompt_tokens, reading.finish_reason, reading.completion_tokens) == (5, "stop", 1)


# --------------------------------------------------------------------------- #
# The runner                                                                  #
# --------------------------------------------------------------------------- #
def test_the_runner_runs_the_correctness_study_end_to_end(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _script

    out = tmp_path / "sim128.json"
    code = _script().main(["--simulate", "--study", "correctness", "--swa-full-variant", "--chat-exposure",
                           "--blocks", "2", "--out", str(out), "--run-id", "r128"])
    record = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and record["status"] == DONE and record["evaluation"] == "prompt-reuse-eval-4"
    assert len(record["trials"]) == 9 * 2 * 2 * 2 and record["cancellation"] == []
    assert record["correctness"]["swa_full_comparison"]["interpretable"] is True
    assert set(record["chat_summary"]["pairs"]) == {p.pair_id for p in PAIRS}
    assert record["study"]["pairs"]["regression-pump-b-unit-7"]["expected"] == "57"


def test_the_runner_refuses_correctness_flags_for_the_reuse_study(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _script

    assert _script().main(["--simulate", "--swa-full-variant", "--out", str(tmp_path / "x.json")]) == 3
    assert not (tmp_path / "x.json").exists()


def test_the_runner_keeps_the_reuse_study_unchanged_by_default(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _script

    out = tmp_path / "sim127.json"
    assert _script().main(["--simulate", "--blocks", "1", "--out", str(out)]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert "correctness" not in record and record["cancellation"] and record["planned_trials"] == 16


def test_a_chat_stop_gives_the_runner_its_stop_status(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _script

    script = _script()

    class LeavesKeyOnChatLoads(SimulatedReuseEngine):
        def chat(self, messages: Any, **kwargs: Any) -> ChatReading:
            self.faults = SimFaults(leave_credential=True)  # the next stop is unclean
            return super().chat(messages, **kwargs)

    engine = LeavesKeyOnChatLoads(oracle=answer_oracle(PAIRS))
    out = tmp_path / "r.json"
    with patch("inferenceos.tune.prompt_reuse_sim.SimulatedReuseEngine", return_value=engine):
        code = script.main(["--simulate", "--study", "correctness", "--chat-exposure", "--blocks", "2",
                            "--cases", "correctness/regression-pump-b-unit-7", "--out", str(out)])
    record = json.loads(out.read_text(encoding="utf-8"))
    assert code == 5 and record["status"] == STOPPED_CLEANUP and len(record["chat_trials"]) == 1
    assert record["correctness"]["pairs"]  # the completed trials' summary is kept
    assert {t["arm"] for t in record["trials"]} == {FRESH, REUSE}


def _run_runner_with(engine: Any, tmp_path: Path) -> tuple[int, dict[str, Any]]:
    from tests.test_prompt_reuse import _script

    out = tmp_path / "probe.json"
    with patch("inferenceos.tune.prompt_reuse_sim.SimulatedReuseEngine", return_value=engine):
        code = _script().main(["--simulate", "--study", "correctness", "--chat-exposure", "--blocks", "2",
                               "--cases", "correctness/regression-pump-b-unit-7", "--out", str(out)])
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_the_runner_never_reports_a_missing_chat_reply_as_done(tmp_path: Path) -> None:
    """Review finding 3, through the real runner: once exit 0, status done, "exposed"."""
    engine = _replace_measured(SimulatedReuseEngine, text=None, finish_reason=None, prompt_tokens=None,
                               completion_tokens=None)(oracle=answer_oracle(PAIRS), sliding_window=True,
                                                       faults=SimFaults(partial_reuse_wrong=True))
    code, record = _run_runner_with(engine, tmp_path)
    assert code == 2 and record["status"] == "incomplete" and "not usable" in record["stop_reason"]
    assert record["chat_summary"]["pairs"]["regression-pump-b-unit-7"]["outcome"] == "not usable"


def test_the_runner_stops_on_priming_reuse_in_the_chat_exposure(tmp_path: Path) -> None:
    """Review finding 5, through the real runner: once exit 0, status done, four starts."""
    class PrimingReuse(SimulatedReuseEngine):
        def chat(self, messages: Any, **kwargs: Any) -> ChatReading:
            reading = super().chat(messages, **kwargs)
            return replace(reading, cached=5) if self._requests == 1 else reading

    engine = PrimingReuse(oracle=answer_oracle(PAIRS))
    code, record = _run_runner_with(engine, tmp_path)
    assert code == 7 and record["status"] == STOPPED_VALIDITY and record["stop_reason"].startswith("E4")
    assert len(record["chat_trials"]) == 1 and record["chat_trials"][0]["client"][0]["cached"] == 5
    assert "matched" not in record["chat_trials"][0]  # nothing dispatched after the stop
