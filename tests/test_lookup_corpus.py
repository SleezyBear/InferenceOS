"""Absolute complete-task oracles, independent of any engine's preferred answer."""
from __future__ import annotations

from collections import Counter
import os

import pytest
from pydantic import ValidationError

from inferenceos.tune.lookup_corpus import (
    CodeTask, ProseTask, Row, RowsTask, TaskCase, TaskCorpus, build_corpus, judge,
)
from inferenceos.tune.lookup_decoding import build_cases as old_cases

CORPUS = build_corpus()
CASES = CORPUS.cases


def test_software_corpus_population_and_digest_are_prospective_and_stable() -> None:
    assert CORPUS.sha256 == "363d3870ad21b072a18cc269fda1cf325a641e28e5fb3a37039910d3840c21ea"
    assert build_corpus() == CORPUS
    assert len(CASES) == 42
    assert Counter(c.task.family for c in CASES) == {"copy-prose": 14, "copy-code": 14, "extract-rows": 14}
    assert Counter(c.split for c in CASES) == {"calibration": 21, "holdout": 21}
    assert Counter(c.length for c in CASES) == {"short": 18, "medium": 12, "long": 12}
    assert Counter(c.opportunity for c in CASES) == {"repeated": 18, "varied": 18, "short-output": 6}
    assert not {c.prompt for c in CASES} & {c.prompt for c in old_cases()}
    calibration = {c.expected for c in CASES if c.split == "calibration"}
    holdout = {c.expected for c in CASES if c.split == "holdout"}
    assert not calibration & holdout
    assert TaskCorpus.model_validate_json(CORPUS.model_dump_json()) == CORPUS


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_entire_expected_answer_and_declared_newline_equivalence(case: TaskCase) -> None:
    for text in (case.expected, case.expected + "\n", case.expected.replace("\n", "\r\n") + "\r\n"):
        assert judge(case, text, ending="eos", truncated=False).passed


@pytest.mark.parametrize("mutation", ("empty", "prefix", "suffix", "missing_tail", "padding", "duplicate",
                                      "bom", "zero_width", "two_newlines", "lone_cr"))
def test_no_substring_or_blanket_whitespace_normalization(mutation: str) -> None:
    for case in CASES:
        expected = case.expected
        text = {
            "empty": "", "prefix": "Here is the answer:\n" + expected,
            "suffix": expected + "\nDone.", "missing_tail": expected[:-1],
            "padding": " " + expected + " ", "duplicate": expected + "\n" + expected,
            "bom": "\ufeff" + expected, "zero_width": expected[:1] + "\u200b" + expected[1:],
            "two_newlines": expected + "\n\n", "lone_cr": expected + "\r",
        }[mutation]
        assert not judge(case, text, ending="eos", truncated=False).passed


@pytest.mark.parametrize("ending,truncated", (("limit", False), ("error", False), ("eos", True),
                                             ("", False), ("EOS", False)))
def test_matching_text_without_complete_work_cannot_pass(ending: str, truncated: bool) -> None:
    for case in CASES:
        assert not judge(case, case.expected, ending=ending, truncated=truncated).passed


def test_omitted_labels_and_selected_rows_are_not_inherited_as_good_baselines() -> None:
    prose = next(c for c in CASES if c.task.family == "copy-prose")
    no_labels = "\n".join(line.split(": ", 1)[1] for line in prose.expected.splitlines())
    assert not judge(prose, no_labels, ending="eos", truncated=False).passed
    rows = next(c for c in CASES if c.task.family == "extract-rows" and c.length == "long")
    assert not judge(rows, "\n".join(reversed(rows.expected.splitlines())), ending="eos", truncated=False).passed
    assert isinstance(rows.task, RowsTask)
    all_rows = "\n".join(r.rendered() for r in rows.task.rows)
    assert not judge(rows, all_rows, ending="eos", truncated=False).passed


def test_numbers_unicode_and_code_indentation_are_not_reinterpreted(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> int:
        raise AssertionError("task data must never be executed")
    monkeypatch.setattr(os, "system", forbidden)
    code = TaskCase(case_id="code-data", split="holdout", length="short", opportunity="varied",
                    task=CodeTask(source="__import__('os').system('not executed')\n    cafÃ© = '-57/7'"))
    assert judge(code, code.expected, ending="eos", truncated=False).passed
    for text in (code.expected.replace("    ", "\t"), code.expected.replace("Ã©", "e\u0301"),
                 code.expected.replace("-57/7", "57")):
        assert not judge(code, text, ending="eos", truncated=False).passed


@pytest.mark.parametrize("bad", ("", " ", "\n", "one\ntwo", "\ufefftext", "text\u200b", "text\r"))
def test_record_input_rejects_ambiguous_line_or_control_boundaries(bad: str) -> None:
    with pytest.raises(ValidationError):
        ProseTask(records=(bad,))


def test_typed_inputs_refuse_duplicates_coercion_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Row(number=True, unit="A", reading="57")
    with pytest.raises(ValidationError):
        RowsTask(rows=(Row(number=2, unit="A", reading="57"), Row(number=2, unit="B", reading="58")))
    with pytest.raises(ValidationError):
        TaskCorpus(cases=(CASES[0], CASES[0]))
    with pytest.raises(ValidationError):
        TaskCorpus(cases=(CASES[0], CASES[0].model_copy(update={"case_id": "another-id"})))
    with pytest.raises(ValidationError):
        CodeTask.model_validate({"source": "print('text')", "execute": True})
    with pytest.raises(ValidationError):
        CASES[0].task = CodeTask(source="changed")
    with pytest.raises(ValidationError):
        CodeTask(source="\t \n ")
