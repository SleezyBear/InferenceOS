"""Known-answer questions for changes that can alter answers (quality-risk changes).

The normal answer check compares a new setup's answers with the current setup's on the
tuning questions (wording, numbers, required phrases). A change that stores numbers at
lower precision (the conversation memory format) needs more than "the text looks
similar": this policy (``POLICY_VERSION``) also asks questions whose right answer is
known, at temperature 0, with every answer of both setups checked:

* an arithmetic question (one exact number);
* a fact far back in a long text (one exact number and one direction);
* a code listed early in a long list (one exact code).

**Answers are read from an explicit answer line** (review 7): every question leads to an
``ANSWER: ...`` line. Only the **first** answer line is judged, and it is judged strictly:
the right value must be there as a whole number / word / code, and nothing that
contradicts it may be there (another number, another direction, a longer code).
Mentioning the right value somewhere is not enough: "1805" is not 805, "southwest" is not
south, "XF9595Y" is not F9595. No answer line is a wrong answer.

Tune sends prompts as plain text to complete (llama-server ``/completion``, no chat
template). Laptop job 110 (6 Oct 2026) showed what models do with that: they continue the
document, often inventing new questions with their own answer lines ("ANSWER: L7109 /
Question: what is the code for item 39? ..."), and Llama 3.2 3B wrote nothing at all after
an instruction-style ending. So the retrieval questions end with ``ANSWER:`` themselves
(the model's first line is its answer), the arithmetic question ends with ``Working:``
(the model works it out, then writes the answer line), and lines after the first answer
line are the model's own continuation, not answers to our question.

Rules (each question measured with both setups, each from a fresh load):

* a measurement is **usable** only with at least ``REQUIRED_RUNS`` good runs, no failed
  run, and (when a checker is given) the requested settings shown by the engine;
* a question counts (is *informative*) only when both measurements are usable and every
  answer of the current setup is right: a model that cannot answer it says nothing;
* **fail** when any answer of the new setup is wrong on an informative question, or any
  of its runs failed to execute;
* **incomplete** when any question that fits the context could not be completed (time
  ran out, a measurement was disturbed or too short, the settings were not shown):
  not eligible for automatic acceptance;
* **pass** only when every question that fits was completed, at least
  ``MIN_INFORMATIVE`` of them are informative, one a long-text question when the context
  holds it, and every new answer is right;
* otherwise **insufficient**: not enough evidence, so not eligible for automatic
  acceptance (reported with the reason, never as a pass).

A few right answers are not proof that answers never change; they are a smoke test that
the general answer check on the tuning questions sits on top of, not a replacement.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from inferenceos.contracts.enums import WorkloadKind
from inferenceos.contracts.experiment import ExperimentMeasurement, RuntimeConfiguration
from inferenceos.profiling.workloads import BenchmarkWorkload

POLICY_VERSION = "known-answers-v3"
MIN_INFORMATIVE = 2
#: Good runs each setup needs per question.
REQUIRED_RUNS = 2

PASS, FAIL, INSUFFICIENT, INCOMPLETE = "pass", "fail", "insufficient", "incomplete"

_NUMBER_PROMPT = (
    "Question: A shop sells pens at 17 rupees each and notebooks at 46 rupees each. Priya "
    "buys 23 pens and 9 notebooks. What is the total cost in rupees?\n"
    "Work it out step by step, then write the final line as 'ANSWER: <the total in rupees, "
    "as one number>'.\n"
    "Working:\n"
)
_DIRECTIONS = ("north", "south", "east", "west")
_NOTES_PROMPT = (
    "Read the notes below and then answer the question at the end.\n\n"
    + "\n".join(
        f"Note {i}: the warehouse in district {i % 7 + 1} received {12 * i + 5} crates of "
        f"supplies on day {i}, and shipped {7 * i + 3} crates to the "
        f"{_DIRECTIONS[i % 4]} depot."
        for i in range(1, 41)
    )
    + "\n\nQuestion: how many crates did the warehouse receive on day 17, and to which "
    "depot were crates shipped that day? Answer as: <number of crates received>, "
    "<depot direction>\n"
    "ANSWER:"
)


def _code(index: int) -> str:
    return f"{chr(65 + index % 26)}{(index * 7919) % 10000:04d}"


_CODES_PROMPT = (
    "Here is a list of item codes.\n"
    + "\n".join(f"Item {i}: code {_code(i)}" for i in range(1, 61))
    + "\n\nQuestion: what is the code for item 5?\n"
    "ANSWER:"
)

_ANSWER_RE = re.compile(r"^[\s>*_#`-]*answer[\s*_`]*:[\s*_`]*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
_WORD_RE = re.compile(r"[a-z]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def primed(question_prompt: str) -> bool:
    """Whether a prompt ends with the answer label itself (the reply is the answer)."""
    return question_prompt.rstrip().endswith("ANSWER:")


def answer_fields(text: str) -> list[str]:
    """Every ``ANSWER: ...`` line of a reply (markdown emphasis and quotes removed). An
    answer label alone on its line takes the next non-empty line as its value."""
    fields = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = _ANSWER_RE.match(line)
        if match is None:
            continue
        value = match.group(1).strip().strip("*_`\"'").strip()
        if not value:
            following = next((item.strip() for item in lines[index + 1:] if item.strip()), "")
            value = following.strip("*_`\"'").strip()
        fields.append(value)
    return fields


def _numbers(field_text: str) -> list[str]:
    """Whole numbers in an answer field (thousands commas removed, "1805" stays 1805)."""
    found = []
    for match in re.finditer(r"\d[\d,]*(?:\.\d+)?", field_text):
        start, end = match.span()
        before = field_text[start - 1] if start else " "
        after = field_text[end] if end < len(field_text) else " "
        if before.isalpha() or after.isalpha():
            # Part of a code or word (e.g. "F9595", "17th" is kept below).
            if not (after.isalpha() and field_text[end:end + 2].lower() in {"st", "nd", "rd", "th"}):
                continue
        number = match.group(0).rstrip(",").replace(",", "")
        if "." in number and set(number.split(".", 1)[1]) <= {"0"}:
            number = number.split(".", 1)[0]
        found.append(number)
    return found


def _check_number(expected: str) -> Callable[[str], tuple[bool, str]]:
    def check(field_text: str) -> tuple[bool, str]:
        numbers = _numbers(field_text)
        if numbers == [expected] or (numbers and all(n == expected for n in numbers)):
            return True, "right"
        if not numbers:
            return False, "no number in the answer line"
        return False, f"answer line has {', '.join(numbers)} (expected only {expected})"

    return check


def _check_fact(expected_number: str, allowed: set[str], direction: str) -> Callable[[str], tuple[bool, str]]:
    compounds = {"northeast", "northwest", "southeast", "southwest"}

    def check(field_text: str) -> tuple[bool, str]:
        numbers = _numbers(field_text)
        words = set(_WORD_RE.findall(field_text.lower()))
        if expected_number not in numbers:
            return False, f"answer line lacks {expected_number}"
        extra = [n for n in numbers if n not in allowed]
        if extra:
            return False, f"answer line also has {', '.join(extra)}"
        if direction not in words:
            return False, f"answer line lacks '{direction}' as a word"
        wrong = sorted((words & (set(_DIRECTIONS) | compounds)) - {direction})
        if wrong:
            return False, f"answer line also names {', '.join(wrong)}"
        return True, "right"

    return check


def _check_code(expected: str, allowed_digits: set[str]) -> Callable[[str], tuple[bool, str]]:
    def check(field_text: str) -> tuple[bool, str]:
        tokens = _TOKEN_RE.findall(field_text)
        codes = [t for t in tokens if t.upper() == expected.upper()]
        if not codes:
            return False, f"answer line lacks the code {expected} as a whole word"
        odd = [
            t for t in tokens
            if any(ch.isdigit() for ch in t) and t.upper() != expected.upper() and t not in allowed_digits
        ]
        if odd:
            return False, f"answer line also has {', '.join(odd)}"
        return True, "right"

    return check


@dataclass(frozen=True)
class KnownQuestion:
    question_id: str
    prompt: str
    #: Judges one answer field: (right, words).
    check: Callable[[str], tuple[bool, str]]
    expected: str
    long_text: bool
    #: Pieces of context needed (question plus answer), a safe over-estimate.
    context_tokens: int


QUESTIONS: tuple[KnownQuestion, ...] = (
    KnownQuestion(
        "arithmetic", _NUMBER_PROMPT, _check_number("805"), "805",
        long_text=False, context_tokens=360,
    ),
    KnownQuestion(
        "fact far back", _NOTES_PROMPT, _check_fact("209", {"209", "17"}, "south"), "209, south",
        long_text=True, context_tokens=1850,
    ),
    KnownQuestion(
        "code early in a list", _CODES_PROMPT, _check_code(_code(5), {"5"}), _code(5),
        long_text=True, context_tokens=950,
    ),
)


def workload_for(question: KnownQuestion) -> BenchmarkWorkload:
    return BenchmarkWorkload(
        workload_id=f"known-{question.question_id.replace(' ', '-')}",
        kind=WorkloadKind.SHORT if not question.long_text else WorkloadKind.LONG,
        description=f"known-answer question: {question.question_id}",
        prompt=question.prompt,
        n_predict=48 if primed(question.prompt) else 220,
        seed=1234,
        temperature=0.0,
        min_context_tokens=question.context_tokens,
    )


def judge_answer(question: KnownQuestion, text: str) -> tuple[bool, str]:
    """Whether one reply is right, from its first answer line only (strictly)."""
    if primed(question.prompt) and not text.lstrip().upper().startswith("ANSWER"):
        # The prompt ended with "ANSWER:", so the reply's first line is the answer.
        text = "ANSWER:" + text
    fields = answer_fields(text)
    if not fields:
        return False, "no 'ANSWER:' line"
    return question.check(fields[0])


def answer_right(question: KnownQuestion, text: str) -> bool:
    return judge_answer(question, text)[0]


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    informative: bool
    start_right: int
    start_answers: int
    new_right: int
    new_answers: int
    failed_runs: int
    notes: list[str] = field(default_factory=list)
    #: Fits the context and was completed with usable measurements of both setups.
    completed: bool = False
    asked: bool = False


@dataclass(frozen=True)
class KnownAnswerCheck:
    status: str
    reason: str
    questions: list[QuestionResult]
    version: str = POLICY_VERSION
    #: What the engine showed on each known-answer measurement (when checked).
    settings_shown: list[dict[str, object]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def record(self) -> dict[str, object]:
        return {
            "policy": self.version,
            "status": self.status,
            "reason": self.reason,
            "questions": [
                {
                    "question": item.question_id,
                    "asked": item.asked,
                    "completed": item.completed,
                    "informative": item.informative,
                    "current_right": f"{item.start_right}/{item.start_answers}",
                    "new_right": f"{item.new_right}/{item.new_answers}",
                    "failed_runs": item.failed_runs,
                    "notes": item.notes,
                }
                for item in self.questions
            ],
            "settings_shown": self.settings_shown,
        }


Measure = Callable[[RuntimeConfiguration, BenchmarkWorkload, str], ExperimentMeasurement]
#: Checks a measurement showed its configuration's settings: (shown, words).
SettingsCheck = Callable[[RuntimeConfiguration, ExperimentMeasurement], tuple[bool, str]]


def _usable(
    measurement: ExperimentMeasurement, config: RuntimeConfiguration, side: str,
    settings_check: SettingsCheck | None, shown: list[dict[str, object]], question: str,
) -> tuple[bool, list[str], list[str]]:
    """(usable, notes, texts of its good runs)."""
    notes: list[str] = []
    texts = list(measurement.generated_texts)
    if measurement.valid_runs < REQUIRED_RUNS:
        notes.append(
            f"{side} setup: {measurement.valid_runs} good run(s), needs {REQUIRED_RUNS}"
            + (f" ({measurement.notes[-1]})" if measurement.notes else "")
        )
        return False, notes, []
    if len(texts) < measurement.valid_runs:
        notes.append(f"{side} setup: answers missing for some good runs")
        return False, notes, []
    texts = texts[: measurement.valid_runs]
    if settings_check is not None:
        ok, words = settings_check(config, measurement)
        shown.append({"question": question, "setup": side, "shown": ok, "words": words})
        if not ok:
            notes.append(f"{side} setup: its settings were not shown by the engine ({words})")
            return False, notes, texts
    return True, notes, texts


def run_known_answer_check(
    *,
    start: RuntimeConfiguration,
    candidate: RuntimeConfiguration,
    context_length: int,
    measure: Measure,
    time_left: Callable[[], bool] = lambda: True,
    settings_check: SettingsCheck | None = None,
) -> KnownAnswerCheck:
    """Ask every known-answer question that fits the context, with both setups.

    ``measure(config, workload, label)`` makes one measurement from a fresh load (the
    tuner's own measuring, so the same harness and settings). ``settings_check`` (when
    given) must confirm each measurement ran with its configuration's verifiable
    settings. Time running out, unusable measurements or unshown settings leave the
    check incomplete, never passed.
    """
    results: list[QuestionResult] = []
    shown: list[dict[str, object]] = []
    skipped_long = False
    incomplete: list[str] = []
    for question in QUESTIONS:
        if question.context_tokens > context_length:
            if question.long_text:
                skipped_long = True
            results.append(
                QuestionResult(question.question_id, False, 0, 0, 0, 0, 0,
                               [f"needs about {question.context_tokens} pieces of context; not asked"])
            )
            continue
        if not time_left():
            incomplete.append(f"'{question.question_id}' not asked (no time left)")
            results.append(
                QuestionResult(question.question_id, False, 0, 0, 0, 0, 0, ["no time left; not asked"])
            )
            continue
        workload = workload_for(question)
        before = measure(start, workload, f"known-{question.question_id}-current")
        after = measure(candidate, replace(workload), f"known-{question.question_id}-new")
        start_ok, start_notes, start_texts = _usable(
            before, start, "current", settings_check, shown, question.question_id
        )
        new_ok, new_notes, new_texts = _usable(
            after, candidate, "new", settings_check, shown, question.question_id
        )
        start_ok = start_ok and before.invalid_runs == 0
        if before.invalid_runs:
            start_notes.append(f"current setup: {before.invalid_runs} run(s) failed")
        start_verdicts = [judge_answer(question, text) for text in start_texts]
        new_verdicts = [judge_answer(question, text) for text in new_texts]
        start_right = sum(1 for ok, _ in start_verdicts if ok)
        new_right = sum(1 for ok, _ in new_verdicts if ok)
        completed = start_ok and new_ok
        counts = completed and start_right == len(start_texts)
        notes = [*start_notes, *new_notes]
        if completed and not counts:
            first_wrong = next(words for ok, words in start_verdicts if not ok)
            notes.append(
                "the current setup did not answer it right every time "
                f"({first_wrong}), so it says nothing here"
            )
        wrong_new = [words for ok, words in new_verdicts if not ok]
        if wrong_new:
            notes.append(f"new setup: {wrong_new[0]}")
        results.append(
            QuestionResult(
                question.question_id,
                counts,
                start_right,
                len(start_texts),
                new_right,
                len(new_texts),
                after.invalid_runs,
                notes,
                completed=completed,
                asked=True,
            )
        )
        if after.invalid_runs:
            return KnownAnswerCheck(
                FAIL, f"{after.invalid_runs} run(s) of the new setup failed on '{question.question_id}'",
                results, settings_shown=shown,
            )
        if counts and new_right < len(new_texts):
            return KnownAnswerCheck(
                FAIL,
                f"the new setup answered '{question.question_id}' wrongly "
                f"({new_right}/{len(new_texts)} right: {wrong_new[0]}; the current setup "
                f"{start_right}/{len(start_texts)})",
                results, settings_shown=shown,
            )
        if not completed:
            incomplete.append(f"'{question.question_id}': " + "; ".join(start_notes + new_notes))
    if incomplete:
        return KnownAnswerCheck(
            INCOMPLETE,
            "not every question could be completed, so this is not eligible for automatic "
            "acceptance: " + "; ".join(incomplete),
            results, settings_shown=shown,
        )
    informative = [item for item in results if item.informative]
    long_informative = any(
        item.informative and next(q for q in QUESTIONS if q.question_id == item.question_id).long_text
        for item in results
    )
    if len(informative) < MIN_INFORMATIVE:
        return KnownAnswerCheck(
            INSUFFICIENT,
            f"only {len(informative)} question(s) the current setup answers right every time "
            f"(needs {MIN_INFORMATIVE})",
            results, settings_shown=shown,
        )
    if not long_informative and not skipped_long:
        return KnownAnswerCheck(
            INSUFFICIENT, "no long-text question the current setup answers right every time",
            results, settings_shown=shown,
        )
    return KnownAnswerCheck(
        PASS,
        f"every answer of the new setup right on {len(informative)} known-answer question(s)"
        + ("; the long-text questions did not fit the context" if skipped_long else ""),
        results, settings_shown=shown,
    )


__all__ = [
    "FAIL",
    "INCOMPLETE",
    "INSUFFICIENT",
    "KnownAnswerCheck",
    "KnownQuestion",
    "MIN_INFORMATIVE",
    "PASS",
    "POLICY_VERSION",
    "QUESTIONS",
    "REQUIRED_RUNS",
    "answer_fields",
    "answer_right",
    "judge_answer",
    "run_known_answer_check",
    "workload_for",
]
