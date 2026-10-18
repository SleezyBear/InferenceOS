"""Reuse correctness: a development study of whether prompt reuse changes answers (job 128,
prepared 17 Oct 2026 after job 127). Not a tune path.

Job 127 saw Gemma 3 1B answer one question wrongly when the engine reused most, but not
all, of an earlier prompt ("The material does not contain the answer." instead of "57
degrees"), in all 4 repeats of that one prompt pair. This module asks, with known answers:

* does it happen for other prompt pairs that share a long start and part near the end?
* does a full-size sliding-window cache (``--swa-full``) change it?
* does a model without sliding-window layers (the control) show it?
* does an ordinary chat client, which leaves ``cache_prompt`` at the engine's default
  (reuse on), get the wrong answer (the exposure of ``start`` users)?

**Pairs** (``build_pairs``): the exact job 127 pair (a regression) plus 8 more, each a
document, an earlier question (the priming request) and a measured question whose answer
is a number stated once in the document. Prompts use job 127's format
(``...\\nQuestion: ...\\nAnswer:``).

**Judging an answer** (``judge``, pre-registered grammar): three verdicts, ``right``,
``wrong`` (an established wrong fact) and ``unscorable`` (the reply does not state one
clear value), from the reply's answer line only. See ``judge`` for the exact rules (signs,
fractions, ranges, alternatives, number roles, negation, spelled-out numbers, a lead-in
line). Exact wording is not judged.

**Trials** are the prompt-reuse trials (``prompt_reuse.run_trial``): a fresh load, the
priming request, the measured request with reuse requested (reuse arm) or not (fresh
arm), stop; with the same stops (cleanup, validity gates E1/E4, power, interruption).

**Chat exposure** (``chat_trial``): per pair and block, three fresh loads, the first two in
alternating order: the **client** (priming question, then the measured question, both
with ``cache_prompt`` left out: the engine default, reuse on), the **matched reference**
(the same priming question the same way, then the measured question with ``cache_prompt``
false), and the **unprimed reference** (the measured question alone, ``cache_prompt``
false; an extra control). Reuse is read from the reply's ``usage`` (b11139:
``prompt_tokens``, ``prompt_tokens_details.cached_tokens``), checked against ``timings``
when present. Every request is checked as soon as it returns (E1/E4) and power at every
load boundary; a stop ends all further work.

**What counts** is pre-registered in ``reviewer-job128-prereg-2026-10-17.md``; this
module implements its rules (``correctness_summary``, ``chat_summary``). Nothing here
writes saved setups or tune history, changes the context size, or uses tune's acceptance
rules. Results are development evidence only.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from functools import cached_property
from typing import Any, Protocol

from inferenceos.tune.prompt_reuse import (
    _INSTRUCTIONS,
    _MANUAL,
    FRESH,
    INTERRUPTED,
    MEASURED_PIECES,
    PRIMING_PIECES,
    REUSE,
    SEED,
    STOPPED_CLEANUP,
    STOPPED_POWER,
    STOPPED_VALIDITY,
    Cleanup,
    EngineVariant,
    ReuseCase,
    ReuseEngine,
    TrialResult,
    _count,
    _document,
    _power,
    _power_allowed,
    _sha,
    pair_problem,
    prompt_counts,
    validity_breach,
    work_evidence,
)

CORRECTNESS_VERSION = "reuse-correctness-8"
#: Repeats per pair and variant (``/completion`` trials), and per pair (chat exposure).
BLOCKS = 3
CHAT_BLOCKS = 2
CHAT_MAX_TOKENS = MEASURED_PIECES
DEFAULT = EngineVariant("default")
SWA_FULL = EngineVariant("swa-full", swa_full=True)

_NOTES = (
    "Note {n}. Orchard row {n} was pruned in week {t}. Apple trees in this row bloom "
    "about {u} days after the last frost. Bees were moved in on day {d}, and the soil "
    "moisture reading was {p} percent at noon. The next visit is planned for week {t}. "
)


def _facts(n: int) -> dict[str, int]:
    """The numbers ``_document`` writes for section ``n`` (the same formulas)."""
    return {"t": 40 + n % 17, "u": 50 + n % 23, "d": 10 + n % 29, "p": 20 + n % 31}


@dataclass(frozen=True)
class KnownAnswerPair:
    """A priming question and a measured question about one document, with the measured
    question's known answer."""

    pair_id: str
    document: str
    question_prime: str
    question_measured: str
    #: The number the measured answer must state.
    expected: str
    #: Where the fact is (for the reader of the results).
    source: str
    #: The exact job 127 pair (kept as a regression).
    regression: bool = False

    @cached_property
    def case(self) -> ReuseCase:
        return ReuseCase(
            f"correctness/{self.pair_id}", f"known answer {self.expected} ({self.source})",
            self.document + f"\nQuestion: {self.question_prime}\nAnswer:",
            self.document + f"\nQuestion: {self.question_measured}\nAnswer:",
            expect_reuse=True,
        )

    @cached_property
    def case_sha256(self) -> str:
        return self.case.sha256

    @property
    def identifiers(self) -> set[str]:
        """The question's own identifying numbers (its unit or row number): a right answer
        may name them as identifiers (``unit 7``), never as the value."""
        return set(re.findall(r"\d+", self.question_measured))

    def chat_messages(self, question: str) -> list[dict[str, str]]:
        """How an ordinary chat client would ask: the document as the system message."""
        return [{"role": "system", "content": self.document}, {"role": "user", "content": question}]


def build_pairs(sections: int = 12) -> tuple[KnownAnswerPair, ...]:
    """The fixed pairs. The regression pair is job 127's ``shared_prefix`` case exactly
    (``prompt_reuse.build_cases(12)``); every expected number occurs once in its document
    for its field (checked by the tests)."""
    if sections < 12:
        raise ValueError("the fixed known-answer pairs require at least 12 sections")
    manual = _INSTRUCTIONS + _document(_MANUAL, sections)
    notes = _INSTRUCTIONS + _document(_NOTES, sections)

    def m(n: int, key: str) -> str:
        return str(_facts(n)[key])

    return (
        KnownAnswerPair("regression-pump-b-unit-7", manual, "how often are the filters of unit 3 changed?",
                        "when does pump B of unit 7 start?", m(7, "u"), "manual, section 7", regression=True),
        KnownAnswerPair("manual-pressure-unit-5", manual, "how often are the filters of unit 3 changed?",
                        "what pressure drop means a filter of unit 5 is blocked, in kilopascals?", m(5, "p"),
                        "manual, section 5"),
        KnownAnswerPair("manual-water-unit-9", manual, "when does pump B of unit 7 start?",
                        "at what temperature in degrees does the cooling loop of unit 9 carry water?", m(9, "t"),
                        "manual, section 9"),
        KnownAnswerPair("manual-filters-unit-10", manual, "what pressure drop means a filter of unit 2 is blocked?",
                        "every how many days are the filters of unit 10 changed?", m(10, "d"), "manual, section 10"),
        KnownAnswerPair("manual-pump-unit-12", manual, "at what temperature does the cooling loop of unit 4 carry "
                        "water?", "above what inlet temperature in degrees does pump B of unit 12 start?",
                        m(12, "u"), "manual, section 12"),
        KnownAnswerPair("notes-bees-row-6", notes, "in which week was orchard row 2 pruned?",
                        "on which day were bees moved in for orchard row 6?", m(6, "d"), "notes, note 6"),
        KnownAnswerPair("notes-moisture-row-8", notes, "on which day were bees moved in for orchard row 4?",
                        "what was the soil moisture reading in orchard row 8, in percent?", m(8, "p"), "notes, note 8"),
        KnownAnswerPair("notes-bloom-row-10", notes, "what was the soil moisture reading in orchard row 1?",
                        "about how many days after the last frost do the apple trees in orchard row 10 bloom?",
                        m(10, "u"), "notes, note 10"),
        KnownAnswerPair("notes-pruned-row-5", notes, "how many days after the last frost do trees in row 3 bloom?",
                        "in which week was orchard row 5 pruned?", m(5, "t"), "notes, note 5"),
    )




# --------------------------------------------------------------------------- #
# The judge (pre-registered grammar)                                          #
# --------------------------------------------------------------------------- #
RIGHT, WRONG, UNSCORABLE = "right", "wrong", "unscorable"

_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
          "sixteen seventeen eighteen nineteen").split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_SPELLED = re.compile(r"\b(" + "|".join(_TENS) + r")(?:[- ](" + "|".join(_UNITS[1:10]) + r"))?\b|\b("
                      + "|".join(_UNITS) + r")\b", re.IGNORECASE)
#: The only number forms the judge reads (pre-registration 3.1, rule 3). Each word that
#: contains a numeric character, once outer punctuation is removed, must be exactly one of these:
#: an optional attached sign (+, -, minus sign, en dash), a whole number without leading
#: zeros, then optionally a decimal part or a fraction (denominator from 1), then
#: optionally an attached unit from a fixed list. Anything else with a numeric character is an
#: unsupported form, and the reply is ``unscorable``.
_NUMBER = re.compile(r"(?P<sign>[-+\u2212\u2013])?(?P<num>0|[1-9]\d*)"
                     r"(?:(?P<dec>\.\d+)|/(?P<den>[1-9]\d*))?"
                     r"(?P<unit>\u00b0[CF]?|%|kPa|th|st|nd|rd)?", re.IGNORECASE | re.ASCII)
_NEGATIVE_SIGNS = ("-", "\u2212", "\u2013")
_STRIP_LEAD = "([{\"'*_`"
_STRIP_TRAIL = ")]}\"'.,;:!?*_`"
_BRACKETS = "()[]{}\"'`"
#: Symbol-only neighbouring tokens are unsupported, except these standalone units.
_STANDALONE_UNITS = frozenset({"%", "\u00b0", "\u00b0C", "\u00b0F"})
_PLAIN_PUNCTUATION = ".,;:!?"
#: A word right after a number that changes its size: unsupported, ``unscorable``. So is
#: an exponent word ("57 e2", "57 E") and any single letter ("57 M", "57 k", "57 m").
_MAGNITUDE = frozenset({"hundred", "hundreds", "thousand", "thousands", "million", "millions", "billion",
                        "billions", "trillion", "trillions", "lakh", "lakhs", "crore", "crores", "dozen", "dozens", "bn", "mn"})
_EXPONENT = re.compile(r"[eE][-+\u2212]?\d*")
_IDENTIFIER_WORDS = frozenset({"unit", "units", "row", "rows", "section", "sections", "note", "notes"})
#: Read sign words beside normalized numeric words, after removing outer formatting.
_SIGN_WORDS = frozenset({"minus", "negative", "plus", "positive"})
#: Arithmetic between two numbers ("57 + 3", "57 times 2", "57 minus 3") states no single value.
_ARITHMETIC = re.compile(r"\d\s*(?:[+*\u00d7]|\b(?:plus|minus|times)\b)\s*[-+\u2212]?\d", re.IGNORECASE)
_RANGE = re.compile(r"\d\s*(?:-|\u2013|\u2014|\bto\b|\band\b)\s*[-\u2212]?\d")
_NEGATION = re.compile(r"\b(?:not|no|never|neither|nor|without|cannot)\b|n't\b", re.IGNORECASE)
_REFUSAL = re.compile(
    r"\b(?:does|do|did)\s*(?:not|n't)\s+(?:contain|say|state|mention|give|provide|include|specify)"
    r"|\bno (?:information|answer|mention)\b|\bnot (?:mentioned|stated|specified|provided|given|available|known)\b"
    r"|\bcannot (?:answer|be determined|determine|tell)\b|\bcan't (?:answer|tell|determine)\b"
    r"|\bunknown\b|\bnot enough information\b", re.IGNORECASE)
_LABEL = re.compile(r"^[\s>*_#`-]*(?:final\s+)?answer[\s*_`]*:[\s*_`]*", re.IGNORECASE)


def _spelled_to_digits(line: str) -> str:
    def number(match: re.Match[str]) -> str:
        if match.group(1):
            value = _TENS.get(match.group(1).lower())
            if value is None or (match.group(2) and match.group(2).lower() not in _UNITS):
                return match.group(0)
            if match.group(2):
                value += _UNITS.index(match.group(2).lower())
            return str(value)
        word = match.group(3).lower()
        return str(_UNITS.index(word)) if word in _UNITS else match.group(0)

    return _SPELLED.sub(number, line)


def _unsupported_spelling(line: str) -> bool:
    """Unicode regex case equivalences must not expand the listed English vocabulary."""
    return any(_spelled_to_digits(m.group(0)) == m.group(0) for m in _SPELLED.finditer(line))


def _has_numeric(text: str) -> bool:
    """Find all Unicode numeric characters; only ASCII forms pass ``_NUMBER``.

    The regex decimal-digit class misses superscripts, fractions and other numeric glyphs.
    Discovery is deliberately broader than the accepted number grammar.
    """
    return any(char.isnumeric() for char in text)


def _unsafe_format(text: str) -> bool:
    return any(unicodedata.category(c) in {"Cf", "Cc"} and not c.isspace() for c in text)


def _detached_symbol(token: str) -> bool:
    """Discover arbitrary standalone symbols, not a blacklist of operator glyphs.

    Empty sentence punctuation is ignored. The letter x is also an existing arithmetic
    operator; symbolic units are allowed only in their explicitly listed forms.
    """
    def wrapper(char: str) -> bool:
        return char in _BRACKETS + _PLAIN_PUNCTUATION or unicodedata.category(char) in {"Ps", "Pe", "Pi", "Pf"}

    core = token
    while core and wrapper(core[0]):
        core = core[1:]
    while core and wrapper(core[-1]):
        core = core[:-1]
    return core.lower() == "x" or bool(core and core not in _STANDALONE_UNITS
                                      and not any(c.isalpha() or c.isnumeric() for c in core))


def _sign_token(token: str) -> tuple[str, bool]:
    """Recognize a sign word even when punctuation makes applying it ambiguous."""
    core = token
    while core and (core[0] in _STRIP_LEAD or unicodedata.category(core[0]) in {"Ps", "Pi"}):
        core = core[1:]
    opened = core
    while core and not core[-1].isalpha():
        core = core[:-1]
    sign = core.lower()
    return (sign, core != opened) if sign in _SIGN_WORDS else ("", False)


def answer_line(text: str | None) -> str | None:
    """The text judged: the first sentence of the reply's first non-empty line, after an
    ``Answer:`` label is removed; one lead-in line (ending with ":" and stating no
    number) is skipped. A sentence ends at ".", "!" or "?" followed by a space."""
    if not text:
        return None
    lines = [_LABEL.sub("", line).strip().strip("*_`\"'").strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if lines and lines[0].endswith(":") and len(lines) > 1:
        lead = _normalise_answer(lines[0])
        # A sign immediately at the lead-in boundary is ambiguous. Ordinary words such
        # as "A positive example follows:" are not a numeric sign at that boundary.
        sign_at_end = bool(lead.split() and _sign_token(lead.split()[-1])[0])
        if not _has_numeric(lead) and not _unsupported_spelling(lead) and not _unsafe_format(lead) and not sign_at_end:
            lines = lines[1:]
    if not lines:
        return None
    return re.split(r"(?<=[.!?])\s+", lines[0], maxsplit=1)[0].strip() or None


def _value(match: re.Match[str]) -> Fraction:
    """A number word's exact value: "57", "57.0", "+57" and "114/2" are all 57."""
    value = Fraction(match["num"] + (match["dec"] or ""))
    if match["den"]:
        value /= int(match["den"])
    return -value if match["sign"] in _NEGATIVE_SIGNS else value


def _unformat_word(word: str) -> str:
    """Remove outer formatting without erasing an operator or a punctuation boundary.

    Keeping the colon in ``**7**:`` preserves ``Unit 7: 57``; keeping a standalone
    ``*`` preserves the arithmetic guard instead of treating it as empty markdown.
    """
    operator = word.strip(_BRACKETS + ".,;:!?")
    formatted_unit = operator.strip("*_`")
    if formatted_unit in _STANDALONE_UNITS:
        return formatted_unit
    if not operator:
        # Preserve boundary-only tokens: erasing a separate ')' would apply a minus
        # across it, while ignoring a separate comma would drop the minus entirely.
        return word
    if _detached_symbol(operator):
        return operator
    leading = word.lstrip(_STRIP_LEAD)
    core = leading.rstrip(_STRIP_TRAIL)
    sign, _ = _sign_token(core)
    if sign:
        # Remove markdown, but retain punctuation and closing delimiters after a sign.
        # In particular, (negative) must not silently become a valid minus word.
        return leading.rstrip("*_`").replace("**", "").replace("__", "")
    punctuation = "".join(c for c in leading[len(core):] if c in ".,;:!?")
    return core + punctuation


def _normalise_answer(text: str) -> str:
    """Use the same formatting/spelling rules for lead-in detection and value scoring."""
    return _spelled_to_digits(" ".join(_unformat_word(word) for word in text.split()))


def _neighbour(raw: list[str], index: int, direction: int) -> tuple[str, bool, bool, int]:
    """Nearest non-delimiter token, whether a sign boundary was crossed, and a colon.

    Open brackets before a value can format the number. Closing brackets or sentence
    punctuation between a sign and value are ambiguous even when written separately.
    """
    boundary = colon = False
    index += direction
    while 0 <= index < len(raw):
        token = raw[index]
        if not all(c in _BRACKETS + _PLAIN_PUNCTUATION or unicodedata.category(c) in {"Ps", "Pe", "Pi", "Pf"}
                   for c in token):
            return token, boundary, colon, index
        colon |= ":" in token
        boundary |= any(c in ")]}\"'.,;:!?`" or unicodedata.category(c) in {"Pe", "Pf"} for c in token)
        index += direction
    return "", boundary, colon, index


def judge(text: str | None, pair: KnownAnswerPair) -> tuple[str, str]:
    """``(verdict, reason)`` for one reply to the measured question, from its answer
    sentence (``answer_line``) only. Pre-registered rules, in order:

    1. no answer line: ``unscorable``;
    2. outer formatting is removed first, retaining punctuation boundaries and standalone
       operators; spelled-out whole numbers up to 99 are read as digits ("fifty-seven" is 57); a sign
       word right before a number gives it that sign ("minus 57", "negative fifty-seven"
       are -57; "plus 57", "positive fifty-seven" are 57); arithmetic between numbers
       ("57 + 3", "57 minus 3", "57 times 2"): ``unscorable``. A sign word plus an attached
       sign, repeated sign words or a trailing sign word is also ``unscorable``.
       Punctuation or a closing delimiter after a sign word makes it ambiguous and
       ``unscorable``; the sign is never silently applied across that boundary;
    3. **accepted number forms only.** Every word containing a numeric character, after outer
       punctuation is removed, must be exactly one ``_NUMBER`` form: an optional attached
       sign (``+``, ``-``, minus sign, en dash), a whole number without leading zeros, an
       optional decimal part or fraction, and an optional attached unit (degree sign,
       degrees C or F, ``%``, ``kPa``, ``th``, ``st``, ``nd``, ``rd``). Its value is exact:
       "57", "57.0", "+57", "57/1" and "114/2" are 57; "-57", "57.5", "57/7" and "1057"
       are not. Anything else with a digit ("57e2", "57k", "57,000", "057", "57-60",
       "57-degree", "unit7", non-ASCII digits) is an unsupported form: ``unscorable``. So
       is a standalone sign or operator next to a number ("- 57", "(- 57)", "57 x 2"),
       two numbers written apart with nothing between ("5 7", "57 000"), and a number
       followed by a magnitude word ("hundred", "thousand", "million"...), an exponent
       word ("e2", "E") or any single letter ("57 M", "57 k", "57 m"). Numeric glyphs
       outside the ASCII grammar (including separate superscripts and fractions) are
       discovered too and rejected, rather than silently ignored. Neighbouring symbol-only
       tokens (no letter or numeric character, after outer brackets/quotes and plain
       punctuation) are unscorable except standalone units %, degree sign, degrees C/F;
    4. a number right after ``unit``, ``row``, ``section`` or ``note`` is an
       **identifier**: it must be a plain whole number among the question's own (else
       ``unscorable``) and is never the value. Its explicit trailing colon separates it
       from a following signed value ("Unit 7: negative 57"); every other number is a **value**;
    5. a refusal ("does not contain the answer", "not stated", "unknown") with no value:
       ``wrong`` (the document states the answer);
    6. no value: ``unscorable``; a range ("57 to 60", "between 57 and 60") involving a
       value: ``unscorable``;
    7. the expected value is the only value: ``right``, unless the line has a negation
       ("not", "no", "never", "n't"...): then ``unscorable``;
    8. the expected value and other values: ``unscorable`` (several values or
       alternatives);
    9. only other values: ``wrong`` (a different fact), unless negated or with a refusal:
       ``unscorable``."""
    line = answer_line(text)
    if line is None:
        return UNSCORABLE, "no answer line"
    if _unsafe_format(line):
        return UNSCORABLE, "unsupported invisible format or control character in the answer"
    line = _normalise_answer(line)
    if _unsupported_spelling(line):
        return UNSCORABLE, "unsupported spelling outside the listed English number forms"
    if _ARITHMETIC.search(line):
        return UNSCORABLE, f"states arithmetic, not a value: {line[:80]!r}"
    raw = line.split()
    words = [w.lstrip(_STRIP_LEAD).rstrip(_STRIP_TRAIL) for w in raw]
    values: list[Fraction] = []
    labels: list[str] = []
    for index, word in enumerate(words):
        if not _has_numeric(word):
            continue
        match = _NUMBER.fullmatch(word)
        if match is None:
            return UNSCORABLE, f"an unsupported number form ({raw[index]!r}): {line[:80]!r}"
        before, before_boundary, _, before_position = _neighbour(raw, index, -1)
        after_context, _, after_colon, _ = _neighbour(raw, index, 1)
        after_raw = raw[index + 1] if index + 1 < len(raw) else ""
        after = after_context.lstrip(_STRIP_LEAD).rstrip(_STRIP_TRAIL)
        identifier = index > 0 and words[index - 1].lower() in _IDENTIFIER_WORDS
        if identifier:
            plain = not (match["sign"] or match["dec"] or match["den"] or match["unit"])
            if not plain or match["num"] not in pair.identifiers:
                return UNSCORABLE, f"names another identifier ({raw[index]!r}): {line[:80]!r}"
        if _detached_symbol(before) or _detached_symbol(after_context):
            return UNSCORABLE, f"a standalone sign or operator next to {raw[index]!r}: {line[:80]!r}"
        if raw[index] == word and re.match(r"[-+\u2212\u2013]?\d", after_raw):
            return UNSCORABLE, f"numbers written apart ({raw[index]!r} {after_raw!r}): {line[:80]!r}"
        if after.lower() in _MAGNITUDE or _EXPONENT.fullmatch(after) or (len(after) == 1 and after.isalpha()):
            return UNSCORABLE, f"an unsupported number form ({raw[index]!r} {after_raw!r}): {line[:80]!r}"
        sign_word, sign_boundary = _sign_token(before)
        after_sign, _ = _sign_token(after_context)
        if after_sign and not (identifier and (raw[index].endswith(":") or after_colon)):
            return UNSCORABLE, f"a trailing sign word next to {raw[index]!r}: {line[:80]!r}"
        if sign_word in _SIGN_WORDS:
            previous = _neighbour(raw, before_position, -1)[0] if before_position >= 0 else ""
            previous_sign, _ = _sign_token(previous)
            if before_boundary or sign_boundary or match["sign"] or previous_sign or _detached_symbol(previous):
                return UNSCORABLE, f"ambiguous or multiple signs next to {raw[index]!r}: {line[:80]!r}"
        if identifier:
            continue
        negative_word = sign_word in ("minus", "negative")
        try:
            value = _value(match)
        except (ValueError, OverflowError):
            return UNSCORABLE, "number exceeds the supported exact parser size"
        values.append(-value if negative_word else value)
        labels.append(("-" if negative_word else "") + word)
    refusal = bool(_REFUSAL.search(line))
    negated = bool(_NEGATION.search(line))
    if refusal and not values:
        return WRONG, f"a refusal; the document states {pair.expected}: {line[:80]!r}"
    if not values:
        return UNSCORABLE, f"states no value: {line[:80]!r}"
    if _RANGE.search(line):
        return UNSCORABLE, f"states a range: {line[:80]!r}"
    if len(set(values)) > 1:
        return UNSCORABLE, f"states several different values: {line[:80]!r}"
    target = Fraction(int(pair.expected))
    expected = [v for v in values if v == target]
    others = [label for v, label in zip(values, labels) if v != target]
    shown = ", ".join(labels)
    if expected and not others:
        if negated or refusal:
            return UNSCORABLE, f"the value is negated or qualified: {line[:80]!r}"
        return RIGHT, "right"
    if expected:
        return UNSCORABLE, f"states several values ({shown}): {line[:80]!r}"
    if negated or refusal:
        return UNSCORABLE, f"a negated other value ({', '.join(others)}): {line[:80]!r}"
    return WRONG, f"states {', '.join(others)}; the document states {pair.expected}: {line[:80]!r}"


# --------------------------------------------------------------------------- #
# /completion trials: the correctness verdict per block                         #
# --------------------------------------------------------------------------- #
def correctness_block(fresh: TrialResult, reuse: TrialResult, pair: KnownAnswerPair) -> dict[str, Any]:
    """One fresh/reuse pair judged for correctness. **Usable** needs: no trial problem,
    mains power throughout, complete work evidence for both arms' priming and measured
    readings, and the reuse arm answering in its priming slot. Unlike the speed study,
    differing answers and missing reuse do not make a block unusable here: they are
    what is measured."""
    reasons: list[str] = []
    for trial in (fresh, reuse):
        reasons.extend(f"{trial.arm}: {p}" for p in trial.problems)
        if trial.interrupted or trial.breach or validity_breach(trial):
            reasons.append(f"{trial.arm}: interrupted or a validity gate fired")
        if (trial.cleanup.get("process_gone") is not True
                or trial.cleanup.get("credential_removed") is False or trial.cleanup.get("problem")):
            reasons.append(f"{trial.arm}: cleanup not confirmed clean")
        if len(trial.power) != 2:
            reasons.append(f"{trial.arm}: before/after power evidence missing")
        identity = trial.identity
        if (identity.get("case_sha256") != pair.case_sha256 or identity.get("case_id") != trial.case_id
                or identity.get("arm") != trial.arm or identity.get("variant_id") != trial.variant_id
                or identity.get("cache_prompt") is not (trial.arm == REUSE)
                or _sha(identity) != trial.identity_sha256):
            reasons.append(f"{trial.arm}: request identity does not match the known-answer case")
        if any(not _power_allowed(p, unchecked_ok=True) for p in trial.power):
            reasons.append(f"{trial.arm}: not on mains power throughout")
        for label, reading, total, pieces in (("priming", trial.prime, trial.prime_tokens, PRIMING_PIECES),
                                              ("measured", trial.measured, trial.measured_tokens, MEASURED_PIECES)):
            reasons.extend(f"{trial.arm} {label}: {p}" for p in work_evidence(
                reading, expected_total=total, expected_generated=pieces)[1])
            cached, _ = prompt_counts(reading, expected_total=total)
            if cached is not None and ((label == "priming" and cached > 0)
                                       or (label == "measured" and trial.arm == FRESH and cached > 0)
                                       or (label == "measured" and cached > trial.bound)):
                reasons.append(f"{trial.arm} {label}: reuse violates the fresh/bound conditions")
    compared = ("evaluation", "case_sha256", "variant_id", "context_length", "configuration_sha256",
                "server_cache_reuse", "server_swa_full")
    if (fresh.prime_tokens != reuse.prime_tokens or fresh.measured_tokens != reuse.measured_tokens
            or any(fresh.identity.get(k) != reuse.identity.get(k) for k in compared)):
        reasons.append("the two arms did not measure the same configuration and workload")
    p_slot, m_slot = _count(reuse.prime.get("slot")), _count(reuse.measured.get("slot"))
    if p_slot is None or m_slot is None or p_slot != m_slot:
        reasons.append(f"reuse arm not in its priming slot ({reuse.prime.get('slot')!r} -> "
                       f"{reuse.measured.get('slot')!r})")
    cached = _count(reuse.measured.get("cached"))
    total = reuse.measured_tokens
    fresh_verdict, fresh_why = judge(fresh.measured.get("text"), pair)
    reuse_verdict, reuse_why = judge(reuse.measured.get("text"), pair)
    return {
        "block": fresh.block, "usable": not reasons, "reasons": list(dict.fromkeys(reasons)),
        "fresh_verdict": fresh_verdict, "fresh_judgement": fresh_why,
        "reuse_verdict": reuse_verdict, "reuse_judgement": reuse_why,
        "fresh_text": fresh.measured.get("text"), "reuse_text": reuse.measured.get("text"),
        "answers_identical": fresh.measured.get("text") == reuse.measured.get("text"),
        "reused": cached, "measured_tokens": total,
        #: Some but not all of the prompt reused: the job 127 condition.
        "partial_reuse": cached is not None and 0 < cached < total - 1,
    }


def pair_outcome(rows: Sequence[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    """The pre-registered outcome of one pair under one variant, and its counts.

    **Informative** needs at least 2 usable blocks with the fresh arm ``right`` in every
    one. Then, from the usable blocks, in this order: ``wrong with partial reuse`` (the
    reuse arm ``wrong`` in a block with partial reuse), ``wrong without partial reuse``
    (``wrong`` only in blocks without it), ``unscorable with reuse``, ``no partial reuse
    happened``, ``changed with reuse, still right`` (every reuse answer ``right``, some
    text different), ``right and identical with reuse``."""
    usable = [r for r in rows if r.get("usable")]
    counts = {
        "usable_blocks": len(usable),
        "partial_reuse_blocks": sum(1 for r in usable if r["partial_reuse"]),
        "reuse_wrong_with_partial": sum(1 for r in usable if r["partial_reuse"] and r["reuse_verdict"] == WRONG),
        "reuse_wrong_without_partial": sum(1 for r in usable
                                           if not r["partial_reuse"] and r["reuse_verdict"] == WRONG),
        "reuse_unscorable": sum(1 for r in usable if r["reuse_verdict"] == UNSCORABLE),
        "fresh_not_right": sum(1 for r in usable if r["fresh_verdict"] != RIGHT),
    }
    if len(usable) < 2:
        return "not informative: fewer than 2 usable blocks", counts
    if counts["fresh_not_right"]:
        return "not informative: the fresh arm is not right in every usable block", counts
    if counts["reuse_wrong_with_partial"]:
        return "wrong with partial reuse", counts
    if counts["reuse_wrong_without_partial"]:
        return "wrong without partial reuse", counts
    if counts["reuse_unscorable"]:
        return "unscorable with reuse", counts
    if not counts["partial_reuse_blocks"]:
        return "no partial reuse happened", counts
    if any(not r["answers_identical"] for r in usable):
        return "changed with reuse, still right", counts
    return "right and identical with reuse", counts


#: What a pair's usable partial-reuse blocks establish (``partial_reuse_evidence``).
EVIDENCE_WRONG, EVIDENCE_RIGHT, EVIDENCE_INCONCLUSIVE = "wrong", "right", "inconclusive"
EVIDENCE_NONE, EVIDENCE_NOT_INFORMATIVE = "no partial reuse", "not informative"


def partial_reuse_evidence(rows: Sequence[dict[str, Any]], outcome: str) -> str:
    """What the job 127 condition (partial reuse) established for one pair and variant,
    from its usable partial-reuse blocks only, kept apart from the outcome's other
    answers:

    * ``not informative``: the pair is not informative (``pair_outcome``);
    * ``wrong``: a reuse answer is **wrong** in some partial-reuse block (an observed
      failure, whatever the other blocks say);
    * ``no partial reuse``: no usable block reused part of the prompt;
    * ``inconclusive``: no wrong answer, but some partial-reuse answer is
      **unscorable**, so the absence of the failure is not established;
    * ``right``: every partial-reuse answer is **right** (at least one block)."""
    if outcome.startswith("not informative"):
        return EVIDENCE_NOT_INFORMATIVE
    partial = [r for r in rows if r.get("usable") and r["partial_reuse"]]
    if any(r["reuse_verdict"] == WRONG for r in partial):
        return EVIDENCE_WRONG
    if not partial:
        return EVIDENCE_NONE
    if any(r["reuse_verdict"] != RIGHT for r in partial):
        return EVIDENCE_INCONCLUSIVE
    return EVIDENCE_RIGHT


def correctness_summary(trials: Sequence[TrialResult], pairs: Sequence[KnownAnswerPair]) -> dict[str, Any]:
    """Per variant and pair: the blocks and the pre-registered outcome (``pair_outcome``);
    per variant, the sliding-window states of **every recorded load** (whatever its
    block's usability); and the default / ``--swa-full`` comparison."""
    by_case = {pair.case.case_id: pair for pair in pairs}
    groups: dict[tuple[str, str], dict[int, list[TrialResult]]] = {}
    load_states: dict[str, list[str]] = {}
    for trial in trials:
        if trial.case_id in by_case:
            groups.setdefault((trial.variant_id, trial.case_id), {}).setdefault(trial.block, []).append(trial)
            load_states.setdefault(trial.variant_id, []).append(str((trial.swa_cache or {}).get("state")))
    out: dict[str, Any] = {"version": CORRECTNESS_VERSION, "pairs": {}, "variants": {}}
    critical = [t for t in trials if t.interrupted or validity_breach(t)
                or t.cleanup.get("process_gone") is not True or t.cleanup.get("credential_removed") is False
                or t.cleanup.get("problem") or any(not _power_allowed(p, unchecked_ok=True) for p in t.power)]
    if critical:
        out["problems"] = ["a recorded trial stopped on validity, cleanup, power or interruption"]
    if any(t.case_id not in by_case for t in trials):
        out.setdefault("problems", []).append("a recorded trial has an unknown known-answer case")
    for (variant_id, case_id), blocks in sorted(groups.items()):
        pair = by_case[case_id]
        rows: list[dict[str, Any]] = []
        # Repeating blocks and changing only SWA are meaningful comparisons only
        # while the remaining request identity stays fixed. Do not pool setups.
        compared = ("evaluation", "case_sha256", "context_length", "configuration_sha256", "server_cache_reuse")
        identities = [{key: trial.identity.get(key) for key in compared}
                      for loads in blocks.values() for trial in loads]
        comparison_identity = identities[0]
        mixed_identity = any(identity != comparison_identity for identity in identities)
        for block, loads in sorted(blocks.items()):
            problem = pair_problem([t.arm for t in loads])
            if problem:
                rows.append({"block": block, "usable": False, "reasons": [f"not exactly one load per arm: {problem}"]})
                continue
            by_arm = {t.arm: t for t in loads}
            row = correctness_block(by_arm[FRESH], by_arm[REUSE], pair)
            if mixed_identity:
                row["usable"] = False
                row["reasons"].append("request identity changed between blocks")
            rows.append(row)
        outcome, counts = pair_outcome(rows)
        out["pairs"][f"{variant_id}/{pair.pair_id}"] = {
            "variant": variant_id, "pair": pair.pair_id, "expected": pair.expected, "regression": pair.regression,
            "blocks": rows, **counts, "informative": not outcome.startswith("not informative"), "outcome": outcome,
            "partial_reuse_evidence": partial_reuse_evidence(rows, outcome),
            "comparison_identity": comparison_identity,
        }
    for variant_id in sorted(load_states):
        items = [v for v in out["pairs"].values() if v["variant"] == variant_id]
        outcomes: dict[str, int] = {}
        for item in items:
            outcomes[item["outcome"]] = outcomes.get(item["outcome"], 0) + 1
        evidence: dict[str, list[str]] = {}
        for item in items:
            evidence.setdefault(item["partial_reuse_evidence"], []).append(item["pair"])
        states = load_states[variant_id]
        out["variants"][variant_id] = {"pairs": len(items), "outcomes": outcomes,
                                       "partial_reuse_evidence": evidence, "loads": len(states),
                                       "swa_cache_states": {s: states.count(s) for s in sorted(set(states))}}
    out["swa_full_comparison"] = swa_full_comparison(out)
    return out


def swa_full_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    """The default / ``--swa-full`` comparison, pair by pair, from each variant's
    ``partial_reuse_evidence``.

    **Logs:** every recorded load of the default variant must show a ``reduced``
    sliding-window cache and every load of ``swa-full`` a ``full`` one, from its own log.

    **Conclusive pair:** the evidence is ``wrong`` or ``right`` under **both** variants.
    Classified as ``wrong only at the default``, ``wrong under both``, ``wrong only with
    swa-full`` or ``wrong under neither``.

    **Inconclusive pair:** partial reuse happened under both variants, but at least one
    side is ``inconclusive`` (unscorable answers). An observed failure is still reported
    (``wrong at the default; swa-full inconclusive``, ``wrong with swa-full; default
    inconclusive``); otherwise ``inconclusive``. No removal or absence is claimed.

    **Not comparable:** not run, not informative, or no partial reuse under a variant.

    **Interpretable** needs verified logs and at least one conclusive pair; only
    conclusive pairs support a G3 claim."""
    variants = summary.get("variants", {})
    if DEFAULT.variant_id not in variants or SWA_FULL.variant_id not in variants:
        return {"interpretable": False, "reason": "both the default and the swa-full variant are needed"}
    default_states = variants[DEFAULT.variant_id]["swa_cache_states"]
    full_states = variants[SWA_FULL.variant_id]["swa_cache_states"]
    logs_ok = set(default_states) == {"reduced"} and set(full_states) == {"full"}
    pairs = summary["pairs"]
    conclusive: dict[str, str] = {}
    inconclusive: dict[str, str] = {}
    not_comparable: dict[str, str] = {}
    scored = (EVIDENCE_WRONG, EVIDENCE_RIGHT)
    for pair_id in sorted({v["pair"] for v in pairs.values()}):
        a = pairs.get(f"{DEFAULT.variant_id}/{pair_id}")
        b = pairs.get(f"{SWA_FULL.variant_id}/{pair_id}")
        problems = []
        for label, item in (("default", a), ("swa-full", b)):
            if item is None:
                problems.append(f"{label}: not run")
            elif item["partial_reuse_evidence"] == EVIDENCE_NOT_INFORMATIVE:
                problems.append(f"{label}: {item['outcome']}")
            elif item["partial_reuse_evidence"] == EVIDENCE_NONE:
                problems.append(f"{label}: no partial reuse happened")
        if problems:
            not_comparable[pair_id] = "; ".join(problems)
            continue
        assert a is not None and b is not None
        if not a.get("comparison_identity") or a["comparison_identity"] != b.get("comparison_identity"):
            not_comparable[pair_id] = "request identity missing or changed between variants"
            continue
        ea, eb = a["partial_reuse_evidence"], b["partial_reuse_evidence"]
        if ea in scored and eb in scored:
            conclusive[pair_id] = {(EVIDENCE_WRONG, EVIDENCE_RIGHT): "wrong only at the default",
                                   (EVIDENCE_WRONG, EVIDENCE_WRONG): "wrong under both",
                                   (EVIDENCE_RIGHT, EVIDENCE_WRONG): "wrong only with swa-full",
                                   (EVIDENCE_RIGHT, EVIDENCE_RIGHT): "wrong under neither"}[(ea, eb)]
        elif ea == EVIDENCE_WRONG:
            inconclusive[pair_id] = "wrong at the default; swa-full inconclusive"
        elif eb == EVIDENCE_WRONG:
            inconclusive[pair_id] = "wrong with swa-full; default inconclusive"
        else:
            inconclusive[pair_id] = "inconclusive"
    if not logs_ok:
        reason = (f"effective sliding-window cache not verified for every load (default {default_states}, "
                  f"swa-full {full_states})")
    elif not conclusive:
        reason = "no conclusive pair"
    else:
        reason = "every load's own log verified its sliding-window cache"
    if summary.get("problems"):
        reason = "; ".join(summary["problems"])
    return {"interpretable": logs_ok and bool(conclusive) and not summary.get("problems"),
            "reason": reason, "logs_verified": logs_ok,
            "comparable": conclusive, "inconclusive": inconclusive, "not_comparable": not_comparable}


# --------------------------------------------------------------------------- #
# Chat exposure: what an ordinary client gets                                   #
# --------------------------------------------------------------------------- #
NO_REUSE, PARTIAL_REUSE, FULL_REUSE, REUSE_UNKNOWN = "none", "partial", "full", "unknown"
CHAT_FINISH = frozenset({"stop", "length"})


@dataclass
class ChatReading:
    """One chat reply as the engine sent it (None: not reported). b11139's non-streamed
    reply has ``usage`` (``prompt_tokens`` the whole prompt, ``prompt_tokens_details.
    cached_tokens`` the part reused, ``completion_tokens``), ``choices[0].finish_reason``,
    and ``timings`` when the engine set them; it has no slot identity."""

    text: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    cached: int | None
    completion_tokens: int | None
    timings: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def chat_reading(payload: dict[str, Any]) -> ChatReading:
    raw_choices = payload.get("choices")
    choices: list[Any] = raw_choices if isinstance(raw_choices, list) else []
    first: dict[str, Any] = choices[0] if choices and isinstance(choices[0], dict) else {}
    raw_message = first.get("message")
    message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
    content = message.get("content")
    raw_usage = payload.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    raw_details = usage.get("prompt_tokens_details")
    details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
    raw_timings = payload.get("timings")
    finish = first.get("finish_reason")
    problems = []
    if len(choices) != 1:
        problems.append("expected exactly one chat choice")
    if "timings" in payload and (not isinstance(raw_timings, dict) or not raw_timings):
        problems.append("reported timings are missing or invalid")
    if "total_tokens" in usage:
        total = _count(usage.get("total_tokens"))
        prompt, completion = _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens"))
        if total is None or prompt is None or completion is None or total != prompt + completion:
            problems.append("chat usage total does not match prompt plus completion")
    return ChatReading(
        text=content if isinstance(content, str) else None,
        finish_reason=finish if isinstance(finish, str) else None,
        prompt_tokens=_count(usage.get("prompt_tokens")), cached=_count(details.get("cached_tokens")),
        completion_tokens=_count(usage.get("completion_tokens")),
        timings=dict(raw_timings) if isinstance(raw_timings, dict) else {},
        problems=problems,
    )


def chat_evidence(reply: dict[str, Any], *, max_tokens: int = MEASURED_PIECES) -> tuple[str, list[str]]:
    """``(reuse category, problems)`` for one chat reply (a ``ChatReading`` as a dict).

    Complete evidence: non-empty text; ``finish_reason`` ``stop`` (ended naturally) or
    ``length`` (the token limit); between 1 and ``max_tokens`` completion tokens; whole,
    non-negative prompt and reused counts with reused below the whole prompt; and, when
    ``timings`` are present, ``cache_n`` equal to the reused count and ``prompt_n`` to the
    rest. Reuse: ``none`` (0), ``partial`` (more than 0, fewer than the whole prompt minus
    one), ``full`` (the whole prompt minus one), ``unknown`` (counters missing or
    inconsistent)."""
    problems: list[str] = list(reply.get("problems", []))
    text = reply.get("text")
    if not isinstance(text, str) or not text.strip():
        problems.append("no reply text")
    if reply.get("finish_reason") not in CHAT_FINISH:
        problems.append(f"finish reason {reply.get('finish_reason')!r}")
    completion = _count(reply.get("completion_tokens"))
    if completion is None or not 1 <= completion <= max_tokens:
        problems.append(f"completion tokens {reply.get('completion_tokens')!r}")
    total, cached = _count(reply.get("prompt_tokens")), _count(reply.get("cached"))
    counters: list[str] = []
    if total is None or cached is None:
        counters.append("prompt or reused count missing or invalid")
    elif cached >= total:
        counters.append(f"reused {cached} of {total} prompt pieces")
    timings = reply.get("timings") or {}
    if not isinstance(timings, dict):
        problems.append("reported timings are not an object")
        timings = {}
    if timings and total is not None and cached is not None and not counters:
        if _count(timings.get("cache_n")) != cached or _count(timings.get("prompt_n")) != total - cached:
            counters.append(f"timings {timings.get('cache_n')!r}/{timings.get('prompt_n')!r} disagree with usage "
                            f"{cached}/{total}")
        if "predicted_n" in timings and _count(timings.get("predicted_n")) != completion:
            counters.append("timings written count disagrees with usage completion tokens")
    problems.extend(counters)
    if counters:
        return REUSE_UNKNOWN, problems
    assert total is not None and cached is not None
    if cached == 0:
        return NO_REUSE, problems
    return (FULL_REUSE if cached >= total - 1 else PARTIAL_REUSE), problems


class ChatEngine(ReuseEngine, Protocol):
    def chat(self, messages: list[dict[str, str]], *, cache_prompt: bool | None, max_tokens: int,
             seed: int) -> ChatReading: ...


#: The three loads of one chat trial: (label, [(question role, cache_prompt)]).
CHAT_LOADS: dict[str, list[tuple[str, bool | None]]] = {
    "client": [("prime", None), ("measured", None)],
    "matched": [("prime", None), ("measured", False)],
    "unprimed": [("measured", False)],
}


def chat_order(block: int) -> list[str]:
    """Client and matched reference alternate (counterbalanced); the unprimed reference,
    an extra control, comes last."""
    return (["client", "matched"] if block % 2 == 0 else ["matched", "client"]) + ["unprimed"]


def _request_breach(reply: dict[str, Any], *, first_on_load: bool, cache_prompt: bool | None) -> str | None:
    cached, total = _count(reply.get("cached")), _count(reply.get("prompt_tokens"))
    if cached is None:
        return None
    if first_on_load and cached > 0:
        return f"E4: the first request on a fresh load reused {cached} pieces (nothing to reuse)"
    if cache_prompt is False and cached > 0:
        return f"E1: a request asked to be fresh reused {cached} pieces"
    if total is not None and cached > max(total - 1, 0):
        return f"E4: reused {cached} of {total} prompt pieces (the last piece is always processed)"
    return None


def chat_trial(engine: ChatEngine, pair: KnownAnswerPair, *, block: int, power: Callable[[], str],
               power_check: bool) -> dict[str, Any]:
    """One chat trial: the three loads (``CHAT_LOADS``) in ``chat_order(block)``.

    Power is checked before each load starts and after it stops; every observed state is
    recorded and any one not allowed stops the trial before the next load. Each request
    is checked as soon as it returns (``_request_breach``: E4 on a fresh load's first
    request or impossible reuse, E1 on a request asked to be fresh); a breach ends that
    load's requests, the engine is stopped, and no further load runs. An unclean cleanup
    stops the trial too. ``stop`` is ``{"status", "reason"}`` or None.

    An interruption anywhere in the trial (a request, a cleanup, or a power reading at a
    load boundary) is recorded with everything gathered so far: every reply is kept in
    the result as soon as it returns, and the trial returns at once for the caller to
    checkpoint and re-raise. The engine's cleanup stays here, in the load's ``finally``."""
    result: dict[str, Any] = {"pair": pair.pair_id, "block": block, "order": chat_order(block), "problems": [],
                              "interrupted": False, "power": [], "cleanup": [], "stop": None}
    questions = {"prime": pair.question_prime, "measured": pair.question_measured}

    def stop(status: str, reason: str) -> dict[str, Any]:
        result["stop"] = {"status": status, "reason": reason}
        return result

    def interrupted(where: str) -> dict[str, Any]:
        result["interrupted"] = True
        result["problems"].append(f"interrupted {where}")
        return result

    def power_state(when: str) -> str | None:
        """The power state, recorded; None when the reading itself was interrupted."""
        try:
            state = _power(power)
        except KeyboardInterrupt:
            interrupted(f"while reading power {when}")
            return None
        result["power"].append({"when": when, "state": state})
        return state

    for label in result["order"]:
        state = power_state(f"before {label}")
        if state is None:
            return result
        if not _power_allowed(state, unchecked_ok=not power_check):
            return stop(STOPPED_POWER, f"power '{state}' before the {label} load of {pair.pair_id}")
        replies: list[dict[str, Any]] = []
        result[label] = replies  # kept as each reply returns
        breach: str | None = None
        cleanup = Cleanup(False, None, "not started")
        try:
            engine.start(DEFAULT)
            for index, (role, cache_prompt) in enumerate(CHAT_LOADS[label]):
                reading = engine.chat(pair.chat_messages(questions[role]), cache_prompt=cache_prompt,
                                      max_tokens=CHAT_MAX_TOKENS, seed=SEED)
                reply = {"role": role, "cache_prompt": cache_prompt, **asdict(reading)}
                replies.append(reply)
                breach = _request_breach(reply, first_on_load=index == 0, cache_prompt=cache_prompt)
                if breach:
                    break
        except KeyboardInterrupt:
            interrupted(f"during the {label} load")
        except Exception as exc:  # noqa: BLE001 - recorded; the engine is stopped below
            result["problems"].append(f"{label}: failed: {type(exc).__name__}: {exc}")
        finally:
            try:
                cleanup = engine.stop()
            except KeyboardInterrupt:
                interrupted(f"while stopping the {label} load")
                cleanup = Cleanup(False, None, "interrupted while stopping")
            except Exception as exc:  # noqa: BLE001
                cleanup = Cleanup(False, None, f"stop failed: {type(exc).__name__}: {exc}")
            result["cleanup"].append({"load": label, **asdict(cleanup), "clean": cleanup.clean})
        if result["interrupted"]:
            return result
        if breach:
            return stop(STOPPED_VALIDITY, f"{breach} ({label} load, {pair.pair_id} block {block + 1})")
        if not cleanup.clean:
            return stop(STOPPED_CLEANUP, f"{label} load of {pair.pair_id}: {cleanup.problem}")
        input_problem = getattr(engine, "input_problem", None)
        if input_problem:
            return stop("error", str(input_problem))
        state = power_state(f"after {label}")
        if state is None:
            return result
        if not _power_allowed(state, unchecked_ok=not power_check):
            return stop(STOPPED_POWER, f"power '{state}' after the {label} load of {pair.pair_id}")
    return result


def run_chat_exposure(engine: ChatEngine, pairs: Sequence[KnownAnswerPair], *, blocks: int = CHAT_BLOCKS,
                      power: Callable[[], str], power_check: bool,
                      checkpoint: Callable[[list[dict[str, Any]], str | None, str | None], None]
                      = lambda _t, _s, _r: None) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Every pair, ``blocks`` times. Returns (trials, stop status or None, stop reason). A
    trial's stop ends the run at once, with nothing further dispatched; an interruption
    is re-raised after the checkpoint."""
    trials: list[dict[str, Any]] = []
    for block in range(blocks):
        for pair in pairs:
            trial = chat_trial(engine, pair, block=block, power=power, power_check=power_check)
            trials.append(trial)
            if trial["interrupted"]:
                checkpoint(trials, INTERRUPTED, f"interrupted during chat {pair.pair_id}")
                raise KeyboardInterrupt
            if trial["stop"]:
                checkpoint(trials, trial["stop"]["status"], trial["stop"]["reason"])
                return trials, trial["stop"]["status"], trial["stop"]["reason"]
            checkpoint(trials, None, None)
    return trials, None, None


def chat_summary(trials: Sequence[dict[str, Any]], pairs: Sequence[KnownAnswerPair]) -> dict[str, Any]:
    """Per pair: is an ordinary client exposed, and is it reuse?

    A trial is **usable** when it did not stop, had no problem, every cleanup was clean,
    every power reading allowed, all three loads answered every request, and every reply
    has complete evidence (``chat_evidence``); the matched reference's measured reply must
    have reused nothing. Per pair (usable trials):

    * ``not usable`` with no usable trial;
    * ``not informative: the matched reference is not right`` unless the matched
      reference is ``right`` in every usable trial (a previous request, or the model,
      already gives the wrong answer without reuse);
    * ``exposed: wrong for the client after partial reuse`` when, in some usable trial,
      the client's measured reply reused part of the prompt and is ``wrong``;
    * ``wrong for the client without partial reuse``; ``unscorable for the client``;
    * ``right for the client after partial reuse`` or ``right for the client, no partial
      reuse happened``.

    The unprimed reference is reported per trial (``unprimed_verdict``) as an extra
    control; it does not decide the outcome."""
    by_id = {pair.pair_id: pair for pair in pairs}
    out: dict[str, Any] = {"version": CORRECTNESS_VERSION, "pairs": {}}
    duplicates = Counter((t.get("pair"), t.get("block")) for t in trials)
    for pair_id in sorted({t["pair"] for t in trials}):
        if pair_id not in by_id:
            out.setdefault("problems", []).append(f"unknown chat pair: {pair_id}")
            continue
        pair = by_id[pair_id]
        rows = []
        for t in (t for t in trials if t["pair"] == pair_id):
            reasons: list[str] = list(t["problems"])
            if duplicates[(pair_id, t.get("block"))] != 1 or _count(t.get("block")) is None:
                reasons.append("duplicate or invalid chat block")
            if t.get("interrupted"):
                reasons.append("interrupted chat trial")
            cleanups = t.get("cleanup", [])
            if (Counter(c.get("load") for c in cleanups) != Counter(CHAT_LOADS.keys())
                    or any(c.get("clean") is not True or c.get("process_gone") is not True
                           or c.get("credential_removed") is False or c.get("problem") for c in cleanups)):
                reasons.append("missing, duplicate or incomplete load cleanup evidence")
            power = t.get("power", [])
            if Counter(p.get("when") for p in power) != Counter(
                    f"{when} {label}" for label in CHAT_LOADS for when in ("before", "after")):
                reasons.append("before/after power evidence missing or duplicated")
            if t.get("stop"):
                reasons.append(f"stopped: {t['stop']['reason']}")
            reasons.extend(f"{c['load']} cleanup not clean" for c in t["cleanup"] if not c["clean"])
            reasons.extend(f"power {p['state']} {p['when']}" for p in t["power"]
                           if not _power_allowed(p["state"], unchecked_ok=True))
            evidence: dict[str, dict[str, Any]] = {}
            for label, plan in CHAT_LOADS.items():
                replies = t.get(label, [])
                if len(replies) != len(plan):
                    reasons.append(f"{label}: {len(replies)} of {len(plan)} replies")
                    continue
                for reply, (role, cache_prompt) in zip(replies, plan):
                    if reply.get("role") != role or reply.get("cache_prompt") is not cache_prompt:
                        reasons.append(f"{label}: reply role or freshness request does not match the plan")
                    breach = _request_breach(reply, first_on_load=role == plan[0][0], cache_prompt=cache_prompt)
                    if breach:
                        reasons.append(f"{label}: {breach}")
                    category, problems = chat_evidence(reply)
                    reasons.extend(f"{label} {reply['role']}: {p}" for p in problems)
                    if reply["role"] == "measured":
                        evidence[label] = {"reuse": category, "verdict": judge(reply.get("text"), pair),
                                           "text": reply.get("text")}
            matched = evidence.get("matched")
            if matched is not None and matched["reuse"] != NO_REUSE:
                reasons.append(f"matched reference reused ({matched['reuse']})")
            row: dict[str, Any] = {"block": t["block"], "order": t.get("order"), "usable": not reasons,
                                   "reasons": list(dict.fromkeys(reasons))}
            for label in CHAT_LOADS:
                item = evidence.get(label)
                row[f"{label}_verdict"] = item["verdict"][0] if item else None
                row[f"{label}_judgement"] = item["verdict"][1] if item else None
                row[f"{label}_reuse"] = item["reuse"] if item else None
                row[f"{label}_text"] = item["text"] if item else None
            rows.append(row)
        usable = [r for r in rows if r["usable"]]
        if not usable:
            outcome = "not usable"
        elif any(r["matched_verdict"] != RIGHT for r in usable):
            outcome = "not informative: the matched reference is not right"
        elif any(r["client_reuse"] == PARTIAL_REUSE and r["client_verdict"] == WRONG for r in usable):
            outcome = "exposed: wrong for the client after partial reuse"
        elif any(r["client_verdict"] == WRONG for r in usable):
            outcome = "wrong for the client without partial reuse"
        elif any(r["client_verdict"] == UNSCORABLE for r in usable):
            outcome = "unscorable for the client"
        elif any(r["client_reuse"] == PARTIAL_REUSE for r in usable):
            outcome = "right for the client after partial reuse"
        else:
            outcome = "right for the client, no partial reuse happened"
        out["pairs"][pair_id] = {"expected": pair.expected, "regression": pair.regression, "trials": rows,
                                 "usable": len(usable), "outcome": outcome}
    return out


def answer_oracle(pairs: Sequence[KnownAnswerPair]) -> Callable[[str], str | None]:
    """For the simulated engine: a right answer to each pair's measured question (also
    asked through the chat path); None (a digest) for anything else."""
    def oracle(prompt: str) -> str | None:
        for pair in pairs:
            if prompt.rstrip().endswith(pair.question_measured) or pair.question_measured + "\nAnswer:" in prompt[-300:]:
                return f" The answer is {pair.expected}."
        return None

    return oracle


__all__ = [
    "BLOCKS", "CHAT_BLOCKS", "CHAT_LOADS", "CORRECTNESS_VERSION", "DEFAULT", "EVIDENCE_INCONCLUSIVE", "EVIDENCE_NONE",
    "EVIDENCE_NOT_INFORMATIVE", "EVIDENCE_RIGHT", "EVIDENCE_WRONG", "FULL_REUSE", "NO_REUSE",
    "PARTIAL_REUSE", "REUSE_UNKNOWN", "RIGHT", "SWA_FULL", "UNSCORABLE", "WRONG", "ChatEngine", "ChatReading",
    "KnownAnswerPair", "answer_line", "answer_oracle", "build_pairs", "chat_evidence", "chat_order", "chat_reading",
    "chat_summary", "chat_trial", "correctness_block", "correctness_summary", "judge", "pair_outcome",
    "partial_reuse_evidence",
    "run_chat_exposure", "swa_full_comparison",
]
