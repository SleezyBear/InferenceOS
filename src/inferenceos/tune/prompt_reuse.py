"""Prompt reuse: a development-time evaluation (16 Oct 2026; revised after review). Not a
tune path.

llama.cpp's server can keep the processed start of a prompt and, when the next prompt
starts the same way, process only the rest (``cache_prompt``, on by default in
llama-server b11139). With ``--cache-reuse N`` it also searches the old work for chunks
of at least N pieces that the new prompt continues with (``server-context.cpp``: the
new-prompt cursor stays at the first mismatch while the old-work cursor scans forward).
Tune never asks for reuse: every tune measurement requests a fresh reading of the whole
prompt. This module measures, on a real engine, what reuse is worth and whether it is
safe, so that a later decision rests on evidence collected before any user's tune.

**One trial** (the unit of evidence):

1. a fresh engine load with the variant's settings (mains power checked first);
2. a **priming** request (``cache_prompt`` True, one written piece), the same in both
   arms, so both arms are equally warm and the engine holds the same earlier work;
3. the **measured** request: the same prompt, seed, length and sampling in both arms;
   only ``cache_prompt`` differs (fresh arm False, reuse arm True);
4. the engine is stopped and its cleanup checked; mains power checked again.

**Blocks** pair one fresh-arm and one reuse-arm trial, in alternating order. A block's
difference counts only with complete work evidence from the engine itself in both
arms (processed + reused = whole prompt = the engine tokenizer's count; the asked
number of pieces written; a completed response; a valid time), reuse shown in the
reuse arm and none in the fresh arm, the reuse within what the prompts can share
(``reuse_bound``), both requests in the slot the priming request used, mains power
throughout, and identical answers. Ordinary warming is equal in both arms by
construction; a difference without that evidence is reported, never counted.

**Stops** (pre-registered, immediate; the partial record is kept):

* *cleanup*: a stop that did not confirm the process gone and its password file
  deleted ends the run; the engine object is kept for the existing retry (``cleanup()``);
* *validity gate E1*: a fresh-arm reading whose own complete counters show reuse;
* *validity gate E4*: any reading whose complete counters show more reuse than possible,
  checked as soon as the reading exists: a priming request on a fresh load (nothing to
  reuse) before the measured request is sent, the measured request, and the cancellation
  check's readings;
* *power*: anything but verified mains power (or no battery) before or after a trial.

Absence of reuse is a completed negative result, not a stop. Completed trials are
checkpointed after each trial, so an interruption keeps them (status "interrupted").

Every reading carries the prompt-reuse identity of what was asked (``ReuseIdentity``).
Nothing here writes saved setups or tune history, changes the context size, or uses
tune's acceptance rules. Results are development evidence only: they cannot admit a
path into Practical mode.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from inferenceos.tune.stats import t_quantile

EVALUATION_VERSION = "prompt-reuse-eval-5"
FRESH = "fresh"
REUSE = "reuse"
ARMS = (FRESH, REUSE)
#: Pieces written by the measured request in both arms (fixed length, no early stop).
MEASURED_PIECES = 24
#: Pieces written by the priming request (the earlier work the engine keeps).
PRIMING_PIECES = 1
SEED = 42
#: Power states a trial may run in. "not checked" is only allowed when the caller
#: disabled the check explicitly (simulation or a labelled test run).
POWER_OK = frozenset({"ac", "no battery"})

DONE = "done"
INCOMPLETE = "incomplete"
STOPPED_CLEANUP = "stopped: cleanup incomplete"
STOPPED_VALIDITY = "stopped: validity gate"
STOPPED_POWER = "stopped: not on mains power"
INTERRUPTED = "interrupted"

#: How an abandoned answer ended. Only ``CUT_BY_CLIENT`` is an interruption.
CUT_BY_CLIENT = "client cutoff"
ENGINE_FINISHED = "engine finished"
ENDED_EARLY = "stream ended early"
REQUEST_FAILED = "request failed"

#: llama-server b11139 (``server-context.cpp``, ``load_model``) resets ``--cache-reuse``
#: to 0 with one of these warnings when the model's memory cannot shift or for
#: multimodal use. The warning level is shown at the default log verbosity (3).
CACHE_REUSE_DISABLED_WARNINGS = (
    "cache_reuse is not supported by this context, it will be disabled",
    "cache_reuse is not supported by multimodal, it will be disabled",
)
#: Logged by b11139 (``server.cpp``, ``SRV_INF``) after ``load_model`` returns: a startup
#: log that contains it covers the point where the warning would have been printed.
MODEL_LOADED_MARKER = "model loaded"


# --------------------------------------------------------------------------- #
# Cases: prompt pairs with a known relation                                     #
# --------------------------------------------------------------------------- #
_INSTRUCTIONS = (
    "You are a careful assistant for a hardware maintenance team. Answer only from the "
    "material given, say when the material does not contain the answer, and keep every "
    "answer to one short paragraph. Use plain words and give numbers with their units. "
)
_MANUAL = (
    "Section {n}. The cooling loop of unit {n} carries water at {t} degrees through two "
    "pumps. Pump A runs continuously; pump B starts when the inlet passes {u} degrees. "
    "Filters are changed every {d} days, and a pressure drop above {p} kilopascals means "
    "a filter is blocked. Log every change in the unit {n} book with the date and name. "
)
_FIELD_NOTES = (
    "Note {n}. Orchard row {n} was pruned in week {t}. Apple trees in this row bloom "
    "about {u} days after the last frost. Bees were moved in on day {d}, and the soil "
    "moisture reading was {p} percent at noon. The next visit is planned for week {t}. "
)


def _document(template: str, count: int, offset: int = 0) -> str:
    parts = []
    for index in range(count):
        n = index + 1 + offset
        parts.append(template.format(n=n, t=40 + n % 17, u=50 + n % 23, d=10 + n % 29, p=20 + n % 31))
    return "".join(parts)


@dataclass(frozen=True)
class ReuseCase:
    """One priming prompt and one measured prompt with a known relation."""

    case_id: str
    description: str
    prime: str
    measured: str
    #: Whether a correct engine should reuse a meaningful part of the measured prompt.
    expect_reuse: bool
    #: Whether shifted-chunk reuse (``--cache-reuse``) can apply to this pair at all.
    shifted_chunks: bool = False

    @property
    def sha256(self) -> str:
        return _sha({"case": self.case_id, "prime": self.prime, "measured": self.measured,
                     "expect_reuse": self.expect_reuse, "shifted_chunks": self.shifted_chunks})


def build_cases(sections: int = 12) -> tuple[ReuseCase, ...]:
    """The fixed cases. ``sections`` sets the length of the shared material (12 sections
    is roughly 1,000-1,300 pieces on common tokenizers, well inside a 4,096 context).

    ``deleted_middle`` is the shifted-chunk **positive control**: removing a section makes
    the new prompt continue, at the first mismatch, with work found later in the old
    prompt, which is what the engine's chunk search can reuse. ``edited_middle`` replaces a
    section instead; the same search cannot reuse its unchanged end (the new-prompt cursor
    stays on the replaced text), so it is expected to show prefix reuse only."""
    manual = _document(_MANUAL, sections)
    head = _INSTRUCTIONS + manual
    question_1 = "\nQuestion: how often are the filters of unit 3 changed?\nAnswer:"
    question_2 = "\nQuestion: when does pump B of unit 7 start?\nAnswer:"
    reply = " The filters of unit 3 are changed every 13 days, and each change is logged."
    edited = _document(_MANUAL, 1, offset=100)
    middle = _document(_FIELD_NOTES, 2, offset=50)
    tail = _document(_MANUAL, sections // 2, offset=sections)
    return (
        ReuseCase("identical", "the same prompt twice (most reuse possible)",
                  head + question_1, head + question_1, expect_reuse=True),
        ReuseCase("shared_prefix", "the same instructions and manual, a different question",
                  head + question_1, head + question_2, expect_reuse=True),
        ReuseCase("continuation", "a conversation continued: the earlier turn, a fixed reply, a new question",
                  head + question_1, head + question_1 + reply + question_2, expect_reuse=True),
        ReuseCase("edited_middle", "one section near the start replaced; the long remainder unchanged",
                  _INSTRUCTIONS + _document(_MANUAL, 1) + tail + question_1,
                  _INSTRUCTIONS + edited + tail + question_1, expect_reuse=True, shifted_chunks=True),
        ReuseCase("deleted_middle", "two notes near the start removed; the long remainder unchanged "
                  "(shifted-chunk positive control)",
                  _INSTRUCTIONS + middle + tail + question_1,
                  _INSTRUCTIONS + tail + question_1, expect_reuse=True, shifted_chunks=True),
        ReuseCase("unrelated", "a different document and question (isolation: almost nothing to reuse)",
                  head + question_1, _document(_FIELD_NOTES, sections) + "\nQuestion: when were bees moved in?\nAnswer:",
                  expect_reuse=False),
    )


@dataclass(frozen=True)
class EngineVariant:
    """Engine settings for one series of trials (everything else is the model's setup)."""

    variant_id: str
    #: ``--cache-reuse`` chunk size; None = engine default (off).
    cache_reuse: int | None = None
    #: ``--swa-full``: a full-size sliding-window cache (False = engine default).
    swa_full: bool = False
    #: Cases this variant runs (an empty tuple means every case).
    cases: tuple[str, ...] = ()


DEFAULT_VARIANTS = (
    EngineVariant("default"),
    EngineVariant("cache-reuse-256", cache_reuse=256, cases=("edited_middle", "deleted_middle")),
)


@dataclass(frozen=True)
class ReuseIdentity:
    """What one reading asked for. Cached and fresh readings never share an identity."""

    evaluation: str
    case_id: str
    case_sha256: str
    arm: str
    cache_prompt: bool
    variant_id: str
    server_cache_reuse: int | None
    context_length: int
    configuration_sha256: str
    server_swa_full: bool = False

    @property
    def sha256(self) -> str:
        return _sha(asdict(self))


# --------------------------------------------------------------------------- #
# The engine interface (real: scripts/evaluate_prompt_reuse.py; tests: a fake)  #
# --------------------------------------------------------------------------- #
@dataclass
class Reading:
    """One answer, exactly as the engine reported it (None: not reported). ``cached`` is
    the engine's own reused-prompt count (``timings.cache_n``); ``occupancy`` is the
    slot's size after the answer (``tokens_cached``), which is NOT a reuse count."""

    ttft_s: float | None
    total_s: float | None
    prefill_s: float | None
    processed: int | None
    cached: int | None
    total: int | None
    text: str
    cache_prompt: bool
    generated: int | None = None
    completed: bool | None = None
    slot: int | None = None
    occupancy: int | None = None
    engine_counters: dict[str, Any] = field(default_factory=dict)
    #: Optional development provenance reference; absent in legacy records.
    load_evidence_id: str | None = None


@dataclass
class Abandoned:
    """What happened when the client set out to walk away from an answer."""

    pieces: int
    #: How the answer's stream ended: ``CUT_BY_CLIENT`` (the client left after the
    #: intended number of pieces; the only interruption), ``ENGINE_FINISHED`` (the engine
    #: said it finished first), ``ENDED_EARLY`` (the stream ended without either) or
    #: ``REQUEST_FAILED`` (the request itself failed; ``error`` says how).
    ending: str
    #: Whether the engine confirmed every slot idle afterwards within the wait (True),
    #: did not (False: still busy, or no reply in time), or could not be asked (None).
    idle_confirmed: bool | None
    idle_detail: str | None = None
    error: str | None = None


@dataclass
class Cleanup:
    process_gone: bool
    credential_removed: bool | None
    problem: str | None = None

    @property
    def clean(self) -> bool:
        return self.process_gone and self.credential_removed is not False and self.problem is None


class ReuseEngine(Protocol):
    def start(self, variant: EngineVariant) -> None: ...
    def tokenize(self, text: str) -> list[int]: ...
    def generate(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int) -> Reading: ...
    def abandon(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int, after_chunks: int) -> Abandoned: ...
    def healthy(self) -> bool: ...
    def stop(self) -> Cleanup: ...
    #: The current load's startup log, or None when it cannot show the engine's warnings.
    def startup_log(self) -> str | None: ...


_SWA_SIZES = re.compile(r"creating\s+(non-SWA|SWA) KV cache, size = (\d+) cells")
#: b11139 ``llama-kv-cache-iswa.cpp``: logged (warning level) when ``--swa-full`` takes effect.
SWA_FULL_WARNING = "using full-size SWA cache"


def swa_cache_evidence(log: str | None, *, requested_full: bool) -> dict[str, Any]:
    """The sliding-window cache this load actually got, from its own startup log.

    b11139 (``llama-kv-cache-iswa.cpp``) logs ``creating non-SWA KV cache, size = N
    cells`` and ``creating SWA KV cache, size = M cells`` for models with sliding-window
    layers, and the warning ``using full-size SWA cache`` when ``--swa-full`` takes effect
    (then M = N). States: ``full`` (warning present and M = N), ``reduced`` (no warning
    and M < N), ``no sliding-window cache`` (a complete log without those lines: the
    model has none), or ``unverified`` (no log, an incomplete log, or contradictory
    lines). A requested setting is never taken as the effective one."""
    evidence: dict[str, Any] = {"requested_full": requested_full, "base_cells": None, "swa_cells": None,
                                "full_warning": None}
    if log is None:
        return {**evidence, "state": "unverified: no startup log with warnings"}
    sizes: dict[str, list[int]] = {}
    for kind, cells in _SWA_SIZES.findall(log):
        sizes.setdefault(kind, []).append(int(cells))
    warning = SWA_FULL_WARNING in log
    evidence["full_warning"] = warning
    complete = any(line.rstrip().endswith(MODEL_LOADED_MARKER) for line in log.splitlines())
    if not complete:
        return {**evidence, "state": "unverified: the startup log does not reach the end of loading"}
    if not sizes:
        state = ("no sliding-window cache" if complete and not warning
                 else "unverified: no cache sizes in the startup log")
        return {**evidence, "state": state}
    base, swa = sizes.get("non-SWA", []), sizes.get("SWA", [])
    if len(base) != 1 or len(swa) != 1:
        return {**evidence, "state": f"unverified: cache sizes logged {len(base)} and {len(swa)} times"}
    evidence["base_cells"], evidence["swa_cells"] = base[0], swa[0]
    if warning and swa[0] == base[0]:
        state = "full"
    elif not warning and swa[0] < base[0]:
        state = "reduced"
    else:
        state = "unverified: the warning and the sizes disagree"
    return {**evidence, "state": state}


def cache_reuse_activation(variant: EngineVariant, log: str | None) -> str:
    """What the engine's own startup log says about ``--cache-reuse`` for one load.

    A requested option is not proof of an active one: b11139 turns it off, with a
    warning, when the model's memory cannot shift (or for multimodal use). The log can
    show that it was turned off; a complete log without the warning shows only that this
    branch was not taken. Direct proof of activity is shifted reuse in the counters."""
    if not variant.cache_reuse:
        return "not requested"
    if log is None:
        return "unverified: no startup log with warnings"
    for warning in CACHE_REUSE_DISABLED_WARNINGS:
        if warning in log:
            return f"disabled by the engine: {warning}"
    if not any(line.rstrip().endswith(MODEL_LOADED_MARKER) for line in log.splitlines()):
        return "unverified: the startup log does not reach the end of loading"
    return "not disabled: complete startup log without the disable warning"


# --------------------------------------------------------------------------- #
# How much reuse is possible                                                    #
# --------------------------------------------------------------------------- #
def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def shifted_reuse(old: Sequence[int], new: Sequence[int], chunk: int) -> int:
    """Pieces of ``new`` that b11139's ``--cache-reuse`` search can take from ``old`` after
    the shared start: the new-prompt cursor stays at the first mismatch while the
    old-work cursor scans forward; a run of at least ``chunk`` equal pieces is shifted
    into place and both cursors move past it."""
    start = common_prefix(old, new)
    head_c = head_p = start
    reused = 0
    while head_c < len(old) and head_p < len(new):
        n = 0
        while head_c + n < len(old) and head_p + n < len(new) and old[head_c + n] == new[head_p + n]:
            n += 1
        if n >= chunk:
            head_c += n
            head_p += n
            reused += n
        else:
            head_c += 1
    return reused


def reuse_bound(case: ReuseCase, prime_tokens: Sequence[int], measured_tokens: Sequence[int],
                variant: EngineVariant) -> int:
    """The most measured-prompt pieces a correct engine could take from the priming
    request: the shared start plus the one written piece (it may match the next piece),
    plus, with ``--cache-reuse`` on, what the engine's chunk search can shift (computed
    from the engine's own tokens; the written piece is unknown and never matched).
    The engine always processes at least the last piece of the prompt itself."""
    bound = common_prefix(prime_tokens, measured_tokens) + PRIMING_PIECES
    if variant.cache_reuse and case.shifted_chunks:
        bound += shifted_reuse(list(prime_tokens), list(measured_tokens), variant.cache_reuse)
    return min(bound, max(len(measured_tokens) - 1, 0))


# --------------------------------------------------------------------------- #
# Work evidence: is a reading complete and internally consistent?               #
# --------------------------------------------------------------------------- #
def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def prompt_counts(reading: dict[str, Any], *, expected_total: int) -> tuple[int | None, list[str]]:
    """``(reused pieces or None, problems)`` from the engine's prompt counters alone.

    Reuse is known only from the engine's own counts: ``cached`` (``timings.cache_n``) or,
    when that is absent, whole prompt (``tokens_evaluated``) minus processed
    (``timings.prompt_n``). They must be whole, non-negative, add up, and the whole prompt
    must equal the engine tokenizer's count. Nothing is assumed when anything is missing."""
    problems: list[str] = []
    processed, total, cached = _count(reading.get("processed")), _count(reading.get("total")), _count(reading.get("cached"))
    for name, raw, value in (("processed", reading.get("processed"), processed),
                             ("whole-prompt", reading.get("total"), total)):
        if value is None:
            problems.append(f"{name} count {'missing' if raw is None else 'invalid'}")
    if reading.get("cached") is not None and cached is None:
        problems.append("reused count invalid")
    if reading.get("cached") is None and processed is not None and total is not None:
        if processed > total:
            problems.append("processed exceeds the whole prompt")
        else:
            cached = total - processed
    if cached is not None and processed is not None and total is not None and processed + cached != total:
        problems.append(f"counts do not add up: {processed} processed + {cached} reused != {total}")
    if total is not None and total != expected_total:
        problems.append(f"engine counted {total} prompt pieces, its tokenizer {expected_total}")
    return (None if problems else cached), problems


def work_evidence(reading: dict[str, Any], *, expected_total: int, expected_generated: int
                  ) -> tuple[int | None, list[str]]:
    """``(reused pieces or None, problems)`` for one reading: the prompt counters
    (``prompt_counts``) plus the rest of complete work evidence: the asked number of
    written pieces as the engine counted them, a completed response and a valid time to
    first word. Every missing or contradictory item is a problem."""
    if not reading:
        return None, ["no reading"]
    cached, problems = prompt_counts(reading, expected_total=expected_total)
    generated = _count(reading.get("generated"))
    raw = reading.get("engine_counters") or {}
    if raw.get("protocol_problem"):
        problems.append(f"engine protocol: {raw['protocol_problem']}")
    if raw:
        for name in ("predicted_n", "tokens_predicted"):
            value = raw.get(name)
            if value is not None and (_count(value) is None or value != generated):
                problems.append(f"engine {name} {value!r} disagrees with written count {generated!r}")
    if generated != expected_generated:
        problems.append(f"wrote {reading.get('generated')!r} pieces, asked for {expected_generated}")
    if reading.get("completed") is not True:
        problems.append("the response did not complete")
    ttft = reading.get("ttft_s")
    if not (isinstance(ttft, (int, float)) and not isinstance(ttft, bool) and math.isfinite(ttft) and ttft > 0):
        problems.append("time to first word missing or invalid")
    return cached, problems


def counters_reuse(reading: dict[str, Any], *, expected_total: int) -> int | None:
    """Reused pieces when the engine's prompt counters alone are complete and consistent
    (the validity gates must not depend on timing or the answer)."""
    return prompt_counts(reading, expected_total=expected_total)[0] if reading else None


@dataclass
class TrialResult:
    case_id: str
    variant_id: str
    arm: str
    block: int
    expect_reuse: bool
    identity: dict[str, Any]
    identity_sha256: str
    prime_tokens: int
    measured_tokens: int
    shared_tokens: int
    bound: int
    prime: dict[str, Any]
    measured: dict[str, Any]
    cleanup: dict[str, Any]
    power: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    interrupted: bool = False
    #: A validity gate (E1/E4) proven by this trial's own complete counters.
    breach: str | None = None
    #: What the engine's startup log says about ``--cache-reuse`` (``cache_reuse_activation``).
    activation: str = "not requested"
    #: Measured-prompt pieces the engine's chunk search could shift from the priming
    #: prompt at this variant's chunk size, on the engine's own tokens (None: not asked).
    shift_eligible: int | None = None
    #: What this load's startup log says about the sliding-window cache
    #: (``swa_cache_evidence``); None in records made before evaluation 4.
    swa_cache: dict[str, Any] | None = None


def _breach(reading: dict[str, Any], *, expected_total: int, bound: int, fresh: bool, what: str) -> str | None:
    """E1 / E4 from one reading's complete, consistent prompt counters (None otherwise:
    missing or contradictory counters make evidence unusable, they prove nothing)."""
    reused = counters_reuse(reading, expected_total=expected_total) if reading else None
    if reused is None:
        return None
    if fresh and reused > 0:
        return f"E1: {what} reused {reused} pieces in a reading asked to be fresh"
    if reused > bound:
        return f"E4: {what} reused {reused} pieces, more than the {bound} possible"
    return None


def run_trial(engine: ReuseEngine, case: ReuseCase, variant: EngineVariant, arm: str, *, block: int,
              context_length: int, configuration_sha256: str, measured_pieces: int = MEASURED_PIECES,
              power: Callable[[], str] = lambda: "not checked") -> TrialResult:
    """One fresh load, priming request, measured request, stop. The engine is always
    stopped, whatever happens, including an interruption, which is recorded on the
    result (``interrupted``) for the caller to re-raise after saving the evidence.

    The priming request runs on a fresh load, so any reuse it reports is impossible
    (E4): it is checked before the measured request is sent, and if it fires the
    measured request is not sent."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    identity = ReuseIdentity(
        evaluation=EVALUATION_VERSION, case_id=case.case_id, case_sha256=case.sha256, arm=arm,
        cache_prompt=arm == REUSE, variant_id=variant.variant_id, server_cache_reuse=variant.cache_reuse,
        context_length=context_length, configuration_sha256=configuration_sha256,
        server_swa_full=variant.swa_full,
    )
    problems: list[str] = []
    interrupted = False
    prime_reading: Reading | None = None
    measured_reading: Reading | None = None
    prime_tokens: list[int] = []
    measured_tokens: list[int] = []
    cleanup = Cleanup(process_gone=False, credential_removed=None, problem="not started")
    started = False
    breach: str | None = None
    activation = "not requested" if not variant.cache_reuse else "unverified: the load did not complete"
    swa_cache: dict[str, Any] = {"state": "unverified: the load did not complete", "requested_full": variant.swa_full}
    label = f"({variant.variant_id}/{case.case_id} block {block + 1}, {arm} arm)"
    try:
        started = True
        engine.start(variant)
        prime_tokens = engine.tokenize(case.prime)
        measured_tokens = engine.tokenize(case.measured)
        prime_reading = engine.generate(case.prime, cache_prompt=True, n_predict=PRIMING_PIECES, seed=SEED)
        breach = _breach(asdict(prime_reading), expected_total=len(prime_tokens), bound=0, fresh=False,
                         what=f"the priming request on a fresh load {label}")
        if breach is None:
            measured_reading = engine.generate(case.measured, cache_prompt=identity.cache_prompt,
                                               n_predict=measured_pieces, seed=SEED)
        else:
            problems.append(f"measured request not sent: {breach}")
        log: str | None = None
        log_error: str | None = None
        try:
            log = engine.startup_log()
        except Exception as exc:  # noqa: BLE001 - evidence only; never fails a trial
            log_error = f"the startup log could not be read ({type(exc).__name__})"
        if variant.cache_reuse:
            activation = (cache_reuse_activation(variant, log) if log_error is None
                          else f"unverified: {log_error}")
        swa_cache = swa_cache_evidence(log, requested_full=variant.swa_full)
    except KeyboardInterrupt:
        interrupted = True
        problems.append("interrupted")
    except Exception as exc:  # noqa: BLE001 - recorded; the engine is stopped below
        problems.append(f"trial failed: {type(exc).__name__}: {exc}")
    finally:
        if started:
            try:
                cleanup = engine.stop()
            except KeyboardInterrupt:
                interrupted = True
                cleanup = Cleanup(False, None, "interrupted while stopping")
            except Exception as exc:  # noqa: BLE001 - recorded, never hidden
                cleanup = Cleanup(process_gone=False, credential_removed=None,
                                  problem=f"stop failed: {type(exc).__name__}: {exc}")
    if not cleanup.clean:
        problems.append(f"cleanup not complete: {cleanup.problem or asdict(cleanup)}")
    try:
        after = _power(power)
    except KeyboardInterrupt:
        # The trial's readings are kept; the caller checkpoints them and re-raises.
        interrupted = True
        problems.append("interrupted while reading power after the trial")
        after = "not read (interrupted)"
    shared = common_prefix(prime_tokens, measured_tokens)
    bound = reuse_bound(case, prime_tokens, measured_tokens, variant) if measured_tokens else 0
    measured = asdict(measured_reading) if measured_reading else {}
    if breach is None and measured:
        breach = _breach(measured, expected_total=len(measured_tokens), bound=bound, fresh=arm == FRESH,
                         what=f"the measured request {label}")
    eligible = (shifted_reuse(prime_tokens, measured_tokens, variant.cache_reuse)
                if variant.cache_reuse and prime_tokens and measured_tokens else None)
    return TrialResult(
        case_id=case.case_id, variant_id=variant.variant_id, arm=arm, block=block,
        expect_reuse=case.expect_reuse, identity=asdict(identity), identity_sha256=identity.sha256,
        prime_tokens=len(prime_tokens), measured_tokens=len(measured_tokens), shared_tokens=shared, bound=bound,
        prime=asdict(prime_reading) if prime_reading else {}, measured=measured,
        cleanup=asdict(cleanup), power=[after], problems=problems, interrupted=interrupted,
        breach=breach, activation=activation, shift_eligible=eligible, swa_cache=swa_cache,
    )


def _power(probe: Callable[[], str]) -> str:
    try:
        return str(probe())
    except Exception as exc:  # noqa: BLE001 - an unknown power state is not mains power
        return f"unknown ({type(exc).__name__})"


def _power_allowed(state: str, *, unchecked_ok: bool) -> bool:
    return state in POWER_OK or (unchecked_ok and state == "not checked")


# --------------------------------------------------------------------------- #
# Judging the evidence                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class BlockVerdict:
    block: int
    usable: bool
    reasons: list[str]
    fresh_ttft_s: float | None
    reuse_ttft_s: float | None
    gain_s: float | None
    relative_gain: float | None
    cached: int | None
    answers_identical: bool | None
    #: Fresh arm stayed fresh: True (complete counters, nothing reused), False (reuse
    #: shown), None (counters could not establish it).
    fresh_stayed_fresh: bool | None = None
    slots: dict[str, Any] = field(default_factory=dict)


def judge_block(fresh: TrialResult, reuse: TrialResult) -> BlockVerdict:
    """Whether one fresh/reuse pair is usable evidence, and the time difference.

    For a case where reuse is expected, reuse must be shown in the reuse arm. For the
    isolation control (``expect_reuse`` False) it need not be: its difference shows what
    the comparison reads when there is (almost) nothing to reuse; it is never a gain."""
    reasons: list[str] = []
    for trial in (fresh, reuse):
        reasons.extend(f"{trial.arm}: {problem}" for problem in trial.problems)
        if any(not _power_allowed(p, unchecked_ok=True) for p in trial.power):
            reasons.append(f"{trial.arm}: not on mains power throughout ({trial.power})")
    if fresh.identity["case_sha256"] != reuse.identity["case_sha256"] or fresh.case_id != reuse.case_id:
        reasons.append("the two arms did not measure the same case")
    if fresh.measured_tokens != reuse.measured_tokens:
        reasons.append("different work: the engine tokenized the measured prompt differently")
    cached_by_arm: dict[str, int | None] = {}
    for trial in (fresh, reuse):
        p_cached, p_problems = work_evidence(trial.prime, expected_total=trial.prime_tokens,
                                             expected_generated=PRIMING_PIECES)
        reasons.extend(f"{trial.arm} priming: {p}" for p in p_problems)
        if p_cached:
            reasons.append(f"{trial.arm} priming reused {p_cached} pieces on a fresh load")
        m_cached, m_problems = work_evidence(trial.measured, expected_total=trial.measured_tokens,
                                             expected_generated=MEASURED_PIECES)
        reasons.extend(f"{trial.arm}: {p}" for p in m_problems)
        cached_by_arm[trial.arm] = m_cached
        if m_cached is not None and m_cached > trial.bound:
            reasons.append(f"{trial.arm} arm reused {m_cached} pieces, more than the {trial.bound} possible")
        if trial.arm == REUSE:
            # Reuse can only come from the slot the priming request used: both slot
            # identities must be known and equal, or neither a gain nor its absence can
            # be attributed to reuse. (The fresh arm's slots are recorded, not required.)
            p_slot, m_slot = _count(trial.prime.get("slot")), _count(trial.measured.get("slot"))
            if p_slot is None or m_slot is None:
                reasons.append("reuse arm: slot identity not reported (priming "
                               f"{trial.prime.get('slot')!r}, measured {trial.measured.get('slot')!r})")
            elif p_slot != m_slot:
                reasons.append(f"reuse arm answered in slot {m_slot}, not the priming slot {p_slot}")
    f_cached, r_cached = cached_by_arm[FRESH], cached_by_arm[REUSE]
    fresh_state = None if f_cached is None else f_cached == 0
    if fresh_state is False:
        reasons.append("the fresh arm reused earlier work (a fresh reading was not fresh)")
    if reuse.expect_reuse and not (r_cached is not None and r_cached > 0):
        reasons.append("reuse not shown in the reuse arm")
    f_text, r_text = fresh.measured.get("text"), reuse.measured.get("text")
    identical = None if f_text is None or r_text is None else f_text == r_text
    if identical is False:
        reasons.append("the answers differ between the arms")
    f_ttft, r_ttft = fresh.measured.get("ttft_s"), reuse.measured.get("ttft_s")
    gain = relative = None
    if isinstance(f_ttft, (int, float)) and isinstance(r_ttft, (int, float)) and f_ttft > 0 and r_ttft > 0:
        gain = float(f_ttft) - float(r_ttft)
        relative = gain / float(f_ttft)
    usable = not reasons and relative is not None
    return BlockVerdict(block=fresh.block, usable=usable, reasons=list(dict.fromkeys(reasons)),
                        fresh_ttft_s=f_ttft, reuse_ttft_s=r_ttft,
                        gain_s=gain if usable else None, relative_gain=relative if usable else None,
                        cached=r_cached, answers_identical=identical, fresh_stayed_fresh=fresh_state,
                        slots={arm: [t.prime.get("slot"), t.measured.get("slot")] for arm, t in
                               ((FRESH, fresh), (REUSE, reuse))})


def pair_problem(arms: Sequence[str]) -> str | None:
    """None when ``arms`` is exactly one fresh-arm and one reuse-arm load; otherwise what
    is wrong, naming duplicates as duplicates (an extra load is not a missing one)."""
    counts = {arm: list(arms).count(arm) for arm in set(arms)}
    if len(arms) == 2 and counts.get(FRESH) == 1 and counts.get(REUSE) == 1:
        return None
    parts = []
    for arm in ARMS:
        n = counts.get(arm, 0)
        if n == 0:
            parts.append(f"no {arm}-arm load")
        elif n > 1:
            parts.append(f"{n} {arm}-arm loads (duplicate)")
    unknown = sorted(a for a in counts if a not in ARMS)
    if unknown:
        parts.append(f"unknown arm(s) {unknown}")
    return ", ".join(parts)


def summarize(trials: Sequence[TrialResult], *, alpha: float = 0.05) -> dict[str, Any]:
    """Per variant and case: the block verdicts, and the mean relative gain with a
    two-sided (1 - 2*alpha) interval over usable blocks only. ``fresh_arm_stayed_fresh``
    is True only when every block established it, False when any block showed reuse,
    otherwise None (unknown)."""
    # Every load is kept (a list per block, never a dict keyed by arm): a block with a
    # duplicate or missing load is reported as such, never resolved by picking one load.
    groups: dict[tuple[str, str], dict[int, list[TrialResult]]] = {}
    for trial in trials:
        groups.setdefault((trial.variant_id, trial.case_id), {}).setdefault(trial.block, []).append(trial)
    out: dict[str, Any] = {}
    for (variant_id, case_id), blocks in sorted(groups.items()):
        verdicts = []
        for block, loads in sorted(blocks.items()):
            problem = pair_problem([t.arm for t in loads])
            if problem:
                verdicts.append(BlockVerdict(block, False, [f"not exactly one load per arm: {problem}"],
                                             None, None, None, None, None, None))
                continue
            by_arm = {t.arm: t for t in loads}
            verdicts.append(judge_block(by_arm[FRESH], by_arm[REUSE]))
        usable = [v for v in verdicts if v.usable and v.relative_gain is not None]
        gains = [float(v.relative_gain) for v in usable if v.relative_gain is not None]
        interval: list[float] | None = None
        mean = statistics.fmean(gains) if gains else None
        if mean is not None and len(gains) >= 2:
            se = statistics.stdev(gains) / math.sqrt(len(gains))
            critical = t_quantile(1.0 - alpha, len(gains) - 1)
            interval = [mean - critical * se, mean + critical * se]
        states = [v.fresh_stayed_fresh for v in verdicts]
        fresh = False if False in states else (True if states and all(s is True for s in states) else None)
        expect = next(iter(blocks.values()))[0].expect_reuse
        out[f"{variant_id}/{case_id}"] = {
            "variant": variant_id, "case": case_id, "role": "reuse" if expect else "isolation control",
            "blocks": len(verdicts), "usable_blocks": len(usable),
            "clean": bool(verdicts) and all(v.usable for v in verdicts),
            "fresh_arm_stayed_fresh": fresh,
            # For the isolation control this is the noise floor, never a reuse gain.
            "mean_relative_ttft_gain": mean, "interval": interval, "interval_level": 1.0 - 2 * alpha,
            "cached_pieces": [v.cached for v in verdicts],
            "answers_identical": [v.answers_identical for v in verdicts],
            "verdicts": [asdict(v) for v in verdicts],
        }
        group = [t for loads in blocks.values() for t in loads]
        if any(t.identity.get("server_cache_reuse") for t in group):
            out[f"{variant_id}/{case_id}"]["shifted_reuse"] = shifted_reuse_status(group)
    return out


#: Version of the E7 classification (``shifted_reuse_status``). 2 (17 Oct, after the job 127
#: results review): complete pairs, complete work evidence and the priming slot are
#: required before anything is called testable. 3 (17 Oct, second results review): every
#: load is kept; a block with a duplicate or missing load is named as such and is never
#: resolved by picking one load (``pair_problem``).
E7_ANALYSIS_VERSION = "e7-status-3"


def shifted_reuse_status(trials: Sequence[TrialResult]) -> dict[str, Any]:
    """Whether E7 can be read for one ``--cache-reuse`` variant and case.

    **Prerequisites, all required before anything is called testable:**

    1. complete pairs: every block has exactly one fresh-arm and one reuse-arm load;
    2. a shiftable chunk in the engine's own tokens in every load (``shift_eligible`` > 0);
    3. complete work evidence in every load, for its priming and its measured reading
       (``work_evidence``: counts that add up to the engine tokenizer's count, the asked
       number of written pieces, a completed response, a valid time);
    4. every reuse-arm load answered in the slot its priming request used, both slot
       identities reported (otherwise the earlier work was never reachable);
    5. per load, its own evidence that the engine kept the option on: shifted reuse in
       that load's own counters (reuse beyond the shared start plus the written piece;
       only a reuse-arm load can show it) or that load's complete startup log without
       the disable warning (``cache_reuse_activation``). One load's evidence never counts
       for another.

    Only then is the group testable, and the status says in how many reuse-arm loads
    shifted reuse was observed: absence of shifted reuse is a valid negative result only
    here. Activation, observed shifted work, answer identity (``answers_identical``, per
    block) and any time difference (the summary's usable-block gain) stay separate."""
    loads: list[dict[str, Any]] = []
    blocks: dict[int, list[str]] = {}
    for t in trials:
        blocks.setdefault(t.block, []).append(t.arm)
        problems: list[str] = []
        for label, reading, total, pieces in (("priming", t.prime, t.prime_tokens, PRIMING_PIECES),
                                              ("measured", t.measured, t.measured_tokens, MEASURED_PIECES)):
            problems.extend(f"{label}: {p}" for p in work_evidence(reading, expected_total=total,
                                                                    expected_generated=pieces)[1])
        slot_ok: bool | None = None
        if t.arm == REUSE:
            p_slot, m_slot = _count(t.prime.get("slot")), _count(t.measured.get("slot"))
            slot_ok = p_slot is not None and m_slot is not None and p_slot == m_slot
        beyond: int | None = None
        if t.arm == REUSE and t.measured:
            cached = counters_reuse(t.measured, expected_total=t.measured_tokens)
            beyond = None if cached is None else cached - (t.shared_tokens + PRIMING_PIECES)
        observed = bool(beyond is not None and beyond > 0 and slot_ok and not problems)
        if t.activation.startswith("disabled"):
            evidence = "turned off by the engine"
        elif observed:
            evidence = "on: shifted reuse in this load's counters"
        elif t.activation.startswith("not disabled"):
            evidence = "not turned off: this load's complete startup log"
        else:
            evidence = "unverified"
        loads.append({"block": t.block, "arm": t.arm, "eligible_pieces": t.shift_eligible,
                      "work_evidence_problems": problems, "in_priming_slot": slot_ok,
                      "slots": [t.prime.get("slot"), t.measured.get("slot")] if t.measured or t.prime else [],
                      "activation": t.activation, "reuse_beyond_shared_start": beyond,
                      "observed": observed, "evidence": evidence})
    reuse_loads = [load for load in loads if load["arm"] == REUSE]
    pair_problems = {b: pair_problem(arms) for b, arms in sorted(blocks.items())}
    incomplete_pairs = [b for b, problem in pair_problems.items() if problem]
    pair_detail = "; ".join(f"block {b + 1}: {pair_problems[b]}" for b in incomplete_pairs)
    bad_evidence = sum(1 for load in loads if load["work_evidence_problems"])
    off_slot = sum(1 for load in reuse_loads if not load["in_priming_slot"])
    observed_count = sum(1 for load in reuse_loads if load["observed"])
    unverified = sum(1 for load in loads if load["evidence"] == "unverified")
    # Every failing prerequisite, in order; the status names the first.
    failed: list[str] = []
    if not loads:
        failed.append("no loads")
    if incomplete_pairs:
        failed.append(f"{len(incomplete_pairs)} of {len(blocks)} blocks are not exactly one fresh-arm and one "
                      f"reuse-arm load ({pair_detail})")
    if any(load["eligible_pieces"] is None for load in loads):
        failed.append("the engine's tokens were not obtained for every load")
    elif any(load["eligible_pieces"] == 0 for load in loads):
        failed.append("no shiftable chunk in the engine's tokens at this chunk size")
    if bad_evidence:
        failed.append(f"incomplete work evidence in {bad_evidence} of {len(loads)} loads")
    if off_slot:
        failed.append(f"{off_slot} of {len(reuse_loads)} reuse-arm loads did not answer in the priming slot "
                      "(or did not report it)")
    if any(load["evidence"] == "turned off by the engine" for load in loads):
        failed.append("the engine turned --cache-reuse off")
    elif unverified:
        failed.append(f"activation unverified in {unverified} of {len(loads)} loads")
    if not loads:
        state = "not testable: no loads"
    elif incomplete_pairs:
        state = (f"not testable: {len(incomplete_pairs)} of {len(blocks)} blocks are not exactly one fresh-arm "
                 f"and one reuse-arm load ({pair_detail})")
    elif any(load["eligible_pieces"] is None for load in loads):
        state = "not testable: the engine's tokens were not obtained for every load"
    elif any(load["eligible_pieces"] == 0 for load in loads):
        state = "not testable: no shiftable chunk in the engine's tokens at this chunk size"
    elif bad_evidence:
        state = f"not testable: incomplete work evidence in {bad_evidence} of {len(loads)} loads"
    elif off_slot:
        state = (f"not testable: {off_slot} of {len(reuse_loads)} reuse-arm loads did not answer in the "
                 "priming slot (or did not report it)")
    elif any(load["evidence"] == "turned off by the engine" for load in loads):
        state = "not testable: the engine turned --cache-reuse off"
    elif unverified:
        state = f"not testable: activation unverified in {unverified} of {len(loads)} loads"
    elif observed_count == len(reuse_loads):
        state = "testable: shifted reuse observed in every reuse-arm load"
    elif observed_count == 0:
        state = "testable: kept on in every load (startup log), no shifted reuse observed"
    else:
        state = (f"testable: kept on in every load; shifted reuse observed in {observed_count} of "
                 f"{len(reuse_loads)} reuse-arm loads")
    answers: dict[int, bool | None] = {}
    for b in blocks:
        if pair_problems[b]:
            answers[b] = None  # never compare against a load picked from a non-unique pair
            continue
        texts = {t.arm: t.measured.get("text") for t in trials if t.block == b}
        f_text, r_text = texts.get(FRESH), texts.get(REUSE)
        answers[b] = None if f_text is None or r_text is None else f_text == r_text
    return {"e7": state, "analysis": E7_ANALYSIS_VERSION, "prerequisites_failed": failed, "loads": loads,
            "eligible_pieces": [load["eligible_pieces"] for load in loads],
            "activation": sorted({load["activation"] for load in loads}),
            "reuse_beyond_shared_start": [load["reuse_beyond_shared_start"] for load in reuse_loads],
            "answers_identical": [answers[b] for b in sorted(answers)]}


def validity_breach(trial: TrialResult) -> str | None:
    """The pre-registered immediate stops E1 and E4 for a trial, from the engine's own
    complete, consistent prompt counters, for every reading: the priming request (a
    fresh load: nothing to reuse) and the measured request. Missing counters are not a
    breach; they make the block unusable."""
    if trial.breach:
        return trial.breach
    label = f"({trial.variant_id}/{trial.case_id} block {trial.block + 1}, {trial.arm} arm)"
    return (_breach(trial.prime, expected_total=trial.prime_tokens, bound=0, fresh=False,
                    what=f"the priming request on a fresh load {label}")
            or _breach(trial.measured, expected_total=trial.measured_tokens, bound=trial.bound,
                       fresh=trial.arm == FRESH, what=f"the measured request {label}"))


# --------------------------------------------------------------------------- #
# Cancellation                                                                  #
# --------------------------------------------------------------------------- #
def cancellation_check(engine: ReuseEngine, case: ReuseCase, variant: EngineVariant, *,
                       reference: dict[str, Any] | None, measured_pieces: int = MEASURED_PIECES,
                       after_chunks: int = 2) -> dict[str, Any]:
    """Prime, start the measured request with reuse and walk away after a few pieces,
    confirm the engine ended that answer, then ask again with reuse.

    ``passed`` only when the client itself cut the answer off (``CUT_BY_CLIENT``, after
    at least one piece), the engine confirmed every slot idle within the wait, the engine
    is healthy, a valid fresh reference answer from this run exists, and the repeat has
    complete work evidence and the reference's exact answer. The repeat is sent only
    when all of that before it holds: once the earlier answer is known to be still
    running, the engine is unhealthy, or a pass is already impossible, the check goes
    straight to cleanup. Missing evidence makes it ``inconclusive``; a busy or unhealthy
    engine, a wrong answer or unclean cleanup makes it ``failed``. A validity gate
    (``breach``) on its readings fails it and stops the run."""
    result: dict[str, Any] = {"case": case.case_id, "variant": variant.variant_id, "problems": [], "missing": [],
                              "breach": None, "repeat": "not sent"}
    problems: list[str] = result["problems"]
    missing: list[str] = result["missing"]
    if reference is None or not reference.get("text") or reference.get("completed") is not True:
        missing.append("no valid fresh reference answer from this run")
    interrupted = False
    try:
        engine.start(variant)
        prime_tokens = engine.tokenize(case.prime)
        measured_tokens = engine.tokenize(case.measured)
        prime = asdict(engine.generate(case.prime, cache_prompt=True, n_predict=PRIMING_PIECES, seed=SEED))
        result["prime"] = prime
        result["breach"] = _breach(prime, expected_total=len(prime_tokens), bound=0, fresh=False,
                                   what="the cancellation check's priming request on a fresh load")
        if result["breach"]:
            problems.append(result["breach"])
        else:
            abandoned = engine.abandon(case.measured, cache_prompt=True, n_predict=measured_pieces, seed=SEED,
                                       after_chunks=after_chunks)
            result["abandoned"] = asdict(abandoned)
            if abandoned.ending != CUT_BY_CLIENT or abandoned.pieces < 1:
                missing.append(f"no answer was interrupted by the client ({abandoned.ending}, "
                               f"{abandoned.pieces} pieces{'; ' + abandoned.error if abandoned.error else ''})")
            if abandoned.idle_confirmed is None:
                missing.append(f"the engine could not confirm the abandoned answer ended ({abandoned.idle_detail})")
            elif not abandoned.idle_confirmed:
                problems.append(f"the abandoned answer was not confirmed ended within the wait ({abandoned.idle_detail})")
            if not problems:
                result["healthy_after_abandon"] = engine.healthy()
                if not result["healthy_after_abandon"]:
                    problems.append("engine not healthy after a cancelled answer")
            if problems or missing:
                result["repeat"] = "not sent: " + "; ".join(problems or missing)
            else:
                result["repeat"] = "sent"
                after = asdict(engine.generate(case.measured, cache_prompt=True, n_predict=measured_pieces, seed=SEED))
                result["after"] = after
                # The engine holds the priming prompt and the abandoned request's prompt:
                # everything but the last piece of the repeat could be reused, no more.
                result["breach"] = _breach(after, expected_total=len(measured_tokens),
                                           bound=max(len(measured_tokens) - 1, 0), fresh=False,
                                           what="the repeat after a cancelled answer")
                if result["breach"]:
                    problems.append(result["breach"])
                cached, work = work_evidence(after, expected_total=len(measured_tokens),
                                             expected_generated=measured_pieces)
                result["cached_after"] = cached
                missing.extend(f"repeat: {p}" for p in work)
                if reference is not None and reference.get("text"):
                    result["answer_matches_fresh"] = after.get("text") == reference["text"]
                    if not result["answer_matches_fresh"]:
                        problems.append("the answer after a cancelled answer differs from the fresh answer")
    except KeyboardInterrupt:
        interrupted = True
        problems.append("interrupted")
    except Exception as exc:  # noqa: BLE001 - recorded; the engine is stopped below
        problems.append(f"cancellation check failed: {type(exc).__name__}: {exc}")
    finally:
        try:
            cleanup = engine.stop()
        except KeyboardInterrupt:
            interrupted = True
            cleanup = Cleanup(False, None, "interrupted while stopping")
        except Exception as exc:  # noqa: BLE001
            cleanup = Cleanup(False, None, f"stop failed: {type(exc).__name__}: {exc}")
        result["cleanup"] = asdict(cleanup)
        result["cleanup_clean"] = cleanup.clean
        if not cleanup.clean:
            problems.append(f"cleanup not complete: {cleanup.problem or asdict(cleanup)}")
    result["interrupted"] = interrupted
    result["verdict"] = "failed" if problems else ("inconclusive" if missing else "passed")
    result["passed"] = result["verdict"] == "passed"
    return result


# --------------------------------------------------------------------------- #
# The whole evaluation for one model                                            #
# --------------------------------------------------------------------------- #
def plan_trials(cases: Sequence[ReuseCase], variants: Sequence[EngineVariant], blocks: int
                ) -> list[tuple[EngineVariant, ReuseCase, int, str]]:
    """Every trial, in run order: for each variant and case, ``blocks`` blocks whose arm
    order alternates (fresh first in even blocks, reuse first in odd ones)."""
    plan = []
    for variant in variants:
        for case in cases:
            if variant.cases and case.case_id not in variant.cases:
                continue
            for block in range(blocks):
                order = ARMS if block % 2 == 0 else tuple(reversed(ARMS))
                for arm in order:
                    plan.append((variant, case, block, arm))
    return plan


def evaluate(engine: ReuseEngine, *, cases: Sequence[ReuseCase], variants: Sequence[EngineVariant], blocks: int,
             context_length: int, configuration: dict[str, Any], run_id: str,
             progress: Callable[[str], None] = lambda _line: None, cancellation: bool = True,
             power: Callable[[], str] = lambda: "not checked", power_check: bool = True,
             checkpoint: Callable[[dict[str, Any]], None] = lambda _record: None) -> dict[str, Any]:
    """Run the planned trials and the cancellation check; return the full record.

    The record is also passed to ``checkpoint`` after every trial, so an interruption
    keeps everything done so far. Status:

    * "done": every planned trial and the cancellation check ran, every cleanup was
      complete, no validity gate fired and the cancellation check passed;
    * "incomplete": a trial failed or the cancellation check was not a pass;
    * "stopped: cleanup incomplete" / "stopped: validity gate" / "stopped: not on mains
      power": the run stopped at once; later trials and the check were not run;
    * "interrupted": stopped by the user (re-raised after the checkpoint).

    ``power_check`` False allows the "not checked" state (simulation or a labelled test
    run); it is recorded."""
    if blocks < 1:
        raise ValueError("at least one block")
    configuration_sha256 = _sha(configuration)
    plan = plan_trials(cases, variants, blocks)
    trials: list[TrialResult] = []
    checks: list[dict[str, Any]] = []
    state: dict[str, Any] = {"status": None, "stop_reason": None}

    def record() -> dict[str, Any]:
        problems = [p for t in trials for p in t.problems] + [p for c in checks for p in c["problems"]]
        return {
            "evaluation": EVALUATION_VERSION, "run_id": run_id, "context_length": context_length,
            "configuration": configuration, "configuration_sha256": configuration_sha256,
            "power_check": "on" if power_check else "off (simulation or labelled test run)",
            "cases": {c.case_id: {"description": c.description, "sha256": c.sha256, "expect_reuse": c.expect_reuse,
                                  "shifted_chunks": c.shifted_chunks} for c in cases},
            "variants": [asdict(v) for v in variants], "blocks": blocks, "planned_trials": len(plan),
            "trials": [asdict(t) for t in trials], "summary": summarize(trials) if trials else {},
            "cancellation": checks, "problems": problems,
            "status": state["status"] or "running", "stop_reason": state["stop_reason"],
            "note": "development evidence only: not acceptance, not Practical admission, not a hardware claim "
                    "beyond this machine, model and setup",
        }

    def stop(status: str, reason: str) -> dict[str, Any]:
        state["status"], state["stop_reason"] = status, reason
        final = record()
        checkpoint(final)
        return final

    try:
        for index, (variant, case, block, arm) in enumerate(plan, start=1):
            before = _power(power)
            if not _power_allowed(before, unchecked_ok=not power_check):
                return stop(STOPPED_POWER, f"power '{before}' before trial {index}")
            progress(f"trial {index}/{len(plan)}: {variant.variant_id} {case.case_id} block {block + 1} {arm}")
            trial = run_trial(engine, case, variant, arm, block=block, context_length=context_length,
                              configuration_sha256=configuration_sha256, power=power)
            trial.power.insert(0, before)
            trials.append(trial)
            checkpoint(record())
            if trial.interrupted:
                stop(INTERRUPTED, f"interrupted during trial {index}")
                raise KeyboardInterrupt
            breach = validity_breach(trial)
            clean = not trial.cleanup or Cleanup(**trial.cleanup).clean
            if breach:
                return stop(STOPPED_VALIDITY, breach + ("" if clean else
                                                       f"; cleanup also incomplete: {trial.cleanup.get('problem')}"))
            if not clean:
                return stop(STOPPED_CLEANUP, f"trial {index}: {trial.cleanup.get('problem') or trial.cleanup}")
            input_problem = getattr(engine, "input_problem", None)
            if input_problem:
                return stop("error", str(input_problem))
            if not _power_allowed(trial.power[-1], unchecked_ok=not power_check):
                return stop(STOPPED_POWER, f"power '{trial.power[-1]}' after trial {index}")
        if cancellation:
            case = next((c for c in cases if c.case_id == "shared_prefix"), cases[0])
            variant = variants[0]
            reference = next((t.measured for t in trials if t.case_id == case.case_id
                              and t.variant_id == variant.variant_id and t.arm == FRESH
                              and not work_evidence(t.measured, expected_total=t.measured_tokens,
                                                    expected_generated=MEASURED_PIECES)[1]), None)
            if not _power_allowed(_power(power), unchecked_ok=not power_check):
                return stop(STOPPED_POWER, "power before the cancellation check")
            progress("cancellation check")
            check = cancellation_check(engine, case, variant, reference=reference)
            checks.append(check)
            checkpoint(record())
            if check["interrupted"]:
                stop(INTERRUPTED, "interrupted during the cancellation check")
                raise KeyboardInterrupt
            if check["breach"]:
                return stop(STOPPED_VALIDITY, check["breach"] + ("" if check["cleanup_clean"] else
                                                                 "; cleanup also incomplete"))
            if not check["cleanup_clean"]:
                return stop(STOPPED_CLEANUP, "cancellation check cleanup")
    except KeyboardInterrupt:
        if state["status"] != INTERRUPTED:
            stop(INTERRUPTED, "interrupted between trials")
        raise
    complete = len(trials) == len(plan) and (not cancellation or bool(checks and checks[0]["passed"]))
    no_problems = not any(t.problems for t in trials)
    return stop(DONE if complete and no_problems else INCOMPLETE,
                "" if complete and no_problems else "a trial failed or the cancellation check did not pass")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


__all__ = [
    "ARMS", "CACHE_REUSE_DISABLED_WARNINGS", "CUT_BY_CLIENT", "DEFAULT_VARIANTS", "DONE", "E7_ANALYSIS_VERSION",
    "ENDED_EARLY",
    "ENGINE_FINISHED", "EVALUATION_VERSION", "FRESH", "INCOMPLETE", "INTERRUPTED", "MODEL_LOADED_MARKER",
    "REQUEST_FAILED", "REUSE", "STOPPED_CLEANUP", "STOPPED_POWER", "STOPPED_VALIDITY", "SWA_FULL_WARNING", "Abandoned", "Cleanup",
    "EngineVariant", "Reading", "ReuseCase", "ReuseEngine", "ReuseIdentity", "TrialResult", "build_cases",
    "cache_reuse_activation", "cancellation_check", "common_prefix", "counters_reuse", "prompt_counts", "evaluate",
    "judge_block", "pair_problem", "plan_trials", "reuse_bound", "run_trial", "shifted_reuse", "shifted_reuse_status", "summarize", "swa_cache_evidence",
    "validity_breach", "work_evidence",
]
