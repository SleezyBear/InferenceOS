"""Optimization paths for ``tune`` (after the graphics card layer path).

Each path looks at one possible source of waste, decides from facts about the model and
the machine (``tune.model_facts``) whether it applies and what is worth trying, tries
those changes from the current best setup, and keeps one only when a fair comparison
(``tune.fair``) shows it really helps. Paths stack: each one starts from the setup the
previous one left.

Paths (Oct 2026):

* ``threads``: processor threads that write the answer. Skipped when the whole model,
  output layer included, is on the graphics card (the processor then only hands work
  to the card). Otherwise every second count from 2 up to all logical cores is tried,
  then the counts next to the best.
* ``reading_threads``: threads for reading the question (``-tb``). Skipped when the
  whole model is on the card. Measured on the long question and must not lose on the
  short one: llama.cpp sends most reading of a long question to the card, but a short
  question is read on the processor.
* ``flash_attention``: on or off (llama.cpp's default is "auto"). Skipped for a model
  with no attention layers. Measured on the long question, where attention matters,
  and must not lose on the short one.
* ``prompt_batch``: how many question pieces are read at once (``-ub``). Sizes from 64
  up to the first size that holds the whole long question; sizes whose working memory
  clearly can't fit on the card are dropped; when the best is at either end of what
  was tried, one more step past it is tried. Must not lose on the short question.

The long test question is sized to the model (see ``Tuner._long_question``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
import re

from inferenceos.profiling.workloads import (
    LONG_PROMPT_PASSAGES,
    BenchmarkWorkload,
    get_workload,
    long_prompt_with,
)
from inferenceos.tune.model_facts import MachineFacts, ModelFacts
from inferenceos.tune.placement import placement

#: Workload id meaning "the long test question sized to this model".
LONG_QUESTION = "long-question"
#: Workload id of the short test question.
SHORT_QUESTION = "short-short"


@dataclass(frozen=True)
class PathContext:
    """What a path may look at when deciding whether it applies and what to try."""

    hardware: HardwareProfile
    model: ModelProfile
    total_layers: int
    current: RuntimeConfiguration
    supported_options: frozenset[str] | None = None
    facts: ModelFacts | None = None
    machine: MachineFacts | None = None
    #: Share of the model (by size) on the processor with the current setup.
    processor_share: float | None = None
    #: Length of the long test question in pieces (tokens), when known.
    question_tokens: int | None = None
    context_length: int | None = None
    #: The engine's full version line and type (e.g. "vulkan"), when known.
    engine_version: str | None = None
    engine: str | None = None
    #: Engine switches that passed the timed checks (took effect, same answers), as
    #: (full switch text to try, plain label) that passed the timed checks; None = not
    #: checked yet (``tune/engine_switches.py``).
    checked_switches: tuple[tuple[str, str], ...] | None = None

    @property
    def physical_cores(self) -> int:
        if self.machine is not None:
            return self.machine.physical_cores
        return max(int(self.hardware.cpu.physical_cores or 1), 1)

    @property
    def logical_cores(self) -> int:
        if self.machine is not None:
            return self.machine.logical_cores
        physical = self.physical_cores
        return max(int(self.hardware.cpu.logical_cores or physical), physical)


@dataclass(frozen=True)
class Candidate:
    value: int | bool | str
    label: str


#: (context, best value so far, values already tried) -> more candidates to try.
Refiner = Callable[[PathContext, "int | bool | str", "set[int | bool | str]"], "list[Candidate]"]


@dataclass(frozen=True)
class OptionPath:
    """One self-contained optimization path over a single setting."""

    name: str
    label: str
    setting: str
    card_id: str
    description: str
    #: llama-server option this path needs (checked against ``--help`` when known).
    option: str
    #: Workload id this path is measured with (None = tune's workload,
    #: ``LONG_QUESTION`` = the long question sized to the model).
    workload_id: str | None
    applies: Callable[[PathContext], tuple[bool, str]]
    candidates: Callable[[PathContext], list[Candidate]]
    #: A second workload a change must not lose on (None = none).
    guard_workload_id: str | None = None
    #: More values to try around the best one (None = none).
    refine: Refiner | None = None
    #: Plain-language notes on what was left out and why.
    notes: Callable[[PathContext], list[str]] | None = None
    #: The value llama.cpp uses when the setting is not given.
    default_value: Callable[[PathContext], int | bool | str | None] | None = None

    def check(self, context: PathContext) -> tuple[bool, str]:
        if (
            self.option
            and context.supported_options is not None
            and self.option not in context.supported_options
        ):
            return False, f"this llama.cpp does not have the {self.option} option"
        return self.applies(context)

    def current_value(self, context: PathContext) -> int | bool | str | None:
        value = getattr(context.current, self.setting)
        if value is None and self.default_value is not None:
            return self.default_value(context)
        return value  # type: ignore[no-any-return]


def _on_card_only(context: PathContext) -> bool:
    return context.processor_share is not None and context.processor_share <= 0.0


_ON_CARD_REASON = (
    "everything, output step included, is on the graphics card; the processor only "
    "hands the work to the card"
)


def _by_closeness(values: set[int], centre: int) -> list[int]:
    return sorted(values, key=lambda value: (abs(value - centre), value))


def _neighbours(value: int, low: int, high: int, tried: set[int | bool | str]) -> list[int]:
    return [
        item
        for item in (value - 1, value + 1)
        if low <= item <= high and item not in tried
    ]


# --------------------------------------------------------------------------- #
# threads                                                                     #
# --------------------------------------------------------------------------- #
def _threads_default(context: PathContext) -> int:
    return context.physical_cores


def thread_candidates(context: PathContext) -> list[Candidate]:
    """Every second count from 2 up to all logical cores (plus the physical count),
    nearest to llama.cpp's default (the physical count) first, so a time limit cuts
    the least likely ones. The current value is never re-tried."""
    physical, logical = context.physical_cores, context.logical_cores
    current = context.current.threads if context.current.threads is not None else physical
    values = set(range(2, logical + 1, 2)) | {physical, logical}
    if logical < 2:
        values = {1}
    values.discard(current)
    return [Candidate(value=value, label=f"{value} threads") for value in _by_closeness(values, physical)]


def thread_refine(
    context: PathContext, best: int | bool | str, tried: set[int | bool | str]
) -> list[Candidate]:
    """The counts next to the best one that were not tried yet."""
    if isinstance(best, bool):
        return []
    return [
        Candidate(value=value, label=f"{value} threads")
        for value in _neighbours(int(best), 1, context.logical_cores, tried)
    ]


def _threads_applies(context: PathContext) -> tuple[bool, str]:
    if context.physical_cores < 2:
        return False, "only one processor core"
    if _on_card_only(context):
        return False, _ON_CARD_REASON
    if context.processor_share is None:
        return True, "the processor may do part of the work"
    return True, f"the processor works through {context.processor_share:.0%} of the model per word"


# --------------------------------------------------------------------------- #
# reading threads (prompt)                                                    #
# --------------------------------------------------------------------------- #
def _reading_default(context: PathContext) -> int:
    return context.current.threads if context.current.threads is not None else context.physical_cores


def reading_thread_candidates(context: PathContext) -> list[Candidate]:
    """Threads for reading the question, separate from the writing threads.

    Unset means "the same as the writing threads", which is never re-tried.
    """
    physical, logical = context.physical_cores, context.logical_cores
    current = (
        context.current.threads_batch
        if context.current.threads_batch is not None
        else _reading_default(context)
    )
    values = {max(physical // 2, 1), max(physical - 2, 1), physical, logical}
    values.discard(current)
    return [
        Candidate(value=value, label=f"{value} reading threads")
        for value in _by_closeness(values, physical)
    ]


def reading_thread_refine(
    context: PathContext, best: int | bool | str, tried: set[int | bool | str]
) -> list[Candidate]:
    if isinstance(best, bool):
        return []
    return [
        Candidate(value=value, label=f"{value} reading threads")
        for value in _neighbours(int(best), 1, context.logical_cores, tried)
    ]


def _reading_threads_applies(context: PathContext) -> tuple[bool, str]:
    if context.physical_cores < 2:
        return False, "only one processor core"
    if _on_card_only(context):
        return False, _ON_CARD_REASON
    return True, "short questions are read on the processor; long ones partly"


# --------------------------------------------------------------------------- #
# flash attention                                                             #
# --------------------------------------------------------------------------- #
def flash_candidates(context: PathContext) -> list[Candidate]:
    current = context.current.flash_attention
    values: list[bool] = [True, False] if current is None else [not current]
    return [Candidate(value=value, label="flash attention " + ("on" if value else "off"))
            for value in values]


def _flash_applies(context: PathContext) -> tuple[bool, str]:
    facts = context.facts
    if facts is not None and facts.attention_layers is not None:
        if facts.attention_layers == 0:
            return False, "this model has no attention layers"
        if facts.other_layers:
            return True, (
                f"{facts.attention_layers} of {facts.total_layers} layers use attention, "
                "so the effect may be small"
            )
        return True, "every layer uses attention"
    return True, "llama.cpp decides flash attention by itself unless told"


# --------------------------------------------------------------------------- #
# prompt batch                                                                #
# --------------------------------------------------------------------------- #
PROMPT_BATCH_DEFAULT = 512
#: llama.cpp's default logical batch; the prompt batch can't be larger.
PROMPT_BATCH_MAX = 2048
PROMPT_BATCH_MIN = 16
PROMPT_BATCH_VALUES = (64, 128, 256, 512, 1024, 2048)


def batch_working_memory_bytes(context: PathContext, size: int) -> int | None:
    """An estimate of the extra card memory one prompt batch of ``size`` needs.

    Only one part is counted: with flash attention turned off, each attention layer on
    the card builds a table of ``size x context x heads`` numbers (4 bytes each) while
    reading. With llama.cpp's "auto" flash attention is usually on and the table is not
    built, so nothing is counted. This is an estimate, not a guaranteed bound; it only
    drops sizes that clearly cannot fit. ``None`` when unknown.
    """
    facts = context.facts
    contexts = context.context_length or context.current.context_length
    if facts is None or not contexts:
        return None
    where = placement(context.current.gpu_offload_layers, facts.total_layers)
    if (
        context.current.flash_attention is False
        and facts.attention_heads
        and where.blocks_on_card
        and facts.attention_layers != 0
    ):
        return size * int(contexts) * int(facts.attention_heads) * 4
    return None


def _batch_sizes(context: PathContext) -> tuple[list[int], list[str]]:
    notes: list[str] = []
    sizes = list(PROMPT_BATCH_VALUES)
    question = context.question_tokens
    if question:
        # Sizes at or above the question's length all read it in one go: keep only the
        # smallest of them.
        whole = next((size for size in sizes if size >= question), PROMPT_BATCH_MAX)
        dropped = [size for size in sizes if size > whole]
        sizes = [size for size in sizes if size <= whole]
        if dropped:
            notes.append(
                f"sizes above {whole} not tried: the long question is about {question} "
                f"pieces, so they would read it the same way as {whole}"
            )
    free = context.machine.free_card_bytes if context.machine is not None else None
    if free is not None and context.machine is not None and context.machine.card:
        too_big = []
        for size in list(sizes):
            need = batch_working_memory_bytes(context, size)
            if need is not None and need > free:
                too_big.append(size)
                sizes.remove(size)
        if too_big:
            notes.append(
                "not tried, their working memory would not fit in the free graphics "
                f"memory ({free / 1024**3:.1f} GB): " + ", ".join(str(s) for s in too_big)
            )
    return sizes, notes


def prompt_batch_candidates(context: PathContext) -> list[Candidate]:
    current = context.current.ubatch_size or PROMPT_BATCH_DEFAULT
    sizes, _ = _batch_sizes(context)
    values = [value for value in sizes if value != current]
    return [Candidate(value=value, label=f"prompt batch {value}") for value in sorted(values)]


def prompt_batch_notes(context: PathContext) -> list[str]:
    return _batch_sizes(context)[1]


def prompt_batch_refine(
    context: PathContext, best: int | bool | str, tried: set[int | bool | str]
) -> list[Candidate]:
    """One step past the best size when it is at either end of what was tried."""
    if isinstance(best, bool):
        return []
    sizes = sorted(int(value) for value in tried if not isinstance(value, bool))
    if not sizes:
        return []
    best = int(best)
    out: list[int] = []
    if best == sizes[0] and best // 2 >= PROMPT_BATCH_MIN:
        out.append(best // 2)
    question = context.question_tokens
    if best == sizes[-1] and best * 2 <= PROMPT_BATCH_MAX and (question is None or best < question):
        larger = best * 2
        need = batch_working_memory_bytes(context, larger)
        free = context.machine.free_card_bytes if context.machine is not None else None
        if need is None or free is None or need <= free:
            out.append(larger)
    return [
        Candidate(value=value, label=f"prompt batch {value}") for value in out if value not in tried
    ]


def _prompt_batch_applies(context: PathContext) -> tuple[bool, str]:
    return True, "how the question is read changes the time to the first word"


THREADS = OptionPath(
    name="threads",
    label="processor threads",
    setting="threads",
    card_id="runtime.reduce_threads",
    description="Processor threads",
    option="--threads",
    workload_id=None,
    applies=_threads_applies,
    candidates=thread_candidates,
    refine=thread_refine,
    default_value=_threads_default,
)
FLASH_ATTENTION = OptionPath(
    name="flash_attention",
    label="flash attention",
    setting="flash_attention",
    card_id="runtime.flash_attention",
    description="Flash attention",
    option="--flash-attn",
    workload_id=LONG_QUESTION,
    applies=_flash_applies,
    candidates=flash_candidates,
    guard_workload_id=SHORT_QUESTION,
)
PROMPT_BATCH = OptionPath(
    name="prompt_batch",
    label="prompt batch size",
    setting="ubatch_size",
    card_id="runtime.prompt_batch",
    description="Prompt batch size",
    option="--ubatch-size",
    workload_id=LONG_QUESTION,
    applies=_prompt_batch_applies,
    candidates=prompt_batch_candidates,
    # A bigger batch can slow writing for every question (found on the 9B, Oct 2026:
    # 2048 read a 2,000-piece question 12% sooner but wrote 5% slower on short ones).
    guard_workload_id=SHORT_QUESTION,
    refine=prompt_batch_refine,
    notes=prompt_batch_notes,
    default_value=lambda context: PROMPT_BATCH_DEFAULT,
)
READING_THREADS = OptionPath(
    name="reading_threads",
    label="reading threads",
    setting="threads_batch",
    card_id="runtime.reading_threads",
    description="Threads for reading the question",
    option="--threads-batch",
    workload_id=LONG_QUESTION,
    applies=_reading_threads_applies,
    candidates=reading_thread_candidates,
    guard_workload_id=SHORT_QUESTION,
    refine=reading_thread_refine,
    default_value=_reading_default,
)

def _switches_apply(context: PathContext) -> tuple[bool, str]:
    from inferenceos.tune.engine_switches import switches_for

    if context.machine is not None and not context.machine.card:
        return False, "no graphics card is used, and the listed switches are for the card's engine"
    if context.current.gpu_offload_layers in (None, 0):
        return False, "nothing runs on the graphics card with the current setup"
    if context.engine is None or not switches_for(context.engine, context.engine_version):
        return False, (
            "no engine switches are listed for this engine build "
            f"({context.engine or 'unknown engine'}, {context.engine_version or 'unknown build'})"
        )
    return True, "engine switches listed for this engine build"


def _switch_candidates(context: PathContext) -> list[Candidate]:
    """The switch setups that passed the timed checks (their full switch text: a switch
    set with its conflicts removed, or one put back to the engine's default)."""
    out: list[Candidate] = []
    for text, label in context.checked_switches or ():
        if (text or None) == (context.current.engine_switches or None):
            continue
        out.append(Candidate(text, label))
    return out


ENGINE_SWITCHES = OptionPath(
    name="engine_switches",
    label="engine switches",
    setting="engine_switches",
    card_id="engine.switches",
    description="Settings inside the engine, from the list for this engine build.",
    option="",
    workload_id=None,
    applies=_switches_apply,
    candidates=_switch_candidates,
    guard_workload_id=LONG_QUESTION,
)


ALL_PATHS: tuple[OptionPath, ...] = (
    THREADS,
    READING_THREADS,
    FLASH_ATTENTION,
    PROMPT_BATCH,
    ENGINE_SWITCHES,
)


STEADY_SUFFIX = "-steady"
_LONG_QUESTION_ID = re.compile(r"long-question-(\d+)x")


def steady_workload(workload_id: str, *, default: BenchmarkWorkload) -> BenchmarkWorkload:
    """A built-in workload by id; ``<id>-steady`` is the same prompt but always writes
    the full answer length (end-of-answer ignored), so a model that stops at once on
    that prompt still gives a first-word time and a writing speed."""
    base_id = workload_id[: -len(STEADY_SUFFIX)] if workload_id.endswith(STEADY_SUFFIX) else workload_id
    match = _LONG_QUESTION_ID.fullmatch(base_id)
    if match is not None:
        long_base = get_workload("long-prefill")
        built = long_question_workload(
            int(match.group(1)),
            base=long_base,
            per_passage=len(long_base.prompt) / 5 / LONG_PROMPT_PASSAGES,
        )
        return built if base_id != workload_id else replace(
            built, workload_id=base_id, ignore_eos=long_base.ignore_eos
        )
    try:
        base = get_workload(base_id)
    except KeyError:
        return default
    if base_id == workload_id:
        return base
    return replace(base, workload_id=workload_id, ignore_eos=True)


def long_question_workload(passages: int, *, base: BenchmarkWorkload, per_passage: float) -> BenchmarkWorkload:
    """The long question with ``passages`` copies of the passage, writing the full answer
    length. Its id (``long-question-<N>x-steady``) can be given to ``steady_workload``
    to rebuild it (e.g. by the re-check script)."""
    passages = max(int(passages), 1)
    return replace(
        base,
        workload_id=f"long-question-{passages}x-steady",
        description=f"long question ({passages} passages), full answer length",
        prompt=long_prompt_with(passages),
        min_context_tokens=int(passages * per_passage) + base.n_predict + 16,
        ignore_eos=True,
    )


def paths_by_name(names: list[str] | tuple[str, ...]) -> list[OptionPath]:
    known = {path.name: path for path in ALL_PATHS}
    chosen: list[OptionPath] = []
    for name in names:
        key = name.strip()
        if key in known and known[key] not in chosen:
            chosen.append(known[key])
    return chosen


def apply_candidate(
    config: RuntimeConfiguration, path: OptionPath, candidate: Candidate
) -> RuntimeConfiguration:
    value: int | bool | str | None = candidate.value
    if path.setting == "engine_switches" and value == "":
        value = None  # no switches: the engine's defaults
    return config.model_copy(update={path.setting: value})


__all__ = [
    "ALL_PATHS",
    "Candidate",
    "ENGINE_SWITCHES",
    "FLASH_ATTENTION",
    "LONG_QUESTION",
    "OptionPath",
    "PROMPT_BATCH",
    "PathContext",
    "READING_THREADS",
    "SHORT_QUESTION",
    "THREADS",
    "apply_candidate",
    "batch_working_memory_bytes",
    "flash_candidates",
    "long_question_workload",
    "paths_by_name",
    "prompt_batch_candidates",
    "prompt_batch_notes",
    "prompt_batch_refine",
    "reading_thread_candidates",
    "reading_thread_refine",
    "steady_workload",
    "thread_candidates",
    "thread_refine",
]
