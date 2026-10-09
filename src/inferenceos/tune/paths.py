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
from typing import TYPE_CHECKING

from inferenceos.contracts.capability import Support
from inferenceos.contracts.experiment import ExperimentMeasurement, RuntimeConfiguration
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
import re

from inferenceos.profiling.workloads import (
    LONG_PROMPT_PASSAGES,
    BenchmarkWorkload,
    get_workload,
    long_prompt_with,
)
from inferenceos.tune import applicability
from inferenceos.tune.declarations import PathDeclaration, declaration
from inferenceos.tune.model_facts import MachineFacts, ModelFacts
from inferenceos.tune.placement import placement

if TYPE_CHECKING:
    from inferenceos.tune.tensor_placement import PlacementPlan as TensorPlan

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
    #: Room on the card for model weights with the current setup's conversation memory
    #: (free memory minus the safety margins); None = unknown.
    card_budget_bytes: int | None = None
    #: The engine's own "does it accept this setting and value" (help text); None =
    #: not known. Answers only the first of the three questions.
    accepts: Callable[[str, object], tuple[Support, str]] | None = None
    #: Engine switches that passed the timed checks (took effect, same answers), as
    #: (full switch text to try, plain label) that passed the timed checks; None = not
    #: checked yet (``tune/engine_switches.py``).
    checked_switches: tuple[tuple[str, str], ...] | None = None
    #: Whether a hardware probe has confirmed block-kind grouped placement runs
    #: correctly on this engine (``tune_grouped_placement_probed``); until then grouped
    #: placements are screened only (reviewer, 6 Oct 2026).
    grouped_placement_probed: bool = False
    #: Which pieces the engine moved, read from a separate load at its debug log level
    #: ("<layers>|<overrides>" -> piece -> buffer; None = that load listed none). Used
    #: as the placement's identity when the measured runs (normal log level) do not
    #: list them; the measured runs' own buffer totals must still agree.
    piece_lines: dict[str, dict[str, str] | None] | None = None

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
    #: Other settings that change together with this one, atomically (e.g. the layer
    #: count that goes with a placement, or flash attention that a quantized value
    #: cache needs). Empty for single-setting candidates.
    also: tuple[tuple[str, int | bool | str | None], ...] = ()
    #: When set, the candidate is measured and reported but never accepted, with this
    #: reason (e.g. a placement kind that needs a hardware probe first). Not part of
    #: the candidate's identity.
    screen_only: str | None = None

    @property
    def key(self) -> tuple[object, ...]:
        """Identity of what is tried (the value alone is not enough with ``also``)."""
        return (self.value, self.also)


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
    #: Workload id this path is measured with (None = tune's workload,
    #: ``LONG_QUESTION`` = the long question sized to the model).
    workload_id: str | None
    candidates: Callable[[PathContext], list[Candidate]]
    #: A second workload a change must not lose on (None = none).
    guard_workload_id: str | None = None
    #: More values to try around the best one (None = none).
    refine: Refiner | None = None
    #: Plain-language notes on what was left out and why.
    notes: Callable[[PathContext], list[str]] | None = None
    #: The value llama.cpp uses when the setting is not given.
    default_value: Callable[[PathContext], int | bool | str | None] | None = None
    #: Did a measured candidate really run as requested? From what the engine printed
    #: (``SHOWN`` / ``NOT_SEEN`` / ``UNVERIFIED``, with words). None = the path's
    #: setting is not observable this way (then it is not checked here).
    verify: Callable[[PathContext, RuntimeConfiguration, ExperimentMeasurement], tuple[str, str]] | None = None
    #: More candidates around the best one, for paths whose candidates carry ``also``.
    refine_candidates: Callable[[PathContext, Candidate, set[tuple[object, ...]]], list[Candidate]] | None = None

    @property
    def declaration(self) -> PathDeclaration:
        """This path's canonical declaration (``tune/declarations.py``), the one authority
        for its engine option and applicability rule (C3)."""
        return declaration(self.name)

    @property
    def option(self) -> str:
        """llama-server option this path needs (checked against ``--help`` when known)."""
        return self.declaration.engine_option

    @property
    def applies(self) -> Callable[[PathContext], tuple[bool, str]]:
        """The path's own applicability rule, resolved from its declared handler id."""
        assert self.declaration.applicability is not None
        return applicability.resolve(self.declaration.applicability)

    def check(self, context: PathContext) -> tuple[bool, str]:
        """A known missing engine option rejects the path before its rule runs; otherwise
        the rule decides (``tune/applicability.check``; unchanged semantics)."""
        return applicability.check(self.declaration, context)

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


@applicability.handler("threads.processor_work")
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


@applicability.handler("reading_threads.processor_reading")
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


@applicability.handler("flash_attention.attention_layers")
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


@applicability.handler("prompt_batch.always")
def _prompt_batch_applies(context: PathContext) -> tuple[bool, str]:
    return True, "how the question is read changes the time to the first word"


THREADS = OptionPath(
    name="threads",
    label="processor threads",
    setting="threads",
    card_id="runtime.reduce_threads",
    description="Processor threads",
    workload_id=None,
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
    workload_id=LONG_QUESTION,
    candidates=flash_candidates,
    guard_workload_id=SHORT_QUESTION,
)
PROMPT_BATCH = OptionPath(
    name="prompt_batch",
    label="prompt batch size",
    setting="ubatch_size",
    card_id="runtime.prompt_batch",
    description="Prompt batch size",
    workload_id=LONG_QUESTION,
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
    workload_id=LONG_QUESTION,
    candidates=reading_thread_candidates,
    guard_workload_id=SHORT_QUESTION,
    refine=reading_thread_refine,
    default_value=_reading_default,
)

@applicability.handler("engine_switches.listed_for_build")
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
    workload_id=None,
    candidates=_switch_candidates,
    guard_workload_id=LONG_QUESTION,
)


# --------------------------------------------------------------------------- #
# per-piece placement                                                          #
# --------------------------------------------------------------------------- #
def observed_settings(measurement: ExperimentMeasurement) -> dict[str, object] | None:
    """What the engine printed about how it was set up, from a measurement's runs
    (the first run that has it; ``None`` when no run does)."""
    for run in measurement.runs:
        observed = getattr(run.context.config, "observed", None)
        if observed:
            return dict(observed)
    return None


@applicability.handler("placement.piece_facts")
def _placement_applies(context: PathContext) -> tuple[bool, str]:
    facts = context.facts
    if context.machine is not None and not context.machine.card:
        return False, "no graphics card is used"
    if facts is None or facts.tensor_bytes is None:
        return False, "the model file's piece names and sizes could not be read"
    if context.current.gpu_offload_layers is None:
        return False, "the layer count is left to llama.cpp, so the starting placement is unknown"
    return True, "pieces can be placed one by one (" + facts.design_words() + ")"


GROUPED_NEEDS_PROBE = (
    "block-kind grouped placement is not yet confirmed on this engine by a hardware "
    "probe (tune_grouped_placement_probed); measured and reported only"
)


def _plan_candidate(plan: TensorPlan, context: PathContext | None = None) -> Candidate:
    screen_only = None
    if plan.kind.startswith("grouped") and not (context is not None and context.grouped_placement_probed):
        screen_only = GROUPED_NEEDS_PROBE
    return Candidate(
        value=plan.overrides or "",
        label=plan.label,
        also=(("gpu_offload_layers", plan.gpu_layers),),
        screen_only=screen_only,
    )


def placement_candidates(context: PathContext) -> list[Candidate]:
    from inferenceos.tune.tensor_placement import placement_plans

    if context.facts is None:
        return []
    options = placement_plans(
        context.facts,
        gpu_layers=context.current.gpu_offload_layers,
        current_overrides=context.current.placement_overrides,
        card_budget_bytes=context.card_budget_bytes,
        card=context.machine.card if context.machine is not None else True,
    )
    return [_plan_candidate(plan, context) for plan in options.plans]


def placement_notes(context: PathContext) -> list[str]:
    from inferenceos.tune.tensor_placement import placement_plans

    if context.facts is None:
        return []
    return placement_plans(
        context.facts,
        gpu_layers=context.current.gpu_offload_layers,
        current_overrides=context.current.placement_overrides,
        card_budget_bytes=context.card_budget_bytes,
        card=context.machine.card if context.machine is not None else True,
    ).notes


def _plan_of(context: PathContext, candidate: Candidate) -> TensorPlan | None:
    from inferenceos.tune.tensor_placement import EXPERT_PATTERN, OUTPUT_PATTERN, PlacementPlan

    layers = dict(candidate.also).get("gpu_offload_layers")
    if not isinstance(layers, int) or isinstance(layers, bool) or not candidate.value:
        return None
    text = str(candidate.value)
    current = context.current.gpu_offload_layers or 0
    if text == f"{OUTPUT_PATTERN}=CPU":
        return PlacementPlan("output", layers, text, candidate.label, None, knob=layers - current)
    if EXPERT_PATTERN in text and context.facts is not None and context.facts.expert_bytes:
        with_experts = sum(1 for size in context.facts.expert_bytes if size > 0)
        found = re.match(r"^\^blk\\\.\(([\d|]+)\)", text)
        count = len(found.group(1).split("|")) if found else with_experts
        return PlacementPlan("experts", layers, text, candidate.label, None, knob=count)
    return None


def placement_refine(
    context: PathContext, best: Candidate, tried: set[tuple[object, ...]]
) -> list[Candidate]:
    from inferenceos.tune.tensor_placement import refine_plans

    plan = _plan_of(context, best)
    if plan is None or context.facts is None:
        return []
    seen: set[tuple[int, str | None]] = set()
    for value, also in (key for key in tried if len(key) == 2):
        layers = dict(also).get("gpu_offload_layers") if isinstance(also, tuple) else None
        if isinstance(layers, int) and isinstance(value, str):
            seen.add((layers, value or None))
    out = [_plan_candidate(item, context) for item in refine_plans(context.facts, plan, seen)]
    return [item for item in out if item.key not in tried]


def piece_lines_key(layers: int | None, overrides: str | None) -> str:
    return f"{layers}|{overrides or ''}"


def placement_verify(
    context: PathContext, config: RuntimeConfiguration, measurement: ExperimentMeasurement
) -> tuple[str, str]:
    from inferenceos.tune.tensor_placement import UNVERIFIED, verify_placement

    if context.facts is None:
        return UNVERIFIED, "the model's pieces are unknown"
    observed = observed_settings(measurement)
    extra = ""
    if config.placement_overrides and not (observed or {}).get("overridden_tensors"):
        lines = (context.piece_lines or {}).get(
            piece_lines_key(config.gpu_offload_layers, config.placement_overrides)
        )
        if lines:
            # Identity from the separate debug-level load of this exact setup; this
            # run's own buffer totals are still checked against the estimate.
            observed = {**(observed or {}), "overridden_tensors": lines}
            extra = " (moved pieces read from a separate debug-level load of this setup)"
    status, words = verify_placement(
        context.facts,
        config.gpu_offload_layers,
        config.placement_overrides,
        observed,
        card=context.machine.card if context.machine is not None else True,
    )
    return status, words + extra


PLACEMENT = OptionPath(
    name="placement",
    label="per-piece placement",
    setting="placement_overrides",
    card_id="runtime.tensor_placement",
    description="Where single pieces of the model are loaded (card or processor)",
    workload_id=None,
    candidates=placement_candidates,
    # Pieces on the processor can slow reading a long question (their work is copied
    # to the card for big batches), so a placement must not lose there.
    guard_workload_id=LONG_QUESTION,
    notes=placement_notes,
    default_value=lambda context: "",
    verify=placement_verify,
    refine_candidates=placement_refine,
)


# --------------------------------------------------------------------------- #
# conversation memory format (KV cache)                                        #
# --------------------------------------------------------------------------- #
@applicability.handler("kv_cache.attention_blocks")
def _cache_applies(context: PathContext) -> tuple[bool, str]:
    facts = context.facts
    if facts is not None and facts.attention_layers == 0:
        return False, "this model has no attention blocks, so no conversation memory of this kind"
    if facts is not None and facts.attention_layers is not None and facts.other_layers:
        return True, (
            f"{facts.attention_layers} of {facts.total_layers} blocks keep conversation memory "
            "(the recurrent state of the others is not changed)"
        )
    return True, "the conversation memory can be stored at lower precision"


def _extra_layers_for(context: PathContext, keys: str, values: str) -> int:
    """How many more blocks the memory a format frees would hold on the card (estimate)."""
    from inferenceos.tune.kv_cache import cache_bytes

    facts = context.facts
    layers = context.current.gpu_offload_layers
    total = context.total_layers
    if facts is None or layers is None or not facts.layer_bytes or not context.context_length:
        return 0
    if layers < 1 or layers > total:
        return 0
    now_k, now_v = context.current.effective_kv_types()
    before = cache_bytes(facts, context.model, context.context_length, now_k, now_v).bytes
    after = cache_bytes(facts, context.model, context.context_length, keys, values).bytes
    if before is None or after is None or after >= before:
        return 0
    freed = before - after
    extra, used = 0, 0
    for index in range(total - layers, -1, -1):
        if index >= len(facts.layer_bytes) or used + facts.layer_bytes[index] > freed:
            break
        used += facts.layer_bytes[index]
        extra += 1
    return extra


def cache_candidates(context: PathContext) -> list[Candidate]:
    from inferenceos.tune.kv_cache import cache_choices

    choices, _ = cache_choices(context.facts, context.current, accepts=context.accepts)
    out: list[Candidate] = []
    for choice in choices:
        also: tuple[tuple[str, int | bool | str | None], ...] = (
            ("kv_cache_type_v", choice.values),
            *((name, value) for name, value in choice.also if isinstance(value, (int, bool, str))),
        )
        out.append(Candidate(choice.keys, choice.label, also))
        extra = _extra_layers_for(context, choice.keys, choice.values)
        layers = context.current.gpu_offload_layers
        if extra > 0 and layers is not None:
            from inferenceos.tune.kv_cache import cache_bytes

            more = min(layers + extra, context.total_layers + 1)
            exact = cache_bytes(
                context.facts, context.model, context.context_length or 1, choice.keys, choice.values
            ).exact_layout
            out.append(
                Candidate(
                    choice.keys,
                    f"{choice.label}, with {more} graphics card layers (room it frees, "
                    + ("estimate" if exact else "rough estimate; the real load decides")
                    + ")",
                    (*also, ("gpu_offload_layers", more)),
                )
            )
    return out


def cache_notes(context: PathContext) -> list[str]:
    from inferenceos.tune.kv_cache import cache_choices

    return cache_choices(context.facts, context.current, accepts=context.accepts)[1]


def cache_verify(
    context: PathContext, config: RuntimeConfiguration, measurement: ExperimentMeasurement
) -> tuple[str, str]:
    from inferenceos.tune.kv_cache import cache_bytes, observed_cache_bytes, verify_cache

    observed = observed_settings(measurement)
    status, words = verify_cache(config, observed)
    seen = observed_cache_bytes(observed)
    if context.context_length:
        keys, values = config.effective_kv_types()
        estimate = cache_bytes(context.facts, context.model, context.context_length, keys, values)
        if estimate.bytes is not None:
            words += f"; memory estimated {estimate.bytes / 1024**2:,.0f} MiB ({estimate.basis})"
    if seen is not None:
        words += f", observed {seen / 1024**2:,.0f} MiB"
    return status, words


KV_CACHE = OptionPath(
    name="kv_cache",
    label="conversation memory format",
    setting="kv_cache_type_k",
    card_id="runtime.kv_cache_format",
    description="Conversation memory (KV cache) format, keys and values",
    workload_id=LONG_QUESTION,
    candidates=cache_candidates,
    guard_workload_id=SHORT_QUESTION,
    notes=cache_notes,
    default_value=lambda context: context.current.effective_kv_types()[0] or "f16",
    verify=cache_verify,
)


#: Every setting a path can verify, with its verifier (used by the final check).
Verifier = Callable[[PathContext, RuntimeConfiguration, ExperimentMeasurement], tuple[str, str]]


def verify_every_run(
    verifier: Verifier,
    context: PathContext,
    config: RuntimeConfiguration,
    measurement: ExperimentMeasurement,
) -> tuple[str, str]:
    """``verifier`` on every good run of ``measurement`` on its own (review 7: one run
    showing the setting must not stand in for the others). The first run that does not
    show it decides; no run at all is "unverified"."""
    runs = list(measurement.runs)
    if not runs:
        return "unverified", "no good run of this setup to check"
    words = ""
    for index, run in enumerate(runs, start=1):
        status, words = verifier(context, config, measurement.model_copy(update={"runs": [run]}))
        if status != "shown":
            return status, (f"run {index} of {len(runs)}: " if len(runs) > 1 else "") + words
    return "shown", words + (f" (every one of {len(runs)} runs)" if len(runs) > 1 else "")


VERIFIERS: dict[str, Verifier] = {
    "placement_overrides": placement_verify,
    "kv_cache_type_k": cache_verify,
    "kv_cache_type_v": cache_verify,
}


ALL_PATHS: tuple[OptionPath, ...] = (
    THREADS,
    READING_THREADS,
    FLASH_ATTENTION,
    PROMPT_BATCH,
    ENGINE_SWITCHES,
    PLACEMENT,
    KV_CACHE,
)


def _check_bindings(paths: tuple[OptionPath, ...]) -> None:
    """Every path is bound to its canonical declaration: same card, own setting and
    questions, and a declared applicability rule that is registered here. Every
    registered rule belongs to exactly one declared path. Raises at import otherwise, so
    the two descriptions can never drift apart."""
    to_question = {None: "short", LONG_QUESTION: "long", SHORT_QUESTION: "short"}
    problems: list[str] = []
    declared: set[str] = set()
    for path in paths:
        d = path.declaration
        if d.applicability is None:
            problems.append(f"{path.name}: its declaration names no applicability rule")
            continue
        declared.add(d.applicability)
        try:
            applicability.resolve(d.applicability)
        except applicability.ApplicabilityError as exc:
            problems.append(f"{path.name}: {exc}")
        if (d.card_id, d.settings_written[0]) != (path.card_id, path.setting):
            problems.append(f"{path.name}: card or setting differs from its declaration")
        guard = None if path.guard_workload_id is None else to_question.get(path.guard_workload_id)
        if d.judged_on != to_question.get(path.workload_id) or d.guarded_on != guard:
            problems.append(f"{path.name}: questions differ from its declaration")
    unbound = applicability.registered() - declared
    if unbound:
        problems.append(f"registered rules no declared path uses: {sorted(unbound)}")
    if problems:
        raise RuntimeError("tune paths and their declarations disagree: " + "; ".join(problems))


_check_bindings(ALL_PATHS)


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


#: Settings stored as text where "" means "none".
_EMPTY_MEANS_NONE = frozenset({"engine_switches", "placement_overrides"})


def apply_candidate(
    config: RuntimeConfiguration, path: OptionPath, candidate: Candidate
) -> RuntimeConfiguration:
    update: dict[str, object] = {}
    for setting, value in (*candidate.also, (path.setting, candidate.value)):
        if setting in _EMPTY_MEANS_NONE and value == "":
            value = None
        update[setting] = value
    return config.model_copy(update=update)


__all__ = [
    "ALL_PATHS",
    "Candidate",
    "ENGINE_SWITCHES",
    "KV_CACHE",
    "PLACEMENT",
    "VERIFIERS",
    "Verifier",
    "observed_settings",
    "piece_lines_key",
    "verify_every_run",
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
