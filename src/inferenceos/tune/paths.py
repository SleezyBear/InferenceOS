"""Optimization paths for ``tune`` (after the graphics card layer path).

Each path looks at one possible source of waste, tries a few changes from the current
best setup, and keeps one only when a fair comparison (``tune.fair``) shows it really
helps. Paths stack: each one starts from the setup the previous one left.

Paths built here (Oct 2026):

* ``threads``: how many processor threads write the answer. llama.cpp's default
  is the number of physical cores, which is not always the fastest; on a laptop
  part of the model often runs on the processor, so this can matter a lot.
* ``flash_attention``: llama.cpp's flash attention on or off (its default is
  "auto"). Whether it helps depends on the graphics card and how much of the model
  is on it. Wording check on the answers.
* ``reading_threads``: threads for reading the question (``-tb``), separate from the
  writing threads; measured with the long-prompt workload.
* ``prompt_batch``: how many prompt words are processed at once (``-ub``, default
  512). This mainly changes the time to the first word on longer prompts, so this
  path measures with the long-prompt workload.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.profiling.workloads import BenchmarkWorkload, get_workload


@dataclass(frozen=True)
class PathContext:
    """What a path may look at when deciding whether it applies and what to try."""

    hardware: HardwareProfile
    model: ModelProfile
    total_layers: int
    current: RuntimeConfiguration
    supported_options: frozenset[str] | None = None


@dataclass(frozen=True)
class Candidate:
    value: int | bool
    label: str


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
    #: Workload id this path is measured with (None = tune's workload).
    workload_id: str | None
    applies: Callable[[PathContext], tuple[bool, str]]
    candidates: Callable[[PathContext], list[Candidate]]

    def check(self, context: PathContext) -> tuple[bool, str]:
        if context.supported_options is not None and self.option not in context.supported_options:
            return False, f"this llama.cpp does not have the {self.option} option"
        return self.applies(context)


# --------------------------------------------------------------------------- #
# threads                                                                     #
# --------------------------------------------------------------------------- #
def thread_candidates(context: PathContext) -> list[Candidate]:
    """Thread counts around the physical core count, plus all logical cores.

    The current value (``None`` = llama.cpp default, which is the physical core count)
    is never re-tried.
    """
    physical = max(int(context.hardware.cpu.physical_cores or 1), 1)
    logical = max(int(context.hardware.cpu.logical_cores or physical), physical)
    current = context.current.threads if context.current.threads is not None else physical
    values = {max(physical - 2, 1), max(physical - 1, 1), physical}
    if logical > physical:
        values.add(min(physical + 2, logical))
        values.add(logical)
    values.discard(current)
    return [Candidate(value=value, label=f"{value} threads") for value in sorted(values)]


def _threads_applies(context: PathContext) -> tuple[bool, str]:
    if int(context.hardware.cpu.physical_cores or 0) < 2:
        return False, "only one processor core"
    return True, "the processor does part or all of the work"


# --------------------------------------------------------------------------- #
# flash attention                                                             #
# --------------------------------------------------------------------------- #
def flash_candidates(context: PathContext) -> list[Candidate]:
    current = context.current.flash_attention
    values: list[bool] = [True, False] if current is None else [not current]
    return [Candidate(value=value, label="flash attention " + ("on" if value else "off"))
            for value in values]


def _flash_applies(context: PathContext) -> tuple[bool, str]:
    return True, "llama.cpp decides flash attention by itself unless told"


# --------------------------------------------------------------------------- #
# prompt batch                                                                #
# --------------------------------------------------------------------------- #
PROMPT_BATCH_DEFAULT = 512
PROMPT_BATCH_VALUES = (128, 256, 1024)


def prompt_batch_candidates(context: PathContext) -> list[Candidate]:
    current = context.current.ubatch_size or PROMPT_BATCH_DEFAULT
    values = [value for value in (*PROMPT_BATCH_VALUES, PROMPT_BATCH_DEFAULT) if value != current]
    return [Candidate(value=value, label=f"prompt batch {value}") for value in sorted(values)]


def _prompt_batch_applies(context: PathContext) -> tuple[bool, str]:
    return True, "how the question is read changes the time to the first word"


# --------------------------------------------------------------------------- #
# reading threads (prompt)                                                    #
# --------------------------------------------------------------------------- #
def reading_thread_candidates(context: PathContext) -> list[Candidate]:
    """Threads for reading the question, separate from the writing threads.

    On the laptop (Oct 2026, 9B, 8 layers on the card) reading the prompt was fastest
    with few threads (3: 40 words/s, 6: 35, 12: 34) while writing was fastest with 8-10.
    Unset means "the same as the writing threads", which is never re-tried.
    """
    physical = max(int(context.hardware.cpu.physical_cores or 1), 1)
    writing = context.current.threads if context.current.threads is not None else physical
    current = context.current.threads_batch if context.current.threads_batch is not None else writing
    values = {max(physical // 2, 1), max(physical - 2, 1), physical}
    values.discard(current)
    return [Candidate(value=value, label=f"{value} reading threads") for value in sorted(values)]


def _reading_threads_applies(context: PathContext) -> tuple[bool, str]:
    if int(context.hardware.cpu.physical_cores or 0) < 2:
        return False, "only one processor core"
    return True, "reading the question may want a different thread count than writing"


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
)
FLASH_ATTENTION = OptionPath(
    name="flash_attention",
    label="flash attention",
    setting="flash_attention",
    card_id="runtime.flash_attention",
    description="Flash attention",
    option="--flash-attn",
    workload_id=None,
    applies=_flash_applies,
    candidates=flash_candidates,
)
PROMPT_BATCH = OptionPath(
    name="prompt_batch",
    label="prompt batch size",
    setting="ubatch_size",
    card_id="runtime.prompt_batch",
    description="Prompt batch size",
    option="--ubatch-size",
    workload_id="long-prefill-steady",
    applies=_prompt_batch_applies,
    candidates=prompt_batch_candidates,
)

READING_THREADS = OptionPath(
    name="reading_threads",
    label="reading threads",
    setting="threads_batch",
    card_id="runtime.reading_threads",
    description="Threads for reading the question",
    option="--threads-batch",
    workload_id="long-prefill-steady",
    applies=_reading_threads_applies,
    candidates=reading_thread_candidates,
)

ALL_PATHS: tuple[OptionPath, ...] = (THREADS, READING_THREADS, FLASH_ATTENTION, PROMPT_BATCH)


STEADY_SUFFIX = "-steady"


def steady_workload(workload_id: str, *, default: BenchmarkWorkload) -> BenchmarkWorkload:
    """A built-in workload by id; ``<id>-steady`` is the same prompt but always writes
    the full answer length (end-of-answer ignored), so a model that stops at once on
    that prompt still gives a first-word time and a writing speed."""
    base_id = workload_id[: -len(STEADY_SUFFIX)] if workload_id.endswith(STEADY_SUFFIX) else workload_id
    try:
        base = get_workload(base_id)
    except KeyError:
        return default
    if base_id == workload_id:
        return base
    return replace(base, workload_id=workload_id, ignore_eos=True)


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
    return config.model_copy(update={path.setting: candidate.value})


__all__ = [
    "ALL_PATHS",
    "Candidate",
    "FLASH_ATTENTION",
    "OptionPath",
    "PROMPT_BATCH",
    "PathContext",
    "READING_THREADS",
    "THREADS",
    "apply_candidate",
    "flash_candidates",
    "paths_by_name",
    "steady_workload",
    "prompt_batch_candidates",
    "reading_thread_candidates",
    "thread_candidates",
]
