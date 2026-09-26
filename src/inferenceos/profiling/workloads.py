"""Deterministic, versioned benchmark workloads.

Prompts are fixed synthetic corpus text (never arbitrary user content). Each
workload declares a conservative context requirement so a resolver can avoid
inventing context sizes larger than a model supports. The exact input token count
is *measured* at run time (from the backend's own counters) and recorded with the
result, not assumed here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

from inferenceos.contracts.enums import MeasurementMode, WorkloadKind
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.workload import WorkloadProfile

BENCHMARK_SUITE_VERSION = "1.0"

_CORPUS = (
    "Inference operating systems schedule model execution across heterogeneous "
    "hardware. A key value cache stores attention state so that previously seen "
    "tokens do not require recomputation. Throughput depends on memory bandwidth, "
    "compute occupancy, and the size of the working set. The latency to the first "
    "token is dominated by prompt processing, while later tokens are memory bound. "
    "Profiling separates loading, prefill, and decode so that each phase can be "
    "compared under identical configurations. "
)

_SHORT_PROMPT = "In one sentence, explain what a key value cache is."
_LONG_PROMPT = (
    _CORPUS * 4
    + "Using the passage above, answer in one sentence: what does a key value cache store?"
)


@dataclass(frozen=True)
class BenchmarkWorkload:
    workload_id: str
    kind: WorkloadKind
    description: str
    prompt: str
    n_predict: int
    seed: int
    temperature: float
    min_context_tokens: int
    ignore_eos: bool = False
    version: str = BENCHMARK_SUITE_VERSION
    # B3 deterministic correctness declarations. ``required_phrases`` are exact
    # factual/arithmetic tokens the candidate must preserve when the baseline produced
    # them; ``required_fields`` are required JSON keys for structured workloads.
    required_phrases: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()

    def to_workload_profile(self) -> WorkloadProfile:
        return WorkloadProfile(
            workload_id=self.workload_id,
            kind=self.kind,
            description=self.description,
            prompt_set_ref=(
                f"{self.workload_id}@{self.version}#{self.prompt_hash[:12]}"
            ),
            prompts=[self.prompt],
            seed=self.seed,
            expected_context_length=self.min_context_tokens,
            concurrency=1,
            modes=[MeasurementMode.COLD, MeasurementMode.WARM],
        )


BENCHMARK_SUITE: tuple[BenchmarkWorkload, ...] = (
    BenchmarkWorkload(
        workload_id="short-short",
        kind=WorkloadKind.SHORT,
        description="short prompt, short generation",
        prompt=_SHORT_PROMPT,
        n_predict=32,
        seed=1234,
        temperature=0.0,
        min_context_tokens=64,
    ),
    BenchmarkWorkload(
        workload_id="long-prefill",
        kind=WorkloadKind.LONG,
        description="long prompt (prefill dominated), short generation",
        prompt=_LONG_PROMPT,
        n_predict=16,
        seed=1234,
        temperature=0.0,
        min_context_tokens=640,
    ),
    BenchmarkWorkload(
        workload_id="short-long-decode",
        kind=WorkloadKind.SHORT,
        description="short prompt, longer generation (decode dominated; EOS ignored)",
        prompt=_SHORT_PROMPT,
        n_predict=128,
        seed=1234,
        temperature=0.0,
        min_context_tokens=256,
        ignore_eos=True,
    ),
)


def get_workload(workload_id: str) -> BenchmarkWorkload:
    for workload in BENCHMARK_SUITE:
        if workload.workload_id == workload_id:
            return workload
    raise KeyError(f"unknown workload id: {workload_id}")


# --------------------------------------------------------------------------- #
# Phase 7R — long-context, task-relevant quality workload                      #
# --------------------------------------------------------------------------- #
# The prompt is sized so it fits a reduced context but leaves insufficient room for
# the requested generation, producing a genuine, reproducible truncation (not a
# fabricated result). The unique leading fact makes a content loss visible too.
_QUALITY_FACT = (
    "The Obsidian Key is stored in cache block ZEPHYR-9137. "
    "Remember that identifier exactly. "
)
_QUALITY_QUESTION = (
    " Using the passage above, answer in one sentence: what does a key value cache "
    "store, and what is the exact cache block identifier for the Obsidian Key?"
)
QUALITY_LONG_WORKLOAD = BenchmarkWorkload(
    workload_id="quality-long-context",
    kind=WorkloadKind.LONG,
    description="long prefill that cannot complete the requested generation in a small context",
    prompt=_QUALITY_FACT + _CORPUS * 5 + _QUALITY_QUESTION,
    n_predict=128,
    seed=1234,
    temperature=0.0,
    min_context_tokens=384,
    ignore_eos=True,
    # The exact identifier is present in the prompt; a candidate that loses it (e.g.
    # through truncation or context loss) is an unambiguous, deterministic failure.
    required_phrases=("ZEPHYR-9137",),
)

QUALITY_WORKLOADS: tuple[BenchmarkWorkload, ...] = (QUALITY_LONG_WORKLOAD,)


def get_quality_workload(workload_id: str = "quality-long-context") -> BenchmarkWorkload:
    for workload in QUALITY_WORKLOADS:
        if workload.workload_id == workload_id:
            return workload
    raise KeyError(f"unknown quality workload id: {workload_id}")


def select_workloads(
    workload_ids: Iterable[str] | None,
    model: ModelProfile,
) -> list[BenchmarkWorkload]:
    """Return requested (or all applicable) workloads that fit the model context."""
    context = model.trained_context_length
    chosen = (
        [get_workload(workload_id) for workload_id in workload_ids]
        if workload_ids is not None
        else list(BENCHMARK_SUITE)
    )
    if context is None:
        return chosen
    return [workload for workload in chosen if workload.min_context_tokens <= context]


def choose_context_length(model: ModelProfile, requested: int | None = None) -> int:
    """Pick a profiler context size that never exceeds the model's trained context."""
    trained = model.trained_context_length
    if requested is not None and requested > 0:
        base = requested
    elif trained is not None:
        base = min(trained, 2048)
    else:
        base = 2048
    return max(base, 64)


def workload_profiles(workloads: Sequence[BenchmarkWorkload]) -> list[WorkloadProfile]:
    return [workload.to_workload_profile() for workload in workloads]
