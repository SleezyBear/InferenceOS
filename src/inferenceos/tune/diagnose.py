"""Diagnosis first: where does the time of one written word go, on this machine?

Before trying settings, tune looks at the model running in its current setup:

* the time of one written word, from a normal measurement (timers off);
* how much of it the graphics card works, from one run with the engine's own
  per-operation timer on (``tune/engine_probe.py``; that run is never used for speed);
* how many bytes of model weights the processor reads per word (from the model file and
  the engine's placement rule), and so how fast the processor part moves memory,
  compared with what this computer can move at best (``tune/memory_speed.py``);
* how much of the time to the first word is reading the question.

From that it **orders** the paths and decides how deeply to look at each one (owner's
decision, 4 Oct 2026):

* a path is skipped only with strong evidence: it is unsupported or clearly does not
  apply (that check stays with each path, e.g. no processor threads to tune when
  everything is on the card);
* a path the diagnosis gives little chance gets a cheap probe (its two most likely
  values) instead of a full search; a probe that looks better turns into a full search;
* one short run and one memory test never establish "cannot help";
* after a kept change the diagnosis is run again, because the bottleneck can move.

Every number here is an estimate with its source named; the fair comparison decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inferenceos.tune.declarations import affects_table, memory_shaping_paths
from inferenceos.tune.model_facts import ModelFacts
from inferenceos.tune.placement import OUTPUT_FIRST, placement

#: Path priorities, best first.
HIGH, NORMAL, PROBE = "high", "normal", "probe"


#: Which finished paths a kept change can affect, so they are looked at again (owner,
#: 4 Oct 2026; review 4 found threads +15% after prompt batch 1024 in a reproduction).
#: From how llama.cpp works: threads and reading threads share the processor and its
#: memory traffic with the question-reading batch; flash attention and the batch size
#: change the reading work; engine switches change what the card does, which moves the
#: split of work between card and processor.
#: Explicit piece placement and the conversation-memory format move work and memory
#: between card and processor, so they affect the processor paths, and each other: a
#: smaller cache can make a placement fit, and a placement changes what the cache
#: competes with on the card. A change on either also makes the layer count worth
#: searching again (``MEMORY_PATHS``).
#: Derived from the path declarations (``tune/declarations.py``, C3 second slice): each
#: path's ``invalidates``. Contents and tuple order are the literal this replaced
#: (``tests/test_path_declarations.py`` keeps that literal and checks equality).
AFFECTS: dict[str, tuple[str, ...]] = affects_table()
#: Paths whose kept change frees or uses card memory: the layer count is searched again
#: (declared ``memory_shaping``).
MEMORY_PATHS: frozenset[str] = memory_shaping_paths()


def affected_paths(changed: str) -> tuple[str, ...]:
    """The paths to look at again after a change kept on path ``changed``."""
    return AFFECTS.get(changed, ())


def processor_bytes_per_word(
    facts: ModelFacts,
    gpu_layers: int | None,
    *,
    card: bool,
    rule: str = OUTPUT_FIRST,
    overrides: str | None = None,
) -> int | None:
    """Bytes of model weights the processor reads for one written word (estimate).

    Blocks on the processor (only the experts used, for expert models) plus the output
    step when it is not on the card. ``None`` without sizes from the model file.
    """
    if facts.layer_bytes is None or facts.output_bytes is None:
        return None
    if overrides and card:
        from inferenceos.tune.tensor_placement import piece_placement

        where_pieces = piece_placement(facts, gpu_layers or 0, overrides, card=card, rule=rule)
        return where_pieces.processor_active_bytes if where_pieces is not None else None
    work = list(facts.active_layer_bytes or facts.layer_bytes)
    if not card:
        return int(sum(work) + facts.output_bytes)
    total = max(int(facts.total_layers), 1)
    where = placement(gpu_layers or 0, total, rule)
    processor = sum(work[index] for index in where.blocks_on_processor if index < len(work))
    if not where.output_on_card:
        processor += facts.output_bytes
    return int(processor)


@dataclass
class Diagnosis:
    """Where one written word's time goes, and what that says about the paths."""

    #: Time of one written word with timers off (seconds).
    word_s: float | None = None
    #: Card time of one written word (seconds), from the timed run.
    card_word_s: float | None = None
    #: The rest: processor work, waiting and copying (seconds).
    rest_word_s: float | None = None
    processor_bytes_per_word: int | None = None
    #: Memory traffic of the processor part (bytes/s) and the machine's best copy rate.
    processor_traffic: float | None = None
    memory_ceiling: float | None = None
    #: Share of the time to the first word spent reading the question.
    reading_share: float | None = None
    findings: list[str] = field(default_factory=list)
    #: Path name -> (priority, reason).
    priorities: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def card_share(self) -> float | None:
        if self.word_s and self.card_word_s is not None:
            return min(self.card_word_s / self.word_s, 1.0)
        return None

    @property
    def memory_use(self) -> float | None:
        """Processor traffic as a share of the machine's best copy rate."""
        if self.processor_traffic and self.memory_ceiling:
            return self.processor_traffic / self.memory_ceiling
        return None

    def order(self, names: list[str]) -> list[str]:
        """Path names, high priority first, the given order kept within a priority."""
        rank = {HIGH: 0, NORMAL: 1, PROBE: 2}
        return sorted(
            names, key=lambda name: rank.get(self.priorities.get(name, (NORMAL, ""))[0], 1)
        )

    def priority(self, name: str) -> str:
        return self.priorities.get(name, (NORMAL, ""))[0]

    def record(self) -> dict[str, object]:
        return {
            "word_s": self.word_s,
            "card_word_s": self.card_word_s,
            "rest_word_s": self.rest_word_s,
            "card_share": self.card_share,
            "processor_bytes_per_word": self.processor_bytes_per_word,
            "processor_traffic_gb_s": (
                round(self.processor_traffic / 1e9, 2) if self.processor_traffic else None
            ),
            "memory_ceiling_gb_s": round(self.memory_ceiling / 1e9, 2) if self.memory_ceiling else None,
            "memory_use": self.memory_use,
            "reading_share": self.reading_share,
            "findings": list(self.findings),
            "priorities": {name: {"priority": p, "why": why} for name, (p, why) in self.priorities.items()},
        }


def diagnose(
    *,
    word_s: float | None,
    first_word_s: float | None,
    reading_s: float | None,
    card_word_s: float | None,
    processor_bytes: int | None,
    memory_ceiling: float | None,
    card: bool,
    processor_share_of_work: float | None,
    long_question: bool = True,
    uses_experts: bool = False,
    hybrid: bool = False,
) -> Diagnosis:
    """Turn the readings into findings and path priorities.

    ``word_s``: one written word, timers off. ``first_word_s`` / ``reading_s``: time to
    the first word and the engine's reading time on the short question. ``card_word_s``:
    card time per written word from the timed run (None when the engine has no timer or
    nothing runs on the card).
    """
    result = Diagnosis(
        word_s=word_s,
        card_word_s=card_word_s,
        processor_bytes_per_word=processor_bytes,
        memory_ceiling=memory_ceiling,
    )
    if word_s and card_word_s is not None:
        result.rest_word_s = max(word_s - card_word_s, 0.0)
    if result.rest_word_s and processor_bytes:
        result.processor_traffic = processor_bytes / result.rest_word_s
    if first_word_s and reading_s is not None and first_word_s > 0:
        result.reading_share = min(reading_s / first_word_s, 1.0)

    card_share = result.card_share
    use = result.memory_use
    if word_s:
        result.findings.append(f"one written word takes {word_s * 1000:.1f} ms")
    if card_share is not None:
        result.findings.append(
            f"in a run with the engine's timer on, the graphics card's share of a word was "
            f"{card_share:.0%} (the timer changes how card and processor overlap, so this "
            "is a clue, not a measurement of a normal run)"
        )
    if use is not None and result.processor_traffic:
        result.findings.append(
            f"rough estimate: the processor part reads model weights at about "
            f"{result.processor_traffic / 1e9:.1f} GB/s, against a copy test of "
            f"{result.memory_ceiling / 1e9:.1f} GB/s ({use:.0%}; the copy test counts "  # type: ignore[operator]
            "reads and writes, and the time includes waiting)"
        )
    if result.reading_share is not None:
        result.findings.append(
            f"reading the short question is {result.reading_share:.0%} of the time to the first word"
        )
    if card and processor_share_of_work is not None and processor_share_of_work > 0:
        # The layer round runs before the diagnosis, so it gets no priority here
        # (review 4: a "layers" priority had no effect).
        result.findings.append(
            "the layer count was searched before this diagnosis"
        )

    # --- Priorities (never a skip: skips stay with each path's own check) ---------- #
    # Provisional thresholds (review 4, 4 Oct 2026): the 50% card share and the 85%
    # memory figure rank where to look first; they are not proven boundaries.
    processor_heavy = card_share is None or card_share < 0.5
    if use is not None and use >= 0.85:
        result.priorities["threads"] = (
            PROBE,
            f"rough estimate: the processor part already moves memory near the copy-test "
            f"rate ({use:.0%}), so more threads may add little; quick probe",
        )
    elif processor_heavy:
        result.priorities["threads"] = (
            HIGH, "the processor part looks like most of the time per word; try threads first"
        )
    reading_matters = long_question or (result.reading_share or 0.0) >= 0.3
    for name in ("prompt_batch", "reading_threads", "flash_attention"):
        result.priorities[name] = (
            NORMAL if reading_matters else PROBE,
            "reading the question is a large part of the first-word time"
            if reading_matters
            else "reading the question is a small part of the first-word time; quick probe",
        )
    if card:
        split = processor_share_of_work is not None and 0.0 < processor_share_of_work < 1.0
        if split or uses_experts or hybrid:
            why = (
                "the model is split between card and processor"
                if split
                else "experts or mixed block kinds can be placed piece by piece"
            )
            result.priorities["placement"] = (HIGH, why + "; try per-piece placement")
        else:
            result.priorities["placement"] = (
                PROBE,
                "the whole model sits on one side; per-piece placement is a quick probe",
            )
        result.priorities["kv_cache"] = (
            NORMAL if split or processor_share_of_work is None else PROBE,
            "a smaller conversation memory leaves card room for more of the model"
            if split or processor_share_of_work is None
            else "the model already fits; a smaller conversation memory is a quick probe",
        )
        result.priorities["engine_switches"] = (
            HIGH if (card_share or 0.0) >= 0.5 else PROBE,
            "in the timed run the graphics card did most of the work per word"
            if (card_share or 0.0) >= 0.5
            else "in the timed run the graphics card did little of the work per word; quick probe",
        )
    return result


__all__ = [
    "AFFECTS",
    "Diagnosis",
    "MEMORY_PATHS",
    "HIGH",
    "NORMAL",
    "PROBE",
    "affected_paths",
    "diagnose",
    "processor_bytes_per_word",
]
