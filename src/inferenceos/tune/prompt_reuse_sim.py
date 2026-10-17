"""A simulated engine for the prompt-reuse evaluation (tests and ``--simulate`` only).

It models one llama-server slot: it keeps the pieces of the last request (prompt plus
written pieces) and, when asked to reuse, processes only what it can take from them:
the shared start, plus, with ``--cache-reuse N``, chunks found by the same search as
llama-server b11139 (``prompt_reuse.shifted_reuse``). Time to the first word is a fixed
cost plus a cost per processed piece. Fault switches let tests show that the evaluation
catches each way an engine (or a harness) could go wrong. It says nothing about real
hardware.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

from inferenceos.tune.prompt_reuse import (
    CACHE_REUSE_DISABLED_WARNINGS,
    SWA_FULL_WARNING,
    CUT_BY_CLIENT,
    ENDED_EARLY,
    ENGINE_FINISHED,
    Abandoned,
    Cleanup,
    EngineVariant,
    Reading,
    common_prefix,
    shifted_reuse,
)
from inferenceos.tune.reuse_correctness import ChatReading


@dataclass
class SimFaults:
    #: Reuse even when the request asked for a fresh reading.
    reuse_when_fresh: bool = False
    #: Report more reused pieces than possible.
    over_reuse: int = 0
    #: A reused answer comes out different from the fresh one.
    corrupt_after_reuse: bool = False
    #: The engine reports neither a reused count nor a whole-prompt count.
    hide_counters: bool = False
    #: Only the whole-prompt count is missing.
    hide_total: bool = False
    #: Counts that do not add up (processed + reused != whole prompt).
    inconsistent_counters: bool = False
    #: The engine's whole-prompt count differs from its tokenizer's.
    wrong_total: int = 0
    #: The reuse arm's measured answer processes only a few pieces and reports a short
    #: prompt (shorter work), with no whole-prompt count.
    shorter_work: bool = False
    #: Fewer written pieces than asked.
    short_generation: bool = False
    #: The response never says it finished.
    incomplete_response: bool = False
    #: The measured request lands in another slot (no reuse possible there).
    other_slot: bool = False
    #: The second request on a load is faster whatever is reused (warming).
    warming_s: float = 0.0
    #: After a cancelled answer the engine is dead.
    die_on_abandon: bool = False
    #: The answer had already finished when the client left (nothing interrupted).
    abandon_after_finish: bool = False
    #: The stream ends before the client's cutoff, without the engine saying it finished.
    abandon_early_eof: bool = False
    #: After the client leaves, the engine's slots stay busy past the wait.
    busy_after_abandon: bool = False
    #: The engine cannot say whether its slots are idle.
    no_slot_info: bool = False
    #: Readings carry no slot identity.
    no_slot_ids: bool = False
    #: The priming request (on a fresh load) reports reused pieces (impossible).
    prime_reuse: int = 0
    #: The engine turns ``--cache-reuse`` off at load, with b11139's warning.
    disable_cache_reuse: bool = False
    #: The engine's startup log cannot be read.
    no_startup_log: bool = False
    #: A partial reuse (some but not all of the prompt) gives a wrong answer, unless the
    #: load has a full-size sliding-window cache (the job 127 Gemma observation, as a fault).
    partial_reuse_wrong: bool = False
    #: ``--swa-full`` is requested but the engine ignores it (the log shows a reduced cache).
    swa_full_ignored: bool = False
    #: Stopping leaves the password file behind.
    leave_credential: bool = False
    #: Starting fails (the backend cleaned up after itself).
    fail_start: bool = False


@dataclass
class SimulatedReuseEngine:
    """One simulated llama-server load at a time."""

    base_s: float = 0.05
    per_piece_s: float = 0.002
    faults: SimFaults = field(default_factory=SimFaults)
    #: The model has sliding-window layers (its startup log shows both cache sizes).
    sliding_window: bool = False
    #: Answers by prompt (the reuse-correctness evaluation's known answers); None: the
    #: answer is a digest of the prompt, as before.
    oracle: Callable[[str], str | None] | None = None
    starts: int = 0
    stops: int = 0
    running: bool = False
    _slot: list[int] = field(default_factory=list)
    _slot_prompt: list[int] = field(default_factory=list)
    _requests: int = 0
    _variant: EngineVariant | None = None

    # -- engine interface ---------------------------------------------------- #
    def start(self, variant: EngineVariant) -> None:
        if self.running:
            raise RuntimeError("already running")
        self.starts += 1
        if self.faults.fail_start:
            raise RuntimeError("the engine did not start")
        self.running, self._slot, self._slot_prompt, self._requests, self._variant = True, [], [], 0, variant

    def tokenize(self, text: str) -> list[int]:
        self._require()
        return [1] + [int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") for word in text.split()]

    def generate(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int) -> Reading:
        self._require()
        tokens = self.tokenize(prompt)
        reuse = (cache_prompt or self.faults.reuse_when_fresh) and not (self.faults.other_slot and self._requests >= 1)
        cached = 0
        if reuse and self._slot:
            cached = common_prefix(self._slot, tokens)
            variant = self._variant
            if variant is not None and variant.cache_reuse and not self.faults.disable_cache_reuse:
                cached += shifted_reuse(self._slot_prompt + [-1], tokens, variant.cache_reuse)
            cached = min(cached + self.faults.over_reuse, len(tokens) - 1)
        if self._requests == 0 and self.faults.prime_reuse:
            cached = min(self.faults.prime_reuse, len(tokens) - 1)  # nothing to reuse on a fresh load
        processed = len(tokens) - cached
        self._requests += 1
        slot = 1 if (self.faults.other_slot and self._requests > 1) else 0
        ttft = self.base_s + self.per_piece_s * processed
        if self._requests > 1:
            ttft = max(ttft - self.faults.warming_s, 0.001)
        text = self._answer(tokens, n_predict, seed, prompt)
        if cached and self.faults.corrupt_after_reuse:
            text = text[::-1]
        if self._wrong_after_partial(cached, len(tokens)):
            text = "The material does not contain the answer."
        self._slot = tokens + [7] * n_predict  # the slot keeps the prompt and what was written
        self._slot_prompt = tokens
        reading = Reading(ttft_s=ttft, total_s=ttft + 0.01 * n_predict, prefill_s=ttft - self.base_s,
                          processed=processed, cached=cached, total=len(tokens), text=text, cache_prompt=cache_prompt,
                          generated=n_predict, completed=True, slot=slot, occupancy=len(self._slot))
        if self.faults.hide_counters:
            reading.cached = reading.total = None
        if self.faults.hide_total:
            reading.total = None
        if self.faults.inconsistent_counters:
            reading.total = (reading.total or 0) + 3
        if self.faults.wrong_total and reading.total is not None:
            reading.total -= self.faults.wrong_total
            reading.processed = (reading.processed or 0) - self.faults.wrong_total
        if self.faults.shorter_work and cache_prompt and self._requests > 1:
            reading.processed, reading.cached, reading.total = 1, 2, None
            reading.ttft_s = 0.001
        if self.faults.short_generation:
            reading.generated = max(n_predict - 1, 0)
        if self.faults.incomplete_response:
            reading.completed = None
        if self.faults.no_slot_ids:
            reading.slot = None
        return reading

    def abandon(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int, after_chunks: int) -> Abandoned:
        self._require()
        if self.faults.die_on_abandon:
            self.running = False
            return Abandoned(pieces=after_chunks, ending=CUT_BY_CLIENT, idle_confirmed=None,
                             idle_detail="slots could not be read")
        # A cancelled request leaves its processed prompt in the slot.
        self._slot = self._slot_prompt = self.tokenize(prompt)
        if self.faults.abandon_after_finish:
            return Abandoned(pieces=0, ending=ENGINE_FINISHED, idle_confirmed=True)
        if self.faults.abandon_early_eof:
            return Abandoned(pieces=max(after_chunks - 1, 0), ending=ENDED_EARLY, idle_confirmed=True)
        if self.faults.busy_after_abandon:
            return Abandoned(pieces=after_chunks, ending=CUT_BY_CLIENT, idle_confirmed=False,
                             idle_detail="a slot was still processing at the limit")
        if self.faults.no_slot_info:
            return Abandoned(pieces=after_chunks, ending=CUT_BY_CLIENT, idle_confirmed=None,
                             idle_detail="slots do not report is_processing")
        return Abandoned(pieces=after_chunks, ending=CUT_BY_CLIENT, idle_confirmed=True)

    def healthy(self) -> bool:
        return self.running

    def chat(self, messages: list[dict[str, str]], *, cache_prompt: bool | None, max_tokens: int,
             seed: int) -> ChatReading:
        """An ordinary chat request: the prompt is the messages joined; ``cache_prompt``
        None means the engine default, which (as in llama-server) is to reuse. The reply
        has what b11139's non-streamed chat reply has: usage counts, the finish reason and
        timings; no slot identity."""
        self._require()
        prompt = "\n".join(m["content"] for m in messages)
        reading = self.generate(prompt, cache_prompt=True if cache_prompt is None else cache_prompt,
                                n_predict=max_tokens, seed=seed)
        return ChatReading(text=reading.text, finish_reason="length", prompt_tokens=reading.total,
                           cached=reading.cached, completion_tokens=reading.generated,
                           timings={"cache_n": reading.cached, "prompt_n": reading.processed})

    def startup_log(self) -> str | None:
        self._require()
        if self.faults.no_startup_log:
            return None
        lines = ["srv    load_model: loading model (simulated)"]
        if self.sliding_window:
            full = self._swa_full_effective()
            if full:
                lines.append(f"llama_kv_cache_iswa: {SWA_FULL_WARNING} (ref: simulated)")
            lines.append("llama_kv_cache_iswa: creating non-SWA KV cache, size = 4096 cells")
            lines.append(f"llama_kv_cache_iswa: creating     SWA KV cache, size = {4096 if full else 3072} cells")
        if self.faults.disable_cache_reuse and self._variant is not None and self._variant.cache_reuse:
            lines.append(f"srv    load_model: {CACHE_REUSE_DISABLED_WARNINGS[0]}")
        lines.append("srv    operator(): model loaded")
        return "\n".join(lines)

    def stop(self) -> Cleanup:
        self.stops += 1
        was = self.running
        self.running = False
        if not was and self.faults.fail_start:
            return Cleanup(process_gone=True, credential_removed=True)  # the failed start cleaned up
        return Cleanup(process_gone=True, credential_removed=not self.faults.leave_credential,
                       problem=None if was else "the engine had already died")

    # -- helpers ------------------------------------------------------------- #
    def _require(self) -> None:
        if not self.running:
            raise RuntimeError("engine not running")

    def _answer(self, tokens: list[int], n_predict: int, seed: int, prompt: str = "") -> str:
        known = self.oracle(prompt) if self.oracle is not None else None
        if known is not None:
            return known
        digest = hashlib.sha256(f"{tokens}|{n_predict}|{seed}".encode()).hexdigest()
        return " ".join(digest[i:i + 4] for i in range(0, min(n_predict, 16) * 4, 4))

    def _swa_full_effective(self) -> bool:
        return bool(self._variant is not None and self._variant.swa_full and not self.faults.swa_full_ignored)

    def _wrong_after_partial(self, cached: int, total: int) -> bool:
        partial = 0 < cached < total - 1
        return bool(self.faults.partial_reuse_wrong and partial and not self._swa_full_effective())


__all__ = ["SimFaults", "SimulatedReuseEngine"]
