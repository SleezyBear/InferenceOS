"""Candidate planner for the budgeted experimental search (owner, 6 Oct 2026; revised
after review 8 the same night).

A pure decision object: it is told the model's structure, the machine, the current setup
and every screening reading so far, and answers "what to measure next, in which exact
setup". It never measures anything and never decides acceptance; one fresh final
confirmation does that (``tune/budgeted.py``).

Order (``claude/search-order-v2-2026-10-06.md``):

1. **Placement first, broadly.** CPU-only (0), output-only (1) and the most that safely
   fits, measured at the starting threads and batch; the start is the reference.
2. **Intermediate counts only when justified**: partial offload (top < everything on the
   card), or an ambiguous coarse result (best two within two noise widths with untried
   counts between them). A winning extreme does not stop this for partial offload. Then
   one step either side of the best.
3. **Threads or batch first, by the processor share** at the chosen placement: threads
   first at >= 50% (when large batches are usable), batch first otherwise; threads are
   only a later option below 20%.
4. **Preferred candidates from hardware, workload and the current setting**; the
   current value is always the reference. A clear winner is refined (one step either
   side) right away, in the same setup, before the next category.
5. **Strongest opportunity next.** Then, ranked by reading size in noise widths:
   rechecks after a setup change (below), refinement of a weak lead, and one fallback for
   a category whose preferred candidates all disappointed (fewer threads, smaller batch:
   deprioritised, never eliminated).
6. **Weak readings stay eligible.** A gain within two noise widths is not put into the
   combined candidate on one reading, but is listed (``weak_candidates``) so the driver
   can re-measure the best one and, if it holds up, confirm it when time allows.

**Evidence is scoped to the setup it was measured in (review 8).** Every reading keeps
the exact setup (layers, threads, batch) it was measured in, against a reference measured
in that same setup with only this category's value changed. A category's choice is made
only from readings whose *other* settings equal the current choice of the other
categories. When another category's choice changes (placement moved, threads changed
after batch was chosen, or batch chosen after threads), the affected category's readings
are out of context: its choice and runner-up are rechecked in the new setup (at most
``MAX_RECHECKS`` times per category, so rechecks never turn into a new search), and an
old reading that was never measured in the new setup is never chosen.

Readings are goal-score gains on the category's question (placement and threads on the
short question, batch on the long one). Classes: *clear* gain > 2 noise widths and at
least the smallest worthwhile gain; *weak* 0 < gain otherwise; *worse* below -1 width.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from inferenceos.tune.declarations import practical_tables

PLACEMENT, THREADS, BATCH = "placement", "threads", "batch"
#: C3 adapter: the categories, their settings, slots and questions, and which evidence a
#: setting change makes stale, come from the path declarations (``tune/declarations.py``).
#: The values are unchanged (``tests/test_path_declarations.py`` checks them against the
#: literals they replaced); the search steps below are still the specialised handlers.
TABLES = practical_tables()
CATEGORIES = TABLES.categories
QUESTION = dict(TABLES.question)
SETTING = dict(TABLES.setting)
SLOT = dict(TABLES.slot)
#: Slots a category's readings are valid for (exact-setup evidence).
SCOPE = dict(TABLES.scope)
MAX_INTERMEDIATE = 3
MAX_NEIGHBOURS = 2
#: Rechecks per category after its setup changed.
MAX_RECHECKS = 1
THREADS_FIRST_SHARE = 0.5
THREADS_RELEVANT_SHARE = 0.2
BATCH_SIZES = (64, 128, 256, 512, 1024, 2048)
BATCH_DEFAULT = 512

#: (layers, threads, batch), with the engine defaults resolved to numbers.
Setup = tuple[int, int, int]


@dataclass(frozen=True)
class Structure:
    """What the planner may know before measuring (no model names)."""

    count_max: int
    top: int
    uses_experts: bool
    mixed_kinds: bool
    physical_cores: int
    logical_cores: int
    long_question_tokens: int | None = None

    @property
    def fits_fully(self) -> bool:
        return self.top >= self.count_max


@dataclass
class Reading:
    category: str
    value: int
    gain: float | None
    noise: float
    #: The exact setup measured (layers, threads, batch).
    setup: Setup
    #: How many readings were averaged into this one.
    count: int = 1

    @property
    def context(self) -> tuple[int, int]:
        """The other two settings: what this reading is valid for."""
        return tuple(self.setup[i] for i in SCOPE[self.category])  # type: ignore[return-value]

    def kind(self, smallest: float) -> str:
        if self.gain is None:
            return "missing"
        if self.gain > 2.0 * self.noise and self.gain >= smallest:
            return "clear"
        if self.gain > 0.0:
            return "weak"
        if self.gain < -self.noise:
            return "worse"
        return "flat"


@dataclass(frozen=True)
class Step:
    category: str
    values: tuple[int, ...]
    reason: str
    #: The setup every value is measured in (this category's slot holds the reference).
    base: Setup = (0, 0, 0)

    @property
    def question(self) -> str:
        return QUESTION[self.category]

    def setup_for(self, value: int) -> Setup:
        out = list(self.base)
        out[SLOT[self.category]] = int(value)
        return (out[0], out[1], out[2])


@dataclass
class Planner:
    structure: Structure
    start_layers: int
    start_threads: int | None
    start_batch: int | None
    processor_share: Callable[[int], float | None]
    batch_fits: Callable[[int, int], bool] = lambda layers, size: True
    smallest_gain: float = 0.015
    readings: dict[str, list[Reading]] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    wants_question_length: bool = False
    waiting_for_length: bool = False
    batch_unavailable: bool = False
    #: Current choice per category (starts at the reference values).
    chosen: dict[str, int] = field(default_factory=dict)
    placement: int | None = None
    _planned: set[tuple[str, int, tuple[int, int]]] = field(default_factory=set)
    _last_base: dict[tuple[str, int], Setup] = field(default_factory=dict)
    _stage: str = "coarse"
    _order: list[str] = field(default_factory=list)
    _first_probed: set[str] = field(default_factory=set)
    _refined: set[tuple[str, tuple[int, int]]] = field(default_factory=set)
    _fell_back: set[str] = field(default_factory=set)
    _rechecks_done: dict[str, int] = field(default_factory=dict)
    _pending: list[Step] = field(default_factory=list)
    #: Optional permutation of an already eligible step; no queue/choice ownership.
    candidate_order: Callable[[Step], tuple[int, ...]] | None = None

    def __post_init__(self) -> None:
        for category in CATEGORIES:
            self.readings.setdefault(category, [])
        self.chosen = {PLACEMENT: self.start_layers, THREADS: self.current_threads, BATCH: self.current_batch}

    # ------------------------------------------------------------------ #
    # References, setups and readings                                      #
    # ------------------------------------------------------------------ #
    @property
    def current_threads(self) -> int:
        return int(self.start_threads or self.structure.physical_cores)

    @property
    def current_batch(self) -> int:
        return int(self.start_batch or BATCH_DEFAULT)

    def reference(self, category: str) -> int:
        return {PLACEMENT: self.start_layers, THREADS: self.current_threads, BATCH: self.current_batch}[category]

    def start_setup(self) -> Setup:
        return (self.start_layers, self.current_threads, self.current_batch)

    def current_setup(self) -> Setup:
        return (self.chosen[PLACEMENT], self.chosen[THREADS], self.chosen[BATCH])

    def base_for(self, category: str) -> Setup:
        """Where a category is measured now: placement at the starting threads and batch;
        threads and batch at the current choices, with their own slot at the reference."""
        if category == PLACEMENT:
            return self.start_setup()
        out = list(self.current_setup())
        out[SLOT[category]] = self.reference(category)
        return (out[0], out[1], out[2])

    def _context_of(self, category: str, setup: Setup) -> tuple[int, int]:
        return tuple(setup[i] for i in SCOPE[category])  # type: ignore[return-value]

    def record(self, category: str, value: int, gain: float | None, noise: float,
               setup: Setup | None = None) -> None:
        """One screening reading: gain vs the reference measured in the same setup."""
        if setup is None:
            base = self._last_base.get((category, int(value)), self.base_for(category))
            out = list(base)
            out[SLOT[category]] = int(value)
            setup = (out[0], out[1], out[2])
        new = Reading(category, int(value), gain, max(noise, 1e-6), tuple(int(v) for v in setup))  # type: ignore[arg-type]
        same = [r for r in self.readings[category] if r.value == new.value and r.context == new.context]
        if same:
            old = same[-1]
            self.readings[category].remove(old)
            if gain is not None and old.gain is not None:
                # A repeat in the same setup (fresh loads, fresh reference): average the
                # readings; the spread of one reading shrinks with the square root.
                count = old.count + 1
                single = max(old.noise * old.count ** 0.5, new.noise)
                new = Reading(category, new.value, (old.gain * old.count + gain) / count,
                              single / count ** 0.5, new.setup, count)
        self.readings[category].append(new)
        self._update_choice(category)

    def in_context(self, category: str) -> list[Reading]:
        context = self._context_of(category, self.base_for(category))
        return [r for r in self.readings[category] if r.context == context]

    def _known(self, category: str) -> dict[int, float]:
        out = {r.value: r.gain for r in self.in_context(category) if r.gain is not None}
        out.setdefault(self.reference(category), 0.0)
        return out

    def _noise(self, category: str) -> float:
        values = [r.noise for r in self.readings[category]]
        return max(values) if values else 0.01

    def best(self, category: str) -> tuple[int, float]:
        known = self._known(category)
        value = max(known, key=lambda v: (known[v], -abs(v - self.reference(category))))
        return value, known[value]

    def reading(self, category: str, value: int) -> Reading | None:
        found = [r for r in self.in_context(category) if r.value == value]
        return found[-1] if found else None

    def kind_of(self, category: str, value: int) -> str:
        reading = self.reading(category, value)
        if reading is None:
            return "reference" if value == self.reference(category) else "missing"
        return reading.kind(self.smallest_gain)

    def _update_choice(self, category: str) -> None:
        if category == PLACEMENT and self.placement is None:
            return  # chosen when placement is settled
        if not self.in_context(category):
            return  # nothing measured in this setup yet: keep the choice, pending recheck
        best, _ = self.best(category)
        new = best if self.kind_of(category, best) == "clear" else self.reference(category)
        if new != self.chosen[category]:
            old = self.chosen[category]
            self.chosen[category] = new
            if category == PLACEMENT:
                self.placement = new
            self.log.append(f"{category}: choice {old} -> {new} (readings in setup {self.base_for(category)})")
            if self._stage == "done":
                self._stage = "follow"  # other settings' setups changed: rechecks may follow

    # ------------------------------------------------------------------ #
    # Candidates                                                           #
    # ------------------------------------------------------------------ #
    def _fresh(self, category: str, values: list[int], base: Setup) -> tuple[int, ...]:
        context = self._context_of(category, base)
        out: list[int] = []
        for value in values:
            value = int(value)
            if (category, value, context) in self._planned or value == self.reference(category) or value in out:
                continue
            out.append(value)
        return tuple(out)

    def _step(self, category: str, values: list[int], reason: str) -> Step | None:
        base = self.base_for(category)
        fresh = self._fresh(category, values, base)
        if not fresh:
            return None
        step = Step(category, fresh, reason, base)
        if self.candidate_order is not None and len(fresh) > 1:
            try:
                ordered = self.candidate_order(step)
                if (not isinstance(ordered, tuple) or any(type(v) is not int for v in ordered)
                        or sorted(ordered) != sorted(fresh)):
                    raise ValueError("candidate hint must be a permutation of eligible values")
                fresh = ordered
                step = replace(step, values=fresh)
            except Exception as exc:
                self.log.append(f"candidate ordering unavailable: {type(exc).__name__}")
        context = self._context_of(category, base)
        for value in fresh:
            self._planned.add((category, value, context))
            self._last_base[(category, value)] = base
        self.log.append(f"{category}: {', '.join(map(str, fresh))} ({reason}; setup {base})")
        return step

    def coarse_placement(self) -> list[int]:
        s = self.structure
        order = [0, 1, s.top] if s.uses_experts else [s.top, 0, 1]
        return [v for v in order if 0 <= v <= s.top]

    def _intermediate(self) -> list[int]:
        s = self.structure
        best, _ = self.best(PLACEMENT)
        known = self._known(PLACEMENT)
        ranked = sorted(known, key=lambda v: known[v], reverse=True)
        noise = self._noise(PLACEMENT)
        ambiguous = (
            len(ranked) > 1
            and known[ranked[0]] - known[ranked[1]] <= 2.0 * noise
            and abs(ranked[0] - ranked[1]) > 1
        )
        if not (not s.fits_fully or ambiguous):
            return []
        step = max(2, round(max(s.top, 1) / 7))
        if best <= 1:
            probes = [best + step, best + 2 * step]
        elif best >= s.top:
            probes = [best - step, best - 2 * step]
        else:
            probes = [best + step, best - step]
        return [p for p in probes if 0 <= p <= s.top][:MAX_INTERMEDIATE]

    def _neighbours(self) -> list[int]:
        best, _ = self.best(PLACEMENT)
        allowed = ("clear", "weak") if self.structure.fits_fully else ("clear", "weak", "reference")
        if self.kind_of(PLACEMENT, best) not in allowed or best in (0, 1, self.structure.top):
            return []
        return [v for v in (best - 1, best + 1) if 0 <= v <= self.structure.top][:MAX_NEIGHBOURS]

    def thread_candidates(self) -> list[int]:
        p, l = self.structure.physical_cores, self.structure.logical_cores
        if l > p:
            span = l - 1 - p
            return [p + int(span * 0.4 + 0.5), p + int(span * 0.8 + 0.5)] if span >= 2 else [p, l - 1]
        return [max(1, p - 1), p]

    def thread_fallback(self) -> list[int]:
        p, l = self.structure.physical_cores, self.structure.logical_cores
        return [v for v in (p, max(1, p // 2), l) if v != self.current_threads][:1]

    def _batch_meaningful(self, size: int) -> bool:
        tokens = self.structure.long_question_tokens
        if tokens is None:
            return True
        whole = next((s for s in BATCH_SIZES if s >= tokens), BATCH_SIZES[-1])
        return size <= whole

    def batch_candidates(self) -> list[int]:
        if self.batch_unavailable:
            return []
        layers = self.chosen[PLACEMENT]
        current = self.current_batch
        larger = [s for s in sorted(BATCH_SIZES, reverse=True) if s > current]
        usable = [s for s in larger if self._batch_meaningful(s) and self.batch_fits(layers, s)]
        if not usable:
            nxt = [s for s in sorted(BATCH_SIZES) if s > current and self._batch_meaningful(s)]
            usable = [s for s in nxt[:1] if self.batch_fits(layers, s)]
        return usable[:2]

    def batch_fallback(self) -> list[int]:
        if self.batch_unavailable:
            return []
        return [s for s in sorted(BATCH_SIZES, reverse=True) if s < self.current_batch][:1]

    def category_order(self, layers: int) -> list[str]:
        share = self.processor_share(layers)
        large_usable = bool(self.batch_candidates())
        threads = share is None or share >= THREADS_RELEVANT_SHARE
        if self.batch_unavailable:
            order = [THREADS] if threads else []
            self.log.append("order: " + (" -> ".join(order) or "nothing") + " (no long question: batch not explored)")
            return order
        if threads and share is not None and share >= THREADS_FIRST_SHARE and large_usable:
            order, why = [THREADS, BATCH], f"the processor carries {share:.0%} of the work: threads first"
        elif threads:
            order = [BATCH, THREADS]
            why = ("large batches are not usable here (memory or question length): batch first"
                   if not large_usable else f"the card carries most of the work ({1 - (share or 0):.0%}): batch first")
        else:
            order, why = [BATCH], f"the processor carries only {share:.0%} of the work: threads deferred"
        self.log.append("order: " + " -> ".join(order) + f" ({why})")
        return order

    # ------------------------------------------------------------------ #
    # The plan                                                             #
    # ------------------------------------------------------------------ #
    def next_step(self) -> Step | None:
        """The next measurements (all in one setup), or None when nothing is left."""
        step: Step | None
        while self._pending:
            queued = self._pending.pop(0)
            fresh = self._step(queued.category, list(queued.values), queued.reason)
            if fresh is not None:
                return fresh
        if self._stage == "coarse":
            self._stage = "intermediate"
            step = self._step(PLACEMENT, self.coarse_placement(), "compare very different placements first")
            if step:
                return step
        if self._stage == "intermediate":
            self._stage = "neighbours"
            why = "partial offload" if not self.structure.fits_fully else "coarse result ambiguous"
            step = self._step(PLACEMENT, self._intermediate(), f"intermediate counts ({why})")
            if step:
                return step
        if self._stage == "neighbours":
            self._stage = "settle"
            step = self._step(PLACEMENT, self._neighbours(), "one step either side of the best count")
            if step:
                return step
        if self._stage == "settle" and self.wants_question_length and self.structure.long_question_tokens is None:
            self.waiting_for_length = True
            return None
        if self._stage == "settle":
            self.waiting_for_length = False
            self._settle_placement()
            self._order = self.category_order(self.chosen[PLACEMENT])
            self._stage = "first"
        if self._stage == "first":
            for category in self._order:
                # A clear winner of an earlier category is refined right away, in the
                # same setup, before the next category is opened.
                refine = self._refinement(category, clear_only=True)
                if refine is not None:
                    return refine
                if category in self._first_probed:
                    continue
                self._first_probed.add(category)
                values = self.thread_candidates() if category == THREADS else self.batch_candidates()
                step = self._step(category, values, "preferred candidates")
                if step:
                    return step
            self._stage = "follow"
        if self._stage == "follow":
            step = self._strongest_follow_up()
            if step is not None:
                return step
            self._stage = "done"
        return None

    def _refinement(self, category: str, *, clear_only: bool) -> Step | None:
        if category not in self._first_probed:
            return None
        context = self._context_of(category, self.base_for(category))
        if (category, context) in self._refined:
            return None
        best, _ = self.best(category)
        kind = self.kind_of(category, best)
        if kind != "clear" and (clear_only or kind != "weak"):
            return None
        around = [best - 1, best + 1] if category == THREADS else [best * 2, best // 2]
        if category == BATCH:
            around = [v for v in around if v in BATCH_SIZES and self._batch_meaningful(v)
                      and self.batch_fits(self.chosen[PLACEMENT], v)]
        self._refined.add((category, context))
        why = "refine a clear winner" if kind == "clear" else "refine a weak lead (lower priority)"
        return self._step(category, around, why)

    def chosen_placement(self) -> int:
        best, _ = self.best(PLACEMENT)
        return best if self.kind_of(PLACEMENT, best) == "clear" else self.start_layers

    def set_question_length(self, tokens: int | None) -> None:
        self.structure = replace(self.structure, long_question_tokens=tokens)
        self.wants_question_length = False
        self.waiting_for_length = False
        if tokens is None:
            self.batch_unavailable = True

    def reopen_batch(self, tokens: int | None) -> bool:
        """The prompt batch was left out earlier for lack of time and the evidence since
        (e.g. a larger reading on the short question) makes it affordable now: open it
        once, as the last category."""
        if not self.batch_unavailable or tokens is None:
            return False
        self.structure = replace(self.structure, long_question_tokens=tokens)
        self.batch_unavailable = False
        if BATCH not in self._order:
            self._order.append(BATCH)
        self._stage = "first"
        self.log.append("batch reopened: its check fits now")
        return True

    def _settle_placement(self) -> None:
        best, _ = self.best(PLACEMENT)
        kind = self.kind_of(PLACEMENT, best)
        self.placement = best if kind == "clear" else self.start_layers
        self.chosen[PLACEMENT] = self.placement
        self.log.append(
            f"placement: {self.placement} layers"
            + ("" if kind == "clear" else f" (best reading {best} is {kind}; kept the current count)")
        )

    def _stale(self, category: str) -> bool:
        """Probed earlier, but no reading in the current setup (another choice changed)."""
        return category in self._first_probed and bool(self.readings[category]) and not self.in_context(category)

    def _recheck(self, category: str) -> Step | None:
        """The category's setup changed: its choice and runner-up from the last setup it
        was measured in, measured again in the current setup (bounded)."""
        if self._rechecks_done.get(category, 0) >= MAX_RECHECKS:
            return None
        latest = self.readings[category][-1].context
        old = [r for r in self.readings[category] if r.context == latest and r.gain is not None]
        ranked = sorted(old, key=lambda r: r.gain or 0.0, reverse=True)
        values = [r.value for r in ranked if r.kind(self.smallest_gain) in ("clear", "weak")][:2]
        if self.chosen[category] != self.reference(category) and self.chosen[category] not in values:
            values = [self.chosen[category], *values][:2]
        if not values:
            return None
        self._rechecks_done[category] = self._rechecks_done.get(category, 0) + 1
        return self._step(category, values, "setup changed: recheck the choice and runner-up here")

    def _strongest_follow_up(self) -> Step | None:
        options: list[tuple[float, Callable[[], Step | None]]] = []
        for category in self._order:
            if self._stale(category):
                options.append((5.0, lambda c=category: self._recheck(c)))  # type: ignore[misc]
                continue
            best, gain = self.best(category)
            kind = self.kind_of(category, best)
            context = self._context_of(category, self.base_for(category))
            if kind in ("clear", "weak") and (category, context) not in self._refined and category in self._first_probed:
                options.append((gain / max(self._noise(category), 1e-6),
                                lambda c=category: self._refinement(c, clear_only=False)))  # type: ignore[misc]
            elif kind not in ("clear", "weak") and category not in self._fell_back and category in self._first_probed:
                def fallback(c: str = category) -> Step | None:
                    self._fell_back.add(c)
                    values = self.thread_fallback() if c == THREADS else self.batch_fallback()
                    return self._step(c, values, "preferred candidates disappointed: one fallback")
                options.append((-1.0, fallback))
        options.sort(key=lambda item: item[0], reverse=True)
        for _, make in options:
            step = make()
            if step is not None:
                return step
        return None

    # ------------------------------------------------------------------ #
    # Setup changes from outside                                           #
    # ------------------------------------------------------------------ #
    def placement_changed(self, layers: int) -> list[Step]:
        """The placement moved after threads or batch were chosen (e.g. a memory-format
        change): only their choices and runners-up are rechecked, in the new setup."""
        self.placement = int(layers)
        self.chosen[PLACEMENT] = int(layers)
        steps: list[Step] = []
        for category in TABLES.stale_after(SETTING[PLACEMENT]):
            if self._stale(category):
                self._rechecks_done[category] = 0
                step = self._recheck(category)
                if step is not None:
                    steps.append(step)
        # They were handed out already (logged); measure them next.
        for step in steps:
            for value in step.values:
                self._planned.discard((step.category, value, self._context_of(step.category, step.base)))
        self._pending.extend(steps)
        if self._stage == "done":
            self._stage = "follow"
        return steps

    # ------------------------------------------------------------------ #
    # Results                                                              #
    # ------------------------------------------------------------------ #
    def best_so_far(self) -> Setup:
        """The setup the evidence supports right now, at any point of the search: the
        best clear placement reading (even before placement is settled), and the
        current threads and batch choices (each chosen from readings in its own setup)."""
        layers = self.chosen[PLACEMENT] if self.placement is not None else self.chosen_placement()
        return (layers, self.chosen[THREADS], self.chosen[BATCH])

    def combined(self) -> dict[str, int]:
        """Settings of the best-so-far setup that differ from the start."""
        setup = self.best_so_far()
        out: dict[str, int] = {}
        for category in CATEGORIES:
            if setup[SLOT[category]] != self.reference(category):
                out[SETTING[category]] = setup[SLOT[category]]
        return out

    def weak_candidates(self) -> list[Reading]:
        """Weak readings measured in the current setup of their category, strongest (in
        noise widths) first: eligible for a repeat and, if they hold up, confirmation."""
        out = []
        for category in CATEGORIES:
            if category == PLACEMENT and self.placement is not None and self.chosen[PLACEMENT] != self.start_layers:
                continue  # a clear placement was chosen; weak alternatives are not stacked on it
            for reading in self.in_context(category):
                if reading.kind(self.smallest_gain) == "weak" and reading.value != self.chosen[category]:
                    out.append(reading)
        return sorted(out, key=lambda r: (r.gain or 0.0) / r.noise, reverse=True)

    def deferred(self) -> list[Reading]:
        """Every weak reading (any setup) not in the best-so-far setup, for the report."""
        setup = self.best_so_far()
        out = [r for c in CATEGORIES for r in self.readings[c]
               if r.kind(self.smallest_gain) == "weak" and setup[SLOT[c]] != r.value]
        return sorted(out, key=lambda r: (r.gain or 0.0) / r.noise, reverse=True)


__all__ = ["BATCH", "CATEGORIES", "PLACEMENT", "Planner", "QUESTION", "Reading", "SETTING", "SLOT",
           "Setup", "Step", "Structure", "THREADS"]
