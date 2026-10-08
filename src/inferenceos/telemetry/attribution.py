"""Attribution of Windows counter readings to answer intervals (experimental
measurement-scoped monitor; reviewer follow-up after job 120).

Counter semantics (from the counter implementation, not from arrival spacing):

* ``\\Memory\\Pages Input/sec`` is a ``PERF_COUNTER_COUNTER`` counter (.NET
  ``PerformanceCounterType.RateOfCountsPerSecond32`` = 272696320). Its raw value is a
  cumulative count of pages read from disk to resolve hard faults, and the documented
  formula for its rate is ``(N1 - N0) / ((D1 - D0) / F)``: the count difference
  between two readings over the time between them. A rate therefore describes exactly
  the interval between two readings, and the *count* difference says how many pages
  were read in it. Job 120's raw data confirms the raw value is cumulative (monotonic)
  and shows why the count matters: Get-Counter's cooked readings cover only its own
  internal ~1 s pair inside each 1.74 s cycle, and one interval of 227 pages, at the
  end of the long answer, showed a cooked reading of 0.00.
* Instantaneous counters (``NumberOfItems32/64``) describe a point in time.
* Other types (e.g. ``% Processor Performance``) are evaluated by .NET's
  ``CounterSample.Calculate(previous, current)``, which by definition describes the
  interval between those two readings.

Timing. The sampler reads each counter with ``NextSample()`` (no Get-Counter one-second
wait) and records the system clock (UTC) just before and just after the read; the
reading happened inside that bracket. Answers are timed with Python's ``time.time()``
(the same system clock). An interval between readings ``r0`` and ``r1`` therefore lies
within ``[r0.before, r1.after]`` and certainly covers ``[r0.after, r1.before]``. A
clock slack (``CLOCK_SLACK_S``) is added on both sides for the clock's resolution in
Windows PowerShell 5.1 (.NET Framework ``DateTime.UtcNow``, ~1-16 ms). A reading
whose arrival in Python is earlier than its own read time minus the slack, or later
than ``MAX_ARRIVAL_LAG_S``, has unusable timing (e.g. the system clock was changed).

Rules for one answer ``[A, B]`` and the paging counter:

* **outside**: the interval certainly ends before ``A`` or starts after ``B``;
  **inside**: it certainly lies within ``[A, B]``; **boundary**: anything else
  (it overlaps an edge, or its uncertainty does). Nothing is split evenly.
* **covered**: a usable reading certainly before ``A`` and one certainly after ``B``
  exist, and every interval between them is usable. Otherwise the paging is
  **unknown** -- never zero.
* A **boundary interval with any pages** makes the answer's paging **unknown**
  (ambiguous): the pages may have fallen inside or outside, and the counter cannot
  tell. Its count and an upper bound are kept as evidence.
* Otherwise the answer's pages are exact (the sum over inside intervals; a boundary
  interval with zero pages proves zero pages in its overlap). The guard's quantity is
  the peak rate over a ``PEAK_SPAN_S`` window (1 s, the time base of Get-Counter's
  default readings on which the paging guard's thresholds were set; the whole answer
  if it is shorter). It is bounded from the counts and the **full** timing
  uncertainty of every interval -- its pages fell somewhere in
  ``[outer_start, outer_end]`` (both read brackets plus the clock slack), never at a
  midpoint (reviewer, monitor round 3): ``lower`` = a peak every consistent placement
  of the pages reaches, ``upper`` = a peak no consistent placement exceeds. The rate is
  **established** only if every interval holding pages spans at most
  ``MAX_PAGED_INTERVAL_S``; a longer paged interval (a sampling stall) leaves it
  **unknown**: a burst could sit anywhere inside it, so dividing by its length would
  dilute it (reviewer: 300 pages over a 4 s gap read 75/s). The exact count is still
  reported.
* A measured answer carries both bounds. Its ``value`` is the peak rate only when the
  bounds coincide (``exact``); otherwise it is ``None`` and consumers must use the
  bounds (the tune's paging guard evaluates its decision over them -- an upper bound
  is never passed off as an exact reading).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

#: .NET PerformanceCounterType values whose raw value is a cumulative count and whose
#: rate is (N1 - N0) / elapsed time between readings.
CUMULATIVE_TYPES: dict[int, str] = {
    272696320: "RateOfCountsPerSecond32 (PERF_COUNTER_COUNTER)",
    272696576: "RateOfCountsPerSecond64 (PERF_COUNTER_BULK_COUNT)",
}
#: Instantaneous counters (a value at the reading's time).
INSTANT_TYPES: dict[int, str] = {65536: "NumberOfItems32", 65792: "NumberOfItems64"}

CLOCK_SLACK_S = 0.02
MAX_ARRIVAL_LAG_S = 2.0
PEAK_SPAN_S = 1.0
#: Longest interval that may hold pages for the peak rate to count as established:
#: a quarter of the 1 s window. A measurement-validity rule (when the evidence can
#: support a rate at all), not an acceptance rule.
MAX_PAGED_INTERVAL_S = 0.25
PAGING_SPEC = "Memory|Pages Input/sec|"
CPU_PERFORMANCE_SPEC = "Processor Information|% Processor Performance|_Total"

#: .NET DateTime ticks at the Unix epoch.
_EPOCH_TICKS = 621355968000000000


def ticks_to_unix(ticks: int) -> float:
    return (int(ticks) - _EPOCH_TICKS) / 1e7


@dataclass(frozen=True)
class RawReading:
    """One ``NextSample()`` of one counter, as the sampler reported it."""

    spec: str
    read_before: float  # UTC seconds since the epoch, just before the read
    read_after: float   # just after the read
    raw: int
    base: int
    ts100: int          # the counter's own 100-ns timestamp
    counter_type: int
    calculated: float | None  # Calculate(previous, this): the interval since the previous reading
    arrival_wall: float  # Python time.time() when the line arrived
    arrival_perf: float

    def timing_problem(self) -> str | None:
        if self.read_after < self.read_before:
            return "read bracket is reversed (clock changed during the read)"
        lag = self.arrival_wall - self.read_after
        if lag < -CLOCK_SLACK_S:
            return f"arrived {-lag:.3f} s before it was read (clock changed)"
        if lag > MAX_ARRIVAL_LAG_S:
            return f"arrived {lag:.3f} s after it was read (delayed or clock changed)"
        return None


@dataclass(frozen=True)
class Interval:
    """The span between two consecutive readings of one counter."""

    spec: str
    outer_start: float  # earliest the span can start (previous read_before - slack)
    inner_start: float  # latest it can start (previous read_after + slack)
    inner_end: float    # earliest it can end (this read_before - slack)
    outer_end: float    # latest it can end (this read_after + slack)
    count: int | None   # pages (cumulative counters only)
    rate: float | None
    value: float | None  # Calculate() value (descriptive counters)
    usable: bool
    problem: str | None = None

    @property
    def nominal_start(self) -> float:
        return (self.outer_start + self.inner_start) / 2

    @property
    def nominal_end(self) -> float:
        return (self.inner_end + self.outer_end) / 2

    def place(self, start: float, end: float) -> str:
        """``inside``, ``outside`` or ``boundary`` relative to ``[start, end]``."""
        if self.outer_end <= start or self.outer_start >= end:
            return "outside"
        if self.outer_start >= start and self.outer_end <= end:
            return "inside"
        return "boundary"

    def record(self, start: float) -> dict[str, object]:
        return {"from_s": round(self.nominal_start - start, 3), "to_s": round(self.nominal_end - start, 3),
                "count": self.count, "rate": None if self.rate is None else round(self.rate, 3),
                "value": None if self.value is None else round(self.value, 3),
                "usable": self.usable, "problem": self.problem}


def intervals_for(readings: Sequence[RawReading], spec: str) -> list[Interval]:
    """Consecutive-reading intervals of one counter, oldest first."""
    mine = sorted((r for r in readings if r.spec == spec), key=lambda r: (r.read_after, r.read_before))
    out: list[Interval] = []
    for previous, current in zip(mine, mine[1:]):
        problem = previous.timing_problem() or current.timing_problem()
        count: int | None = None
        rate: float | None = None
        if problem is None and current.counter_type != previous.counter_type:
            problem = "counter type changed between readings"
        if problem is None and current.counter_type in CUMULATIVE_TYPES:
            delta = current.raw - previous.raw
            seconds = (current.ts100 - previous.ts100) / 1e7
            if delta < 0:
                problem = f"raw count went backwards by {-delta} (counter reset)"
            elif seconds <= 0:
                problem = "counter timestamp did not advance"
            else:
                count, rate = int(delta), delta / seconds
        elif problem is None and current.counter_type not in INSTANT_TYPES and current.calculated is None:
            problem = "no calculated value for this interval"
        value = None
        if current.calculated is not None and math.isfinite(current.calculated):
            value = float(current.calculated)
        out.append(Interval(
            spec=spec,
            outer_start=previous.read_before - CLOCK_SLACK_S,
            inner_start=previous.read_after + CLOCK_SLACK_S,
            inner_end=current.read_before - CLOCK_SLACK_S,
            outer_end=current.read_after + CLOCK_SLACK_S,
            count=count, rate=rate, value=value,
            usable=problem is None, problem=problem,
        ))
    return out


@dataclass
class PagingVerdict:
    """The paging evidence of one answer."""

    status: str  # "measured" | "unknown" | "pending"
    value: float | None
    pages: int | None
    reason: str
    inside: int = 0
    boundary: int = 0
    boundary_pages: int = 0
    #: Bounds on the peak rate the evidence allows (pages/s); when boundary pages make
    #: it ambiguous, the upper bound assumes they all fell inside.
    upper_bound_rate: float | None = None
    lower_bound_rate: float | None = None
    intervals: list[dict[str, object]] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        """Measured, and the bounds coincide (``value`` is then the peak rate)."""
        return (self.status == "measured" and self.lower_bound_rate is not None
                and self.upper_bound_rate is not None
                and abs(self.upper_bound_rate - self.lower_bound_rate) <= 1e-9)

    def record(self) -> dict[str, object]:
        return {"status": self.status, "value": None if self.value is None else round(self.value, 3),
                "pages": self.pages, "reason": self.reason, "inside": self.inside,
                "boundary": self.boundary, "boundary_pages": self.boundary_pages,
                "upper_bound_rate": None if self.upper_bound_rate is None else round(self.upper_bound_rate, 3),
                "lower_bound_rate": None if self.lower_bound_rate is None else round(self.lower_bound_rate, 3),
                "exact": self.exact, "intervals": self.intervals}


def paging_verdict(readings: Sequence[RawReading], start: float, end: float, *,
                   spec: str = PAGING_SPEC, peak_span_s: float = PEAK_SPAN_S) -> PagingVerdict:
    """Paging during the answer ``[start, end]`` (wall-clock seconds) -- see module docs."""
    mine = [r for r in readings if r.spec == spec]
    if not mine:
        return PagingVerdict("unknown", None, None, "no paging reading at all")
    types = {r.counter_type for r in mine}
    if not types <= set(CUMULATIVE_TYPES):
        return PagingVerdict("unknown", None, None,
                             f"counter type {sorted(types)} has no count semantics; not interpreted")
    usable = [r for r in mine if r.timing_problem() is None]
    if not any(r.read_after + CLOCK_SLACK_S <= start for r in usable):
        return PagingVerdict("unknown", None, None, "no usable reading certainly before the answer started")
    if not any(r.read_before - CLOCK_SLACK_S >= end for r in usable):
        return PagingVerdict("pending", None, None, "no reading certainly after the answer ended (yet)")
    spans = intervals_for(mine, spec)
    relevant = [iv for iv in spans if iv.place(start, end) != "outside"]
    records = [iv.record(start) | {"place": iv.place(start, end)} for iv in relevant]
    bad = [iv for iv in relevant if not iv.usable]
    if bad or not relevant:
        why = bad[0].problem if bad else "no interval overlaps the answer"
        return PagingVerdict("unknown", None, None, f"unusable evidence: {why}", intervals=records)
    inside = [iv for iv in relevant if iv.place(start, end) == "inside"]
    boundary = [iv for iv in relevant if iv.place(start, end) == "boundary"]
    paged_edges = [iv for iv in boundary if (iv.count or 0) > 0]
    if paged_edges:
        upper = None
        for iv in paged_edges:
            overlap = min(iv.inner_end, end) - max(iv.inner_start, start)  # certain overlap
            bound = math.inf if overlap <= 0 else (iv.count or 0) / overlap
            upper = bound if upper is None else max(upper, bound)
        pages = sum(iv.count or 0 for iv in paged_edges)
        return PagingVerdict(
            "unknown", None, None,
            f"ambiguous: {pages} page(s) read in an interval overlapping the answer's edge; "
            "the counter cannot tell whether they fell inside",
            inside=len(inside), boundary=len(boundary), boundary_pages=pages,
            upper_bound_rate=upper, intervals=records)
    # Exact pages: they come only from inside intervals (whose whole possible span lies
    # within the answer); zero-page edges prove zero. Each paged interval keeps its full
    # possible span -- both read brackets and the clock slack -- for the rate bounds.
    pieces = [(iv.outer_start, iv.outer_end, iv.count or 0)
              for iv in sorted(inside, key=lambda i: (i.outer_start, i.outer_end)) if (iv.count or 0) > 0]
    pages = sum(c for _, _, c in pieces)
    length = end - start
    window = min(peak_span_s, length) if length > 0 else 0.0
    lower, upper = peak_rate_bounds(pieces, window)
    # The guard's quantity is the peak rate over a 1 s window. A count is exact, but a
    # rate is only established when every interval holding pages is short enough to
    # place those pages within such a window; otherwise the pages may have come in a
    # burst anywhere inside a long interval, and dividing by its length would dilute
    # it. Then the rate is unknown, never a diluted "measured" value.
    coarse = [(hi - lo, c) for lo, hi, c in pieces if hi - lo > MAX_PAGED_INTERVAL_S + 1e-9]
    if coarse:
        longest, count = max(coarse)
        return PagingVerdict(
            "unknown", None, pages,
            f"rate not resolved: {count} page(s) within a {longest:.2f} s interval (longer than "
            f"{MAX_PAGED_INTERVAL_S:g} s); the {window:g} s peak could be anywhere from "
            f"{lower:.1f}/s to {upper:.1f}/s",
            inside=len(inside), boundary=len(boundary), intervals=records,
            lower_bound_rate=lower, upper_bound_rate=upper)
    exact = abs(upper - lower) <= 1e-9
    return PagingVerdict("measured", upper if exact else None, pages,
                         f"{pages} page(s) inside the answer; every overlapping edge read 0"
                         + ("" if exact else f"; the {window:g} s peak rate is between {lower:.1f}/s and "
                            f"{upper:.1f}/s (bounds only, not an exact reading)"),
                         inside=len(inside), boundary=len(boundary), intervals=records,
                         lower_bound_rate=lower, upper_bound_rate=upper)


def peak_rate_bounds(pieces: Sequence[tuple[float, float, int]], window: float) -> tuple[float, float]:
    """Bounds on the highest page rate over any closed ``window``-long span.

    ``pieces`` are ``(earliest, latest, pages)``: each interval's pages fell somewhere
    in ``[earliest, latest]`` -- its full possible span, read brackets and clock slack
    included (no midpoints, no even spreading). Ordered by time. For every run of
    consecutive pieces ``i..j``:

    * upper: a window can hold pages of ``i..j`` together only if one window can touch
      all their spans (``earliest_j - latest_i <= window``); no consistent placement
      puts more in a window than the largest such sum;
    * lower: all pages of ``i..j`` lie within ``[earliest_i, latest_j]``, which
      ``ceil(span / window)`` windows cover, so some window holds at least
      ``sum / ceil(span / window)`` pages in every consistent placement.

    Both are divided by the window. Pages-free input gives (0, 0)."""
    if window <= 0 or not pieces:
        return 0.0, 0.0
    ordered = sorted(pieces, key=lambda p: (p[0], p[1]))
    lower = upper = 0.0
    n = len(ordered)
    for i in range(n):
        total = 0
        latest_end = earliest_end = ordered[i][1]
        for j in range(i, n):
            total += ordered[j][2]
            latest_end = max(latest_end, ordered[j][1])
            earliest_end = min(earliest_end, ordered[j][1])
            span = latest_end - ordered[i][0]
            windows = max(1, math.ceil(span / window - 1e-9))
            lower = max(lower, total / windows)
            # One window touches every span of i..j iff the latest start minus the
            # earliest end fits in it.
            if ordered[j][0] - earliest_end <= window + 1e-9:
                upper = max(upper, float(total))
    return lower / window, upper / window


def interval_mean(readings: Sequence[RawReading], start: float, end: float, spec: str) -> float | None:
    """Mean ``Calculate()`` value of the usable intervals overlapping ``[start, end]``
    (descriptive measures such as processor performance; not a guard)."""
    values = [iv.value for iv in intervals_for(readings, spec)
              if iv.usable and iv.value is not None and iv.place(start, end) != "outside"]
    return sum(values) / len(values) if values else None


__all__ = [
    "CLOCK_SLACK_S", "CPU_PERFORMANCE_SPEC", "CUMULATIVE_TYPES", "INSTANT_TYPES", "Interval",
    "MAX_ARRIVAL_LAG_S", "MAX_PAGED_INTERVAL_S", "PAGING_SPEC", "PEAK_SPAN_S", "PagingVerdict",
    "RawReading", "interval_mean", "intervals_for", "paging_verdict", "peak_rate_bounds", "ticks_to_unix",
]
