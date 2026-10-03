"""Heat check for ``tune``: does the laptop slow down from heat under steady use?

InferenceOS cannot cool the laptop; this only *finds out and reports*. The chosen
setup answers several longer questions in a row on one load, and the processor's
real speed (Windows "% Processor Performance") and the writing speed are compared
between the first and the last part of the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inferenceos.contracts.experiment import ExperimentMeasurement


@dataclass
class HeatCheck:
    runs: int = 0
    seconds: float = 0.0
    processor_speeds: list[float | None] = field(default_factory=list)
    writing_speeds: list[float | None] = field(default_factory=list)
    processor_start: float | None = None
    processor_end: float | None = None
    writing_start: float | None = None
    writing_end: float | None = None
    throttled: bool = False
    processor_slowed: bool = False
    words: str = ""

    def record(self) -> dict[str, object]:
        return {
            "runs": self.runs,
            "seconds": round(self.seconds, 1),
            "processor_speeds": self.processor_speeds,
            "writing_speeds": self.writing_speeds,
            "processor_start": self.processor_start,
            "processor_end": self.processor_end,
            "writing_start": self.writing_start,
            "writing_end": self.writing_end,
            "throttled": self.throttled,
            "processor_slowed": self.processor_slowed,
            "words": self.words,
        }


# The writing speed must fall by at least this much (first third -> last third) for a
# steady slowdown to count as heat. On the laptop (9B, Oct 2026) the processor speed
# fell 92% -> 81% after 5 minutes while the writing speed stayed at 4.32 words/s: a
# change the user never feels is not reported as heat.
WRITING_DROP = 0.03


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def analyse_heat(
    measurement: ExperimentMeasurement, *, tolerance: float, seconds: float = 0.0
) -> HeatCheck:
    """Compare the first and last third of a run of answers on one load.

    The laptop counts as slowing down from heat only when it actually got slower:

    * steady: the processor speed in the last third is more than ``tolerance`` points
      below the first third **and** the writing speed fell by at least 3%; or
    * sudden: an answer ran more than twice ``tolerance`` points below the fastest one
      **and** that answer was also written at least 5% slower than the middle answer
      (the first answer after a load is skipped: it is always a little slower).

    A run that speeds up, or whose processor reading wobbles without the answers
    getting slower, is never reported as heat (seen on Gemma, Oct 2026: one reading
    of 72% while writing went 29 -> 32 words/s). When only the processor slowed and the
    answers did not, that is reported as a note (``processor_slowed``), not as heat.
    The first answer after a load is left out of the thirds: it is sometimes faster
    (cool start) and sometimes slower (warm-up) than the rest.
    """
    clocks = [run.cpu_performance_percent for run in measurement.run_metrics]
    writing = [run.decode_tokens_per_sec for run in measurement.run_metrics]
    check = HeatCheck(
        runs=len(measurement.run_metrics),
        seconds=seconds,
        processor_speeds=[None if c is None else round(float(c), 1) for c in clocks],
        writing_speeds=[None if w is None else round(float(w), 3) for w in writing],
    )
    known = [float(c) for c in clocks if c is not None]
    if len(known) < 3:
        check.words = (
            "heat check: the processor speed could not be read, so heat could not be "
            "checked"
        )
        return check
    skip = 1 if len(clocks) >= 4 else 0
    body_clocks, body_writing = clocks[skip:], writing[skip:]
    third = max(len(body_clocks) // 3, 1)
    first = [float(c) for c in body_clocks[:third] if c is not None]
    last = [float(c) for c in body_clocks[-third:] if c is not None]
    first_w = [float(w) for w in body_writing[:third] if w is not None]
    last_w = [float(w) for w in body_writing[-third:] if w is not None]
    check.processor_start, check.processor_end = _mean(first), _mean(last)
    check.writing_start, check.writing_end = _mean(first_w), _mean(last_w)
    drop = (
        check.processor_start - check.processor_end
        if check.processor_start is not None and check.processor_end is not None
        else 0.0
    )
    writing_fell = bool(
        check.writing_start
        and check.writing_end
        and check.writing_end < (1.0 - WRITING_DROP) * check.writing_start
    )
    steady = drop > tolerance and writing_fell
    known_writing = sorted(float(w) for w in writing if w is not None)
    middle_writing = known_writing[len(known_writing) // 2] if known_writing else None
    fastest = max(known)
    # The first answer after a load is always a little slower (warm-up, measured on
    # the laptop), so it never counts as a sudden dip.
    later = list(zip(clocks, writing))[1:]
    sudden = any(
        c is not None
        and w is not None
        and middle_writing is not None
        and float(c) < fastest - 2.0 * tolerance
        and float(w) < 0.95 * middle_writing
        for c, w in later
    )
    check.throttled = steady or sudden
    check.processor_slowed = not check.throttled and (
        drop > tolerance
        or any(c is not None and float(c) < fastest - 2.0 * tolerance for c, _ in later)
    )
    writing_text = ""
    if check.writing_start and check.writing_end:
        change = check.writing_end / check.writing_start - 1.0
        writing_text = (
            f"; writing speed {check.writing_start:.2f} -> {check.writing_end:.2f} "
            f"words/s ({change:+.0%})"
        )
    span = f"{check.runs} answers in a row" + (
        f", {check.seconds / 60:.1f} min" if check.seconds else ""
    )
    if check.throttled:
        check.words = (
            f"HEAT: this laptop slows down from heat under steady use ({span}): processor "
            f"speed {check.processor_start:.0f}% -> {check.processor_end:.0f}% "
            f"(lowest {min(known):.0f}%){writing_text}. This is the laptop's cooling; "
            "InferenceOS cannot change it. Expect long sessions to run slower than the "
            "speeds measured here."
        )
    elif check.processor_slowed:
        check.words = (
            f"heat check: the processor slowed down during steady use ({span}): "
            f"{check.processor_start:.0f}% -> {check.processor_end:.0f}% (lowest "
            f"{min(known):.0f}%), probably from heat, but the answers did not get slower"
            f"{writing_text}, so this model is not held back by it"
        )
    else:
        check.words = (
            f"heat check: no slowdown from heat seen ({span}): processor speed "
            f"{check.processor_start:.0f}% -> {check.processor_end:.0f}%{writing_text}"
        )
    return check


__all__ = ["HeatCheck", "analyse_heat"]
