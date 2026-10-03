"""Find the real largest graphics-card layer count that loads (prompt B).

The formula estimate of how many layers fit is deliberately careful, and lands well
below the real top on some models. Before the speed search runs, this does a few
*load-only* tries (no measuring) to find the largest count that truly loads and still
leaves the safety margins free. Everything that touches the machine is injected, so the
search can be tested without a model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

TryLoad = Callable[[int], "LoadAttempt"]


@dataclass(frozen=True)
class LoadAttempt:
    """The outcome of one load-only try.

    ``loaded`` says whether the model actually came up at that layer count; ``fits``
    adds the safety-margin test on top. ``error`` keeps the full failure reason when the
    load raised (or the model did not report itself loaded) so a "does not fit" is never
    silent.
    """

    loaded: bool
    fits: bool
    error: str | None = None
    used_bytes: int | None = None
    free_bytes: int | None = None


@dataclass(frozen=True)
class LayerTopResult:
    """The largest count that fits, plus evidence for the report."""

    top: int
    tries: list[dict[str, object]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    used_estimate: bool = False


def _gib(value: object) -> str:
    if not isinstance(value, int):
        return "unknown"
    return f"{value / 1024**3:.2f} GB"


def find_layer_top(
    *,
    n_layers: int,
    estimate: int | None,
    try_load: TryLoad | None,
    memory_readable: bool = True,
    say: Callable[[str], None] = print,
) -> LayerTopResult:
    """Return the largest layer count that loads, or the formula estimate.

    ``try_load(N)`` must start the model at N layers at the fixed context and return a
    :class:`LoadAttempt`; a failed load (including out of graphics memory) is simply
    "does not fit". When ``try_load`` is missing or the graphics memory cannot be read,
    the formula ``estimate`` is used and the reason is recorded.
    """
    if n_layers < 1:
        return LayerTopResult(0, reasons=["the model reports no layers"], used_estimate=True)

    if try_load is None or not memory_readable:
        fallback = 0 if estimate is None else max(0, min(int(estimate), int(n_layers)))
        reason = (
            "graphics memory could not be read; using the formula estimate"
            if not memory_readable
            else "load-only tries are not possible; using the formula estimate"
        )
        return LayerTopResult(fallback, reasons=[reason], used_estimate=True)

    tries: list[dict[str, object]] = []
    reasons: list[str] = []

    def attempt(count: int) -> bool:
        outcome = try_load(count)
        entry: dict[str, object] = {
            "layers": count,
            "loaded": bool(outcome.loaded),
            "fits": bool(outcome.fits),
            "error": outcome.error,
            "used_bytes": outcome.used_bytes,
            "free_bytes": outcome.free_bytes,
        }
        tries.append(entry)
        state = "fits" if outcome.fits else (
            "loaded but does not fit" if outcome.loaded else "did not load"
        )
        say(
            f"  load-only: {count} layers {state} "
            f"(used {_gib(outcome.used_bytes)}, free {_gib(outcome.free_bytes)})"
        )
        if outcome.error:
            say(f"    load error: {outcome.error}")
        return outcome.fits

    if attempt(n_layers):
        reasons.append(f"all {n_layers} layers load")
        return LayerTopResult(n_layers, tries=tries, reasons=reasons)

    high = n_layers
    low = 0
    if estimate is not None and 0 < int(estimate) < n_layers:
        if attempt(int(estimate)):
            low = int(estimate)
        else:
            high = int(estimate)
    reasons.append(f"the top is between {low} and {high} layers")
    while high - low > 1:
        mid = (low + high) // 2
        if any(entry["layers"] == mid for entry in tries):
            break
        if attempt(mid):
            low = mid
        else:
            high = mid
    reasons.append(f"{low} layers is the largest count that loads")
    return LayerTopResult(low, tries=tries, reasons=reasons)


__all__ = ["LayerTopResult", "LoadAttempt", "TryLoad", "find_layer_top"]
