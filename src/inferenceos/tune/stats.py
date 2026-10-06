"""Small statistics for the fair comparison (no outside packages).

The fair comparison measures the current and the new setup in independent *blocks*
(each block: both setups, each from a fresh load, the order alternating between
blocks). Each block gives one score: how much better the new setup was in that block.
The block scores are treated as independent draws, and Student's t distribution gives
how sure we can be about their mean.

* "better": the lower confidence bound of the mean is above 0 (at most ``alpha`` chance
  of calling a change better when it really is not).
* "not worse": the lower confidence bound is above ``-margin`` (at most ``alpha``
  chance of calling a change not worse when it really loses ``margin`` or more).

When several looks at the data are allowed (stop early when the answer is already
clear), ``alpha`` is split evenly between the looks (Bonferroni), which keeps the
overall chance of a wrong "better" at or below ``alpha``.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    max_iterations, eps, fpmin = 300, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    x = df / (df + t * t)
    tail = 0.5 * _incomplete_beta(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p: float, df: float) -> float:
    """The t value with P(T <= t) = ``p``."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be between 0 and 1")
    low, high = -1.0e3, 1.0e3
    for _ in range(200):
        mid = (low + high) / 2.0
        if t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


@dataclass(frozen=True)
class BlockTest:
    """What the block scores say."""

    blocks: int
    mean: float | None
    standard_error: float | None
    #: Lower and upper one-sided confidence bounds of the mean at the look's alpha.
    lower: float | None
    upper: float | None
    alpha: float

    @property
    def known(self) -> bool:
        return self.mean is not None and self.lower is not None


def block_bounds(values: Sequence[float], *, alpha: float) -> BlockTest:
    """One-sided bounds for the mean of ``values`` at level ``alpha``.

    Needs at least 2 blocks; with fewer, nothing is known.
    """
    items = [float(value) for value in values]
    count = len(items)
    if count < 2:
        mean = items[0] if items else None
        return BlockTest(count, mean, None, None, None, alpha)
    mean = statistics.fmean(items)
    sd = statistics.stdev(items)
    se = sd / math.sqrt(count)
    critical = t_quantile(1.0 - alpha, count - 1)
    return BlockTest(count, mean, se, mean - critical * se, mean + critical * se, alpha)


__all__ = ["BlockTest", "block_bounds", "t_cdf", "t_quantile"]
