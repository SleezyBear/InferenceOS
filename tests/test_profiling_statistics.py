"""Aggregation statistics and reproducibility bands."""

from __future__ import annotations

import pytest

from inferenceos.profiling.statistics import (
    REPRODUCIBILITY_TOLERANCE,
    summarize,
    within_tolerance,
)


def test_summarize_basic_statistics() -> None:
    stats = summarize("decode_tokens_per_sec", [10.0, 12.0, 14.0])
    assert stats.metric == "decode_tokens_per_sec"
    assert stats.count == 3
    assert stats.mean == pytest.approx(12.0)
    assert stats.median == pytest.approx(12.0)
    assert stats.min == 10.0
    assert stats.max == 14.0
    assert stats.stdev is not None
    assert stats.cv is not None and stats.cv > 0.0


def test_summarize_ignores_none_values() -> None:
    stats = summarize("x", [None, 4.0, None, 6.0])
    assert stats.count == 2
    assert stats.mean == pytest.approx(5.0)


def test_summarize_single_sample_has_no_spread() -> None:
    stats = summarize("x", [3.0])
    assert stats.count == 1
    assert stats.stdev is None
    assert stats.cv is None
    assert within_tolerance(stats, 0.1) is True


def test_summarize_empty() -> None:
    stats = summarize("x", [None, None])
    assert stats.count == 0
    assert stats.mean is None


def test_within_tolerance_respects_band() -> None:
    tight = summarize("x", [100.0, 101.0, 100.5])
    wide = summarize("x", [100.0, 200.0, 50.0])
    assert within_tolerance(tight, 0.05) is True
    assert within_tolerance(wide, 0.05) is False


def test_reproducibility_bands_are_positive() -> None:
    assert REPRODUCIBILITY_TOLERANCE
    assert all(band > 0 for band in REPRODUCIBILITY_TOLERANCE.values())
