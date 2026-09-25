"""R2: peak RSS and final RSS are distinct, correctly tracked measurements."""

from __future__ import annotations

from inferenceos.backends.process_control import PeakRssAccumulator


def test_accumulator_reports_peak_distinct_from_final() -> None:
    acc = PeakRssAccumulator()
    for sample in (100, 250, 175, 120):
        acc.add(rss_bytes=sample)
    assert acc.peak_rss_bytes == 250
    assert acc.final_rss_bytes == 120
    assert acc.peak_rss_bytes != acc.final_rss_bytes
    assert acc.sample_count == 4


def test_os_peak_working_set_raises_the_reported_peak() -> None:
    acc = PeakRssAccumulator()
    acc.add(rss_bytes=100)
    acc.add(rss_bytes=110)
    acc.add(rss_bytes=105, os_peak_rss_bytes=400)
    assert acc.peak_rss_bytes == 400
    assert acc.final_rss_bytes == 105
    assert acc.os_peak_available is True


def test_missing_samples_are_ignored_not_guessed() -> None:
    acc = PeakRssAccumulator()
    acc.add(rss_bytes=None)
    assert acc.peak_rss_bytes is None
    assert acc.final_rss_bytes is None
    assert acc.sample_count == 0
