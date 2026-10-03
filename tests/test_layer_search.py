"""The layer search: sampling, fill-in, grouping, race and safety cut-offs."""

from __future__ import annotations

from inferenceos.tune.layer_search import (
    LayerSearchSettings,
    choose_winner,
    group_counts,
    lift_lone_dips,
    run_race,
    score,
    search_layers,
    trimmed_mean,
)


def test_step_two_measures_expected_points() -> None:
    calls: list[int] = []

    def measure(count: int) -> float | None:
        calls.append(count)
        return None

    result = search_layers(
        top=12,
        n_layers=12,
        layer_kinds=None,
        measure=measure,
        should_stop=lambda: False,
    )

    assert calls == [0, 12, 1, 4, 7, 10, 11]
    assert result.tests == 7
    assert result.complete is True


def test_fill_in_splits_until_no_gaps_remain() -> None:
    measured: set[int] = set()

    def measure(count: int) -> float | None:
        measured.add(count)
        return 10.0

    search_layers(
        top=12,
        n_layers=12,
        layer_kinds=None,
        measure=measure,
        should_stop=lambda: False,
    )
    assert measured == set(range(0, 13))


def test_lone_dip_is_lifted_to_its_neighbours() -> None:
    assert lift_lone_dips([10.0, 1.0, 10.0]) == [10.0, 10.0, 10.0]
    assert lift_lone_dips([5.0, 1.0, 2.0]) == [5.0, 2.0, 2.0]
    assert lift_lone_dips([1.0, 1.0, 1.0]) == [1.0, 1.0, 1.0]


def test_grouping_by_kind_and_its_measure_order() -> None:
    kinds = ["a", "a", "a", "a", "b", "b", "b", "b"]
    groups = group_counts(8, 8, kinds)
    assert [0] in groups
    assert [8] in groups
    assert [1, 2, 3, 4] in groups
    assert [5, 6, 7] in groups

    calls: list[int] = []

    def measure(count: int) -> float | None:
        calls.append(count)
        return None

    search_layers(
        top=8,
        n_layers=8,
        layer_kinds=kinds,
        measure=measure,
        should_stop=lambda: False,
    )
    assert calls == [0, 8, 1, 4, 5, 7]


def test_race_drops_a_clearly_worse_count_and_keeps_near_equal() -> None:
    settings = LayerSearchSettings()
    readings: dict[int, list[float]] = {0: [10.0] * 5, 1: [10.0] * 5, 2: [5.0] * 5}

    def measure_once(count: int) -> bool:
        readings[count].append(readings[count][0])
        return True

    remaining = run_race(
        [0, 1, 2],
        readings=readings,
        measure_once=measure_once,
        score_of=lambda count: score(readings[count], settings),
        settings=settings,
    )
    assert 2 not in remaining
    assert set(remaining) == {0, 1}


def test_tie_rule_prefers_the_smaller_when_close() -> None:
    close = {3: 99.5, 5: 100.0}
    assert choose_winner([5, 3], lambda count: close[count]) == 3
    apart = {3: 99.0, 5: 100.0}
    assert choose_winner([5, 3], lambda count: apart[count]) == 5


def test_race_keeps_going_when_noise_is_large() -> None:
    # Near-equal scores, but each reading swings by 20%, so the race must not trust
    # the "they measure the same" signal and must keep measuring to the round cap.
    settings = LayerSearchSettings()
    readings: dict[int, list[float]] = {0: [10.0], 1: [10.0]}
    calls = {"n": 0}
    cycle = [8.0, 12.0]

    def measure_once(count: int) -> bool:
        calls["n"] += 1
        readings[count].append(cycle[calls["n"] % 2])
        return True

    remaining = run_race(
        [0, 1],
        readings=readings,
        measure_once=measure_once,
        score_of=lambda count: score(readings[count], settings),
        settings=settings,
    )
    assert calls["n"] == 2 * settings.race_max_rounds
    assert set(remaining) == {0, 1}


def test_jumpy_count_is_judged_by_its_lower_readings() -> None:
    assert trimmed_mean([10.0, 10.0, 1.0]) == 10.0
    assert score([20.0, 20.0, 10.0, 10.0]) == 10.0
    assert score([10.0, 10.0, 10.0]) == 10.0


def test_should_stop_cuts_the_search_short() -> None:
    calls: list[int] = []
    counter = {"calls": 0}

    def measure(count: int) -> float | None:
        calls.append(count)
        return 10.0

    def should_stop() -> bool:
        counter["calls"] += 1
        return counter["calls"] > 3

    result = search_layers(
        top=12,
        n_layers=12,
        layer_kinds=None,
        measure=measure,
        should_stop=should_stop,
    )
    assert result.complete is False
    assert result.chosen is not None
    assert len(calls) == 3


def test_failed_counts_are_skipped() -> None:
    calls: list[int] = []

    def measure(count: int) -> float | None:
        calls.append(count)
        return None if count == 4 else 10.0

    result = search_layers(
        top=12,
        n_layers=12,
        layer_kinds=None,
        measure=measure,
        should_stop=lambda: False,
    )
    assert 4 in result.failed
    assert calls.count(4) == 1
