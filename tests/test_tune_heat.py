"""The heat check reports a real slowdown from heat, and nothing else."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from inferenceos.contracts.experiment import ExperimentMeasurement, RuntimeConfiguration
from inferenceos.contracts.observation import RunMetrics
from inferenceos.tune.heat import analyse_heat


def _measurement(clocks: list[float], writing: list[float]) -> ExperimentMeasurement:
    return ExperimentMeasurement(
        measurement_id=f"m-{uuid.uuid4().hex[:6]}",
        created_at=datetime.now(timezone.utc),
        phase="heat",
        configuration=RuntimeConfiguration(context_length=4096),
        repetitions=len(clocks),
        valid_runs=len(clocks),
        invalid_runs=0,
        warmup_runs=1,
        total_runs=len(clocks) + 1,
        run_metrics=[
            RunMetrics(cpu_performance_percent=c, decode_tokens_per_sec=w)
            for c, w in zip(clocks, writing)
        ],
    )


def test_a_steady_slowdown_from_heat_is_reported() -> None:
    check = analyse_heat(
        _measurement([110, 108, 104, 98, 92, 88, 85, 84], [5.0, 5.0, 4.9, 4.8, 4.6, 4.5, 4.4, 4.4]),
        tolerance=8.0,
    )
    assert check.throttled and check.words.startswith("HEAT:")


def test_a_sudden_dip_that_slows_an_answer_is_reported() -> None:
    check = analyse_heat(
        _measurement([99, 99, 98, 70, 99, 98, 99, 99], [20, 20, 20, 16, 20, 20, 20, 20]),
        tolerance=8.0,
    )
    assert check.throttled


def test_speeding_up_is_never_reported_as_heat() -> None:
    # Gemma on the laptop: one 72% reading while writing went 29 -> 32 words/s.
    check = analyse_heat(
        _measurement([79, 72, 85, 84, 86, 85, 86, 87], [29.0, 29.3, 30.5, 31.0, 31.8, 32.0, 32.1, 32.2]),
        tolerance=8.0,
    )
    assert not check.throttled and not check.processor_slowed
    assert check.words.startswith("heat check: no slowdown")


def test_a_processor_reading_wobble_without_slower_answers_is_not_heat() -> None:
    check = analyse_heat(
        _measurement([99, 99, 75, 99, 99, 98, 99, 99], [20, 20, 20, 20, 20, 20, 20, 20]),
        tolerance=8.0,
    )
    assert not check.throttled


def test_steady_cool_running_is_not_heat() -> None:
    check = analyse_heat(
        _measurement([99, 98, 99, 97, 98, 99, 98, 97], [20, 20.1, 19.9, 20, 20, 19.9, 20, 20.1]),
        tolerance=8.0,
    )
    assert not check.throttled


def test_no_processor_reading_means_no_verdict() -> None:
    measurement = _measurement([0, 0, 0], [20, 20, 20])
    measurement = measurement.model_copy(
        update={"run_metrics": [RunMetrics(decode_tokens_per_sec=20.0) for _ in range(3)]}
    )
    check = analyse_heat(measurement, tolerance=8.0)
    assert not check.throttled and "could not be read" in check.words


def test_the_slower_first_answer_after_a_load_is_not_heat() -> None:
    # Gemma's real heat check on the laptop (Oct 2026): the first answer ran at 71.5%
    # and 28.5 words/s, then it sped up to 89% and 32 words/s.
    check = analyse_heat(
        _measurement(
            [71.5, 85.6, 78.8, 82.5, 80.8, 85.1, 82.5, 88.8],
            [28.472, 29.81, 29.848, 29.216, 31.349, 30.504, 32.603, 31.757],
        ),
        tolerance=8.0,
    )
    assert not check.throttled


def test_processor_slowing_without_slower_answers_is_a_note_not_heat() -> None:
    # The 9B on the laptop after 15 minutes idle (Oct 2026): the processor fell from
    # about 93% to 81% after answer 8 and stayed there; writing stayed at 4.3 words/s.
    clocks = [95.4, 88.2, 96.4, 93.3, 93.5, 93.0, 92.5, 92.5, 82.2, 78.4, 84.6, 83.1, 82.1, 82.0,
              81.5, 82.1, 82.3, 81.4, 81.2, 79.8, 81.2, 81.1, 80.8, 79.7, 79.7, 83.7, 81.1]
    writing = [4.645, 4.284, 4.228, 4.262, 4.331, 4.181, 4.399, 4.235, 4.344, 4.259, 4.423,
               4.359, 4.404, 4.402, 4.356, 4.376, 4.376, 4.364, 4.349, 4.317, 4.334, 4.333,
               4.325, 4.275, 4.285, 4.312, 4.34]
    check = analyse_heat(_measurement(clocks, writing), tolerance=8.0)
    assert not check.throttled
    assert check.processor_slowed
    assert "did not get slower" in check.words


def test_a_hot_but_steady_laptop_is_not_heat() -> None:
    # The 9B on the laptop right after 8 minutes of full load (Oct 2026): already hot,
    # flat at about 95% and 4.2-4.4 words/s for 15 minutes.
    clocks = [97.3, 95.7, 95.6, 94.7, 94.8, 94.2, 95.6, 95.1, 92.6, 95.1, 95.2, 95.1, 94.0, 96.6,
              93.8, 96.6, 96.4, 97.2, 95.4, 94.0, 94.3, 95.3, 94.3, 93.1, 93.0, 94.7, 95.5]
    writing = [3.747, 4.377, 4.194, 4.345, 4.326, 4.301, 4.12, 4.282, 4.268, 4.073, 4.173, 4.23,
               4.404, 4.049, 4.248, 4.161, 4.046, 4.237, 4.425, 4.315, 4.36, 4.332, 4.288, 4.122,
               4.368, 4.313, 4.198]
    check = analyse_heat(_measurement(clocks, writing), tolerance=8.0)
    assert not check.throttled and not check.processor_slowed


def test_a_dip_with_a_faster_end_is_not_called_a_slowdown() -> None:
    # Llama 3.2 3B on the laptop (job 101): 85% -> 104% with one low reading; writing flat.
    check = analyse_heat(
        _measurement([86, 84, 80, 112, 95, 100, 104, 104], [19.1, 19.1, 19.1, 19.2, 19.1, 19.1, 19.1, 19.1]),
        tolerance=8.0,
    )
    assert not check.throttled
    assert "slowed down" not in check.words and "probably from heat" not in check.words
