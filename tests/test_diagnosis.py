"""Diagnosis first (4 Oct 2026)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.tune.diagnose import HIGH, NORMAL, PROBE, diagnose, processor_bytes_per_word
from inferenceos.tune.engine_probe import Graph, ProbeResult
from inferenceos.tune.memory_speed import MemorySpeed, cached_memory_speed
from inferenceos.tune.model_facts import ModelFacts
from tests.experiment_helpers import SampleResult
from tests.test_tune import _settings, _tuner


def _facts() -> ModelFacts:
    return ModelFacts(
        total_layers=4, from_file=True, attention_layers=4,
        layer_bytes=(100, 100, 100, 100), output_bytes=50, other_bytes=50,
    )


def test_processor_bytes_follow_the_engine_order() -> None:
    facts = _facts()
    assert processor_bytes_per_word(facts, 0, card=True) == 450  # all blocks + output
    assert processor_bytes_per_word(facts, 1, card=True) == 400  # output on the card
    assert processor_bytes_per_word(facts, 3, card=True) == 200  # output + 2 blocks
    assert processor_bytes_per_word(facts, 5, card=True) == 0  # everything
    assert processor_bytes_per_word(facts, 3, card=False) == 450


def test_a_processor_at_its_memory_limit_gets_only_a_quick_thread_probe() -> None:
    # 40 ms per word, card 4 ms; 1.6 GB on the processor -> 44 GB/s... at a 40 GB/s
    # ceiling that is at the limit: threads are probed, never skipped.
    result = diagnose(
        word_s=0.040, first_word_s=0.20, reading_s=0.05, card_word_s=0.004,
        processor_bytes=1_600_000_000, memory_ceiling=40e9, card=True,
        processor_share_of_work=0.9,
    )
    assert result.priority("threads") == PROBE
    assert "layers" not in result.priorities  # the layer round runs before the diagnosis
    assert "threads" in result.priorities and result.memory_use is not None
    assert result.memory_use > 0.85
    names = ["threads", "reading_threads", "flash_attention", "prompt_batch", "engine_switches"]
    order = result.order(names)
    assert order[-1] in ("threads", "engine_switches") and set(order) == set(names)


def test_room_to_move_memory_makes_threads_a_high_priority() -> None:
    result = diagnose(
        word_s=0.040, first_word_s=0.20, reading_s=0.05, card_word_s=0.004,
        processor_bytes=400_000_000, memory_ceiling=40e9, card=True,
        processor_share_of_work=0.9,
    )
    assert result.priority("threads") == HIGH
    assert result.priority("engine_switches") == PROBE  # the card does little


def test_a_card_heavy_model_puts_engine_switches_first() -> None:
    result = diagnose(
        word_s=0.020, first_word_s=0.10, reading_s=0.02, card_word_s=0.016,
        processor_bytes=10_000_000, memory_ceiling=40e9, card=True,
        processor_share_of_work=0.05,
    )
    assert result.priority("engine_switches") == HIGH
    assert "layers" not in result.priorities


def test_the_memory_speed_is_kept_per_machine(tmp_path: Path) -> None:
    path = tmp_path / "memory_speed.json"
    first = cached_memory_speed(path, "cpu|16GB", max_threads=1)
    again = cached_memory_speed(path, "cpu|16GB", max_threads=1)
    assert first.best_bytes_per_s == again.best_bytes_per_s
    assert path.exists()


def test_tune_diagnoses_first_and_again_after_a_kept_change(tmp_path: Path) -> None:
    from tests.test_tune_paths import stacking_responder

    probes: list[RuntimeConfiguration] = []

    def probe_runner(config: RuntimeConfiguration) -> ProbeResult:
        probes.append(config)
        result = ProbeResult(config=config, ready=True)
        result.graphs = [Graph("writing", 4000.0, ())] * 3
        result.writing_ms_per_word = [40.0]  # the timed run: card 4 ms of 40 ms = 10%
        return result

    settings = _settings(tmp_path, optimization_mode="performance_first")
    lines: list[str] = []
    tuner, _, _, _ = _tuner(
        tmp_path, stacking_responder, answers=[True, True, True], settings=settings, lines=lines
    )
    tuner.probe_runner = probe_runner
    tuner.memory_speed_reader = lambda: MemorySpeed(40e9, 8, {8: 40e9})
    result = tuner.run()
    diagnoses = [r for r in result.rounds if str(r.get("round", "")).startswith("diagnosis")]
    assert diagnoses and diagnoses[0]["round"] == "diagnosis (start)"
    assert len(diagnoses) >= 2, "the diagnosis runs again after a kept change"
    assert any("Diagnosis (start)" in line for line in lines)
    assert probes, "the timed run was used"
    first: Any = diagnoses[0]
    assert first["card_share"] is not None and abs(first["card_share"] - 0.10) < 1e-6
    assert first["memory_ceiling_gb_s"] == 40.0


def test_a_quick_probe_tries_two_values_and_widens_only_when_one_looks_better(
    tmp_path: Path,
) -> None:
    from inferenceos.tune.paths import THREADS

    def flat(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)

    settings = _settings(tmp_path, tune_paths="threads", tune_layer_path=False)
    tuner, _, _, _ = _tuner(tmp_path, flat, answers=[], settings=settings)
    from inferenceos.tune.measure import DisturbanceWatch

    watch = DisturbanceWatch(logical_cores=12, limit_percent=1000.0, max_repeats=0)
    start = RuntimeConfiguration(context_length=2048, gpu_offload_layers=0, threads=4)
    record, change, _ = tuner._path_round(THREADS, start, watch, None, depth=PROBE)
    assert record["depth"] == PROBE and change is None
    candidates: Any = record["candidates"]
    assert len(candidates) == 2 and record.get("probe_left_out")
    assert not record.get("probe_became_full")
