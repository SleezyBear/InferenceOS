"""The placement and conversation-memory paths inside a whole (scripted) tune.

A scripted engine places pieces exactly as asked and reports it in its "log" (model
buffer sizes, cache formats), unless a test says otherwise. Speeds follow how much of
the work stays on the processor, so moving work to the card is faster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferenceos.backends.llama_server_toolchain import build_server_command
from inferenceos.backends.tensor_overrides import check_overrides
from inferenceos.backends.resident_server import _options_from
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.experiment.restoration import runtime_params_from
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.tune.model_facts import ModelFacts
from inferenceos.tune.pipeline import Tuner
from inferenceos.tune.start import check_started
from inferenceos.tune.tensor_placement import piece_placement
from tests.experiment_helpers import SampleResult
from tests.test_placement_and_cache import dense_facts
from tests.test_tune import CONTEXT, _settings, _tuner

MIB = 1024 * 1024
FACTS = dense_facts(blocks=24, block_mib=30, output_mib=150)
RIGHT = {
    "known-arithmetic": "391 + 414 = 805.\nANSWER: 805",
    "known-fact-far-back": "ANSWER: 209, south",
    "known-code-early-in-a-list": "ANSWER: F9595",
}


def moved_pieces(overrides: str | None) -> dict[str, str]:
    """What the engine's per-piece lines list for an override text (first match wins)."""
    if not overrides:
        return {}
    return dict(check_overrides(overrides, list(FACTS.tensor_bytes or {})).assigned)


class _FactsTuner(Tuner):
    budget: int | None = 420 * MIB

    def _model_facts(self) -> ModelFacts:
        return FACTS

    def _card_budget(self, config: RuntimeConfiguration) -> int | None:
        return type(self).budget


def engine(
    *, report: bool = True, wrong_known_with_q8: bool = False
) -> Any:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        assert task.context_length == CONTEXT
        where = piece_placement(FACTS, options.gpu_offload_layers, options.placement_overrides)
        assert where is not None
        share = where.processor_share or 0.0
        speed = 10.0 + 20.0 * (1.0 - share)
        keys = options.kv_cache_type_k or options.kv_cache_type or "f16"
        values = options.kv_cache_type_v or options.kv_cache_type or "f16"
        observed: dict[str, object] = {}
        if report:
            observed = {
                "model_buffers": {"Vulkan0": where.card_bytes, "CPU_Mapped": where.processor_bytes},
                "cache_type_k": keys,
                "cache_type_v": values,
            }
            if options.placement_overrides:
                observed["overridden_tensors"] = moved_pieces(options.placement_overrides)
        text = "A key value cache stores attention state."
        workload_id = task.workload.workload_id
        if workload_id in RIGHT:
            text = RIGHT[workload_id]
            if wrong_known_with_q8 and keys == "q8_0":
                text = "I do not know."
        return SampleResult(decode_tps=speed, total_latency_s=40.0 / speed, text=text, observed=observed)

    return responder


def _run(tmp_path: Path, paths: str, responder: Any, *, budget: int | None = 420 * MIB) -> tuple[Any, Any, Any]:
    settings = _settings(
        tmp_path,
        optimization_mode="performance_first",
        tune_layer_path=False,
        tune_diagnosis=False,
        tune_heat_check_runs=0,
        tune_paths=paths,
    )
    _FactsTuner.budget = budget
    tuner, _, approver, store = _tuner(
        tmp_path, responder, answers=[True] * 6, settings=settings, tuner_cls=_FactsTuner
    )
    return tuner, approver, store


def test_the_output_step_moves_to_the_processor_when_that_lets_more_blocks_on(tmp_path: Path) -> None:
    tuner, _, store = _run(tmp_path, "placement", engine())
    result = tuner.run()
    assert result.outcome == "saved", result.outcome
    change = next(item for item in result.proposed_changes if item.setting == "placement_overrides")
    assert change.new_value == "^output=CPU"
    assert int(str(change.related["gpu_offload_layers"])) > 8
    assert change.observed["placement_overrides"].startswith("shown")
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None
    assert saved.configuration.placement_overrides == "^output=CPU"
    assert saved.configuration.gpu_offload_layers == change.related["gpu_offload_layers"]
    final = next(item for item in result.rounds if item.get("round") == "final check")
    first = final["checks"][0]
    assert first["observed"] and first["observed"][0]["status"] == "shown"
    assert "writing_speed" in first["measures"] and "first_word" in first["measures"]
    assert first["baseline"].startswith("8 graphics card layers")
    command = " ".join(
        build_server_command(
            Path("llama-server.exe"),
            _options_from(Path("m.gguf"), runtime_params_from(saved.configuration)),
            host="127.0.0.1", port=8080, api_key_file=Path("key"),
        )
    )
    assert "-ot ^output=CPU" in command
    where = piece_placement(FACTS, saved.configuration.gpu_offload_layers, saved.configuration.placement_overrides)
    assert where is not None
    shown = check_started(
        saved.configuration,
        {
            "model_buffers": {"Vulkan0": where.card_bytes},
            "overridden_tensors": moved_pieces(saved.configuration.placement_overrides),
        },
        FACTS,
    )
    assert shown == [("piece placement", "shown", shown[0][2])]
    totals_only = check_started(saved.configuration, {"model_buffers": {"Vulkan0": where.card_bytes}}, FACTS)
    assert totals_only[0][1] == "unverified"
    missing = check_started(saved.configuration, {}, FACTS)
    assert missing[0][1] == "unverified"


def test_a_placement_the_engine_does_not_show_is_never_kept(tmp_path: Path) -> None:
    tuner, _, store = _run(tmp_path, "placement", engine(report=False))
    result = tuner.run()
    assert result.outcome == "no change"
    record = next(item for item in result.rounds if item.get("path") == "placement")
    assert all(entry.get("result") == "effect unverified" for entry in record["screened"])
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.placement_overrides is None


def test_a_smaller_cache_that_lets_more_layers_fit_is_kept_after_known_answers(tmp_path: Path) -> None:
    tuner, _, store = _run(tmp_path, "kv_cache", engine())
    result = tuner.run()
    assert result.outcome == "saved", result.outcome
    record = next(item for item in result.rounds if item.get("path") == "kv_cache")
    assert record["known_answers"]["status"] == "pass"
    assert record["quality_requirement"]["level"] == "precision"
    final = next(item for item in result.rounds if item.get("round") == "final check")
    assert final["checks"][0]["quality_card"] == "runtime.kv_cache_format"
    assert any(check.get("workload") == "known answers" and check["passed"] for check in final["checks"])
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None
    config = saved.configuration
    assert config.kv_cache_type_k == "q8_0"
    assert config.context_length == CONTEXT  # never a shorter conversation window
    assert int(config.gpu_offload_layers or 0) > 8
    if config.kv_cache_type_v == "q8_0":
        assert config.flash_attention is True  # a quantized value cache needs it


def test_a_cache_format_that_gets_known_answers_wrong_is_not_kept(tmp_path: Path) -> None:
    tuner, _, store = _run(tmp_path, "kv_cache", engine(wrong_known_with_q8=True))
    result = tuner.run()
    record = next(item for item in result.rounds if item.get("path") == "kv_cache")
    assert record["known_answers"]["status"] == "fail"
    assert "known answers wrong" in str(record["reason"])
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.kv_cache_type_k is None


def test_an_unreported_cache_format_is_unverified_not_kept(tmp_path: Path) -> None:
    tuner, _, store = _run(tmp_path, "kv_cache", engine(report=False))
    result = tuner.run()
    assert result.outcome == "no change"
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.kv_cache_type_k is None


LIMIT = 620 * MIB


def _fits(layers: int | None, overrides: str | None, keys: str | None, values: str | None) -> bool:
    from inferenceos.selftest import sample_model
    from inferenceos.tune.kv_cache import cache_bytes

    where = piece_placement(FACTS, layers, overrides)
    cache = cache_bytes(FACTS, sample_model(), CONTEXT, keys, values).bytes
    assert where is not None and cache is not None
    return where.card_bytes + cache <= LIMIT


def limited_engine() -> Any:
    inner = engine()

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        keys = options.kv_cache_type_k or options.kv_cache_type
        values = options.kv_cache_type_v or options.kv_cache_type
        if not _fits(options.gpu_offload_layers, options.placement_overrides, keys, values):
            return SampleResult(fail=True)
        return inner(task, options, call)  # type: ignore[no-any-return]

    return responder


def test_a_memory_change_brings_the_layer_search_back(tmp_path: Path) -> None:
    from inferenceos.tune.layer_top import LoadAttempt

    settings = _settings(
        tmp_path,
        optimization_mode="performance_first",
        tune_layer_path=True,
        tune_diagnosis=False,
        tune_heat_check_runs=0,
        tune_paths="kv_cache",
    )
    _FactsTuner.budget = 420 * MIB
    tuner, _, _, store = _tuner(
        tmp_path, limited_engine(), answers=[True] * 6, settings=settings, tuner_cls=_FactsTuner
    )

    def try_load(layers: int, base: RuntimeConfiguration | None = None) -> LoadAttempt:
        config = base or RuntimeConfiguration()
        keys, values = config.effective_kv_types()
        return LoadAttempt(loaded=True, fits=_fits(layers, config.placement_overrides, keys, values))

    tuner.try_load = try_load
    result = tuner.run()
    first = result.rounds[0]
    assert first["round"] == "graphics card layers" and first["top"] == 10
    kv_round = next(item for item in result.rounds if item.get("path") == "kv_cache")
    assert kv_round.get("selected") == "q8_0", kv_round.get("reason")
    again = [item for item in result.rounds if item.get("round") == "graphics card layers" and item.get("again")]
    assert again and again[0]["revisit"] is True and again[0]["top"] == 13
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None
    assert saved.configuration.kv_cache_type_k == "q8_0"
    assert saved.configuration.gpu_offload_layers == 13
    # One change per setting group: the layer steps are merged into one change.
    layer_changes = [
        item for item in result.proposed_changes
        if item.setting == "gpu_offload_layers" or "gpu_offload_layers" in item.related
    ]
    assert len(layer_changes) == 1
