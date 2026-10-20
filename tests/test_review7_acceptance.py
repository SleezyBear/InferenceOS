"""Review 7 (6 Oct 2026): six acceptance defects found by the adversarial review.

1. A memory-safety correction must show fresh resource evidence (fits, every resource
   reading on every run, no severe paging), not just running and answering.
2. Merging changes must not lose the strictest answer requirement (a layer step merged
   with a cache-format change still needs the known-answer questions).
3. Known answers are judged from an explicit answer line, strictly ("1805" is not 805).
4. Unusable or unfinished known-answer measurements never pass.
5. Fresh trade-off and known-answer measurements must show the requested settings.
6. Buffer totals alone never prove a placement (covered in test_placement_and_cache).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.experiment import ExperimentMeasurement, RuntimeConfiguration
from inferenceos.contracts.observation import RunMetrics
from inferenceos.contracts.tune import ProposedChange
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.tune.acceptance_evidence import safety_resource_evidence
from inferenceos.tune.approval import ScriptedChangeApprover
from inferenceos.tune.layer_top import LoadAttempt
from inferenceos.tune.paths import KV_CACHE, PLACEMENT, Candidate, PathContext, placement_candidates
from inferenceos.tune.pipeline import Tuner, _candidate_requirement, merge_change
from inferenceos.tune.quality_risk import requirement_of_change, strictest
from tests.experiment_helpers import SampleResult
from tests.test_placement_and_cache import hybrid_facts
from tests.test_review6_acceptance import RIGHT as RIGHT_TEXT
from tests.test_review6_acceptance import _monotonic, _saved_high
from tests.test_tune import CONTEXT, _settings, _tuner
from tests.test_tune_new_paths import (
    FACTS,
    MIB,
    _FactsTuner,
    engine,
    limited_engine,
    _fits,
)


# --------------------------------------------------------------------------- #
# 1. Safety corrections need fresh resource evidence                          #
# --------------------------------------------------------------------------- #
def _monotonic_with(**sample: Any) -> Any:
    inner = _monotonic(RIGHT_TEXT)

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        result: SampleResult = inner(task, options, call)
        if (options.gpu_offload_layers or 0) <= 10:
            for name, value in sample.items():
                setattr(result, name, value)
        return result

    return responder


def test_a_safety_correction_without_paging_readings_is_not_saved(tmp_path: Path) -> None:
    """The reviewer's reproduction: saved 25-ish layers, load checks allow 10, paging
    readings missing. It used to save 10 layers with the safety check marked passed."""
    high = _saved_high(tmp_path)
    second, _, _, store = _tuner(tmp_path, _monotonic_with(hard_faults_per_sec=None), answers=[True] * 3)
    second.try_load = lambda layers: LoadAttempt(loaded=True, fits=layers <= 10)
    result = second.run()
    check: Any = next(item for item in result.rounds if item.get("round") == "memory-safety check")
    assert check["passed"] is False
    assert any("paging was read in 0 of" in str(reason) for reason in check["why"])
    assert result.outcome == "not saved: memory-safety correction unverified"
    assert result.saved_setup is None
    saved = store.find(second.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.gpu_offload_layers == high


def test_a_safety_correction_that_pages_severely_is_not_saved(tmp_path: Path) -> None:
    _saved_high(tmp_path)
    second, _, _, _ = _tuner(tmp_path, _monotonic_with(hard_faults_per_sec=5_000.0), answers=[True] * 3)
    second.try_load = lambda layers: LoadAttempt(loaded=True, fits=layers <= 10)
    result = second.run()
    check: Any = next(item for item in result.rounds if item.get("round") == "memory-safety check")
    assert check["passed"] is False
    assert any("severe paging" in str(reason) for reason in check["why"])
    assert result.saved_setup is None


def test_a_safety_correction_that_does_not_fit_on_a_fresh_load_is_not_saved(tmp_path: Path) -> None:
    _saved_high(tmp_path)
    second, _, _, _ = _tuner(tmp_path, _monotonic(RIGHT_TEXT), answers=[True] * 3)

    def try_load(layers: int) -> LoadAttempt:
        # The search sees 10 fit; by the fresh safety check memory has tightened.
        return LoadAttempt(loaded=True, fits=layers <= (10 if second._layer_safety_change is None else 9))

    second.try_load = try_load
    result = second.run()
    check: Any = next(item for item in result.rounds if item.get("round") == "memory-safety check")
    assert check["passed"] is False
    assert check["resources"]["load_check"]["fits"] is False
    assert result.saved_setup is None


def _measurement(paging: list[float | None], rss: int | None = 1) -> ExperimentMeasurement:
    return ExperimentMeasurement(
        measurement_id="m", phase="t", configuration=RuntimeConfiguration(context_length=2048),
        valid_runs=len(paging),
        run_metrics=[RunMetrics(hard_page_faults_per_sec=value, peak_rss_bytes=rss) for value in paging],
    )


def test_safety_resource_evidence_rules() -> None:
    fits = LoadAttempt(loaded=True, fits=True)
    ok = safety_resource_evidence(
        measurement=_measurement([1.0, 2.0]), reference=_measurement([0.0, 1.0]), guarded=[], load_check=fits
    )
    assert ok.passed
    assert not safety_resource_evidence(
        measurement=_measurement([1.0, 2.0]), reference=None, guarded=[], load_check=None
    ).passed  # no fresh load check
    assert not safety_resource_evidence(
        measurement=_measurement([1.0, None]), reference=None, guarded=[], load_check=fits
    ).passed  # read in only some runs
    assert not safety_resource_evidence(
        measurement=_measurement([1.0, 2.0], rss=None), reference=None, guarded=[], load_check=fits
    ).passed  # memory missing
    assert not safety_resource_evidence(
        measurement=_measurement([300.0, 300.0]), reference=_measurement([10.0, 10.0]), guarded=[], load_check=fits
    ).passed  # pages more than the start plus the tolerance
    assert not safety_resource_evidence(
        measurement=_measurement([1.0, 2.0]), reference=None, guarded=["total_latency_s"], load_check=fits
    ).passed  # a guarded measure the goal needs is missing


# --------------------------------------------------------------------------- #
# 2. Merging keeps the strictest requirement                                   #
# --------------------------------------------------------------------------- #
def _change(card: str, setting: str, value: object, **related: object) -> ProposedChange:
    return ProposedChange(
        change_id=f"{card}:{value}", card_id=card, setting=setting, description=setting,
        old_value=None, new_value=value, related=related,  # type: ignore[arg-type]
    )


def test_a_layer_step_merged_with_a_cache_change_still_needs_known_answers() -> None:
    start = RuntimeConfiguration(context_length=2048, gpu_offload_layers=8)
    cache = _change("runtime.kv_cache_format", "kv_cache_type_k", "q8_0", gpu_offload_layers=10)
    layers = _change("runtime.increase_gpu_offload_layers", "gpu_offload_layers", 14)
    merged = merge_change([cache], layers, {}, start)
    assert len(merged) == 1
    only = merged[0]
    assert only.setting == "gpu_offload_layers" and only.related.get("kv_cache_type_k") == "q8_0"
    card, requirement = strictest(merged)
    assert requirement.known_answers and card == "runtime.kv_cache_format"
    assert only.card_id == "runtime.kv_cache_format"
    # Even a change whose card says "layers" is classified by what it changes.
    disguised = _change("runtime.increase_gpu_offload_layers", "gpu_offload_layers", 14, kv_cache_type_v="q8_0")
    assert requirement_of_change(disguised)[1].known_answers
    # Going back to 16-bit is not a precision change; layer counts are scheduling only.
    back = _change("runtime.kv_cache_format", "kv_cache_type_k", "f16")
    assert not requirement_of_change(back)[1].known_answers
    assert strictest([layers])[1].name == "runtime"


def test_a_candidate_requirement_counts_every_setting_it_changes() -> None:
    plain = Candidate(value="^output=CPU", label="x", also=(("gpu_offload_layers", 12),))
    assert _candidate_requirement(PLACEMENT, plain)[1].name == "engine"
    with_cache = Candidate(value="^output=CPU", label="y", also=(("kv_cache_type_v", "q8_0"),))
    assert _candidate_requirement(PLACEMENT, with_cache)[1].known_answers
    assert _candidate_requirement(KV_CACHE, Candidate(value="q8_0", label="z"))[1].known_answers


class _UnderestimatingTuner(_FactsTuner):
    """Reads narrower cache rows than the model really has, so the cache path's joint
    "q8 plus more layers" estimate is too small and the layer revisit finds more: the
    revisit's layer change then merges with the cache change."""

    def _model_facts(self) -> Any:
        import dataclasses

        return dataclasses.replace(
            FACTS,
            key_widths=tuple(768 for _ in FACTS.key_widths or ()),
            value_widths=tuple(768 for _ in FACTS.value_widths or ()),
        )


def test_the_merged_cache_and_layer_change_gets_known_answers_in_the_final_check(tmp_path: Path) -> None:
    """The reviewer's reproduction: a q8 cache lets more layers fit, the layer revisit
    finds more, and the merged proposal (a layer change carrying q8) used to be saved
    with zero known-answer calls in the final confirmation."""
    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_layer_path=True,
        tune_diagnosis=False, tune_heat_check_runs=0, tune_paths="kv_cache",
    )
    _FactsTuner.budget = 420 * MIB
    tuner, _, _, store = _tuner(
        tmp_path, limited_engine(), answers=[True] * 6, settings=settings, tuner_cls=_UnderestimatingTuner
    )

    def try_load(layers: int, base: RuntimeConfiguration | None = None) -> LoadAttempt:
        config = base or RuntimeConfiguration()
        keys, values = config.effective_kv_types()
        return LoadAttempt(loaded=True, fits=_fits(layers, config.placement_overrides, keys, values))

    tuner.try_load = try_load
    result = tuner.run()
    assert len(result.proposed_changes) == 1
    merged = result.proposed_changes[0]
    # The revisit's layer step came last, so the merged change is a layer change...
    assert merged.setting == "gpu_offload_layers" and merged.related.get("kv_cache_type_k") == "q8_0"
    # ...that keeps the cache format's requirement.
    assert merged.card_id == "runtime.kv_cache_format"
    final: Any = next(item for item in result.rounds if item.get("round") == "final check")
    assert final["checks"][0]["quality_requirement"]["level"] == "precision"
    known = [check for check in final["checks"] if check.get("workload") == "known answers"]
    assert known and known[0]["passed"]
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.kv_cache_type_k == "q8_0"


# --------------------------------------------------------------------------- #
# 3. and 5. Known answers inside a whole tune                                  #
# --------------------------------------------------------------------------- #
def _run_cache(tmp_path: Path, responder: Any, *, tuner_cls: type[Tuner] = _FactsTuner, approver: Any = None) -> Any:
    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_layer_path=False,
        tune_diagnosis=False, tune_heat_check_runs=0, tune_paths="kv_cache",
    )
    _FactsTuner.budget = 420 * MIB
    tuner, _, _, store = _tuner(tmp_path, responder, answers=[True] * 6, settings=settings, tuner_cls=tuner_cls)
    if approver is not None:
        tuner.approver = approver
    return tuner, tuner.run(), store


def _known_texts_with_q8(texts: dict[str, str]) -> Any:
    inner = engine()

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        sample: SampleResult = inner(task, options, call)
        keys = options.kv_cache_type_k or options.kv_cache_type
        if keys == "q8_0" and task.workload.workload_id in texts:
            sample.text = texts[task.workload.workload_id]
        return sample

    return responder


def test_wrong_known_answers_that_mention_the_right_value_are_not_kept(tmp_path: Path) -> None:
    """The reviewer's reproduction: a whole tune saved q8 when its known answers were
    1805, "1209 crates to the southwest depot" and XF9595Y."""
    wrong = {
        "known-arithmetic": "ANSWER: 1805",
        "known-fact-far-back": "ANSWER: 1209 crates to the southwest depot",
        "known-code-early-in-a-list": "ANSWER: XF9595Y",
    }
    tuner, result, store = _run_cache(tmp_path, _known_texts_with_q8(wrong))
    record: Any = next(item for item in result.rounds if item.get("path") == "kv_cache")
    assert record["known_answers"]["status"] == "fail"
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.kv_cache_type_k is None


def _known_answers_measured_as_f16() -> Any:
    inner = engine()

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        sample: SampleResult = inner(task, options, call)
        if task.workload.workload_id.startswith("known-") and sample.observed:
            sample.observed = {**sample.observed, "cache_type_k": "f16", "cache_type_v": "f16"}
        return sample

    return responder


def test_known_answers_from_a_load_that_did_not_use_the_format_are_not_evidence(tmp_path: Path) -> None:
    """The reviewer's reproduction: speed measurements showed q8, the known-answer
    measurements showed f16, and the tune still saved q8."""
    tuner, result, store = _run_cache(tmp_path, _known_answers_measured_as_f16())
    record: Any = next(item for item in result.rounds if item.get("path") == "kv_cache")
    assert record["known_answers"]["status"] == "incomplete"
    assert "not shown" in record["known_answers"]["reason"]
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.kv_cache_type_k is None


# --------------------------------------------------------------------------- #
# 5. Trade-offs must show the requested settings on their fresh loads          #
# --------------------------------------------------------------------------- #
class _TradeoffTuner(_FactsTuner):
    """Records when the trade-off's fresh measurements begin."""

    measuring_tradeoff = False

    def _offer_tradeoff(self, result: Any) -> Any:
        type(self).measuring_tradeoff = True
        try:
            return super()._offer_tradeoff(result)
        finally:
            type(self).measuring_tradeoff = False


def _cache_tradeoff_world(*, hide_effect_in_tradeoff: bool) -> Any:
    """q8 is clearly faster on the long question and clearly slower on the short one."""
    inner = engine()

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        sample: SampleResult = inner(task, options, call)
        keys = options.kv_cache_type_k or options.kv_cache_type
        values = options.kv_cache_type_v or options.kv_cache_type
        quantized = "q8_0" in (keys, values)
        workload_id = task.workload.workload_id
        if quantized and not workload_id.startswith("known-"):
            factor = 1.25 if workload_id.startswith("long-") else 0.85
            sample.decode_tps *= factor
            sample.total_latency_s /= factor
        if hide_effect_in_tradeoff and _TradeoffTuner.measuring_tradeoff and sample.observed:
            sample.observed = {
                key: value for key, value in sample.observed.items() if not key.startswith("cache_type")
            }
        return sample

    return responder


def test_a_cache_tradeoff_is_offered_when_its_fresh_loads_show_the_format(tmp_path: Path) -> None:
    approver = ScriptedChangeApprover([True] * 6, tradeoff_answers=[True])
    tuner, result, store = _run_cache(
        tmp_path, _cache_tradeoff_world(hide_effect_in_tradeoff=False), tuner_cls=_TradeoffTuner, approver=approver
    )
    offer: Any = next(item for item in result.rounds if item.get("round") == "trade-off")
    assert offer["quality_requirement"]["level"] == "precision"
    assert offer.get("known_answers", {}).get("status") == "pass"
    assert approver.tradeoffs_asked, offer.get("reason")
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and "q8_0" in (saved.configuration.kv_cache_type_k, saved.configuration.kv_cache_type_v)


def test_a_cache_tradeoff_whose_fresh_loads_show_nothing_is_never_offered(tmp_path: Path) -> None:
    """The reviewer's reproduction: screening showed q8, the fresh trade-off loads gave no
    effect evidence, and an explicit yes still saved q8."""
    approver = ScriptedChangeApprover([True] * 6, tradeoff_answers=[True])
    tuner, result, store = _run_cache(
        tmp_path, _cache_tradeoff_world(hide_effect_in_tradeoff=True), tuner_cls=_TradeoffTuner, approver=approver
    )
    offer: Any = next(item for item in result.rounds if item.get("round") == "trade-off")
    assert "not shown by the engine" in str(offer.get("reason"))
    assert approver.tradeoffs_asked == []
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None
    assert saved.configuration.kv_cache_type_k is None and saved.configuration.kv_cache_type_v is None


# --------------------------------------------------------------------------- #
# 6. Placement inside a whole tune; grouped placements wait for a probe        #
# --------------------------------------------------------------------------- #
def _totals_without_piece_lines() -> Any:
    inner = engine()

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        sample: SampleResult = inner(task, options, call)
        if sample.observed:
            sample.observed = {key: value for key, value in sample.observed.items() if key != "overridden_tensors"}
        return sample

    return responder


def test_a_placement_shown_only_by_buffer_totals_is_never_kept(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_layer_path=False,
        tune_diagnosis=False, tune_heat_check_runs=0, tune_paths="placement",
    )
    _FactsTuner.budget = 420 * MIB
    tuner, _, _, store = _tuner(tmp_path, _totals_without_piece_lines(), answers=[True] * 6, settings=settings, tuner_cls=_FactsTuner)
    result = tuner.run()
    record: Any = next(item for item in result.rounds if item.get("path") == "placement")
    assert record["screened"] and all(entry.get("result") == "effect unverified" for entry in record["screened"])
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.placement_overrides is None


def test_grouped_placements_are_screened_only_until_a_hardware_probe() -> None:
    from inferenceos.selftest import sample_hardware, sample_model

    context = PathContext(
        hardware=sample_hardware(), model=sample_model(), total_layers=5,
        current=RuntimeConfiguration(context_length=2048, gpu_offload_layers=5),
        facts=hybrid_facts(),
    )
    grouped = [item for item in placement_candidates(context) if "blocks on the card, the other" in item.label]
    assert grouped and all(item.screen_only for item in grouped)
    probed = PathContext(**{**context.__dict__, "grouped_placement_probed": True})
    again = [item for item in placement_candidates(probed) if "blocks on the card, the other" in item.label]
    assert again and not any(item.screen_only for item in again)





def test_inherited_engine_arguments_are_not_passed_on() -> None:
    from inferenceos.backends.engine_env import overridden_by_setups, process_environment

    parent = {"PATH": "x", "LLAMA_ARG_CACHE_TYPE_K": "q4_0", "llama_arg_fit": "off", "GGML_VK_DISABLE_F16": "1"}
    env = process_environment(parent, {"GGML_VK_DISABLE_FUSION": "1"})
    assert env == {"PATH": "x", "GGML_VK_DISABLE_FUSION": "1"}
    assert set(overridden_by_setups(parent)) == {"LLAMA_ARG_CACHE_TYPE_K", "llama_arg_fit", "GGML_VK_DISABLE_F16"}


# --------------------------------------------------------------------------- #
# Laptop job 110/112: per-piece lines only at the engine's debug level          #
# --------------------------------------------------------------------------- #
def _placement_run(tmp_path: Path, reader: Any) -> Any:
    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_layer_path=False,
        tune_diagnosis=False, tune_heat_check_runs=0, tune_paths="placement",
    )
    _FactsTuner.budget = 420 * MIB
    tuner, _, _, store = _tuner(
        tmp_path, _totals_without_piece_lines(), answers=[True] * 6, settings=settings, tuner_cls=_FactsTuner
    )
    tuner.piece_lines_reader = reader
    return tuner, tuner.run(), store


def test_a_placement_is_shown_by_a_separate_debug_level_load_plus_its_own_totals(tmp_path: Path) -> None:
    from tests.test_tune_new_paths import moved_pieces

    reads: list[tuple[int | None, str | None]] = []

    def reader(config: RuntimeConfiguration) -> dict[str, object]:
        reads.append((config.gpu_offload_layers, config.placement_overrides))
        return {"overridden_tensors": moved_pieces(config.placement_overrides)}

    tuner, result, store = _placement_run(tmp_path, reader)
    assert result.outcome == "saved", result.outcome
    assert reads and len(reads) == len(set(reads))  # one read per exact setup
    record: Any = next(item for item in result.rounds if item.get("path") == "placement")
    kept = [entry for entry in record["screened"] if entry.get("took_effect", {}).get("status") == "shown"]
    assert kept and "separate debug-level load" in kept[0]["took_effect"]["why"]
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.placement_overrides == "^output=CPU"


def test_a_debug_level_load_that_lists_nothing_leaves_the_placement_unverified(tmp_path: Path) -> None:
    tuner, result, store = _placement_run(tmp_path, lambda config: {"model_buffers": {}})
    record: Any = next(item for item in result.rounds if item.get("path") == "placement")
    assert all(entry.get("result") == "effect unverified" for entry in record["screened"])
    saved = store.find(tuner.model_path, context_length=CONTEXT, power=PowerState.AC)
    assert saved is not None and saved.configuration.placement_overrides is None


@pytest.mark.parametrize("failed_stop", [False, True])
def test_the_debug_level_applies_to_one_start_only(tmp_path: Path, failed_stop: bool) -> None:
    from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
    from inferenceos.contracts.lifecycle import ResidentLoadMeasurement, ResidentUnloadMeasurement
    from inferenceos.contracts.model import ModelProfile
    from inferenceos.interfaces.backend_adapter import RuntimeParams
    from inferenceos.selftest import sample_model
    from tests.resident_helpers import FakeResidentBackend

    started: list[int | None] = []
    failure = RuntimeError("stop failed before another start was attempted")

    class _Backend(FakeResidentBackend):
        stops = 0

        def start(self, model: ModelProfile, params: RuntimeParams) -> ResidentLoadMeasurement:
            started.append(params.log_verbosity)
            return super().start(model, params)

        def stop(self) -> ResidentUnloadMeasurement:
            self.stops += 1
            if failed_stop and self.stops == 1:
                raise failure
            return super().stop()

    path = tmp_path / "model.gguf"
    path.write_bytes(b"\0")
    model = sample_model().model_copy(update={"path": str(path)})
    adapter = LlamaCppResidentAdapter(server=Path("llama-server.exe"), backend=_Backend())
    adapter.load(model, RuntimeParams(gpu_offload_layers=4, log_verbosity=4))
    adapter.use_log_verbosity_once(5)
    if failed_stop:
        with pytest.raises(RuntimeError) as caught:
            adapter.restart()
        assert caught.value is failure and adapter.is_loaded
    adapter.restart()
    adapter.restart()
    assert started == [4, 5, 4]
    assert adapter.params.log_verbosity == 4
