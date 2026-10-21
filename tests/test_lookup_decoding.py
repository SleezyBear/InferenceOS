"""Lookup work, quality, ownership and ordering adversaries; no real engine."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from inferenceos.interfaces.backend_adapter import GenerationRequest, LookupDecodingParams, RuntimeParams
from inferenceos.backends.llama_server_toolchain import ServerOptions, build_server_command
from inferenceos.tune.lookup_decoding import (
    ALPHA, ARMS, BLOCKS, CONTROL, LOOKUP, TOKENS, build_cases, evaluate, plan,
    reading_problems, summarize,
)
from inferenceos.tune.lookup_sim import LookupSimulator
from tests.test_prompt_reuse import _backend_with, _quiet_backend, _client


def run(engine: LookupSimulator | None = None) -> dict[str, Any]:
    return evaluate(engine or LookupSimulator(), config_sha="config", power=lambda: "simulated",
                    checkpoint=lambda r: None, simulated=True)


def reading(arm: str = "lookup") -> dict[str, Any]:
    engine = LookupSimulator()
    engine.start(CONTROL if arm == "control" else LOOKUP)
    return engine.request(build_cases()[0])


def test_new_capability_does_not_add_a_tune_or_saved_setup_setting() -> None:
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.experiment.restoration import runtime_params_from
    from inferenceos.tune.declarations import DECLARATIONS

    assert "lookup_decoding" not in RuntimeConfiguration.model_fields
    assert len(DECLARATIONS) == 8
    assert runtime_params_from(RuntimeConfiguration(context_length=4096)).lookup_decoding is None
    assert RuntimeParams.model_validate({"threads": 3}).lookup_decoding is None
    assert not GenerationRequest.model_validate({"prompt": "p"}).return_tokens


@pytest.mark.parametrize("algorithm", ["draft-simple", "ngram-cache", "ngram-map-k", "a; command", "", None])
def test_only_explicit_none_and_model_free_simple_are_exposed(algorithm: Any) -> None:
    with pytest.raises(ValidationError):
        LookupDecodingParams(algorithm=algorithm)


@pytest.mark.parametrize("field", ["lookup_length", "draft_length"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 65, 1.0, "12", None])
def test_size_bounds_are_strict(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        LookupDecodingParams.model_validate({"algorithm": "ngram-simple", field: value})


def test_ordinary_commands_are_unchanged_and_lookup_uses_algorithm_specific_sizes(tmp_path: Path) -> None:
    def command(group: LookupDecodingParams | None) -> list[str]:
        return build_server_command(Path("server"), ServerOptions(tmp_path / "m", lookup_decoding=group),
                                    host="127.0.0.1", port=1, api_key_file=tmp_path / "key")
    assert "--spec-type" not in command(None)
    assert command(LookupDecodingParams(algorithm="none"))[-2:] == ["--spec-type", "none"]
    active = command(LookupDecodingParams(algorithm="ngram-simple"))
    assert active[-6:] == ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "12", "--spec-ngram-simple-size-m", "16"]
    assert "--spec-draft-n-max" not in active and "--draft" not in active and "-md" not in active
    bad = LookupDecodingParams(algorithm="ngram-simple").model_copy(update={"draft_length": 100000})
    with pytest.raises(ValidationError):
        command(bad)


def test_return_tokens_is_sent_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    client, sent = _client(monkeypatch, [b'data: {"stop":true}\n'])
    kwargs = dict(prompt="p", n_predict=1, seed=42, temperature=0.0, stop=[], json_schema=None)
    list(client.stream_completion(**kwargs))
    list(client.stream_completion(**kwargs, return_tokens=True))
    assert "return_tokens" not in sent[0]["body"] and sent[1]["body"]["return_tokens"] is True


def native(chunks: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> Any:
    from inferenceos.backends import resident_server as rs
    from inferenceos.contracts.lifecycle import ResourceSnapshot
    _quiet_backend(monkeypatch, rs, ResourceSnapshot)
    return _backend_with(chunks)


def terminal() -> dict[str, Any]:
    return {"content": "", "tokens": [], "stop": True, "tokens_evaluated": 9, "tokens_predicted": 4,
            "timings": {"prompt_n": 9, "cache_n": 0, "predicted_n": 4, "draft_n": 6, "draft_n_accepted": 2},
            "generation_settings": {"speculative.types": "ngram-simple"}, "truncated": False}


def test_ids_are_incremental_and_the_empty_final_chunk_does_not_erase_them(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = native([{"content": "a", "tokens": [11, 12]}, {"content": "b", "tokens": [13]},
                      {"content": "c", "tokens": [14]}, terminal()], monkeypatch)
    result = backend.generate(GenerationRequest(prompt="p", n_predict=4, ignore_eos=True, return_tokens=True))
    assert result.text == "abc"
    raw = backend.last_response_counters
    assert raw["generated_token_ids"] == [11, 12, 13, 14]
    assert raw["draft_n"] == 6 and raw["draft_n_accepted"] == 2
    assert raw["generation_settings"]["speculative.types"] == "ngram-simple"


@pytest.mark.parametrize("ids", [[1, True, 3, 4], [1, -1, 3, 4], [1, 2.0, 3, 4], [1, 2, 3], None, "1 2 3 4"])
def test_bad_or_incomplete_native_ids_refuse_the_opt_in_reading(ids: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.backends.resident_server import ResidentGenerationProtocolError
    backend = native([{"content": "a", "tokens": ids}, terminal()], monkeypatch)
    with pytest.raises(ResidentGenerationProtocolError):
        backend.generate(GenerationRequest(prompt="p", n_predict=4, ignore_eos=True, return_tokens=True))
    assert "token IDs" in backend.last_response_counters["protocol_problem"]


def test_non_opt_in_native_evidence_has_no_new_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = native([{"content": "a"}, terminal()], monkeypatch)
    backend.generate(GenerationRequest(prompt="p", n_predict=4, ignore_eos=True))
    assert "generated_token_ids" not in backend.last_response_counters
    assert "draft_n" not in backend.last_response_counters


def test_clean_simulation_retains_exact_plan_and_does_not_claim_hardware() -> None:
    engine = LookupSimulator()
    result = run(engine)
    assert result["status"] == "done"
    assert engine.starts == engine.requests == engine.stops == 40
    assert not engine.active
    assert [(r["case_id"], r["block"], r["arm"]) for r in result["trials"]] == plan()
    assert result["summary"]["simulated"] is True
    for name, case in result["summary"]["cases"].items():
        assert case["complete"] and case["reference_repeatable"]
        assert case["latency_bounds"]["alpha"] == ALPHA
        assert "no accepted draft" in case["outcome"] if name == "compose-prose" else "lower request latency" in case["outcome"]
    assert [a for _, _, a in plan()[:10]] == ["control", "lookup", "lookup", "control", "control", "lookup", "lookup", "control", "control", "lookup"]


@pytest.mark.parametrize("field,value", [
    ("tokens_evaluated", 2), ("prompt_n", 2), ("cache_n", 1), ("cache_n", None),
    ("tokens_predicted", 127), ("predicted_n", 127), ("stop_seen", False), ("stop_type", "word"),
    ("truncated", True), ("truncated", None), ("id_slot", True), ("id_slot", -1),
    ("draft_n", True), ("draft_n", -1), ("draft_n_accepted", 999), ("draft_n_accepted", None),
    ("protocol_problem", "bad stream"),
])
def test_shorter_unverified_or_contradictory_work_is_invalid(field: str, value: Any) -> None:
    r = reading()
    r["counters"][field] = value
    assert reading_problems(r, "lookup")


@pytest.mark.parametrize("field,value", [("total_s", 0), ("total_s", float("nan")), ("total_s", float("inf")),
                                        ("ttft_s", True), ("ttft_s", 20), ("text", ""),
                                        ("token_ids", None), ("token_ids", [1]), ("prompt_tokens", True),
                                        ("peak_rss_bytes", -1)])
def test_bad_wall_time_or_identity_is_never_a_gain(field: str, value: Any) -> None:
    r = reading()
    r[field] = value
    assert reading_problems(r, "lookup")


@pytest.mark.parametrize("field,value", [("speculative.types", "none"), ("seed", True), ("seed", -1),
                                        ("n_predict", 12), ("temperature", 1), ("ignore_eos", True)])
def test_effective_generation_settings_must_match(field: str, value: Any) -> None:
    r = reading()
    r["counters"]["generation_settings"][field] = value
    assert reading_problems(r, "lookup")


def test_omitted_zero_draft_counters_are_honest_but_do_not_prove_a_gain() -> None:
    r = reading()
    del r["counters"]["draft_n"]
    del r["counters"]["draft_n_accepted"]
    assert not reading_problems(r, "lookup")
    r["counters"]["draft_n"] = 12
    assert reading_problems(r, "lookup")
    assert reading_problems(reading(), "control")


@pytest.mark.parametrize("fault,status", [("quality", "stopped-quality"), ("ids", "stopped-quality"),
                                         ("short", "stopped-validity"), ("reuse", "stopped-validity"),
                                         ("counter", "stopped-validity"), ("inactive", "stopped-validity"),
                                         ("cleanup", "stopped-cleanup")])
def test_stop_means_no_next_load(fault: str, status: str) -> None:
    engine = LookupSimulator(fault=fault)
    result = run(engine)
    assert result["status"] == status
    assert engine.starts == engine.stops == (1 if fault == "cleanup" else 2)
    assert not any(c["complete"] for c in result["summary"]["cases"].values())


@pytest.mark.parametrize("fault,error", [("start-error", RuntimeError), ("request-error", RuntimeError),
                                       ("start-interrupt", KeyboardInterrupt), ("request-interrupt", KeyboardInterrupt)])
def test_failed_start_or_request_keeps_row_and_stops_same_engine(fault: str, error: type[BaseException]) -> None:
    engine = LookupSimulator(fault=fault)
    checkpoints: list[dict[str, Any]] = []
    with pytest.raises(error):
        evaluate(engine, config_sha="config", power=lambda: "simulated", simulated=True,
                 checkpoint=lambda r: checkpoints.append(deepcopy(r)))
    assert engine.starts == engine.stops == 1 and not engine.active
    assert len(checkpoints[-1]["trials"]) == 1
    assert checkpoints[-1]["trials"][0]["errors"]


@pytest.mark.parametrize("boundary", [0, 1, 2, 3])
def test_power_loss_stops_at_the_existing_load_boundary(boundary: int) -> None:
    calls = 0
    engine = LookupSimulator()
    def power() -> str:
        nonlocal calls
        calls += 1
        return "battery" if calls - 1 == boundary else "ac"
    result = evaluate(engine, config_sha="config", power=power, checkpoint=lambda r: None)
    assert result["status"] == "stopped-power"
    assert engine.starts == engine.stops == (boundary + 1) // 2


def test_power_after_interrupt_keeps_completed_reading_and_same_exception() -> None:
    original = KeyboardInterrupt("after load")
    calls = 0
    snapshots: list[dict[str, Any]] = []
    def power() -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise original
        return "ac"
    engine = LookupSimulator()
    with pytest.raises(KeyboardInterrupt) as caught:
        evaluate(engine, config_sha="config", power=power, checkpoint=lambda r: snapshots.append(deepcopy(r)))
    assert caught.value is original and engine.starts == engine.stops == 1
    assert snapshots[-1]["trials"][0]["reading"]["text"]
    assert snapshots[-1]["trials"][0]["cleanup"]["process_gone"]


@pytest.mark.parametrize("mutation", ["duplicate-first", "duplicate-last", "missing", "reverse", "unknown-arm", "unknown-case", "wrong-config"])
def test_population_and_configuration_are_not_silently_repaired(mutation: str) -> None:
    rows = deepcopy(run()["trials"])
    if mutation == "duplicate-first": rows.insert(0, deepcopy(rows[0]))
    elif mutation == "duplicate-last": rows.append(deepcopy(rows[1]))
    elif mutation == "missing": rows.pop()
    elif mutation == "reverse": rows.reverse()
    elif mutation == "unknown-arm": rows[0]["arm"] = "other"
    elif mutation == "unknown-case": rows[0]["case_id"] = "other"
    else:
        for row in rows: row["configuration_sha256"] = "different"
    summary = summarize(rows, "config", simulated=True)
    assert not any(c["complete"] for c in summary["cases"].values()) if mutation != "wrong-config" else all(not c["complete"] for c in summary["cases"].values())


def test_reference_variation_cannot_be_pooled_as_repeatable_preservation() -> None:
    rows = deepcopy(run()["trials"])
    for row in rows:
        if row["case_id"] == "copy-prose" and row["block"] == 1:
            row["reading"]["text"] += " block variation"
    case = summarize(rows, "config", simulated=True)["cases"]["copy-prose"]
    assert not case["complete"] and not case["reference_repeatable"]


def test_complete_request_latency_not_draft_acceptance_is_the_primary_endpoint() -> None:
    rows = deepcopy(run()["trials"])
    for row in rows:
        if row["arm"] == "lookup": row["reading"]["total_s"] = 12
    case = summarize(rows, "config", simulated=True)["cases"]["copy-prose"]
    assert case["acceptance_ratio"] > .7
    assert "no demonstrated" in case["outcome"]


def test_a_first_word_regression_is_reported_as_a_tradeoff() -> None:
    rows = deepcopy(run()["trials"])
    for row in rows:
        if row["arm"] == "lookup": row["reading"]["ttft_s"] = 2
    case = summarize(rows, "config", simulated=True)["cases"]["copy-prose"]
    assert "trade-off" in case["outcome"]


def test_natural_eos_is_valid_only_with_complete_equal_work() -> None:
    rows = deepcopy(run()["trials"])
    for row in rows:
        reading = row["reading"]
        reading["token_ids"] = reading["token_ids"][:32]
        reading["counters"].update(tokens_predicted=32, predicted_n=32, stop_type="eos")
        if row["arm"] == "lookup" and "draft_n" in reading["counters"]:
            reading["counters"].update(draft_n=30, draft_n_accepted=20)
    assert summarize(rows, "config", simulated=True)["cases"]["copy-prose"]["complete"]
    rows[1]["reading"]["token_ids"].pop()
    rows[1]["reading"]["counters"].update(tokens_predicted=31, predicted_n=31)
    case = summarize(rows, "config", simulated=True)["cases"]["copy-prose"]
    assert not case["complete"]


def test_limit_cannot_disguise_shortened_work() -> None:
    r = reading()
    r["token_ids"] = r["token_ids"][:64]
    r["counters"].update(tokens_predicted=64, predicted_n=64, draft_n=50, draft_n_accepted=40)
    assert reading_problems(r, "lookup")


def test_native_acceptance_can_include_verified_tokens_after_a_natural_eos() -> None:
    # b11139 increments n_draft_accepted for the verification batch BEFORE
    # process_token stops at EOS. It is not an emitted-token count.
    r = reading()
    r["token_ids"] = r["token_ids"][:32]
    r["counters"].update(tokens_predicted=32, predicted_n=32, stop_type="eos", draft_n=48, draft_n_accepted=40)
    assert not reading_problems(r, "lookup")


@pytest.mark.parametrize("arm,types,valid", [("control", "none,none", True), ("lookup", "none,ngram-simple", True),
                                            ("lookup", "ngram-simple,ngram-simple", False),
                                            ("lookup", "draft-simple,ngram-simple", False),
                                            ("control", "none,ngram-simple", False), ("lookup", "", False)])
def test_explicit_type_append_semantics_accept_only_the_registered_algorithm(arm: str, types: str, valid: bool) -> None:
    r = reading(arm)
    r["counters"]["generation_settings"]["speculative.types"] = types
    assert bool(reading_problems(r, arm)) is not valid


@pytest.mark.parametrize("algorithm", ["none", "ngram-simple"])
@pytest.mark.parametrize("operation", ["load", "update"])
def test_cli_refuses_server_only_lookup_before_changing_state(algorithm: str, operation: str, tmp_path: Path) -> None:
    from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
    from tests.resident_helpers import synthetic_model
    adapter = LlamaCppAdapter(cli=Path("never-executed"))
    model_file = tmp_path / "not-weights.gguf"; model_file.write_bytes(b"fixture")
    model = synthetic_model(str(model_file))
    prior = RuntimeParams(threads=3)
    if operation == "update": adapter.load(model, prior)
    request = RuntimeParams(lookup_decoding=LookupDecodingParams.model_validate({"algorithm": algorithm}))
    with pytest.raises(NotImplementedError, match="native server"):
        if operation == "load": adapter.load(model, request)
        else: adapter.set_runtime_params(request)
    assert adapter.is_loaded is (operation == "update")
    assert adapter.params == (prior if operation == "update" else RuntimeParams())


def test_cli_rejects_native_ids_before_any_process(tmp_path: Path) -> None:
    from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
    from tests.resident_helpers import synthetic_model
    adapter = LlamaCppAdapter(cli=Path("never-executed"))
    model_file = tmp_path / "not-weights.gguf"; model_file.write_bytes(b"fixture")
    adapter.load(synthetic_model(str(model_file)), RuntimeParams())
    with pytest.raises(NotImplementedError, match="generated token IDs"):
        adapter.generate(GenerationRequest(prompt="p", return_tokens=True))


def test_native_adapter_applies_and_clears_lookup_by_reloading(tmp_path: Path) -> None:
    from tests.test_resident_adapter import _adapter, _model
    adapter, backend = _adapter(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(threads=3))
    initial = adapter.pid
    group = LookupDecodingParams(algorithm="ngram-simple")
    adapter.set_runtime_params(RuntimeParams(lookup_decoding=group))
    assert adapter.pid != initial and adapter.params.lookup_decoding == group
    assert adapter.last_param_application and adapter.last_param_application.reloaded
    # The generic changes record must still serialize the nested capability.
    changes = adapter.last_param_application.model_dump(mode="json")["changes"]
    assert changes[0]["applied_value"] == group.model_dump(mode="json")
    adapter.set_runtime_params(RuntimeParams(lookup_decoding=None))
    assert adapter.params.lookup_decoding is None and adapter.params.threads == 3
    assert backend.is_running()
    adapter.unload()


def test_native_failed_lookup_reload_restores_the_prior_group(tmp_path: Path) -> None:
    from tests.test_resident_adapter import _adapter, _model
    from inferenceos.backends.resident_server import ResidentBackendError
    adapter, backend = _adapter(tmp_path)
    group = LookupDecodingParams(algorithm="ngram-simple")
    adapter.load(_model(tmp_path), RuntimeParams(threads=3, lookup_decoding=group))
    backend.fail_start_times = 1
    with pytest.raises(ResidentBackendError): adapter.set_runtime_params(RuntimeParams(lookup_decoding=None))
    assert adapter.params.lookup_decoding == group and adapter.params.threads == 3 and adapter.is_loaded
    assert adapter.last_param_application and not adapter.last_param_application.verified
    adapter.unload()
