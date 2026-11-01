"""Draft identity compatibility and scripted reports never confer execution permission."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import ast
import importlib
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError, create_model

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.tune.lookup_corpus import build_corpus, digest
from inferenceos.tune.lookup_qualification import (
    ConfigurationSnapshot, CostSamples, QualificationDraft, ScopeBinding, match_draft, parse_draft,
)
from inferenceos.tune.lookup_qualification_sim import evaluate_fixtures, fixture_draft, fixture_rows, simulate

CORPUS = build_corpus()
DRAFT = fixture_draft(CORPUS)


def runner() -> Any:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("evaluate_lookup_qualification")


def binding_change(path: tuple[str, ...], value: object) -> ScopeBinding:
    payload = DRAFT.binding.model_dump(mode="json")
    node = payload
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return ScopeBinding.model_validate_json(json.dumps(payload))


def test_matching_supplied_facts_never_means_qualified_or_allowed() -> None:
    result = match_draft(DRAFT, DRAFT.binding, CORPUS)
    assert result.same_scope and result.supplied_effects_complete and result.has_each_cost_component
    assert not result.tuning_allowed and not result.deployment_allowed
    assert result.status == "development-draft-only"
    assert "not independently accepted" in result.problems[-1]
    assert parse_draft(DRAFT.model_dump_json()) == DRAFT
    assert DRAFT.sha256 == parse_draft(DRAFT.model_dump_json()).sha256


@pytest.mark.parametrize("name,value", (
    ("threads", 5), ("threads_batch", 5), ("batch_size", 512), ("ubatch_size", 512),
    ("gpu_offload_layers", 9), ("use_mmap", False), ("flash_attention", True),
    ("device", "different-gpu"), ("engine_switches", "SOME_SWITCH=1"),
    ("placement_overrides", "piece=CPU"), ("kv_cache_type", "q8_0"),
    ("kv_cache_type_k", "q8_0"), ("kv_cache_type_v", "q8_0"),
))
def test_every_path_and_other_base_field_changes_scope(name: str, value: object) -> None:
    changed = binding_change(("effective", name), value)
    result = match_draft(DRAFT, changed, CORPUS)
    assert not result.same_scope
    assert any(name in p for p in result.problems)
    assert not result.tuning_allowed


def test_context_and_requested_null_defaults_are_bound_separately() -> None:
    changed = DRAFT.binding.model_dump(mode="json")
    changed["requested"]["context_length"] = changed["effective"]["context_length"] = 8192
    changed["request"]["context_per_slot"] = 8192
    assert not match_draft(DRAFT, ScopeBinding.model_validate_json(json.dumps(changed)), CORPUS).same_scope
    requested_change = binding_change(("requested", "threads"), 4)
    assert not match_draft(DRAFT, requested_change, CORPUS).same_scope  # same effect, different requested default
    with pytest.raises(ValidationError):
        ConfigurationSnapshot(context_length=4096)
    with pytest.raises(ValidationError):
        binding_change(("effective", "threads"), "4")


@pytest.mark.parametrize("path", (
    ("model_sha256",), ("tokenizer_sha256",), ("corpus_sha256",),
    ("machine", "hostname"), ("machine", "os_build"), ("machine", "cpu_name"),
    ("machine", "driver_version"), ("machine", "power_plan"), ("machine", "gpu_name"),
    ("machine", "device_name"), ("toolchain", "environment_sha256"),
))
def test_identity_transfer_is_never_inferred(path: tuple[str, ...]) -> None:
    value = "f" * 64 if path[-1].endswith("sha256") else "different-identity"
    changed = binding_change(path, value)
    assert not match_draft(DRAFT, changed, CORPUS).same_scope


def test_actual_executable_and_every_required_library_are_bound() -> None:
    data = DRAFT.binding.model_dump(mode="json")
    old = data["toolchain"]["executable"]["path"]
    data["toolchain"]["executable"]["path"] = "C:/other/llama-server.exe"
    next(p for p in data["toolchain"]["files"] if p["path"] == old)["path"] = "C:/other/llama-server.exe"
    assert not match_draft(DRAFT, ScopeBinding.model_validate_json(json.dumps(data)), CORPUS).same_scope
    for index in range(len(DRAFT.binding.toolchain.files)):
        data = DRAFT.binding.model_dump(mode="json")
        pin = data["toolchain"]["files"][index]
        pin["sha256"] = "f" * 64
        if pin["path"] == old:
            data["toolchain"]["executable"]["sha256"] = pin["sha256"]
        assert not match_draft(DRAFT, ScopeBinding.model_validate_json(json.dumps(data)), CORPUS).same_scope
        data["toolchain"]["files"].pop(index)
        with pytest.raises(ValidationError):
            ScopeBinding.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("path,value", (
    (("machine", "usable_gpu_count"), 2), (("machine", "physical_cores"), True),
    (("machine", "pci_device_id"), "unknown id"), (("machine", "driver_version"), "unknown"),
    (("machine", "gpu_vram_bytes"), 0), (("machine", "logical_cores"), 1),
    (("request", "wire_format"), "chat"), (("request", "lifecycle"), "warm"),
    (("request", "cache_prompt"), True), (("request", "cache_prompt"), 0),
    (("request", "cache_reuse"), 0), (("request", "swa_full"), True),
    (("request", "ignore_eos"), True), (("request", "in_flight_requests"), 2),
    (("request", "generated_ids"), False), (("request", "lookup_length"), 12.0),
    (("request", "draft_length"), 32), (("request", "offline"), False),
))
def test_unknown_ambiguous_or_out_of_scope_metadata_refused(path: tuple[str, ...], value: object) -> None:
    with pytest.raises(ValidationError):
        binding_change(path, value)


def test_unknown_and_contradictory_observations_do_not_become_shown() -> None:
    changed = binding_change(("unknown_effects",), ["threads"])
    assert not match_draft(DRAFT, changed, CORPUS).supplied_effects_complete
    changed = binding_change(("effective", "threads"), None)
    assert not match_draft(DRAFT, changed, CORPUS).supplied_effects_complete
    changed = binding_change(("requested", "threads"), 5)
    result = match_draft(DRAFT, changed, CORPUS)
    assert not result.supplied_effects_complete and any("contradicted" in p for p in result.problems)
    changed = binding_change(("toolchain", "ambient_config_absent"), False)
    assert not match_draft(DRAFT, changed, CORPUS).supplied_effects_complete


def test_shared_and_per_side_kv_precedence_is_not_raw_field_equality() -> None:
    data = DRAFT.binding.model_dump(mode="json")
    data["requested"].update(kv_cache_type="f16", kv_cache_type_k="q8_0")
    data["effective"].update(kv_cache_type=None, kv_cache_type_k="q8_0", kv_cache_type_v="f16")
    binding = ScopeBinding.model_validate_json(json.dumps(data))
    draft = DRAFT.model_copy(update={"binding": binding})
    assert match_draft(draft, binding, CORPUS).supplied_effects_complete
    data["effective"]["kv_cache_type_v"] = "q8_0"
    assert not match_draft(draft, ScopeBinding.model_validate_json(json.dumps(data)), CORPUS).supplied_effects_complete


def test_missing_costs_wrong_task_costs_and_revocation_are_visible() -> None:
    missing = DRAFT.model_copy(update={"costs": ()})
    assert not match_draft(missing, missing.binding, CORPUS).has_each_cost_component
    wrong = DRAFT.costs[0].model_copy(update={"case_sha256": digest("another task")})
    draft = DRAFT.model_copy(update={"costs": (wrong, *DRAFT.costs[1:]), "revoked": True})
    result = match_draft(draft, draft.binding, CORPUS)
    assert not result.has_each_cost_component and not result.tuning_allowed
    assert "development draft revoked" in result.problems
    assert any("different task" in p for p in result.problems)


@pytest.mark.parametrize("bad", ("NaN", "Infinity", "-Infinity", "1e999", "1.00000000000000005", "1e-999"))
def test_draft_parser_refuses_nonfinite_or_rounded_values(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_draft(DRAFT.model_dump_json().replace('"seconds":[1.0,1.1]', '"seconds":[' + bad + ',1.1]', 1))


def test_signed_zero_does_not_alias_a_different_serialized_scope() -> None:
    changed = binding_change(("request", "temperature"), -0.0)
    assert digest(changed.model_dump(mode="json")) != digest(DRAFT.binding.model_dump(mode="json"))
    assert not match_draft(DRAFT, changed, CORPUS).same_scope


def test_own_finite_float_serialization_round_trips_without_precision_loss() -> None:
    costs = tuple(c.model_copy(update={"seconds": (5e-324, .1, 1.2345678901234567, 1e250)}) for c in DRAFT.costs)
    draft = DRAFT.model_copy(update={"costs": costs})
    restored = parse_draft(draft.model_dump_json())
    assert restored == draft and restored.sha256 == draft.sha256


def test_draft_cannot_impersonate_reviewed_certificate_or_mutate_nested_state() -> None:
    data = DRAFT.model_dump(mode="json")
    data["independent_review"] = "accepted"
    with pytest.raises(ValidationError):
        parse_draft(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_draft(DRAFT.model_dump_json().replace('"simulated":true', '"simulated":true,"simulated":false'))
    with pytest.raises(ValidationError):
        DRAFT.binding.machine.hostname = "other"
    broken = DRAFT.model_copy(update={"simulated": "false"})
    with pytest.raises(ValidationError):
        match_draft(broken, DRAFT.binding, CORPUS)


@pytest.mark.parametrize("side", ("requested", "effective"))
@pytest.mark.parametrize("boundary", ("draft", "observed"))
@pytest.mark.parametrize("field", tuple(ConfigurationSnapshot.model_fields))
def test_nested_omissions_are_refused_before_defaults_can_be_serialized(side: str, boundary: str, field: str) -> None:
    supplied = getattr(DRAFT.binding, side).model_dump()
    del supplied[field]
    snapshot = ConfigurationSnapshot.model_construct(**supplied)
    binding = DRAFT.binding.model_copy(update={side: snapshot})
    draft = DRAFT.model_copy(update={"binding": binding}) if boundary == "draft" else DRAFT
    observed = binding if boundary == "observed" else DRAFT.binding
    with pytest.raises(ValueError, match="all configuration fields"):
        match_draft(draft, observed, CORPUS)
    if boundary == "draft":
        with pytest.raises(ValueError, match="all configuration fields"):
            _ = draft.sha256


@pytest.mark.parametrize("side", ("requested", "effective"))
def test_explicit_nulls_and_fully_supplied_constructs_remain_valid(side: str) -> None:
    supplied = getattr(DRAFT.binding, side).model_dump()
    snapshot = ConfigurationSnapshot.model_construct(**supplied)
    binding = DRAFT.binding.model_copy(update={side: snapshot})
    draft = DRAFT.model_copy(update={"binding": binding})
    assert snapshot.model_fields_set == set(ConfigurationSnapshot.model_fields)
    assert snapshot == getattr(DRAFT.binding, side)
    assert match_draft(draft, binding, CORPUS).same_scope
    assert draft.sha256 == DRAFT.sha256
    assert parse_draft(draft.model_dump_json()).sha256 == DRAFT.sha256


@pytest.mark.parametrize("side", ("requested", "effective"))
@pytest.mark.parametrize("representation", ("parent-model", "raw-dict"))
def test_unchecked_substitutions_cannot_synthesize_nested_configuration_provenance(side: str, representation: str) -> None:
    partial: object = RuntimeConfiguration(context_length=4096) if representation == "parent-model" else {"context_length": 4096}
    binding = DRAFT.binding.model_copy(update={side: partial})
    draft = DRAFT.model_copy(update={"binding": binding})
    for left, right in ((DRAFT, binding), (draft, DRAFT.binding)):
        with pytest.raises(ValueError, match="all configuration fields"):
            match_draft(left, right, CORPUS)
    with pytest.raises(ValueError, match="all configuration fields"):
        _ = draft.sha256


def foreign_configuration_type() -> Any:
    fields: dict[str, Any] = {n: (f.annotation, f.default) for n, f in RuntimeConfiguration.model_fields.items()}
    return create_model("ForeignConfiguration", **fields)


@pytest.mark.parametrize("side", ("requested", "effective"))
@pytest.mark.parametrize("boundary", ("draft", "observed"))
def test_foreign_models_cannot_supply_implicit_configuration_fields(side: str, boundary: str) -> None:
    supplied = {n: v for n, v in getattr(DRAFT.binding, side).model_dump().items() if v is not None}
    foreign = foreign_configuration_type()(**supplied)
    assert foreign.model_fields_set != set(RuntimeConfiguration.model_fields)
    binding_dict = {n: getattr(DRAFT.binding, n) for n in ScopeBinding.model_fields}
    binding_dict[side] = foreign
    bindings: tuple[Any, ...] = (DRAFT.binding.model_copy(update={side: foreign}), binding_dict)
    for binding in bindings:
        draft = DRAFT.model_copy(update={"binding": binding}) if boundary == "draft" else DRAFT
        observed = binding if boundary == "observed" else DRAFT.binding
        with pytest.raises(ValueError, match="all configuration fields"):
            match_draft(draft, observed, CORPUS)
        if boundary == "draft":
            with pytest.raises(ValueError, match="all configuration fields"):
                _ = draft.sha256


def test_complete_explicit_foreign_model_values_keep_the_same_draft_identity() -> None:
    for side in ("requested", "effective"):
        foreign = foreign_configuration_type()(**getattr(DRAFT.binding, side).model_dump())
        binding = DRAFT.binding.model_copy(update={side: foreign})
        draft = DRAFT.model_copy(update={"binding": binding})
        assert match_draft(draft, binding, CORPUS).same_scope
        assert draft.sha256 == DRAFT.sha256


@pytest.mark.parametrize("token", ("9007199254740993", "9007199254740993.0", "9.007199254740993e15",
                                 "18014398509481985", "9007199254740995", "1" + "0" * 309))
def test_cost_parser_refuses_inexact_integers_as_well_as_decimal_spellings(token: str) -> None:
    payload = DRAFT.model_dump_json().replace('"seconds":[1.0,1.1]', '"seconds":[' + token + ',1.1]', 1)
    with pytest.raises(ValueError):
        parse_draft(payload)


@pytest.mark.parametrize("sample", (2**53 + 1, 2**53 + 3, 2**54 + 1, 10**309))
def test_costs_refuse_rounding_through_constructor_unchecked_copy_matcher_and_hash(sample: int) -> None:
    data = DRAFT.costs[0].model_dump()
    data["seconds"] = (sample,)
    with pytest.raises(ValidationError, match="represented exactly"):
        CostSamples(**data)
    for cost in (DRAFT.costs[0].model_copy(update={"seconds": (sample,)}), CostSamples.model_construct(**data)):
        draft = DRAFT.model_copy(update={"costs": (cost, *DRAFT.costs[1:])})
        with pytest.raises(ValueError, match="represented exactly"):
            match_draft(draft, DRAFT.binding, CORPUS)
        with pytest.raises(ValueError, match="represented exactly"):
            _ = draft.sha256


@pytest.mark.parametrize("sample", (1, 2**53 - 1, 2**53, 2**53 + 2, 2**54, 2**100))
def test_exact_large_cost_integers_round_trip_without_a_blanket_size_cutoff(sample: int) -> None:
    payload = DRAFT.model_dump(mode="json")
    payload["costs"][0]["seconds"][0] = sample
    draft = parse_draft(json.dumps(payload))
    assert int(draft.costs[0].seconds[0]) == sample
    assert parse_draft(draft.model_dump_json()).sha256 == draft.sha256
    assert match_draft(draft, DRAFT.binding, CORPUS).same_scope


@pytest.mark.parametrize("sample", (True, "1", Decimal("1.00000000000000005"), Fraction(1, 3)))
def test_costs_do_not_accept_other_numeric_types_via_float_coercion(sample: object) -> None:
    data = DRAFT.costs[0].model_dump()
    data["seconds"] = (sample,)
    with pytest.raises(ValidationError):
        CostSamples(**data)


def test_large_integer_identity_fields_remain_exact_and_distinct() -> None:
    payload = DRAFT.model_dump(mode="json")
    digests = []
    for ram in (2**53, 2**53 + 1):
        payload["binding"]["machine"]["ram_bytes"] = ram
        payload["binding"]["machine"]["gpu_vram_bytes"] = 10**100 + ram
        draft = parse_draft(json.dumps(payload))
        assert draft.binding.machine.ram_bytes == ram
        assert draft.binding.machine.gpu_vram_bytes == 10**100 + ram
        digests.append(draft.sha256)
    assert digests[0] != digests[1]


@pytest.mark.parametrize("boundary", ("draft", "binding", "cost"))
def test_revalidation_does_not_drop_unknown_fields_from_unchecked_copies(boundary: str) -> None:
    draft = DRAFT
    if boundary == "draft":
        draft = draft.model_copy(update={"accepted": True})
    elif boundary == "binding":
        draft = draft.model_copy(update={"binding": draft.binding.model_copy(update={"accepted": True})})
    else:
        cost = draft.costs[0].model_copy(update={"accepted": True})
        draft = draft.model_copy(update={"costs": (cost, *draft.costs[1:])})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        match_draft(draft, DRAFT.binding, CORPUS)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _ = draft.sha256


def test_matching_is_pure_and_collects_no_machine_or_file_evidence() -> None:
    with patch("subprocess.Popen", side_effect=AssertionError("no processes")), \
         patch.object(Path, "read_bytes", side_effect=AssertionError("no file access")):
        assert match_draft(DRAFT, DRAFT.binding, CORPUS).same_scope


def test_draft_schema_cannot_assign_or_pass_swa_full_to_runtime() -> None:
    path = Path(__file__).resolve().parents[1] / "src/inferenceos/tune/lookup_qualification.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    declarations = [n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                    and isinstance(n.target, ast.Name) and n.target.id == "swa_full"]
    assert len(declarations) == 1
    assert isinstance(declarations[0].value, ast.Constant) and declarations[0].value.value is False
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == "swa_full"]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                and n.attr == "swa_full" and isinstance(n.ctx, ast.Store)]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id in ("RuntimeParams", "setattr", "exec", "eval")]


def test_simulation_retains_all_pairs_and_explicitly_has_no_hardware_meaning() -> None:
    report = simulate()
    assert report["all_fixture_pairs_pass"] and len(report["rows"]) == 84
    assert len(report["cases"]) == 42 and report["corpus_sha256"] == CORPUS.sha256
    for name in ("hardware_qualification", "automatic_admission", "deployment_permission", "native_evidence_collected",
                 "token_fit_verified", "D2_protocol_frozen"):
        assert report[name] is False
    assert report["performance_conclusion"] is None


@pytest.mark.parametrize("defect", ("missing", "duplicate_control", "duplicate_lookup", "reordered", "unknown", "wrong_hash",
                                    "wrong_control", "wrong_lookup", "both_wrong", "limit", "different_correct_newline"))
def test_fixture_report_does_not_pick_a_good_duplicate_or_drop_failure(defect: str) -> None:
    rows = list(fixture_rows(CORPUS))
    first = rows[0]
    if defect == "missing":
        rows.pop(0)
    elif defect in ("duplicate_control", "duplicate_lookup"):
        arm = "control" if defect.endswith("control") else "lookup"
        rows.insert(0, next(r for r in rows[:2] if r.arm == arm).model_copy(update={"text": "WRONG"}))
    elif defect == "reordered":
        rows[:2] = reversed(rows[:2])
    elif defect == "unknown":
        rows.append(first.model_copy(update={"case_id": "unknown-case"}))
    elif defect == "wrong_hash":
        rows[0] = first.model_copy(update={"case_sha256": "f" * 64})
    elif defect in ("wrong_control", "wrong_lookup", "both_wrong"):
        for i in range(2):
            if defect == "both_wrong" or rows[i].arm == defect.removeprefix("wrong_"):
                rows[i] = rows[i].model_copy(update={"text": "WRONG"})
    elif defect == "limit":
        rows[0] = first.model_copy(update={"ending": "limit"})
    else:
        rows[0] = first.model_copy(update={"text": first.text + "\n"})
    report = evaluate_fixtures(CORPUS, rows)
    assert not report["all_fixture_pairs_pass"]
    assert len(report["rows"]) == len(rows) and len(report["cases"]) == 42
    assert not report["automatic_admission"]
    if defect in ("both_wrong", "wrong_control"):
        assert report["cases"][0]["outcome"].startswith("task unqualified: control")


def test_software_only_cli_refuses_real_options_and_overwrite(tmp_path: Path) -> None:
    module = runner()
    output = tmp_path / "fixture.json"
    with patch("subprocess.Popen", side_effect=AssertionError("no child processes")):
        assert module.main(["--simulate", "--out", str(output)]) == 0
    before = output.read_bytes()
    assert json.loads(before)["simulated"] is True
    assert module.main(["--simulate", "--out", str(output)]) == 4
    assert output.read_bytes() == before
    with pytest.raises(SystemExit):
        module.main(["--out", str(tmp_path / "missing-flag.json")])
    with pytest.raises(SystemExit):
        module.main(["--simulate", "--model", "anything.gguf", "--out", str(tmp_path / "real.json")])


def test_exclusive_publication_refuses_concurrent_writer_and_reports_io_failure(tmp_path: Path) -> None:
    module = runner()
    output = tmp_path / "fixture.json"
    report = simulate()
    def concurrent() -> dict[str, Any]:
        output.write_text("another writer", encoding="utf-8")
        return report
    with patch.object(module, "simulate", side_effect=concurrent):
        assert module.main(["--simulate", "--out", str(output)]) == 4
    assert output.read_text() == "another writer"
    with patch.object(module, "simulate", return_value=deepcopy(report)), \
         patch.object(Path, "open", side_effect=PermissionError("denied")):
        assert module.main(["--simulate", "--out", str(tmp_path / "denied.json")]) == 7
