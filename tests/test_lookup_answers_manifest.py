"""Stage-B provenance tests use tiny synthetic GGUFs and fabricated A evidence.

No ignored artifact, real model, native backend or process is used. The production
accepted protocol/proposal hashes are monkeypatched ONLY to bind this synthetic
fixture's bytes; the public parser has no pin-override argument.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from inferenceos.tune import lookup_answers_manifest as am
from inferenceos.tune.lookup_answers import VERSION as CORE_VERSION, plan, suites
from inferenceos.tune.lookup_corpus import digest
from inferenceos.tune.lookup_fit import audit_fit, collect_fit
from inferenceos.tune.lookup_fit_observers import machine_from_raw
from inferenceos.tune.lookup_fit_sim import FitSimulator
from inferenceos.tune.lookup_qualification import ConfigurationSnapshot
from inferenceos.tune.lookup_role_cost import VERSION as COST_VERSION
from tests.test_lookup_fit import native_fixture


def encoded(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, indent=2).encode()


@pytest.fixture(autouse=True)
def no_process() -> Any:
    with patch("subprocess.Popen", side_effect=AssertionError("no native process in manifest tests")):
        yield


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    runner, original, _, proposal_path, settings = native_fixture(tmp_path)
    frozen = runner.FrozenInputs
    def windows(*args: Any, **kwargs: Any) -> Any:
        return frozen(*args, **kwargs, platform="win32", environment=lambda: dict(APPDATA="", PROGRAMDATA=""))
    monkeypatch.setattr(runner, "FrozenInputs", windows)
    original_rules = tmp_path / "rules.json"
    original_raw = original_rules.read_bytes()
    gate = runner.Preflight(tmp_path, settings, original, original_rules, am.sha256(original_raw),
                            proposal_path, original.proposal_sha256)
    record = collect_fit(FitSimulator(original), original, run_id="MANIFEST_TEST_FAKE_A",
                         power=lambda: "ac", guard=lambda: None, checkpoint=lambda _: None)
    selected = str(gate.selected)
    for row in record["loads"]:
        for snap in row["snapshots"]:
            for file in snap["image"]["files"]:
                file["path"] = str(tmp_path / file["path"].removeprefix("C:/D2-FIT-FIXTURE/"))
            snap["image"]["executable"] = selected
            snap["selected_executable"] = selected
            snap["command"][0] = selected
    record["initial_power"] = "ac"
    record["preservation"] = gate.preservation()
    raw_record = encoded(record)
    audit = audit_fit(record, original)
    assert audit["observations_complete"], audit
    raw_audit = encoded(dict(audit, source_sha256=am.sha256(raw_record), rules_sha256=am.sha256(original_raw),
                              proposal_sha256=original.proposal_sha256))
    review = dict(verdict="accept Stage-A results", findings=[], run_id=record["run_id"],
                  record_sha256=am.sha256(raw_record), audit_sha256=am.sha256(raw_audit),
                  manifest_sha256=am.sha256(original_raw), proposal_sha256=original.proposal_sha256,
                  fresh_offline_CLI_exit=0, fresh_offline_CLI_audit_byte_identical=True, pure_recomputation_exact=True)
    protocol_path = tmp_path / "docs/lookup-qualification-protocol.md"
    paths = dict(proposal="proposal.json", protocol_document="docs/lookup-qualification-protocol.md",
                 stage_a_manifest="_jobs/out/original_A_manifest.json", stage_a_record="_jobs/out/original_A_record.json",
                 stage_a_audit="_jobs/out/original_A_audit.json", stage_a_review="_jobs/out/independent_A_review.json")
    by_role = dict(proposal=proposal_path.read_bytes(), protocol_document=protocol_path.read_bytes(),
                   stage_a_manifest=original_raw, stage_a_record=raw_record, stage_a_audit=raw_audit,
                   stage_a_review=encoded(review))
    artifacts = {paths[role]: raw for role, raw in by_role.items()}
    for relative, raw in artifacts.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    extra_source = tmp_path / "scripts/current.py"
    extra_source.parent.mkdir(exist_ok=True)
    extra_source.write_bytes(b"# added software only; never executed\n")
    # Stage B may contain reviewed repairs to files that existed during A. Keep
    # the original artifacts untouched; only the execution file/map advances.
    changed_source = next(iter(original.python_files))
    (tmp_path / changed_source).write_bytes(b"# revised Stage-B source; never executed\n")
    sources = {**original.python_files, changed_source: am.sha256((tmp_path / changed_source).read_bytes()),
               "scripts/current.py": am.sha256(extra_source.read_bytes())}
    bases = {row["model"]: row["snapshots"][0]["effects"]["configuration"] for row in record["loads"]}
    snap = record["loads"][0]["snapshots"][0]
    manifest = dict(version=am.VERSION, stage="B-complete-answers", protocol=original.protocol,
                    **{role: dict(path=paths[role], sha256=am.sha256(raw)) for role, raw in by_role.items()},
                    answer_core_version=CORE_VERSION, cost_schema_version=COST_VERSION,
                    corpus_sha256=original.corpus_sha256, plan_sha256=digest(plan()), suites_sha256=digest(suites()),
                    models=[m.model_dump(mode="json") for m in original.models], effective_bases=bases,
                    machine=machine_from_raw(snap["machine"], bases["llama3b"]["device"]),
                    python_files=sources, engine_files=deepcopy(original.engine_files), hardware_authorized=False,
                    automatic_admission=False, deployment_allowed=False)
    monkeypatch.setattr(am, "ACCEPTED_PROTOCOL_SHA256", original.protocol_document_sha256)
    monkeypatch.setattr(am, "ACCEPTED_PROPOSAL_SHA256", original.proposal_sha256)
    rules = tmp_path / "answers_rules.json"
    rules.write_bytes(encoded(manifest))
    return dict(root=tmp_path, runner=runner, settings=settings, manifest=manifest, artifacts=artifacts,
                original=original, record=record, rules=rules)


def parsed(bundle: dict[str, Any]) -> am.AnswerInputs:
    return am.parse_answer_manifest(encoded(bundle["manifest"]), bundle["artifacts"])


def change_artifact(bundle: dict[str, Any], role: str, change: Any) -> None:
    ref = bundle["manifest"][role]
    raw = json.loads(bundle["artifacts"][ref["path"]])
    change(raw)
    updated = encoded(raw)
    bundle["artifacts"][ref["path"]] = updated
    ref["sha256"] = am.sha256(updated)


def preflight(bundle: dict[str, Any], *, settings: Any = None, factory: Any = None) -> am.AnswerPreflight:
    bundle["rules"].write_bytes(encoded(bundle["manifest"]))
    return am.AnswerPreflight(bundle["root"], settings or bundle["settings"], parsed(bundle), bundle["rules"],
                              am.sha256(bundle["rules"].read_bytes()),
                              preflight_factory=factory or bundle["runner"].Preflight)


def test_original_A_manifest_is_not_refrozen_as_current_B_evidence(bundle: dict[str, Any]) -> None:
    before = deepcopy(bundle["artifacts"])
    inputs = parsed(bundle)
    assert inputs.original_stage_a_manifest == bundle["original"]
    assert inputs.original_stage_a_manifest.python_files != inputs.manifest.python_files
    assert inputs.derived_fit_manifest().python_files == inputs.manifest.python_files
    assert inputs.derived_fit_manifest().generation_allowed is False  # preflight adapter only
    assert inputs.provenance()["original_stage_a_manifest_sha256"] == am.sha256(before[inputs.manifest.stage_a_manifest.path])
    scope = inputs.derived_scope()
    assert scope["executor_sha256"] == digest(inputs.manifest.python_files)
    assert all(len(m["fit_tokens"]) == 200 for m in scope["models"].values())
    scope["models"]["llama3b"]["effective"]["threads"] = 999
    assert inputs.derived_scope()["models"]["llama3b"]["effective"]["threads"] == 4
    assert bundle["artifacts"] == before
    assert not any(inputs.provenance()[k] for k in ("hardware_authorized", "automatic_admission", "deployment_allowed"))


@pytest.mark.parametrize("role", am._ARTIFACT_ROLES)
@pytest.mark.parametrize("fault", ("missing", "bytes"))
def test_every_original_artifact_is_required_and_byte_bound(bundle: dict[str, Any], role: str, fault: str) -> None:
    path = bundle["manifest"][role]["path"]
    if fault == "missing":
        bundle["artifacts"].pop(path)
    else:
        bundle["artifacts"][path] += b"\n"
    with pytest.raises(ValueError):
        parsed(bundle)


@pytest.mark.parametrize("field", ("answer_core_version", "cost_schema_version", "corpus_sha256", "plan_sha256", "suites_sha256"))
def test_versions_and_fixed_scientific_inputs_cannot_drift(bundle: dict[str, Any], field: str) -> None:
    bundle["manifest"][field] = "0"*64
    with pytest.raises(ValueError):
        parsed(bundle)


@pytest.mark.parametrize("field", ("hardware_authorized", "automatic_admission", "deployment_allowed"))
@pytest.mark.parametrize("value", (0, True, None, "false"))
def test_permission_declarations_are_explicit_booleans(bundle: dict[str, Any], field: str, value: Any) -> None:
    bundle["manifest"][field] = value
    with pytest.raises(ValueError):
        parsed(bundle)


@pytest.mark.parametrize("target", ("requested", "effective"))
@pytest.mark.parametrize("field", tuple(ConfigurationSnapshot.model_fields))
def test_missing_nested_configuration_is_never_filled(bundle: dict[str, Any], target: str, field: str) -> None:
    data = bundle["manifest"]
    config = data["models"][0]["configuration"] if target == "requested" else data["effective_bases"]["llama3b"]
    config.pop(field)
    with pytest.raises(ValueError):
        parsed(bundle)


def test_constructed_incomplete_configuration_refused_before_serialization(bundle: dict[str, Any]) -> None:
    inputs = parsed(bundle)
    full = inputs.manifest
    bad_config = ConfigurationSnapshot.model_construct(context_length=4096)
    bad_model = full.models[0].model_copy(update=dict(configuration=bad_config))
    bad = full.model_copy(update=dict(models=(bad_model, full.models[1])))
    with pytest.raises(ValueError, match="all configuration fields"):
        am.validated_answer_manifest(bad)
    with pytest.raises(ValueError, match="all configuration fields"):
        replace(inputs, manifest=bad).revalidated()
    bad = full.model_copy(update=dict(effective_bases={**full.effective_bases, "llama3b": bad_config}))
    with pytest.raises(ValueError, match="all configuration fields"):
        am.validated_answer_manifest(bad)


def test_unchecked_input_copy_cannot_replace_derived_scope_or_source_map(bundle: dict[str, Any]) -> None:
    inputs = parsed(bundle)
    manifest = inputs.manifest.model_copy(update=dict(python_files={"src/foreign.py": "0"*64}))
    with pytest.raises(ValueError):
        replace(inputs, manifest=manifest).revalidated()
    scope = inputs.derived_scope()
    scope["models"]["llama3b"]["effective"]["threads"] = 10
    with pytest.raises(ValueError):
        replace(inputs, _scope_json=encoded(scope)).revalidated()


@pytest.mark.parametrize("fault", ("effective", "machine", "model", "engine", "model_order"))
def test_accepted_A_observation_scope_cannot_be_substituted(bundle: dict[str, Any], fault: str) -> None:
    data = bundle["manifest"]
    if fault == "effective": data["effective_bases"]["llama3b"]["threads"] = 7
    elif fault == "machine": data["machine"]["hostname"] = "replacement host"
    elif fault == "model": data["models"][0]["sha256"] = "0"*64
    elif fault == "engine": data["engine_files"][next(iter(data["engine_files"]))] = "0"*64
    elif fault == "model_order": data["models"].reverse()
    with pytest.raises(ValueError):
        parsed(bundle)


@pytest.mark.parametrize("fault", ("observation", "token_ids", "saved_receipt", "release", "simulated", "source_population"))
def test_original_raw_A_evidence_is_recomputed_not_its_pass_label(bundle: dict[str, Any], fault: str) -> None:
    def change(raw: Any) -> None:
        raw["observations_complete"] = True
        if fault == "observation": raw["loads"][0]["snapshots"][0]["slots"][0]["speculative"] = True
        elif fault == "token_ids": raw["loads"][0]["tokens"][0]["ids"][0] += 1
        elif fault == "saved_receipt": raw["preservation"]["models"][0]["model_file"]["sha256"] = "0"*64
        elif fault == "release": raw["loads"][0]["cleanup"]["credential_removed"] = False
        elif fault == "simulated": raw["simulated"] = True
        elif fault == "source_population": raw["preservation"]["frozen_hashes"]["src/extra.py"] = "0"*64
    change_artifact(bundle, "stage_a_record", change)
    with pytest.raises(ValueError, match="observations cannot be reproduced"):
        parsed(bundle)


@pytest.mark.parametrize("field", ("verdict", "findings", "run_id", "record_sha256", "audit_sha256", "manifest_sha256",
                                   "proposal_sha256", "fresh_offline_CLI_exit", "fresh_offline_CLI_audit_byte_identical",
                                   "pure_recomputation_exact"))
def test_exact_external_acceptance_receipt_required(bundle: dict[str, Any], field: str) -> None:
    def change(raw: Any) -> None:
        raw[field] = ["unresolved"] if field == "findings" else False
    change_artifact(bundle, "stage_a_review", change)
    with pytest.raises(ValueError, match="independent acceptance"):
        parsed(bundle)


def test_stored_A_public_audit_cannot_be_replaced_with_a_pass_boolean(bundle: dict[str, Any]) -> None:
    change_artifact(bundle, "stage_a_audit", lambda raw: raw.update(observations_complete=1))
    with pytest.raises(ValueError):
        parsed(bundle)


@pytest.mark.parametrize("path", ("../outside.json", "a//b", "C:/outside.json", "a\\b", "a/CON.json", "a/b.",
                                  "a/b ", "a/?b", "a/\nfile", "/absolute", "a/./b"))
def test_paths_are_canonical_and_safe(path: str) -> None:
    with pytest.raises(ValueError):
        am.ArtifactPin(path=path, sha256="0"*64)


@pytest.mark.parametrize("field", ("threads", "gpu_offload_layers", "context_length"))
@pytest.mark.parametrize("number", (True, 4.0))
def test_integer_configuration_identity_is_not_coerced(bundle: dict[str, Any], field: str, number: Any) -> None:
    bundle["manifest"]["effective_bases"]["llama3b"][field] = number
    with pytest.raises(ValueError):
        parsed(bundle)


def test_machine_integer_identity_remains_exact_beyond_binary_float_range(bundle: dict[str, Any]) -> None:
    raw = deepcopy(bundle["manifest"])
    raw["machine"]["ram_bytes"] = 9007199254740993
    manifest = am.AnswerManifest.model_validate_json(encoded(raw))
    assert manifest.machine.ram_bytes == 9007199254740993
    assert am.validated_answer_manifest(manifest).machine.ram_bytes == 9007199254740993
    # The supplied value remains exact; it still contradicts the actual A machine.
    with pytest.raises(ValueError, match="machine scope differs"):
        am.parse_answer_manifest(encoded(raw), bundle["artifacts"])


def test_duplicate_or_case_colliding_json_keys_and_extra_artifacts_are_refused(bundle: dict[str, Any]) -> None:
    raw = encoded(bundle["manifest"])
    for changed in (raw.replace(b'"version":', b'"version":"duplicate", "version":', 1),
                    raw.replace(b'"version":', b'"Version":"duplicate", "version":', 1)):
        with pytest.raises(ValueError): am.parse_answer_manifest(changed, bundle["artifacts"])
    bundle["artifacts"]["unlisted.json"] = b"{}"
    with pytest.raises(ValueError): parsed(bundle)


@pytest.mark.parametrize("role", ("rules", "stage_a_review"))
def test_linked_rules_or_provenance_are_refused_before_factory(bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                                              role: str) -> None:
    target = bundle["rules"] if role == "rules" else bundle["root"] / bundle["manifest"][role]["path"]
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    factory = Mock(side_effect=AssertionError("zero preflight inspection"))
    with pytest.raises(ValueError, match="linked"):
        preflight(bundle, factory=factory)
    factory.assert_not_called()


def test_read_only_B_preflight_uses_current_sources_and_preserves_original_A_bytes(bundle: dict[str, Any]) -> None:
    before = deepcopy(bundle["artifacts"])
    gate = preflight(bundle)
    gate.check(full=True)
    assert len(gate.models) == 2 and gate.selected.name == "llama-server.exe"
    assert gate.preservation()["frozen_hashes"] == {**bundle["manifest"]["python_files"], **bundle["manifest"]["engine_files"]}
    assert gate.answer_inputs.original_stage_a_manifest.python_files == bundle["original"].python_files
    assert bundle["artifacts"] == before


@pytest.mark.parametrize("fault", ("model", "saved_model", "saved_configuration", "missing_setup", "duplicate_setup",
                                   "alternate_executable", "identical_executable_copy", "source", "extra_source", "source_pin_omitted", "engine"))
def test_actual_preflight_refusals_make_zero_backend_constructions_or_starts(bundle: dict[str, Any], fault: str) -> None:
    root = bundle["root"]; settings = bundle["settings"]
    store = root / "data/saved_setups.json"
    if fault == "model": (root / bundle["original"].models[0].path).write_bytes(b"replaced")
    elif fault.startswith("saved_") or fault.endswith("setup"):
        value = json.loads(store.read_bytes())
        if fault == "saved_model": value["setups"][0]["model"]["sha256"] = "0"*64
        elif fault == "saved_configuration": value["setups"][0]["configuration"]["threads"] = 19
        elif fault == "missing_setup": value["setups"].pop(0)
        elif fault == "duplicate_setup": value["setups"].append(deepcopy(value["setups"][0]))
        store.write_bytes(encoded(value))
    elif fault in ("alternate_executable", "identical_executable_copy"):
        directory = root / "alternate"; directory.mkdir()
        original = next(p for p in bundle["manifest"]["engine_files"] if p.endswith("/llama-server.exe"))
        (directory / "llama-server.exe").write_bytes((root / original).read_bytes() if fault == "identical_executable_copy" else b"different")
        settings = settings.model_copy(update=dict(llama_cpp_dir=directory))
    elif fault == "source": (root / "scripts/current.py").write_bytes(b"changed")
    elif fault == "extra_source": (root / "scripts/unlisted.py").write_bytes(b"added")
    elif fault == "source_pin_omitted": bundle["manifest"]["python_files"].pop("scripts/current.py")
    elif fault == "engine": (root / next(iter(bundle["manifest"]["engine_files"]))).write_bytes(b"changed")
    backend_factory = Mock(side_effect=AssertionError("zero backend constructions"))
    with pytest.raises((ValueError, LookupError, OSError)):
        preflight(bundle, settings=settings)
        backend_factory()
    backend_factory.assert_not_called()


@pytest.mark.parametrize("role", am._ARTIFACT_ROLES)
def test_changed_artifact_stops_before_preflight_factory(bundle: dict[str, Any], role: str) -> None:
    (bundle["root"] / bundle["manifest"][role]["path"]).write_bytes(b"changed after parsing")
    factory = Mock(side_effect=AssertionError("zero preflight inspection"))
    with pytest.raises(ValueError): preflight(bundle, factory=factory)
    factory.assert_not_called()


@pytest.mark.parametrize("fault", ("model", "deleted_model", "saved", "source", "extra_source", "review", "protocol", "rules", "executable"))
def test_per_load_recheck_prevents_another_start(bundle: dict[str, Any], fault: str) -> None:
    gate = preflight(bundle)
    starts = []
    def start() -> None:
        gate.check()
        starts.append("start permitted, not actually dispatched")
    start()
    root = bundle["root"]
    if fault == "model": (root / bundle["original"].models[0].path).write_bytes(b"replaced")
    elif fault == "deleted_model": (root / bundle["original"].models[0].path).unlink()
    elif fault == "saved": (root / "data/saved_setups.json").write_bytes(b"{}")
    elif fault == "source": (root / "scripts/current.py").write_bytes(b"changed")
    elif fault == "extra_source": (root / "scripts/extra.py").write_bytes(b"extra")
    elif fault in ("review", "protocol"):
        role = "stage_a_review" if fault == "review" else "protocol_document"
        (root / bundle["manifest"][role]["path"]).write_bytes(b"changed")
    elif fault == "rules": bundle["rules"].write_bytes(b"{}")
    elif fault == "executable": (root / next(p for p in bundle["manifest"]["engine_files"] if p.endswith("/llama-server.exe"))).write_bytes(b"changed")
    with pytest.raises((ValueError, LookupError, OSError)):
        start()
    assert len(starts) == 1


@pytest.mark.parametrize("layout", ("changed", "renamed", "removed"))
def test_historical_and_execution_source_contexts_are_separate(bundle: dict[str, Any], layout: str) -> None:
    before = deepcopy(bundle["artifacts"])
    original = bundle["original"]
    path = next(iter(original.python_files))
    current = bundle["root"] / path
    assert am.sha256(current.read_bytes()) != original.python_files[path]
    if layout != "changed":
        if layout == "renamed":
            replacement = "src/revised.py"
            (bundle["root"] / replacement).write_bytes(current.read_bytes())
            bundle["manifest"]["python_files"][replacement] = bundle["manifest"]["python_files"][path]
        current.unlink()  # This is a tiny synthetic fixture, never product source.
        bundle["manifest"]["python_files"].pop(path)
    inputs = parsed(bundle).revalidated()
    gate = preflight(bundle)
    gate.check(full=True)
    assert inputs.original_stage_a_manifest.python_files == original.python_files
    assert inputs.derived_fit_manifest().python_files == bundle["manifest"]["python_files"]
    assert gate.preservation()["frozen_hashes"] == {
        **bundle["manifest"]["python_files"], **bundle["manifest"]["engine_files"]}
    provenance = inputs.provenance()
    assert provenance["original_stage_a_python_files_sha256"] == digest(original.python_files)
    assert provenance["current_python_files_sha256"] == digest(bundle["manifest"]["python_files"])
    assert provenance["current_python_files_sha256"] != provenance["original_stage_a_python_files_sha256"]
    assert inputs.derived_scope()["executor_sha256"] == digest(bundle["manifest"]["python_files"])
    recomputed = audit_fit(json.loads(before[bundle["manifest"]["stage_a_record"]["path"]]), original)
    assert recomputed == json.loads(inputs._stage_a_audit_json)
    assert bundle["artifacts"] == before
    assert all((bundle["root"] / name).read_bytes() == raw for name, raw in before.items())


@pytest.mark.parametrize("fault", ("historical_pin", "false_pin", "omitted_pin", "mutated_bytes", "deleted_bytes", "extra_file"))
def test_current_source_faults_are_checked_against_execution_not_historical_files(bundle: dict[str, Any], fault: str) -> None:
    path = next(iter(bundle["original"].python_files))
    if fault == "historical_pin": bundle["manifest"]["python_files"][path] = bundle["original"].python_files[path]
    elif fault == "false_pin": bundle["manifest"]["python_files"][path] = "0"*64
    elif fault == "omitted_pin": bundle["manifest"]["python_files"].pop(path)
    elif fault == "mutated_bytes": (bundle["root"] / path).write_bytes(b"unreviewed change")
    elif fault == "deleted_bytes": (bundle["root"] / path).unlink()
    elif fault == "extra_file": (bundle["root"] / "src/unlisted.py").write_bytes(b"unlisted")
    before = deepcopy(bundle["artifacts"])
    # Parsing verifies historical evidence; only current-file preflight can tell
    # whether a syntactically valid execution map actually matches this machine.
    inputs = parsed(bundle)
    assert inputs.original_stage_a_manifest == bundle["original"]
    backend = Mock(side_effect=AssertionError("zero backend constructions"))
    with pytest.raises((ValueError, OSError)):
        preflight(bundle)
        backend()
    backend.assert_not_called()
    assert bundle["artifacts"] == before


@pytest.mark.parametrize("fault", ("manifest_source", "record_source", "record_executor", "audit_rules", "review_manifest"))
def test_execution_pins_cannot_replace_historical_audit_context(bundle: dict[str, Any], fault: str) -> None:
    path = next(iter(bundle["original"].python_files))
    newer = bundle["manifest"]["python_files"][path]
    assert newer != bundle["original"].python_files[path]
    role = {"manifest_source": "stage_a_manifest", "record_source": "stage_a_record",
            "record_executor": "stage_a_record", "audit_rules": "stage_a_audit", "review_manifest": "stage_a_review"}[fault]
    def change(raw: Any) -> None:
        if fault == "manifest_source": raw["python_files"][path] = newer
        elif fault == "record_source": raw["preservation"]["frozen_hashes"][path] = newer
        elif fault == "record_executor": raw["manifest_sha256"] = digest(bundle["manifest"])
        elif fault == "audit_rules": raw["rules_sha256"] = digest(bundle["manifest"])
        elif fault == "review_manifest": raw["manifest_sha256"] = digest(bundle["manifest"])
    change_artifact(bundle, role, change)
    with pytest.raises(ValueError):
        parsed(bundle)
