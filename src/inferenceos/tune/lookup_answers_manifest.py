"""Prospective Stage-B provenance and read-only preflight, never run permission.

The accepted Stage-A manifest is audited with its ORIGINAL source pins.  A
separate adapter containing the current Stage-B sources is used only by the
existing read-only native preflight.  Neither object rewrites Stage-A evidence.
All parser inputs are supplied bytes; this module never constructs an engine.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from inferenceos.tune.lookup_answers import VERSION as CORE_VERSION, plan, suites, scope_problems
from inferenceos.tune.lookup_corpus import FrozenInput, build_corpus, digest
from inferenceos.tune.lookup_fit import (FitManifest, FitModel, FitRefused, PROTOCOL, audit_fit,
                                      parse_manifest, relative_path, strict_json, validated_manifest)
from inferenceos.tune.lookup_fit_observers import machine_from_raw
from inferenceos.tune.lookup_qualification import (ConfigurationSnapshot, MachineBinding, SHA256,
                                                _supplied_data)
from inferenceos.tune.lookup_role_cost import VERSION as COST_VERSION

VERSION = "lookup-answer-manifest-1"
ACCEPTED_PROTOCOL_SHA256 = "e87845654c8d87f05667ef1e204292903984cee51b0c580c0958914e25434076"
ACCEPTED_PROPOSAL_SHA256 = "1a5be53a11340d6f3c2cd050a26521322d6a76bdd600d8ffe3646f4784db8e9d"
GOLDEN_PLAN_SHA256 = "920c394cb9050a7c5cd8dc0b73ba95c4cae64f962615e486aea06b5d81d66543"
GOLDEN_SUITES_SHA256 = "8eb43663f6290ef7c40d3c21b9cb574f7c00310f0a7877a2e16bf38219bbb82b"
_ARTIFACT_ROLES = ("proposal", "protocol_document", "stage_a_manifest", "stage_a_record",
                   "stage_a_audit", "stage_a_review")
_RESERVED = {"con", "prn", "aux", "nul", *("com"+str(i) for i in range(1, 10)),
             *("lpt"+str(i) for i in range(1, 10))}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_relative(value: str) -> None:
    relative_path(value)
    if (any(ord(c) < 32 or c in '<>"|?*' for c in value)
            or any(part.endswith((" ", ".")) or part.split(".", 1)[0].casefold() in _RESERVED
                   for part in value.split("/"))):
        raise FitRefused("unambiguous canonical repository-relative path required")


class ArtifactPin(FrozenInput):
    path: str
    sha256: SHA256

    @model_validator(mode="after")
    def canonical(self) -> Self:
        safe_relative(self.path)
        return self


class AnswerManifest(FrozenInput):
    # Every field is required. In particular no parser inserts absent permissions,
    # model defaults, effective settings or evidence receipt references.
    version: Literal["lookup-answer-manifest-1"]
    stage: Literal["B-complete-answers"]
    protocol: Literal["lookup-qualification-protocol-2"]
    proposal: ArtifactPin
    protocol_document: ArtifactPin
    stage_a_manifest: ArtifactPin
    stage_a_record: ArtifactPin
    stage_a_audit: ArtifactPin
    stage_a_review: ArtifactPin
    answer_core_version: str
    cost_schema_version: str
    corpus_sha256: SHA256
    plan_sha256: SHA256
    suites_sha256: SHA256
    models: tuple[FitModel, FitModel]
    effective_bases: dict[Literal["llama3b", "gemma1b"], ConfigurationSnapshot]
    machine: MachineBinding
    python_files: dict[str, SHA256] = Field(min_length=1)
    engine_files: dict[str, SHA256] = Field(min_length=5, max_length=128)
    hardware_authorized: Literal[False]
    automatic_admission: Literal[False]
    deployment_allowed: Literal[False]

    @field_validator("hardware_authorized", "automatic_admission", "deployment_allowed", mode="before")
    @classmethod
    def literal_denials(cls, value: Any) -> Any:
        # Literal[False] alone follows Python equality and can accept numeric 0.
        # These declarations are explicit JSON booleans, never numeric aliases.
        if value is not False:
            raise FitRefused("explicit false permission declarations required")
        return value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (self.answer_core_version != CORE_VERSION or self.cost_schema_version != COST_VERSION
                or self.corpus_sha256 != build_corpus().sha256
                or self.plan_sha256 != GOLDEN_PLAN_SHA256 or digest(plan()) != GOLDEN_PLAN_SHA256
                or self.suites_sha256 != GOLDEN_SUITES_SHA256 or digest(suites()) != GOLDEN_SUITES_SHA256):
            raise FitRefused("unchanged complete D1 corpus, fixed plan/suites and current evidence versions required")
        if (self.protocol_document.sha256 != ACCEPTED_PROTOCOL_SHA256
                or self.proposal.sha256 != ACCEPTED_PROPOSAL_SHA256):
            raise FitRefused("unchanged accepted protocol-2/proposal bytes required")
        if tuple(m.label for m in self.models) != ("llama3b", "gemma1b"):
            raise FitRefused("original fixed two-model order required")
        if set(self.effective_bases) != {"llama3b", "gemma1b"}:
            raise FitRefused("both complete effective bases required")
        pins = [getattr(self, name) for name in _ARTIFACT_ROLES]
        names = [p.path.casefold() for p in pins]
        if len(set(names)) != len(names):
            raise FitRefused("distinct provenance artifact paths required")
        all_names = [*self.python_files, *self.engine_files]
        if len(set(p.casefold() for p in all_names)) != len(all_names):
            raise FitRefused("case-colliding frozen input paths")
        if {p.path.casefold() for p in pins} & {p.casefold() for p in all_names}:
            raise FitRefused("provenance artifacts cannot alias source/engine pins")
        for name in all_names:
            safe_relative(name)
        if any(name.split("/", 1)[0] not in ("src", "scripts", "tests") or not name.endswith(".py")
               for name in self.python_files):
            raise FitRefused("only complete src/scripts/tests Python source pins are supported")
        # Reuse all existing fixed model/toolchain/path checks without claiming that
        # this current-source adapter was the historical Stage-A manifest.
        self._fit_adapter()
        return self

    def _fit_adapter(self) -> FitManifest:
        return FitManifest.model_validate_json(json.dumps(dict(
            proposal_sha256=self.proposal.sha256,
            protocol_document_sha256=self.protocol_document.sha256,
            corpus_sha256=self.corpus_sha256,
            models=_supplied_data(self.models), python_files=self.python_files,
            engine_files=self.engine_files), allow_nan=False))


def validated_answer_manifest(value: AnswerManifest) -> AnswerManifest:
    """Check explicit nested provenance BEFORE any serializer can fill defaults."""
    return AnswerManifest.model_validate_json(json.dumps(_supplied_data(value), allow_nan=False))


@dataclass(frozen=True)
class AnswerInputs:
    manifest: AnswerManifest
    original_stage_a_manifest: FitManifest
    _raw_manifest: bytes
    _artifacts: tuple[tuple[str, bytes], ...]
    _scope_json: bytes
    _stage_a_audit_json: bytes

    def derived_scope(self) -> dict[str, Any]:
        """A fresh copy, derived only from the independently accepted raw A run."""
        value = strict_json(self._scope_json)
        if not isinstance(value, dict):
            raise FitRefused("invalid derived Stage-B scope")
        return value

    def derived_fit_manifest(self) -> FitManifest:
        """Current-source B preflight adapter; NEVER the original A manifest."""
        return validated_answer_manifest(self.manifest)._fit_adapter()

    def artifact_bytes(self) -> dict[str, bytes]:
        return dict(self._artifacts)

    def revalidated(self) -> AnswerInputs:
        # A caller may have built/unchecked-copied this dataclass or nested models.
        # Inspect the supplied field provenance before returning to byte authority;
        # never silently repair an omitted/defaulted configuration along the way.
        supplied = validated_answer_manifest(self.manifest)
        original = validated_manifest(self.original_stage_a_manifest)
        if (digest(supplied.model_dump(mode="json")) != digest(strict_json(self._raw_manifest))
                or len(self._artifacts) != len(self.artifact_bytes())
                or digest(original.model_dump(mode="json")) != digest(strict_json(
                    self.artifact_bytes()[supplied.stage_a_manifest.path]))):
            raise FitRefused("supplied provenance object differs from its original bytes")
        result = parse_answer_manifest(self._raw_manifest, self.artifact_bytes())
        if (digest(self.derived_scope()) != digest(result.derived_scope())
                or digest(strict_json(self._stage_a_audit_json)) != digest(strict_json(result._stage_a_audit_json))):
            raise FitRefused("supplied derived scope/audit differs from original evidence")
        return result

    def provenance(self) -> dict[str, Any]:
        manifest = validated_answer_manifest(self.manifest)
        return dict(version=VERSION, manifest_sha256=sha256(self._raw_manifest),
                    artifacts={name: getattr(manifest, name).model_dump(mode="json") for name in _ARTIFACT_ROLES},
                    original_stage_a_manifest_sha256=manifest.stage_a_manifest.sha256,
                    original_stage_a_python_files_sha256=digest(self.original_stage_a_manifest.python_files),
                    current_python_files_sha256=digest(manifest.python_files),
                    scope_sha256=digest(self.derived_scope()),
                    stage_a_recomputed_audit=strict_json(self._stage_a_audit_json),
                    hardware_authorized=False, automatic_admission=False, deployment_allowed=False)


def parse_answer_manifest(text: str | bytes, artifacts: Mapping[str, bytes]) -> AnswerInputs:
    """Bind supplied bytes, recompute original A evidence, derive the exact B scope.

    The review artifact is an explicitly pinned external acceptance statement, not
    a signature, owner authorization or a certificate minted by this parser.
    """
    raw = text.encode("utf-8") if isinstance(text, str) else text
    if not isinstance(raw, bytes) or not isinstance(artifacts, Mapping):
        raise FitRefused("manifest and supplied byte artifacts required")
    manifest = AnswerManifest.model_validate_json(json.dumps(strict_json(raw), allow_nan=False))
    required = {getattr(manifest, role).path for role in _ARTIFACT_ROLES}
    if set(artifacts) != required:
        raise FitRefused("exact six original provenance artifacts required")
    for role in _ARTIFACT_ROLES:
        pin = getattr(manifest, role)
        content = artifacts[pin.path]
        if not isinstance(content, bytes) or sha256(content) != pin.sha256:
            raise FitRefused("provenance artifact differs: " + role)
    proposal = strict_json(artifacts[manifest.proposal.path])
    if not isinstance(proposal, dict):
        raise FitRefused("original proposal object required")
    original = parse_manifest(artifacts[manifest.stage_a_manifest.path], proposal, manifest.proposal.sha256)
    if (original.protocol_document_sha256 != manifest.protocol_document.sha256
            or original.corpus_sha256 != manifest.corpus_sha256
            or _supplied_data(original.models) != _supplied_data(manifest.models)
            or original.engine_files != manifest.engine_files):
        raise FitRefused("Stage-B scope differs from original accepted Stage-A inputs")
    # Historical source hashes describe how A was collected, not which bytes B
    # must execute. audit_fit below checks the preserved record against ORIGINAL
    # pins. AnswerPreflight independently checks the complete CURRENT B map on
    # disk, including changed/renamed/removed sources, before engine construction
    # and subsequent loads. Cross-map equality would reject reviewed repairs or
    # force stale execution pins without strengthening either context's checks.
    record = strict_json(artifacts[manifest.stage_a_record.path])
    stored_audit = strict_json(artifacts[manifest.stage_a_audit.path])
    review = strict_json(artifacts[manifest.stage_a_review.path])
    if not all(isinstance(v, dict) for v in (record, stored_audit, review)):
        raise FitRefused("original record, audit and independent review objects required")
    if record.get("simulated") is not False or record.get("status") != "done":
        raise FitRefused("original complete Stage-A observations cannot be reproduced")
    recomputed = audit_fit(record, original)
    if (recomputed.get("observations_complete") is not True or recomputed.get("admission_power_readings") != 1
            or recomputed.get("load_boundary_power_readings") != 8
            or recomputed.get("inventory_power_readings") != 8):
        raise FitRefused("original complete Stage-A observations cannot be reproduced")
    expected_audit = dict(recomputed, source_sha256=manifest.stage_a_record.sha256,
                          rules_sha256=manifest.stage_a_manifest.sha256,
                          proposal_sha256=manifest.proposal.sha256)
    if digest(stored_audit) != digest(expected_audit):
        raise FitRefused("original Stage-A public audit differs from recomputation")
    if (review.get("verdict") != "accept Stage-A results" or review.get("findings") != []
            or review.get("run_id") != record.get("run_id")
            or any(review.get(name) != expected for name, expected in (
                ("record_sha256", manifest.stage_a_record.sha256), ("audit_sha256", manifest.stage_a_audit.sha256),
                ("manifest_sha256", manifest.stage_a_manifest.sha256), ("proposal_sha256", manifest.proposal.sha256)))
            or review.get("fresh_offline_CLI_exit") != 0
            or type(review.get("fresh_offline_CLI_exit")) is not int
            or review.get("fresh_offline_CLI_audit_byte_identical") is not True
            or review.get("pure_recomputation_exact") is not True):
        raise FitRefused("explicit independent acceptance of these exact original A bytes required")
    scope: dict[str, Any] = dict(models={}, machine=None, engine_files=deepcopy(original.engine_files),
                                 executor_sha256=digest(manifest.python_files))
    for model in original.models:
        row = next(r for r in record["loads"] if r["model"] == model.label)
        ready = row["snapshots"][0]
        effective = ready["effects"]["configuration"]
        if effective != manifest.effective_bases[model.label].model_dump(mode="json"):
            raise FitRefused("effective base differs from original raw Stage-A evidence")
        machine = machine_from_raw(ready["machine"], effective["device"])
        if machine != manifest.machine.model_dump(mode="json"):
            raise FitRefused("machine scope differs from original raw Stage-A evidence")
        scope["machine"] = machine
        scope["models"][model.label] = dict(requested=model.configuration.model_dump(mode="json"),
                                             effective=effective, fit_tokens=row["tokens"], model_sha256=model.sha256)
    problems = scope_problems(scope)
    if problems:
        raise FitRefused("derived Stage-B scope invalid: " + "; ".join(problems))
    return AnswerInputs(manifest, original, raw, tuple((name, artifacts[name]) for name in sorted(required)),
                        json.dumps(scope, allow_nan=False).encode("utf-8"),
                        json.dumps(recomputed, allow_nan=False).encode("utf-8"))


def _local_path(root: Path, relative: str) -> Path:
    safe_relative(relative)
    path = root / relative
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise FitRefused("frozen provenance path escapes repository")
    for item in (path, *path.parents):
        if item.is_symlink() or getattr(item, "is_junction", lambda: False)():
            raise FitRefused("linked provenance/root paths are unsupported")
    return resolved


class AnswerPreflight:
    """The existing native preflight plus all original acceptance-artifact pins.

    ``preflight_factory`` must be the existing Stage-A Preflight (or a software
    double).  It is invoked only after byte and scope revalidation, and constructs
    no backend.  Its normal checks enforce exact current sources, executable,
    model bytes, saved setup identity and the canonical data/environment boundary.
    """
    def __init__(self, root: Path, settings: Any, inputs: AnswerInputs, rules: Path,
                 rules_sha: str, *, preflight_factory: Callable[..., Any]) -> None:
        from inferenceos.tune.lookup_fit_io import bounded_file_bytes
        self.root = root
        self.answer_inputs = inputs.revalidated()
        self.manifest = self.answer_inputs.derived_fit_manifest()
        self.settings = settings
        rules_path = rules if rules.is_absolute() else root / rules
        try:
            self._rules_relative = rules_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise FitRefused("Stage-B rules must be inside the repository") from exc
        self._rules = _local_path(root, self._rules_relative)
        self._rules_sha = rules_sha
        self._pins = [(getattr(self.answer_inputs.manifest, role).path,
                       getattr(self.answer_inputs.manifest, role).sha256) for role in _ARTIFACT_ROLES]
        if (not re.fullmatch(r"[0-9a-f]{64}", rules_sha)
                or sha256(bounded_file_bytes(self._rules, maximum_bytes=64*1024*1024)) != rules_sha
                or rules_sha != sha256(inputs._raw_manifest)):
            raise FitRefused("actual current Stage-B rules bytes differ")
        # Check every historical artifact before the factory may inspect model or
        # toolchain resources.  Revalidation uses the ORIGINAL A source manifest.
        self._check_artifacts()
        proposal = self.answer_inputs.manifest.proposal
        self._delegate = preflight_factory(root, settings, self.manifest, self._rules, rules_sha,
                                            _local_path(root, proposal.path), proposal.sha256)
        self.check(full=True)

    @property
    def selected(self) -> Any:
        return self._delegate.selected

    @property
    def inputs(self) -> Any:
        return self._delegate.inputs

    @property
    def models(self) -> Any:
        return self._delegate.models

    def _check_artifacts(self) -> None:
        from inferenceos.tune.lookup_fit_io import bounded_file_bytes
        for relative, expected in self._pins:
            if sha256(bounded_file_bytes(_local_path(self.root, relative), maximum_bytes=64*1024*1024)) != expected:
                raise FitRefused("governing original provenance changed")
        rules = _local_path(self.root, self._rules_relative)
        if rules != self._rules or sha256(bounded_file_bytes(rules, maximum_bytes=64*1024*1024)) != self._rules_sha:
            raise FitRefused("Stage-B governing rules changed")

    def check(self, *, full: bool = False) -> None:
        self._check_artifacts()
        self._delegate.check(full=full)

    def preservation(self) -> dict[str, Any]:
        self.check(full=True)
        return self._delegate.preservation()  # type: ignore[no-any-return]
