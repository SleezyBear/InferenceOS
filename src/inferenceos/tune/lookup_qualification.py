"""D1 development DRAFT bindings, never a tuning-admission certificate.

Pure supplied-data validation/comparison. No machine scan, file pin collection,
performance inference, store, registry, dispatcher or accepted-certificate reader.
"""
from __future__ import annotations

from decimal import Decimal
import json
import math
from pathlib import PureWindowsPath
from typing import Annotated, Any, Final, Literal, Self, TypeVar

from pydantic import BaseModel, BeforeValidator, Field, StringConstraints, field_validator, model_validator

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.tune.lookup_audit import REQUIRED_ENGINE_FILES
from inferenceos.tune.lookup_corpus import CORPUS_VERSION, ORACLE_VERSION, FrozenInput, TaskCorpus, digest

DRAFT_VERSION: Final = "lookup-qualification-draft-1"
SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, StringConstraints(min_length=1, max_length=256)]
CostKind = Literal["identity", "oracle", "control", "lookup", "guard", "final", "cleanup"]
COST_COMPONENTS: tuple[CostKind, ...] = ("identity", "oracle", "control", "lookup", "guard", "final", "cleanup")


def _lossless_float(value: object) -> object:
    # Strict Pydantic floats still accept integers. Check before that conversion,
    # without narrowing integer-valued identity fields elsewhere in the binding.
    if type(value) not in (int, float):
        raise ValueError("an explicit JSON integer or float is required")
    if type(value) is int:
        try:
            converted = float(value)
        except OverflowError as exc:
            raise ValueError("integer cannot be represented exactly as a float") from exc
        if int(converted) != value:
            raise ValueError("integer cannot be represented exactly as a float")
    return value


LosslessFloat = Annotated[float, BeforeValidator(_lossless_float)]
Model = TypeVar("Model", bound=BaseModel)


class ConfigurationSnapshot(RuntimeConfiguration):
    model_config = FrozenInput.model_config

    @model_validator(mode="after")
    def explicit_fields(self) -> Self:
        _require_configuration_fields(self)
        return self


def _require_configuration_fields(value: BaseModel) -> None:
    if value.model_fields_set != set(RuntimeConfiguration.model_fields):
        raise ValueError("all configuration fields, including explicit null defaults, are required")


def _supplied_data(value: Any) -> Any:
    """Read original values/provenance before a serializer can fill or coerce them."""
    if isinstance(value, RuntimeConfiguration):
        _require_configuration_fields(value)
    if isinstance(value, BaseModel):
        fields = type(value).model_fields
        if any(not hasattr(value, name) for name in fields):
            raise ValueError("incomplete constructed model: " + type(value).__name__)
        # Unchecked copies can also attach forbidden fields. Retain them so that
        # validation refuses them, instead of silently dropping the assertion.
        names = set(fields) | set(value.__dict__) | set(value.model_extra or {})
        if names == set(RuntimeConfiguration.model_fields):
            # Unchecked input can substitute a foreign model with this exact
            # shape. Its defaulted fields are no more explicit than ours.
            _require_configuration_fields(value)
        return {name: _supplied_data(getattr(value, name)) for name in names}
    if isinstance(value, dict):
        return {key: _supplied_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_supplied_data(item) for item in value]
    return value


def _revalidate(model: type[Model], value: Model) -> Model:
    # Use JSON validation for its tuple input rules, but retain original integers
    # and check nested explicit-field provenance before producing that JSON.
    return model.model_validate_json(json.dumps(_supplied_data(value), allow_nan=False))


class FilePin(FrozenInput):
    path: str
    sha256: SHA256

    @field_validator("path")
    @classmethod
    def canonical_local_path(cls, value: str) -> str:
        parts = value.split("/")
        if (not PureWindowsPath(value).is_absolute() or len(PureWindowsPath(value).drive) != 2
                or "\\" in value or any(p in ("", ".", "..") for p in parts)
                or ":" in value[2:] or any(ord(c) < 32 for c in value)):
            raise ValueError("an explicit local drive path with forward slashes and no dot/empty segments is required")
        return value


class ToolchainBinding(FrozenInput):
    executable: FilePin
    files: tuple[FilePin, ...] = Field(min_length=5, max_length=128)
    ambient_config_absent: bool
    environment_sha256: SHA256

    @field_validator("files")
    @classmethod
    def ordered(cls, value: tuple[FilePin, ...]) -> tuple[FilePin, ...]:
        names = [PureWindowsPath(p.path).name.lower() for p in value]
        if len(set(names)) != len(names) or not REQUIRED_ENGINE_FILES.issubset(names):
            raise ValueError("all required engine components with unambiguous names are required")
        return tuple(sorted(value, key=lambda p: p.path))

    @model_validator(mode="after")
    def executable_is_bound(self) -> Self:
        if PureWindowsPath(self.executable.path).name.lower() != "llama-server.exe" or self.executable not in self.files:
            raise ValueError("the actual selected executable must be in the pinned file set")
        return self


class MachineBinding(FrozenInput):
    version: Literal["lookup-machine-local-1"] = "lookup-machine-local-1"
    hostname: Name
    os_build: Name
    architecture: Name
    cpu_name: Name
    cpu_vendor: Name
    physical_cores: int = Field(ge=1)
    logical_cores: int = Field(ge=1)
    instruction_sets: tuple[Name, ...] = Field(min_length=1, max_length=128)
    ram_bytes: int = Field(ge=1)
    gpu_name: Name
    gpu_vendor: Name
    gpu_vram_bytes: int = Field(ge=1)
    driver_version: Name
    pci_device_id: Name | None
    usable_gpu_count: int = Field(ge=1)
    backend: Literal["vulkan"] = "vulkan"
    device_name: Name
    device_index: int = Field(ge=0)
    power_plan: Name
    power_source: Literal["ac"] = "ac"

    @field_validator("instruction_sets")
    @classmethod
    def instructions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate instruction set")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def known_machine(self) -> Self:
        for value in self.model_dump().values():
            values = value if isinstance(value, tuple) else (value,)
            for item in values:
                if isinstance(item, str) and (not item.strip() or item != item.strip()
                                             or item.lower() in ("unknown", "unknown id", "unconfirmed", "none")):
                    raise ValueError("unknown machine metadata is not a verified identity")
        if self.logical_cores < self.physical_cores or self.usable_gpu_count != 1:
            raise ValueError("a consistent CPU topology and one unambiguous usable GPU are required")
        return self


class RequestBinding(FrozenInput):
    version: Literal["lookup-native-first-request-1"] = "lookup-native-first-request-1"
    wire_format: Literal["native-completion"] = "native-completion"
    lifecycle: Literal["fresh-load-one-request-per-arm"] = "fresh-load-one-request-per-arm"
    algorithm: Literal["ngram-simple"] = "ngram-simple"
    lookup_length: int = Field(default=12, ge=1)
    draft_length: int = Field(default=16, ge=1)
    seed: int = 42
    temperature: LosslessFloat = 0.0
    max_output_tokens: int = Field(ge=1, le=4096)
    context_per_slot: int = Field(ge=1)
    server_slots: int = Field(ge=1, le=512)
    log_verbosity: int = 4
    cache_prompt: bool = False
    cache_reuse: int | None = None
    swa_full: bool = False
    ignore_eos: bool = False
    offline: bool = True
    streaming: bool = True
    generated_ids: bool = True
    in_flight_requests: int = 1

    @model_validator(mode="after")
    def qualified_scope(self) -> Self:
        if (self.lookup_length != 12 or self.draft_length != 16 or self.seed != 42
                or self.temperature != 0 or self.log_verbosity != 4 or self.cache_prompt
                or self.cache_reuse is not None or self.swa_full or self.ignore_eos
                or not self.offline or not self.streaming or not self.generated_ids
                or self.in_flight_requests != 1):
            raise ValueError("this draft format only describes the explicit proposed fresh 12/16 native scope")
        return self


class ScopeBinding(FrozenInput):
    model_sha256: SHA256
    tokenizer_sha256: SHA256
    corpus_version: Literal["lookup-task-corpus-1"] = CORPUS_VERSION
    oracle_version: Literal["lookup-task-oracle-1"] = ORACLE_VERSION
    corpus_sha256: SHA256
    machine: MachineBinding
    toolchain: ToolchainBinding
    requested: ConfigurationSnapshot
    effective: ConfigurationSnapshot
    unknown_effects: tuple[str, ...]
    request: RequestBinding

    @field_validator("unknown_effects")
    @classmethod
    def known_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or set(value) - set(RuntimeConfiguration.model_fields):
            raise ValueError("unknown/duplicate configuration field name")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def context_is_consistent(self) -> Self:
        if self.requested.context_length != self.request.context_per_slot:
            raise ValueError("requested context must equal the bound per-slot context")
        if self.effective.context_length is not None and self.effective.context_length != self.request.context_per_slot:
            raise ValueError("observed context contradicts the bound per-slot context")
        return self


class CostSamples(FrozenInput):
    component: CostKind
    case_sha256: SHA256
    source_sha256: SHA256
    seconds: tuple[Annotated[LosslessFloat, Field(gt=0)], ...] = Field(min_length=1, max_length=4096)


class QualificationDraft(FrozenInput):
    version: Literal["lookup-qualification-draft-1"] = DRAFT_VERSION
    state: Literal["development-draft"] = "development-draft"
    independent_review: Literal["pending"] = "pending"
    simulated: bool
    revoked: bool = False
    binding: ScopeBinding
    protocol_sha256: SHA256 | None = None
    record_sha256: SHA256 | None = None
    audit_sha256: SHA256 | None = None
    costs: tuple[CostSamples, ...] = Field(default=(), max_length=4096)

    @property
    def sha256(self) -> str:
        return digest(_revalidate(QualificationDraft, self).model_dump(mode="json"))


class DraftMatch(FrozenInput):
    same_scope: bool
    supplied_effects_complete: bool
    has_each_cost_component: bool
    problems: tuple[str, ...]
    tuning_allowed: Literal[False] = False
    deployment_allowed: Literal[False] = False
    status: Literal["development-draft-only"] = "development-draft-only"


def _differences(left: Any, right: Any, path: str = "binding") -> list[str]:
    if type(left) is dict and type(right) is dict:
        result = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append(f"{path}.{key}")
            else:
                result.extend(_differences(left[key], right[key], f"{path}.{key}"))
        return result
    if type(left) is float and type(right) is float:
        return [] if left.hex() == right.hex() else [path]
    return [] if type(left) is type(right) and left == right else [path]


def match_draft(draft: QualificationDraft, observed: ScopeBinding, corpus: TaskCorpus) -> DraftMatch:
    """Compare supplied facts, never collect/attest them or grant execution permission."""
    # model_copy/model_construct can bypass Pydantic validation; do not trust them.
    draft = _revalidate(QualificationDraft, draft)
    observed = _revalidate(ScopeBinding, observed)
    corpus = _revalidate(TaskCorpus, corpus)
    mismatches = _differences(draft.binding.model_dump(mode="json"), observed.model_dump(mode="json"))
    if draft.binding.corpus_sha256 != corpus.sha256:
        mismatches.append("binding.corpus_sha256 differs from the supplied full corpus")
    problems = [f"scope differs: {name}" for name in mismatches]
    required = ("threads", "threads_batch", "batch_size", "ubatch_size", "gpu_offload_layers",
                "use_mmap", "context_length", "flash_attention", "device", "kv_cache_type_k", "kv_cache_type_v")
    missing = sorted(set(observed.unknown_effects) | {name for name in required if getattr(observed.effective, name) is None})
    if missing:
        problems.append("effective settings unconfirmed: " + ", ".join(missing))
    contradictory = []
    kv_fields = {"kv_cache_type", "kv_cache_type_k", "kv_cache_type_v"}
    for name in RuntimeConfiguration.model_fields:
        requested = getattr(observed.requested, name)
        if name not in kv_fields and requested is not None and requested != getattr(observed.effective, name):
            contradictory.append(name)
    for name, requested in zip(("kv_cache_type_k", "kv_cache_type_v"), observed.requested.effective_kv_types()):
        if requested is not None and requested != getattr(observed.effective, name):
            contradictory.append(name)
    if contradictory:
        problems.append("requested settings contradicted: " + ", ".join(contradictory))
    if not observed.toolchain.ambient_config_absent:
        problems.append("ambient engine configuration absence unconfirmed")
    case_hashes = {c.sha256 for c in corpus.cases}
    cost_valid = all(c.case_sha256 in case_hashes for c in draft.costs)
    kinds = {c.component for c in draft.costs}
    if not cost_valid:
        problems.append("cost sample belongs to a different task")
    missing_costs = sorted(set(COST_COMPONENTS) - kinds)
    if missing_costs:
        problems.append("cost components missing: " + ", ".join(missing_costs))
    if draft.revoked:
        problems.append("development draft revoked")
    problems.append("development draft is not independently accepted complete-task performance/cost qualification")
    return DraftMatch(same_scope=not mismatches, supplied_effects_complete=not missing and not contradictory and observed.toolchain.ambient_config_absent,
                      has_each_cost_component=cost_valid and not missing_costs, problems=tuple(problems))


def parse_draft(text: str) -> QualificationDraft:
    """Refuse duplicate keys, nonfinite/rounded JSON numbers and unknown fields."""
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def number(value: str) -> float:
        result = float(value)
        if not math.isfinite(result) or Decimal(repr(result)) != Decimal(value):
            raise ValueError("JSON number cannot be represented exactly by this format")
        return result

    def nonfinite(value: str) -> object:
        raise ValueError("nonfinite JSON value: " + value)

    parsed = json.loads(text, object_pairs_hook=pairs, parse_float=number, parse_constant=nonfinite)
    return QualificationDraft.model_validate_json(json.dumps(parsed, allow_nan=False))
