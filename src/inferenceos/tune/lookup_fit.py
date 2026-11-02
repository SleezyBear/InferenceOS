"""D2 Stage A: four fit-only loads. No answers, qualification or tuning permission.

The collector's only engine operations are start, observe, tokenize and release.
The auditor recomputes the full population, observations and capacity checks.
"""
from __future__ import annotations

from copy import deepcopy
import base64
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Literal, Protocol, Self

from pydantic import Field, model_validator

from inferenceos.tune.lookup_audit import engine_identity_problems, REQUIRED_ENGINE_FILES
from inferenceos.tune.lookup_corpus import FrozenInput, build_corpus, digest
from inferenceos.tune.lookup_decoding import activation_problems
from inferenceos.tune.lookup_qualification import ConfigurationSnapshot, MachineBinding, SHA256, _supplied_data

VERSION = "lookup-fit-2"
PROTOCOL: Literal["lookup-qualification-protocol-2"] = "lookup-qualification-protocol-2"
INITIALIZATION_POLICY: dict[str, Any] = {
    "client_warmup_priming_discovery_allowed": False,
    "engine_startup": "pinned-default-internal-empty-run",
    "retain_complete_startup_log": True,
    "stage_A_completion_or_chat_requests": 0,
    "stage_B_cycle_includes_startup": True,
    "stage_B_subtract_startup": False,
}
STAGE_A = (("llama3b", "control"), ("llama3b", "lookup"),
           ("gemma1b", "lookup"), ("gemma1b", "control"))
REQUIRED_EFFECTS = ("threads", "threads_batch", "batch_size", "ubatch_size", "gpu_offload_layers",
                    "context_length", "use_mmap", "flash_attention", "device", "kv_cache_type_k", "kv_cache_type_v")


class FitRefused(ValueError):
    pass


def strict_json(text: str | bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key.casefold() in {k.casefold() for k in result}:
                raise FitRefused("duplicate/case-colliding JSON key")
            result[key] = value
        return result

    def invalid(value: str) -> Any:
        raise FitRefused("nonfinite JSON number: " + value)

    def exact_float(text: str) -> float:
        value = float(text)
        if not math.isfinite(value) or Decimal(repr(value)) != Decimal(text):
            raise FitRefused('JSON number cannot be retained losslessly')
        return value

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid,parse_float=exact_float)


def relative_path(value: str) -> None:
    if (not value or value.startswith(("/", "\\")) or "\\" in value or ":" in value
            or any(p in ("", ".", "..") for p in value.split("/"))):
        raise FitRefused("canonical relative path required")


class FitModel(FrozenInput):
    label: Literal["llama3b", "gemma1b"]
    path: str
    sha256: SHA256
    configuration: ConfigurationSnapshot
    configuration_sha256: SHA256
    saved_setup_source: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        relative_path(self.path)
        if self.configuration.context_length != 4096 or digest(self.configuration.model_dump(mode="json")) != self.configuration_sha256:
            raise FitRefused("explicit 4096 configuration digest required")
        return self


class FitManifest(FrozenInput):
    version: Literal["lookup-fit-manifest-2"] = "lookup-fit-manifest-2"
    stage: Literal["A-fit-only"] = "A-fit-only"
    protocol: Literal["lookup-qualification-protocol-2"] = PROTOCOL
    proposal_sha256: SHA256
    protocol_document_sha256: SHA256
    corpus_sha256: SHA256
    data_root: Literal["data"] = "data"
    models: tuple[FitModel, FitModel]
    python_files: dict[str, SHA256] = Field(min_length=1)
    engine_files: dict[str, SHA256] = Field(min_length=5, max_length=128)
    automatic_admission: Literal[False] = False
    generation_allowed: Literal[False] = False

    @model_validator(mode="after")
    def population(self) -> Self:
        if tuple(m.label for m in self.models) != ("llama3b", "gemma1b") or len({m.path.casefold() for m in self.models}) != 2:
            raise FitRefused("fixed two-model order/identities required")
        if self.corpus_sha256 != build_corpus().sha256:
            raise FitRefused("accepted complete D1 corpus required")
        for group in (self.python_files, self.engine_files):
            if len({p.casefold() for p in group}) != len(group):
                raise FitRefused("case-colliding frozen file paths")
            for path in group:
                relative_path(path)
        names = [p.rsplit('/',1)[-1].lower() for p in self.engine_files]
        if len(set(names)) != len(names) or not REQUIRED_ENGINE_FILES.issubset(names):
            raise FitRefused('unambiguous required engine component pins required')
        return self


def parse_manifest(text: str | bytes, proposal: dict[str, Any], proposal_sha256: str) -> FitManifest:
    manifest = FitManifest.model_validate_json(json.dumps(strict_json(text), allow_nan=False))
    if (manifest.proposal_sha256 != proposal_sha256 or proposal.get("protocol") != PROTOCOL
            or proposal.get("executable") is not False or proposal.get("hardware_authorized") is not False
            or manifest.protocol_document_sha256 != proposal.get("protocol_document_sha256")
            or manifest.corpus_sha256 != proposal.get("corpus_sha256")
            or manifest.engine_files != proposal.get("proposed_engine_files")
            or digest(proposal.get("initialization")) != digest(INITIALIZATION_POLICY)):
        raise FitRefused("manifest differs from accepted non-executable proposal")
    expected = [{"label": m.label, "path": m.path, "sha256": m.sha256,
                 "configuration": m.configuration.model_dump(mode="json"),
                 "configuration_sha256": m.configuration_sha256, "source": m.saved_setup_source,
                 "current_bytes_verified": False} for m in manifest.models]
    if proposal.get("models") != expected:
        raise FitRefused("model/configuration/source pins differ from proposal")
    return manifest


def validated_manifest(value: FitManifest) -> FitManifest:
    # Reuse D1's accepted recursive provenance check before any serializer can fill
    # nested constructed/foreign configuration defaults or coerce original values.
    return FitManifest.model_validate_json(json.dumps(_supplied_data(value),allow_nan=False))


def token_inputs() -> list[dict[str, Any]]:
    """Exact ordered request population, including deduplicated canonical outputs."""
    result = []
    for case in build_corpus().cases:
        result.append(dict(case_id=case.case_id, case_sha256=case.sha256, kind="prompt", labels=["prompt"],
                           text=case.prompt, add_special=True))
        forms = {"lf": case.expected, "lf_terminal": case.expected+"\n",
                 "crlf": case.expected.replace("\n", "\r\n"),
                 "crlf_terminal": case.expected.replace("\n", "\r\n")+"\r\n"}
        unique: dict[str, list[str]] = {}
        for label, text in forms.items():
            unique.setdefault(text, []).append(label)
        for text, labels in unique.items():
            result.append(dict(case_id=case.case_id, case_sha256=case.sha256, kind="expected", labels=labels,
                               text=text, add_special=False))
    return result


def valid_ids(ids: Any) -> bool:
    return isinstance(ids, list) and bool(ids) and len(ids) <= 65536 and all(type(t) is int and t >= 0 for t in ids)


def token_problems(rows: Any) -> list[str]:
    expected = token_inputs()
    if not isinstance(rows, list) or len(rows) != len(expected):
        return ["complete ordered tokenizer population required"]
    problems = []
    for row, source in zip(rows, expected):
        if (not isinstance(row, dict) or type(row.get('add_special')) is not bool
                or {k: row.get(k) for k in source} != source):
            problems.append("tokenizer input identity/order differs")
            continue
        ids = row.get("ids")
        if not valid_ids(ids) or row.get("ids_sha256") != digest(ids):
            problems.append("complete nonempty integer token IDs/hash required")
    if problems:
        return list(dict.fromkeys(problems))
    for case in build_corpus().cases:
        selected = [r for r in rows if r["case_id"] == case.case_id]
        prompt = next(r for r in selected if r["kind"] == "prompt")
        maximum = max(len(r["ids"]) for r in selected if r["kind"] == "expected")
        if len(prompt["ids"])+512+16+64 > 4096 or maximum+32 > 512:
            problems.append(f"{case.case_id}: fixed context/output capacity screen fails")
    return problems


def observation_problems(raw: Any, requested: ConfigurationSnapshot, arm: str,
                         engine_files: dict[str, str]) -> list[str]:
    if not isinstance(raw, dict):
        return ["live observation missing"]
    problems = []
    startup = raw.get("startup", {})
    if (not isinstance(startup, dict) or startup.get("available") is not True
            or startup.get("generation_current") is not True or startup.get("truncated") is not False
            or startup.get("reader_errors") or type(startup.get("dropped_lines")) is not int or startup.get("dropped_lines") != 0
            or not isinstance(startup.get("text"), str)
            or type(startup.get("lines")) is not int or type(startup.get("line_limit")) is not int
            or not 0 < startup["lines"] <= startup["line_limit"]
            or startup.get("text_sha256") != hashlib.sha256(startup.get("text", "").encode()).hexdigest()):
        problems.append("complete current startup log required")
    else:
        problems.extend(activation_problems(startup["text"], arm))
        if "warming up the model with an empty run" not in startup["text"]:
            problems.append("pinned default internal initialization is unconfirmed")
        if "using config file:" in startup["text"]:
            problems.append("ambient engine configuration was loaded")
    problems.extend(engine_identity_problems(raw.get("image"), engine_files))
    image = raw.get('image',{})
    birth = image.get('create_time_s')
    if (type(image.get('pid')) is not int or image['pid']<=0 or type(birth) not in (int,float)
            or not math.isfinite(birth) or birth<=0):
        problems.append('positive current process identity required')
    if raw.get("image", {}).get("executable") != raw.get("selected_executable"):
        problems.append("mapped executable differs from selected path")
    if type(raw.get('engine_environment')) is not dict or raw['engine_environment']:
        problems.append('unregistered engine environment settings observed')
    command = raw.get('command')
    if (not isinstance(command,list) or not command or any(not isinstance(x,str) for x in command)
            or command[0] != raw.get('selected_executable') or command.count('--offline') != 1
            or command.count('--spec-type') != 1 or '--cache-reuse' in command or '--swa-full' in command
            or any(arg.split('=',1)[0] in ('--warmup','--no-warmup') for arg in command)):
        problems.append('explicit fresh lookup command evidence missing')
    else:
        algorithm = 'none' if arm=='control' else 'ngram-simple'
        if command[command.index('--spec-type')+1:command.index('--spec-type')+2] != [algorithm]:
            problems.append('command algorithm differs from arm')
        for flag,command_value in (('--spec-ngram-simple-size-n','12'),('--spec-ngram-simple-size-m','16')):
            if command.count(flag) != (1 if arm=='lookup' else 0) or (arm=='lookup' and
                    command[command.index(flag)+1:command.index(flag)+2] != [command_value]):
                problems.append('command lookup sizes differ')
    if raw.get("process_current") is not True:
        problems.append("process generation not current")
    try:
        from inferenceos.tune.lookup_fit_observers import observe_effects, machine_from_raw, slot_activation_problems
        problems.extend(slot_activation_problems(raw.get("slots"), arm))
        observed = observe_effects(startup.get("text", ""), raw.get("slots"))
        if digest(raw.get("effects")) != digest(observed):
            problems.append("stored effects differ from raw live evidence")
        if observed["problems"]:
            problems.extend(observed["problems"])
        effective = observed["configuration"]
        for name in REQUIRED_EFFECTS:
            if effective.get(name) is None:
                problems.append("effective setting unconfirmed: " + name)
        for name, value in requested.model_dump(mode="json").items():
            if name in ("kv_cache_type", "kv_cache_type_k", "kv_cache_type_v"):
                continue
            if value is not None and value != effective.get(name):
                problems.append("requested setting contradicted: " + name)
        for name, value in zip(("kv_cache_type_k", "kv_cache_type_v"), requested.effective_kv_types()):
            if value is not None and value != effective.get(name):
                problems.append("requested KV setting contradicted: " + name)
        machine = machine_from_raw(raw.get("machine"), effective.get("device"))
        MachineBinding.model_validate_json(json.dumps(machine))
    except (ValueError, TypeError, KeyError) as exc:
        problems.append("machine/effective evidence invalid: " + str(exc))
    return list(dict.fromkeys(problems))


class FitEngine(Protocol):
    def start(self, model: FitModel, arm: str) -> None: ...
    def observe(self) -> dict[str, Any]: ...
    def tokenize(self, text: str, *, add_special: bool) -> list[int]: ...
    def release(self, *, abort: bool) -> dict[str, Any]: ...


def release_complete(raw: Any) -> bool:
    """Require literal JSON booleans, never numeric 0/1 aliases, as exit evidence."""
    return (isinstance(raw, dict) and set(raw) == {"process_gone", "credential_removed", "problem"}
            and raw["process_gone"] is True and raw["credential_removed"] is True and raw["problem"] is None)


def preservation_problems(raw: Any, manifest: FitManifest) -> list[str]:
    """Recompute maps, retained saved bytes and model/setup bindings, not a pass bit."""
    from inferenceos.contracts.tune import SavedSetup
    problems = []
    if not isinstance(raw, dict):
        return ["full input/data evidence required"]
    try:
        root = Path(raw["root"])
        if not root.is_absolute():
            raise ValueError("absolute original root required")
        before, after = raw["data_before"], raw["data_after"]
        if not isinstance(before, dict) or not before or digest(before) != digest(after):
            raise ValueError("complete equal data maps required")
        for name, item in before.items():
            relative_path(name)
            if (set(item) != {"size", "mtime_ns", "sha256"} or type(item["size"]) is not int or item["size"] < 0
                    or type(item["mtime_ns"]) is not int or not re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"]))):
                raise ValueError("malformed data map entry")
        saved = raw["saved_file"]
        saved_bytes = base64.b64decode(saved["bytes_base64"], validate=True)
        saved_hash = hashlib.sha256(saved_bytes).hexdigest()
        if (saved_hash != saved["sha256"] or before["saved_setups.json"]["sha256"] != saved_hash
                or before["saved_setups.json"]["size"] != len(saved_bytes)
                or Path(saved["path"]) != root / "data/saved_setups.json"):
            raise ValueError("retained saved bytes differ from map")
        stored = strict_json(saved_bytes)["setups"]
        if not isinstance(stored, list):
            raise ValueError("saved setup layout invalid")
        if raw["frozen_hashes"] != {**manifest.python_files, **manifest.engine_files}:
            raise ValueError("frozen file population/digests differ")
        if not re.fullmatch(r'[0-9a-f]{64}',str(raw.get('environment_sha256'))):
            raise ValueError('guarded parent environment fingerprint required')
        executables = [name for name in manifest.engine_files if name.rsplit("/", 1)[-1] == "llama-server.exe"]
        if len(executables) != 1 or Path(raw["selected_executable"]) != root / executables[0]:
            raise ValueError("actual selected executable path differs")
        receipts = raw["models"]
        if len(receipts) != 2:
            raise ValueError("two model receipts required")
        for receipt, pin in zip(receipts, manifest.models):
            identity = receipt["model_file"]
            if (receipt["label"] != pin.label or identity["sha256"] != pin.sha256
                    or Path(identity["path"]) != root / pin.path or type(identity["size_bytes"]) is not int
                    or identity["size_bytes"] <= 0 or type(identity["modified_ns"]) is not int
                    or receipt["source"] != pin.saved_setup_source
                    or receipt["configuration"] != pin.configuration.model_dump(mode="json")):
                raise ValueError("actual model/setup receipt differs")
            setup_id = pin.saved_setup_source.removeprefix("saved setup ").removesuffix(" (read only)")
            matches = [s for s in stored if isinstance(s, dict) and s.get("setup_id") == setup_id]
            if len(matches) != 1:
                raise ValueError("unique saved setup source required")
            setup = SavedSetup.model_validate(matches[0])
            if (setup.model.sha256 != pin.sha256 or Path(setup.model.path) != root / pin.path
                    or setup.model.size_bytes != identity["size_bytes"]
                    or setup.context_length != 4096 or setup.configuration.model_dump(mode="json") != receipt["configuration"]
                    or setup.power_source.value != "ac" or setup.program != "llama.cpp"):
                raise ValueError("retained saved setup/model binding differs")
        ambient = raw["ambient"]
        env = ambient["environment"]
        if (ambient["platform"] != "win32" or ambient["checked"] is not True or set(env) != {"APPDATA", "PROGRAMDATA"}
                or set(ambient["absent_paths"]) != {str(Path(v) / "llama.cpp/config.ini") for v in env.values() if v}):
            raise ValueError("ambient configuration absence evidence incomplete")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        problems.append("input/data evidence invalid: " + str(exc))
    return problems


def _audit_fit(record: dict[str, Any], manifest: FitManifest) -> dict[str, Any]:
    """Read-only raw-evidence audit. Never uses stored pass/fit labels as evidence."""
    problems = []
    if (record.get("version") != VERSION or record.get("stage") != "A-fit-only"
            or record.get("manifest_sha256") != digest(manifest.model_dump(mode="json"))
            or not isinstance(record.get('run_id'),str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", record.get("run_id", ""))):
        problems.append("record/run/manifest binding differs")
    if type(record.get('generation_requests')) is not int or record.get("generation_requests") != 0 or record.get("generation_allowed") is not False:
        problems.append("fit-only no-generation contract missing")
    if record.get("simulated") is not False:
        problems.append("simulated observations are not live fit evidence")
    if 'initial_power' in record and record['initial_power'] != 'ac':
        problems.append('initial admission power was not AC')
    if record.get("status") != "done":
        problems.append("four-load run not complete")
    expected_plan = [dict(load_id=f"{record.get('run_id')}/A/{i}",model=m,arm=a) for i,(m,a) in enumerate(STAGE_A)]
    if record.get('plan') != expected_plan or record.get('not_run') != []:
        problems.append('complete four-slot plan/accounting required')
    rows = record.get("loads")
    if not isinstance(rows, list):
        rows = []
    if [(r.get("model"), r.get("arm")) for r in rows] != list(STAGE_A):
        problems.append("four-load population/order differs")
    by_model = {m.label: m for m in manifest.models}
    seen_processes: set[tuple[Any, Any]] = set()
    tokens_by_model: dict[str, Any] = {}
    observations_by_model: dict[str, Any] = {}
    machine_binding: Any = None
    for i, row in enumerate(rows):
        model = by_model.get(row.get("model"))
        if model is None:
            problems.append("unknown model load")
            continue
        if (row.get("load_id") != f"{record.get('run_id')}/A/{i}"
                or row.get("power") != ["ac", "ac"] or row.get("errors") or row.get("complete") is not True):
            problems.append(f"load {i}: incomplete run/AC evidence")
        if not release_complete(row.get("cleanup")):
            problems.append(f"load {i}: process/credential release unconfirmed")
        problems.extend(token_problems(row.get("tokens")))
        snapshots = row.get("snapshots", [])
        if not isinstance(snapshots, list) or [s.get("phase") for s in snapshots] != ["ready", "before_stop"]:
            problems.append(f"load {i}: two ordered observations required")
            continue
        identities = set()
        for snap in snapshots:
            problems.extend(observation_problems(snap, model.configuration, str(row.get("arm")), manifest.engine_files))
            image = snap.get("image", {})
            identity = (image.get("pid"), image.get("create_time_s"))
            if (type(identity[0]) is not int or identity[0] <= 0 or type(identity[1]) not in (int, float)
                    or not math.isfinite(identity[1]) or identity[1] <= 0):
                problems.append("process identity missing")
            identities.add(identity)
            from inferenceos.tune.lookup_fit_observers import machine_from_raw
            try:
                machine = machine_from_raw(snap.get("machine"),snap.get("effects", {}).get("configuration", {}).get("device"))
            except (ValueError, TypeError, KeyError):
                machine = None
            if machine_binding is None:
                machine_binding = machine
            elif machine != machine_binding:
                problems.append("machine observation changed between loads")
        if len(identities) != 1 or identities & seen_processes:
            problems.append("changed or reused process identity")
        seen_processes.update(identities)
        projected = [{k:v for k,v in t.items() if k in ("case_id","kind","labels","ids","ids_sha256")} for t in row.get("tokens", [])]
        label = model.label
        if label in tokens_by_model and projected != tokens_by_model[label]:
            problems.append(label + ": token IDs differ between arms")
        tokens_by_model[label] = projected
        effective = snapshots[0].get("effects")
        if snapshots[1].get("effects") != effective:
            problems.append(label + ": effective base changed within load")
        if label in observations_by_model and effective != observations_by_model[label]:
            problems.append(label + ": effective base changed between arms")
        observations_by_model[label] = effective
    problems.extend(preservation_problems(record.get("preservation"),manifest))
    selected = record.get("preservation", {}).get("selected_executable")
    for row in rows:
        for snap in row.get("snapshots", []):
            if snap.get("selected_executable") != selected:
                problems.append("snapshot selected path differs from preflight")
    return dict(version=VERSION, observations_complete=not problems, problems=list(dict.fromkeys(problems)),
                generation_allowed=False, stage_B_authorized=False, automatic_admission=False,
                deployment_allowed=False, performance_measured=False,
                load_boundary_power_readings=sum(len(r.get('power',[])) for r in rows),
                inventory_power_readings=sum(len(r.get('snapshots',[])) for r in rows),
                admission_power_readings=int('initial_power' in record))


def audit_fit(record: dict[str, Any], manifest: FitManifest) -> dict[str, Any]:
    try:
        return _audit_fit(record,validated_manifest(manifest))
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
        return dict(version=VERSION,observations_complete=False,problems=["malformed evidence: "+type(exc).__name__],
                    generation_allowed=False,stage_B_authorized=False,automatic_admission=False,
                    deployment_allowed=False,performance_measured=False)


def collect_fit(engine: FitEngine, manifest: FitManifest, *, run_id: str,
                power: Callable[[], str], guard: Callable[[], None],
                checkpoint: Callable[[dict[str, Any]], None], simulated: bool = False) -> dict[str, Any]:
    manifest = validated_manifest(manifest)
    if not isinstance(run_id,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',run_id):
        raise FitRefused('explicit bounded run identity required')
    record: dict[str, Any] = dict(version=VERSION, stage="A-fit-only", run_id=run_id, simulated=simulated,
                                 manifest_sha256=digest(manifest.model_dump(mode="json")), status="running", loads=[],
                                 generation_requests=0,generation_allowed=False,
                                 plan=[dict(load_id=f'{run_id}/A/{i}',model=m,arm=a) for i,(m,a) in enumerate(STAGE_A)],not_run=[])
    by_model: dict[str, FitModel] = {m.label:m for m in manifest.models}
    pending: BaseException | None = None
    for i, (label, arm) in enumerate(STAGE_A):
        row: dict[str, Any] = dict(load_id=f"{run_id}/A/{i}", model=label, arm=arm,
                                   power=[], snapshots=[], tokens=[], errors=[], complete=False, cleanup=None)
        record["loads"].append(row)
        owed = False
        failed = False
        try:
            checkpoint(deepcopy(record))
            row["power"].append(power())
            if row["power"] != ["ac"]:
                raise FitRefused("AC required before load")
            guard()
            owed = True  # even a failed or interrupted start belongs to this load
            engine.start(by_model[label], arm)
            snap = deepcopy(engine.observe())
            row["snapshots"].append(dict(snap, phase="ready"))
            issues = observation_problems(snap, by_model[label].configuration, arm, manifest.engine_files)
            if issues:
                raise FitRefused("; ".join(issues))
            from inferenceos.tune.lookup_fit_observers import machine_from_raw
            machine = machine_from_raw(snap["machine"],snap["effects"]["configuration"]["device"])
            identity = (snap["image"]["pid"], snap["image"]["create_time_s"])
            for earlier in record["loads"][:-1]:
                if any((old["image"]["pid"], old["image"]["create_time_s"]) == identity for old in earlier["snapshots"]):
                    raise FitRefused("previously observed process identity reused; no tokenization or further load")
                first = earlier["snapshots"][0]
                if machine_from_raw(first["machine"],first["effects"]["configuration"]["device"]) != machine:
                    raise FitRefused("machine changed between loads")
                if earlier["model"] == label and first["effects"] != snap["effects"]:
                    raise FitRefused("effective base changed between arms")
            for source in token_inputs():
                guard()
                token_row = deepcopy(source)
                row["tokens"].append(token_row)  # partial request kept if interrupted
                ids = deepcopy(engine.tokenize(source["text"], add_special=source["add_special"]))
                token_row.update(ids=ids, ids_sha256=digest(ids))
                if not valid_ids(ids):
                    raise FitRefused("tokenizer returned invalid/empty IDs")
                if (source["kind"] == "prompt" and len(ids)+512+16+64 > 4096
                        or source["kind"] == "expected" and len(ids)+32 > 512):
                    raise FitRefused("fixed token capacity screen fails")
            issues = token_problems(row["tokens"])
            if issues:
                raise FitRefused("; ".join(issues))
            previous = [r for r in record["loads"][:-1] if r["model"] == label]
            if previous and previous[0]["tokens"] != row["tokens"]:
                raise FitRefused("token IDs differ between arms; no further load")
            row["snapshots"].append(dict(deepcopy(engine.observe()), phase="before_stop"))
            issues = observation_problems(row["snapshots"][-1], by_model[label].configuration, arm, manifest.engine_files)
            if issues:
                raise FitRefused("; ".join(issues))
            before_stop = row['snapshots'][-1]
            first_image, last_image = snap['image'],before_stop['image']
            if ((first_image.get('pid'),first_image.get('create_time_s')) != (last_image.get('pid'),last_image.get('create_time_s'))
                    or digest(snap['effects']) != digest(before_stop['effects'])
                    or machine_from_raw(before_stop['machine'],before_stop['effects']['configuration']['device']) != machine):
                raise FitRefused('machine, process or effective base changed within load; no further load')
            guard()
        except BaseException as exc:
            failed = True
            pending = exc
            row["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
        finally:
            if owed:
                try:
                    row["cleanup"] = engine.release(abort=failed)
                    if not release_complete(row["cleanup"]):
                        failed = True
                        row["errors"].append(dict(type="CleanupUnconfirmed", message="owned resource release not confirmed"))
                except BaseException as exc:
                    failed = True
                    pending = pending if isinstance(pending, (KeyboardInterrupt, SystemExit)) else exc
                    row["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
                try:
                    row["power"].append(power())
                    if row["power"][-1] != "ac":
                        raise FitRefused("AC required after attempted load")
                    guard()
                except BaseException as exc:
                    failed = True
                    pending = pending if isinstance(pending, (KeyboardInterrupt, SystemExit)) else exc
                    row["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
            row["complete"] = owed and not failed
            record["status"] = "interrupted" if isinstance(pending, (KeyboardInterrupt, SystemExit)) else "stopped" if failed else "running"
            if failed:
                record['not_run'] = [dict(slot,cause=f"{record['status']} at {row['load_id']}")
                                     for slot in record['plan'][i+1:]]
            try:
                checkpoint(deepcopy(record))
            except BaseException:
                if pending is not None:
                    raise pending
                raise
        if failed:
            break
    if isinstance(pending, (KeyboardInterrupt, SystemExit)):
        raise pending
    if record["status"] == "running":
        record["status"] = "done"
    if len(record['loads']) < len(STAGE_A):
        cause = f"{record['status']} at {record['loads'][-1]['load_id']}"
        record['not_run'] = [dict(slot,cause=cause) for slot in record['plan'][len(record['loads']):]]
    checkpoint(deepcopy(record))
    return record
