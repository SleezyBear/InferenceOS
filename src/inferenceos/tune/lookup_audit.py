"""Recompute lookup-study observations. Stored winners/usable labels confer nothing."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import re
from typing import Any

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.experiment.restoration import runtime_params_from
from inferenceos.interfaces.backend_adapter import LookupDecodingParams
from inferenceos.tune.lookup_decoding import (
    PROTOCOL, VERSION, LOOKUP_LENGTH, DRAFT_LENGTH, CONTROL, LOOKUP, digest, summarize, activation_problems,
)
from inferenceos.tune.reuse_evidence import assess_record


REQUIRED_ENGINE_FILES = frozenset({
    "llama-server.exe", "llama-server-impl.dll", "llama.dll", "ggml-base.dll", "ggml-vulkan.dll",
})


def engine_identity_problems(image: Any, frozen_files: dict[str, str]) -> list[str]:
    """Shared live/offline backing-file check; no file IO or process control.

    Windows system libraries need not belong to this toolchain. Every observed
    frozen engine component must match, and the registered executable plus required
    implementation/Vulkan libraries must all be present. These are observations of
    backing bytes, not an attestation of resident instructions.
    """
    problems: list[str] = []
    frozen: dict[str, str] = {}
    for path, expected in frozen_files.items():
        name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if not name.endswith((".exe", ".dll")):
            continue
        if name in frozen:
            problems.append("ambiguous frozen engine/library name")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            problems.append("invalid frozen engine/library hash")
        frozen[name] = expected
    if not REQUIRED_ENGINE_FILES.issubset(frozen):
        problems.append("frozen engine/library identities missing")
    if not isinstance(image, dict):
        return problems + ["engine process image missing"]
    observed = image.get("files")
    if image.get("status") != "observed" or image.get("problems") or not isinstance(observed, list):
        problems.append("engine process image incomplete")
    if not isinstance(observed, list):
        observed = []
    names: set[str] = set()
    executable = str(image.get("executable", "")).replace("\\", "/").lower()
    executable_seen = False
    for item in observed:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            problems.append("invalid observed engine/library identity")
            continue
        path = item["path"].replace("\\", "/").lower()
        name = path.rsplit("/", 1)[-1]
        if name in frozen:
            if name in names:
                problems.append("duplicate observed engine/library name")
            if item.get("sha256") != frozen[name]:
                problems.append("observed engine bytes differ from frozen input")
        names.add(name)
        if path == executable and name == "llama-server.exe":
            executable_seen = True
    if not REQUIRED_ENGINE_FILES.issubset(names):
        problems.append("engine implementation/Vulkan libraries not observed")
    if not executable_seen:
        problems.append("frozen engine executable not identified in mapped files")
    return list(dict.fromkeys(problems))


def audit(record: dict[str, Any], rules: dict[str, Any], rules_sha: str) -> dict[str, Any]:
    problems: list[str] = []
    if (record.get("protocol") != PROTOCOL or record.get("version") != VERSION
            or record.get("preregistration", {}).get("sha256") != rules_sha):
        problems.append("study protocol/version/rules identity differs")
    if record.get("parameters") != rules.get("parameters"):
        problems.append("parameters differ from the frozen plan")
    configuration = record.get("configuration", {})
    config_sha = digest(configuration)
    if record.get("configuration_sha256") != config_sha:
        problems.append("configuration digest differs")
    summary = summarize(record.get("trials", []), config_sha, simulated=record.get("simulated") is True)
    if summary != record.get("summary"):
        problems.append("stored summary does not recompute identically")
    if not summary["population_complete"] or any(not c["complete"] for c in summary["cases"].values()):
        problems.append("complete usable population not demonstrated")
    if record.get("status") != "done" or record.get("stop_reason") or record.get("held_engine_retry"):
        problems.append("execution stopped, errored or needed held cleanup")
    if record.get("simulated") is not False:
        problems.append("simulation is not real hardware evidence")
        return {"protocol": PROTOCOL, "observations_complete": False, "performance_interpretable": False,
                "problems": problems, "summary": summary, "workload_findings": {}, "automatic_admission": False}
    identity = record.get("model_file_identity", {})
    path = str(record.get("model", "")).replace("\\", "/").lower()
    matches = [m for name, m in rules.get("models", {}).items()
               if path.endswith("/" + name.replace("\\", "/").lower()) and m.get("sha256") == identity.get("sha256")]
    receipt = record.get("model_preflight", {})
    if (len(matches) != 1 or receipt.get("pin_verified") is not True
            or receipt.get("saved_setup_verified") is not True
            or receipt.get("configuration_verified") is not True
            or receipt.get("expected_sha256") != identity.get("sha256")
            or record.get("model_file_unchanged") is not True
            or record.get("frozen_inputs_unchanged") is not True):
        problems.append("model/setup/frozen-input identity not demonstrated")
    if len(matches) == 1 and matches[0].get("configuration_sha256") != config_sha:
        problems.append("configuration differs from the frozen baseline")
    ambient = record.get("ambient_config", {})
    environment = ambient.get("environment")
    if (not isinstance(environment, dict) or set(environment) != {"APPDATA", "PROGRAMDATA"}
            or any(not isinstance(v, str) for v in environment.values())):
        problems.append("ambient engine configuration environment missing")
    else:
        expected_paths = {(str(v).replace("\\", "/").rstrip("/") + "/llama.cpp/config.ini").lower()
                          for v in environment.values() if v}
        paths = ambient.get("absent_paths", [])
        actual_paths = {str(p).replace("\\", "/").lower() for p in paths} if isinstance(paths, list) else set()
        if (ambient.get("platform") != "win32" or ambient.get("checked") is not True
                or actual_paths != expected_paths):
            problems.append("ambient engine configuration absence not demonstrated")
    # Reuse the shared capture/data-map checks with a read-only shape adapter.
    # Each lookup reading is the sole request in its own fresh load.
    evidence_record = {**record, "trials": [
        {**r, "variant_id": r.get("arm"), "prime": r.get("reading") or {}, "measured": r.get("reading") or {}}
        for r in record.get("trials", [])]}
    evidence = assess_record(evidence_record)
    problems.extend(evidence["problems"])
    loads = record.get("load_evidence", {}).get("loads", [])
    trials = record.get("trials", [])
    for row in trials:
        reading = row.get("reading") or {}
        if reading.get("counters", {}).get("generated_token_ids") != reading.get("token_ids"):
            problems.append("raw native generated token IDs missing or contradictory")
    base_params = runtime_params_from(RuntimeConfiguration.model_validate(configuration))
    for index, load in enumerate(loads):
        label = str(load.get("load_id"))
        row = trials[index] if index < len(trials) else {}
        arm = row.get("arm")
        variant = CONTROL if arm == "control" else LOOKUP
        expected_params = base_params.model_copy(update={
            "log_verbosity": 4,
            "lookup_decoding": LookupDecodingParams(algorithm="none" if arm == "control" else "ngram-simple",
                                                    lookup_length=LOOKUP_LENGTH, draft_length=DRAFT_LENGTH),
        }).model_dump(mode="json")
        expected_variant = json.loads(json.dumps(asdict(variant)))
        if json.loads(json.dumps(load.get("variant"))) != expected_variant or load.get("requested_params") != expected_params:
            problems.append(f"{label}: load parameters/variant differ from the fixed comparison")
        if label != f"{record.get('run_id')}/load-{index + 1:06d}":
            problems.append(f"{label}: load is not from this run/order")
        command = load.get("command", [])
        algorithm = "none" if arm == "control" else "ngram-simple"
        if (not isinstance(command, list) or command.count("--offline") != 1
                or command.count("--spec-type") != 1
                or command[command.index("--spec-type") + 1:command.index("--spec-type") + 2] != [algorithm]):
            problems.append(f"{label}: explicit offline/algorithm command evidence missing")
        if isinstance(command, list):
            for flag, value in (("--spec-ngram-simple-size-n", str(LOOKUP_LENGTH)),
                                ("--spec-ngram-simple-size-m", str(DRAFT_LENGTH))):
                expected = 1 if arm == "lookup" else 0
                if command.count(flag) != expected or (expected and
                        command[command.index(flag) + 1:command.index(flag) + 2] != [value]):
                    problems.append(f"{label}: explicit lookup length command differs")
        for snapshot in load.get("snapshots", []):
            text = snapshot.get("startup", {}).get("text", "")
            if isinstance(text, str) and "using config file:" in text:
                problems.append(f"{label}: engine loaded ambient configuration")
            problems.extend(f"{label}: {p}" for p in activation_problems(text, str(arm)))
            problems.extend(f"{label}: {p}" for p in engine_identity_problems(
                snapshot.get("image"), rules.get("engine_files", {})))
    problems = list(dict.fromkeys(problems))
    complete = not problems
    return {"protocol": PROTOCOL, "observations_complete": complete, "performance_interpretable": complete,
            "problems": problems, "summary": summary, "load_evidence": evidence,
            "workload_findings": {name: row["outcome"] for name, row in summary["cases"].items()} if complete else {},
            "automatic_admission": False,
            "limits": ["fixed greedy requests; exact preservation does not certify factual correctness",
                       "paired intervals assume representative independent block noise; no whole-product guarantee",
                       "no tune/default/history admission; memory/TTFT observations are not a general safety proof",
                       "mapped backing files and file generations are bounded observations, not locks or memory attestation"]}
