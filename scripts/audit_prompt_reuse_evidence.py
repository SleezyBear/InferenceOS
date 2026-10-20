"""Read-only audit of the focused G3 follow-up; never contact an engine.

Exit 0 means the registered observations are complete, NOT that G3 is interpretable
or that a mitigation works. Exit 2 keeps an audit with explicit gaps; 4 refuses a
bad input/output. Original records, rules, saved setups and history stay unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inferenceos.tune.prompt_reuse import TrialResult, plan_trials, swa_cache_evidence
from inferenceos.tune.reuse_correctness import CORRECTNESS_VERSION, DEFAULT, SWA_FULL, build_pairs, correctness_summary
from inferenceos.tune.reuse_evidence import assess_record

PROTOCOL = "g3-evidence-2"
REQUIRED_LIBRARIES = {"llama-server-impl.dll", "llama.dll", "ggml-base.dll", "ggml-vulkan.dll"}
MODEL_SHA256 = "12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d"


def audit(record: dict[str, Any], rules_sha256: str, *, engine_files: dict[str, str] | None = None) -> dict[str, Any]:
    """Frozen v8 judge, one regression pair, three blocks, both engine variants.

    No cache-state inference from command flags or filenames. Incomplete logs, absent
    libraries or ambiguous replies leave G3 unanswered, even on operational exit 0.
    """
    evidence = assess_record(record)
    problems = list(evidence["problems"])
    pairs = tuple(p for p in build_pairs() if p.regression)
    trials = [TrialResult(**t) for t in record.get("trials", [])]
    summary = correctness_summary(trials, pairs)
    if summary != record.get("correctness"):
        problems.append("stored correctness summary does not recompute identically")
    expected = [(v.variant_id, c.case_id, b, a) for v, c, b, a in
                plan_trials([p.case for p in pairs], (DEFAULT, SWA_FULL), 3)]
    actual = [(t.variant_id, t.case_id, t.block, t.arm) for t in trials]
    if actual != expected or record.get("planned_trials") != len(expected) or record.get("chat_trials"):
        problems.append("trial population/order differs from the focused twelve-load protocol")
    if record.get("variants") != json.loads(json.dumps([asdict(DEFAULT), asdict(SWA_FULL)])):
        problems.append("engine variants differ from the frozen protocol")
    if (record.get("study", {}).get("version") != CORRECTNESS_VERSION
            or record.get("preregistration", {}).get("sha256") != rules_sha256):
        problems.append("study version or frozen rules identity differs")
    if (record.get("configuration", {}).get("context_length") != 4096
            or record.get("model_file_identity", {}).get("sha256") != MODEL_SHA256
            or record.get("model_file_unchanged") is not True or record.get("problems")
            or record.get("stop_reason") or record.get("held_engine_retry")):
        problems.append("study execution/input conditions not demonstrated")
    preflight = record.get("model_preflight", {})
    if (preflight.get("expected_sha256") != MODEL_SHA256 or preflight.get("pin_verified") is not True
            or preflight.get("saved_setup_required") is not True or preflight.get("saved_setup_verified") is not True):
        problems.append("frozen model and saved-setup preflight not demonstrated")
    if record.get("configuration", {}).get("device") not in (None, "Vulkan0"):
        problems.append("this protocol requires the saved Vulkan configuration")
    loads = {load["load_id"]: load for load in record.get("load_evidence", {}).get("loads", [])}
    for trial in trials:
        label = f"{trial.variant_id}/{trial.block}/{trial.arm}"
        config_digest = hashlib.sha256(json.dumps(record.get("configuration"), sort_keys=True,
                                                  separators=(",", ":")).encode()).hexdigest()
        if (trial.identity.get("configuration_sha256") != config_digest
                or trial.identity.get("context_length") != 4096):
            problems.append(f"{label}: request identity differs from recorded configuration")
        if trial.power != ["ac", "ac"] or trial.interrupted or trial.problems:
            problems.append(f"{label}: trial did not complete on mains power")
        load = loads.get((trial.prime or {}).get("load_evidence_id"), {})
        variant = json.loads(json.dumps(asdict(DEFAULT if trial.variant_id == DEFAULT.variant_id else SWA_FULL)))
        if load.get("variant") != variant:
            problems.append(f"{label}: captured variant differs")
        params = load.get("requested_params", {})
        if (params.get("swa_full") is not variant["swa_full"] or params.get("cache_reuse") is not None
                or params.get("log_verbosity") != 4):
            problems.append(f"{label}: captured request differs from the variant")
        for snapshot in load.get("snapshots", []):
            executable = str(snapshot.get("image", {}).get("executable", "")).replace("\\", "/").rsplit("/", 1)[-1]
            if executable.lower() != "llama-server.exe":
                problems.append(f"{label}: expected launcher executable not observed")
            names = {Path(f.get("path", "")).name.lower() for f in snapshot.get("image", {}).get("files", [])}
            # Windows paths must work when an audit is repeated on another platform.
            names |= {str(f.get("path", "")).replace("\\", "/").rsplit("/", 1)[-1].lower()
                      for f in snapshot.get("image", {}).get("files", [])}
            if not REQUIRED_LIBRARIES.issubset(names):
                problems.append(f"{label}: expected implementation/Vulkan libraries not observed")
            if engine_files is not None:
                for file in snapshot.get("image", {}).get("files", []):
                    name = str(file.get("path", "")).replace("\\", "/").rsplit("/", 1)[-1].lower()
                    if name in engine_files and file.get("sha256") != engine_files[name]:
                        problems.append(f"{label}: mapped {name} differs from the frozen engine input")
            log = snapshot.get("startup", {}).get("text")
            parsed = swa_cache_evidence(log, requested_full=bool(variant["swa_full"]))
            if snapshot.get("phase") == "before_stop" and parsed != trial.swa_cache:
                problems.append(f"{label}: raw cache-state log differs from trial evidence")
    rows = [row for pair in summary["pairs"].values() for row in pair["blocks"]]
    if len(rows) != 6 or any(row.get("usable") is not True for row in rows):
        problems.append("not all six fresh/reuse comparisons are usable")
    if any(not str(load_id).startswith(f"{record.get('run_id')}/load-") for load_id in loads):
        problems.append("captured load identity is not from this run")
    problems = list(dict.fromkeys(problems))
    comparison = summary["swa_full_comparison"]
    return {"protocol": PROTOCOL, "observations_complete": not problems,
            "problems": problems, "load_evidence": evidence,
            "correctness": summary, "g3_interpretable": not problems and comparison["interpretable"],
            "limits": ["observed backing files, not in-memory code attestation",
                       "frozen first-answer judge; no general semantic or mitigation claim",
                       "unknown cache states leave G3 unanswered; no default change"]}


def main(argv: list[str] | None = None) -> int:
    from evaluate_prompt_reuse import output_problem
    from inferenceos.config import Settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--rules-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rules = args.rules.read_bytes()
        if (hashlib.sha256(rules).hexdigest() != args.rules_sha256.lower()
                or PROTOCOL not in rules.decode("utf-8") or CORRECTNESS_VERSION not in rules.decode("utf-8")):
            raise ValueError("governing rules digest/version mismatch")
        frozen = json.loads(rules)
        if frozen.get("protocol") != PROTOCOL or frozen.get("study_version") != CORRECTNESS_VERSION:
            raise ValueError("governing rules protocol/version mismatch")
        if frozen.get("model_sha256") != MODEL_SHA256 or frozen.get("require_saved_setup") is not True:
            raise ValueError("governing rules must pin the G3 model and require a verified saved setup")
        engine_files = {str(path).replace("\\", "/").rsplit("/", 1)[-1].lower(): digest
                        for path, digest in frozen["engine_files"].items()}
        if not (REQUIRED_LIBRARIES | {"llama-server.exe"}).issubset(engine_files):
            raise ValueError("frozen engine identities missing")
        raw = args.source.read_bytes()
        record = json.loads(raw)
        protected = [args.source, args.rules, Path(Settings().data_dir)]
        if record.get("model"):
            protected.append(Path(record["model"]))
        if record.get("data_folder_evidence", {}).get("root"):
            protected.append(Path(record["data_folder_evidence"]["root"]))
        problem = output_problem(args.out, protected=protected)
        if problem:
            raise ValueError(problem)
        result = audit(record, args.rules_sha256.lower(), engine_files=engine_files)
        result.update(source=str(args.source), source_sha256=hashlib.sha256(raw).hexdigest(),
                      rules_sha256=args.rules_sha256.lower())
        if args.source.read_bytes() != raw or args.rules.read_bytes() != rules:
            raise ValueError("source or rules changed during audit")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps({k: result[k] for k in ("observations_complete", "g3_interpretable", "problems")}))
        return 0 if result["observations_complete"] else 2
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"refused: {type(exc).__name__}: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
