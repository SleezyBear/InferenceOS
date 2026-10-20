"""Opt-in development-study provenance. No process control or measurement policy."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
from typing import Any

import psutil

EVIDENCE_VERSION = "reuse-load-evidence-1"


def file_identity(path: Path) -> dict[str, Any]:
    """Hash an open file and reject path/descriptor changes while it is read."""
    def signature(stat: os.stat_result) -> tuple[int, ...]:
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    path = path.resolve(strict=True)
    before = path.stat()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        closed = os.fstat(handle.fileno())
    after = path.stat()
    # On this Windows/Python 3.12 host, path stat and descriptor fstat can disagree
    # about ctime for a stable file. Check ctime within each API, not between APIs.
    if (len({signature(s) for s in (before, opened, closed, after)}) != 1
            or before.st_ctime_ns != after.st_ctime_ns or opened.st_ctime_ns != closed.st_ctime_ns):
        raise OSError(f"study input changed while hashing: {path}")
    return {"path": str(path), "size_bytes": after.st_size,
            "modified_ns": after.st_mtime_ns, "sha256": digest}


def process_image(pid: int | None, *, process: Callable[[int], Any] = psutil.Process) -> dict[str, Any]:
    """Observe executable/mapped library files of one PID/create-time identity.

    This hashes backing files, not resident instruction bytes. Incomplete enumeration,
    unreadable files and identity changes remain explicit; none confer attestation.
    A new Process object checks the final identity (create_time may be cached on one).
    """
    result: dict[str, Any] = {"captured_at": datetime.now(timezone.utc).isoformat(),
                              "pid": pid, "status": "incomplete", "files": [], "problems": [],
                              "scope": "observed executable/mapped library files; not in-memory code attestation"}
    problems: list[str] = result["problems"]
    if pid is None:
        problems.append("no active process identity")
        return result
    try:
        owner = process(pid)
        birth = float(owner.create_time())
        executable = str(owner.exe())
        result.update(create_time_s=birth, executable=executable)
        paths = {executable}
        try:
            mappings = owner.memory_maps(grouped=True)
        except (psutil.Error, OSError) as exc:
            problems.append(f"module enumeration failed: {type(exc).__name__}")
            mappings = []
        for mapping in mappings:
            name = str(mapping.path)
            base = name.lower().removesuffix(" (deleted)")
            if base.endswith((".exe", ".dll", ".dylib")) or re.search(r"\.so(?:\.\d+)*$", base):
                if name.endswith(" (deleted)"):
                    problems.append(f"deleted library mapping: {name}")
                else:
                    paths.add(name)
        if not executable:
            problems.append("no executable path")
        for name in sorted(paths):
            if not name:
                continue
            try:
                result["files"].append(file_identity(Path(name)))
            except OSError as exc:
                result["files"].append({"path": name, "error": type(exc).__name__})
                problems.append(f"unreadable mapped file: {name} ({type(exc).__name__})")
        final = process(pid)
        if float(final.create_time()) != birth or str(final.exe()) != executable:
            problems.append("process identity changed during capture")
        if not problems:
            result["status"] = "observed"
    except (psutil.Error, OSError, ValueError) as exc:
        problems.append(f"process enumeration failed: {type(exc).__name__}")
    return result


def capture_problems(loads: list[dict[str, Any]]) -> list[str]:
    """Whether required per-load observations exist; never infer engine settings."""
    problems: list[str] = []
    seen: set[str] = set()
    for load in loads:
        label = str(load.get("load_id"))
        if not load.get("load_id") or label in seen:
            problems.append(f"{label}: absent or duplicate load identity")
        seen.add(label)
        snapshots = load.get("snapshots", [])
        for phase in ("ready", "before_stop"):
            selected = [s for s in snapshots if s.get("phase") == phase]
            if len(selected) != 1:
                problems.append(f"{label}: expected one {phase} snapshot")
                continue
            snapshot = selected[0]
            log = snapshot.get("startup", {})
            if (log.get("available") is not True or log.get("truncated") is not False
                    or log.get("reader_errors") or not isinstance(log.get("text"), str)):
                problems.append(f"{label}/{phase}: startup capture incomplete")
            image = snapshot.get("image", {})
            if image.get("status") != "observed" or not image.get("files"):
                problems.append(f"{label}/{phase}: process image capture incomplete")
            if snapshot.get("process_current") is not True:
                problems.append(f"{label}/{phase}: process identity not current")
        if load.get("capture_errors"):
            problems.append(f"{label}: capture errors: {load['capture_errors']}")
    return problems


def assess_record(record: dict[str, Any]) -> dict[str, Any]:
    """Audit the new observations and their reading bindings; legacy data stays unknown.

    This grants no cache/quality verdict. That still requires the registered study's
    own comparison. Changes in library *population* may be lazy loading, but changed
    bytes for a path already observed are a study-input discrepancy.
    """
    evidence = record.get("load_evidence", {})
    loads = evidence.get("loads", [])
    problems = capture_problems(loads)
    if not loads or evidence.get("version") != EVIDENCE_VERSION:
        problems.append("required versioned load observations missing")
    if record.get("simulated") is not False or evidence.get("simulated") is not False:
        problems.append("not explicitly real process observations")
    if record.get("status") != "done":
        problems.append("study is not complete")
    inventory = record.get("data_folder_evidence", {})
    if (inventory.get("version") != "file-fingerprint-1"
            or inventory.get("before_complete") is not True
            or inventory.get("after_complete") is not True
            or not isinstance(inventory.get("before"), dict)
            or not isinstance(inventory.get("after"), dict)):
        problems.append("both complete data-folder maps are required")
    else:
        for phase in ("before", "after"):
            for name, item in inventory[phase].items():
                if (not isinstance(item, dict) or type(item.get("size")) is not int
                        or type(item.get("mtime_ns")) is not int
                        or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256")))):
                    problems.append(f"{phase}/{name}: invalid file fingerprint")
        if inventory["before"] != inventory["after"] or record.get("data_folder_unchanged") is not True:
            problems.append("data-folder preservation is not demonstrated")
    by_id = {load.get("load_id"): load for load in loads}
    referenced: list[str] = []

    def bind(replies: list[dict[str, Any]], label: str, variant: str) -> None:
        ids = {reply.get("load_evidence_id") for reply in replies}
        if len(ids) != 1 or None in ids or not replies:
            problems.append(f"{label}: readings do not bind one captured load")
            return
        load_id = str(next(iter(ids)))
        referenced.append(load_id)
        load = by_id.get(load_id)
        if load is None:
            problems.append(f"{label}: unknown load reference {load_id}")
            return
        if load.get("variant", {}).get("variant_id") != variant:
            problems.append(f"{label}: load variant differs from the reading")
        params = load.get("requested_params", {})
        for key, value in record.get("configuration", {}).items():
            if key not in params or params[key] != value:
                problems.append(f"{label}: requested {key} differs from the study configuration")
    for trial in record.get("trials", []):
        bind([trial.get("prime", {}), trial.get("measured", {})],
             f"{trial.get('case_id')}/{trial.get('block')}/{trial.get('arm')}", str(trial.get("variant_id")))
    for trial in record.get("chat_trials", []):
        for role in trial.get("order", []):
            bind(trial.get(role, []), f"chat {trial.get('pair')}/{trial.get('block')}/{role}", "default")
    if referenced != [load.get("load_id") for load in loads] or len(set(referenced)) != len(referenced):
        problems.append("captured load order/population differs from reading owners")
    file_hashes: dict[str, str] = {}
    process_owners: set[tuple[Any, Any]] = set()
    for load in loads:
        if load.get("state") != "released" or load.get("cleanup", {}).get("process_gone") is not True:
            problems.append(f"{load.get('load_id')}: release not confirmed")
        cleanup = load.get("cleanup", {})
        if cleanup.get("credential_removed") is not True or cleanup.get("problem"):
            problems.append(f"{load.get('load_id')}: credential release not confirmed")
        identities: set[tuple[Any, Any]] = set()
        for snapshot in load.get("snapshots", []):
            image = snapshot.get("image", {})
            identities.add((image.get("pid"), image.get("create_time_s")))
            if image.get("problems") or type(image.get("pid")) is not int or image.get("pid", 0) <= 0:
                problems.append(f"{load.get('load_id')}: invalid process evidence")
            if not any(f.get("path") == image.get("executable") for f in image.get("files", [])):
                problems.append(f"{load.get('load_id')}: executable not identified in mapped files")
            for file in image.get("files", []):
                path, digest = file.get("path"), file.get("sha256")
                if not path or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                    problems.append(f"{load.get('load_id')}: incomplete mapped-file identity")
                elif path in file_hashes and file_hashes[path] != digest:
                    problems.append(f"mapped-file bytes changed: {path}")
                else:
                    file_hashes[path] = digest
            log = snapshot.get("startup", {})
            text = log.get("text")
            if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != log.get("text_sha256"):
                problems.append(f"{load.get('load_id')}: startup text digest missing or mismatched")
            if (log.get("dropped_lines") != 0 or type(log.get("lines")) is not int
                    or type(log.get("line_limit")) is not int or log["lines"] > log["line_limit"]):
                problems.append(f"{load.get('load_id')}: startup loss metadata missing or invalid")
        if len(identities) != 1 or not identities or None in next(iter(identities)):
            problems.append(f"{load.get('load_id')}: process identity changed or missing between captures")
        elif identities & process_owners:
            problems.append(f"{load.get('load_id')}: fresh loads share a process identity")
        process_owners.update(identities)
    return {"version": EVIDENCE_VERSION, "complete": not problems,
            "problems": problems, "loads": len(loads), "mapped_file_hashes": file_hashes,
            "attestation": "observed backing files, not resident instruction bytes"}
