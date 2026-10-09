"""Run one practical-mode (time-target) tune as a hardware evaluation, fully isolated from
the owner's real stores (practical-mode evaluation, 8 Oct 2026).

What it guarantees before the tune starts (otherwise it refuses, exit 7):

* **Isolation by configuration, not backup-and-restore.** Every store ``tune`` reads or
  writes lives under ``data_dir`` (saved setups, the learned load-to-load spread, the last
  machine check, the memory-speed cache), and ``data_dir``, ``reports_dir`` and
  ``logs_dir`` all point into a fresh scratch folder that must not exist yet. The same
  folders are also set in the environment, so a stray ``Settings()`` built anywhere else
  points there too. Each store's resolved path is checked to be inside the scratch folder
  and outside the real data folder.
* **Fixed rules.** Every setting except the evaluation's own (``PERMITTED_OVERRIDES``: the
  three scratch folders, effort, goal, the time target and the fixed start) must equal its
  code default. That covers every setting the goal, effort, quality, measurement,
  resource-safety and payback policies read (``goal_*``, ``optimization_*``, ``phase7*``,
  ``phase6_*``, ``payback_*``, ``telemetry_*``, ``tune_*`` ...), so acceptance thresholds,
  guards, quality limits, workloads and the monitor default cannot be changed by the
  environment or a ``.env`` file. A difference refuses the run before any engine starts;
  nothing is reset silently. The resolved policies are recorded as evidence. The scoped
  paging monitor must be off (the default per-answer monitor is used).
* **The intended start.** The fixed starting configuration parses, sets no context size,
  and the model file's hash matches the expected one.

What it checks afterwards (always, in ``finally``, including after an error or a stop):
every file in the real data folder (path, size, modification time and SHA-256) is
compared with the copy taken before; any difference is an isolation failure (exit 6).
The real folder is only ever read.

Outputs: the tune report, a summary JSON (settings, isolation evidence, outcome) and an
explanation (JSON + Markdown) from ``scripts/explain_practical_run.py``.

Completion needs its evidence (reviewer, 8 Oct 2026): status "done" (exit 0) only when the
report exists, was written by this run, is a valid tune report for this evaluation (model,
context, modes, the fixed start, a time-budget round), the explanation (.json and .md) was
written and reads back, the isolation check completed, and the tune's exit code agrees with
the report's outcome. A tune that failed and wrote a valid failure report is "failed"
(tune's exit code). Missing, malformed or contradictory evidence is a reporting error
(exit 8): nothing is deleted, and the summary lists every problem.

Exit codes: 0 done, tune's own non-zero (failed, with a valid failure report), 3 runner
error, 6 isolation breach, 7 refused before starting, 8 reporting error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

#: Settings this evaluation sets on purpose; EVERY other setting must equal its code default.
PERMITTED_OVERRIDES = frozenset({
    "data_dir", "reports_dir", "logs_dir",            # the scratch folders (checked separately)
    "effort_mode", "goal_mode",                        # the evaluation's modes (recorded)
    "tune_experimental_budget_s", "tune_experimental_start_config",  # target and fixed start
})
#: Which policy reads which settings (by name prefix), for the evidence record only: the
#: check itself covers every setting, so a setting missing from this table is still frozen.
POLICY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("goal", ("goal_",)),
    ("effort", ("effort_mode", "optimization_")),
    ("quality", ("phase7r_", "phase7_quality", "phase7_require_quality")),
    ("measurement", ("phase7_", "telemetry_", "phase_validity_", "resident_", "tune_")),
    ("resource safety", ("phase6_gpu_offload_", "phase10_gpu_search_")),
    ("payback", ("payback_",)),
)


def _default(field: Any) -> Any:
    return field.default_factory() if field.default_factory is not None else field.default


def frozen_rule_differences(settings: Any) -> dict[str, dict[str, Any]]:
    """Every setting outside ``PERMITTED_OVERRIDES`` whose value is not its code default."""
    from inferenceos.config import Settings

    differing: dict[str, dict[str, Any]] = {}
    for name, field in Settings.model_fields.items():
        if name in PERMITTED_OVERRIDES:
            continue
        default = _default(field)
        if getattr(settings, name) != default:
            differing[name] = {"value": getattr(settings, name), "default": default}
    return differing


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses

        return {k: _plain(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return getattr(value, "value", str(value))


def resolved_policies(settings: Any) -> dict[str, Any]:
    """The policies the tune will apply, resolved from these settings (evidence only)."""
    from inferenceos.config import Settings
    from inferenceos.contracts.enums import Situation
    from inferenceos.optimization.goals import build_goal_preference
    from inferenceos.optimization.payback import answering_seconds
    from inferenceos.optimization.preference import build_optimization_preference

    out: dict[str, Any] = {}
    for label, build in (("goal", lambda: build_goal_preference(settings)),
                         ("effort", lambda: build_optimization_preference(settings)),
                         ("payback_answering_s", lambda: {s.value: answering_seconds(settings, s) for s in Situation})):
        try:
            out[label] = _plain(build())
        except Exception as exc:  # noqa: BLE001 - recorded; the setting check decides
            out[label] = f"could not resolve: {type(exc).__name__}: {exc}"
    groups: dict[str, dict[str, Any]] = {}
    for name in sorted(Settings.model_fields):
        group = next((g for g, prefixes in POLICY_GROUPS if name.startswith(prefixes)), "other")
        groups.setdefault(group, {})[name] = _plain(getattr(settings, name))
    out["settings_by_policy"] = groups
    return out
#: Exit codes.
EXIT_RUNNER_ERROR, EXIT_ISOLATION, EXIT_REFUSED, EXIT_REPORTING = 3, 6, 7, 8


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    return h.hexdigest()


def fingerprint(folder: Path) -> dict[str, dict[str, Any]]:
    """Every file under ``folder``: size, modification time (ns) and SHA-256. Read only."""
    out: dict[str, dict[str, Any]] = {}
    if not folder.is_dir():
        return out
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        stat = path.stat()
        out[path.relative_to(folder).as_posix()] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                                     "sha256": sha256(path)}
    return out


def compare(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    for name in sorted(set(before) | set(after)):
        if name not in after:
            problems.append(f"{name}: removed")
        elif name not in before:
            problems.append(f"{name}: created")
        elif before[name] != after[name]:
            changed = [k for k in ("size", "mtime_ns", "sha256") if before[name][k] != after[name][k]]
            problems.append(f"{name}: changed ({', '.join(changed)})")
    return problems


def inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


def default_data_dir() -> Path:
    """The real data folder: the ``data_dir`` default, independent of the environment."""
    from inferenceos.config import Settings

    return Path(_default(Settings.model_fields["data_dir"]))


def preflight(settings: Any, *, scratch: Path, real_data: Path, start_config: str, context: int,
              model: Path, expect_sha256: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every check, as (name, passed, detail); and evidence to keep."""
    from inferenceos.config import Settings
    from inferenceos.tune.budgeted import fixed_start
    from inferenceos.tune.machine import last_check_path
    from inferenceos.tune.profiles import SetupStore

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    stores = {
        "saved setups": SetupStore(settings)._path,  # noqa: SLF001 - the exact path the tune will write
        "load-to-load spread": settings.data_dir / "load_spread.json",
        "last machine check": last_check_path(settings),
        "memory-speed cache": settings.data_dir / "memory_speed.json",
    }
    for name, path in stores.items():
        check(f"{name} store is in the scratch folder", inside(path, scratch) and not inside(path, real_data),
              str(path))
    for name in ("data_dir", "reports_dir", "logs_dir"):
        path = Path(getattr(settings, name))
        check(f"{name} is in the scratch folder", inside(path, scratch) and not inside(path, real_data), str(path))
    check("the scratch folder is not the real data folder or inside it",
          not inside(scratch, real_data) and not inside(real_data, scratch), f"{scratch} vs {real_data}")
    check("the scratch data folder starts empty", not any(Path(settings.data_dir).iterdir()), str(settings.data_dir))

    differing = frozen_rule_differences(settings)
    check("every setting except the evaluation's own is its code default (goal, effort, quality, "
          "measurement, resource safety, payback, thresholds, guards, workloads, monitor)",
          not differing, differing or f"all default ({len(Settings.model_fields) - len(PERMITTED_OVERRIDES)} settings)")
    check("the scoped paging monitor is off (default per-answer monitor)",
          settings.tune_experimental_measurement_monitor is False, settings.tune_experimental_measurement_monitor)
    check("the time target is set", float(settings.tune_experimental_budget_s) > 0, settings.tune_experimental_budget_s)
    try:
        config, source = fixed_start(start_config, context)
        check("the fixed starting configuration parses (no context size in it)", True, source)
        start = config.model_dump()
    except Exception as exc:  # noqa: BLE001 - reported as a refusal
        check("the fixed starting configuration parses (no context size in it)", False, f"{type(exc).__name__}: {exc}")
        start = None
    digest = sha256(model) if model.is_file() else None
    check("the model file exists", digest is not None, str(model))
    if expect_sha256:
        check("the model file is the expected one (SHA-256)", digest == expect_sha256,
              {"found": digest, "expected": expect_sha256})
    env = {k: v for k, v in os.environ.items() if k.startswith("INFERENCEOS_")}
    dotenv = ROOT / ".env"
    evidence = {
        "stores": {k: str(v) for k, v in stores.items()},
        "starting_configuration": start,
        "model_sha256": digest,
        "inferenceos_environment": env,
        "dotenv_present": dotenv.is_file(),
        "dotenv_keys": sorted(line.split("=", 1)[0].strip() for line in dotenv.read_text(encoding="utf-8").splitlines()
                              if "=" in line and not line.lstrip().startswith("#")) if dotenv.is_file() else [],
        "permitted_overrides": {name: _plain(getattr(settings, name)) for name in sorted(PERMITTED_OVERRIDES)},
        "resolved_policies": resolved_policies(settings),
    }
    return checks, evidence


def check_report(path: Path, *, began_ns: int, code: int, model: Path, context: int, effort: str, goal: str,
                 start_config: str) -> tuple[dict[str, Any] | None, list[str]]:
    """The tune report, if it is a valid report written by this evaluation; and every
    problem found. A non-zero tune exit may come with a short failure report."""
    from inferenceos.contracts.tune import TuneResult
    from inferenceos.tune.budgeted import fixed_start

    if not path.is_file():
        return None, [f"the tune report is missing ({path})"]
    problems: list[str] = []
    if path.stat().st_mtime_ns < began_ns:
        problems.append("the tune report was not written by this run (older than the run)")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, problems + [f"the tune report is not readable JSON ({type(exc).__name__}: {exc})"]
    if not isinstance(report, dict):
        return None, problems + ["the tune report is not a JSON object"]
    outcome = report.get("outcome")
    if not isinstance(outcome, str) or not outcome.strip():
        problems.append("the tune report has no outcome")
        outcome = ""
    short_failure = code != 0 and outcome.startswith("failed") and "error" in report and "rounds" not in report
    if short_failure:
        # ``run_tune``'s own short failure report (an unresolved monitor cleanup).
        return report, problems
    try:
        TuneResult.model_validate(report)
    except Exception as exc:  # noqa: BLE001 - recorded as a reporting problem
        return None, problems + [f"the tune report is not a valid tune report ({type(exc).__name__}: "
                                 f"{str(exc).splitlines()[0]})"]
    if Path(str(report.get("model_path"))).resolve() != model.resolve():
        problems.append(f"the tune report is for another model ({report.get('model_path')})")
    if report.get("context_length") != context:
        problems.append(f"the tune report's context size is {report.get('context_length')}, not {context}")
    for key, want in (("effort_mode", effort), ("goal_mode", goal)):
        if str(report.get(key) or "").replace("-", "_") != want.replace("-", "_"):
            problems.append(f"the tune report's {key} is {report.get(key)!r}, not {want!r}")
    try:
        expected, _ = fixed_start(start_config, context)
        given = {k: v for k, v in (report.get("starting_configuration") or {}).items() if v is not None}
        if given != expected.model_dump(exclude_none=True):
            problems.append(f"the tune report's start {given} is not the fixed start "
                            f"{expected.model_dump(exclude_none=True)}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the fixed start could not be read back ({exc})")
    if not any(isinstance(r, dict) and r.get("round") == "time budget" for r in report.get("rounds") or []):
        problems.append("the tune report has no time-budget round (practical mode did not run)")
    if code == 0 and outcome.startswith("failed"):
        problems.append(f"the tune exited 0 but its outcome is {outcome!r}")
    if code != 0 and not outcome.startswith("failed"):
        problems.append(f"the tune exited {code} but its outcome is {outcome!r}")
    return report, problems


def start_config_json(text: str) -> str:
    """The fixed start as JSON. Accepts a JSON object, or ``key=value,key=value`` (what a
    PowerShell job can pass without quoting trouble); values are read as JSON where they
    can be (numbers, true/false/null), otherwise as text."""
    text = text.strip()
    if text.startswith("{"):
        return text
    fields: dict[str, object] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        if "=" not in part:
            raise ValueError(f"start setting {part!r} is not key=value")
        key, value = (x.strip() for x in part.split("=", 1))
        try:
            fields[key] = json.loads(value)
        except ValueError:
            fields[key] = value
    return json.dumps(fields, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--budget-s", type=float, required=True)
    parser.add_argument("--start-config", required=True, help='JSON object, e.g. {"gpu_offload_layers": 23}')
    parser.add_argument("--effort", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--scratch", required=True, help="a folder that must not exist yet")
    parser.add_argument("--out", required=True, help="the tune report")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--explain", required=True, help="explanation path stem (.json and .md are written)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--expect-model-sha256", default=None)
    parser.add_argument("--historical", default=None, help="Unbounded report to compare with, only where it matches")
    parser.add_argument("--historical-label", default="historical Unbounded run")
    args = parser.parse_args(argv)

    scratch = Path(args.scratch)
    summary: dict[str, Any] = {"run_id": args.run_id, "status": "starting", "args": vars(args)}
    try:
        args.start_config = start_config_json(args.start_config)
    except ValueError as exc:
        args.start_config = f"invalid: {exc}"
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        tmp = summary_path.with_suffix(".partial")
        tmp.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, summary_path)

    if scratch.exists():
        summary["status"] = f"refused: the scratch folder already exists ({scratch})"
        save()
        print(summary["status"])
        return EXIT_REFUSED
    stem = Path(args.explain)
    targets = [Path(args.out), stem.with_suffix(".json"), stem.with_suffix(".md")]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        summary["status"] = f"refused: an output of this run already exists ({', '.join(existing)})"
        save()
        print(summary["status"])
        return EXIT_REFUSED
    folders = {name: scratch / name for name in ("data", "reports", "logs")}
    for folder in folders.values():
        folder.mkdir(parents=True)
    # Before anything builds Settings: a stray Settings() anywhere points to scratch too.
    os.environ["INFERENCEOS_DATA_DIR"] = str(folders["data"])
    os.environ["INFERENCEOS_REPORTS_DIR"] = str(folders["reports"])
    os.environ["INFERENCEOS_LOGS_DIR"] = str(folders["logs"])

    from inferenceos.config import Settings

    real_data = default_data_dir()
    settings = Settings(
        data_dir=folders["data"], reports_dir=folders["reports"], logs_dir=folders["logs"],
        effort_mode=args.effort, goal_mode=args.goal,
        tune_experimental_budget_s=float(args.budget_s), tune_experimental_start_config=args.start_config,
    )
    checks, evidence = preflight(settings, scratch=scratch, real_data=real_data, start_config=args.start_config,
                                 context=args.context, model=Path(args.model), expect_sha256=args.expect_model_sha256)
    summary.update({"real_data_dir": str(real_data), "scratch": str(scratch), "preflight": checks, "evidence": evidence})
    failed = [c for c in checks if not c["passed"]]
    if failed:
        summary["status"] = "refused: " + "; ".join(f"{c['check']} ({c['detail']})" for c in failed)
        save()
        print(summary["status"])
        return EXIT_REFUSED
    print("preflight: " + ", ".join(c["check"] for c in checks))
    before = fingerprint(real_data)
    summary["real_data_files"] = len(before)
    summary["status"] = "running"
    save()

    from inferenceos.tune.commands import run_tune

    code = EXIT_RUNNER_ERROR
    began = time.perf_counter()
    began_ns = time.time_ns() - 2_000_000_000  # file times may be coarser than the clock
    raised = False
    try:
        code = run_tune(settings, model_path=Path(args.model), context_length=int(args.context),
                        approve_all=True, out=Path(args.out))
    except BaseException as exc:  # noqa: BLE001 - recorded; isolation still checked below
        summary["error"] = {"type": type(exc).__name__, "message": str(exc),
                            "traceback": traceback.format_exc()[-4000:]}
        code = EXIT_RUNNER_ERROR
        raised = True
    finally:
        summary["wall_s"] = round(time.perf_counter() - began, 1)
        after = fingerprint(real_data)
        problems = compare(before, after)
        summary["isolation"] = {"real_data_unchanged": not problems, "problems": problems,
                                "files_checked": len(after),
                                "scratch_files": sorted(p.relative_to(scratch).as_posix()
                                                        for p in scratch.rglob("*") if p.is_file())}
        save()

    summary["tune_exit"] = None if raised else code
    report_path = Path(args.out)
    report, problems = (None, []) if raised else check_report(
        report_path, began_ns=began_ns, code=code, model=Path(args.model), context=int(args.context),
        effort=args.effort, goal=args.goal, start_config=args.start_config)
    if report is not None:
        summary["outcome"] = report.get("outcome")
    explanation: list[str] = []
    if report is not None and "rounds" in report:
        try:
            from explain_practical_run import explain, historical_from_unbounded_report, render_markdown

            historical = None
            if args.historical:
                if not Path(args.historical).is_file():
                    problems.append(f"the historical report is missing ({args.historical})")
                else:
                    historical = historical_from_unbounded_report(
                        json.loads(Path(args.historical).read_text(encoding="utf-8")), label=args.historical_label)
            e = explain(report, historical=historical)
            stem.with_suffix(".json").write_text(json.dumps(e, indent=2, default=str), encoding="utf-8")
            stem.with_suffix(".md").write_text(render_markdown(e), encoding="utf-8")
            json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))  # reads back
            if not stem.with_suffix(".md").read_text(encoding="utf-8").strip():
                raise ValueError("the explanation Markdown is empty")
            explanation = [str(stem.with_suffix(".json")), str(stem.with_suffix(".md"))]
        except Exception as exc:  # noqa: BLE001 - the report itself is kept
            summary["explain_error"] = f"{type(exc).__name__}: {exc}"
            problems.append(f"the explanation could not be written ({type(exc).__name__}: {exc})")
    elif report is not None:
        explanation = []  # a short failure report has nothing to explain; the tune failed
    isolation = summary.get("isolation") or {}
    if "real_data_unchanged" not in isolation or not isinstance(isolation.get("files_checked"), int):
        problems.append("the isolation check did not complete")
    summary["completion"] = {
        "complete": not raised and report is not None and not problems
        and (bool(explanation) or (code != 0 and "rounds" not in report)),
        "problems": problems, "report": str(report_path) if report is not None else None,
        "explanation": explanation, "isolation_checked": "real_data_unchanged" in isolation,
    }
    if not summary["isolation"]["real_data_unchanged"]:
        summary["status"] = "isolation breach: " + "; ".join(summary["isolation"]["problems"])
        save()
        print(summary["status"])
        return EXIT_ISOLATION
    if raised:
        summary["status"] = "error"
        save()
        print(f"status: error | {summary['error']['type']}: {summary['error']['message']}")
        return EXIT_RUNNER_ERROR
    if not summary["completion"]["complete"]:
        summary["status"] = "reporting error: " + "; ".join(problems or ["required evidence missing"])
        save()
        print(summary["status"])
        return EXIT_REPORTING
    summary["status"] = "failed" if code != 0 else "done"
    save()
    print(f"status: {summary['status']} | outcome: {summary.get('outcome')} | real data unchanged "
          f"({summary['isolation']['files_checked']} files)")
    return code if code != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
