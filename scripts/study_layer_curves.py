"""Run the layer-range check on eight models, one after another, unattended.

For each model this launches a *separate* ``check_layer_range.py`` process (so a crash
or a leftover server cannot affect the next model), streams its output to the console
and to a per-model log, records the machine conditions, and waits a cool-down before
the next model. Nothing is optimised or kept: it only measures writing speed against
the number of graphics-card layers.

Start the whole study (from the repo root, one line in PowerShell; leave the computer
alone and keep it plugged in until it finishes):

    .\\.venv\\Scripts\\python.exe scripts\\study_layer_curves.py

It writes ``reports\\layer_study_<UTC timestamp>\\`` with one JSON report and one log
per model, ``curves.csv``, ``study.json`` and ``summary.txt``. Re-running with
``--resume <folder>`` reuses an earlier folder and skips models that already finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import psutil

from inferenceos.backends.llama_server_toolchain import server_process_names
from inferenceos.config import Settings
from inferenceos.contracts.enums import PowerState
from inferenceos.hardware.profiler import detect_hardware_profile
from inferenceos.models.loader import load_model_profile
from inferenceos.tune.machine import check_machine

REPO_ROOT = Path(__file__).resolve().parents[1]
RANGE_SCRIPT = REPO_ROOT / "scripts" / "check_layer_range.py"

CSV_COLUMNS = (
    "label",
    "design",
    "layers",
    "pass",
    "writing_speed_mean",
    "slowest_run",
    "fastest_run",
    "first_word_s",
    "graphics_memory_bytes",
    "status",
)


# --------------------------------------------------------------------------- #
# The study models (run in this order).                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """One model in the study: label, file path (relative to the repo) and design."""

    label: str
    path: str
    design: str


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("qwen2.5-1.5b", "models\\qwen2.5-1.5b-instruct-q4_k_m.gguf", "standard"),
    ModelSpec("lfm2-1.2b", "models\\study\\LFM2-1.2B-Q4_K_M.gguf", "convolution + attention"),
    ModelSpec("gemma3-1b", "models\\study\\google_gemma-3-1b-it-Q4_K_M.gguf", "mostly recent-text attention"),
    ModelSpec("granite3.1-moe-3b", "models\\study\\granite-3.1-3b-a800m-instruct-Q4_K_M.gguf", "mixture of experts"),
    ModelSpec("llama3.2-3b", "models\\study\\Llama-3.2-3B-Instruct-Q4_K_M.gguf", "standard"),
    ModelSpec("phi4-mini", "models\\study\\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf", "standard (Phi)"),
    ModelSpec("granite4-h-tiny", "models\\study\\granite-4.0-h-tiny-Q4_K_M.gguf", "Mamba-2 + mixture of experts"),
    ModelSpec("qwen3.5-9b", "models\\ab\\Qwen3.5-9B-Q4_K_M.gguf", "hybrid linear attention"),
)
# --------------------------------------------------------------------------- #


RunOne = Callable[[ModelSpec, Path, Path], int]
Sleep = Callable[[float], None]
Emit = Callable[[str], None]
Clock = Callable[[], datetime]
ConditionsReader = Callable[[], "ModelConditions"]
ReportLoader = Callable[[Path], "dict[str, object] | None"]


@dataclass(frozen=True)
class ModelConditions:
    """What the machine looked like just before a model started."""

    checked_at: str
    power_source: str
    on_battery: bool
    free_graphics_memory_bytes: int | None
    total_graphics_memory_bytes: int | None


@dataclass(frozen=True)
class StudySettings:
    """Fixed (but overridable) settings for a study."""

    output_dir: Path
    context: int = 4096
    step: int = 1
    runs: int = 5
    passes: int = 2
    workload: str | None = None
    cooldown_min: float = 10.0
    only: tuple[str, ...] = ()
    resume: Path | None = None
    dry_run: bool = False


@dataclass
class ModelRecord:
    """What happened to one model."""

    label: str
    design: str
    path: str
    report_path: str
    log_path: str
    conditions: ModelConditions | None = None
    started_at: str | None = None
    ended_at: str | None = None
    exit_code: int | None = None
    status: str = "pending"
    on_battery: bool = False
    error: str | None = None


@dataclass
class StudyResult:
    """The outcome of one study run."""

    output_dir: Path
    started_at: str
    ended_at: str
    records: list[ModelRecord] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    interrupted: bool = False


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _num(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: object) -> int | None:
    number = _num(value)
    return None if number is None else int(number)


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _split_only(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_plan(models: Sequence[ModelSpec], only: Sequence[str] = ()) -> list[ModelSpec]:
    """Filter the model list by ``--only`` labels, keeping the declared run order."""
    if not only:
        return list(models)
    wanted = {label.strip() for label in only if label.strip()}
    return [model for model in models if model.label in wanted]


def missing_model_files(models: Sequence[ModelSpec], *, root: Path = REPO_ROOT) -> list[str]:
    """Paths (as declared) whose files do not exist."""
    return [model.path for model in models if not (root / model.path).is_file()]


def estimate_model_counts(total_layers: int, step: int) -> int:
    counts = set(range(0, total_layers + 1, max(step, 1))) | {total_layers}
    return len(counts)


def estimate_total_seconds(
    plans: Sequence[tuple[ModelSpec, int]], settings: StudySettings
) -> float:
    total = 0.0
    for _model, layers in plans:
        counts = estimate_model_counts(layers, settings.step)
        per_count_s = settings.runs * 1.5 + 4.0
        total += counts * max(settings.passes, 1) * per_count_s + 30.0
    total += settings.cooldown_min * 60.0 * max(len(plans) - 1, 0)
    return total


def cool_down(
    seconds: float,
    *,
    sleep: Sleep,
    emit: Emit = print,
    now: Clock = _utcnow,
    tick_s: float = 60.0,
) -> None:
    """Wait ``seconds``, printing the time left once a minute, and nothing else."""
    _ = now
    remaining = max(0.0, seconds)
    while remaining > 0:
        emit(f"  cooling down: {remaining / 60.0:.1f} min left")
        step = min(tick_s, remaining)
        sleep(step)
        remaining -= step


# --------------------------------------------------------------------------- #
# Conditions                                                                  #
# --------------------------------------------------------------------------- #
def read_machine_conditions() -> ModelConditions:
    """Read the time, power source and free graphics memory via the tune machine check."""
    settings = Settings()
    hardware = detect_hardware_profile(settings)
    machine = check_machine(settings, hardware=hardware)
    return ModelConditions(
        checked_at=_iso(_utcnow()),
        power_source=machine.power_source.value,
        on_battery=machine.power_source is PowerState.BATTERY,
        free_graphics_memory_bytes=machine.free_graphics_memory_bytes,
        total_graphics_memory_bytes=machine.total_graphics_memory_bytes,
    )


# --------------------------------------------------------------------------- #
# Orchestration (dependency-injected so it can be tested without real models) #
# --------------------------------------------------------------------------- #
def run_study(
    models: Sequence[ModelSpec],
    settings: StudySettings,
    *,
    run_one: RunOne,
    sleep: Sleep,
    now: Clock = _utcnow,
    emit: Emit = print,
    read_conditions: ConditionsReader | None = None,
    root: Path = REPO_ROOT,
) -> StudyResult:
    """Run every model in order, cooling down between them, and write the outputs."""
    started_at = _iso(now())
    plan = build_plan(models, settings.only)
    missing = missing_model_files(plan, root=root)
    if missing:
        return StudyResult(settings.output_dir, started_at, _iso(now()), missing_files=missing)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    conditions_reader = read_conditions or read_machine_conditions
    finished = _load_finished(settings.resume) if settings.resume is not None else {}
    records: list[ModelRecord] = []
    total = len(plan)

    for index, model in enumerate(plan):
        report_path = settings.output_dir / f"{model.label}.json"
        log_path = settings.output_dir / f"{model.label}.log"

        if model.label in finished:
            emit(f"[{index + 1}/{total}] {model.label}: already finished (resume); skipping")
            records.append(_resumed_record(model, finished[model.label], report_path, log_path))
            _write_study_json(settings, plan, records, started_at, _iso(now()), interrupted=False)
            continue

        conditions = conditions_reader()
        record = ModelRecord(
            label=model.label,
            design=model.design,
            path=model.path,
            report_path=str(report_path),
            log_path=str(log_path),
            conditions=conditions,
            on_battery=conditions.on_battery,
        )
        if conditions.on_battery:
            emit("!" * 74)
            emit(f"WARNING: {model.label} is running on BATTERY.")
            emit("         Plug the computer in now; results may be slower and noisier.")
            emit("         This model is marked 'on battery' in the results.")
            emit("!" * 74)

        _remove(report_path)
        record.started_at = _iso(now())
        emit(f"[{index + 1}/{total}] {model.label} ({model.design}) ...")
        try:
            code = run_one(model, report_path, log_path)
            record.exit_code = code
            record.status = "finished" if code == 0 else "failed"
        except KeyboardInterrupt:
            record.status = "interrupted"
            record.error = "interrupted by user"
            record.ended_at = _iso(now())
            records.append(record)
            _write_study_json(
                settings, plan, records, started_at, record.ended_at, interrupted=True
            )
            return StudyResult(
                settings.output_dir,
                started_at,
                record.ended_at,
                records=records,
                interrupted=True,
            )
        except Exception as exc:  # noqa: BLE001 - a bad model must not stop the study
            record.exit_code = None
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
        record.ended_at = _iso(now())
        records.append(record)
        _write_study_json(settings, plan, records, started_at, record.ended_at, interrupted=False)

        if index != total - 1:
            emit(f"cooling down for {settings.cooldown_min:g} min before the next model ...")
            cool_down(settings.cooldown_min * 60.0, sleep=sleep, emit=emit, now=now)

    ended_at = _iso(now())
    write_curves_csv(settings.output_dir / "curves.csv", records)
    write_summary(settings.output_dir / "summary.txt", records)
    _write_study_json(settings, plan, records, started_at, ended_at, interrupted=False)
    return StudyResult(settings.output_dir, started_at, ended_at, records=records)


def _load_finished(folder: Path | None) -> dict[str, Mapping[str, object]]:
    if folder is None:
        return {}
    study_path = folder / "study.json"
    if not study_path.is_file():
        return {}
    try:
        data: object = json.loads(study_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    records = data.get("records")
    if not isinstance(records, list):
        return {}
    finished: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        label = record.get("label")
        report_path = record.get("report_path")
        if (
            record.get("exit_code") == 0
            and isinstance(label, str)
            and isinstance(report_path, str)
            and Path(report_path).is_file()
        ):
            finished[label] = record
    return finished


def _resumed_record(
    model: ModelSpec,
    previous: Mapping[str, object],
    report_path: Path,
    log_path: Path,
) -> ModelRecord:
    return ModelRecord(
        label=model.label,
        design=model.design,
        path=model.path,
        report_path=str(report_path),
        log_path=str(log_path),
        started_at=_opt_str(previous.get("started_at")),
        ended_at=_opt_str(previous.get("ended_at")),
        exit_code=0,
        status="skipped",
        on_battery=bool(previous.get("on_battery", False)),
    )


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _settings_dict(settings: StudySettings) -> dict[str, object]:
    return {
        "context": settings.context,
        "step": settings.step,
        "runs": settings.runs,
        "passes": settings.passes,
        "workload": settings.workload,
        "cooldown_min": settings.cooldown_min,
        "output_dir": str(settings.output_dir),
        "only": list(settings.only),
        "resume": str(settings.resume) if settings.resume is not None else None,
    }


def _record_dict(record: ModelRecord) -> dict[str, object]:
    return {
        "label": record.label,
        "design": record.design,
        "path": record.path,
        "conditions": asdict(record.conditions) if record.conditions is not None else None,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "exit_code": record.exit_code,
        "status": record.status,
        "on_battery": record.on_battery,
        "report_path": record.report_path,
        "log_path": record.log_path,
        "error": record.error,
    }


def _write_study_json(
    settings: StudySettings,
    models: Sequence[ModelSpec],
    records: Sequence[ModelRecord],
    started_at: str,
    ended_at: str | None,
    *,
    interrupted: bool,
) -> None:
    payload = {
        "schema_version": "1.0",
        "settings": _settings_dict(settings),
        "models": [
            {"label": model.label, "path": model.path, "design": model.design}
            for model in models
        ],
        "started_at": started_at,
        "ended_at": ended_at,
        "interrupted": interrupted,
        "records": [_record_dict(record) for record in records],
    }
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "study.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# curves.csv and summary.txt                                                  #
# --------------------------------------------------------------------------- #
def load_report(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _row_status(row: Mapping[str, object]) -> str:
    if row.get("skipped") is not None:
        return "skipped"
    if row.get("error") is not None:
        return "failed"
    notes = row.get("notes")
    if isinstance(notes, list) and any("out of memory" in str(note) for note in notes):
        return "does not fit"
    if row.get("ok") is True:
        return "measured"
    return "failed"


def curves_rows(
    records: Sequence[ModelRecord], *, read_report: ReportLoader = load_report
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        report = read_report(Path(record.report_path))
        if report is None:
            continue
        raw_rows = report.get("runs")
        if not isinstance(raw_rows, list):
            continue
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            runs = row.get("writing_speed_runs")
            run_values = (
                [value for value in (_num(item) for item in runs) if value is not None]
                if isinstance(runs, list)
                else []
            )
            rows.append(
                {
                    "label": record.label,
                    "design": record.design,
                    "layers": row.get("layers"),
                    "pass": row.get("pass"),
                    "writing_speed_mean": _num(row.get("writing_speed")),
                    "slowest_run": min(run_values) if run_values else None,
                    "fastest_run": max(run_values) if run_values else None,
                    "first_word_s": _num(row.get("first_word_s")),
                    "graphics_memory_bytes": _num(row.get("graphics_memory_bytes")),
                    "status": _row_status(row),
                }
            )
    return rows


def write_curves_csv(path: Path, records: Sequence[ModelRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(curves_rows(records))


def records_from_study(folder: Path) -> list[ModelRecord]:
    """Build report records for a study folder without measuring anything."""
    records: list[ModelRecord] = []
    study = load_report(folder / "study.json")
    models: list[Mapping[str, object]] = []
    raw_models: object = study.get("models") if study is not None else None
    if isinstance(raw_models, list):
        models = [item for item in raw_models if isinstance(item, dict)]
    if models:
        for item in models:
            label = str(item.get("label", ""))
            records.append(
                ModelRecord(
                    label=label,
                    design=str(item.get("design", "")),
                    path=str(item.get("path", "")),
                    report_path=str(folder / f"{label}.json"),
                    log_path=str(folder / f"{label}.log"),
                )
            )
        return records
    for path in sorted(folder.glob("*.json")):
        if path.name in {"study.json", "replay.json"}:
            continue
        records.append(
            ModelRecord(
                label=path.stem,
                design="",
                path="",
                report_path=str(path),
                log_path=str(path.with_suffix(".log")),
            )
        )
    return records


def rebuild_outputs(folder: Path) -> tuple[int, int]:
    """Rewrite ``curves.csv`` and ``summary.txt`` from existing per-model JSON files."""
    records = records_from_study(folder)
    write_curves_csv(folder / "curves.csv", records)
    write_summary(folder / "summary.txt", records)
    return len(curves_rows(records)), len(summary_lines(records))


def _report_entries(report: Mapping[str, object]) -> list[dict[str, object]]:
    entries = report.get("summary")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _summary_line(record: ModelRecord, report: Mapping[str, object] | None) -> str:
    battery = "yes" if record.on_battery else "no"
    if report is None:
        reason = record.error or f"exit code {record.exit_code}"
        return f"{record.label} [{record.design}]: no report ({reason}); on battery: {battery}"

    entries = _report_entries(report)
    measured = [entry for entry in entries if _num(entry.get("writing_speed")) is not None]
    layers_seen = [value for value in (_as_int(entry.get("layers")) for entry in entries)]
    total_layers = max((value for value in layers_seen if value is not None), default=None)
    if not measured:
        return (
            f"{record.label} [{record.design}]: {total_layers} layers; "
            f"no layer count could be measured; on battery: {battery}"
        )

    def speed(entry: Mapping[str, object]) -> float:
        return _num(entry.get("writing_speed")) or 0.0

    fastest = max(measured, key=speed)
    best = speed(fastest)
    fastest_layers = _as_int(fastest.get("layers"))
    zero = next((entry for entry in measured if _as_int(entry.get("layers")) == 0), None)
    zero_speed = _num(zero.get("writing_speed")) if zero is not None else None
    fitted = [entry for entry in measured if not bool(entry.get("does_not_fit"))]
    most = max(fitted, key=lambda entry: _as_int(entry.get("layers")) or 0) if fitted else None
    most_layers = _as_int(most.get("layers")) if most is not None else None
    most_speed = _num(most.get("writing_speed")) if most is not None else None
    first_fail = min(
        (value for value in (_as_int(entry.get("layers")) for entry in entries if bool(entry.get("does_not_fit"))) if value is not None),
        default=None,
    )
    within = sorted(
        value
        for value in (_as_int(entry.get("layers")) for entry in measured if speed(entry) >= best * 0.97)
        if value is not None
    )
    return (
        f"{record.label} [{record.design}]: {total_layers} layers; "
        f"fastest {fastest_layers} @ {best:.2f} tok/s; "
        f"0 layers @ {_fmt(zero_speed)}; "
        f"most that fit {most_layers} @ {_fmt(most_speed)}; "
        f"first that doesn't fit: {first_fail if first_fail is not None else 'none'}; "
        f"within 3% of fastest: {', '.join(str(v) for v in within) if within else 'none'}; "
        f"on battery: {battery}"
    )


def summary_lines(
    records: Sequence[ModelRecord], *, read_report: ReportLoader = load_report
) -> list[str]:
    return [_summary_line(record, read_report(Path(record.report_path))) for record in records]


def write_summary(path: Path, records: Sequence[ModelRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = summary_lines(records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Real per-model runner (subprocess)                                          #
# --------------------------------------------------------------------------- #
def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - give up
            pass


def stop_leftover_server(*, emit: Emit = print) -> int:
    """Best-effort: terminate any ``llama-server`` process left behind."""
    names = {name.lower() for name in server_process_names()}
    stopped = 0
    for proc in psutil.process_iter(["name"]):
        try:
            name = str(proc.info.get("name") or "").lower()
        except (psutil.Error, OSError):  # pragma: no cover - process vanished
            continue
        if name in names:
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except (psutil.Error, OSError):  # pragma: no cover - already gone
                pass
            stopped += 1
    if stopped:
        emit(f"stopped {stopped} leftover llama-server process(es)")
    return stopped


def run_one_model(
    model: ModelSpec,
    report_path: Path,
    log_path: Path,
    *,
    settings: StudySettings,
    root: Path = REPO_ROOT,
) -> int:
    """Run ``check_layer_range.py`` for one model, streaming output to console and log."""
    command = [
        sys.executable,
        str(root / "scripts" / "check_layer_range.py"),
        "--model",
        str(root / model.path),
        "--context",
        str(settings.context),
        "--step",
        str(settings.step),
        "--runs",
        str(settings.runs),
        "--passes",
        str(settings.passes),
        "--out",
        str(report_path),
    ]
    if settings.workload:
        command += ["--workload", settings.workload]

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            print("\ninterrupted; stopping the current model ...", flush=True)
            _terminate(proc)
            stop_leftover_server()
            raise
        finally:
            _terminate(proc)
    stop_leftover_server()
    return code


# --------------------------------------------------------------------------- #
# Command line                                                                #
# --------------------------------------------------------------------------- #
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=4096, help="fixed context size (default 4096)")
    parser.add_argument("--step", type=int, default=1, help="layer step (default 1)")
    parser.add_argument("--runs", type=int, default=5, help="measured runs per layer count (default 5)")
    parser.add_argument("--passes", type=int, default=2, help="1 = up only, 2 = up then down (default 2)")
    parser.add_argument("--workload", default=None, help="test prompt (default: tune's)")
    parser.add_argument(
        "--cooldown-min",
        dest="cooldown_min",
        type=float,
        default=10.0,
        help="minutes to wait after each model (default 10)",
    )
    parser.add_argument("--only", default=None, help="comma-separated labels to run")
    parser.add_argument("--resume", default=None, help="reuse an earlier study folder")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="print the plan and check the files only")
    parser.add_argument(
        "--rebuild-csv",
        dest="rebuild_csv",
        default=None,
        help="rewrite curves.csv and summary.txt from an existing study folder",
    )
    return parser.parse_args(argv)


def _layer_counts(models: Sequence[ModelSpec]) -> list[int]:
    counts: list[int] = []
    for model in models:
        try:
            counts.append(int(load_model_profile(REPO_ROOT / model.path).n_layers))
        except Exception:  # noqa: BLE001 - a rough estimate is enough for the plan
            counts.append(0)
    return counts


def print_plan(
    models: Sequence[ModelSpec],
    settings: StudySettings,
    layer_counts: Sequence[int],
    missing: Sequence[str],
) -> None:
    print("=== Layer curve study plan ===")
    print(f"output     : {settings.output_dir}")
    print(f"context    : {settings.context} (fixed)")
    print(f"step       : {settings.step}")
    print(f"runs/count : {settings.runs}")
    print(f"passes     : {settings.passes}")
    print(f"workload   : {settings.workload or 'default'}")
    print(f"cool-down  : {settings.cooldown_min:g} min between models (none after the last)")
    print(f"models     : {len(models)}")
    print()
    print(f"{'#':>2}  {'label':18} {'layers':>6} {'counts':>6}  design")
    for index, (model, layers) in enumerate(zip(models, layer_counts), start=1):
        counts = estimate_model_counts(layers, settings.step)
        print(f"{index:>2}  {model.label:18} {layers:>6} {counts:>6}  {model.design}")
    rough = estimate_total_seconds(list(zip(models, layer_counts)), settings)
    print()
    print(f"rough total time (including cool-downs): ~{rough / 3600.0:.1f} h")
    if missing:
        print()
        print("MISSING FILES:")
        for path in missing:
            print(f"  - {path}")


def print_notice() -> None:
    print()
    print("IMPORTANT: this study runs for a long time.")
    print("Please leave the computer alone and keep it plugged in until it finishes.")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.rebuild_csv:
        folder = Path(args.rebuild_csv)
        if not folder.is_dir():
            print(f"study folder not found: {folder}", file=sys.stderr)
            return 2
        csv_rows, summary_lines_count = rebuild_outputs(folder)
        print(f"rebuilt {folder / 'curves.csv'} ({csv_rows} row(s))")
        print(f"rebuilt {folder / 'summary.txt'} ({summary_lines_count} line(s))")
        return 0

    only = _split_only(args.only)
    models = build_plan(MODELS, only)
    if not models:
        print("no models selected (check the --only labels)", file=sys.stderr)
        return 2

    timestamp = _utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.resume) if args.resume else Path("reports") / f"layer_study_{timestamp}"
    settings = StudySettings(
        output_dir=output_dir,
        context=args.context,
        step=args.step,
        runs=args.runs,
        passes=args.passes,
        workload=args.workload,
        cooldown_min=args.cooldown_min,
        only=only,
        resume=Path(args.resume) if args.resume else None,
        dry_run=args.dry_run,
    )

    missing = missing_model_files(models)
    layer_counts = _layer_counts(models)
    print_plan(models, settings, layer_counts, missing)
    if missing:
        print()
        print("Stopping before running anything: model file(s) missing.", file=sys.stderr)
        return 2
    if args.dry_run:
        print()
        print("dry run: all model files are present; nothing was measured.")
        return 0

    print_notice()
    result = run_study(
        models,
        settings,
        run_one=lambda model, report_path, log_path: run_one_model(
            model, report_path, log_path, settings=settings
        ),
        sleep=time.sleep,
        emit=print,
    )
    if result.missing_files:
        for path in result.missing_files:
            print(f"missing: {path}", file=sys.stderr)
        return 2

    print()
    print("=== Study summary ===")
    lines = summary_lines(result.records)
    for line in lines:
        print(line)
    print()
    print(f"output folder: {result.output_dir}")
    print(f"curves.csv   : {result.output_dir / 'curves.csv'}")
    print(f"summary.txt  : {result.output_dir / 'summary.txt'}")
    if result.interrupted:
        print("interrupted by user; partial study.json was saved.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
