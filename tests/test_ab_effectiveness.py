"""Unit tests for the root-level A/B effectiveness experiment runner.

The runner is a single self-contained script at the project root, so it is loaded here
by path rather than imported as a package. No test starts a real server, downloads a
model, or runs the A/B experiment.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "run_ab_effectiveness.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_ab_effectiveness", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_ab_effectiveness"] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _outcome(
    *,
    test: str,
    index: int,
    decode_tps: float | None,
    latency: float | None = 5.0,
    ttft: float | None = 0.5,
    rss: int | None = 1_000_000,
    success: bool = True,
    post: bool = False,
    submitted_at: str = "2026-01-01T00:00:00+00:00",
    prompt_sha256: str = "deadbeef",
) -> dict[str, Any]:
    payload = {
        "test": test,
        "index": index,
        "request_id": f"{test}-{index}",
        "prompt": "fixed prompt",
        "prompt_sha256": prompt_sha256,
        "n_predict": 64,
        "seed": 1234,
        "temperature": 0.0,
        "submitted_at": submitted_at,
        "first_token_at": submitted_at,
        "completed_at": submitted_at,
        "ttft_s": ttft,
        "total_latency_s": latency,
        "generated_tokens": 64,
        "prompt_tokens": 16,
        "decode_tokens_per_sec": decode_tps,
        "peak_rss_bytes": rss,
        "final_rss_bytes": rss,
        "success": success,
        "error": None,
        "output": "ok" if success else None,
        "post_intervention": post,
    }
    return payload


def _summary(
    *,
    outcomes: list[dict[str, Any]],
    telemetry_duration: float = 120.0,
    intervention_count: int = 0,
    intervention_end: str | None = None,
    overhead_s: float | None = None,
    model_sha256: str = "abc",
    intended: int | None = None,
    termination: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "outcomes": outcomes,
        "telemetry": {
            "duration_s": telemetry_duration,
            "peak_cpu_percent": 80.0,
            "peak_target_rss_bytes": 4_000_000_000,
            "peak_gpu_engine_percent": 20.0,
            "monitor_overhead_ms_mean": 5.0,
        },
        "intervention_count": intervention_count,
        "intervention_end": intervention_end,
        "monitor_overhead_s": overhead_s,
        "model_sha256": model_sha256,
        "monitor_termination": termination,
    }
    if intended is not None:
        payload["intended_request_count"] = intended
    return payload


def _settings() -> Any:
    from inferenceos.config import Settings

    return Settings()


def _stabilization(*, shortened: bool = False, completed: bool = True) -> dict[str, Any]:
    return {
        "label": "stabilization",
        "shortened": shortened,
        "completed_successfully": completed,
        "requested_seconds": 1200.0,
        "actual_seconds": 1200.1,
        "machine_state_after": {"process_inventory_count": 200},
        "returned_to_baseline": {"approximately_returned": True},
        "telemetry": {"sample_count": 600},
        "process_count_evolution": {},
    }


def _analyze(test_a: dict[str, Any], test_b: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    cfg = runner.parse_args(["--run-dir", "unused", "--cooldown-seconds", "1200"])
    cleanup = overrides.pop(
        "cleanup",
        {"cleanup_success": True, "unexpected_survivors": []},
    )
    pre_a_stabilization = overrides.pop("pre_a_stabilization", _stabilization())
    post_a_stabilization = overrides.pop("post_a_stabilization", _stabilization())
    result: dict[str, Any] = runner.analyze_ab(
        test_a=test_a,
        test_b=test_b,
        cfg=cfg,
        cleanup=cleanup,
        pre_a_stabilization=pre_a_stabilization,
        post_a_stabilization=post_a_stabilization,
        preflight={"system_state": {"process_count": 200}},
        settings=_settings(),
        pre_a_cleanup=overrides.pop("pre_a_cleanup", None),
    )
    return result


# --------------------------------------------------------------------------- #
# Workload construction                                                         #
# --------------------------------------------------------------------------- #
def test_fixed_requests_are_deterministic() -> None:
    first = runner.build_fixed_requests(rounds=3)
    second = runner.build_fixed_requests(rounds=3)
    assert len(first) == 3 * len(runner.FIXED_PROMPTS)
    assert [r.request_id for r in first] == [r.request_id for r in second]
    assert [r.prompt_sha256 for r in first] == [r.prompt_sha256 for r in second]
    assert all(r.n_predict == runner.DEFAULT_N_PREDICT for r in first)
    assert all(r.temperature == 0.0 for r in first)


def test_workload_artifact_lists_every_request() -> None:
    cfg = runner.parse_args(["--run-dir", "unused"])
    requests = runner.build_fixed_requests(rounds=2)
    artifact = runner._workload_artifact(cfg, requests)
    assert artifact["rounds"] == 2
    assert len(artifact["requests"]) == len(requests)
    assert artifact["requests"][0]["prompt"] in runner.FIXED_PROMPTS


def test_build_unmanaged_command_mirrors_configuration() -> None:
    command = runner.build_unmanaged_command(
        server=Path("llama-server.exe"),
        model_path=Path("model.gguf"),
        host="127.0.0.1",
        port=1234,
        context_length=4096,
        gpu_layers=0,
        threads=None,
    )
    assert "-ngl" in command and command[command.index("-ngl") + 1] == "0"
    assert "-c" in command and command[command.index("-c") + 1] == "4096"
    assert "-t" not in command


# --------------------------------------------------------------------------- #
# Statistics                                                                    #
# --------------------------------------------------------------------------- #
def test_summarize_values_empty() -> None:
    stats = runner.summarize_values([])
    assert stats["n"] == 0
    assert stats["mean"] is None


def test_summarize_values_basic() -> None:
    stats = runner.summarize_values([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["p95"] is not None and stats["p95"] >= 3.4


def test_percentile_single_value() -> None:
    assert runner.percentile([7.0], 95.0) == 7.0


def test_mean_difference_ci_detects_clear_shift() -> None:
    diff, low, high = runner.mean_difference_ci([1.0, 1.1, 0.9, 1.0], [2.0, 2.1, 1.9, 2.0])
    assert diff is not None and diff > 0
    assert low is not None and high is not None and low > 0


def test_metric_verdict_no_improvement_when_identical() -> None:
    values = [10.0, 10.2, 9.8, 10.1, 9.9]
    verdict = runner.metric_verdict(
        baseline_values=values, measured_values=list(values), higher_is_better=True, min_effect=0.05
    )
    assert verdict["improved"] is False
    assert verdict["regressed"] is False


def test_metric_verdict_improvement_beyond_variability() -> None:
    baseline = [10.0, 10.1, 9.9, 10.0, 10.05]
    measured = [12.0, 12.1, 11.9, 12.0, 12.05]
    verdict = runner.metric_verdict(
        baseline_values=baseline, measured_values=measured, higher_is_better=True, min_effect=0.05
    )
    assert verdict["improved"] is True
    assert verdict["beyond_variability"] is True


def test_metric_verdict_regression_for_lower_is_better() -> None:
    baseline = [1.0, 1.05, 0.95, 1.0]
    measured = [2.0, 2.1, 1.9, 2.0]
    verdict = runner.metric_verdict(
        baseline_values=baseline, measured_values=measured, higher_is_better=False, min_effect=0.05
    )
    assert verdict["regressed"] is True


# --------------------------------------------------------------------------- #
# A/B analysis                                                                  #
# --------------------------------------------------------------------------- #
def test_analysis_reports_no_improvement_when_equal() -> None:
    a = _summary(outcomes=[_outcome(test="A", index=i, decode_tps=10.0) for i in range(5)])
    b = _summary(outcomes=[_outcome(test="B", index=i, decode_tps=10.0) for i in range(5)])
    analysis = _analyze(a, b)
    assert analysis["interpretation"]["verdict"] == "no_demonstrated_improvement"
    assert "No demonstrated improvement" in analysis["interpretation"]["statement"]


def test_analysis_detects_retained_improvement_after_intervention() -> None:
    a = _summary(outcomes=[_outcome(test="A", index=i, decode_tps=10.0) for i in range(6)])
    b = _summary(
        outcomes=[
            _outcome(test="B", index=i, decode_tps=10.0, post=False) for i in range(3)
        ]
        + [_outcome(test="B", index=i + 3, decode_tps=12.5, post=True) for i in range(6)],
        intervention_count=1,
        intervention_end="2026-01-01T00:00:00+00:00",
        overhead_s=10.0,
    )
    analysis = _analyze(a, b)
    assert analysis["interpretation"]["verdict"] == "improvement_demonstrated"
    assert analysis["interpretation"]["retained_improvement"] is True
    assert analysis["interpretation"]["used_post_intervention_only"] is True


def test_analysis_does_not_claim_improvement_without_intervention() -> None:
    # B is faster but InferenceOS changed nothing: this is variability, not an effect.
    a = _summary(outcomes=[_outcome(test="A", index=i, decode_tps=10.0) for i in range(6)])
    b = _summary(outcomes=[_outcome(test="B", index=i, decode_tps=12.0) for i in range(6)])
    analysis = _analyze(a, b)
    assert analysis["interpretation"]["verdict"] == "no_demonstrated_improvement"
    assert analysis["interpretation"]["retained_improvement"] is False


def test_analysis_is_inconclusive_on_surviving_process() -> None:
    a = _summary(outcomes=[_outcome(test="A", index=i, decode_tps=10.0) for i in range(5)])
    b = _summary(outcomes=[_outcome(test="B", index=i, decode_tps=10.0) for i in range(5)])
    analysis = _analyze(
        a,
        b,
        cleanup={"cleanup_success": False, "unexpected_survivors": [{"pid": 999}]},
    )
    assert analysis["validity"]["major_confound"] is True
    assert analysis["interpretation"]["verdict"] == "inconclusive"
    assert analysis["interpretation"]["verdict_text"] == "INCONCLUSIVE"


def test_analysis_flags_shortened_stabilization() -> None:
    a = _summary(outcomes=[_outcome(test="A", index=i, decode_tps=10.0) for i in range(5)])
    b = _summary(outcomes=[_outcome(test="B", index=i, decode_tps=10.0) for i in range(5)])
    analysis = _analyze(
        a,
        b,
        pre_a_stabilization=_stabilization(shortened=True),
    )
    assert analysis["validity"]["major_confound"] is True
    assert analysis["interpretation"]["verdict"] == "inconclusive"


def test_analysis_flags_incomplete_stabilization() -> None:
    a = _summary(outcomes=[_outcome(test="A", index=i, decode_tps=10.0) for i in range(5)])
    b = _summary(outcomes=[_outcome(test="B", index=i, decode_tps=10.0) for i in range(5)])
    analysis = _analyze(
        a,
        b,
        post_a_stabilization=_stabilization(completed=False),
    )
    checks = {item["check"]: item["ok"] for item in analysis["validity"]["checks"]}
    assert checks["post-A stabilization ran for the full requested duration"] is False


def test_cross_run_net_benefit_present() -> None:
    a = _summary(outcomes=[_outcome(test="A", index=i, decode_tps=10.0) for i in range(5)])
    b = _summary(outcomes=[_outcome(test="B", index=i, decode_tps=12.0) for i in range(5)])
    analysis = _analyze(a, b)
    assert "status" in analysis["cross_run_net_benefit"]


def test_analysis_distinguishes_pre_a_and_post_a_stabilization() -> None:
    a = _summary(outcomes=[_outcome(test="A", index=i, decode_tps=10.0) for i in range(5)])
    b = _summary(outcomes=[_outcome(test="B", index=i, decode_tps=10.0) for i in range(5)])
    pre = {
        "label": "pre_a",
        "cleanup_success": True,
        "cleanup_actions": ["pre_a: verified no llama-server process remains"],
        "remaining_inference": [],
        "remaining_owned": [],
        "checks": {},
    }
    analysis = _analyze(
        a,
        b,
        pre_a_cleanup=pre,
        pre_a_stabilization=_stabilization(),
        post_a_stabilization=_stabilization(),
    )
    assert analysis["cleanup"]["pre_a"]["cleanup_success"] is True
    assert "pre_a_stabilization" in analysis["stabilization"]
    assert "post_a_stabilization" in analysis["stabilization"]
    assert "machine_state_comparison" in analysis
    checks = {item["check"]: item["ok"] for item in analysis["validity"]["checks"]}
    assert checks["pre-A cleanup established a clean starting state"] is True
    assert checks["pre-A stabilization ran for the full requested duration"] is True
    assert checks["post-A stabilization ran for the full requested duration"] is True


# --------------------------------------------------------------------------- #
# Equal-work A/B validity gate                                                  #
# --------------------------------------------------------------------------- #
def _equal_work_pair(
    *, a_count: int = 6, b_count: int = 6, b_fail: int = 0, termination: str | None = "duration_reached"
) -> tuple[dict[str, Any], dict[str, Any]]:
    a = _summary(
        outcomes=[_outcome(test="A", index=i, decode_tps=10.0) for i in range(a_count)],
        intended=a_count,
    )
    b_outcomes = [_outcome(test="B", index=i, decode_tps=10.0) for i in range(b_count)]
    for i in range(b_fail):
        b_outcomes[i]["success"] = False
        b_outcomes[i]["output"] = None
    b = _summary(outcomes=b_outcomes, intended=a_count, termination=termination)
    return a, b


def test_equal_work_valid_when_counts_match() -> None:
    a, b = _equal_work_pair(a_count=6, b_count=6)
    analysis = _analyze(a, b)
    assert analysis["equal_work"]["valid"] is True
    assert analysis["interpretation"]["effective_comparison"] is True
    assert analysis["interpretation"]["verdict"] != "invalid_comparison"


def test_equal_work_invalid_when_b_truncated() -> None:
    a, b = _equal_work_pair(a_count=6, b_count=3)
    analysis = _analyze(a, b)
    assert analysis["equal_work"]["valid"] is False
    assert any(
        "B completed the full fixed workload" in name
        for name in analysis["equal_work"]["failed_checks"]
    )
    assert analysis["interpretation"]["verdict"] == "invalid_comparison"
    assert analysis["interpretation"]["verdict_text"].startswith("INVALID COMPARISON")


def test_equal_work_invalid_on_failed_request() -> None:
    a, b = _equal_work_pair(a_count=6, b_count=6, b_fail=1)
    analysis = _analyze(a, b)
    assert analysis["equal_work"]["valid"] is False
    assert "no failed requests" in analysis["equal_work"]["failed_checks"]
    assert analysis["interpretation"]["verdict"] == "invalid_comparison"


def test_equal_work_invalid_on_monitor_self_termination() -> None:
    a, b = _equal_work_pair(a_count=6, b_count=6, termination="monitoring_budget_exhausted")
    analysis = _analyze(a, b)
    assert analysis["equal_work"]["valid"] is False
    assert "monitor did not terminate the managed run early" in analysis["equal_work"]["failed_checks"]
    assert analysis["interpretation"]["verdict"] == "invalid_comparison"


def test_equal_work_invalid_on_workload_spec_mismatch() -> None:
    a = _summary(
        outcomes=[_outcome(test="A", index=i, decode_tps=10.0) for i in range(6)],
        intended=6,
    )
    b = _summary(
        outcomes=[
            _outcome(test="B", index=i, decode_tps=10.0, prompt_sha256="different")
            for i in range(6)
        ],
        intended=6,
        termination="duration_reached",
    )
    analysis = _analyze(a, b)
    assert analysis["equal_work"]["valid"] is False
    assert (
        "workload specification identical between A and B"
        in analysis["equal_work"]["failed_checks"]
    )
    assert analysis["interpretation"]["verdict"] == "invalid_comparison"


# --------------------------------------------------------------------------- #
# Process classification / comparison                                           #
# --------------------------------------------------------------------------- #
def test_classify_process_categories() -> None:
    assert runner.classify_process("llama-server.exe", None, self_pid=1) == "llama_server"
    assert runner.classify_process("llama-cli.exe", None, self_pid=1) == "llama_cpp"
    assert runner.classify_process("ollama.exe", None, self_pid=1) == "ollama"
    assert runner.classify_process("python.exe", "vllm serve", self_pid=1) == "model_server"
    assert runner.classify_process("explorer.exe", None, self_pid=1) == "other"
    assert runner.classify_process("inferenceos.exe", None, self_pid=1) == "inferenceos"
    assert (
        runner.classify_process("python.exe", "python -m inferenceos optimize", self_pid=1)
        == "inferenceos"
    )
    # A shell whose command line merely contains the project path is not a runtime.
    assert (
        runner.classify_process(
            "powershell.exe", "Set-Location C:\\Projects\\InferenceOS", self_pid=1
        )
        == "other"
    )
    # A shell that merely mentions a server name in its arguments is not a server.
    assert (
        runner.classify_process(
            "powershell.exe", 'Get-Process -Name "llama-server"', self_pid=1
        )
        == "other"
    )


def test_compare_process_sets_identifies_created_and_remaining() -> None:
    before = [
        runner.ProcessRecord(1, "explorer.exe", 1.0, None, None, 10, "other"),
        runner.ProcessRecord(2, "llama-server.exe", 2.0, None, None, 20, "llama_server"),
    ]
    after = before + [
        runner.ProcessRecord(3, "llama-server.exe", 3.0, None, None, 30, "llama_server"),
    ]
    comparison = runner.compare_process_sets(before, after, launched_pids=[3])
    assert [record.pid for record in comparison.created] == [3]
    assert [record.pid for record in comparison.remaining] == [3]
    assert [record.pid for record in comparison.unexpected_survivors] == [3]


# --------------------------------------------------------------------------- #
# Shared cleanup (pre-A and post-A)                                             #
# --------------------------------------------------------------------------- #
def test_run_cleanup_writes_labeled_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    run = runner.RunDirectory(tmp_path / "run")
    records = [runner.ProcessRecord(1, "explorer.exe", 1.0, None, None, 10, "other")]
    monkeypatch.setattr(runner, "snapshot_processes", lambda: records)
    payload = runner.run_cleanup(
        run,
        before_records=[],
        launched_pids=[],
        label="pre_a",
        phase_noun="before Test A",
    )
    assert payload["label"] == "pre_a"
    assert payload["cleanup_success"] is True
    for suffix in ("process_before", "process_created", "process_remaining", "cleanup"):
        assert (tmp_path / "run" / "cleanup" / f"pre_a_{suffix}.json").is_file()


def test_pre_a_cleanup_fails_closed_with_inference_process(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run = runner.RunDirectory(tmp_path / "run")
    llama = runner.ProcessRecord(
        42, "llama-server.exe", 2.0, None, None, 20, "llama_server"
    )
    monkeypatch.setattr(runner, "snapshot_processes", lambda: [llama])
    payload = runner.run_cleanup(
        run,
        before_records=[],
        launched_pids=[],
        label="pre_a",
        phase_noun="before Test A",
    )
    assert payload["cleanup_success"] is False
    assert payload["remaining_inference"][0]["pid"] == 42
    # No experiment-owned process was launched, so nothing was terminated.
    assert all("terminating tree" not in action for action in payload["cleanup_actions"])


def test_post_a_cleanup_reports_inference_categories(tmp_path: Path, monkeypatch: Any) -> None:
    run = runner.RunDirectory(tmp_path / "run")
    after = [
        runner.ProcessRecord(7, "ollama.exe", 3.0, None, None, 30, "ollama"),
        runner.ProcessRecord(8, "explorer.exe", 4.0, None, None, 5, "other"),
    ]
    monkeypatch.setattr(runner, "snapshot_processes", lambda: after)
    payload = runner.run_cleanup(
        run,
        before_records=[],
        launched_pids=[],
        label="post_a",
        phase_noun="after Test A",
    )
    assert payload["cleanup_success"] is False
    assert payload["checks"]["ollama"]
    assert payload["checks"]["llama_server"] == []


# --------------------------------------------------------------------------- #
# Artifacts / CLI                                                               #
# --------------------------------------------------------------------------- #
def test_atomic_write_json_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "data.json"
    runner.atomic_write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}


def test_run_directory_phase_state(tmp_path: Path) -> None:
    run = runner.RunDirectory(tmp_path / "run")
    run.mark_complete("preflight", "hash-1")
    assert run.is_complete("preflight", "hash-1") is True
    assert run.is_complete("preflight", "hash-2") is False
    assert run.is_complete("test_a", "hash-1") is False


def test_parse_args_overrides_and_defaults() -> None:
    cfg = runner.parse_args(
        [
            "--run-dir",
            "somewhere",
            "--dry-run",
            "--gpu-layers",
            "7",
            "--threads",
            "4",
            "--rounds",
            "2",
            "--cooldown-seconds",
            "60",
        ]
    )
    assert cfg.dry_run is True
    assert cfg.gpu_layers == 7
    assert cfg.threads == 4
    assert cfg.rounds == 2
    assert cfg.cooldown_s == 60.0
    assert cfg.auto_approve is True
    assert cfg.run_dir == Path("somewhere")


def test_config_hash_changes_with_model() -> None:
    first = runner.parse_args(["--run-dir", "x", "--model-path", "a.gguf"])
    second = runner.parse_args(["--run-dir", "x", "--model-path", "b.gguf"])
    assert first.config_hash() != second.config_hash()


# --------------------------------------------------------------------------- #
# Model acquisition                                                             #
# --------------------------------------------------------------------------- #
def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    payload = b"inferenceos-ab-test"
    source = tmp_path / "blob.bin"
    source.write_bytes(payload)
    assert runner.sha256_file(source) == hashlib.sha256(payload).hexdigest()


def test_ensure_model_refuses_checksum_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    source.write_bytes(b"model-bytes")
    wrong = "0" * 64
    cfg = runner.parse_args(["--run-dir", "unused"])
    cfg.model_path = tmp_path / "target.gguf"
    cfg.model_url = source.resolve().as_uri()
    cfg.model_sha256 = wrong
    result = runner.ensure_model_available(cfg)
    assert result is None
    assert not cfg.model_path.exists()


def test_ensure_model_downloads_when_url_and_checksum_match(tmp_path: Path) -> None:
    import hashlib

    payload = b"model-bytes-v2"
    source = tmp_path / "source.gguf"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    cfg = runner.parse_args(["--run-dir", "unused"])
    cfg.model_path = tmp_path / "target.gguf"
    cfg.model_url = source.resolve().as_uri()
    cfg.model_sha256 = digest
    result = runner.ensure_model_available(cfg)
    assert result is not None
    assert cfg.model_path.read_bytes() == payload


# --------------------------------------------------------------------------- #
# Stabilization                                                                 #
# --------------------------------------------------------------------------- #
class _FakeTelemetryCollector:
    """Deterministic stand-in for TelemetryCollector (no real sampling)."""

    def __init__(
        self,
        *,
        jsonl_path: Path,
        interval_s: float,
        target_pid_provider: Any,
        include_gpu: bool = True,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.include_gpu = include_gpu

    def start(self) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.write_text(
            '{"process_count": 100, "cpu_percent_total": 5.0, "ram_available_bytes": 1000}\n'
            '{"process_count": 101, "cpu_percent_total": 6.0, "ram_available_bytes": 900}\n',
            encoding="utf-8",
        )

    def stop(self) -> Any:
        return runner.TelemetrySummary(
            sample_count=2,
            duration_s=1.0,
            peak_cpu_percent=6.0,
            mean_cpu_percent=5.5,
            peak_target_rss_bytes=None,
            final_target_rss_bytes=None,
            peak_gpu_engine_percent=None,
            peak_gpu_memory_bytes=None,
            monitor_overhead_ms_mean=1.0,
            thermal_note="unavailable: test",
        )


def _fake_snapshot(path: Path, *, target_pid: int | None, settings: Any) -> dict[str, Any]:
    payload = {
        "process_inventory_count": 100,
        "cpu_percent_total": 5.0,
        "available_ram_bytes": 1000,
        "memory_pressure": 0.2,
    }
    runner.atomic_write_json(path, payload)
    return payload


def test_run_stabilization_writes_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    run = runner.RunDirectory(tmp_path / "run")
    cfg = runner.parse_args(["--run-dir", str(tmp_path / "run"), "--cooldown-seconds", "1"])
    monkeypatch.setattr(runner, "TelemetryCollector", _FakeTelemetryCollector)
    monkeypatch.setattr(runner, "_write_machine_snapshot", _fake_snapshot)
    summary = runner.run_stabilization(
        run,
        cfg,
        label="pre_a_stabilization",
        before_snapshot=tmp_path / "nonexistent_before.json",
        after_snapshot=run.path("stabilization", "pre_a_after.json"),
        settings=_settings(),
        duration_s=0.2,
        dry_run=False,
    )
    assert summary["label"] == "pre_a_stabilization"
    assert summary["requested_seconds"] == 0.2
    assert summary["actual_seconds"] >= 0.2
    assert summary["completed_successfully"] is True
    # 0.2 s is below the 20-minute default, so it is recorded as shortened.
    assert summary["shortened"] is True
    assert summary["process_count_evolution"]["sample_count"] == 2
    assert (tmp_path / "run" / "stabilization" / "pre_a_stabilization_summary.json").is_file()
    assert (tmp_path / "run" / "stabilization" / "pre_a_stabilization_telemetry.jsonl").is_file()


def test_run_stabilization_dry_run_is_not_marked_shortened(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run = runner.RunDirectory(tmp_path / "run")
    cfg = runner.parse_args(["--run-dir", str(tmp_path / "run"), "--cooldown-seconds", "1"])
    monkeypatch.setattr(runner, "TelemetryCollector", _FakeTelemetryCollector)
    monkeypatch.setattr(runner, "_write_machine_snapshot", _fake_snapshot)
    summary = runner.run_stabilization(
        run,
        cfg,
        label="post_a_stabilization",
        before_snapshot=tmp_path / "nonexistent_before.json",
        after_snapshot=run.path("stabilization", "post_a_after.json"),
        settings=_settings(),
        duration_s=0.1,
        dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["shortened"] is False


def test_run_stabilization_default_duration_is_20_minutes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run = runner.RunDirectory(tmp_path / "run")
    cfg = runner.parse_args(["--run-dir", str(tmp_path / "run")])
    monkeypatch.setattr(runner, "TelemetryCollector", _FakeTelemetryCollector)
    monkeypatch.setattr(runner, "_write_machine_snapshot", _fake_snapshot)
    # Do not actually wait 20 minutes: replace the sleep loop's duration with a fast
    # value by patching run_stabilization's requested duration is not possible here, so
    # assert the configured default instead (used by execute for the real run).
    assert cfg.cooldown_s == runner.STABILIZATION_SECONDS_DEFAULT
    assert runner.STABILIZATION_SECONDS_DEFAULT == 20 * 60.0


# --------------------------------------------------------------------------- #
# execute() ordering and fail-closed behaviour (with lightweight fakes)         #
# --------------------------------------------------------------------------- #
def _install_execute_fakes(
    monkeypatch: Any,
    tmp_path: Path,
    *,
    pre_a_cleanup_ok: bool = True,
    post_a_cleanup_ok: bool = True,
    pre_a_stab_completed: bool = True,
    post_a_stab_completed: bool = True,
) -> list[str]:
    events: list[str] = []

    def fake_ensure(cfg: Any) -> Path:
        return Path("model.gguf")

    def fake_preflight(cfg: Any, settings: Any) -> Any:
        return runner.PreflightReport(
            passed=True,
            blockers=[],
            warnings=[],
            setup_commands=[],
            data={"system_state": {"process_count": 200}},
        )

    def fake_acquire(cfg: Any) -> tuple[Path, dict[str, Any]]:
        return Path("model.gguf"), {"sha256": "abc", "size_bytes": 1}

    def fake_find_server(llama_dir: Path) -> Path:
        return Path("llama-server.exe")

    def fake_snapshot_processes() -> list[Any]:
        return []

    def fake_cleanup(run_arg: Any, **kwargs: Any) -> dict[str, Any]:
        label = kwargs["label"]
        events.append(f"{label}_cleanup")
        ok = pre_a_cleanup_ok if label == "pre_a" else post_a_cleanup_ok
        return {
            "label": label,
            "cleanup_success": ok,
            "remaining_inference": [],
            "remaining_owned": [],
            "cleanup_actions": [],
            "unexpected_survivors": [],
            "checks": {},
        }

    def fake_stabilization(run_arg: Any, cfg: Any, **kwargs: Any) -> dict[str, Any]:
        label = kwargs["label"]
        events.append(label)
        completed = (
            pre_a_stab_completed if label == "pre_a_stabilization" else post_a_stab_completed
        )
        return {**_stabilization(completed=completed), "label": label}

    def fake_write_snapshot(path: Path, *, target_pid: int | None, settings: Any) -> dict[str, Any]:
        payload = {"process_inventory_count": 100}
        runner.atomic_write_json(path, payload)
        return payload

    def fake_test_a(cfg: Any, run_arg: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("test_a")
        return {
            "test": "A",
            "outcomes": [_outcome(test="A", index=i, decode_tps=10.0) for i in range(5)],
            "telemetry": {"duration_s": 10.0},
            "server_pid": 111,
            "startup_s": 1.0,
            "stop": {"exited": True},
            "machine_after_process_count": 100,
            "command": ["llama-server"],
        }

    def fake_test_b(cfg: Any, run_arg: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("test_b")
        return {
            "test": "B",
            "outcomes": [_outcome(test="B", index=i, decode_tps=10.0) for i in range(5)],
            "telemetry": {"duration_s": 10.0, "monitor_overhead_ms_mean": 5.0},
            "intervention_count": 0,
            "intervention_end": None,
            "recovery_count": 0,
            "monitor_termination": "duration_reached",
        }

    def fake_final_cleanup(run_arg: Any) -> dict[str, Any]:
        events.append("final_cleanup")
        return {"cleanup_success": True, "inference_process_survivors": []}

    monkeypatch.setattr(runner, "ensure_model_available", fake_ensure)
    monkeypatch.setattr(runner, "collect_preflight", fake_preflight)
    monkeypatch.setattr(runner, "acquire_model", fake_acquire)
    monkeypatch.setattr(runner, "find_server", fake_find_server)
    monkeypatch.setattr(runner, "snapshot_processes", fake_snapshot_processes)
    monkeypatch.setattr(runner, "run_cleanup", fake_cleanup)
    monkeypatch.setattr(runner, "run_stabilization", fake_stabilization)
    monkeypatch.setattr(runner, "_write_machine_snapshot", fake_write_snapshot)
    monkeypatch.setattr(runner, "run_test_a", fake_test_a)
    monkeypatch.setattr(runner, "run_test_b", fake_test_b)
    monkeypatch.setattr(runner, "run_final_cleanup", fake_final_cleanup)
    monkeypatch.setattr(runner, "load_model_profile", lambda path: object())
    return events


def _execute_cfg(tmp_path: Path) -> Any:
    return runner.parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--rounds",
            "1",
            "--cooldown-seconds",
            "1200",
        ]
    )


def test_execute_runs_stages_in_required_order(tmp_path: Path, monkeypatch: Any) -> None:
    events = _install_execute_fakes(monkeypatch, tmp_path)
    exit_code = runner.execute(_execute_cfg(tmp_path))
    assert exit_code == 0
    assert events == [
        "pre_a_cleanup",
        "pre_a_stabilization",
        "test_a",
        "post_a_cleanup",
        "post_a_stabilization",
        "test_b",
        "final_cleanup",
    ]


def test_pre_a_cleanup_failure_prevents_stabilization_and_a(
    tmp_path: Path, monkeypatch: Any
) -> None:
    events = _install_execute_fakes(monkeypatch, tmp_path, pre_a_cleanup_ok=False)
    exit_code = runner.execute(_execute_cfg(tmp_path))
    assert exit_code == 5
    assert events == ["pre_a_cleanup"]


def test_a_cannot_start_before_pre_a_stabilization(tmp_path: Path, monkeypatch: Any) -> None:
    events = _install_execute_fakes(monkeypatch, tmp_path, pre_a_stab_completed=False)
    exit_code = runner.execute(_execute_cfg(tmp_path))
    assert exit_code == 6
    assert "test_a" not in events
    assert events == ["pre_a_cleanup", "pre_a_stabilization"]


def test_post_a_cleanup_failure_prevents_stabilization_and_b(
    tmp_path: Path, monkeypatch: Any
) -> None:
    events = _install_execute_fakes(monkeypatch, tmp_path, post_a_cleanup_ok=False)
    exit_code = runner.execute(_execute_cfg(tmp_path))
    assert exit_code == 4
    assert "test_b" not in events
    assert "post_a_stabilization" not in events
    assert events == ["pre_a_cleanup", "pre_a_stabilization", "test_a", "post_a_cleanup"]


def test_b_cannot_start_before_post_a_stabilization(tmp_path: Path, monkeypatch: Any) -> None:
    events = _install_execute_fakes(monkeypatch, tmp_path, post_a_stab_completed=False)
    exit_code = runner.execute(_execute_cfg(tmp_path))
    assert exit_code == 6
    assert "test_b" not in events
    assert events == [
        "pre_a_cleanup",
        "pre_a_stabilization",
        "test_a",
        "post_a_cleanup",
        "post_a_stabilization",
    ]
