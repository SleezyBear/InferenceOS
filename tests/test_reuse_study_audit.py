"""Full-path adversarial contracts: parser composition, evidence, publication and ownership."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from inferenceos.tune import reuse_correctness as r
from inferenceos.tune.prompt_reuse import TrialResult, _sha, evaluate, swa_cache_evidence, work_evidence
from inferenceos.tune.prompt_reuse_sim import SimulatedReuseEngine

PAIRS = r.build_pairs()


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p.pair_id)
@pytest.mark.parametrize("prefix", ["negative , ", "Negative : ", "minus ; ", "positive , ",
                                    "( negative ) ", "negative ) ", "negative ] ", "negative } ",
                                    "negative\t,\t", "negative , **"])
def test_sign_boundary_composition(pair: Any, prefix: str) -> None:
    value = pair.expected + ("**" if prefix.endswith("**") else "")
    assert r.judge(f"The reading is {prefix}{value} degrees.", pair)[0] == r.UNSCORABLE


@pytest.mark.parametrize("word", ["\u017fixty", "fifty-\u017feven", "S\u0130X", "F\u0130FTY-SEVEN", "\u017feven", "\u017fix"])
def test_unknown_number_spellings_never_raise_or_disappear(word: str) -> None:
    for text in (word, word + " 57 degrees.", word + ":\n57 degrees."):
        assert r.judge(text, PAIRS[0])[0] == r.UNSCORABLE


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p.pair_id)
def test_units_compose_with_formatting_but_operators_do_not(pair: Any) -> None:
    for unit in ("%", "\u00b0", "\u00b0C", "\u00b0F"):
        for mark in ("**", "_", "`"):
            assert r.judge(f"{pair.expected} {mark}{unit}{mark}.", pair)[0] == r.RIGHT
    for op in ("(*)", "(**)", "\u2260", "+", "\u2012"):
        assert r.judge(f"{op} , {pair.expected} degrees.", pair)[0] == r.UNSCORABLE
    assert r.judge(f"negative ( {pair.expected} ) degrees.", pair)[0] == r.WRONG


def test_numeric_parser_limits_and_lead_in_scope() -> None:
    assert r.judge("9" * 5000, PAIRS[0])[0] == r.UNSCORABLE
    assert r.judge("A positive example follows:\n57 degrees.", PAIRS[0])[0] == r.RIGHT
    assert r.judge("The answer is negative:\n57 degrees.", PAIRS[0])[0] == r.UNSCORABLE
    assert r.judge("FIFTY-SEVEN degrees.", PAIRS[0])[0] == r.RIGHT
    for text in ("56 or 58 degrees.", "56, 58 degrees.", "5 , 7 degrees.", "57 , million degrees.", "57 , M degrees."):
        assert r.judge(text, PAIRS[0])[0] == r.UNSCORABLE


@pytest.mark.parametrize("hidden", ["\u200b", "\u200d", "\u202e", "\u2066", "\u00ad", "\ufeff", "\0"])
def test_hidden_control_characters_cannot_hide_signs_or_lead_in_numbers(hidden: str) -> None:
    for pair in PAIRS:
        for text in (f"{hidden}negative {pair.expected} degrees.",
                     f"{hidden}Negative:\n{pair.expected} degrees."):
            assert r.judge(text, pair)[0] == r.UNSCORABLE


def native_record() -> dict[str, Any]:
    return evaluate(SimulatedReuseEngine(oracle=r.answer_oracle(PAIRS), sliding_window=True),
                    cases=(PAIRS[0].case,), variants=(r.DEFAULT, r.SWA_FULL), blocks=3,
                    context_length=4096, configuration={}, run_id="audit", power_check=False, cancellation=False)


@pytest.mark.parametrize("scope", ["blocks", "variants"])
def test_request_identity_cannot_change_between_comparisons(scope: str) -> None:
    data = native_record()
    for trial in data["trials"]:
        changed = trial["block"] == 2 if scope == "blocks" else trial["variant_id"] == "swa-full"
        if changed:
            trial["identity"]["configuration_sha256"] = "another setup"
            trial["identity_sha256"] = _sha(trial["identity"])
    summary = r.correctness_summary([TrialResult(**t) for t in data["trials"]], PAIRS[:1])
    assert not summary["swa_full_comparison"]["interpretable"]
    if scope == "blocks":
        assert all(not row["usable"] for item in summary["pairs"].values() for row in item["blocks"])
    else:
        assert "identity" in summary["swa_full_comparison"]["not_comparable"][PAIRS[0].pair_id]


@pytest.mark.parametrize("problem", ["breach", "cleanup", "power", "identity", "configuration", "fresh_reuse", "interrupted"])
def test_native_integrity_gates_are_independent_of_problem_strings(problem: str) -> None:
    data = native_record()
    t = data["trials"][0]
    if problem == "breach":
        t["breach"] = "E4: injected validity failure"
    elif problem == "cleanup":
        t["cleanup"] = {}
    elif problem == "power":
        t["power"] = []
    elif problem == "identity":
        t["identity"]["case_sha256"] = "other workload"
    elif problem == "configuration":
        t["identity"]["configuration_sha256"] = "other setup"
        t["identity_sha256"] = _sha(t["identity"])
    elif problem == "fresh_reuse":
        t["measured"]["cached"] = 1
        t["measured"]["processed"] = t["measured"]["total"] - 1
    else:
        t["interrupted"] = True
    untouched = deepcopy(data)
    summary = r.correctness_summary([TrialResult(**x) for x in data["trials"]], PAIRS[:1])
    assert not summary["pairs"]["default/" + PAIRS[0].pair_id]["blocks"][0]["usable"]
    if problem in ("breach", "cleanup", "fresh_reuse", "interrupted"):
        # Even two earlier complete blocks cannot bless an experiment that hit a stop.
        assert not summary["swa_full_comparison"]["interpretable"]
    assert data == untouched


@pytest.mark.parametrize("problem", ["role", "cache_prompt", "cleanup", "clean_flag", "power", "duplicate", "interrupted"])
def test_chat_integrity_cannot_create_an_absence_claim(problem: str) -> None:
    trials, _, _ = r.run_chat_exposure(SimulatedReuseEngine(oracle=r.answer_oracle(PAIRS)), PAIRS[:1],
                                     blocks=1, power=lambda: "not checked", power_check=False)
    t = trials[0]
    if problem == "role":
        t["client"][1]["role"] = "prime"
    elif problem == "cache_prompt":
        t["matched"][1]["cache_prompt"] = None
    elif problem == "cleanup":
        t["cleanup"] = []
    elif problem == "clean_flag":
        t["cleanup"][0]["clean"] = "true"
    elif problem == "power":
        t["power"] = []
    elif problem == "duplicate":
        trials.append(deepcopy(t))
    else:
        t["interrupted"] = True
    untouched = deepcopy(trials)
    row = r.chat_summary(trials, PAIRS[:1])["pairs"][PAIRS[0].pair_id]
    assert row["outcome"] == "not usable" and row["usable"] == 0
    assert trials == untouched


@pytest.mark.parametrize("timings", [[], "bad", None, {}])
def test_reported_malformed_chat_timings_are_not_discarded(timings: Any) -> None:
    payload = {"choices": [{"message": {"content": "57 degrees."}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 0}},
               "timings": timings}
    assert r.chat_evidence(asdict(r.chat_reading(payload)))[1]


def test_swa_mode_needs_a_complete_log_and_real_numeric_time() -> None:
    log = "creating non-SWA KV cache, size = 4096 cells\ncreating SWA KV cache, size = 3072 cells"
    assert swa_cache_evidence(log, requested_full=False)["state"].startswith("unverified")
    assert swa_cache_evidence(log + "\nsrv model loaded", requested_full=False)["state"] == "reduced"
    trial = native_record()["trials"][0]
    reading = trial["measured"]
    reading["ttft_s"] = True
    assert work_evidence(reading, expected_total=trial["measured_tokens"], expected_generated=24)[1]
    reading["ttft_s"] = 1.0
    for raw in ({"predicted_n": 24, "tokens_predicted": 23}, {"predicted_n": True, "tokens_predicted": 24}):
        reading["engine_counters"] = raw
        assert work_evidence(reading, expected_total=trial["measured_tokens"], expected_generated=24)[1]


def test_all_reported_chat_work_counters_must_agree() -> None:
    reply = {"text": "57 degrees.", "finish_reason": "stop", "prompt_tokens": 10,
             "cached": 0, "completion_tokens": 3, "timings": {"prompt_n": 10, "cache_n": 0, "predicted_n": 2}}
    assert r.chat_evidence(reply)[1]
    payload = {"choices": [{"message": {"content": "57"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 99,
                         "prompt_tokens_details": {"cached_tokens": 0}}}
    assert r.chat_evidence(asdict(r.chat_reading(payload)))[1]


def test_runner_refuses_invalid_fixed_documents_before_start(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _script
    from inferenceos.tune.reuse_correctness import build_pairs

    with pytest.raises(ValueError):
        build_pairs(11)
    for args in (["--sections", "11"], ["--blocks", "0"]):
        out = tmp_path / (args[0][2:] + ".json")
        assert _script().main(["--simulate", "--study", "correctness", "--out", str(out), *args]) == 4
        assert not out.exists()


def test_runner_pins_rules_in_every_checkpoint(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _script

    rules = tmp_path / "rules.txt"
    rules.write_text(r.CORRECTNESS_VERSION, encoding="utf-8")
    sha = hashlib.sha256(rules.read_bytes()).hexdigest()
    args = ["--simulate", "--study", "correctness", "--blocks", "1", "--preregistration", str(rules)]
    bad = tmp_path / "bad.json"
    assert _script().main([*args, "--out", str(bad), "--preregistration-sha256", "wrong"]) == 4
    assert not bad.exists()
    out = tmp_path / "good.json"
    assert _script().main([*args, "--out", str(out), "--preregistration-sha256", sha]) == 0
    record = json.loads(out.read_text())
    assert record["preregistration"]["sha256"] == sha


def test_concurrent_runners_cannot_share_a_results_file(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_prompt_reuse.py"
    out = tmp_path / "race.json"

    def run(name: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-B", str(script), "--simulate", "--study", "correctness",
                               "--blocks", "1", "--out", str(out), "--run-id", name],
                              capture_output=True, text=True, timeout=120)

    with ThreadPoolExecutor(2) as pool:
        a, b = list(pool.map(run, ("first", "second")))
    assert sorted((a.returncode, b.returncode)) == [0, 4]
    assert json.loads(out.read_text())["run_id"] == ("first" if a.returncode == 0 else "second")


def test_failed_publication_does_not_swallow_an_interrupt(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _script

    class Interrupted(SimulatedReuseEngine):
        def generate(self, *args: Any, **kwargs: Any) -> Any:
            raise KeyboardInterrupt

    engine = Interrupted()
    script = _script()
    with patch("inferenceos.tune.prompt_reuse_sim.SimulatedReuseEngine", return_value=engine), \
            patch.object(script, "_write", side_effect=OSError("disk full")), pytest.raises(KeyboardInterrupt):
        script.main(["--simulate", "--study", "correctness", "--blocks", "1", "--out", str(tmp_path / "r.json")])
    assert engine.starts == engine.stops == 1


def test_interrupted_native_checkpoints_keep_the_study_identity(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _script

    class Interrupted(SimulatedReuseEngine):
        def generate(self, *args: Any, **kwargs: Any) -> Any:
            raise KeyboardInterrupt

    engine = Interrupted()
    out = tmp_path / "interrupt.json"
    with patch("inferenceos.tune.prompt_reuse_sim.SimulatedReuseEngine", return_value=engine), pytest.raises(KeyboardInterrupt):
        _script().main(["--simulate", "--study", "correctness", "--blocks", "1", "--out", str(out)])
    record = json.loads(out.read_text())
    assert record["status"] == "interrupted" and record["study"]["version"] == r.CORRECTNESS_VERSION
    assert len(record["trials"]) == 1 and engine.starts == engine.stops == 1


@pytest.mark.parametrize("chat", [False, True])
def test_only_the_final_publication_can_claim_a_completed_correctness_study(tmp_path: Path, chat: bool) -> None:
    from tests.test_prompt_reuse import _script

    script = _script()
    captured: list[dict[str, Any]] = []
    write = script._write

    def capture(path: Path, record: dict[str, Any]) -> None:
        captured.append(deepcopy(record))
        write(path, record)

    out = tmp_path / "phases.json"
    with patch.object(script, "_write", side_effect=capture):
        assert script.main(["--simulate", "--study", "correctness", "--blocks", "1", "--out", str(out),
                            *(["--chat-exposure"] if chat else [])]) == 0
    assert captured[-1]["status"] == "done" and "correctness" in captured[-1]
    assert all(record["status"] == "running" for record in captured[:-1])


def test_an_interrupt_during_final_input_validation_cannot_publish_done(tmp_path: Path) -> None:
    import tests.test_prompt_reuse as helpers

    script = helpers._script()
    real_identity = script._file_identity
    calls = 0

    def identity(path: Path) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return real_identity(path)

    out = tmp_path / "postcheck-interrupt.json"
    with patch.object(helpers, "_script", return_value=script), \
            patch.object(script, "_file_identity", side_effect=identity), pytest.raises(KeyboardInterrupt):
        helpers._real_mode(tmp_path, out)
    assert json.loads(out.read_text())["status"] == "interrupted"


def test_an_unsupported_reply_keeps_the_entire_study_running(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _script

    class UnicodeReply(SimulatedReuseEngine):
        def generate(self, *args: Any, **kwargs: Any) -> Any:
            reading = super().generate(*args, **kwargs)
            if kwargs["cache_prompt"] and self._requests == 2:
                reading.text = "\u017fixty degrees."
            return reading

    engine = UnicodeReply(oracle=r.answer_oracle(PAIRS))
    out = tmp_path / "unicode.json"
    with patch("inferenceos.tune.prompt_reuse_sim.SimulatedReuseEngine", return_value=engine):
        assert _script().main(["--simulate", "--study", "correctness", "--chat-exposure", "--blocks", "2",
                               "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert len(record["trials"]) == 36 and len(record["chat_trials"]) == 18
    assert {p["partial_reuse_evidence"] for p in record["correctness"]["pairs"].values()} == {r.EVIDENCE_INCONCLUSIVE}


def test_a_stale_exists_check_cannot_overwrite_other_evidence(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _script

    out = tmp_path / "claimed.json"
    out.write_bytes(b"existing evidence")
    original = Path.exists
    with patch.object(Path, "exists", lambda p: False if p == out else original(p)):
        assert _script().main(["--simulate", "--study", "correctness", "--blocks", "1", "--out", str(out)]) == 4
    assert out.read_bytes() == b"existing evidence"


def test_model_bytes_are_identified_and_changes_refuse_a_completed_run(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _real_mode

    class Changed(SimulatedReuseEngine):
        def generate(self, *args: Any, **kwargs: Any) -> Any:
            reply = super().generate(*args, **kwargs)
            (tmp_path / "dummy.gguf").write_bytes(b"different weights")
            return reply

    out = tmp_path / "changed.json"
    code, _ = _real_mode(tmp_path, out, engine=Changed())
    record = json.loads(out.read_text())
    assert code == 6 and record["model_file_unchanged"] is False
    assert record["model_file_identity"]["sha256"] == hashlib.sha256(b"not a model").hexdigest()


def test_engine_binary_changes_stop_before_another_measurement(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from tests.test_prompt_reuse import _real_engine

    exe = tmp_path / "not-executed.bin"
    exe.write_bytes(b"first build")

    class Backend:
        active = False
        last_command = [str(exe)]
        undeleted_key_files: tuple[Any, ...] = ()
        unconfirmed_pid = None
        api_key_file = None

        def start(self, *args: Any) -> None:
            self.active = True

        def is_active(self) -> bool:
            return self.active

        def stop(self) -> Any:
            self.active = False
            return SimpleNamespace(cleanup_complete=True, credential_removed=True)

    engine = _real_engine(Backend)
    engine.start(r.DEFAULT)
    assert engine.stop().clean
    exe.write_bytes(b"different executable build")
    with pytest.raises(RuntimeError, match="changed"):
        engine.start(r.DEFAULT)
    assert engine.stop().clean and engine.input_problem
    assert len(engine.engine_binaries) == 2
    with pytest.raises(RuntimeError, match="changed"):
        engine.start(r.DEFAULT)
    assert not engine._backend.is_active()

    class Changed(SimulatedReuseEngine):
        input_problem = "the engine executable changed during the study"

    simulated = Changed()
    record = evaluate(simulated, cases=(PAIRS[0].case,), variants=(r.DEFAULT,), blocks=3,
                      context_length=4096, configuration={}, run_id="changed", power_check=False)
    assert record["status"] == "error" and simulated.starts == simulated.stops == 1


def test_changed_engine_input_in_chat_stops_all_further_dispatch() -> None:
    class Changed(SimulatedReuseEngine):
        input_problem: str | None = None

        def start(self, variant: Any) -> None:
            super().start(variant)
            self.input_problem = "engine executable changed"
            raise RuntimeError(self.input_problem)

    engine = Changed(oracle=r.answer_oracle(PAIRS))
    trials, status, reason = r.run_chat_exposure(engine, PAIRS, blocks=2,
                                               power=lambda: "not checked", power_check=False)
    assert status == "error" and reason == "engine executable changed"
    assert len(trials) == 1 and engine.starts == engine.stops == 1


@pytest.mark.parametrize("exception", [OSError("no executable"), KeyboardInterrupt()])
def test_launch_failure_releases_the_new_guard(exception: BaseException) -> None:
    from inferenceos.backends import llama_server_toolchain as srv

    class Guard:
        closed = False

        def close(self) -> None:
            self.closed = True

    guard = Guard()
    process = srv.LlamaServerProcess(["not-run"])
    with patch.object(srv, "create_process_guard", return_value=guard), \
            patch.object(srv.subprocess, "Popen", side_effect=exception):
        with pytest.raises((srv.ServerStartError, KeyboardInterrupt)):
            process.start()
    assert guard.closed and process.pid is None


def test_failed_containment_never_returns_a_usable_server() -> None:
    from inferenceos.backends import llama_server_toolchain as srv

    guard = MagicMock()
    guard.assign.return_value = False
    child = MagicMock(pid=4242)
    tree = MagicMock()
    process = srv.LlamaServerProcess(["not-run"])
    with patch.object(srv, "create_process_guard", return_value=guard), \
            patch.object(srv.subprocess, "Popen", return_value=child), \
            patch.object(srv, "PsutilProcessGuard", return_value=tree) as owner:
        with pytest.raises(srv.ServerStartError, match="containment"):
            process.start()
        owner.assert_called_once_with(4242)
    guard.close.assert_called_once()
    assert process._guard is tree  # retained for the backend's existing failed-start unwind


@pytest.mark.parametrize("tokens", [[True], [-1], [1.5], ["1"]])
def test_tokenizer_ids_are_not_coerced_into_valid_work(tokens: list[Any]) -> None:
    from inferenceos.backends import llama_server_toolchain as srv

    client = srv.LlamaServerClient(host="127.0.0.1", port=1, api_key="test")
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps({"tokens": tokens}).encode()
    with patch.object(client, "_open", return_value=response), pytest.raises(srv.ServerRequestError):
        client.tokenize("no request sent")


def test_windows_guard_uses_pointer_sized_handles() -> None:
    from inferenceos.backends import process_control as pc
    from ctypes import wintypes

    if sys.platform != "win32":
        pytest.skip("native Windows ABI")
    guard = pc.WindowsJobObject()
    try:
        assert guard._kernel32.CreateJobObjectW.restype is wintypes.HANDLE
        assert guard._kernel32.OpenProcess.restype is wintypes.HANDLE
    finally:
        guard.close()
