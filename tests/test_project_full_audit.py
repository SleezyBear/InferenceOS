"""Whole-project boundary regressions: interactions, inputs, persistence and execution."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch
from types import SimpleNamespace

import pytest

from inferenceos.contracts.enums import ParameterType
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.optimization import ParameterSpec
from inferenceos.deployment import validate_deployment, write_deployment_state
from inferenceos.experiment.executor import validate_parameters
from inferenceos.optimization.actions import build_default_registry
from inferenceos.tune.profiles import model_identity
from tests.experiment_helpers import SampleResult
from tests.test_coordinator import ScriptedTuner, _drive, _kept, _layer_round, _policy
from tests.test_deployment_state import _profile, _settings, _state
from tests.test_tune import _settings as tune_settings, _tuner


def test_a_later_layer_change_reconsiders_finished_engine_switches() -> None:
    tuner = ScriptedTuner("engine_switches,kv_cache", path_round=_kept("kv_cache_type_k", "q8_0"),
                          layer_round=_layer_round(13))
    policy, _, _ = _policy(tuner)
    _drive(policy)
    assert [c[2]["gpu_offload_layers"] for c in tuner.calls
            if c[0] == "path" and c[1] == "engine_switches"] == [8, 13]


@pytest.mark.parametrize("budget", [0, 1200])
def test_changed_model_cannot_become_the_identity_of_approved_old_measurements(tmp_path: Path, budget: int) -> None:
    tuner: Any = None
    changed = False

    def response(task: Any, options: Any, call: int) -> SampleResult:
        nonlocal changed
        if not changed:
            changed = True
            tuner.model_path.write_bytes(b"GGUF a different model after measurement began")
        return SampleResult(decode_tps=24.0, total_latency_s=1.7) if (options.gpu_offload_layers or 0) >= 20 else SampleResult()

    tuner, _, _, store = _tuner(tmp_path, response, answers=[True] * 8, settings=tune_settings(
        tmp_path, tune_layer_path=False, tune_paths="", tune_heat_check_runs=0,
        tune_offer_tradeoffs=False, tune_experimental_budget_s=budget))
    with pytest.raises(RuntimeError, match="model.*changed"):
        tuner.run()
    assert store.all() == []


def test_model_identity_refuses_a_file_changing_while_it_is_hashed(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"before")

    def changed(path: Path, **kwargs: Any) -> str:
        path.write_bytes(b"changed during hashing")
        return "digest-from-the-wrong-generation"

    with patch("inferenceos.tune.profiles.file_sha256", side_effect=changed), \
            pytest.raises(RuntimeError, match="model.*changed"):
        model_identity(model)


def test_deployment_accepts_the_separate_output_layer(tmp_path: Path) -> None:
    settings, profile = _settings(tmp_path), _profile(tmp_path)
    config = RuntimeConfiguration(context_length=1024, gpu_offload_layers=profile.n_layers + 1)
    with patch("inferenceos.deployment.cli_version", return_value="fake"):
        result = validate_deployment(_state(profile, final_configuration=config),
                                     settings=settings, model_profile=profile)
    assert result.valid, result.reasons


def test_failed_deployment_publication_keeps_previous_state_bytes(tmp_path: Path) -> None:
    state = _state(_profile(tmp_path))
    path = tmp_path / "deployment.json"
    path.write_bytes(b"previous owner-approved deployment")
    with patch("inferenceos.storage.os.replace", side_effect=PermissionError("publication denied")), \
            pytest.raises(PermissionError, match="publication denied"):
        write_deployment_state(path, state)
    assert path.read_bytes() == b"previous owner-approved deployment"


@pytest.mark.parametrize("kind", [ParameterType.INTEGER, ParameterType.FLOAT])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_action_values_are_rejected_without_raising(kind: ParameterType, value: float) -> None:
    action = build_default_registry().get("runtime.reduce_threads").model_copy(update={
        "parameters": [ParameterSpec(name="threads", type=kind, minimum=1, maximum=8)]})
    assert validate_parameters(action, {"threads": value})[0] is False


@pytest.mark.parametrize("value", [-0.1, 9.0])
def test_float_action_values_obey_registered_bounds(value: float) -> None:
    action = build_default_registry().get("runtime.reduce_threads").model_copy(update={
        "parameters": [ParameterSpec(name="threads", type=ParameterType.FLOAT, minimum=1, maximum=8)]})
    assert validate_parameters(action, {"threads": value})[0] is False


def test_empty_parameters_do_not_claim_an_action_was_applied() -> None:
    action = build_default_registry().get("runtime.reduce_threads")
    assert validate_parameters(action, {})[0] is False


@pytest.mark.parametrize("kind", [ParameterType.INTEGER, ParameterType.FLOAT])
def test_huge_integer_action_values_are_rejected_by_the_bounds(kind: ParameterType) -> None:
    action = build_default_registry().get("runtime.reduce_threads").model_copy(update={
        "parameters": [ParameterSpec(name="threads", type=kind, minimum=1, maximum=8)]})
    assert validate_parameters(action, {"threads": 10 ** 400})[0] is False


@pytest.mark.parametrize("state", ["active", "pid", "key", "unknown"])
def test_release_requires_actual_process_and_credential_evidence(state: str) -> None:
    from inferenceos.backends.release import release_evidence

    backend = SimpleNamespace(is_running=lambda: state == "active",
                              unconfirmed_pid=91 if state == "pid" else None,
                              api_key_file=Path("retained.txt") if state == "key" else None,
                              undeleted_key_files=[])
    if state == "unknown":
        backend.is_running = lambda: None
    assert not release_evidence(backend, stopped=SimpleNamespace(cleanup_complete=True)).clean


def test_cleanup_evidence_does_not_invent_a_credential_failure_for_an_old_adapter() -> None:
    from inferenceos.backends.release import release_evidence

    evidence = release_evidence(SimpleNamespace(is_running=lambda: False))
    assert evidence.clean and evidence.credential_removed is None


def test_failed_start_releases_an_older_adapter_with_only_stop(tmp_path: Path) -> None:
    from inferenceos.tune.start import serve_until_stopped
    from tests.test_project_review_fixes import _setup
    from tests.test_start import FakeBackend
    from inferenceos.selftest import sample_model

    class PartialStart(FakeBackend):
        def start(self, model: Any, params: Any) -> object:
            super().start(model, params)
            raise RuntimeError("failed after acquiring resources")

    backend = PartialStart()
    result = serve_until_stopped(setup=_setup(tmp_path), model=sample_model(), backend=backend)
    assert not result.started and result.stopped_cleanly
    assert backend.stops == 1 and backend.api_key_file is None


@pytest.mark.parametrize("seen", [None, 4])
def test_recovery_never_verifies_the_request_against_itself(seen: int | None) -> None:
    from inferenceos.interfaces.backend_adapter import BackendTelemetry
    from tests.test_resident_recovery import RecoverableFakeAdapter, _workload, _provision

    class WrongEngine(RecoverableFakeAdapter):
        def telemetry(self) -> BackendTelemetry:
            return BackendTelemetry(gpu_layers_offloaded=seen,
                                    context_length=2048, observed={"context_length": 2048})

    adapter = WrongEngine()
    workload = _workload(adapter)
    _provision(workload)
    adapter.kill()
    result = workload.recover_target(reason="process_exited")
    assert not result.configuration_verified and not result.recovered
    assert not adapter.is_loaded


def test_recovery_does_not_load_over_resources_whose_cleanup_failed() -> None:
    from tests.test_resident_recovery import RecoverableFakeAdapter, _workload, _provision

    class Retained(RecoverableFakeAdapter):
        def release_evidence(self) -> Any:
            return SimpleNamespace(clean=False)

    adapter = Retained()
    workload = _workload(adapter)
    _provision(workload)
    adapter.kill()
    calls = adapter.load_calls
    result = workload.recover_target(reason="process_exited")
    assert not result.recovered and not result.cleanup_complete
    assert adapter.load_calls == calls


def test_failed_new_restoration_capture_cannot_reuse_an_earlier_success() -> None:
    from tests.test_restoration_verifier import _FakeAdapter, _verifier

    adapter = _FakeAdapter()
    verifier = _verifier(adapter)
    config = RuntimeConfiguration(context_length=2048)
    verifier.capture(config)
    adapter.fail = True
    verifier.capture(config)
    adapter.fail = False
    assert not verifier.verify(config).verified


@pytest.mark.parametrize("credential", [False, None])
def test_reasoner_incomplete_unload_cannot_launch_the_target(credential: bool | None) -> None:
    from tests.test_resident_lifecycle import _setup, _runner

    _, diagnosis, report, model, params, target, reasoner = _setup()
    original = reasoner.stop

    def incomplete() -> Any:
        result = original()
        return result.model_copy(update={"cleanup_complete": credential is False,
                                         "credential_removed": credential})

    with patch.object(reasoner, "stop", side_effect=incomplete):
        outcome = _runner(target, model, params, reasoner).run(diagnosis, report=report)
    assert outcome.errors and not target.is_running()
    assert target._starts == 1


@pytest.mark.parametrize("value", [True, -0.1, float("nan"), float("inf"), 10 ** 400])
def test_invalid_noise_never_reaches_measurement_planning(tmp_path: Path, value: Any) -> None:
    import json
    from inferenceos.tune.noise_store import NoiseStore

    path = tmp_path / "noise.json"
    path.write_text(json.dumps({"bad": [value, value], "good": [0.0, 0.02]}), encoding="utf-8")
    store = NoiseStore(path)
    assert store.estimate("bad") is None
    assert store.estimate("good") == 0.01
    before = path.read_bytes()
    store.add("bad", value)
    assert path.read_bytes() == before


def test_good_answers_cannot_average_away_a_bad_quality_run() -> None:
    from tests.test_quality_suite import _evaluate, REFERENCE
    from inferenceos.contracts.enums import QualityStatus

    result = _evaluate([REFERENCE] * 5 + ["A large purple elephant dances beside a house."])
    assert result.status is QualityStatus.QUALITY_REGRESSION


def test_good_answers_cannot_average_away_an_excessively_long_answer() -> None:
    from tests.test_quality_suite import _evaluate, REFERENCE
    from inferenceos.contracts.enums import QualityStatus

    result = _evaluate([REFERENCE] * 5 + [REFERENCE * 4])
    assert result.status is QualityStatus.QUALITY_REGRESSION


def test_unreadable_process_existence_is_not_proof_of_exit() -> None:
    from inferenceos.backends import process_control as pc

    with patch.object(pc.psutil, "pid_exists", side_effect=PermissionError("process query denied")):
        assert pc.process_exists(91) is True
        assert pc.wait_process_exit(91, timeout_s=0) is False


def test_unreadable_memory_is_not_proof_that_a_process_disappeared() -> None:
    from inferenceos.backends import process_control as pc

    with patch.object(pc.psutil, "Process", side_effect=PermissionError("memory query denied")), \
            patch.object(pc.psutil, "pid_exists", return_value=True):
        snapshot = pc.resource_snapshot(91)
    assert snapshot.exists and snapshot.rss_bytes is None


def test_gguf_impossible_string_length_is_a_metadata_error(tmp_path: Path) -> None:
    import struct
    from inferenceos.models.gguf import read_gguf, GgufError

    path = tmp_path / "invalid.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 2 ** 64 - 1))
    with pytest.raises(GgufError):
        read_gguf(path)


@pytest.mark.parametrize("capacity,contradicts", [(256, True), (1024, False), (1280, False)])
def test_requested_context_is_a_minimum_capacity_not_a_requested_echo(capacity: int, contradicts: bool) -> None:
    from inferenceos.backends.llama_cpp_toolchain import parse_effective_settings
    from inferenceos.backends.observed import observed_configuration
    from inferenceos.interfaces.backend_adapter import BackendTelemetry

    observed = parse_effective_settings(f"srv init: initializing, n_slots = 4, n_ctx_slot = {capacity}, kv_unified = 'true'")
    _, problems, unknown = observed_configuration(RuntimeConfiguration(context_length=1024),
        BackendTelemetry(context_length=1024, observed=observed))
    assert bool(problems) is contradicts
    assert unknown == []


def test_engine_cannot_publish_ready_with_a_smaller_requested_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.backends import llama_server_toolchain as srv
    from inferenceos.backends.resident_server import LlamaServerResidentBackend, ResidentBackendError
    from inferenceos.interfaces.backend_adapter import RuntimeParams
    from tests.test_resident_races_timeouts import Proc, Client, _model

    monkeypatch.setattr(srv, "LlamaServerClient", Client)
    monkeypatch.setattr(Proc, "log_text", lambda self: "initializing, n_slots = 4, n_ctx_slot = 256")
    backend = LlamaServerResidentBackend(server=Path("unused"), process_factory=Proc)  # type: ignore[arg-type]
    with pytest.raises(ResidentBackendError, match="below requested context"):
        backend.start(_model(tmp_path), RuntimeParams(context_length=1024))
    assert not backend.is_active() and backend.api_key_file is None
    assert not backend.undeleted_key_files


def test_outer_loop_cannot_continue_after_reasoner_release_failure() -> None:
    from inferenceos.loop.resident import ResidentDeepDiagnosisRunner
    from inferenceos.reasoning.resident_lifecycle import ResidentLifecycleError
    from inferenceos.contracts.lifecycle import LifecyclePolicy
    from tests.test_resident_lifecycle import _setup

    _, diagnosis, report, model, params, target, reasoner = _setup()
    reasoner.fail_stop = True
    runner = ResidentDeepDiagnosisRunner(backend=target, target_model=model, target_params=params,  # type: ignore[arg-type]
                                        reasoner=reasoner, policy=LifecyclePolicy())  # type: ignore[arg-type]
    with pytest.raises(ResidentLifecycleError, match="optimization stopped"):
        runner(diagnosis, report)
    assert target._starts == 1 and not target.is_running()


def test_failed_restore_cannot_claim_queued_requests_were_drained() -> None:
    from tests.test_resident_lifecycle import _setup, _runner, _submit
    from inferenceos.contracts.lifecycle import RequestPolicy

    _, diagnosis, report, model, params, target, reasoner = _setup()
    reasoner.fail_stop = True
    result = _runner(target, model, params, reasoner, request_policy=RequestPolicy.QUEUE,
                     on_diagnosis_window=_submit).run(diagnosis, report=report)
    assert result.requests and not any(request.drained for request in result.requests)
    assert not result.transitions[-1].success


def test_probe_cannot_abandon_a_backend_with_a_retained_password() -> None:
    from inferenceos.tune.engine_probe import run_probe, ProbeNotSafe
    from inferenceos.interfaces.backend_adapter import GenerationResult
    from inferenceos.config import Settings
    from inferenceos.selftest import sample_model

    class ProbeBackend:
        api_key_file = Path("undeleted-probe-password.txt")
        undeleted_key_files = [api_key_file]

        def start(self, model: Any, params: Any) -> None:
            pass

        def generate(self, request: Any) -> Any:
            return GenerationResult(text="OK")

        def server_log(self) -> str:
            return ""

        def is_running(self) -> bool:
            return False

        def stop(self) -> object:
            return SimpleNamespace(cleanup_complete=True, credential_removed=False)

    with pytest.raises(ProbeNotSafe):
        run_probe(settings=Settings(), model=sample_model(), config=RuntimeConfiguration(context_length=1024),
                  repeats=1, backend_factory=ProbeBackend)  # type: ignore[arg-type]


def test_unresolved_probe_stops_the_entire_tune_and_cannot_save(tmp_path: Path) -> None:
    from inferenceos.tune.engine_probe import ProbeNotSafe

    unsafe = False
    after_failure: list[int] = []
    error = ProbeNotSafe("probe process exit was not confirmed")

    def respond(task: Any, options: Any, call: int) -> SampleResult:
        if unsafe:
            after_failure.append(call)
        return SampleResult()

    tuner, _, _, store = _tuner(tmp_path, respond, answers=[], settings=tune_settings(
        tmp_path, tune_layer_path=False, tune_paths="threads", tune_diagnosis=True, tune_heat_check_runs=0))

    def fail(*args: Any, **kwargs: Any) -> Any:
        nonlocal unsafe
        unsafe = True
        raise error

    tuner.probe_runner = fail
    starting = RuntimeConfiguration(context_length=tuner.context_length, gpu_offload_layers=1)
    with patch.object(tuner, "_starting_point", return_value=(starting, "scripted on-card start")), \
            pytest.raises(ProbeNotSafe) as caught:
        tuner.run()
    assert caught.value is error and not after_failure and not store.all()


def test_command_writes_existing_failure_report_after_an_unsafe_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from inferenceos.tune.engine_probe import ProbeNotSafe
    from tests.test_coordinator import _run_tune

    tuner, _, engine_calls, out = _run_tune(tmp_path, monkeypatch, ProbeNotSafe("unreleased probe"))
    assert tuner._run_tune() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["saved_setup"] is None and report["error"]["type"] == "ProbeNotSafe"
    assert engine_calls.count("stop") >= 1


def test_nan_paging_cannot_pass_a_memory_safety_check() -> None:
    from inferenceos.tune.acceptance_evidence import safety_resource_evidence
    from inferenceos.tune.layer_top import LoadAttempt
    from tests.test_review7_acceptance import _measurement

    measurement = _measurement([0.0, 0.0])
    measurement = measurement.model_copy(update={"run_metrics": [
        run.model_copy(update={"hard_page_faults_per_sec": float("nan")}) for run in measurement.run_metrics]})
    result = safety_resource_evidence(measurement=measurement,
                                     reference=None, guarded=[], load_check=LoadAttempt(loaded=True, fits=True))
    assert not result.passed


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_metrics_cannot_become_a_scored_measurement(value: float) -> None:
    from inferenceos.contracts.observation import RunMetrics

    with pytest.raises(ValueError, match="finite and nonnegative"):
        RunMetrics(decode_tokens_per_sec=value)


@pytest.mark.parametrize("gain", [float("nan"), float("inf")])
def test_nonfinite_goal_gain_is_never_kept(gain: float) -> None:
    from inferenceos.contracts.goal import GoalAssessment
    from inferenceos.contracts.enums import Situation
    from inferenceos.config import Settings
    from inferenceos.optimization.keep import ModeRules, judge_assessment

    settings = Settings()
    rules = ModeRules.from_settings(settings)
    result = judge_assessment(GoalAssessment(gain=gain, measured=True), technically_better=True,
                              effort=rules.effort, goal=rules.goal, situation=Situation.BEFORE_LOADING,
                              settings=settings)
    assert not result.keep


@pytest.mark.parametrize("stage", ["log", "stop"])
def test_probe_cleanup_survives_interruptions_in_its_own_unwind(stage: str) -> None:
    from inferenceos.tune.engine_probe import run_probe
    from inferenceos.interfaces.backend_adapter import GenerationResult
    from inferenceos.config import Settings
    from inferenceos.selftest import sample_model

    error = KeyboardInterrupt(stage)

    class Backend:
        running = True
        stops = 0
        cleanups = 0

        def start(self, model: Any, params: Any) -> None:
            pass

        def generate(self, request: Any) -> Any:
            return GenerationResult(text="OK")

        def server_log(self) -> str:
            if stage == "log":
                raise error
            return ""

        def is_running(self) -> bool:
            return self.running

        def stop(self) -> None:
            self.stops += 1
            if stage == "stop":
                raise error
            self.running = False

        def cleanup(self) -> None:
            self.cleanups += 1
            self.running = False

    backend = Backend()
    with pytest.raises(KeyboardInterrupt) as caught:
        run_probe(settings=Settings(), model=sample_model(), config=RuntimeConfiguration(context_length=1024),
                  repeats=1, backend_factory=lambda: backend)  # type: ignore[arg-type, return-value]
    assert caught.value is error and backend.stops == backend.cleanups == 1 and not backend.running


def test_profiling_and_experiment_round_trip_preserves_all_path_settings() -> None:
    from inferenceos.profiling.runner import _requested_config, ProfileTask
    from inferenceos.experiment.measurement import options_for
    from inferenceos.experiment.service import baseline_configuration_from
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.selftest import sample_model

    config = RuntimeConfiguration(context_length=2048, gpu_offload_layers=29, threads=6,
        threads_batch=3, engine_switches="GGML_VK_DISABLE_FUSION=1", batch_size=1024, ubatch_size=128,
        flash_attention=True, use_mmap=False, kv_cache_type="f16", kv_cache_type_k="q8_0",
        kv_cache_type_v="f16", placement_overrides="^blk\\.(1|2)\\.=CPU", device="Vulkan0")
    model = sample_model()
    profile = _requested_config(ProfileTask(model, get_workload("short-short"), 2048), options_for(config))
    assert baseline_configuration_from(profile, model) == config


def test_monitor_does_not_relabel_an_old_observation_with_todays_setup() -> None:
    from tests.test_runtime_paging_telemetry import _monitor, _runtime_observation, _model, _workload_profile, _hardware
    from inferenceos.experiment.service import baseline_configuration_from

    captured = RuntimeConfiguration(context_length=2048, gpu_offload_layers=29,
        threads_batch=3, engine_switches="GGML_VK_DISABLE_FUSION=1", ubatch_size=128,
        flash_attention=True, kv_cache_type_k="q8_0", placement_overrides="^output=CPU", device="Vulkan0")
    observation = _runtime_observation(faults=3.0).model_copy(update={"requested_configuration": captured})
    run = _monitor()._to_run(observation, condition="observed", model=_model(),
                              workload=_workload_profile(), hardware=_hardware())
    assert baseline_configuration_from(run.context.config, _model()) == captured


def test_gpu_envelope_does_not_count_the_output_step_twice() -> None:
    from datetime import datetime, timezone
    from inferenceos.optimization.search.resources import envelope_from_context
    from inferenceos.optimization.search.policy import SearchPolicy
    from inferenceos.selftest import sample_model, sample_hardware

    model = sample_model().model_copy(update={"n_layers": 28})
    ctx: Any = SimpleNamespace(model=model, hardware=sample_hardware(), metric=lambda _: 0.0,
        config=SimpleNamespace(context_length=1024, gpu_layers_offloaded=0, gpu_layers_total=29))
    now = datetime.now(timezone.utc)
    observed = SimpleNamespace(gpu_telemetry_available=True, gpu_observed_at=now, gpu_source="scripted", gpu_memory=[])
    with patch("inferenceos.optimization.search.resources._latest_gpu_system_state", return_value=observed), \
            patch("inferenceos.optimization.search.resources.total_vram_from_hardware", return_value=64 * 1024 ** 3):
        envelope = envelope_from_context(ctx, SearchPolicy(), now=now)
    assert envelope.feasible_max == 29 and envelope.total_layers == 28


def test_engine_cannot_measure_a_known_contradictory_thread_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.backends import llama_server_toolchain as srv
    from inferenceos.backends.resident_server import LlamaServerResidentBackend, ResidentBackendError
    from inferenceos.interfaces.backend_adapter import RuntimeParams
    from tests.test_resident_races_timeouts import Proc, Client, _model

    monkeypatch.setattr(srv, "LlamaServerClient", Client)
    monkeypatch.setattr(Proc, "log_text", lambda self: "llama threadpool init, n_threads = 3")
    backend = LlamaServerResidentBackend(server=Path("unused"), process_factory=Proc)  # type: ignore[arg-type]
    with pytest.raises(ResidentBackendError, match="threads: requested 6"):
        backend.start(_model(tmp_path), RuntimeParams(context_length=1024, threads=6))
    assert not backend.is_active() and backend.api_key_file is None and not backend.undeleted_key_files


@pytest.mark.parametrize("tail", [{}, {"stop": False}, {"timings": {"predicted_n": 4}},
                                   {"stop": 1, "timings": {"predicted_n": 4}}])
def test_native_generation_refuses_eof_without_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        tail: dict[str, Any]) -> None:
    from inferenceos.backends import llama_server_toolchain as srv
    from inferenceos.backends.resident_server import LlamaServerResidentBackend, ResidentBackendError
    from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
    from tests.test_resident_races_timeouts import Proc, Client, _model

    class Truncated(Client):
        def stream_completion(self, **kwargs: Any) -> Any:
            yield {"content": "A plausible partial answer"}
            yield tail

    monkeypatch.setattr(srv, "LlamaServerClient", Truncated)
    backend = LlamaServerResidentBackend(server=Path("unused"), process_factory=Proc)  # type: ignore[arg-type]
    try:
        backend.start(_model(tmp_path), RuntimeParams(context_length=1024))
        with pytest.raises(ResidentBackendError, match="completion"):
            backend.generate(GenerationRequest(prompt="test", n_predict=16))
    finally:
        backend.cleanup()


def test_native_generation_does_not_invent_tokens_from_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.backends import llama_server_toolchain as srv
    from inferenceos.backends.resident_server import LlamaServerResidentBackend
    from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
    from tests.test_resident_races_timeouts import Proc, Client, _model

    class Uncounted(Client):
        def stream_completion(self, **kwargs: Any) -> Any:
            yield {"content": "two words"}
            yield {"stop": True}

    monkeypatch.setattr(srv, "LlamaServerClient", Uncounted)
    backend = LlamaServerResidentBackend(server=Path("unused"), process_factory=Proc)  # type: ignore[arg-type]
    try:
        backend.start(_model(tmp_path), RuntimeParams(context_length=1024))
        result = backend.generate(GenerationRequest(prompt="test"))
        assert result.text == "two words" and result.generated_tokens is None
    finally:
        backend.cleanup()


@pytest.mark.parametrize("shown", [None, 512])
def test_restoration_requires_observed_settings_not_just_the_same_answer(shown: int | None) -> None:
    from tests.test_restoration_verifier import _FakeAdapter, _verifier
    from inferenceos.interfaces.backend_adapter import BackendTelemetry

    class Adapter(_FakeAdapter):
        def telemetry(self) -> BackendTelemetry:
            return BackendTelemetry(observed={} if shown is None else {"context_length": shown})

    adapter = Adapter()
    verifier = _verifier(adapter)
    config = RuntimeConfiguration(context_length=2048)
    verifier.capture(config)
    result = verifier.verify(config)
    assert result.probe_match and not result.configuration_match and not result.verified


def test_restoration_releases_a_partially_failed_load_and_preserves_the_error() -> None:
    from tests.test_restoration_verifier import _FakeAdapter, _verifier

    error = RuntimeError("partially started")

    class Adapter(_FakeAdapter):
        def load(self, model: Any, params: Any) -> None:
            self._loaded = True
            raise error

    adapter = Adapter()
    with pytest.raises(RuntimeError) as caught:
        _verifier(adapter)._probe_once(RuntimeConfiguration(context_length=2048))
    assert caught.value is error and not adapter.is_loaded


def test_orphan_check_cannot_use_an_old_captured_pid_as_proof_of_ownership() -> None:
    from tests.test_resident_recovery import RecoverableFakeAdapter, _workload, _provision

    adapter = RecoverableFakeAdapter()
    workload = _workload(adapter)
    _provision(workload)
    old_pid = workload.target_pid()
    adapter.kill()
    with patch("inferenceos.runtime.resident_workload.process_exists", return_value=True):
        assert not workload._no_orphan(old_pid)


@pytest.mark.parametrize("shown", [None, 512])
def test_reasoner_reload_cannot_claim_matching_configuration_from_liveness(shown: int | None) -> None:
    from tests.test_resident_lifecycle import _setup, _runner
    from inferenceos.interfaces.backend_adapter import BackendTelemetry, GenerationRequest

    _, _, _, model, params, target, reasoner = _setup()
    runner = _runner(target, model, params, reasoner)
    before = target.identity
    target.stop()
    target.start(model, params)
    with patch.object(target, "telemetry", return_value=BackendTelemetry(
            observed={} if shown is None else {"context_length": shown})):
        result = runner._verify_reload(before, before.pid if before else None, None, False,
                                       GenerationRequest(prompt="OK"))
    assert result.generation_ok and not result.configuration_match


@pytest.mark.parametrize("stage", ["load", "generate", "telemetry"])
def test_profiler_releases_failed_attempts_through_the_existing_adapter(stage: str) -> None:
    from tests.test_restoration_verifier import _FakeAdapter
    from tests.test_measurement_monitor import _Platform
    from inferenceos.profiling.runner import BaselineProfiler, ProfilerOptions, ProfileTask
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.selftest import sample_model, sample_hardware
    from inferenceos.contracts.enums import MeasurementMode

    error = RuntimeError(f"failed {stage}")

    class Adapter(_FakeAdapter):
        unloads = 0

        def load(self, model: Any, params: Any) -> None:
            super().load(model, params)
            if stage == "load":
                raise error

        def generate(self, request: Any) -> Any:
            if stage == "generate":
                raise error
            return super().generate(request)

        def telemetry(self) -> Any:
            if stage == "telemetry":
                raise error
            return super().telemetry()

        def unload(self) -> None:
            self.unloads += 1
            super().unload()

    adapter = Adapter()
    profiler = BaselineProfiler(adapter=adapter, hardware=sample_hardware())
    with patch("inferenceos.profiling.runner.get_platform", return_value=_Platform()), pytest.raises(RuntimeError) as caught:
        profiler.run_once(ProfileTask(sample_model(), get_workload("short-short"), 1024),
                          mode=MeasurementMode.WARM, sequence=1, repetition=0, warmup=False,
                          options=ProfilerOptions(telemetry_sampling=False))
    assert caught.value is error and adapter.unloads == 1 and not adapter.is_loaded


@pytest.mark.parametrize("interrupted", [False, True])
def test_profiler_unresolved_release_is_sticky_and_never_retried(interrupted: bool) -> None:
    from tests.test_restoration_verifier import _FakeAdapter
    from tests.test_measurement_monitor import _Platform
    from inferenceos.profiling.runner import BaselineProfiler, ProfilerOptions, ProfileTask
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.selftest import sample_model, sample_hardware
    from inferenceos.contracts.enums import MeasurementMode
    from inferenceos.interfaces.backend_adapter import BackendCleanupUnresolved

    error = KeyboardInterrupt("user stop") if interrupted else RuntimeError("answer failed")

    class Adapter(_FakeAdapter):
        loads = 0
        unloads = 0

        def load(self, model: Any, params: Any) -> None:
            self.loads += 1
            super().load(model, params)

        def generate(self, request: Any) -> Any:
            raise error

        def unload(self) -> None:
            self.unloads += 1
            raise PermissionError("cleanup denied")

        def release_evidence(self) -> Any:
            return SimpleNamespace(clean=False, reason="process still owned")

    adapter = Adapter()
    profiler = BaselineProfiler(adapter=adapter, hardware=sample_hardware())
    task = ProfileTask(sample_model(), get_workload("short-short"), 1024)
    kwargs: Any = dict(mode=MeasurementMode.WARM, sequence=1, repetition=0, warmup=False,
                       options=ProfilerOptions(telemetry_sampling=False))
    with patch("inferenceos.profiling.runner.get_platform", return_value=_Platform()):
        with pytest.raises(KeyboardInterrupt if interrupted else BackendCleanupUnresolved) as caught:
            profiler.run_once(task, **kwargs)
        assert (caught.value is error if interrupted else caught.value.__cause__ is error)
        with pytest.raises(BackendCleanupUnresolved):
            profiler.run_once(task, **kwargs)
    assert adapter.loads == adapter.unloads == 1


@pytest.mark.parametrize("budget", [0, 1200])
def test_unresolved_backend_cleanup_stops_both_tune_modes_without_saving(tmp_path: Path, budget: int) -> None:
    from inferenceos.interfaces.backend_adapter import BackendCleanupUnresolved

    calls = 0
    error = BackendCleanupUnresolved("process or credential retained")

    def failed(task: Any, options: Any, call: int) -> SampleResult:
        nonlocal calls
        calls += 1
        raise error

    tuner, _, _, store = _tuner(tmp_path, failed, answers=[True] * 8, settings=tune_settings(
        tmp_path, tune_layer_path=False, tune_paths="", tune_heat_check_runs=0,
        tune_offer_tradeoffs=False, tune_experimental_budget_s=budget))
    if budget:
        result = tuner.run()
        assert result.outcome.startswith("failed")
    else:
        with pytest.raises(BackendCleanupUnresolved) as caught:
            tuner.run()
        assert caught.value is error
    assert calls == 1 and store.all() == []


def test_profiler_keeps_the_sampler_owner_when_setup_fails() -> None:
    from tests.test_restoration_verifier import _FakeAdapter
    from inferenceos.profiling.runner import BaselineProfiler, ProfilerOptions, ProfileTask
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.selftest import sample_model, sample_hardware
    from inferenceos.contracts.enums import MeasurementMode
    from inferenceos.telemetry.counters import MonitorCleanupUnresolved

    error = RuntimeError("first counter read failed")

    class Sampler:
        cleanup_confirmed = False
        stops = 0

        def start(self) -> None:
            pass

        def wait_for_first_sample(self, **kwargs: Any) -> None:
            raise error

        def stop(self) -> list[Any]:
            self.stops += 1
            return []

    sampler = Sampler()
    adapter = _FakeAdapter()
    profiler = BaselineProfiler(adapter=adapter, hardware=sample_hardware())
    task = ProfileTask(sample_model(), get_workload("short-short"), 1024)
    kwargs: Any = dict(mode=MeasurementMode.WARM, sequence=1, repetition=0, warmup=False,
                       options=ProfilerOptions(telemetry_sampling=True))
    with patch("inferenceos.profiling.runner.CounterSampler", return_value=sampler):
        with pytest.raises(RuntimeError) as caught:
            profiler.run_once(task, **kwargs)
        assert caught.value is error
        with pytest.raises(MonitorCleanupUnresolved):
            profiler.run_once(task, **kwargs)
    assert sampler.stops == 1 and not adapter.is_loaded
    assert profiler._answer_sampler is sampler


def test_late_approval_cannot_be_accepted_when_the_caller_wakes_up_late() -> None:
    import threading
    import time
    from inferenceos.runtime.approval import _call_with_timeout, _TimedOut

    real_join = threading.Thread.join

    def delayed_join(worker: threading.Thread, timeout: float | None = None) -> None:
        real_join(worker, 0.2)  # caller was descheduled beyond its approval deadline

    def answer() -> bool:
        time.sleep(0.02)
        return True

    with patch.object(threading.Thread, "join", delayed_join), pytest.raises(_TimedOut):
        _call_with_timeout(answer, 0.001)


@pytest.mark.parametrize("old,new", [({"memory_pressure": float("nan")}, {"memory_pressure": 0.1}),
                                   ({"other": 1.0}, {"available_ram_bytes": 9.0}),
                                   ({"available_ram_bytes": 0.0}, {"available_ram_bytes": 9.0})])
def test_unknown_or_changed_history_conditions_are_not_a_match(old: dict[str, float], new: dict[str, float]) -> None:
    from inferenceos.knowledge.applicability import _conditions_match
    from inferenceos.knowledge.policy import KnowledgePolicy
    from inferenceos.contracts.enums import DimensionStatus

    assert _conditions_match(old, new, KnowledgePolicy()) is not DimensionStatus.MATCH


def test_missing_history_dimensions_and_future_dates_do_not_claim_direct_applicability(tmp_path: Path) -> None:
    from datetime import datetime, timezone, timedelta
    from tests.knowledge_helpers import base_result, make_service, current_context
    from inferenceos.knowledge.applicability import evaluate_applicability
    from inferenceos.contracts.enums import ApplicabilityClass

    service = make_service(tmp_path / "knowledge.db")
    record = service.build_record(base_result())
    now = datetime.now(timezone.utc)
    changes: list[dict[str, Any]] = [{"created_at": now + timedelta(days=1)},
                   {"created_at": now.replace(tzinfo=None)},
                   {"context": record.context.model_copy(update={"fingerprint":
                        record.context.fingerprint.model_copy(update={"diagnosis": None})})}]
    for change in changes:
        result = evaluate_applicability(record.model_copy(update=change), current_context(record), service.policy, now=now)
        assert result.classification is not ApplicabilityClass.DIRECTLY_APPLICABLE


def test_rollback_does_not_erase_a_new_setting_outside_its_original_change() -> None:
    from tests.test_rollback_executor import _setup
    from inferenceos.contracts.enums import RollbackExecutionStatus

    executor, adapter, record, _ = _setup()
    adapter._params = adapter._params.model_copy(update={"threads": 9})
    pid = adapter.pid
    result = executor.execute_record(record, run_id="full-audit-new-setting")
    assert result.status is RollbackExecutionStatus.REJECTED_INAPPLICABLE
    assert adapter.pid == pid and adapter.params.threads == 9


def test_rollback_cannot_verify_a_requested_setup_against_itself() -> None:
    from tests.test_rollback_executor import _setup
    from inferenceos.interfaces.backend_adapter import BackendTelemetry
    from inferenceos.contracts.enums import RollbackExecutionStatus

    executor, adapter, record, _ = _setup()
    with patch.object(adapter, "telemetry", return_value=BackendTelemetry(observed={"context_length": 512})):
        result = executor.execute_record(record, run_id="full-audit-wrong-engine")
    assert result.status is RollbackExecutionStatus.FAILED_UNRECOVERED
    assert not result.configuration_restored and not result.recovery_verified


def test_rollback_refuses_new_load_when_the_previous_owner_is_not_released() -> None:
    from tests.test_rollback_executor import _setup

    executor, adapter, record, _ = _setup()
    with patch.object(adapter, "release_evidence", create=True,
                      return_value=SimpleNamespace(clean=False, reason="credential retained")), \
            patch.object(adapter, "load", wraps=adapter.load) as load, pytest.raises(RuntimeError, match="release"):
        executor._apply(record.previous_state)
    assert load.call_count == 0


def test_dead_target_is_not_an_already_restored_rollback() -> None:
    from tests.test_rollback_executor import _setup
    from inferenceos.experiment.restoration import runtime_params_from
    from inferenceos.contracts.enums import RollbackExecutionStatus

    executor, adapter, record, _ = _setup()
    adapter._params = runtime_params_from(record.previous_state)
    adapter._loaded = False
    result = executor.execute_record(record, run_id="full-audit-dead-rollback")
    assert result.status is not RollbackExecutionStatus.ALREADY_RESTORED and not result.configuration_restored


def test_stale_supplied_rollback_context_cannot_authorize_overwriting_live_settings() -> None:
    from tests.test_rollback_executor import _setup
    from inferenceos.contracts.enums import RollbackExecutionStatus

    executor, adapter, record, _ = _setup()
    before = executor.current_context()
    adapter._params = adapter._params.model_copy(update={"threads": 9})
    pid = adapter.pid
    result = executor.execute_record(record, run_id="stale-owner-context", current_context=before)
    assert result.status is RollbackExecutionStatus.REJECTED_INAPPLICABLE
    assert adapter.pid == pid and adapter.params.threads == 9


@pytest.mark.parametrize("change", ["setup", "expiry"])
def test_monitor_approval_requires_the_current_live_setup_and_unexpired_proposal(change: str) -> None:
    from datetime import datetime, timedelta, timezone
    from tests.test_runtime_paging_telemetry import _monitor
    from inferenceos.contracts.enums import ApprovalOutcome
    from inferenceos.contracts.runtime import ApprovalRecord, RuntimeRecommendation

    monitor = _monitor()
    context = monitor._context_identity()
    recommendation = RuntimeRecommendation(proposal_id="test", opportunity_id="test", condition="test",
        why_it_matters="test", analysis_summary="test", context_identity=context,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=10))
    approval = ApprovalRecord(approval_id="test", proposal_id="test", context_identity=context,
                              outcome=ApprovalOutcome.APPROVED)
    assert monitor._approval_valid(approval, recommendation)
    if change == "setup":
        with patch.object(monitor, "_context_identity", return_value="changed-live-setup"):
            assert not monitor._approval_valid(approval, recommendation)
    else:
        recommendation = recommendation.model_copy(update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)})
        assert not monitor._approval_valid(approval, recommendation)


def test_monitor_does_not_resume_or_report_clean_after_a_failed_checkpoint_restore() -> None:
    from tests.test_runtime_intervention import (_monitor, _budget, _quality_regression_responder,
        FakeClock, ScriptedHarness, ScriptedApprovalProvider)
    from tests.runtime_helpers import make_runtime_policy
    from inferenceos.runtime.verification import ScriptedWorkload
    from inferenceos.contracts.enums import ApprovalOutcome

    clock = FakeClock()
    workload = ScriptedWorkload()
    monitor = _monitor(workload, make_runtime_policy(budget=_budget(30.0)),
        ScriptedApprovalProvider([ApprovalOutcome.APPROVED]),
        ScriptedHarness(_quality_regression_responder), clock)
    with patch("inferenceos.runtime.monitor.restore_runtime_checkpoint", return_value=False) as restore:
        result = monitor.run()
    assert restore.call_count == 1 and workload.resume_calls == 0
    assert not result.run.termination.clean and result.run.termination.safety
    assert not any(i.resume_verified for i in result.run.interventions)


def test_gpu_search_failed_checkpoint_restore_never_resumes_the_workload() -> None:
    from tests.test_runtime_gpu_search_integration import (_monitor, _gpu_workload,
        FakeClock, ScriptedHarness, _improving_responder, ScriptedApprovalProvider)
    from inferenceos.contracts.enums import ApprovalOutcome

    clock = FakeClock()
    workload = _gpu_workload()
    monitor = _monitor(workload, harness=ScriptedHarness(_improving_responder),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.APPROVED]), clock=clock)
    with patch("inferenceos.runtime.monitor.restore_runtime_checkpoint", return_value=False) as restore:
        result = monitor.run()
    assert restore.called and workload.resume_calls == 0
    assert result.run.termination.safety and not result.run.termination.clean


def test_recovery_identity_without_live_confirmation_is_not_verified() -> None:
    from tests.test_resident_recovery import RecoverableFakeAdapter, _workload, _provision

    adapter = RecoverableFakeAdapter()
    workload = _workload(adapter)
    _provision(workload)
    adapter.kill()
    with patch.object(adapter, "verify_process_identity", return_value=None):
        result = workload.recover_target(reason="process_exited")
    assert not result.identity_verified and not result.recovered and not adapter.is_loaded


def test_invalid_reference_paging_cannot_hide_a_resource_regression() -> None:
    from inferenceos.tune.acceptance_evidence import safety_resource_evidence
    from inferenceos.tune.layer_top import LoadAttempt
    from tests.test_review7_acceptance import _measurement

    reference = _measurement([0.0, 0.0])
    reference = reference.model_copy(update={"run_metrics": [
        run.model_copy(update={"hard_page_faults_per_sec": float("nan")}) for run in reference.run_metrics]})
    candidate = _measurement([500.0, 500.0])
    result = safety_resource_evidence(measurement=candidate, reference=reference, guarded=[],
                                     load_check=LoadAttempt(loaded=True, fits=True))
    assert not result.passed


@pytest.mark.parametrize("field", ["generated_tokens", "total_latency_s", "hard_page_faults_per_sec"])
def test_boolean_values_are_not_measurements(field: str) -> None:
    from inferenceos.contracts.observation import RunMetrics

    with pytest.raises(ValueError):
        RunMetrics.model_validate({field: True})


@pytest.mark.parametrize("count", [None, 3, 17])
def test_fixed_length_native_work_cannot_succeed_with_short_or_unknown_output(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, count: int | None) -> None:
    from inferenceos.backends import llama_server_toolchain as srv
    from inferenceos.backends.resident_server import LlamaServerResidentBackend, ResidentBackendError
    from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
    from tests.test_resident_races_timeouts import Proc, Client, _model

    class Incomplete(Client):
        def stream_completion(self, **kwargs: Any) -> Any:
            yield {"content": "plausible answer"}
            yield {"stop": True, "tokens_predicted": count, "timings": {"predicted_n": count}}

    monkeypatch.setattr(srv, "LlamaServerClient", Incomplete)
    backend = LlamaServerResidentBackend(server=Path("unused"), process_factory=Proc)  # type: ignore[arg-type]
    try:
        backend.start(_model(tmp_path), RuntimeParams(context_length=1024))
        with pytest.raises(ResidentBackendError, match="fixed-length"):
            backend.generate(GenerationRequest(prompt="test", n_predict=16, ignore_eos=True))
    finally:
        backend.cleanup()


def test_cli_backend_nonzero_exit_is_not_a_successful_measurement() -> None:
    from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
    from inferenceos.interfaces.backend_adapter import RuntimeParams, GenerationRequest
    from inferenceos.selftest import sample_model

    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter._loaded, adapter._model = True, sample_model()
    adapter._params, adapter._cli = RuntimeParams(), Path("unused")
    adapter._timeout_s = 1.0
    from inferenceos.backends.llama_cpp_toolchain import LlamaRunResult

    failed = LlamaRunResult(pid=1, returncode=1, stderr="engine out of memory",
                           stdout="plausible partial answer", duration_s=1.0, command=[])
    with patch("inferenceos.backends.llama_cpp_adapter.run_cli", return_value=failed), \
            pytest.raises(RuntimeError, match="exited"):
        adapter.generate(GenerationRequest(prompt="test"))


@pytest.mark.parametrize("failure", [KeyboardInterrupt("cancel"), OSError("pipe broke")])
def test_cli_process_is_reaped_when_communication_is_interrupted(failure: BaseException) -> None:
    from inferenceos.backends import llama_cpp_toolchain as toolchain

    class Process:
        pid = 2_000_000_001
        alive = True
        calls = 0
        returncode: int | None = None

        def communicate(self, **kwargs: Any) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise failure
            return "", ""

        def poll(self) -> int | None:
            return None if self.alive else self.returncode

        def kill(self) -> None:
            self.alive, self.returncode = False, -1

    process = Process()
    with patch.object(toolchain.subprocess, "Popen", return_value=process), \
            pytest.raises(type(failure)) as caught:
        toolchain.run_cli(Path("unused"), Path("model.gguf"), "test")
    assert caught.value is failure and not process.alive and process.calls == 2


def test_cli_interrupt_reaps_an_actual_owned_python_child() -> None:
    import subprocess
    import sys
    from inferenceos.backends import llama_cpp_toolchain as toolchain

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    original = process.communicate
    error = KeyboardInterrupt("user cancelled CLI read")
    calls = 0

    def interrupted(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return original(**kwargs)

    try:
        with patch.object(toolchain.subprocess, "Popen", return_value=process), \
                patch.object(process, "communicate", side_effect=interrupted), \
                pytest.raises(KeyboardInterrupt) as caught:
            toolchain.run_cli(Path("unused"), Path("model.gguf"), "test")
        assert caught.value is error and process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5.0)


def test_cli_preserves_reading_threads_switches_and_request_length_semantics() -> None:
    from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
    from inferenceos.backends.llama_cpp_toolchain import LlamaRunResult
    from inferenceos.interfaces.backend_adapter import RuntimeParams, GenerationRequest
    from inferenceos.selftest import sample_model

    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter._loaded, adapter._model, adapter._cli, adapter._timeout_s = True, sample_model(), Path("unused"), 1.0
    adapter._params = RuntimeParams(threads_batch=3, engine_switches="GGML_VK_DISABLE_FUSION=1")
    result = LlamaRunResult(pid=1, returncode=0, stdout="OK", stderr="", duration_s=1.0, command=[])
    with patch("inferenceos.backends.llama_cpp_adapter.run_cli", return_value=result) as run:
        adapter.generate(GenerationRequest(prompt="test", ignore_eos=True))
    assert run.call_args.kwargs["threads_batch"] == 3
    assert run.call_args.kwargs["engine_switches"] == "GGML_VK_DISABLE_FUSION=1"
    assert run.call_args.kwargs["ignore_eos"] is True


def test_cli_launch_applies_reading_threads_and_explicit_switch_without_logging_the_prompt() -> None:
    from inferenceos.backends import llama_cpp_toolchain as toolchain
    import os

    process = SimpleNamespace(pid=2_000_000_001, returncode=0,
                              communicate=lambda **kwargs: ("OK", ""))
    with patch.object(toolchain.subprocess, "Popen", return_value=process) as launch, \
            patch.object(toolchain.logger, "info") as log, \
            patch.dict(os.environ, {"GGML_VK_FORCE_MMVQ": "1"}):
        toolchain.run_cli(Path("unused"), Path("model.gguf"), "private prompt marker",
            threads_batch=3, engine_switches="GGML_VK_DISABLE_FUSION=1")
    command = launch.call_args.args[0]
    assert command[command.index("-tb") + 1] == "3" and "private prompt marker" in command
    environment = launch.call_args.kwargs["env"]
    assert environment["GGML_VK_DISABLE_FUSION"] == "1" and "GGML_VK_FORCE_MMVQ" not in environment
    assert "private prompt marker" not in str(log.call_args)


def test_terminal_backend_cleanup_cannot_be_swallowed_as_missing_path_evidence(tmp_path: Path) -> None:
    from inferenceos.interfaces.backend_adapter import BackendCleanupUnresolved

    error = BackendCleanupUnresolved("prior process retained")
    tuner, _, _, _ = _tuner(tmp_path, lambda *args: SampleResult(), answers=[])

    def blocked(config: Any) -> Any:
        raise error

    tuner.piece_lines_reader = blocked
    with pytest.raises(BackendCleanupUnresolved) as caught:
        tuner._read_piece_lines(RuntimeConfiguration(gpu_offload_layers=8, placement_overrides="^blk\\.1\\.=CPU"))
    assert caught.value is error


def test_interrupted_scoped_sampler_start_still_reaches_its_cleanup_owner() -> None:
    from inferenceos.telemetry.counters import MeasurementMonitor, HARD_FAULT_PATH

    error = KeyboardInterrupt("partial sampler start")

    class Sampler:
        stops = 0

        def start(self) -> None:
            raise error

        def stop(self) -> dict[str, Any]:
            self.stops += 1
            return {"resolved": True}

    sampler = Sampler()
    monitor = MeasurementMonitor([HARD_FAULT_PATH], sampler_factory=lambda *args, **kwargs: sampler)  # type: ignore[arg-type]
    with pytest.raises(KeyboardInterrupt) as caught:
        monitor.open()
    assert caught.value is error
    monitor.close()
    assert sampler.stops == 1 and monitor.stop_record["resolved"] is True


def test_profiler_retains_a_monitor_whose_close_was_interrupted() -> None:
    from inferenceos.profiling.runner import BaselineProfiler
    from tests.test_restoration_verifier import _FakeAdapter
    from inferenceos.selftest import sample_hardware

    error = KeyboardInterrupt("interrupted close")
    monitor = SimpleNamespace(closed=False, close=lambda: (_ for _ in ()).throw(error))
    profiler = BaselineProfiler(adapter=_FakeAdapter(), hardware=sample_hardware())
    profiler._monitor = monitor  # type: ignore[assignment]
    with pytest.raises(KeyboardInterrupt) as caught:
        profiler.end_measurement()
    assert caught.value is error and profiler._monitor is monitor and profiler.monitor_blocked


def test_actual_cli_timing_dialect_is_parsed_without_inventing_ttft_or_written_tokens() -> None:
    from inferenceos.backends.llama_cpp_toolchain import parse_llama_timings

    log = ("llama_perf_context_print:        load time = 1200.00 ms\n"
           "llama_perf_context_print: prompt eval time = 439.01 ms / 35 tokens (12.54 ms per token, 79.72 tokens per second)\n"
           "llama_perf_context_print:        eval time = 289.88 ms / 7 runs (41.41 ms per token, 24.15 tokens per second)\n"
           "llama_perf_context_print:       total time = 728.89 ms / 42 tokens\n")
    result = parse_llama_timings(log)
    assert result is not None
    assert result.prompt_tokens == 35 and result.decode_evaluations == 7
    assert result.model_load_s == pytest.approx(1.2)
    assert result.decode_tokens_per_sec == pytest.approx(24.15)
    assert result.total_s == pytest.approx(0.72889)
    assert result.ttft_s is None and result.generated_tokens is None


@pytest.mark.parametrize("completed", [False, True])
def test_unusable_native_protocol_work_still_exposes_prompt_validity_breaches(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, completed: bool) -> None:
    from dataclasses import asdict
    from inferenceos.backends import llama_server_toolchain as srv
    from inferenceos.backends.resident_server import LlamaServerResidentBackend
    from inferenceos.interfaces.backend_adapter import RuntimeParams
    from inferenceos.tune.prompt_reuse import work_evidence, _breach
    from tests.test_resident_races_timeouts import Proc, Client, _model
    from tests.test_prompt_reuse import _real_engine

    class Invalid(Client):
        def stream_completion(self, **kwargs: Any) -> Any:
            yield {"content": "answer", "stop": completed, "id_slot": 0,
                   "tokens_evaluated": 10, "tokens_predicted": 3,
                   "timings": {"prompt_n": 9, "cache_n": 1, "predicted_n": 3}}

    monkeypatch.setattr(srv, "LlamaServerClient", Invalid)
    backend = LlamaServerResidentBackend(server=Path("unused"), process_factory=Proc)  # type: ignore[arg-type]
    backend.start(_model(tmp_path), RuntimeParams(context_length=1024))
    engine = _real_engine(lambda: backend)
    engine._backend, engine._loaded = backend, True
    try:
        reading = asdict(engine.generate("test", cache_prompt=False, n_predict=16, seed=1))
        assert work_evidence(reading, expected_total=10, expected_generated=16)[1]
        assert reading["engine_counters"]["protocol_problem"]
        assert _breach(reading, expected_total=10, bound=0, fresh=True, what="fresh") is not None
        assert backend.last_generation is None
    finally:
        assert engine.stop().clean


def test_deep_valid_gguf_arrays_are_skipped_without_python_recursion(tmp_path: Path) -> None:
    import struct
    from tests.test_gguf import _write_gguf, _gstr
    from inferenceos.models.gguf import read_gguf, GgufArray

    nested = struct.pack("<IQ", 9, 1) * 2000 + struct.pack("<IQI", 4, 1, 7)
    path = tmp_path / "deep.gguf"
    _write_gguf(path, 3, [_gstr("vendor.nested") + struct.pack("<I", 9) + nested], [])
    assert read_gguf(path).get("vendor.nested") == GgufArray(element_type=9, count=1)


@pytest.mark.parametrize("defect", ["duplicate", "boolean", "utf8"])
def test_ambiguous_or_malformed_gguf_metadata_is_not_normalized_into_model_facts(tmp_path: Path, defect: str) -> None:
    import struct
    from tests.test_gguf import _write_gguf, _gstr, _kv_u32
    from inferenceos.models.gguf import read_gguf, GgufError

    if defect == "duplicate":
        metadata = [_kv_u32("llama.block_count", 2), _kv_u32("llama.block_count", 3)]
    elif defect == "boolean":
        metadata = [_gstr("vendor.flag") + struct.pack("<IB", 7, 2)]
    else:
        metadata = [_gstr("general.name") + struct.pack("<IQ", 8, 1) + b"\xff"]
    path = tmp_path / "ambiguous.gguf"
    _write_gguf(path, 3, metadata, [])
    with pytest.raises(GgufError):
        read_gguf(path)


@pytest.mark.parametrize("budget", [0, 1200])
def test_tuner_cannot_measure_one_model_profile_and_save_another_file_identity(tmp_path: Path, budget: int) -> None:
    tuner, harness, _, store = _tuner(tmp_path, lambda *args: SampleResult(), answers=[],
        settings=tune_settings(tmp_path, tune_experimental_budget_s=budget))
    other = tmp_path / "other.gguf"
    other.write_bytes(b"a different model file")
    tuner.model = tuner.model.model_copy(update={"path": str(other)})
    with pytest.raises(RuntimeError, match="model.*(path|file|profile)"):
        tuner.run()
    assert not harness.calls and store.all() == []


def test_model_profile_refuses_a_file_changing_during_metadata_read(tmp_path: Path) -> None:
    from inferenceos.models import loader
    from tests.test_gguf import _write_gguf, _kv_u32, _kv_string

    path = tmp_path / "changed.gguf"
    _write_gguf(path, 3, [_kv_string("general.architecture", "llama"),
        _kv_u32("llama.block_count", 28), _kv_u32("llama.embedding_length", 128),
        _kv_u32("llama.attention.head_count", 2)], [])
    real_reader = loader.read_gguf

    def changed(path: Any) -> Any:
        result = real_reader(path)
        Path(path).write_bytes(b"a changed model")
        return result

    with patch.object(loader, "read_gguf", side_effect=changed), \
            pytest.raises(loader.ModelMetadataError, match="changed"):
        loader.load_model_profile(path)


def test_model_profile_signature_binds_metadata_to_the_file_at_tune_entry(tmp_path: Path) -> None:
    from inferenceos.models.loader import load_model_profile
    from tests.test_gguf import _write_gguf, _kv_u32, _kv_string

    tuner, harness, _, store = _tuner(tmp_path, lambda *args: SampleResult(), answers=[])
    _write_gguf(tuner.model_path, 3, [_kv_string("general.architecture", "llama"),
        _kv_u32("llama.block_count", 28), _kv_u32("llama.embedding_length", 128),
        _kv_u32("llama.attention.head_count", 2)], [])
    tuner.model = load_model_profile(tuner.model_path)
    assert tuner.model.metadata["_source_file_signature"][1] == tuner.model_path.stat().st_size
    tuner.model_path.write_bytes(tuner.model_path.read_bytes() + b"replacement")
    with pytest.raises(RuntimeError, match="model file changed since its profile"):
        tuner.run()
    assert not harness.calls and store.all() == []


@pytest.mark.parametrize("key", ["qwen2.block_count", "qwen2.attention.head_count_kv"])
@pytest.mark.parametrize("value", [True, 2.5, float("nan"), -2, "2"])
def test_invalid_model_dimensions_are_not_coerced_or_replaced_by_a_default(key: str, value: Any) -> None:
    from inferenceos.models.loader import ModelMetadataError, build_model_profile
    from tests.test_model_architecture import GQA_METADATA, _gguf

    with pytest.raises(ModelMetadataError):
        build_model_profile(_gguf({**GQA_METADATA, key: value}), file_size_bytes=1)


def test_ambiguous_architecture_facts_are_not_picked_by_dictionary_order() -> None:
    from inferenceos.models.loader import ModelMetadataError, build_model_profile
    from tests.test_model_architecture import MHA_METADATA, _gguf

    metadata = {k: v for k, v in MHA_METADATA.items() if k != "general.architecture"}
    metadata["other.block_count"] = 80
    with pytest.raises(ModelMetadataError, match="ambiguous"):
        build_model_profile(_gguf(metadata), file_size_bytes=1)


@pytest.mark.parametrize("changed", ["configuration", "identity"])
def test_rollback_refuses_a_target_that_changes_during_the_checkpoint_probe(changed: str) -> None:
    from tests.test_rollback_executor import _setup
    from inferenceos.contracts.enums import RollbackExecutionStatus
    from inferenceos.interfaces.backend_adapter import RuntimeParams
    from inferenceos.contracts.lifecycle import ProcessIdentity

    executor, adapter, record, _ = _setup()
    generate = adapter.generate
    unload = adapter.unload
    unloads: list[bool] = []

    def replacement(request: Any) -> Any:
        reply = generate(request)
        if changed == "configuration":
            adapter.set_runtime_params(RuntimeParams(threads=987))
        else:
            adapter.pid += 20
            adapter.identity = ProcessIdentity(pid=adapter.pid, create_time=800.0)
        return reply

    def counted_unload() -> None:
        unloads.append(True)
        unload()

    with patch.object(adapter, "generate", side_effect=replacement), \
            patch.object(adapter, "unload", side_effect=counted_unload):
        result = executor.execute_record(record, run_id="checkpoint-change")
    assert result.status is RollbackExecutionStatus.REJECTED_INAPPLICABLE
    assert not unloads and adapter.is_loaded
    if changed == "configuration":
        assert adapter.params.threads == 987


def _two_gpu_hardware() -> Any:
    from inferenceos.selftest import sample_hardware
    from inferenceos.contracts.hardware import BackendDevice, BackendAvailability
    from inferenceos.contracts.enums import BackendType

    hardware = sample_hardware()
    return hardware.model_copy(update={
        "gpus": [hardware.gpus[0].model_copy(update={"name": name, "vram_bytes": size * 1024**3})
                 for name, size in (("small", 4), ("large", 16))],
        "backend_availability": [BackendAvailability(backend=BackendType.VULKAN, usable=True,
            devices=[BackendDevice(name=name, device_index=i, vram_bytes=size * 1024**3, usable=True)
                     for i, (name, size) in enumerate((("small", 4), ("large", 16)))])],
    })


def test_scalar_gpu_planning_does_not_assume_the_largest_device_is_selected(tmp_path: Path) -> None:
    from inferenceos.optimization.search.resources import total_vram_from_hardware
    from inferenceos.tune.machine import check_machine
    from inferenceos.contracts.tune import PowerPlan
    from inferenceos.contracts.enums import PowerState

    hardware = _two_gpu_hardware()
    assert total_vram_from_hardware(hardware) is None
    machine = check_machine(tune_settings(tmp_path), hardware=hardware,
        live_reader=lambda: (2 * 1024**3, PowerState.AC), power_plan_reader=PowerPlan,
        version_reader=lambda: "scripted")
    assert machine.total_graphics_memory_bytes is None
    assert machine.free_graphics_memory_bytes is None


def test_live_gpu_memory_does_not_merge_unrelated_adapters() -> None:
    from inferenceos.tune.machine import read_live_state
    from inferenceos.contracts.enums import PowerState

    state = SimpleNamespace(gpu_telemetry_available=True, power_state=PowerState.AC,
        gpu_memory=[SimpleNamespace(memory_used_bytes=10), SimpleNamespace(memory_used_bytes=20)])
    with patch("inferenceos.telemetry.system.sample_system_state", return_value=state):
        used, power = read_live_state()
    assert used is None and power is PowerState.AC


def test_conflicting_backend_gpu_evidence_cannot_authorize_a_memory_fit(tmp_path: Path) -> None:
    from inferenceos.selftest import sample_hardware
    from inferenceos.contracts.hardware import BackendAvailability, BackendDevice
    from inferenceos.contracts.enums import BackendType, GPUDeviceType, PowerState
    from inferenceos.contracts.tune import PowerPlan
    from inferenceos.optimization.search.resources import total_vram_from_hardware
    from inferenceos.tune.machine import check_machine

    hardware = sample_hardware()
    hardware.backend_availability.append(BackendAvailability(
        backend=BackendType.CUDA, usable=True,
        devices=[BackendDevice(name="unrelated large GPU", device_type=GPUDeviceType.DISCRETE,
                               vram_bytes=16 * 1024**3, usable=True)],
    ))
    assert len(hardware.gpus) == 1  # Incomplete OS inventory cannot bind the other API.
    assert total_vram_from_hardware(hardware) is None
    machine = check_machine(tune_settings(tmp_path), hardware=hardware,
        live_reader=lambda: (2 * 1024**3, PowerState.AC), power_plan_reader=PowerPlan,
        version_reader=lambda: "scripted")
    assert machine.total_graphics_memory_bytes is None
    assert machine.free_graphics_memory_bytes is None


@pytest.mark.parametrize("counter", ["disagrees", "boolean", "reported_fresh_reuse"])
def test_native_counter_contradictions_cannot_become_measurement_evidence(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, counter: str) -> None:
    from inferenceos.backends import llama_server_toolchain as srv
    from inferenceos.backends.resident_server import LlamaServerResidentBackend, ResidentGenerationProtocolError
    from inferenceos.interfaces.backend_adapter import RuntimeParams, GenerationRequest
    from tests.test_resident_races_timeouts import Proc, Client, _model

    class Invalid(Client):
        def stream_completion(self, **kwargs: Any) -> Any:
            yield {"content": "answer", "stop": True, "tokens_evaluated": 10,
                   "tokens_predicted": {"disagrees": 3, "boolean": True}.get(counter, 16),
                   "timings": {"predicted_n": 16,
                       "prompt_n": 9 if counter == "reported_fresh_reuse" else 10,
                       "cache_n": 1 if counter == "reported_fresh_reuse" else 0}}

    monkeypatch.setattr(srv, "LlamaServerClient", Invalid)
    backend = LlamaServerResidentBackend(server=Path("unused"), process_factory=Proc)  # type: ignore[arg-type]
    backend.start(_model(tmp_path), RuntimeParams(context_length=1024))
    try:
        with pytest.raises(ResidentGenerationProtocolError):
            backend.generate(GenerationRequest(prompt="test", n_predict=16, ignore_eos=True))
        assert backend.last_generation is None
    finally:
        backend.cleanup()
