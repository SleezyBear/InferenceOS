"""Real rollback executor over a deterministic fake adapter (error branches)."""

from __future__ import annotations

from types import SimpleNamespace

from inferenceos.contracts.enums import RollbackExecutionStatus
from inferenceos.contracts.experiment import RestorationVerification, RuntimeConfiguration
from inferenceos.contracts.lifecycle import ProcessIdentity
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.rollback import RollbackExecutionPolicy
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    BackendCapabilities,
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.knowledge import (
    RollbackExecutor,
    build_rollback_record,
    seal_rollback_record,
)
from inferenceos.optimization.actions import build_default_registry
from tests.knowledge_helpers import base_result, make_service


def _probe_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


class FakeRollbackAdapter(BackendAdapter):
    def __init__(self) -> None:
        self._loaded = False
        self._params = RuntimeParams()
        self.pid = 100
        self.identity = ProcessIdentity(pid=100, create_time=1.0)
        self.last_load = SimpleNamespace(ready=True, resident_verified=True)
        self.text = "OK"
        self.rss = 1_000_000_000
        self.tps = 20.0

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="fake")

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        self._loaded = True
        self._params = params
        self.pid += 1
        self.identity = ProcessIdentity(pid=self.pid, create_time=float(self.pid))

    def unload(self) -> None:
        self._loaded = False

    def set_runtime_params(self, params: RuntimeParams) -> None:
        updates = {name: getattr(params, name) for name in params.model_fields_set}
        self._params = self._params.model_copy(update=updates)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            text=self.text, peak_rss_bytes=self.rss, decode_tokens_per_sec=self.tps
        )

    def telemetry(self) -> BackendTelemetry:
        return BackendTelemetry(gpu_layers_offloaded=self._params.gpu_offload_layers,
                                mmap_effective=self._params.use_mmap,
                                observed={"context_length": self._params.context_length})

    @property
    def params(self) -> RuntimeParams:
        return self._params

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def _setup(*, fault_stage: str | None = None) -> tuple:
    service = make_service()
    result = base_result()
    service.persist_result(result)
    record = build_rollback_record(result, registry=build_default_registry())
    assert record is not None and record.restoration_verification is None
    record = seal_rollback_record(
        record.model_copy(
            update={
                "restoration_verification": RestorationVerification(
                    attempted=True,
                    running=True,
                    configuration_match=True,
                    probe_match=True,
                    verified=True,
                    baseline_probe_hash=_probe_hash("OK"),
                )
            }
        )
    )
    baseline = service.store.get_by_experiment(result.experiment_id)
    adapter = FakeRollbackAdapter()
    if baseline is not None and "peak_rss_bytes" in baseline.baseline_metrics:
        mean = baseline.baseline_metrics["peak_rss_bytes"].mean
        if mean is not None:
            adapter.rss = int(mean)
    # Target is in the optimized (resulting) state before rollback.
    adapter._loaded = True
    adapter._params = RuntimeParams(
        **{
            name: value
            for name, value in record.resulting_state.model_dump().items()
            if name in RuntimeParams.model_fields
        }
    )
    executor = RollbackExecutor(
        service=service,
        registry=build_default_registry(),
        adapter=adapter,
        model=result.request.model,  # type: ignore[union-attr]
        hardware=result.request.hardware,  # type: ignore[union-attr]
        backend=result.request.backend,  # type: ignore[union-attr]
        policy=RollbackExecutionPolicy(fault_stage=fault_stage),
    )
    return executor, adapter, record, service


def test_real_rollback_applies_and_verifies() -> None:
    executor, adapter, record, service = _setup()
    result = executor.execute_record(record, run_id="run-1")
    assert result.status is RollbackExecutionStatus.APPLIED_VERIFIED
    assert result.integrity_ok is True
    assert result.applicability.classification.value == "applicable"
    assert result.configuration_restored is True
    assert result.behavioural_probe_passed is True
    assert result.pid_before != result.pid_after
    assert result.process_reloaded is True
    assert result.ready is True
    assert result.event_ids
    assert adapter.params.use_mmap == record.previous_state.use_mmap
    assert adapter.params.context_length == record.previous_state.context_length


def test_idempotent_already_restored() -> None:
    executor, adapter, record, service = _setup()
    executor.execute_record(record, run_id="run-1")
    # Second invocation: target is now already at the previous configuration.
    second = executor.execute_record(record, run_id="run-2")
    assert second.status is RollbackExecutionStatus.ALREADY_RESTORED
    assert second.process_reloaded is False


def test_integrity_failure_rejected_without_change() -> None:
    executor, adapter, record, service = _setup()
    pid_before = adapter.pid
    tampered = record.model_copy(update={"previous_state": RuntimeConfiguration(threads=1)})
    result = executor.execute_record(tampered, run_id="run-1")
    assert result.status is RollbackExecutionStatus.REJECTED_INTEGRITY
    assert adapter.pid == pid_before


def test_incompatible_model_rejected_without_change() -> None:
    executor, adapter, record, service = _setup()
    pid_before = adapter.pid
    incompatible = seal_rollback_record(record.model_copy(update={"model_key": "other"}))
    result = executor.execute_record(incompatible, run_id="run-1")
    assert result.status is RollbackExecutionStatus.REJECTED_INAPPLICABLE
    assert result.applicability.classification.value == "incompatible_model"
    assert adapter.pid == pid_before


def test_failure_containment_restores_checkpoint() -> None:
    executor, adapter, record, service = _setup(fault_stage="after_apply")
    checkpoint_mmap = record.resulting_state.use_mmap
    result = executor.execute_record(record, run_id="run-1")
    assert result.status is RollbackExecutionStatus.FAILED_RESTORED
    assert result.recovery_attempted is True
    assert result.recovery_verified is True
    # Checkpoint (resulting state) restored.
    assert adapter.params.use_mmap == checkpoint_mmap


def test_insufficient_evidence_without_model() -> None:
    executor, adapter, record, service = _setup()
    context = executor.current_context()
    from inferenceos.contracts.knowledge import ApplicabilityContext

    no_model = ApplicabilityContext(
        fingerprint=context.fingerprint.model_copy(update={"model_key": "unknown"}),
        hardware=context.hardware,
        model=None,
        backend=context.backend,
        runtime_configuration=record.resulting_state,
    )
    result = executor.execute_record(record, run_id="run-1", current_context=no_model)
    assert result.status is RollbackExecutionStatus.REJECTED_INAPPLICABLE
    assert result.applicability.classification.value == "insufficient_evidence"


# --------------------------------------------------------------------------- #
# Step 2 bug: a failed undo must never be labelled "restored".                #
# --------------------------------------------------------------------------- #
class _RecoveryFailsAdapter(FakeRollbackAdapter):
    """First load (the undo) works; the recovery load afterwards crashes."""

    def __init__(self) -> None:
        super().__init__()
        self.loads = 0

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        self.loads += 1
        if self.loads >= 2:
            self._loaded = False
            raise RuntimeError("server failed to start")
        super().load(model, params)


class _RecoveryWrongSettingsAdapter(FakeRollbackAdapter):
    """Recovery load comes up and replies, but not with the checkpoint settings."""

    def __init__(self) -> None:
        super().__init__()
        self.loads = 0

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        self.loads += 1
        super().load(model, params if self.loads < 2 else RuntimeParams())


def _swap_adapter(executor: RollbackExecutor, adapter: FakeRollbackAdapter, old: FakeRollbackAdapter) -> None:
    adapter._loaded = True
    adapter._params = old._params
    adapter.rss = old.rss
    executor.adapter = adapter


def test_failed_verification_with_failed_recovery_is_unrecovered() -> None:
    executor, old, record, service = _setup()
    adapter = _RecoveryFailsAdapter()
    _swap_adapter(executor, adapter, old)
    adapter.text = "a different answer"  # undo comes up but fails the behaviour check
    result = executor.execute_record(record, run_id="run-1")
    assert result.recovery_attempted is True
    assert result.recovery_verified is False
    assert result.status is RollbackExecutionStatus.FAILED_UNRECOVERED


def test_recovery_with_wrong_settings_is_not_counted_as_restored() -> None:
    executor, old, record, service = _setup()
    adapter = _RecoveryWrongSettingsAdapter()
    _swap_adapter(executor, adapter, old)
    adapter.text = "a different answer"
    result = executor.execute_record(record, run_id="run-1")
    assert result.recovery_attempted is True
    assert result.recovery_verified is False
    assert result.status is RollbackExecutionStatus.FAILED_UNRECOVERED


def test_failed_apply_with_wrong_settings_recovery_is_unrecovered() -> None:
    executor, old, record, service = _setup(fault_stage="after_apply")
    adapter = _RecoveryWrongSettingsAdapter()
    _swap_adapter(executor, adapter, old)
    result = executor.execute_record(record, run_id="run-1")
    assert result.recovery_verified is False
    assert result.status is RollbackExecutionStatus.FAILED_UNRECOVERED
