"""Real rollback executor (Phase 8R).

Consumes a persisted :class:`RollbackRecord` and actually applies its stored previous
configuration to a live target through the Phase 5R resident backend, restarting the
target as required, then verifies the restoration behaviourally and by measurement,
and records the outcome as append-only knowledge events.

It does **not** merely assign Python configuration values: the target is unloaded and
reloaded, a new process identity is captured, a real generation is performed, and the
result is persisted.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from inferenceos.contracts.enums import KnowledgeEventType, RollbackApplicabilityClass
from inferenceos.contracts.enums import RollbackExecutionStatus
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.knowledge import ApplicabilityContext, RollbackRecord
from inferenceos.contracts.lifecycle import ProcessIdentity
from inferenceos.backends.observed import ObservedConfigurationBaseline, observed_configuration
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import ProfilingConfig
from inferenceos.contracts.rollback import (
    RollbackApplicability,
    RollbackExecutionPolicy,
    RollbackExecutionResult,
)
from inferenceos.experiment.restoration import (
    runtime_params_from,
    runtime_configuration_from,
)
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    GenerationRequest,
    RuntimeParams,
)
from inferenceos.knowledge.fingerprint import build_current_context
from inferenceos.knowledge.rollback import (
    classify_rollback_applicability,
    verify_rollback_integrity,
)
from inferenceos.knowledge.service import KnowledgeService
from inferenceos.logging_config import get_logger
from inferenceos.optimization.registry import ActionRegistry

logger = get_logger("knowledge.rollback_executor")


class RollbackExecutionError(RuntimeError):
    """A rollback could not be completed; the caller must not report success."""


def _hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class RollbackExecutor:
    """Applies a persisted rollback to a live target and verifies it."""

    service: KnowledgeService
    registry: ActionRegistry
    adapter: BackendAdapter
    model: ModelProfile
    hardware: HardwareProfile
    backend: ProfilingConfig
    policy: RollbackExecutionPolicy = field(default_factory=RollbackExecutionPolicy)
    probe: GenerationRequest | None = None

    # ------------------------------------------------------------------ #
    # Entry points                                                         #
    # ------------------------------------------------------------------ #
    def execute(
        self,
        rollback_id: str,
        *,
        run_id: str,
        current_context: ApplicabilityContext | None = None,
    ) -> RollbackExecutionResult:
        record = self.service.get_rollback(rollback_id)
        if record is None:
            raise RollbackExecutionError(f"no persisted rollback record: {rollback_id!r}")
        return self.execute_record(record, run_id=run_id, current_context=current_context)

    def execute_record(
        self,
        record: RollbackRecord,
        *,
        run_id: str,
        current_context: ApplicabilityContext | None = None,
    ) -> RollbackExecutionResult:
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        execution_id = f"rbx-{uuid.uuid4().hex[:12]}"
        record_id = f"opt-{record.experiment_id}"
        event_ids: list[str] = []
        errors: list[str] = []

        def emit(event_type: KnowledgeEventType, detail: str, data: dict[str, object]) -> None:
            try:
                event = self.service.store.append_event(record_id, event_type, detail, data)
                event_ids.append(event.event_id)
            except Exception as exc:  # noqa: BLE001 - never let audit failure raise
                errors.append(f"audit event failed: {type(exc).__name__}: {exc}")

        emit(
            KnowledgeEventType.ROLLBACK_REQUESTED,
            f"rollback {record.rollback_id} requested",
            {"rollback_id": record.rollback_id, "run_id": run_id, "execution_id": execution_id},
        )

        # --- integrity --- #
        if not verify_rollback_integrity(record):
            result = self._finish(
                execution_id,
                run_id,
                record,
                status=RollbackExecutionStatus.REJECTED_INTEGRITY,
                applicability=RollbackApplicability(
                    classification=RollbackApplicabilityClass.INTEGRITY_FAILURE,
                    rollback_id=record.rollback_id,
                    experiment_id=record.experiment_id,
                    integrity_ok=False,
                    reasons=["integrity hash mismatch"],
                ),
                integrity_ok=False,
                started=started,
                started_at=started_at,
                errors=["integrity hash mismatch; rollback not executed"],
                event_ids=event_ids,
                summary="rollback rejected: integrity failure (no target change)",
            )
            emit(
                KnowledgeEventType.ROLLBACK_REJECTED,
                "integrity failure",
                result.model_dump(mode="json"),
            )
            return result

        # --- applicability --- #
        current = current_context or self.current_context()
        stale_context = current.runtime_configuration != self._current_runtime()
        applicability = classify_rollback_applicability(
            record, current, self.service.policy, registry=self.registry
        )
        if stale_context:
            applicability = applicability.model_copy(update={
                "classification": RollbackApplicabilityClass.INCOMPATIBLE_TARGET,
                "reasons": [*applicability.reasons, "supplied context is stale: live settings changed"],
            })
        if applicability.classification is RollbackApplicabilityClass.ALREADY_RESTORED:
            observed, observed_ok = self._observed_configuration(record.previous_state)
            restored = bool(self.adapter.is_loaded and observed_ok)
            result = self._finish(
                execution_id,
                run_id,
                record,
                status=(RollbackExecutionStatus.ALREADY_RESTORED if restored else
                        RollbackExecutionStatus.REJECTED_INAPPLICABLE),
                applicability=applicability,
                integrity_ok=True,
                started=started,
                started_at=started_at,
                errors=[],
                event_ids=event_ids,
                checkpoint_configuration=current.runtime_configuration,
                actual_configuration=observed,
                configuration_restored=restored,
                summary=("rollback not needed: live target shows the previous settings" if restored else
                         "previous setup was requested, but live settings are unverified; no target change"),
            )
            emit(
                KnowledgeEventType.ROLLBACK_EXECUTED,
                "already_restored" if restored else "live_settings_unverified",
                result.model_dump(mode="json"),
            )
            return result
        if not applicability.executable:
            result = self._finish(
                execution_id,
                run_id,
                record,
                status=RollbackExecutionStatus.REJECTED_INAPPLICABLE,
                applicability=applicability,
                integrity_ok=True,
                started=started,
                started_at=started_at,
                errors=[],
                event_ids=event_ids,
                summary=(
                    "rollback rejected before execution: "
                    + applicability.classification.value
                ),
            )
            emit(
                KnowledgeEventType.ROLLBACK_REJECTED,
                applicability.classification.value,
                result.model_dump(mode="json"),
            )
            return result

        # --- checkpoint --- #
        pid_before = getattr(self.adapter, "pid", None)
        identity_before = getattr(self.adapter, "identity", None)
        if isinstance(identity_before, ProcessIdentity):
            identity_before = identity_before.model_copy(deep=True)
        checkpoint_config = current.runtime_configuration
        checkpoint_text, checkpoint_latency, checkpoint_metrics = self._probe()
        try:
            checkpoint_observed = ObservedConfigurationBaseline.capture(
                checkpoint_config, self.adapter.telemetry())
        except Exception:
            checkpoint_observed = ObservedConfigurationBaseline(
                RuntimeConfiguration(), ("baseline telemetry unavailable",))
        emit(
            KnowledgeEventType.ROLLBACK_CHECKPOINT,
            "checkpoint captured before rollback",
            {
                "checkpoint_configuration": checkpoint_config.model_dump(),
                "checkpoint_observed": checkpoint_observed.observed.model_dump(),
                "checkpoint_observation_problems": list(checkpoint_observed.problems),
                "probe_hash": _hash(checkpoint_text),
                "latency_s": checkpoint_latency,
                "pid": pid_before,
            },
        )
        self._maybe_fault("after_checkpoint", errors)

        # A checkpoint probe is real work. If another controller changed/reloaded
        # the target while it ran, neither this rollback nor checkpoint recovery
        # may overwrite that new owner state. Recheck before the first unload.
        if (self._current_runtime() != checkpoint_config
                or getattr(self.adapter, "pid", None) != pid_before
                or getattr(self.adapter, "identity", None) != identity_before):
            result = self._finish(
                execution_id, run_id, record,
                status=RollbackExecutionStatus.REJECTED_INAPPLICABLE,
                applicability=applicability.model_copy(update={
                    "classification": RollbackApplicabilityClass.INCOMPATIBLE_TARGET,
                    "reasons": [*applicability.reasons, "target changed during checkpoint probe"],
                }),
                integrity_ok=True, started=started, started_at=started_at,
                errors=[*errors, "target changed during checkpoint; no rollback or recovery attempted"],
                event_ids=event_ids, checkpoint_configuration=checkpoint_config,
                summary="rollback rejected: target changed during checkpoint (no target change)",
            )
            emit(KnowledgeEventType.ROLLBACK_REJECTED, "target_changed_during_checkpoint",
                 result.model_dump(mode="json"))
            return result

        # --- apply + reload + verify --- #
        result = self._apply_and_verify(
            record,
            execution_id=execution_id,
            run_id=run_id,
            applicability=applicability,
            checkpoint_config=checkpoint_config,
            checkpoint_observed=checkpoint_observed,
            checkpoint_text=checkpoint_text,
            checkpoint_metrics=checkpoint_metrics,
            pid_before=pid_before,
            identity_before=identity_before,
            started=started,
            started_at=started_at,
            event_ids=event_ids,
            errors=errors,
        )
        return result

    # ------------------------------------------------------------------ #
    # Core apply/verify with failure containment                           #
    # ------------------------------------------------------------------ #
    def _apply_and_verify(
        self,
        record: RollbackRecord,
        *,
        execution_id: str,
        run_id: str,
        applicability: RollbackApplicability,
        checkpoint_config: RuntimeConfiguration,
        checkpoint_observed: ObservedConfigurationBaseline,
        checkpoint_text: str,
        checkpoint_metrics: dict[str, float],
        pid_before: int | None,
        identity_before: ProcessIdentity | None,
        started: float,
        started_at: datetime,
        event_ids: list[str],
        errors: list[str],
    ) -> RollbackExecutionResult:
        reload_s: float | None = None
        ready: bool | None = None
        pid_after: int | None = None
        identity_after: ProcessIdentity | None = None
        restored_text = ""
        restored_metrics: dict[str, float] = {}
        try:
            rename_started = time.perf_counter()
            pid_after, identity_after, ready = self._apply(record.previous_state)
            reload_s = time.perf_counter() - rename_started
            self._maybe_fault("after_apply", errors)
            restored_text, restored_latency, restored_metrics = self._probe()
        except Exception as exc:  # noqa: BLE001 - contain and attempt recovery
            errors.append(f"{type(exc).__name__}: {exc}")
            recovery = self._recover(checkpoint_config, record, checkpoint_observed)
            status = (
                RollbackExecutionStatus.FAILED_RESTORED
                if recovery
                else RollbackExecutionStatus.FAILED_UNRECOVERED
            )
            result = self._finish(
                execution_id,
                run_id,
                record,
                status=status,
                applicability=applicability,
                integrity_ok=True,
                started=started,
                started_at=started_at,
                errors=errors,
                event_ids=event_ids,
                checkpoint_configuration=checkpoint_config,
                requested_configuration=record.previous_state,
                pid_before=pid_before,
                pid_after=getattr(self.adapter, "pid", None),
                identity_before=identity_before,
                identity_after=getattr(self.adapter, "identity", None),
                process_reloaded=True,
                reload_s=reload_s,
                ready=ready,
                recovery_attempted=True,
                recovery_verified=recovery,
                summary=(
                    "rollback failed; checkpoint recovery "
                    + ("succeeded" if recovery else "FAILED")
                ),
            )
            self._emit_final(record, KnowledgeEventType.ROLLBACK_FAILED, result)
            return result

        actual_config = self._current_runtime()
        observed_config, observed_ok = self._observed_configuration(record.previous_state)
        configuration_restored = self._configuration_matches(
            actual_config, record.previous_state, record.changes
        ) and observed_ok
        baseline_hash = (
            record.restoration_verification.baseline_probe_hash
            if record.restoration_verification is not None
            else None
        )
        restored_hash = _hash(restored_text)
        behavioural_passed = self._behaviour_matches(baseline_hash, restored_text)
        measurement_passed, measurement_detail = self._measurement_check(
            restored_metrics, record
        )
        process_reloaded = bool(identity_before is not None and identity_after is not None
            and (identity_before.pid != identity_after.pid
                 or abs(identity_before.create_time - identity_after.create_time) > 0.01))
        success = bool(
            configuration_restored
            and behavioural_passed
            and measurement_passed is not False
            and ready is not False
            and process_reloaded
            and restored_text
        )
        if not success:
            recovery = self._recover(checkpoint_config, record, checkpoint_observed)
        else:
            recovery = False
        # Only claim "restored" when the checkpoint recovery was actually verified.
        if success:
            status = RollbackExecutionStatus.APPLIED_VERIFIED
        elif recovery:
            status = RollbackExecutionStatus.FAILED_RESTORED
        else:
            status = RollbackExecutionStatus.FAILED_UNRECOVERED
        result = self._finish(
            execution_id,
            run_id,
            record,
            status=status,
            applicability=applicability,
            integrity_ok=True,
            started=started,
            started_at=started_at,
            errors=errors,
            event_ids=event_ids,
            checkpoint_configuration=checkpoint_config,
            requested_configuration=record.previous_state,
            applied_configuration=record.previous_state,
            actual_configuration=observed_config,
            pid_before=pid_before,
            pid_after=pid_after,
            identity_before=identity_before,
            identity_after=identity_after,
            process_reloaded=process_reloaded,
            reload_s=reload_s,
            ready=ready,
            configuration_restored=configuration_restored,
            behavioural_probe_attempted=True,
            behavioural_probe_passed=behavioural_passed,
            baseline_probe_hash=baseline_hash,
            checkpoint_probe_hash=_hash(checkpoint_text),
            restored_probe_hash=restored_hash,
            measurement_verification_passed=measurement_passed,
            measurement_detail=measurement_detail,
            restored_metrics=restored_metrics,
            baseline_metrics=self._baseline_metrics(record),
            recovery_attempted=not success,
            recovery_verified=recovery,
            summary=(
                "rollback applied and behaviourally verified"
                if success
                else "rollback produced a state that failed verification; checkpoint "
                "recovery " + ("succeeded" if recovery else "FAILED")
            ),
        )
        self._emit_final(
            record,
            KnowledgeEventType.ROLLBACK_EXECUTED
            if success
            else KnowledgeEventType.ROLLBACK_FAILED,
            result,
        )
        return result

    # ------------------------------------------------------------------ #
    # Target operations                                                    #
    # ------------------------------------------------------------------ #
    def _apply(
        self, configuration: RuntimeConfiguration
    ) -> tuple[int | None, ProcessIdentity | None, bool | None]:
        adapter = self.adapter
        adapter.unload()
        reader = getattr(adapter, "release_evidence", None)
        if callable(reader) and not reader().clean:
            raise RollbackExecutionError("previous target release unconfirmed; refusing rollback load")
        if adapter.is_loaded:
            raise RollbackExecutionError("previous target release unconfirmed; refusing rollback load")
        params = runtime_params_from(configuration)
        adapter.load(self.model, params)
        pid = getattr(adapter, "pid", None)
        identity = getattr(adapter, "identity", None)
        ready: bool | None = adapter.is_loaded
        last_load = getattr(adapter, "last_load", None)
        if last_load is not None:
            ready = bool(last_load.ready and last_load.resident_verified)
        return pid, identity, ready

    def _recover(
        self, checkpoint_config: RuntimeConfiguration, record: RollbackRecord,
        checkpoint_observed: ObservedConfigurationBaseline,
    ) -> bool:
        """Put the checkpoint back and report whether that was *verified*.

        Recovery only counts when the target is up, answers, and is running the
        checkpoint settings again. Anything less is reported as unrecovered, so a
        failed undo is never labelled "restored".
        """
        try:
            _pid, _identity, ready = self._apply(checkpoint_config)
            text, _latency, _metrics = self._probe()
        except Exception as exc:  # noqa: BLE001
            logger.warning("rollback checkpoint recovery failed: %s", exc)
            return False
        if ready is False or not text:
            logger.warning("rollback checkpoint recovery: target not ready or silent")
            return False
        shown, observed_ok = self._observed_configuration(checkpoint_config)
        problems = checkpoint_observed.problems_after(shown)
        if not self._configuration_matches(
            self._current_runtime(), checkpoint_config, record.changes
        ) or not observed_ok or problems:
            logger.warning("rollback checkpoint recovery: settings do not match checkpoint: %s", problems)
            return False
        return True

    def _probe(self) -> tuple[str, float, dict[str, float]]:
        request = self.probe or GenerationRequest(
            prompt="Reply with the single word: OK.",
            n_predict=self.policy.probe_n_predict,
            seed=self.policy.probe_seed,
            temperature=0.0,
        )
        started = time.perf_counter()
        result = self.adapter.generate(request)
        latency = time.perf_counter() - started
        metrics: dict[str, float] = {}
        if result.peak_rss_bytes is not None:
            metrics["peak_rss_bytes"] = float(result.peak_rss_bytes)
        if result.decode_tokens_per_sec is not None:
            metrics["decode_tokens_per_sec"] = float(result.decode_tokens_per_sec)
        if result.ttft_s is not None:
            metrics["ttft_s"] = float(result.ttft_s)
        return result.text or "", latency, metrics

    def _current_runtime(self) -> RuntimeConfiguration:
        params = getattr(self.adapter, "params", RuntimeParams())
        if not isinstance(params, RuntimeParams):  # pragma: no cover - defensive
            return RuntimeConfiguration()
        return runtime_configuration_from(params)

    def current_context(self) -> ApplicabilityContext:
        return build_current_context(
            hardware=self.hardware,
            model=self.model,
            workload=None,
            backend=self.backend,
            runtime=self._current_runtime(),
            diagnosis=None,
            conditions=None,
        )

    # ------------------------------------------------------------------ #
    # Verification helpers                                                 #
    # ------------------------------------------------------------------ #
    def _observed_configuration(self, expected: RuntimeConfiguration) -> tuple[RuntimeConfiguration, bool]:
        try:
            shown, contradictions, unknown = observed_configuration(expected, self.adapter.telemetry())
        except Exception:
            return RuntimeConfiguration(), False
        return shown, not contradictions and not unknown

    @staticmethod
    def _configuration_matches(
        actual: RuntimeConfiguration,
        expected: RuntimeConfiguration,
        changes: list,
    ) -> bool:
        return actual == expected

    def _behaviour_matches(self, baseline_hash: str | None, restored_text: str) -> bool:
        if not restored_text:
            return False
        if baseline_hash is None:
            # No recorded baseline probe: configuration + process reload stand in.
            return True
        return _hash(restored_text) == baseline_hash

    def _baseline_metrics(self, record: RollbackRecord) -> dict[str, float]:
        optimization = self.service.store.get_by_experiment(record.experiment_id)
        if optimization is None:
            return {}
        metrics: dict[str, float] = {}
        for name in ("peak_rss_bytes", "decode_tokens_per_sec", "ttft_s"):
            stats = optimization.baseline_metrics.get(name)
            if stats is not None and stats.mean is not None:
                metrics[name] = float(stats.mean)
        return metrics

    def _measurement_check(
        self, restored: dict[str, float], record: RollbackRecord
    ) -> tuple[bool | None, str]:
        baseline = self._baseline_metrics(record)
        if not baseline or not restored:
            return None, "baseline or restored measurement unavailable; not assessed"
        checks: list[str] = []
        passed = True
        for metric, expected in baseline.items():
            observed = restored.get(metric)
            if observed is None or expected == 0:
                continue
            relative = abs(observed - expected) / abs(expected)
            ok = relative <= self.policy.measurement_tolerance
            passed = passed and ok
            checks.append(
                f"{metric}: restored={observed:.4g} baseline={expected:.4g} "
                f"rel={relative:.3f} ok={ok}"
            )
        if not checks:
            return None, "no comparable measurement metrics"
        return passed, "; ".join(checks)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _maybe_fault(self, stage: str, errors: list[str]) -> None:
        if self.policy.fault_stage == stage:
            raise RollbackExecutionError(f"injected rollback fault at stage {stage!r}")

    def _emit_final(
        self,
        record: RollbackRecord,
        event_type: KnowledgeEventType,
        result: RollbackExecutionResult,
    ) -> None:
        try:
            event = self.service.store.append_event(
                f"opt-{record.experiment_id}",
                event_type,
                result.status.value,
                result.model_dump(mode="json"),
            )
            result.event_ids.append(event.event_id)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"final audit event failed: {type(exc).__name__}: {exc}")

    def _finish(
        self,
        execution_id: str,
        run_id: str,
        record: RollbackRecord,
        *,
        status: RollbackExecutionStatus,
        applicability: RollbackApplicability,
        integrity_ok: bool,
        started: float,
        started_at: datetime,
        errors: list[str],
        event_ids: list[str],
        summary: str,
        **fields: object,
    ) -> RollbackExecutionResult:
        finished = datetime.now(timezone.utc)
        data: dict[str, object] = {
            "execution_id": execution_id,
            "run_id": run_id,
            "rollback_id": record.rollback_id,
            "experiment_id": record.experiment_id,
            "status": status,
            "applicability": applicability,
            "integrity_ok": integrity_ok,
            "event_ids": list(event_ids),
            "errors": list(errors),
            "started_at": started_at,
            "finished_at": finished,
            "duration_s": time.perf_counter() - started,
            "summary": summary,
        }
        data.update(fields)
        return RollbackExecutionResult.model_validate(data)
