"""Phase 9R bridge to the Phase 5R resident deep-diagnosis lifecycle.

The Phase 9 loop normally invokes the process-per-generation ``run_deep_diagnosis``.
For a fresh resident run we instead drive the real Phase 5R lifecycle (unload target,
load reasoner, diagnose, unload reasoner, reload target, verify) through the injected
``deep_diagnosis_runner`` hook. This does not create a second lifecycle implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferenceos.backends.resident_server import LlamaServerResidentBackend
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.lifecycle import LifecyclePolicy, RequestPolicy
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.contracts.reasoning import DeepDiagnosisOutcome
from inferenceos.interfaces.backend_adapter import RuntimeParams
from inferenceos.logging_config import get_logger
from inferenceos.reasoning.resident_lifecycle import ResidentLifecycleRunner
from inferenceos.reasoning.resident_reasoner import LlamaServerReasoner

logger = get_logger("loop.resident")


@dataclass
class ResidentDeepDiagnosisRunner:
    """Runs the real resident deep-diagnosis lifecycle when escalation fires."""

    backend: LlamaServerResidentBackend
    target_model: ModelProfile
    target_params: RuntimeParams
    reasoner: LlamaServerReasoner
    policy: LifecyclePolicy
    request_policy: str = RequestPolicy.REJECT
    interruption_cost_s: float = 0.0
    confidence_threshold: float = 0.6

    def __call__(
        self, diagnosis: BottleneckDiagnosis, report: BaselineReport
    ) -> DeepDiagnosisOutcome | None:
        started_here = False
        if not self.backend.is_running():
            self.backend.start(self.target_model, self.target_params)
            started_here = True
        try:
            runner = ResidentLifecycleRunner(
                target=self.backend,
                target_model=self.target_model,
                target_params=self.target_params,
                policy=self.policy,
                reasoner=self.reasoner,
                request_policy=self.request_policy,
                interruption_cost_s=self.interruption_cost_s,
            )
            outcome = runner.run(
                diagnosis,
                report=report,
                confidence_threshold=self.confidence_threshold,
            )
        finally:
            # Leave the target unloaded so the Phase 3 profiler can own it per run.
            try:
                if self.backend.is_running():
                    self.backend.stop()
            except Exception as exc:  # noqa: BLE001 - cleanup must not crash the loop
                logger.warning("resident deep-diagnosis cleanup failed: %s", exc)
        _ = started_here
        return outcome.diagnosis if outcome.entered else None
