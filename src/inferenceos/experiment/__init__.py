"""Phase 7 experiment execution & quality validation.

Applies exactly one eligible Phase 6 candidate through a controlled executor,
measures real before/after inference via the Phase 3 harness, validates output
quality, decides keep/reject on transparent statistical grounds, and always restores
the protected baseline. It does not persist deployments or learn from history.
"""

from __future__ import annotations

from inferenceos.experiment.audit import build_audit_record
from inferenceos.experiment.eligibility import RevalidationResult, revalidate_candidate
from inferenceos.experiment.engine import run_experiment
from inferenceos.experiment.executor import (
    ActionExecutor,
    RuntimeConfigurationExecutor,
    apply_to_configuration,
    supported_actions,
    validate_parameters,
)
from inferenceos.experiment.policy import build_experiment_policy
from inferenceos.experiment.report import render_experiment
from inferenceos.experiment.service import ExperimentInputError, run_candidate_experiment

__all__ = [
    "ActionExecutor",
    "ExperimentInputError",
    "RevalidationResult",
    "RuntimeConfigurationExecutor",
    "apply_to_configuration",
    "build_audit_record",
    "build_experiment_policy",
    "render_experiment",
    "revalidate_candidate",
    "run_candidate_experiment",
    "run_experiment",
    "supported_actions",
    "validate_parameters",
]
