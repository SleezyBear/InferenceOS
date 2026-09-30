"""Controlled action-execution abstraction.

The experiment engine never mutates configuration ad hoc and never executes a
candidate's data as a command. It resolves a registered :class:`CandidateAction`,
validates its parameters against the registry schema, captures the prior state,
applies exactly that action, and can restore the prior state.

For the current process-per-generation llama.cpp backend there is no long-lived
inference process to mutate; the mutated object is the *target model's runtime
configuration* used for the real measurement runs. The executor is an
inference-level, reversible, single-target surface: it cannot touch OS state,
other processes, or the model file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from inferenceos.contracts.enums import ParameterType, RestorationStatus
from inferenceos.contracts.experiment import (
    ApplicationResult,
    ExperimentChange,
    RestorationResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.optimization import CandidateAction, ParameterSpec

from inferenceos.cards.base import LLAMA_CPP
from inferenceos.cards.catalog import default_catalog, get_card
from inferenceos.cards.safety import ContextSafetyError, check_context_unchanged


def _card_setting(action_id: str, program: str = LLAMA_CPP) -> str | None:
    """The runtime setting a card changes for ``program`` (``None``: no surface).

    Suggest-only cards never have a surface: InferenceOS never applies them.
    """
    card = get_card(action_id)
    if card is None or card.suggest_only:
        return None
    return card.setting_for(program)


def supported_actions() -> set[str]:
    """Every card InferenceOS can apply and undo on the llama.cpp runtime."""
    return {
        card.card_id
        for card in default_catalog().applied_cards()
        if card.setting_for(LLAMA_CPP) is not None
    }


def action_fields(action_id: str) -> tuple[str, ...]:
    """The runtime-configuration fields changed by a card (empty: no surface)."""
    field = _card_setting(action_id)
    return (field,) if field is not None else ()


def _spec(action: CandidateAction, name: str) -> ParameterSpec | None:
    for spec in action.parameters:
        if spec.name == name:
            return spec
    return None


def validate_parameters(
    action: CandidateAction, parameters: dict[str, object]
) -> tuple[bool, str]:
    """Validate candidate parameters against the registry schema (never trust data)."""
    for name, value in parameters.items():
        spec = _spec(action, name)
        if spec is None:
            return False, f"undeclared parameter {name!r} for action {action.action_id}"
        if spec.type is ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"parameter {name!r} must be boolean"
        elif spec.type is ParameterType.INTEGER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False, f"parameter {name!r} must be an integer"
            if float(value) != int(value):
                return False, f"parameter {name!r} must be an integer"
            integer = int(value)
            if spec.minimum is not None and integer < spec.minimum:
                return False, f"parameter {name!r} below minimum {spec.minimum:g}"
            if spec.maximum is not None and integer > spec.maximum:
                return False, f"parameter {name!r} above maximum {spec.maximum:g}"
        elif spec.type is ParameterType.FLOAT:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False, f"parameter {name!r} must be numeric"
        elif spec.type is ParameterType.STRING:
            if not isinstance(value, str):
                return False, f"parameter {name!r} must be a string"
            if spec.choices and value not in spec.choices:
                return False, f"parameter {name!r} not in allowed choices"
    return True, ""


def apply_to_configuration(
    action_id: str,
    parameters: dict[str, object],
    configuration: RuntimeConfiguration,
) -> RuntimeConfiguration:
    """Return a copy of ``configuration`` with the card's change applied.

    Raises :class:`ContextSafetyError` if the result would have a different context
    size (the automatic context-safety check), and ``ValueError`` if the card has no
    runtime surface.
    """
    field = _card_setting(action_id)
    if field is None:
        raise ValueError(f"action {action_id!r} has no runtime surface")
    updates: dict[str, object] = {field: value for value in parameters.values()}
    updated = configuration.model_copy(update=updates)
    check_context_unchanged(configuration, updated, card_id=action_id)
    return updated


class ActionExecutor(ABC):
    """Applies/restores exactly one registered action's runtime change."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Executor identity (e.g. the backend it configures)."""

    @abstractmethod
    def supports(self, action_id: str) -> bool:
        """Whether this executor has a real, safe surface for the action."""

    @abstractmethod
    def set_configuration(self, configuration: RuntimeConfiguration) -> None:
        """Reset the working configuration to a known baseline."""

    @abstractmethod
    def current(self) -> RuntimeConfiguration:
        """The current working configuration."""

    @abstractmethod
    def apply(
        self, action: CandidateAction, parameters: dict[str, object]
    ) -> ApplicationResult:
        """Capture the prior state and apply the registered action."""

    @abstractmethod
    def restore(self, expected: RuntimeConfiguration) -> RestorationResult:
        """Restore and verify the exact expected configuration."""


class RuntimeConfigurationExecutor(ActionExecutor):
    """Real, reversible executor over the target model's runtime configuration."""

    def __init__(
        self,
        *,
        backend_name: str = "llama.cpp",
        working: RuntimeConfiguration | None = None,
        supported: set[str] | None = None,
    ) -> None:
        self._backend_name = backend_name
        self._working = working or RuntimeConfiguration()
        # ``None`` means "whatever the card catalog supports right now", so a card
        # registered later is picked up without touching this class.
        self._supported = supported

    @property
    def name(self) -> str:
        return f"runtime-configuration[{self._backend_name}]"

    def supports(self, action_id: str) -> bool:
        if self._supported is not None:
            return action_id in self._supported
        return action_id in supported_actions()

    def set_configuration(self, configuration: RuntimeConfiguration) -> None:
        self._working = configuration

    def current(self) -> RuntimeConfiguration:
        return self._working

    def apply(
        self, action: CandidateAction, parameters: dict[str, object]
    ) -> ApplicationResult:
        now = datetime.now(timezone.utc)
        if not self.supports(action.action_id):
            return ApplicationResult(
                applied=False,
                action_id=action.action_id,
                supported=False,
                reason=(
                    f"action {action.action_id!r} has no safe runtime surface in this "
                    "backend and must not be executed"
                ),
                applied_at=now,
            )
        ok, reason = validate_parameters(action, parameters)
        if not ok:
            return ApplicationResult(
                applied=False,
                action_id=action.action_id,
                supported=True,
                reason=reason,
                applied_at=now,
            )
        prior = self._working
        try:
            self._working = apply_to_configuration(action.action_id, parameters, prior)
        except ContextSafetyError as exc:
            return ApplicationResult(
                applied=False,
                action_id=action.action_id,
                supported=True,
                reason=str(exc),
                applied_at=now,
            )
        field = _card_setting(action.action_id)
        changes = [
            ExperimentChange(
                parameter=name,
                prior_value=getattr(prior, field, None) if field is not None else None,
                applied_value=value,
            )
            for name, value in parameters.items()
        ]
        return ApplicationResult(
            applied=True,
            action_id=action.action_id,
            supported=True,
            changes=changes,
            reason="applied via registered runtime surface",
            applied_at=now,
        )

    def restore(self, expected: RuntimeConfiguration) -> RestorationResult:
        now = datetime.now(timezone.utc)
        before = self._working
        changes = _diff(before, expected)
        self._working = expected
        verified = self._working == expected
        return RestorationResult(
            attempted=True,
            status=RestorationStatus.RESTORED if verified else RestorationStatus.FAILED,
            expected_configuration=expected,
            restored_configuration=self._working,
            verified=verified,
            changes=changes,
            reason=(
                "baseline runtime configuration restored and verified"
                if verified
                else "restored configuration does not match the protected baseline"
            ),
            restored_at=now,
        )


def _diff(
    before: RuntimeConfiguration, after: RuntimeConfiguration
) -> list[ExperimentChange]:
    changes: list[ExperimentChange] = []
    for field in type(before).model_fields:
        old = getattr(before, field)
        new = getattr(after, field)
        if old != new:
            changes.append(
                ExperimentChange(parameter=field, prior_value=old, applied_value=new)
            )
    return changes
