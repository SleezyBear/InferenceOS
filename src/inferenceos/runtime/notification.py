"""Structured notification abstraction (CLI now; a desktop/UI can replace it later)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from inferenceos.contracts.runtime import RuntimeNotification


class NotificationSink(ABC):
    """Where structured runtime notifications are delivered."""

    @abstractmethod
    def notify(self, notification: RuntimeNotification) -> None:
        ...


class CollectingNotificationSink(NotificationSink):
    """Records notifications in memory (tests / structured reports)."""

    def __init__(self) -> None:
        self.notifications: list[RuntimeNotification] = []

    def notify(self, notification: RuntimeNotification) -> None:
        self.notifications.append(notification)


class ConsoleNotificationSink(NotificationSink):
    """Human-readable CLI notification with the full structured cost/benefit breakdown."""

    def __init__(self) -> None:
        self.notifications: list[RuntimeNotification] = []

    def notify(self, notification: RuntimeNotification) -> None:
        self.notifications.append(notification)
        rec = notification.recommendation
        print("")
        print("================ InferenceOS runtime recommendation ================")
        print(f"proposal   : {rec.proposal_id}")
        print(f"detected   : {rec.condition}")
        print(f"why        : {rec.why_it_matters}")
        print(f"analysis   : {rec.analysis_summary}")
        if rec.proposed_action_id is not None:
            print(f"proposed   : {rec.proposed_action_id} {rec.proposed_parameters}")
        print(f"benefit    : {rec.expected_benefit}")
        print(f"cost/disrupt: ~{rec.expected_cost_s:.1f}s (disruption ~{rec.expected_disruption_s:.1f}s)")
        print(f"pause      : {rec.pause_required} checkpoint: {rec.checkpoint_planned} reversible: {rec.reversible}")
        print(f"learning   : {rec.expected_learning}")
        print(f"approval   : required={rec.requires_approval} class={rec.intervention_class.value}")
        print(f"context    : {rec.context_identity[:16]}... expires={rec.expires_at}")
        print("===================================================================")
