"""Step 5 fix: past-result lookups keep the newest records, not the oldest."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from inferenceos.config import Settings
from inferenceos.knowledge import KnowledgePolicy, KnowledgeService, build_knowledge_policy
from inferenceos.contracts.knowledge import OptimizationRecord
from tests.knowledge_helpers import base_result, current_context, make_service


def _store_five(service_limit: int = 2) -> tuple[KnowledgeService, OptimizationRecord]:
    policy = build_knowledge_policy(Settings()).model_copy(update={"lookup_limit": service_limit})
    service = make_service(policy=policy)
    template = service.build_record(base_result())
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    for index in range(5):
        record = template.model_copy(
            update={
                "record_id": f"rec-{index}",
                "experiment_id": f"exp-{index}",
                "created_at": start + timedelta(days=index),
            }
        )
        assert service.store.record_optimization(record).record_id == f"rec-{index}"
    return service, template


def test_applicable_lookup_uses_the_newest_records() -> None:
    service, template = _store_five(service_limit=2)
    results = service.query_applicable(template.action_id, current_context(template))
    assert sorted(result.experiment_id for result in results) == ["exp-3", "exp-4"]


def test_history_is_the_newest_records_oldest_first() -> None:
    service, template = _store_five()
    history = service.store.history_for_action(template.action_id)
    assert [record.experiment_id for record in history] == [f"exp-{i}" for i in range(5)]


def test_policy_type_is_unchanged() -> None:
    assert isinstance(build_knowledge_policy(Settings()), KnowledgePolicy)
