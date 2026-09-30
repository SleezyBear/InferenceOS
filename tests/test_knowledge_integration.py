"""Integration: a real Phase 7 artifact persists into a real durable store."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.config import Settings
from inferenceos.contracts.experiment import ExperimentResult
from inferenceos.knowledge import evaluate_applicability
from tests.knowledge_helpers import make_service, temp_db_path


def _latest_real_experiment() -> Path | None:
    settings = Settings()
    paths = sorted(
        list(settings.reports_dir.glob("phase7_experiment_*.json"))
        + list(settings.reports_dir.glob("experiment_*.json")),
        key=lambda item: item.stat().st_mtime,
    )
    paths = [path for path in paths if "audit" not in path.name]
    return paths[-1] if paths else None


def test_real_experiment_persists_and_retrieves_after_reopen() -> None:
    path = _latest_real_experiment()
    if path is None:
        pytest.skip("no real Phase 7 experiment artifact available")
    result = ExperimentResult.from_json(path.read_text(encoding="utf-8"))
    service = make_service(temp_db_path())
    outcome = service.persist_result(result)
    assert outcome.persisted

    reopened = make_service(service.store.db_path)
    record = reopened.store.get_by_experiment(result.experiment_id)
    assert record is not None
    assert record.experiment_id == result.experiment_id
    assert record.context.fingerprint.identity_hash

    applicable = evaluate_applicability(record, record.context, reopened.policy)
    if record.action_registered:
        # The change still has a card: the saved result is reusable as-is.
        assert applicable.classification.value == "directly_applicable"
    else:
        # The change no longer has a card (for example an old context-size test,
        # deleted in step 3): the saved result must never be reused.
        assert applicable.classification.value == "not_applicable"

    rollback = reopened.rollback_for_experiment(result.experiment_id)
    if result.application is not None and result.application.applied:
        assert rollback is not None
        # A saved undo is only trusted while its change still has a card.
        assert rollback.verification.verified is record.action_registered
