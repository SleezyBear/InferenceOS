"""H6 optimizer self-target exclusion tests."""

from __future__ import annotations

import os

from inferenceos.contracts.enums import ExternalResource
from inferenceos.external.attribution import assess_external_attribution
from inferenceos.external.engine import ExternalRecommendationEngine
from inferenceos.external.observer import WindowsProcessObserver
from inferenceos.optimization.actions import build_default_registry
from inferenceos.runtime.approval import AutoRejectApprovalProvider
from tests.test_external_engine import _policy  # reuse the controlled external policy
from tests.external_helpers import make_process_observation, make_report


class _RecordingObserver:
    """Observer that records the own_pid the engine passes it."""

    def __init__(self, report: object) -> None:
        self._report = report
        self.own_pid: int | None = None

    def observe(self, **kwargs: object) -> object:
        value = kwargs.get("own_pid")
        self.own_pid = value if isinstance(value, int) else None
        return self._report


def test_engine_defaults_own_pid_to_this_process() -> None:
    observer = _RecordingObserver(make_report(processes=[], target_pid=None))
    engine = ExternalRecommendationEngine(
        observer=observer,  # type: ignore[arg-type]
        registry=build_default_registry(),
        policy=_policy(),
        approval_provider=AutoRejectApprovalProvider(),
    )
    engine.observe_once()
    assert observer.own_pid == os.getpid()


def test_optimizer_pid_excluded_but_external_and_target_kept() -> None:
    own_pid = os.getpid()
    target = make_process_observation(
        pid=100, name="llama-server.exe", rss_bytes=1_300_000_000, is_target=True
    )
    optimizer_self = make_process_observation(
        pid=own_pid, name="python.exe", rss_bytes=6 * 1024**3, cpu_percent=180.0
    )
    external = make_process_observation(
        pid=4242, name="external.exe", rss_bytes=6 * 1024**3, cpu_percent=180.0
    )
    report = make_report(
        processes=[target, optimizer_self, external], target_pid=100, memory_pressure=0.9
    ).model_copy(update={"own_pid": own_pid})

    assessments = assess_external_attribution(report, _policy(), target_degraded=True)
    pids = {assessment.identity.pid for assessment in assessments}

    assert own_pid not in pids  # the optimizer itself is never a candidate
    assert 4242 in pids  # a legitimate external process is still selectable
    # The target is never an external candidate either.
    assert 100 not in pids


def test_observer_never_returns_the_optimizer_process() -> None:
    observer = WindowsProcessObserver()
    report = observer.observe(own_pid=os.getpid(), max_processes=200)
    assert all(observation.identity.pid != os.getpid() for observation in report.processes)
    assert report.own_pid == os.getpid()
