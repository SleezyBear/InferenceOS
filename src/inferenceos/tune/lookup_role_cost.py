"""Measured inclusive intervals for D2 role work. No forecast or tune permission."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Iterator, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from inferenceos.tune.lookup_corpus import FrozenInput
from inferenceos.tune.lookup_qualification import SHA256

VERSION = "lookup-role-intervals-1"
KINDS = ("job", "pair", "suite-arm", "cycle", "gate", "start", "ready-identity",
         "tokenize", "request", "oracle", "release-identity", "cleanup", "power", "publication", "rest")


class RoleScope(FrozenInput):
    model: Literal["llama3b", "gemma1b"]
    family: Literal["copy-prose", "copy-code", "extract-rows"]
    role: Literal["primary", "calibration"]
    arm: Literal["control", "lookup"]
    model_sha256: SHA256
    engine_files_sha256: SHA256
    suite_sha256: SHA256
    requested_sha256: SHA256
    effective_sha256: SHA256
    machine_sha256: SHA256
    request_sha256: SHA256
    executor_sha256: SHA256
    ordered_cases: tuple[tuple[str, SHA256], ...] = Field(min_length=7, max_length=7)

    @model_validator(mode='after')
    def distinct_cases(self) -> Self:
        if len({c[0] for c in self.ordered_cases})!=7:raise ValueError('seven distinct ordered suite cases required')
        return self


class Interval(FrozenInput):
    sample_id: str = Field(min_length=1, max_length=180)
    parent_id: str | None
    kind: str
    start_ns: int = Field(ge=0)
    end_ns: int | None = Field(ge=0)
    outcome: Literal["running", "complete", "failed"]
    scope: RoleScope | None
    case_id: str | None

    @model_validator(mode="after")
    def valid(self) -> Self:
        if self.kind not in KINDS or ((self.end_ns is None) != (self.outcome == "running")):
            raise ValueError("known measured interval and explicit completion required")
        if self.end_ns is not None and self.end_ns <= self.start_ns:
            raise ValueError("positive integer nanosecond interval required")
        if self.kind in ("suite-arm", "cycle") and self.scope is None:
            raise ValueError("role/case work needs its complete suite scope")
        if self.kind == "cycle" and (self.case_id is None or self.case_id not in dict(self.scope.ordered_cases if self.scope else ())):
            raise ValueError("cycle must identify one case of its enclosing suite")
        return self


class Ledger:
    """Single append-only owner. Failed spans never become completed cost samples."""
    def __init__(self, run_id: str, clock: Callable[[], int]) -> None:
        self.run_id, self.clock = run_id, clock
        self.rows: list[dict[str, Any]] = []
        self.stack: list[str] = []

    @contextmanager
    def measure(self, kind: str, *, scope: RoleScope | None = None, case_id: str | None = None) -> Iterator[dict[str, Any]]:
        row: dict[str, Any] = dict(sample_id=f"{self.run_id}/cost/{len(self.rows)}", parent_id=self.stack[-1] if self.stack else None,
                                   kind=kind, start_ns=self.clock(), end_ns=None, outcome="running",
                                   scope=scope.model_dump(mode="json") if scope else None, case_id=case_id)
        self.rows.append(row); self.stack.append(row["sample_id"])
        try:
            yield row
        except BaseException:
            row["outcome"] = "failed"
            raise
        else:
            if row["outcome"] == "running":
                row["outcome"] = "complete"
        finally:
            row["end_ns"] = self.clock()
            self.stack.pop()

    def snapshot(self) -> list[dict[str, Any]]:
        return deepcopy(self.rows)


def audit_intervals(raw: Any) -> dict[str, Any]:
    """Check containment/order and report gaps; inclusive costs are never added twice."""
    try:
        if not isinstance(raw, list) or not raw:
            raise ValueError("nonempty interval population required")
        import json
        rows = [Interval.model_validate_json(json.dumps(r, allow_nan=False)) for r in raw]
        ids = {r.sample_id: r for r in rows}
        if len(ids) != len(rows) or sum(r.parent_id is None for r in rows) != 1 or sum(r.kind=='job' for r in rows)!=1 or rows[0].kind != "job":
            raise ValueError("one job root and unique interval IDs required")
        if any(b.start_ns < a.start_ns for a,b in zip(rows,rows[1:])):
            raise ValueError("intervals must retain start order")
        children_by_parent: dict[str | None,list[Interval]] = {}
        for item in rows:children_by_parent.setdefault(item.parent_id,[]).append(item)
        gaps: dict[str, int | None] = {}
        for row in rows:
            if row.parent_id is not None:
                parent = ids.get(row.parent_id)
                if parent is None or parent.start_ns > row.start_ns or parent.sample_id == row.sample_id:
                    raise ValueError("invalid interval parent")
                if parent.end_ns is not None and (row.end_ns is None or row.end_ns > parent.end_ns):
                    raise ValueError("child outside measured parent")
            ancestry=set();cursor=row
            while cursor.parent_id is not None:
                if cursor.sample_id in ancestry or cursor.parent_id not in ids:raise ValueError('cyclic/disconnected cost ancestry')
                ancestry.add(cursor.sample_id);cursor=ids[cursor.parent_id]
            if cursor.sample_id!=rows[0].sample_id:raise ValueError('cost outside root job')
            children = children_by_parent.get(row.sample_id,[])
            if children and row.kind not in ('job','pair','suite-arm','cycle'):
                raise ValueError('operation intervals must be leaves')
            if any(a.end_ns is None or a.end_ns > b.start_ns for a,b in zip(children,children[1:])):
                raise ValueError("sibling work overlaps or remains open")
            gaps[row.sample_id] = (None if row.end_ns is None or any(c.end_ns is None for c in children) else
                                   row.end_ns-row.start_ns-sum(c.end_ns-c.start_ns for c in children if c.end_ns is not None))
            # A completed role/case cannot hide an unsuccessful comparison,
            # checkpoint, or other operation merely because its cycles finished.
            # Job/pair containers may contain an intentionally futile suite;
            # the study auditor separately proves that exception from answers.
            if row.kind in ('suite-arm','cycle') and row.outcome == 'complete' and any(c.outcome != 'complete' for c in children):
                raise ValueError('completed role/case cannot hide unsuccessful operations')
            if row.kind == "suite-arm" and row.outcome == "complete":
                cycles = [c for c in children if c.kind == "cycle"]
                if row.scope is None or [c.case_id for c in cycles] != [c[0] for c in row.scope.ordered_cases]:
                    raise ValueError("complete role cost must contain all seven ordered case cycles")
                if any(c.outcome != "complete" or c.scope != row.scope for c in cycles):
                    raise ValueError("role cost cannot hide incomplete/different scoped cycles")
        distributions = {}
        for kind in KINDS:
            values = [int(r.end_ns)-r.start_ns for r in rows if r.kind == kind and r.outcome == "complete" and r.end_ns is not None]
            ordered=sorted(values);n=len(ordered)
            median=(dict(numerator=ordered[n//2],denominator=1) if n%2 else
                    dict(numerator=ordered[n//2-1]+ordered[n//2],denominator=2)) if n else None
            distributions[kind] = dict(count=n,min_ns=min(values) if values else None,
                                       median_ns=median,max_ns=max(values) if values else None)
        return dict(version=VERSION, valid=True, problems=[], outcomes=dict(Counter(r.outcome for r in rows)),
                    unassigned_ns=gaps, distributions=distributions, inclusive_do_not_sum=True, forecast_available=False)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        return dict(version=VERSION, valid=False, problems=[str(exc)],forecast_available=False)
