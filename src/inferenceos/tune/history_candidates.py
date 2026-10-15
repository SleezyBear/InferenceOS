"""Historical screening hints: a permutation, never current performance evidence."""
from __future__ import annotations

import math
import statistics
from typing import Any

from inferenceos.tune.coordinator import PRACTICAL_PATH_IDS
from inferenceos.tune.history_cost import HistoricalCosts, MAX_RUNS, MIN_RUNS, compatible_request
from inferenceos.tune.learning import request_identity_problems


class HistoricalCandidates:
    """Reuse the cost snapshot's host/model/date/origin gate without owning a queue.

    Require three independent runs with complete, linked screening measurements and
    the same goal, comparison base, candidate and workload. Live reference runtime
    identity is a near-match hint for an as-yet unmeasured candidate, never proof of
    that candidate's effective runtime. Only existing values can be promoted.
    The latest same-scope attempt supersedes older readings even when unusable.
    Clean readings from other run outcomes can suppress, but cannot promote.
    """

    def __init__(self, snapshot: HistoricalCosts) -> None:
        self.snapshot = snapshot

    def rank(self, category: str, values: tuple[int, ...], reference: dict[str, Any],
             candidates: dict[int, dict[str, Any]], goal: dict[str, Any],
             smallest_gain: float) -> tuple[tuple[int, ...], dict[str, Any]]:
        scores: dict[int, float] = {}
        provenance: dict[str, Any] = {}
        suppressed: dict[str, list[str]] = {}
        if any(request_identity_problems(req) or req["configuration"]["context_length"] != self.snapshot.context
               for req in (reference, *candidates.values())):
            return values, {"hints": {}, "why": "incomplete or incompatible request context"}
        for value in values:
            samples: list[tuple[str, float, float, int, int, bool]] = []
            for run in self.snapshot.runs:
                may_promote = run.disposition.get("outcome") in ("saved", "no change")
                observations = {o.operation_id: o for o in run.observations}
                for event in reversed(run.comparisons):
                    try:
                        if (event.get("hint_version") != "candidate-hint-1" or event.get("goal") != goal
                                or event.get("category") != category or type(event.get("value")) is not int
                                or event["value"] != value or event.get("path") != PRACTICAL_PATH_IDS[category]
                                or event.get("kind") not in ("screen", "weak repeat")):
                            continue
                        if (event["reference"]["configuration"] != reference["configuration"]
                                or event["candidate"]["configuration"] != candidates[value]["configuration"]
                                or event["question"] != reference["workload"]["id"]):
                            continue
                    except (KeyError, TypeError, ValueError, AttributeError):
                        continue  # cannot establish which comparison this describes
                    # This is the latest attempt for this exact comparison scope.
                    # Unusable evidence must not resurrect an earlier positive read.
                    valid = False
                    try:
                        gain, noise = event["gain"], event["raw_noise"]
                        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (gain, noise)) or noise < 0:
                            raise ValueError("unusable latest gain/noise")
                        ids = [event[side]["measurement"] for side in ("reference", "candidate")]
                        if any(type(i) is not int for i in ids) or ids[0] == ids[1]:
                            raise ValueError("unusable latest measurement links")
                        valid = True
                        for side, op_id, req in zip(("reference", "candidate"), ids,
                                                   (reference, candidates[value])):
                            op = observations[op_id]
                            if (event[side]["configuration"] != req["configuration"]
                                    or event["question"] != req["workload"]["id"]
                                    or op.kind != "screening" or op.status != "completed" or op.failure is not None
                                    or op.retries != 0 or op.valid_runs is None
                                    or op.valid_runs < req["policy"]["min_valid_samples"]
                                    or op.phases.get("raw", {}).get("invalid_runs") != 0
                                    or not math.isfinite(op.wall_s) or op.wall_s <= 0
                                    or not compatible_request(op.request, req, op.wall_s)):
                                valid = False
                                break
                        if valid:
                            samples.append((run.run_id, gain, noise, ids[0], ids[1], may_promote))
                    except (KeyError, TypeError, ValueError, AttributeError):
                        valid = False
                    if not valid:
                        suppressed.setdefault(str(value), []).append(run.run_id)
                    break  # latest attempt only, including failed/unknown repeats
                if len(samples) >= MAX_RUNS:
                    break
            if len(samples) < MIN_RUNS:
                continue
            # Provisional conservative promotion rule. Negative recent readings are
            # included; an old success is not a permanent preference or blacklist.
            # Unconfirmed/interrupted runs may supply contrary clean evidence, but
            # never a positive promotion contribution. Do not invent a failure gain.
            margins = [g - max(smallest_gain, 2 * n) for _, g, n, _, _, _ in samples]
            margin = float(statistics.median(m if s[-1] else min(m, 0.0) for m, s in zip(margins, samples)))
            provenance[str(value)] = {"run_ids": [r for r, *_ in samples], "run_count": len(samples),
                "promotion_run_count": sum(s[-1] for s in samples),
                "median_margin": margin, "comparisons": [
                    {"run_id": r, "gain": g, "noise": n, "reference_operation": a, "candidate_operation": b,
                     "may_promote": p}
                    for r, g, n, a, b, p in samples]}
            if margin > 0 and sum(s[-1] for s in samples) >= MIN_RUNS:
                scores[value] = margin
        ordered = tuple(sorted(values, key=lambda v: -scores.get(v, 0.0)))
        return ordered, {"rule": "candidate-hint-2", "provisional": True,
            "match": "same comparison; candidate runtime inferred from observed live reference, requires fresh screening",
            "goal": goal, "reference_request": reference, "candidate_requests": candidates,
            "before": list(values), "after": list(ordered), "hints": provenance,
            "suppressed_latest_attempts": suppressed}
