"""Model-free lookup decoding: fixed development experiment, never a tune path.

Only native b11139 ngram-simple versus explicit none. Independent fresh loads,
one fresh bounded natural request per load, exact generated IDs AND text preservation.
Missing drafting evidence cannot become an attributed speed gain. No settings save,
history consumer, scheduler, semantic judge or automatic admission lives here.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Protocol, TypeGuard

from inferenceos.tune.prompt_reuse import Cleanup, EngineVariant
from inferenceos.tune.stats import block_bounds

VERSION = "lookup-decoding-1"
PROTOCOL = "lookup-study-1"
BLOCKS = 5
TOKENS = 128
SEED = 42
LOOKUP_LENGTH = 12
DRAFT_LENGTH = 16
# One final look, eight registered model/workload comparisons, family alpha .05.
ALPHA = 0.05 / 8
CONTROL = EngineVariant("control")
LOOKUP = EngineVariant("lookup")
ARMS = (CONTROL.variant_id, LOOKUP.variant_id)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class LookupCase:
    case_id: str
    description: str
    prompt: str

    @property
    def sha256(self) -> str:
        return digest(asdict(self))


def build_cases() -> tuple[LookupCase, ...]:
    prose = "\n".join(f"Record {n}: inspect the inlet, clean the filter, and write the reading in the log."
                       for n in range(1, 17))
    code = "\n".join(f"def reading_{n}(value):\n    return {{'sensor': {n}, 'reading': value, 'valid': value >= 0}}\n"
                      for n in range(1, 13))
    table = "\n".join(f"row {n}: unit=unit_{n}; inlet={41+n}; outlet={30+n}; state=ready"
                       for n in range(1, 21))
    return (
        LookupCase("copy-prose", "Copy repeated prose; intended lookup opportunity",
                   "Copy the following records exactly, starting at Record 1. Do not explain.\n" + prose + "\nCopy:\n"),
        LookupCase("copy-code", "Copy repeated code; intended lookup opportunity",
                   "Copy this Python source exactly, without code fences or explanation:\n" + code + "\nCopy:\n"),
        LookupCase("extract-rows", "Extract selected structured rows, rather than copy everything",
                   "Return only the even-numbered rows exactly as written, one per line.\n" + table + "\nAnswer:\n"),
        LookupCase("compose-prose", "Less copying; a negative-opportunity control, not a promised zero",
                   "Write a short original explanation for a beginner of why a maintenance log helps "
                   "a team diagnose intermittent faults. Discuss an example, a limitation and a practical "
                   "next step. Do not quote a supplied document.\nAnswer:\n"),
    )


def plan(cases: Sequence[LookupCase] | None = None) -> list[tuple[str, int, str]]:
    return [(case.case_id, block, arm)
            for case in (build_cases() if cases is None else cases)
            for block in range(BLOCKS)
            for arm in (ARMS if block % 2 == 0 else tuple(reversed(ARMS)))]


def activation_problems(text: Any, arm: str) -> list[str]:
    """Observed b11139 implementation initialization, not request-as-observation."""
    if not isinstance(text, str) or "model loaded" not in text:
        return ["startup log does not reach model readiness"]
    if arm == "control":
        return ["none control has an initialized speculative context"] if "speculative decoding context initialized" in text else []
    if arm != "lookup":
        return ["unknown lookup arm"]
    if ("adding speculative implementation 'ngram-simple'" not in text
            or not re.search(r"size_n=12,\s*size_m=16,\s*min_hits=1", text)
            or "speculative decoding context initialized" not in text
            or "speculative decoding not supported by this context" in text):
        return ["lookup implementation/size activation unverified"]
    return []


class LookupEngine(Protocol):
    def start(self, variant: EngineVariant) -> None: ...
    def request(self, case: LookupCase) -> dict[str, Any]: ...
    def stop(self) -> Cleanup: ...
    def load_problems(self) -> list[str]: ...


def _integer(value: Any, *, minimum: int = 0) -> TypeGuard[int]:
    return type(value) is int and value >= minimum


def _positive(value: Any) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def reading_problems(reading: Any, arm: str) -> list[str]:
    """Validate original work evidence, not stored usability labels or timing alone."""
    if not isinstance(reading, dict):
        return ["reading missing or not an object"]
    problems = []
    raw = reading.get("counters")
    if not isinstance(raw, dict):
        return ["native counters missing or not an object"]
    if not isinstance(reading.get("text"), str) or not reading["text"]:
        problems.append("no written text")
    ids = reading.get("token_ids")
    if not isinstance(ids, list) or not 1 <= len(ids) <= TOKENS or any(not _integer(x) for x in ids):
        problems.append("complete generated token IDs not demonstrated")
    if "generated_token_ids" in raw and raw["generated_token_ids"] != ids:
        problems.append("generated token IDs disagree with the raw native evidence")
    expected = reading.get("prompt_tokens")
    if not _integer(expected, minimum=1) or not re.fullmatch(r"[0-9a-f]{64}", str(reading.get("prompt_ids_sha256"))):
        problems.append("tokenized prompt identity missing")
    for key in ("tokens_evaluated", "prompt_n"):
        if not _integer(raw.get(key), minimum=1) or raw[key] != expected:
            problems.append(f"{key} differs from the tokenized prompt")
    if type(raw.get("cache_n")) is not int or raw["cache_n"] != 0:
        problems.append("fresh zero-reuse prompt work not demonstrated")
    for key in ("tokens_predicted", "predicted_n"):
        if type(raw.get(key)) is not int or not isinstance(ids, list) or raw[key] != len(ids):
            problems.append(f"{key} does not match the complete generated IDs")
    if (raw.get("stop_seen") is not True or raw.get("stop_type") not in ("limit", "eos")
            or (raw.get("stop_type") == "limit" and (not isinstance(ids, list) or len(ids) != TOKENS))):
        problems.append("natural EOS or full-budget terminal engine completion not demonstrated")
    if raw.get("truncated") is not False:
        problems.append("no-truncation evidence missing")
    if raw.get("protocol_problem"):
        problems.append(f"engine protocol problem: {raw['protocol_problem']}")
    if not _integer(raw.get("id_slot")):
        problems.append("native slot identity missing")
    settings = raw.get("generation_settings")
    if not isinstance(settings, dict):
        problems.append("effective generation settings missing")
    else:
        # b11139 starts with {none} and APPENDS the explicit --spec-type. The
        # native report can therefore contain inert none before the chosen type.
        allowed_types = ({"none", "none,none"} if arm == CONTROL.variant_id
                         else {"ngram-simple", "none,ngram-simple"})
        observed_type = settings.get("speculative.types")
        if not isinstance(observed_type, str) or observed_type not in allowed_types:
            problems.append("effective speculative type differs from the arm")
        if (type(settings.get("seed")) is not int or settings["seed"] != SEED
                or type(settings.get("n_predict")) is not int or settings["n_predict"] != TOKENS
                or type(settings.get("temperature")) not in (int, float) or settings["temperature"] != 0
                or settings.get("ignore_eos") is not False):
            problems.append("effective sampling/length settings differ")
    if not _positive(reading.get("total_s")) or not _positive(reading.get("ttft_s")):
        problems.append("complete-request wall time or TTFT invalid")
    elif reading["ttft_s"] > reading["total_s"]:
        problems.append("TTFT exceeds the whole request")
    if reading.get("peak_rss_bytes") is not None and not _integer(reading["peak_rss_bytes"], minimum=1):
        problems.append("reported peak RSS invalid")
    drafted, accepted = raw.get("draft_n"), raw.get("draft_n_accepted")
    if drafted is not None or accepted is not None:
        if not _integer(drafted) or not _integer(accepted) or accepted > drafted:
            problems.append("draft counters incomplete or contradictory")
        elif arm == CONTROL.variant_id and (drafted or accepted):
            problems.append("the none control reports draft activity")
    # b11139 legitimately omits BOTH draft counters when none were proposed. Do not
    # invent a positive activity/acceptance rate from the requested algorithm.
    return problems


def trial_problems(row: dict[str, Any], case: LookupCase, config_sha: str,
                   *, simulated: bool = False) -> list[str]:
    problems = reading_problems(row.get("reading"), str(row.get("arm")))
    if row.get("case_sha256") != case.sha256 or row.get("configuration_sha256") != config_sha:
        problems.append("trial prompt/configuration identity differs")
    if row.get("complete") is not True or row.get("errors"):
        problems.append("trial did not complete normally")
    if row.get("observation_problems"):
        problems.append("per-load observations incomplete")
    expected_power = ["simulated", "simulated"] if simulated else ["ac", "ac"]
    if row.get("power") != expected_power:
        problems.append("both load-boundary power checks not demonstrated")
    if row.get("cleanup") != {"process_gone": True, "credential_removed": True, "problem": None}:
        problems.append("process and credential release not demonstrated")
    return problems


def summarize(rows: list[dict[str, Any]], config_sha: str, *, simulated: bool = False) -> dict[str, Any]:
    """Recompute from the complete raw population; duplicates are never selected away."""
    expected = plan()
    actual = [(r.get("case_id"), r.get("block"), r.get("arm")) for r in rows]
    population_ok = actual == expected and all(type(r.get("block")) is int for r in rows)
    all_problems = [] if population_ok else ["trial population/order is not the full frozen plan"]
    results: dict[str, Any] = {}
    for case in build_cases():
        block_rows = []
        gains, first_word_gains = [], []
        drafts, accepts = 0, 0
        reference_outputs: list[tuple[str, list[int]]] = []
        for block in range(BLOCKS):
            selected = [r for r in rows if r.get("case_id") == case.case_id and r.get("block") == block]
            counts = Counter(str(r.get("arm")) for r in selected)
            problems = []
            if counts != Counter(ARMS):
                problems.append(f"expected exactly one control and lookup; arms={dict(counts)}")
            for row in selected:
                problems.extend(trial_problems(row, case, config_sha, simulated=simulated))
            if not problems:
                base, candidate = (next(r["reading"] for r in selected if r["arm"] == arm) for arm in ARMS)
                if (base["prompt_ids_sha256"] != candidate["prompt_ids_sha256"]
                        or base["prompt_tokens"] != candidate["prompt_tokens"]):
                    problems.append("tokenized prompts differ between arms")
                if base["text"] != candidate["text"] or base["token_ids"] != candidate["token_ids"]:
                    problems.append("output IDs or raw text differ between arms")
                if base["counters"]["stop_type"] != candidate["counters"]["stop_type"]:
                    problems.append("engine ending differs between arms")
                if not all(math.isfinite((base[key] - candidate[key]) / base[key]) for key in ("total_s", "ttft_s")):
                    problems.append("paired relative timing is nonfinite")
            detail: dict[str, Any] = {"block": block, "usable": not problems, "problems": list(dict.fromkeys(problems))}
            if not problems:
                gain = (base["total_s"] - candidate["total_s"]) / base["total_s"]
                first = (base["ttft_s"] - candidate["ttft_s"]) / base["ttft_s"]
                gains.append(gain)
                first_word_gains.append(first)
                raw = candidate["counters"]
                drafts += raw.get("draft_n") or 0
                accepts += raw.get("draft_n_accepted") or 0
                reference_outputs.append((base["text"], base["token_ids"]))
                detail.update(latency_reduction=gain, ttft_reduction=first,
                              draft_n=raw.get("draft_n"), draft_n_accepted=raw.get("draft_n_accepted"))
            block_rows.append(detail)
        repeatable = bool(reference_outputs) and all(r == reference_outputs[0] for r in reference_outputs)
        complete = population_ok and len(gains) == BLOCKS and repeatable
        bounds, ttft_bounds = None, None
        if complete:
            try:
                bounds = asdict(block_bounds(gains, alpha=ALPHA))
                ttft_bounds = asdict(block_bounds(first_word_gains, alpha=ALPHA))
                if any(type(v) is float and not math.isfinite(v) for b in (bounds, ttft_bounds) for v in b.values()):
                    raise ValueError("nonfinite timing statistics")
            except (ArithmeticError, ValueError):
                complete = False
                bounds, ttft_bounds = None, None
        if not complete:
            outcome = "incomplete, mismatched or nonrepeatable evidence; no performance conclusion"
        elif not accepts:
            outcome = "no accepted draft work demonstrated; no attributed lookup gain"
        elif bounds is not None and bounds["lower"] > 0:
            outcome = ("lower request latency with a first-word trade-off"
                       if ttft_bounds is not None and ttft_bounds["upper"] < 0
                       else "lower request latency on this workload")
        else:
            outcome = "no demonstrated request-latency improvement"
        results[case.case_id] = {"blocks": block_rows, "complete": complete,
                                 "reference_repeatable": repeatable, "outcome": outcome,
                                 "latency_bounds": bounds, "ttft_bounds": ttft_bounds,
                                 "draft_n": drafts, "draft_n_accepted": accepts,
                                 "acceptance_ratio": accepts / drafts if drafts else None}
    return {"version": VERSION, "population_complete": population_ok, "problems": all_problems,
            "cases": results, "simulated": simulated,
            "scope": "fixed workloads and exact outputs; no general quality, tune admission or performance guarantee"}


def evaluate(engine: LookupEngine, *, config_sha: str, power: Callable[[], str],
             checkpoint: Callable[[dict[str, Any]], None], simulated: bool = False) -> dict[str, Any]:
    """Execute the finite frozen plan. Every attempted row is retained, including errors."""
    result: dict[str, Any] = {"version": VERSION, "status": "running", "stop_reason": "",
                              "planned_trials": len(plan()), "trials": [], "simulated": simulated}
    cases = {c.case_id: c for c in build_cases()}

    def publish() -> None:
        result["summary"] = summarize(result["trials"], config_sha, simulated=simulated)
        checkpoint(result)

    publish()
    for case_id, block, arm in plan():
        case = cases[case_id]
        row: dict[str, Any] = {"case_id": case_id, "case_sha256": case.sha256,
                               "configuration_sha256": config_sha, "block": block, "arm": arm,
                               "power": [], "reading": None, "cleanup": None, "errors": [],
                               "observation_problems": [],
                               "stage": "power-before", "complete": False}
        result["trials"].append(row)
        owed = False
        pending: BaseException | None = None

        def remember(exc: BaseException) -> None:
            nonlocal pending
            row["errors"].append({"stage": row["stage"], "type": type(exc).__name__, "message": str(exc)})
            if pending is None or isinstance(exc, KeyboardInterrupt):
                if pending is not None:
                    exc.__context__ = pending
                pending = exc

        try:
            row["power"].append(power())
            if row["power"][-1] != ("simulated" if simulated else "ac"):
                result.update(status="stopped-power", stop_reason="mains power not demonstrated before load")
            else:
                row["stage"] = "start"
                owed = True
                engine.start(CONTROL if arm == CONTROL.variant_id else LOOKUP)
                row["stage"] = "request"
                row["reading"] = engine.request(case)
        except BaseException as exc:
            remember(exc)
        finally:
            if owed:
                row["stage"] = "cleanup"
                try:
                    row["cleanup"] = asdict(engine.stop())
                except BaseException as exc:
                    remember(exc)
                row["stage"] = "power-after"
                try:
                    row["power"].append(power())
                except BaseException as exc:
                    remember(exc)
        if owed and pending is None:
            row["stage"] = "observations"
            try:
                issues = engine.load_problems()
                if not isinstance(issues, list) or any(not isinstance(p, str) or not p for p in issues):
                    raise TypeError("load observation problems must be a list of nonempty strings")
                row["observation_problems"] = issues
            except BaseException as exc:
                remember(exc)
        if pending is not None:
            result.update(status="interrupted" if isinstance(pending, KeyboardInterrupt) else "error",
                          stop_reason=f"{type(pending).__name__}: {pending}")
            try:
                publish()
            except Exception as exc:
                row["checkpoint_problem"] = f"{type(exc).__name__}: {exc}"
            raise pending
        if owed:
            if row["cleanup"] != {"process_gone": True, "credential_removed": True, "problem": None}:
                result.update(status="stopped-cleanup", stop_reason="process/credential cleanup incomplete")
            elif row["power"] != (["simulated", "simulated"] if simulated else ["ac", "ac"]):
                result.update(status="stopped-power", stop_reason="mains power not demonstrated after load")
            elif row["observation_problems"]:
                result.update(status="stopped-observation", stop_reason="; ".join(row["observation_problems"]))
            else:
                issues = reading_problems(row["reading"], arm)
                if issues:
                    result.update(status="stopped-validity", stop_reason="; ".join(issues))
                else:
                    row.update(stage="complete", complete=True)
        # Check paired preservation as soon as the second arm returns. Do not run
        # later blocks after a demonstrated output discrepancy.
        matched = [r for r in result["trials"] if r["case_id"] == case_id and r["block"] == block]
        if result["status"] == "running" and len(matched) == 2 and all(r["complete"] for r in matched):
            first, second = (r["reading"] for r in matched)
            if (first["text"] != second["text"] or first["token_ids"] != second["token_ids"]
                    or first["prompt_ids_sha256"] != second["prompt_ids_sha256"]
                    or first["counters"]["stop_type"] != second["counters"]["stop_type"]):
                result.update(status="stopped-quality", stop_reason="paired output or prompt identity differs")
            elif not all(math.isfinite(first[key] / second[key]) and math.isfinite(second[key] / first[key])
                         for key in ("total_s", "ttft_s")):
                result.update(status="stopped-validity", stop_reason="paired relative timing is nonfinite")
        publish()
        if result["status"] != "running":
            return result
    final_summary = summarize(result["trials"], config_sha, simulated=simulated)
    if any(not c["complete"] for c in final_summary["cases"].values()):
        result.update(status="incomplete", stop_reason="nonrepeatable or unusable complete-request evidence")
    else:
        result.update(status="done", stop_reason="")
    publish()
    return result
