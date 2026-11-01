"""Scripted complete-task oracle exercises. NO native engine, timing or qualification."""
from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Sequence

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.tune.lookup_audit import REQUIRED_ENGINE_FILES
from inferenceos.tune.lookup_corpus import FrozenInput, TaskCorpus, build_corpus, digest, judge
from inferenceos.tune.lookup_qualification import (
    COST_COMPONENTS, ConfigurationSnapshot, CostSamples, FilePin, MachineBinding,
    QualificationDraft, RequestBinding, ScopeBinding, ToolchainBinding, match_draft,
)

SIM_VERSION = "lookup-qualification-fixtures-1"


class FixtureRow(FrozenInput):
    case_id: str
    case_sha256: str
    arm: Literal["control", "lookup"]
    text: str
    ending: Literal["eos", "limit", "error"]
    truncated: bool


def fixture_draft(corpus: TaskCorpus) -> QualificationDraft:
    """Deliberately fabricated identity/cost data; not read from the owner's machine."""
    pins = tuple(FilePin(path="C:/D1-FIXTURE/" + name, sha256=digest("fixture bytes: " + name))
                 for name in sorted(REQUIRED_ENGINE_FILES))
    requested = RuntimeConfiguration(context_length=4096).model_dump()
    effective = RuntimeConfiguration(context_length=4096, threads=4, threads_batch=4,
                                    batch_size=1024, ubatch_size=256, gpu_offload_layers=8,
                                    use_mmap=True, flash_attention=False, device="fixture-gpu",
                                    kv_cache_type_k="f16", kv_cache_type_v="f16").model_dump()
    binding = ScopeBinding(
        model_sha256=digest("fixture model, not a GGUF"), tokenizer_sha256=digest("fixture tokenizer"),
        corpus_sha256=corpus.sha256,
        machine=MachineBinding(hostname="fixture-host", os_build="fixture-windows", architecture="x86_64",
                               cpu_name="fixture-cpu", cpu_vendor="fixture-vendor", physical_cores=4,
                               logical_cores=8, instruction_sets=("fixture-isa",), ram_bytes=16 * 2**30,
                               gpu_name="fixture-gpu", gpu_vendor="fixture-vendor", gpu_vram_bytes=2 * 2**30,
                               driver_version="fixture-driver", pci_device_id=None, usable_gpu_count=1,
                               device_name="fixture-gpu", device_index=0, power_plan="fixture-power-plan"),
        toolchain=ToolchainBinding(executable=next(p for p in pins if p.path.endswith("/llama-server.exe")),
                                   files=pins, ambient_config_absent=True,
                                   environment_sha256=digest("fixture environment")),
        requested=ConfigurationSnapshot(**requested), effective=ConfigurationSnapshot(**effective),
        unknown_effects=(), request=RequestBinding(max_output_tokens=512, context_per_slot=4096, server_slots=4),
    )
    return QualificationDraft(
        simulated=True, binding=binding,
        costs=tuple(CostSamples(component=k, case_sha256=corpus.cases[0].sha256,
                                source_sha256=digest("fabricated cost: " + k), seconds=(1.0, 1.1))
                    for k in COST_COMPONENTS),
    )


def fixture_rows(corpus: TaskCorpus) -> tuple[FixtureRow, ...]:
    result = []
    for index, case in enumerate(corpus.cases):
        arms: tuple[Literal["control", "lookup"], ...] = (("control", "lookup") if index % 2 == 0 else ("lookup", "control"))
        for arm in arms:
            result.append(FixtureRow(case_id=case.case_id, case_sha256=case.sha256, arm=arm,
                                     text=case.expected, ending="eos", truncated=False))
    return tuple(result)


def evaluate_fixtures(corpus: TaskCorpus, rows: Sequence[FixtureRow]) -> dict[str, Any]:
    """Recompute every row and case, including wrong controls and duplicates."""
    corpus = TaskCorpus.model_validate_json(corpus.model_dump_json())
    rows = tuple(FixtureRow.model_validate_json(r.model_dump_json()) for r in rows)
    expected_order = [(r.case_id, r.arm) for r in fixture_rows(corpus)]
    actual_order = [(r.case_id, r.arm) for r in rows]
    population_problems = [] if actual_order == expected_order else ["fixture population/order differs"]
    known = {c.case_id: c for c in corpus.cases}
    findings: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        case = known.get(row.case_id)
        issues = []
        if case is None:
            issues.append("unknown case")
        elif row.case_sha256 != case.sha256:
            issues.append("case content identity differs")
        judgement = judge(case, row.text, ending=row.ending, truncated=row.truncated) if case is not None else None
        if judgement is not None:
            issues.extend(judgement.problems)
        entry = {"raw": row.model_dump(mode="json"), "task_correct": judgement.passed if judgement else False,
                 "usable_fixture": not issues, "problems": issues}
        findings.append(entry)
        by_case.setdefault(row.case_id, []).append(entry)
    cases = []
    for case in corpus.cases:
        selected = by_case.get(case.case_id, [])
        counts = Counter(r["raw"]["arm"] for r in selected)
        problems = []
        outcome = "incomplete"
        if counts != Counter({"control": 1, "lookup": 1}):
            problems.append("exactly one control and lookup fixture required; no duplicate load selected")
        else:
            control = next(r for r in selected if r["raw"]["arm"] == "control")
            candidate = next(r for r in selected if r["raw"]["arm"] == "lookup")
            if not control["usable_fixture"]:
                outcome = "task unqualified: control wrong or unusable"
                problems.append(outcome)
            elif not candidate["usable_fixture"]:
                outcome = "lookup wrong or unusable on a correct control"
                problems.append(outcome)
            elif control["raw"]["text"] != candidate["raw"]["text"]:
                outcome = "both task-correct but raw outputs differ"
                problems.append(outcome)
            else:
                outcome = "correct scripted full-answer pair; not hardware evidence"
        cases.append({"case_id": case.case_id, "case_sha256": case.sha256, "split": case.split,
                      "family": case.task.family, "length": case.length, "opportunity": case.opportunity,
                      "passed_fixture_pair": not problems, "outcome": outcome, "problems": problems})
    complete = not population_problems and all(c["passed_fixture_pair"] for c in cases)
    draft = fixture_draft(corpus)
    return {"version": SIM_VERSION, "simulated": True,
            "status": "complete software fixtures" if complete else "incomplete or failed software fixtures",
            "all_fixture_pairs_pass": complete, "corpus_sha256": corpus.sha256,
            "corpus": corpus.model_dump(mode="json"), "expected": {c.case_id: c.expected for c in corpus.cases},
            "prompts": {c.case_id: c.prompt for c in corpus.cases},
            "population_problems": population_problems, "rows": findings, "cases": cases,
            "draft": draft.model_dump(mode="json"),
            "draft_match": match_draft(draft, draft.binding, corpus).model_dump(mode="json"),
            "hardware_qualification": False, "automatic_admission": False, "deployment_permission": False,
            "native_evidence_collected": False, "token_fit_verified": False,
            "performance_conclusion": None, "D2_protocol_frozen": False}


def simulate() -> dict[str, Any]:
    corpus = build_corpus()
    return evaluate_fixtures(corpus, fixture_rows(corpus))
