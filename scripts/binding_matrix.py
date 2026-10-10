"""What every ``OptionPath`` produces over a fixed matrix of path contexts: candidates,
notes, defaults and both refinement mechanisms (no model, no laptop; C3, 10 Oct 2026).

``record ROOT OUT.json`` loads ``tune/paths.py`` from the code under ``ROOT`` and writes,
for every context and path: the candidates (value, label, settings changed with it,
screen-only reason, in order), the notes, the current value (with the default
resolver), the scalar refinement around the first two candidate values, and the
compound refinement around the first two candidates. An exception is recorded as its
type and message, so it is compared too. It also records the path's binding facts
(card, setting, label, questions, option) and the specialised verifier map.
``compare A B`` reports every difference.

The matrix crosses: the model facts (none; dense; dense with a tied output; experts;
hybrid; no piece sizes), the current setup (layer count unset, 0, 2, all, all + 1;
memory format 16-bit or q8_0 with flash on; threads, batch, switches and placement
set or not), the long question (unknown, short, long), the engine's checked switches
(not checked, none passed, two passed), the room on the graphics card (unknown, small,
large), the machine (graphics card or not), the engine's own value check (unknown,
accepts everything, refuses 4-bit), and the grouped-placement probe flag.

    python scripts/binding_matrix.py record . OUT.json
    python scripts/binding_matrix.py compare BEFORE.json AFTER.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

MIB = 1024 * 1024


def _facts() -> dict[str, Any]:
    from inferenceos.tune.model_facts import ModelFacts

    def dense(blocks: int = 4, tied: bool = False) -> ModelFacts:
        tensors: dict[str, int] = {"token_embd.weight": 50 * MIB}
        for i in range(blocks):
            tensors[f"blk.{i}.attn_q.weight"] = 40 * MIB
            tensors[f"blk.{i}.ffn_up.weight"] = 60 * MIB
        if not tied:
            tensors["output_norm.weight"] = 0
            tensors["output.weight"] = 200 * MIB
        return ModelFacts(
            total_layers=blocks, from_file=True, attention_layers=blocks,
            layer_bytes=tuple(100 * MIB for _ in range(blocks)), active_layer_bytes=tuple(100 * MIB for _ in range(blocks)),
            output_bytes=(50 if tied else 200) * MIB, other_bytes=50 * MIB, tensor_bytes=tensors,
            layer_kinds=tuple("attention" for _ in range(blocks)), tied_output=tied,
            expert_bytes=tuple(0 for _ in range(blocks)), key_widths=tuple(1024 for _ in range(blocks)),
            value_widths=tuple(1024 for _ in range(blocks)))

    moe_tensors: dict[str, int] = {"token_embd.weight": 50 * MIB, "output.weight": 100 * MIB}
    for i in range(4):
        moe_tensors[f"blk.{i}.attn_q.weight"] = 20 * MIB
        for part in ("up", "down", "gate"):
            moe_tensors[f"blk.{i}.ffn_{part}_exps.weight"] = 100 * MIB
    moe = ModelFacts(
        total_layers=4, from_file=True, attention_layers=4, experts=32, experts_used=8,
        layer_bytes=tuple(320 * MIB for _ in range(4)), active_layer_bytes=tuple(95 * MIB for _ in range(4)),
        output_bytes=100 * MIB, other_bytes=50 * MIB, tensor_bytes=moe_tensors,
        layer_kinds=tuple("attention" for _ in range(4)), tied_output=False,
        expert_bytes=tuple(300 * MIB for _ in range(4)), key_widths=tuple(512 for _ in range(4)),
        value_widths=tuple(512 for _ in range(4)))
    kinds = ("recurrent (state-space)", "recurrent (state-space)", "attention", "recurrent (state-space)")
    hybrid_tensors: dict[str, int] = {"token_embd.weight": 50 * MIB, "output.weight": 100 * MIB}
    for i, kind in enumerate(kinds):
        hybrid_tensors[f"blk.{i}.{'attn_q' if kind == 'attention' else 'ssm_in'}.weight"] = 100 * MIB
    hybrid = ModelFacts(
        total_layers=4, from_file=True, attention_layers=1, other_layers={"recurrent (state-space)": 3},
        layer_bytes=tuple(100 * MIB for _ in kinds), active_layer_bytes=tuple(100 * MIB for _ in kinds),
        output_bytes=100 * MIB, other_bytes=50 * MIB, tensor_bytes=hybrid_tensors, layer_kinds=kinds,
        tied_output=False, expert_bytes=(0, 0, 0, 0), key_widths=(0, 0, 512, 0), value_widths=(0, 0, 512, 0))
    return {"no facts": None, "dense": dense(), "dense, tied output": dense(tied=True), "experts": moe,
            "hybrid": hybrid, "no piece sizes": ModelFacts(total_layers=4, from_file=True, attention_layers=4)}


def contexts(paths: Any) -> list[tuple[str, Any]]:
    from inferenceos.contracts.capability import Support
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.selftest import sample_hardware, sample_model
    from inferenceos.tune.model_facts import MachineFacts

    def accept_all(setting: str, value: object) -> tuple[Any, str]:
        return Support.YES, "accepted"

    def refuse_4bit(setting: str, value: object) -> tuple[Any, str]:
        return (Support.NO, f"{value} refused") if str(value).startswith("q4") else (Support.YES, "accepted")

    base = RuntimeConfiguration(context_length=4096)
    currents = {
        "layers unset": base,
        "0 layers": base.model_copy(update={"gpu_offload_layers": 0}),
        "2 layers": base.model_copy(update={"gpu_offload_layers": 2}),
        "all layers": base.model_copy(update={"gpu_offload_layers": 4}),
        "all + 1": base.model_copy(update={"gpu_offload_layers": 5}),
        "2 layers, q8_0, flash on": base.model_copy(update={"gpu_offload_layers": 2, "kv_cache_type_k": "q8_0",
                                                            "kv_cache_type_v": "q8_0", "flash_attention": True}),
        "2 layers, 8 threads, batch 1024, switch, placement": base.model_copy(update={
            "gpu_offload_layers": 2, "threads": 8, "threads_batch": 4, "ubatch_size": 1024,
            "engine_switches": "GGML_VK_DISABLE_FUSION=1", "placement_overrides": "^output=CPU"}),
    }
    switches = {"switches not checked": None, "no switch passed": (),
                "two switches passed": (("GGML_VK_DISABLE_FUSION=1", "fusion off"), ("", "back to default"))}
    budgets = {"card room unknown": None, "small card room": 150 * MIB, "large card room": 3000 * MIB}
    machines = {"card": MachineFacts(6, 12, 2400, 2, True, 3000 * MIB),
                "no card": MachineFacts(6, 12, 2400, 2, False, None)}
    accepts = {"engine check unknown": None, "accepts all": accept_all, "refuses 4-bit": refuse_4bit}
    hardware, model = sample_hardware(), sample_model()
    out: list[tuple[str, Any]] = []
    grid = itertools.product(_facts().items(), currents.items(), (None, 100, 2000), switches.items(),
                             budgets.items(), machines.items(), accepts.items(), (False, True))
    for (f_label, facts), (c_label, current), tokens, (s_label, checked), (b_label, budget), \
            (m_label, machine), (a_label, accept), grouped in grid:
        label = (f"{f_label} | {c_label} | question {tokens} | {s_label} | {b_label} | {m_label} | {a_label} | "
                 f"grouped probed {grouped}")
        share = None if facts is None else 0.5
        out.append((label, paths.PathContext(
            hardware=hardware, model=model, total_layers=4, current=current, facts=facts, machine=machine,
            processor_share=share, question_tokens=tokens, context_length=4096, engine="vulkan",
            engine_version="version: 11139 (42916d83f)", card_budget_bytes=budget, accepts=accept,
            checked_switches=checked, grouped_placement_probed=grouped)))
    return out


def _call(fn: Any, *args: Any) -> Any:
    try:
        return {"ok": _plain(fn(*args))}
    except Exception as exc:  # noqa: BLE001 - an exception is an observable outcome too
        return {"raised": f"{type(exc).__name__}: {exc}"}


def _plain(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "value") and hasattr(value, "label") and hasattr(value, "also"):
        return {"value": value.value, "label": value.label, "also": _plain(value.also),
                "screen_only": value.screen_only}
    if isinstance(value, set):
        return sorted(repr(v) for v in value)
    return value


def record_paths(paths: Any) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for label, context in contexts(paths):
        row: dict[str, Any] = {}
        for path in paths.ALL_PATHS:
            found = _call(path.candidates, context)
            entry: dict[str, Any] = {"candidates": found, "current_value": _call(path.current_value, context)}
            entry["notes"] = None if path.notes is None else _call(path.notes, context)
            entry["default"] = None if path.default_value is None else _call(path.default_value, context)
            cands = path.candidates(context) if "ok" in found else []
            if path.refine is not None:
                tried = {c.value for c in cands}
                entry["refine"] = [_call(path.refine, context, c.value, tried) for c in cands[:2]]
            if path.refine_candidates is not None:
                entry["refine_candidates"] = [_call(path.refine_candidates, context, c, {c.key}) for c in cands[:2]]
            row[path.name] = entry
        results[label] = row
    return results


def binding_facts(paths: Any) -> dict[str, Any]:
    facts = {}
    for path in paths.ALL_PATHS:
        facts[path.name] = {
            "card_id": path.card_id, "setting": path.setting, "label": path.label, "workload_id": path.workload_id,
            "guard_workload_id": path.guard_workload_id, "option": path.option, "description": path.description,
            "candidates": path.candidates.__qualname__,
            "refine": None if path.refine is None else path.refine.__qualname__,
            "refine_candidates": None if path.refine_candidates is None else path.refine_candidates.__qualname__,
            "notes": None if path.notes is None else path.notes.__qualname__,
            "default_value": None if path.default_value is None else path.default_value.__qualname__,
            "verify": None if path.verify is None else path.verify.__qualname__,
        }
    return {"order": [p.name for p in paths.ALL_PATHS], "paths": facts,
            "verifiers": [[setting, fn.__qualname__] for setting, fn in paths.VERIFIERS.items()],
            "paths_by_name": {",".join(names): [p.name for p in paths.paths_by_name(list(names))] for names in (
                ("threads",), ("threads", "threads"), (" threads ", "unknown", "kv_cache"), ("layers",), (),
                ("kv_cache", "threads", "prompt_batch", "flash_attention", "reading_threads", "engine_switches",
                 "placement"))}}


def record(root: Path, out: Path) -> int:
    root = root.resolve()
    sys.path[:0] = [str(root / "src"), str(root)]
    import inferenceos.tune.paths as paths
    if not Path(paths.__file__).resolve().is_relative_to(root):
        raise SystemExit(f"imported {paths.__file__}, not the code under {root}")
    data = {"bindings": binding_facts(paths), "contexts": record_paths(paths)}
    out.write_text(json.dumps(data, sort_keys=True, default=repr), encoding="utf-8")
    print(f"{len(data['contexts'])} contexts x {len(paths.ALL_PATHS)} paths recorded from {paths.__file__}")
    return 0


def compare(a_path: Path, b_path: Path) -> int:
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    differ = 0
    if a["bindings"] != b["bindings"]:
        differ += 1
        for key in sorted(set(a["bindings"]) | set(b["bindings"])):
            if a["bindings"].get(key) != b["bindings"].get(key):
                print(f"  BINDINGS DIFFER: {key}")
    if set(a["contexts"]) != set(b["contexts"]):
        print("different context sets")
        return 1
    entries = 0
    for label in sorted(a["contexts"]):
        for name in sorted(set(a["contexts"][label]) | set(b["contexts"][label])):
            entries += 1
            x, y = a["contexts"][label].get(name), b["contexts"][label].get(name)
            if x != y:
                differ += 1
                if differ <= 10:
                    print(f"  DIFFERS {name} @ {label}")
    raised = sum("raised" in json.dumps(v) for row in a["contexts"].values() for v in row.values())
    print(f"{len(a['contexts'])} contexts, {entries} path entries ({raised} contain a recorded exception); "
          f"bindings {'equal' if a['bindings'] == b['bindings'] else 'DIFFER'}; {differ} differences")
    return 0 if differ == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("root", type=Path)
    rec.add_argument("out", type=Path)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("before", type=Path)
    cmp_.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    if args.command == "record":
        return record(args.root, args.out)
    return compare(args.before, args.after)


if __name__ == "__main__":
    raise SystemExit(main())
