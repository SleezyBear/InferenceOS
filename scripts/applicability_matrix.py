"""Applicability of every ``OptionPath`` over a fixed matrix of path contexts (no model,
no laptop; C3, 9 Oct 2026).

``record ROOT OUT.json`` loads ``tune/paths.py`` from the code under ``ROOT`` and writes,
for every context in the matrix and every path, whether ``path.check`` says the path
applies and its reason. ``compare A B`` reports every context whose outcome or reason
differs. Used to show that moving applicability onto the declarations changed nothing.

The matrix crosses the facts the rules read: the engine's option list (unknown, complete,
each path's option missing, empty), the processor's share of the model (unknown, none,
part, all), the machine (from the hardware profile; one core; no graphics card), the model
facts (none; kinds unknown; no attention; all attention; mixed kinds; piece sizes known or
not), the current layer count (unset, 0, 8) and memory format, the long question's length,
and the engine build (unknown, listed, unlisted, another engine) with or without checked
switches.

    python scripts/applicability_matrix.py record . OUT.json
    python scripts/applicability_matrix.py compare BEFORE.json AFTER.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

OPTIONS = ("--threads", "--threads-batch", "--flash-attn", "--ubatch-size", "--override-tensor", "--cache-type-k")


def contexts(paths: Any) -> list[tuple[str, Any]]:
    """(label, PathContext) for the whole matrix, built only from public constructors."""
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.selftest import sample_hardware, sample_model
    from inferenceos.tune.model_facts import MachineFacts, ModelFacts

    hardware, model = sample_hardware(), sample_model()
    option_sets: dict[str, frozenset[str] | None] = {"options unknown": None, "all options": frozenset(OPTIONS),
                                                     "no options": frozenset()}
    for option in OPTIONS:
        option_sets[f"missing {option}"] = frozenset(OPTIONS) - {option}
    machines: dict[str, MachineFacts | None] = {
        "machine from hardware": None,
        "6 cores, card": MachineFacts(6, 12, 2400, 2, True, 3 * 1024**3),
        "1 core, card": MachineFacts(1, 1, 2400, 1, True, 3 * 1024**3),
        "6 cores, no card": MachineFacts(6, 12, 2400, 2, False, None),
    }
    pieces = {"blk.0.attn_q.weight": 1000, "blk.0.ffn_up.weight": 4000, "output.weight": 2000}
    facts: dict[str, ModelFacts | None] = {
        "no facts": None,
        "kinds unknown": ModelFacts(total_layers=24),
        "no attention": ModelFacts(total_layers=24, from_file=True, attention_layers=0, other_layers={"recurrent": 24},
                                   tensor_bytes=pieces),
        "all attention": ModelFacts(total_layers=24, from_file=True, attention_layers=24, tensor_bytes=pieces),
        "mixed, no pieces": ModelFacts(total_layers=24, from_file=True, attention_layers=4,
                                       other_layers={"recurrent": 20}),
        "mixed, experts": ModelFacts(total_layers=24, from_file=True, attention_layers=4, other_layers={"recurrent": 20},
                                     experts=32, experts_used=8, tensor_bytes=pieces),
    }
    currents = {
        "layers unset": RuntimeConfiguration(context_length=4096),
        "0 layers": RuntimeConfiguration(context_length=4096, gpu_offload_layers=0),
        "8 layers": RuntimeConfiguration(context_length=4096, gpu_offload_layers=8),
        "8 layers, q8_0 memory": RuntimeConfiguration(context_length=4096, gpu_offload_layers=8, kv_cache_type_k="q8_0",
                                                      kv_cache_type_v="q8_0", flash_attention=True),
    }
    engines: dict[str, tuple[str | None, str | None, tuple[tuple[str, str], ...] | None]] = {
        "engine unknown": (None, None, None),
        "listed build": ("vulkan", "version: 11139 (42916d83f)", None),
        "listed build, checked switches": ("vulkan", "version: 11139 (42916d83f)", (("GGML_VK_DISABLE_FUSION=1", "fusion off"),)),
        "unlisted build": ("vulkan", "version: 1 (0000000)", None),
        "other engine": ("cuda", "version: 11139 (42916d83f)", None),
    }
    out: list[tuple[str, Any]] = []
    grid = itertools.product(option_sets.items(), (None, 0.0, 0.2, 1.0), machines.items(), facts.items(),
                             currents.items(), (None, 2000), engines.items())
    for (o_label, options), share, (m_label, machine), (f_label, fact), (c_label, current), tokens, (e_label, engine) in grid:
        label = f"{o_label} | share {share} | {m_label} | {f_label} | {c_label} | question {tokens} | {e_label}"
        out.append((label, paths.PathContext(
            hardware=hardware, model=model, total_layers=24, current=current, supported_options=options,
            facts=fact, machine=machine, processor_share=share, question_tokens=tokens, context_length=4096,
            engine=engine[0], engine_version=engine[1], checked_switches=engine[2])))
    return out


def record(root: Path, out: Path) -> int:
    root = root.resolve()
    sys.path[:0] = [str(root / "src"), str(root)]
    import inferenceos.tune.paths as paths
    if not Path(paths.__file__).resolve().is_relative_to(root):
        raise SystemExit(f"imported {paths.__file__}, not the code under {root}")
    results: dict[str, dict[str, list[object]]] = {}
    for label, context in contexts(paths):
        results[label] = {path.name: list(path.check(context)) for path in paths.ALL_PATHS}
    out.write_text(json.dumps(results, sort_keys=True), encoding="utf-8")
    print(f"{len(results)} contexts x {len(paths.ALL_PATHS)} paths recorded from {paths.__file__}")
    return 0


def compare(a_path: Path, b_path: Path) -> int:
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    if set(a) != set(b):
        print("different context sets")
        return 1
    checks = differ = 0
    applies = {True: 0, False: 0}
    for label in sorted(a):
        if set(a[label]) != set(b[label]):
            print(f"different paths at {label}")
            return 1
        for name in sorted(a[label]):
            checks += 1
            applies[bool(a[label][name][0])] += 1
            if a[label][name] != b[label][name]:
                differ += 1
                if differ <= 10:
                    print(f"  DIFFERS {name} @ {label}: {a[label][name]} -> {b[label][name]}")
    print(f"{len(a)} contexts, {checks} path checks ({applies[True]} apply, {applies[False]} do not); "
          f"{differ} differ in outcome or reason")
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
