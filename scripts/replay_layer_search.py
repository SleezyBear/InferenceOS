"""Replay the layer search against a recorded study (thin command line).

    .\\.venv\\Scripts\\python.exe scripts\\replay_layer_search.py --study reports\\layer_study_20261001T110201Z

Writes ``replay.json`` and ``replay.txt`` into the study folder (or next to ``--out``)
and prints the PASS/FAIL lines.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from inferenceos.tune.layer_replay import render_report, run_replay, write_outputs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, help="study folder with per-model JSON")
    parser.add_argument("--repeats", type=int, default=300, help="repeats per curve (default 300)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    parser.add_argument(
        "--out",
        default=None,
        help="output basename or folder (default: study folder)",
    )
    parser.add_argument("--quick", action="store_true", help="real curves only, 50 repeats")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    study = Path(args.study)
    if not study.is_dir():
        print(f"study folder not found: {study}", file=sys.stderr)
        return 2

    result = run_replay(study, repeats=args.repeats, seed=args.seed, quick=args.quick)

    if args.out is None:
        json_path = study / "replay.json"
        txt_path = study / "replay.txt"
    else:
        base = Path(args.out)
        if base.is_dir() or str(args.out).endswith(("/", "\\")):
            base = base / "replay"
        json_path = base.with_suffix(".json")
        txt_path = base.with_suffix(".txt")

    write_outputs(result, json_path, txt_path)
    print(render_report(result))
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")
    return 0 if all(ok for _name, ok, _detail in result.passes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
