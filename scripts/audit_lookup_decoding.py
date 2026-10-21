"""Non-destructive offline audit of a frozen lookup study. Never starts an engine.

0 = complete registered real observations; 2 = explicit gaps; 4 = refused input/output.
No return code authorizes a tune path or production change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inferenceos.config import Settings
from inferenceos.tune.lookup_audit import audit
from evaluate_lookup_decoding import read_rules
from evaluate_prompt_reuse import output_problem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--rules-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rules, rules_sha = read_rules(args.rules, args.rules_sha256)
        rules_raw = args.rules.read_bytes()
        if hashlib.sha256(rules_raw).hexdigest() != rules_sha:
            raise ValueError("rules changed before offline audit")
        raw = args.source.read_bytes()
        record = json.loads(raw)
        protected = [args.source, args.rules, Path(Settings().data_dir), ROOT / "src", ROOT / "scripts", ROOT / "tests", ROOT / "tools"]
        if record.get("model"):
            protected.append(Path(record["model"]))
        if record.get("data_folder_evidence", {}).get("root"):
            protected.append(Path(record["data_folder_evidence"]["root"]))
        problem = output_problem(args.out, protected=protected)
        if problem:
            raise ValueError(problem)
        checked = audit(record, rules, rules_sha)
        checked.update(source=str(args.source), source_sha256=hashlib.sha256(raw).hexdigest(), rules_sha256=rules_sha)
        if args.source.read_bytes() != raw or args.rules.read_bytes() != rules_raw:
            raise ValueError("source or rules changed during offline audit")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(checked, handle, indent=2, allow_nan=False)
        print(json.dumps({k: checked[k] for k in ("observations_complete", "performance_interpretable", "problems")}))
        return 0 if checked["observations_complete"] else 2
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"refused: {type(exc).__name__}: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
