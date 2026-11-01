"""D1 SOFTWARE-ONLY fixture evaluation. No real-engine option exists.

Exit 0=all scripted pairs correct, 2=fixture failure, 4=refused, 7=write failed.
This cannot grant hardware qualification, tuning admission or deployment permission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inferenceos.tune.lookup_qualification_sim import simulate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out: Path = args.out
    if out.exists():
        print("refused: output exists", file=sys.stderr)
        return 4
    report = simulate()
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        # Exclusive create also refuses a file created after the early check.
        # Serialize first; no partial report is ever treated as complete evidence.
        with out.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    except FileExistsError:
        print("refused: output was created concurrently", file=sys.stderr)
        return 4
    except OSError as exc:
        print(f"report write failed: {exc}", file=sys.stderr)
        return 7
    print(f"Wrote SOFTWARE-ONLY fixtures to {out}; no hardware or tuning qualification.")
    return 0 if report["all_fixture_pairs_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
