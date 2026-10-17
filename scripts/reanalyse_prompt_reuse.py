"""Re-summarise prompt-reuse results files with the current analysis, without touching them.

Reads each given results file (written by ``scripts/evaluate_prompt_reuse.py``), rebuilds
its trials exactly as recorded, and writes the current ``summarize`` output to a NEW file
next to it: ``<name>.<analysis version>.json``. The source file is only read; its SHA-256
is recorded in the output and checked again afterwards. An existing output file is never
overwritten.

The output also says whether anything other than the E7 classification
(``shifted_reuse``) differs from the summary stored in the source, so a corrected
analysis can be told apart from changed evidence, and lists any duplicate loads (the
same variant, case, block and arm recorded more than once). Every recorded load is kept:
a duplicate makes its block not a unique pair; no load is ever dropped or picked.

    python scripts/reanalyse_prompt_reuse.py _jobs/out/127_reuse_llama3b_<stamp>.json [...]

Exit status: 0 every file re-analysed; 4 an output file already exists (nothing written
for that file); 3 a source file could not be read or changed while it was read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from inferenceos.tune.prompt_reuse import E7_ANALYSIS_VERSION, TrialResult, summarize  # noqa: E402


def _without_e7(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: {k: v for k, v in item.items() if k != "shifted_reuse"} for key, item in summary.items()}


def reanalyse(source: Path) -> tuple[Path, dict[str, Any]]:
    """The output path and record for one results file (nothing is written here)."""
    raw = source.read_bytes()
    record = json.loads(raw)
    trials = [TrialResult(**trial) for trial in record.get("trials", [])]
    seen: dict[str, int] = {}
    for t in trials:
        key = f"{t.variant_id}/{t.case_id} block {t.block + 1} {t.arm}"
        seen[key] = seen.get(key, 0) + 1
    duplicates = {key: n for key, n in sorted(seen.items()) if n > 1}
    summary = summarize(trials) if trials else {}
    stored = record.get("summary") or {}
    e7_changes = {
        key: {"stored": (stored.get(key) or {}).get("shifted_reuse", {}).get("e7"),
              "now": item["shifted_reuse"]["e7"]}
        for key, item in summary.items() if "shifted_reuse" in item
    }
    out = source.with_name(f"{source.stem}.{E7_ANALYSIS_VERSION}.json")
    return out, {
        "source": source.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_evaluation": record.get("evaluation"),
        "source_run_id": record.get("run_id"),
        "source_status": record.get("status"),
        "analysis": E7_ANALYSIS_VERSION,
        "analysed_at": datetime.now(timezone.utc).isoformat(),
        "trials": len(trials),
        "duplicate_loads": duplicates,
        "unchanged_outside_e7": _without_e7(summary) == _without_e7(stored),
        "e7_changes": e7_changes,
        "summary": summary,
        "note": "re-analysis of recorded trials only; the source file is unchanged and remains the evidence",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args(argv)
    code = 0
    for source in args.results:
        try:
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            out, record = reanalyse(source)
        except (OSError, ValueError, TypeError) as exc:
            print(f"{source}: could not be re-analysed: {type(exc).__name__}: {exc}")
            code = max(code, 3)
            continue
        if out.exists() or out.is_symlink():
            print(f"{source}: refused, {out} already exists (never overwritten)")
            code = max(code, 4)
            continue
        if hashlib.sha256(source.read_bytes()).hexdigest() != before or record["source_sha256"] != before:
            print(f"{source}: changed while it was read; nothing written")
            code = max(code, 3)
            continue
        with out.open("x", encoding="utf-8") as stream:  # exclusive: never replaces a file
            stream.write(json.dumps(record, indent=2) + "\n")
        print(f"{source.name}: {out.name}; unchanged outside E7: {record['unchanged_outside_e7']}")
        for key, count in record["duplicate_loads"].items():
            print(f"  DUPLICATE: {key} recorded {count} times (its block is not a unique pair)")
        for key, change in record["e7_changes"].items():
            print(f"  {key}: {change['stored']!r} -> {change['now']!r}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
