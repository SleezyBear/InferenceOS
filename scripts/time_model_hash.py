"""Wall-clock time of the saved-setup model-file hash (reviewer question 3).

Runs ``file_sha256`` (the exact function ``Tuner._save`` uses) on a model file a few
times and prints JSON: size, seconds per run, MB/s. The first run may read from disk,
later runs may come from the file cache; both are reported.

    .\\.venv\\Scripts\\python.exe scripts\\time_model_hash.py --model models\\study\\Llama-3.2-3B-Instruct-Q4_K_M.gguf --repeats 3 --out _jobs\\out\\120_hash_time.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    from inferenceos.tune.profiles import file_sha256

    path = Path(args.model).resolve()
    size = path.stat().st_size
    runs = []
    digest = None
    for index in range(max(args.repeats, 1)):
        began = time.perf_counter()
        digest = file_sha256(path)
        took = time.perf_counter() - began
        runs.append({"run": index + 1, "seconds": round(took, 3), "mb_per_s": round(size / 1e6 / max(took, 1e-9), 1)})
        print(json.dumps(runs[-1]), flush=True)
    result = {"model": str(path), "size_bytes": size, "sha256": digest, "runs": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
