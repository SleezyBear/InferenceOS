"""Print what tune's paths learn from each model file (design, sizes, processor share).

Usage: python scripts/show_model_facts.py model1.gguf [model2.gguf ...] [--layers N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferenceos.models.loader import load_model_profile
from inferenceos.tune.model_facts import processor_share, read_model_facts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out: list[dict[str, object]] = []
    for name in args.models:
        path = Path(name)
        model = load_model_profile(path)
        facts = read_model_facts(path, model)
        total = facts.total_layers
        shares = {
            str(n): (None if (s := processor_share(facts, n, card=True)) is None else round(s, 3))
            for n in sorted({0, total // 4, total // 2, total, total + 1})
        }
        row = {
            "model": path.name,
            "architecture": model.architecture,
            "design": facts.design_words(),
            "attention_layers": facts.attention_layers,
            "other_layers": facts.other_layers,
            "experts": facts.experts,
            "experts_used": facts.experts_used,
            "layer_mb": [round(b / 1e6, 1) for b in (facts.layer_bytes or ())],
            "output_mb": None if facts.output_bytes is None else round(facts.output_bytes / 1e6, 1),
            "other_mb": None if facts.other_bytes is None else round(facts.other_bytes / 1e6, 1),
            "file_mb": round(path.stat().st_size / 1e6, 1),
            "vocab": facts.vocabulary_size,
            "embedding": facts.embedding_dim,
            "heads": facts.attention_heads,
            "processor_share_at_layers": shares,
        }
        out.append(row)
        print(json.dumps(row))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
