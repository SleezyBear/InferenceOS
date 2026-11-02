"""Offline Stage-A recomputation; never constructs a backend or a Stage-B manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'src'))

from inferenceos.config import Settings
from inferenceos.tune.lookup_fit import audit_fit, parse_manifest, strict_json
from inferenceos.tune.lookup_fit_io import bounded_file_bytes
from evaluate_prompt_reuse import output_problem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record',type=Path)
    parser.add_argument('--rules',type=Path,required=True)
    parser.add_argument('--rules-sha256',required=True)
    parser.add_argument('--proposal',type=Path,required=True)
    parser.add_argument('--proposal-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(argv)
    protected = [ROOT / n for n in ('src','scripts','tests','docs','models','tools','data')]
    protected += [Settings().data_dir,args.record,args.rules,args.proposal,ROOT/'reviewer-implementation-handoff-current.md']
    if output_problem(args.out,protected=protected) or args.out.exists() or args.out.is_symlink():
        return 4
    try:
        rules_raw,proposal_raw = bounded_file_bytes(args.rules),bounded_file_bytes(args.proposal)
        if (hashlib.sha256(rules_raw).hexdigest()!=args.rules_sha256
                or hashlib.sha256(proposal_raw).hexdigest()!=args.proposal_sha256):
            raise ValueError('manifest/proposal bytes differ')
        manifest = parse_manifest(rules_raw,strict_json(proposal_raw),args.proposal_sha256)
        raw = bounded_file_bytes(args.record,maximum_bytes=64*1024*1024)
        result: dict[str,Any] = audit_fit(strict_json(raw),manifest)
        result.update(source_sha256=hashlib.sha256(raw).hexdigest(),rules_sha256=args.rules_sha256,
                      proposal_sha256=args.proposal_sha256)
        if (bounded_file_bytes(args.record,maximum_bytes=64*1024*1024)!=raw
                or bounded_file_bytes(args.rules)!=rules_raw or bounded_file_bytes(args.proposal)!=proposal_raw):
            raise ValueError('record or governing input changed during audit')
        serialized = json.dumps(result,indent=2,allow_nan=False)+'\n'
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8') as stream:
            stream.write(serialized)
        return 0 if result['observations_complete'] else 2
    except (OSError,ValueError,TypeError) as exc:
        print('refused: '+str(exc))
        return 4


if __name__=='__main__':
    raise SystemExit(main())
