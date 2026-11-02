"""Offline audit of simulated or provenance-bound native Stage-B evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from inferenceos.config import Settings
from inferenceos.tune.lookup_fit import strict_json
from inferenceos.tune.lookup_fit_io import bounded_file_bytes
from inferenceos.tune.lookup_answers_audit import audit_answers
from inferenceos.tune.lookup_answer_publication import bounded_study_bytes,MAX_JOURNAL_BYTES,sha
from evaluate_prompt_reuse import output_problem


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rules',type=Path);parser.add_argument('--rules-sha256')
    parser.add_argument('--closure',type=Path);parser.add_argument('--journal',type=Path)
    args=parser.parse_args(argv)
    protected=[ROOT/n for n in ('src','scripts','tests','docs','models','tools','data')]
    protected += [Settings().data_dir,args.record,ROOT/'reviewer-implementation-handoff-current.md']
    protected += [p for p in (args.rules,args.closure,args.journal) if p is not None]
    if output_problem(args.out,protected=protected) or args.out.exists() or args.out.is_symlink():return 4
    try:
        raw=bounded_study_bytes(args.record)
        if args.rules is None:
            if any((args.rules_sha256,args.closure,args.journal)):return 4
            result=audit_answers(strict_json(raw))
        else:
            if not args.rules_sha256:return 4
            from inferenceos.tune.lookup_answers_runner import load_inputs
            from inferenceos.tune.lookup_answers_study import audit_native_study
            closure=args.closure or args.record.with_name(args.record.name+'.closure.json')
            journal=args.journal or args.record.with_name(args.record.name+'.journal')
            if args.out.resolve() in (closure.resolve(),journal.resolve()):return 4
            rules_raw=bounded_file_bytes(args.rules,maximum_bytes=64*1024*1024)
            inputs=load_inputs(ROOT,args.rules,args.rules_sha256)
            closure_raw=bounded_file_bytes(closure,maximum_bytes=16*1024*1024)
            journal_raw=bounded_study_bytes(journal,maximum_bytes=MAX_JOURNAL_BYTES)
            result=audit_native_study(raw,closure_raw,journal_raw,inputs)
            if (bounded_file_bytes(args.rules,maximum_bytes=64*1024*1024)!=rules_raw
                    or sha(bounded_file_bytes(closure,maximum_bytes=16*1024*1024))!=sha(closure_raw)
                    or sha(bounded_study_bytes(journal,maximum_bytes=MAX_JOURNAL_BYTES))!=sha(journal_raw)):
                raise ValueError('governing native evidence changed during audit')
            # Historical provenance is reread through the same pinned loader.
            load_inputs(ROOT,args.rules,args.rules_sha256)
        result['source_sha256']=hashlib.sha256(raw).hexdigest()
        if sha(bounded_study_bytes(args.record))!=sha(raw):raise ValueError('record changed during audit')
        serialized=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8',newline='\n') as stream:stream.write(serialized)
    except (ValueError,OSError,TypeError):return 4
    return 0 if result['evidence_valid'] and result.get('study_complete') else 2


if __name__=='__main__':raise SystemExit(main())
