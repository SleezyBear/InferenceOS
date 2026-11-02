"""Stage-B development collection: simulation or explicitly authorized frozen inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from inferenceos.config import Settings
from inferenceos.tune.lookup_answers_sim import simulate_answers
from inferenceos.tune.lookup_answers_audit import audit_answers
from evaluate_prompt_reuse import output_problem


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--simulate',action='store_true')
    mode.add_argument('--rules',type=Path)
    parser.add_argument('--rules-sha256')
    parser.add_argument('--authorized-stage-b',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    protected=[ROOT/n for n in ('src','scripts','tests','docs','models','tools','data')]
    protected += [Settings().data_dir,ROOT/'reviewer-implementation-handoff-current.md']
    if args.rules is not None:protected.append(args.rules)
    if output_problem(args.out,protected=protected) or args.out.exists() or args.out.is_symlink():return 4
    if not args.simulate:
        if not args.authorized_stage_b or not args.rules_sha256:return 4
        from collect_lookup_fit import Preflight,LiveFitEngine
        from evaluate_prompt_reuse import power_state
        from inferenceos.tune.lookup_answers_runner import run_native
        return run_native(root=ROOT,settings=Settings(),rules=args.rules,rules_sha=args.rules_sha256,
                          out=args.out,owner_authorized=True,preflight_factory=Preflight,
                          fit_engine_factory=LiveFitEngine,power=power_state)
    if args.authorized_stage_b or args.rules_sha256:return 4
    record=simulate_answers()
    record['audit']=audit_answers(record)
    if not record['audit']['evidence_valid']:return 2
    serialized=json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    try:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x',encoding='utf-8',newline='\n') as stream:stream.write(serialized)
    except OSError:return 4
    print(f"SIMULATED: {len(record['rows'])} invented answer loads; no native correctness/performance meaning")
    return 0


if __name__=='__main__':raise SystemExit(main())
