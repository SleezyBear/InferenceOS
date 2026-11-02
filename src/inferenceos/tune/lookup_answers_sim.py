"""Invented answer/cost observations. Never native correctness or performance evidence."""
from __future__ import annotations

from copy import deepcopy
from typing import Any,Callable

from inferenceos.tune.lookup_answers import VERSION,request_body
from inferenceos.tune.lookup_answers_collect import collect_answers,NO_CHECKPOINT
from inferenceos.tune.lookup_corpus import build_corpus,digest
from inferenceos.tune.lookup_fit_sim import FitSimulator,fixture_manifest
from inferenceos.tune.lookup_fit_observers import machine_from_raw


class FakeClock:
    def __init__(self) -> None:self.now=1_000_000_000
    def __call__(self) -> int:
        self.now+=1_000_000
        return self.now
    def advance(self,ns: int) -> None:self.now+=ns


def fixture_scope() -> dict[str,Any]:
    manifest=fixture_manifest();models={};machine=None
    for model in manifest.models:
        engine=FitSimulator(manifest);engine.start(model,'control');raw=engine.observe()
        from inferenceos.tune.lookup_fit import token_inputs
        tokens=[dict(t,ids=engine.tokenize(t['text'],add_special=t['add_special'])) for t in token_inputs()]
        for t in tokens:t['ids_sha256']=digest(t['ids'])
        models[model.label]=dict(requested=model.configuration.model_dump(mode='json'),effective=raw['effects']['configuration'],
                                 fit_tokens=tokens,model_sha256=model.sha256)
        machine=machine_from_raw(raw['machine'],raw['effects']['configuration']['device'])
    return dict(models=models,machine=machine,engine_files=manifest.engine_files,executor_sha256=digest('SIMULATED '+VERSION))


class AnswerSimulator:
    def __init__(self,clock: FakeClock) -> None:
        self.clock=clock;self.sim=FitSimulator(fixture_manifest());self.model='';self.arm='';self.prompt_ids: list[int]=[]
        self.answers=0;self.tokens=0
    def start(self,model: str,arm: str) -> None:
        self.model,self.arm=model,arm
        self.sim.start(next(m for m in self.sim.manifest.models if m.label==model),arm)
    def observe(self) -> dict[str,Any]:return self.sim.observe()
    def tokenize(self,text: str,*,add_special: bool) -> list[int]:
        self.tokens+=1;self.prompt_ids=self.sim.tokenize(text,add_special=add_special)
        return self.prompt_ids
    def answer(self,body: dict[str,Any],chunk: Callable[[dict[str,Any]],None]) -> None:
        assert self.sim.active
        case=next(c for c in build_corpus().cases if c.prompt==body['prompt'])
        assert body==request_body(case)
        self.answers+=1
        self.clock.advance(100_000_000 if self.arm=='control' else 65_000_000+(self.sim.starts%8)*400_000)
        ids=list(range(100,100+max(1,len(case.expected)//4)))
        chunk(dict(content=case.expected,tokens=ids,stop=False))
        timings=dict(prompt_n=len(self.prompt_ids),cache_n=0,predicted_n=len(ids),prompt_ms=1.25,predicted_ms=10.5)
        if self.arm=='lookup':timings.update(draft_n=8,draft_n_accepted=4)
        chunk(dict(content='',tokens=[],stop=True,stop_type='eos',truncated=False,id_slot=0,
                   tokens_evaluated=len(self.prompt_ids),tokens_predicted=len(ids),timings=timings,
                   generation_settings={'seed':42,'n_predict':512,'temperature':0,'ignore_eos':False,
                                        'speculative.types':'none' if self.arm=='control' else 'none,ngram-simple'}))
    def release(self,*,abort: bool) -> dict[str,Any]:return self.sim.release(abort=abort)


def simulate_answers(*,checkpoint: Callable[[dict[str,Any]],None]=NO_CHECKPOINT) -> dict[str,Any]:
    clock=FakeClock();engine=AnswerSimulator(clock)
    raw=collect_answers(engine,fixture_scope(),run_id='STAGE_B_SIMULATED',clock=clock,power=lambda:'ac',guard=lambda:None,
                        rest=lambda seconds:clock.advance(seconds*1_000_000_000),checkpoint=checkpoint,simulated=True)
    raw['fixture_only']=True
    raw['hardware_or_native_meaning']=False
    raw['native_engine_starts']=0
    return raw
