"""Deterministic lookup-study test engine. No model, network or hardware meaning."""
from __future__ import annotations

from typing import Any

from inferenceos.tune.lookup_decoding import TOKENS, SEED, LookupCase, digest
from inferenceos.tune.prompt_reuse import Cleanup, EngineVariant


class LookupSimulator:
    def __init__(self, *, fault: str = "") -> None:
        self.fault = fault
        self.starts = 0
        self.requests = 0
        self.stops = 0
        self.active = False
        self.arm = ""
        self.commands: list[list[str]] = []
        self.load_evidence: list[dict[str, Any]] = []

    def start(self, variant: EngineVariant) -> None:
        if self.active:
            raise RuntimeError("simulation: previous load still active")
        self.active = True
        self.starts += 1
        self.arm = variant.variant_id
        if self.fault == "start-error":
            raise RuntimeError("injected start error")
        if self.fault == "start-interrupt":
            raise KeyboardInterrupt

    def request(self, case: LookupCase) -> dict[str, Any]:
        if not self.active:
            raise RuntimeError("simulation: no load")
        self.requests += 1
        if self.fault == "request-error":
            raise RuntimeError("injected request error")
        if self.fault == "request-interrupt":
            raise KeyboardInterrupt
        ids = list(range(TOKENS))
        candidate = self.arm == "lookup"
        opportunity = case.case_id != "compose-prose"
        prompt_ids = [ord(c) for c in case.prompt]
        result: dict[str, Any] = {
            "text": f"SIMULATED OUTPUT for {case.case_id}. " + " ".join(map(str, ids)),
            "token_ids": ids, "prompt_tokens": len(prompt_ids), "prompt_ids_sha256": digest(prompt_ids),
            "total_s": 8.0 if candidate and opportunity else 10.0,
            "ttft_s": 1.0, "peak_rss_bytes": 1000000, "load_evidence_id": None,
            "counters": {"tokens_evaluated": len(prompt_ids), "prompt_n": len(prompt_ids), "cache_n": 0,
                         "tokens_predicted": TOKENS, "predicted_n": TOKENS, "stop_seen": True,
                         "stop_type": "limit", "truncated": False, "id_slot": 0,
                         "generation_settings": {"speculative.types": "ngram-simple" if candidate else "none",
                                                 "n_predict": TOKENS, "seed": SEED, "temperature": 0.0,
                                                 "ignore_eos": False}},
        }
        if candidate and opportunity:
            result["counters"].update(draft_n=120, draft_n_accepted=96)
        if candidate:
            if self.fault == "quality":
                result["text"] += " changed"
            elif self.fault == "ids":
                result["token_ids"][-1] += 1
            elif self.fault == "short":
                result["counters"]["tokens_predicted"] = 64
            elif self.fault == "reuse":
                result["counters"]["cache_n"] = 5
            elif self.fault == "counter":
                result["counters"]["draft_n_accepted"] = 1000
            elif self.fault == "inactive":
                result["counters"]["generation_settings"]["speculative.types"] = "none"
        return result

    def stop(self) -> Cleanup:
        self.stops += 1
        if self.fault == "cleanup":
            return Cleanup(process_gone=False, credential_removed=False, problem="injected held process/key")
        self.active = False
        return Cleanup(process_gone=True, credential_removed=True)

    def retry_held(self, *, propagate_interrupt: bool = False) -> None:
        # A simulator cannot claim that a real cleanup or diagnostic retry occurred.
        return None

    def load_problems(self) -> list[str]:
        # A simulation has no live observations to attest. The offline auditor
        # always refuses to turn it into hardware evidence.
        return []
