"""Clearly fabricated Stage-A fixtures, never model, fit or qualification evidence."""
from __future__ import annotations

from copy import deepcopy
import json
import hashlib
from typing import Any, Callable

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.tune.lookup_audit import REQUIRED_ENGINE_FILES
from inferenceos.tune.lookup_corpus import build_corpus, digest
from inferenceos.tune.lookup_fit import FitManifest, FitModel, audit_fit, collect_fit
from inferenceos.tune.lookup_fit_observers import observe_effects
from inferenceos.tune.lookup_qualification import ConfigurationSnapshot


def fixture_manifest() -> FitManifest:
    configuration = ConfigurationSnapshot(**RuntimeConfiguration(context_length=4096, gpu_offload_layers=8,
                                                                 ubatch_size=256).model_dump())
    models = tuple(FitModel(label=label, path="models/"+label+".gguf", sha256=digest(label),
                            configuration=configuration, configuration_sha256=digest(configuration.model_dump(mode="json")),
                            saved_setup_source="saved setup fixture-"+label+" (read only)")
                   for label in ("llama3b", "gemma1b"))
    return FitManifest.model_validate_json(json.dumps(dict(
        proposal_sha256=digest("simulated proposal"),protocol_document_sha256=digest("simulated document"),
        corpus_sha256=build_corpus().sha256,models=[m.model_dump(mode="json") for m in models],
        python_files={"src/fixture.py":digest("never executed")},
        engine_files={"tools/llama.cpp/"+name:digest(name) for name in sorted(REQUIRED_ENGINE_FILES)})))


def fixture_machine() -> dict[str, Any]:
    return dict(hostname="D2-FIT-FIXTURE",cim=dict(processors=[dict(Name="fixture CPU",Manufacturer="fixture vendor",
                Architecture=9,NumberOfCores=4,NumberOfLogicalProcessors=8)],
                os=dict(Caption="fixture Windows",Version="10.0",BuildNumber="fixture",OSArchitecture="64-bit"),
                video_controllers=[dict(Name="fixture GPU",DriverVersion="fixture driver")]),
                cpuid={"avx":True,"avx2":True},os_flags={"avx":True,"avx2":True},memory=dict(total_physical_bytes=16*2**30),
                vulkan=[dict(index=0,name="fixture GPU",vendor_id=0x1002,device_type=2,device_local_bytes=4*2**30)],
                power_plan="11111111-2222-3333-4444-555555555555",power_source="ac")


def fixture_log(arm: str) -> str:
    text = """system_info: n_threads = 4 (n_threads_batch = 4) / 8
llama threadpool init, n_threads = 4
llama_context: n_batch = 1024
llama_context: n_ubatch = 256
load_tensors: offloaded 8/29 layers to GPU
load_tensors: loading model tensors (load_mode = mmap)
load_model: initializing, n_slots = 4, n_ctx_slot = 4096
resolve_fused_ops: Flash Attention enabled
llama_prepare_model_devices: using device Vulkan0 (fixture GPU)
llama_kv_cache: K (f16): 1 MiB, V (f16): 1 MiB
common_init_: warming up the model with an empty run - please wait ... (--no-warmup to disable)
"""
    if arm == "lookup":
        text += "adding speculative implementation 'ngram-simple'\nsize_n=12, size_m=16, min_hits=1\nspeculative decoding context initialized\n"
    return text+"model loaded\n"


class FitSimulator:
    def __init__(self, manifest: FitManifest) -> None:
        self.manifest = manifest
        self.starts = 0
        self.releases = 0
        self.arm = ""
        self.active = False
        self.calls: list[Any] = []

    def start(self, model: FitModel, arm: str) -> None:
        assert not self.active
        self.starts += 1
        self.active = True
        self.arm = arm
        self.calls.append(("start",model.label,arm))

    def observe(self) -> dict[str, Any]:
        assert self.active
        text = fixture_log(self.arm)
        startup = dict(available=True,text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest(),generation_current=True,truncated=False,
                       dropped_lines=0,reader_errors=[],lines=len(text.splitlines()),line_limit=1000)
        files: list[dict[str, Any]] = [dict(path="C:/D2-FIT-FIXTURE/"+p,sha256=h,size_bytes=1,modified_ns=1) for p,h in self.manifest.engine_files.items()]
        image = dict(status="observed",files=files,executable=next(f["path"] for f in files if f["path"].endswith("llama-server.exe")),
                     pid=10000+self.starts,create_time_s=1000+self.starts,problems=[])
        slots = [dict(id=i,n_ctx=4096,is_processing=False,speculative=self.arm == "lookup") for i in range(4)]
        command=[image['executable'],'--offline','--spec-type','none' if self.arm=='control' else 'ngram-simple']
        if self.arm=='lookup':command+=['--spec-ngram-simple-size-n','12','--spec-ngram-simple-size-m','16']
        return dict(startup=startup,image=image,command=command,engine_environment={},selected_executable=image["executable"],process_current=True,slots=slots,machine=fixture_machine(),
                    effects=observe_effects(text,slots))

    def tokenize(self, text: str, *, add_special: bool) -> list[int]:
        assert self.active
        self.calls.append(("tokenize",add_special))
        # Deliberately invented, stable fake tokens, not a tokenizer or token fit.
        return ([1] if add_special else [])+[int(digest(text[i:i+32])[:8],16) for i in range(0,len(text),32)]

    def release(self, *, abort: bool) -> dict[str, Any]:
        self.calls.append(("release",abort))
        self.active = False
        self.releases += 1
        return dict(process_gone=True,credential_removed=True,problem=None)


def simulate_fit(*, checkpoint: Callable[[dict[str, Any]], None] = lambda _:None) -> dict[str, Any]:
    manifest = fixture_manifest()
    engine = FitSimulator(manifest)
    result = collect_fit(engine,manifest,run_id="D2_FIT_SIMULATED",power=lambda:"ac",guard=lambda:None,
                         checkpoint=checkpoint,simulated=True)
    result["manifest"] = manifest.model_dump(mode="json")
    result["audit"] = audit_fit(deepcopy(result),manifest)
    result["fixture_only"] = True
    result["measured_token_fit"] = False
    result["hardware_authorized"] = False
    return result
