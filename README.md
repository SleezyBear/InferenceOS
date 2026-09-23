# InferenceOS

Autonomous adaptive local LLM optimization system. The authoritative specification is
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

**Phase 0 — Contracts & Environment: complete.**

Phase 0 establishes the vocabulary and interfaces every later phase builds against,
and the real environment baseline (llama.cpp + Vulkan on the AMD Radeon, a real GGUF
model, logging, config, and the audit-record schema).

## Layout

```
src/inferenceos/
  contracts/      core data contracts (HardwareProfile, ModelProfile, WorkloadProfile,
                  SystemState, Observation, Bottleneck, Candidate, ExperimentResult,
                  ReasoningRequest, ReasoningResponse)
  interfaces/     abstract ReasoningEngine and BackendAdapter interfaces
  reasoning/      StubReasoningEngine (interface conformance, no LLM)
  backends/       thin real process wrapper around the llama.cpp CLI
  telemetry/      Windows GPU Engine utilization sampler
  audit/          audit-record schema (Phase 8/9 persistence + rollback)
  config.py       runtime configuration (INFERENCEOS_* env overrides)
  logging_config.py
  selftest.py     contract round-trips + reasoning interface self-test
  cli.py
tests/            pytest suite
tools/            llama.cpp Vulkan build + app-local VC++ runtime
models/           real GGUF models
reports/          human-verifiable verification reports (JSON)
```

## Environment baseline (this machine)

- Windows Boot Camp, Intel i7-8850H (6C/12T), 16 GB RAM, Apple SSD.
- GPU: **Radeon Pro 560X**, AMD driver 31.0.21921.1000 (Adrenalin 24.9.1), Vulkan 1.3.260.
- llama.cpp: `tools/llama.cpp` (release **b11139**, `llama-*-bin-win-vulkan-x64`).
- Microsoft VC++ 2015–2022 x64 runtime deployed app-local in `tools/llama.cpp`
  (extracted from the official `vc_redist.x64.exe`; the Boot Camp image lacked
  `vcruntime140_1.dll`). Reference copy in `tools/vcredist_x64/`.
- Model: `models/qwen2.5-1.5b-instruct-q4_k_m.gguf`.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Human-verifiable Phase 0 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase0
```

This loads the real model, runs one prompt through the real Vulkan-enabled
llama.cpp binary, and prints:

1. the **generated text**,
2. the **backend/device** that executed it (Vulkan0 / Radeon Pro 560X, layers offloaded),
3. **GPU utilization** observed via Windows GPU Engine counters during the call,

then writes a JSON report to `reports/`. It exits non-zero unless every acceptance
check passes.

Additional checks:

```powershell
.\.venv\Scripts\python.exe -m inferenceos contracts-check      # contract JSON round-trips
.\.venv\Scripts\python.exe -m inferenceos reasoning-selftest   # stub ReasoningEngine round-trip
.\.venv\Scripts\python.exe -m mypy                             # all contracts type-check
.\.venv\Scripts\python.exe -m pytest                           # unit + deterministic tests
```
