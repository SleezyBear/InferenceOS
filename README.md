# InferenceOS

Autonomous adaptive local LLM optimization system. The authoritative specification is
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

**Phase 0 — Contracts & Environment: complete.**
**Phase 1 — Hardware & System Characterization: complete.**

Phase 0 establishes the vocabulary and interfaces every later phase builds against,
and the real environment baseline (llama.cpp + Vulkan on the AMD Radeon, a real GGUF
model, logging, config, and the audit-record schema).

Phase 1 populates the `HardwareProfile` and live `SystemState` contracts from real
Windows probes: CPUID-derived CPU features, real Vulkan device enumeration plus
CUDA/ROCm presence checks, `GlobalMemoryStatusEx` RAM, physical-disk/media
classification, and live per-process CPU/RSS, hard page-fault and GPU engine
telemetry.

## Layout

```
src/inferenceos/
  contracts/      core data contracts (HardwareProfile, SystemState, ... +
                  BackendAvailability/BackendDevice for detected-vs-usable backends)
  interfaces/     abstract ReasoningEngine and BackendAdapter interfaces
  reasoning/      StubReasoningEngine (interface conformance, no LLM)
  native/         platform-specific probes (WMI/CIM, CPUID, Vulkan, Win32 memory,
                  Windows performance counters) kept behind one module
  hardware/       host-independent characterization (CPU/GPU/RAM/storage/OS)
  telemetry/      Windows GPU Engine utilization sampler + live SystemState sampler
  backends/       thin real process wrapper around the llama.cpp CLI
  audit/          audit-record schema (Phase 8/9 persistence + rollback)
  config.py       runtime configuration (INFERENCEOS_* env overrides)
  logging_config.py
  selftest.py     contract round-trips + reasoning interface self-test
  cli.py
tests/            pytest suite (deterministic + real-machine integration)
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

## Human-verifiable Phase 1 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos profile
```

Prints the full real `HardwareProfile` and a live `SystemState` snapshot for this
machine — CPU identity/cores/caches/instruction sets, detected GPUs and usable
backends, physical RAM and modules, physical storage with SSD/HDD media type, the
storage backing the model file, top processes by RSS with CPU%, hard page-fault
rate, and GPU engine/adapter-memory telemetry — then writes a JSON report to
`reports/`. Cross-check the CPU/cores and RAM against Task Manager, and the GPU
against Device Manager.

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase1
```

Runs the real Phase 1 acceptance: it builds the profile, captures live telemetry,
cross-checks the CPU/RAM figures against `psutil` (the OS), identifies the storage
backing the model, and runs a short real llama.cpp Vulkan inference while observing
real Windows GPU Engine utilization. It prints every check as PASS/FAIL and writes
`reports/phase1_verification_*.json`; it exits non-zero unless all checks pass.

## Phase 1 detection notes / limitations

- **CPU features** come from a real `CPUID` probe (cross-checked against Windows
  `IsProcessorFeaturePresent`). AVX-family features are only reported when the OS
  has enabled the extended state. Cache sizes are as reported by CPUID leaf 4 for
  the calling processor (L1/L2 per-core, L3 physical), with a WMI fallback.
- **GPU usability** is decided by real Vulkan physical-device enumeration; a GPU
  visible in WMI but not enumerated by Vulkan is recorded as detected-but-not-usable.
  CUDA/ROCm are reported as *detected* when their runtimes are loadable; their
  device usability is not validated in Phase 1.
- **GPU total VRAM** is only available from Vulkan device-local heaps; the Windows
  `GPU Adapter Memory` counters expose *dedicated usage* but not a total, so the
  live `gpu_memory[*].memory_total_bytes` is `None` rather than a fabricated value.
- **Hard page faults** are read from `\Memory\Pages Input/sec` (real hard faults);
  Windows has no swap/swappiness equivalent, so commit ratio and hard-fault rate are
  the pressure signals.
- **Storage media** (SSD/HDD) is read from the Storage management namespace
  (`MSFT_PhysicalDisk`); direct physical-drive IOCTL classification was rejected
  because it requires elevation (standard-user default).
- **GPU `device_index`** in live telemetry follows Windows LUID ordering, which is
  not necessarily the same ordering as Vulkan device indices.
