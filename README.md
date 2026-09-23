# InferenceOS

Autonomous adaptive local LLM optimization system. The authoritative specification is
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

**Phase 0 — Contracts & Environment: complete.**
**Phase 1 — Hardware & System Characterization: complete.**
**Phase 2 — Model & Backend Abstraction: complete.**
**Phase 3 — Baseline Profiling & Telemetry: complete.**

Phase 0 establishes the vocabulary and interfaces every later phase builds against,
and the real environment baseline (llama.cpp + Vulkan on the AMD Radeon, a real GGUF
model, logging, config, and the audit-record schema).

Phase 1 populates the `HardwareProfile` and live `SystemState` contracts from real
Windows probes: CPUID-derived CPU features, real Vulkan device enumeration plus
CUDA/ROCm presence checks, `GlobalMemoryStatusEx` RAM, physical-disk/media
classification, and live per-process CPU/RSS, hard page-fault and GPU engine
telemetry.

Phase 2 adds real GGUF metadata introspection and the concrete llama.cpp backend
adapter: it reads the actual GGUF metadata (architecture, layers, attention/KV
heads, head dimension, context length, quantization, parameter count) into
`ModelProfile`, computes an architecture-derived KV-cache/footprint estimate that is
correct for GQA/MQA and MHA, and runs real inference through `BackendAdapter`.

Phase 3 adds the baseline profiling harness: a deterministic, versioned workload
suite, real repeated runs through the backend adapter with per-phase timing (load,
TTFT, prefill, decode), throughput, token counts, peak RSS, CPU/GPU/page-fault
telemetry sampled over the inference interval, explicit metric provenance, and
persistent self-contained baseline reports with observed run-to-run variance.

## Layout

```
src/inferenceos/
  contracts/      core data contracts (HardwareProfile, ModelProfile, ModelFootprint,
                  SystemState, ... + BackendAvailability/BackendDevice)
  interfaces/     abstract ReasoningEngine and BackendAdapter interfaces
  reasoning/      StubReasoningEngine (interface conformance, no LLM)
  native/         platform-specific probes (WMI/CIM, CPUID, Vulkan, Win32 memory,
                  Windows performance counters) kept behind one module
  hardware/       host-independent characterization (CPU/GPU/RAM/storage/OS)
  models/         real GGUF reader + architecture/quantization normalization +
                  KV/footprint estimator + ModelProfile loader
  profiling/      deterministic benchmark workloads + repeated-run harness +
                  aggregation statistics + baseline reports
  telemetry/      persistent Windows performance-counter sampler + live SystemState
  backends/       real llama.cpp adapter (+ low-level CLI toolchain/timing/memory parsing)
  audit/          audit-record schema (Phase 8/9 persistence + rollback)
  config.py       runtime configuration (INFERENCEOS_* env overrides)
  logging_config.py
  selftest.py     contract round-trips + reasoning interface self-test
  cli.py
tests/            pytest suite (deterministic + real-machine integration)
tools/            llama.cpp Vulkan build + app-local VC++ runtime
models/           real GGUF models (gitignored)
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

## Human-verifiable Phase 2 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos model-info
```

Parses the real GGUF metadata of every model in `models/` and prints the normalized
`ModelProfile` (format, architecture, quantization, parameter count, context length,
layers, attention heads, KV heads, head dimension, embedding/vocab) plus the KV-cache
and total-footprint estimates at several context lengths. `--json` emits a
machine-readable form; `--model <path>` targets one file.

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase2
```

Runs the real Phase 2 acceptance over at least two actual GGUF models:

1. parses each model's real metadata into `ModelProfile` and reports the
   architecture-derived KV/footprint estimates;
2. exercises the concrete `BackendAdapter` (real `llama-cli`), prints the backend
   capabilities, and generates real text through the **Vulkan/Radeon** path with
   layer offload;
3. measures **predicted vs measured** memory across 2–3 context lengths on the CPU
   (so the KV cache is in system RAM) and compares the measured peak-RSS delta
   against the predicted KV delta, alongside the KV size the backend itself reports.

It prints every check PASS/FAIL and writes `reports/phase2_verification_*.json`;
it exits non-zero unless all checks pass.

## Phase 2 model/estimation notes / limitations

- **KV formula** (per sequence position):
  `n_layers * n_kv_heads * (key_length + value_length) * element_bytes`, with
  `element_bytes = 2` for the default f16 cache. The **KV-head** count is used, not
  the attention-head count, so GQA/MQA models are not overestimated.
- **`weight_bytes` is an estimate** derived from the real (already-quantized) file
  size; it is deliberately kept distinct from `file_size_bytes` (the fs size), the
  KV estimate, and from **measured** process RSS. Phase 2 only performs the limited
  measurements needed to validate the estimate; it does not build the Phase 3
  profiling harness.
- **Backend execution model:** the acquired llama.cpp build is a one-shot CLI, so
  the adapter's `load` pins the model + runtime configuration (and probes real
  devices) and `generate` runs a real `llama-cli` process. Residency is per
  invocation; a resident-server adapter can implement the same `BackendAdapter`
  interface later without changing callers.
- **KV-cache type:** the estimator assumes an f16 cache; `ModelProfile`/
  `ModelFootprint` scale linearly for other element widths, but quantized KV-cache
  types (block scales) are not modelled and remain estimates.
- **Real models used for acceptance:** `qwen2.5-1.5b-instruct-q4_k_m.gguf`
  (Qwen2, GQA, 28 layers / 12 heads / 2 KV) and `stories15M-q4_0.gguf`
  (Llama, standard MHA, 6 layers / 6 heads / 6 KV). Models are gitignored and not
  committed.

## Human-verifiable Phase 3 evidence

```powershell
.\.venv\Scripts\python.exe -m inferenceos baseline --repetitions 3
```

Runs the deterministic benchmark suite (`short-short`, `long-prefill`,
`short-long-decode`) against every model in `models/` through the real backend and
prints a per-run table plus aggregated statistics (mean/stddev/min/max/coefficient
of variation), then writes a self-contained `reports/baseline_*.json`. Options:
`--model`, `--workload` (repeatable), `--repetitions`, `--mode cold|warm|both`,
`--context`, `--gpu-layers`, `--no-telemetry`, `--out`.

```powershell
.\.venv\Scripts\python.exe -m inferenceos verify-phase3
```

Runs the real Phase 3 acceptance: it profiles both structurally different real
models (GQA + MHA) through the **Vulkan/Radeon** path, captures per-phase timings
(load, TTFT, prefill, decode), throughput, token counts, peak RSS and interval
CPU/GPU/hard-fault/memory telemetry, demonstrates repeated-run reproducibility
against agreed variance bands, performs a cold run and a profiler-overhead
comparison, and prints every check PASS/FAIL. It writes
`reports/phase3_verification_*.json` and exits non-zero unless all checks pass.

## Phase 3 measurement notes / limitations

- **Only measurements** are produced; there is no diagnosis, ranking, or automatic
  configuration change (those are Phase 4+).
- **Cold vs warm semantics are honest:** every measured run is a fresh `llama-cli`
  process. `COLD` = no preceding warm-up (model likely not in the OS page cache);
  `WARM` = measured after an explicit warm-up so the model file is OS-page-cache
  resident. `WARM` does **not** mean a resident model process or reused KV/weights;
  resident-process warm inference is not supported by this adapter and is reported
  as `resident_warm_supported: false`.
- **TTFT is derived** (prefill + one decode step) because llama.cpp exposes no
  first-token timestamp; it is labelled `derived` in each run's metric-quality map.
- **Unavailable telemetry is explicit:** GPU total VRAM is not exposed by Windows
  counters and is reported as `None` with quality `unavailable`.
- **Profiler overhead** is quantified in acceptance (decode tokens/sec with external
  counter sampling on vs off; observed ~1–5%).
- **COLD runs are performed once per model/workload** because the OS page cache
  cannot be evicted without elevation; repeating them would not be genuinely cold.
